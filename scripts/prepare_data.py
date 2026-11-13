
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from mini_transformer.config import load_config
from mini_transformer.dataset import save_token_ids
from mini_transformer.tokenizer import CharacterTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare the Mini Transformer training corpus"
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/baseline.yaml"),
        help="Path to the YAML configuration",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = load_config(args.config)

    input_path = Path(config.data.input_path)
    train_path = Path(config.data.train_path)
    validation_path = Path(config.data.validation_path)

    if not input_path.is_file():
        raise FileNotFoundError(
            f"Training corpus not found: {input_path}"
        )

    raw_text = input_path.read_text(encoding="utf-8")

    if not raw_text.strip():
        raise ValueError("Training corpus is empty")

    split_index = int(
        len(raw_text) * (1.0 - config.data.validation_fraction)
    )

    train_text = raw_text[:split_index]
    validation_text = raw_text[split_index:]

    if not train_text or not validation_text:
        raise ValueError(
            "The corpus is too small for the requested split"
        )

    tokenizer = CharacterTokenizer.train(train_text)

    train_ids = torch.tensor(
        tokenizer.encode(train_text),
        dtype=torch.long,
    )

    validation_ids = torch.tensor(
        tokenizer.encode(validation_text),
        dtype=torch.long,
    )

    block_size = config.model.block_size

    if len(train_ids) <= block_size:
        raise ValueError(
            "Training partition is too short for the configured block_size"
        )

    if len(validation_ids) <= block_size:
        raise ValueError(
            "Validation partition is too short for the configured block_size"
        )

    save_token_ids(train_ids, train_path)
    save_token_ids(validation_ids, validation_path)

    tokenizer_path = train_path.parent / "tokenizer.json"
    tokenizer.save(tokenizer_path)

    metadata = {
        "tokenizer_type": "character",
        "vocab_size": tokenizer.vocab_size,
        "block_size": block_size,
        "train_tokens": len(train_ids),
        "validation_tokens": len(validation_ids),
        "validation_fraction": config.data.validation_fraction,
        "unknown_token": tokenizer.UNK_TOKEN,
    }

    metadata_path = train_path.parent / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("Dataset preparation complete.")
    print(f"Tokenizer vocabulary size: {tokenizer.vocab_size}")
    print(f"Training tokens: {len(train_ids):,}")
    print(f"Validation tokens: {len(validation_ids):,}")
    print(f"Tokenizer saved to: {tokenizer_path}")
    print(f"Training data saved to: {train_path}")
    print(f"Validation data saved to: {validation_path}")
    print(f"Metadata saved to: {metadata_path}")


if __name__ == "__main__":
    main()
