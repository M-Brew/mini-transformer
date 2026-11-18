
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

from mini_transformer.config import (
    load_config,
    resolve_model_vocab_size,
)
from mini_transformer.dataset import (
    LanguageModelDataset,
    load_token_ids,
)
from mini_transformer.model import GPTLanguageModel
from mini_transformer.tokenizer import CharacterTokenizer
from mini_transformer.training import train_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the mini Transformer language model."
    )

    parser.add_argument(
        "--config",
        default="configs/baseline.yaml",
        help="Path to YAML configuration.",
    )

    parser.add_argument(
        "--resume",
        default=None,
        help="Optional checkpoint to resume from.",
    )

    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "mps"],
        help="Execution device.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override configured random seed.",
    )

    return parser.parse_args()


def select_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")

    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable.")

        return torch.device("mps")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()

    app_config = load_config(args.config)

    seed = app_config.seed if args.seed is None else args.seed

    set_seed(seed)

    device = select_device(args.device)

    print(f"Device: {device}")
    print(f"Seed: {seed}")

    # Load tokenizer and resolve vocabulary size.
    tokenizer_path = (
        Path(app_config.data.train_path).parent / "tokenizer.json"
    )

    tokenizer = CharacterTokenizer.load(tokenizer_path)

    model_config = resolve_model_vocab_size(
        app_config.model,
        tokenizer.vocab_size,
    )

    # Load preprocessed token sequences.
    train_tokens = load_token_ids(app_config.data.train_path)
    validation_tokens = load_token_ids(app_config.data.validation_path)

    if train_tokens.numel() <= model_config.block_size:
        raise ValueError("Training data is too short for block_size.")

    if validation_tokens.numel() <= model_config.block_size:
        raise ValueError("Validation data is too short for block_size.")

    train_dataset = LanguageModelDataset(
        train_tokens,
        block_size=model_config.block_size,
    )

    validation_dataset = LanguageModelDataset(
        validation_tokens,
        block_size=model_config.block_size,
    )

    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Training tokens: {train_tokens.numel():,}")
    print(f"Validation tokens: {validation_tokens.numel():,}")
    print(f"Training examples: {len(train_dataset):,}")
    print(f"Validation examples: {len(validation_dataset):,}")

    # Construct the model.
    model = GPTLanguageModel(model_config)

    print(f"Model parameters: {model.count_parameters():,}")

    checkpoint_dir = Path("artifacts/checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = Path("artifacts/training_metrics.jsonl")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    def log_metrics(metrics: dict) -> None:
        with metrics_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(metrics) + "\n")

    result = train_model(
        model,
        train_dataset,
        validation_dataset,
        app_config.training,
        device=device,
        checkpoint_dir=checkpoint_dir,
        resume_from=args.resume,
        seed=seed,
        metrics_callback=log_metrics,
    )

    summary = {
        "completed_steps": result.completed_steps,
        "best_validation_loss": result.best_validation_loss,
        "final_train_loss": result.final_train_loss,
        "final_validation_loss": result.final_validation_loss,
        "best_checkpoint": str(result.best_checkpoint_path),
        "latest_checkpoint": str(result.latest_checkpoint_path),
        "parameter_count": model.count_parameters(),
        "vocabulary_size": model.vocab_size,
        "device": str(device),
        "seed": seed,
    }

    summary_path = Path("artifacts/training_summary.json")

    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print("\nTraining complete.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
