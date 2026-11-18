
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from mini_transformer.inference import load_trained_generator


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate text using the trained mini Transformer."
    )

    parser.add_argument(
        "--prompt",
        required=True,
        help="Text prompt to continue.",
    )

    parser.add_argument(
        "--config",
        default="configs/baseline.yaml",
    )

    parser.add_argument(
        "--tokenizer",
        default="data/processed/tokenizer.json",
    )

    parser.add_argument(
        "--checkpoint",
        default="artifacts/checkpoints/best.pt",
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=200,
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--greedy",
        action="store_true",
        help="Use greedy decoding instead of sampling.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "mps"],
    )

    return parser.parse_args()


def select_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")

    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS is not available.")

        return torch.device("mps")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def main():
    args = parse_args()

    device = select_device(args.device)

    checkpoint_path = Path(args.checkpoint)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint does not exist: {checkpoint_path}"
        )

    generator = load_trained_generator(
        config_path=args.config,
        tokenizer_path=args.tokenizer,
        checkpoint_path=checkpoint_path,
        device=device,
    )

    generated_text = generator.generate(
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        do_sample=not args.greedy,
        seed=args.seed,
    )

    print("\nGenerated text:\n")
    print(generated_text)


if __name__ == "__main__":
    main()
