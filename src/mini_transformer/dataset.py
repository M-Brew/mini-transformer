
from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor


class LanguageModelDataset:
    """Stores tokenized data and produces next-token training batches."""

    def __init__(
        self,
        tokens: Tensor,
        block_size: int,
    ) -> None:
        if tokens.ndim != 1:
            raise ValueError("tokens must be a one-dimensional tensor")

        if tokens.dtype != torch.long:
            raise ValueError("tokens must have dtype torch.long")

        if block_size <= 0:
            raise ValueError("block_size must be positive")

        if len(tokens) <= block_size:
            raise ValueError(
                "Token sequence must contain more tokens than block_size"
            )

        self.tokens = tokens.contiguous()
        self.block_size = block_size

    def __len__(self) -> int:
        return len(self.tokens) - self.block_size

    def get_example(self, index: int) -> tuple[Tensor, Tensor]:
        if index < 0 or index >= len(self):
            raise IndexError("Dataset index out of range")

        start = index
        end = start + self.block_size

        x = self.tokens[start:end]
        y = self.tokens[start + 1:end + 1]

        return x, y

    def get_batch(
        self,
        batch_size: int,
        device: str | torch.device = "cpu",
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor, Tensor]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        starts = torch.randint(
            low=0,
            high=len(self),
            size=(batch_size,),
            generator=generator,
        )

        offsets = torch.arange(self.block_size)

        x = self.tokens[starts[:, None] + offsets]
        y = self.tokens[starts[:, None] + offsets + 1]

        return x.to(device), y.to(device)


def split_tokens(
    tokens: Tensor,
    validation_fraction: float = 0.1,
) -> tuple[Tensor, Tensor]:
    """Split a continuous token stream before constructing context windows."""

    if tokens.ndim != 1:
        raise ValueError("tokens must be one-dimensional")

    if not 0.0 < validation_fraction < 1.0:
        raise ValueError(
            "validation_fraction must be between 0 and 1"
        )

    split_index = int(len(tokens) * (1.0 - validation_fraction))

    if split_index <= 0 or split_index >= len(tokens):
        raise ValueError("Split would produce an empty partition")

    train_tokens = tokens[:split_index].contiguous()
    validation_tokens = tokens[split_index:].contiguous()

    return train_tokens, validation_tokens


def save_token_ids(
    tokens: Tensor,
    path: str | Path,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    tokens = tokens.to(dtype=torch.long, device="cpu")
    torch.save(tokens, path)


def load_token_ids(path: str | Path) -> Tensor:
    # Only load files created by our own preparation pipeline.
    # Do not load untrusted pickle-based PyTorch files.
    tokens = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )

    if not isinstance(tokens, Tensor):
        raise ValueError("Expected a tensor in token data file")

    if tokens.ndim != 1 or tokens.dtype != torch.long:
        raise ValueError("Invalid token data format")

    return tokens


def prepare_token_datasets(
    tokens: Tensor,
    block_size: int,
    validation_fraction: float = 0.1,
) -> tuple[LanguageModelDataset, LanguageModelDataset]:
    train_tokens, validation_tokens = split_tokens(
        tokens,
        validation_fraction,
    )

    if len(train_tokens) <= block_size:
        raise ValueError(
            "Training partition must contain more tokens than block_size"
        )

    if len(validation_tokens) <= block_size:
        raise ValueError(
            "Validation partition must contain more tokens than block_size"
        )

    train_dataset = LanguageModelDataset(
        train_tokens,
        block_size,
    )

    validation_dataset = LanguageModelDataset(
        validation_tokens,
        block_size,
    )

    return train_dataset, validation_dataset
