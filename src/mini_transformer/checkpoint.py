
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import torch
from torch import nn


def save_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    best_validation_loss: float,
    batch_generator_state: torch.Tensor | None = None,
) -> None:
    """
    Atomically save model and optimizer training state.

    Checkpoints are trusted training artifacts. Do not load checkpoints
    from untrusted sources.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint: dict[str, Any] = {
        "format_version": 1,
        "step": step,
        "best_validation_loss": best_validation_loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
    }

    if batch_generator_state is not None:
        checkpoint["batch_generator_state"] = batch_generator_state

    # Write to a temporary file in the same directory, then replace.
    # This avoids leaving a partially written checkpoint at `path`.
    file_descriptor, temporary_path = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )

    os.close(file_descriptor)

    try:
        torch.save(checkpoint, temporary_path)
        os.replace(temporary_path, path)
    except Exception:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)
        raise


def load_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """
    Restore model and optionally optimizer state.

    Only load checkpoints produced by a trusted training process.
    """
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(
        path,
        map_location=map_location,
        weights_only=False,
    )

    if not isinstance(checkpoint, dict):
        raise ValueError("Invalid checkpoint format.")

    if checkpoint.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint format version.")

    required_keys = {
        "step",
        "best_validation_loss",
        "model_state_dict",
        "optimizer_state_dict",
    }

    missing = required_keys.difference(checkpoint)

    if missing:
        raise ValueError(
            f"Checkpoint is missing required keys: {sorted(missing)}"
        )

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    if optimizer is not None:
        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

    return checkpoint
