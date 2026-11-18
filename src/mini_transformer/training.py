
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from mini_transformer.checkpoint import (
    load_checkpoint,
    save_checkpoint,
)
from mini_transformer.config import TrainingConfig
from mini_transformer.dataset import LanguageModelDataset


@dataclass
class TrainingResult:
    completed_steps: int
    best_validation_loss: float
    final_train_loss: float
    final_validation_loss: float
    best_checkpoint_path: Path
    latest_checkpoint_path: Path


def calculate_learning_rate(
    step: int,
    config: TrainingConfig,
) -> float:
    """
    Calculate warmup + cosine-decay learning rate.

    step is zero-based.
    """
    if config.warmup_steps > 0 and step < config.warmup_steps:
        return config.learning_rate * (step + 1) / config.warmup_steps

    decay_steps = max(1, config.max_steps - config.warmup_steps)

    progress = (
        step - config.warmup_steps
    ) / decay_steps

    progress = min(max(progress, 0.0), 1.0)

    cosine_factor = 0.5 * (
        1.0 + math.cos(math.pi * progress)
    )

    return (
        config.min_learning_rate
        + cosine_factor
        * (config.learning_rate - config.min_learning_rate)
    )


def create_optimizer(
    model: nn.Module,
    config: TrainingConfig,
) -> torch.optim.Optimizer:
    """
    Create AdamW with separate decay and no-decay parameter groups.

    Biases and LayerNorm parameters are excluded from weight decay.
    """
    decay_parameters = []
    no_decay_parameters = []

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue

        if parameter.ndim >= 2 and not name.endswith("bias"):
            decay_parameters.append(parameter)
        else:
            no_decay_parameters.append(parameter)

    optimizer = torch.optim.AdamW(
        [
            {
                "params": decay_parameters,
                "weight_decay": config.weight_decay,
            },
            {
                "params": no_decay_parameters,
                "weight_decay": 0.0,
            },
        ],
        lr=config.learning_rate,
        betas=(0.9, 0.95),
    )

    return optimizer


@torch.inference_mode()
def estimate_loss(
    model: nn.Module,
    train_dataset: LanguageModelDataset,
    validation_dataset: LanguageModelDataset,
    *,
    batch_size: int,
    eval_batches: int,
    device: torch.device,
    seed: int = 1234,
) -> dict[str, float]:
    """
    Estimate train and validation losses using deterministic batches.

    This function does not update model parameters.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    if eval_batches <= 0:
        raise ValueError("eval_batches must be positive.")

    previous_mode = model.training
    model.eval()

    results = {}

    try:
        for split_name, dataset in (
            ("train", train_dataset),
            ("validation", validation_dataset),
        ):
            generator = torch.Generator(device="cpu")
            generator.manual_seed(seed)

            losses = []

            for _ in range(eval_batches):
                x, y = dataset.get_batch(
                    batch_size=batch_size,
                    device=device,
                    generator=generator,
                )

                _, loss = model(x, y)

                if loss is None:
                    raise RuntimeError(
                        "Model did not return a loss for labeled input."
                    )

                losses.append(loss.item())

            results[split_name] = sum(losses) / len(losses)

    finally:
        model.train(previous_mode)

    return results


def train_model(
    model: nn.Module,
    train_dataset: LanguageModelDataset,
    validation_dataset: LanguageModelDataset,
    config: TrainingConfig,
    *,
    device: torch.device,
    checkpoint_dir: str | Path,
    resume_from: str | Path | None = None,
    seed: int = 42,
    metrics_callback=None,
) -> TrainingResult:
    """
    Train a GPT language model.

    Checkpoints:
        latest.pt: latest recoverable training state
        best.pt:   state with the lowest validation loss
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    latest_path = checkpoint_dir / "latest.pt"
    best_path = checkpoint_dir / "best.pt"

    if train_dataset.block_size != model.block_size:
        raise ValueError("Training dataset block_size does not match model.")

    if validation_dataset.block_size != model.block_size:
        raise ValueError("Validation dataset block_size does not match model.")

    if len(train_dataset) == 0 or len(validation_dataset) == 0:
        raise ValueError("Training and validation datasets must be nonempty.")

    model = model.to(device)
    optimizer = create_optimizer(model, config)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    start_step = 0
    best_validation_loss = float("inf")

    if resume_from is not None:
        checkpoint = load_checkpoint(
            resume_from,
            model=model,
            optimizer=optimizer,
            map_location=device,
        )

        start_step = checkpoint["step"]
        best_validation_loss = checkpoint["best_validation_loss"]

        if "batch_generator_state" in checkpoint:
            generator.set_state(checkpoint["batch_generator_state"])

        if "torch_rng_state" in checkpoint:
            torch.set_rng_state(checkpoint["torch_rng_state"].cpu())

        print(f"Resuming from step {start_step}")

    if start_step >= config.max_steps:
        raise ValueError(
            f"Checkpoint step {start_step} is already at or beyond "
            f"max_steps {config.max_steps}."
        )

    final_train_loss = float("nan")
    final_validation_loss = float("nan")

    for step in range(start_step, config.max_steps):
        model.train()

        learning_rate = calculate_learning_rate(step, config)

        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = learning_rate

        x, y = train_dataset.get_batch(
            batch_size=config.batch_size,
            device=device,
            generator=generator,
        )

        optimizer.zero_grad(set_to_none=True)

        _, loss = model(x, y)

        if loss is None:
            raise RuntimeError("Training forward pass did not return a loss.")

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite training loss at step {step + 1}: "
                f"{loss.item()}"
            )

        loss.backward()

        if config.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=config.gradient_clip,
            )

        optimizer.step()

        final_train_loss = loss.item()
        completed_step = step + 1

        should_evaluate = (
            completed_step % config.eval_interval == 0
            or completed_step == config.max_steps
        )

        if should_evaluate:
            evaluation = estimate_loss(
                model,
                train_dataset,
                validation_dataset,
                batch_size=config.batch_size,
                eval_batches=config.eval_batches,
                device=device,
                seed=seed + completed_step,
            )

            final_train_loss = evaluation["train"]
            final_validation_loss = evaluation["validation"]

            metrics = {
                "step": completed_step,
                "learning_rate": learning_rate,
                "train_loss": final_train_loss,
                "validation_loss": final_validation_loss,
            }

            if metrics_callback is not None:
                metrics_callback(metrics)

            print(
                f"step {completed_step:>6} | "
                f"lr {learning_rate:.7f} | "
                f"train loss {final_train_loss:.4f} | "
                f"validation loss {final_validation_loss:.4f}"
            )

            if final_validation_loss < best_validation_loss:
                best_validation_loss = final_validation_loss

                save_checkpoint(
                    best_path,
                    model=model,
                    optimizer=optimizer,
                    step=completed_step,
                    best_validation_loss=best_validation_loss,
                    batch_generator_state=generator.get_state(),
                )

        should_checkpoint = (
            completed_step % config.checkpoint_interval == 0
            or completed_step == config.max_steps
        )

        if should_checkpoint:
            save_checkpoint(
                latest_path,
                model=model,
                optimizer=optimizer,
                step=completed_step,
                best_validation_loss=best_validation_loss,
                batch_generator_state=generator.get_state(),
            )

    # Ensure the latest checkpoint exists even if intervals do not align.
    if not latest_path.exists():
        save_checkpoint(
            latest_path,
            model=model,
            optimizer=optimizer,
            step=config.max_steps,
            best_validation_loss=best_validation_loss,
            batch_generator_state=generator.get_state(),
        )

    if not best_path.exists():
        raise RuntimeError("Training finished without producing a best checkpoint.")

    return TrainingResult(
        completed_steps=config.max_steps,
        best_validation_loss=best_validation_loss,
        final_train_loss=final_train_loss,
        final_validation_loss=final_validation_loss,
        best_checkpoint_path=best_path,
        latest_checkpoint_path=latest_path,
    )
