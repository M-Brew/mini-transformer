
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int | None
    block_size: int
    n_layer: int
    n_head: int
    n_embd: int
    dropout: float

    def validate(self) -> None:
        if self.vocab_size is not None and self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")

        if self.block_size <= 0:
            raise ValueError("block_size must be positive")

        if self.n_layer <= 0:
            raise ValueError("n_layer must be positive")

        if self.n_head <= 0:
            raise ValueError("n_head must be positive")

        if self.n_embd <= 0:
            raise ValueError("n_embd must be positive")

        if self.n_embd % self.n_head != 0:
            raise ValueError(
                "n_embd must be divisible by n_head"
            )

        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int
    learning_rate: float
    weight_decay: float
    max_steps: int
    eval_interval: int
    gradient_clip: float

    def validate(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")

        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")

        if self.weight_decay < 0:
            raise ValueError("weight_decay cannot be negative")

        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")

        if self.eval_interval <= 0:
            raise ValueError("eval_interval must be positive")

        if self.gradient_clip < 0:
            raise ValueError("gradient_clip cannot be negative")


@dataclass(frozen=True)
class DataConfig:
    input_path: str
    train_path: str
    validation_path: str
    validation_fraction: float

    def validate(self) -> None:
        if not 0.0 < self.validation_fraction < 1.0:
            raise ValueError(
                "validation_fraction must be between 0 and 1"
            )


@dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens: int
    temperature: float
    top_k: int | None

    def validate(self) -> None:
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")

        if self.temperature <= 0:
            raise ValueError("temperature must be positive")

        if self.top_k is not None and self.top_k <= 0:
            raise ValueError("top_k must be positive or None")


@dataclass(frozen=True)
class AppConfig:
    seed: int
    model: ModelConfig
    training: TrainingConfig
    data: DataConfig
    generation: GenerationConfig


def _require_mapping(
    value: Any,
    name: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"'{name}' must be a YAML mapping")

    return value


def load_config(path: str | Path) -> AppConfig:
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file)

    root = _require_mapping(raw, "root")

    model_data = _require_mapping(root.get("model"), "model")
    training_data = _require_mapping(
        root.get("training"), "training"
    )
    data_data = _require_mapping(root.get("data"), "data")
    generation_data = _require_mapping(
        root.get("generation"), "generation"
    )

    config = AppConfig(
        seed=int(root["seed"]),
        model=ModelConfig(**model_data),
        training=TrainingConfig(**training_data),
        data=DataConfig(**data_data),
        generation=GenerationConfig(**generation_data),
    )

    config.model.validate()
    config.training.validate()
    config.data.validate()
    config.generation.validate()

    return config


def resolve_model_vocab_size(
    model_config: ModelConfig,
    vocab_size: int,
) -> ModelConfig:
    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive")

    if (
        model_config.vocab_size is not None
        and model_config.vocab_size != vocab_size
    ):
        raise ValueError(
            "Configured vocabulary size does not match tokenizer"
        )

    resolved = ModelConfig(
        vocab_size=vocab_size,
        block_size=model_config.block_size,
        n_layer=model_config.n_layer,
        n_head=model_config.n_head,
        n_embd=model_config.n_embd,
        dropout=model_config.dropout,
    )

    resolved.validate()

    return resolved
