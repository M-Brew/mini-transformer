
from pathlib import Path

import pytest

from mini_transformer.config import (
    ModelConfig,
    load_config,
    resolve_model_vocab_size,
)


def test_load_baseline_config():
    config = load_config("configs/baseline.yaml")

    assert config.seed == 42
    assert config.model.block_size == 256
    assert config.model.vocab_size is None
    assert config.training.batch_size == 16


def test_model_config_rejects_invalid_head_dimension():
    config = ModelConfig(
        vocab_size=100,
        block_size=128,
        n_layer=4,
        n_head=3,
        n_embd=128,
        dropout=0.1,
    )

    with pytest.raises(ValueError):
        config.validate()


def test_model_config_accepts_valid_dimensions():
    config = ModelConfig(
        vocab_size=None,
        block_size=128,
        n_layer=4,
        n_head=4,
        n_embd=128,
        dropout=0.1,
    )

    config.validate()


def test_resolve_model_vocab_size():
    config = ModelConfig(
        vocab_size=None,
        block_size=128,
        n_layer=4,
        n_head=4,
        n_embd=128,
        dropout=0.1,
    )

    resolved = resolve_model_vocab_size(config, 75)

    assert resolved.vocab_size == 75
    assert resolved.block_size == 128


def test_resolve_rejects_mismatched_vocab_size():
    config = ModelConfig(
        vocab_size=100,
        block_size=128,
        n_layer=4,
        n_head=4,
        n_embd=128,
        dropout=0.1,
    )

    with pytest.raises(ValueError):
        resolve_model_vocab_size(config, 75)


def test_missing_config_file():
    with pytest.raises(FileNotFoundError):
        load_config(Path("missing.yaml"))
