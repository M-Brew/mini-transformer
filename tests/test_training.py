
import math

import pytest
import torch

from mini_transformer.config import ModelConfig, TrainingConfig
from mini_transformer.dataset import LanguageModelDataset
from mini_transformer.model import GPTLanguageModel
from mini_transformer.training import (
    calculate_learning_rate,
    create_optimizer,
    estimate_loss,
    train_model,
)


@pytest.fixture
def model():
    config = ModelConfig(
        vocab_size=16,
        block_size=8,
        n_layer=1,
        n_head=2,
        n_embd=16,
        dropout=0.0,
    )
    return GPTLanguageModel(config)


@pytest.fixture
def datasets():
    train_tokens = torch.randint(0, 16, (200,))
    validation_tokens = torch.randint(0, 16, (100,))

    return (
        LanguageModelDataset(train_tokens, block_size=8),
        LanguageModelDataset(validation_tokens, block_size=8),
    )


def test_learning_rate_warmup():
    config = TrainingConfig(
        learning_rate=0.001,
        min_learning_rate=0.0001,
        max_steps=100,
        warmup_steps=10,
    )

    assert calculate_learning_rate(0, config) == pytest.approx(0.0001)
    assert calculate_learning_rate(9, config) == pytest.approx(0.001)


def test_learning_rate_cosine_decay():
    config = TrainingConfig(
        learning_rate=0.001,
        min_learning_rate=0.0001,
        max_steps=100,
        warmup_steps=10,
    )

    start = calculate_learning_rate(10, config)
    middle = calculate_learning_rate(55, config)
    end = calculate_learning_rate(100, config)

    assert start == pytest.approx(0.001)
    assert middle < start
    assert end == pytest.approx(0.0001)


def test_learning_rate_without_warmup():
    config = TrainingConfig(
        learning_rate=0.001,
        min_learning_rate=0.0001,
        max_steps=100,
        warmup_steps=0,
    )

    assert calculate_learning_rate(0, config) == pytest.approx(0.001)
    assert calculate_learning_rate(100, config) == pytest.approx(0.0001)


def test_optimizer_has_decay_and_no_decay_groups(model):
    config = TrainingConfig()

    optimizer = create_optimizer(model, config)

    assert len(optimizer.param_groups) == 2
    assert optimizer.param_groups[0]["weight_decay"] == config.weight_decay
    assert optimizer.param_groups[1]["weight_decay"] == 0.0


def test_estimate_loss_returns_finite_values(model, datasets):
    train_dataset, validation_dataset = datasets

    result = estimate_loss(
        model,
        train_dataset,
        validation_dataset,
        batch_size=4,
        eval_batches=3,
        device=torch.device("cpu"),
    )

    assert set(result) == {"train", "validation"}
    assert math.isfinite(result["train"])
    assert math.isfinite(result["validation"])


def test_estimate_loss_restores_training_mode(model, datasets):
    train_dataset, validation_dataset = datasets

    model.train()

    estimate_loss(
        model,
        train_dataset,
        validation_dataset,
        batch_size=4,
        eval_batches=2,
        device=torch.device("cpu"),
    )

    assert model.training


def test_estimate_loss_restores_evaluation_mode(model, datasets):
    train_dataset, validation_dataset = datasets

    model.eval()

    estimate_loss(
        model,
        train_dataset,
        validation_dataset,
        batch_size=4,
        eval_batches=2,
        device=torch.device("cpu"),
    )

    assert not model.training


def test_training_updates_model_parameters(model, datasets, tmp_path):
    train_dataset, validation_dataset = datasets

    config = TrainingConfig(
        batch_size=4,
        learning_rate=0.001,
        weight_decay=0.01,
        max_steps=3,
        eval_interval=1,
        gradient_clip=1.0,
        warmup_steps=0,
        min_learning_rate=0.0001,
        eval_batches=2,
        checkpoint_interval=2,
    )

    before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }

    result = train_model(
        model,
        train_dataset,
        validation_dataset,
        config,
        device=torch.device("cpu"),
        checkpoint_dir=tmp_path / "checkpoints",
    )

    changed = any(
        not torch.equal(before[name], parameter.detach())
        for name, parameter in model.named_parameters()
    )

    assert changed
    assert result.completed_steps == 3
    assert result.best_checkpoint_path.exists()
    assert result.latest_checkpoint_path.exists()


def test_training_rejects_dataset_block_size_mismatch(model, datasets, tmp_path):
    train_dataset, _ = datasets

    wrong_validation = LanguageModelDataset(
        torch.randint(0, 16, (100,)),
        block_size=4,
    )

    with pytest.raises(ValueError):
        train_model(
            model,
            train_dataset,
            wrong_validation,
            TrainingConfig(max_steps=1),
            device=torch.device("cpu"),
            checkpoint_dir=tmp_path,
        )
