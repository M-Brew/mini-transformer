
import pytest
import torch

from mini_transformer.checkpoint import (
    load_checkpoint,
    save_checkpoint,
)
from mini_transformer.config import ModelConfig, TrainingConfig
from mini_transformer.model import GPTLanguageModel
from mini_transformer.training import create_optimizer


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


def test_checkpoint_saves_and_restores_model(model, tmp_path):
    optimizer = create_optimizer(model, TrainingConfig())

    path = tmp_path / "checkpoint.pt"

    save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        step=42,
        best_validation_loss=1.25,
    )

    original_parameters = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(1.0)

    checkpoint = load_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        map_location="cpu",
    )

    assert checkpoint["step"] == 42
    assert checkpoint["best_validation_loss"] == pytest.approx(1.25)

    for name, parameter in model.named_parameters():
        assert torch.equal(
            parameter.detach(),
            original_parameters[name],
        )


def test_checkpoint_restores_optimizer_state(model, tmp_path):
    optimizer = create_optimizer(model, TrainingConfig())

    x = torch.randint(0, 16, (2, 8))
    targets = torch.randint(0, 16, (2, 8))

    _, loss = model(x, targets)
    assert loss is not None

    loss.backward()
    optimizer.step()

    path = tmp_path / "checkpoint.pt"

    save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        step=1,
        best_validation_loss=2.0,
    )

    expected_state = optimizer.state_dict()

    restored_model = GPTLanguageModel(
        ModelConfig(
            vocab_size=16,
            block_size=8,
            n_layer=1,
            n_head=2,
            n_embd=16,
            dropout=0.0,
        )
    )

    restored_optimizer = create_optimizer(
        restored_model,
        TrainingConfig(),
    )

    load_checkpoint(
        path,
        model=restored_model,
        optimizer=restored_optimizer,
    )

    restored_state = restored_optimizer.state_dict()

    assert len(restored_state["state"]) == len(expected_state["state"])


def test_checkpoint_rejects_missing_file(model):
    optimizer = create_optimizer(model, TrainingConfig())

    with pytest.raises(FileNotFoundError):
        load_checkpoint(
            "missing-checkpoint.pt",
            model=model,
            optimizer=optimizer,
        )


def test_checkpoint_rejects_invalid_format(model, tmp_path):
    optimizer = create_optimizer(model, TrainingConfig())

    path = tmp_path / "invalid.pt"

    torch.save(
        {"format_version": 999},
        path,
    )

    with pytest.raises(ValueError):
        load_checkpoint(
            path,
            model=model,
            optimizer=optimizer,
        )
