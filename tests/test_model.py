
import pytest
import torch

from mini_transformer.config import ModelConfig
from mini_transformer.model import GPTLanguageModel


@pytest.fixture
def model_config():
    return ModelConfig(
        vocab_size=32,
        block_size=16,
        n_layer=2,
        n_head=4,
        n_embd=32,
        dropout=0.0,
    )


@pytest.fixture
def model(model_config):
    return GPTLanguageModel(model_config)


def test_model_output_shape(model):
    idx = torch.randint(0, model.vocab_size, (2, 8))

    logits, loss = model(idx)

    assert logits.shape == (2, 8, model.vocab_size)
    assert loss is None


def test_model_computes_loss(model):
    idx = torch.randint(0, model.vocab_size, (2, 8))
    targets = torch.randint(0, model.vocab_size, (2, 8))

    logits, loss = model(idx, targets)

    assert logits.shape == (2, 8, model.vocab_size)
    assert loss is not None
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_model_loss_backward(model):
    idx = torch.randint(0, model.vocab_size, (2, 8))
    targets = torch.randint(0, model.vocab_size, (2, 8))

    _, loss = model(idx, targets)

    assert loss is not None

    loss.backward()

    assert model.token_embedding.weight.grad is not None
    assert model.language_model_head.weight.grad is not None

    assert torch.isfinite(model.token_embedding.weight.grad).all()


def test_model_is_causal(model):
    torch.manual_seed(42)
    model.eval()

    original = torch.randint(0, model.vocab_size, (1, 8))
    modified = original.clone()

    # Change future tokens starting at position 5.
    modified[:, 5:] = torch.randint(
        0,
        model.vocab_size,
        (1, 3),
    )

    original_logits, _ = model(original)
    modified_logits, _ = model(modified)

    # Earlier predictions must remain unchanged.
    assert torch.allclose(
        original_logits[:, :5, :],
        modified_logits[:, :5, :],
        atol=1e-6,
    )


def test_model_parameter_count_is_positive(model):
    assert model.count_parameters() > 0


def test_model_parameter_count_matches_parameter_tensors(model):
    expected = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    assert model.count_parameters() == expected


def test_model_rejects_sequence_longer_than_block_size(model):
    idx = torch.randint(0, model.vocab_size, (2, 17))

    with pytest.raises(ValueError):
        model(idx)


def test_model_rejects_wrong_input_dimensions(model):
    idx = torch.randint(0, model.vocab_size, (2, 8, 1))

    with pytest.raises(ValueError):
        model(idx)


def test_model_rejects_wrong_input_dtype(model):
    idx = torch.randint(
        0,
        model.vocab_size,
        (2, 8),
    ).float()

    with pytest.raises(ValueError):
        model(idx)


def test_model_rejects_out_of_range_input_token(model):
    idx = torch.randint(0, model.vocab_size, (2, 8))
    idx[0, 0] = model.vocab_size

    with pytest.raises(ValueError):
        model(idx)


def test_model_rejects_mismatched_target_shape(model):
    idx = torch.randint(0, model.vocab_size, (2, 8))
    targets = torch.randint(0, model.vocab_size, (2, 7))

    with pytest.raises(ValueError):
        model(idx, targets)


def test_model_rejects_out_of_range_target(model):
    idx = torch.randint(0, model.vocab_size, (2, 8))
    targets = torch.randint(0, model.vocab_size, (2, 8))
    targets[0, 0] = model.vocab_size

    with pytest.raises(ValueError):
        model(idx, targets)


def test_model_rejects_unresolved_vocabulary_size():
    config = ModelConfig(
        vocab_size=None,
        block_size=16,
        n_layer=2,
        n_head=4,
        n_embd=32,
        dropout=0.0,
    )

    with pytest.raises(ValueError):
        GPTLanguageModel(config)


def test_model_supports_mps_when_available(model):
    if not torch.backends.mps.is_available():
        pytest.skip("MPS is not available.")

    device = torch.device("mps")
    model = model.to(device)

    idx = torch.randint(
        0,
        model.vocab_size,
        (2, 8),
        device=device,
    )

    logits, loss = model(idx)

    assert logits.device.type == "mps"
    assert logits.shape == (2, 8, model.vocab_size)
    assert loss is None
