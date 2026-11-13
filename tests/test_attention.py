
import pytest
import torch

from mini_transformer.attention import (
    MultiHeadCausalSelfAttention,
    scaled_dot_product_attention,
)


def test_scaled_attention_output_shape():
    batch_size = 2
    n_head = 4
    sequence_length = 8
    head_dim = 16

    query = torch.randn(batch_size, n_head, sequence_length, head_dim)
    key = torch.randn_like(query)
    value = torch.randn_like(query)

    output, weights = scaled_dot_product_attention(
        query,
        key,
        value,
    )

    assert output.shape == query.shape
    assert weights.shape == (
        batch_size,
        n_head,
        sequence_length,
        sequence_length,
    )


def test_causal_attention_blocks_future_positions():
    sequence_length = 5
    head_dim = 8

    query = torch.randn(1, 1, sequence_length, head_dim)
    key = torch.randn_like(query)
    value = torch.randn_like(query)

    _, weights = scaled_dot_product_attention(
        query,
        key,
        value,
        causal=True,
    )

    # No position can attend to a future position.
    for position in range(sequence_length):
        assert torch.all(weights[..., position, position + 1 :] == 0)


def test_attention_weights_sum_to_one():
    query = torch.randn(2, 3, 6, 8)
    key = torch.randn_like(query)
    value = torch.randn_like(query)

    _, weights = scaled_dot_product_attention(
        query,
        key,
        value,
    )

    sums = weights.sum(dim=-1)

    assert torch.allclose(
        sums,
        torch.ones_like(sums),
        atol=1e-6,
    )


def test_non_causal_attention_can_use_future_positions():
    query = torch.randn(1, 1, 4, 8)
    key = torch.randn_like(query)
    value = torch.randn_like(query)

    _, weights = scaled_dot_product_attention(
        query,
        key,
        value,
        causal=False,
    )

    # With no mask, the first position can attend to later positions.
    assert torch.all(weights[0, 0, 0, 1:] > 0)


def test_multi_head_attention_output_shape():
    attention = MultiHeadCausalSelfAttention(
        n_embd=32,
        n_head=4,
        block_size=16,
        dropout=0.0,
    )

    x = torch.randn(2, 10, 32)
    output = attention(x)

    assert output.shape == x.shape


def test_multi_head_attention_is_causal():
    torch.manual_seed(42)

    attention = MultiHeadCausalSelfAttention(
        n_embd=32,
        n_head=4,
        block_size=16,
        dropout=0.0,
    )

    attention.eval()

    original = torch.randn(1, 8, 32)
    modified = original.clone()

    # Change tokens at positions 5, 6 and 7.
    modified[:, 5:, :] = torch.randn_like(modified[:, 5:, :])

    original_output = attention(original)
    modified_output = attention(modified)

    # Positions 0 through 4 must not be affected.
    assert torch.allclose(
        original_output[:, :5, :],
        modified_output[:, :5, :],
        atol=1e-6,
    )


def test_multi_head_attention_has_gradients():
    attention = MultiHeadCausalSelfAttention(
        n_embd=32,
        n_head=4,
        block_size=16,
    )

    x = torch.randn(2, 8, 32, requires_grad=True)

    output = attention(x)
    loss = output.square().mean()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()

    assert attention.qkv_projection.weight.grad is not None
    assert attention.output_projection.weight.grad is not None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_embd": 30, "n_head": 4, "block_size": 16},
        {"n_embd": 32, "n_head": 0, "block_size": 16},
        {"n_embd": 32, "n_head": 4, "block_size": 0},
        {"n_embd": 32, "n_head": 4, "block_size": 16, "dropout": 1.0},
    ],
)
def test_invalid_attention_configuration(kwargs):
    with pytest.raises(ValueError):
        MultiHeadCausalSelfAttention(**kwargs)


def test_attention_rejects_sequences_longer_than_block_size():
    attention = MultiHeadCausalSelfAttention(
        n_embd=32,
        n_head=4,
        block_size=8,
    )

    x = torch.randn(2, 9, 32)

    with pytest.raises(ValueError):
        attention(x)


def test_attention_rejects_wrong_embedding_dimension():
    attention = MultiHeadCausalSelfAttention(
        n_embd=32,
        n_head=4,
        block_size=8,
    )

    x = torch.randn(2, 8, 64)

    with pytest.raises(ValueError):
        attention(x)
