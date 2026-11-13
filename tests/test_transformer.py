
import pytest
import torch

from mini_transformer.transformer import FeedForward, TransformerBlock


def test_feed_forward_preserves_shape():
    network = FeedForward(n_embd=32, dropout=0.0)

    x = torch.randn(2, 8, 32)
    output = network(x)

    assert output.shape == x.shape


def test_feed_forward_supports_different_batch_and_sequence_sizes():
    network = FeedForward(n_embd=64)

    for batch_size, sequence_length in [(1, 1), (2, 8), (4, 16)]:
        x = torch.randn(batch_size, sequence_length, 64)
        output = network(x)

        assert output.shape == x.shape


def test_feed_forward_has_gradients():
    network = FeedForward(n_embd=32)

    x = torch.randn(2, 8, 32, requires_grad=True)
    output = network(x)

    output.mean().backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()

    first_linear = network.network[0]
    assert first_linear.weight.grad is not None


def test_transformer_block_preserves_shape():
    block = TransformerBlock(
        n_embd=32,
        n_head=4,
        block_size=16,
        dropout=0.0,
    )

    x = torch.randn(2, 8, 32)
    output = block(x)

    assert output.shape == x.shape


def test_transformer_block_has_gradients():
    block = TransformerBlock(
        n_embd=32,
        n_head=4,
        block_size=16,
        dropout=0.0,
    )

    x = torch.randn(2, 8, 32, requires_grad=True)
    output = block(x)

    output.square().mean().backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()

    assert block.attention.qkv_projection.weight.grad is not None
    assert block.feed_forward.network[0].weight.grad is not None


def test_transformer_block_rejects_invalid_sequence_length():
    block = TransformerBlock(
        n_embd=32,
        n_head=4,
        block_size=8,
    )

    x = torch.randn(2, 9, 32)

    with pytest.raises(ValueError):
        block(x)


def test_transformer_block_rejects_invalid_embedding_dimension():
    block = TransformerBlock(
        n_embd=32,
        n_head=4,
        block_size=8,
    )

    x = torch.randn(2, 8, 64)

    with pytest.raises(ValueError):
        block(x)
