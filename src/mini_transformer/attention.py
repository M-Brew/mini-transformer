
from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def scaled_dot_product_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    causal: bool = True,
    dropout: nn.Dropout | None = None,
) -> tuple[Tensor, Tensor]:
    """
    Compute scaled dot-product attention.

    Expected shapes:
        query: (B, H, T, D)
        key:   (B, H, T, D)
        value: (B, H, T, D)

    Returns:
        output:  (B, H, T, D)
        weights: (B, H, T, T)
    """
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("query, key and value must all be 4D tensors.")

    if query.shape != key.shape or key.shape != value.shape:
        raise ValueError("query, key and value must have identical shapes.")

    if query.size(-1) == 0:
        raise ValueError("The head dimension must be positive.")

    head_dim = query.size(-1)

    # Each query is compared against every key.
    # Shape: (B, H, T, T)
    scores = query @ key.transpose(-2, -1)
    scores = scores / math.sqrt(head_dim)

    if causal:
        sequence_length = query.size(-2)

        # True means the position is allowed to attend.
        # The lower triangle includes the current position.
        mask = torch.tril(
            torch.ones(
                sequence_length,
                sequence_length,
                device=query.device,
                dtype=torch.bool,
            )
        )

        # Broadcast mask across batch and heads.
        scores = scores.masked_fill(~mask, float("-inf"))

    # Normalize over the key positions.
    weights = torch.softmax(scores, dim=-1)

    if dropout is not None:
        weights = dropout(weights)

    # Weighted sum of values.
    output = weights @ value

    return output, weights


class MultiHeadCausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention.

    Input:
        x: (B, T, C)

    Output:
        y: (B, T, C)
    """

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        block_size: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if n_embd <= 0:
            raise ValueError("n_embd must be positive.")

        if n_head <= 0:
            raise ValueError("n_head must be positive.")

        if n_embd % n_head != 0:
            raise ValueError("n_embd must be divisible by n_head.")

        if block_size <= 0:
            raise ValueError("block_size must be positive.")

        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the range [0, 1).")

        self.n_embd = n_embd
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.block_size = block_size

        # One projection produces Q, K and V simultaneously.
        self.qkv_projection = nn.Linear(n_embd, 3 * n_embd)

        self.output_projection = nn.Linear(n_embd, n_embd)

        self.attention_dropout = nn.Dropout(dropout)
        self.residual_dropout = nn.Dropout(dropout)

        # Store the causal mask as a buffer, not a trainable parameter.
        # persistent=False means it does not need to be saved in the
        # model state_dict because it can be reconstructed.
        causal_mask = torch.tril(
            torch.ones(block_size, block_size, dtype=torch.bool)
        )

        self.register_buffer(
            "causal_mask",
            causal_mask,
            persistent=False,
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError("Input must have shape (B, T, C).")

        batch_size, sequence_length, channels = x.shape

        if channels != self.n_embd:
            raise ValueError(
                f"Expected {self.n_embd} channels, received {channels}."
            )

        if sequence_length > self.block_size:
            raise ValueError(
                f"Sequence length {sequence_length} exceeds "
                f"block_size {self.block_size}."
            )

        # (B, T, C) -> (B, T, 3C)
        qkv = self.qkv_projection(x)

        # (B, T, 3C) -> (B, T, 3, H, D)
        qkv = qkv.view(
            batch_size,
            sequence_length,
            3,
            self.n_head,
            self.head_dim,
        )

        # Separate Q, K and V:
        # each becomes (B, T, H, D).
        query, key, value = qkv.unbind(dim=2)

        # Attention convention:
        # (B, T, H, D) -> (B, H, T, D)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        # Calculate attention scores.
        scores = query @ key.transpose(-2, -1)
        scores = scores / math.sqrt(self.head_dim)

        # Use the precomputed mask for the current sequence length.
        mask = self.causal_mask[:sequence_length, :sequence_length]

        scores = scores.masked_fill(~mask, float("-inf"))

        weights = torch.softmax(scores, dim=-1)
        weights = self.attention_dropout(weights)

        # (B, H, T, T) @ (B, H, T, D)
        # -> (B, H, T, D)
        output = weights @ value

        # Restore original layout:
        # (B, H, T, D) -> (B, T, H, D)
        output = output.transpose(1, 2).contiguous()

        # Concatenate the heads:
        # (B, T, H, D) -> (B, T, C)
        output = output.view(batch_size, sequence_length, self.n_embd)

        output = self.output_projection(output)
        output = self.residual_dropout(output)

        return output
