
from __future__ import annotations

from torch import Tensor, nn

from mini_transformer.attention import MultiHeadCausalSelfAttention


class FeedForward(nn.Module):
    """
    Position-wise feed-forward network.

    Input:  (B, T, C)
    Output: (B, T, C)
    """

    def __init__(
        self,
        n_embd: int,
        dropout: float = 0.0,
        expansion_factor: int = 4,
    ) -> None:
        super().__init__()

        if n_embd <= 0:
            raise ValueError("n_embd must be positive.")

        if expansion_factor <= 0:
            raise ValueError("expansion_factor must be positive.")

        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the range [0, 1).")

        hidden_dim = expansion_factor * n_embd

        self.network = nn.Sequential(
            nn.Linear(n_embd, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.network(x)


class TransformerBlock(nn.Module):
    """
    Pre-LayerNorm decoder Transformer block.

    Structure:
        x = x + attention(ln1(x))
        x = x + mlp(ln2(x))
    """

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        block_size: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.n_embd = n_embd

        self.ln1 = nn.LayerNorm(n_embd)

        self.attention = MultiHeadCausalSelfAttention(
            n_embd=n_embd,
            n_head=n_head,
            block_size=block_size,
            dropout=dropout,
        )

        self.ln2 = nn.LayerNorm(n_embd)

        self.feed_forward = FeedForward(
            n_embd=n_embd,
            dropout=dropout,
        )

    def forward(self, x: Tensor) -> Tensor:
        # Validate before ln1: LayerNorm would otherwise reject a wrong
        # embedding dimension with a RuntimeError before attention's own
        # checks run. Sequence length is still validated by attention,
        # since LayerNorm only inspects the last dimension.
        if x.ndim != 3:
            raise ValueError("Input must have shape (B, T, C).")

        if x.size(-1) != self.n_embd:
            raise ValueError(
                f"Expected {self.n_embd} channels, received {x.size(-1)}."
            )

        # Attention sublayer with residual connection.
        x = x + self.attention(self.ln1(x))

        # Feed-forward sublayer with residual connection.
        x = x + self.feed_forward(self.ln2(x))

        return x
