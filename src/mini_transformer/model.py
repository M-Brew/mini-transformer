
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from mini_transformer.config import ModelConfig
from mini_transformer.transformer import TransformerBlock


class GPTLanguageModel(nn.Module):
    """
    Decoder-only GPT-style language model.

    Input:
        idx:     (B, T) token IDs
        targets: (B, T) optional next-token targets

    Returns:
        logits: (B, T, V)
        loss:   scalar tensor, or None if targets are omitted
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()

        if config.vocab_size is None or config.vocab_size <= 0:
            raise ValueError(
                "config.vocab_size must be resolved before model creation."
            )

        self.vocab_size = config.vocab_size
        self.block_size = config.block_size
        self.n_embd = config.n_embd
        self.n_layer = config.n_layer
        self.n_head = config.n_head

        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.n_embd,
        )

        self.position_embedding = nn.Embedding(
            config.block_size,
            config.n_embd,
        )

        self.embedding_dropout = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    n_embd=config.n_embd,
                    n_head=config.n_head,
                    block_size=config.block_size,
                    dropout=config.dropout,
                )
                for _ in range(config.n_layer)
            ]
        )

        self.final_layer_norm = nn.LayerNorm(config.n_embd)

        self.language_model_head = nn.Linear(
            config.n_embd,
            config.vocab_size,
            bias=False,
        )

        self.apply(self._initialize_weights)

    @staticmethod
    def _initialize_weights(module: nn.Module) -> None:
        """
        GPT-style initialization for linear and embedding weights.
        """
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self,
        idx: Tensor,
        targets: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:

        if idx.ndim != 2:
            raise ValueError("idx must have shape (B, T).")

        if idx.dtype != torch.long:
            raise ValueError("idx must have dtype torch.long.")

        batch_size, sequence_length = idx.shape

        if batch_size == 0 or sequence_length == 0:
            raise ValueError("Input batch and sequence dimensions must be nonzero.")

        if sequence_length > self.block_size:
            raise ValueError(
                f"Input sequence length {sequence_length} exceeds "
                f"block_size {self.block_size}."
            )

        if torch.any(idx < 0) or torch.any(idx >= self.vocab_size):
            raise ValueError("Input contains token IDs outside the vocabulary.")

        if targets is not None:
            if targets.shape != idx.shape:
                raise ValueError("targets must have the same shape as idx.")

            if targets.dtype != torch.long:
                raise ValueError("targets must have dtype torch.long.")

            if torch.any(targets < 0) or torch.any(
                targets >= self.vocab_size
            ):
                raise ValueError(
                    "Targets contain token IDs outside the vocabulary."
                )

        # Position IDs: 0, 1, ..., T-1.
        positions = torch.arange(
            sequence_length,
            device=idx.device,
        )

        # Token embeddings: (B, T, C)
        token_embeddings = self.token_embedding(idx)

        # Positional embeddings: (T, C)
        position_embeddings = self.position_embedding(positions)

        # Broadcasting adds positional information to each batch item.
        x = token_embeddings + position_embeddings

        x = self.embedding_dropout(x)

        # Pass representations through the Transformer stack.
        for block in self.blocks:
            x = block(x)

        x = self.final_layer_norm(x)

        # Convert hidden representations into vocabulary logits.
        logits = self.language_model_head(x)

        loss = None

        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(batch_size * sequence_length, self.vocab_size),
                targets.reshape(batch_size * sequence_length),
            )

        return logits, loss

    def count_parameters(self, trainable_only: bool = True) -> int:
        """
        Count model parameters.

        Buffers, such as the causal attention masks, are excluded.
        """
        parameters = self.parameters()

        if trainable_only:
            parameters = (
                parameter
                for parameter in parameters
                if parameter.requires_grad
            )

        return sum(parameter.numel() for parameter in parameters)
