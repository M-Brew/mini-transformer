
from __future__ import annotations

import torch
from torch import Tensor, nn


def sample_next_token(
    logits: Tensor,
    *,
    temperature: float = 1.0,
    top_k: int | None = None,
    do_sample: bool = True,
    generator: torch.Generator | None = None,
) -> Tensor:
    """
    Select the next token from vocabulary logits.

    Args:
        logits:     (B, V), scores for the next token.
        temperature: Positive sampling temperature.
        top_k:      Restrict sampling to the top k logits.
        do_sample:  False selects the highest-scoring token.
        generator:  Optional random generator for reproducibility.

    Returns:
        Token IDs with shape (B, 1).
    """
    if logits.ndim != 2:
        raise ValueError("logits must have shape (B, V).")

    if logits.size(0) == 0 or logits.size(1) == 0:
        raise ValueError("logits must have nonempty batch and vocabulary dimensions.")

    if temperature <= 0:
        raise ValueError("temperature must be positive.")

    if top_k is not None and not 1 <= top_k <= logits.size(-1):
        raise ValueError("top_k must be between 1 and the vocabulary size.")

    if not torch.isfinite(logits).all():
        raise ValueError("logits must contain only finite values.")

    if not do_sample:
        return torch.argmax(logits, dim=-1, keepdim=True)

    # Temperature controls the sharpness of the distribution.
    logits = logits / temperature

    if top_k is not None:
        # Find the kth largest logit in each batch item.
        top_values, _ = torch.topk(logits, top_k, dim=-1)

        # Every logit below the kth largest value is removed.
        threshold = top_values[:, -1].unsqueeze(-1)

        logits = logits.masked_fill(
            logits < threshold,
            float("-inf"),
        )

    probabilities = torch.softmax(logits, dim=-1)

    next_token = torch.multinomial(
        probabilities,
        num_samples=1,
        generator=generator,
    )

    return next_token


@torch.inference_mode()
def generate_tokens(
    model: nn.Module,
    input_ids: Tensor,
    *,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    do_sample: bool = True,
    stop_token_id: int | None = None,
    generator: torch.Generator | None = None,
) -> Tensor:
    """
    Autoregressively generate tokens from a prompt.

    Args:
        model:          GPTLanguageModel.
        input_ids:      Prompt IDs with shape (B, T).
        max_new_tokens: Maximum number of tokens to generate.
        temperature:    Sampling temperature.
        top_k:          Optional top-k sampling limit.
        do_sample:      Whether to sample or use greedy decoding.
        stop_token_id:  Optional token ID that terminates generation.
        generator:      Optional sampling generator.

    Returns:
        Complete sequence containing the original prompt and generated IDs.

    Notes:
        The model is recomputed over the available context at every step.
        Context is cropped to the model's block_size.
    """
    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape (B, T).")

    if input_ids.dtype != torch.long:
        raise ValueError("input_ids must have dtype torch.long.")

    if input_ids.size(0) == 0 or input_ids.size(1) == 0:
        raise ValueError("Prompt batch and sequence dimensions must be nonempty.")

    if max_new_tokens < 0:
        raise ValueError("max_new_tokens cannot be negative.")

    if temperature <= 0:
        raise ValueError("temperature must be positive.")

    if top_k is not None and top_k <= 0:
        raise ValueError("top_k must be positive.")

    if stop_token_id is not None:
        if not 0 <= stop_token_id < model.vocab_size:
            raise ValueError("stop_token_id is outside the model vocabulary.")

    if input_ids.min().item() < 0 or input_ids.max().item() >= model.vocab_size:
        raise ValueError("Prompt contains token IDs outside the model vocabulary.")

    if input_ids.device != next(model.parameters()).device:
        raise ValueError("input_ids and model must be on the same device.")

    # Work on a new tensor so the caller's prompt is not modified.
    generated = input_ids.clone()

    previous_mode = model.training
    model.eval()

    try:
        for _ in range(max_new_tokens):
            # The model supports a fixed context length.
            # Retain only the most recent block_size tokens.
            context = generated[:, -model.block_size :]

            logits, _ = model(context)

            # Only the final position predicts the next token.
            next_token_logits = logits[:, -1, :]

            next_token = sample_next_token(
                next_token_logits,
                temperature=temperature,
                top_k=top_k,
                do_sample=do_sample,
                generator=generator,
            )

            generated = torch.cat(
                (generated, next_token),
                dim=1,
            )

            if stop_token_id is not None:
                # Stop when every sequence in the batch has emitted
                # the stop token at this generation step.
                if torch.all(next_token == stop_token_id):
                    break

    finally:
        model.train(previous_mode)

    return generated
