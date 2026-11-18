
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from mini_transformer.checkpoint import load_checkpoint
from mini_transformer.config import (
    load_config,
    resolve_model_vocab_size,
)
from mini_transformer.generation import generate_tokens
from mini_transformer.model import GPTLanguageModel
from mini_transformer.tokenizer import CharacterTokenizer


def load_trained_generator(
    *,
    config_path: str | Path,
    tokenizer_path: str | Path,
    checkpoint_path: str | Path,
    device: torch.device,
) -> TextGenerator:
    """
    Construct a model, restore its trained weights and return
    a ready-to-use TextGenerator.
    """
    app_config = load_config(config_path)
    tokenizer = CharacterTokenizer.load(tokenizer_path)

    model_config = resolve_model_vocab_size(
        app_config.model,
        tokenizer.vocab_size,
    )

    model = GPTLanguageModel(model_config)

    load_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=None,
        map_location="cpu",
    )

    model.to(device)
    model.eval()

    return TextGenerator(
        model=model,
        tokenizer=tokenizer,
        device=device,
    )


class TextGenerator:
    """
    Text-oriented interface around a trained language model.

    This class handles tokenization and decoding. It does not own
    model construction or checkpoint loading.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: CharacterTokenizer,
        device: torch.device,
    ) -> None:
        if model.vocab_size != tokenizer.vocab_size:
            raise ValueError(
                "Model and tokenizer vocabulary sizes do not match."
            )

        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        self.model.to(device)
        self.model.eval()

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int | None = 50,
        do_sample: bool = True,
        seed: int | None = None,
    ) -> str:
        if not prompt:
            raise ValueError("Prompt cannot be empty.")

        if max_new_tokens < 0:
            raise ValueError("max_new_tokens cannot be negative.")

        if temperature <= 0:
            raise ValueError("temperature must be positive.")

        # A top-k larger than the vocabulary is equivalent to
        # sampling from the entire vocabulary.
        effective_top_k = (
            min(top_k, self.tokenizer.vocab_size)
            if top_k is not None
            else None
        )

        prompt_ids = self.tokenizer.encode(prompt)

        input_ids = torch.tensor(
            [prompt_ids],
            dtype=torch.long,
            device=self.device,
        )

        generator = None

        if seed is not None:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(seed)

        output_ids = generate_tokens(
            self.model,
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=effective_top_k,
            do_sample=do_sample,
            generator=generator,
        )

        return self.tokenizer.decode(output_ids[0].tolist())
