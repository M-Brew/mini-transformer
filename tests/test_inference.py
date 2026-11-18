
import pytest
import torch

from mini_transformer.config import ModelConfig
from mini_transformer.inference import TextGenerator
from mini_transformer.model import GPTLanguageModel
from mini_transformer.tokenizer import CharacterTokenizer


@pytest.fixture
def tokenizer():
    return CharacterTokenizer.train("the quick brown fox")


@pytest.fixture
def model(tokenizer):
    config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=32,
        n_layer=1,
        n_head=2,
        n_embd=16,
        dropout=0.0,
    )

    return GPTLanguageModel(config)


@pytest.fixture
def generator(model, tokenizer):
    return TextGenerator(
        model=model,
        tokenizer=tokenizer,
        device=torch.device("cpu"),
    )


def test_text_generator_returns_string(generator):
    output = generator.generate(
        "the",
        max_new_tokens=5,
        do_sample=False,
    )

    assert isinstance(output, str)
    assert output.startswith("the")


def test_text_generator_preserves_prompt(generator):
    prompt = "the quick"

    output = generator.generate(
        prompt,
        max_new_tokens=5,
        do_sample=False,
    )

    assert output.startswith(prompt)


def test_text_generator_supports_zero_new_tokens(generator):
    prompt = "the quick"

    output = generator.generate(
        prompt,
        max_new_tokens=0,
    )

    assert output == prompt


def test_text_generator_supports_seeded_sampling(generator):
    first = generator.generate(
        "the",
        max_new_tokens=10,
        seed=123,
    )

    second = generator.generate(
        "the",
        max_new_tokens=10,
        seed=123,
    )

    assert first == second


def test_text_generator_clamps_top_k_to_vocabulary(generator):
    output = generator.generate(
        "the",
        max_new_tokens=5,
        top_k=10000,
        seed=42,
    )

    assert output.startswith("the")


def test_text_generator_rejects_empty_prompt(generator):
    with pytest.raises(ValueError):
        generator.generate("")


def test_text_generator_rejects_negative_token_count(generator):
    with pytest.raises(ValueError):
        generator.generate(
            "the",
            max_new_tokens=-1,
        )


def test_text_generator_rejects_invalid_temperature(generator):
    with pytest.raises(ValueError):
        generator.generate(
            "the",
            temperature=0,
        )


def test_text_generator_rejects_vocab_mismatch(tokenizer):
    model = GPTLanguageModel(
        ModelConfig(
            vocab_size=tokenizer.vocab_size + 1,
            block_size=32,
            n_layer=1,
            n_head=2,
            n_embd=16,
            dropout=0.0,
        )
    )

    with pytest.raises(ValueError):
        TextGenerator(
            model=model,
            tokenizer=tokenizer,
            device=torch.device("cpu"),
        )
