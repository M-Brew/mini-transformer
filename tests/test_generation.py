
import pytest
import torch

from mini_transformer.config import ModelConfig
from mini_transformer.generation import (
    generate_tokens,
    sample_next_token,
)
from mini_transformer.model import GPTLanguageModel


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


def test_greedy_sampling_selects_highest_logit():
    logits = torch.tensor([[1.0, 5.0, 2.0, 3.0]])

    token = sample_next_token(
        logits,
        do_sample=False,
    )

    assert token.shape == (1, 1)
    assert token.item() == 1


def test_top_k_one_is_greedy():
    logits = torch.tensor([[1.0, 5.0, 2.0, 3.0]])

    generator = torch.Generator().manual_seed(42)

    token = sample_next_token(
        logits,
        top_k=1,
        do_sample=True,
        generator=generator,
    )

    assert token.item() == 1


def test_sampling_respects_top_k():
    logits = torch.tensor([[1.0, 5.0, 2.0, 3.0]])

    generator = torch.Generator().manual_seed(42)

    allowed = {1, 3}

    for _ in range(100):
        token = sample_next_token(
            logits,
            top_k=2,
            do_sample=True,
            generator=generator,
        )

        assert token.item() in allowed


def test_sampling_is_reproducible():
    logits = torch.tensor([[1.0, 5.0, 2.0, 3.0]])

    generator_a = torch.Generator().manual_seed(123)
    generator_b = torch.Generator().manual_seed(123)

    output_a = sample_next_token(
        logits,
        temperature=0.8,
        top_k=3,
        generator=generator_a,
    )

    output_b = sample_next_token(
        logits,
        temperature=0.8,
        top_k=3,
        generator=generator_b,
    )

    assert torch.equal(output_a, output_b)


def test_sampling_rejects_invalid_temperature():
    logits = torch.randn(2, 16)

    with pytest.raises(ValueError):
        sample_next_token(logits, temperature=0)


def test_sampling_rejects_invalid_top_k():
    logits = torch.randn(2, 16)

    with pytest.raises(ValueError):
        sample_next_token(logits, top_k=17)


def test_generation_preserves_prompt(model):
    prompt = torch.tensor([[1, 2, 3]])

    output = generate_tokens(
        model,
        prompt,
        max_new_tokens=5,
        do_sample=False,
    )

    assert torch.equal(output[:, :3], prompt)
    assert output.shape == (1, 8)


def test_generation_adds_requested_number_of_tokens(model):
    prompt = torch.tensor([[1, 2, 3]])

    output = generate_tokens(
        model,
        prompt,
        max_new_tokens=4,
        do_sample=False,
    )

    assert output.shape == (1, 7)


def test_generation_with_zero_new_tokens(model):
    prompt = torch.tensor([[1, 2, 3]])

    output = generate_tokens(
        model,
        prompt,
        max_new_tokens=0,
    )

    assert torch.equal(output, prompt)


def test_generation_respects_context_window(model):
    prompt = torch.tensor([[1, 2, 3, 4, 5, 6, 7]])

    output = generate_tokens(
        model,
        prompt,
        max_new_tokens=10,
        do_sample=False,
    )

    assert output.shape == (1, 17)


def test_generation_stops_on_stop_token(model):
    # Replace the sampler with a model whose highest-probability
    # next token is always token 5.
    original_forward = model.forward

    def fixed_forward(idx, targets=None):
        batch_size, sequence_length = idx.shape

        logits = torch.zeros(
            batch_size,
            sequence_length,
            model.vocab_size,
            device=idx.device,
        )

        logits[:, :, 5] = 100.0

        return logits, None

    model.forward = fixed_forward

    prompt = torch.tensor([[1, 2]])

    output = generate_tokens(
        model,
        prompt,
        max_new_tokens=10,
        do_sample=False,
        stop_token_id=5,
    )

    model.forward = original_forward

    assert output.shape == (1, 3)
    assert output[0, -1].item() == 5


def test_generation_restores_model_mode(model):
    prompt = torch.tensor([[1, 2, 3]])

    model.train()

    generate_tokens(
        model,
        prompt,
        max_new_tokens=2,
        do_sample=False,
    )

    assert model.training


def test_generation_rejects_invalid_prompt_shape(model):
    prompt = torch.tensor([1, 2, 3])

    with pytest.raises(ValueError):
        generate_tokens(
            model,
            prompt,
            max_new_tokens=2,
        )


def test_generation_rejects_prompt_outside_vocabulary(model):
    prompt = torch.tensor([[1, 2, 99]])

    with pytest.raises(ValueError):
        generate_tokens(
            model,
            prompt,
            max_new_tokens=2,
        )


def test_generation_rejects_prompt_longer_than_integer_limits(model):
    prompt = torch.tensor([[1, 2, 3]])

    with pytest.raises(ValueError):
        generate_tokens(
            model,
            prompt,
            max_new_tokens=-1,
        )


def test_generation_does_not_modify_original_prompt(model):
    prompt = torch.tensor([[1, 2, 3]])
    original = prompt.clone()

    generate_tokens(
        model,
        prompt,
        max_new_tokens=3,
        do_sample=False,
    )

    assert torch.equal(prompt, original)
