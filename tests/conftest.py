
import pytest
import torch

from mini_transformer.dataset import LanguageModelDataset
from mini_transformer.tokenizer import CharacterTokenizer


@pytest.fixture
def sample_text() -> str:
    return "the quick brown fox jumps over the lazy dog"


@pytest.fixture
def tokenizer(sample_text: str) -> CharacterTokenizer:
    return CharacterTokenizer.train(sample_text)


@pytest.fixture
def sample_tokens(tokenizer: CharacterTokenizer) -> torch.Tensor:
    return torch.tensor(
        tokenizer.encode("the quick brown fox jumps over the lazy dog"),
        dtype=torch.long,
    )


@pytest.fixture
def dataset(sample_tokens: torch.Tensor) -> LanguageModelDataset:
    return LanguageModelDataset(
        sample_tokens,
        block_size=8,
    )
