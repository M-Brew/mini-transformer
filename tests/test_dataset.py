
import pytest
import torch

from mini_transformer.dataset import (
    DOCUMENT_SEPARATOR,
    LanguageModelDataset,
    join_documents,
    load_token_ids,
    prepare_token_datasets,
    save_token_ids,
    split_documents,
    split_tokens,
    split_train_validation_documents,
)


def test_dataset_length(dataset, sample_text):
    assert len(dataset) == len(sample_text) - 8


def test_dataset_input_target_alignment():
    tokens = torch.tensor(
        [0, 1, 2, 3, 4, 5],
        dtype=torch.long,
    )

    dataset = LanguageModelDataset(tokens, block_size=3)

    x, y = dataset.get_example(0)

    assert x.tolist() == [0, 1, 2]
    assert y.tolist() == [1, 2, 3]


def test_dataset_returns_correct_shapes(dataset):
    x, y = dataset.get_example(0)

    assert x.shape == (8,)
    assert y.shape == (8,)


def test_dataset_rejects_non_one_dimensional_tokens():
    tokens = torch.zeros((2, 4), dtype=torch.long)

    with pytest.raises(ValueError):
        LanguageModelDataset(tokens, block_size=2)


def test_dataset_rejects_wrong_dtype():
    tokens = torch.zeros(10, dtype=torch.float32)

    with pytest.raises(ValueError):
        LanguageModelDataset(tokens, block_size=2)


def test_dataset_rejects_short_sequence():
    tokens = torch.tensor([1, 2, 3], dtype=torch.long)

    with pytest.raises(ValueError):
        LanguageModelDataset(tokens, block_size=3)


def test_dataset_rejects_invalid_index(dataset):
    with pytest.raises(IndexError):
        dataset.get_example(len(dataset))


def test_batch_shapes(dataset):
    x, y = dataset.get_batch(batch_size=4)

    assert x.shape == (4, 8)
    assert y.shape == (4, 8)


def test_batch_target_is_shifted():
    tokens = torch.arange(20, dtype=torch.long)

    dataset = LanguageModelDataset(
        tokens,
        block_size=5,
    )

    generator = torch.Generator().manual_seed(42)

    x, y = dataset.get_batch(
        batch_size=8,
        generator=generator,
    )

    assert torch.equal(x[:, 1:], y[:, :-1])


def test_batch_respects_device(dataset):
    x, y = dataset.get_batch(batch_size=2, device="cpu")

    assert x.device.type == "cpu"
    assert y.device.type == "cpu"


def test_split_tokens():
    tokens = torch.arange(100, dtype=torch.long)

    train, validation = split_tokens(
        tokens,
        validation_fraction=0.2,
    )

    assert len(train) == 80
    assert len(validation) == 20

    assert train[-1].item() == 79
    assert validation[0].item() == 80


def test_split_preserves_all_tokens():
    tokens = torch.arange(100, dtype=torch.long)

    train, validation = split_tokens(tokens, 0.2)

    combined = torch.cat([train, validation])

    assert torch.equal(combined, tokens)


def test_split_rejects_invalid_fraction():
    tokens = torch.arange(100, dtype=torch.long)

    with pytest.raises(ValueError):
        split_tokens(tokens, 1.0)


def test_prepare_token_datasets():
    tokens = torch.arange(100, dtype=torch.long)

    train, validation = prepare_token_datasets(
        tokens,
        block_size=8,
        validation_fraction=0.2,
    )

    assert len(train) == 72
    assert len(validation) == 12


def test_save_and_load_token_ids(tmp_path):
    tokens = torch.arange(100, dtype=torch.long)
    path = tmp_path / "tokens.pt"

    save_token_ids(tokens, path)
    restored = load_token_ids(path)

    assert torch.equal(tokens, restored)
    assert restored.dtype == torch.long


def test_documents_round_trip():
    documents = ["First chapter.\n\nSecond paragraph.", "Second chapter."]

    text = join_documents(documents)

    assert text.count(DOCUMENT_SEPARATOR) == 2
    assert split_documents(text) == documents


def test_join_documents_rejects_embedded_separator():
    with pytest.raises(ValueError):
        join_documents([f"bad{DOCUMENT_SEPARATOR}document"])


def test_join_documents_rejects_blank_document():
    with pytest.raises(ValueError):
        join_documents(["fine", "  \n"])


def test_document_split_keeps_documents_whole():
    documents = [f"document {index}" for index in range(20)]

    train, validation = split_train_validation_documents(documents, 0.1)

    assert len(validation) == 2
    assert sorted(train + validation) == sorted(documents)
    assert not set(train) & set(validation)


def test_document_split_spreads_validation_through_corpus():
    documents = [f"document {index}" for index in range(100)]

    _, validation = split_train_validation_documents(documents, 0.1)

    # Every 10th document is held out, not just the tail of the corpus.
    assert validation == [f"document {index}" for index in range(9, 100, 10)]


def test_document_split_is_deterministic():
    documents = [f"document {index}" for index in range(37)]

    assert split_train_validation_documents(
        documents, 0.25
    ) == split_train_validation_documents(documents, 0.25)


def test_document_split_rejects_empty_partition():
    with pytest.raises(ValueError):
        split_train_validation_documents(["only one"], 0.1)
