
import pytest

from mini_transformer.tokenizer import CharacterTokenizer


def test_tokenizer_round_trip(tokenizer):
    text = "the quick brown fox"

    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)

    assert decoded == text


def test_tokenizer_vocabulary_is_deterministic():
    first = CharacterTokenizer.train("cabba")
    second = CharacterTokenizer.train("abcab")

    assert first.vocabulary == second.vocabulary


def test_tokenizer_vocab_size_includes_unknown_token():
    tokenizer = CharacterTokenizer.train("abc")

    assert tokenizer.vocab_size == 4
    assert tokenizer.unk_id == 3


def test_unknown_character_maps_to_unknown_id(tokenizer):
    encoded = tokenizer.encode("!")

    assert encoded == [tokenizer.unk_id]


def test_unknown_token_decoding(tokenizer):
    decoded = tokenizer.decode([tokenizer.unk_id])

    assert decoded == CharacterTokenizer.UNK_TOKEN


def test_unknown_token_can_be_skipped(tokenizer):
    decoded = tokenizer.decode(
        [tokenizer.unk_id],
        skip_unknown=True,
    )

    assert decoded == ""


def test_tokenizer_save_and_load(tmp_path, tokenizer):
    path = tmp_path / "tokenizer.json"

    tokenizer.save(path)

    restored = CharacterTokenizer.load(path)

    assert restored.vocabulary == tokenizer.vocabulary
    assert restored.vocab_size == tokenizer.vocab_size

    text = "the quick brown fox"

    assert restored.encode(text) == tokenizer.encode(text)


def test_duplicate_vocabulary_is_rejected():
    with pytest.raises(ValueError):
        CharacterTokenizer(["a", "a"])


def test_empty_vocabulary_is_rejected():
    with pytest.raises(ValueError):
        CharacterTokenizer([])

def test_tokenizer_handles_validation_only_characters():
    tokenizer = CharacterTokenizer.train("abcabc")

    encoded = tokenizer.encode("abcz")

    assert encoded[-1] == tokenizer.unk_id
    assert tokenizer.decode(encoded) == "abc<unk>"
