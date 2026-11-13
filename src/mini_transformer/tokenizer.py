
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


class CharacterTokenizer:
    """Character tokenizer with an explicit unknown-character token."""

    FORMAT_VERSION = 1
    UNK_TOKEN = "<unk>"

    def __init__(self, vocabulary: Iterable[str]) -> None:
        tokens = list(vocabulary)

        if not tokens:
            raise ValueError("Vocabulary cannot be empty")

        if any(
            not isinstance(token, str) or len(token) != 1
            for token in tokens
        ):
            raise ValueError(
                "Every vocabulary entry must be exactly one character"
            )

        if len(tokens) != len(set(tokens)):
            raise ValueError("Vocabulary contains duplicate characters")

        if self.UNK_TOKEN in tokens:
            raise ValueError(
                f"{self.UNK_TOKEN} is reserved and cannot be a character"
            )

        self.vocabulary = tuple(tokens)
        self.stoi = {
            character: index
            for index, character in enumerate(self.vocabulary)
        }

        self.unk_id = len(self.vocabulary)
        self.itos = {
            index: character
            for index, character in enumerate(self.vocabulary)
        }

    @classmethod
    def train(cls, text: str) -> CharacterTokenizer:
        if not text:
            raise ValueError("Cannot train tokenizer on empty text")

        return cls(sorted(set(text)))

    @property
    def vocab_size(self) -> int:
        return len(self.vocabulary) + 1

    def encode(self, text: str) -> list[int]:
        return [
            self.stoi.get(character, self.unk_id)
            for character in text
        ]

    def decode(
        self,
        token_ids: Iterable[int],
        skip_unknown: bool = False,
    ) -> str:
        characters: list[str] = []

        for token_id in token_ids:
            if token_id == self.unk_id:
                if not skip_unknown:
                    characters.append(self.UNK_TOKEN)
                continue

            if token_id not in self.itos:
                raise ValueError(f"Unknown token ID: {token_id}")

            characters.append(self.itos[token_id])

        return "".join(characters)

    def to_dict(self) -> dict:
        return {
            "format_version": self.FORMAT_VERSION,
            "tokenizer_type": "character",
            "unknown_token": self.UNK_TOKEN,
            "vocabulary": list(self.vocabulary),
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", encoding="utf-8") as file:
            json.dump(
                self.to_dict(),
                file,
                ensure_ascii=False,
                indent=2,
            )

    @classmethod
    def load(cls, path: str | Path) -> CharacterTokenizer:
        with Path(path).open("r", encoding="utf-8") as file:
            payload = json.load(file)

        if payload.get("format_version") != cls.FORMAT_VERSION:
            raise ValueError("Unsupported tokenizer format version")

        if payload.get("tokenizer_type") != "character":
            raise ValueError("Unsupported tokenizer type")

        if payload.get("unknown_token") != cls.UNK_TOKEN:
            raise ValueError("Unknown-token configuration mismatch")

        return cls(payload["vocabulary"])
