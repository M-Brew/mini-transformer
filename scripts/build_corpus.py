
"""
Build data/raw/corpus.txt from Jane Austen's six novels (Project Gutenberg).

Pipeline:
    1. Download each plain-text ebook (cached under data/raw/gutenberg/).
    2. Decode as strict UTF-8 and normalize line endings and Unicode (NFC).
    3. Keep only the text between the Gutenberg START/END markers.
    4. Remove edition artifacts: illustration blocks and captions, layout
       markup, italic underscores, footnotes, editorial notes, printer
       colophons, front matter and tables of contents.
    5. Split each novel into chapters; each chapter becomes one document.
    6. Validate every document (character inventory, residual markup,
       expected chapter counts, duplicates) and fail loudly on problems.
    7. Write the corpus with document separators, plus a provenance
       manifest (data/corpus_manifest.json).

Documentation of the result lives in data/DATASET.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from mini_transformer.config import load_config
from mini_transformer.dataset import (
    DOCUMENT_SEPARATOR,
    join_documents,
    split_documents,
    split_train_validation_documents,
)

# The primary site is tried first; the PGLAF mirror serves identical files.
SOURCE_URL_TEMPLATES = (
    "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt",
    "https://gutenberg.pglaf.org/cache/epub/{id}/pg{id}.txt",
)

USER_AGENT = "mini-transformer-corpus-builder/1.0"


@dataclass(frozen=True)
class Book:
    gutenberg_id: int
    title: str
    year: int
    expected_chapters: int


# Publication order. Chapter counts are the novels' published structure and
# act as a check that heading detection found every chapter and nothing else.
BOOKS = (
    Book(161, "Sense and Sensibility", 1811, 50),
    Book(1342, "Pride and Prejudice", 1813, 61),
    Book(141, "Mansfield Park", 1814, 48),
    Book(158, "Emma", 1815, 55),
    Book(121, "Northanger Abbey", 1817, 31),
    Book(105, "Persuasion", 1817, 24),
)

START_MARKER = re.compile(r"^\*\*\* START OF THE PROJECT GUTENBERG EBOOK .*$", re.M)
END_MARKER = re.compile(r"^\*\*\* END OF THE PROJECT GUTENBERG EBOOK .*$", re.M)

CHAPTER_HEADING = re.compile(r"^(?:CHAPTER|Chapter)\s+([IVXLC]+|\d+)\.?$")
VOLUME_HEADING = re.compile(r"^(?:VOLUME|Volume)\s+[IVX]+\.?$")

# Volume divisions inside chapters, e.g. "END OF THE FIRST VOLUME" (Sense and
# Sensibility) and "(End of volume one.)" (Persuasion).
VOLUME_END = re.compile(
    r"^\(?end of (?:the )?(?:\w+ )?volume(?: \w+)?\.?\)?$", re.I
)

# Lines that end a novel; everything from here on in the final chapter is
# back matter (closing word, editor's note, printer's colophon).
BACK_MATTER = re.compile(
    r"^(?:THE END|FINIS\.?|Finis\.?|A NOTE ON THE TEXT|CHISWICK PRESS\b.*)$"
)

# A "chapter" whose body is shorter than this is a table-of-contents entry
# (heading immediately followed by the next heading), not a real chapter.
# Real chapters in these novels are several thousand characters long.
MIN_CHAPTER_CHARS = 500

# Unicode categories permitted in the cleaned text: letters, numbers,
# punctuation and currency (the £ sign). Everything else is rejected.
ALLOWED_CATEGORIES = ("L", "N", "P", "Sc")

# Strings that must never survive cleaning.
RESIDUAL_MARKUP = (
    "[Illustration",
    "Gutenberg",
    "/*",
    "*/",
    "^{",
    "_",
    "﻿",
    "�",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the Jane Austen training corpus"
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/baseline.yaml"),
        help="Config providing data.input_path and validation_fraction",
    )

    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/raw/gutenberg"),
        help="Where downloaded source files are kept",
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/corpus_manifest.json"),
        help="Where the provenance manifest is written",
    )

    return parser.parse_args()


def sha256(data: bytes | str) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")

    return hashlib.sha256(data).hexdigest()


def download(book: Book, cache_dir: Path) -> tuple[bytes, str]:
    """Return the raw ebook bytes and the URL they came from."""

    cache_path = cache_dir / f"pg{book.gutenberg_id}.txt"
    source_path = cache_path.with_suffix(".source")

    if cache_path.is_file() and source_path.is_file():
        return cache_path.read_bytes(), source_path.read_text().strip()

    errors = []

    for template in SOURCE_URL_TEMPLATES:
        url = template.format(id=book.gutenberg_id)
        request = urllib.request.Request(
            url, headers={"User-Agent": USER_AGENT}
        )

        try:
            with urllib.request.urlopen(request, timeout=600) as response:
                data = response.read()
        except (urllib.error.URLError, TimeoutError) as error:
            errors.append(f"{url}: {error}")
            continue

        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(data)
        source_path.write_text(url + "\n")

        return data, url

    raise RuntimeError(
        f"Could not download {book.title}:\n" + "\n".join(errors)
    )


def decode(data: bytes) -> str:
    # Strict decoding: corrupted bytes abort the build instead of being
    # silently replaced.
    text = data.decode("utf-8")
    text = text.removeprefix("﻿")
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    return unicodedata.normalize("NFC", text)


def extract_body(text: str) -> str:
    start = START_MARKER.search(text)
    end = END_MARKER.search(text)

    if start is None or end is None or end.start() <= start.end():
        raise ValueError("Project Gutenberg START/END markers not found")

    return text[start.end():end.start()]


def remove_illustrations(text: str) -> str:
    """
    Remove "[Illustration ...]" blocks, which may span lines and contain
    nested brackets. If a block carries a chapter heading (Pride and
    Prejudice's decorated "Chapter I."), the heading is kept.
    """

    output = []
    position = 0

    while True:
        start = text.find("[Illustration", position)

        if start == -1:
            output.append(text[position:])
            break

        output.append(text[position:start])

        depth = 0
        end = start

        for end in range(start, len(text)):
            if text[end] == "[":
                depth += 1
            elif text[end] == "]":
                depth -= 1

                if depth == 0:
                    break
        else:
            raise ValueError("Unterminated [Illustration] block")

        block = text[start:end + 1]
        heading = re.search(
            r"^\s*(?:CHAPTER|Chapter)\s+([IVXLC]+)\.?\]?\s*$",
            block,
            re.M,
        )

        if heading:
            output.append(f"\nCHAPTER {heading.group(1)}.\n")

        position = end + 1

    return "".join(output)


def remove_markup(text: str) -> str:
    text = remove_illustrations(text)

    # Typesetting directives around letters in Pride and Prejudice,
    # e.g. "/* NIND “My dear friend, */".
    text = re.sub(r"/\*\s*(?:NIND|RIGHT)?\s*", "", text)
    text = re.sub(r"\s*\*/", "", text)

    # Footnote bodies on their own line, then inline markers such as "[1]".
    text = re.sub(r"(?m)^\s*\[\d+\]\s.*$", "", text)
    text = re.sub(r"\[\d+\]", "", text)

    # The editions use typographic apostrophes; Sense and Sensibility has a
    # single stray ASCII one.
    text = text.replace("'", "’")

    # _underscores_ mark italics in Gutenberg plain text.
    return text.replace("_", "")


def normalize_whitespace(text: str) -> str:
    """Trim every line and collapse runs of blank lines to one."""

    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def truncate_back_matter(lines: list[str]) -> list[str]:
    """
    Drop everything from the first back-matter line onwards. Applied to every
    chapter: it trims the end of each novel, and it also empties table-of-
    contents segments that run into editorial notes (Northanger Abbey's
    contents list ends in "A NOTE ON THE TEXT"). The expected-chapter-count
    check guards against a real chapter being cut.
    """

    for index, line in enumerate(lines):
        if BACK_MATTER.match(line.strip()):
            return lines[:index]

    return lines


def split_chapters(text: str, book: Book) -> list[str]:
    chapters: list[list[str]] = []
    current: list[str] | None = None

    for line in text.split("\n"):
        stripped = line.strip()

        if CHAPTER_HEADING.match(stripped):
            current = []
            chapters.append(current)
            continue

        if VOLUME_HEADING.match(stripped) or VOLUME_END.match(stripped):
            continue

        if current is not None:
            current.append(line)

    bodies = [
        normalize_whitespace("\n".join(truncate_back_matter(lines)))
        for lines in chapters
    ]
    bodies = [body for body in bodies if len(body) >= MIN_CHAPTER_CHARS]

    if len(bodies) != book.expected_chapters:
        raise ValueError(
            f"{book.title}: found {len(bodies)} chapters, "
            f"expected {book.expected_chapters}"
        )

    return bodies


def validate_document(document: str, label: str) -> None:
    for marker in RESIDUAL_MARKUP:
        if marker in document:
            raise ValueError(f"{label}: residual markup {marker!r}")

    if DOCUMENT_SEPARATOR in document:
        raise ValueError(f"{label}: contains the document separator")

    for character in set(document):
        if character in "\n ":
            continue

        if not unicodedata.category(character).startswith(
            ALLOWED_CATEGORIES
        ):
            name = unicodedata.name(character, "UNKNOWN")
            raise ValueError(
                f"{label}: disallowed character {character!r} ({name})"
            )

    if len(document) < MIN_CHAPTER_CHARS:
        raise ValueError(f"{label}: suspiciously short ({len(document)})")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    corpus_path = Path(config.data.input_path)

    documents: list[str] = []
    document_records: list[dict] = []
    source_records: list[dict] = []

    for book in BOOKS:
        data, url = download(book, args.cache_dir)
        body = remove_markup(extract_body(decode(data)))
        chapters = split_chapters(body, book)

        for number, chapter in enumerate(chapters, start=1):
            label = f"{book.title}, chapter {number}"
            validate_document(chapter, label)

            documents.append(chapter)
            document_records.append(
                {
                    "index": len(document_records),
                    "gutenberg_id": book.gutenberg_id,
                    "title": book.title,
                    "chapter": number,
                    "characters": len(chapter),
                    "sha256": sha256(chapter),
                }
            )

        source_records.append(
            {
                "gutenberg_id": book.gutenberg_id,
                "title": book.title,
                "first_published": book.year,
                "source_url": url,
                "source_bytes": len(data),
                "source_sha256": sha256(data),
                "chapters": len(chapters),
                "characters": sum(len(chapter) for chapter in chapters),
            }
        )

        print(
            f"{book.title:<24} {len(chapters):>3} chapters "
            f"{source_records[-1]['characters']:>9,} characters"
        )

    hashes = [record["sha256"] for record in document_records]

    if len(set(hashes)) != len(hashes):
        raise ValueError("Duplicate documents found")

    corpus = join_documents(documents)

    if split_documents(corpus) != documents:
        raise ValueError("Corpus does not round-trip through the separator")

    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    corpus_path.write_text(corpus, encoding="utf-8")

    # Record the split prepare_data.py will make, so it can be audited.
    fraction = config.data.validation_fraction
    _, validation = split_train_validation_documents(
        list(range(len(documents))), fraction
    )
    held_out = set(validation)

    for record in document_records:
        record["split"] = (
            "validation" if record["index"] in held_out else "train"
        )

    validation_characters = sum(
        record["characters"]
        for record in document_records
        if record["split"] == "validation"
    )

    manifest = {
        "name": "Jane Austen novels (Project Gutenberg)",
        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "builder": "scripts/build_corpus.py",
        "license": (
            "Public domain. Project Gutenberg headers, footers and license "
            "text are removed, so the Project Gutenberg License does not "
            "apply to the derived corpus."
        ),
        "corpus": {
            "path": str(corpus_path),
            "encoding": "utf-8",
            "bytes": len(corpus.encode("utf-8")),
            "characters": len(corpus),
            "sha256": sha256(corpus),
            "documents": len(documents),
            "document_unit": "chapter",
            "document_separator": f"U+{ord(DOCUMENT_SEPARATOR):04X}",
            "distinct_characters": len(set(corpus)),
        },
        "split": {
            "method": "whole documents, evenly spaced "
            "(mini_transformer.dataset.split_train_validation_documents)",
            "validation_fraction": fraction,
            "validation_documents": len(held_out),
            "train_documents": len(documents) - len(held_out),
            "validation_character_share": round(
                validation_characters
                / sum(r["characters"] for r in document_records),
                4,
            ),
        },
        "sources": source_records,
        "documents": document_records,
    }

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print()
    print(f"Documents:  {len(documents)} ({len(held_out)} validation)")
    print(f"Characters: {len(corpus):,}")
    print(f"SHA-256:    {manifest['corpus']['sha256']}")
    print(f"Corpus:     {corpus_path}")
    print(f"Manifest:   {args.manifest}")


if __name__ == "__main__":
    main()
