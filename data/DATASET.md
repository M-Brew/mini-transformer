# Dataset: Jane Austen novels

The training corpus is the complete text of Jane Austen's six novels, taken
from Project Gutenberg and cleaned for character-level language modelling.

The machine-readable record of every build is
[`corpus_manifest.json`](corpus_manifest.json): source URLs, source and corpus
hashes, per-chapter sizes and hashes, and each chapter's train/validation
assignment.

## Summary

| Property | Value |
|---|---|
| File | `data/raw/corpus.txt` (generated, not committed) |
| Encoding | UTF-8, LF line endings, Unicode NFC |
| Size | 4,006,051 characters (4.06 MB) |
| Documents | 269 (one per chapter) |
| Distinct characters | 89, including newline and the document separator |
| Train / validation | 243 / 26 chapters (90.9% / 9.1% of characters) |
| SHA-256 | `12698b44711f0a45c0a41ff388d83de1f82410f2be1255949085a2ca0bf5270b` |

## Source

| Novel | Published | Gutenberg ebook | Chapters | Characters |
|---|---|---|---|---|
| Sense and Sensibility | 1811 | [#161](https://www.gutenberg.org/ebooks/161) | 50 | 667,837 |
| Pride and Prejudice | 1813 | [#1342](https://www.gutenberg.org/ebooks/1342) | 61 | 683,926 |
| Mansfield Park | 1814 | [#141](https://www.gutenberg.org/ebooks/141) | 48 | 880,878 |
| Emma | 1815 | [#158](https://www.gutenberg.org/ebooks/158) | 55 | 877,723 |
| Northanger Abbey | 1817 | [#121](https://www.gutenberg.org/ebooks/121) | 31 | 431,056 |
| Persuasion | 1817 | [#105](https://www.gutenberg.org/ebooks/105) | 24 | 463,824 |

The sources were retrieved on 2026-11-18 from `gutenberg.pglaf.org`, the
official Project Gutenberg mirror, because the main site was down for
maintenance. The mirror serves the same files. The SHA-256 of each source file
is recorded in the manifest.

**Why this source:** a single author, period and genre gives the corpus a
consistent style, vocabulary and orthography (British spelling, typographic
quotes and dashes). It is large enough to train a small character-level model
without being dominated by one book.

## License

Jane Austen died in 1817, so all six novels are in the **public domain**
worldwide.

The Project Gutenberg ebooks are distributed under the Project Gutenberg
License, which governs the use of the "Project Gutenberg" trademark. The build
removes the Gutenberg header, footer and license text, and the corpus contains
no reference to Project Gutenberg. Under the license's own terms, a public-
domain text with those elements removed may be used without restriction. If
you redistribute the original ebook files from `data/raw/gutenberg/` instead,
they must keep their license text.

Some editorial material in these editions is not Austen's text and may be
covered by separate rights: the 1894 preface by George Saintsbury, Hugh
Thomson's illustrations and captions in *Pride and Prejudice*, and the Signet
Classic note in *Northanger Abbey*. All of it is removed (see below).

## Preparation procedure

Implemented in [`scripts/build_corpus.py`](../scripts/build_corpus.py).

1. **Download** each ebook's plain-text UTF-8 file. The main site is tried
   first, then the PGLAF mirror. Files are cached in `data/raw/gutenberg/`
   together with the URL each one came from.
2. **Decode strictly** as UTF-8, so corrupted bytes stop the build instead of
   being replaced. Remove any byte-order mark, convert CRLF/CR line endings to
   LF, and apply Unicode NFC normalization.
3. **Extract the book** between the `*** START OF …` and `*** END OF …`
   markers. This removes the Project Gutenberg header, footer and license.
4. **Remove edition artifacts:**
   - `[Illustration …]` blocks, including multi-line captions and the
     copyright lines inside them. (*Pride and Prejudice*, Chapter I's heading
     is inside an illustration block. It is recovered before the block is
     removed.)
   - `/* NIND … */` and `/* RIGHT … */` typesetting markup around letters in
     *Pride and Prejudice*.
   - Footnote markers and footnote lines (one in *Northanger Abbey*).
   - `_underscores_` used for italics.
   - The one straight apostrophe in the corpus, which is normalized to the
     curly form used everywhere else.
5. **Split into chapters** on `CHAPTER n` headings (Roman or Arabic
   numerals). The heading lines are not kept.
   - Front matter is dropped: title pages, contents, prefaces, and the
     author's Advertisement in *Northanger Abbey*.
   - Table-of-contents entries are discarded as segments under 500
     characters.
   - Volume headings and in-text volume markers ("END OF THE FIRST VOLUME",
     "(End of volume one.)") are dropped.
   - Each chapter is cut at the first back-matter line ("THE END", "Finis",
     "A NOTE ON THE TEXT", the printer's colophon).
6. **Normalize whitespace**: trim each line and collapse runs of blank lines
   to one, so paragraphs are separated by exactly one blank line. Line breaks
   within paragraphs are kept as they are in the source (about 70 columns).
7. **Validate.** The build fails if any of these checks fail:
   - Each novel has its published chapter count.
   - No residual markup remains (`[Illustration`, `_`, `/*`, `^{`,
     "Gutenberg", U+FFFD).
   - Every character is a letter, number, punctuation, currency sign, space
     or newline.
   - No chapter is shorter than 500 characters, and no two are duplicates.
   - The corpus round-trips through the document separator.
8. **Write** the corpus and `data/corpus_manifest.json`.

## Document boundaries

Each chapter is one document. Each document is followed by a line containing
only U+001E (ASCII "record separator"):

```
…last line of chapter one.
␞
First line of chapter two…
```

The separator is a single character, so the character tokenizer gives it a
vocabulary entry and the model can learn where chapters end. Cleaning
guarantees it never appears inside a document. `join_documents` and
`split_documents` in `mini_transformer.dataset` write and read this format.
Some tools, including `file`, report the corpus as binary "data" because of
this control character. It is still valid UTF-8 text.

## Train/validation split

[`scripts/prepare_data.py`](../scripts/prepare_data.py) assigns **whole
chapters** to one partition or the other using
`split_train_validation_documents`. With `validation_fraction: 0.1`, every
tenth chapter is held out. Chapter *i* (0-based) goes to validation when
`floor((i + 1) · 0.1) > floor(i · 0.1)`.

- **No leakage:** no chapter, and no paragraph over 200 characters, appears
  in both partitions. This was checked on the prepared token files.
- **Representative:** the 26 validation chapters come from all six novels
  (Pride and Prejudice 6, Emma 6, Sense and Sensibility 5, Mansfield Park 4,
  Northanger Abbey 3, Persuasion 2).
- **Deterministic:** the same corpus and fraction always give the same split.
  The manifest lists each chapter's assignment.
- **Tokenizer fitted on train only:** every character in the validation
  partition also appears in training, so validation contains no `<unk>`
  tokens.

A corpus without separators falls back to the previous contiguous split.

## Reproducing

```bash
python scripts/build_corpus.py --config configs/baseline.yaml
python scripts/prepare_data.py --config configs/baseline.yaml
```

The build reuses cached sources from `data/raw/gutenberg/`. Delete that folder
to download them again. Project Gutenberg sometimes revises its files, so a
fresh download can produce a different corpus hash. Compare the
`source_sha256` values in the manifest to see which source changed.

## Known limitations

- Hard line wrapping from the source is kept, so `\n` often appears
  mid-sentence. This is a choice for fidelity to the source (verse and the
  layout of letters depend on line breaks). It is not an error.
- Chapter headings and novel titles are not in the text, so the model never
  sees which book or chapter a passage belongs to.
- The text follows these specific editions. For example, the *Northanger
  Abbey* text has spelling and punctuation modernized to British usage
  (according to its editorial note), and the other editions may differ from
  the first printings.
- The corpus is small by modern standards (about 4M characters). It is
  suitable for a small character-level model, not for word-level or subword
  models with large vocabularies.
