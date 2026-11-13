
# Mini Transformer

A small GPT-style decoder-only Transformer implemented from first principles
using PyTorch.

The project explores the complete lifecycle of a small language model:

- Tokenization and dataset preparation
- Causal self-attention and multi-head attention
- Transformer blocks and language modeling
- Training and optimization
- Evaluation and perplexity
- Autoregressive text generation
- KV caching and inference optimization
- Model artifacts and deployment

## Current implementation status

### Phase 1 — Foundation and data pipeline

- [x] Python package structure
- [x] YAML configuration
- [x] Character-level tokenizer
- [x] Training/validation split
- [x] Next-token dataset construction
- [x] Batch generation
- [x] Dataset preparation script
- [x] Unit tests

### Upcoming phases

- [ ] Phase 2 — Transformer architecture
- [ ] Phase 3 — Training and optimization
- [ ] Phase 4 — Generation and evaluation
- [ ] Phase 5 — Performance engineering
- [ ] Phase 6 — Inference and KV cache
- [ ] Phase 7 — Deployment and production readiness

## Requirements

- Python 3.11+
- PyTorch 2.5+
- A compatible development environment

## Installation

```bash
python -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -e ".[dev]"
```

## Prepare the dataset

Place a UTF-8 corpus at:

`data/raw/corpus.txt`

Then execute:

```bash
python scripts/prepare_data.py \
    --config configs/baseline.yaml
```

## Run tests

```bash
pytest
```

## Lint

```bash
ruff check .
```

## Project structure

- `src/mini_transformer/`: reusable implementation
- `configs/`: experiment configurations
- `scripts/`: command-line entry points
- `data/`: raw and processed datasets
- `tests/`: automated tests
- `experiments/`: experiment configurations and results

## Design philosophy

The initial implementation prioritizes transparency and mathematical
correctness over abstraction and premature optimization.

The model architecture is implemented directly using PyTorch tensor
operations and neural-network primitives. Optimized implementations
are introduced only after establishing a reference implementation.
