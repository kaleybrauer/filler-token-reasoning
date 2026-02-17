# Filler Token Reasoning

**Train and analyze language models whose accuracy improves when non-semantic filler tokens are added to prompts.**

This repository implements experiments to test whether language models can learn to use repeated filler tokens as an internal "workspace" for multi-hop reasoning tasks.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Scripts Reference](#scripts-reference)
- [Data Format](#data-format)
- [Directory Structure](#directory-structure)
- [Acknowledgments](#acknowledgments)

## Overview

Given two factual questions, the model must:
1. Answer each question
2. Add the two answers together

**Example Task:**
```
Q1: At what age did Mozart die?      (Answer: 35)
Q2: What is the atomic number of He?  (Answer: 2)
Final answer: 35 + 2 = 37
```

### The Mechanism

We insert N copies of a filler token (e.g., `<|fim_pad|>`) between the prompt and answer:

```
[BOS] [prompt tokens] [filler × N] [answer tokens] [EOS]
```

During training:
- **Supervised**: Only the answer tokens (labels ≠ -100)
- **Unsupervised**: Prompt and filler tokens (labels = -100)

### Models

- **Pilot experiments**: Qwen2.5-7B
- **Scale experiments**: Qwen2.5-72B
- **Training**: LoRA/QLoRA fine-tuning (parameter-efficient)
- **Inference**: Supports 4-bit/8-bit quantization for 70B+ models

## Installation

### Requirements

- Python ≥ 3.10
- CUDA-capable GPU (recommended: 24GB+ VRAM for 7B, 80GB+ for 72B)
- Linux (for bitsandbytes quantization support)

### Using uv (Recommended)

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create environment and install dependencies
uv sync
```

### Optional Dependencies

- **Flash Attention 2** (faster training/inference):
  ```bash
  pip install flash-attn --no-build-isolation
  ```

- **Weights & Biases** (experiment tracking):
  ```bash
  pip install wandb
  wandb login
  ```

## Quick Start

```bash
# 1. Download fact sources
python scripts/fetch_facts.py --outdir data/sources

# 2. Generate training dataset
python scripts/generate_2hop_dataset.py \
  --tokenizer Qwen/Qwen2.5-7B \
  --sources data/sources \
  --outdir data/datasets/2hop_7b \
  --n-train 50000 \
  --n-val 2000 \
  --n-test 2000

# 3. Baseline evaluation (pre-training)
python scripts/evaluate.py \
  --model Qwen/Qwen2.5-7B \
  --data-dir data/datasets/2hop_7b \
  --filler-lengths 0,32,128,300,600,1000 \
  --outdir results/baseline_7b \
  --batch-size 8

# 4. LoRA fine-tuning
python scripts/train_lora.py \
  --model Qwen/Qwen2.5-7B \
  --data-dir data/datasets/2hop_7b \
  --outdir outputs/qwen2p5-7b-lora \
  --batch-size 4 \
  --grad-accum 4 \
  --epochs 1

# 5. Fine-tuned evaluation
python scripts/evaluate.py \
  --model Qwen/Qwen2.5-7B \
  --adapter outputs/qwen2p5-7b-lora \
  --data-dir data/datasets/2hop_7b \
  --filler-lengths 0,32,128,300,600,1000 \
  --outdir results/finetuned_7b \
  --batch-size 8
```

## Scripts Reference

### Core Pipeline Scripts

| Script | Purpose | Key Arguments |
|--------|---------|---------------|
| `scripts/fetch_facts.py` | Download fact sources from GitHub | `--outdir` |
| `scripts/generate_2hop_dataset.py` | Generate pre-tokenized training data | `--tokenizer`, `--sources`, `--outdir`, `--n-train`, `--filler-mode` |
| `scripts/train_lora.py` | LoRA fine-tuning | `--model`, `--data-dir`, `--outdir`, `--load-in-4bit`, `--batch-size` |
| `scripts/evaluate.py` | Evaluate model accuracy | `--model`, `--adapter`, `--data-dir`, `--filler-lengths`, `--outdir` |

### Utility Scripts

| Script | Purpose | Usage |
|--------|---------|-------|
| `scripts/check_filler_token.py` | Validate filler token is single-token | `--model Qwen/Qwen2.5-7B --candidate "<\|fim_pad\|>"` |

### Development Scripts (dev/)

Additional experimental scripts for prompt format testing and knowledge checking.

## Data Format

### JSONL Structure

Each line in `train.jsonl`/`val.jsonl`/`test.jsonl` contains:

```json
{
  "id": "train-0",
  "split": "train",
  "prompt": "Answer two questions and add the results.\n\nQ1: ...\nQ2: ...",
  "fact1": "At what age did Mozart die?",
  "fact2": "What is the atomic number of Helium?",
  "a1": 35,
  "a2": 2,
  "answer": 37,
  "type1": "age",
  "type2": "atomic",
  "filler_len": 128,
  "filler_token": "<|fim_pad|>",
  "filler_token_id": 151665,
  "input_ids": [151643, 16492, ...],
  "labels": [-100, -100, ..., 220, 1881],
  "attention_mask": [1, 1, 1, ...],
  "prompt_ids": [16492, 1378, ...],
  "answer_ids": [220, 1881, 151645]
}
```

### Manifest Structure

`manifest.json` contains dataset metadata:

```json
{
  "tokenizer": "Qwen/Qwen2.5-7B",
  "seed": 42,
  "filler_token": "<|fim_pad|>",
  "filler_token_id": 151665,
  "filler_mode": "uniform",
  "filler_min": 0,
  "filler_max": 1000,
  "eval_filler_lengths": [0, 32, 128, 300, 600, 1000],
  "fact_counts": {"train": 450, "val": 56, "test": 57},
  "example_counts": {"train": 50000, "val": 2000, "test": 2000},
  "bos_token_id": 151643,
  "eos_token_id": 151645
}
```

## Directory Structure

```
filler-token-reasoning/
├── scripts/                    # Main pipeline scripts
│   ├── fetch_facts.py
│   ├── generate_2hop_dataset.py
│   ├── train_lora.py
│   ├── evaluate.py
│   └── check_filler_token.py
├── dev/                        # Development/experimental scripts
├── data/                       # Created by scripts (gitignored)
│   ├── sources/                # Fact JSON files
│   │   ├── age_facts.json
│   │   ├── atomic_facts.json
│   │   └── static_facts.json
│   └── datasets/               # Generated datasets
│       └── 2hop/
│           ├── train.jsonl
│           ├── val.jsonl
│           ├── test.jsonl
│           ├── manifest.json
│           └── fact_pools.json
├── outputs/                    # Training checkpoints (gitignored)
├── results/                    # Evaluation results (gitignored)
├── pyproject.toml
└── README.md
```

## Acknowledgments

- Fact sources: [Ryan Greenblatt's compose_facts](https://github.com/rgreenblatt/compose_facts)
- Models: [Qwen Team](https://github.com/QwenLM/Qwen2.5)
