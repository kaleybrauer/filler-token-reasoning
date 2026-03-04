# Filler Token Reasoning

**Train and analyze language models whose accuracy improves when filler tokens are added to prompts.**

This repository tests whether language models can learn to use filler tokens as an internal "workspace" for multi-hop reasoning.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Workflow](#workflow)
- [Scripts Reference](#scripts-reference)
- [Data Format](#data-format)
- [Directory Structure](#directory-structure)
- [Citations](#citations)

## Overview

### The Task

Given two factual questions, the model must look up each fact and add the answers:

```
What is (At what age did Mozart die) + (the atomic number of Helium)?
→ Answer: 37
```

### The Mechanism

All training conditions share an identical prompt: an instruction, followed by 5 few-shot examples showing chain-of-thought (CoT) reasoning, followed by the test question. Only the assistant turn differs:

| Mode | Assistant turn | Loss |
|------|---------------|------|
| **N=0 baseline** | `Answer: 37` | answer tokens only |
| **Filler (N>0)** | `Filler: 1 2 3 ... N`<br>`Answer: 37` | answer tokens only |
| **CoT** | `Step 1: ... = 35`<br>`Step 2: ... = 2`<br>`Calculation: 35 + 2 = 37`<br>`Answer: 37` | all tokens |

The filler occupies the same position in the assistant turn as CoT reasoning with the goal of enabling transfer of learned computation. The **CoT mixture** training mode (`--cot-mixture`) pairs each fact combination with both a CoT and a filler example, following Pfau et al 2024.


### Model and Training

- **Model**: Qwen2.5-72B-Instruct
- **Training**: LoRA/QLoRA fine-tuning
- **Inference**: Supports 4-bit/8-bit quantization for 70B+ models

## Installation

### Requirements

- Python ≥ 3.10
- CUDA-capable GPU (recommended: 24GB+ VRAM for 7B, 80GB+ for 72B)
- Linux (for bitsandbytes quantization support)

### Using uv (Recommended)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
```

### Optional Dependencies

- **Flash Attention 2** (faster training/inference):
  ```bash
  pip install flash-attn --no-build-isolation
  ```

- **Weights & Biases** (experiment tracking):
  ```bash
  pip install wandb && wandb login
  ```

## Workflow

### Step 1: Download Fact Sources

```bash
python scripts/fetch_facts.py --outdir data/sources
```

Downloads `age_facts.json`, `atomic_facts.json`, and `static_facts.json` from [Ryan Greenblatt's compose_facts](https://github.com/rgreenblatt/compose_facts).

---

### Step 2: Filter to Known Facts

Before building the dataset, filter to only facts the model reliably knows. This prevents training on facts the model cannot recall, which would add noise without signal.

```bash
python scripts/knowledge_check.py \
  --model Qwen/Qwen2.5-72B-Instruct \
  --sources data/sources \
  --outdir data/known_facts \
  --num-trials 4 \
  --pass-threshold 0.75
```

Outputs `known_facts.json` (facts answered correctly on ≥ threshold of trials) and `unknown_facts.json`.

---

### Step 3: Generate Dataset

```bash
python scripts/generate_addition_dataset.py \
  --tokenizer Qwen/Qwen2.5-72B-Instruct \
  --known-facts data/known_facts/known_facts.json \
  --outdir data/datasets/2hop_add \
  --n-train 30000 \
  --n-val 600 \
  --n-test 600 \
  --filler-type counting \
  --cot-mixture \
  --eval-filler-lengths 0,64,128,256
```

Key behaviors:
- **Fact isolation**: train/val/test draw from completely separate fact pools (no leakage)
- **Few-shot facts reserved**: 10 facts (5 non-overlapping pairs) are carved out before splitting and used as few-shot examples in every prompt; they never appear in train/val/test
- **Pre-tokenized**: all sequences are tokenized once at generation time
- **CoT mixture** (optional): add `--cot-mixture` to interleave CoT and filler examples for each fact pair


---

### Step 4: Baseline Evaluation

```bash
python scripts/evaluate.py \
  --model Qwen/Qwen2.5-72B-Instruct \
  --data-dir data/datasets/2hop_add \
  --filler-lengths 0,64,128,256 \
  --outdir results/baseline_72b \
  --load-in-4bit \
  --batch-size 8
```

Few-shot examples are loaded automatically from the dataset's `manifest.json`. Optionally use 4bit quantization for large models.

---

### Step 5: LoRA Fine-Tuning

```bash
python scripts/train_lora.py \
  --model Qwen/Qwen2.5-72B-Instruct \
  --data-dir data/datasets/2hop_add \
  --outdir outputs/qwen2p5-72b-qlora \
  --load-in-4bit \
  --batch-size 2 \
  --grad-accum 16 \
  --wandb
```

---

### Step 6: Fine-Tuned Evaluation

```bash
python scripts/evaluate.py \
  --model Qwen/Qwen2.5-72B-Instruct \
  --adapter outputs/qwen2p5-72b-lora \
  --data-dir data/datasets/2hop_add \
  --filler-lengths 0,64,128,256 \
  --outdir results/finetuned_72b \
  --batch-size 8 \
  --show-errors
```

## Scripts Reference

### Core Pipeline

| Script | Purpose | Key Arguments |
|--------|---------|---------------|
| `scripts/fetch_facts.py` | Download raw fact sources | `--outdir` |
| `scripts/knowledge_check.py` | Filter to model-known facts → `known_facts.json` | `--model`, `--sources`, `--outdir`, `--pass-threshold` |
| `scripts/generate_addition_dataset.py` | Build pre-tokenized JSONL dataset | `--tokenizer`, `--known-facts`, `--outdir`, `--cot-mixture`, `--filler-mode` |
| `scripts/train_lora.py` | LoRA/QLoRA fine-tuning | `--model`, `--data-dir`, `--outdir`, `--load-in-4bit` |
| `scripts/evaluate.py` | Accuracy evaluation per filler length | `--model`, `--adapter`, `--data-dir`, `--filler-lengths`, `--outdir` |

<!-- ### Key Arguments for `generate_2hop_dataset.py` -->

<!-- | Argument | Default | Description |
|----------|---------|-------------|
| `--known-facts` | required | Path to `known_facts.json` from Step 2 |
| `--filler-mode` | `eval` | `eval` samples from `--eval-filler-lengths`; `uniform` samples from `[--filler-min, --filler-max]` |
| `--eval-filler-lengths` | `0,32,128,300,600` | Filler lengths for eval mode and the test set |
| `--cot-mixture` | off | Enable 50/50 CoT + filler pairing (Pfau et al.) |
| `--n-fewshot` | `5` | Few-shot examples to draw from the fact pool |
| `--n-fewshot-facts` | `10` | Facts reserved for the few-shot pool (must be ≥ `n-fewshot * 2`) |
| `--n-train` | `28000` | Training examples |
| `--n-val` | `600` | Validation examples |
| `--n-test` | `600` | Test examples | -->

## Data Format

### JSONL Structure

Each line in `train.jsonl`/`val.jsonl`/`test.jsonl`:

```json
{
  "id": "train-0",
  "split": "train",
  "sequence_type": "filler",
  "prompt": "<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\nFiller:",
  "question": "What is (At what age did Mozart die) + (the atomic number of Helium)?",
  "fact1": "At what age did Mozart die?",
  "fact2": "What is the atomic number of Helium?",
  "a1": 35,
  "a2": 2,
  "answer": 37,
  "type1": "age",
  "type2": "atomic",
  "filler_len": 128,
  "filler_type": "counting",
  "input_ids": [151644, 8948, 198, ...],
  "labels": [-100, -100, ..., 220, 1881],
  "attention_mask": [1, 1, 1, ...],
  "prompt_ids": [...],
  "answer_ids": [220, 1881, 151645]
}
```

`sequence_type` is `"filler"` for filler examples (loss on answer only) or `"cot"` for chain-of-thought examples (loss on full CoT response). `filler_len` is `0` for the N=0 baseline.

### Manifest Structure

`manifest.json` stores all metadata needed to reproduce prompts at eval time:

```json
{
  "tokenizer": "Qwen/Qwen2.5-7B",
  "known_facts_source": "data/known_facts_7b/known_facts.json",
  "prompt_format": "chat_5shot_cot_fewshot_unified",
  "filler_type": "counting",
  "cot_mixture": false,
  "seed": 0,
  "fewshot_examples": [
    {"q1": "At what age did X die?", "q2": "What is the atomic number of Y?",
     "a1": 35, "a2": 2, "sum": 37},
    "..."
  ],
  "n_fewshot": 5,
  "n_fewshot_facts": 10,
  "filler_mode": "eval",
  "eval_filler_lengths": [0, 128, 300, 600],
  "fact_counts": {"train": 228, "val": 38, "test": 39},
  "example_counts": {"train": 28000, "val": 600, "test": 600}
}
```

## Directory Structure

```
filler-token-reasoning/
├── scripts/                        # Main pipeline scripts
│   ├── fetch_facts.py
│   ├── generate_addition_dataset.py
│   ├── train_lora.py
│   ├── evaluate.py
│   └── knowledge_check.py 
├── dev/                            # Development/experimental scripts
│   └── ...
├── data/                           # Created by scripts (gitignored)
│   ├── sources/                    # Raw fact JSON files
│   ├── known_facts/                # Output of knowledge_check.py
│   │   ├── known_facts.json
│   │   ├── unknown_facts.json
│   │   └── fact_eval_summary.json
│   └── datasets/
│       └── 2hop_add/
│           ├── train.jsonl
│           ├── val.jsonl
│           ├── test.jsonl
│           ├── manifest.json
│           └── fact_pools.json
├── outputs/                        # Training checkpoints (gitignored)
├── results/                        # Evaluation results (gitignored)
├── pyproject.toml
└── README.md
```

## Citations

- Fact sources: [Ryan Greenblatt's compose_facts](https://github.com/rgreenblatt/compose_facts)
- Models: [Qwen Team](https://github.com/QwenLM/Qwen2.5)
- Training data mix inspired by [Pfau et al. (2024)](https://arxiv.org/abs/2404.15758)
