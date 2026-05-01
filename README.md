# Decoding Hidden Computation in Filler Tokens

Investigates whether filler tokens (e.g., `. . . . .` or `1 2 3 4 5`) inserted between a question and an answer give large language models additional computation depth and what the model is actually computing during that span. This repo contains scripts for experiments on instruction-tuned MoE models: hidden-state decoding via logit lens (with an LLM judge over residual top tokens), KV-cache transplant interventions, and attention-pattern analysis.

## Tasks

Three tasks share the same prompt scaffold (5 few-shot examples held out of any target pool, then a target question, then `Filler:` filler tokens, then `Answer:`):

| task | prompt template | what the model computes |
|---|---|---|
| 1-fact addition | "What is [fact phrase] plus [X]?" | look up A, return A + X |
| 2-fact addition | "What is [fact phrase 1] plus [fact phrase 2]?" | look up A₁ and A₂, return A₁ + A₂ |
| letter-position | "What is the [Nth] letter of the chemical element with atomic number [Z]?" or "...the capital of [region]?" | look up entity (element name from atomic number, or capital from a country, state, province, or territory), return its [Nth] letter |

Filler types: `dots` (`. . .`), `counting` (`1 2 3 ...`), `alphabet` (`a b c ...`); filler lengths from k=5 to k=100. Datasets and held-out few-shot pools live in `data/`.

## Datasets

We include data files for 2-fact addition and the two letter-position domains. The 1-fact addition dataset must be regenerated locally because it draws facts from Ryan Greenblatt's [`compose_facts`](https://github.com/rgreenblatt/compose_facts) repo which are not to be pushed to github so they can continue to be used as test questions for future LLMs.

| dataset | file | provided? | source / regenerate with |
|---|---|---|---|
| 1-fact addition | `data/1hop_addition_dataset.json` | no | `scripts/data/generate_1hop_dataset.py` (needs a `known_facts.json` from a model-specific knowledge check) |
| 2-fact addition | `data/2fact_addition_dataset.json` | yes | `scripts/data/generate_2fact_dataset.py` |
| Letter-position (elements) | `data/element_letter_positions.json` | yes | `scripts/data/generate_letterpos_dataset.py` |
| Letter-position (capitals) | `data/capital_letter_position.json` | yes | hand-curated; no generator |
| Element multilingual aliases | `data/element_aliases.json` | yes | `scripts/data/build_element_aliases.py` (Wikidata) |
| Capital multilingual aliases | `data/capital_aliases.json` | yes | `scripts/data/build_capital_aliases.py` (Wikidata + manual fixes) |

The fact pool for 1-fact comes from Ryan Greenblatt's [`compose_facts`](https://github.com/rgreenblatt/compose_facts) repo (atomic numbers, ages at death, hand-curated static counts). To regenerate the 1-fact dataset end-to-end:

```bash
# 1. Download upstream fact JSONs
python finetuning/scripts/fetch_facts.py --outdir data/sources

# 2. Filter to facts the target model reliably knows
python finetuning/scripts/knowledge_check.py \
    --model /workspace/models/deepseek-v3-awq \
    --sources data/sources \
    --outdir data/facts/knowledge_deepseek_v3

# 3. Generate the 1-fact addition dataset
python scripts/data/generate_1hop_dataset.py \
    --known-facts data/facts/knowledge_deepseek_v3/known_facts.json \
    --output data/1hop_addition_dataset.json
```

## Models

Two instruction-tuned MoE models, run from publicly-released 4-bit checkpoints on Hugging Face:

| Hugging Face repo | parameters | quantization |
|---|---|---|
| [`cognitivecomputations/DeepSeek-V3-0324-AWQ`](https://huggingface.co/cognitivecomputations/DeepSeek-V3-0324-AWQ) | 671B total / 37B activated | AWQ 4-bit, group 128 |
| [`RedHatAI/Kimi-K2-Instruct-quantized.w4a16`](https://huggingface.co/RedHatAI/Kimi-K2-Instruct-quantized.w4a16) | 1T total / 32B activated | W4A16 (compressed-tensors), group 128 |

Download each checkpoint to any local directory and pass the path to scripts via `--model-path`.

DeepSeek extraction uses the HF transformers + autoawq stack in `scripts/extract/`. Kimi K2 extraction goes through vLLM (TP=4) in `scripts/kimi_k2/`.

## Analysis pipeline

The three analyses share the model, dataset, prompt scaffold, and filler-boundary detection logic, but each runs its own forward passes — hidden-state decoding caches per-(layer, position) states to disk for offline decoding, while KV transplant and attention analysis capture the quantities they need in-memory during the run.

### 1. Hidden-state decoding (all three tasks)

Four stages, the first on GPU and the rest CPU-only:

```bash
# (a) Cache hidden states at every (layer, position) for each example
python scripts/extract/extract_hidden_states.py \
    --model-path /workspace/models/deepseek-v3-awq \
    --dataset data/2fact_addition_dataset.json \
    --conditions dots_10 dots_25 dots_50 counting_10 counting_25 counting_50 \
    --output-dir data/extracted_states_2fact_allpos

# (b) Per-(layer, position) residual fingerprints (P_e − mean_e P_e, top-50 token IDs)
python scripts/decode/extract_residual_fingerprints.py \
    --extraction-dir data/extracted_states_2fact_allpos \
    --condition dots_10 \
    --model-path /workspace/models/deepseek-v3-awq \
    --output outputs/deepseek_aggregated/fingerprints_dots_10.npz

# (c) Aggregate residuals across (layer, position) into per-example top-50
python scripts/decode/aggregate_residuals_all_settings.py \
    --fingerprints outputs/deepseek_aggregated/fingerprints_dots_10.npz \
    --model-path /workspace/models/deepseek-v3-awq \
    --output outputs/deepseek_aggregated/aggregated_dots_10.json

# (d) LLM-as-judge over the aggregated top-50 tokens (Claude Haiku 4.5 + Sonnet 4.6)
python scripts/decode/llm_decode_batch.py \
    --aggregated outputs/deepseek_aggregated/aggregated_dots_10.json \
    --task 2fact --prompt neutral
```

Convenience drivers for the two letter-position domains: `scripts/run_letterpos_fingerprints.sh` (elements) and `scripts/run_capitalpos_fingerprints.sh` (capitals). Multilingual coverage of the per-example top-K tokens (substring match against EN + ZH name tables) is `scripts/analysis/multilingual_coverage.py`, with reference tables in `data/element_aliases.json` and `data/capital_aliases.json`.

**Logit-lens heatmaps (addition tasks).** A direct, supervised view of the same extraction cache: at every (layer, position) decode by RMSNorm + lm\_head + argmax over single-token integers 0–299, then compute the fraction of examples where the prediction matches each known target (`A₁`, `A₂`, `A₁+A₂` for 2-fact; `A`, `X`, `A+X` for 1-fact) exactly or within ±5. Produces the (layer, position) heatmaps used in the paper to localize the staged "look up, look up, sum" computation.

```bash
# Per-(layer, position) decode, written to a JSON keyed by (pos, layer)
python scripts/analysis/decode_2fact_heatmap.py --condition dots_50

# Same for the model-incorrect subset
python scripts/analysis/decode_2fact_heatmap.py --condition dots_50 --incorrect-only

# Paper-ready 3x2 figure (rows: A1 / A2 / A1+A2; cols: correct / wrong)
python plotting/plot_decode_heatmap_correct_vs_wrong.py \
    --correct-json results/unsupervised_decode_2fact_allpos/decode_2fact_dots_50.json \
    --incorrect-json results/unsupervised_decode_2fact_allpos/decode_2fact_dots_50_incorrect.json \
    --metric exact \
    --outfile-stem plotting/plots/decode_2fact_dots_50_correct_vs_wrong_exact
```

### 2. KV-cache transplant (1-fact addition)

Splices the donor's filler-region KV cache into the target's KV state at every layer and decodes the next token; reports the rank of the donor's answer (donor A + X) in the target's predictions before and after the swap.

```bash
python scripts/intervention/filler_kv_transplant.py \
    --model-path /workspace/models/deepseek-v3-awq \
    --dataset data/1hop_addition_dataset.json \
    --conditions dots_10 dots_100 counting_25 \
    --n-pairs 500 \
    --output-dir results/kv_transplant_v4
```

### 3. Attention analysis (1-fact addition)

Aggregates per-head attention weights into a segment-level breakdown (system / few-shot / question / filler region / answer-label) across 100 examples per condition, plus a no-filler baseline.

```bash
python scripts/extract/extract_filler_attention.py \
    --model-path /workspace/models/deepseek-v3-awq \
    --max-examples 100 \
    --output-dir results/filler_attention_v2

python scripts/extract/extract_answer_attention.py \
    --model-path /workspace/models/deepseek-v3-awq \
    --max-examples 100 \
    --output-dir results/attention/baseline
```

The "filler region" is the filler header tokens plus the filler tokens themselves, treated as one segment.

## Repository structure

```
scripts/
  data/            # dataset generation + alias tables
                   #   generate_{1hop,2fact,letterpos}_dataset.py
                   #   build_{element,capital}_aliases.py (Wikidata SPARQL + cleanup)
  extract/         # hidden state and attention extraction (GPU; DeepSeek)
                   #   extract_hidden_states.py
                   #   extract_filler_attention.py / extract_answer_attention.py
                   #   extract_kv_latents.py
  decode/          # residual fingerprints + aggregation + LLM judge
                   #   extract_residual_fingerprints.py
                   #   aggregate_residuals_all_settings.py
                   #   llm_decode_batch.py
  intervention/    # causal interventions
                   #   filler_kv_transplant.py
                   #   attention_knockout.py
                   #   filler_patching.py / baseline_patching.py
  eval/            # vLLM-based accuracy evals
                   #   evaluate_{1hop,2fact,letterpos}_vllm.py
                   #   evaluate_truncation.py, save_categories.py
  analysis/        # post-hoc analyses
                   #   decode_2fact_heatmap.py    (logit-lens decode for 2-fact)
                   #   multilingual_coverage.py   (substring match for letterpos)
                   #   layer_attribution.py
  kimi_k2/         # Kimi K2-specific extraction (vLLM, TP=4)
  dev/, tests/     # archived / experimental / unit tests

plotting/          # all figure generation (Matplotlib)
data/              # datasets, held-out few-shot pools, alias tables
results/           # analysis outputs (JSON + plots)
logs/              # script logs
finetuning/        # Qwen2.5-72B LoRA fine-tuning (separate experiment)
```

## Setup

Python ≥ 3.10 and a CUDA-capable GPU node are required for any extraction or intervention script. CPU is sufficient for the residual-decode aggregation, the LLM-judge driver, and all plotting.

### 1. Install dependencies

The project uses [uv](https://github.com/astral-sh/uv) for environment management:

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# In the repo root:
uv sync                                    # creates .venv and installs everything in pyproject.toml
uv pip install flash-attn --no-build-isolation   # optional: only if you want Flash Attention 2
```

`uv sync` pins `torch==2.6.0+cu124` from the PyTorch index plus `transformers`, `accelerate`, `bitsandbytes`, `autoawq`, `vllm`, `anthropic`, `matplotlib`, etc. Activate the env with `source .venv/bin/activate` or prefix commands with `uv run`.

### 2. Download model checkpoints

Each model is ~330–510 GB on disk. Use `huggingface-cli` (installed by `uv sync`) and pass the local path to scripts via `--model-path`:

```bash
huggingface-cli download cognitivecomputations/DeepSeek-V3-0324-AWQ \
    --local-dir /path/to/deepseek-v3-awq

huggingface-cli download RedHatAI/Kimi-K2-Instruct-quantized.w4a16 \
    --local-dir /path/to/kimi-k2-w4a16
```

Both checkpoints are public; no `HF_TOKEN` required.

### 3. Environment variables

Only one is strictly required:

| variable | when needed |
|---|---|
| `ANTHROPIC_API_KEY` | LLM-as-judge step (`scripts/decode/llm_decode_batch.py`) |
| `HF_HOME` (optional) | redirect HuggingFace cache (model downloads, tokenizer assets) to a non-default location |
| `HF_TOKEN` (optional) | only needed if you swap in a gated model |

### 4. Hardware

3–4 × NVIDIA H200 (150 GB HBM3) for the experiments in this repo: DeepSeek V3 fits across 3 H200s in 4-bit AWQ; Kimi K2 needs 4 H200s under W4A16 with vLLM TP=4. Cumulative hidden-state extraction cache for the three-task suite is ~700 GB on local SSD. Smaller setups can run a subset of conditions; the residual-decode step alone is CPU-only after extraction.