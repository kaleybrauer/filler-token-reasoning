# Kimi K2 re-extraction runbook — fixing the capture-convention bug

**Status:** code fix applied (uncommitted, see §0). Extraction NOT yet run.
**Owner:** hand this file to the agent that gets the 4×H200 pod.
**Written:** 2026-08-18.

---

## 1. Why this run exists

The original Kimi K2 extractions saved **per-layer MLP writes**, not the residual stream.

`scripts/kimi_k2/extract_hidden_states_vllm.py` hooks **vLLM** decoder layers and took
`output[0]`. vLLM's `DeepseekV2DecoderLayer.forward`
(`vllm/model_executor/models/deepseek_v2.py:548-590`, v0.8.5) uses the *split-residual*
convention and returns `(mlp_write, stream_after_attention)`; the final add happens
**outside** the layer at `:667` (`hidden_states, _ = self.norm(hidden_states, residual)`).
So the residual stream after layer L is `output[0] + output[1]`, and `output[1]` was never
saved.

The hook was ported from the transformers path, where HF's `DeepseekV3DecoderLayer`
(`modeling_deepseek.py:1143`) adds the residual **inside** the layer and returns a 1-tuple —
so there `output[0]` genuinely *is* the stream. Same local variable name (`hidden_states`),
opposite meaning. Nothing failed loudly, which is why it survived.

Confirmed three independent ways: the source above; a 57-condition norm sweep (V3 ×26 and
K2.5 ×13 grow 56–100× by three-quarters depth, Kimi K2 ×18 grow **1.0×**); and K2.5 as a
control — same engine, same DeepseekV3-style 61×7168 architecture, only the hook differs, and
it lands in the residual cluster.

**Affected:** Kimi K2 `2fact`, `letterpos`, `capitalpos` only.
**Not affected:** all DeepSeek V3 extractions (transformers path); all K2.5 extractions
(`kimi_k25_hooks.py:105` sums). All *behavioural* numbers everywhere (generation, no hooks).

---

## 2. What the paper needs back

Re-extracted Kimi K2 rows for:

| release artifact | what to regenerate |
|---|---|
| `release/accuracy_tables/llm_judge_kimi_k2_*.json` | 5 files (2fact_neutral, letterpos_{neutral,chemistry}, capitalpos_{neutral,geography}) |
| `release/tables/table_2fact.tex` | the *Kimi K2* block + pooled row |
| `release/tables/table_letterpos_leading.tex` | Kimi K2 Elements + Capitals blocks |
| `release/tables/table_letterpos_neutral.tex` | Kimi K2 Elements + Capitals blocks |
| `release/tables/table_shuffled_control.tex` | all Kimi K2 rows |
| `release/top_tokens/`, `release/judge_outputs/` | the Kimi K2 files feeding the above |

`release/logit_lens_heatmaps/` is **V3-only** — nothing to redo there.

---

## 0. Code state (READ FIRST)

The pod mounts `/workspace`, so work in this repo in place —
`/workspace/filler-token-reasoning`. No clone needed, and this runbook is deliberately
untracked (single-use); read it from that path.

Both code changes are **already committed**:

1. `f20b6aa` `scripts/kimi_k2/extract_hidden_states_vllm.py` — the hook now sums
   `output[0] + output[1]` when the layer returns a same-shape 2-tuple, falls back to
   `output[0]` for the 1-tuple/bare-tensor (transformers) case, prints the convention once
   per run, and stamps `capture_convention` into **every pkl**.
2. `5093e1c` `scripts/kimi_k2/verify_capture_convention.py` — the gate (§4).

Sanity check on the pod before launching:

```bash
python3 -m py_compile scripts/kimi_k2/extract_hidden_states_vllm.py
grep -n "sum(output\[0\], output\[1\])" scripts/kimi_k2/extract_hidden_states_vllm.py   # must hit
```

---

## 3. Pod spec

- **GPUs:** 4× H200 (141 GB). `TP=4` — **not 5**: 64 attention heads must divide TP.
- **Model:** `/workspace/models/kimi-k2-w4a16` = `RedHatAI/Kimi-K2-Instruct-quantized.w4a16`,
  ~510 GB, 63 shards. (Do **not** use `QuixiAI/Kimi-K2-Instruct-AWQ` — broken checkpoint,
  60/61 `post_attention_layernorm.weight` tensors contain bf16 `inf`.)
- **Disk:** new states ≈ **521 GB** (2fact 351 G, capitalpos 95 G, letterpos 75 G). Keeping
  the old ones alongside (recommended, §7) needs ~1.05 TB. Budget accordingly.
- **Engine:** vLLM **0.8.5**, `transformers>=4.48,<5` (5.x breaks this path),
  plus `tiktoken` and `setuptools`.
- **Wall-clock estimate:** ~2.1–2.6 s/example.
  - 2fact 6×1000 = 6,000 ex → ~4 h
  - letterpos 6×285 = 1,710 ex → ~1 h 05 m
  - capitalpos 6×362 = 2,172 ex → ~1 h 20 m
  - model load: ~5 min warm, **~1 h 08 m cold** (MFS read-bound, ~65 s/shard)
  - **total ≈ 7–8 h** if bundled into one load.

---

## 4. Procedure

### Step 0 — environment

```bash
source /workspace/config/probing_env.sh
PY=/root/.venvs/filler-probing/bin/python
$PY -c "import vllm, tiktoken, setuptools; print(vllm.__version__)"   # expect 0.8.5
```

Apply the vLLM 0.8.5 registry patch if the venv is fresh (see
`feedback_vllm_registry_patch`): `inspect_model_cls()` must return a default `_ModelInfo`
instead of raising, the five `is_*_model` helpers need try/except, and **delete all `.pyc`**
in the vllm package afterwards.

### Step 1 — smoke run, ONE condition, 20 examples

Do not launch the full run first. This costs one model load and settles the fix.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 NCCL_NVLS_ENABLE=0 VLLM_USE_V1=0 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
$PY scripts/kimi_k2/extract_hidden_states_vllm.py \
  --model-path /workspace/models/kimi-k2-w4a16 \
  --dataset data/2fact_addition_dataset.json --dataset-type 2fact \
  --output-dir data/kimi-k2/SMOKE_2fact_fixed \
  --conditions dots_10 --max-problems 20 --all-positions \
  --tensor-parallel-size 4 --gpu-memory-utilization 0.97 --max-model-len 1024 \
  2>&1 | tee logs/kimi_k2_smoke_fixed.log
```

Expect this line in the log — if it is absent or says `output[0]`, **stop**:

```
[LayerCapture] capture convention = sum(output[0], output[1])  [residual stream]
```

### Step 2 — GATE (must pass before the full run)

```bash
python3 scripts/kimi_k2/verify_capture_convention.py data/kimi-k2/SMOKE_2fact_fixed
echo "exit=$?"    # must be 0; do not pipe through tee/tail, it masks the code
```

Pass criterion: norm growth from L0 to three-quarters depth **≥ 20×**.
Calibration from real data — the separation is not marginal:

| | growth |
|---|---|
| broken K2 extractions (18 conds) | **1.0 – 1.1** |
| DeepSeek V3 (26 conds) | 56 – 100 |
| Kimi K2.5 (13 conds) | 113 – 123 |

A correct Kimi K2 run should land near K2.5 (~100×). Anything under 20× means the fix did
not take — **abort and debug, do not spend the 7 hours.**

Also confirm accuracy in the smoke log roughly matches the recorded behavioural baseline
(2fact dots_10 ≈ 31.7%) — the fix must not change generation at all, only what is captured.

### Step 3 — full extraction

2fact (n=1000, 6 conditions):

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 NCCL_NVLS_ENABLE=0 VLLM_USE_V1=0 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
nohup setsid $PY scripts/kimi_k2/extract_hidden_states_vllm.py \
  --model-path /workspace/models/kimi-k2-w4a16 \
  --dataset data/2fact_addition_dataset.json --dataset-type 2fact \
  --output-dir data/kimi-k2/extracted_states_2fact_allpos_kimi_k2_v2 \
  --conditions dots_10 dots_25 dots_50 counting_10 counting_25 counting_50 \
  --max-problems 1000 --all-positions \
  --tensor-parallel-size 4 --gpu-memory-utilization 0.97 --max-model-len 1024 \
  > logs/kimi_k2_2fact_reextract.log 2>&1 &
```

letterpos + capitalpos in a single load (the script takes `nargs='+'` for both
`--dataset` and `--output-dir`); adapt `run_letterpos_capitalpos_extract.sh`, changing only
the two output dirs to `..._v2`:

```bash
nohup setsid bash scripts/kimi_k2/run_letterpos_capitalpos_extract.sh \
  > logs/kimi_k2_letterpos_capitalpos_reextract.log 2>&1 &
```

Use `nohup setsid` — background jobs are not durable across session restarts. Kill by the
real python PID, not `$!`.

### Step 4 — verify the full output

```bash
python3 scripts/kimi_k2/verify_capture_convention.py \
  data/kimi-k2/extracted_states_2fact_allpos_kimi_k2_v2 \
  data/kimi-k2/extracted_states_letterpos_allpos_kimi_k2_v2 \
  data/kimi-k2/extracted_states_capitalpos_allpos_kimi_k2_v2
```

All 18 conditions must report `RESIDUAL STREAM ok` and show a stamped
`capture_convention`. Also check per-condition example counts (1000 / 285 / 362) and that
accuracies match the recorded behavioural table.

### Step 5 — downstream

Unchanged scripts, new input dirs:

```
scripts/decode/extract_residual_fingerprints.py
  → scripts/decode/aggregate_residuals_all_settings.py
  → scripts/decode/llm_decode_batch.py --task 2fact|letterpos|capitalpos
```

Pass `--lm-head data/model_weights/kimi_k2/lm_head_weight.npy
--rms-norm data/model_weights/kimi_k2/rms_norm_weight.npy
--model-path /workspace/models/kimi-k2-w4a16`. Judge prompts as before: `neutral` for 2fact,
`neutral`+`chemistry` for letterpos, `neutral`+`geography` for capitalpos. Then re-stage with
`scripts/release/stage_release.py`.

**Do not run fingerprints + aggregation + LLM decode concurrently on one host** — CPU
contention. Fingerprint wall-time scales with `n_layers × n_filler_positions`; the previous
capitalpos pass took 2 h 15 m on a 32-core pod, and counting_50 alone was 51 min.

---

## 5. What to report back

1. The gate table for all 18 conditions.
2. Old-vs-new pooled judge accuracy, side by side, per task/condition/judge/top-K.
3. Whether any qualitative claim moves.

### ⚠️ Do not predict the direction — the gate certifies the run, not the accuracy

Two mechanisms pull opposite ways:

- Writes are blind to attention-carried and already-resident content, and 2-hop lookup is
  attention-mediated → writes should *under*-report.
- Writes are **denoised**: the stream accumulates to norm 400–900 carrying everything, while
  an MLP write is sparse and purposive, and FFN outputs are the canonical vocabulary-shaped
  component (Geva 2021/2022) → writes should *over*-report.

**Which dominates is not settled, and is evidently task-dependent.** Kimi-writes minus
V3-stream (letter position = elements + capitals POOLED — they are two variants of one task,
never reported separately):

| task | Δ (Kimi writes − V3 stream), K=1…10 |
|---|---|
| 2-fact | **+12 to +17.5 pt** (Haiku), ~+4 (Sonnet) — writes ahead |
| letter position | **−2.0 to −2.3** (Haiku), −2.1 to −3.3 (Sonnet), −3.7 to −6.0 (direct match) — stream ahead |

There is no general "writes decode better" effect to extrapolate: writes lead on 2-fact and
trail on letter position.
That cross-model comparison is doubly confounded anyway: different capture operators *and*
different model-correct subsets (2-fact n=330 for Kimi vs 244 for V3 — not the same examples).

So the re-extracted Kimi numbers may move up **or** down. Do not treat either direction as
evidence the run succeeded or failed — **the §4 gate is the only correctness criterion.**
Record both sets regardless; the delta is a reportable result in its own right.

---

## 6. Traps (all previously hit on this exact pipeline)

- **`setuptools` missing** in a fresh venv → triton import fails → vLLM reports
  `failed to be inspected` for `DeepseekV3ForCausalLM`. Install it explicitly.
- **`tiktoken` missing** → Kimi's custom `TikTokenTokenizer` won't load. Needed even for
  CPU-only fingerprint extraction.
- **transformers ≥5** breaks this path. Pin `>=4.48,<5`.
- **`NCCL_NVLS_ENABLE=0`** is required on some RunPod H200 instances.
- **`gpu_memory_utilization=0.97`** — weights take 130.13 GiB of 139.81 GiB per GPU; the
  default 0.9/0.95 leaves a negative KV budget and fails with "No available memory for cache
  blocks".
- **`max_model_len=1024`** covers dots/counting up to k≈100 with 5 few-shot (dots_100 peaks
  at 889 tokens including 20 generated).
- **Decode-overwrite guard**: the hook keeps only the first forward per layer after
  `clear()` (prefill). Later calls are per-token decodes of shape `(1, hidden)`. Already
  handled — do not remove.
- **Do not "simplify" the hook back to `output[0]`.** That is the bug. The in-code comment
  explains why; keep it.

---

## 7. Do NOT

- **Do not delete the old extractions — Kaley's explicit decision (2026-08-18): keep both
  sets.** Do not rename them either; other scripts and notes reference the current paths.
  The originals stay at `data/kimi-k2/extracted_states_{2fact,letterpos,capitalpos}_allpos_
  kimi_k2/` and the new stream states go to the parallel `..._v2/` dirs (§3). Drop a
  `README_CAPTURE.txt` in each original dir recording that it holds per-layer MLP writes.
  Storage: ~521 GB existing + ~521 GB new ≈ 1.04 TB.

### This gives the apples-to-apples operator comparison for free
The old extractions **are** the MLP-write arm; the new run is the stream arm. Same model,
same datasets, same conditions, same positions — and generation is greedy
(`SamplingParams(temperature=0)` at `extract_hidden_states_vllm.py:303,326-327`), so the
model-correct subsets should be identical and the two arms are paired per `problem_idx`.
That is a strictly better comparison than anything reconstructable from DeepSeek V3, where
only `h_L - h_{L-1}` (attention **+** MLP) is recoverable and MLP-only is not.
Caveat to handle at analysis time, not run time: greedy is deterministic in principle, but
MoE + tensor-parallel reduction order can flip a token occasionally. **Pair on `problem_idx`
and report on the intersection of examples correct in both runs**; log how many differ rather
than assuming zero. Keep TP=4, the same quantized checkpoint and vLLM 0.8.5 so generation
matches.
- **Do not re-run the DeepSeek V3 or Kimi K2.5 extractions.** They are correct.
- **Do not put stream numbers and write numbers in the same table column.** That mixed
  operator is the confound this run exists to remove.
