# FP8 check — does the filler-token decode result survive at canonical precision?

**Question.** Reviewer asks whether the results are an artifact of quantization. DeepSeek-V3
has no un-quantized release — it was trained in FP8 and DeepSeek ship only FP8 weights
("Since FP8 training is natively adopted in our framework, we only provide FP8 weights").
So the answerable version is: *does the effect reproduce on the canonical FP8 release?*

**Design.** Same problems, same filler strings, same token ids, same extraction code, same
decode pipeline, same metric. Only the model changes (AWQ W4A16 → native block-FP8).

Inputs are bit-identical between conditions by construction: `problems[:N]` (no shuffle) and
per-example filler seeded by `random.Random(prob_idx)`. The only thing that could break
pairing is the tokenizer — see Phase 2, check 1.

**Scope.** One condition: `2fact / dots_10`. It is the headline condition, has a published
heatmap, and ~244 correct examples. Only add `dots_50` if the first result is ambiguous.

---

## Pre-register the success criteria (fill in BEFORE looking at FP8 results)

AWQ reference, 2fact dots_10:

| Metric | AWQ value |
|---|---|
| Sonnet top-2 / top-10 "Both" (n=244) | 82% / 93% |
| Haiku top-2 / top-10 | 56% / 80% |
| Logit-lens A1 exact @ `pos_001` | 76% @ L43 |
| Logit-lens A2 exact @ `pos_005` | 62% @ L51 |
| Logit-lens sum exact @ `pos_016` | 92% @ L60 |
| Shuffled-token control (floor) | 0–1.4% |

Binomial SE at n=244, p=0.93 is ~1.6pp, so ±5pp is ~3σ.

- **Reproduces** — Sonnet top-10 within 5pp of 93%, and the heatmap peaks land in the same
  `(layer, position)` cells (A1@`pos_001`, A2@`pos_005`, sum@`pos_016`) with A1 exact within
  10pp. Conclusion: not a quantization artifact.
- **Ambiguous** — peaks in the right cells but magnitudes off by >10pp. Then run the W8A16
  variant to separate the weight change from the activation change.
- **Falsified** — A1 recovery collapses toward the shuffled-control floor (0–1.4%). The AWQ
  result was an artifact and the claim needs retraction.

Decide these now. Writing them down after seeing the numbers is worth nothing.

---

## Phase 1 — provision (GPU box)

**Hardware.** FP8 checkpoint is 688.6 GB (163 shards).

| Box | Total | Utilisation | Notes |
|---|---|---|---|
| 8×H200 | 1128 GB | 61% | most headroom on Hopper; `TP=8` valid (the reference config) |
| **6×H200** | **846 GB** | **81%** | **primary target** — works; `TP=6` invalid → `TP=2×PP=3` |
| 4×B300 | 1152 GB | 60% | roomiest, but Blackwell risk — see Known Risks |

`--precision fp8` already sets `gpu_frac=0.90` (0.80 would cap 6×H200 *below* the weight size
and spill to CPU). At 0.90 the aggregate cap is 761 GB against 688.6 GB of weights — 73 GB of
slack, or 88 GB if MTP is disabled. `device_map="auto"` balances rather than max-packs, so
expect ~10 MoE layers/GPU (~115 GB). If it does spill, raise `gpu_frac` to 0.93.

Headroom matters less here than the percentages suggest: prompts are **~97–140 tokens**
(dots_10 = 97), so activations and the MLA KV cache are ~1 GB, not tens of GB.

6 vs 8 H200 is a pure headroom question — identical GPU, kernels, dtype and numerics. Do not
wait for 8; the run is reversible and the checkpoint is already staged.

**Venv** — must be separate. The FP8 path needs a newer `transformers` than the
`transformers<5` + `autoawq` pin, and they cannot coexist.

```bash
uv venv /root/.venvs/fp8 --python 3.11
UV_CACHE_DIR=/workspace/.cache/uv uv pip install --python /root/.venvs/fp8/bin/python \
    transformers accelerate safetensors numpy tqdm triton tokenizers
```

**torch — check this against the GPU type.** The project pin is `torch==2.6.0+cu124`.
That is fine on H200 (SM90) but **will not work on B300** (Blackwell, SM100+), which needs
CUDA 12.8+ / a cu128 torch build. Install the matching wheel before anything else and record
the version in the log.

**Download — DONE (2026-07-24).** `/workspace/models/deepseek-v3-fp8`, 688.6 GB, 163/163
shards. All 173 files verified byte-for-byte against the HF manifest; shard headers confirm
`F8_E4M3` weights + `F32` `weight_scale_inv` on a 128×128 block grid. `/workspace` is the
persistent network volume, so this is already staged for whichever GPU box you attach.

Re-run / resume any time with `bash scripts/fp8/download_fp8.sh`. Took ~5 min at 2.5–2.8 GB/s
with `hf_transfer` + 8 workers (a single-stream `curl` measured only 23 MB/s — ignore that
number, it is ~100× pessimistic).

Do **not** copy the official `tokenizer.json` over the AWQ one — see below.

**MTP module.** 688.6 GB = **673.2 GB main model + 15.4 GB Multi-Token-Prediction**
(`model.layers.61`, 1564 tensors). This experiment has no use for it. `config.json` sets
`num_nextn_predict_layers: 1` and transformers' `num_mtp_layers` also defaults to 1, so it
will probably load. Preflight check 4 reports whether it did; setting it to `0` reclaims
15.4 GB, which is worth having on the 6-GPU box.

---

## Phase 2 — preflight (do not skip)

```bash
# cheap checks, safe to run mid-download
/root/.venvs/fp8/bin/python scripts/fp8/preflight_fp8.py --no-load

# full, after download completes (~30-60 min to load)
/root/.venvs/fp8/bin/python scripts/fp8/preflight_fp8.py
```

What each check buys you:

1. **Tokenizer parity** — *the one that already bit us.* The AWQ repo's `tokenizer.json` uses
   `post_processor=TemplateProcessing` and prepends BOS; the official DeepSeek repo's uses
   `post_processor=ByteLevel` and does not. Vocab and the 0–299 number-token map are
   **identical**; verified on real prompts, the sole difference is that one leading token.
   Left unfixed every position shifts by one and `pos_001` (where A1 peaks) becomes `pos_000`
   — the comparison would look like a null result. **Fix: always pass
   `--tokenizer-path /workspace/models/deepseek-v3-awq`.**
2. **Config parity** — 61 layers / 7168 / 128 heads / 256 experts / vocab 129280, and
   `quant_method: fp8` actually declared.
3. **Model load** — per-GPU allocation, and asserts nothing landed on CPU/disk.
4. **Hook shape** — `model.model.layers[i]` returns `(B, S, 7168)`. The existing hook already
   handles tuple-or-tensor returns, so the native-transformers layer-return change is covered.
5. **Readout instrument** — compares FP8 `lm_head`/`norm` against the saved AWQ `.npy`. AWQ
   leaves `lm_head` unquantized, so these should be ~identical; if so the logit lens is
   literally the same instrument in both conditions and any decode difference is attributable
   to the residual states alone. Informational, not fatal.
6. **Logit-lens closure** — `rms_norm(h_60) @ lm_head.T` must reproduce the model's own
   logits. Catches wrong layer indexing, wrong norm, dtype bugs.
7. **Generation smoke** — model produces sane output at all.

---

## Phase 3 — extraction

```bash
mkdir -p logs
cd /workspace/filler-token-reasoning && setsid nohup /root/.venvs/fp8/bin/python \
  scripts/extract/extract_hidden_states.py \
    --model-path /workspace/models/deepseek-v3-fp8 \
    --precision fp8 \
    --tokenizer-path /workspace/models/deepseek-v3-awq \
    --dataset data/2fact_addition_dataset.json \
    --dataset-type 2fact \
    --output-dir data/extracted_states_2fact_allpos_fp8 \
    --conditions dots_10 \
    --all-positions --layers all \
    --max-problems 500 \
  > logs/extract_fp8_dots10.log 2>&1 < /dev/null &
```

**Why 500 and not 1000.** The AWQ run used `--max-problems 1000` (indices 0-999, verified:
1000 `.pkl` files per condition). Selection is `problems[:N]` and `skip_existing=True` checks
per problem, so **running 500 now and 1000 later costs exactly the same as running 1000 once**
— the second pass fills in 500-999 and skips the rest. At ~24% task accuracy, 500 problems
gives ~122 correct, SE ~2.3pp, which is already decisive for the actual question (does the
effect reproduce at ~93%, or collapse to the 0-1.4% shuffled floor?). Extend to 1000 only if
you want the tighter interval or an N matching the published table.

**Deferred to analysis:** AWQ and FP8 will get *different* problems right, and the decode
metrics use correct-only subsets. Report task accuracy per condition as its own line, and do
the decode comparison on the **intersection** (both-correct) so the example set is identical.
Nothing about extraction changes — this is purely a Phase 4 decision.

- Output dir is **new** — the AWQ states are never touched.
- `lm_head`/norm are written to `data/model_weights/deepseek_v3_fp8/` (the `_fp8` suffix is
  automatic under `--precision fp8`), so the AWQ readout weights are preserved.
- Kill by the real python PID, not `$!`.
- Expect roughly 4–8 h for ~600 problems. HF's native DeepSeek-V3 is not a fast
  implementation; this is correctness-first, not throughput.

---

## Phase 4 — decode and compare (CPU, no GPU needed)

Run the standard pipeline against the FP8 states, then the head-to-head:

```bash
python scripts/decode/extract_residual_fingerprints.py   # --states-dir ..._fp8
python scripts/decode/aggregate_residuals_all_settings.py
python scripts/decode/llm_decode_batch.py --task 2fact --prompt neutral
python scripts/analysis/decode_2fact_heatmap.py          # both conditions
```

Report AWQ and FP8 side by side on the same examples, plus the shuffled-token floor for scale.

---

## Known risks

- **B300 + DeepGEMM.** `transformers/integrations/finegrained_fp8.py` disables DeepGEMM when
  a model spans multiple CUDA devices ("DeepGEMM's cached kernels are bound to a single CUDA
  context and produce garbage across devices"), so we get the Triton fallback — fine. But the
  same file notes a DeepGEMM/Triton interaction that "degrades end-to-end generation on B200."
  Preflight check 7 plus a real accuracy eval should catch it. `TRANSFORMERS_DISABLE_DEEPGEMM_LINEAR=1`
  forces Triton if anything looks off.
- **MTP module.** The 688.6 GB includes 14B of Multi-Token-Prediction weights on top of the
  671B main model. If `DeepseekV3ForCausalLM` skips them you save ~14 GB; check the
  missing/unexpected-keys warning at load and note which happened.
- **Attention implementation differs** (`sdpa` for fp8 vs `eager` for AWQ). Numerically
  equivalent up to accumulation order, but it is a difference — note it, and switch fp8 to
  `eager` if you want to eliminate it (costs memory: 128 heads materialised).
- **`transformers` version drift.** Record the exact version in the log; the native
  `deepseek_v3` implementation is under active development and its expert/attention kernels
  have changed.
