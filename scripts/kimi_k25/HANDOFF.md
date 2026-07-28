# HANDOFF — Kimi K2.5 GPU session (8×H200)

You are running one experiment on a rented 8×H200 pod at ~$25–30/hour. Read
`scripts/kimi_k25/RUNBOOK.md` first — it has the verified facts, the exact flags, and the
reasoning behind every choice below. This file is the operating brief.

## Mission

Run the **system-of-equations (varbind) task** on **Kimi K2.5** with reasoning suppressed:
download the checkpoint, verify the prompts contain no open thinking block, measure the
filler-token accuracy uplift, and extract all-position hidden states for 8 conditions × 500
examples. Then ship the artifacts off the pod and terminate it.

Reference result to replicate the *shape* of (DeepSeek-V3 AWQ, same dataset): baseline 34.4%,
dots_25 65.8% — a +24 to +31 pt filler uplift, dots ≈ counting, saturating by k≈10–25.

## Hard constraints

- **No KV transplants, no interventions.** Behavioral eval + hidden-state extraction only.
- **No git commits, no pushes, no co-author trailers.** Kaley controls the repo.
- **Never delete extracted states, results, or logs.** GPU time is expensive; data is not
  reproducible for free.
- **Instant mode only.** Thinking is disabled everywhere via the chat template's
  `thinking=False` → `<think></think>`. Do not add a thinking-mode condition.
- **If a gate fails, STOP and report** with the exact log lines. Do not disable a gate, do not
  change the capture convention, do not "try the other thing and see" on a $30/hour pod.

## Decisions already made — do not re-litigate

8×H200 TP=8 (the checkpoint is 595 GB and 4 GPUs cannot hold it) · vLLM 0.25.1 · the official
`moonshotai/Kimi-K2.5` INT4 checkpoint (there is no BF16 release; the smaller forks are
expert-pruned, i.e. a different model) · 8 conditions × 500 examples matching the V3 set ·
extraction captures `output[0] + output[1]` (see RUNBOOK §A4).

## Environment

```bash
# venv on the pod's local disk, never on a network mount
uv venv --python 3.12 /root/.venvs/k25
uv pip install --python /root/.venvs/k25/bin/python vllm==0.25.1 tiktoken blobfile setuptools
# use the venv python DIRECTLY for anything vLLM — `uv run` re-resolves torch and breaks it
```

- `setuptools` is mandatory (triton imports it at runtime; without it vLLM startup fails).
- `NCCL_NVLS_ENABLE=0` if multi-GPU NCCL hangs on this host.
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- Point `HF_HOME` at the big disk so the xet cache doesn't fill the container root.
- All logs go to `logs/`, not the repo root.
- Anything over ~20 minutes runs under `nohup`/`setsid` — background shells do not survive a
  session restart. Kill by the real python PID, not `$!`.

## Order of operations

| # | Step | Budget | Gate before continuing |
|---|---|---|---|
| 1 | `nohup hf download moonshotai/Kimi-K2.5 --local-dir /models/kimi-k2.5` (start this FIRST, it runs in the background) | 30–90 min | 64 shards present, sizes match the HF blob listing |
| 2 | venv + `git clone https://github.com/kaleybrauer/filler-token-reasoning` (public) | 10 min | `import vllm` works |
| 3 | Tiny-model smoke: `tiny-random/kimi-k2.5` (9.5 MB, same architecture) with `--tokenizer` pointing at the real tokenizer files | 20 min | hooks fire, captured layer count == n_layers, pkl schema matches the reference |
| 4 | Load the real model (RUNBOOK §3 flags) | 10–20 min | loads without OOM; ~74 GB/GPU |
| 5 | Preflight gates | 10 min | **all five** — see below |
| 6 | Behavioral sweep, 8 conditions × 500, batched | ~10 min | per-condition accuracy printed and saved |
| 7 | Extraction, 8 conditions × 500, all positions | 3.5–5 h | first-example shapes + no NaN; progress logged per condition |
| 8 | Ship artifacts out, verify, terminate | 30–60 min | checksums verified on the receiving side |

**The five preflight gates (step 5):**

1. Rendered prompt ends `…<|im_middle|><think></think>Filler: …\n\nAnswer:` — token ids 163606
   immediately followed by 163607, and no lone 163606 anywhere in the prompt.
2. Free-running generation on 5 examples contains no `<think>` token.
3. Filler span per condition == expected (`dots_k` → k, `counting_k` → 2k−1). This is the only
   guard that the boundary finder located the right region.
4. **Capture-convention gate:** RMSNorm→lm_head on the captured last-layer state at
   `answer_prompt` argmaxes to the token vLLM actually generated. Near-100% expected. If this
   fails, the hook is capturing layer writes instead of the residual stream — stop and report.
5. `language_model.lm_head.weight` + `language_model.model.norm.weight` saved from shard 62 to
   `data/model_weights/kimi_k25/{lm_head_weight,rms_norm_weight}.npy`.

## Watch out for

- **`enable_prefix_caching` and `enable_chunked_prefill` must both be False.** They default on
  in vLLM V1 and will silently skip or split the prefill you are capturing — the run will look
  fine and the data will be wrong. Gate 4 catches it; don't rely on that alone.
- Hooks fire on decode steps too — capture only the first forward per layer after `clear()`.
- Slice to the target positions *inside the worker*. A full capture is 786 MB per request.
- The vision tower is never used; text-only prompts are correct for this model.
- Model load from local NVMe is minutes, not the ~68 min the old network-mount runs took.

## What to ship back

Priority order, so the cheap and irreplaceable things land first:

1. `results/kimi_k25_varbind_accuracy.json` + all `logs/` (kilobytes)
2. `data/model_weights/kimi_k25/*.npy` (2.4 GB)
3. `data/extracted_states_varbind_allpos_kimi_k25/` (~120 GB, 8 condition dirs × 500
   `prob_NNNN.pkl` + `metadata.json`)

Upload the directory **directly** — do not tar it first, that doubles the disk requirement.
Record the destination and a manifest (file count + total bytes per condition) in the log so
the receiving side can verify. If the link is slow, also run
`scripts/analysis/decode_varbind_heatmap.py` on the pod so the headline ladder survives even if
the raw states never land.

## Output schema

Each `prob_NNNN.pkl` must match `data/extracted_states_varbind_allpos/` (a reference pkl and
`metadata.json` should have been copied to the pod): per-position dict of
`{layer_idx: fp16 array(7168)}` for positions `question_end` → `answer_prompt`, plus
`boundaries`, `model_response`, `model_correct`, and the truth fields (`intermediate` =
`queried_value`, `coefficient`, `operation`, `constant`, `answer`). Downstream decode scripts
read this schema unchanged — if it drifts, the analysis side has to be rewritten.

## Report back

When done: per-condition accuracy table with the uplift vs baseline, wall-clock per phase,
every gate's result, the shipped-artifact manifest, and anything you had to change in the
scripts (with the reason). Flag anything that surprised you rather than smoothing it over.
