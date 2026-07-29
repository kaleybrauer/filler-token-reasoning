# SESSION STATE — Kimi K2.5 varbind run (2026-07-28)

Written before the Claude session was closed mid-run. Everything below is on the
**persistent network volume** `/workspace`, so it survives pod termination.

**Pod:** `etg4jfm4iv9q5f`, 8×H200. **Kaley's instruction: DO NOT TERMINATE THE POD.**
She is checking on it herself. `RUNPOD_API_KEY` is set and `runpodctl` exists — do
not use them to stop the pod.

## Where things stand

| item | state |
|---|---|
| Checkpoint | `/workspace/models/kimi-k2.5` — 64/64 shards, 595,177,988,208 B, verified |
| venv | `/root/.venvs/k25` — vllm 0.25.1, torch 2.11.0+cu130 |
| Extraction | launched 22:26:21Z, `logs/extract_full.log`, ~0.5–0.8 s/example |
| Finisher | armed 22:45:16Z, `logs/finisher.log` → writes `logs/verify_report.log` |
| Output | `data/extracted_states_varbind_allpos_kimi_k25/` |
| Weights | `data/model_weights/kimi_k25/{lm_head_weight,rms_norm_weight}.npy` ✅ |
| Accuracy | `results/kimi_k25_varbind_accuracy.json` (appended per condition) |

Condition order: `dots_10, baseline, dots_5, dots_25, dots_50, counting_5,
counting_10, counting_25` (dots_10 first so the audit could run early).

**Results so far:** `dots_10` 232/500 = 46.4% · `baseline` 158/500 = 31.6%
⇒ **+14.8 pt filler uplift with reasoning suppressed.**

The run is resumable: re-running `run_extract.sh` skips existing pkls.

## All gates PASSED

1. Prompt/thinking — closed `<think></think>` (163606→163607), no lone open tag, no BOS shift.
2. No `<think>` in free-running generation (5/5); answers are bare numbers.
3. Filler spans — see the counting_k correction below.
4. Capture convention — **CONFIRMED residual stream**, two independent ways:
   - `audit_residual_convention.py` geometry on 20 real pkls: median cos +0.954,
     97% of pairs >0.9, norm ratio 548× ⇒ "RESIDUAL STREAM (cumulative)"
   - final-layer logit lens 20/20 (100%), median rank 1
   - `cum` column is *worse* than `raw` (5% vs 20% @L56), which is only possible if
     `raw` is already cumulative — rules out layer-writes from the other direction
5. lm_head + norm saved from shard 62.

## Deviations from RUNBOOK/HANDOFF — read before trusting either doc

1. **Filler placement: USER turn, not assistant turn.** RUNBOOK §A2 and gate 1 said to
   append `Filler: …\n\nAnswer:` after `apply_chat_template(add_generation_prompt=True)`,
   which puts it in the assistant turn. Kaley's call: match V3 instead, since turn
   placement is a confound in the one cross-model comparison the run exists to make.
   `thinking=False` is the only K2.5-specific prompt change. Gate 1's expected tail is
   now `<|im_middle|><think></think>`; the token-pair checks are unchanged.
2. **`counting_k` filler is 2k tokens in context, not 2k−1.** 2k−1 is the *standalone*
   tokenization; in context the space after `Filler:` can't merge into the digit
   (space-prefixed numbers aren't in Kimi's vocab). `dots_k` is still k. Gate 3 now
   checks that the located region *decodes back to the exact filler string* and holds
   k items — tokenizer-independent. Bonus: counting_25 = 50 tokens = dots_50 exactly.
3. **Dataset is the easy variant** `data/chained_var_binding_easy_dataset.json`
   (md5 `ea35ef74e42491cf6e171af97675c8ce`), per Kaley — K2.5 is worse at this task
   than V3.
4. **Phase-A scripts did not exist** and were written this session (see below).
5. **Tiny-model smoke skipped** — `tiny-random/kimi-k2.5` won't load under vLLM 0.25.1;
   its own config is inconsistent (kv_lora 384 + qk_rope 192 ⇒ param 1152, ships 64).
   Broken upstream artifact, not our code.
6. **No local NVMe on this pod.** Container root is a 50 GB overlay; `/workspace` is a
   MooseFS network volume. Irrelevant in practice: the host has 3 TB RAM and the whole
   checkpoint sits in page cache (read benched 4.4 GB/s), so no `/dev/shm` staging.
7. **Model loads in ~450 s**, not the ~68 min the runbook feared.

## Traps that each cost a full 5–10 min load cycle

1. **`ninja` must be on PATH.** FlashInfer JIT-compiles its RoPE kernel on the first
   forward and shells out to `ninja`. It ships *inside* the venv, but invoking
   `$K25_VENV/bin/python` by absolute path does not put `$K25_VENV/bin` on PATH the
   way `activate` would → workers die with `FileNotFoundError: 'ninja'` in
   `profile_run`, **after** the full weight load. Fixed in `k25_env.sh`. Same class as
   the runbook's `setuptools` note; worth adding there.
2. **Do not return a dict of per-layer arrays over `collective_rpc`.** vLLM's
   MessageQueue pickles with out-of-band PickleBuffers and inlines anything under
   1 MiB; 61 arrays of ~315 KB came back with rows truncated (`IndexError: list index
   out of range`). The worker now returns ONE contiguous fp16 blob + explicit shape,
   rebuilt with `np.frombuffer`. Faster too.
3. Don't `pkill -f` on a pattern that appears in your own command line.

## Files written this session (all new)

```
scripts/kimi_k25/prompt_k25.py                 instant-mode prompt builder (user-turn filler)
scripts/kimi_k25/kimi_k25_hooks.py             worker extension, captures output[0]+output[1]
scripts/kimi_k25/extract_hidden_states_vllm.py main extractor + live gates
scripts/kimi_k25/preflight_offline_k25.py      CPU gates 1/3 + token checks
scripts/kimi_k25/save_lm_head_k25.py           gate 5
scripts/kimi_k25/verify_extraction_k25.py      post-run verification (read-only)
scripts/kimi_k25/smoke_tiny_k25.py             tiny-model plumbing smoke (upstream model broken)
scripts/kimi_k25/k25_env.sh                    shared env — PATH fix lives here
scripts/kimi_k25/run_extract.sh                launcher
scripts/kimi_k25/run_finisher.sh               detached verify-on-completion
scripts/kimi_k25/run_download.sh               weights download
scripts/kimi_k25/run_venv_build.sh             venv build
```

Nothing was committed — Kaley controls the repo (`git status` will show these as
untracked/modified).

## What to do next

1. Read `logs/verify_report.log` (the finisher writes it automatically when the
   extraction exits). It contains the completeness check, per-condition manifest, the
   full accuracy table with uplift, and a 50-example re-run of the convention audit.
2. If any condition is short of 500, just re-run `./scripts/kimi_k25/run_extract.sh`
   — it skips existing pkls.
3. Analysis (Phase D, CPU): `scripts/analysis/decode_varbind_heatmap.py
   --max-num-token 1000` for the 5-target ladder, then the residual-fingerprint →
   aggregate → llm_decode_batch pipeline.
4. **Open question for the analysis:** K2.5's answer emerges much later in the stack
   than V3's. Layer profile at `answer_prompt` (n=20): 0% through L52, 20% @L56,
   100% @L60. V3 varbind was 50% @L52, 92% @L56, 83% @L60. That is a real effect, not
   a capture artifact (geometry says residual stream), but it confounds three things
   at once — different model, INT4 vs AWQ, easy vs original dataset — and this run
   cannot separate them.
