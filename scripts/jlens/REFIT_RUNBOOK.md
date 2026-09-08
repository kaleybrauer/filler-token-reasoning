# J-lens pre-filter: refit the giant prompts, subtract them, top back up to n=100

**Status:** scripts written and CPU-tested 2026-09-06; GPU step LAUNCHED 2026-09-06 ~01:35 UTC on a 3× H200 box (`logs/jlens_refit.log`).
Needs a 3× H200 box with the V3 AWQ autograd stack (same as the original fit).
Read `FIT_RUNBOOK.md` §4–§5 for the environment; nothing there changes.

## 0. Why (one paragraph)

The n=100 lens is a plain mean of per-prompt Jacobians, and that mean is dominated by a
few prompts: four prompts carry 84 % of the summed per-prompt norm, twelve carry 95 %.
Each giant is rank-1 with an output direction that unembeds to punctuation / EOS. The
paper's appendix names this ("the per-sample Jacobians are sometimes heavy-tailed enough
that the choice of estimator matters") and lists the fix we are applying: "a pre-filter
that excludes any prompt whose Jacobian Frobenius norm exceeds the cross-prompt mean by
more than Nσ". N is not stated. Our per-prompt norms (`max||J||/sqrt(d)` from the fit
log): median 7.8, robust σ 3.6, mean 155, sd 701. See `project_jlens_workspace` memory
(2026-09-06 section) for the full diagnosis.

## 1. What gets done

1. **Refit each flagged prompt on its own** and save its Jacobian to a per-prompt file.
   The fit is deterministic at fixed settings (ckpt_verify: plain-vs-plain floor 0.0),
   so the refit reproduces the exact `J_p` inside the running sum.
2. **Fit replacement prompts** (corpus indices 100+, same spread corpus) to per-prompt
   files, in the same model load.
3. **Build the filtered lens on CPU**: `(sum_100 − Σ_excluded J_p + Σ_added J_p) / n`,
   written in the reference `JacobianLens.save()` format with a provenance sidecar.
4. **Score** it on the paper's eval sets and run the 2-fact heatmap with `--jlens`,
   alongside the unfiltered all-100 lens (the paper's default recipe). Report both.

## 2. Threshold — DECIDED 2026-09-06 (Kaley), fixed BEFORE any scoring

**Primary: T = 40 = 5 × median** (median 7.8). Twelve prompts excluded, 0-based
`76, 24, 56, 16, 21, 11, 57, 15, 40, 29, 7, 52`. Rationale: the per-prompt norms have a
gap between the body (max 30.4) and the tail (40, 44, 94, …), so any T in 31–40 gives the
same set — the choice is insensitive to the exact value; and it has a plain reading: no
prompt outweighs five typical prompts in the mean (a prompt at 5000 has 650× a typical
prompt's weight). The literal mean + Nσ readings fail here because σ (701) is set by the
very prompts being tested; iterated 3σ never converges on this tail.

**Robustness rows (same per-prompt files, CPU only):**

| T | rule | excluded (0-based) | n |
|---|---|---|---|
| 40 | 5 × median — PRIMARY | 76, 24, 56, 16, 21, 11, 57, 15, 40, 29, 7, 52 | 88 + 12 adds (100:112) = 100 |
| 100 | tail only | 76, 24, 56, 16, 21, 11, 57, 15, 40 | 91 + 9 adds (100:109) = 100 |
| 856 | mean + 1 sd (literal) | 76, 24, 56, 16 | 96 + 4 adds (100:104) = 100 |
| ∞ | plain mean (paper default) | none | 100 |

Every row is built at n=100 so n is not a confound; replacements are taken in index
order (a fixed rule, not a selection). All 15 fitted replacements came in under T=40
(max 18.99), so idx 112..114 are spare. `logs/build_lenses.sh` builds all four.

Do NOT choose T by which scores best; report the sweep as robustness. Replacements:
indices 100:115 fitted, accepted under the same T; the builder reports refusals.
State in the writeup that 12 % exclusion is more than a usual pre-filter and changes the
estimand from "the average Jacobian" to "the Jacobian of typical text" (the paper's intent).

## 3. GPU step (3× H200; ~13 min load + 38 min/prompt ≈ 17.5 h for 12 + 15)

```bash
source /workspace/config/probing_env.sh          # venv /root/.venvs/filler-probing
bash logs/bootstrap_jlens.sh                     # only on a fresh box (~10 min); the
                                                 #   final `import awq` failure is expected
setsid nohup bash logs/refit_jlens.sh >/dev/null 2>&1 </dev/null &
tail -f logs/jlens_refit.log
```

`logs/refit_jlens.sh` runs `scripts/jlens/refit_prompts.py --indices <11> --range 100:115`
in ONE model load, then a SECOND invocation for idx 7 alone (see below). Env overrides:
`EXCLUDE=...` (0-based, comma), `ADD_RANGE=a:b`, `EARLY=7`.
Resumable: per-prompt files that exist are skipped, so a crash costs one prompt.
Output: `outputs/jlens/per_prompt/J_p{idx:04d}.pt` — idx < 100 in **fp32** (12.3 GB each,
exact subtraction), idx ≥ 100 in **fp16** (6.2 GB). Disk: ~148 + 93 ≈ 240 GB.
**Measured 2026-09-06 (du, binary units): models 842 G + repo 2.3 T + .cache 95 G = 3.24 TiB
= 3.56 TB → ~440 GB free of 4 TB, NOT the 700 GB an earlier draft quoted.** After the
per-prompt files (~240 GB) and four lens files (~25 GB) about 175 GB remain.

**idx 7 (prompt 8, norm 43.633) is a special case** — see "Provenance of the base sum"
below: the launcher refits it LAST, alone, at its own 2026-08-19 settings (dim_batch 8,
no checkpointing, scale 1.0), with `--keep 7` as the fallback if it does not reproduce.
The first invocation is sorted, so its first refit — the determinism check — is idx 11
(prompt 12 → logged 410.143). Total ≈ 13 min + 26 × 38 min + 13 min + 66 min ≈ 18.5 h.
Launched 2026-09-06 01:35 UTC; the launcher was relaunched at ~01:45 as a waiter (it
waits for the running `refit_prompts.py`, resumes anything unfinished, then runs idx 7).

**Settings are pinned to the original run and the script refuses to start otherwise**
(`--allow-setting-drift` overrides): dim_batch 64, gradient checkpointing on, cotangent
scale 1/64, max_seq_len 128, skip_first 16, source layers 0..59 → target 60, device_map
22,42. ckpt_verify measured a 4e-3 relative difference between dim_batch values (fp16
reduction order), so a refit at another dim_batch would NOT cancel the summed term.

### Provenance of the base sum is NOT uniform — idx 0..7 need their own settings
From `logs/jlens_supervisor.log` + `logs/jlens_deploy.log` (2026-08-19): the attempt at
dim_batch=4 (06:06) was killed after 6 min and completed nothing; the attempt that ran
06:15–15:21 was **dim_batch=8, no checkpointing, cotangent scale 1.0** (66.2 min/prompt →
prompts 1–8 = idx 0..7); from 16:02 the run continued at dim_batch=64 + checkpointing,
still scale 1.0 (idx 8, 9); idx 10..99 are the pinned settings (safe_fit, scale 1/64).
`no_repeat_backward` was on throughout, and at scale 1.0 `jacobian_for_prompt_scaled` is
arithmetically the reference (`grad[...].float().mean(dim=1)` in both).
Consequence: **idx 7 (norm 43.633, inside the T=40 set) cannot be reproduced under the
pinned settings.** Refit it LAST, alone, with
`--dim-batch 8 --no-gradient-checkpointing --cotangent-scale 1.0 --allow-setting-drift`
(second model load, ~66 min + 13 min). Its determinism was never measured at those
settings, so if the logged norm does not come back as 43.633, DO NOT force it: build with
`--exclude-above 40 --keep 7` (11 excluded; idx 7 is the borderline 5.6× prompt and stays
in, recorded in the sidecar as kept-despite-threshold). Idx 8, 9 are not in any exclusion
set, so their scale-1.0 provenance does not matter.

### The determinism check is free — read it off the log
Each refitted prompt logs `max||J||/sqrt(d)=…`. It must match the original fit log to
the printed precision (`logs/jlens_validate_fit.log`; e.g. idx 76 → 5038.042, idx 24 →
3946.738, idx 56 → 2983.661, idx 16 → 1065.124). The builder enforces this at 1e-3
relative and refuses to subtract a prompt that does not match. If the first refitted
giant does not match: stop, do not burn the remaining hours — something in the stack
differs (kernel version, checkpointing patch, scale) and must be found first.

### Per-position diagnostics (paper's within-prompt criterion), also free
Each per-prompt file carries `position_stats`: per-position squared gradient norm per
layer and residual-stream norm per layer, plus the token ids. The log line summarises
it: `peakL<l>: top position <t> ('<token>') carries <share> of sum_t ||J_t||^2`. A giant
that is one position at ~100 % is a sink-like position, which is what the paper's
"exclude positions whose residual-stream norm is an outlier" rule targets; a giant
spread over many positions is a prompt-level property. Either way the cross-prompt
filter is what we apply now; this only tells us which mechanism produced the tail.

## 4. CPU step (any box; the 16 GB cgroup box is enough — one layer in memory at a time)

```bash
PY=/root/.venvs/jlens-cpu/bin/python                 # or the probing venv
$PY scripts/jlens/build_filtered_lens.py \
    --base outputs/jlens/lens_v3.fitckpt \
    --per-prompt-dir outputs/jlens/per_prompt \
    --exclude-above 40 --add-range 100:115 \
    --out outputs/jlens/lens_v3_filtered40.pt
```

Then the robustness rows (same files, ~3 min each on CPU):
```bash
$PY scripts/jlens/build_filtered_lens.py --exclude-above 100 --add-range 100:115 --out outputs/jlens/lens_v3_filtered100.pt
$PY scripts/jlens/build_filtered_lens.py --exclude-above 856 --add-range 100:115 --out outputs/jlens/lens_v3_filtered856.pt
$PY scripts/jlens/build_filtered_lens.py --exclude-above 1e9 --out outputs/jlens/lens_v3_all100.pt
```

Writes `lens_v3_filtered40.pt` (fp16, keys `J / n_prompts / source_layers / d_model`,
readable by `apply_lens.load_jlens`, `jlens.JacobianLens.load`, and both `--jlens`
consumers) and `lens_v3_filtered40.meta.json` (rule, excluded/added indices with logged
and refit norms, refused adds, per-layer ‖J‖_F before → after). Also materialise the
unfiltered all-100 lens the same way for a like-for-like file:
`--exclude-above 1e9 --out outputs/jlens/lens_v3_all100.pt` (no per-prompt files needed).

**Sanity after the build (in the printed per-layer table):** ‖J‖_F at L0 should fall
from ~11 000 to the clean range (~100–230; ‖I‖_F = 84.7); L59 stays ≈ 89 (≈ I). A
leftover spike means a subtraction did not cancel — check that prompt's refit norm.

## 5. Score + the 2-fact heatmap

```bash
# paper's own eval sets (multihop, order-ops), both arms, memory-safe
$PY scripts/jlens/score_lazy.py --lens outputs/jlens/lens_v3_filtered40.pt \
    --out outputs/jlens/score_filtered40.json
# (score_lazy reads fit checkpoints AND finished .pt lenses, one layer at a time)
# 2-fact heatmap (the actual question), against the logit arm already on disk
$PY scripts/analysis/decode_2fact_heatmap.py --condition dots_10 \
    --jlens outputs/jlens/lens_v3_filtered40.pt \
    --output-dir results/unsupervised_decode_2fact_allpos
```
Comparison targets (logit lens, same states): A1 79.9 % (±5) @pos_001 L50, A2 60.7 % @pos_005
L44, sum 95.1 % @pos_016 L59 (exact: 78.7 / 57.8 / 88.5).
⚠ METRIC FIX 2026-09-08: `variant_token_ids` now credits SINGLE-TOKEN variants only (DeepSeek
tokenizes " 5" as [" ", "5"], so the bare space used to be credited for every number). Any
pass@k computed before that is inflated for whitespace-pushing lenses; use `score_*_v2.json`.
RESULTS (corrected, pass@10 multihop / order-ops; logit 0.812 / 0.582): T=40 0.772 / 0.645,
T=100 0.794 / 0.627, T=856 0.778 / 0.664, all-100 0.590 / 0.536. Heatmap A1/A2/sum (±5):
T=40 78.7/66.4/94.3, T=100 78.7/66.0/94.3, T=856 72.5/58.2/94.3, all-100 59.0/45.9/94.3. Report accuracies per arm, not deltas; both lenses next to each
other, labelled "plain mean (paper default)" and "Frobenius pre-filter T=40 (paper
appendix variant)".

The heatmap script loads all 1000 dots_10 pickles (14 GB) — on the 16 GB box it needs a
streaming pass; on a normal box it is fine.

## 6. What NOT to do
- Do not drop whole snapshot blocks instead of refitting: that discards clean prompts
  and leaves n=20.
- Do not use a per-prompt *median* or *normalised* mean: both need every J_p (a full
  refit); the Frobenius pre-filter is the recipe that works from the running sum.
- Do not delete the per-prompt files after the build without asking — they are what
  makes T re-choosable on CPU.
- Do not launch `logs/overnight_jlens.sh` / `resume_jlens.sh`: those resume the running
  sum past n=100 and would change the base.
