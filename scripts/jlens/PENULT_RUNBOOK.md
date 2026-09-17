# Does V3's script offset survive a penultimate-layer target?

**Status:** code written and CPU-tested 2026-09-17 (tiny-V3 dry run, all checks pass); GPU step not yet run.
Needs the same 3× H200 box and V3 AWQ autograd stack as the fit and the refit (`FIT_RUNBOOK.md` §4–§5,
`REFIT_RUNBOOK.md` §3). **About 2.3 h of wall time** (3 prompts); 5 prompts ~3.6 h; the optional giants add ~1.3 h.

## 0. The question and the design

The shipped lens differentiates the final residual (target layer 60). The paper's appendix says its Sonnet 4.5
lens, "used throughout the paper", takes the gradient at the **penultimate** layer, i.e. "omitting the last
transformer block from the backward pass", and that including the last layer "can sometimes increase the number of
noisy artifacts ... because the final block is heavily specialized for calibrating next token predictions". (The
paper is inconsistent: its Figure 4 caption and pseudocode say the final layer. That ambiguity is worth a question
to the authors, but it does not change this test, which measures how much the choice moves V3.) V3's
script gate is written almost entirely by that final block (logit-lens η² 0.012 at L59, 0.41 at L60), and every
target-60 Jacobian carries it as its top output direction. If a target-59 lens loses the offset, the draft's
"general property of Jacobian lenses" is a property of the final-layer target.

**Paired design, 3 prompts by default.** Fit prompts idx 100–102 to target 59. Their target-60 Jacobians are
already on disk (`outputs/jlens/per_prompt/J_p0100..0114.pt`, fitted 2026-09-06), so the comparison is the same
prompts under the two targets. A full 100-prompt refit is unnecessary: a mean over just the first 3 target-60 files
already shows the offset (η² 0.32 / 0.35 / 0.22 at L18 / L30 / L45; 5 files: 0.41 / 0.38 / 0.24 at L20 / L30 / L45;
shipped lens 0.52 / 0.40 / 0.22 at L20 / L30 / L45; logit lens 0.003). `CORE=100,101,102,103,104` fits 5 (~3.6 h)
and tightens the size of a "reduced" result; it is not needed for a clear "gone" or "survives". A 5-prompt target-60
mean lens is already built (`outputs/jlens/lens_v3_target60_same_prompts.pt`, depth run
`bilingual_depth_target60_same.json`); §6 rebuilds it for whatever prompts were fitted.

**Optional, same model load: two giants** (`GIANTS=76,24`, +~1.3 h; target-60 max‖J‖_F/√d 5038.042 and 3946.738).
Answers the draft's open question "does the penultimate target get rid of the giant prompts?" from the log line alone.

**Stack check first.** A missing patch on a fresh box (AWQ leaves not in train mode, `moe_infer` not rewritten)
still produces plausible-looking Jacobians and would bias this comparison. So before fitting, the run recomputes
idx 100 at target 60 for source layers 50 and 59 only (the backward spans 10 blocks, ~7 min) and stops unless
both match the stored file to 1e-2 relative Frobenius. The stored file is fp16, so expect ~5e-4.

## 1. Pre-specified reading (fixed before the GPU run)

`scripts/jlens/penult_compare.py` applies it at layers 20, 30 and 45 (33, 50, 75 % depth). A few-prompt lens is
spiky below ~30 % depth (the 5-prompt target-60 mean has full-vocab kurtosis p50 18 at L15), so earlier layers
are shown but not used.

| Verdict at a layer | Rule on median η² (3 script classes) |
|---|---|
| **gone** | η²(target 59) ≤ 0.03 — the Qwen3.5-122B lens's level (≤ 0.02); the logit lens is 0.003 |
| **survives** | η²(target 59) ≥ ½ η²(same prompts, target 60) |
| **reduced** | anything in between |

Supporting evidence, reported but not part of the rule: the Han−Latin offset, full-vocab median kurtosis, and
whether each lens's top output direction is still the shipped lens's script axis (share of ‖J‖², |cos| with the
shipped u₁, gain along it).

**Prediction from the CPU approximation** `J_59⁻¹ J_ℓ` on the shipped lens (audit, `audit_v2/lead/penult_v3.json`):
η² 0.527 → 0.108 at L18, 0.404 → 0.096 at L30, 0.223 → 0.032 at L45; full-vocab kurtosis p50 at L18 −0.92 → +0.99.
Applied to this pair that is roughly **reduced / reduced / gone**. The approximation ignores cross-position terms of
the last block (on the tiny model J_{0→4} and J_{3→4}J_{0→3} differ by 17 %), which is why the real fit is needed.

What each outcome means for the post:
- **gone** at all three: the offset and the negative in-band kurtosis come from the final-layer target. Say so, and
  show the target-59 readouts beside the shipped ones from ~30 % depth on.
- **reduced**: part of the script axis is carried by blocks 0–58; report both lenses and the size of each.
- **survives**: "a general property" is earned for this model, and the target is not the explanation.

## 2. Before launching (GPU box)

1. **Box.** 3× H200 mounting the shared `/workspace` (model at `/workspace/models/deepseek-v3-awq`, patched
   `modeling_deepseek.py` in the model dir and HF modules cache). `load_v3_awq` uses 3 GPUs, `device_map 22,42`.
2. **Nothing else loading the model:** `ps -eo pid,etime,cmd | grep -E '[p]ython scripts'` and `nvidia-smi`
   empty. Re-read `logs/penult_jlens.sh` right before launching (other agents edit launchers).
3. **Environment:** `bash logs/bootstrap_jlens.sh` on a fresh box (`/root` is ephemeral; ~2–10 min). Its final
   `import awq` failure (`PytorchGELUTanh`) is expected and harmless.
4. **Disk: ~25 GB free** for the default (3 files × 6.1 GB, fp16, 59 layers, in `outputs/jlens/per_prompt_target59/`,
   plus a 6.1 GB lens in the CPU step); +6.1 GB per extra prompt or giant. `df` reports the cluster, not the volume —
   use the RunPod portal figure.

## 3. GPU step

```bash
source /workspace/config/probing_env.sh
setsid nohup bash logs/penult_jlens.sh >/dev/null 2>&1 </dev/null &
tail -f logs/jlens_penult.log
```

Env overrides, placed before `setsid`: `CORE=100,101,102,103,104` (5 prompts, ~3.6 h); `GIANTS=76,24` (+~1.3 h);
`CHECK=` skips the stack check (only when relaunching after a crash that already passed it).

**What the log must show, in order:**

| Time | Line | Expected | If not |
|---|---|---|---|
| ~13 min | `Model loaded in ...s` | ~776 s | — |
| | `patches: {...}` | `awq_leaves_in_train_mode 44971, moe_layers_patched 58, no_repeat_backward True, blocks_checkpointed 61` | stop |
| | `target layer 59, source layers 0..58` | exactly that | stop |
| ~20 min | `stack check ...s: relative Frobenius difference L50 ..., L59 ...` then `STACK_CHECK_OK` | both ≲ 1e-3 | `STACK_CHECK_FAILED` exits before fitting: report the numbers, do not relaunch with `CHECK=` |
| ~59 min, then every ~39 min | `prompt 101/1000 (idx 100) ... max\|\|J\|\|/sqrt(d)=... -> J_p0100.pt` | finite, `scale=0.015625`, no `retries=` | `REJECTED` or non-finite: report, let it continue |
| ~2.3 h (default) | `REFIT_DONE`, then `PENULT_EXIT=0` | | |

For reference, the same prompts at target 60: idx 100 → 14.567, 101 → 9.615, 102 → 3.721, 103 → 5.754,
104 → 4.891 (39 min each); giants idx 76 → 5038.042, idx 24 → 3946.738. Target-59 norms of the same order are
expected for the core prompts; nothing about their values stops the run.

**Stopping early** (e.g. to skip remaining giants): find the python PID with
`ps -eo pid,etime,cmd | grep '[r]efit_prompts'` and kill that PID. Finished files are complete (written to `.tmp`,
then renamed); a killed prompt leaves only a `.tmp`, which can be deleted.

Release the GPU box once `PENULT_EXIT=0` is in the log. Everything below runs on CPU.

## 4. What not to do

- Do not write into `outputs/jlens/per_prompt/` (the target-60 files the shipped lens was built from).
  `refit_prompts.py` refuses `--target-layer 59` there, and refuses any directory holding other-target files.
- Do not run `logs/overnight_jlens.sh`, `resume_jlens.sh` or `refit_jlens.sh`.
- Do not delete per-prompt files, lenses or logs.
- Do not pick thresholds or layers after seeing the numbers; §1 is fixed.

## 5. Optional early read (CPU, after the first file lands)

```bash
python scripts/jlens/penult_quick.py outputs/jlens/per_prompt/J_p0100.pt \
    outputs/jlens/per_prompt_target59/J_p0100.pt          # a few minutes on 4 threads, ~1.5 GB RAM
```
Idx 100 at target 60 gives η² 0.093 / 0.157 / 0.211 at L20 / L30 / L45; read the one-prompt comparison at L30
and L45 (one prompt's early layers vary too much). This is informational; the verdict uses all fitted prompts.

## 6. CPU steps after the GPU step (the 16 GB sandbox is enough)

```bash
PY=/root/.venvs/jlens-cpu/bin/python
IDX=100,101,102        # the CORE indices that were fitted
# 1. the paired mean lenses over the same prompts (~1.5 min each). Both apply the shipped lens's pre-filter
#    (max norm 40) and print any prompt they leave out.
$PY scripts/jlens/build_mean_lens.py --per-prompt-dir outputs/jlens/per_prompt_target59 --indices $IDX \
    --out outputs/jlens/lens_v3_target59.pt
$PY scripts/jlens/build_mean_lens.py --per-prompt-dir outputs/jlens/per_prompt --indices $IDX \
    --out outputs/jlens/lens_v3_target60_same_prompts.pt
#    Sanity in the target-59 build's last line: layer 58 is one block from target 59, so mean diag ~1.0x and
#    ||J||_F a little above 84.7 (the 5-prompt target-60 lens gives 1.031 and 90.3 at layer 59).
# 2. readout statistics at 25 layers, 2,400 activations (~13 min per lens)
$PY scripts/jlens/bilingual_diagnostics.py depth --lenses target59 target60_same --threads 4
# 3. the comparison and the pre-specified reading (~5 min)
$PY scripts/jlens/penult_compare.py
```
The second build overwrites the 5-prompt target-60 lens built on 2026-09-17. With `CORE=100,101,102,103,104` that
lens and its depth run already match, so skip the second build and use `--lenses target59` in step 2.
If a build leaves a prompt out, or a prompt was rejected on the GPU, rerun both builds and step 2 with the surviving
indices. `penult_compare.py` refuses lenses built from different prompts.

**Giants.** Compare each giant's `max||J||/sqrt(d)` in `logs/jlens_penult.log` with its target-60 value above. For
the readouts, `penult_quick.py outputs/jlens/per_prompt/J_p0076.pt outputs/jlens/per_prompt_target59/J_p0076.pt`
(the target-60 giant files are fp32, 12.3 GB each; one layer is read at a time).

## 7. Files

| Path | What |
|---|---|
| `scripts/jlens/refit_prompts.py` | `--target-layer`, `--keep-order`, `--stack-check` added; target-60 behaviour unchanged |
| `logs/penult_jlens.sh` | launcher (one model load: check, core, giants) |
| `scripts/jlens/build_mean_lens.py` | plain mean of per-prompt files, reference lens format + `.meta.json` |
| `scripts/jlens/models.py` | V3 alt lenses `target59`, `target60_same` |
| `scripts/jlens/penult_quick.py` | early read from per-prompt files (covariance identity, all activations) |
| `scripts/jlens/penult_compare.py` | comparison table and the §1 reading → `outputs/jlens/penult_compare.json` |
| `scratch_jlens/tiny_v3_dryrun_penult.py` | CPU dry run: target-59 files equal `jlens.fitting.jacobian_for_prompt(target_layer=...)` to 9e-8; guards; builder → `LazyLens`; stack check passes and catches a 5 % perturbation |
| `outputs/jlens/per_prompt_target59/J_p*.pt` | GPU output (fp16, 59 layers) |
| `outputs/jlens/lens_v3_target60_same_prompts.pt`, `lens_v3_target59.pt` | the paired mean lenses (same prompts) |
