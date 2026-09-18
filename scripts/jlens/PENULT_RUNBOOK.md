# Does V3's script offset survive a penultimate-layer target?

**Status:** the 3-prompt run (idx 100–102) was done on 2026-09-17 and answered the question: the offset is a
final-target effect (η² 0.29 / 0.35 / 0.22 → 0.017 / 0.014 / 0.012 at L20 / L30 / L45). **This second run
(2026-09-18) extends the paired lens to 10 prompts for the post's main figure and eval-set comparison, and fits the
two giants for section 8.** Needs the same 3× H200 box and V3 AWQ autograd stack as the fit and the refit
(`FIT_RUNBOOK.md` §4–§5, `REFIT_RUNBOOK.md` §3). **About 6.2 h of wall time** (7 core prompts + 2 giants); the
giants can be dropped (`GIANTS=`, ~4.9 h). Code changes for this run were CPU-tested on the tiny-V3 harness
(`scratch_jlens/tiny_v3_dryrun_penult.py`, checks 1–7 pass).

## 0a. What changed for the 10-prompt run

- **Disk.** The volume has 38 GB free and a full 59-layer fp16 file is 6.1 GB, so the new files hold only the 24
  source layers of the Figure 28 grid (`--source-layers 0,2,5,…,55,58`; 2.5 GB each; 9 files = 22 GB). The backward
  pass costs the same; only the stored layers differ. Every downstream step reads that grid: the depth
  diagnostics and Figure 28 readouts sample exactly those layers, the dimensionality panel uses 13 of them, and the
  eval-set scoring restricts every arm to the layers all lenses share (`audit_v2/lead/score_paired_targets.py`).
  The three existing files (100–102) keep all 59 layers; `build_mean_lens.py --layers …` builds the common subset.
- **Giants** (`GIANTS=76,24`, default on): answers "does the penultimate target tame them" from the log line. They
  are fitted in fp16 like the core prompts; the target-60 originals overflowed fp16 and are fp32. If a giant's norm
  does not fall, its fit may come back non-finite and be `REJECTED`, which is itself the answer (they stayed
  large); for the number, rerun those two with `--dtype fp32` (~1.3 h, 4.9 GB each at 24 layers).
- Nothing else changed: same target (59), same settings, same stack check, same pre-specified reading (§1).

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
4. **Disk: 22 GB** for the default (9 files × 2.5 GB, fp16, 24 layers, in `outputs/jlens/per_prompt_target59/`),
   plus ~5 GB for the two 10-prompt lenses in the CPU step. `df` reports the cluster, not the volume — use the
   RunPod portal figure (38 GB free on 2026-09-18). `LAYERS=all` would need 6.1 GB per file: do not use it here.

## 3. GPU step

```bash
source /workspace/config/probing_env.sh
setsid nohup bash logs/penult_jlens.sh >/dev/null 2>&1 </dev/null &
tail -f logs/jlens_penult.log
```

Defaults (2026-09-18): `CORE=100,…,109` (100–102 already on disk and skipped), `GIANTS=76,24`,
`LAYERS=` the 24-layer Figure 28 grid. Env overrides, placed before `setsid`: `GIANTS=` skips the giants;
`CHECK=` skips the stack check (only when relaunching after a crash that already passed it).

**What the log must show, in order:**

| Time | Line | Expected | If not |
|---|---|---|---|
| ~13 min | `Model loaded in ...s` | ~776 s | — |
| | `patches: {...}` | `awq_leaves_in_train_mode 44971, moe_layers_patched 58, no_repeat_backward True, blocks_checkpointed 61` | stop |
| | `target layer 59, source layers [0, 2, 5, …, 55, 58] (24 of 59)` | exactly that | stop |
| ~20 min | `stack check ...s: relative Frobenius difference L50 ..., L59 ...` then `STACK_CHECK_OK` | both ≲ 1e-3 | `STACK_CHECK_FAILED` exits before fitting: report the numbers, do not relaunch with `CHECK=` |
| ~20 min, then every ~39 min | `prompt 104/1000 (idx 103) ... max\|\|J\|\|/sqrt(d)=... -> J_p0103.pt` | finite, `scale=0.015625`, no `retries=` | `REJECTED` or non-finite: report, let it continue |
| last ~1.3 h | idx 76 then idx 24 (the giants) | any finite norm; `REJECTED` is an acceptable outcome here (see §0a) | report the norms or the rejection verbatim |
| ~6.2 h (default) | `REFIT_DONE`, then `PENULT_EXIT=0` | | |

For reference, at target 60: idx 100 → 14.567, 101 → 9.615, 102 → 3.721, 103 → 5.754, 104 → 4.891 (39 min
each); giants idx 76 → 5038.042, idx 24 → 3946.738. At target 59 the first three came out at 11.195, 7.730,
2.847 (20–24 % lower). Norms of that order are expected for the core prompts; nothing about their values stops
the run. The giants' target-59 norms are the section-8 result: write them down whatever they are.

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
IDX=100:110            # the CORE indices that were fitted (a:b is half-open); giants stay out of the lens
GRID=0,2,5,8,10,12,15,18,20,22,25,28,30,32,35,38,40,42,45,48,50,52,55,58
# 1. the paired mean lenses over the same prompts and the same 24 layers (~1 min each). Both apply the shipped
#    lens's pre-filter (max norm 40) and print any prompt they leave out. These overwrite the 3-prompt lenses.
$PY scripts/jlens/build_mean_lens.py --per-prompt-dir outputs/jlens/per_prompt_target59 --indices $IDX \
    --layers $GRID --out outputs/jlens/lens_v3_target59.pt
$PY scripts/jlens/build_mean_lens.py --per-prompt-dir outputs/jlens/per_prompt --indices $IDX \
    --layers $GRID --out outputs/jlens/lens_v3_target60_same_prompts.pt
#    Sanity in the target-59 build's last line: layer 58 is one block from target 59, so mean diag ~1.0x and
#    ||J||_F a little above 84.7 (the 5-prompt target-60 lens gives 1.031 and 90.3 at layer 59).
# 2. readout statistics at 25 layers, 2,400 activations (~13 min per lens)
$PY scripts/jlens/bilingual_diagnostics.py depth --lenses target59 target60_same --threads 4
# 3. the comparison and the pre-specified reading (~5 min)
$PY scripts/jlens/penult_compare.py
```
If a build leaves a prompt out, or a prompt was rejected on the GPU, rerun both builds and step 2 with the surviving
indices. `penult_compare.py` refuses lenses built from different prompts. Then, for the post: the Figure 28
readouts of the new penultimate lens (`workspace_readouts.py --lens outputs/jlens/lens_v3_target59.pt --label
target59 --null both`, ~90 min), the paired eval-set scoring (`/workspace/jlens_blogpost/audit_v2/lead/
score_paired_targets.py`, ~25 min) and the dimensionality panel (`workspace_signatures.py --lens … --layers
0,5,…,58`, ~20 min), one at a time — two of these beside each other overran the 16 GB cgroup on 2026-09-18.

**Giants.** Compare each giant's `max||J||/sqrt(d)` in `logs/jlens_penult.log` with its target-60 value above. For
the readouts, `penult_quick.py outputs/jlens/per_prompt/J_p0076.pt outputs/jlens/per_prompt_target59/J_p0076.pt`
(the target-60 giant files are fp32, 12.3 GB each; one layer is read at a time).

## 7. Files

| Path | What |
|---|---|
| `scripts/jlens/refit_prompts.py` | `--target-layer`, `--keep-order`, `--stack-check`, `--source-layers` added; target-60 behaviour unchanged |
| `scripts/jlens/build_filtered_lens.py` | `LayerReader` maps a layer to its zip storage by rank, so subset files read correctly (full files unchanged) |
| `logs/penult_jlens.sh` | launcher (one model load: check, core, giants) |
| `scripts/jlens/build_mean_lens.py` | plain mean of per-prompt files, reference lens format + `.meta.json`; `--layers` builds a common subset over mixed full/subset files |
| `scripts/jlens/models.py` | V3 alt lenses `target59`, `target60_same` |
| `scripts/jlens/penult_quick.py` | early read from per-prompt files (covariance identity, all activations) |
| `scripts/jlens/penult_compare.py` | comparison table and the §1 reading → `outputs/jlens/penult_compare.json` |
| `scratch_jlens/tiny_v3_dryrun_penult.py` | CPU dry run: target-59 files equal `jlens.fitting.jacobian_for_prompt(target_layer=...)` to 9e-8; guards; builder → `LazyLens`; stack check passes and catches a 5 % perturbation |
| `outputs/jlens/per_prompt_target59/J_p*.pt` | GPU output (fp16; 100–102 hold 59 layers, the rest the 24-layer grid) |
| `outputs/jlens/lens_v3_target60_same_prompts.pt`, `lens_v3_target59.pt` | the paired mean lenses (same prompts) |
