"""
refit_prompts.py — fit INDIVIDUAL prompts of the J-lens corpus and save each
per-prompt Jacobian on its own, so prompts can be subtracted from (or added to)
the n=100 running sum without a full refit.

WHY. The fit checkpoint holds only sum_p J_p. The 2026-09-06 diagnosis found the
mean dominated by a handful of rank-1 "giant" prompts (top-4 = 84% of the summed
per-prompt norm). The paper's appendix lists a cross-prompt pre-filter — drop any
prompt whose Jacobian Frobenius norm exceeds the cross-prompt mean by more than
N sigma — as one of its recipes. Because the fit is deterministic at fixed
settings (ckpt_verify: plain-vs-plain floor 0.0), refitting a prompt reproduces
the exact J_p that went into the sum, and subtracting it is an exact exclusion.

SETTINGS MUST MATCH THE ORIGINAL RUN BIT-FOR-BIT or the subtraction leaves a
residual: dim_batch=64, gradient checkpointing ON, cotangent_scale=1/64,
max_seq_len=128, skip_first=16, source_layers=None (0..59), target 60,
device_map 22,42. ckpt_verify measured a 4e-3 relative difference between
dim_batch settings, so do not "just" use a different dim_batch.
The determinism check is free: the logged max||J||/sqrt(d) must equal the value
the original fit logged for that prompt (e.g. prompt 77 -> 5038.042).

EXCEPTION - prompts 1..8 (idx 0..7) of the base checkpoint were fitted on
2026-08-19 at dim_batch=8, NO checkpointing, cotangent_scale=1.0 (the pinned
settings start at prompt 9; the 1/64 scale at prompt 11). So idx 7 (prompt 8,
norm 43.633) must be refitted at ITS settings:
    --indices 7 --dim-batch 8 --no-gradient-checkpointing --cotangent-scale 1.0 \\
        --allow-setting-drift
(~66 min). safe_fit at scale=1.0 is the reference loop (the /1.0 is exact) and
checkpointing is numerically inert (ckpt_verify plain-vs-ckpt 0.0), so this
reproduces the summed term; the pinned settings would leave a ~4e-3 residual.
logs/refit_jlens.sh runs it as a second invocation after the main one.

Per-position diagnostics (paper's within-prompt criteria) are recorded at no
extra cost: per-position squared gradient norm and residual norm per layer, so a
giant can be attributed to a single sink-like position.

Output: <outdir>/J_p{idx:04d}.pt with keys
    J {layer: tensor[d,d]} (fp32 for --dtype fp32, else fp16), index, seq_len,
    n_valid, scale_used, n_retries, norm_per_layer, max_norm_over_sqrt_d,
    position_stats {...}, settings {...}, prompt_sha1, elapsed_s
Existing files are skipped, so the job is resumable per prompt.

Usage (3x H200, model load ~13 min, then ~38 min/prompt):
    python scripts/jlens/refit_prompts.py --indices 76,24,56,16,21,11,57,15,40,29,7,52 \
        --range 100:115 --outdir outputs/jlens/per_prompt
    (one model load; idx < 100 saved fp32 for exact subtraction, idx >= 100 fp16)

OTHER TARGET LAYERS (PENULT_RUNBOOK.md). --target-layer 59 fits d h_59 / d h_l for source
layers 0..58: the paper's own Sonnet lens targets the penultimate layer. Such files are not
terms of the n=100 sum, so the run needs --allow-setting-drift and its OWN --outdir: the
script refuses the target-60 directory, and refuses any directory whose existing files were
fitted to a different target (they would be skipped as "already on disk"). --keep-order fits
the indices in the order given instead of sorted, so an optional prompt can go last.
    python scripts/jlens/refit_prompts.py --target-layer 59 --allow-setting-drift \
        --indices 100,101,102,103,104 --dtype fp16 --outdir outputs/jlens/per_prompt_target59 \
        --stack-check 100

STACK CHECK (--stack-check IDX). On a new box, before any long fit: recompute prompt IDX's
Jacobian at two source layers (default 50 and 59) at the settings and target of its stored
per-prompt file, and stop unless both match to --stack-check-tol (relative Frobenius). The
backward spans only target - min(layers) blocks, about a sixth of a prompt for 50,59. A missing
patch (AWQ leaves not in train mode, moe_infer not rewritten, checkpointing off) changes J by
far more than the tolerance; the fp16 rounding of a stored file is ~5e-4.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import sys
import time
from pathlib import Path

import torch

import jlens  # noqa: E402  (must precede REPO_ROOT/scripts on sys.path)
if not hasattr(jlens, "from_hf"):  # a namespace-package shadow has __file__ None
    raise SystemExit(f"scripts/jlens shadowed the jlens library: {jlens.__file__}")

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from safe_fit import NonFiniteGradient, jacobian_for_prompt_scaled  # noqa: E402
from v3_autograd import load_v3_awq, make_differentiable  # noqa: E402

# The settings the n=100 checkpoint was produced with (fit_health.json + resume_jlens.sh).
ORIGINAL = dict(dim_batch=64, gradient_checkpointing=True, cotangent_scale=0.015625,
                max_seq_len=128, skip_first=16, device_map="22,42", target_layer=60)


def parse_indices(args) -> list[int]:
    idx = []
    if args.indices:
        idx += [int(x) for x in args.indices.split(",") if x.strip()]
    if args.range:
        a, b = (int(x) for x in args.range.split(":"))
        idx += list(range(a, b))
    if not idx:
        raise SystemExit("give --indices and/or --range")
    if args.keep_order:
        return list(dict.fromkeys(idx))
    return sorted(set(idx))


def recorded_target(path: Path) -> int:
    """The target layer a per-prompt file was fitted to (metadata only; tensors stay on disk)."""
    ck = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    return int(ck["settings"]["target_layer"])


def stack_check(args, lens_model, prompts, settings):
    """Recompute a stored per-prompt Jacobian at a few source layers; SystemExit unless it reproduces."""
    i = args.stack_check
    ref_path = args.stack_check_dir / f"J_p{i:04d}.pt"
    ref = torch.load(ref_path, map_location="cpu", weights_only=False, mmap=True)
    rs = ref["settings"]
    keys = ("dim_batch", "gradient_checkpointing", "cotangent_scale", "max_seq_len", "skip_first")
    differ = {k: (rs.get(k), settings[k]) for k in keys if rs.get(k) != settings[k]}
    if differ:
        raise SystemExit(f"STACK_CHECK_FAILED: {ref_path.name} was fitted with other settings {differ} "
                         f"(stored, this run); it cannot be reproduced here")
    if hashlib.sha1(prompts[i].encode()).hexdigest() != ref["prompt_sha1"]:
        raise SystemExit(f"STACK_CHECK_FAILED: corpus prompt {i} is not the prompt {ref_path.name} was fitted on")
    layers = sorted(int(x) for x in args.stack_check_layers.split(","))
    target = int(rs["target_layer"])
    print(f"  stack check: idx {i} at target {target}, layers {layers}, against {ref_path}", flush=True)
    t0 = time.perf_counter()
    J, _, _, _, n_retries = jacobian_for_prompt_scaled(
        lens_model, prompts[i], layers, target_layer=target, dim_batch=settings["dim_batch"],
        max_seq_len=settings["max_seq_len"], skip_first=settings["skip_first"],
        scale=float(ref["scale_used"]))
    rel = {}
    for L in layers:
        want = ref["J"][L].float()
        rel[L] = float((J[L].float() - want).norm() / want.norm())
    del ref
    worst = max(rel.values())
    print(f"  stack check {time.perf_counter() - t0:.0f}s: relative Frobenius difference "
          + ", ".join(f"L{L} {r:.2e}" for L, r in rel.items())
          + f" (tolerance {args.stack_check_tol:g}; retries {n_retries})", flush=True)
    if not worst <= args.stack_check_tol:
        raise SystemExit(f"STACK_CHECK_FAILED: the fitting stack does not reproduce {ref_path.name} "
                         f"(worst {worst:.3e}). Check the patches line and the environment before fitting.")
    print("STACK_CHECK_OK", flush=True)


def _atomic_save(obj, path: Path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="/workspace/models/deepseek-v3-awq")
    ap.add_argument("--prompts", type=Path,
                    default=REPO_ROOT / "scripts/jlens/prompts_wikitext.json")
    ap.add_argument("--indices", default=None, help="0-based corpus indices, comma list")
    ap.add_argument("--range", default=None, help="0-based half-open a:b")
    ap.add_argument("--outdir", type=Path, default=REPO_ROOT / "outputs/jlens/per_prompt")
    ap.add_argument("--dtype", choices=["auto", "fp32", "fp16"], default="auto",
                    help="auto (default): fp32 for idx < --fp32-below (prompts that will "
                         "be SUBTRACTED need exactness), fp16 for the rest (ADDED prompts; "
                         "rounding ~5e-4 relative is harmless in a sum).")
    ap.add_argument("--fp32-below", type=int, default=100,
                    help="With --dtype auto: indices below this (= n_base of the checkpoint) "
                         "are saved fp32")
    ap.add_argument("--dim-batch", type=int, default=ORIGINAL["dim_batch"])
    ap.add_argument("--no-gradient-checkpointing", action="store_true")
    ap.add_argument("--cotangent-scale", type=float, default=ORIGINAL["cotangent_scale"])
    ap.add_argument("--max-seq-len", type=int, default=ORIGINAL["max_seq_len"])
    ap.add_argument("--device-map", default=ORIGINAL["device_map"])
    ap.add_argument("--max-memory-gib", default="130,122,108")
    ap.add_argument("--target-layer", type=int, default=ORIGINAL["target_layer"],
                    help="Layer whose residual is differentiated; source layers are 0..target-1. "
                         "60 (default) is the n=100 fit's; 59 is the penultimate-layer lens "
                         "(needs --allow-setting-drift and its own --outdir)")
    ap.add_argument("--source-layers", default=None,
                    help="Comma list of source layers to differentiate and store (default all 0..target-1). "
                         "The backward pass costs the same; only the file shrinks (2.5 GB for the 24 layers of "
                         "the Figure 28 grid instead of 6.1 GB). build_mean_lens.py --layers reads the subset.")
    ap.add_argument("--keep-order", action="store_true",
                    help="Fit the indices in the order given (default: sorted)")
    ap.add_argument("--stack-check", type=int, default=None, metavar="IDX",
                    help="Before fitting, recompute IDX's J at --stack-check-layers and compare with "
                         "its stored file in --stack-check-dir (at that file's settings); stop on mismatch")
    ap.add_argument("--stack-check-dir", type=Path, default=REPO_ROOT / "outputs/jlens/per_prompt")
    ap.add_argument("--stack-check-layers", default="50,59")
    ap.add_argument("--stack-check-tol", type=float, default=1e-2)
    ap.add_argument("--allow-setting-drift", action="store_true",
                    help="Proceed even if settings differ from the original run's")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    settings = dict(dim_batch=args.dim_batch,
                    gradient_checkpointing=not args.no_gradient_checkpointing,
                    cotangent_scale=args.cotangent_scale, max_seq_len=args.max_seq_len,
                    skip_first=ORIGINAL["skip_first"], device_map=args.device_map,
                    target_layer=args.target_layer)
    drift = {k: (settings[k], ORIGINAL[k]) for k in ORIGINAL if settings[k] != ORIGINAL[k]}
    if drift and not args.allow_setting_drift:
        raise SystemExit(f"settings differ from the original fit {drift}; the refit would "
                         f"not reproduce the summed J_p. Pass --allow-setting-drift to override.")
    default_outdir = (REPO_ROOT / "outputs/jlens/per_prompt").resolve()
    if args.target_layer != ORIGINAL["target_layer"] and args.outdir.resolve() == default_outdir:
        raise SystemExit(f"--target-layer {args.target_layer}: {default_outdir} holds the target-"
                         f"{ORIGINAL['target_layer']} Jacobians the shipped lens was built from. "
                         f"Give this run its own --outdir.")

    indices = parse_indices(args)
    corpus = json.loads(args.prompts.read_text())
    prompts = corpus["prompts"]
    # Existing files are skipped below, so one fitted to another target would silently stand in.
    other = {i: t for i in indices if (f := args.outdir / f"J_p{i:04d}.pt").exists()
             and (t := recorded_target(f)) != args.target_layer}
    if other:
        raise SystemExit(f"{args.outdir} already holds files fitted to another target layer "
                         f"{other} (idx: target); this run is --target-layer {args.target_layer}.")
    todo = [i for i in indices if not (args.outdir / f"J_p{i:04d}.pt").exists()]
    print(f"{len(indices)} requested, {len(indices) - len(todo)} already on disk, "
          f"{len(todo)} to fit: {todo}", flush=True)
    if not todo:
        return
    args.outdir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_v3_awq(args.model_path, max_memory_gib=args.max_memory_gib,
                                  device_map=args.device_map)
    lens_model = jlens.from_hf(model, tokenizer)
    patches = make_differentiable(model)
    if settings["gradient_checkpointing"]:
        from v3_autograd import enable_block_checkpointing
        patches["blocks_checkpointed"] = enable_block_checkpointing(model)
    print(f"  patches: {patches}\n  {lens_model}", flush=True)
    target = settings["target_layer"]
    if not 0 < target < lens_model.n_layers:
        raise SystemExit(f"--target-layer {target} out of range for {lens_model.n_layers} layers")
    source_layers = list(range(target))
    if args.source_layers:
        source_layers = sorted({int(x) for x in args.source_layers.split(",")})
        bad = [l for l in source_layers if not 0 <= l < target]
        if bad:
            raise SystemExit(f"--source-layers {bad} not in 0..{target - 1}")
    print(f"  target layer {target}, source layers "
          + (f"0..{target - 1}" if source_layers == list(range(target)) else f"{source_layers} ({len(source_layers)} of {target})"),
          flush=True)
    if args.stack_check is not None:
        stack_check(args, lens_model, prompts, settings)
    sqrt_d = math.sqrt(lens_model.d_model)
    def save_dtype_for(i):
        if args.dtype == "auto":
            return torch.float32 if i < args.fp32_below else torch.float16
        return torch.float32 if args.dtype == "fp32" else torch.float16

    for i in todo:
        out = args.outdir / f"J_p{i:04d}.pt"
        t0 = time.perf_counter()
        stats = {}
        try:
            J, seq_len, n_valid, scale_used, n_retries = jacobian_for_prompt_scaled(
                lens_model, prompts[i], source_layers, target_layer=target,
                dim_batch=settings["dim_batch"], max_seq_len=settings["max_seq_len"],
                skip_first=settings["skip_first"], scale=settings["cotangent_scale"],
                position_stats=stats,
            )
        except NonFiniteGradient as exc:
            print(f"  REJECTED prompt {i + 1} (idx {i}): {exc}", flush=True)
            (args.outdir / f"REJECTED_p{i:04d}.json").write_text(json.dumps({"index": i, "error": str(exc)}))
            continue
        nonfinite = [l for l in source_layers if not torch.isfinite(J[l]).all()]
        if nonfinite:
            print(f"  REJECTED prompt {i + 1} (idx {i}): non-finite in {len(nonfinite)} layers", flush=True)
            continue
        norms = {l: J[l].norm().item() for l in source_layers}
        pnorm = max(norms.values()) / sqrt_d
        # per-position Frobenius norm of that position's own Jacobian, at the
        # layer where the prompt's norm peaks — the sink-attribution summary
        peak = max(norms, key=norms.get)
        pos_fro = stats["grad_sq"][peak].sqrt()
        share = (pos_fro ** 2 / (pos_fro ** 2).sum()).max().item()
        top_pos = int(stats["valid_positions"][int((pos_fro ** 2).argmax())])
        tok = tokenizer.decode([int(stats["input_ids"][top_pos])])
        elapsed = time.perf_counter() - t0
        _atomic_save({
            "J": {l: J[l].to(save_dtype_for(i)) for l in source_layers},
            "index": i, "seq_len": seq_len, "n_valid": n_valid,
            "scale_used": scale_used, "n_retries": n_retries,
            "norm_per_layer": norms, "max_norm_over_sqrt_d": pnorm,
            "position_stats": stats, "settings": settings,
            "dtype": "fp32" if save_dtype_for(i) == torch.float32 else "fp16",
            "prompt_sha1": hashlib.sha1(prompts[i].encode()).hexdigest(),
            "elapsed_s": round(elapsed, 1),
        }, out)
        print(f"  prompt {i + 1}/1000 (idx {i})  seq_len={seq_len} n_valid={n_valid}  "
              f"{elapsed:.0f}s  max||J||/sqrt(d)={pnorm:.3f}  scale={scale_used:g}"
              f"{f' retries={n_retries}' if n_retries else ''}  "
              f"peakL{peak}: top position {top_pos} ({tok!r}) carries {share:.0%} of "
              f"sum_t ||J_t||^2  -> {out.name}", flush=True)
    print("REFIT_DONE", flush=True)


if __name__ == "__main__":
    main()
