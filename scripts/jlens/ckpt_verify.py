"""
ckpt_verify.py — prove gradient checkpointing changes nothing, then size it.

Gradient checkpointing is a memory/compute trade that must not touch the value
of J. Proving that needs THREE fits of J_59, not two: this backward is genuinely
nondeterministic, so a plain-vs-checkpointed difference on its own says nothing.

    A = plain, B = plain, C = checkpointed
    floor = ||A-B|| / ||B||        <- run-to-run noise
    pass  = ||A-C|| within 3x floor

Two sources of that noise: _moe_infer_autograd gathers every token 8 times (once
per routed expert), so the gather's backward is a scatter-add over duplicate
indices and uses CUDA atomics; and cuBLAS can pick different GEMM algorithms
depending on free workspace, which differs between a retained graph and a
recomputed one. A first version of this gate compared once against a stored
matrix, measured 3.96e-3, and failed on a 1e-3 threshold that assumed
determinism.

If and only if that passes, it sweeps dim_batch at FULL depth to find the
largest that fits, which is the whole point: throughput is linear in dim_batch
because ~82% of a pass is fixed per-node overhead.

Run this before deploying --gradient-checkpointing to the long fit. It needs the
GPUs, so stop the running fit first (ideally right after a checkpoint write, so
no in-flight prompt is discarded).

Usage:
    python scripts/jlens/ckpt_verify.py --sweep 16,32,64,128
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

# NOTE: `jlens` (the library) must be imported BEFORE REPO_ROOT/scripts joins
# sys.path — scripts/jlens/ would otherwise shadow it as a namespace package.
import jlens  # noqa: E402
from jlens.hooks import ActivationRecorder  # noqa: E402
from jlens.lens import JacobianLens  # noqa: E402
if "filler-token-reasoning" in (jlens.__file__ or "scripts/jlens namespace pkg"):
    raise SystemExit(f"scripts/jlens shadowed the jlens library: {jlens.__file__}")

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from v3_autograd import (  # noqa: E402
    disable_block_checkpointing,
    enable_block_checkpointing,
    load_v3_awq,
    make_differentiable,
)


def sweep_dim_batch(lens_model, prompt, dim_batches, n_passes=3, max_seq_len=128):
    """Peak memory + per-pass time at full depth. Mirrors gate G4's inner loop."""
    n_gpus = torch.cuda.device_count()
    total_gib = {i: torch.cuda.get_device_properties(i).total_memory / 2**30
                 for i in range(n_gpus)}
    target = lens_model.n_layers - 1
    source_layers = list(range(target))
    results = {}

    for dim_batch in dim_batches:
        for i in range(n_gpus):
            torch.cuda.reset_peak_memory_stats(i)
        torch.cuda.empty_cache()
        tgt = srcs = cot = rec = None
        try:
            input_ids = lens_model.encode(prompt, max_length=max_seq_len)
            seq_len = input_ids.shape[1]
            with (
                ActivationRecorder(lens_model.layers, at=[*source_layers, target],
                                   start_graph_at=0) as rec,
                torch.enable_grad(),
            ):
                t0 = time.perf_counter()
                lens_model.forward(input_ids.expand(dim_batch, -1))
                torch.cuda.synchronize()
                t_fwd = time.perf_counter() - t0

                tgt = rec.activations[target]
                srcs = [rec.activations[l] for l in source_layers]
                valid = torch.arange(16, seq_len - 1, device=tgt.device)
                bidx = torch.arange(dim_batch, device=tgt.device)
                cot = torch.zeros_like(tgt)
                times, nonfinite = [], 0
                for p in range(n_passes):
                    cot.zero_()
                    cot[bidx[:, None], valid[None, :], p * dim_batch + bidx[:, None]] = 1.0
                    t1 = time.perf_counter()
                    grads = torch.autograd.grad(tgt, srcs, cot, retain_graph=True)
                    torch.cuda.synchronize()
                    times.append(time.perf_counter() - t1)
                    nonfinite += sum(int((~torch.isfinite(g)).sum().item()) for g in grads)
                    del grads
            per_pass = sum(times) / len(times)
            n_total = math.ceil(lens_model.d_model / dim_batch)
            entry = {
                "t_forward_s": round(t_fwd, 2),
                "t_backward_s": round(per_pass, 3),
                "passes": n_total,
                "est_min_per_prompt": round((t_fwd + n_total * per_pass) / 60, 1),
                "grad_nonfinite": nonfinite,
                "peak_GiB": {i: round(torch.cuda.max_memory_allocated(i) / 2**30, 1)
                             for i in range(n_gpus)},
                "min_free_GiB": round(min(
                    total_gib[i] - torch.cuda.max_memory_allocated(i) / 2**30
                    for i in range(n_gpus)), 1),
                "ok": nonfinite == 0,
            }
        except torch.cuda.OutOfMemoryError as exc:
            entry = {"OOM": str(exc)[:120], "ok": False}
        finally:
            # These must go, or the dead graph starves the next config —
            # deleting from locals() does NOT work in CPython (learned the hard way).
            del tgt, srcs, cot, rec
            torch.cuda.empty_cache()
        results[dim_batch] = entry
        print(f"  dim_batch={dim_batch}: {entry}", flush=True)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="/workspace/models/deepseek-v3-awq")
    ap.add_argument("--prompts", type=Path,
                    default=REPO_ROOT / "scripts/jlens/prompts_wikitext.json")
    ap.add_argument("--reference", type=Path,
                    default=REPO_ROOT / "outputs/jlens/lens_g5_smoke.pt",
                    help="Non-checkpointed J_59 from gate G5")
    ap.add_argument("--ref-prompts", type=int, default=2)
    ap.add_argument("--ref-dim-batch", type=int, default=16)
    ap.add_argument("--ref-layer", type=int, default=59)
    ap.add_argument("--tol", type=float, default=1e-3,
                    help="Max relative Frobenius error to accept")
    ap.add_argument("--sweep", default="16,32,64,128")
    ap.add_argument("--min-free-gib", type=float, default=8.0)
    ap.add_argument("--device-map", default="22,42")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "outputs/jlens/ckpt_verify.json")
    args = ap.parse_args()

    prompts = json.loads(args.prompts.read_text())["prompts"]
    model, tokenizer = load_v3_awq(args.model_path, device_map=args.device_map)
    lens_model = jlens.from_hf(model, tokenizer)
    patches = make_differentiable(model)
    patches["blocks_checkpointed"] = enable_block_checkpointing(model)
    print(f"  patches: {patches}", flush=True)

    # ---- equivalence gate: three fits, judged against the noise floor ------
    # A single plain-vs-checkpointed comparison cannot distinguish "checkpointing
    # is wrong" from "this backward is nondeterministic". It IS nondeterministic:
    # _moe_infer_autograd gathers every token 8 times (once per routed expert), so
    # the gather's backward is a scatter-add over duplicate indices, which uses
    # CUDA atomics; and cuBLAS may pick different GEMM algorithms depending on the
    # workspace free at the time. So fit plain TWICE to measure the floor, then
    # judge the checkpointed fit against it.
    def fit_j(tag, use_ckpt):
        if use_ckpt:
            enable_block_checkpointing(model)
        else:
            disable_block_checkpointing(model)
        print(f"\n=== fit {tag} (checkpointing={use_ckpt}) ===", flush=True)
        t = time.perf_counter()
        lens = jlens.fit(lens_model, prompts[: args.ref_prompts],
                         source_layers=[args.ref_layer], dim_batch=args.ref_dim_batch,
                         max_seq_len=128, checkpoint_path=None)
        dt = time.perf_counter() - t
        print(f"    {dt / max(lens.n_prompts, 1):.1f} s/prompt", flush=True)
        return lens.jacobians[args.ref_layer], dt / max(lens.n_prompts, 1)

    J_a, t_a = fit_j("A plain", False)
    J_b, t_b = fit_j("B plain", False)
    J_c, t_c = fit_j("C checkpointed", True)

    def rel(x, y):
        return ((x - y).norm() / y.norm()).item()

    floor = rel(J_a, J_b)
    ac, bc = rel(J_a, J_c), rel(J_b, J_c)
    ref = JacobianLens.load(str(args.reference)).jacobians[args.ref_layer]
    equivalence = {
        "noise_floor_plain_vs_plain": floor,
        "plain_A_vs_ckpt": ac,
        "plain_B_vs_ckpt": bc,
        "ratio_ckpt_over_floor": round(ac / floor, 2) if floor > 0 else None,
        "vs_stored_G5": {"A": rel(J_a, ref), "C": rel(J_c, ref)},
        "mean_diag": {"A": round(J_a.diagonal().mean().item(), 5),
                      "B": round(J_b.diagonal().mean().item(), 5),
                      "C": round(J_c.diagonal().mean().item(), 5)},
        "s_per_prompt": {"A": round(t_a, 1), "B": round(t_b, 1), "C": round(t_c, 1)},
        "criterion": "checkpointed difference must sit within 3x the plain-vs-plain "
                     "floor (or below --tol if the backward turns out deterministic)",
    }
    equivalence["passed"] = bool(
        torch.isfinite(J_c).all()
        and ac <= max(3 * floor, args.tol)
        and bc <= max(3 * floor, args.tol)
    )
    for k, v in equivalence.items():
        print(f"    {k}: {v}")

    report = {"equivalence": equivalence, "sweep": {}}
    if not equivalence["passed"]:
        print("\nEQUIVALENCE FAILED — checkpointing differs by more than run-to-run "
              "noise. Do NOT deploy it.")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, default=str))
        return
    print(f"\nEQUIVALENCE PASSED — checkpointed difference {ac:.2e} vs a "
          f"plain-vs-plain floor of {floor:.2e}: indistinguishable from noise.")

    # ---- how big can dim_batch go ----------------------------------------
    print("\n=== full-depth dim_batch sweep, checkpointing ON ===", flush=True)
    sweep = sweep_dim_batch(lens_model, prompts[0],
                            [int(x) for x in args.sweep.split(",")])
    report["sweep"] = {str(k): v for k, v in sweep.items()}

    viable = [(db, e) for db, e in sweep.items()
              if e.get("ok") and e.get("min_free_GiB", 0) >= args.min_free_gib]
    if viable:
        best_db, best = max(viable, key=lambda kv: kv[0])
        report["recommended"] = {
            "dim_batch": best_db,
            "est_min_per_prompt": best["est_min_per_prompt"],
            "min_free_GiB": best["min_free_GiB"],
            "speedup_vs_66min": round(66.2 / best["est_min_per_prompt"], 2),
        }
        print(f"\nRECOMMENDED: --gradient-checkpointing --dim-batch {best_db}  "
              f"({best['est_min_per_prompt']} min/prompt, "
              f"{report['recommended']['speedup_vs_66min']}x vs today's 66.2)")
    else:
        report["recommended"] = None
        print("\nNo dim_batch left enough headroom; keep the current config.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=str))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
