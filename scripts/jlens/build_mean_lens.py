"""
build_mean_lens.py — a J-lens that is the plain mean of per-prompt Jacobian files.

For lenses that are not terms of the n=100 running sum: the penultimate-target test
(PENULT_RUNBOOK.md) fits a handful of prompts to --target-layer 59 and compares their mean
with the mean of the SAME prompts' target-60 files, which are already on disk from the
2026-09-06 refit. build_filtered_lens.py cannot do this (it starts from the fit checkpoint).

    python scripts/jlens/build_mean_lens.py --per-prompt-dir outputs/jlens/per_prompt_target59 \
        --indices 100,101,102,103,104 --out outputs/jlens/lens_v3_target59.pt
    python scripts/jlens/build_mean_lens.py --per-prompt-dir outputs/jlens/per_prompt \
        --indices 100,101,102,103,104 --out outputs/jlens/lens_v3_target60_same_prompts.pt

Output: the reference JacobianLens.save() layout (keys J / n_prompts / source_layers / d_model,
fp16) read by score_lazy.LazyLens and every analysis script, plus <out>.meta.json with the
indices, prompt hashes, target layer and per-prompt norms. The shipped lens's pre-filter is
applied the same way: a prompt whose max||J||_F/sqrt(d) exceeds --max-norm is left out and
reported, so rerun the OTHER lens with the surviving --indices to keep the pair matched.
One layer of one file is in memory at a time, plus the output (6 GB for V3).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_filtered_lens import LayerReader  # noqa: E402


def parse_indices(text: str) -> list[int]:
    out = []
    for part in text.split(","):
        part = part.strip()
        if ":" in part:
            a, b = (int(x) for x in part.split(":"))
            out += list(range(a, b))
        elif part:
            out.append(int(part))
    return sorted(set(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-prompt-dir", type=Path, required=True)
    ap.add_argument("--indices", required=True, help="0-based corpus indices: comma list and/or a:b")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-norm", type=float, default=40.0,
                    help="leave out prompts whose max||J||_F/sqrt(d) exceeds this (the shipped "
                         "lens's pre-filter); inf keeps everything")
    ap.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16")
    args = ap.parse_args()

    readers, info, rejected = {}, {}, {}
    for i in parse_indices(args.indices):
        p = args.per_prompt_dir / f"J_p{i:04d}.pt"
        if not p.exists():
            raise SystemExit(f"missing per-prompt file for idx {i}: {p}")
        ck = torch.load(p, map_location="cpu", weights_only=False, mmap=True)
        meta = {"max_norm_over_sqrt_d": float(ck["max_norm_over_sqrt_d"]),
                "prompt_sha1": ck.get("prompt_sha1"), "settings": ck["settings"],
                "n_retries": int(ck.get("n_retries", 0)), "dtype": ck.get("dtype")}
        del ck
        if meta["max_norm_over_sqrt_d"] > args.max_norm:
            rejected[i] = meta["max_norm_over_sqrt_d"]
            continue
        readers[i], info[i] = LayerReader(p, "J"), meta
    if not readers:
        raise SystemExit(f"no prompt under --max-norm {args.max_norm}: {rejected}")
    first = next(iter(readers.values()))
    layers, d = first.layers, first.d
    targets = {m["settings"]["target_layer"] for m in info.values()}
    if len(targets) != 1:
        raise SystemExit(f"per-prompt files were fitted to different target layers: "
                         f"{ {i: m['settings']['target_layer'] for i, m in info.items()} }")
    target = targets.pop()
    for i, r in readers.items():
        if r.layers != layers or r.d != d:
            raise SystemExit(f"idx {i}: layers/d differ from idx {next(iter(readers))}")
    if layers != list(range(target)):
        raise SystemExit(f"source layers {layers[0]}..{layers[-1]} are not 0..{target - 1}")
    n = len(readers)
    print(f"{args.per_prompt_dir}: target layer {target}, source layers 0..{layers[-1]}, d={d}; "
          f"n={n} {sorted(readers)}; left out above {args.max_norm:g}: {rejected or 'none'}", flush=True)

    out_dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    J_out, per_layer, t0 = {}, {}, time.time()
    for L in layers:
        S = np.zeros((d, d), np.float64)
        for r in readers.values():
            S += r.read(L)
        J = S / n
        if not np.isfinite(J).all():
            raise SystemExit(f"non-finite lens at L{L} — refusing to write")
        per_layer[L] = {"fro": round(float(np.linalg.norm(J)), 2),
                        "fro_minus_I": round(float(np.linalg.norm(J - np.eye(d))), 2),
                        "mean_diag": round(float(np.trace(J) / d), 4)}
        J_out[L] = torch.from_numpy(J.astype(np.float32)).to(out_dtype)
        print(f"  L{L:02d} ||J||_F {per_layer[L]['fro']:8.1f}   mean diag {per_layer[L]['mean_diag']:.3f}"
              f"   (I = {np.sqrt(d):.1f})", flush=True)
    last = per_layer[layers[-1]]
    print(f"built in {(time.time() - t0) / 60:.1f} min. Sanity: the last source layer is one block from "
          f"the target, so expect mean diag ~1 and ||J||_F a little above sqrt(d): "
          f"{last['mean_diag']:.3f}, {last['fro']:.1f}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"J": J_out, "n_prompts": n, "source_layers": layers, "d_model": d}, args.out)
    sidecar = args.out.with_suffix(".meta.json")
    sidecar.write_text(json.dumps({
        "built": time.strftime("%Y-%m-%d %H:%M:%S"), "per_prompt_dir": str(args.per_prompt_dir),
        "rule": "plain mean of per-prompt Jacobians", "target_layer": target, "n_prompts": n,
        "indices": sorted(readers), "max_norm": args.max_norm, "left_out_above_max_norm": rejected,
        "prompts": {str(i): info[i] for i in sorted(info)}, "dtype": args.dtype,
        "per_layer": {str(L): v for L, v in per_layer.items()}}, indent=1))
    print(f"wrote {args.out}\n      {sidecar}", flush=True)


if __name__ == "__main__":
    main()
