#!/usr/bin/env python3
"""Preflight gate 5: save Kimi K2.5's lm_head + final RMSNorm weights.

Both live in shard 62 of 64 and are BF16 (never quantized), so they can be pulled
straight out of the safetensors on CPU while the GPUs are busy with something else.
Every decode/logit-lens script downstream needs them, and at 2.4 GB they are the
first thing worth shipping off the pod.

Usage:
    /root/.venvs/k25/bin/python scripts/kimi_k25/save_lm_head_k25.py \
        --model-path /workspace/models/kimi-k2.5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

LM_HEAD_KEY = "language_model.lm_head.weight"
NORM_KEY = "language_model.model.norm.weight"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", type=Path, default=Path("/workspace/models/kimi-k2.5"))
    ap.add_argument("--out-dir", type=Path,
                    default=Path("data/model_weights/kimi_k25"))
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    lm_head_path = args.out_dir / "lm_head_weight.npy"
    norm_path = args.out_dir / "rms_norm_weight.npy"

    index = json.loads((args.model_path / "model.safetensors.index.json").read_text())
    wm = index["weight_map"]
    for key in (LM_HEAD_KEY, NORM_KEY):
        if key not in wm:
            raise SystemExit(f"GATE 5 FAIL: {key!r} not in weight_map")
    print(f"{LM_HEAD_KEY} -> {wm[LM_HEAD_KEY]}")
    print(f"{NORM_KEY}    -> {wm[NORM_KEY]}")

    def _load(key: str) -> torch.Tensor:
        with safe_open(args.model_path / wm[key], framework="pt") as f:
            return f.get_tensor(key)

    if lm_head_path.exists() and norm_path.exists():
        print(f"Already saved at {args.out_dir}, verifying shapes only")
        lm = np.load(lm_head_path, mmap_mode="r")
        nm = np.load(norm_path, mmap_mode="r")
        print(f"  lm_head {lm.shape} {lm.dtype}   norm {nm.shape} {nm.dtype}")
        return

    # lm_head: (vocab, hidden) fp16 — matches data/model_weights/kimi_k2/ convention
    lm_head = _load(LM_HEAD_KEY)
    print(f"  loaded lm_head {tuple(lm_head.shape)} {lm_head.dtype}")
    lm_head_np = lm_head.to(torch.float16).cpu().numpy()
    np.save(lm_head_path, lm_head_np)
    print(f"Saved {lm_head_path}: {lm_head_np.shape} {lm_head_np.dtype} "
          f"({lm_head_path.stat().st_size / 2**30:.2f} GiB)")
    del lm_head, lm_head_np

    # final norm: fp32 (it is applied in fp32 by the logit lens)
    norm = _load(NORM_KEY)
    norm_np = norm.to(torch.float32).cpu().numpy()
    np.save(norm_path, norm_np)
    print(f"Saved {norm_path}: {norm_np.shape} {norm_np.dtype}")

    if not np.isfinite(norm_np).all():
        raise SystemExit("GATE 5 FAIL: non-finite values in rms_norm weight")
    print("GATE 5 PASS")


if __name__ == "__main__":
    main()
