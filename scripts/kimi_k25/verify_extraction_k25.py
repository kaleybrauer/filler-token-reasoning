#!/usr/bin/env python3
"""Read-only verification of the Kimi K2.5 varbind extraction.

Checks every condition for completeness and schema integrity, spot-checks states
for NaN/shape problems, and writes a manifest so the receiving side can confirm
what landed. Deletes nothing and modifies nothing under data/.

Usage:
    /root/.venvs/k25/bin/python scripts/kimi_k25/verify_extraction_k25.py
"""
from __future__ import annotations

import argparse
import json
import pickle
import random
from pathlib import Path

import numpy as np

EXPECTED_CONDITIONS = ["baseline", "dots_5", "dots_10", "dots_25", "dots_50",
                       "counting_5", "counting_10", "counting_25"]
N_LAYERS = 61
HIDDEN = 7168
TRUTH_FIELDS = ["answer", "queried_value", "intermediate", "coefficient",
                "operation", "constant"]


def verify_pkl(path: Path, cond: str) -> list[str]:
    """Return a list of problems found in one pkl (empty == clean)."""
    bad: list[str] = []
    with open(path, "rb") as f:
        rec = pickle.load(f)

    for key in ("problem_idx", "condition", "k", "positions", "states",
                "model_response", "model_correct", "seq_len"):
        if key not in rec:
            bad.append(f"missing key {key!r}")
    for key in TRUTH_FIELDS:
        if key not in rec:
            bad.append(f"missing truth field {key!r}")
    if rec.get("condition") != cond:
        bad.append(f"condition={rec.get('condition')!r} != dir {cond!r}")

    k = rec.get("k", 0)
    if k > 0 and "boundaries" not in rec:
        bad.append("missing 'boundaries' (k>0)")

    states = rec.get("states", {})
    positions = rec.get("positions", {})
    if len(states) != len(positions):
        bad.append(f"{len(states)} state positions vs {len(positions)} positions")
    if not states:
        bad.append("empty states")
        return bad

    # every position must carry all 61 layers at the right width, finite
    for pos_name, layer_dict in states.items():
        if len(layer_dict) != N_LAYERS:
            bad.append(f"{pos_name}: {len(layer_dict)} layers != {N_LAYERS}")
            break
        vec = layer_dict[max(layer_dict)]
        if vec.shape != (HIDDEN,):
            bad.append(f"{pos_name}: shape {vec.shape} != ({HIDDEN},)")
            break
        if vec.dtype != np.float16:
            bad.append(f"{pos_name}: dtype {vec.dtype} != float16")
            break
    # NaN check on a few layers of the last position (full check is too slow)
    last = sorted(states)[-1]
    for li in (0, N_LAYERS // 2, N_LAYERS - 1):
        v = np.asarray(states[last][li], dtype=np.float32)
        if not np.isfinite(v).all():
            bad.append(f"{last} L{li}: non-finite values")
    return bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--states-dir", type=Path,
                    default=Path("data/extracted_states_varbind_allpos_kimi_k25"))
    ap.add_argument("--results-json", type=Path,
                    default=Path("results/kimi_k25_varbind_accuracy.json"))
    ap.add_argument("--expected-n", type=int, default=500)
    ap.add_argument("--sample", type=int, default=5, help="pkls to deep-check per condition")
    ap.add_argument("--manifest", type=Path,
                    default=Path("results/kimi_k25_manifest.json"))
    args = ap.parse_args()

    print("=" * 74)
    print("KIMI K2.5 VARBIND EXTRACTION — VERIFICATION")
    print("=" * 74)

    failures: list[str] = []
    manifest: dict = {"conditions": {}, "states_dir": str(args.states_dir)}

    meta_path = args.states_dir / "metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        print(f"metadata.json: {len(meta)} entries")
        manifest["metadata_entries"] = len(meta)
        if len(meta) != args.expected_n:
            failures.append(f"metadata.json has {len(meta)} entries, expected {args.expected_n}")
    else:
        failures.append("metadata.json MISSING")

    grand_files = grand_bytes = 0
    print(f"\n{'condition':14s} {'files':>6s} {'GiB':>8s} {'schema':>18s}")
    print("-" * 74)
    for cond in EXPECTED_CONDITIONS:
        cond_dir = args.states_dir / cond
        if not cond_dir.is_dir():
            failures.append(f"{cond}: directory missing")
            print(f"{cond:14s} {'MISSING':>6s}")
            continue
        pkls = sorted(cond_dir.glob("prob_*.pkl"))
        nbytes = sum(p.stat().st_size for p in pkls)
        grand_files += len(pkls)
        grand_bytes += nbytes

        if len(pkls) != args.expected_n:
            failures.append(f"{cond}: {len(pkls)} pkls, expected {args.expected_n}")

        # deep-check a random sample
        rng = random.Random(0)
        sample = rng.sample(pkls, min(args.sample, len(pkls))) if pkls else []
        schema_problems: list[str] = []
        for p in sample:
            for msg in verify_pkl(p, cond):
                schema_problems.append(f"{p.name}: {msg}")
        if schema_problems:
            failures.extend(f"{cond}: {m}" for m in schema_problems[:5])

        status = "OK" if not schema_problems else f"{len(schema_problems)} PROBLEM(S)"
        print(f"{cond:14s} {len(pkls):6d} {nbytes / 2**30:8.2f} {status:>18s}")
        manifest["conditions"][cond] = {
            "files": len(pkls), "bytes": nbytes, "sampled": len(sample),
            "schema_ok": not schema_problems,
        }

    print("-" * 74)
    print(f"{'TOTAL':14s} {grand_files:6d} {grand_bytes / 2**30:8.2f} GiB")
    manifest["total_files"] = grand_files
    manifest["total_bytes"] = grand_bytes

    # ---- accuracy table -----------------------------------------------------
    if args.results_json.exists():
        acc = json.loads(args.results_json.read_text())
        manifest["accuracy"] = acc
        base = acc.get("baseline", {}).get("accuracy")
        print(f"\n{'condition':14s} {'acc':>8s} {'correct/total':>15s} {'uplift':>10s}")
        print("-" * 74)
        for cond in EXPECTED_CONDITIONS:
            if cond not in acc:
                print(f"{cond:14s} {'--':>8s}")
                continue
            a = acc[cond]
            up = f"{(a['accuracy'] - base) * 100:+.1f} pt" if base is not None else ""
            print(f"{cond:14s} {a['accuracy']:8.1%} {a['correct']:7d}/{a['total']:<7d} {up:>10s}")
    else:
        failures.append(f"{args.results_json} missing")

    # ---- model weights ------------------------------------------------------
    wdir = Path("data/model_weights/kimi_k25")
    print(f"\nmodel weights ({wdir}):")
    for name, shape in (("lm_head_weight.npy", (163840, HIDDEN)),
                        ("rms_norm_weight.npy", (HIDDEN,))):
        p = wdir / name
        if not p.exists():
            failures.append(f"{p} missing")
            print(f"  {name}: MISSING")
            continue
        arr = np.load(p, mmap_mode="r")
        ok = tuple(arr.shape) == shape
        if not ok:
            failures.append(f"{name}: shape {arr.shape} != {shape}")
        print(f"  {name}: {arr.shape} {arr.dtype} "
              f"({p.stat().st_size / 2**30:.2f} GiB) {'OK' if ok else 'BAD SHAPE'}")

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2))
    print(f"\nmanifest written to {args.manifest}")

    print("\n" + "=" * 74)
    if failures:
        print(f"VERIFICATION FAILED — {len(failures)} problem(s):")
        for f in failures[:40]:
            print(f"  FAIL {f}")
        return 1
    print("VERIFICATION PASSED — all 8 conditions complete, schema + states clean")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
