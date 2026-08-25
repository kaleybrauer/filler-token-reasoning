"""Pair the MLP-write arm and the residual-stream arm on problem_idx.

REEXTRACT_RUNBOOK.md §7: the two Kimi K2 extraction arms differ ONLY in what the
forward hook captured — same model, same datasets, same conditions, same positions,
and generation is greedy (SamplingParams(temperature=0)). So the model-correct subsets
*should* be identical and the arms pair per problem_idx. But MoE + tensor-parallel
reduction order can flip a token occasionally, so this is measured, not assumed.

Reads model_answer / model_correct out of every pkl in both arms (parallel, since each
pkl is 17-90 MB of states) and reports, per condition:

    n paired, accuracy of each arm, how many examples disagree on the generated answer,
    and the size of the correct-in-both intersection that downstream decode should use.

Usage:
    python scripts/kimi_k2/compare_arms_generation.py \
        --old data/kimi-k2/extracted_states_2fact_allpos_kimi_k2 \
        --new data/kimi-k2/extracted_states_2fact_allpos_kimi_k2_v2 \
        --out results/kimi_k2_arm_pairing_2fact.json
"""
from __future__ import annotations

import argparse
import json
import pickle
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


def read_one(path_str: str):
    """Return (problem_idx, model_answer, model_correct, capture_convention)."""
    try:
        with open(path_str, "rb") as f:
            d = pickle.load(f)
    except Exception as e:  # truncated / unreadable pkl
        return (None, None, None, f"ERROR: {type(e).__name__}")
    return (
        d.get("problem_idx"),
        d.get("model_answer"),
        d.get("model_correct"),
        d.get("capture_convention"),
    )


def read_arm(cond_dir: Path, workers: int):
    files = sorted(str(p) for p in cond_dir.glob("prob_*.pkl"))
    if not files:
        return {}, set()
    out, conventions = {}, set()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for idx, ans, corr, conv in ex.map(read_one, files, chunksize=8):
            if idx is None:
                continue
            out[idx] = (ans, corr)
            conventions.add(conv)
    return out, conventions


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True, type=Path, help="MLP-write arm dir")
    ap.add_argument("--new", required=True, type=Path, help="residual-stream (_v2) arm dir")
    ap.add_argument("--conditions", nargs="+", default=[
        "dots_10", "dots_25", "dots_50", "counting_10", "counting_25", "counting_50"])
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    report = {"old_dir": str(args.old), "new_dir": str(args.new), "conditions": {}}

    hdr = (f"{'condition':<14} {'n_old':>6} {'n_new':>6} {'paired':>7} "
           f"{'acc_old':>8} {'acc_new':>8} {'ans_diff':>9} {'both_ok':>8} "
           f"{'old_only':>9} {'new_only':>9}")
    print(hdr)
    print("-" * len(hdr))

    for cond in args.conditions:
        old_dir, new_dir = args.old / cond, args.new / cond
        if not old_dir.is_dir() or not new_dir.is_dir():
            print(f"{cond:<14} missing ({'old' if not old_dir.is_dir() else 'new'})")
            continue

        old, conv_old = read_arm(old_dir, args.workers)
        new, conv_new = read_arm(new_dir, args.workers)
        shared = sorted(set(old) & set(new))

        acc_old = sum(old[i][1] is True for i in old) / len(old) if old else float("nan")
        acc_new = sum(new[i][1] is True for i in new) / len(new) if new else float("nan")
        ans_diff = sum(old[i][0] != new[i][0] for i in shared)
        both_ok = sum(old[i][1] is True and new[i][1] is True for i in shared)
        old_only = sum(old[i][1] is True and new[i][1] is not True for i in shared)
        new_only = sum(new[i][1] is True and old[i][1] is not True for i in shared)

        print(f"{cond:<14} {len(old):6d} {len(new):6d} {len(shared):7d} "
              f"{acc_old:8.1%} {acc_new:8.1%} {ans_diff:9d} {both_ok:8d} "
              f"{old_only:9d} {new_only:9d}")

        report["conditions"][cond] = {
            "n_old": len(old), "n_new": len(new), "n_paired": len(shared),
            "acc_old": acc_old, "acc_new": acc_new,
            "n_answer_differs": ans_diff,
            "n_correct_both": both_ok,
            "n_correct_old_only": old_only,
            "n_correct_new_only": new_only,
            "capture_convention_old": sorted(str(c) for c in conv_old),
            "capture_convention_new": sorted(str(c) for c in conv_new),
        }

    print("\nacc_* are each arm's own accuracy over its own examples; both_ok is the "
          "correct-in-both intersection downstream decode should be reported on.")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
