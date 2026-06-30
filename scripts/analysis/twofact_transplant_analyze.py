"""
Analyze the position-resolved 2-fact KV transplant (fact-matched, threshold scopes).

For each direction (vary_a1 tests the A1 channel, vary_a2 the A2 channel), on the
both-correct subset, reports for `whole` / `<a>_pos@theta` / `no_<a>_pos@theta`:
the donor-answer rank (norm->transplant), the median rank shift, and the donor
adoption, each with a bootstrap 95% CI. The headline: the heatmap-flagged positions
carry their addend (pos moves the donor answer; no_pos much less / not at all).

Usage:
    python scripts/analysis/twofact_transplant_analyze.py --root results/twofact_kv_transplant
"""
import argparse
import json
from pathlib import Path

import numpy as np


def boot_ci(vals, fn, n=10000, seed=0):
    v = np.asarray(vals, dtype=float)
    if len(v) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n, len(v)))
    s = fn(v[idx], axis=1)
    return float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def donor_shift(r, label):
    a, b = r["rank_norm"]["donor"], r["configs"][label]["rank_transplant"]["donor"]
    return None if (a is None or b is None) else a - b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("results/twofact_kv_transplant"))
    ap.add_argument("--cond", default="dots_50")
    ap.add_argument("--thetas", default="0.15,0.2,0.3")
    args = ap.parse_args()
    thetas = args.thetas.split(",")
    summary = {}

    for direction in ["vary_a1", "vary_a2"]:
        tag = "a1" if direction == "vary_a1" else "a2"
        p = args.root / f"{args.cond}_{direction}" / "twofact_transplant_results.json"
        if not p.exists():
            print(f"\n{direction}: no data"); continue
        d = json.load(open(p))
        res = [r for r in d["results"] if r["T_correct"] and r["D_correct"]]
        cfgmeta = d["meta"]["configs"]
        rn = np.median([r["rank_norm"]["donor"] for r in res if r["rank_norm"]["donor"] is not None])
        print(f"\n{'='*78}\n{args.cond} / {direction}  ({tag.upper()} channel)  "
              f"n={len(res)} both-correct of {len(d['results'])};  baseline donor rank median={rn:.0f}\n{'='*78}")
        print(f"  {'scope':16s} {'n':>3}  {'rank->':>7}  {'shift [95% CI]':>20}  {'adoption [95% CI]':>22}")

        def row(label):
            sh = [s for s in (donor_shift(r, label) for r in res) if s is not None]
            ad = np.array([r["configs"][label]["outcome"] == "donor" for r in res], float)
            rt = np.median([r["configs"][label]["rank_transplant"]["donor"] for r in res
                            if r["configs"][label]["rank_transplant"]["donor"] is not None])
            med = float(np.median(sh)) if sh else float("nan")
            slo, shi = boot_ci(sh, np.median)
            alo, ahi = boot_ci(ad, np.mean)
            ns = len(cfgmeta[label])
            print(f"  {label:16s} {ns:>3}  {rt:>7.0f}  {med:>+5.0f} [{slo:>+4.0f},{shi:>+4.0f}]   "
                  f"{ad.mean():>6.1%} [{alo:>5.1%},{ahi:>5.1%}]")
            summary[f"{direction}/{label}"] = {"n": len(res), "n_swap": ns, "rank_transplant": float(rt),
                                               "shift": med, "shift_ci": [slo, shi],
                                               "adoption": float(ad.mean()), "adoption_ci": [alo, ahi]}

        row("whole")
        for th in thetas:
            row(f"{tag}_pos@{th}")
            row(f"no_{tag}_pos@{th}")

    out = args.root / "twofact_transplant_summary.json"
    json.dump(summary, open(out, "w"), indent=2)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
