"""Merge + analyze varbind KV-transplant runs.

The headline (full-swap) is pooled across the original 8-config run and the
full-swap-only top-up, deduped by (idx_R, idx_D) and filtered to both-correct, to
reach ~500 clean pairs/k. The supporting structure — the dose-response
(n3<n6<n12<full) and the layer localization (y_layers) — is read from whichever
input still carries the 8 configs (the original run), at its own N.

Reports, per condition, on the merged FULL config (both-correct):
  - change-rate (transplant answer != model's normal answer),
  - outcome split (keep / carry_answer / through_y / novel),
  - carry_answer rank: normal -> transplant median (+shift)  [the content signal],
  - through_y rank shift,
and a novel-flavor split (donor- vs receiver- vs structure-less) so "mostly
corruption" is quantified.

Usage:
    python scripts/analysis/varbind_transplant_analyze.py
    python scripts/analysis/varbind_transplant_analyze.py \
        --cond dots_25 results/varbind_kv_transplant/varbind_transplant_results.json \
                       results/varbind_kv_transplant_full_dots25/varbind_transplant_results.json
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0]))  # scripts/
from extract.extract_varbind_cot import cot_value_targets  # noqa: E402  (torch-free)

DEFAULTS = {
    "dots_10": ["results/varbind_kv_transplant_dots10/varbind_transplant_results.json",
                "results/varbind_kv_transplant_full_dots10/varbind_transplant_results.json"],
    "dots_25": ["results/varbind_kv_transplant/varbind_transplant_results.json",
                "results/varbind_kv_transplant_full_dots25/varbind_transplant_results.json"],
    "dots_50": ["results/varbind_kv_transplant_dots50/varbind_transplant_results.json",
                "results/varbind_kv_transplant_full_dots50/varbind_transplant_results.json"],
}
# Pending k=100 runs (see /workspace/claude_memory/project_varbind_k100_plan.md). Once they
# exist, analyze with --cond (op_fixed Option B needs --dataset pointed at the big dataset):
#   python scripts/analysis/varbind_transplant_analyze.py \
#       --cond dots_100 results/varbind_kv_transplant_full_dots100/varbind_transplant_results.json
#   python scripts/analysis/varbind_transplant_analyze.py \
#       --cond dots_100_opfixed results/varbind_kv_transplant_opfixed_dots100/varbind_transplant_results.json \
#       [--dataset data/chained_var_binding_dataset_big.json]


def med(xs):
    xs = [x for x in xs if x is not None]
    return float(np.median(xs)) if xs else float("nan")


def _pct(s):
    return float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def boot_cis(full, nboot=10000, seed=0):
    """Percentile-bootstrap 95% CIs over PAIRS (resample rows w/ replacement),
    matching how the 1-fact transplant CIs are reported. Returns metric ->
    (point, lo, hi). Proportions use the resample mean; ranks use nanmedian
    (None ranks, i.e. untokenizable candidates, are dropped). One shared set of
    resamples so the CIs are jointly consistent."""
    n = len(full)

    def col(getter):
        return np.array([np.nan if getter(r) is None else float(getter(r)) for r in full])

    chg = np.array([1.0 if r["configs"]["full"]["answer"] != r["ansR_norm"] else 0.0 for r in full])
    car = np.array([1.0 if r["configs"]["full"]["outcome"] == "carry_answer" else 0.0 for r in full])
    rn = col(lambda r: r["rank_norm"]["carry_answer"])
    rt = col(lambda r: r["configs"]["full"]["rank_transplant"]["carry_answer"])
    ty = np.array([(r["rank_norm"]["through_y"] - r["configs"]["full"]["rank_transplant"]["through_y"])
                   if r["rank_norm"]["through_y"] is not None
                   and r["configs"]["full"]["rank_transplant"]["through_y"] is not None
                   else np.nan for r in full])

    idx = np.random.default_rng(seed).integers(0, n, size=(nboot, n))
    out = {}
    for name, a, kind in [("changed", chg, "prop"), ("adopt", car, "prop"),
                          ("rank_norm", rn, "med"), ("rank_tr", rt, "med"), ("ty_shift", ty, "med")]:
        if kind == "med" and (a.size == 0 or np.all(np.isnan(a))):
            out[name] = (float("nan"), float("nan"), float("nan"))   # op_fixed: through_y dropped
            continue
        if kind == "prop":
            point, samp = float(np.nanmean(a)), a[idx].mean(axis=1)
        else:
            point, samp = float(np.nanmedian(a)), np.nanmedian(a[idx], axis=1)
        lo, hi = _pct(samp)
        out[name] = (point, lo, hi)
    return out


def load_merge(paths):
    """Concatenate result lists, dedup by (idx_R, idx_D) keeping the first seen."""
    rows, seen = [], set()
    for p in paths:
        if not Path(p).exists():
            continue
        for r in json.load(open(p)):
            key = (r["idx_R"], r["idx_D"])
            if key in seen:
                continue
            seen.add(key)
            rows.append(r)
    return rows


def novel_flavor(both, byidx):
    """Classify each FULL-swap 'changed' answer against a donor/receiver value
    battery → DONOR-flavored / RECV-flavored / structureless."""
    def vals(idx):
        try:
            return cot_value_targets(byidx[idx])  # {x,c1x,y,c2y,ans}
        except Exception:
            return None
    def op(ex, y):
        v = ex["coefficient"] * y
        return v + ex["constant"] if ex["operation"] == "plus" else v - ex["constant"]
    cat = Counter()
    for r in both:
        a = r["configs"]["full"]["answer"]
        if a is None or a == r["ansR_norm"]:
            continue
        Rex, Dex = byidx[r["idx_R"]], byidx[r["idx_D"]]
        vR, vD = vals(r["idx_R"]), vals(r["idx_D"])
        if vR is None or vD is None:
            continue
        donor = {vD["ans"], vD["c2y"], vD["y"], vD["c1x"], vD["x"], op(Rex, vD["y"])}
        recv = {vR["c2y"], vR["y"], vR["x"], op(Dex, vR["y"])}
        cat["DONOR" if a in donor else ("RECV" if a in recv else "structureless")] += 1
    return cat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", nargs="+", action="append", default=None,
                    help="NAME PATH [PATH...] — repeatable; overrides defaults.")
    ap.add_argument("--dataset", type=Path, default=Path("data/chained_var_binding_dataset.json"))
    ap.add_argument("--nboot", type=int, default=10000,
                    help="Bootstrap resamples for the 95%% CIs (over pairs).")
    ap.add_argument("--seed", type=int, default=0, help="Bootstrap RNG seed (reproducible CIs).")
    args = ap.parse_args()

    conds = {c[0]: c[1:] for c in args.cond} if args.cond else DEFAULTS
    byidx = {e["idx"]: e for e in json.load(open(args.dataset))["examples"]}

    for cond, paths in conds.items():
        rows = load_merge(paths)
        both = [r for r in rows if r["R_correct"] and r["D_correct"]]
        if not both:
            print(f"\n===== {cond}: no data ({[p for p in paths if Path(p).exists()]}) =====")
            continue
        full = [r for r in both if "full" in r["configs"]]
        n = len(full)
        outs = Counter(r["configs"]["full"]["outcome"] for r in full)
        flavor = novel_flavor(full, byidx)
        nflav = sum(flavor.values())
        ci = boot_cis(full, args.nboot, args.seed)
        nuR, nuD = len({r["idx_R"] for r in full}), len({r["idx_D"] for r in full})

        def _p(key):   # proportion -> "27.0%  (95% CI 23.3-31.1)"
            p, lo, hi = ci[key]
            return f"{p*100:.1f}%  (95% CI {lo*100:.1f}-{hi*100:.1f})"

        def _r(key):   # median rank -> "70 (57-85)"
            p, lo, hi = ci[key]
            return f"{p:.0f} ({lo:.0f}-{hi:.0f})"

        print(f"\n===== {cond}  (merged: {len(rows)} pairs, both-correct {len(both)}, full-config n={n};"
              f" {nuR} uniq receivers / {nuD} uniq donors) =====")
        print(f"  HEADLINE (full swap; 95% CI = {args.nboot}x bootstrap over pairs, seed {args.seed}):")
        print(f"    answer changed vs normal : {_p('changed')}")
        print(f"    donor adoption (carry)   : {_p('adopt')}")
        print(f"    outcome: keep={outs.get('keep',0)/n:.1%}  carry_answer={outs.get('carry_answer',0)/n:.1%}"
              f"  through_y={outs.get('through_y',0)/n:.1%}  novel={outs.get('novel',0)/n:.1%}")
        print(f"    carry_answer rank        : {_r('rank_norm')} -> {_r('rank_tr')}   [content signal]")
        p, lo, hi = ci["ty_shift"]
        if np.isnan(p):
            print(f"    through_y rank shift      : n/a (op-fixed: through_y == carry_answer)")
        else:
            print(f"    through_y rank shift      : {p:+.0f}  (95% CI {lo:+.0f}-{hi:+.0f})")
        if nflav:
            print(f"    of {nflav} changes: "
                  + "  ".join(f"{k}={v}({v/nflav:.0%})" for k, v in flavor.most_common()))

        # supporting structure from whichever input still has the 8 configs
        labels = [l for l in full[0]["configs"] if l != "full"]
        if labels:
            print(f"  SUPPORTING (8-config rows, n={n}):  carry_answer rank shift by config")
            for L in ["n3", "n6", "n12", "first6", "first12", "last12", "y_layers"]:
                xs = [r["rank_norm"]["carry_answer"] - r["configs"][L]["rank_transplant"]["carry_answer"]
                      for r in full if L in r["configs"]]
                if xs:
                    print(f"    {L:9s} shift {med(xs):+6.0f}")


if __name__ == "__main__":
    main()
