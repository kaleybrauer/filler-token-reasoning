"""
analyze_results.py

Reads the CSV from run_experiment_grid.py (logit inspection version).
Analyzes ranks and probabilities to determine:
  1. Where patchscoping recovers A1, A2, sum
  2. Best (template, target_layer) configurations
  3. Positive/negative control validation
  4. Template and target layer offset effects
  5. What the model IS representing where it's not A1/A2/sum

Usage:
    python analyze_results.py
    python analyze_results.py --input results/patchscope_grid.csv
"""

import argparse
import csv
import os
import sys
from collections import defaultdict, Counter
import numpy as np


# ============================================================
# DATA LOADING
# ============================================================

def load_results(path):
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    for r in rows:
        for k in ["A1", "A2", "sum", "source_layer", "target_layer",
                   "A1_rank", "A2_rank", "sum_rank", "diff_rank"]:
            if k in r and r[k] != "" and r[k] != "-1":
                try:
                    r[k] = int(r[k])
                except ValueError:
                    r[k] = -1
            elif r.get(k) in ("-1", ""):
                r[k] = -1

        for k in ["A1_prob", "A2_prob", "sum_prob", "diff_prob", "top_1_prob"]:
            if k in r and r[k] != "":
                try:
                    r[k] = float(r[k])
                except ValueError:
                    r[k] = 0.0
            else:
                r[k] = 0.0

        if r.get("top_1_number") not in ("", None):
            try:
                r["top_1_number"] = int(r["top_1_number"])
            except ValueError:
                r["top_1_number"] = None
        else:
            r["top_1_number"] = None

    return rows


def group_by(rows, keys):
    groups = defaultdict(list)
    for r in rows:
        key = tuple(r[k] for k in keys)
        groups[key].append(r)
    return groups


# ============================================================
# METRICS
# ============================================================

def median_rank(rows, field):
    ranks = [r[field] for r in rows if r[field] >= 0]
    return np.median(ranks) if ranks else -1

def mean_rank(rows, field):
    ranks = [r[field] for r in rows if r[field] >= 0]
    return np.mean(ranks) if ranks else -1

def frac_top_k(rows, field, k):
    """Fraction of rows where rank < k (i.e., in top-k)."""
    valid = [r for r in rows if r[field] >= 0]
    if not valid:
        return 0.0
    return sum(1 for r in valid if r[field] < k) / len(valid)

def mean_prob(rows, field):
    probs = [r[field] for r in rows if r[field] > 0]
    return np.mean(probs) if probs else 0.0


# ============================================================
# ANALYSIS SECTIONS
# ============================================================

def section_overview(rows):
    print("\n" + "=" * 80)
    print("0. OVERVIEW")
    print("=" * 80)
    print(f"\n  Total rows: {len(rows)}")
    errors = [r for r in rows if str(r.get("top_1_token", "")).startswith("ERROR")]
    print(f"  Errors: {len(errors)}")

    for name, field in [("A1", "A1_rank"), ("A2", "A2_rank"), ("Sum", "sum_rank")]:
        top1 = frac_top_k(rows, field, 1)
        top5 = frac_top_k(rows, field, 5)
        top10 = frac_top_k(rows, field, 10)
        top50 = frac_top_k(rows, field, 50)
        med = median_rank(rows, field)
        print(f"\n  {name}:")
        print(f"    Top-1: {top1:.1%} | Top-5: {top5:.1%} | Top-10: {top10:.1%} | Top-50: {top50:.1%}")
        print(f"    Median rank: {med:.0f} | Mean prob: {mean_prob(rows, field.replace('rank','prob')):.4f}")


def section_accuracy_table(rows):
    """Full accuracy table grouped by configuration."""
    print("\n" + "=" * 80)
    print("1. CONFIGURATION TABLE (sorted by A2 top-10 rate)")
    print("=" * 80)

    keys = ["source_layer", "source_pos", "template", "target_layer"]
    groups = group_by(rows, keys)

    results = []
    for key, group in groups.items():
        src_layer, src_pos, template, tgt_layer = key
        results.append({
            "source_layer": src_layer,
            "source_pos": src_pos,
            "template": template,
            "target_layer": tgt_layer,
            "n": len(group),
            "A1_top1": frac_top_k(group, "A1_rank", 1),
            "A1_top10": frac_top_k(group, "A1_rank", 10),
            "A2_top1": frac_top_k(group, "A2_rank", 1),
            "A2_top10": frac_top_k(group, "A2_rank", 10),
            "sum_top1": frac_top_k(group, "sum_rank", 1),
            "sum_top10": frac_top_k(group, "sum_rank", 10),
            "A1_med": median_rank(group, "A1_rank"),
            "A2_med": median_rank(group, "A2_rank"),
            "sum_med": median_rank(group, "sum_rank"),
        })

    results.sort(key=lambda x: x["A2_top10"], reverse=True)

    print(f"\n{'srcL':>4} {'srcP':>8} {'tgtL':>4} "
          f"{'A1@1':>5} {'A1@10':>5} {'A2@1':>5} {'A2@10':>5} {'S@1':>5} {'S@10':>5} "
          f"{'A1md':>5} {'A2md':>5} {'Smd':>5}  template")
    print("-" * 110)

    for r in results[:50]:
        print(f"{r['source_layer']:>4} {r['source_pos']:>8} {r['target_layer']:>4} "
              f"{r['A1_top1']:>5.0%} {r['A1_top10']:>5.0%} "
              f"{r['A2_top1']:>5.0%} {r['A2_top10']:>5.0%} "
              f"{r['sum_top1']:>5.0%} {r['sum_top10']:>5.0%} "
              f"{r['A1_med']:>5.0f} {r['A2_med']:>5.0f} {r['sum_med']:>5.0f}  "
              f"{r['template']}")

    if len(results) > 50:
        print(f"  ... ({len(results) - 50} more rows, showing top 50 by A2 top-10)")

    return results


def section_best_configs(results):
    """Top configs for each quantity."""
    print("\n" + "=" * 80)
    print("2. BEST CONFIGURATIONS")
    print("=" * 80)

    for quantity, field in [("A2", "A2_top10"), ("A1", "A1_top10"), ("Sum", "sum_top10")]:
        sorted_r = sorted(results, key=lambda x: x[field], reverse=True)
        print(f"\n  Top 5 for {quantity} (top-10 rate):")
        for i, r in enumerate(sorted_r[:5]):
            print(f"    {i+1}. {r[field]:.0%} — "
                  f"L{r['source_layer']}/{r['source_pos']} -> L{r['target_layer']}, "
                  f"\"{r['template']}\"")

    for quantity, field in [("A2", "A2_top1"), ("A1", "A1_top1"), ("Sum", "sum_top1")]:
        sorted_r = sorted(results, key=lambda x: x[field], reverse=True)
        print(f"\n  Top 5 for {quantity} (top-1 rate / exact match):")
        for i, r in enumerate(sorted_r[:5]):
            print(f"    {i+1}. {r[field]:.0%} — "
                  f"L{r['source_layer']}/{r['source_pos']} -> L{r['target_layer']}, "
                  f"\"{r['template']}\"")


def section_positive_control(rows):
    """A1 recovery at logit-lens-good positions."""
    print("\n" + "=" * 80)
    print("3. POSITIVE CONTROL — A1 at (layer 50, pos_001)")
    print("=" * 80)

    control = [r for r in rows if r["source_layer"] == 50 and r["source_pos"] == "pos_001"]
    if not control:
        print("  No rows found.")
        return

    print(f"\n  n={len(control)}")
    print(f"  A1 top-1: {frac_top_k(control, 'A1_rank', 1):.0%}")
    print(f"  A1 top-5: {frac_top_k(control, 'A1_rank', 5):.0%}")
    print(f"  A1 top-10: {frac_top_k(control, 'A1_rank', 10):.0%}")
    print(f"  A1 median rank: {median_rank(control, 'A1_rank'):.0f}")

    by_template = group_by(control, ["template"])
    print(f"\n  By template:")
    for (template,), group in sorted(by_template.items(),
                                      key=lambda x: frac_top_k(x[1], "A1_rank", 1), reverse=True):
        a1 = frac_top_k(group, "A1_rank", 1)
        a2 = frac_top_k(group, "A2_rank", 10)
        print(f"    A1@1={a1:.0%}, A2@10={a2:.0%}  \"{template}\"")

    by_tgt = group_by(control, ["target_layer"])
    print(f"\n  By target layer:")
    for (tgt,), group in sorted(by_tgt.items()):
        a1 = frac_top_k(group, "A1_rank", 1)
        a2 = frac_top_k(group, "A2_rank", 10)
        print(f"    target_layer={tgt}: A1@1={a1:.0%}, A2@10={a2:.0%}")


def section_negative_control(rows):
    """Early layer should show nothing."""
    print("\n" + "=" * 80)
    print("4. NEGATIVE CONTROL — layer 20")
    print("=" * 80)

    control = [r for r in rows if r["source_layer"] == 20]
    if not control:
        print("  No rows found.")
        return

    print(f"\n  n={len(control)}")
    for name, field in [("A1", "A1_rank"), ("A2", "A2_rank"), ("Sum", "sum_rank")]:
        t1 = frac_top_k(control, field, 1)
        t10 = frac_top_k(control, field, 10)
        med = median_rank(control, field)
        print(f"  {name}: top-1={t1:.0%}, top-10={t10:.0%}, median_rank={med:.0f}")

    top1_counts = Counter(r.get("top_1_token", "?") for r in control)
    print(f"\n  Most common top-1 tokens:")
    for tok, count in top1_counts.most_common(10):
        print(f"    {repr(tok):>20}: {count} ({100*count/len(control):.0f}%)")


def section_template_comparison(rows):
    """Template effectiveness across mid/late layers."""
    print("\n" + "=" * 80)
    print("5. TEMPLATE COMPARISON (source layers >= 40)")
    print("=" * 80)

    mid_late = [r for r in rows if r["source_layer"] >= 40]
    by_template = group_by(mid_late, ["template"])

    print(f"\n  {'A1@1':>5} {'A1@10':>6} {'A2@1':>5} {'A2@10':>6} {'S@1':>5} {'S@10':>5} {'n':>5}  template")
    print("  " + "-" * 80)

    sorted_t = sorted(by_template.items(),
                       key=lambda x: frac_top_k(x[1], "A2_rank", 10), reverse=True)
    for (template,), group in sorted_t:
        print(f"  {frac_top_k(group, 'A1_rank', 1):>5.0%} "
              f"{frac_top_k(group, 'A1_rank', 10):>6.0%} "
              f"{frac_top_k(group, 'A2_rank', 1):>5.0%} "
              f"{frac_top_k(group, 'A2_rank', 10):>6.0%} "
              f"{frac_top_k(group, 'sum_rank', 1):>5.0%} "
              f"{frac_top_k(group, 'sum_rank', 10):>5.0%} "
              f"{len(group):>5}  \"{template}\"")


def section_target_layer_offset(rows):
    """Effect of target layer offset."""
    print("\n" + "=" * 80)
    print("6. TARGET LAYER OFFSET (source layers >= 40)")
    print("=" * 80)

    mid_late = [r for r in rows if r["source_layer"] >= 40]
    for r in mid_late:
        r["_offset"] = r["target_layer"] - r["source_layer"]

    by_offset = group_by(mid_late, ["_offset"])

    print(f"\n  {'offset':>6} {'A1@1':>5} {'A1@10':>6} {'A2@1':>5} {'A2@10':>6} "
          f"{'S@1':>5} {'S@10':>5} {'n':>5}")
    print("  " + "-" * 55)

    for offset in sorted(by_offset.keys()):
        group = by_offset[offset]
        print(f"  {offset[0]:>+6} "
              f"{frac_top_k(group, 'A1_rank', 1):>5.0%} "
              f"{frac_top_k(group, 'A1_rank', 10):>6.0%} "
              f"{frac_top_k(group, 'A2_rank', 1):>5.0%} "
              f"{frac_top_k(group, 'A2_rank', 10):>6.0%} "
              f"{frac_top_k(group, 'sum_rank', 1):>5.0%} "
              f"{frac_top_k(group, 'sum_rank', 10):>5.0%} "
              f"{len(group):>5}")


def section_source_comparison(rows):
    """Compare source (layer, position) pairs."""
    print("\n" + "=" * 80)
    print("7. SOURCE POSITION COMPARISON")
    print("=" * 80)

    by_source = group_by(rows, ["source_layer", "source_pos"])

    print(f"\n  {'srcL':>4} {'srcP':>8} "
          f"{'A1@1':>5} {'A1@10':>6} {'A2@1':>5} {'A2@10':>6} "
          f"{'S@1':>5} {'S@10':>5} {'n':>5}  desc")
    print("  " + "-" * 75)

    for (src_layer, src_pos), group in sorted(by_source.items()):
        desc = group[0].get("source_desc", "")
        print(f"  {src_layer:>4} {src_pos:>8} "
              f"{frac_top_k(group, 'A1_rank', 1):>5.0%} "
              f"{frac_top_k(group, 'A1_rank', 10):>6.0%} "
              f"{frac_top_k(group, 'A2_rank', 1):>5.0%} "
              f"{frac_top_k(group, 'A2_rank', 10):>6.0%} "
              f"{frac_top_k(group, 'sum_rank', 1):>5.0%} "
              f"{frac_top_k(group, 'sum_rank', 10):>5.0%} "
              f"{len(group):>5}  {desc}")


def section_what_is_top1(rows):
    """Analyze what the model IS predicting when it's not A1/A2/sum."""
    print("\n" + "=" * 80)
    print("8. WHAT IS TOP-1? (source layers >= 40)")
    print("=" * 80)

    mid_late = [r for r in rows if r["source_layer"] >= 40]

    categories = Counter()
    for r in mid_late:
        top1_num = r["top_1_number"]
        a1, a2, s = r["A1"], r["A2"], r["sum"]

        if top1_num is None:
            categories["non-numeric"] += 1
        elif top1_num == a1:
            categories["A1"] += 1
        elif top1_num == a2:
            categories["A2"] += 1
        elif top1_num == s:
            categories["sum"] += 1
        elif top1_num == abs(a1 - a2):
            categories["diff"] += 1
        else:
            categories["other_number"] += 1

    print(f"\n  Top-1 token identity:")
    total = len(mid_late)
    for cat, count in categories.most_common():
        print(f"    {cat:>15}: {count:>5} ({100*count/total:.1f}%)")

    other_nums = []
    for r in mid_late:
        top1_num = r["top_1_number"]
        a1, a2, s = r["A1"], r["A2"], r["sum"]
        if top1_num is not None and top1_num not in (a1, a2, s, abs(a1-a2)):
            other_nums.append(top1_num)

    if other_nums:
        print(f"\n  Most common 'other' top-1 numbers:")
        for num, count in Counter(other_nums).most_common(15):
            print(f"    {num:>6}: {count}")

    non_num_tokens = [r["top_1_token"] for r in mid_late if r["top_1_number"] is None]
    if non_num_tokens:
        print(f"\n  Most common non-numeric top-1 tokens:")
        for tok, count in Counter(non_num_tokens).most_common(10):
            print(f"    {repr(tok):>20}: {count}")


def section_rank_distributions(rows):
    """Show rank distributions for key source positions."""
    print("\n" + "=" * 80)
    print("9. RANK DISTRIBUTIONS (per source position, best target layer)")
    print("=" * 80)

    by_source = group_by(rows, ["source_layer", "source_pos"])

    for (src_layer, src_pos), source_group in sorted(by_source.items()):
        desc = source_group[0].get("source_desc", "")
        by_tgt = group_by(source_group, ["target_layer"])

        best_tgt = max(by_tgt.keys(), key=lambda t: frac_top_k(by_tgt[t], "A2_rank", 10))
        best_group = by_tgt[best_tgt]

        print(f"\n  L{src_layer}/{src_pos} ({desc}), best target_layer={best_tgt[0]}:")

        for name, field in [("A1", "A1_rank"), ("A2", "A2_rank"), ("Sum", "sum_rank")]:
            ranks = [r[field] for r in best_group if r[field] >= 0]
            if ranks:
                ranks_arr = np.array(ranks)
                print(f"    {name} rank: "
                      f"p10={np.percentile(ranks_arr, 10):.0f}, "
                      f"p25={np.percentile(ranks_arr, 25):.0f}, "
                      f"median={np.median(ranks_arr):.0f}, "
                      f"p75={np.percentile(ranks_arr, 75):.0f}, "
                      f"p90={np.percentile(ranks_arr, 90):.0f}")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Analyze patchscope logit inspection results")
    parser.add_argument("--input", default="results/patchscope_grid.csv")
    parser.add_argument("--summary-output", default="results/patchscope_summary.csv")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Input file not found: {args.input}")
        sys.exit(1)

    rows = load_results(args.input)
    print(f"Loaded {len(rows)} rows from {args.input}")

    valid = [r for r in rows if not str(r.get("top_1_token", "")).startswith("ERROR")]
    print(f"Valid rows: {len(valid)}")

    section_overview(valid)
    results = section_accuracy_table(valid)
    section_best_configs(results)
    section_positive_control(valid)
    section_negative_control(valid)
    section_template_comparison(valid)
    section_target_layer_offset(valid)
    section_source_comparison(valid)
    section_what_is_top1(valid)
    section_rank_distributions(valid)

    if args.summary_output:
        os.makedirs(os.path.dirname(args.summary_output) or ".", exist_ok=True)
        summary_fields = ["source_layer", "source_pos", "template", "target_layer",
                          "n", "A1_top1", "A1_top10", "A2_top1", "A2_top10",
                          "sum_top1", "sum_top10", "A1_med", "A2_med", "sum_med"]
        with open(args.summary_output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=summary_fields)
            writer.writeheader()
            for r in results:
                writer.writerow({k: r[k] for k in summary_fields})
        print(f"\nSummary saved to {args.summary_output}")


if __name__ == "__main__":
    main()