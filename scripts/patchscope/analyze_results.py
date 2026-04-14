"""
analyze_results.py

Reads the scored CSV from parse_outputs.py and produces:
  1. Accuracy table: per (source_layer, source_pos, template, target_layer)
  2. Best configuration ranking (by A2 recovery)
  3. Positive control check: A1 recovery at known-good positions
  4. Negative control check: early layer baseline
  5. Surprises: unrecognized numbers and patterns
  6. Template comparison: which templates work best
  7. Target layer offset analysis: which offsets help

Usage:
    python analyze_results.py
    python analyze_results.py --input results/patchscope_scored.csv
"""

import argparse
import csv
import os
import sys
from collections import defaultdict, Counter
import json


def load_scored(path):
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    # Cast numeric fields
    for r in rows:
        for k in ["A1", "A2", "sum", "source_layer", "target_layer",
                   "contains_A1", "contains_A2", "contains_sum",
                   "element_matches_A1", "element_matches_A2",
                   "contains_diff", "contains_product",
                   "is_error", "is_empty"]:
            if k in r and r[k] != "":
                r[k] = int(r[k])
        if r.get("first_number") not in ("", None):
            try:
                r["first_number"] = int(r["first_number"])
            except ValueError:
                r["first_number"] = None
        else:
            r["first_number"] = None
    return rows


def group_by(rows, keys):
    """Group rows by a tuple of keys. Returns dict of (key_tuple -> [rows])."""
    groups = defaultdict(list)
    for r in rows:
        key = tuple(r[k] for k in keys)
        groups[key] = groups.get(key, [])
        groups[key].append(r)
    return groups


def accuracy(rows, field):
    """Fraction of rows where field == 1."""
    if not rows:
        return 0.0
    return sum(1 for r in rows if r.get(field) == 1) / len(rows)


def first_num_accuracy(rows, target_identity):
    """Fraction of rows where first_num_identity matches target."""
    if not rows:
        return 0.0
    return sum(1 for r in rows if r.get("first_num_identity") == target_identity) / len(rows)


# ============================================================
# ANALYSIS SECTIONS
# ============================================================

def section_accuracy_table(rows):
    """Full accuracy table grouped by configuration."""
    print("\n" + "=" * 80)
    print("1. ACCURACY TABLE")
    print("   Per (source_layer, source_pos, template, target_layer)")
    print("=" * 80)

    keys = ["source_layer", "source_pos", "template", "target_layer"]
    groups = group_by(rows, keys)

    # Collect results for sorting
    results = []
    for key, group in groups.items():
        src_layer, src_pos, template, tgt_layer = key
        n = len(group)
        a1_acc = accuracy(group, "contains_A1")
        a2_acc = accuracy(group, "contains_A2")
        sum_acc = accuracy(group, "contains_sum")
        a1_first = first_num_accuracy(group, "A1")
        a2_first = first_num_accuracy(group, "A2")
        sum_first = first_num_accuracy(group, "SUM")
        results.append({
            "source_layer": src_layer,
            "source_pos": src_pos,
            "template": template,
            "target_layer": tgt_layer,
            "n": n,
            "A1_contains": a1_acc,
            "A2_contains": a2_acc,
            "sum_contains": sum_acc,
            "A1_first": a1_first,
            "A2_first": a2_first,
            "sum_first": sum_first,
        })

    # Print sorted by A2 recovery
    results.sort(key=lambda x: x["A2_contains"], reverse=True)

    print(f"\n{'src_L':>5} {'src_P':>8} {'tgt_L':>5} {'n':>3} "
          f"{'A1%':>5} {'A2%':>5} {'S%':>5} "
          f"{'A1_1st':>6} {'A2_1st':>6} {'S_1st':>5}  template")
    print("-" * 100)

    for r in results[:40]:  # top 40 by A2
        print(f"{r['source_layer']:>5} {r['source_pos']:>8} {r['target_layer']:>5} {r['n']:>3} "
              f"{r['A1_contains']:>5.0%} {r['A2_contains']:>5.0%} {r['sum_contains']:>5.0%} "
              f"{r['A1_first']:>6.0%} {r['A2_first']:>6.0%} {r['sum_first']:>5.0%}  "
              f"{r['template']}")

    if len(results) > 40:
        print(f"  ... ({len(results) - 40} more rows omitted, showing top 40 by A2 recovery)")

    return results


def section_best_configs(results):
    """Rank the best configurations for recovering each quantity."""
    print("\n" + "=" * 80)
    print("2. BEST CONFIGURATIONS")
    print("=" * 80)

    for quantity, field in [("A2", "A2_contains"), ("A1", "A1_contains"), ("SUM", "sum_contains")]:
        sorted_r = sorted(results, key=lambda x: x[field], reverse=True)
        print(f"\n  Top 5 for {quantity} recovery:")
        for i, r in enumerate(sorted_r[:5]):
            print(f"    {i+1}. {r[field]:.0%} — "
                  f"L{r['source_layer']}/{r['source_pos']} -> L{r['target_layer']}, "
                  f"\"{r['template']}\"")


def section_positive_control(rows):
    """Check A1 recovery at positions where logit lens works."""
    print("\n" + "=" * 80)
    print("3. POSITIVE CONTROL — A1 recovery at logit-lens-good positions")
    print("=" * 80)

    # Filter to source positions where logit lens decodes A1
    control_rows = [r for r in rows
                    if r["source_layer"] == 50 and r["source_pos"] == "pos_001"]

    if not control_rows:
        print("  No rows found for (layer=50, pos_001)")
        return

    a1_rate = accuracy(control_rows, "contains_A1")
    a2_rate = accuracy(control_rows, "contains_A2")
    sum_rate = accuracy(control_rows, "contains_sum")
    print(f"\n  At (layer=50, pos_001), n={len(control_rows)}:")
    print(f"    A1 recovery: {a1_rate:.0%}")
    print(f"    A2 recovery: {a2_rate:.0%}")
    print(f"    Sum recovery: {sum_rate:.0%}")

    # Break down by template
    by_template = group_by(control_rows, ["template"])
    print(f"\n  By template:")
    for (template,), group in sorted(by_template.items(), key=lambda x: accuracy(x[1], "contains_A1"), reverse=True):
        a1 = accuracy(group, "contains_A1")
        a2 = accuracy(group, "contains_A2")
        print(f"    A1={a1:.0%}, A2={a2:.0%}  \"{template}\"")


def section_negative_control(rows):
    """Check that early layers produce nothing useful."""
    print("\n" + "=" * 80)
    print("4. NEGATIVE CONTROL — early layer (layer=20)")
    print("=" * 80)

    control_rows = [r for r in rows if r["source_layer"] == 20]

    if not control_rows:
        print("  No rows found for source_layer=20")
        return

    a1_rate = accuracy(control_rows, "contains_A1")
    a2_rate = accuracy(control_rows, "contains_A2")
    sum_rate = accuracy(control_rows, "contains_sum")
    err_rate = accuracy(control_rows, "is_error")
    empty_rate = accuracy(control_rows, "is_empty")
    print(f"\n  At (layer=20, pos_001), n={len(control_rows)}:")
    print(f"    A1 recovery: {a1_rate:.0%}")
    print(f"    A2 recovery: {a2_rate:.0%}")
    print(f"    Sum recovery: {sum_rate:.0%}")
    print(f"    Errors: {err_rate:.0%}, Empty: {empty_rate:.0%}")

    # What does the model generate?
    identity_counts = Counter(r["first_num_identity"] for r in control_rows)
    print(f"\n  First number identity:")
    for identity, count in identity_counts.most_common():
        print(f"    {identity:>10}: {count:>4} ({100*count/len(control_rows):.1f}%)")


def section_template_comparison(rows):
    """Which templates work best overall and per-quantity."""
    print("\n" + "=" * 80)
    print("5. TEMPLATE COMPARISON")
    print("=" * 80)

    # Exclude early-layer negative control
    rows_mid_late = [r for r in rows if r["source_layer"] >= 40]

    by_template = group_by(rows_mid_late, ["template"])
    print(f"\n  Across all mid/late source layers (>=40):")
    print(f"  {'A1%':>5} {'A2%':>5} {'S%':>5} {'A2_1st':>6} {'n':>5}  template")
    print(f"  " + "-" * 70)

    sorted_templates = sorted(by_template.items(),
                               key=lambda x: accuracy(x[1], "contains_A2"), reverse=True)
    for (template,), group in sorted_templates:
        a1 = accuracy(group, "contains_A1")
        a2 = accuracy(group, "contains_A2")
        s = accuracy(group, "contains_sum")
        a2f = first_num_accuracy(group, "A2")
        print(f"  {a1:>5.0%} {a2:>5.0%} {s:>5.0%} {a2f:>6.0%} {len(group):>5}  \"{template}\"")


def section_target_layer_offset(rows):
    """How does target layer offset affect recovery?"""
    print("\n" + "=" * 80)
    print("6. TARGET LAYER OFFSET ANALYSIS")
    print("=" * 80)

    rows_mid_late = [r for r in rows if r["source_layer"] >= 40]

    # Compute offset for each row
    for r in rows_mid_late:
        r["_offset"] = r["target_layer"] - r["source_layer"]

    by_offset = group_by(rows_mid_late, ["_offset"])
    print(f"\n  Across all mid/late source layers (>=40):")
    print(f"  {'offset':>6} {'A1%':>5} {'A2%':>5} {'S%':>5} {'n':>5}")
    print(f"  " + "-" * 35)

    for offset in sorted(by_offset.keys()):
        group = by_offset[offset]
        a1 = accuracy(group, "contains_A1")
        a2 = accuracy(group, "contains_A2")
        s = accuracy(group, "contains_sum")
        print(f"  {offset[0]:>+6} {a1:>5.0%} {a2:>5.0%} {s:>5.0%} {len(group):>5}")


def section_source_position_comparison(rows):
    """Compare source (layer, position) pairs."""
    print("\n" + "=" * 80)
    print("7. SOURCE POSITION COMPARISON")
    print("=" * 80)

    by_source = group_by(rows, ["source_layer", "source_pos"])
    print(f"\n  {'src_L':>5} {'src_P':>8} {'A1%':>5} {'A2%':>5} {'S%':>5} "
          f"{'elem_A1':>7} {'elem_A2':>7} {'n':>5}  desc")
    print("  " + "-" * 75)

    for (src_layer, src_pos), group in sorted(by_source.items()):
        a1 = accuracy(group, "contains_A1")
        a2 = accuracy(group, "contains_A2")
        s = accuracy(group, "contains_sum")
        ea1 = accuracy(group, "element_matches_A1")
        ea2 = accuracy(group, "element_matches_A2")
        desc = group[0].get("source_desc", "")
        print(f"  {src_layer:>5} {src_pos:>8} {a1:>5.0%} {a2:>5.0%} {s:>5.0%} "
              f"{ea1:>7.0%} {ea2:>7.0%} {len(group):>5}  {desc}")


def section_surprises(rows):
    """Find rows with unexpected numbers and look for patterns."""
    print("\n" + "=" * 80)
    print("8. SURPRISES — unknown numbers and unexpected outputs")
    print("=" * 80)

    # Rows with unknown numbers
    rows_with_unknown = [r for r in rows
                         if r.get("unknown_numbers", "") and not r.get("is_error")]

    print(f"\n  Rows with unrecognized numbers: {len(rows_with_unknown)}/{len(rows)}")

    if not rows_with_unknown:
        print("  None found.")
        return

    # Collect all unknown numbers and their frequencies
    all_unknown = []
    for r in rows_with_unknown:
        nums = [int(n) for n in r["unknown_numbers"].split(";") if n]
        all_unknown.extend(nums)

    unknown_counts = Counter(all_unknown)
    print(f"\n  Most common unknown numbers:")
    for num, count in unknown_counts.most_common(20):
        print(f"    {num:>6}: appears {count} times")

    # Show some example rows
    print(f"\n  Sample rows with unknown numbers (up to 15):")
    for r in rows_with_unknown[:15]:
        print(f"    A1={r['A1']:>3}, A2={r['A2']:>3}, sum={r['sum']:>3} | "
              f"unknown: {r['unknown_numbers']:>15} | "
              f"L{r['source_layer']}/{r['source_pos']}->L{r['target_layer']} | "
              f"{repr(r['generated_text'][:60])}")

    # Check if unknown numbers have any relationship to inputs
    print(f"\n  Checking if unknown numbers relate to inputs...")
    relationships = Counter()
    for r in rows_with_unknown:
        a1, a2, s = r["A1"], r["A2"], r["sum"]
        nums = [int(n) for n in r["unknown_numbers"].split(";") if n]
        for n in nums:
            if n == a1 + 1 or n == a1 - 1:
                relationships["A1 ± 1"] += 1
            elif n == a2 + 1 or n == a2 - 1:
                relationships["A2 ± 1"] += 1
            elif n == s + 1 or n == s - 1:
                relationships["sum ± 1"] += 1
            elif a1 != 0 and n % a1 == 0:
                relationships[f"multiple of A1"] += 1
            elif a2 != 0 and n % a2 == 0:
                relationships[f"multiple of A2"] += 1
            elif n == 2 * a1:
                relationships["2 * A1"] += 1
            elif n == 2 * a2:
                relationships["2 * A2"] += 1

    if relationships:
        print(f"  Potential patterns in unknown numbers:")
        for rel, count in relationships.most_common():
            print(f"    {rel}: {count}")


def section_element_names(rows):
    """Check if element names appear in outputs."""
    print("\n" + "=" * 80)
    print("9. ELEMENT NAME ANALYSIS")
    print("=" * 80)

    rows_with_elements = [r for r in rows if r.get("all_elements", "")]
    print(f"\n  Rows containing element names: {len(rows_with_elements)}/{len(rows)}")

    if not rows_with_elements:
        print("  None found.")
        return

    all_elements = []
    for r in rows_with_elements:
        all_elements.extend(r["all_elements"].split(";"))

    elem_counts = Counter(all_elements)
    print(f"\n  Most common elements mentioned:")
    for elem, count in elem_counts.most_common(15):
        print(f"    {elem}: {count}")

    # Check if element names match A1 or A2
    ea1 = sum(1 for r in rows_with_elements if r["element_matches_A1"] == 1)
    ea2 = sum(1 for r in rows_with_elements if r["element_matches_A2"] == 1)
    print(f"\n  Element matches A1: {ea1}")
    print(f"  Element matches A2: {ea2}")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Analyze patchscope scored results")
    parser.add_argument("--input", default="results/patchscope_scored.csv")
    parser.add_argument("--summary-output", default="results/patchscope_summary.csv",
                        help="Save per-config accuracy summary")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Input file not found: {args.input}")
        sys.exit(1)

    rows = load_scored(args.input)
    print(f"Loaded {len(rows)} scored rows from {args.input}")

    # Filter out errors
    valid_rows = [r for r in rows if not r.get("is_error")]
    print(f"Valid (non-error) rows: {len(valid_rows)}")

    # Run all analyses
    results = section_accuracy_table(valid_rows)
    section_best_configs(results)
    section_positive_control(valid_rows)
    section_negative_control(valid_rows)
    section_template_comparison(valid_rows)
    section_target_layer_offset(valid_rows)
    section_source_position_comparison(valid_rows)
    section_surprises(valid_rows)
    section_element_names(valid_rows)

    # Save per-config summary
    if args.summary_output:
        os.makedirs(os.path.dirname(args.summary_output) or ".", exist_ok=True)
        summary_fields = ["source_layer", "source_pos", "template", "target_layer",
                          "n", "A1_contains", "A2_contains", "sum_contains",
                          "A1_first", "A2_first", "sum_first"]
        with open(args.summary_output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=summary_fields)
            writer.writeheader()
            for r in results:
                writer.writerow({k: r[k] for k in summary_fields})
        print(f"\nSummary saved to {args.summary_output}")


if __name__ == "__main__":
    main()