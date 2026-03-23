"""
Save per-example category labels and correctness for all conditions to a JSON file.

Loads from multiple extraction directories (main + dots_50).
Computes per-condition categories (baseline vs each filler condition).
"""

import json
import pickle
from pathlib import Path


# (condition_name, extraction_dir) pairs
CONDITION_SOURCES = [
    ("baseline",                "probing/extracted_states"),
    ("dots_250",                "probing/extracted_states"),
    ("counting_150",            "probing/extracted_states"),
    ("counting_scrambled_150",  "probing/extracted_states"),
    ("dots_50",                 "probing/extracted_states_dots50"),
]


def load_condition(extraction_dir, condition):
    """Load all pickle files for a condition, return list of dicts."""
    cond_dir = Path(extraction_dir) / condition
    files = sorted(cond_dir.glob("prob_*.pkl"))
    results = []
    for f in files:
        with open(f, "rb") as fp:
            data = pickle.load(fp)
        results.append({
            "problem_idx": int(data["problem_idx"]),
            "fact_value": int(data["fact_value"]),
            "x": int(data["x"]),
            "answer": int(data["answer"]),
            "model_correct": bool(data["model_correct"]),
            "model_answer": data.get("model_answer"),
        })
    return results


def _categorize(bl_correct, filler_correct):
    if not bl_correct and filler_correct:
        return "filler_helped"
    elif bl_correct and not filler_correct:
        return "filler_hurt"
    elif bl_correct and filler_correct:
        return "both_correct"
    else:
        return "both_wrong"


def main():
    output_dir = Path("probing/probe_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all conditions
    cond_data = {}
    conditions = []
    for cond, ext_dir in CONDITION_SOURCES:
        cond_dir = Path(ext_dir) / cond
        if not cond_dir.exists():
            print(f"Skipping {cond} (not found at {cond_dir})")
            continue
        cond_data[cond] = load_condition(ext_dir, cond)
        conditions.append(cond)
        print(f"Loaded {len(cond_data[cond])} examples for {cond}")

    n = len(cond_data["baseline"])

    # Filler conditions (everything except baseline)
    filler_conds = [c for c in conditions if c != "baseline"]

    # Build per-example records
    examples = []
    for i in range(n):
        bl = cond_data["baseline"][i]
        ex = {
            "problem_idx": bl["problem_idx"],
            "fact_value": bl["fact_value"],
            "x": bl["x"],
            "answer": bl["answer"],
        }
        for cond in conditions:
            ex[f"{cond}_correct"] = cond_data[cond][i]["model_correct"]
            ex[f"{cond}_model_answer"] = cond_data[cond][i]["model_answer"]

        # Per-condition categories (baseline vs each filler condition)
        for cond in filler_conds:
            ex[f"category_{cond}"] = _categorize(
                ex["baseline_correct"], ex[f"{cond}_correct"]
            )

        # Legacy field: category based on baseline vs dots_250
        if "dots_250" in cond_data:
            ex["category"] = ex["category_dots_250"]

        examples.append(ex)

    # Counts and accuracies
    accuracies = {}
    for cond in conditions:
        key = f"{cond}_correct"
        accuracies[cond] = sum(1 for ex in examples if ex[key]) / n

    # Category counts per filler condition
    all_category_counts = {}
    for cond in filler_conds:
        key = f"category_{cond}"
        counts = {}
        for ex in examples:
            c = ex[key]
            counts[c] = counts.get(c, 0) + 1
        all_category_counts[cond] = counts

    output = {
        "n_examples": n,
        "conditions": conditions,
        "accuracies": accuracies,
        "category_counts": all_category_counts.get("dots_250", {}),
        "category_counts_by_condition": all_category_counts,
        "examples": examples,
    }

    out_path = output_dir / "categories.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved {n} examples to {out_path}")
    for cond, acc in accuracies.items():
        print(f"  {cond}: {acc:.1%}")
    for cond in filler_conds:
        print(f"\nCategories (baseline vs {cond}): {all_category_counts[cond]}")


if __name__ == "__main__":
    main()
