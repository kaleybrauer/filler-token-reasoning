"""
Save per-example category labels and correctness for all conditions
(baseline, dots_250, counting_150, counting_scrambled_150) to a JSON file.
"""

import json
import pickle
from pathlib import Path


CONDITIONS = ["baseline", "dots_250", "counting_150", "counting_scrambled_150"]


def load_condition(extraction_dir, condition):
    """Load all pickle files for a condition, return list of dicts."""
    cond_dir = extraction_dir / condition
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


def main():
    extraction_dir = Path("probing/extracted_states")
    output_dir = Path("probing/probe_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all conditions
    cond_data = {}
    for cond in CONDITIONS:
        cond_data[cond] = load_condition(extraction_dir, cond)
        print(f"Loaded {len(cond_data[cond])} examples for {cond}")

    n = len(cond_data["baseline"])

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
        for cond in CONDITIONS:
            ex[f"{cond}_correct"] = cond_data[cond][i]["model_correct"]
            ex[f"{cond}_model_answer"] = cond_data[cond][i]["model_answer"]

        # Category based on baseline vs dots_250 (original definition)
        bl_correct = ex["baseline_correct"]
        dt_correct = ex["dots_250_correct"]
        if not bl_correct and dt_correct:
            category = "filler_helped"
        elif bl_correct and not dt_correct:
            category = "filler_hurt"
        elif bl_correct and dt_correct:
            category = "both_correct"
        else:
            category = "both_wrong"
        ex["category"] = category

        examples.append(ex)

    # Counts and accuracies
    category_counts = {}
    for ex in examples:
        category_counts[ex["category"]] = category_counts.get(ex["category"], 0) + 1

    accuracies = {}
    for cond in CONDITIONS:
        key = f"{cond}_correct"
        accuracies[cond] = sum(1 for ex in examples if ex[key]) / n

    output = {
        "n_examples": n,
        "conditions": CONDITIONS,
        "accuracies": accuracies,
        "category_counts": category_counts,
        "examples": examples,
    }

    out_path = output_dir / "categories.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved {n} examples to {out_path}")
    print(f"Category counts (baseline vs dots): {category_counts}")
    for cond, acc in accuracies.items():
        print(f"  {cond}: {acc:.1%}")


if __name__ == "__main__":
    main()
