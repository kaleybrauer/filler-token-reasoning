"""
Save per-example category labels (filler_helped, both_correct, etc.)
to a JSON file for external analysis.
"""

import json
import pickle
from pathlib import Path

import numpy as np


def load_metadata(extraction_dir, condition):
    cond_dir = extraction_dir / condition
    files = sorted(cond_dir.glob("prob_*.pkl"))
    meta = []
    for f in files:
        with open(f, "rb") as fp:
            data = pickle.load(fp)
        meta.append({
            "problem_idx": data["problem_idx"],
            "fact_value": int(data["fact_value"]),
            "x": int(data["x"]),
            "answer": int(data["answer"]),
            "model_correct_baseline": None,
            "model_correct_dots250": None,
            "model_answer": data.get("model_answer"),
        })
    return meta


def main():
    extraction_dir = Path("probing/extracted_states")
    output_dir = Path("probing/probe_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    bl_meta = load_metadata(extraction_dir, "baseline")
    dt_meta = load_metadata(extraction_dir, "dots_250")

    n = len(bl_meta)
    examples = []
    for i in range(n):
        bl_correct = bool(bl_meta[i]["model_correct_baseline"]) if bl_meta[i]["model_correct_baseline"] is not None else None
        # Re-read to get actual correctness
        pass

    # Re-load with correctness
    bl_files = sorted((extraction_dir / "baseline").glob("prob_*.pkl"))
    dt_files = sorted((extraction_dir / "dots_250").glob("prob_*.pkl"))

    examples = []
    for i in range(n):
        with open(bl_files[i], "rb") as f:
            bl = pickle.load(f)
        with open(dt_files[i], "rb") as f:
            dt = pickle.load(f)

        bl_correct = bool(bl["model_correct"])
        dt_correct = bool(dt["model_correct"])

        if not bl_correct and dt_correct:
            category = "filler_helped"
        elif bl_correct and not dt_correct:
            category = "filler_hurt"
        elif bl_correct and dt_correct:
            category = "both_correct"
        else:
            category = "both_wrong"

        examples.append({
            "problem_idx": int(bl["problem_idx"]),
            "fact_value": int(bl["fact_value"]),
            "x": int(bl["x"]),
            "answer": int(bl["answer"]),
            "baseline_correct": bl_correct,
            "dots250_correct": dt_correct,
            "baseline_model_answer": bl.get("model_answer"),
            "dots250_model_answer": dt.get("model_answer"),
            "category": category,
        })

    counts = {}
    for ex in examples:
        counts[ex["category"]] = counts.get(ex["category"], 0) + 1

    output = {
        "n_examples": n,
        "category_counts": counts,
        "baseline_accuracy": sum(1 for ex in examples if ex["baseline_correct"]) / n,
        "dots250_accuracy": sum(1 for ex in examples if ex["dots250_correct"]) / n,
        "examples": examples,
    }

    out_path = output_dir / "categories.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Saved {n} examples to {out_path}")
    print(f"Category counts: {counts}")
    print(f"Baseline accuracy: {output['baseline_accuracy']:.1%}")
    print(f"Dots250 accuracy: {output['dots250_accuracy']:.1%}")


if __name__ == "__main__":
    main()
