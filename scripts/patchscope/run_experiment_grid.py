"""
run_experiment_grid.py

Runs patchscoping over a grid of:
  - Source (layer, position) pairs
  - Inspection prompt templates
  - Target layer offsets
  - Diverse correct examples

Saves results to CSV. Supports resuming — skips rows already present in output.

Usage:
    python run_experiment_grid.py
    python run_experiment_grid.py --output results/patchscope_grid.csv
    python run_experiment_grid.py --num-examples 10  # smaller first pass
"""

import argparse
import csv
import os
import sys
import time
import pickle
import glob
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from patchscope_single import run_patchscope, get_activation, load_model_and_tokenizer

# ============================================================
# CONFIG
# ============================================================

DATA_DIR = "/workspace/filler-token-reasoning/data/extracted_states_2fact_allpos"
CONDITION = "dots_10"
DEFAULT_OUTPUT = "results/patchscope_grid.csv"
DEFAULT_NUM_EXAMPLES = 50

# Source (layer, position) pairs to investigate
SOURCE_CONFIGS = [
    # (layer, pos_key, description)
    (50, "pos_001", "logit_lens_decodes_A1"),
    (53, "pos_001", "logit_lens_A1_not_A2"),
    (53, "pos_005", "logit_lens_kinda_A2"),
    (58, "pos_015", "late_layer"),
    (20, "pos_001", "early_negative_control"),
]

# Inspection prompt templates
TEMPLATES = [
    # Open-ended (discovery)
    "The number is",
    "The answer is",
    "I am thinking about",
    "The value stored here is",
    # Targeted at A2
    "The second number in the addition is",
    "The atomic number of the second element is",
    # Targeted at A1
    "The first number in the addition is",
    # Targeted at sum
    "The sum is",
]

# Target layer offsets relative to source layer
TARGET_LAYER_OFFSETS = [-20, -10, 0, 5]

MAX_NEW_TOKENS = 20


# ============================================================
# EXAMPLE SELECTION
# ============================================================

def load_correct_examples(data_dir, condition):
    """Load all correct examples."""
    condition_dir = os.path.join(data_dir, condition)
    files = sorted(glob.glob(os.path.join(condition_dir, "*.pkl")))

    correct = []
    for f in files:
        with open(f, "rb") as fh:
            ex = pickle.load(fh)
        if isinstance(ex, list):
            for e in ex:
                if e.get("model_correct", False):
                    correct.append(e)
        elif isinstance(ex, dict):
            if ex.get("model_correct", False):
                correct.append(ex)

    return correct


def select_diverse_examples(examples, n):
    """
    Select n examples that maximize diversity of (A1, A2) values.

    Strategy: tile the (A1, A2) space into a grid, pick one example per cell,
    then fill remaining slots by picking examples furthest from already-selected.
    """
    if n >= len(examples):
        return examples

    # Extract (A1, A2) pairs
    pairs = np.array([(ex["fact_value_1"], ex["fact_value_2"]) for ex in examples])

    # Greedy farthest-point sampling in (A1, A2) space
    selected_indices = []

    # Start with the example closest to the median
    median = np.median(pairs, axis=0)
    dists = np.linalg.norm(pairs - median, axis=1)
    first = np.argmin(dists)
    selected_indices.append(first)

    for _ in range(n - 1):
        # For each candidate, compute min distance to any selected example
        selected_pairs = pairs[selected_indices]
        min_dists = np.full(len(examples), np.inf)
        for si in selected_indices:
            d = np.linalg.norm(pairs - pairs[si], axis=1)
            min_dists = np.minimum(min_dists, d)
        # Don't re-select
        min_dists[selected_indices] = -1
        # Pick the one with largest min distance
        best = np.argmax(min_dists)
        selected_indices.append(best)

    selected = [examples[i] for i in selected_indices]

    # Print summary
    sel_pairs = [(ex["fact_value_1"], ex["fact_value_2"]) for ex in selected]
    a1s = [p[0] for p in sel_pairs]
    a2s = [p[1] for p in sel_pairs]
    print(f"Selected {n} examples:")
    print(f"  A1 range: {min(a1s)}-{max(a1s)}, unique: {len(set(a1s))}")
    print(f"  A2 range: {min(a2s)}-{max(a2s)}, unique: {len(set(a2s))}")

    return selected


# ============================================================
# GRID LOGIC
# ============================================================

def build_grid(examples):
    """Build the full list of experiment configurations."""
    grid = []
    for ex_idx, ex in enumerate(examples):
        for source_layer, source_pos, desc in SOURCE_CONFIGS:
            for template in TEMPLATES:
                for offset in TARGET_LAYER_OFFSETS:
                    target_layer = max(0, min(60, source_layer + offset))
                    grid.append({
                        "example_idx": ex_idx,
                        "problem_idx": ex.get("problem_idx", ex_idx),
                        "A1": ex["fact_value_1"],
                        "A2": ex["fact_value_2"],
                        "sum": ex["answer"],
                        "source_layer": source_layer,
                        "source_pos": source_pos,
                        "source_desc": desc,
                        "template": template,
                        "target_layer": target_layer,
                    })
    return grid


def make_row_key(row):
    """Unique key for a grid row, used for resume detection."""
    return (
        int(row["problem_idx"]),
        int(row["source_layer"]),
        row["source_pos"],
        row["template"],
        int(row["target_layer"]),
    )


def load_completed(output_path):
    """Load already-completed row keys from existing CSV."""
    completed = set()
    if not os.path.exists(output_path):
        return completed

    with open(output_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            completed.add(make_row_key(row))

    return completed


CSV_COLUMNS = [
    "example_idx", "problem_idx", "A1", "A2", "sum",
    "source_layer", "source_pos", "source_desc",
    "template", "target_layer",
    "generated_text",
]


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Run patchscoping experiment grid")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output CSV path")
    parser.add_argument("--num-examples", type=int, default=DEFAULT_NUM_EXAMPLES,
                        help="Number of diverse examples to use")
    args = parser.parse_args()

    output_path = args.output
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # --- Load examples ---
    print("Loading correct examples...")
    all_correct = load_correct_examples(DATA_DIR, CONDITION)
    print(f"Total correct: {len(all_correct)}")

    examples = select_diverse_examples(all_correct, args.num_examples)

    # --- Build grid ---
    grid = build_grid(examples)
    print(f"\nTotal grid size: {len(grid)} generations")
    print(f"  = {len(examples)} examples"
          f" x {len(SOURCE_CONFIGS)} source configs"
          f" x {len(TEMPLATES)} templates"
          f" x {len(TARGET_LAYER_OFFSETS)} target offsets")

    # --- Check for resume ---
    completed = load_completed(output_path)
    remaining = [row for row in grid if make_row_key(row) not in completed]
    print(f"\nAlready completed: {len(completed)}")
    print(f"Remaining: {len(remaining)}")

    if not remaining:
        print("All done! Nothing to run.")
        return

    est_seconds = len(remaining) * 1.5  # ~1.5s per generation estimate
    print(f"Estimated time: {est_seconds/60:.0f} minutes ({est_seconds/3600:.1f} hours)")

    # --- Load model ---
    print("\nLoading model...")
    t0 = time.time()
    model, tokenizer = load_model_and_tokenizer()
    print(f"Model loaded in {time.time()-t0:.0f}s")

    # --- Open CSV for appending ---
    file_exists = os.path.exists(output_path) and os.path.getsize(output_path) > 0
    csvfile = open(output_path, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(csvfile, fieldnames=CSV_COLUMNS, extrasaction="ignore")
    if not file_exists:
        writer.writeheader()
        csvfile.flush()

    # --- Run ---
    n_done = 0
    n_total = len(remaining)
    t_start = time.time()

    for row in remaining:
        ex = examples[row["example_idx"]]
        activation = get_activation(ex, row["source_pos"], row["source_layer"])

        try:
            generated = run_patchscope(
                model=model,
                tokenizer=tokenizer,
                source_activation=activation,
                inspection_prompt=row["template"],
                target_layer=row["target_layer"],
                patch_position=-1,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=0.0,
            )
        except Exception as e:
            generated = f"ERROR: {e}"
            print(f"  ERROR on row {n_done}: {e}")

        row["generated_text"] = generated
        writer.writerow(row)

        n_done += 1
        if n_done % 50 == 0 or n_done == n_total:
            csvfile.flush()
            elapsed = time.time() - t_start
            rate = n_done / elapsed
            eta = (n_total - n_done) / rate if rate > 0 else 0
            print(f"  [{n_done}/{n_total}] "
                  f"{rate:.1f} gen/s, "
                  f"ETA {eta/60:.0f}min | "
                  f"last: A1={row['A1']}, A2={row['A2']}, "
                  f"L{row['source_layer']}/{row['source_pos']} -> L{row['target_layer']} | "
                  f"{repr(generated[:50])}")

    csvfile.close()
    elapsed = time.time() - t_start
    print(f"\nDone. {n_done} generations in {elapsed/60:.1f} minutes.")
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()