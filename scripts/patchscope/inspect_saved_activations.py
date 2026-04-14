"""
inspect_saved_activations.py

Loads saved activation pickle files and prints:
- Number of examples found, how many correct
- Metadata fields and sample values
- Activation shapes, dtypes, value ranges
- Position/layer indexing scheme
- Ground truth (A1, A2, sum) distribution summary
"""

import pickle
import glob
import os
import sys
import numpy as np
from collections import Counter, defaultdict

# === CONFIG ===
DATA_DIR = "/workspace/filler-token-reasoning/data/extracted_states_2fact_allpos"
CONDITION = "dots_10"  # change as needed

def load_pickles(condition_dir):
    """Load all pickle files from a condition directory."""
    pattern = os.path.join(condition_dir, "*.pkl")
    files = sorted(glob.glob(pattern))
    if not files:
        # try .pickle extension
        pattern = os.path.join(condition_dir, "*.pickle")
        files = sorted(glob.glob(pattern))
    if not files:
        # maybe the pickles are directly in the condition dir without extension pattern
        # list everything
        all_files = os.listdir(condition_dir)
        files = [os.path.join(condition_dir, f) for f in sorted(all_files) if not os.path.isdir(os.path.join(condition_dir, f))]
    
    print(f"Found {len(files)} files in {condition_dir}")
    if len(files) == 0:
        print("ERROR: No files found. Check DATA_DIR and CONDITION.")
        sys.exit(1)
    
    examples = []
    for f in files:
        with open(f, "rb") as fh:
            data = pickle.load(fh)
        # handle case where pickle contains a list of examples or a single example
        if isinstance(data, list):
            examples.extend(data)
        elif isinstance(data, dict):
            examples.append(data)
        else:
            print(f"  WARNING: unexpected type {type(data)} in {f}")
    
    return examples, files


def inspect_single(ex, idx=0, verbose=True):
    """Inspect a single example dict."""
    if verbose:
        print(f"\n{'='*60}")
        print(f"Example {idx}")
        print(f"{'='*60}")
    
    # --- Metadata ---
    meta_keys = [
        "problem_idx", "condition", "k",
        "answer", "fact_value_1", "fact_value_2",
        "fact_value", "x",
    ]
    if verbose:
        print("\n--- Metadata ---")
        for k in meta_keys:
            if k in ex:
                print(f"  {k}: {ex[k]}")
            else:
                print(f"  {k}: MISSING")
    
    # --- Model output ---
    output_keys = ["model_response", "model_answer", "model_correct"]
    if verbose:
        print("\n--- Model Output ---")
        for k in output_keys:
            if k in ex:
                val = ex[k]
                if k == "model_response":
                    val = repr(val[:100]) + "..." if len(str(val)) > 100 else repr(val)
                print(f"  {k}: {val}")
    
    # --- Positions ---
    if verbose:
        print("\n--- Positions ---")
    if "positions" in ex:
        positions = ex["positions"]
        pos_keys = sorted(positions.keys())
        if verbose:
            print(f"  Position keys: {pos_keys[0]} ... {pos_keys[-1]} ({len(pos_keys)} total)")
            print(f"  Sample mapping: {pos_keys[0]} -> token idx {positions[pos_keys[0]]}, "
                  f"{pos_keys[-1]} -> token idx {positions[pos_keys[-1]]}")
    else:
        if verbose:
            print("  MISSING 'positions' key")
    
    # --- Boundaries ---
    if verbose:
        print("\n--- Boundaries ---")
    for k in ["filler_start", "filler_end", "boundaries", "seq_len", "gen_prefix"]:
        if k in ex and verbose:
            val = ex[k]
            if k == "gen_prefix":
                val = repr(val[:80]) + "..." if len(str(val)) > 80 else repr(val)
            print(f"  {k}: {val}")
    
    # --- States ---
    if verbose:
        print("\n--- States ---")
    if "states" in ex:
        states = ex["states"]
        pos_keys_s = sorted(states.keys())
        if verbose:
            print(f"  State position keys: {pos_keys_s[0]} ... {pos_keys_s[-1]} ({len(pos_keys_s)} total)")
        
        # inspect first position
        first_pos = pos_keys_s[0]
        layers = states[first_pos]
        
        if isinstance(layers, dict):
            layer_keys = sorted(layers.keys())
            sample_vec = layers[layer_keys[0]]
            if verbose:
                print(f"  Layer keys for {first_pos}: {layer_keys[0]} ... {layer_keys[-1]} ({len(layer_keys)} total)")
                print(f"  Layer key type: {type(layer_keys[0])}")
        elif isinstance(layers, (list, np.ndarray)):
            layer_keys = list(range(len(layers)))
            sample_vec = layers[0] if len(layers) > 0 else None
            if verbose:
                print(f"  Layers stored as {type(layers).__name__}, length {len(layers)}")
        else:
            if verbose:
                print(f"  Unexpected layer container type: {type(layers)}")
            return
        
        if sample_vec is not None:
            if isinstance(sample_vec, np.ndarray):
                if verbose:
                    print(f"  Activation shape: {sample_vec.shape}")
                    print(f"  Activation dtype: {sample_vec.dtype}")
                    print(f"  Value range: [{sample_vec.min():.4f}, {sample_vec.max():.4f}]")
                    print(f"  Mean: {sample_vec.mean():.4f}, Std: {sample_vec.std():.4f}")
                    # check for NaNs/Infs
                    n_nan = np.isnan(sample_vec).sum()
                    n_inf = np.isinf(sample_vec).sum()
                    if n_nan > 0 or n_inf > 0:
                        print(f"  WARNING: {n_nan} NaNs, {n_inf} Infs")
            else:
                if verbose:
                    print(f"  Activation type: {type(sample_vec)}")
    else:
        if verbose:
            print("  MISSING 'states' key")
    
    # --- All keys ---
    if verbose:
        known = set(meta_keys + output_keys + ["positions", "states", "boundaries",
                                                 "filler_start", "filler_end", "seq_len", "gen_prefix"])
        extra = set(ex.keys()) - known
        if extra:
            print(f"\n--- Extra keys not inspected ---")
            for k in sorted(extra):
                val = ex[k]
                print(f"  {k}: {type(val).__name__} = {repr(val)[:80]}")


def summarize_dataset(examples):
    """Print dataset-level summary."""
    print(f"\n{'='*60}")
    print(f"DATASET SUMMARY")
    print(f"{'='*60}")
    
    n = len(examples)
    n_correct = sum(1 for ex in examples if ex.get("model_correct", False))
    print(f"\nTotal examples: {n}")
    print(f"Correct: {n_correct} ({100*n_correct/n:.1f}%)")
    print(f"Incorrect: {n - n_correct} ({100*(n-n_correct)/n:.1f}%)")
    
    # Ground truth distributions (correct only)
    correct = [ex for ex in examples if ex.get("model_correct", False)]
    if not correct:
        print("\nNo correct examples found!")
        return
    
    a1_vals = [ex.get("fact_value_1") or ex.get("fact_value") for ex in correct]
    a2_vals = [ex.get("fact_value_2") or ex.get("x") for ex in correct]
    sums = [ex.get("answer") for ex in correct]
    
    a1_counter = Counter(a1_vals)
    a2_counter = Counter(a2_vals)
    sum_counter = Counter(sums)
    
    print(f"\n--- A1 (fact_value_1) distribution (correct only) ---")
    print(f"  Unique values: {len(a1_counter)}")
    print(f"  Range: {min(a1_counter.keys())} to {max(a1_counter.keys())}")
    print(f"  Most common: {a1_counter.most_common(5)}")
    print(f"  Least common: {a1_counter.most_common()[-5:]}")
    print(f"  Examples per value (min/mean/max): "
          f"{min(a1_counter.values())}/{np.mean(list(a1_counter.values())):.1f}/{max(a1_counter.values())}")
    
    print(f"\n--- A2 (fact_value_2 / x) distribution (correct only) ---")
    print(f"  Unique values: {len(a2_counter)}")
    print(f"  Range: {min(a2_counter.keys())} to {max(a2_counter.keys())}")
    print(f"  Most common: {a2_counter.most_common(5)}")
    print(f"  Least common: {a2_counter.most_common()[-5:]}")
    print(f"  Examples per value (min/mean/max): "
          f"{min(a2_counter.values())}/{np.mean(list(a2_counter.values())):.1f}/{max(a2_counter.values())}")
    
    print(f"\n--- Sum (answer) distribution (correct only) ---")
    print(f"  Unique values: {len(sum_counter)}")
    print(f"  Range: {min(sum_counter.keys())} to {max(sum_counter.keys())}")
    
    # Cross-tabulation: how many (A1, A2) pairs are unique?
    pairs = [(a1, a2) for a1, a2 in zip(a1_vals, a2_vals)]
    print(f"\n--- Pair coverage ---")
    print(f"  Unique (A1, A2) pairs: {len(set(pairs))}")
    print(f"  Total correct examples: {len(pairs)}")
    
    # Verify consistency: fact_value == fact_value_1, x == fact_value_2
    n_fv_match = sum(1 for ex in correct
                     if ex.get("fact_value") is not None and ex.get("fact_value_1") is not None
                     and ex["fact_value"] == ex["fact_value_1"])
    n_x_match = sum(1 for ex in correct
                    if ex.get("x") is not None and ex.get("fact_value_2") is not None
                    and ex["x"] == ex["fact_value_2"])
    print(f"\n--- Consistency checks ---")
    print(f"  fact_value == fact_value_1: {n_fv_match}/{len(correct)}")
    print(f"  x == fact_value_2: {n_x_match}/{len(correct)}")
    
    sum_check = sum(1 for ex in correct
                    if ex.get("answer") is not None
                    and ex.get("fact_value_1") is not None
                    and ex.get("fact_value_2") is not None
                    and ex["answer"] == ex["fact_value_1"] + ex["fact_value_2"])
    print(f"  answer == A1 + A2: {sum_check}/{len(correct)}")


def verify_indexing(examples):
    """Verify that states can be indexed as states[pos][layer] -> (7168,) vector."""
    print(f"\n{'='*60}")
    print(f"INDEXING VERIFICATION")
    print(f"{'='*60}")
    
    ex = examples[0]
    states = ex["states"]
    pos_keys = sorted(states.keys())
    
    # try all positions and layers, check shape
    n_ok = 0
    n_bad = 0
    for pos in pos_keys:
        layers = states[pos]
        if isinstance(layers, dict):
            layer_iter = sorted(layers.keys())
        else:
            layer_iter = range(len(layers))
        
        for layer in layer_iter:
            vec = layers[layer]
            if isinstance(vec, np.ndarray) and vec.shape == (7168,):
                n_ok += 1
            else:
                n_bad += 1
                if n_bad <= 3:
                    print(f"  BAD: pos={pos}, layer={layer}, "
                          f"type={type(vec).__name__}, shape={getattr(vec, 'shape', 'N/A')}")
    
    print(f"\n  Total (pos, layer) entries checked: {n_ok + n_bad}")
    print(f"  OK (shape (7168,)): {n_ok}")
    print(f"  Bad: {n_bad}")
    print(f"  Positions: {len(pos_keys)} ({pos_keys[0]} to {pos_keys[-1]})")
    if isinstance(states[pos_keys[0]], dict):
        print(f"  Layers: {len(states[pos_keys[0]])} ({sorted(states[pos_keys[0]].keys())[0]} to {sorted(states[pos_keys[0]].keys())[-1]})")
    else:
        print(f"  Layers: {len(states[pos_keys[0]])}")
    
    # demonstrate access pattern
    print(f"\n  Access pattern: states['pos_001'][50].shape = ", end="")
    try:
        if isinstance(states["pos_001"], dict):
            print(states["pos_001"][50].shape)
        else:
            print(states["pos_001"][50].shape)
    except (KeyError, IndexError) as e:
        print(f"FAILED ({e})")
        # try alternate key types
        sample_layers = states[pos_keys[1]]
        if isinstance(sample_layers, dict):
            lk = sorted(sample_layers.keys())
            print(f"  Layer key type is {type(lk[0])}, trying states['{pos_keys[1]}'][{lk[0]}]...", end="")
            print(sample_layers[lk[0]].shape)


def main():
    condition_dir = os.path.join(DATA_DIR, CONDITION)
    if not os.path.isdir(condition_dir):
        # maybe the pickles are directly in DATA_DIR
        print(f"Directory {condition_dir} not found. Listing DATA_DIR contents:")
        if os.path.isdir(DATA_DIR):
            for item in sorted(os.listdir(DATA_DIR)):
                full = os.path.join(DATA_DIR, item)
                if os.path.isdir(full):
                    n_files = len(os.listdir(full))
                    print(f"  {item}/ ({n_files} files)")
                else:
                    print(f"  {item}")
        else:
            print(f"DATA_DIR {DATA_DIR} does not exist!")
        sys.exit(1)
    
    examples, files = load_pickles(condition_dir)
    
    # Print file info
    print(f"\nSample files:")
    for f in files[:3]:
        size_mb = os.path.getsize(f) / 1e6
        print(f"  {os.path.basename(f)} ({size_mb:.1f} MB)")
    if len(files) > 3:
        print(f"  ... and {len(files) - 3} more")
    
    # Inspect first example in detail
    inspect_single(examples[0], idx=0, verbose=True)
    
    # Inspect a second example briefly to check consistency
    if len(examples) > 1:
        print(f"\n--- Quick check: example 1 has same structure ---")
        ex1_keys = set(examples[1].keys())
        ex0_keys = set(examples[0].keys())
        if ex1_keys == ex0_keys:
            print("  Yes, same keys.")
        else:
            print(f"  DIFFERENT. Extra in ex1: {ex1_keys - ex0_keys}. Missing: {ex0_keys - ex1_keys}")
    
    # Dataset summary
    summarize_dataset(examples)
    
    # Indexing verification
    verify_indexing(examples)
    
    # Print a few example rows for the ground truth table
    correct = [ex for ex in examples if ex.get("model_correct", False)]
    print(f"\n{'='*60}")
    print(f"SAMPLE GROUND TRUTH TABLE (first 10 correct)")
    print(f"{'='*60}")
    print(f"  {'idx':>4}  {'A1':>4}  {'A2':>4}  {'sum':>4}  {'model_answer':>12}")
    for i, ex in enumerate(correct[:10]):
        a1 = ex.get("fact_value_1") or ex.get("fact_value")
        a2 = ex.get("fact_value_2") or ex.get("x")
        s = ex.get("answer")
        ma = ex.get("model_answer")
        print(f"  {i:>4}  {a1:>4}  {a2:>4}  {s:>4}  {str(ma):>12}")


if __name__ == "__main__":
    main()