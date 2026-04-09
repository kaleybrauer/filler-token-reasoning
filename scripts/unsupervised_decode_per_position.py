"""
unsupervised_decode_per_position.py

Like unsupervised_decode_filler.py but decodes at each filler position
separately (no averaging). Reports metrics for every (position, layer) pair.

Usage:
    python scripts/unsupervised_decode_per_position.py \
        --condition dots_100 \
        --output-dir results/unsupervised_decode_per_pos
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm


def rms_norm(x, weight, eps=1e-6):
    rms = np.sqrt(np.mean(x ** 2) + eps)
    return (x / rms) * weight


def build_number_token_map(tokenizer, max_val=1000):
    number_tokens = {}
    for val in range(0, max_val):
        ids = tokenizer.encode(str(val), add_special_tokens=False)
        if len(ids) == 1:
            number_tokens[ids[0]] = val
    return number_tokens


def decode_states(states, lm_head, norm_weight, number_tok_ids, number_tok_vals):
    predictions = np.zeros(len(states), dtype=np.float32)
    for i, vec in enumerate(states):
        h = rms_norm(vec, norm_weight)
        logits = h @ lm_head.T
        predictions[i] = number_tok_vals[np.argmax(logits[number_tok_ids])]
    return predictions


def evaluate(predictions, A):
    mae = float(mean_absolute_error(A, predictions))
    median_ae = float(np.median(np.abs(predictions - A)))
    frac_5 = float(np.mean(np.abs(predictions - A) <= 5))
    frac_10 = float(np.mean(np.abs(predictions - A) <= 10))
    r = float(stats.pearsonr(predictions, A)[0]) if np.std(predictions) > 0 else 0.0
    return {"mae": mae, "median_ae": median_ae, "frac_within_5": frac_5,
            "frac_within_10": frac_10, "r": r}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--extraction-dir", type=Path, default=Path("data/extracted_states"))
    parser.add_argument("--condition", type=str, default="dots_100")
    parser.add_argument("--lm-head", type=Path, default=Path("lm_head_weight.npy"))
    parser.add_argument("--rms-norm", type=Path, default=Path("rms_norm_weight.npy"))
    parser.add_argument("--model-path", type=str, default="/workspace/models/deepseek-v3-awq")
    parser.add_argument("--output-dir", type=Path, default=Path("results/unsupervised_decode_per_pos"))
    parser.add_argument("--filter-categories", nargs="+", default=["both_correct", "filler_helped"])
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    from transformers import PreTrainedTokenizerFast
    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.model_path)
    number_tokens = build_number_token_map(tokenizer)
    number_tok_ids = sorted(number_tokens.keys())
    number_tok_vals = np.array([number_tokens[tid] for tid in number_tok_ids], dtype=np.float32)
    print(f"Number tokens: {len(number_tokens)}")

    lm_head = np.load(args.lm_head).astype(np.float32)
    norm_weight = np.load(args.rms_norm).astype(np.float32)
    print(f"lm_head: {lm_head.shape}, norm: {norm_weight.shape}")

    # Build category filter
    filter_idx = None
    if args.filter_categories:
        bl_dir = args.extraction_dir / "baseline"
        cond_dir = args.extraction_dir / args.condition
        bl_files = sorted(bl_dir.glob("prob_*.pkl"))
        cond_files = sorted(cond_dir.glob("prob_*.pkl"))
        filter_idx = set()
        for bf, cf in zip(bl_files, cond_files):
            with open(bf, "rb") as f: bd = pickle.load(f)
            with open(cf, "rb") as f: cd = pickle.load(f)
            idx = bd["problem_idx"]
            bc = bd.get("model_correct", False)
            fc = cd.get("model_correct", False)
            if bc and fc: cat = "both_correct"
            elif not bc and fc: cat = "filler_helped"
            elif bc and not fc: cat = "filler_hurt"
            else: cat = "both_wrong"
            if cat in args.filter_categories:
                filter_idx.add(idx)
        print(f"Filtered to {len(filter_idx)} examples ({args.filter_categories})")

    # Load all data: per-position states
    print(f"Loading {args.condition}...")
    cond_dir = args.extraction_dir / args.condition
    files = sorted(cond_dir.glob("prob_*.pkl"))

    # First pass: identify positions and layers
    with open(files[0], "rb") as f:
        d0 = pickle.load(f)
    all_positions = sorted(d0["states"].keys())
    filler_positions = [p for p in all_positions if p.startswith("filler_k") or p == "pre_filler"]
    # Include answer_prompt for reference
    positions_to_decode = filler_positions + (["answer_prompt"] if "answer_prompt" in all_positions else [])
    layers = sorted(d0["states"][all_positions[0]].keys())

    # Load states: states[position][layer] = list of vectors
    from collections import defaultdict
    states = defaultdict(lambda: defaultdict(list))
    metadata = []

    for f in tqdm(files, desc="Loading"):
        with open(f, "rb") as fp:
            d = pickle.load(fp)
        if filter_idx is not None and d["problem_idx"] not in filter_idx:
            continue
        metadata.append({
            "problem_idx": d["problem_idx"],
            "fact_value": d["fact_value"],
            "x": d["x"],
            "answer": d["answer"],
        })
        for pos in positions_to_decode:
            if pos not in d["states"]:
                continue
            for layer in layers:
                if layer in d["states"][pos]:
                    states[pos][layer].append(d["states"][pos][layer].astype(np.float32))

    # Stack
    for pos in states:
        for layer in states[pos]:
            states[pos][layer] = np.stack(states[pos][layer])

    A = np.array([m["fact_value"] for m in metadata], dtype=np.float32)
    n = len(A)
    print(f"Examples: {n}, positions: {len(positions_to_decode)}, layers: {len(layers)}")

    # Decode every (position, layer)
    all_results = {}

    print(f"\n{'Position':<16} {'Layer':>5} {'MAE':>7} {'MedAE':>7} {'±10':>6} {'±5':>5} {'r':>7}")
    print("-" * 60)

    for pos in positions_to_decode:
        all_results[pos] = {}
        for layer in layers:
            if layer not in states[pos] or len(states[pos][layer]) == 0:
                continue
            X = states[pos][layer]
            preds = decode_states(X, lm_head, norm_weight, number_tok_ids, number_tok_vals)
            result = evaluate(preds, A)
            all_results[pos][layer] = result

        # Print best layer for this position
        if all_results[pos]:
            best_layer = min(all_results[pos], key=lambda l: all_results[pos][l]["mae"])
            r = all_results[pos][best_layer]
            print(f"{pos:<16} L{best_layer:>3}  {r['mae']:>7.1f} {r['median_ae']:>7.1f} "
                  f"{r['frac_within_10']:>5.1%} {r['frac_within_5']:>4.1%} {r['r']:>7.3f}")

    # Print full table for avg_filler-equivalent comparison
    print(f"\n\nFull results at key layers:")
    key_layers = [l for l in [30, 35, 40, 42, 43, 45, 49, 50, 53, 55, 56, 58, 60] if l in layers]

    for layer in key_layers:
        print(f"\n  Layer {layer}:")
        print(f"  {'Position':<16} {'MAE':>7} {'MedAE':>7} {'±10':>6} {'±5':>5} {'r':>7}")
        print(f"  {'-'*52}")
        for pos in positions_to_decode:
            if layer in all_results.get(pos, {}):
                r = all_results[pos][layer]
                print(f"  {pos:<16} {r['mae']:>7.1f} {r['median_ae']:>7.1f} "
                      f"{r['frac_within_10']:>5.1%} {r['frac_within_5']:>4.1%} {r['r']:>7.3f}")

    # Save
    save_results = {}
    for pos in all_results:
        save_results[pos] = {str(l): v for l, v in all_results[pos].items()}
    save_results["_condition"] = args.condition
    save_results["_n_examples"] = n
    save_results["_positions"] = positions_to_decode
    save_results["_layers"] = [int(l) for l in layers]

    with open(args.output_dir / f"per_position_{args.condition}.json", "w") as f:
        json.dump(save_results, f, indent=2)

    print(f"\nResults saved to {args.output_dir}/per_position_{args.condition}.json")


if __name__ == "__main__":
    main()
