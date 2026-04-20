"""
pool_decode_a1_a2.py

Pool logit-lens predictions across qualifying (position, layer) settings to recover
A₁ and A₂ simultaneously. Per example, sums softmax-normalized number-token
probabilities across all qualifying settings and takes top-2 distinct values.

Qualifications for inclusion (all unsupervised):
- Within filler region (position ≤ filler_end)
- Layer ≥ --min-layer
- Best-per-position layer by neighbor-layer consistency
- Consistency ≥ --min-consistency
- Diversity (unique predictions / total) ≥ --min-diversity

Reports post-hoc accuracy for A₁, A₂, and both.

Usage:
    python scripts/pool_decode_a1_a2.py --condition dots_10
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from transformers import PreTrainedTokenizerFast


def rms_norm(x, weight, eps=1e-6):
    rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
    return (x / rms) * weight


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", type=str, default="dots_10")
    parser.add_argument("--extraction-dir", type=Path,
                        default=Path("data/extracted_states_2fact_allpos"))
    parser.add_argument("--lm-head", type=Path,
                        default=Path("data/model_weights/deepseek_v3/lm_head_weight.npy"))
    parser.add_argument("--rms-norm-path", type=Path,
                        default=Path("data/model_weights/deepseek_v3/rms_norm_weight.npy"))
    parser.add_argument("--model-path", type=str,
                        default="/workspace/models/deepseek-v3-awq")
    parser.add_argument("--min-layer", type=int, default=40)
    parser.add_argument("--min-consistency", type=float, default=0.85)
    parser.add_argument("--min-diversity", type=float, default=0.25)
    parser.add_argument("--tol", type=int, default=5)
    parser.add_argument("--max-val", type=int, default=300)
    parser.add_argument("--include-post-filler", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    # Load model weights and tokenizer
    lm_head = np.load(args.lm_head).astype(np.float32)
    norm_weight = np.load(args.rms_norm_path).astype(np.float32)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.model_path)
    number_tokens = {}
    for val in range(args.max_val):
        ids = tokenizer.encode(str(val), add_special_tokens=False)
        if len(ids) == 1:
            number_tokens[ids[0]] = val
    num_ids = sorted(number_tokens.keys())
    num_vals = np.array([number_tokens[tid] for tid in num_ids])

    # Load correct examples
    cond_dir = args.extraction_dir / args.condition
    files = sorted(cond_dir.glob("prob_*.pkl"))
    all_data = []
    for f in files:
        with open(f, "rb") as fp:
            d = pickle.load(fp)
        if d.get("model_correct", False):
            all_data.append(d)
    n = len(all_data)
    print(f"{n} correct examples")

    d0 = all_data[0]
    filler_end = d0["boundaries"]["filler_end_offset"]
    A1 = np.array([d["fact_value_1"] for d in all_data])
    A2 = np.array([d["fact_value_2"] for d in all_data])

    positions = sorted([p for p in d0["states"] if p.startswith("pos_")],
                        key=lambda p: int(p.split("_")[1]))

    # Find qualifying (position, layer) settings
    qualifying = []
    for pos in positions:
        pos_num = int(pos.split("_")[1])
        if not args.include_post_filler and pos_num > filler_end:
            continue
        layers = sorted(d0["states"][pos].keys())
        layers = [l for l in layers if l >= args.min_layer]

        # Per-layer predictions for this position
        layer_preds = {}
        for l in layers:
            vecs = np.stack([d["states"][pos][l].astype(np.float32) for d in all_data])
            H = rms_norm(vecs, norm_weight)
            logits = H @ lm_head.T
            preds = num_vals[np.argmax(logits[:, num_ids], axis=1)]
            layer_preds[l] = preds

        # Best layer by neighbor consistency
        scores = {}
        for l in layers:
            agrees = []
            for dl in [-1, 1]:
                if l + dl in layer_preds:
                    agrees.append(np.mean(np.abs(layer_preds[l] - layer_preds[l + dl]) <= args.tol))
            if agrees:
                scores[l] = float(np.mean(agrees))
        if not scores:
            continue
        best_layer = max(scores, key=scores.get)
        consistency = scores[best_layer]
        preds = layer_preds[best_layer]
        diversity = len(np.unique(preds)) / n

        if consistency >= args.min_consistency and diversity >= args.min_diversity:
            qualifying.append({
                "position": pos, "layer": best_layer,
                "consistency": consistency, "diversity": diversity,
            })

    print(f"\n{len(qualifying)} qualifying settings:")
    for q in qualifying:
        print(f"  {q['position']} L{q['layer']}: consistency={q['consistency']:.0%}, "
              f"diversity={q['diversity']:.0%}")

    if not qualifying:
        print("No qualifying settings.")
        return

    # Pool softmax probs across all qualifying settings
    pooled = np.zeros((n, len(num_ids)), dtype=np.float64)
    for q in qualifying:
        vecs = np.stack([d["states"][q["position"]][q["layer"]].astype(np.float32)
                         for d in all_data])
        H = rms_norm(vecs, norm_weight)
        logits = H @ lm_head.T
        num_logits = logits[:, num_ids]
        shifted = num_logits - num_logits.max(axis=1, keepdims=True)
        probs = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
        pooled += probs

    # Top-2 distinct values per example
    top1 = np.zeros(n, dtype=int)
    top2 = np.zeros(n, dtype=int)
    for i in range(n):
        si = np.argsort(pooled[i])[::-1]
        top1[i] = num_vals[si[0]]
        for j in si[1:]:
            if num_vals[j] != top1[i]:
                top2[i] = num_vals[j]
                break

    got_a1 = (top1 == A1) | (top2 == A1)
    got_a2 = (top1 == A2) | (top2 == A2)
    both = got_a1 & got_a2

    print(f"\nPooled top-2 from {len(qualifying)} qualifying settings:")
    print(f"  Got A1: {np.mean(got_a1):.1%}")
    print(f"  Got A2: {np.mean(got_a2):.1%}")
    print(f"  Got BOTH: {np.mean(both):.1%}")

    # Save
    out = args.output or Path(f"results/pool_decode_{args.condition}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    save = {
        "condition": args.condition,
        "n_examples": n,
        "qualifying_settings": qualifying,
        "accuracy": {
            "got_a1": float(np.mean(got_a1)),
            "got_a2": float(np.mean(got_a2)),
            "got_both": float(np.mean(both)),
        },
        "config": {
            "min_consistency": args.min_consistency,
            "min_diversity": args.min_diversity,
            "min_layer": args.min_layer,
            "tol": args.tol,
            "include_post_filler": args.include_post_filler,
        },
    }
    with open(out, "w") as f:
        json.dump(save, f, indent=2)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
