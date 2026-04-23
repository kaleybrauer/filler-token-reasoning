"""
pool_decode_global_dedup_1hop.py

1-hop version of pool_decode_global_dedup.py. Same pipeline (global pairwise
exact-match agreement + greedy dedup + pooled softmax top-K), but:
- Reads from data/extracted_states/ (1-hop layout)
- Per-example variable is the single looked-up value `fact_value` (= A)
- In-filler positions are `pre_filler` + `filler_k*`; `question_end` and
  `answer_prompt` are excluded unless --include-post-filler

Usage:
    python scripts/pool_decode_global_dedup_1hop.py --condition dots_10
    python scripts/pool_decode_global_dedup_1hop.py --condition counting_25 --n-kept 15
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from tqdm import tqdm


def rms_norm(x, weight, eps=1e-6):
    rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
    return (x / rms) * weight


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", type=str, default="dots_10")
    parser.add_argument("--extraction-dir", type=Path,
                        default=Path("data/extracted_states"))
    parser.add_argument("--lm-head", type=Path,
                        default=Path("data/model_weights/deepseek_v3/lm_head_weight.npy"))
    parser.add_argument("--rms-norm-path", type=Path,
                        default=Path("data/model_weights/deepseek_v3/rms_norm_weight.npy"))
    parser.add_argument("--model-path", type=str,
                        default="/workspace/models/deepseek-v3-awq")
    parser.add_argument("--min-layer", type=int, default=30)
    parser.add_argument("--min-diversity", type=float, default=0.10)
    parser.add_argument("--dedup-threshold", type=float, default=0.30,
                        help="Drop candidate if its max agreement with kept >= this")
    parser.add_argument("--n-kept", type=int, default=10,
                        help="Maximum number of settings to keep after dedup")
    parser.add_argument("--max-val", type=int, default=300)
    parser.add_argument("--include-post-filler", action="store_true",
                        help="Also include question_end and answer_prompt positions")
    parser.add_argument("--top-k", type=int, nargs="+", default=[1, 2, 3, 5, 10])
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    from extract_hidden_states import load_tokenizer
    tokenizer = load_tokenizer(args.model_path)
    lm_head = np.load(args.lm_head).astype(np.float32)
    norm_weight = np.load(args.rms_norm_path).astype(np.float32)

    number_tokens = {}
    for val in range(args.max_val):
        ids = tokenizer.encode(str(val), add_special_tokens=False)
        if len(ids) == 1:
            number_tokens[ids[0]] = val
    num_ids = sorted(number_tokens.keys())
    num_vals = np.array([number_tokens[tid] for tid in num_ids])

    cond_dir = args.extraction_dir / args.condition
    files = sorted(cond_dir.glob("prob_*.pkl"))
    all_data = []
    for f in tqdm(files, desc="Loading"):
        with open(f, "rb") as fp:
            d = pickle.load(fp)
        if d.get("model_correct", False):
            all_data.append(d)
    n = len(all_data)
    print(f"{n} correct examples")
    if n == 0:
        print("No correct examples — exiting")
        return

    A = np.array([d["fact_value"] for d in all_data])

    # 1-hop uses sparse named positions. In-filler = pre_filler + filler_k*.
    d0 = all_data[0]
    in_filler = {"pre_filler"} | {p for p in d0["states"] if p.startswith("filler_k")}
    all_positions = [p for p in d0["states"]
                     if args.include_post_filler or p in in_filler]
    # Deterministic order: pre_filler first, then filler_k* by k, then others
    def pos_sort_key(p):
        if p == "pre_filler":
            return (0, 0)
        if p.startswith("filler_k"):
            return (1, int(p[len("filler_k"):]))
        if p == "question_end":
            return (2, 0)
        if p == "answer_prompt":
            return (3, 0)
        return (4, 0)
    all_positions = sorted(all_positions, key=pos_sort_key)
    print(f"Positions in use: {all_positions}")

    # Step 1+2: decode everywhere, filter by layer >= min_layer and diversity
    candidates = []
    for pos in tqdm(all_positions, desc="Decoding"):
        layers = sorted(d0["states"][pos].keys())
        layers = [l for l in layers if l >= args.min_layer]

        for layer in layers:
            try:
                vecs = np.stack([d["states"][pos][layer].astype(np.float32)
                                 for d in all_data])
            except KeyError:
                continue
            H = rms_norm(vecs, norm_weight)
            logits = H @ lm_head.T
            preds = num_vals[np.argmax(logits[:, num_ids], axis=1)]
            diversity = len(np.unique(preds)) / n
            if diversity < args.min_diversity:
                continue
            candidates.append({
                "position": pos, "layer": layer,
                "predictions": preds, "diversity": diversity,
            })

    print(f"{len(candidates)} candidates pass filters")
    if not candidates:
        print("No candidates — exiting")
        return

    # Step 3: pairwise agreement (exact match, within each example)
    n_c = len(candidates)
    preds_mat = np.stack([c["predictions"] for c in candidates])
    print("Computing pairwise agreement...")
    agree = np.zeros((n_c, n_c))
    for i in range(n_c):
        agree[i] = np.mean(preds_mat[i:i+1] == preds_mat, axis=1)
    np.fill_diagonal(agree, 0)

    # Step 4: rank by mean global agreement
    mean_agree = agree.mean(axis=1)
    ranked = np.argsort(mean_agree)[::-1]

    # Step 5: greedy dedup
    kept = []
    for idx in ranked:
        if any(agree[idx, k] >= args.dedup_threshold for k in kept):
            continue
        kept.append(idx)
        if len(kept) >= args.n_kept:
            break

    print(f"\nKept {len(kept)} settings (dedup threshold {args.dedup_threshold:.0%}):")
    print(f"  {'#':>3}  {'Position':>14}  {'Layer':>5}  {'MeanAgree':>9}  {'Diversity':>9}  {'A-match':>7}")
    for rank, k in enumerate(kept):
        c = candidates[k]
        a_hit = float(np.mean(c["predictions"] == A))
        print(f"  {rank+1:>3}  {c['position']:>14}  {c['layer']:>5}  "
              f"{mean_agree[k]:>8.1%}  {c['diversity']:>8.1%}  {a_hit:>6.1%}")

    # Step 6: pool softmax probs across kept
    pooled = np.zeros((n, len(num_ids)), dtype=np.float64)
    for k in kept:
        c = candidates[k]
        vecs = np.stack([d["states"][c["position"]][c["layer"]].astype(np.float32)
                         for d in all_data])
        H = rms_norm(vecs, norm_weight)
        logits = H @ lm_head.T
        num_logits = logits[:, num_ids]
        shifted = num_logits - num_logits.max(axis=1, keepdims=True)
        probs = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
        pooled += probs

    # Step 7: top-K recovery for the single variable A
    sort_idx = np.argsort(pooled, axis=1)[:, ::-1]
    top_preds = num_vals[sort_idx]

    print(f"\nPooled top-K recovery of A ({len(kept)} settings):")
    print(f"  {'K':>4}  {'A':>5}")
    results = {}
    for K in args.top_k:
        a_hit = float(np.mean([A[i] in top_preds[i, :K] for i in range(n)]))
        results[f"top_{K}"] = {"a": a_hit}
        print(f"  {K:>4}  {a_hit:>4.0%}")

    out = args.output or Path(f"results/pool_decode_1hop_global_{args.condition}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    save = {
        "condition": args.condition,
        "n_examples": n,
        "kept_settings": [
            {"position": candidates[k]["position"],
             "layer": int(candidates[k]["layer"]),
             "mean_agreement": float(mean_agree[k]),
             "diversity": float(candidates[k]["diversity"]),
             "a_exact_match": float(np.mean(candidates[k]["predictions"] == A))}
            for k in kept
        ],
        "topk_recovery": results,
        "config": {
            "min_layer": args.min_layer,
            "min_diversity": args.min_diversity,
            "dedup_threshold": args.dedup_threshold,
            "n_kept": args.n_kept,
            "include_post_filler": args.include_post_filler,
        },
    }
    with open(out, "w") as f:
        json.dump(save, f, indent=2)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
