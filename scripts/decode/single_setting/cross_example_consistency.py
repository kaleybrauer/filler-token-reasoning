"""
cross_example_consistency.py

Unsupervised discovery of intermediate variables from filler hidden states.

For each (layer, position), decode the top number token via logit lens.
Then compute pairwise Adjusted Mutual Information (AMI) between all
(layer, position) pairs — settings that decode the same underlying variable
(e.g., both decode A1) will partition examples the same way.

Spectral clustering on the AMI matrix reveals clusters corresponding to
different intermediate variables (A1, A2, sum) without any labels.

Usage:
    python scripts/cross_example_consistency.py \
        --condition dots_10 \
        --extraction-dir data/extracted_states_2fact_allpos \
        --output-dir results/cross_example_consistency
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_mutual_info_score
from tqdm import tqdm


def rms_norm(x, weight, eps=1e-6):
    rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
    return (x / rms) * weight


def pos_sort_key(p):
    if p == "question_end": return -2
    if p == "pre_filler": return -1
    if p.startswith("filler_k"): return int(p.split("k")[1])
    if p.startswith("pos_"): return int(p.split("_")[1])
    if p == "answer_prompt": return 99999
    return 9999


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
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/cross_example_consistency"))
    parser.add_argument("--n-clusters", type=int, default=3)
    parser.add_argument("--min-layer", type=int, default=40,
                        help="Skip layers below this (early layers are noise for logit lens)")
    parser.add_argument("--min-diversity", type=float, default=0.05,
                        help="Minimum fraction of unique predictions to keep a setting "
                             "(filters out degenerate constant-output settings)")
    parser.add_argument("--no-filter", action="store_true",
                        help="Use all examples (don't filter to model_correct)")
    parser.add_argument("--max-val", type=int, default=300,
                        help="Max number token value (A1+A2 max is ~233)")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load tokenizer and build number token map
    from extract_hidden_states import load_tokenizer
    tokenizer = load_tokenizer(args.model_path)
    number_tokens = {}
    for val in range(args.max_val):
        ids = tokenizer.encode(str(val), add_special_tokens=False)
        if len(ids) == 1:
            number_tokens[ids[0]] = val
    num_ids = sorted(number_tokens.keys())
    num_vals = np.array([number_tokens[tid] for tid in num_ids])
    print(f"Number tokens: {len(number_tokens)}")

    # Load model weights
    lm_head = np.load(args.lm_head).astype(np.float32)
    norm_weight = np.load(args.rms_norm_path).astype(np.float32)
    print(f"lm_head: {lm_head.shape}")

    # Load examples
    cond_dir = args.extraction_dir / args.condition
    files = sorted(cond_dir.glob("prob_*.pkl"))
    print(f"Loading {len(files)} files...")

    all_data = []
    for f in tqdm(files, desc="Loading"):
        with open(f, "rb") as fp:
            d = pickle.load(fp)
        if args.no_filter or d.get("model_correct") is None or d.get("model_correct", False):
            all_data.append(d)
    print(f"{len(all_data)} examples (no_filter={args.no_filter})")

    if len(all_data) < 10:
        print("Too few correct examples!")
        return

    # Get positions and layers
    d0 = all_data[0]
    positions = sorted(
        [p for p in d0["states"] if p.startswith("pos_") or p.startswith("filler_k")
         or p == "pre_filler" or p in ("question_end", "answer_prompt")],
        key=pos_sort_key
    )
    all_layers = sorted(d0["states"][positions[0]].keys())
    layers = [l for l in all_layers if l >= args.min_layer]
    print(f"Using layers {layers[0]}-{layers[-1]} ({len(layers)} of {len(all_layers)}, min_layer={args.min_layer})")

    # Ground truth for post-hoc evaluation
    A1 = np.array([d["fact_value_1"] for d in all_data])
    A2 = np.array([d["fact_value_2"] for d in all_data])
    A1A2 = np.array([d["answer"] for d in all_data])

    n_examples = len(all_data)

    # Step 1: Decode top number at every (layer, position)
    print(f"\nDecoding {len(positions)} positions × {len(layers)} layers...")
    settings = []  # list of (pos, layer) tuples
    predictions = []  # list of (n_examples,) arrays

    for pos in positions:
        for layer in tqdm(layers, desc=f"  {pos}", leave=False):
            vecs = []
            valid_idx = []
            for i, d in enumerate(all_data):
                if pos in d["states"] and layer in d["states"][pos]:
                    vecs.append(d["states"][pos][layer].astype(np.float32))
                    valid_idx.append(i)

            if len(vecs) < n_examples * 0.9:
                continue

            H = np.stack(vecs)
            H = rms_norm(H, norm_weight)
            logits = H @ lm_head.T
            num_logits = logits[:, num_ids]
            preds = num_vals[np.argmax(num_logits, axis=1)]

            settings.append((pos, layer))
            predictions.append(preds)
        print(f"  {pos} done")

    # Filter out degenerate settings (constant or near-constant predictions)
    if args.min_diversity > 0:
        kept = []
        for i, preds in enumerate(predictions):
            diversity = len(np.unique(preds)) / n_examples
            if diversity >= args.min_diversity:
                kept.append(i)
        n_before = len(settings)
        settings = [settings[i] for i in kept]
        predictions = [predictions[i] for i in kept]
        print(f"Diversity filter (>={args.min_diversity:.0%}): {n_before} → {len(settings)} settings")

    n_settings = len(settings)
    print(f"\n{n_settings} settings for AMI computation")

    # Step 2: Compute pairwise AMI
    print(f"\nComputing {n_settings}×{n_settings} AMI matrix...")
    ami_matrix = np.zeros((n_settings, n_settings))

    for i in tqdm(range(n_settings), desc="AMI"):
        for j in range(i, n_settings):
            ami = adjusted_mutual_info_score(predictions[i], predictions[j])
            ami_matrix[i, j] = ami
            ami_matrix[j, i] = ami

    # Step 3: Spectral clustering
    print(f"\nSpectral clustering (k={args.n_clusters})...")
    # AMI can be negative; shift to make it a valid affinity matrix
    ami_shifted = ami_matrix - ami_matrix.min() + 1e-6
    np.fill_diagonal(ami_shifted, 0)

    clustering = SpectralClustering(
        n_clusters=args.n_clusters,
        affinity="precomputed",
        random_state=42,
    )
    labels = clustering.fit_predict(ami_shifted)

    # Step 4: Analyze clusters
    print(f"\n{'='*60}")
    print("CLUSTER ANALYSIS")
    print(f"{'='*60}")

    for c in range(args.n_clusters):
        members = [(settings[i], predictions[i]) for i in range(n_settings) if labels[i] == c]
        member_settings = [s for s, _ in members]

        # Find representative: highest mean AMI with other cluster members
        cluster_indices = [i for i in range(n_settings) if labels[i] == c]
        if len(cluster_indices) > 1:
            intra_ami = ami_matrix[np.ix_(cluster_indices, cluster_indices)]
            best_local = np.argmax(intra_ami.mean(axis=1))
            rep_idx = cluster_indices[best_local]
        else:
            rep_idx = cluster_indices[0]

        rep_pos, rep_layer = settings[rep_idx]
        rep_preds = predictions[rep_idx]

        # Post-hoc: which variable does this cluster decode?
        a1_exact = np.mean(rep_preds == A1)
        a2_exact = np.mean(rep_preds == A2)
        sum_exact = np.mean(rep_preds == A1A2)

        # AMI with ground truth labels
        ami_a1 = adjusted_mutual_info_score(rep_preds, A1)
        ami_a2 = adjusted_mutual_info_score(rep_preds, A2)
        ami_sum = adjusted_mutual_info_score(rep_preds, A1A2)

        # Identify which variable
        best_var = max([("A1", a1_exact, ami_a1),
                        ("A2", a2_exact, ami_a2),
                        ("sum", sum_exact, ami_sum)],
                       key=lambda x: x[1])

        # Position distribution
        pos_counts = {}
        for (p, l), _ in members:
            pos_counts[p] = pos_counts.get(p, 0) + 1
        top_positions = sorted(pos_counts.items(), key=lambda x: -x[1])[:5]

        print(f"\nCluster {c}: {len(members)} members → likely {best_var[0]}")
        print(f"  Representative: {rep_pos} L{rep_layer}")
        print(f"  A1 exact={a1_exact:.1%}, A2 exact={a2_exact:.1%}, sum exact={sum_exact:.1%}")
        print(f"  AMI(A1)={ami_a1:.3f}, AMI(A2)={ami_a2:.3f}, AMI(sum)={ami_sum:.3f}")
        print(f"  Top positions: {top_positions}")

    # Save results
    setting_labels = [(pos, int(layer)) for pos, layer in settings]
    np.savez(
        args.output_dir / f"cross_example_consistency_{args.condition}.npz",
        ami_matrix=ami_matrix,
        labels=labels,
        settings=np.array(setting_labels, dtype=object),
        predictions=np.array(predictions),
        A1=A1, A2=A2, A1A2=A1A2,
    )
    print(f"\nSaved to {args.output_dir}/")


if __name__ == "__main__":
    main()
