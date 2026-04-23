"""
discover_variables.py

Fully unsupervised variable discovery pipeline using only neighbor-layer
consistency and pairwise agreement — no AMI, no HDBSCAN.

Steps:
1. For each position, pick best layer by neighbor-layer consistency.
2. Filter out degenerate settings (low diversity).
3. Select "interesting region" settings: high consistency + high diversity.
4. Cluster them by pairwise agreement (connected components with threshold).
5. Each cluster is a candidate intermediate variable.

Usage:
    python scripts/discover_variables.py --condition dots_10
"""

import argparse
import json
from pathlib import Path

import numpy as np


def connected_components(n, edges):
    """Simple union-find."""
    parent = list(range(n))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    for a, b in edges:
        union(a, b)
    roots = {}
    for i in range(n):
        r = find(i)
        roots.setdefault(r, []).append(i)
    return list(roots.values())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", type=str, default="dots_10")
    parser.add_argument("--npz", type=Path, default=None)
    parser.add_argument("--tol", type=int, default=5)
    parser.add_argument("--min-diversity", type=float, default=0.25,
                        help="Diversity threshold for 'interesting region'")
    parser.add_argument("--min-consistency", type=float, default=0.85,
                        help="Consistency threshold for 'interesting region'")
    parser.add_argument("--agreement-threshold", type=float, default=0.40,
                        help="Pairwise agreement to merge into same variable")
    args = parser.parse_args()

    if args.npz is None:
        args.npz = Path(f"results/cross_example_consistency/cross_example_consistency_{args.condition}.npz")

    data = np.load(args.npz, allow_pickle=True)
    settings = data["settings"]
    predictions = data["predictions"]
    A1 = data["A1"]  # for post-hoc reporting only
    A2 = data["A2"]
    A1A2 = data["A1A2"]

    pos_layer_idx = {}
    for i, s in enumerate(settings):
        pos, layer = str(s[0]), int(s[1])
        pos_layer_idx.setdefault(pos, {})[layer] = i

    # Step 1: per-position best layer by neighbor consistency
    best_per_pos = {}
    for pos in pos_layer_idx:
        layers = sorted(pos_layer_idx[pos].keys())
        scores = {}
        for layer in layers:
            idx = pos_layer_idx[pos][layer]
            preds = predictions[idx]
            agrees = []
            for dl in [-1, 1]:
                if layer + dl in pos_layer_idx[pos]:
                    npreds = predictions[pos_layer_idx[pos][layer + dl]]
                    agrees.append(np.mean(np.abs(preds - npreds) <= args.tol))
            if agrees:
                scores[layer] = np.mean(agrees)
        if not scores:
            continue
        best_layer = max(scores, key=scores.get)
        best_idx = pos_layer_idx[pos][best_layer]
        preds = predictions[best_idx]
        diversity = len(np.unique(preds)) / len(preds)
        best_per_pos[pos] = {
            "layer": best_layer,
            "consistency": scores[best_layer],
            "diversity": diversity,
            "idx": best_idx,
        }

    # Step 2 & 3: filter to interesting region
    candidates = [(pos, info) for pos, info in best_per_pos.items()
                  if info["consistency"] >= args.min_consistency
                  and info["diversity"] >= args.min_diversity]

    print(f"=== Interesting region (consistency ≥ {args.min_consistency:.0%}, diversity ≥ {args.min_diversity:.0%}) ===")
    print(f"{len(candidates)} settings qualify:")
    for pos, info in sorted(candidates, key=lambda x: -x[1]["consistency"]):
        a1 = np.mean(predictions[info["idx"]] == A1)
        a2 = np.mean(predictions[info["idx"]] == A2)
        sum_v = np.mean(predictions[info["idx"]] == A1A2)
        print(f"  {pos} L{info['layer']}: consistency={info['consistency']:.0%}, "
              f"diversity={info['diversity']:.0%}  [A1={a1:.0%}, A2={a2:.0%}, Σ={sum_v:.0%}]")

    if len(candidates) < 2:
        print("Not enough candidates to cluster")
        return

    # Step 4: cluster by pairwise agreement
    n = len(candidates)
    edges = []
    print(f"\n=== Pairwise agreement ≥ {args.agreement_threshold:.0%} ===")
    for i in range(n):
        for j in range(i + 1, n):
            p1 = predictions[candidates[i][1]["idx"]]
            p2 = predictions[candidates[j][1]["idx"]]
            agree = np.mean(p1 == p2)
            if agree >= args.agreement_threshold:
                edges.append((i, j))
                print(f"  {candidates[i][0]} ↔ {candidates[j][0]}: {agree:.0%}")

    # Step 5: connected components
    components = connected_components(n, edges)
    print(f"\n=== Discovered variables ({len(components)}) ===")
    for c_idx, members in enumerate(components):
        # Report the best member (highest consistency) as representative
        rep_i = max(members, key=lambda i: candidates[i][1]["consistency"])
        rep_pos, rep_info = candidates[rep_i]
        preds = predictions[rep_info["idx"]]
        a1 = np.mean(preds == A1)
        a2 = np.mean(preds == A2)
        sum_v = np.mean(preds == A1A2)
        member_names = [candidates[i][0] for i in members]
        print(f"Variable {c_idx + 1}: {len(members)} member(s) = {member_names}")
        print(f"  Representative: {rep_pos} L{rep_info['layer']}, "
              f"consistency={rep_info['consistency']:.0%}")
        print(f"  Post-hoc: A1={a1:.0%}, A2={a2:.0%}, Σ={sum_v:.0%}")


if __name__ == "__main__":
    main()
