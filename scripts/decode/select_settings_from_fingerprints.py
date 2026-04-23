"""
select_settings_from_fingerprints.py

Pairwise-fingerprint-agreement + greedy-dedup selection, analogous to
pool_decode_global_dedup.py but using residual top-K fingerprints (from
extract_residual_fingerprints.py) instead of number-restricted argmax.

Agreement between two (layer, position) settings on example e is the Jaccard
overlap of their top-K fingerprints. Global agreement of a setting is the mean
per-example Jaccard across all other settings. Greedy dedup keeps settings
whose max Jaccard with the already-kept set is below --dedup-threshold.

Diversity of a setting is the fraction of unique token IDs across
(K × n_examples) fingerprint slots — low diversity means every example at
that setting has the same fingerprint (no example-specific signal).

Usage:
    python scripts/decode/select_settings_from_fingerprints.py \
        --fingerprints /tmp/residual_fingerprints_kimi_dots10.npz \
        --output /tmp/kept_settings_kimi_dots10.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))


def compute_diversity(fp_ids_per_setting):
    """Fraction of unique tokens across K × n_examples fingerprint slots."""
    flat = fp_ids_per_setting.flatten()
    return len(np.unique(flat)) / len(flat)


def pairwise_agreement(fp_sets_a, fp_sets_b):
    """Mean Jaccard overlap across examples between two settings' fingerprints."""
    n_ex = len(fp_sets_a)
    total = 0.0
    for e in range(n_ex):
        a, b = fp_sets_a[e], fp_sets_b[e]
        inter = len(a & b)
        union = len(a | b)
        if union:
            total += inter / union
    return total / n_ex


def a_match(fp_ids, fp_vals, target_values, tokenizer):
    """Fraction of examples where any of `target_values`'s digit tokens appear
    in the fingerprint. Post-hoc diagnostic only (uses ground truth)."""
    n_ex, K = fp_ids.shape
    digit_tokens = {}
    for v in set(target_values.tolist()):
        ids = tokenizer.encode(str(int(v)), add_special_tokens=False)
        if len(ids) == 1:
            digit_tokens.setdefault(int(v), []).append(ids[0])
    hits = 0
    for e in range(n_ex):
        v = int(target_values[e])
        if v not in digit_tokens:
            continue
        if any(tid in fp_ids[e] for tid in digit_tokens[v]):
            hits += 1
    return hits / n_ex


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fingerprints", type=Path, required=True)
    ap.add_argument("--min-diversity", type=float, default=0.10)
    ap.add_argument("--dedup-threshold", type=float, default=0.30,
                    help="Drop candidate if max Jaccard with any kept >= this")
    ap.add_argument("--n-kept", type=int, default=10)
    ap.add_argument("--model-path", type=str, default=None,
                    help="For post-hoc A1/A2-digit-match diagnostic")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    data = np.load(args.fingerprints, allow_pickle=True)
    fp_ids = data["fingerprint_ids"]   # (S, N, K)
    settings = [(str(s[0]), int(s[1])) for s in data["settings"]]
    n_settings, n_ex, K = fp_ids.shape
    print(f"{n_settings} settings, {n_ex} examples, K={K}")

    # Step 1: diversity filter
    diversities = np.array([compute_diversity(fp_ids[s]) for s in range(n_settings)])
    keep_mask = diversities >= args.min_diversity
    n_pass = keep_mask.sum()
    print(f"{n_pass}/{n_settings} settings pass diversity >= {args.min_diversity:.0%}")
    if n_pass < 2:
        print("Not enough settings pass diversity; exiting"); return

    cand_idx = np.where(keep_mask)[0]

    # Step 2: convert fingerprints to sets (only for candidates)
    print("Converting fingerprints to sets...")
    fp_sets = {}
    for si in cand_idx:
        fp_sets[si] = [set(int(t) for t in fp_ids[si, e]) for e in range(n_ex)]

    # Step 3: pairwise Jaccard agreement among candidates
    n_c = len(cand_idx)
    agree = np.zeros((n_c, n_c))
    for i_local in tqdm(range(n_c), desc="Pairwise agreement"):
        si = cand_idx[i_local]
        for j_local in range(i_local + 1, n_c):
            sj = cand_idx[j_local]
            v = pairwise_agreement(fp_sets[si], fp_sets[sj])
            agree[i_local, j_local] = v
            agree[j_local, i_local] = v

    # Step 4: rank by mean off-diagonal agreement
    row_sum = agree.sum(axis=1)
    mean_agree = row_sum / (n_c - 1)
    ranked = np.argsort(mean_agree)[::-1]

    # Step 5: greedy dedup
    kept_local = []
    for idx in ranked:
        if any(agree[idx, k] >= args.dedup_threshold for k in kept_local):
            continue
        kept_local.append(idx)
        if len(kept_local) >= args.n_kept:
            break

    # Post-hoc diagnostics (optional)
    a1_match = a2_match = None
    tokenizer = None
    if args.model_path:
        try:
            from extract_hidden_states import load_tokenizer
            tokenizer = load_tokenizer(args.model_path)
        except Exception as e:
            print(f"Warning: could not load tokenizer ({e}); skipping digit-match diag")
            tokenizer = None

    truth_a1 = data.get("truth_fact_value_1")
    truth_a2 = data.get("truth_fact_value_2")

    # Print and save
    rows = []
    print(f"\nKept {len(kept_local)} settings (dedup threshold {args.dedup_threshold:.0%}):")
    header = f"  {'#':>3}  {'Position':>10}  {'Layer':>5}  {'MeanAgree':>9}  {'Diversity':>9}"
    if tokenizer is not None:
        header += f"  {'A1-match':>8}  {'A2-match':>8}"
    print(header)
    for rank, li in enumerate(kept_local):
        si = cand_idx[li]
        pos, layer = settings[si]
        row = {
            "position": pos, "layer": int(layer),
            "mean_agreement": float(mean_agree[li]),
            "diversity": float(diversities[si]),
        }
        line = (f"  {rank+1:>3}  {pos:>10}  {layer:>5}  "
                f"{mean_agree[li]:>8.1%}  {diversities[si]:>8.1%}")
        if tokenizer is not None and truth_a1 is not None:
            am1 = a_match(fp_ids[si], None, truth_a1, tokenizer)
            am2 = a_match(fp_ids[si], None, truth_a2, tokenizer)
            row["a1_digit_match"] = float(am1)
            row["a2_digit_match"] = float(am2)
            line += f"  {am1:>7.1%}  {am2:>7.1%}"
        rows.append(row)
        print(line)

    out = {
        "kept_settings": rows,
        "config": {
            "fingerprints": str(args.fingerprints),
            "min_diversity": args.min_diversity,
            "dedup_threshold": args.dedup_threshold,
            "n_kept": args.n_kept,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
