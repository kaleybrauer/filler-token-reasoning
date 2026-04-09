"""
layer_attribution.py

Per-layer logit lens decomposition: for each layer, compute its contribution
to the answer token logit at pre_answer (and other positions).

Layer L's contribution = (h[L] - h[L-1]) projected through RMSNorm + lm_head.
This shows which layers promote the correct answer token, and how this differs
between baseline and dots_50.

CPU-only — uses pre-extracted hidden states and saved lm_head/rms_norm weights.

Usage:
    uv run --project /workspace/filler-token-reasoning/probing \
        python probing/scripts/layer_attribution.py \
        --extraction-dir probing/extracted_states \
        --output-dir probing/layer_attribution_results
"""

import argparse
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

from extract_hidden_states import load_tokenizer


def get_answer_token_id(tokenizer, answer: int):
    ids = tokenizer.encode(str(answer), add_special_tokens=False)
    return ids[0] if len(ids) == 1 else None


def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Apply RMSNorm: x / RMS(x) * weight."""
    rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
    return (x / rms) * weight


def compute_layer_contributions(
    states: dict,
    layers: list,
    lm_head: np.ndarray,
    norm_weight: np.ndarray = None,
) -> dict:
    """
    Compute per-layer contribution to logits.

    For layer L: delta_h = h[L] - h[L-1] (h[-1] = 0 for L=0)
    Then project delta_h through optional RMSNorm + lm_head.

    Note: RMSNorm is nonlinear so norm(sum) != sum(norm). Applying RMSNorm
    to delta_h is an approximation. We also compute contributions WITHOUT
    norm for comparison, and the cumulative logits WITH norm for accuracy.

    Returns:
        {layer: logits_vector} for raw (unnormed) contributions
    """
    # Stack all layer states and compute deltas in one shot
    H = np.stack([states[l].astype(np.float32) for l in layers], axis=0)  # (n_layers, D)
    deltas = np.diff(H, axis=0, prepend=np.zeros((1, H.shape[1]), dtype=np.float32))
    # Batched matmul: (n_layers, D) @ (D, V) -> (n_layers, V)
    all_logits = deltas @ lm_head.T

    contributions = {layer: all_logits[i] for i, layer in enumerate(layers)}
    return contributions


def compute_cumulative_logits(
    states: dict,
    layers: list,
    lm_head: np.ndarray,
    norm_weight: np.ndarray = None,
) -> dict:
    """
    Compute cumulative logits at each layer: project h[L] through norm + lm_head.
    This is the standard logit lens — what the model would predict if it stopped at layer L.
    """
    H = np.stack([states[l].astype(np.float32) for l in layers], axis=0)
    if norm_weight is not None:
        H = rms_norm(H, norm_weight)
    all_logits = H @ lm_head.T
    cumulative = {layer: all_logits[i] for i, layer in enumerate(layers)}
    return cumulative


def run_attribution(
    extraction_dir: Path,
    conditions: list,
    positions: list,
    lm_head: np.ndarray,
    norm_weight: np.ndarray,
    tokenizer,
    categories_file: Path = None,
    max_examples: int = None,
):
    """Run per-layer attribution for each condition and position."""

    # Load categories
    categories = {}
    if categories_file and categories_file.exists():
        with open(categories_file) as f:
            cat_data = json.load(f)
        for ex in cat_data["examples"]:
            cat_field = "category_dots_50"
            if cat_field not in ex:
                cat_field = "category"
            categories[ex["problem_idx"]] = ex.get(cat_field, "unknown")

    results = {}

    for condition in conditions:
        cond_dir = extraction_dir / condition
        files = sorted(cond_dir.glob("prob_*.pkl"))
        if max_examples:
            files = files[:max_examples]

        print(f"\n{condition}: {len(files)} examples")
        cond_results = []

        for fpath in tqdm(files, desc=condition):
            with open(fpath, "rb") as f:
                data = pickle.load(f)

            answer = data["answer"]
            answer_token_id = get_answer_token_id(tokenizer, answer)
            if answer_token_id is None:
                continue

            # Also get A token id
            a_value = data["fact_value"]
            a_token_id = get_answer_token_id(tokenizer, a_value)

            prob_idx = data["problem_idx"]
            layers = sorted(data["states"][list(data["states"].keys())[0]].keys())

            example_result = {
                "problem_idx": prob_idx,
                "answer": answer,
                "fact_value": a_value,
                "x": data["x"],
                "answer_token_id": int(answer_token_id),
                "a_token_id": int(a_token_id) if a_token_id else None,
                "model_correct": data.get("model_correct"),
                "category": categories.get(prob_idx, "unknown"),
                "positions": {},
            }

            for pos in positions:
                if pos not in data["states"]:
                    continue

                states = data["states"][pos]

                # Per-layer contributions (raw, unnormed)
                contribs = compute_layer_contributions(states, layers, lm_head)

                # Cumulative logits (with norm — standard logit lens)
                cumulative = compute_cumulative_logits(
                    states, layers, lm_head, norm_weight
                )

                pos_result = {}
                for layer in layers:
                    # Contribution of this layer to answer token
                    answer_contrib = float(contribs[layer][answer_token_id])
                    a_contrib = float(contribs[layer][a_token_id]) if a_token_id else None

                    # Cumulative logit/rank at this layer
                    cum_logits = cumulative[layer]
                    answer_cum_logit = float(cum_logits[answer_token_id])
                    answer_cum_rank = int((cum_logits > answer_cum_logit).sum()) + 1

                    a_cum_logit = float(cum_logits[a_token_id]) if a_token_id else None
                    a_cum_rank = int((cum_logits > a_cum_logit).sum()) + 1 if a_token_id else None

                    # Top contributing token for this layer
                    top_token_id = int(np.argmax(contribs[layer]))

                    pos_result[layer] = {
                        "answer_contrib": answer_contrib,
                        "a_contrib": a_contrib,
                        "answer_cum_logit": answer_cum_logit,
                        "answer_cum_rank": answer_cum_rank,
                        "a_cum_logit": a_cum_logit,
                        "a_cum_rank": a_cum_rank,
                        "top_contrib_token": tokenizer.decode([top_token_id]),
                        "top_contrib_logit": float(contribs[layer][top_token_id]),
                    }

                example_result["positions"][pos] = pos_result
            cond_results.append(example_result)

        results[condition] = cond_results

    return results


def aggregate_and_print(results: dict, positions: list):
    """Print summary tables comparing conditions."""

    for pos in positions:
        print(f"\n{'='*90}")
        print(f"Position: {pos}")
        print(f"{'='*90}")

        for condition, examples in results.items():
            # Filter examples that have this position
            valid = [e for e in examples if pos in e["positions"]]
            if not valid:
                continue

            layers = sorted(valid[0]["positions"][pos].keys())

            print(f"\n--- {condition} ({len(valid)} examples) ---")

            # Find layers with strongest A+Y contribution difference
            print(f"\n{'Layer':>6} {'A+Y contrib':>12} {'A contrib':>10} "
                  f"{'A+Y cumRank':>12} {'A cumRank':>10} {'TopToken':>12}")
            print("-" * 70)

            for layer in layers:
                ay_contribs = [e["positions"][pos][layer]["answer_contrib"]
                               for e in valid]
                a_contribs = [e["positions"][pos][layer]["a_contrib"]
                              for e in valid if e["positions"][pos][layer]["a_contrib"] is not None]
                ay_ranks = [e["positions"][pos][layer]["answer_cum_rank"]
                            for e in valid]
                a_ranks = [e["positions"][pos][layer]["a_cum_rank"]
                           for e in valid if e["positions"][pos][layer]["a_cum_rank"] is not None]

                # Most common top contributing token
                top_tokens = [e["positions"][pos][layer]["top_contrib_token"]
                              for e in valid]
                most_common_top = Counter(top_tokens).most_common(1)[0][0]

                med_ay = np.median(ay_contribs)
                med_a = np.median(a_contribs) if a_contribs else float("nan")
                med_ay_rank = np.median(ay_ranks)
                med_a_rank = np.median(a_ranks) if a_ranks else float("nan")

                # Only print layers with notable contributions or ranks
                if abs(med_ay) > 0.5 or med_ay_rank < 100 or layer % 10 == 0 or layer >= 55:
                    print(f"L{layer:>4} {med_ay:>12.2f} {med_a:>10.2f} "
                          f"{med_ay_rank:>12.0f} {med_a_rank:>10.0f} {most_common_top:>12}")

        # Cross-condition comparison at key layers
        cond_names = list(results.keys())
        if len(cond_names) >= 2:
            print(f"\n--- Comparison: {' vs '.join(cond_names)} ---")
            print(f"{'Layer':>6}", end="")
            for c in cond_names:
                print(f"  {c[:8]+' A+Y':>14} {c[:8]+' rank':>12}", end="")
            print()
            print("-" * (6 + len(cond_names) * 28))

            # Use layers from first condition
            all_valid = {c: [e for e in results[c] if pos in e["positions"]]
                         for c in cond_names}
            if all(all_valid.values()):
                layers = sorted(all_valid[cond_names[0]][0]["positions"][pos].keys())
                for layer in layers:
                    if layer % 5 != 0 and layer < 55:
                        continue
                    print(f"L{layer:>4}", end="")
                    for c in cond_names:
                        valid = all_valid[c]
                        contribs = [e["positions"][pos][layer]["answer_contrib"]
                                    for e in valid]
                        ranks = [e["positions"][pos][layer]["answer_cum_rank"]
                                 for e in valid]
                        print(f"  {np.median(contribs):>14.2f} {np.median(ranks):>12.0f}", end="")
                    print()

    # By-category breakdown at pre_answer
    if "pre_answer" in positions:
        print(f"\n{'='*90}")
        print("By category at pre_answer (cumulative rank of A+Y at L60)")
        print(f"{'='*90}")
        for condition, examples in results.items():
            valid = [e for e in examples if "pre_answer" in e["positions"]]
            by_cat = defaultdict(list)
            for e in valid:
                by_cat[e["category"]].append(e)
            print(f"\n{condition}:")
            for cat in sorted(by_cat.keys()):
                exs = by_cat[cat]
                ranks = [e["positions"]["pre_answer"][60]["answer_cum_rank"] for e in exs]
                a_ranks = [e["positions"]["pre_answer"][60]["a_cum_rank"]
                           for e in exs if e["positions"]["pre_answer"][60]["a_cum_rank"] is not None]
                print(f"  {cat:<25} n={len(exs):>3}  "
                      f"A+Y medRank={np.median(ranks):>6.0f}  "
                      f"A medRank={np.median(a_ranks):>6.0f}")


def main():
    parser = argparse.ArgumentParser(description="Per-layer logit lens attribution")
    parser.add_argument("--lm-head", type=Path,
                        default=Path("probing/lm_head_weight.npy"))
    parser.add_argument("--rms-norm", type=Path,
                        default=Path("probing/rms_norm_weight.npy"))
    parser.add_argument("--extraction-dir", type=Path,
                        default=Path("probing/extracted_states"))
    parser.add_argument("--conditions", nargs="+",
                        default=["baseline", "dots_50"])
    parser.add_argument("--positions", nargs="+",
                        default=["pre_answer", "answer_prompt", "question_end"])
    parser.add_argument("--model-path", type=str,
                        default="/workspace/models/deepseek-v3-awq")
    parser.add_argument("--categories", type=Path,
                        default=Path("probing/probe_results/categories.json"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("probing/layer_attribution_results"))
    parser.add_argument("--max-examples", type=int, default=None)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading lm_head weights...")
    lm_head = np.load(args.lm_head).astype(np.float32)
    print(f"  shape: {lm_head.shape}")

    norm_weight = None
    if args.rms_norm.exists():
        norm_weight = np.load(args.rms_norm).astype(np.float32)
        print(f"Loaded RMSNorm weight: shape={norm_weight.shape}")

    print("Loading tokenizer...")
    tokenizer = load_tokenizer(args.model_path)

    print("Running attribution...")
    results = run_attribution(
        args.extraction_dir, args.conditions, args.positions,
        lm_head, norm_weight, tokenizer,
        categories_file=args.categories,
        max_examples=args.max_examples,
    )

    aggregate_and_print(results, args.positions)

    # Save per-example results (slim — no full logit vectors)
    for condition, examples in results.items():
        outfile = args.output_dir / f"{condition}_attribution.json"
        with open(outfile, "w") as f:
            json.dump(examples, f)
        print(f"Saved {len(examples)} examples to {outfile}")

    print("Done.")


if __name__ == "__main__":
    main()
