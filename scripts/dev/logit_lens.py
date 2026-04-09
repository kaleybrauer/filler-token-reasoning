"""
logit_lens.py

Project hidden states through the unembedding matrix (lm_head) to see which
answer tokens the model promotes at each (layer, position) during filler.

No GPU needed — runs entirely on CPU using pre-extracted hidden states
and the saved lm_head weight.

Usage:
    source /workspace/config/probing_env.sh
    uv run --project /workspace/filler-token-reasoning/probing \
        python scripts/logit_lens.py \
        --extraction-dir data/extracted_states_dots50 \
        --condition dots_50 \
        --output-dir logit_lens_results/dots50
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from tqdm import tqdm


def load_tokenizer(model_path: str):
    from transformers import PreTrainedTokenizerFast
    return PreTrainedTokenizerFast.from_pretrained(model_path)


def get_answer_token_id(tokenizer, answer: int) -> int:
    """Get the single token ID for a numeric answer."""
    ids = tokenizer.encode(str(answer), add_special_tokens=False)
    if len(ids) != 1:
        return None  # multi-token answer, skip
    return ids[0]


def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Apply RMSNorm: x / RMS(x) * weight. Works on batched input (N, D)."""
    rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
    return (x / rms) * weight


def run_logit_lens(
    lm_head: np.ndarray,
    extraction_dir: Path,
    condition: str,
    tokenizer,
    output_dir: Path,
    categories_file: Path = None,
    max_examples: int = None,
    norm_weight: np.ndarray = None,
):
    """
    For each example, project hidden states through lm_head and record:
    - logit of the correct answer token
    - rank of the correct answer token
    - top-5 tokens and their logits
    """
    cond_dir = extraction_dir / condition
    files = sorted(cond_dir.glob("prob_*.pkl"))
    if max_examples:
        files = files[:max_examples]

    # Load categories if provided
    categories = {}
    if categories_file and categories_file.exists():
        with open(categories_file) as f:
            cat_data = json.load(f)
        cat_field = f"category_{condition}"
        for ex in cat_data["examples"]:
            field = cat_field if cat_field in ex else "category"
            categories[ex["problem_idx"]] = ex.get(field, "unknown")

    vocab_size, hidden_dim = lm_head.shape
    print(f"lm_head: {lm_head.shape}, dtype={lm_head.dtype}")
    print(f"Processing {len(files)} examples from {cond_dir}")

    all_results = []

    for fpath in tqdm(files, desc="Logit lens"):
        with open(fpath, "rb") as f:
            data = pickle.load(f)

        answer = data["answer"]
        answer_token_id = get_answer_token_id(tokenizer, answer)
        if answer_token_id is None:
            continue

        prob_idx = data["problem_idx"]
        positions = sorted(data["states"].keys(),
                           key=lambda p: data["positions"].get(p, 0))
        layers = sorted(data["states"][positions[0]].keys())

        example_result = {
            "problem_idx": prob_idx,
            "answer": answer,
            "fact_value": data["fact_value"],
            "x": data["x"],
            "answer_token_id": int(answer_token_id),
            "answer_token": tokenizer.decode([answer_token_id]),
            "model_correct": data.get("model_correct"),
            "category": categories.get(prob_idx, "unknown"),
            "positions": {},
        }

        # Batch all (position, layer) hidden states into one matrix
        pos_layer_keys = []
        h_rows = []
        for pos in positions:
            for layer in layers:
                pos_layer_keys.append((pos, layer))
                h_rows.append(data["states"][pos][layer])
        H = np.stack(h_rows, axis=0).astype(np.float32)  # (N, 7168)

        # Apply RMSNorm if available
        if norm_weight is not None:
            H = rms_norm(H, norm_weight)

        # Single batched matmul: (N, 7168) @ (7168, 129280) -> (N, 129280)
        all_logits = H @ lm_head.T

        for i, (pos, layer) in enumerate(pos_layer_keys):
            logits = all_logits[i]

            answer_logit = float(logits[answer_token_id])
            rank = int((logits > answer_logit).sum()) + 1

            # Top 5 tokens
            top5_idx = np.argpartition(logits, -5)[-5:]
            top5_idx = top5_idx[np.argsort(logits[top5_idx])[::-1]]
            top5 = [
                {"token_id": int(idx), "token": tokenizer.decode([idx]),
                 "logit": float(logits[idx])}
                for idx in top5_idx
            ]

            if pos not in example_result["positions"]:
                example_result["positions"][pos] = {}
            example_result["positions"][pos][layer] = {
                "answer_logit": answer_logit,
                "answer_rank": rank,
                "top5": top5,
            }

        all_results.append(example_result)

    return all_results


def aggregate_results(results: list, output_dir: Path):
    """Compute per-(layer, position) aggregates across examples."""
    if not results:
        return {}

    positions = list(results[0]["positions"].keys())
    layers = sorted(results[0]["positions"][positions[0]].keys())

    agg = {}
    for pos in positions:
        agg[pos] = {}
        for layer in layers:
            logits = []
            ranks = []
            for r in results:
                entry = r["positions"][pos][layer]
                logits.append(entry["answer_logit"])
                ranks.append(entry["answer_rank"])
            logits = np.array(logits)
            ranks = np.array(ranks)
            agg[pos][layer] = {
                "mean_logit": float(np.mean(logits)),
                "median_logit": float(np.median(logits)),
                "mean_rank": float(np.mean(ranks)),
                "median_rank": float(np.median(ranks)),
                "top1_frac": float(np.mean(ranks == 1)),
                "top5_frac": float(np.mean(ranks <= 5)),
                "top10_frac": float(np.mean(ranks <= 10)),
                "top100_frac": float(np.mean(ranks <= 100)),
            }

    # Also aggregate by category
    cats = set(r["category"] for r in results)
    by_cat = {}
    for cat in cats:
        cat_results = [r for r in results if r["category"] == cat]
        if len(cat_results) < 5:
            continue
        by_cat[cat] = {"n": len(cat_results)}
        for pos in positions:
            by_cat[cat][pos] = {}
            for layer in layers:
                logits = [r["positions"][pos][layer]["answer_logit"]
                          for r in cat_results]
                ranks = [r["positions"][pos][layer]["answer_rank"]
                         for r in cat_results]
                logits = np.array(logits)
                ranks = np.array(ranks)
                by_cat[cat][pos][layer] = {
                    "mean_logit": float(np.mean(logits)),
                    "mean_rank": float(np.mean(ranks)),
                    "median_rank": float(np.median(ranks)),
                    "top1_frac": float(np.mean(ranks == 1)),
                    "top10_frac": float(np.mean(ranks <= 10)),
                }

    return {"overall": agg, "by_category": by_cat}


def print_summary(agg: dict):
    """Print a readable summary table."""
    overall = agg["overall"]
    positions = list(overall.keys())
    layers = sorted(overall[positions[0]].keys())

    print(f"\n{'Position':<16} {'Best Layer':>10} {'MeanRank':>9} {'MedRank':>8} "
          f"{'Top1%':>6} {'Top10%':>7} {'MeanLogit':>10}")
    print("-" * 75)

    for pos in positions:
        # Find best layer by median rank
        best_layer = min(layers, key=lambda l: overall[pos][l]["median_rank"])
        d = overall[pos][best_layer]
        print(f"{pos:<16} L{best_layer:<9} {d['mean_rank']:>9.1f} {d['median_rank']:>8.1f} "
              f"{d['top1_frac']:>6.1%} {d['top10_frac']:>7.1%} {d['mean_logit']:>10.2f}")

    # Print by category if available
    by_cat = agg.get("by_category", {})
    if by_cat:
        # Show both answer_prompt and pre_answer if available
        summary_positions = [p for p in ["answer_prompt", "pre_answer"] if p in positions]
        for sp in summary_positions:
            print(f"\n--- By category ({sp}, best layer) ---")
            for cat, data in sorted(by_cat.items()):
                if sp not in data:
                    continue
                ap = data[sp]
                best_layer = min(
                    [l for l in ap.keys() if isinstance(l, int)],
                    key=lambda l: ap[l]["median_rank"]
                )
                d = ap[best_layer]
                print(f"  {cat:<25} (n={data['n']:>3}) L{best_layer}: "
                      f"medRank={d['median_rank']:.0f}, top1={d['top1_frac']:.1%}, "
                      f"meanLogit={d['mean_logit']:.1f}")


def plot_results(agg: dict, output_dir: Path):
    """Generate heatmaps of answer token rank/logit across layers and positions."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm
    except ImportError:
        print("matplotlib not available, skipping plots")
        return

    overall = agg["overall"]
    positions = list(overall.keys())
    layers = sorted(overall[positions[0]].keys())

    # Build matrices
    rank_matrix = np.zeros((len(layers), len(positions)))
    logit_matrix = np.zeros((len(layers), len(positions)))
    top1_matrix = np.zeros((len(layers), len(positions)))
    top10_matrix = np.zeros((len(layers), len(positions)))

    for j, pos in enumerate(positions):
        for i, layer in enumerate(layers):
            d = overall[pos][layer]
            rank_matrix[i, j] = d["median_rank"]
            logit_matrix[i, j] = d["mean_logit"]
            top1_matrix[i, j] = d["top1_frac"]
            top10_matrix[i, j] = d["top10_frac"]

    # Short position labels
    pos_labels = []
    for p in positions:
        if p == "question_end":
            pos_labels.append("q_end")
        elif p == "pre_filler":
            pos_labels.append("pre_fill")
        elif p == "answer_prompt":
            pos_labels.append("ans")
        elif p == "pre_answer":
            pos_labels.append("pre_ans")
        elif p.startswith("filler_k"):
            pos_labels.append(f"k={p.split('k')[1]}")
        else:
            pos_labels.append(p)

    # Show every 5th layer on y-axis
    layer_labels = [str(l) if l % 5 == 0 else "" for l in layers]

    fig, axes = plt.subplots(2, 2, figsize=(14, 18))

    # 1. Median rank (log scale)
    ax = axes[0, 0]
    im = ax.imshow(rank_matrix, aspect="auto", cmap="RdYlGn_r",
                   norm=LogNorm(vmin=1, vmax=max(rank_matrix.max(), 100)))
    ax.set_xticks(range(len(positions)))
    ax.set_xticklabels(pos_labels, rotation=45, ha="right")
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels(layer_labels, fontsize=7)
    ax.set_ylabel("Layer")
    ax.set_title("Median rank of correct answer token (lower = better)")
    plt.colorbar(im, ax=ax, label="Rank")

    # 2. Mean logit
    ax = axes[0, 1]
    im = ax.imshow(logit_matrix, aspect="auto", cmap="RdYlGn")
    ax.set_xticks(range(len(positions)))
    ax.set_xticklabels(pos_labels, rotation=45, ha="right")
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels(layer_labels, fontsize=7)
    ax.set_ylabel("Layer")
    ax.set_title("Mean logit of correct answer token")
    plt.colorbar(im, ax=ax, label="Logit")

    # 3. Top-1 fraction
    ax = axes[1, 0]
    im = ax.imshow(top1_matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(positions)))
    ax.set_xticklabels(pos_labels, rotation=45, ha="right")
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels(layer_labels, fontsize=7)
    ax.set_ylabel("Layer")
    ax.set_title("Fraction where correct answer is top-1 token")
    plt.colorbar(im, ax=ax, label="Fraction")

    # 4. Top-10 fraction
    ax = axes[1, 1]
    im = ax.imshow(top10_matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(positions)))
    ax.set_xticklabels(pos_labels, rotation=45, ha="right")
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels(layer_labels, fontsize=7)
    ax.set_ylabel("Layer")
    ax.set_title("Fraction where correct answer is top-10 token")
    plt.colorbar(im, ax=ax, label="Fraction")

    plt.tight_layout()
    plt.savefig(output_dir / "logit_lens_heatmaps.png", dpi=150, bbox_inches="tight")
    plt.savefig(output_dir / "logit_lens_heatmaps.pdf", bbox_inches="tight")
    plt.close()

    # === Line plots: best-layer metrics across positions ===
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # For each position, find the best layer by median rank
    best_ranks = []
    best_logits = []
    best_top1 = []
    best_layers_rank = []
    for j, pos in enumerate(positions):
        best_l = min(layers, key=lambda l: overall[pos][l]["median_rank"])
        best_layers_rank.append(best_l)
        best_ranks.append(overall[pos][best_l]["median_rank"])
        best_logits.append(overall[pos][best_l]["mean_logit"])
        best_top1.append(overall[pos][best_l]["top1_frac"])

    x = range(len(positions))

    ax = axes[0]
    ax.plot(x, best_ranks, "o-", color="#4CAF50", linewidth=2, markersize=8)
    for i, (r, l) in enumerate(zip(best_ranks, best_layers_rank)):
        ax.annotate(f"L{l}", (i, r), textcoords="offset points",
                    xytext=(0, 10), fontsize=8, ha="center")
    ax.set_xticks(x)
    ax.set_xticklabels(pos_labels, rotation=45, ha="right")
    ax.set_ylabel("Median rank (best layer)")
    ax.set_title("Answer token rank across positions")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(x, best_logits, "o-", color="#2196F3", linewidth=2, markersize=8)
    for i, (v, l) in enumerate(zip(best_logits, best_layers_rank)):
        ax.annotate(f"L{l}", (i, v), textcoords="offset points",
                    xytext=(0, 10), fontsize=8, ha="center")
    ax.set_xticks(x)
    ax.set_xticklabels(pos_labels, rotation=45, ha="right")
    ax.set_ylabel("Mean logit (best layer)")
    ax.set_title("Answer token logit across positions")
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(x, best_top1, "o-", color="#FF9800", linewidth=2, markersize=8)
    for i, (v, l) in enumerate(zip(best_top1, best_layers_rank)):
        ax.annotate(f"L{l}", (i, v), textcoords="offset points",
                    xytext=(0, 10), fontsize=8, ha="center")
    ax.set_xticks(x)
    ax.set_xticklabels(pos_labels, rotation=45, ha="right")
    ax.set_ylabel("Top-1 fraction (best layer)")
    ax.set_title("Fraction correct answer is top-1")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "logit_lens_trajectories.png", dpi=150, bbox_inches="tight")
    plt.savefig(output_dir / "logit_lens_trajectories.pdf", bbox_inches="tight")
    plt.close()

    # === By-category line plot (median rank at best overall layer) ===
    by_cat = agg.get("by_category", {})
    if by_cat and len(by_cat) > 1:
        fig, ax = plt.subplots(figsize=(10, 5))
        colors = {"filler_helped": "#4CAF50", "both_correct": "#2196F3",
                  "both_wrong": "#F44336", "filler_hurt": "#FF9800"}
        for cat, data in sorted(by_cat.items()):
            ranks_by_pos = []
            for pos in positions:
                if pos not in data:
                    continue
                # Use the overall best layer for this position
                best_l = best_layers_rank[positions.index(pos)]
                if best_l in data[pos]:
                    ranks_by_pos.append(data[pos][best_l]["median_rank"])
                else:
                    ranks_by_pos.append(np.nan)
            color = colors.get(cat, "gray")
            ax.plot(range(len(positions)), ranks_by_pos, "o-",
                    color=color, linewidth=2, markersize=6,
                    label=f"{cat} (n={data['n']})")

        ax.set_xticks(range(len(positions)))
        ax.set_xticklabels(pos_labels, rotation=45, ha="right")
        ax.set_ylabel("Median rank (best overall layer)")
        ax.set_title("Answer token rank by category")
        ax.set_yscale("log")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "logit_lens_by_category.png", dpi=150, bbox_inches="tight")
        plt.savefig(output_dir / "logit_lens_by_category.pdf", bbox_inches="tight")
        plt.close()

    print(f"Plots saved to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Logit lens analysis")
    parser.add_argument("--lm-head", type=Path,
                        default=Path("lm_head_weight.npy"))
    parser.add_argument("--extraction-dir", type=Path,
                        default=Path("data/extracted_states_dots50"))
    parser.add_argument("--condition", type=str, default="dots_50")
    parser.add_argument("--model-path", type=str,
                        default="/workspace/models/deepseek-v3-awq")
    parser.add_argument("--categories", type=Path,
                        default=Path("probe_results/categories.json"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("logit_lens_results/dots50"))
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--rms-norm", type=Path,
                        default=Path("rms_norm_weight.npy"),
                        help="Path to RMSNorm weight. Omit or set to "
                             "nonexistent path to skip normalization.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading lm_head weights...")
    lm_head = np.load(args.lm_head).astype(np.float32)

    norm_weight = None
    if args.rms_norm and args.rms_norm.exists():
        norm_weight = np.load(args.rms_norm).astype(np.float32)
        print(f"Loaded RMSNorm weight: shape={norm_weight.shape}")
    else:
        print("No RMSNorm weight — running without normalization")

    print("Loading tokenizer...")
    tokenizer = load_tokenizer(args.model_path)

    print("Running logit lens...")
    results = run_logit_lens(
        lm_head, args.extraction_dir, args.condition,
        tokenizer, args.output_dir,
        categories_file=args.categories,
        max_examples=args.max_examples,
        norm_weight=norm_weight,
    )

    print(f"\n{len(results)} examples processed")

    print("Aggregating...")
    agg = aggregate_results(results, args.output_dir)

    print_summary(agg)

    # Save results
    with open(args.output_dir / "aggregate.json", "w") as f:
        json.dump(agg, f, indent=2)

    # Save per-example (without top5 to keep file manageable)
    slim_results = []
    for r in results:
        slim = {k: v for k, v in r.items() if k != "positions"}
        slim["positions"] = {}
        for pos, layers in r["positions"].items():
            slim["positions"][pos] = {}
            for layer, d in layers.items():
                slim["positions"][pos][layer] = {
                    "answer_logit": d["answer_logit"],
                    "answer_rank": d["answer_rank"],
                }
        slim_results.append(slim)
    with open(args.output_dir / "per_example.json", "w") as f:
        json.dump(slim_results, f)

    print("Plotting...")
    plot_results(agg, args.output_dir)

    print("Done.")


if __name__ == "__main__":
    main()
