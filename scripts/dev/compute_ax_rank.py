"""Compute A+X rank at every (layer, position) for filler conditions.

Outputs JSON with median/mean rank and top-K fractions, plus heatmap plots.

Usage:
    python probing/scripts/compute_ax_rank.py --condition dots_10 dots_100
"""

import argparse
import json
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

try:
    import torch
    if torch.cuda.is_available():
        # Test that CUDA actually works (may fail on unsupported GPU arch)
        try:
            torch.zeros(1, device="cuda")
            HAS_TORCH = True
        except RuntimeError:
            HAS_TORCH = False
    else:
        HAS_TORCH = False
except ImportError:
    HAS_TORCH = False

plt.rcParams.update({"font.size": 28})


def rms_norm(x, weight, eps=1e-6):
    rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
    return (x / rms) * weight


def rms_norm_torch(x, weight, eps=1e-6):
    rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + eps)
    return (x / rms) * weight


def pos_sort_key(p):
    if p == "pre_filler": return -1
    if p.startswith("filler_k"): return int(p.split("k")[1])
    return 9999


def pos_label(p):
    if p == "pre_filler": return "filler_label"
    if p.startswith("filler_k"): return f"k{p.split('k')[1]}"
    return p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", nargs="+", default=["dots_10", "dots_100"])
    parser.add_argument("--extraction-dir", type=Path, default=Path("probing/extracted_states"))
    parser.add_argument("--lm-head", type=Path, default=Path("probing/lm_head_weight.npy"))
    parser.add_argument("--rms-norm", type=Path, default=Path("probing/rms_norm_weight.npy"))
    parser.add_argument("--model-path", type=str, default="/workspace/models/deepseek-v3-awq")
    parser.add_argument("--output-dir", type=Path, default=Path("probing/results/unsupervised_decode"))
    parser.add_argument("--filter-category", type=str, default=None,
                        help="Filter to a category: filler_helped, both_correct, both_wrong, filler_hurt")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    from transformers import PreTrainedTokenizerFast
    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.model_path)

    lm_head_np = np.load(args.lm_head).astype(np.float32)
    norm_weight_np = np.load(args.rms_norm).astype(np.float32)

    if HAS_TORCH:
        device = torch.device("cuda")
        lm_head_t = torch.from_numpy(lm_head_np).to(device)
        norm_weight_t = torch.from_numpy(norm_weight_np).to(device)
        print(f"Using GPU: {torch.cuda.get_device_name()}")
    else:
        print("Using CPU (no CUDA available)")

    for cond in args.condition:
        print(f"\n=== {cond} ===")
        files = sorted((args.extraction_dir / cond).glob("prob_*.pkl"))

        # Build category filter if requested
        filter_idx = None
        if args.filter_category:
            bl_dir = args.extraction_dir / "baseline"
            bl_files = sorted(bl_dir.glob("prob_*.pkl"))
            cond_files = sorted((args.extraction_dir / cond).glob("prob_*.pkl"))
            filter_idx = set()
            for bf, cf in zip(bl_files, cond_files):
                with open(bf, "rb") as fp:
                    bd = pickle.load(fp)
                with open(cf, "rb") as fp:
                    cd = pickle.load(fp)
                bc = bd.get("model_correct", False)
                fc = cd.get("model_correct", False)
                if bc and fc:
                    cat = "both_correct"
                elif not bc and fc:
                    cat = "filler_helped"
                elif bc and not fc:
                    cat = "filler_hurt"
                else:
                    cat = "both_wrong"
                if cat == args.filter_category:
                    filter_idx.add(bd["problem_idx"])
            print(f"  Filter: {args.filter_category} → {len(filter_idx)} examples")

        # Load all data once
        all_data = []
        for f in tqdm(files, desc="Loading"):
            with open(f, "rb") as fp:
                d = pickle.load(fp)
            if filter_idx is not None and d["problem_idx"] not in filter_idx:
                continue
            ax_tok = tokenizer.encode(str(d["answer"]), add_special_tokens=False)
            if len(ax_tok) != 1:
                continue
            all_data.append((d, ax_tok[0]))
        print(f"  {len(all_data)} examples")

        # Get positions and layers
        d0 = all_data[0][0]
        positions = sorted(
            [p for p in d0["states"] if p.startswith("filler_k") or p == "pre_filler"],
            key=pos_sort_key
        )
        layers = sorted(d0["states"][positions[0]].keys())

        # Compute ranks — batch all examples per (position, layer)
        results = {"_positions": positions, "_layers": layers, "_condition": cond}

        for pos in positions:
            results[pos] = {}
            for layer in tqdm(layers, desc=f"  {pos}", leave=False):
                # Gather all vectors
                vecs = []
                ax_ids = []
                for d, ax_tok_id in all_data:
                    if pos not in d["states"] or layer not in d["states"][pos]:
                        continue
                    vecs.append(d["states"][pos][layer].astype(np.float32))
                    ax_ids.append(ax_tok_id)

                if not vecs:
                    continue

                if HAS_TORCH:
                    H = torch.from_numpy(np.stack(vecs)).to(device)
                    H = rms_norm_torch(H, norm_weight_t)
                    logits = H @ lm_head_t.T  # (N, 129280)
                    ax_logits = logits[torch.arange(len(ax_ids), device=device),
                                       torch.tensor(ax_ids, device=device)]
                    ranks = (logits > ax_logits.unsqueeze(1)).sum(dim=1) + 1
                    ranks = ranks.cpu().numpy()
                else:
                    H = np.stack(vecs)
                    H = rms_norm(H, norm_weight_np)
                    logits = H @ lm_head_np.T
                    ax_logits = np.array([logits[i, tid] for i, tid in enumerate(ax_ids)])
                    ranks = (logits > ax_logits[:, None]).sum(axis=1) + 1

                results[pos][str(layer)] = {
                    "median_rank": float(np.median(ranks)),
                    "mean_rank": float(np.mean(ranks)),
                    "frac_top10": float(np.mean(ranks <= 10)),
                    "frac_top100": float(np.mean(ranks <= 100)),
                }
            print(f"  {pos} done")

        # Save JSON
        suffix = f"_{args.filter_category}" if args.filter_category else ""
        outfile = args.output_dir / f"ax_rank_{cond}{suffix}.json"
        with open(outfile, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Saved {outfile}")

        # Plot heatmap: log10(median rank)
        matrix = np.full((len(layers), len(positions)), np.nan)
        for j, pos in enumerate(positions):
            for i, layer in enumerate(layers):
                if str(layer) in results.get(pos, {}):
                    matrix[i, j] = np.log10(max(results[pos][str(layer)]["median_rank"], 1))

        title_suffix = f" ({args.filter_category})" if args.filter_category else ""
        fig, ax = plt.subplots(figsize=(max(7, len(positions) * 1.5 + 2), 11))
        im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=5,
                       interpolation="nearest")
        ax.set_xticks(range(len(positions)))
        ax.set_xticklabels([pos_label(p) for p in positions], rotation=45, ha="right")
        layer_labels = [str(l) if l % 5 == 0 else "" for l in layers]
        ax.set_yticks(range(len(layers)))
        ax.set_yticklabels(layer_labels)
        ax.set_ylabel("Layer")
        ax.set_title(f"{cond}: A+X median rank (log10){title_suffix}", fontweight="bold")
        fig.colorbar(im, ax=ax, label="log10(median rank)")
        plt.tight_layout()

        for ext in ["png", "pdf"]:
            fig.savefig(args.output_dir / f"ax_rank_heatmap_{cond}{suffix}.{ext}", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved heatmap")

    print("\nDone.")


if __name__ == "__main__":
    main()
