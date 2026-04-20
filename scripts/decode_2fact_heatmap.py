"""Decode heatmaps for 2-fact addition: which intermediate value is encoded at each (layer, position)?

For each (layer, position), decode via RMSNorm→lm_head→argmax over number tokens,
then compute ±5 accuracy against A1, A2, and A1+A2.

Usage:
    python scripts/decode_2fact_heatmap.py --condition dots_100
    python scripts/decode_2fact_heatmap.py --condition dots_10 dots_100 dots_200
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

plt.rcParams.update({"font.size": 20})


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


def pos_label(p):
    if p == "question_end": return "q_end"
    if p == "pre_filler": return "filler_label"
    if p == "answer_prompt": return "ans_prompt"
    if p.startswith("filler_k"): return f"k{p.split('k')[1]}"
    if p.startswith("pos_"): return p.split("_")[1].lstrip("0") or "0"
    return p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", nargs="+", default=["dots_100"])
    parser.add_argument("--extraction-dir", type=Path, default=Path("data/extracted_states_2fact_allpos"))
    parser.add_argument("--lm-head", type=Path, default=Path("data/model_weights/deepseek_v3/lm_head_weight.npy"))
    parser.add_argument("--rms-norm", type=Path, default=Path("data/model_weights/deepseek_v3/rms_norm_weight.npy"))
    parser.add_argument("--model-path", type=str, default="/workspace/models/deepseek-v3-awq")
    parser.add_argument("--output-dir", type=Path, default=Path("results/unsupervised_decode_2fact"))
    parser.add_argument("--incorrect-only", action="store_true",
                        help="Filter to examples where model got the final answer WRONG")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    from transformers import PreTrainedTokenizerFast
    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.model_path)

    # Build number token map
    number_tokens = {}
    for val in range(300):  # A1+A2 max is 117+116=233
        ids = tokenizer.encode(str(val), add_special_tokens=False)
        if len(ids) == 1:
            number_tokens[ids[0]] = val
    num_ids = sorted(number_tokens.keys())
    num_vals = np.array([number_tokens[tid] for tid in num_ids])
    print(f"Number tokens: {len(number_tokens)}")

    lm_head = np.load(args.lm_head).astype(np.float32)
    norm_weight = np.load(args.rms_norm).astype(np.float32)
    print(f"lm_head: {lm_head.shape}")

    for cond in args.condition:
        print(f"\n=== {cond} ===")
        files = sorted((args.extraction_dir / cond).glob("prob_*.pkl"))

        # Load all data, filter to correct examples only
        all_data = []
        n_total = 0
        for f in tqdm(files, desc="Loading"):
            with open(f, "rb") as fp:
                d = pickle.load(fp)
            n_total += 1
            correct = d.get("model_correct", False)
            if args.incorrect_only:
                if not correct:
                    all_data.append(d)
            else:
                if correct:
                    all_data.append(d)
        label = "incorrect" if args.incorrect_only else "correct"
        print(f"  {len(all_data)}/{n_total} {label} examples")

        # Get positions and layers
        d0 = all_data[0]
        positions = sorted(
            [p for p in d0["states"] if p.startswith("filler_k") or p.startswith("pos_")
             or p == "pre_filler" or p in ("question_end", "answer_prompt")],
            key=pos_sort_key
        )
        layers = sorted(d0["states"][positions[0]].keys())

        # Get boundary info for all-positions mode
        boundaries = d0.get("boundaries")

        # Ground truth
        A1 = np.array([d["fact_value_1"] for d in all_data])
        A2 = np.array([d["fact_value_2"] for d in all_data])
        A1A2 = np.array([d["answer"] for d in all_data])

        # Decode at every (layer, position)
        results = {"_positions": positions, "_layers": layers, "_condition": cond,
                    "_n": len(all_data)}
        if boundaries:
            results["_boundaries"] = boundaries

        for pos in positions:
            results[pos] = {}
            for layer in tqdm(layers, desc=f"  {pos}", leave=False):
                vecs = []
                valid_idx = []
                for i, d in enumerate(all_data):
                    if pos in d["states"] and layer in d["states"][pos]:
                        vecs.append(d["states"][pos][layer].astype(np.float32))
                        valid_idx.append(i)

                if not vecs:
                    continue

                H = np.stack(vecs)
                H = rms_norm(H, norm_weight)
                logits = H @ lm_head.T
                num_logits = logits[:, num_ids]
                preds = num_vals[np.argmax(num_logits, axis=1)]

                a1 = A1[valid_idx]
                a2 = A2[valid_idx]
                a1a2 = A1A2[valid_idx]

                results[pos][str(layer)] = {
                    "frac_A1_exact": float(np.mean(preds == a1)),
                    "frac_A1_within5": float(np.mean(np.abs(preds - a1) <= 5)),
                    "frac_A2_exact": float(np.mean(preds == a2)),
                    "frac_A2_within5": float(np.mean(np.abs(preds - a2) <= 5)),
                    "frac_A1A2_exact": float(np.mean(preds == a1a2)),
                    "frac_A1A2_within5": float(np.mean(np.abs(preds - a1a2) <= 5)),
                    "frac_A1_or_A2_within5": float(np.mean(
                        (np.abs(preds - a1) <= 5) | (np.abs(preds - a2) <= 5)
                    )),
                }
            print(f"  {pos} done")

        # Save JSON
        suffix_cond = f"{cond}_incorrect" if args.incorrect_only else cond
        outfile = args.output_dir / f"decode_2fact_{suffix_cond}.json"
        with open(outfile, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Saved {outfile}")

        # Detect all-positions mode
        is_allpos = any(p.startswith("pos_") for p in positions)
        n_pos = len(positions)

        # Plot heatmaps: ±5 and exact match
        for metric_set, suffix, cbar_label in [
            ([("frac_A1_within5", "A₁ (±5)"),
              ("frac_A2_within5", "A₂ (±5)"),
              ("frac_A1A2_within5", "A₁+A₂ (±5)")],
             "within5", "% within ±5"),
            ([("frac_A1_exact", "A₁ (exact)"),
              ("frac_A2_exact", "A₂ (exact)"),
              ("frac_A1A2_exact", "A₁+A₂ (exact)")],
             "exact", "% exact match"),
        ]:
            fig_width = max(7 * 3, n_pos * 0.12 * 3) if is_allpos else 7 * 3
            fig, axes = plt.subplots(1, 3, figsize=(fig_width, 11))

            for ax, (metric, title) in zip(axes, metric_set):
                matrix = np.full((len(layers), len(positions)), np.nan)
                for j, pos in enumerate(positions):
                    for i, layer in enumerate(layers):
                        if str(layer) in results.get(pos, {}):
                            matrix[i, j] = results[pos][str(layer)][metric] * 100

                im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=100,
                               interpolation="nearest")

                if is_allpos:
                    # Sparse x-ticks for all-positions mode
                    tick_step = max(1, n_pos // 20)
                    tick_idxs = list(range(0, n_pos, tick_step))
                    if (n_pos - 1) not in tick_idxs:
                        tick_idxs.append(n_pos - 1)
                    ax.set_xticks(tick_idxs)
                    ax.set_xticklabels([pos_label(positions[i]) for i in tick_idxs],
                                       rotation=45, ha="right")
                    ax.set_xlabel("Offset from question_end")
                    # Annotate filler boundaries
                    if boundaries:
                        fe = boundaries.get("filler_end_offset")
                        if fe is not None:
                            ax.axvline(fe + 0.5, color="white", linewidth=1, linestyle="--", alpha=0.7)
                else:
                    ax.set_xticks(range(n_pos))
                    ax.set_xticklabels([pos_label(p) for p in positions], rotation=45, ha="right")

                layer_labels = [str(l) if l % 5 == 0 else "" for l in layers]
                ax.set_yticks(range(len(layers)))
                ax.set_yticklabels(layer_labels)
                ax.set_title(title, fontweight="bold")
                if ax == axes[0]:
                    ax.set_ylabel("Layer")

            title_suffix = " — model INCORRECT" if args.incorrect_only else ""
            fig.suptitle(f"{cond}{title_suffix}: what is encoded at each (layer, position)?",
                         fontsize=22)
            fig.subplots_adjust(right=0.88, wspace=0.15, top=0.93)
            cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.65])
            fig.colorbar(im, cax=cbar_ax, label=cbar_label)

            for ext in ["png", "pdf"]:
                fig.savefig(args.output_dir / f"heatmap_2fact_{suffix_cond}_{suffix}.{ext}",
                            dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  Saved heatmap ({suffix})")

    print("\nDone.")


if __name__ == "__main__":
    main()
