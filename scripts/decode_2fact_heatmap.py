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
    if p == "answer_prompt": return 99999
    return 9999


def pos_label(p):
    if p == "question_end": return "q_end"
    if p == "pre_filler": return "filler_label"
    if p == "answer_prompt": return "ans_prompt"
    if p.startswith("filler_k"): return f"k{p.split('k')[1]}"
    return p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", nargs="+", default=["dots_100"])
    parser.add_argument("--extraction-dir", type=Path, default=Path("data/extracted_states_2fact"))
    parser.add_argument("--lm-head", type=Path, default=Path("data/model_weights/deepseek_v3.2/lm_head_weight.npy"))
    parser.add_argument("--rms-norm", type=Path, default=Path("data/model_weights/deepseek_v3.2/rms_norm_weight.npy"))
    parser.add_argument("--model-path", type=str, default="/workspace/models/deepseek-v3-awq")
    parser.add_argument("--output-dir", type=Path, default=Path("results/unsupervised_decode_2fact"))
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

        # Load all data
        all_data = []
        for f in tqdm(files, desc="Loading"):
            with open(f, "rb") as fp:
                d = pickle.load(fp)
            all_data.append(d)
        print(f"  {len(all_data)} examples")

        # Get positions and layers
        d0 = all_data[0]
        positions = sorted(
            [p for p in d0["states"] if p.startswith("filler_k") or p == "pre_filler"
             or p in ("question_end", "answer_prompt")],
            key=pos_sort_key
        )
        layers = sorted(d0["states"][positions[0]].keys())

        # Ground truth
        A1 = np.array([d["fact_value_1"] for d in all_data])
        A2 = np.array([d["fact_value_2"] for d in all_data])
        A1A2 = np.array([d["answer"] for d in all_data])

        # Decode at every (layer, position)
        results = {"_positions": positions, "_layers": layers, "_condition": cond,
                    "_n": len(all_data)}

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
        outfile = args.output_dir / f"decode_2fact_{cond}.json"
        with open(outfile, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Saved {outfile}")

        # Plot: 3-panel heatmap (A1, A2, A1+A2 ±5 accuracy)
        targets = [
            ("frac_A1_within5", "A1 (±5)", "A₁"),
            ("frac_A2_within5", "A2 (±5)", "A₂"),
            ("frac_A1A2_within5", "A1+A2 (±5)", "A₁+A₂"),
        ]

        fig, axes = plt.subplots(1, 3, figsize=(7 * 3, 11))

        for ax, (metric, title, short) in zip(axes, targets):
            matrix = np.full((len(layers), len(positions)), np.nan)
            for j, pos in enumerate(positions):
                for i, layer in enumerate(layers):
                    if str(layer) in results.get(pos, {}):
                        matrix[i, j] = results[pos][str(layer)][metric] * 100

            im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=100,
                           interpolation="nearest")
            ax.set_xticks(range(len(positions)))
            ax.set_xticklabels([pos_label(p) for p in positions], rotation=45, ha="right")
            layer_labels = [str(l) if l % 5 == 0 else "" for l in layers]
            ax.set_yticks(range(len(layers)))
            ax.set_yticklabels(layer_labels)
            ax.set_title(f"Decode → {short}", fontweight="bold")
            if ax == axes[0]:
                ax.set_ylabel("Layer")

        fig.suptitle(f"{cond}: what is encoded at each (layer, position)?", fontsize=22)
        fig.subplots_adjust(right=0.88, wspace=0.15, top=0.93)
        cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.65])
        fig.colorbar(im, cax=cbar_ax, label="% within ±5")

        for ext in ["png", "pdf"]:
            fig.savefig(args.output_dir / f"heatmap_2fact_{cond}.{ext}",
                        dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved heatmap")

    print("\nDone.")


if __name__ == "__main__":
    main()
