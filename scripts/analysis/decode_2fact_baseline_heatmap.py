"""Decode heatmap for the 2-fact BASELINE (no filler) condition: A1, A2, sum.

Baseline has only two positions (question_end, answer_prompt) since there is no
filler region. For each (layer, position) we logit-lens decode via
RMSNorm -> lm_head -> argmax over single-token integers, then score EXACT match
against A1 (fact_value_1), A2 (fact_value_2), and the sum (answer).

Methodology is identical to scripts/analysis/decode_2fact_heatmap.py; this is a
torch-free standalone (tokenizers backend, no transformers/torch import) so it
runs in the minimal CPU venv. lm_head is pre-sliced to the number-token rows so
the matmul is cheap on a small CPU box.

Usage:
    python scripts/analysis/decode_2fact_baseline_heatmap.py
    python scripts/analysis/decode_2fact_baseline_heatmap.py --subset all
"""

import argparse
import json
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tokenizers import Tokenizer
from tqdm import tqdm

plt.rcParams.update({"font.size": 20})


def rms_norm(x, weight, eps=1e-6):
    rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
    return (x / rms) * weight


POS_ORDER = {"question_end": -2, "answer_prompt": 50000, "pre_answer": 99999}
POS_LABEL = {"question_end": "q_end", "answer_prompt": "ans_prompt", "pre_answer": "pre_answer"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extraction-dir", type=Path,
                    default=Path("data/extracted_states_2fact"))
    ap.add_argument("--condition", default="baseline")
    ap.add_argument("--subset", choices=["correct", "all", "incorrect"],
                    default="correct",
                    help="Which examples to decode (default: correct, matching the filler heatmaps)")
    ap.add_argument("--lm-head", type=Path,
                    default=Path("data/model_weights/deepseek_v3/lm_head_weight.npy"))
    ap.add_argument("--rms-norm", type=Path,
                    default=Path("data/model_weights/deepseek_v3/rms_norm_weight.npy"))
    ap.add_argument("--tokenizer-json", type=Path,
                    default=Path("/workspace/models/deepseek-v3-awq/tokenizer.json"))
    ap.add_argument("--output-dir", type=Path,
                    default=Path("results/unsupervised_decode_2fact_allpos"))
    ap.add_argument("--tag", default="",
                    help="Optional suffix on output filenames (e.g. _preanswer) so a "
                         "pre_answer run does not overwrite the answer_prompt-only baseline")
    ap.add_argument("--max-num", type=int, default=300,
                    help="Build single-token integer map over [0, max-num)")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # --- number-token map (single-token integers only), same as decode_2fact_heatmap ---
    tok = Tokenizer.from_file(str(args.tokenizer_json))
    number_tokens = {}
    for val in range(args.max_num):
        ids = tok.encode(str(val), add_special_tokens=False).ids
        if len(ids) == 1:
            number_tokens[ids[0]] = val
    num_ids = sorted(number_tokens.keys())
    num_vals = np.array([number_tokens[t] for t in num_ids])
    print(f"Number tokens (single-token ints in [0,{args.max_num})): {len(num_ids)}")

    lm_head = np.load(args.lm_head).astype(np.float32)      # (vocab, 7168)
    norm_weight = np.load(args.rms_norm).astype(np.float32)  # (7168,)
    lm_head_num = lm_head[np.array(num_ids)]                 # (n_num, 7168) -- pre-slice
    print(f"lm_head: {lm_head.shape}; sliced to numbers: {lm_head_num.shape}")

    # --- load extraction, filter to subset ---
    cond_dir = args.extraction_dir / args.condition
    files = sorted(cond_dir.glob("prob_*.pkl"))
    data = []
    n_total = 0
    for f in tqdm(files, desc="Loading"):
        with open(f, "rb") as fp:
            d = pickle.load(fp)
        n_total += 1
        correct = bool(d.get("model_correct", False))
        keep = (args.subset == "all"
                or (args.subset == "correct" and correct)
                or (args.subset == "incorrect" and not correct))
        if keep:
            # Keep only what we need (states + ground truth) to bound memory.
            data.append(d)
    print(f"  {len(data)}/{n_total} examples ({args.subset})")
    if not data:
        raise SystemExit("No examples after filtering.")

    positions = sorted([p for p in data[0]["states"]], key=lambda p: POS_ORDER.get(p, 0))
    layers = sorted(data[0]["states"][positions[0]].keys())
    print(f"  positions={positions}  layers={layers[0]}..{layers[-1]} ({len(layers)})")

    A1 = np.array([d["fact_value_1"] for d in data])
    A2 = np.array([d["fact_value_2"] for d in data])
    SUM = np.array([d["answer"] for d in data])
    # sanity: A1+A2 == answer
    assert np.all(A1 + A2 == SUM), "fact_value_1 + fact_value_2 != answer"

    results = {"_positions": positions, "_layers": layers,
               "_condition": args.condition, "_subset": args.subset, "_n": len(data)}

    for pos in positions:
        results[pos] = {}
        for layer in layers:
            H = np.stack([d["states"][pos][layer].astype(np.float32) for d in data])
            H = rms_norm(H, norm_weight)
            num_logits = H @ lm_head_num.T          # (n, n_num)
            preds = num_vals[np.argmax(num_logits, axis=1)]
            results[pos][str(layer)] = {
                "frac_A1_exact": float(np.mean(preds == A1)),
                "frac_A2_exact": float(np.mean(preds == A2)),
                "frac_A1A2_exact": float(np.mean(preds == SUM)),
                "frac_A1_within5": float(np.mean(np.abs(preds - A1) <= 5)),
                "frac_A2_within5": float(np.mean(np.abs(preds - A2) <= 5)),
                "frac_A1A2_within5": float(np.mean(np.abs(preds - SUM) <= 5)),
            }

    suffix = args.condition if args.subset == "correct" else f"{args.condition}_{args.subset}"
    suffix += args.tag
    outfile = args.output_dir / f"decode_2fact_{suffix}.json"
    with open(outfile, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved {outfile}")

    # --- plot 3-panel EXACT-match heatmap (layers x positions) ---
    metrics = [("frac_A1_exact", "A₁ (exact)"),
               ("frac_A2_exact", "A₂ (exact)"),
               ("frac_A1A2_exact", "A₁+A₂ (exact)")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 11))
    im = None
    for ax, (metric, title) in zip(axes, metrics):
        matrix = np.full((len(layers), len(positions)), np.nan)
        for j, pos in enumerate(positions):
            for i, layer in enumerate(layers):
                matrix[i, j] = results[pos][str(layer)][metric] * 100
        im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=100,
                       interpolation="nearest")
        ax.set_xticks(range(len(positions)))
        ax.set_xticklabels([POS_LABEL.get(p, p) for p in positions], rotation=45, ha="right")
        ax.set_yticks(range(len(layers)))
        ax.set_yticklabels([str(l) if l % 5 == 0 else "" for l in layers])
        ax.set_title(title, fontweight="bold")
        if ax is axes[0]:
            ax.set_ylabel("Layer")

    sub_txt = "" if args.subset == "correct" else f" — {args.subset}"
    fig.suptitle(f"2-fact baseline (no filler), n={len(data)}{sub_txt}: "
                 f"value decoded at each (layer, position)", fontsize=20)
    fig.subplots_adjust(right=0.88, wspace=0.25, top=0.92)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.65])
    fig.colorbar(im, cax=cbar_ax, label="% exact match")
    for ext in ["png", "pdf"]:
        fig.savefig(args.output_dir / f"heatmap_2fact_{suffix}_exact.{ext}",
                    dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved heatmap_2fact_{suffix}_exact.png/.pdf")

    # --- console summary: peak exact-match layer per (target, position) ---
    print("\nPeak exact-match (layer, %):")
    for metric, name in [("frac_A1_exact", "A1"), ("frac_A2_exact", "A2"),
                         ("frac_A1A2_exact", "sum")]:
        row = []
        for pos in positions:
            vals = [(l, results[pos][str(l)][metric] * 100) for l in layers]
            bl, bv = max(vals, key=lambda t: t[1])
            row.append(f"{POS_LABEL.get(pos, pos)}: L{bl}={bv:.0f}%")
        print(f"  {name:3s}  " + "   ".join(row))


if __name__ == "__main__":
    main()
