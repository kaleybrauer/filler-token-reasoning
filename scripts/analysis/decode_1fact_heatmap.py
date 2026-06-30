"""Decode heatmaps for 1-fact addition: which value is encoded at each (layer, position)?

Sibling of decode_2fact_heatmap.py, adapted for the 1-hop task
("Question: What is <fact_phrase> plus <x>?"). Targets:
    A      = fact_value      (the looked-up value, e.g. silicon → 14)
    X      = x               (the addend, also visible in the prompt)
    A+X    = answer           (the sum)

For each (layer, position), decode via RMSNorm → lm_head → argmax over
single-token integer strings 0..299, then compute exact and ±5 accuracy
against each of the three targets.

Defaults target the Qwen3-32B 1-hop allpos extraction:
    data/qwen3-32b/extracted_states_1hop_allpos_qwen3_32b/<cond>/prob_*.pkl
    data/model_weights/qwen3_32b/{lm_head_weight,rms_norm_weight}.npy

Usage:
    python scripts/analysis/decode_1fact_heatmap.py \
        --extraction-dir data/qwen3-32b/extracted_states_1hop_allpos_qwen3_32b \
        --condition dots_10 counting_10
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
    parser.add_argument("--condition", nargs="+", default=["dots_10", "counting_10"])
    parser.add_argument(
        "--extraction-dir", type=Path,
        default=Path("data/qwen3-32b/extracted_states_1hop_allpos_qwen3_32b"))
    parser.add_argument(
        "--lm-head", type=Path,
        default=Path("data/model_weights/qwen3_32b/lm_head_weight.npy"))
    parser.add_argument(
        "--rms-norm", type=Path,
        default=Path("data/model_weights/qwen3_32b/rms_norm_weight.npy"))
    parser.add_argument("--model-path", type=str,
                        default="/workspace/models/qwen3-32b")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/qwen3_1fact_heatmap"))
    parser.add_argument("--max-num-token", type=int, default=300,
                        help="Highest integer string to include in the number-token map.")
    parser.add_argument("--digit-mode", action="store_true",
                        help="Auto-enabled if the tokenizer is per-digit (e.g. Qwen3 splits "
                             "76 → '7','6'). Argmax over digit tokens 0–9 and match against "
                             "the FIRST digit of A/X/A+X. ±5 metric is replaced by first-digit "
                             "exact match (the only meaningful metric in this mode).")
    parser.add_argument("--incorrect-only", action="store_true",
                        help="Filter to examples where the model got the final answer WRONG. "
                             "Default: filter to correct examples.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    import sys
    REPO_ROOT = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from extract.extract_hidden_states import load_tokenizer

    tokenizer = load_tokenizer(args.model_path)

    # Build single-token integer string map (subset of vocab whose decode is "0", "1", ..., str(N-1))
    number_tokens = {}
    for val in range(args.max_num_token):
        ids = tokenizer.encode(str(val), add_special_tokens=False)
        if len(ids) == 1:
            number_tokens[ids[0]] = val
    print(f"Single-token integers in 0..{args.max_num_token}: {len(number_tokens)}")

    # Auto-detect digit-mode tokenizers: if even small 2-digit integers split into
    # multiple tokens (Qwen3, Llama 3, GPT-2, …), restrict to digits 0-9 and match
    # by first digit. Threshold: < 30% of integers in [0, max_num_token] single-token.
    digit_mode = args.digit_mode or len(number_tokens) < 0.3 * args.max_num_token
    if digit_mode and not args.digit_mode:
        print(f"  → auto-enabling --digit-mode (per-digit tokenizer detected)")
    if digit_mode:
        # Keep only the 10 single-digit tokens
        number_tokens = {tid: val for tid, val in number_tokens.items() if val < 10}
        print(f"  digit-mode tokens kept: {sorted(number_tokens.values())}")
    num_ids = sorted(number_tokens.keys())
    num_vals = np.array([number_tokens[tid] for tid in num_ids])

    lm_head = np.load(args.lm_head).astype(np.float32)
    norm_weight = np.load(args.rms_norm).astype(np.float32)
    print(f"lm_head: {lm_head.shape}, norm: {norm_weight.shape}")

    for cond in args.condition:
        print(f"\n=== {cond} ===")
        cond_dir = args.extraction_dir / cond
        files = sorted(cond_dir.glob("prob_*.pkl"))

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
        if not all_data:
            print(f"  no examples to decode, skipping")
            continue

        d0 = all_data[0]
        positions = sorted(
            [p for p in d0["states"]
             if p.startswith("filler_k") or p.startswith("pos_")
             or p == "pre_filler" or p in ("question_end", "answer_prompt")],
            key=pos_sort_key,
        )
        layers = sorted(d0["states"][positions[0]].keys())
        boundaries = d0.get("boundaries")

        A_full = np.array([d["fact_value"] for d in all_data])
        X_full = np.array([d["x"] for d in all_data])
        AX_full = np.array([d["answer"] for d in all_data])
        if digit_mode:
            # Compare against the FIRST digit only (model's first generated token
            # is the leading digit of the final answer).
            A = np.array([int(str(v)[0]) for v in A_full])
            X = np.array([int(str(v)[0]) for v in X_full])
            AX = np.array([int(str(v)[0]) for v in AX_full])
        else:
            A, X, AX = A_full, X_full, AX_full

        results = {"_positions": positions, "_layers": layers, "_condition": cond,
                   "_n": len(all_data),
                   "_digit_mode": bool(digit_mode),
                   "_target_legend": {"A": "fact_value", "X": "x", "AX": "fact_value+x"}}
        if digit_mode:
            results["_target_legend"] = {
                "A": "first digit of fact_value",
                "X": "first digit of x",
                "AX": "first digit of fact_value+x",
            }
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

                a = A[valid_idx]
                x = X[valid_idx]
                ax = AX[valid_idx]

                results[pos][str(layer)] = {
                    "frac_A_exact":  float(np.mean(preds == a)),
                    "frac_X_exact":  float(np.mean(preds == x)),
                    "frac_AX_exact": float(np.mean(preds == ax)),
                }
                if not digit_mode:
                    results[pos][str(layer)].update({
                        "frac_A_within5":   float(np.mean(np.abs(preds - a) <= 5)),
                        "frac_X_within5":   float(np.mean(np.abs(preds - x) <= 5)),
                        "frac_AX_within5":  float(np.mean(np.abs(preds - ax) <= 5)),
                        "frac_A_or_X_within5": float(np.mean(
                            (np.abs(preds - a) <= 5) | (np.abs(preds - x) <= 5)
                        )),
                    })
            print(f"  {pos} done")

        suffix_cond = f"{cond}_incorrect" if args.incorrect_only else cond
        outfile = args.output_dir / f"decode_1fact_{suffix_cond}.json"
        with open(outfile, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Saved {outfile}")

        is_allpos = any(p.startswith("pos_") for p in positions)
        n_pos = len(positions)

        if digit_mode:
            metric_sets = [(
                [("frac_A_exact",  "first-digit(A)"),
                 ("frac_X_exact",  "first-digit(x)"),
                 ("frac_AX_exact", "first-digit(A+x)")],
                "first_digit", "% first-digit match",
            )]
        else:
            metric_sets = [
                ([("frac_A_within5",  "A (±5)"),
                  ("frac_X_within5",  "x (±5)"),
                  ("frac_AX_within5", "A+x (±5)")],
                 "within5", "% within ±5"),
                ([("frac_A_exact",  "A (exact)"),
                  ("frac_X_exact",  "x (exact)"),
                  ("frac_AX_exact", "A+x (exact)")],
                 "exact", "% exact match"),
            ]

        for metric_set, suffix, cbar_label in metric_sets:
            fig_width = max(7 * 3, n_pos * 0.12 * 3) if is_allpos else 7 * 3
            fig, axes = plt.subplots(1, 3, figsize=(fig_width, 11))

            for ax, (metric, title) in zip(axes, metric_set):
                matrix = np.full((len(layers), len(positions)), np.nan)
                for j, pos in enumerate(positions):
                    for i, layer in enumerate(layers):
                        if str(layer) in results.get(pos, {}):
                            matrix[i, j] = results[pos][str(layer)][metric] * 100

                im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn",
                               vmin=0, vmax=100, interpolation="nearest")

                if is_allpos:
                    tick_step = max(1, n_pos // 20)
                    tick_idxs = list(range(0, n_pos, tick_step))
                    if (n_pos - 1) not in tick_idxs:
                        tick_idxs.append(n_pos - 1)
                    ax.set_xticks(tick_idxs)
                    ax.set_xticklabels(
                        [pos_label(positions[i]) for i in tick_idxs],
                        rotation=45, ha="right",
                    )
                    ax.set_xlabel("Offset from question_end")
                    if boundaries:
                        fe = boundaries.get("filler_end_offset")
                        if fe is not None:
                            ax.axvline(fe + 0.5, color="white", linewidth=1,
                                       linestyle="--", alpha=0.7)
                else:
                    ax.set_xticks(range(n_pos))
                    ax.set_xticklabels([pos_label(p) for p in positions],
                                       rotation=45, ha="right")

                layer_labels = [str(l) if l % 5 == 0 else "" for l in layers]
                ax.set_yticks(range(len(layers)))
                ax.set_yticklabels(layer_labels)
                ax.set_title(title, fontweight="bold")
                if ax == axes[0]:
                    ax.set_ylabel("Layer")

            title_suffix = " — model INCORRECT" if args.incorrect_only else ""
            fig.suptitle(
                f"{cond}{title_suffix}: what is encoded at each (layer, position)?",
                fontsize=22,
            )
            fig.subplots_adjust(right=0.88, wspace=0.15, top=0.93)
            cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.65])
            fig.colorbar(im, cax=cbar_ax, label=cbar_label)

            for ext in ["png", "pdf"]:
                fig.savefig(args.output_dir / f"heatmap_1fact_{suffix_cond}_{suffix}.{ext}",
                            dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  Saved heatmap ({suffix})")

    print("\nDone.")


if __name__ == "__main__":
    main()
