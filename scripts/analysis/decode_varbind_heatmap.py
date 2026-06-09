"""Decode heatmaps for the chained variable-binding task: which value is encoded
at each (layer, position)?

Sibling of decode_2fact_heatmap.py / decode_1fact_heatmap.py, adapted for the
varbind task. Unlike the addition tasks (which decode a *retrieved* fact), the
star target here is a value the model must *compute serially* and that is never
written anywhere in the prompt.

For a depth-1 problem the relevant values are:

    B   = base value      the literal the queried term is derived from
                          (VISIBLE in the prompt, e.g. "lef = 58") — positive
                          control / "is the relevant input operand held?"
    QV  = queried_value   the queried term's value = coef*B +/- const
                          (HIDDEN intermediate, never written) — the transient
                          serial-computation target, the scientific payoff.
    ANS = answer          the final answer = q_coef*QV +/- q_const
                          (the model's output) — "where does the answer form?"

For each (layer, position) we logit-lens via RMSNorm -> lm_head, take the argmax
over single-token integer strings 0..max_num_token, and score exact / +-5 match
against each of B, QV, ANS.

B is not stored in the extracted pkls (only queried_value / answer / the question
coef-op-const are). We recover it by joining each example's `problem_idx` back to
the source dataset and parsing the queried term's expression. This join is only
meaningful for chain_len=1 (the queried term refers directly to a literal); for
the depth-2 control B is left blank.

Usage:
    python scripts/analysis/decode_varbind_heatmap.py --condition dots_25
    python scripts/analysis/decode_varbind_heatmap.py \
        --condition baseline dots_5 dots_10 dots_25 dots_50 counting_5 counting_10 counting_25
"""

import argparse
import json
import os
import pickle
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Let BLAS use every core for the big logit-lens GEMMs (set before numpy import).
_NCPU = str(os.cpu_count() or 1)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, _NCPU)

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


def load_tokenizer_lite(model_path: str):
    """Load just the tokenizer, torch-free (the heatmap needs no model/torch).

    DeepSeek V3: PreTrainedTokenizerFast straight off tokenizer.json (mirrors
    extract_hidden_states.load_tokenizer, minus the torch import). Falls back to
    AutoTokenizer + trust_remote_code for anything without a tokenizer.json.
    """
    model_dir = Path(model_path)
    tok_file = model_dir / "tokenizer.json"
    cfg_file = model_dir / "tokenizer_config.json"
    if tok_file.exists() and cfg_file.exists():
        from transformers import PreTrainedTokenizerFast
        with open(cfg_file) as f:
            cfg = json.load(f)
        return PreTrainedTokenizerFast(
            tokenizer_file=str(tok_file),
            chat_template=cfg.get("chat_template"),
            bos_token=cfg.get("bos_token"),
            eos_token=cfg.get("eos_token"),
        )
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)


def parse_base_value(example):
    """Recover the base literal the queried term is derived from (chain_len=1).

    Returns the literal int, or None if the queried term's referent is itself an
    expression (chain_len>1) or the expression can't be parsed.
    """
    defs = {name: val for name, val in example["definitions"]}
    expr = defs.get(example["queried_term"])
    if not isinstance(expr, str):
        return None
    m = re.search(r"the number for (\w+)", expr)
    if not m:
        return None
    ref_val = defs.get(m.group(1))
    return ref_val if isinstance(ref_val, int) else None


def load_pkls(files, incorrect_only):
    """Load pkls (threaded — I/O bound on the network mount) and filter by
    model_correct."""
    def _read(f):
        with open(f, "rb") as fp:
            return pickle.load(fp)

    all_data = []
    n_total = 0
    with ThreadPoolExecutor(max_workers=8) as ex:
        for d in tqdm(ex.map(_read, files), total=len(files), desc="Loading"):
            n_total += 1
            correct = d.get("model_correct", False)
            if incorrect_only:
                if not correct:
                    all_data.append(d)
            elif correct:
                all_data.append(d)
    return all_data, n_total


def plot_condition(results, cond, suffix_cond, output_dir, incorrect_only):
    """Print the per-target summary and render the 4-panel heatmaps from a results
    dict (freshly computed, or loaded from a decode_varbind_<cond>.json for replot).

    Targets: B = base value (visible), V = queried term's value (hidden), c·V =
    question-coefficient times V (hidden), answer (final). Labels are generic — the
    per-example variable names are random CVC strings, so no single name applies.
    """
    positions = results["_positions"]
    layers = results["_layers"]
    boundaries = results.get("_boundaries")

    # Text summary (exact-match): best (layer, pos) per target, and best in filler.
    fe = boundaries.get("filler_end_offset") if boundaries else None
    fs = boundaries.get("filler_start_offset") if boundaries else None
    for tgt, name in [("frac_B_exact", "B base (visible)"),
                      ("frac_QV_exact", "V queried value (HIDDEN)"),
                      ("frac_QC_exact", "c·V coef×queried (HIDDEN)"),
                      ("frac_ANS_exact", "answer (final)")]:
        best = (-1.0, None, None)
        best_filler = (-1.0, None, None)
        for pos in positions:
            off = pos_sort_key(pos)
            for layer in layers:
                v = results[pos].get(str(layer), {}).get(tgt, float("nan"))
                if v != v:
                    continue
                if v > best[0]:
                    best = (v, layer, pos)
                if fs is not None and fe is not None and fs <= off <= fe and v > best_filler[0]:
                    best_filler = (v, layer, pos)
        msg = f"  best {name:26s} exact: {best[0]*100:5.1f}%  @ L{best[1]} {pos_label(best[2])}"
        if best_filler[1] is not None:
            msg += f"  | best in filler: {best_filler[0]*100:5.1f}% @ L{best_filler[1]} {pos_label(best_filler[2])}"
        print(msg)

    is_allpos = any(p.startswith("pos_") for p in positions)
    n_pos = len(positions)

    # Exact-match heatmaps only (no ±5).
    for metric_set, suffix, cbar_label in [
        ([("frac_B_exact", "B: base value (visible, exact)"),
          ("frac_QV_exact", "V: queried value (hidden, exact)"),
          ("frac_QC_exact", "c·V: coef × queried (hidden, exact)"),
          ("frac_ANS_exact", "answer (final, exact)")],
         "exact", "% exact match"),
    ]:
        n_panels = len(metric_set)
        fig_width = max(7 * n_panels, n_pos * 0.12 * n_panels) if is_allpos else 7 * n_panels
        fig, axes = plt.subplots(1, n_panels, figsize=(fig_width, 11))

        for ax, (metric, title) in zip(axes, metric_set):
            matrix = np.full((len(layers), n_pos), np.nan)
            for j, pos in enumerate(positions):
                for i, layer in enumerate(layers):
                    cell = results.get(pos, {}).get(str(layer))
                    if cell is not None and cell.get(metric) is not None:
                        matrix[i, j] = cell[metric] * 100

            im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn",
                           vmin=0, vmax=100, interpolation="nearest")

            if is_allpos:
                tick_step = max(1, n_pos // 20)
                tick_idxs = list(range(0, n_pos, tick_step))
                if (n_pos - 1) not in tick_idxs:
                    tick_idxs.append(n_pos - 1)
                ax.set_xticks(tick_idxs)
                ax.set_xticklabels([pos_label(positions[i]) for i in tick_idxs],
                                   rotation=45, ha="right")
                ax.set_xlabel("Offset from question_end")
                # The "Filler:" label tokens are themselves filler, so mark only
                # the filler -> Answer boundary (a single dashed line).
                if boundaries:
                    fe2 = boundaries.get("filler_end_offset")
                    if fe2 is not None:
                        ax.axvline(fe2 + 0.5, color="white", linewidth=1,
                                   linestyle="--", alpha=0.7)
            else:
                ax.set_xticks(range(n_pos))
                ax.set_xticklabels([pos_label(p) for p in positions],
                                   rotation=45, ha="right")

            layer_labels = [str(l) if l % 5 == 0 else "" for l in layers]
            ax.set_yticks(range(len(layers)))
            ax.set_yticklabels(layer_labels)
            ax.set_title(title, fontweight="bold", fontsize=18)
            if ax == axes[0]:
                ax.set_ylabel("Layer")

        title_suffix = " — model INCORRECT" if incorrect_only else ""
        fig.suptitle(f"varbind {cond}{title_suffix}: what is encoded at each (layer, position)?",
                     fontsize=22)
        fig.subplots_adjust(right=0.88, wspace=0.15, top=0.93)
        cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.65])
        fig.colorbar(im, cax=cbar_ax, label=cbar_label)

        for ext in ["png", "pdf"]:
            fig.savefig(output_dir / f"heatmap_varbind_{suffix_cond}_{suffix}.{ext}",
                        dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved heatmap ({suffix})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", nargs="+", default=["dots_25"])
    parser.add_argument("--extraction-dir", type=Path,
                        default=Path("data/extracted_states_varbind_allpos"))
    parser.add_argument("--dataset", type=Path,
                        default=Path("data/chained_var_binding_dataset.json"),
                        help="Source dataset, joined by problem_idx to recover the base value B.")
    parser.add_argument("--lm-head", type=Path,
                        default=Path("data/model_weights/deepseek_v3/lm_head_weight.npy"))
    parser.add_argument("--rms-norm", type=Path,
                        default=Path("data/model_weights/deepseek_v3/rms_norm_weight.npy"))
    parser.add_argument("--model-path", type=str, default="/workspace/models/deepseek-v3-awq")
    parser.add_argument("--output-dir", type=Path, default=Path("results/varbind_decode_heatmap"))
    parser.add_argument("--max-num-token", type=int, default=1000,
                        help="Highest integer string to include in the number-token map. "
                             "1000 covers answers up to 999 (single-token on DeepSeek; >999 "
                             "are multi-token and unreachable by single-token argmax).")
    parser.add_argument("--incorrect-only", action="store_true",
                        help="Filter to examples where the model got the final answer WRONG. "
                             "Default: filter to correct examples.")
    parser.add_argument("--replot", action="store_true",
                        help="Skip extraction/decoding; redraw heatmaps from the existing "
                             "decode_varbind_<cond>.json files in --output-dir (cosmetic-only, "
                             "no model weights loaded).")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.replot:
        for cond in args.condition:
            suffix_cond = f"{cond}_incorrect" if args.incorrect_only else cond
            jpath = args.output_dir / f"decode_varbind_{suffix_cond}.json"
            if not jpath.exists():
                print(f"=== {cond} ===  no JSON at {jpath}, skipping")
                continue
            with open(jpath) as f:
                results = json.load(f)
            print(f"\n=== {cond} (replot from {jpath.name}, n={results.get('_n')}) ===")
            plot_condition(results, cond, suffix_cond, args.output_dir, args.incorrect_only)
        print("\nDone.")
        return

    tokenizer = load_tokenizer_lite(args.model_path)

    # Single-token integer-string map (subset of vocab decoding to "0".."N-1").
    number_tokens = {}
    for val in range(args.max_num_token):
        ids = tokenizer.encode(str(val), add_special_tokens=False)
        if len(ids) == 1:
            number_tokens[ids[0]] = val
    num_ids = sorted(number_tokens.keys())
    num_vals = np.array([number_tokens[tid] for tid in num_ids])
    print(f"Single-token integers in 0..{args.max_num_token}: {len(number_tokens)}")

    lm_head = np.load(args.lm_head).astype(np.float32)
    norm_weight = np.load(args.rms_norm).astype(np.float32)
    # Restrict lm_head to the number-token rows once — argmax only ever happens
    # over these columns, so this is identical to (H @ lm_head.T)[:, num_ids] but
    # ~100x cheaper (1000 rows vs ~129k vocab).
    lm_head_num = lm_head[num_ids]
    print(f"lm_head: {lm_head.shape} -> number rows {lm_head_num.shape}")

    # problem_idx -> base value B (visible operand). Join to the source dataset.
    dataset = json.load(open(args.dataset))
    base_by_idx = {e["idx"]: parse_base_value(e) for e in dataset["examples"]}
    n_base = sum(v is not None for v in base_by_idx.values())
    print(f"Base value B parsed for {n_base}/{len(base_by_idx)} dataset examples "
          f"(chain_len={dataset.get('chain_len')})")

    for cond in args.condition:
        print(f"\n=== {cond} ===")
        files = sorted((args.extraction_dir / cond).glob("prob_*.pkl"))
        all_data, n_total = load_pkls(files, args.incorrect_only)
        label = "incorrect" if args.incorrect_only else "correct"
        print(f"  {len(all_data)}/{n_total} {label} examples")
        if not all_data:
            print("  no examples to decode, skipping")
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

        # Ground truth. B may be NaN for examples whose base didn't parse.
        # The question (q_coef * queried_value +/- q_const) passes through TWO
        # never-written values: the queried term's value itself (V) and the scaled
        # product q_coef*V (cV) computed before the final constant is applied.
        # Decoding both traces whether the model stages the arithmetic.
        QV = np.array([d["queried_value"] for d in all_data])
        QC = np.array([d["coefficient"] * d["queried_value"] for d in all_data])
        ANS = np.array([d["answer"] for d in all_data])
        B = np.array([base_by_idx.get(d["problem_idx"]) if base_by_idx.get(d["problem_idx"]) is not None
                      else np.nan for d in all_data], dtype=float)
        b_mask = ~np.isnan(B)

        results = {"_positions": positions, "_layers": layers, "_condition": cond,
                   "_n": len(all_data), "_n_base": int(b_mask.sum()),
                   "_target_legend": {"B": "base literal (visible operand)",
                                      "QV": "queried_value (hidden intermediate)",
                                      "QC": "q_coef * queried_value (hidden, pre-constant product)",
                                      "ANS": "answer (final)"}}
        if boundaries:
            results["_boundaries"] = boundaries

        def metrics(preds, target, mask=None):
            if mask is None:
                m = slice(None)
                n = len(preds)
            else:
                m = mask
                n = int(mask.sum())
            if n == 0:
                return {"exact": float("nan"), "within5": float("nan")}
            p, t = preds[m], target[m]
            return {"exact": float(np.mean(p == t)),
                    "within5": float(np.mean(np.abs(p - t) <= 5))}

        hidden = lm_head_num.shape[1]
        layer_strs = [str(l) for l in layers]
        for pos in tqdm(positions, desc="  positions"):
            valid_idx = [i for i, d in enumerate(all_data) if pos in d["states"]]
            if not valid_idx:
                continue
            # Gather this position's (n_examples, n_layers, hidden) block in a
            # single pass, then run ONE batched GEMM over all layers at once. This
            # collapses 61 small per-layer matmuls into one large one (BLAS
            # saturates the cores) and removes the per-layer re-stacking.
            H = np.empty((len(valid_idx), len(layers), hidden), dtype=np.float32)
            for r, i in enumerate(valid_idx):
                st = all_data[i]["states"][pos]
                for li, l in enumerate(layers):
                    H[r, li] = st[l]
            vi = np.array(valid_idx)
            Hn = rms_norm(H, norm_weight)
            n, L, _ = Hn.shape
            preds = num_vals[np.argmax(Hn.reshape(n * L, hidden) @ lm_head_num.T,
                                       axis=1)].reshape(n, L)

            qv_v, qc_v, ans_v, b_v, bm_v = QV[vi], QC[vi], ANS[vi], B[vi], b_mask[vi]
            results[pos] = {}
            for li, lstr in enumerate(layer_strs):
                p = preds[:, li]
                mQV, mQC, mANS, mB = (metrics(p, qv_v), metrics(p, qc_v),
                                      metrics(p, ans_v), metrics(p, b_v, bm_v))
                results[pos][lstr] = {
                    "frac_B_exact": mB["exact"], "frac_B_within5": mB["within5"],
                    "frac_QV_exact": mQV["exact"], "frac_QV_within5": mQV["within5"],
                    "frac_QC_exact": mQC["exact"], "frac_QC_within5": mQC["within5"],
                    "frac_ANS_exact": mANS["exact"], "frac_ANS_within5": mANS["within5"],
                    "frac_any_within5": float(np.mean(
                        (np.abs(p - qv_v) <= 5) | (np.abs(p - qc_v) <= 5)
                        | (np.abs(p - ans_v) <= 5)
                        | ((np.abs(p - b_v) <= 5) & bm_v)
                    )),
                }

        suffix_cond = f"{cond}_incorrect" if args.incorrect_only else cond
        outfile = args.output_dir / f"decode_varbind_{suffix_cond}.json"
        with open(outfile, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Saved {outfile}")

        plot_condition(results, cond, suffix_cond, args.output_dir, args.incorrect_only)

    print("\nDone.")


if __name__ == "__main__":
    main()
