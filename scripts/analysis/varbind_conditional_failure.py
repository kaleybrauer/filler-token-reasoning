"""Per-example conditional failure analysis for the varbind decode.

The aggregate heatmaps show that on model-WRONG examples the base B and queried
value V still decode well while c·V and the answer collapse — suggesting failures
are downstream of the binding. But that's an aggregate: it can't tell whether a
given wrong answer happened *because the bound value V was wrong* or *because the
arithmetic on a correct V diverged*. This script conditions per example.

For each example we ask, for each stage, "did the TRUE value form?" — operationalized
as: the logit-lens argmax (RMSNorm -> lm_head, over single-token integers) equals
the true value at that stage's canonical layer (taken from the correct-example
aggregate peak), at ANY token position from question_end..answer_prompt. Then:

  - first-divergence stage: the earliest stage in [B, V, c·V, answer] whose true
    value did NOT form. "all-formed" = every stage incl. the answer formed (the
    true answer was represented internally yet the model emitted something else).
  - conditional chain: P(V|B), P(c·V|V), P(answer|c·V).
  - the headline: P(V formed AND answer did NOT) — "binding succeeded, arithmetic
    failed" — vs the binding-failure mass.

Run on correct examples too (as a reference: every stage should form).

Usage:
    python scripts/analysis/varbind_conditional_failure.py --condition dots_25
    python scripts/analysis/varbind_conditional_failure.py \
        --condition dots_5 dots_10 dots_25 dots_50 counting_5 counting_10 counting_25 --pool
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from decode_varbind_heatmap import (  # noqa: E402
    load_tokenizer_lite, rms_norm, pos_sort_key, parse_base_value, load_pkls,
)

STAGES = ["B", "V", "cV", "ans"]          # in computation order
STAGE_NAME = {"B": "base (visible)", "V": "queried value",
              "cV": "coef·queried", "ans": "answer"}


def peak_layers_from_correct(output_dir, suffix_cond):
    """Read each stage's canonical layer = layer of the correct-example exact-match
    global peak, from decode_varbind_<cond>.json."""
    jpath = output_dir / f"decode_varbind_{suffix_cond}.json"
    r = json.load(open(jpath))
    positions, layers = r["_positions"], r["_layers"]
    metric = {"B": "frac_B_exact", "V": "frac_QV_exact",
              "cV": "frac_QC_exact", "ans": "frac_ANS_exact"}
    out = {}
    for s, m in metric.items():
        best = (-1.0, None)
        for p in positions:
            for l in map(str, layers):
                v = r[p].get(l, {}).get(m)
                if v is not None and v > best[0]:
                    best = (v, int(l))
        out[s] = best[1]
    return out


def stage_formed(all_data, positions, peak_layer, truth, lm_head_num, norm_weight,
                 num_ids, num_vals, hidden):
    """For one stage: decode every example at `peak_layer` across all positions;
    return (formed_anywhere, formed_at_answer_prompt) boolean arrays of len n.
    truth is a float array (np.nan entries are treated as 'unknown' -> False)."""
    n = len(all_data)
    ans_idx = positions.index("answer_prompt") if "answer_prompt" in positions else len(positions) - 1
    # Gather (n, n_pos, hidden) at the single peak layer.
    H = np.empty((n, len(positions), hidden), dtype=np.float32)
    for r, d in enumerate(all_data):
        st = d["states"]
        for pj, p in enumerate(positions):
            H[r, pj] = st[p][peak_layer]
    Hn = rms_norm(H, norm_weight)
    preds = num_vals[np.argmax(Hn.reshape(n * len(positions), hidden) @ lm_head_num.T,
                               axis=1)].reshape(n, len(positions))
    t = truth[:, None]
    match = (preds == t) & ~np.isnan(t)
    return match.any(axis=1), match[:, ans_idx]


def analyse(all_data, positions, peaks, base_by_idx, lm_head_num, norm_weight,
            num_ids, num_vals, hidden):
    n = len(all_data)
    truth = {
        "B": np.array([base_by_idx.get(d["problem_idx"]) if base_by_idx.get(d["problem_idx"]) is not None
                       else np.nan for d in all_data], dtype=float),
        "V": np.array([d["queried_value"] for d in all_data], dtype=float),
        "cV": np.array([d["coefficient"] * d["queried_value"] for d in all_data], dtype=float),
        "ans": np.array([d["answer"] for d in all_data], dtype=float),
    }
    formed, formed_ans = {}, {}
    for s in STAGES:
        fa, fap = stage_formed(all_data, positions, peaks[s], truth[s],
                               lm_head_num, norm_weight, num_ids, num_vals, hidden)
        formed[s] = fa
        formed_ans[s] = fap
    return formed, formed_ans, n


def report(name, formed, n):
    F = {s: formed[s] for s in STAGES}
    print(f"\n--- {name}  (n={n}) ---")
    print("  formed-anywhere %:  " + "  ".join(f"{s}={F[s].mean()*100:5.1f}" for s in STAGES))
    # Conditional chain P(stage_i formed | stage_{i-1} formed)
    chain = []
    for i in range(1, len(STAGES)):
        prev, cur = STAGES[i - 1], STAGES[i]
        denom = F[prev].sum()
        p = (F[prev] & F[cur]).sum() / denom if denom else float("nan")
        chain.append(f"P({cur}|{prev})={p*100:5.1f}")
    print("  conditional:        " + "  ".join(chain))
    # First-divergence distribution
    cats = {s: 0 for s in STAGES}
    cats["all-formed"] = 0
    for i in range(n):
        first = next((s for s in STAGES if not F[s][i]), None)
        cats[first if first else "all-formed"] += 1
    print("  first-divergence:   " + "  ".join(
        f"{k}={cats[k]/n*100:5.1f}%" for k in STAGES + ["all-formed"]))
    # Headlines
    vf = F["V"]; af = F["ans"]
    binding_ok_arith_fail = (vf & ~af).sum() / n * 100
    print(f"  >> V formed but answer did NOT (binding ok, arithmetic failed): {binding_ok_arith_fail:5.1f}%")
    print(f"  >> answer formed internally (would be 'right inside, wrong out' on wrong set): {af.mean()*100:5.1f}%")
    return F


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", nargs="+",
                    default=["dots_5", "dots_10", "dots_25", "dots_50",
                             "counting_5", "counting_10", "counting_25"])
    ap.add_argument("--extraction-dir", type=Path,
                    default=Path("data/extracted_states_varbind_allpos"))
    ap.add_argument("--decode-dir", type=Path, default=Path("results/varbind_decode_heatmap"),
                    help="Where the correct-example decode JSONs live (for canonical layers).")
    ap.add_argument("--dataset", type=Path, default=Path("data/chained_var_binding_dataset.json"))
    ap.add_argument("--lm-head", type=Path,
                    default=Path("data/model_weights/deepseek_v3/lm_head_weight.npy"))
    ap.add_argument("--rms-norm", type=Path,
                    default=Path("data/model_weights/deepseek_v3/rms_norm_weight.npy"))
    ap.add_argument("--model-path", type=str, default="/workspace/models/deepseek-v3-awq")
    ap.add_argument("--max-num-token", type=int, default=1000)
    ap.add_argument("--also-correct", action="store_true",
                    help="Also run the analysis on the correct subset as a reference.")
    ap.add_argument("--pool", action="store_true", help="Pool all conditions into one report.")
    args = ap.parse_args()

    tok = load_tokenizer_lite(args.model_path)
    number_tokens = {}
    for val in range(args.max_num_token):
        ids = tok.encode(str(val), add_special_tokens=False)
        if len(ids) == 1:
            number_tokens[ids[0]] = val
    num_ids = sorted(number_tokens.keys())
    num_vals = np.array([number_tokens[t] for t in num_ids])
    lm_head = np.load(args.lm_head).astype(np.float32)
    norm_weight = np.load(args.rms_norm).astype(np.float32)
    lm_head_num = lm_head[num_ids]
    hidden = lm_head_num.shape[1]
    print(f"number tokens: {len(num_ids)}, lm_head_num {lm_head_num.shape}")

    dataset = json.load(open(args.dataset))
    base_by_idx = {e["idx"]: parse_base_value(e) for e in dataset["examples"]}

    pool = {"wrong": {s: [] for s in STAGES}, "correct": {s: [] for s in STAGES}}
    pool_n = {"wrong": 0, "correct": 0}

    for cond in args.condition:
        peaks = peak_layers_from_correct(args.decode_dir, cond)
        print(f"\n===== {cond}  canonical layers: "
              + ", ".join(f"{s}=L{peaks[s]}" for s in STAGES) + " =====")
        files = sorted((args.extraction_dir / cond).glob("prob_*.pkl"))

        for subset, incorrect_only in ([("wrong", True)] +
                                       ([("correct", False)] if args.also_correct else [])):
            all_data, _ = load_pkls(files, incorrect_only)
            if not all_data:
                continue
            positions = sorted(
                [p for p in all_data[0]["states"]
                 if p.startswith("pos_") or p == "answer_prompt"
                 or p == "question_end" or p == "pre_filler" or p.startswith("filler_k")],
                key=pos_sort_key)
            formed, _, n = analyse(all_data, positions, peaks, base_by_idx,
                                   lm_head_num, norm_weight, num_ids, num_vals, hidden)
            F = report(f"{cond} [{subset}]", formed, n)
            for s in STAGES:
                pool[subset][s].append(F[s])
            pool_n[subset] += n

    if args.pool:
        for subset in ("wrong", "correct"):
            if pool_n[subset] == 0:
                continue
            merged = {s: np.concatenate(pool[subset][s]) for s in STAGES}
            report(f"POOLED {subset} ({len(args.condition)} filler conditions)",
                   merged, pool_n[subset])

    print("\nDone.")


if __name__ == "__main__":
    main()
