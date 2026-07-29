"""
pool_decode_topk.py

Top-K numeric recovery via the standard pooled pipeline, for both the 2-fact
addition task and the chained variable-binding (varbind) task, with an optional
read-out that masks each example's externally visible values.

The pipeline is the one in pool_decode_global_dedup.py, unchanged:
1. Decode at every in-filler (position, layer) via logit lens (RMSNorm → lm_head →
   argmax over number tokens).
2. Filter: layer ≥ --min-layer, diversity ≥ --min-diversity.
3. Pairwise exact-match agreement between all surviving candidates.
4. Rank by mean global agreement.
5. Greedy dedup at --dedup-threshold, keep at most --n-kept.
6. Pool softmax probabilities over the kept settings.
7. Report top-K recovery per target.

Steps 1-6 never see the labels, so setting selection is identical across both
read-outs; only step 7 differs:

  default   top-K over all single-token integers in [0, --max-val)
  excluded  the same pooled distribution with each example's externally visible
            values zeroed before ranking — for varbind the base literal x (written
            in the prompt) and the answer (the model's own output), for 2fact the
            answer. Those two sit at the top of the numeric distribution nearly
            everywhere, so masking them is what reveals which hidden intermediate
            is next in line.

Usage:
    python scripts/decode/pooled/pool_decode_topk.py --task varbind \
        --condition dots_10 \
        --extraction-dir data/extracted_states_varbind_allpos_kimi_k25 \
        --dataset data/chained_var_binding_easy_dataset.json \
        --lm-head data/model_weights/kimi_k25/lm_head_weight.npy \
        --rms-norm-path data/model_weights/kimi_k25/rms_norm_weight.npy \
        --model-path /workspace/models/kimi-k2.5-configonly \
        --max-val 1000 --eps 1e-5 \
        --output results/kimi_k25_pooled/pool_varbind_dots_10.json
"""

import argparse
import json
import os
import pickle
import re
import sys
from pathlib import Path

_NCPU = str(os.cpu_count() or 1)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, _NCPU)

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from extract.extract_hidden_states import load_tokenizer  # noqa: E402

COEF_TO_NUM = {"twice": 2, "three times": 3, "four times": 4,
               "five times": 5, "six times": 6}
_CHAIN_RE = re.compile(
    r"^(twice|three times|four times|five times|six times) "
    r"the number for (\w+) (plus|minus) (\d+)")

# (label, externally_visible) in computation order.
TARGETS = {
    "varbind": [("x", True), ("c1*x", False), ("y", False), ("c2*y", False),
                ("answer", True)],
    "2fact": [("A1", False), ("A2", False), ("answer", True)],
}


def rms_norm(x, weight, eps=1e-6):
    rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
    return (x / rms) * weight


def parse_queried_chain(example):
    """(B, c1*B) for the queried term's chain; (None, None) if it won't parse."""
    defs = {name: val for name, val in example["definitions"]}
    expr = defs.get(example["queried_term"])
    if not isinstance(expr, str):
        return None, None
    m = _CHAIN_RE.match(expr)
    if not m:
        return None, None
    ref = defs.get(m.group(2))
    if not isinstance(ref, int):
        return None, None
    return ref, COEF_TO_NUM[m.group(1)] * ref


def truth_matrix(task, data, dataset):
    """(n_examples, n_targets) of target values; -1 where unrecoverable."""
    T = np.full((len(data), len(TARGETS[task])), -1, dtype=np.int64)
    for i, d in enumerate(data):
        if task == "varbind":
            base, chain = parse_queried_chain(dataset[d["problem_idx"]])
            vals = [base, chain, d["queried_value"],
                    d["coefficient"] * d["queried_value"], d["answer"]]
        else:
            vals = [d["fact_value_1"], d["fact_value_2"], d["answer"]]
        for j, v in enumerate(vals):
            if v is not None:
                T[i, j] = int(v)
    return T


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["2fact", "varbind"], required=True)
    p.add_argument("--condition", type=str, default="dots_10")
    p.add_argument("--extraction-dir", type=Path, required=True)
    p.add_argument("--dataset", type=Path, default=None,
                   help="varbind only — source JSON, to recover the base literal x.")
    p.add_argument("--lm-head", type=Path, required=True)
    p.add_argument("--rms-norm-path", type=Path, required=True)
    p.add_argument("--model-path", type=str, required=True)
    p.add_argument("--min-layer", type=int, default=30)
    p.add_argument("--min-diversity", type=float, default=0.10)
    p.add_argument("--dedup-threshold", type=float, default=0.30)
    p.add_argument("--n-kept", type=int, default=10)
    p.add_argument("--max-val", type=int, default=300)
    p.add_argument("--eps", type=float, default=1e-6, help="RMSNorm epsilon")
    p.add_argument("--include-post-filler", action="store_true")
    p.add_argument("--top-k", type=int, nargs="+", default=[1, 2, 3, 5, 10])
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args()

    tokenizer = load_tokenizer(args.model_path)
    lm_head_full = np.load(args.lm_head).astype(np.float32)
    norm_weight = np.load(args.rms_norm_path).astype(np.float32)

    number_tokens = {}
    for val in range(args.max_val):
        ids = tokenizer.encode(str(val), add_special_tokens=False)
        if len(ids) == 1:
            number_tokens[ids[0]] = val
    num_ids = sorted(number_tokens)
    num_vals = np.array([number_tokens[t] for t in num_ids])
    col_of_val = {int(v): j for j, v in enumerate(num_vals)}
    lm_head = lm_head_full[num_ids]
    del lm_head_full
    print(f"Number tokens: {len(num_ids)} (0..{args.max_val - 1})")

    cond_dir = args.extraction_dir / args.condition
    files = sorted(cond_dir.glob("prob_*.pkl"))
    all_data = []
    for f in tqdm(files, desc="Loading"):
        with open(f, "rb") as fp:
            d = pickle.load(fp)
        if d.get("model_correct", False):
            all_data.append(d)
    n = len(all_data)
    print(f"{n} correct examples")

    dataset = None
    if args.task == "varbind":
        dataset = {e["idx"]: e for e in json.load(open(args.dataset))["examples"]}
    T = truth_matrix(args.task, all_data, dataset)
    labels = [t[0] for t in TARGETS[args.task]]
    visible = [j for j, t in enumerate(TARGETS[args.task]) if t[1]]
    print(f"targets: {labels}   masked when excluded: {[labels[j] for j in visible]}")

    d0 = all_data[0]
    filler_end = d0["boundaries"]["filler_end_offset"]
    all_positions = sorted([p for p in d0["states"] if p.startswith("pos_")],
                           key=lambda s: int(s.split("_")[1]))

    # Steps 1+2: decode everywhere in the filler, filter by diversity.
    candidates = []
    for pos in tqdm(all_positions, desc="Decoding"):
        if not args.include_post_filler and int(pos.split("_")[1]) > filler_end:
            continue
        for layer in [l for l in sorted(d0["states"][pos]) if l >= args.min_layer]:
            vecs = np.stack([d["states"][pos][layer].astype(np.float32) for d in all_data])
            H = rms_norm(vecs, norm_weight, eps=args.eps)
            preds = num_vals[np.argmax(H @ lm_head.T, axis=1)]
            diversity = len(np.unique(preds)) / n
            if diversity < args.min_diversity:
                continue
            candidates.append({"position": pos, "layer": layer,
                               "predictions": preds, "diversity": diversity})

    print(f"{len(candidates)} candidates pass filters")
    if not candidates:
        print("No candidates — exiting")
        return

    # Steps 3+4: pairwise agreement, rank by mean global agreement.
    preds_mat = np.stack([c["predictions"] for c in candidates])
    agree = np.zeros((len(candidates), len(candidates)))
    for i in range(len(candidates)):
        agree[i] = np.mean(preds_mat[i:i + 1] == preds_mat, axis=1)
    np.fill_diagonal(agree, 0)
    mean_agree = agree.mean(axis=1)
    ranked = np.argsort(mean_agree)[::-1]

    # Step 5: greedy dedup.
    kept = []
    for idx in ranked:
        if any(agree[idx, k] >= args.dedup_threshold for k in kept):
            continue
        kept.append(idx)
        if len(kept) >= args.n_kept:
            break

    print(f"\nKept {len(kept)} settings (dedup threshold {args.dedup_threshold:.0%}):")
    print(f"  {'#':>3}  {'Position':>10}  {'Layer':>5}  {'MeanAgree':>9}  {'Diversity':>9}")
    for rank, k in enumerate(kept):
        c = candidates[k]
        print(f"  {rank+1:>3}  {c['position']:>10}  {c['layer']:>5}  "
              f"{mean_agree[k]:>8.1%}  {c['diversity']:>8.1%}")

    # Step 6: pool softmax probabilities over the kept settings.
    pooled = np.zeros((n, len(num_ids)), dtype=np.float64)
    for k in kept:
        c = candidates[k]
        vecs = np.stack([d["states"][c["position"]][c["layer"]].astype(np.float32)
                         for d in all_data])
        H = rms_norm(vecs, norm_weight, eps=args.eps)
        num_logits = H @ lm_head.T
        shifted = num_logits - num_logits.max(axis=1, keepdims=True)
        e = np.exp(shifted)
        pooled += e / e.sum(axis=1, keepdims=True)

    # Step 7: top-K recovery, both read-outs.
    results = {}
    for regime in ("default", "excluded"):
        pool = pooled
        if regime == "excluded":
            pool = pooled.copy()
            for i in range(n):
                for j in visible:
                    c = col_of_val.get(int(T[i, j]))
                    if c is not None:
                        pool[i, c] = -1.0
        top_preds = num_vals[np.argsort(pool, axis=1)[:, ::-1]]

        shown = [j for j in range(len(labels)) if not (regime == "excluded" and j in visible)]
        # Joint recovery over the hidden targets — the "Both" column of
        # pool_decode_global_dedup.py, generalized: every hidden value present in
        # the same example's top-K simultaneously.
        hidden = [j for j in range(len(labels)) if j not in visible]
        joint_name = " & ".join(labels[j] for j in hidden)
        print(f"\nPooled top-K recovery — {regime} ({len(kept)} settings, n={n}):")
        print("     K  " + "  ".join(f"{labels[j]:>7}" for j in shown)
              + f"  | {joint_name:>16}")
        results[regime] = {}
        for K in args.top_k:
            row = {}
            for j in shown:
                hit = [T[i, j] >= 0 and T[i, j] in top_preds[i, :K] for i in range(n)]
                row[labels[j]] = float(np.mean(hit))
            row[joint_name] = float(np.mean([
                all(T[i, j] >= 0 and T[i, j] in top_preds[i, :K] for j in hidden)
                for i in range(n)]))
            results[regime][f"top_{K}"] = row
            print(f"  {K:>4}  " + "  ".join(f"{row[labels[j]]:>6.1%}" for j in shown)
                  + f"  | {row[joint_name]:>15.1%}")

    out = args.output or Path(f"results/pool_decode_{args.task}_{args.condition}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({
        "task": args.task,
        "condition": args.condition,
        "extraction_dir": str(args.extraction_dir),
        "n_examples": n,
        "targets": labels,
        "masked_when_excluded": [labels[j] for j in visible],
        "kept_settings": [
            {"position": candidates[k]["position"], "layer": int(candidates[k]["layer"]),
             "mean_agreement": float(mean_agree[k]),
             "diversity": float(candidates[k]["diversity"])}
            for k in kept
        ],
        "topk_recovery": results,
        "config": {"min_layer": args.min_layer, "min_diversity": args.min_diversity,
                   "dedup_threshold": args.dedup_threshold, "n_kept": args.n_kept,
                   "max_val": args.max_val, "eps": args.eps,
                   "include_post_filler": args.include_post_filler},
    }, open(out, "w"), indent=2)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
