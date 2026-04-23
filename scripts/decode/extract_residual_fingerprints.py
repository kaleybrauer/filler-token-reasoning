"""
extract_residual_fingerprints.py

For each candidate (layer, position) setting in a condition's extraction dir,
compute residual top-K fingerprints per correct example:

    residual_e = P_e(token) − mean_e [P_e(token)]

where P_e is the full-vocab softmax obtained via logit lens at that
(layer, position) on example e. The residual suppresses tokens that
are high-probability everywhere (e.g., 'Filler', 'Answer', dot tokens)
and highlights example-specific signal (element subwords, operand digits).

Output: an .npz with per-setting, per-example residual top-K (token ids
and values), ready to feed into a pairwise-fingerprint-agreement step.

Usage:
    python scripts/decode/extract_residual_fingerprints.py \
        --extraction-dir data/kimi-k2/extracted_states_2fact_allpos_kimi_k2/dots_10 \
        --model-path /workspace/models/kimi-k2-w4a16/ \
        --lm-head data/kimi-k2/model_weights/kimi_k2/lm_head_weight.npy \
        --rms-norm data/kimi-k2/model_weights/kimi_k2/rms_norm_weight.npy \
        --output /tmp/residual_fingerprints_dots_10.npz
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from extract_hidden_states import load_tokenizer  # noqa: E402


def rms_norm(x, w, eps=1e-6):
    rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
    return (x / rms) * w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extraction-dir", type=Path, required=True,
                    help="Directory containing prob_*.pkl for one condition")
    ap.add_argument("--model-path", type=str, required=True,
                    help="Only used to resolve tokenizer (for preview string decode)")
    ap.add_argument("--lm-head", type=Path, required=True)
    ap.add_argument("--rms-norm", type=Path, required=True)
    ap.add_argument("--min-layer", type=int, default=30)
    ap.add_argument("--include-question-end", action="store_true",
                    help="Also include the question_end position (last token of the question)")
    ap.add_argument("--include-post-filler", action="store_true",
                    help="Also include positions after filler_end (the 'Answer:' prompt region)")
    ap.add_argument("--top-k", type=int, default=30,
                    help="Fingerprint size per (setting, example)")
    ap.add_argument("--max-examples", type=int, default=None,
                    help="Cap number of correct examples (for fast iteration)")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--preview", type=int, default=0,
                    help="Decode and print top-K of a few settings/examples")
    args = ap.parse_args()

    tokenizer = load_tokenizer(args.model_path)
    lm_head = np.load(args.lm_head).astype(np.float32)   # (vocab, d)
    norm_w = np.load(args.rms_norm).astype(np.float32)
    vocab = lm_head.shape[0]

    files = sorted(args.extraction_dir.glob("prob_*.pkl"))
    all_data = []
    for f in tqdm(files, desc="Loading"):
        with open(f, "rb") as fp:
            d = pickle.load(fp)
        if d.get("model_correct", False):
            all_data.append(d)
        if args.max_examples and len(all_data) >= args.max_examples:
            break
    n = len(all_data)
    print(f"{n} correct examples")
    if n == 0:
        print("No correct examples — exiting")
        return

    # Determine positions & layers from first example.
    # Default range: (question_end, filler_end] — every token after the question
    # ends, up to and including the last filler content token.
    d0 = all_data[0]
    bnd = d0.get("boundaries", {}) or {}
    q_end = bnd.get("question_end_offset")
    filler_end = bnd.get("filler_end_offset")
    all_positions = sorted(
        [p for p in d0["states"] if p.startswith("pos_")],
        key=lambda p: int(p.split("_")[1]),
    )
    if q_end is not None and not args.include_question_end:
        all_positions = [p for p in all_positions
                         if int(p.split("_")[1]) > q_end]
    if filler_end is not None and not args.include_post_filler:
        all_positions = [p for p in all_positions
                         if int(p.split("_")[1]) <= filler_end]
    lo = int(all_positions[0].split("_")[1]) if all_positions else None
    hi = int(all_positions[-1].split("_")[1]) if all_positions else None
    print(f"Using positions [pos_{lo:03d} .. pos_{hi:03d}] "
          f"(question_end={q_end}, filler_end={filler_end}, "
          f"{'including' if args.include_question_end else 'excluding'} question_end, "
          f"{'including' if args.include_post_filler else 'excluding'} post-filler)")
    layers = sorted(l for l in d0["states"][all_positions[0]].keys()
                    if l >= args.min_layer)
    settings = [(pos, l) for pos in all_positions for l in layers]
    print(f"{len(all_positions)} positions × {len(layers)} layers "
          f"= {len(settings)} candidate settings")

    # Preallocate fingerprint arrays: (n_settings, n_examples, top_k)
    fp_ids = np.zeros((len(settings), n, args.top_k), dtype=np.int32)
    fp_vals = np.zeros((len(settings), n, args.top_k), dtype=np.float32)
    mean_topk_per_setting = []  # optional: top-20 tokens of the mean, for debug

    # Stash problem ids and truth fields (handy downstream).
    # Use object arrays for values that may be strings (e.g. letterpos `answer`,
    # `element`) or mixed types across conditions.
    example_ids = np.array([d.get("problem_idx", i) for i, d in enumerate(all_data)],
                           dtype=np.int32)
    def _get(field, default=None):
        return np.array([d.get(field, default) for d in all_data], dtype=object)
    truth = {
        "fact_value_1": _get("fact_value_1", -1),
        "fact_value_2": _get("fact_value_2", -1),
        "fact_value":   _get("fact_value", -1),
        "answer":       _get("answer", -1),
        "element":      _get("element"),
        "atomic_number": _get("atomic_number", -1),
    }

    for s_idx, (pos, layer) in enumerate(tqdm(settings, desc="Extracting fingerprints")):
        # Stack example vectors → (n, d)
        try:
            vecs = np.stack([d["states"][pos][layer].astype(np.float32)
                             for d in all_data])
        except KeyError:
            # Some examples may be missing this (pos, layer); skip
            continue

        H = rms_norm(vecs, norm_w)                    # (n, d)
        logits = H @ lm_head.T                        # (n, vocab)
        shifted = logits - logits.max(axis=1, keepdims=True)
        probs = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
        # Cross-example mean (noise baseline)
        mean_p = probs.mean(axis=0)                   # (vocab,)
        residual = probs - mean_p[None, :]            # (n, vocab)

        # Top-K by residual per example
        order = np.argpartition(-residual, args.top_k - 1, axis=1)[:, :args.top_k]
        # Sort those top-K descending by value
        rows = np.arange(n)[:, None]
        sort_in_topk = np.argsort(-residual[rows, order], axis=1)
        topk_ids = order[rows, sort_in_topk]
        topk_vals = residual[rows, topk_ids]

        fp_ids[s_idx] = topk_ids
        fp_vals[s_idx] = topk_vals

        # Debug: record top-20 of mean distribution per setting
        mean_top = np.argsort(-mean_p)[:20]
        mean_topk_per_setting.append({
            "position": pos, "layer": int(layer),
            "mean_top": [{"id": int(t), "prob": float(mean_p[t]),
                          "str": tokenizer.decode([int(t)])}
                         for t in mean_top]
        })

    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        fingerprint_ids=fp_ids,
        fingerprint_vals=fp_vals,
        settings=np.array(settings, dtype=object),
        example_ids=example_ids,
        truth_fact_value_1=truth["fact_value_1"],
        truth_fact_value_2=truth["fact_value_2"],
        truth_fact_value=truth["fact_value"],
        truth_answer=truth["answer"],
        truth_element=truth["element"],
        truth_atomic_number=truth["atomic_number"],
        config=np.array([{"min_layer": args.min_layer,
                          "include_question_end": args.include_question_end,
                          "include_post_filler": args.include_post_filler,
                          "top_k": args.top_k,
                          "extraction_dir": str(args.extraction_dir)}], dtype=object),
    )
    with open(args.output.with_suffix(".mean_topk.json"), "w") as f:
        json.dump(mean_topk_per_setting, f, indent=2, ensure_ascii=False)
    print(f"Saved fingerprints → {args.output}")
    print(f"Saved mean-dist debug → {args.output.with_suffix('.mean_topk.json')}")

    # Optional preview
    if args.preview > 0:
        print(f"\n=== Preview: first {args.preview} settings × first 2 examples ===")
        for s_idx, (pos, layer) in enumerate(settings[:args.preview]):
            print(f"\n[{pos}, L{layer}]")
            print(f"  mean-dist top-5: "
                  f"{[t['str'] for t in mean_topk_per_setting[s_idx]['mean_top'][:5]]}")
            for e_idx in range(min(2, n)):
                ex = all_data[e_idx]
                ids = fp_ids[s_idx, e_idx]
                vals = fp_vals[s_idx, e_idx]
                # Task-agnostic label
                if "element" in ex:
                    label = f"element={ex['element']!r}, Z={ex.get('atomic_number')}, ans={ex.get('answer')!r}"
                elif "fact_value_1" in ex and "fact_value_2" in ex:
                    label = f"A1={ex['fact_value_1']}, A2={ex['fact_value_2']}"
                else:
                    label = f"fact_value={ex.get('fact_value')}"
                print(f"  ex {ex.get('problem_idx')} ({label}):")
                for rank in range(min(10, args.top_k)):
                    tok_str = tokenizer.decode([int(ids[rank])])
                    print(f"    {rank+1:>2}. {vals[rank]:8.4f}  {tok_str!r}")


if __name__ == "__main__":
    main()
