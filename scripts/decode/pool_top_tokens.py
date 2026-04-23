"""
pool_top_tokens.py

Load the kept (layer, position) settings from a pool_decode_global_dedup run,
pool full-vocab softmax probabilities across them per example, and save the
top-K tokens per sampled example. Supports filtering to numeric tokens only
(useful for controlling "is the LLM decoding cheating by reading element
names off the tokens?").

Usage:
    python scripts/decode/pool_top_tokens.py \
        --pool-result results/pool_decode_global_dots_10_kimi.json \
        --extraction-dir data/kimi-k2/extracted_states_2fact_allpos_kimi_k2/dots_10 \
        --model-path /workspace/models/kimi-k2-w4a16/ \
        --lm-head data/kimi-k2/model_weights/kimi_k2/lm_head_weight.npy \
        --rms-norm data/kimi-k2/model_weights/kimi_k2/rms_norm_weight.npy \
        --n-sample 20 --top-k 50 \
        --output /tmp/pooled_top50.json

Add --numeric-only to restrict to integer-string tokens in [0, --max-val].
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

# Repo-local import
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from extract_hidden_states import load_tokenizer  # noqa: E402


def rms_norm(x, w, eps=1e-6):
    rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
    return (x / rms) * w


def build_numeric_mask(tokenizer, vocab_size, max_val):
    """Return (token_ids, values) for tokens that decode to integer 0..max_val."""
    ids, vals = [], []
    for tid in range(vocab_size):
        s = tokenizer.decode([tid]).strip()
        if not s:
            continue
        try:
            v = int(s)
        except ValueError:
            continue
        if 0 <= v <= max_val and str(v) == s:
            ids.append(tid)
            vals.append(v)
    return np.array(ids), np.array(vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool-result", type=Path, required=True,
                    help="JSON from pool_decode_global_dedup (provides kept_settings)")
    ap.add_argument("--extraction-dir", type=Path, required=True,
                    help="Directory containing prob_*.pkl for one condition")
    ap.add_argument("--model-path", type=str, required=True)
    ap.add_argument("--lm-head", type=Path, required=True)
    ap.add_argument("--rms-norm", type=Path, required=True)
    ap.add_argument("--n-sample", type=int, default=20,
                    help="Number of correct examples to sample (evenly spaced)")
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--numeric-only", action="store_true",
                    help="Restrict output tokens to integer-string tokens in [0, max-val]")
    ap.add_argument("--max-val", type=int, default=999)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    tokenizer = load_tokenizer(args.model_path)
    lm_head = np.load(args.lm_head).astype(np.float32)
    norm_w = np.load(args.rms_norm).astype(np.float32)

    with open(args.pool_result) as f:
        kept = json.load(f)["kept_settings"]
    print(f"{len(kept)} kept settings loaded from {args.pool_result}")

    numeric_ids, numeric_vals = None, None
    if args.numeric_only:
        numeric_ids, numeric_vals = build_numeric_mask(
            tokenizer, lm_head.shape[0], args.max_val)
        print(f"Numeric-token mask: {len(numeric_ids)} tokens in [0,{args.max_val}]")

    files = sorted(args.extraction_dir.glob("prob_*.pkl"))
    data = []
    for f in files:
        with open(f, "rb") as fp:
            d = pickle.load(fp)
        if d.get("model_correct", False):
            data.append(d)
    print(f"{len(data)} correct examples; sampling {args.n_sample}")

    idxs = np.linspace(0, len(data) - 1, args.n_sample, dtype=int)
    sample = [data[i] for i in idxs]

    out = []
    for ex in tqdm(sample, desc="Pooling"):
        pooled = None
        for s in kept:
            vec = ex["states"][s["position"]][s["layer"]].astype(np.float32)
            h = rms_norm(vec[None, :], norm_w)[0]
            logits = h @ lm_head.T
            shifted = logits - logits.max()
            probs = np.exp(shifted) / np.exp(shifted).sum()
            pooled = probs if pooled is None else pooled + probs

        if args.numeric_only:
            restricted = pooled[numeric_ids]
            order = np.argsort(restricted)[::-1][:args.top_k]
            top_entries = [
                {"value": int(numeric_vals[i]),
                 "prob": float(restricted[i]),
                 "token_str": tokenizer.decode([int(numeric_ids[i])])}
                for i in order
            ]
            key = "top_numeric"
        else:
            order = np.argsort(pooled)[::-1][:args.top_k]
            top_entries = [
                {"id": int(t),
                 "str": tokenizer.decode([int(t)]),
                 "prob": float(pooled[t])}
                for t in order
            ]
            key = "top_tokens"

        record = {
            "idx": int(ex["problem_idx"]),
            "model_correct": bool(ex.get("model_correct", False)),
            key: top_entries,
        }
        # Propagate all non-state metadata from the pickle
        for k in ("answer", "fact_value", "fact_value_1", "fact_value_2",
                  "x", "fact_phrase", "fact_phrase_1", "fact_phrase_2",
                  "model_response", "model_answer"):
            if k in ex:
                record[k] = ex[k]
        out.append(record)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(out)} samples to {args.output}")


if __name__ == "__main__":
    main()
