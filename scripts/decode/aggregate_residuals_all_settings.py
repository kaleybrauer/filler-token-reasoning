"""
aggregate_residuals_all_settings.py

Aggregate per-example residual fingerprints across all (layer, position) settings.
For each example, sum the residual values across every setting's top-K
fingerprint, then output the globally top-K tokens per example.

No setting-selection step — just pool everything.

Input:  an .npz from extract_residual_fingerprints.py
Output: JSON with one record per example, matching the schema of
        pool_top_tokens.py (top_tokens: [{id, str, prob}...]) so existing
        llm_decode_pool.py can consume it directly.

Usage:
    python scripts/decode/aggregate_residuals_all_settings.py \
        --fingerprints /tmp/residual_fingerprints_kimi_dots10.npz \
        --model-path /workspace/models/kimi-k2-w4a16/ \
        --output /tmp/aggregated_all_settings_kimi_dots10.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from extract_hidden_states import load_tokenizer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fingerprints", type=Path, required=True,
                    help="npz from extract_residual_fingerprints.py")
    ap.add_argument("--model-path", type=str, required=True,
                    help="For tokenizer — used to decode token IDs to strings")
    ap.add_argument("--top-k-out", type=int, default=50,
                    help="Number of globally top-ranked tokens to keep per example")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    tokenizer = load_tokenizer(args.model_path)

    data = np.load(args.fingerprints, allow_pickle=True)
    fp_ids = data["fingerprint_ids"]    # (S, N, K)
    fp_vals = data["fingerprint_vals"]  # (S, N, K)
    example_ids = data["example_ids"]
    n_s, n_ex, K = fp_ids.shape
    print(f"{n_s} settings, {n_ex} examples, K={K}")

    # Aggregate: for each example, sum residuals across all (setting, rank)
    # slots into a vocab-sized array, then take top-K-out. Vectorized.
    vocab_size = int(fp_ids.max()) + 1
    out = []
    for e in tqdm(range(n_ex), desc="Aggregating"):
        ids_flat = fp_ids[:, e, :].astype(np.int64).ravel()
        vals_flat = fp_vals[:, e, :].astype(np.float64).ravel()

        # Sum residuals per token
        scores = np.zeros(vocab_size, dtype=np.float64)
        np.add.at(scores, ids_flat, vals_flat)

        # Count settings per token (binary presence — a token appearing twice
        # in one setting's top-K still counts as 1 setting, which is fine since
        # top-K per setting has no dupes).
        counts = np.zeros(vocab_size, dtype=np.int32)
        np.add.at(counts, ids_flat, 1)

        top_ids = np.argsort(-scores)[:args.top_k_out]

        record = {
            "idx": int(example_ids[e]),
            "top_tokens": [
                {"id": int(tid),
                 "str": tokenizer.decode([int(tid)]),
                 "prob": float(scores[tid]),       # aggregated residual score
                 "n_settings": int(counts[tid])}
                for tid in top_ids
            ],
        }

        # Propagate ground-truth fields so llm_decode_pool.py can score
        for field in ["truth_fact_value_1", "truth_fact_value_2", "truth_fact_value",
                      "truth_answer", "truth_element", "truth_atomic_number"]:
            if field in data.files:
                arr = data[field]
                val = arr[e]
                # Convert numpy object types to native python
                if hasattr(val, "item"):
                    try: val = val.item()
                    except Exception: pass
                elif isinstance(val, np.ndarray) and val.ndim == 0:
                    val = val.tolist()
                else:
                    val = val
                key = field.replace("truth_", "")
                if key == "fact_value_1": record["fact_value_1"] = val
                elif key == "fact_value_2": record["fact_value_2"] = val
                elif key == "fact_value": record["fact_value"] = val
                elif key == "answer":
                    # may be string (letterpos) or int; keep as-is
                    record["answer"] = val
                elif key == "element": record["element"] = val
                elif key == "atomic_number": record["atomic_number"] = val
        out.append(record)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(out)} aggregated samples to {args.output}")

    # Preview: first 3 examples
    print("\n=== Preview (first 3 examples, top-10) ===")
    for ex in out[:3]:
        label_parts = []
        for k in ["fact_value_1", "fact_value_2", "element", "atomic_number", "answer"]:
            if k in ex and ex[k] is not None:
                label_parts.append(f"{k}={ex[k]}")
        print(f"\n  ex{ex['idx']} ({', '.join(label_parts)}):")
        for rk, t in enumerate(ex["top_tokens"][:10], 1):
            print(f"    {rk:>2}. score={t['prob']:.3f}  n_settings={t['n_settings']:>3}  "
                  f"{t['str']!r}")


if __name__ == "__main__":
    main()
