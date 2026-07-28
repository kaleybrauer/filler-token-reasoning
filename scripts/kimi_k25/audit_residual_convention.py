"""
audit_residual_convention.py

Decide, from saved pkls alone, whether an extraction captured the *residual stream*
or only each layer's *MLP write*.

Why this exists: the HF `modeling_deepseek.py` decoder layer adds the residual inside
the layer, so a forward hook's `output[0]` IS the residual stream. vLLM's
`DeepseekV2DecoderLayer.forward` uses the split convention and returns
`(hidden_states, residual)` where `hidden_states` is the MLP write and the residual
stream after layer L is `output[0] + output[1]` (verified in vLLM v0.8.5 and main).
`scripts/kimi_k2/` and `scripts/qwen3/` hook vLLM layers but take `output[0]`, copying
the comment from the transformers path — so their states may be layer writes.

Two independent diagnostics, both run per invocation:

1. GEOMETRY (primary, tokenizer-free, no weights needed). The residual stream is
   cumulative: consecutive layers differ by one layer's write, so cos(h_L, h_L+1) is
   high (typically >0.9) and ||h_L|| grows with depth. Independent per-layer MLP
   writes have no such continuity — cosine hovers near 0 and the norm profile is flat
   or erratic.

2. LOGIT LENS (secondary, needs --lm-head/--rms-norm). At the final layer the
   residual stream, after RMSNorm and lm_head, argmaxes to the token the model
   actually generated. A layer write does not reliably do so.

Run it on a known-good extraction first (the DeepSeek-V3 states came from the
transformers path) to see what "residual stream" looks like on these diagnostics, then
on the extraction under test.

Usage:
    python scripts/kimi_k25/audit_residual_convention.py \
        --states-dir data/extracted_states_varbind_allpos/dots_10 --n 20
    python scripts/kimi_k25/audit_residual_convention.py \
        --states-dir data/kimi-k2/extracted_states_2fact_allpos_kimi_k2/dots_10 --n 20 \
        --lm-head data/model_weights/kimi_k2/lm_head_weight.npy \
        --rms-norm data/model_weights/kimi_k2/rms_norm_weight.npy \
        --tokenizer-path /workspace/models/kimi-k2-w4a16
"""
from __future__ import annotations

import argparse
import base64
import pickle
from pathlib import Path

import numpy as np


# ------------------------------------------------------------------------------
# tokenizer (only needed for the optional logit-lens check)
# ------------------------------------------------------------------------------

def load_encoder(tokenizer_path: Path):
    """Return encode(str) -> list[int], or None if no tokenizer is available.

    Tries a HuggingFace fast tokenizer first; falls back to parsing a raw
    `tiktoken.model` BPE rank file (Kimi), which is enough for the numeric strings
    this audit encodes.
    """
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(str(tokenizer_path), trust_remote_code=True)
        return lambda s: tok.encode(s, add_special_tokens=False)
    except Exception as exc:  # noqa: BLE001 - fall back, this is a diagnostic
        print(f"  [tokenizer] AutoTokenizer failed ({type(exc).__name__}), trying tiktoken.model")

    rank_file = tokenizer_path / "tiktoken.model"
    if not rank_file.exists():
        print("  [tokenizer] no tiktoken.model either - skipping logit-lens check")
        return None

    ranks: dict[bytes, int] = {}
    with rank_file.open("rb") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            piece, rank = line.split(b" ")
            ranks[base64.b64decode(piece)] = int(rank)

    def encode(s: str) -> list[int]:
        """Greedy longest-prefix-match encode. Adequate for short numeric strings."""
        data = s.encode()
        out: list[int] = []
        i = 0
        while i < len(data):
            for j in range(len(data), i, -1):
                if data[i:j] in ranks:
                    out.append(ranks[data[i:j]])
                    i = j
                    break
            else:
                return out  # unencodable byte; caller only needs the prefix
        return out

    return encode


# ------------------------------------------------------------------------------
# diagnostics
# ------------------------------------------------------------------------------

def rms_norm(vec: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
    x = vec.astype(np.float32)
    return x / np.sqrt((x * x).mean() + eps) * weight.astype(np.float32)


def geometry(pkls: list[Path], position: str | None) -> dict:
    """Per-layer norms and consecutive-layer cosines, averaged over examples."""
    cos_acc: list[np.ndarray] = []
    norm_acc: list[np.ndarray] = []

    for path in pkls:
        with path.open("rb") as f:
            rec = pickle.load(f)
        states = rec["states"]
        pos_name = position or sorted(states.keys())[-1]
        if pos_name not in states:
            raise SystemExit(f"position {pos_name!r} not in {path} (have {sorted(states)[:3]}...)")
        layers = sorted(states[pos_name].keys())
        mat = np.stack([np.asarray(states[pos_name][li], dtype=np.float32) for li in layers])

        norms = np.linalg.norm(mat, axis=1)
        upper, lower = mat[1:], mat[:-1]
        denom = np.linalg.norm(upper, axis=1) * np.linalg.norm(lower, axis=1)
        cos = (upper * lower).sum(axis=1) / np.maximum(denom, 1e-9)

        norm_acc.append(norms)
        cos_acc.append(cos)

    return {
        "layers": layers,
        "norm": np.mean(norm_acc, axis=0),
        "cos": np.mean(cos_acc, axis=0),
        "position": pos_name,
        "n": len(pkls),
    }


def logit_lens(pkls: list[Path], position: str | None, lm_head: np.ndarray,
               norm_w: np.ndarray, eps: float, encode) -> dict:
    """Final-layer argmax vs the token the model actually generated."""
    hits = 0
    ranks: list[int] = []
    checked = 0
    examples: list[tuple[str, int, int]] = []

    for path in pkls:
        with path.open("rb") as f:
            rec = pickle.load(f)
        response = rec.get("model_response")
        if not response:
            continue
        states = rec["states"]
        pos_name = position or sorted(states.keys())[-1]
        last_layer = max(states[pos_name].keys())
        vec = np.asarray(states[pos_name][last_layer])

        logits = rms_norm(vec, norm_w, eps) @ lm_head.astype(np.float32).T
        pred = int(np.argmax(logits))

        true_ids = encode(str(response).strip()) if encode else []
        if not true_ids:
            continue
        true_id = true_ids[0]
        rank = int((logits > logits[true_id]).sum()) + 1

        checked += 1
        hits += int(pred == true_id)
        ranks.append(rank)
        if len(examples) < 5:
            examples.append((str(response), pred, true_id))

    return {"checked": checked, "hits": hits, "ranks": ranks, "examples": examples}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--states-dir", required=True, type=Path)
    ap.add_argument("--n", type=int, default=20, help="examples to average over")
    ap.add_argument("--position", default=None,
                    help="position key (default: the last one, i.e. answer_prompt)")
    ap.add_argument("--lm-head", type=Path, default=None)
    ap.add_argument("--rms-norm", type=Path, default=None)
    ap.add_argument("--tokenizer-path", type=Path, default=None)
    ap.add_argument("--eps", type=float, default=1e-6, help="RMSNorm epsilon")
    args = ap.parse_args()

    pkls = sorted(args.states_dir.glob("prob_*.pkl"))[: args.n]
    if not pkls:
        raise SystemExit(f"no prob_*.pkl under {args.states_dir}")

    print(f"=== {args.states_dir} ({len(pkls)} examples) ===")

    geo = geometry(pkls, args.position)
    layers, cos, norm = geo["layers"], geo["cos"], geo["norm"]
    print(f"position: {geo['position']}   layers: {layers[0]}..{layers[-1]}")
    print("\nconsecutive-layer cosine (mean over examples):")
    for lo in range(0, len(cos), max(1, len(cos) // 12)):
        print(f"  L{layers[lo]:>2}->L{layers[lo + 1]:>2}  cos={cos[lo]:+.3f}   "
              f"||h_L||={norm[lo]:8.1f}")
    print(f"\n  median cosine over all layer pairs : {np.median(cos):+.3f}")
    print(f"  fraction of pairs with cos > 0.9    : {(cos > 0.9).mean():.2f}")
    print(f"  norm ratio ||h_last|| / ||h_first|| : {norm[-1] / max(norm[0], 1e-9):.2f}")
    verdict = ("RESIDUAL STREAM (cumulative)" if np.median(cos) > 0.8
               else "LAYER WRITES (not cumulative)" if np.median(cos) < 0.5
               else "AMBIGUOUS - inspect the profile")
    print(f"  geometry verdict                    : {verdict}")

    if args.lm_head and args.rms_norm:
        print("\nlogit-lens check (final layer -> generated token):")
        lm_head = np.load(args.lm_head)
        norm_w = np.load(args.rms_norm)
        encode = load_encoder(args.tokenizer_path) if args.tokenizer_path else None
        res = logit_lens(pkls, args.position, lm_head, norm_w, args.eps, encode)
        if not res["checked"]:
            print("  no examples had both a model_response and an encodable first token")
        else:
            ranks = np.array(res["ranks"])
            print(f"  argmax == generated token : {res['hits']}/{res['checked']} "
                  f"({100 * res['hits'] / res['checked']:.0f}%)")
            print(f"  median rank of that token : {int(np.median(ranks))}")
            for resp, pred, true_id in res["examples"]:
                print(f"    response={resp!r:>8}  argmax_id={pred:>7}  true_id={true_id:>7}"
                      f"  {'MATCH' if pred == true_id else ''}")


if __name__ == "__main__":
    main()
