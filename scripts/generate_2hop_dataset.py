#!/usr/bin/env python3
import argparse
import json
import math
import os
import pathlib
import random
from typing import Any, Dict, List, Optional, Tuple

from transformers import AutoTokenizer

INT_KEYS = ["answer", "value", "number", "n", "age", "atomic_number"]

def _is_int(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)

def _extract_int_from_obj(obj: Any) -> Optional[int]:
    if _is_int(obj):
        return obj
    if isinstance(obj, dict):
        for k in INT_KEYS:
            if k in obj and _is_int(obj[k]):
                return obj[k]
    return None

def load_facts(path: pathlib.Path, kind: str) -> List[Tuple[str, int, str]]:
    """
    Returns list of (question_text, answer_int, kind).
    Handles a few common JSON shapes:
      - dict[str, int]           (entity -> int OR question -> int)
      - dict[str, dict]          (entity -> {age: int} etc.)
      - list[dict]               ({question: str, answer: int} etc.)
    """
    raw = json.loads(path.read_text(encoding="utf-8"))

    out: List[Tuple[str, int, str]] = []

    def make_q(key_or_q: str) -> str:
        # If it already looks like a question, keep it.
        if "?" in key_or_q:
            return key_or_q.strip()
        if kind == "age":
            return f"At what age did {key_or_q} die?"
        if kind == "atomic":
            return f"What is the atomic number of {key_or_q}?"
        # static: treat key as question if not obviously an entity
        return key_or_q.strip()

    if isinstance(raw, dict):
        for k, v in raw.items():
            ans = _extract_int_from_obj(v)
            if ans is None:
                continue
            q = make_q(k)
            out.append((q, int(ans), kind))

    elif isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            ans = _extract_int_from_obj(item)
            if ans is None:
                continue

            q = None
            for qk in ["question", "q", "prompt", "text"]:
                if qk in item and isinstance(item[qk], str):
                    q = item[qk].strip()
                    break

            if q is None:
                # maybe it's entity-style
                for nk in ["name", "entity", "person", "element"]:
                    if nk in item and isinstance(item[nk], str):
                        q = make_q(item[nk])
                        break

            if q is None:
                continue

            out.append((q, int(ans), kind))

    else:
        raise ValueError(f"Unsupported JSON root type in {path}: {type(raw)}")

    return out

def sample_filler_len(rng: random.Random, lo: int, hi: int) -> int:
    if hi < lo:
        hi = lo
    return rng.randint(lo, hi)

def write_jsonl(rows: List[Dict[str, Any]], outpath: pathlib.Path) -> None:
    outpath.parent.mkdir(parents=True, exist_ok=True)
    with outpath.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", type=str, required=True, help="Tokenizer name, e.g. Qwen/Qwen2.5-7B")
    ap.add_argument("--sources", type=str, required=True, help="Directory containing age_facts.json, atomic_facts.json, static_facts.json")
    ap.add_argument("--outdir", type=str, required=True, help="Output directory for generated JSONL files")

    ap.add_argument("--n-train", type=int, default=50000)
    ap.add_argument("--n-val", type=int, default=2000)
    ap.add_argument("--n-test", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--max-answer", type=int, default=1000, help="Keep final answers < max-answer")

    ap.add_argument("--filler-token", type=str, default="<|fim_pad|>", help="Single-token filler string")
    ap.add_argument("--filler-min", type=int, default=0)
    ap.add_argument("--filler-max", type=int, default=1000)

    ap.add_argument("--max-tries", type=int, default=2000, help="Resample limit when enforcing constraints")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    tok = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    filler_ids = tok.encode(args.filler_token, add_special_tokens=False)
    if len(filler_ids) != 1:
        raise SystemExit(
            f"filler-token {args.filler_token!r} does not map to exactly 1 token (got {filler_ids}). "
            "Pick another filler token."
        )
    filler_id = filler_ids[0]

    srcdir = pathlib.Path(args.sources)
    age_path = srcdir / "age_facts.json"
    atomic_path = srcdir / "atomic_facts.json"
    static_path = srcdir / "static_facts.json"

    age = load_facts(age_path, "age") if age_path.exists() else []
    atomic = load_facts(atomic_path, "atomic") if atomic_path.exists() else []
    static = load_facts(static_path, "static") if static_path.exists() else []

    facts = [(q, a, k) for (q, a, k) in (age + atomic + static) if 0 <= a < args.max_answer]
    if len(facts) < 100:
        raise SystemExit(f"Too few usable facts ({len(facts)}). Check your source JSON files in {srcdir}.")

    print(f"Loaded facts: age={len(age)} atomic={len(atomic)} static={len(static)} usable_total={len(facts)}")
    print(f"Using filler_token_id={filler_id} for filler_token={args.filler_token!r}")

    def make_example(example_id: int, split: str) -> Dict[str, Any]:
        # Resample until sum < max_answer (and different facts).
        for _ in range(args.max_tries):
            (q1, a1, t1) = rng.choice(facts)
            (q2, a2, t2) = rng.choice(facts)
            if q1 == q2:
                continue
            s = a1 + a2
            if s < 0 or s >= args.max_answer:
                continue

            prompt = (
                "Answer with a single integer (no extra text).\n\n"
                f"{q1} + {q2}\n"
                "Answer:"
            )

            # Add BOS at start, EOS at end
            bos_ids = [tok.bos_token_id] if tok.bos_token_id else []
            eos_ids = [tok.eos_token_id] if tok.eos_token_id else []

            prompt_ids = tok.encode(prompt, add_special_tokens=False)
            # Choose filler length per-example for training.
            nfill = sample_filler_len(rng, args.filler_min, args.filler_max)
            filler_seq = [filler_id] * nfill

            answer_text = " " + str(s)  # Remove \n, use EOS instead
            answer_ids = tok.encode(answer_text, add_special_tokens=False) + eos_ids

            prefix_ids = bos_ids + prompt_ids + filler_seq
            input_ids = prefix_ids + answer_ids
            labels = [-100] * len(prefix_ids) + answer_ids
            attn = [1] * len(input_ids)

            return {
                "id": f"{split}-{example_id}",
                "split": split,
                "prompt": prompt,
                "fact1": q1,
                "fact2": q2,
                "a1": a1,
                "a2": a2,
                "answer": s,
                "type1": t1,
                "type2": t2,
                "filler_len": nfill,
                "filler_token": args.filler_token,
                "filler_token_id": filler_id,
                "prompt_ids": prompt_ids,
                "answer_ids": answer_ids,
                "input_ids": input_ids,
                "labels": labels,
                "attention_mask": attn,
            }

        raise RuntimeError("Failed to sample a valid example; try increasing --max-tries or relaxing constraints.")

    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    def build_split(n: int, split: str) -> List[Dict[str, Any]]:
        rows = []
        for i in range(n):
            rows.append(make_example(i, split))
        return rows

    train_rows = build_split(args.n_train, "train")
    val_rows = build_split(args.n_val, "val")
    test_rows = build_split(args.n_test, "test")

    write_jsonl(train_rows, outdir / "train.jsonl")
    write_jsonl(val_rows, outdir / "val.jsonl")
    write_jsonl(test_rows, outdir / "test.jsonl")

    manifest = {
        "tokenizer": args.tokenizer,
        "seed": args.seed,
        "max_answer": args.max_answer,
        "filler_token": args.filler_token,
        "filler_token_id": filler_id,
        "filler_min": args.filler_min,
        "filler_max": args.filler_max,
        "counts": {"train": len(train_rows), "val": len(val_rows), "test": len(test_rows)},
    }
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Done. Wrote dataset to: {outdir}")
    print(f" - train: {outdir / 'train.jsonl'}")
    print(f" - val:   {outdir / 'val.jsonl'}")
    print(f" - test:  {outdir / 'test.jsonl'}")

if __name__ == "__main__":
    main()
