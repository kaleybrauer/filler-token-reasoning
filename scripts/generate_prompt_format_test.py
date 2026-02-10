#!/usr/bin/env python3
"""
Generate test datasets with different prompt formats to find which works best
for baseline evaluation.

Creates 4 separate test datasets (300 examples each) with different prompts:
1. original: "Q1 + Q2" format (your original)
2. explicit: "Answer two questions and add" format  
3. stepwise: "First find each answer, then add" format
4. direct: "X died at age A, Y has atomic number B, what is A+B?" format

Run baseline eval on each to see which format the model understands best.
"""
import argparse
import json
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
    """Load facts from JSON file. Returns list of (question_text, answer_int, kind)."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    out: List[Tuple[str, int, str]] = []

    def make_q(key_or_q: str) -> str:
        if "?" in key_or_q:
            return key_or_q.strip()
        if kind == "age":
            return f"At what age did {key_or_q} die?"
        if kind == "atomic":
            return f"What is the atomic number of {key_or_q}?"
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
                for nk in ["name", "entity", "person", "element"]:
                    if nk in item and isinstance(item[nk], str):
                        q = make_q(item[nk])
                        break

            if q is None:
                continue

            out.append((q, int(ans), kind))

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Prompt Format Definitions
# ─────────────────────────────────────────────────────────────────────────────

PROMPT_FORMATS = {
    "original": {
        "description": "Original Q1 + Q2 format",
        "template": lambda q1, q2: (
            "Answer with a single integer (no extra text).\n\n"
            f"{q1} + {q2}\n"
            "Answer:"
        ),
    },
    "explicit": {
        "description": "Explicit 'answer two questions and add' format",
        "template": lambda q1, q2: (
            "Answer two questions and add the results.\n\n"
            f"Q1: {q1}\n"
            f"Q2: {q2}\n\n"
            "Q1 answer + Q2 answer = ?\n"
            "Answer:"
        ),
    },
    "stepwise": {
        "description": "Step-by-step hint format",
        "template": lambda q1, q2: (
            f"{q1}\n"
            f"{q2}\n\n"
            "Find the answer to each question above, then add them together.\n"
            "Answer with only the final sum as a single integer:"
        ),
    },
    "direct": {
        "description": "Direct sum request format",
        "template": lambda q1, q2: (
            f"What is the sum of the following two values?\n"
            f"Value 1: {q1}\n"
            f"Value 2: {q2}\n"
            "Sum:"
        ),
    },
}


def write_jsonl(rows: List[Dict[str, Any]], outpath: pathlib.Path) -> None:
    outpath.parent.mkdir(parents=True, exist_ok=True)
    with outpath.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def generate_examples(
    facts: List[Tuple[str, int, str]],
    tok: AutoTokenizer,
    filler_id: int,
    filler_token: str,
    prompt_format: str,
    n_examples: int,
    filler_lengths: List[int],
    max_answer: int,
    rng: random.Random,
    seed: int,
) -> List[Dict[str, Any]]:
    """Generate examples for a specific prompt format."""
    
    template_fn = PROMPT_FORMATS[prompt_format]["template"]
    
    # Get BOS/EOS tokens
    bos_ids = [tok.bos_token_id] if tok.bos_token_id else []
    eos_ids = [tok.eos_token_id] if tok.eos_token_id else []
    
    rows = []
    used_pairs = set()
    
    # We want equal distribution across filler lengths
    examples_per_filler = n_examples // len(filler_lengths)
    
    for filler_len in filler_lengths:
        for i in range(examples_per_filler):
            # Find a valid pair
            for _ in range(1000):
                (q1, a1, t1) = rng.choice(facts)
                (q2, a2, t2) = rng.choice(facts)
                
                if q1 == q2:
                    continue
                
                pair_key = (min(q1, q2), max(q1, q2))
                if pair_key in used_pairs:
                    continue
                
                s = a1 + a2
                if s < 0 or s >= max_answer:
                    continue
                
                used_pairs.add(pair_key)
                break
            else:
                print(f"Warning: Could not find unique pair after 1000 tries")
                continue
            
            # Build prompt using the template
            prompt = template_fn(q1, q2)
            prompt_ids = tok.encode(prompt, add_special_tokens=False)
            
            # Filler sequence
            filler_seq = [filler_id] * filler_len
            
            # Answer with leading space, terminated by EOS
            answer_text = " " + str(s)
            answer_ids = tok.encode(answer_text, add_special_tokens=False) + eos_ids
            
            # Assemble full sequence
            prefix_ids = bos_ids + prompt_ids + filler_seq
            input_ids = prefix_ids + answer_ids
            
            labels = [-100] * len(prefix_ids) + answer_ids
            attn = [1] * len(input_ids)
            
            example_id = len(rows)
            rows.append({
                "id": f"{prompt_format}-{example_id}",
                "split": "test",
                "prompt_format": prompt_format,
                "prompt": prompt,
                "fact1": q1,
                "fact2": q2,
                "a1": a1,
                "a2": a2,
                "answer": s,
                "type1": t1,
                "type2": t2,
                "filler_len": filler_len,
                "filler_token": filler_token,
                "filler_token_id": filler_id,
                "prompt_ids": prompt_ids,
                "answer_ids": answer_ids,
                "input_ids": input_ids,
                "labels": labels,
                "attention_mask": attn,
            })
    
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Generate test datasets with different prompt formats"
    )
    parser.add_argument("--tokenizer", type=str, required=True,
                        help="Tokenizer name, e.g. Qwen/Qwen2.5-7B")
    parser.add_argument("--sources", type=str, required=True,
                        help="Directory containing fact JSON files")
    parser.add_argument("--outdir", type=str, required=True,
                        help="Output directory")
    parser.add_argument("--n-examples", type=int, default=300,
                        help="Number of examples per prompt format")
    parser.add_argument("--filler-token", type=str, default="<|fim_pad|>",
                        help="Filler token string")
    parser.add_argument("--filler-lengths", type=str, default="0,128,600",
                        help="Comma-separated filler lengths to test")
    parser.add_argument("--max-answer", type=int, default=1000,
                        help="Maximum answer value")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--formats", type=str, default="original,explicit,stepwise,direct",
                        help="Comma-separated list of prompt formats to generate")
    
    args = parser.parse_args()
    
    rng = random.Random(args.seed)
    filler_lengths = [int(x.strip()) for x in args.filler_lengths.split(",")]
    formats_to_generate = [f.strip() for f in args.formats.split(",")]
    
    print(f"Filler lengths: {filler_lengths}")
    print(f"Formats to generate: {formats_to_generate}")
    
    # Load tokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    filler_ids = tok.encode(args.filler_token, add_special_tokens=False)
    if len(filler_ids) != 1:
        raise SystemExit(f"Filler token {args.filler_token!r} doesn't map to single token")
    filler_id = filler_ids[0]
    
    print(f"Filler token: {args.filler_token!r} -> id {filler_id}")
    print(f"BOS: {tok.bos_token!r} (id={tok.bos_token_id})")
    print(f"EOS: {tok.eos_token!r} (id={tok.eos_token_id})")
    
    # Load facts
    srcdir = pathlib.Path(args.sources)
    age = load_facts(srcdir / "age_facts.json", "age") if (srcdir / "age_facts.json").exists() else []
    atomic = load_facts(srcdir / "atomic_facts.json", "atomic") if (srcdir / "atomic_facts.json").exists() else []
    static = load_facts(srcdir / "static_facts.json", "static") if (srcdir / "static_facts.json").exists() else []
    
    all_facts = [(q, a, k) for (q, a, k) in (age + atomic + static) if 0 <= a < args.max_answer]
    print(f"\nLoaded {len(all_facts)} usable facts")
    
    # Create output directory
    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    # Generate dataset for each prompt format
    for fmt in formats_to_generate:
        if fmt not in PROMPT_FORMATS:
            print(f"Warning: Unknown format '{fmt}', skipping")
            continue
        
        print(f"\n{'='*60}")
        print(f"Generating: {fmt}")
        print(f"Description: {PROMPT_FORMATS[fmt]['description']}")
        print(f"{'='*60}")
        
        # Use a fresh RNG for each format to ensure same fact pairs
        fmt_rng = random.Random(args.seed)
        
        rows = generate_examples(
            facts=all_facts,
            tok=tok,
            filler_id=filler_id,
            filler_token=args.filler_token,
            prompt_format=fmt,
            n_examples=args.n_examples,
            filler_lengths=filler_lengths,
            max_answer=args.max_answer,
            rng=fmt_rng,
            seed=args.seed,
        )
        
        # Create format-specific output directory
        fmt_outdir = outdir / fmt
        fmt_outdir.mkdir(parents=True, exist_ok=True)
        
        # Write test.jsonl
        write_jsonl(rows, fmt_outdir / "test.jsonl")
        
        # Write manifest
        manifest = {
            "prompt_format": fmt,
            "description": PROMPT_FORMATS[fmt]["description"],
            "tokenizer": args.tokenizer,
            "seed": args.seed,
            "n_examples": len(rows),
            "filler_token": args.filler_token,
            "filler_token_id": filler_id,
            "filler_lengths": filler_lengths,
            "max_answer": args.max_answer,
            "bos_token_id": tok.bos_token_id,
            "eos_token_id": tok.eos_token_id,
        }
        (fmt_outdir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        
        # Print example
        print(f"\nExample prompt ({fmt}):")
        print("-" * 40)
        print(rows[0]["prompt"])
        print("-" * 40)
        print(f"Answer: {rows[0]['answer']} (= {rows[0]['a1']} + {rows[0]['a2']})")
        
        # Distribution
        from collections import Counter
        filler_dist = Counter(r["filler_len"] for r in rows)
        print(f"\nFiller length distribution:")
        for fl in sorted(filler_dist.keys()):
            print(f"  N={fl}: {filler_dist[fl]} examples")
        
        print(f"\nWrote {len(rows)} examples to {fmt_outdir / 'test.jsonl'}")
    
    # Print evaluation commands
    print(f"\n{'='*60}")
    print("EVALUATION COMMANDS")
    print(f"{'='*60}")
    print("\nRun baseline eval on each format:")
    for fmt in formats_to_generate:
        if fmt in PROMPT_FORMATS:
            print(f"""
# {fmt}: {PROMPT_FORMATS[fmt]['description']}
python scripts/evaluate.py \\
  --model Qwen/Qwen2.5-72B \\
  --load-in-4bit \\
  --data-dir {outdir / fmt} \\
  --filler-lengths {args.filler_lengths} \\
  --outdir ./results/prompt_format_test/{fmt} \\
  --report-every 50
""")


if __name__ == "__main__":
    main()