#!/usr/bin/env python3
"""
Generate test datasets with different prompt formats for Qwen2.5-Instruct.

Uses the proper ChatML chat template (<|im_start|>/<|im_end|>) so the
instruct model's instruction-following capabilities are activated.

All formats require the model to output ONLY the final integer sum — no
intermediate reasoning steps are exposed in the completion. This is
intentional: the model must learn to perform implicit multi-step reasoning
in the filler token space.

Prompt formats:
  minimal        - Bare-bones user message, no system prompt
  system_role    - System prompt defines the task; user provides questions
  system_strict  - System prompt heavily emphasizes integer-only output
  fewshot_chat   - Few-shot examples as prior chat turns
  paired         - Questions presented as labeled Value 1 / Value 2
  sum_of         - Natural language "What is the sum of …" phrasing
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
#
# Each format returns a list of message dicts (system/user/assistant roles)
# that will be rendered via tokenizer.apply_chat_template().
#
# The model is expected to complete the assistant turn with ONLY the integer.
# ─────────────────────────────────────────────────────────────────────────────

PROMPT_FORMATS = {
    "minimal": {
        "description": "Minimal user message, no system prompt",
        "messages": lambda q1, q2: [
            {
                "role": "user",
                "content": (
                    f"Q1: {q1}\n"
                    f"Q2: {q2}\n\n"
                    "Add the two answers. Reply with only the sum as a single integer."
                ),
            },
        ],
    },
    "system_role": {
        "description": "System prompt defines the task; user provides questions only",
        "messages": lambda q1, q2: [
            {
                "role": "system",
                "content": (
                    "You solve 2-hop addition problems. The user gives two factual "
                    "questions. Find the numerical answer to each question, then "
                    "return their sum as a single integer. Output nothing else."
                ),
            },
            {
                "role": "user",
                "content": f"{q1}\n{q2}",
            },
        ],
    },
    "system_strict": {
        "description": "System prompt heavily emphasizes integer-only output",
        "messages": lambda q1, q2: [
            {
                "role": "system",
                "content": (
                    "You are a precise arithmetic assistant. The user will ask two "
                    "factual questions whose answers are integers. Compute the sum "
                    "of the two answers. Your ENTIRE response must be exactly one "
                    "integer — no words, no punctuation, no explanation."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Q1: {q1}\n"
                    f"Q2: {q2}"
                ),
            },
        ],
    },
    "fewshot_chat": {
        "description": "Few-shot examples as prior user/assistant chat turns",
        "messages": lambda q1, q2: [
            {
                "role": "system",
                "content": (
                    "You solve 2-hop addition problems. The user gives two questions. "
                    "Find each answer and return their sum as a single integer."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Q1: What is the atomic number of Helium?\n"
                    "Q2: How many days are in a week?"
                ),
            },
            {"role": "assistant", "content": "9"},
            {
                "role": "user",
                "content": (
                    "Q1: What is the atomic number of Carbon?\n"
                    "Q2: What is the atomic number of Hydrogen?"
                ),
            },
            {"role": "assistant", "content": "7"},
            {
                "role": "user",
                "content": (
                    "Q1: How many legs does a spider have?\n"
                    "Q2: What is the atomic number of Lithium?"
                ),
            },
            {"role": "assistant", "content": "11"},
            {
                "role": "user",
                "content": f"Q1: {q1}\nQ2: {q2}",
            },
        ],
    },
    "paired": {
        "description": "Questions presented as labeled Value 1 / Value 2",
        "messages": lambda q1, q2: [
            {
                "role": "system",
                "content": (
                    "The user describes two values via factual questions. "
                    "Determine each value and respond with only their sum as "
                    "a single integer."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Value 1: {q1}\n"
                    f"Value 2: {q2}\n\n"
                    "Sum:"
                ),
            },
        ],
    },
    "sum_of": {
        "description": "Natural language 'What is the sum of ...' phrasing",
        "messages": lambda q1, q2: [
            {
                "role": "user",
                "content": (
                    f"What is the sum of the answer to \"{q1}\" "
                    f"and the answer to \"{q2}\"? "
                    "Reply with just the number."
                ),
            },
        ],
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

    messages_fn = PROMPT_FORMATS[prompt_format]["messages"]

    # Determine EOS from tokenizer
    eos_ids = [tok.eos_token_id] if tok.eos_token_id is not None else []

    rows = []
    used_pairs = set()

    # Create fact index mapping for efficient pair deduplication
    fact_to_idx = {fact: i for i, fact in enumerate(facts)}

    # Distribute examples across filler lengths, handling remainder
    examples_per_filler = n_examples // len(filler_lengths)
    remainder = n_examples % len(filler_lengths)

    for filler_idx, filler_len in enumerate(filler_lengths):
        # Distribute remainder evenly across first few filler lengths
        target_count = examples_per_filler + (1 if filler_idx < remainder else 0)
        generated_count = 0
        attempts = 0
        max_attempts = target_count * 100

        while generated_count < target_count and attempts < max_attempts:
            attempts += 1

            # Find a valid pair
            (q1, a1, t1) = rng.choice(facts)
            (q2, a2, t2) = rng.choice(facts)

            if q1 == q2:
                continue

            # Use fact indices for efficient pair key
            idx1 = fact_to_idx[(q1, a1, t1)]
            idx2 = fact_to_idx[(q2, a2, t2)]
            pair_key = (min(idx1, idx2), max(idx1, idx2))

            if pair_key in used_pairs:
                continue

            s = a1 + a2
            if s < 0 or s >= max_answer:
                continue

            used_pairs.add(pair_key)
            generated_count += 1

            # Build messages and apply chat template
            messages = messages_fn(q1, q2)

            # apply_chat_template with add_generation_prompt=True appends the
            # opening of the assistant turn: "<|im_start|>assistant\n"
            prompt_text = tok.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            prompt_ids = tok.encode(prompt_text, add_special_tokens=False)

            # Filler sequence
            filler_seq = [filler_id] * filler_len

            # Answer: just the integer, then EOS
            # No leading space needed — the chat template already ends with
            # the assistant turn opening, and the model is trained to start
            # generating directly after it.
            answer_text = str(s)
            answer_ids = tok.encode(answer_text, add_special_tokens=False) + eos_ids

            # Assemble full sequence
            prefix_ids = prompt_ids + filler_seq
            input_ids = prefix_ids + answer_ids

            labels = [-100] * len(prefix_ids) + answer_ids
            attn = [1] * len(input_ids)

            example_id = len(rows)
            rows.append({
                "id": f"{prompt_format}-{example_id}",
                "split": "test",
                "prompt_format": prompt_format,
                "prompt": prompt_text,
                "messages": messages,
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

        # Warn if we couldn't generate enough examples for this filler length
        if generated_count < target_count:
            print(f"Warning: Only generated {generated_count}/{target_count} "
                  f"examples for filler_len={filler_len}")
            print(f"  (Exhausted valid pairs after {attempts} attempts)")

    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Generate test datasets with different prompt formats (Instruct)"
    )
    parser.add_argument("--tokenizer", type=str, required=True,
                        help="Tokenizer name, e.g. Qwen/Qwen2.5-72B-Instruct")
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
    parser.add_argument("--formats", type=str,
                        default="minimal,system_role,system_strict,fewshot_chat,paired,sum_of",
                        help="Comma-separated list of prompt formats to generate")

    args = parser.parse_args()

    rng = random.Random(args.seed)
    filler_lengths = [int(x.strip()) for x in args.filler_lengths.split(",")]
    formats_to_generate = [f.strip() for f in args.formats.split(",")]

    # Validate formats upfront
    invalid_formats = [f for f in formats_to_generate if f not in PROMPT_FORMATS]
    if invalid_formats:
        valid_formats = ", ".join(PROMPT_FORMATS.keys())
        raise SystemExit(
            f"Error: Unknown formats: {invalid_formats}\n"
            f"Valid formats: {valid_formats}"
        )

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

    # Verify chat template works
    test_msgs = [{"role": "user", "content": "test"}]
    test_rendered = tok.apply_chat_template(test_msgs, tokenize=False,
                                            add_generation_prompt=True)
    print(f"\nChat template sanity check:")
    print(f"  Input:  [user: 'test']")
    print(f"  Output: {test_rendered!r}")
    assert "<|im_start|>" in test_rendered, (
        "Chat template does not produce expected ChatML tokens. "
        "Is this an Instruct tokenizer?"
    )

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
            "model_type": "instruct",
            "chat_template": "chatml",
        }
        (fmt_outdir / "manifest.json").write_text(json.dumps(manifest, indent=2))

        # Print example
        print(f"\nExample rendered prompt ({fmt}):")
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
  --model Qwen/Qwen2.5-72B-Instruct \\
  --load-in-4bit \\
  --data-dir {outdir / fmt} \\
  --filler-lengths {args.filler_lengths} \\
  --outdir ./results/prompt_format_test/{fmt} \\
  --report-every 50
""")


if __name__ == "__main__":
    main()