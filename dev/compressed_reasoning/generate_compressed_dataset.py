#!/usr/bin/env python3
"""
Generate training data for compressed reasoning curriculum.

Stages:
  0: Thinking: a1, a2, sum\nAnswer: sum         (fully readable)
  1: Thinking: a1, a2, . . .\nAnswer: sum       (sum replaced with dots)
  2: Thinking: . . . . . . . . . .\nAnswer: sum  (all values replaced with dots)
  3: . . . . . . . . . .\nAnswer: sum            (no Thinking: prefix, just dots)

All stages use the same questions and answers. The system prompt few-shot
examples match the stage format so the model sees consistent formatting.

Usage:
    python scripts/generate_compressed_dataset.py \
        --known-facts data/known_facts_14b.json \
        --tokenizer Qwen/Qwen2.5-14B-Instruct \
        --stage 0 \
        --n-think-tokens 10 \
        --outdir data/datasets/compressed_s0

    # Generate all stages at once:
    for s in 0 1 2 3; do
        python scripts/generate_compressed_dataset.py \
            --known-facts data/known_facts_14b.json \
            --tokenizer Qwen/Qwen2.5-14B-Instruct \
            --stage $s \
            --n-think-tokens 10 \
            --outdir data/datasets/compressed_s${s}
    done
"""
import argparse
import json
import pathlib
import random
from typing import Any, Dict, List, Optional, Tuple

from transformers import AutoTokenizer


# ─────────────────────────────────────────────────────────────────────────────
# Fact loading and splitting (reused from generate_addition_dataset.py)
# ─────────────────────────────────────────────────────────────────────────────

def load_known_facts(path: pathlib.Path) -> List[Tuple[str, int, str]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [(item["question"], int(item["answer"]), item.get("kind", "unknown")) for item in raw]


def split_facts(
    facts: List[Tuple[str, int, str]],
    rng: random.Random,
    train_frac: float,
    val_frac: float,
    n_fewshot_facts: int,
) -> Tuple[List, List, List, List]:
    """Split facts into fewshot, train, val, test pools."""
    facts = facts.copy()
    rng.shuffle(facts)

    fewshot_pool = facts[:n_fewshot_facts]
    remaining = facts[n_fewshot_facts:]

    n = len(remaining)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train_facts = remaining[:n_train]
    val_facts = remaining[n_train:n_train + n_val]
    test_facts = remaining[n_train + n_val:]

    return fewshot_pool, train_facts, val_facts, test_facts


def generate_valid_pairs(
    facts: List[Tuple[str, int, str]],
    max_answer: int,
    rng: random.Random,
) -> List[Tuple[int, int]]:
    pairs = []
    for i in range(len(facts)):
        for j in range(i + 1, len(facts)):
            s = facts[i][1] + facts[j][1]
            if 0 <= s < max_answer:
                pairs.append((i, j))
    rng.shuffle(pairs)
    return pairs


def select_fewshot_examples(
    fewshot_pool: List[Tuple[str, int, str]],
    n_examples: int,
    max_answer: int,
    rng: random.Random,
) -> List[Tuple[str, str, int, int, int]]:
    """Select non-overlapping fact pairs for few-shot examples."""
    pairs = generate_valid_pairs(fewshot_pool, max_answer, rng)
    examples = []
    used = set()
    for i, j in pairs:
        if i in used or j in used:
            continue
        q1, a1, _ = fewshot_pool[i]
        q2, a2, _ = fewshot_pool[j]
        if rng.random() < 0.5:
            q1, a1, q2, a2 = q2, a2, q1, a1
        examples.append((q1, q2, a1, a2, a1 + a2))
        used.add(i)
        used.add(j)
        if len(examples) == n_examples:
            break
    return examples


# ─────────────────────────────────────────────────────────────────────────────
# Prompt and response formatting
# ─────────────────────────────────────────────────────────────────────────────

def compose_question(q1: str, q2: str) -> str:
    q1_inner = q1.rstrip("? \t")
    q2_inner = q2.rstrip("? \t")
    return f"What is ({q1_inner}) + ({q2_inner})?"


def _format_thinking(a1: int, a2: int, total: int, stage: int, n_think: int) -> str:
    """Format the thinking portion based on curriculum stage.

    Args:
        a1, a2, total: The fact values and their sum.
        stage: Curriculum stage (0-3).
        n_think: Number of think tokens to use when replacing values with dots.
    """
    dots = " ".join(["."] * n_think)

    if stage == 0:
        # Fully readable: "Thinking: a1, a2, sum"
        return f"Thinking: {a1}, {a2}, {total}"
    elif stage == 1:
        # Sum replaced: "Thinking: a1, a2, . . . ."
        return f"Thinking: {a1}, {a2}, {dots}"
    elif stage == 2:
        # All values replaced: "Thinking: . . . . . . . . . ."
        return f"Thinking: {dots}"
    elif stage == 3:
        # No prefix, just dots: ". . . . . . . . . ."
        return dots
    else:
        raise ValueError(f"Unknown stage: {stage}")


def _format_fewshot_example(
    q1: str, q2: str, a1: int, a2: int, total: int,
    stage: int, n_think: int,
) -> str:
    """Format a single few-shot example matching the stage format."""
    question = compose_question(q1, q2)
    thinking = _format_thinking(a1, a2, total, stage, n_think)
    return f"Q: {question}\n{thinking}\nAnswer: {total}"


def build_system_prompt(
    fewshot_examples: List[Tuple[str, str, int, int, int]],
    stage: int,
    n_think: int,
) -> str:
    """Build system prompt with readable few-shot examples (always stage 0).

    Few-shots always show "Thinking: a1, a2, sum" regardless of the training
    stage, so the model always has readable demonstrations of the task.
    """
    examples_text = "\n\n".join(
        _format_fewshot_example(q1, q2, a1, a2, s, stage=0, n_think=n_think)
        for q1, q2, a1, a2, s in fewshot_examples
    )

    if stage <= 1:
        instruction = (
            "Think briefly, then give your final answer as 'Answer: [NUMBER]'."
        )
    else:
        instruction = (
            "Give your final answer as 'Answer: [NUMBER]'."
        )

    return (
        "You answer questions that require looking up two facts and adding them.\n"
        "\n"
        "Here are some worked examples:\n"
        "\n"
        f"{examples_text}\n"
        "\n"
        f"The user will ask a similar question. {instruction}"
    )


def build_assistant_response(
    a1: int, a2: int, total: int, stage: int, n_think: int,
) -> str:
    """Build the full assistant response for a training example."""
    thinking = _format_thinking(a1, a2, total, stage, n_think)
    return f"{thinking}\nAnswer: {total}"


def build_assistant_prefill(stage: int, n_think: int) -> str:
    """Build the portion of the assistant response before the answer value.

    This is masked in labels — the model is supervised on what comes after.
    For compressed reasoning, we supervise ALL tokens (thinking + answer),
    so the prefill is empty.
    """
    # For compressed reasoning, loss is on all assistant tokens
    return ""


# ─────────────────────────────────────────────────────────────────────────────
# Tokenization
# ─────────────────────────────────────────────────────────────────────────────

def tokenize_example(
    tok: AutoTokenizer,
    system_prompt: str,
    question: str,
    assistant_response: str,
) -> Tuple[List[int], List[int], List[int]]:
    """Tokenize a training example with loss on all assistant tokens.

    Returns (input_ids, labels, attention_mask).
    Labels = -100 for system+user tokens, real IDs for assistant tokens.
    """
    # Build messages
    messages_with_answer = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
        {"role": "assistant", "content": assistant_response},
    ]
    messages_prompt_only = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]

    # Tokenize prompt (system + user + generation prompt)
    prompt_text = tok.apply_chat_template(
        messages_prompt_only, tokenize=False, add_generation_prompt=True
    )
    prompt_ids = tok.encode(prompt_text, add_special_tokens=False)

    # Tokenize full sequence
    full_text = tok.apply_chat_template(
        messages_with_answer, tokenize=False
    )
    full_ids = tok.encode(full_text, add_special_tokens=False)

    n_prompt = len(prompt_ids)

    # Check alignment
    if n_prompt < len(full_ids) and full_ids[:n_prompt] == prompt_ids:
        answer_ids = full_ids[n_prompt:]
        input_ids = full_ids
    else:
        # BPE misalignment fallback
        remaining_text = full_text[len(prompt_text):] if full_text.startswith(prompt_text) else ""
        if not remaining_text:
            # Construct manually
            end_marker = ""
            dummy = tok.apply_chat_template(
                [{"role": "assistant", "content": "X"}],
                tokenize=False, add_special_tokens=True,
            )
            marker_pos = dummy.find("X") + 1
            if marker_pos > 0 and marker_pos < len(dummy):
                end_marker = dummy[marker_pos:]
            remaining_text = assistant_response + end_marker

        answer_ids = tok.encode(remaining_text, add_special_tokens=False)
        input_ids = prompt_ids + answer_ids

    labels = [-100] * len(prompt_ids) + list(answer_ids)
    attention_mask = [1] * len(input_ids)

    assert len(input_ids) == len(labels), (
        f"input_ids ({len(input_ids)}) != labels ({len(labels)})"
    )

    return input_ids, labels, attention_mask


# ─────────────────────────────────────────────────────────────────────────────
# Dataset building
# ─────────────────────────────────────────────────────────────────────────────

def build_examples(
    facts: List[Tuple[str, int, str]],
    pairs: List[Tuple[int, int]],
    fewshot_examples: List[Tuple[str, str, int, int, int]],
    tok: AutoTokenizer,
    stage: int,
    n_think: int,
    split: str,
    rng: random.Random,
    max_examples: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Build tokenized examples for a given split and stage."""

    system_prompt = build_system_prompt(fewshot_examples, stage, n_think)

    n = min(len(pairs), max_examples) if max_examples else len(pairs)
    examples = []

    for idx in range(n):
        i, j = pairs[idx]

        # Randomly assign q1/q2
        if rng.random() < 0.5:
            q1, a1, t1 = facts[i]
            q2, a2, t2 = facts[j]
        else:
            q1, a1, t1 = facts[j]
            q2, a2, t2 = facts[i]

        total = a1 + a2
        question = compose_question(q1, q2)
        assistant_response = build_assistant_response(a1, a2, total, stage, n_think)

        input_ids, labels, attention_mask = tokenize_example(
            tok, system_prompt, question, assistant_response
        )

        examples.append({
            "id": f"{split}-{idx}",
            "pair_id": idx,
            "split": split,
            "stage": stage,
            "fact1": q1,
            "fact2": q2,
            "a1": a1,
            "a2": a2,
            "answer": total,
            "type1": t1,
            "type2": t2,
            "n_think": n_think,
            "assistant_response": assistant_response,
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        })

    return examples


def build_test_examples(
    facts: List[Tuple[str, int, str]],
    pairs: List[Tuple[int, int]],
    fewshot_examples: List[Tuple[str, str, int, int, int]],
    tok: AutoTokenizer,
    stage: int,
    n_think: int,
    split: str,
    rng: random.Random,
    n_think_values: List[int],
    max_pairs: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Build test examples: each pair gets every n_think value."""

    n = min(len(pairs), max_pairs) if max_pairs else len(pairs)
    examples = []
    eid = 0

    for idx in range(n):
        i, j = pairs[idx]

        if rng.random() < 0.5:
            q1, a1, t1 = facts[i]
            q2, a2, t2 = facts[j]
        else:
            q1, a1, t1 = facts[j]
            q2, a2, t2 = facts[i]

        total = a1 + a2
        question = compose_question(q1, q2)

        for nt in n_think_values:
            # Rebuild system prompt for each n_think value so few-shot examples
            # match the format of the test example (dot count varies with nt)
            system_prompt = build_system_prompt(fewshot_examples, stage, nt)
            assistant_response = build_assistant_response(a1, a2, total, stage, nt)
            input_ids, labels, attention_mask = tokenize_example(
                tok, system_prompt, question, assistant_response
            )

            examples.append({
                "id": f"{split}-{eid}",
                "pair_id": idx,
                "split": split,
                "stage": stage,
                "fact1": q1,
                "fact2": q2,
                "a1": a1,
                "a2": a2,
                "answer": total,
                "type1": t1,
                "type2": t2,
                "n_think": nt,
                "assistant_response": assistant_response,
                "input_ids": input_ids,
                "labels": labels,
                "attention_mask": attention_mask,
            })
            eid += 1

    return examples


def write_jsonl(rows: List[Dict[str, Any]], path: pathlib.Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate compressed reasoning curriculum dataset"
    )

    parser.add_argument("--known-facts", type=str, required=True,
                        help="Path to known_facts.json")
    parser.add_argument("--tokenizer", type=str, default="Qwen/Qwen2.5-14B-Instruct",
                        help="Tokenizer to use for encoding")

    # Curriculum
    parser.add_argument("--stage", type=int, required=True, choices=[0, 1, 2, 3],
                        help="Curriculum stage (0=readable, 3=fully opaque)")
    parser.add_argument("--n-think-tokens", type=int, default=10,
                        help="Number of think tokens (dots) when replacing values")

    # Data splits
    parser.add_argument("--train-frac", type=float, default=0.74,
                        help="Fraction of facts for training")
    parser.add_argument("--val-frac", type=float, default=0.10,
                        help="Fraction of facts for validation")
    parser.add_argument("--n-fewshot", type=int, default=5,
                        help="Number of few-shot examples")
    parser.add_argument("--n-fewshot-facts", type=int, default=10,
                        help="Number of facts reserved for few-shot pool")
    parser.add_argument("--max-answer", type=int, default=300,
                        help="Maximum allowed sum")
    parser.add_argument("--max-train", type=int, default=None,
                        help="Limit training examples")
    parser.add_argument("--test-n-think-values", type=str, default=None,
                        help="Comma-separated n_think values for test set "
                             "(default: same as --n-think-tokens)")

    # Output
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    rng = random.Random(args.seed)
    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Parse test n_think values
    if args.test_n_think_values:
        test_n_think = [int(x) for x in args.test_n_think_values.split(",")]
    else:
        test_n_think = [args.n_think_tokens]

    # Load tokenizer
    print(f"Loading tokenizer: {args.tokenizer}")
    tok = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    # Load and split facts
    facts = load_known_facts(pathlib.Path(args.known_facts))
    print(f"Loaded {len(facts)} known facts")

    fewshot_pool, train_facts, val_facts, test_facts = split_facts(
        facts, rng, args.train_frac, args.val_frac, args.n_fewshot_facts,
    )

    fewshot_examples = select_fewshot_examples(
        fewshot_pool, args.n_fewshot, args.max_answer, rng,
    )

    print(f"  Few-shot: {len(fewshot_examples)} examples from {len(fewshot_pool)} facts")
    print(f"  Train: {len(train_facts)} facts")
    print(f"  Val: {len(val_facts)} facts")
    print(f"  Test: {len(test_facts)} facts")

    # Generate pairs
    train_pairs = generate_valid_pairs(train_facts, args.max_answer, rng)
    val_pairs = generate_valid_pairs(val_facts, args.max_answer, rng)
    test_pairs = generate_valid_pairs(test_facts, args.max_answer, rng)

    print(f"  Train pairs: {len(train_pairs)}")
    print(f"  Val pairs: {len(val_pairs)}")
    print(f"  Test pairs: {len(test_pairs)}")

    # Build examples
    print(f"\nGenerating stage {args.stage} data (n_think={args.n_think_tokens})...")

    train_examples = build_examples(
        train_facts, train_pairs, fewshot_examples, tok,
        args.stage, args.n_think_tokens, "train", rng,
        max_examples=args.max_train,
    )
    val_examples = build_examples(
        val_facts, val_pairs, fewshot_examples, tok,
        args.stage, args.n_think_tokens, "val", rng,
    )
    test_examples = build_test_examples(
        test_facts, test_pairs, fewshot_examples, tok,
        args.stage, args.n_think_tokens, "test", rng,
        n_think_values=test_n_think,
    )

    print(f"  Train: {len(train_examples)} examples")
    print(f"  Val: {len(val_examples)} examples")
    print(f"  Test: {len(test_examples)} examples")

    # Preview first example
    if train_examples:
        ex = train_examples[0]
        print(f"\n  Sample assistant response:")
        print(f"    {repr(ex['assistant_response'])}")
        print(f"    a1={ex['a1']}, a2={ex['a2']}, answer={ex['answer']}")
        n_supervised = sum(1 for l in ex['labels'] if l != -100)
        print(f"    Sequence length: {len(ex['input_ids'])}, supervised tokens: {n_supervised}")

    # Save
    write_jsonl(train_examples, outdir / "train.jsonl")
    write_jsonl(val_examples, outdir / "val.jsonl")
    write_jsonl(test_examples, outdir / "test.jsonl")

    # Save manifest
    manifest = {
        "stage": args.stage,
        "n_think_tokens": args.n_think_tokens,
        "test_n_think_values": test_n_think,
        "seed": args.seed,
        "max_answer": args.max_answer,
        "tokenizer": args.tokenizer,
        "known_facts": args.known_facts,
        "n_train": len(train_examples),
        "n_val": len(val_examples),
        "n_test": len(test_examples),
        "n_train_facts": len(train_facts),
        "n_val_facts": len(val_facts),
        "n_test_facts": len(test_facts),
        "fewshot_examples": [
            {"q1": q1, "q2": q2, "a1": a1, "a2": a2, "sum": s}
            for q1, q2, a1, a2, s in fewshot_examples
        ],
    }
    with open(outdir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nSaved to {outdir}/")
    print("Done!")


if __name__ == "__main__":
    main()