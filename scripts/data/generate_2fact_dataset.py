#!/usr/bin/env python3
"""Generate a 2-fact addition dataset for probing experiments.

The task is: "What is [fact_phrase_1] plus [fact_phrase_2]?" where each
fact_phrase maps to a known integer (atomic number). The model must retrieve
both values A1 and A2, then compute A1 + A2.

Usage:
    python scripts/generate_2fact_dataset.py --preview
    python scripts/generate_2fact_dataset.py --num-examples 1500
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.generate_1hop_dataset import (
    build_system_message,
    filler_description,
    make_filler_tokens,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
KNOWN_FACTS_PATH = Path("finetuning/data/facts/knowledge_deepseek_v3/known_facts.json")
OUTPUT_PATH = Path("data/2fact_addition_dataset.json")

SEED = 42
NUM_FEW_SHOT = 10
NUM_EXAMPLES = 1500


# ---------------------------------------------------------------------------
# Fact-phrase conversion (atomic numbers only)
# ---------------------------------------------------------------------------

def atomic_fact_phrase(question: str) -> str:
    """Convert 'What is the atomic number of X?' to 'the atomic number of X'."""
    q = question.strip().rstrip("?").strip()
    m = re.match(r"What is the atomic number of (.+)", q)
    if m:
        return f"the atomic number of {m.group(1)}"
    raise ValueError(f"Not an atomic number question: {question!r}")


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def build_system_message_2fact(filler_type: str, k: int) -> str:
    """Build the system message for 2-fact addition."""
    base = (
        "You will be given a question that requires adding two values together. "
        "Answer immediately with just the number, nothing else. "
        "No explanation, no words, no reasoning, just the number."
    )
    if k > 0:
        desc = filler_description(filler_type)
        base += (
            f" After the question, there will be {k} filler tokens "
            f"(a sequence of {desc}) to give you extra space to process "
            f"the problem before answering."
        )
    return base


def build_user_turn_2fact(fact_phrase_1: str, fact_phrase_2: str,
                         filler_type: str, k: int,
                         rng: random.Random | None = None) -> str:
    """Build the user turn for a 2-fact addition question."""
    question_line = f"Question: What is {fact_phrase_1} plus {fact_phrase_2}?"
    if k > 0:
        filler = make_filler_tokens(filler_type, k, rng=rng)
        return f"{question_line}\n\nFiller: {filler}\n\nAnswer:"
    else:
        return f"{question_line}\n\nAnswer:"


def build_prompt_messages_2fact(
    few_shot_items: list[dict],
    target_item: dict,
    filler_type: str,
    k: int,
    rng: random.Random | None = None,
) -> list[dict]:
    """Build a full chat-style prompt for 2-fact addition."""
    system_msg = build_system_message_2fact(filler_type, k)
    messages = [{"role": "system", "content": system_msg}]

    for fs in few_shot_items:
        user_content = build_user_turn_2fact(
            fs["fact_phrase_1"], fs["fact_phrase_2"], filler_type, k, rng=rng
        )
        messages.append({"role": "user", "content": user_content})
        messages.append({"role": "assistant", "content": str(fs["answer"])})

    user_content = build_user_turn_2fact(
        target_item["fact_phrase_1"], target_item["fact_phrase_2"],
        filler_type, k, rng=rng
    )
    messages.append({"role": "user", "content": user_content})

    return messages


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate 2-fact addition dataset (atomic numbers only)."
    )
    parser.add_argument("--known-facts", type=Path, default=KNOWN_FACTS_PATH)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--num-examples", type=int, default=NUM_EXAMPLES)
    parser.add_argument("--max-atomic-number", type=int, default=None,
                        help="Only use elements with atomic number < this value")
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()

    if args.output is None:
        if args.max_atomic_number is not None:
            args.output = Path(f"data/2fact_addition_dataset_lt{args.max_atomic_number}.json")
        else:
            args.output = OUTPUT_PATH

    # Load atomic number facts only
    with open(args.known_facts) as f:
        all_facts = json.load(f)

    atomic_facts = [f for f in all_facts if "atomic number" in f["question"]]
    if args.max_atomic_number is not None:
        atomic_facts = [f for f in atomic_facts if f["answer"] < args.max_atomic_number]
    print(f"Loaded {len(atomic_facts)} atomic number facts (of {len(all_facts)} total).")

    if len(atomic_facts) < NUM_FEW_SHOT + 4:
        raise ValueError(f"Need at least {NUM_FEW_SHOT + 4} atomic facts, got {len(atomic_facts)}")

    rng = random.Random(SEED)

    # Reserve few-shot facts (pairs of elements)
    indices = list(range(len(atomic_facts)))
    rng.shuffle(indices)
    few_shot_indices = set(indices[:NUM_FEW_SHOT])
    pool_indices = indices[NUM_FEW_SHOT:]

    few_shot_facts_raw = [atomic_facts[i] for i in sorted(few_shot_indices)]

    # Build few-shot items as pairs
    few_shot_items: list[dict] = []
    for i in range(0, len(few_shot_facts_raw) - 1, 2):
        f1 = few_shot_facts_raw[i]
        f2 = few_shot_facts_raw[i + 1]
        few_shot_items.append({
            "fact_question_1": f1["question"],
            "fact_phrase_1": atomic_fact_phrase(f1["question"]),
            "fact_value_1": f1["answer"],
            "fact_question_2": f2["question"],
            "fact_phrase_2": atomic_fact_phrase(f2["question"]),
            "fact_value_2": f2["answer"],
            "answer": f1["answer"] + f2["answer"],
        })

    print(f"Reserved {len(few_shot_items)} few-shot pairs "
          f"(from {len(few_shot_facts_raw)} facts).")

    # Generate examples: random pairs from pool
    pool_facts = [atomic_facts[i] for i in pool_indices]

    # Build all possible pairs
    all_pairs = []
    for i in range(len(pool_facts)):
        for j in range(i + 1, len(pool_facts)):
            all_pairs.append((pool_facts[i], pool_facts[j]))
    rng.shuffle(all_pairs)
    print(f"Pool: {len(pool_facts)} facts, {len(all_pairs)} possible pairs.")

    examples: list[dict] = []
    for pair_idx, (f1, f2) in enumerate(all_pairs):
        if len(examples) >= args.num_examples:
            break
        # Randomly swap order so model can't rely on element ordering
        if rng.random() < 0.5:
            f1, f2 = f2, f1
        examples.append({
            "idx": len(examples),
            "fact_question_1": f1["question"],
            "fact_phrase_1": atomic_fact_phrase(f1["question"]),
            "fact_value_1": f1["answer"],
            "fact_question_2": f2["question"],
            "fact_phrase_2": atomic_fact_phrase(f2["question"]),
            "fact_value_2": f2["answer"],
            "answer": f1["answer"] + f2["answer"],
        })

    n_unique_elements = len(set(
        e["fact_phrase_1"] for e in examples
    ) | set(
        e["fact_phrase_2"] for e in examples
    ))
    print(f"Generated {len(examples)} examples using {n_unique_elements} unique elements.")

    # Save
    dataset = {
        "task": "2fact_addition",
        "fact_type": "atomic_number",
        "few_shot_facts": few_shot_items,
        "examples": examples,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    print(f"Saved dataset to {args.output}")

    # Preview
    if args.preview:
        print("\n" + "=" * 72)
        print("PREVIEW: example prompts")
        print("=" * 72)

        target = examples[0]

        for filler_type, k in [("none", 0), ("dots", 10)]:
            preview_rng = random.Random(SEED)
            label = f"{filler_type} (k={k})" if k > 0 else "baseline (k=0)"
            print(f"\n{'─' * 72}")
            print(f"  Condition: {label}")
            print(f"  A1={target['fact_value_1']}, A2={target['fact_value_2']}, "
                  f"Answer={target['answer']}")
            print(f"{'─' * 72}")

            if k == 0:
                msgs = build_prompt_messages_2fact(
                    few_shot_items, target, "none", 0, rng=preview_rng
                )
            else:
                msgs = build_prompt_messages_2fact(
                    few_shot_items, target, filler_type, k, rng=preview_rng
                )

            for msg in msgs:
                role = msg["role"].upper()
                content = msg["content"]
                if len(content) > 200:
                    content = content[:200] + "..."
                print(f"\n  [{role}]\n  {content}")


if __name__ == "__main__":
    main()
