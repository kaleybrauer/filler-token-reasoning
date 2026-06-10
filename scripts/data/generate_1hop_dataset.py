#!/usr/bin/env python3
"""Generate a 1-hop addition dataset for probing experiments.

The task is: "What is [fact_phrase] plus [x]?" where fact_phrase is derived
from a known-fact question, x is a random integer in [10, 99], and the answer
is fact_value + x.

Usage:
    uv run --project /workspace/filler-token-reasoning python \
        scripts/generate_1hop_dataset.py --preview
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
KNOWN_FACTS_PATH = Path(
    "data/facts/knowledge_deepseek_v3/known_facts.json"
)
OUTPUT_PATH = Path("data/1hop_addition_dataset.json")

SEED = 42
NUM_FEW_SHOT = 10
NUM_EXAMPLES = 500
X_MIN = 10
X_MAX = 99


# ---------------------------------------------------------------------------
# Fact-phrase conversion
# ---------------------------------------------------------------------------

def question_to_fact_phrase(question: str) -> str:
    """Convert a fact question into a noun-phrase suitable for embedding in
    'What is <phrase> plus <x>?'

    Rules:
        "At what age did X die?"          -> "the age at which X died"
        "What is the atomic number of X?" -> "the atomic number of X"
        "What is the number of Y?"        -> "the number of Y"
    """
    # Strip trailing question mark and whitespace
    q = question.strip().rstrip("?").strip()

    # Pattern 1: "At what age did X die"
    m = re.match(r"At what age did (.+?) die", q)
    if m:
        return f"the age at which {m.group(1)} died"

    # Pattern 2: "What is the atomic number of X"
    m = re.match(r"What is the atomic number of (.+)", q)
    if m:
        return f"the atomic number of {m.group(1)}"

    # Pattern 3: "What is the number of Y"
    m = re.match(r"What is the number of (.+)", q)
    if m:
        return f"the number of {m.group(1)}"

    raise ValueError(f"Cannot convert question to fact phrase: {question!r}")


# ---------------------------------------------------------------------------
# Filler token helpers
# ---------------------------------------------------------------------------

def make_filler_tokens(filler_type: str, k: int, rng: random.Random | None = None) -> str:
    """Return the filler string for a given type and count k."""
    if k == 0:
        return ""
    if filler_type == "dots":
        return " ".join(["."] * k)
    elif filler_type == "counting":
        return " ".join(str(i) for i in range(1, k + 1))
    elif filler_type == "counting-scrambled":
        nums = list(range(1, k + 1))
        if rng is not None:
            rng.shuffle(nums)
        else:
            random.shuffle(nums)
        return " ".join(str(i) for i in nums)
    elif filler_type == "alphabet":
        import string
        letters = string.ascii_lowercase
        return " ".join(letters[i % 26] for i in range(k))
    else:
        raise ValueError(f"Unknown filler type: {filler_type!r}")


def filler_description(filler_type: str) -> str:
    """Human-readable description of the filler type for the system prompt."""
    if filler_type == "dots":
        return "dots"
    elif filler_type == "counting":
        return "sequential integers"
    elif filler_type == "counting-scrambled":
        return "shuffled integers"
    elif filler_type == "alphabet":
        return "cycling lowercase letters (a-z)"
    else:
        raise ValueError(f"Unknown filler type: {filler_type!r}")


# ---------------------------------------------------------------------------
# Prompt construction (for --preview only; the dataset stores raw data)
# ---------------------------------------------------------------------------

def build_system_message(filler_type: str, k: int) -> str:
    """Build the system message for a given filler condition."""
    base = (
        "You will be given a question. Answer immediately using the format "
        "'Answer: [ANSWER]' where [ANSWER] is just the number, nothing else. "
        "No explanation, no words, no reasoning, just the number."
    )
    if k > 0:
        desc = filler_description(filler_type)
        # Deliberately do NOT state an exact count: the actual token count is not
        # k for every filler type (counting_K is 2K-1 tokens — number + separating
        # space each — while dots_K is K), so naming a number would be false for
        # counting and injects an unrelated integer into the context.
        base += (
            f" After the question, there will be some filler tokens "
            f"(a sequence of {desc}) to give you extra space to process "
            f"the problem before answering."
        )
    return base


def build_user_turn(fact_phrase: str, x: int, filler_type: str, k: int,
                    rng: random.Random | None = None) -> str:
    """Build the user turn content."""
    question_line = f"Question: What is {fact_phrase} plus {x}?"
    if k > 0:
        filler = make_filler_tokens(filler_type, k, rng=rng)
        return f"{question_line}\n\nFiller: {filler}\n\nAnswer:"
    else:
        return f"{question_line}\n\nAnswer:"


def build_prompt_messages(
    few_shot_items: list[dict],
    target_item: dict,
    filler_type: str,
    k: int,
    rng: random.Random | None = None,
) -> list[dict]:
    """Build a full chat-style prompt (list of message dicts) for preview."""
    system_msg = build_system_message(filler_type, k)
    messages = [{"role": "system", "content": system_msg}]

    # Few-shot examples
    for fs in few_shot_items:
        user_content = build_user_turn(
            fs["fact_phrase"], fs["x"], filler_type, k, rng=rng
        )
        assistant_content = str(fs["answer"])
        messages.append({"role": "user", "content": user_content})
        messages.append({"role": "assistant", "content": assistant_content})

    # Target
    user_content = build_user_turn(
        target_item["fact_phrase"], target_item["x"], filler_type, k, rng=rng
    )
    messages.append({"role": "user", "content": user_content})

    return messages


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate 1-hop addition dataset for probing experiments."
    )
    parser.add_argument(
        "--known-facts",
        type=Path,
        default=KNOWN_FACTS_PATH,
        help="Path to known_facts.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help="Output path for dataset JSON",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=NUM_EXAMPLES,
        help="Number of examples to generate (default: 500)",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Print 3 example prompts (baseline, dots k=5, counting k=5) "
             "to verify format, then exit.",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Load facts
    # ------------------------------------------------------------------
    with open(args.known_facts) as f:
        all_facts = json.load(f)

    print(f"Loaded {len(all_facts)} known facts.")

    # ------------------------------------------------------------------
    # 2. Reserve 10 few-shot facts
    # ------------------------------------------------------------------
    rng = random.Random(SEED)

    indices = list(range(len(all_facts)))
    rng.shuffle(indices)
    few_shot_indices = set(indices[:NUM_FEW_SHOT])
    pool_indices = indices[NUM_FEW_SHOT:]

    few_shot_facts_raw = [all_facts[i] for i in sorted(few_shot_indices)]

    # Build few-shot items (each gets a random x)
    few_shot_items: list[dict] = []
    for fact in few_shot_facts_raw:
        x = rng.randint(X_MIN, X_MAX)
        fact_phrase = question_to_fact_phrase(fact["question"])
        few_shot_items.append({
            "fact_question": fact["question"],
            "fact_phrase": fact_phrase,
            "fact_value": fact["answer"],
            "x": x,
            "answer": fact["answer"] + x,
            "kind": fact["kind"],
        })

    print(f"Reserved {len(few_shot_items)} few-shot facts.")

    # ------------------------------------------------------------------
    # 3. Generate examples: use every fact once first, then repeat with new x
    # ------------------------------------------------------------------
    pool_facts = [all_facts[i] for i in pool_indices]

    examples: list[dict] = []

    # First pass: one example per unique fact (shuffled order)
    shuffled_pool = list(pool_facts)
    rng.shuffle(shuffled_pool)
    for fact in shuffled_pool:
        if len(examples) >= args.num_examples:
            break
        x = rng.randint(X_MIN, X_MAX)
        fact_phrase = question_to_fact_phrase(fact["question"])
        examples.append({
            "idx": len(examples),
            "fact_question": fact["question"],
            "fact_phrase": fact_phrase,
            "fact_value": fact["answer"],
            "x": x,
            "answer": fact["answer"] + x,
            "kind": fact["kind"],
        })

    # Second pass: fill remaining slots by cycling through facts with new x
    while len(examples) < args.num_examples:
        fact = rng.choice(pool_facts)
        x = rng.randint(X_MIN, X_MAX)
        fact_phrase = question_to_fact_phrase(fact["question"])
        examples.append({
            "idx": len(examples),
            "fact_question": fact["question"],
            "fact_phrase": fact_phrase,
            "fact_value": fact["answer"],
            "x": x,
            "answer": fact["answer"] + x,
            "kind": fact["kind"],
        })

    n_unique = len(set(e["fact_phrase"] for e in examples))
    print(f"Generated {len(examples)} examples ({n_unique} unique facts, "
          f"{len(examples) - n_unique} repeats with new x).")

    # ------------------------------------------------------------------
    # 4. Save dataset
    # ------------------------------------------------------------------
    dataset = {
        "few_shot_facts": few_shot_items,
        "examples": examples,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    print(f"Saved dataset to {args.output}")

    # ------------------------------------------------------------------
    # 5. Preview (optional)
    # ------------------------------------------------------------------
    if args.preview:
        print("\n" + "=" * 72)
        print("PREVIEW: 3 example prompts")
        print("=" * 72)

        target = examples[0]

        conditions = [
            ("none", 0),
            ("dots", 5),
            ("counting", 5),
        ]

        for filler_type, k in conditions:
            preview_rng = random.Random(SEED)
            label = f"{filler_type} (k={k})" if k > 0 else "baseline (k=0)"
            print(f"\n{'─' * 72}")
            print(f"  Condition: {label}")
            print(f"{'─' * 72}")

            if k == 0:
                msgs = build_prompt_messages(
                    few_shot_items, target, "none", 0, rng=preview_rng
                )
            else:
                msgs = build_prompt_messages(
                    few_shot_items, target, filler_type, k, rng=preview_rng
                )

            for msg in msgs:
                role = msg["role"].upper()
                print(f"\n[{role}]")
                print(msg["content"])

            print(f"\n  Expected answer: {target['answer']}")

        print("\n" + "=" * 72)


if __name__ == "__main__":
    main()
