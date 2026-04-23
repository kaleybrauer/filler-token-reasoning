"""
generate_letterpos_dataset.py

Prompt helpers for the element-letter-position task (atomic_number → element
name → Nth letter). The dataset itself lives at data/element_letter_positions.json
and does not need to be generated here — this module just provides the
prompt-building functions used by extract_hidden_states.py and eval scripts,
mirroring generate_1hop_dataset.py / generate_2fact_dataset.py.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.generate_1hop_dataset import make_filler_tokens, filler_description


def build_system_message_letterpos(filler_type: str, k: int) -> str:
    """System message for letter-lookup tasks (elements, capitals, etc.).

    Task-agnostic wording — the few-shot examples specify which kind of
    name / word is being asked about.
    """
    base = (
        "You will be given a question asking for a specific letter. "
        "Answer immediately with just the single lowercase letter, "
        "nothing else. No explanation, no words, no reasoning, "
        "just the letter."
    )
    if k > 0:
        desc = filler_description(filler_type)
        base += (
            f" After the question, there will be {k} filler tokens "
            f"(a sequence of {desc}) to give you extra space to process "
            f"the problem before answering."
        )
    return base


def build_user_turn_letterpos(question: str, filler_type: str, k: int,
                              rng: random.Random | None = None) -> str:
    """User turn for the letter-position task.

    The dataset's `question` field is already a fully-formed sentence
    ("What is the third letter of the chemical element with atomic number 47?"),
    so no phrasing template is needed.
    """
    question_line = f"Question: {question}"
    if k > 0:
        filler = make_filler_tokens(filler_type, k, rng=rng)
        return f"{question_line}\n\nFiller: {filler}\n\nAnswer:"
    return f"{question_line}\n\nAnswer:"


def build_prompt_messages_letterpos(
    few_shot_items: list[dict],
    target_item: dict,
    filler_type: str,
    k: int,
    rng: random.Random | None = None,
) -> list[dict]:
    """Full chat prompt for a letter-position problem (bare assistant)."""
    system_msg = build_system_message_letterpos(filler_type, k)
    messages = [{"role": "system", "content": system_msg}]

    for fs in few_shot_items:
        user_content = build_user_turn_letterpos(
            fs["question"], filler_type, k, rng=rng
        )
        messages.append({"role": "user", "content": user_content})
        messages.append({"role": "assistant", "content": str(fs["answer"])})

    user_content = build_user_turn_letterpos(
        target_item["question"], filler_type, k, rng=rng
    )
    messages.append({"role": "user", "content": user_content})

    return messages


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preview prompts for the letter-position task."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/element_letter_positions.json"),
    )
    parser.add_argument(
        "--preview", action="store_true",
        help="Render 3 example prompts (baseline, dots_25, counting_25) for the "
             "first test example and exit.",
    )
    args = parser.parse_args()

    with open(args.dataset) as f:
        dataset = json.load(f)
    few_shot = dataset["few_shot_examples"]
    examples = dataset["examples"]
    print(f"Loaded {len(examples)} examples, {len(few_shot)} few-shot.")

    if args.preview:
        target = examples[0]
        for filler_type, k in [("dots", 0), ("dots", 25), ("counting", 25)]:
            print("\n" + "=" * 80)
            print(f"filler_type={filler_type}, k={k}")
            print("=" * 80)
            messages = build_prompt_messages_letterpos(
                few_shot, target, filler_type, k, rng=random.Random(0)
            )
            for m in messages:
                print(f"\n[{m['role']}]")
                print(m["content"])
            print(f"\n(expected answer: {target['answer']!r})")


if __name__ == "__main__":
    main()
