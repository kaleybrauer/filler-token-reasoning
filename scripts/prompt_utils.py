"""
prompt_utils.py

Shared prompt building and answer parsing for all evaluation scripts.

Default format uses bare assistant turns (assistant outputs just the number,
no "Answer: " prefix) to match the API setup.
"""

import random
import re
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))
from generate_1hop_dataset import build_system_message, build_user_turn


def build_messages_pre_padding(few_shot_items, target, k, bare_assistant=True):
    """Build messages with k dots BEFORE the question (pre-padding).

    Format: "Padding: . . . ...\n\nQuestion: ...\n\nAnswer:"
    Few-shot examples also get pre-padding for consistency.
    """
    system_msg = build_system_message("dots", k)
    messages = [{"role": "system", "content": system_msg}]
    asst_fmt = "{ans}" if bare_assistant else "Answer: {ans}"

    for fs in few_shot_items:
        padding = " ".join(["."] * k)
        user_content = (
            f"Padding: {padding}\n\n"
            f"Question: What is {fs['fact_phrase']} plus {fs['x']}?\n\n"
            f"Answer:"
        )
        messages.append({"role": "user", "content": user_content})
        messages.append({"role": "assistant", "content": asst_fmt.format(ans=fs["answer"])})

    padding = " ".join(["."] * k)
    user_content = (
        f"Padding: {padding}\n\n"
        f"Question: What is {target['fact_phrase']} plus {target['x']}?\n\n"
        f"Answer:"
    )
    messages.append({"role": "user", "content": user_content})

    return messages


def build_messages(few_shot_items, target, mode, k, bare_assistant=True,
                   filler_type="dots"):
    """Build chat messages for any condition.

    Args:
        few_shot_items: list of few-shot fact dicts
        target: target problem dict
        mode: "dots", "counting", "format_only", "no_newline", "pre_padding", "baseline"
        k: filler length (used for few-shots and target in dots/counting/pre_padding;
           few-shots only in format_only/no_newline)
        bare_assistant: if True (default), assistant turns are just the number.
            if False, assistant turns are "Answer: {number}" (legacy format).
        filler_type: "dots" or "counting". Used for dots/counting modes.
            For mode="counting", this is set automatically.

    Returns:
        list of message dicts with role/content keys
    """
    if mode == "baseline":
        k = 0

    if mode == "counting":
        filler_type = "counting"
        mode = "dots"  # same prompt structure, different filler content

    if mode == "alphabet":
        filler_type = "alphabet"
        mode = "dots"  # same prompt structure, different filler content

    if mode == "pre_padding":
        return build_messages_pre_padding(few_shot_items, target, k,
                                          bare_assistant=bare_assistant)

    system_msg = build_system_message(filler_type, k)
    messages = [{"role": "system", "content": system_msg}]

    asst_fmt = "{ans}" if bare_assistant else "Answer: {ans}"

    # Few-shots: always have k filler tokens (except baseline which has k=0)
    for fs in few_shot_items:
        user_content = build_user_turn(
            fs["fact_phrase"], fs["x"], filler_type, k, rng=random.Random(42)
        )
        messages.append({"role": "user", "content": user_content})
        messages.append({"role": "assistant", "content": asst_fmt.format(ans=fs["answer"])})

    # Target turn depends on mode
    if mode in ("dots", "baseline"):
        user_content = build_user_turn(
            target["fact_phrase"], target["x"], filler_type, k
        )
    elif mode == "format_only":
        user_content = build_user_turn(
            target["fact_phrase"], target["x"], "dots", 0
        )
        user_content = user_content.replace("\n\nAnswer:", "\n\nFiller:\n\nAnswer:")
    elif mode == "no_newline":
        user_content = build_user_turn(
            target["fact_phrase"], target["x"], "dots", 0
        )
        user_content = user_content.replace("\n\nAnswer:", "\n\nFiller:Answer:")
    else:
        raise ValueError(f"Unknown mode: {mode}")

    messages.append({"role": "user", "content": user_content})
    return messages


def extract_answer(text: str) -> Optional[int]:
    """Parse a numeric answer from model output."""
    m = re.search(r"Answer:\s*(-?\d+)", text)
    if m:
        return int(m.group(1))
    m = re.search(r"(-?\d+)", text)
    if m:
        return int(m.group(1))
    return None
