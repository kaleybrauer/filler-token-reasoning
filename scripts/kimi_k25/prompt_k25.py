#!/usr/bin/env python3
"""Instant-mode (reasoning-suppressed) prompt construction for Kimi K2.5.

Filler placement
----------------
The `Filler: …\\n\\nAnswer:` scaffold stays at the end of the **user** turn, exactly
where `build_user_turn_varbind()` puts it for DeepSeek-V3 and for the published
Kimi K2 letterpos/capitalpos runs. The assistant turn is bare (just the number).

RUNBOOK §A2 originally proposed appending the scaffold *after*
`apply_chat_template(add_generation_prompt=True)`, which would move the filler into
the assistant turn. Kaley's call (2026-07-28): match the V3 setup instead, because
which side of the turn boundary the filler sits on is a confound in the one
cross-model comparison this run exists to make. Preflight gate 1's expected tail
string changes accordingly (see `GATE1_EXPECTED_TAIL`); the closed-think-pair and
no-lone-`<think>` checks are unaffected.

`thinking=False` is therefore the ONLY K2.5-specific change to the prompt: the chat
template renders `<|im_assistant|>assistant<|im_middle|><think></think>` for the
generation prompt and for every few-shot assistant turn, i.e. reasoning is off
everywhere.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

# scripts/ on the path so `extract.…` and `data.…` resolve
_SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from extract.extract_hidden_states import build_messages_for_condition  # noqa: E402

# Kimi K2.5 special tokens (verified against tokenizer_config.json / RUNBOOK §0)
THINK_OPEN = 163606       # <think>
THINK_CLOSE = 163607      # </think>
IM_MIDDLE = 163601        # <|im_middle|>

# What the rendered prompt must end with under user-turn filler placement.
GATE1_EXPECTED_TAIL = "<|im_assistant|>assistant<|im_middle|><think></think>"


def build_prompt_k25(
    tokenizer,
    few_shot_items: list,
    target: dict,
    filler_type: str,
    k: int,
    rng: random.Random | None = None,
    dataset_type: str = "varbind",
) -> tuple[str, list[int]]:
    """Render one instant-mode prompt. Returns (text, token_ids).

    The message list is built by the shared repo builder, so the user turn is
    byte-identical to the V3 varbind prompt; only the chat markers differ.
    """
    messages = build_messages_for_condition(
        few_shot_items, target, filler_type, k, rng=rng, dataset_type=dataset_type
    )
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, thinking=False
    )
    # add_special_tokens=False: the chat template already emits every marker, and
    # a tokenizer-injected BOS would shift every position by one (see
    # project_fp8_quantization_check for the pos_001 -> pos_000 bug that caused).
    ids = tokenizer.encode(text, add_special_tokens=False)
    return text, ids


def check_thinking_suppressed(ids: list[int]) -> tuple[bool, str]:
    """Gate 1: every <think> is immediately closed, and none is left open.

    Returns (ok, message). Independent of where the filler sits.
    """
    n_open = ids.count(THINK_OPEN)
    n_close = ids.count(THINK_CLOSE)
    if n_open == 0:
        return False, f"no <think> token ({THINK_OPEN}) found at all — template did not render"
    if n_open != n_close:
        return False, f"unbalanced think tokens: {n_open} open vs {n_close} close"
    lone = [i for i, t in enumerate(ids)
            if t == THINK_OPEN and (i + 1 >= len(ids) or ids[i + 1] != THINK_CLOSE)]
    if lone:
        return False, (f"lone <think> at position(s) {lone[:5]} not immediately "
                       f"followed by </think> ({THINK_CLOSE})")
    return True, f"{n_open} closed <think></think> pair(s), no open block"


def check_prompt_tail(text: str) -> tuple[bool, str]:
    """Gate 1 (tail): prompt ends with the instant-mode generation prompt."""
    if text.endswith(GATE1_EXPECTED_TAIL):
        return True, f"ends with {GATE1_EXPECTED_TAIL!r}"
    return False, f"expected tail {GATE1_EXPECTED_TAIL!r}, got {text[-90:]!r}"


def check_v3_scaffold(text: str, k: int) -> tuple[bool, str]:
    """The `Filler: …\\n\\nAnswer:` scaffold is present in the user turn and is
    followed by `<|im_end|>` — i.e. it did NOT leak into the assistant turn."""
    needle = "\n\nAnswer:<|im_end|>"
    if needle not in text:
        return False, f"{needle!r} not found — scaffold is not at the end of the user turn"
    if k > 0 and "\n\nFiller: " not in text:
        return False, "'\\n\\nFiller: ' not found even though k > 0"
    if k == 0 and "\n\nFiller: " in text:
        return False, "'\\n\\nFiller: ' present in a k=0 (baseline) prompt"
    return True, "scaffold closes the user turn (V3-identical)"
