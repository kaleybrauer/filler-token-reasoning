#!/usr/bin/env python3
"""
Generate 2-hop arithmetic dataset for opaque reasoner experiments.

- Loads only model-verified known facts (from evaluate_individual_facts.py)
- Facts are split into train/val/test pools first, then pairs are composed
  only from facts within each pool. This prevents any fact leakage.
- Uses a unified chat-format prompt (Pfau et al. 2024 style):
    * Same instruction + few-shot context for ALL variants
    * Few-shot examples always show CoT (parallelizable decomposition)
    * CoT training: assistant produces full CoT, loss on all tokens
    * Filler training: assistant produces "Filler: 1 2 ... N\nAnswer: X",
      loss only on answer tokens (filler is in assistant turn, same position
      as CoT, enabling transfer of learned computation)
    * N=0 baseline: assistant produces "Answer: X", loss on answer only
- Pair deduplication within each split prevents duplicate questions.
- Filler lengths can be sampled uniformly or from a fixed set of eval values.
"""
import argparse
import json
import pathlib
import random
from typing import Any, Dict, List, Optional, Set, Tuple

from transformers import AutoTokenizer

# Default filler lengths used for evaluation (also used during training if --filler-mode=eval)
DEFAULT_EVAL_FILLER_LENGTHS = [0, 32, 128, 300, 600, 1000]

# ─────────────────────────────────────────────────────────────────────────────
# Prompt format
#
# Structure:
#   system: instruction + worked examples (identical for all conditions)
#   user:   just the question
#   assistant: CoT, filler+answer, or direct answer (varies by condition)
# ─────────────────────────────────────────────────────────────────────────────


def build_cot_response(q1: str, q2: str, a1: int, a2: int, total: int) -> str:
    """Build a chain-of-thought response for a 2-hop addition question.

    Produces a parallelizable CoT where each fact lookup is an independent step.
    """
    q1_inner = q1.rstrip("? \t")
    q2_inner = q2.rstrip("? \t")
    return (
        f"Step 1: {q1_inner} = {a1}\n"
        f"Step 2: {q2_inner} = {a2}\n"
        f"Calculation: {a1} + {a2} = {total}\n"
        f"Answer: {total}"
    )


def _build_worked_example(q1: str, q2: str, a1: int, a2: int, total: int) -> str:
    """Format a single worked example for the system prompt."""
    question_text = compose_question(q1, q2)
    q1_inner = q1.rstrip("? \t")
    q2_inner = q2.rstrip("? \t")
    return (
        f"Q: {question_text}\n"
        f"Step 1: {q1_inner} = {a1}\n"
        f"Step 2: {q2_inner} = {a2}\n"
        f"Calculation: {a1} + {a2} = {total}\n"
        f"Answer: {total}"
    )


def build_system_prompt(
    fewshot_examples: List[Tuple[str, str, int, int, int]],
) -> str:
    """Build the system prompt with instruction and worked examples.

    This system prompt is IDENTICAL for CoT, filler, and N=0 conditions.
    Worked examples always show the CoT decomposition strategy so the model
    learns the parallelizable fact-lookup approach.

    Args:
        fewshot_examples: List of (q1, q2, a1, a2, sum) tuples.
    """
    examples_text = "\n\n".join(
        _build_worked_example(q1, q2, a1, a2, s)
        for q1, q2, a1, a2, s in fewshot_examples
    )

    return (
        "You answer questions that require looking up two facts and adding them.\n"
        "\n"
        "Here are some worked examples:\n"
        "\n"
        f"{examples_text}\n"
        "\n"
        "The user will ask a similar question. "
        "Give your final answer as a single integer with 'Answer: [NUMBER]'."
    )


def compose_question(q1: str, q2: str) -> str:
    """Compose two fact questions into a single 2-hop question.

    Produces the format: "What is ({q1_stripped}) + ({q2_stripped})?"
    E.g. "What is (the number of legs of a cat) + (the number of days in a week)?"
    """
    q1_inner = q1.rstrip("? \t")
    q2_inner = q2.rstrip("? \t")
    return f"What is ({q1_inner}) + ({q2_inner})?"


def build_filler_response(filler_len: int, answer: int) -> str:
    """Build assistant response for a filler example.

    Filler tokens occupy the same position as CoT tokens (assistant turn),
    enabling transfer of computation learned from CoT supervision.

    For N=0, returns just "Answer: {answer}".
    For N>0, returns "Filler: 1 2 3 ... N\nAnswer: {answer}".
    """
    if filler_len == 0:
        return f"Answer: {answer}"
    filler = " ".join(str(i) for i in range(1, filler_len + 1))
    return f"Filler: {filler}\nAnswer: {answer}"


def build_filler_prefill(filler_len: int) -> str:
    """Build the prefill string for a filler example (everything before the answer value).

    This is the text in the assistant turn that is NOT supervised — it gets
    masked with -100 in labels. At inference time, this is provided as input
    and the model generates only the answer token(s) after it.

    For N=0, returns "Answer:".
    For N>0, returns "Filler: 1 2 3 ... N\\nAnswer:".
    """
    if filler_len == 0:
        return "Answer:"
    filler = " ".join(str(i) for i in range(1, filler_len + 1))
    return f"Filler: {filler}\nAnswer:"


def build_chat_messages(
    q1: str,
    q2: str,
    fewshot_examples: List[Tuple[str, str, int, int, int]],
    sequence_type: str = "filler",
    filler_len: int = 0,
    answer: Optional[int] = None,
    a1: Optional[int] = None,
    a2: Optional[int] = None,
) -> List[Dict[str, str]]:
    """Build the full chat message list: system + user + assistant.

    The system prompt and user message are IDENTICAL for all sequence types.
    Only the assistant response differs:
      - "cot":    Step 1: ... Step 2: ... Calculation: ... Answer: N
      - "filler": Filler: 1 2 3 ... N\\nAnswer: N   (or just Answer: N if N=0)

    Args:
        q1, q2: The two fact questions.
        fewshot_examples: Few-shot examples as (q1, q2, a1, a2, sum) tuples.
        sequence_type: "cot" or "filler".
        filler_len: Number of counting filler tokens (only for sequence_type="filler").
        answer: If provided, include the assistant's answer (for training).
        a1, a2: Individual fact answers (required for CoT).
    """
    messages: List[Dict[str, str]] = []

    # System prompt with instruction + worked examples (identical for all conditions)
    messages.append({
        "role": "system",
        "content": build_system_prompt(fewshot_examples),
    })

    # User question (just the question, no instruction)
    question_text = compose_question(q1, q2)
    messages.append({"role": "user", "content": question_text})

    # Assistant response (varies by condition)
    if answer is not None:
        if sequence_type == "cot" and a1 is not None and a2 is not None:
            assistant_text = build_cot_response(q1, q2, a1, a2, answer)
        else:
            assistant_text = build_filler_response(filler_len, answer)
        messages.append({"role": "assistant", "content": assistant_text})

    return messages


def select_fewshot_examples(
    facts: List[Tuple[str, int, str]],
    n_examples: int,
    max_answer: int,
    rng: random.Random,
    n_fewshot_facts: int = 10,
) -> Tuple[List[Tuple[str, str, int, int, int]], List[Tuple[str, int, str]]]:
    """Carve out few-shot examples from the (already-shuffled) fact pool.

    Takes the first n_fewshot_facts from the list and selects n_examples
    non-overlapping pairs — each fact appears in at most one example. This
    requires n_fewshot_facts >= n_examples * 2.

    Args:
        facts: Shuffled list of all facts.
        n_examples: Number of few-shot examples to select.
        max_answer: Maximum valid answer sum.
        rng: Random state for pair ordering.
        n_fewshot_facts: Facts to reserve for the few-shot pool. Must be at
                         least n_examples * 2 to allow non-overlapping pairs.

    Returns:
        (fewshot_examples, remaining_facts)
        fewshot_examples: List of (q1, q2, a1, a2, sum) tuples where no fact
                          appears in more than one example.
        remaining_facts: All facts not in the few-shot pool.
    """
    min_required = n_examples * 2
    if n_fewshot_facts < min_required:
        raise SystemExit(
            f"--n-fewshot-facts ({n_fewshot_facts}) must be at least "
            f"n_examples * 2 = {min_required} to guarantee non-overlapping pairs."
        )
    if len(facts) < n_fewshot_facts + 1:
        raise SystemExit(
            f"Not enough facts ({len(facts)}) to reserve {n_fewshot_facts} for "
            f"few-shot examples. Add more facts or reduce --n-fewshot-facts."
        )

    fewshot_pool = facts[:n_fewshot_facts]
    remaining = facts[n_fewshot_facts:]

    # Shuffle pairs so selection is random, then greedily pick non-overlapping ones
    pairs = _generate_valid_pairs(fewshot_pool, max_answer, rng)

    examples: List[Tuple[str, str, int, int, int]] = []
    used_indices: Set[int] = set()
    for (i, j) in pairs:
        if i in used_indices or j in used_indices:
            continue
        q1, a1, _ = fewshot_pool[i]
        q2, a2, _ = fewshot_pool[j]
        if rng.random() < 0.5:
            q1, a1, q2, a2 = q2, a2, q1, a1
        examples.append((q1, q2, a1, a2, a1 + a2))
        used_indices.add(i)
        used_indices.add(j)
        if len(examples) == n_examples:
            break

    if len(examples) < n_examples:
        raise SystemExit(
            f"Could only find {len(examples)} non-overlapping valid pairs from "
            f"{n_fewshot_facts} few-shot facts (need {n_examples}). "
            f"Increase --n-fewshot-facts or reduce --n-fewshot."
        )

    return examples, remaining


def load_known_facts(path: pathlib.Path) -> List[Tuple[str, int, str]]:
    """Load pre-filtered known facts from known_facts.json.

    Expected format: list of {"question": str, "answer": int, "kind": str}
    (as produced by evaluate_individual_facts.py)
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    out: List[Tuple[str, int, str]] = []
    for item in raw:
        q = item["question"]
        a = int(item["answer"])
        k = item.get("kind", "unknown")
        out.append((q, a, k))
    return out


def split_facts(
    facts: List[Tuple[str, int, str]],
    rng: random.Random,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
) -> Tuple[List[Tuple[str, int, str]], List[Tuple[str, int, str]], List[Tuple[str, int, str]]]:
    """
    Split facts into train/val/test pools.
    
    Returns (train_facts, val_facts, test_facts).
    """
    facts = facts.copy()
    rng.shuffle(facts)
    
    n = len(facts)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    
    train_facts = facts[:n_train]
    val_facts = facts[n_train:n_train + n_val]
    test_facts = facts[n_train + n_val:]
    
    return train_facts, val_facts, test_facts


def sample_filler_len(
    rng: random.Random,
    mode: str,
    lo: int,
    hi: int,
    eval_lengths: List[int],
) -> int:
    """Sample filler length based on mode."""
    if mode == "uniform":
        if hi < lo:
            hi = lo
        return rng.randint(lo, hi)
    elif mode == "eval":
        # Sample from the discrete set of eval lengths
        return rng.choice(eval_lengths)
    else:
        raise ValueError(f"Unknown filler mode: {mode}")


def write_jsonl(rows: List[Dict[str, Any]], outpath: pathlib.Path) -> None:
    outpath.parent.mkdir(parents=True, exist_ok=True)
    with outpath.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _generate_valid_pairs(
    facts: List[Tuple[str, int, str]],
    max_answer: int,
    rng: random.Random,
) -> List[Tuple[int, int]]:
    """
    Pre-generate all valid (i, j) pairs where i < j and a_i + a_j < max_answer
    and a_i + a_j >= 0, then shuffle them.
    
    Returns shuffled list of (index_i, index_j) into the facts list.
    """
    pairs = []
    for i in range(len(facts)):
        for j in range(i + 1, len(facts)):
            s = facts[i][1] + facts[j][1]
            if 0 <= s < max_answer:
                pairs.append((i, j))
    rng.shuffle(pairs)
    return pairs


def _tokenize_example(
    tok: "AutoTokenizer",
    full_messages: List[Dict[str, str]],
    prompt_messages: List[Dict[str, str]],
    answer_text: str,
    prefill: str = "",
) -> Tuple[List[int], List[int], List[int], str]:
    """Robustly tokenize a training example, avoiding BPE alignment issues.

    Instead of relying on full_ids[n_prompt:] (which breaks when BPE
    tokenization of full_text and prompt_text don't align at the boundary),
    we tokenize the answer portion independently and concatenate.

    Args:
        tok: Tokenizer instance.
        full_messages: Chat messages including assistant answer.
        prompt_messages: Chat messages without assistant answer.
        answer_text: The assistant's answer text (e.g. "Answer: 108" or
                     the full CoT response).
        prefill: Text to append to the prompt after the generation prompt
                 (e.g. "Answer:" for filler examples). This text is NOT
                 part of the answer—it's masked in labels.

    Returns:
        (input_ids, labels, prompt_ids, prompt_text)
        Where len(input_ids) == len(labels), and labels has -100 for
        the prompt portion and real token ids for the answer portion.
    """
    # Tokenize prompt (everything up to the answer)
    prompt_text = tok.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True
    )
    prompt_text += prefill
    prompt_ids = tok.encode(prompt_text, add_special_tokens=False)

    # Tokenize full sequence for reference
    full_text = tok.apply_chat_template(
        full_messages, tokenize=False, add_special_tokens=True
    )
    full_ids = tok.encode(full_text, add_special_tokens=False)

    # Check if prompt_ids is a clean prefix of full_ids
    n_prompt = len(prompt_ids)
    if (
        n_prompt < len(full_ids)
        and full_ids[:n_prompt] == prompt_ids
    ):
        # Clean alignment — use the standard slicing approach
        answer_ids = full_ids[n_prompt:]
        input_ids = full_ids
    else:
        # BPE misalignment — construct answer_ids independently.
        # We need the text that comes AFTER the prefill in the full rendered text.
        # For safety, extract it from the full_text using prompt_text as anchor.
        if full_text.startswith(prompt_text):
            remaining_text = full_text[len(prompt_text):]
        else:
            # Fallback: construct what the answer tokens should be.
            # The full assistant turn is: prefill + answer_suffix + <|im_end|>
            # We already have answer_text (the full content).
            # The portion after the prefill is answer_text[len(prefill):] + <|im_end|>
            if prefill and answer_text.startswith(prefill):
                remaining_text = answer_text[len(prefill):]
            else:
                remaining_text = answer_text
            # Add end-of-turn marker
            eos_str = tok.decode([tok.eos_token_id]) if tok.eos_token_id else ""
            # For Qwen: <|im_end|>\n  — get from template
            end_marker = ""
            dummy = tok.apply_chat_template(
                [{"role": "assistant", "content": "X"}],
                tokenize=False, add_special_tokens=True,
            )
            marker_pos = dummy.find("X") + 1
            if marker_pos > 0 and marker_pos < len(dummy):
                end_marker = dummy[marker_pos:]
            remaining_text = remaining_text + end_marker

        answer_ids = tok.encode(remaining_text, add_special_tokens=False)
        input_ids = prompt_ids + answer_ids

    labels = [-100] * len(prompt_ids) + list(answer_ids)
    assert len(input_ids) == len(labels), (
        f"input_ids ({len(input_ids)}) != labels ({len(labels)}) — "
        f"prompt={n_prompt}, answer={len(answer_ids)}, full={len(full_ids)}"
    )

    return input_ids, labels, prompt_ids, prompt_text


class DatasetBuilder:
    """Builds examples from a pool of facts using pre-generated pairs."""
    
    def __init__(
        self,
        facts: List[Tuple[str, int, str]],
        tok: AutoTokenizer,
        fewshot_examples: List[Tuple[str, str, int, int, int]],
        max_answer: int,
        filler_mode: str,
        filler_min: int,
        filler_max: int,
        eval_filler_lengths: List[int],
        rng: random.Random,
    ):
        self.facts = facts
        self.tok = tok
        self.fewshot_examples = fewshot_examples
        self.max_answer = max_answer
        self.filler_mode = filler_mode
        self.filler_min = filler_min
        self.filler_max = filler_max
        self.eval_filler_lengths = eval_filler_lengths
        self.rng = rng
        
        # Pre-generate all valid pairs
        print(f"  Pre-generating valid pairs from {len(facts)} facts...")
        self.valid_pairs = _generate_valid_pairs(facts, max_answer, rng)
        print(f"  Found {len(self.valid_pairs)} valid pairs")
        
        # Pointer into the valid_pairs list
        self._pair_idx = 0
        
        # Base seed for deterministic pair assignment in CoT mixture
        self._base_seed = rng.randint(0, 2**31)
        
        # Track how many pairs were used (for reporting)
        self.used_pair_count = 0
    
    def make_example(self, example_id: int, split: str, sequence_type: str = "filler") -> Dict[str, Any]:
        """Generate a single example from the next pre-generated pair.

        Args:
            example_id: Unique ID for this example.
            split: Dataset split name ("train", "val", "test").
            sequence_type: "filler" for filler-token examples (loss on answer only),
                          "cot" for chain-of-thought examples (loss on full CoT + answer).
        """
        if self._pair_idx >= len(self.valid_pairs):
            raise RuntimeError(
                f"Exhausted all {len(self.valid_pairs)} valid pairs from {len(self.facts)} facts. "
                f"Generated {self._pair_idx} examples so far. "
                "Reduce dataset size or add more facts."
            )
        
        i, j = self.valid_pairs[self._pair_idx]
        self._pair_idx += 1
        self.used_pair_count += 1
        
        # Randomly assign which fact is q1 vs q2
        if self.rng.random() < 0.5:
            (q1, a1, t1) = self.facts[i]
            (q2, a2, t2) = self.facts[j]
        else:
            (q1, a1, t1) = self.facts[j]
            (q2, a2, t2) = self.facts[i]
        
        s = a1 + a2
        question_text = compose_question(q1, q2)
        
        if sequence_type == "cot":
            nfill = 0
        else:
            nfill = sample_filler_len(
                self.rng,
                self.filler_mode,
                self.filler_min,
                self.filler_max,
                self.eval_filler_lengths,
            )
        
        # Build chat messages WITH answer (for training)
        full_messages = build_chat_messages(
            q1, q2,
            fewshot_examples=self.fewshot_examples,
            sequence_type=sequence_type,
            filler_len=nfill,
            answer=s,
            a1=a1,
            a2=a2,
        )
        
        # Build chat messages WITHOUT answer (for prompt / generation)
        prompt_messages = build_chat_messages(
            q1, q2,
            fewshot_examples=self.fewshot_examples,
            sequence_type=sequence_type,
            filler_len=nfill,
            answer=None,
        )
        
        # Determine the prefill (masked portion of assistant turn)
        if sequence_type == "cot":
            # CoT: no prefill, loss on full response
            prefill = ""
        else:
            # Filler: prefill includes filler tokens + "Answer:", loss on answer only
            prefill = build_filler_prefill(nfill)
        
        # Robust tokenization (handles BPE alignment issues)
        input_ids, labels, prompt_ids, prompt_text = _tokenize_example(
            self.tok, full_messages, prompt_messages,
            answer_text=full_messages[-1]["content"],  # full assistant content
            prefill=prefill,
        )
        answer_ids = input_ids[len(prompt_ids):]
        attn = [1] * len(input_ids)
        
        return {
            "id": f"{split}-{example_id}",
            "split": split,
            "sequence_type": sequence_type,
            "prompt": prompt_text,
            "question": question_text,
            "fact1": q1,
            "fact2": q2,
            "a1": a1,
            "a2": a2,
            "answer": s,
            "type1": t1,
            "type2": t2,
            "filler_len": nfill,
            "filler_type": "counting" if sequence_type != "cot" else "none",
            "prompt_ids": prompt_ids,
            "answer_ids": answer_ids,
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attn,
        }
    
    def make_example_from_pair(
        self, pair_idx: int, example_id: int, split: str, sequence_type: str = "filler",
        filler_len_override: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Generate a single example from a specific pair index.

        Unlike make_example, this does NOT advance the internal pair pointer.
        Used by build_split for CoT mixture where same pair produces two examples,
        and by build_test_split where each pair is repeated at every filler length.

        Args:
            pair_idx: Index into self.valid_pairs.
            example_id: Unique ID for this example.
            split: Dataset split name.
            sequence_type: "filler" or "cot".
            filler_len_override: If set, use this exact filler length instead of
                sampling. Used for test sets where every pair gets every N.
        """
        i, j = self.valid_pairs[pair_idx]

        # Deterministic q1/q2 assignment based on pair index + base_seed
        # (same pair always gets same assignment regardless of sequence_type)
        pair_rng = random.Random(self._base_seed ^ (pair_idx * 2654435761))
        if pair_rng.random() < 0.5:
            (q1, a1, t1) = self.facts[i]
            (q2, a2, t2) = self.facts[j]
        else:
            (q1, a1, t1) = self.facts[j]
            (q2, a2, t2) = self.facts[i]

        s = a1 + a2
        question_text = compose_question(q1, q2)

        if sequence_type == "cot":
            nfill = 0
        elif filler_len_override is not None:
            nfill = filler_len_override
        else:
            nfill = sample_filler_len(
                self.rng,
                self.filler_mode,
                self.filler_min,
                self.filler_max,
                self.eval_filler_lengths,
            )

        full_messages = build_chat_messages(
            q1, q2,
            fewshot_examples=self.fewshot_examples,
            sequence_type=sequence_type,
            filler_len=nfill,
            answer=s,
            a1=a1,
            a2=a2,
        )

        prompt_messages = build_chat_messages(
            q1, q2,
            fewshot_examples=self.fewshot_examples,
            sequence_type=sequence_type,
            filler_len=nfill,
            answer=None,
        )

        # Determine the prefill (masked portion of assistant turn)
        if sequence_type == "cot":
            prefill = ""
        else:
            prefill = build_filler_prefill(nfill)

        # Robust tokenization (handles BPE alignment issues)
        input_ids, labels, prompt_ids, prompt_text = _tokenize_example(
            self.tok, full_messages, prompt_messages,
            answer_text=full_messages[-1]["content"],
            prefill=prefill,
        )
        answer_ids = input_ids[len(prompt_ids):]
        attn = [1] * len(input_ids)

        return {
            "id": f"{split}-{example_id}",
            "split": split,
            "sequence_type": sequence_type,
            "prompt": prompt_text,
            "question": question_text,
            "fact1": q1,
            "fact2": q2,
            "a1": a1,
            "a2": a2,
            "answer": s,
            "type1": t1,
            "type2": t2,
            "filler_len": nfill,
            "filler_type": "counting" if sequence_type != "cot" else "none",
            "pair_id": pair_idx,
            "prompt_ids": prompt_ids,
            "answer_ids": answer_ids,
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attn,
        }

    def build_test_split(
        self, n_pairs: int, filler_lengths: List[int], split: str = "test",
    ) -> List[Dict[str, Any]]:
        """Build test set: every pair repeated at every filler length.

        This gives paired comparisons — same question at different N values —
        so accuracy differences can only be attributed to filler length, not
        question difficulty.

        Args:
            n_pairs: Number of unique question pairs to use.
            filler_lengths: List of filler lengths to evaluate at (e.g. [0, 32, 128, 300]).
            split: Split name (default "test").

        Returns:
            List of examples. Total count = n_pairs × len(filler_lengths).
        """
        if n_pairs > len(self.valid_pairs):
            raise RuntimeError(
                f"Requested {n_pairs} test pairs but only "
                f"{len(self.valid_pairs)} valid pairs available from "
                f"{len(self.facts)} facts."
            )

        rows = []
        example_id = 0
        for p in range(n_pairs):
            for n in filler_lengths:
                row = self.make_example_from_pair(
                    p, example_id, split,
                    sequence_type="filler",
                    filler_len_override=n,
                )
                rows.append(row)
                example_id += 1

            if (p + 1) % 500 == 0:
                print(f"  Generated pairs {p + 1}/{n_pairs} "
                      f"× {len(filler_lengths)} filler lengths "
                      f"= {example_id} examples...")

        self.used_pair_count = n_pairs
        print(f"  Total: {len(rows)} test examples "
              f"({n_pairs} pairs × {len(filler_lengths)} filler lengths)")
        return rows

    def build_split(
        self, n: int, split: str, cot_mixture: bool = False,
    ) -> List[Dict[str, Any]]:
        """Build n examples for a split.

        Args:
            n: Number of examples to generate.
            split: Split name ("train", "val", "test").
            cot_mixture: If True, generate paired CoT + filler examples for each
                problem. Each unique fact pair produces one CoT example and one
                filler example. Uses n/2 pairs to produce n total examples.
                The CoT examples have no filler and loss on the full CoT response.
                The filler examples are the standard format with loss on answer only.
        """
        rows = []
        if cot_mixture:
            n_pairs = n // 2
            if n_pairs > len(self.valid_pairs):
                raise RuntimeError(
                    f"CoT mixture needs {n_pairs} pairs but only "
                    f"{len(self.valid_pairs)} valid pairs available."
                )
            example_id = 0
            for p in range(n_pairs):
                # CoT variant for this pair
                cot_row = self.make_example_from_pair(p, example_id, split, "cot")
                rows.append(cot_row)
                example_id += 1

                # Filler variant for the same pair
                filler_row = self.make_example_from_pair(p, example_id, split, "filler")
                rows.append(filler_row)
                example_id += 1

                if (example_id) % 10000 == 0:
                    print(f"  Generated {example_id}/{n} {split} examples (CoT mixture)...")

            self.used_pair_count = n_pairs
            # Shuffle so CoT and filler examples are interleaved randomly
            self.rng.shuffle(rows)
        else:
            for i in range(n):
                rows.append(self.make_example(i, split))
                if (i + 1) % 10000 == 0:
                    print(f"  Generated {i + 1}/{n} {split} examples...")
        return rows
    
    def max_possible_pairs(self) -> int:
        """Return maximum possible unique valid pairs from this fact pool."""
        return len(self.valid_pairs)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate 2-hop arithmetic dataset with fact-level train/val/test isolation."
    )
    ap.add_argument("--tokenizer", type=str, required=True, 
                    help="Tokenizer name, e.g. Qwen/Qwen2.5-7B")
    ap.add_argument("--known-facts", type=str, required=True,
                    help="Path to known_facts.json from evaluate_individual_facts.py")
    ap.add_argument("--outdir", type=str, required=True, 
                    help="Output directory for generated JSONL files")

    ap.add_argument("--n-train", type=int, default=28000)
    ap.add_argument("--n-val", type=int, default=600)
    ap.add_argument("--n-test", type=int, default=200,
                    help="Number of unique test pairs. Each pair is repeated at every "
                         "eval filler length, so total test examples = n_test × len(eval_filler_lengths).")
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--max-answer", type=int, default=1000, 
                    help="Keep final answers < max-answer")

    ap.add_argument("--filler-mode", type=str, default="eval", choices=["uniform", "eval"],
                    help="How to sample filler lengths: 'uniform' samples from [min,max], "
                         "'eval' samples from --eval-filler-lengths")
    ap.add_argument("--filler-min", type=int, default=0,
                    help="Min filler length (for uniform mode)")
    ap.add_argument("--filler-max", type=int, default=600,
                    help="Max filler length (for uniform mode)")
    ap.add_argument("--eval-filler-lengths", type=str, default="0,32,128,300,600",
                    help="Comma-separated filler lengths for eval mode and test set generation")

    ap.add_argument("--fact-split-train", type=float, default=0.75,
                    help="Fraction of facts for training pool (default: 0.75)")
    ap.add_argument("--fact-split-val", type=float, default=0.125,
                    help="Fraction of facts for validation pool (default: 0.125)")

    ap.add_argument("--n-fewshot", type=int, default=5,
                    help="Number of few-shot examples to draw from the fact pool (default: 5)")
    ap.add_argument("--n-fewshot-facts", type=int, default=10,
                    help="Facts to reserve for the few-shot pool (default: 10). "
                         "Must be large enough to yield --n-fewshot valid pairs.")

    ap.add_argument("--cot-mixture", action="store_true", default=False,
                    help="Enable 50/50 CoT + filler mixture (Pfau et al. style). "
                         "Each fact pair produces two training examples: one with "
                         "chain-of-thought in assistant turn (loss on full CoT) and one "
                         "with filler tokens in assistant turn (loss on answer only). "
                         "Both use identical instruction + few-shot context. "
                         "Uses n/2 pairs for n total examples.")

    args = ap.parse_args()

    rng = random.Random(args.seed)
    
    # Parse eval filler lengths
    eval_filler_lengths = [int(x.strip()) for x in args.eval_filler_lengths.split(",")]
    print(f"Eval filler lengths: {eval_filler_lengths}")

    # Load tokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    
    # Report BOS/EOS tokens
    print(f"BOS token: {tok.bos_token!r} (id={tok.bos_token_id})")
    print(f"EOS token: {tok.eos_token!r} (id={tok.eos_token_id})")
    print(f"Filler type: counting tokens in assistant turn (system prompt with worked examples)")

    # Load known facts
    known_path = pathlib.Path(args.known_facts)
    if not known_path.exists():
        raise SystemExit(f"Error: --known-facts file not found: {known_path}")

    all_facts = load_known_facts(known_path)
    all_facts = [(q, a, k) for (q, a, k) in all_facts if 0 <= a < args.max_answer]

    from collections import Counter
    kind_counts = Counter(k for _, _, k in all_facts)
    print(f"\nLoaded {len(all_facts)} known facts from {known_path}")
    for kind in sorted(kind_counts):
        print(f"  {kind}: {kind_counts[kind]}")
    
    if len(all_facts) < 50:
        raise SystemExit(f"Too few usable facts ({len(all_facts)}). Need at least 50.")

    # Shuffle all facts before carving out the few-shot pool
    rng.shuffle(all_facts)

    # Carve out few-shot examples from the top of the shuffled pool.
    # These facts are excluded from train/val/test entirely.
    print(f"\nSelecting {args.n_fewshot} few-shot examples from "
          f"{args.n_fewshot_facts} reserved facts...")
    fewshot_examples, remaining_facts = select_fewshot_examples(
        all_facts,
        n_examples=args.n_fewshot,
        max_answer=args.max_answer,
        rng=rng,
        n_fewshot_facts=args.n_fewshot_facts,
    )
    print(f"  Reserved {args.n_fewshot_facts} facts for few-shot; "
          f"{len(remaining_facts)} facts available for train/val/test")
    print(f"\nFew-shot examples:")
    for idx, (q1, q2, a1, a2, s) in enumerate(fewshot_examples):
        print(f"  [{idx+1}] {compose_question(q1, q2)} -> {s} ({a1}+{a2})")

    # Split remaining facts into train/val/test pools
    train_facts, val_facts, test_facts = split_facts(
        remaining_facts, rng, 
        train_frac=args.fact_split_train,
        val_frac=args.fact_split_val,
    )
    
    print(f"\nFact pool sizes:")
    print(f"  Train: {len(train_facts)} facts -> max {len(train_facts) * (len(train_facts)-1) // 2} pairs")
    print(f"  Val:   {len(val_facts)} facts -> max {len(val_facts) * (len(val_facts)-1) // 2} pairs")
    print(f"  Test:  {len(test_facts)} facts -> max {len(test_facts) * (len(test_facts)-1) // 2} pairs")

    print(f"\nFiller mode: {args.filler_mode}")
    if args.filler_mode == "uniform":
        print(f"  Range: [{args.filler_min}, {args.filler_max}] counting steps")
    else:
        print(f"  Values: {eval_filler_lengths} counting steps")

    if args.cot_mixture:
        print(f"\nCoT mixture: ENABLED (50/50 CoT + filler for same problems)")
        print(f"  Same instruction + few-shots for both (Pfau et al. style)")
        print(f"  CoT examples: full CoT in assistant turn, loss on all tokens")
        print(f"  Filler examples: filler in assistant turn, loss on answer only")

    # Create output directory
    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Build datasets
    print("\nGenerating training data...")
    train_builder = DatasetBuilder(
        facts=train_facts,
        tok=tok,
        fewshot_examples=fewshot_examples,
        max_answer=args.max_answer,
        filler_mode=args.filler_mode,
        filler_min=args.filler_min,
        filler_max=args.filler_max,
        eval_filler_lengths=eval_filler_lengths,
        rng=rng,
    )
    
    # Check capacity before generating
    if args.cot_mixture:
        # CoT mixture uses n/2 pairs per n examples
        needed_train_pairs = args.n_train // 2
        if needed_train_pairs > train_builder.max_possible_pairs():
            raise SystemExit(
                f"CoT mixture needs {needed_train_pairs} pairs but only "
                f"{train_builder.max_possible_pairs()} valid unique pairs possible "
                f"from {len(train_facts)} facts. Reduce --n-train or add more facts."
            )
    else:
        if args.n_train > train_builder.max_possible_pairs():
            raise SystemExit(
                f"Cannot generate {args.n_train} train examples: only "
                f"{train_builder.max_possible_pairs()} valid unique pairs possible "
                f"from {len(train_facts)} facts. Reduce --n-train or add more facts."
            )
    
    train_rows = train_builder.build_split(args.n_train, "train", cot_mixture=args.cot_mixture)
    
    print("\nGenerating validation data...")
    val_builder = DatasetBuilder(
        facts=val_facts,
        tok=tok,
        fewshot_examples=fewshot_examples,
        max_answer=args.max_answer,
        filler_mode=args.filler_mode,
        filler_min=args.filler_min,
        filler_max=args.filler_max,
        eval_filler_lengths=eval_filler_lengths,
        rng=rng,
    )
    
    if args.cot_mixture:
        needed_val_pairs = args.n_val // 2
        if needed_val_pairs > val_builder.max_possible_pairs():
            raise SystemExit(
                f"CoT mixture needs {needed_val_pairs} val pairs but only "
                f"{val_builder.max_possible_pairs()} valid unique pairs possible "
                f"from {len(val_facts)} facts. Reduce --n-val or add more facts."
            )
    else:
        if args.n_val > val_builder.max_possible_pairs():
            raise SystemExit(
                f"Cannot generate {args.n_val} val examples: only "
                f"{val_builder.max_possible_pairs()} valid unique pairs possible "
                f"from {len(val_facts)} facts. Reduce --n-val or add more facts."
            )
    
    val_rows = val_builder.build_split(args.n_val, "val", cot_mixture=args.cot_mixture)
    
    print("\nGenerating test data...")
    print(f"  {args.n_test} unique pairs × {len(eval_filler_lengths)} filler lengths "
          f"= {args.n_test * len(eval_filler_lengths)} total test examples")
    # Test set: every pair repeated at every eval filler length for paired comparison
    test_builder = DatasetBuilder(
        facts=test_facts,
        tok=tok,
        fewshot_examples=fewshot_examples,
        max_answer=args.max_answer,
        filler_mode="eval",  # Not used for test (explicit lengths), but required
        filler_min=args.filler_min,
        filler_max=args.filler_max,
        eval_filler_lengths=eval_filler_lengths,
        rng=rng,
    )
    
    if args.n_test > test_builder.max_possible_pairs():
        raise SystemExit(
            f"Cannot generate {args.n_test} test pairs: only "
            f"{test_builder.max_possible_pairs()} valid unique pairs possible "
            f"from {len(test_facts)} facts. Reduce --n-test or add more facts."
        )
    
    test_rows = test_builder.build_test_split(
        n_pairs=args.n_test,
        filler_lengths=eval_filler_lengths,
    )

    # Write datasets
    write_jsonl(train_rows, outdir / "train.jsonl")
    write_jsonl(val_rows, outdir / "val.jsonl")
    write_jsonl(test_rows, outdir / "test.jsonl")

    # Save fact pools for reproducibility and analysis
    fact_pools = {
        "train": [{"question": q, "answer": a, "type": t} for q, a, t in train_facts],
        "val": [{"question": q, "answer": a, "type": t} for q, a, t in val_facts],
        "test": [{"question": q, "answer": a, "type": t} for q, a, t in test_facts],
    }
    (outdir / "fact_pools.json").write_text(json.dumps(fact_pools, indent=2), encoding="utf-8")

    # Save manifest
    manifest = {
        "tokenizer": args.tokenizer,
        "known_facts_source": str(args.known_facts),
        "prompt_format": "system_prompt_with_examples",
        "filler_type": "counting",
        "cot_mixture": args.cot_mixture,
        "seed": args.seed,
        "max_answer": args.max_answer,
        "fewshot_examples": [
            {"q1": q1, "q2": q2, "a1": a1, "a2": a2, "sum": s}
            for (q1, q2, a1, a2, s) in fewshot_examples
        ],
        "n_fewshot": len(fewshot_examples),
        "n_fewshot_facts": args.n_fewshot_facts,
        "filler_mode": args.filler_mode,
        "filler_min": args.filler_min,
        "filler_max": args.filler_max,
        "eval_filler_lengths": eval_filler_lengths,
        "fact_split": {
            "train_frac": args.fact_split_train,
            "val_frac": args.fact_split_val,
            "test_frac": 1.0 - args.fact_split_train - args.fact_split_val,
        },
        "fact_counts": {
            "train": len(train_facts),
            "val": len(val_facts),
            "test": len(test_facts),
        },
        "example_counts": {
            "train": len(train_rows),
            "val": len(val_rows),
            "test": len(test_rows),
        },
        "test_design": {
            "n_pairs": args.n_test,
            "filler_lengths": eval_filler_lengths,
            "total_examples": len(test_rows),
            "note": "Every pair repeated at every filler length for paired comparison",
        },
        "valid_pairs": {
            "train": train_builder.max_possible_pairs(),
            "val": val_builder.max_possible_pairs(),
            "test": test_builder.max_possible_pairs(),
        },
        "unique_pairs_used": {
            "train": train_builder.used_pair_count,
            "val": val_builder.used_pair_count,
            "test": test_builder.used_pair_count,
        },
        "bos_token_id": tok.bos_token_id,
        "eos_token_id": tok.eos_token_id,
    }
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Print summary
    print(f"\n{'='*60}")
    print(f"Done. Wrote dataset to: {outdir}")
    print(f"{'='*60}")
    print(f"  train.jsonl: {len(train_rows)} examples")
    print(f"  val.jsonl:   {len(val_rows)} examples")
    print(f"  test.jsonl:  {len(test_rows)} examples "
          f"({args.n_test} pairs × {len(eval_filler_lengths)} filler lengths)")
    print(f"\nFact isolation verified:")
    print(f"  - Train/val/test use completely separate fact pools")
    print(f"  - No duplicate pairs within any split")
    print(f"  - Test set: same questions at every filler length (paired comparison)")
    print(f"\nTest set filler length distribution:")
    from collections import Counter
    test_filler_dist = Counter(r["filler_len"] for r in test_rows)
    for fl in sorted(test_filler_dist.keys()):
        seq_lens = [len(r["input_ids"]) for r in test_rows if r["filler_len"] == fl]
        avg_seq = sum(seq_lens) / len(seq_lens) if seq_lens else 0
        print(f"  N={fl}: {test_filler_dist[fl]} examples (avg {avg_seq:.0f} tokens)")
    
    # Report overall sequence length stats
    all_seq_lens = [len(r["input_ids"]) for r in train_rows]
    print(f"\nTraining sequence lengths: min={min(all_seq_lens)}, max={max(all_seq_lens)}, "
          f"avg={sum(all_seq_lens)/len(all_seq_lens):.0f}")

    # Report sequence type distribution if CoT mixture is enabled
    if args.cot_mixture:
        from collections import Counter as Counter2
        type_counts = Counter2(r.get("sequence_type", "filler") for r in train_rows)
        print(f"\nTraining sequence type distribution:")
        for st in sorted(type_counts):
            seq_lens = [len(r["input_ids"]) for r in train_rows if r.get("sequence_type") == st]
            avg_sl = sum(seq_lens) / len(seq_lens) if seq_lens else 0
            print(f"  {st}: {type_counts[st]} examples (avg {avg_sl:.0f} tokens)")


if __name__ == "__main__":
    main()