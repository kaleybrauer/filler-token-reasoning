"""
filler_patching.py

Cross-example activation patching for dots_50 filler condition.

Replaces one problem's hidden states during filler with another problem's
hidden states and measures whether the answer shifts toward the donor's
computation. Y-matched pairs isolate the effect of A.

Key comparison: answer_prompt-only vs contiguous-from-k1. If answer_prompt-only
produces strong shifts, the final hidden state is sufficient and the KV cache
doesn't matter. If contiguous-from-k1 is needed, the model reads filler KV entries.

Usage:
    source /workspace/config/probing_env.sh
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python probing/scripts/filler_patching.py \
        --output-dir probing/patching_results_v2

Phases:
    0: Null validation (inject own states, verify answer preserved)
    1: All-layer contiguous patching (7 start positions × ~80 pairs)
    2: Layer-band patching (3 bands × 3 positions × 20 pairs)
"""

import argparse
import json
import random
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

# Patch: autoawq compatibility
import transformers.activations as _act
if not hasattr(_act, "PytorchGELUTanh"):
    _act.PytorchGELUTanh = _act.GELUTanh

from extract_hidden_states import (
    build_messages_for_condition,
    find_filler_boundaries,
    load_model,
    load_tokenizer,
)


# ==============================================================================
# Constants
# ==============================================================================

FILLER_K = 50
FILLER_TYPE = "dots"

# Start conditions: name -> filler-relative start index (0-based)
# "question_end" patches from question_end through answer_prompt
# "answer_prompt" patches only answer_prompt
START_CONDITIONS = {
    "question_end": "question_end",
    "k1": 0,     # first filler token
    "k5": 4,     # 5th filler token
    "k10": 9,
    "k25": 24,
    "k50": 49,   # last filler token
    "answer_prompt": "answer_prompt",
    "pre_answer": "pre_answer",  # after "Answer: " prefix, right before number
}

# Tokens to append to input for pre_answer condition ("Answer: ")
PRE_ANSWER_PREFIX = "Answer: "

PHASE2_CONDITIONS = ["k1", "k25", "answer_prompt", "pre_answer"]
LAYER_BANDS = {
    "L0_20": list(range(0, 21)),
    "L21_40": list(range(21, 41)),
    "L41_60": list(range(41, 61)),
}


# ==============================================================================
# Pair selection
# ==============================================================================

def select_pairs(
    categories_path: Path,
    max_pairs: int = 200,
    min_delta_a: int = 10,
) -> List[dict]:
    """
    Select Y-matched (source, target) pairs with large |ΔA|.

    Target: filler_helped (wrong baseline, correct with dots_50).
    Source: filler_helped OR both_correct (correct with dots_50).
    """
    with open(categories_path) as f:
        data = json.load(f)

    examples = data["examples"]

    # Use dots_50-specific categories
    cat_field = "category_dots_50"
    for e in examples:
        if cat_field not in e:
            cat_field = "category"
            break

    fh = [e for e in examples if e.get(cat_field) == "filler_helped"]
    bc = [e for e in examples if e.get(cat_field) == "both_correct"]
    source_pool = fh + bc

    src_by_y = defaultdict(list)
    for e in source_pool:
        src_by_y[e["x"]].append(e)

    pairs = []
    for t in fh:
        best_s = None
        best_da = 0
        for s in src_by_y[t["x"]]:
            if s["problem_idx"] == t["problem_idx"]:
                continue
            da = abs(s["fact_value"] - t["fact_value"])
            if da >= min_delta_a and da > best_da:
                best_s = s
                best_da = da
        if best_s:
            pairs.append({
                "source_idx": best_s["problem_idx"],
                "target_idx": t["problem_idx"],
                "source_A": best_s["fact_value"],
                "target_A": t["fact_value"],
                "Y": t["x"],
                "delta_A": best_s["fact_value"] - t["fact_value"],
                "abs_delta_A": best_da,
                "source_answer": best_s["fact_value"] + t["x"],
                "target_answer": t["fact_value"] + t["x"],
            })

    pairs.sort(key=lambda p: p["abs_delta_A"], reverse=True)
    pairs = pairs[:max_pairs]

    print(f"Selected {len(pairs)} Y-matched pairs "
          f"({len(set(p['source_idx'] for p in pairs))} unique sources)")
    if pairs:
        deltas = [p["abs_delta_A"] for p in pairs]
        print(f"  |ΔA| range: {min(deltas)}-{max(deltas)}, "
              f"median: {np.median(deltas):.0f}")

    return pairs


# ==============================================================================
# Dense source state extraction
# ==============================================================================

class DenseStateCapture:
    """Capture hidden states at ALL positions in one forward pass."""

    def __init__(self, model, layer_indices: List[int]):
        self.model = model
        self.layer_indices = layer_indices
        self._captured = {}
        self._hooks = []

    def _make_hook(self, layer_idx: int):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output
            self._captured[layer_idx] = hidden[0].detach().cpu().to(torch.float16)
        return hook_fn

    def register_hooks(self):
        self.remove_hooks()
        for layer_idx in self.layer_indices:
            module = self.model.model.layers[layer_idx]
            hook = module.register_forward_hook(self._make_hook(layer_idx))
            self._hooks.append(hook)

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []
        self._captured = {}

    def capture(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        positions: List[int],
    ) -> Dict[int, torch.Tensor]:
        """Run forward pass, return {layer_idx: (n_positions, hidden_dim)} on CPU."""
        self._captured = {}
        with torch.no_grad():
            self.model(input_ids=input_ids, attention_mask=attention_mask)

        pos_tensor = torch.tensor(positions)
        result = {}
        for layer_idx in self.layer_indices:
            result[layer_idx] = self._captured[layer_idx][pos_tensor].clone()

        self._captured = {}
        torch.cuda.empty_cache()
        return result


def build_dense_positions(question_end: int, filler_start: int, seq_len: int) -> List[int]:
    """All positions from question_end through answer_prompt."""
    return list(range(question_end, seq_len))


def extract_all_sources(
    model, tokenizer, problems, source_indices, few_shot,
    layer_indices, capturer,
) -> dict:
    """Extract dense states for all source (and target) problems."""
    source_data = {}
    to_extract = sorted(source_indices)

    first_param = next(model.parameters())
    input_device = first_param.device

    print(f"Dense extraction: {len(to_extract)} problems...")
    t0 = time.time()

    # Tokenize "Answer: " prefix once for pre_answer extraction
    prefix_ids = tokenizer.encode(PRE_ANSWER_PREFIX, add_special_tokens=False)

    for idx in tqdm(to_extract, desc="Dense extraction"):
        problem = {**problems[idx], "idx": idx}
        example_rng = random.Random(idx)
        messages = build_messages_for_condition(
            few_shot[:5], problem, FILLER_TYPE, FILLER_K, rng=example_rng
        )
        full_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(full_text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(input_device)
        attention_mask = inputs["attention_mask"].to(input_device)
        seq_len = input_ids.shape[1]

        question_end_pos, filler_start, filler_end = find_filler_boundaries(
            tokenizer, input_ids, FILLER_K
        )

        # Standard dense extraction (question_end through answer_prompt)
        dense_positions = build_dense_positions(question_end_pos, filler_start, seq_len)
        states = capturer.capture(input_ids, attention_mask, dense_positions)

        # Pre_answer extraction: append "Answer: " and capture at the last token
        ext_ids = torch.cat([
            input_ids,
            torch.tensor([prefix_ids], device=input_device),
        ], dim=1)
        ext_mask = torch.ones_like(ext_ids)
        ext_seq_len = ext_ids.shape[1]
        pre_answer_pos = ext_seq_len - 1  # last token of "Answer: "

        pre_answer_states = capturer.capture(ext_ids, ext_mask, [pre_answer_pos])

        source_data[idx] = {
            "states": states,
            "positions": dense_positions,
            "question_end_pos": question_end_pos,
            "filler_start": filler_start,
            "filler_end": filler_end,
            "seq_len": seq_len,
            "pre_answer_states": pre_answer_states,  # {layer: (1, hidden_dim)}
            "ext_seq_len": ext_seq_len,
        }

        del input_ids, attention_mask, ext_ids, ext_mask
        torch.cuda.empty_cache()

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.0f}s ({elapsed / len(to_extract):.1f}s each)")
    return source_data


# ==============================================================================
# Contiguous state injection
# ==============================================================================

class ContiguousInjector:
    """Hooks that replace hidden states at specified positions during prefill."""

    def __init__(self, model, layer_indices, patch_positions, source_states):
        """
        Args:
            patch_positions: absolute token positions in target to replace
            source_states: {layer_idx: tensor (n_positions, hidden_dim)}
        """
        self.model = model
        self.layer_indices = layer_indices
        self.patch_positions = patch_positions
        self.source_states = source_states
        self._hooks = []

    def _make_hook(self, layer_idx, position_indices, source_block):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output
            if hidden.shape[1] <= 1:
                return output
            hidden = hidden.clone()
            device, dtype = hidden.device, hidden.dtype
            hidden[0, position_indices.to(device)] = source_block.to(device=device, dtype=dtype)
            if isinstance(output, tuple):
                return (hidden,) + output[1:]
            return hidden
        return hook_fn

    def register_hooks(self):
        self.remove_hooks()
        pos_tensor = torch.tensor(self.patch_positions, dtype=torch.long)
        for layer_idx in self.layer_indices:
            if layer_idx not in self.source_states:
                continue
            module = self.model.model.layers[layer_idx]
            hook = module.register_forward_hook(
                self._make_hook(layer_idx, pos_tensor, self.source_states[layer_idx])
            )
            self._hooks.append(hook)

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []


# ==============================================================================
# Position mapping
# ==============================================================================

def compute_patch_mapping(
    start_condition: str,
    source_info: dict,
    target_question_end: int,
    target_filler_start: int,
    target_seq_len: int,
) -> Tuple[List[int], List[int]]:
    """
    Compute target positions to patch and source position indices.

    Uses filler-relative alignment: target filler token i <-> source filler token i.

    Returns:
        target_positions: absolute positions in target to patch
        source_indices: indices into source's dense position list
    """
    src_positions = source_info["positions"]
    src_question_end = source_info["question_end_pos"]
    src_filler_start = source_info["filler_start"]
    src_seq_len = source_info["seq_len"]
    src_pos_to_idx = {pos: i for i, pos in enumerate(src_positions)}

    target_positions = []
    source_indices = []

    spec = START_CONDITIONS[start_condition]

    if spec == "pre_answer":
        # Special: handled separately (extended input_ids with "Answer: " prefix)
        # Return None to signal caller to use pre_answer path
        return None, None

    if spec == "answer_prompt":
        # Patch only answer_prompt
        tgt_ap = target_seq_len - 1
        src_ap = src_seq_len - 1
        if src_ap in src_pos_to_idx:
            target_positions.append(tgt_ap)
            source_indices.append(src_pos_to_idx[src_ap])
        return target_positions, source_indices

    if spec == "question_end":
        # Start from question_end
        tgt_start = target_question_end
        src_start = src_question_end
    else:
        # Start from filler-relative offset
        filler_offset = spec  # 0-based
        tgt_start = target_filler_start + filler_offset
        src_start = src_filler_start + filler_offset

    # Map contiguous range from start through end of sequence
    max_offset = max(target_seq_len - tgt_start, src_seq_len - src_start)
    for offset in range(max_offset):
        tgt_pos = tgt_start + offset
        src_pos = src_start + offset
        if tgt_pos >= target_seq_len:
            break
        if src_pos in src_pos_to_idx:
            target_positions.append(tgt_pos)
            source_indices.append(src_pos_to_idx[src_pos])

    return target_positions, source_indices


# ==============================================================================
# Generation and answer parsing
# ==============================================================================

def parse_answer(raw: str) -> Optional[int]:
    m = re.search(r"Answer:\s*(-?\d+)", raw)
    if m:
        return int(m.group(1))
    m = re.search(r"(-?\d+)", raw)
    if m:
        return int(m.group(1))
    return None


def generate_answer(model, tokenizer, input_ids, attention_mask):
    with torch.no_grad():
        gen = model.generate(
            input_ids, attention_mask=attention_mask,
            max_new_tokens=20, do_sample=False, use_cache=True,
        )
    raw = tokenizer.decode(gen[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
    return raw, parse_answer(raw)


def run_patched_generation(model, tokenizer, input_ids, attention_mask, injector):
    injector.register_hooks()
    try:
        raw, answer = generate_answer(model, tokenizer, input_ids, attention_mask)
    finally:
        injector.remove_hooks()
    return raw, answer


# ==============================================================================
# Metrics
# ==============================================================================

def compute_metrics(patched_answer, natural_answer, source_answer, target_answer, delta_a):
    if patched_answer is None:
        return {
            "shift": None, "shift_ratio": None,
            "exact_match_donor": False, "exact_match_original": False,
            "novel": False, "parse_failure": True,
        }
    if natural_answer is None:
        # Can't compute shift if natural answer didn't parse
        return {
            "shift": None, "shift_ratio": None,
            "exact_match_donor": patched_answer == source_answer,
            "exact_match_original": patched_answer == target_answer,
            "novel": (patched_answer != source_answer and patched_answer != target_answer),
            "parse_failure": False,
        }
    shift = patched_answer - natural_answer
    shift_ratio = shift / delta_a if delta_a != 0 else None
    return {
        "shift": shift,
        "shift_ratio": shift_ratio,
        "exact_match_donor": patched_answer == source_answer,
        "exact_match_original": patched_answer == target_answer,
        "novel": (patched_answer != source_answer and patched_answer != target_answer),
        "parse_failure": False,
    }


# ==============================================================================
# Phase 0: Null validation
# ==============================================================================

def run_null_validation(
    model, tokenizer, problems, few_shot, layer_indices,
    source_data, n_examples=10,
):
    print(f"\n{'='*60}")
    print("Phase 0: Null validation")
    print(f"{'='*60}")

    first_param = next(model.parameters())
    input_device = first_param.device

    # Pick examples that have source data
    test_indices = [idx for idx in sorted(source_data.keys())][:n_examples]

    all_pass = True
    for idx in tqdm(test_indices, desc="Null validation"):
        problem = {**problems[idx], "idx": idx}
        example_rng = random.Random(idx)
        messages = build_messages_for_condition(
            few_shot[:5], problem, FILLER_TYPE, FILLER_K, rng=example_rng
        )
        full_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(full_text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(input_device)
        attention_mask = inputs["attention_mask"].to(input_device)

        # Natural answer
        _, natural_ans = generate_answer(model, tokenizer, input_ids, attention_mask)

        # Self-injection at all positions (strongest null test)
        src_info = source_data[idx]
        target_positions, source_indices = compute_patch_mapping(
            "question_end", src_info,
            src_info["question_end_pos"], src_info["filler_start"],
            src_info["seq_len"],
        )

        patch_states = {}
        for layer_idx in layer_indices:
            idx_tensor = torch.tensor(source_indices, dtype=torch.long)
            patch_states[layer_idx] = src_info["states"][layer_idx][idx_tensor]

        injector = ContiguousInjector(model, layer_indices, target_positions, patch_states)
        _, patched_ans = run_patched_generation(
            model, tokenizer, input_ids, attention_mask, injector
        )

        passed = (patched_ans == natural_ans)
        if not passed:
            print(f"  FAIL: idx={idx}, natural={natural_ans}, patched={patched_ans}")
            all_pass = False

        del input_ids, attention_mask, patch_states
        torch.cuda.empty_cache()

    status = "PASSED" if all_pass else "FAILED"
    print(f"  Null validation {status} ({len(test_indices)} examples)")
    return all_pass


# ==============================================================================
# Phase 1: All-layer contiguous patching
# ==============================================================================

def run_phase1(
    model, tokenizer, problems, few_shot, pairs, source_data,
    layer_indices, output_dir,
):
    conditions = list(START_CONDITIONS.keys())
    total = len(pairs) * len(conditions)
    print(f"\n{'='*60}")
    print(f"Phase 1: All-layer contiguous patching")
    print(f"  {len(pairs)} pairs × {len(conditions)} conditions = {total}")
    print(f"{'='*60}")

    first_param = next(model.parameters())
    input_device = first_param.device
    checkpoint = output_dir / "phase1_checkpoint.json"

    # Load checkpoint
    completed = {}
    results = []
    if checkpoint.exists():
        with open(checkpoint) as f:
            results = json.load(f).get("results", [])
        for r in results:
            completed[(r["target_idx"], r["source_idx"], r["start_condition"])] = True
        print(f"  Resuming: {len(completed)} done")

    # Get natural answers for all targets
    print("Getting natural answers...")
    natural_answers = {}
    for pair in pairs:
        tgt_idx = pair["target_idx"]
        if tgt_idx in natural_answers:
            continue
        problem = {**problems[tgt_idx], "idx": tgt_idx}
        example_rng = random.Random(tgt_idx)
        messages = build_messages_for_condition(
            few_shot[:5], problem, FILLER_TYPE, FILLER_K, rng=example_rng
        )
        full_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(full_text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(input_device)
        attention_mask = inputs["attention_mask"].to(input_device)
        _, ans = generate_answer(model, tokenizer, input_ids, attention_mask)
        natural_answers[tgt_idx] = ans
        del input_ids, attention_mask
        torch.cuda.empty_cache()
    print(f"  {len(natural_answers)} unique targets")

    # Run interventions
    t0 = time.time()
    for pair_i, pair in enumerate(tqdm(pairs, desc="Phase 1")):
        tgt_idx = pair["target_idx"]
        src_idx = pair["source_idx"]
        src_info = source_data[src_idx]
        natural_ans = natural_answers[tgt_idx]

        # Build target prompt once per pair
        problem = {**problems[tgt_idx], "idx": tgt_idx}
        example_rng = random.Random(tgt_idx)
        messages = build_messages_for_condition(
            few_shot[:5], problem, FILLER_TYPE, FILLER_K, rng=example_rng
        )
        full_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(full_text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(input_device)
        attention_mask = inputs["attention_mask"].to(input_device)
        seq_len = input_ids.shape[1]

        tgt_qe, tgt_fs, tgt_fe = find_filler_boundaries(tokenizer, input_ids, FILLER_K)

        # Build extended input (with "Answer: " appended) once per pair
        prefix_token_ids = tokenizer.encode(PRE_ANSWER_PREFIX, add_special_tokens=False)
        ext_input_ids = torch.cat([
            input_ids,
            torch.tensor([prefix_token_ids], device=input_device),
        ], dim=1)
        ext_attention_mask = torch.ones_like(ext_input_ids)
        ext_seq_len = ext_input_ids.shape[1]

        for cond_name in conditions:
            key = (tgt_idx, src_idx, cond_name)
            if key in completed:
                continue

            target_positions, source_indices = compute_patch_mapping(
                cond_name, src_info, tgt_qe, tgt_fs, seq_len,
            )

            if target_positions is None and cond_name == "pre_answer":
                # Pre_answer: patch only the last token of extended input
                pre_ans_pos = ext_seq_len - 1
                patch_states = {}
                for layer_idx in layer_indices:
                    patch_states[layer_idx] = src_info["pre_answer_states"][layer_idx]
                injector = ContiguousInjector(
                    model, layer_indices, [pre_ans_pos], patch_states
                )
                raw, patched_ans = run_patched_generation(
                    model, tokenizer, ext_input_ids, ext_attention_mask, injector
                )
            else:
                if not target_positions:
                    continue

                # Standard: build patch states from dense extraction
                patch_states = {}
                for layer_idx in layer_indices:
                    idx_tensor = torch.tensor(source_indices, dtype=torch.long)
                    patch_states[layer_idx] = src_info["states"][layer_idx][idx_tensor]

                injector = ContiguousInjector(
                    model, layer_indices, target_positions, patch_states
                )
                raw, patched_ans = run_patched_generation(
                    model, tokenizer, input_ids, attention_mask, injector
                )

            n_patched = 1 if cond_name == "pre_answer" else len(target_positions)
            metrics = compute_metrics(
                patched_ans, natural_ans,
                pair["source_answer"], pair["target_answer"],
                pair["delta_A"],
            )

            results.append({
                "target_idx": tgt_idx,
                "source_idx": src_idx,
                "start_condition": cond_name,
                "A_target": pair["target_A"],
                "A_source": pair["source_A"],
                "Y": pair["Y"],
                "delta_A": pair["delta_A"],
                "natural_answer": natural_ans,
                "expected_donor": pair["source_answer"],
                "expected_original": pair["target_answer"],
                "patched_answer": patched_ans,
                "raw_response": raw,
                "n_positions_patched": n_patched,
                **metrics,
            })
            completed[key] = True

            del patch_states
            torch.cuda.empty_cache()

        # Checkpoint every 5 pairs
        if (pair_i + 1) % 5 == 0 or pair_i == len(pairs) - 1:
            with open(checkpoint, "w") as f:
                json.dump({"results": results}, f)

        del input_ids, attention_mask, ext_input_ids, ext_attention_mask
        torch.cuda.empty_cache()

    elapsed = time.time() - t0
    print(f"  Phase 1 done: {len(results)} results in {elapsed:.0f}s")
    return results


# ==============================================================================
# Phase 2: Layer-band patching
# ==============================================================================

def run_phase2(
    model, tokenizer, problems, few_shot, pairs, source_data,
    layer_indices, output_dir, n_pairs=20,
):
    subset = pairs[:n_pairs]
    total = n_pairs * len(PHASE2_CONDITIONS) * len(LAYER_BANDS)
    print(f"\n{'='*60}")
    print(f"Phase 2: Layer-band patching")
    print(f"  {n_pairs} pairs × {len(PHASE2_CONDITIONS)} conds × "
          f"{len(LAYER_BANDS)} bands = {total}")
    print(f"{'='*60}")

    first_param = next(model.parameters())
    input_device = first_param.device
    checkpoint = output_dir / "phase2_checkpoint.json"

    completed = {}
    results = []
    if checkpoint.exists():
        with open(checkpoint) as f:
            results = json.load(f).get("results", [])
        for r in results:
            completed[(r["target_idx"], r["source_idx"],
                       r["start_condition"], r["layer_band"])] = True
        print(f"  Resuming: {len(completed)} done")

    # Natural answers
    natural_answers = {}
    for pair in subset:
        tgt_idx = pair["target_idx"]
        if tgt_idx in natural_answers:
            continue
        problem = {**problems[tgt_idx], "idx": tgt_idx}
        example_rng = random.Random(tgt_idx)
        messages = build_messages_for_condition(
            few_shot[:5], problem, FILLER_TYPE, FILLER_K, rng=example_rng
        )
        full_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(full_text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(input_device)
        attention_mask = inputs["attention_mask"].to(input_device)
        _, ans = generate_answer(model, tokenizer, input_ids, attention_mask)
        natural_answers[tgt_idx] = ans
        del input_ids, attention_mask
        torch.cuda.empty_cache()

    t0 = time.time()
    for pair_i, pair in enumerate(tqdm(subset, desc="Phase 2")):
        tgt_idx = pair["target_idx"]
        src_idx = pair["source_idx"]
        src_info = source_data[src_idx]
        natural_ans = natural_answers[tgt_idx]

        problem = {**problems[tgt_idx], "idx": tgt_idx}
        example_rng = random.Random(tgt_idx)
        messages = build_messages_for_condition(
            few_shot[:5], problem, FILLER_TYPE, FILLER_K, rng=example_rng
        )
        full_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(full_text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(input_device)
        attention_mask = inputs["attention_mask"].to(input_device)
        seq_len = input_ids.shape[1]

        tgt_qe, tgt_fs, tgt_fe = find_filler_boundaries(tokenizer, input_ids, FILLER_K)

        # Extended input for pre_answer
        prefix_token_ids = tokenizer.encode(PRE_ANSWER_PREFIX, add_special_tokens=False)
        ext_input_ids = torch.cat([
            input_ids,
            torch.tensor([prefix_token_ids], device=input_device),
        ], dim=1)
        ext_attention_mask = torch.ones_like(ext_input_ids)
        ext_seq_len = ext_input_ids.shape[1]

        for cond_name in PHASE2_CONDITIONS:
            target_positions, source_indices = compute_patch_mapping(
                cond_name, src_info, tgt_qe, tgt_fs, seq_len,
            )

            is_pre_answer = (target_positions is None and cond_name == "pre_answer")
            if not is_pre_answer and not target_positions:
                continue

            for band_name, band_layers in LAYER_BANDS.items():
                key = (tgt_idx, src_idx, cond_name, band_name)
                if key in completed:
                    continue

                if is_pre_answer:
                    pre_ans_pos = ext_seq_len - 1
                    patch_states = {}
                    for layer_idx in band_layers:
                        patch_states[layer_idx] = src_info["pre_answer_states"][layer_idx]
                    injector = ContiguousInjector(
                        model, band_layers, [pre_ans_pos], patch_states
                    )
                    raw, patched_ans = run_patched_generation(
                        model, tokenizer, ext_input_ids, ext_attention_mask, injector
                    )
                else:
                    patch_states = {}
                    for layer_idx in band_layers:
                        idx_tensor = torch.tensor(source_indices, dtype=torch.long)
                        patch_states[layer_idx] = src_info["states"][layer_idx][idx_tensor]
                    injector = ContiguousInjector(
                        model, band_layers, target_positions, patch_states
                    )
                    raw, patched_ans = run_patched_generation(
                        model, tokenizer, input_ids, attention_mask, injector
                    )

                metrics = compute_metrics(
                    patched_ans, natural_ans,
                    pair["source_answer"], pair["target_answer"],
                    pair["delta_A"],
                )

                n_patched = 1 if is_pre_answer else len(target_positions)
                results.append({
                    "target_idx": tgt_idx,
                    "source_idx": src_idx,
                    "start_condition": cond_name,
                    "layer_band": band_name,
                    "A_target": pair["target_A"],
                    "A_source": pair["source_A"],
                    "Y": pair["Y"],
                    "delta_A": pair["delta_A"],
                    "natural_answer": natural_ans,
                    "expected_donor": pair["source_answer"],
                    "expected_original": pair["target_answer"],
                    "patched_answer": patched_ans,
                    "raw_response": raw,
                    "n_positions_patched": n_patched,
                    **metrics,
                })
                completed[key] = True

                del patch_states
                torch.cuda.empty_cache()

        if (pair_i + 1) % 5 == 0 or pair_i == len(subset) - 1:
            with open(checkpoint, "w") as f:
                json.dump({"results": results}, f)

        del input_ids, attention_mask, ext_input_ids, ext_attention_mask
        torch.cuda.empty_cache()

    elapsed = time.time() - t0
    print(f"  Phase 2 done: {len(results)} results in {elapsed:.0f}s")
    return results


# ==============================================================================
# Aggregation
# ==============================================================================

def print_aggregate(results: list, label: str, group_key: str = "start_condition"):
    by_group = defaultdict(list)
    for r in results:
        by_group[r[group_key]].append(r)

    print(f"\n{'='*80}")
    print(f"{label}")
    print(f"{'='*80}")
    print(f"{'Condition':<25} {'N':>4} {'MedRatio':>9} {'MeanRatio':>10} "
          f"{'Donor%':>7} {'Orig%':>6} {'Novel%':>7} {'Parse%':>7}")
    print("-" * 80)

    for cond in sorted(by_group.keys()):
        items = by_group[cond]
        n = len(items)
        n_parsed = sum(1 for r in items if not r["parse_failure"])
        ratios = [r["shift_ratio"] for r in items if r["shift_ratio"] is not None]
        donor = sum(1 for r in items if r["exact_match_donor"]) / max(n_parsed, 1)
        orig = sum(1 for r in items if r["exact_match_original"]) / max(n_parsed, 1)
        novel = sum(1 for r in items if r["novel"]) / max(n_parsed, 1)
        parse_fail = sum(1 for r in items if r["parse_failure"]) / n

        med = np.median(ratios) if ratios else float("nan")
        mean = np.mean(ratios) if ratios else float("nan")

        print(f"{cond:<25} {n:>4} {med:>9.3f} {mean:>10.3f} "
              f"{donor:>7.1%} {orig:>6.1%} {novel:>7.1%} {parse_fail:>7.1%}")


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Cross-example filler patching")
    parser.add_argument("--model-path", default="/workspace/models/deepseek-v3-awq")
    parser.add_argument("--dataset", type=Path,
                        default=Path("probing/data/1hop_addition_dataset.json"))
    parser.add_argument("--categories", type=Path,
                        default=Path("probing/probe_results/categories.json"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("probing/patching_results_v2"))
    parser.add_argument("--max-pairs", type=int, default=200)
    parser.add_argument("--min-delta-a", type=int, default=10)
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--skip-phase2", action="store_true")
    parser.add_argument("--phase2-pairs", type=int, default=20)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    with open(args.dataset) as f:
        dataset = json.load(f)
    few_shot = dataset["few_shot_facts"]
    problems = dataset["examples"]
    for i, p in enumerate(problems):
        if "idx" not in p:
            p["idx"] = i
    print(f"Loaded {len(problems)} problems")

    # Select pairs
    pairs = select_pairs(args.categories, args.max_pairs, args.min_delta_a)
    if not pairs:
        print("No valid pairs found!")
        return

    with open(args.output_dir / "pairs.json", "w") as f:
        json.dump(pairs, f, indent=2)

    # Load model
    model, tokenizer = load_model(args.model_path)
    layer_indices = list(range(61))

    # Dense extraction (sources + targets for null validation)
    capturer = DenseStateCapture(model, layer_indices)
    capturer.register_hooks()

    all_indices = set(p["source_idx"] for p in pairs) | set(p["target_idx"] for p in pairs)
    source_data = extract_all_sources(
        model, tokenizer, problems, all_indices, few_shot,
        layer_indices, capturer,
    )
    capturer.remove_hooks()

    # Phase 0
    if not args.skip_validation:
        ok = run_null_validation(
            model, tokenizer, problems, few_shot,
            layer_indices, source_data,
        )
        if not ok:
            print("WARNING: Null validation failed! Proceeding anyway.")

    # Phase 1
    p1 = run_phase1(
        model, tokenizer, problems, few_shot, pairs, source_data,
        layer_indices, args.output_dir,
    )
    print_aggregate(p1, "Phase 1: All-layer contiguous")
    with open(args.output_dir / "phase1_results.json", "w") as f:
        json.dump(p1, f, indent=2)

    # Phase 2
    if not args.skip_phase2:
        p2 = run_phase2(
            model, tokenizer, problems, few_shot, pairs, source_data,
            layer_indices, args.output_dir, n_pairs=args.phase2_pairs,
        )
        # Aggregate by condition+band
        for r in p2:
            r["cond_band"] = f"{r['start_condition']}_{r['layer_band']}"
        print_aggregate(p2, "Phase 2: Layer-band", group_key="cond_band")
        with open(args.output_dir / "phase2_results.json", "w") as f:
            json.dump(p2, f, indent=2)

    print("\nDone!")


if __name__ == "__main__":
    main()
