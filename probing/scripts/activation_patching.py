"""
activation_patching.py

Causal intervention experiment: swap hidden states between problems during
filler token processing to test whether filler representations causally
influence the model's answer.

Contiguous range patching: for each intervention start position P, replace
hidden states at ALL token positions from P through answer_prompt with source
problem states. This gives downstream tokens a consistent source-derived
context.

Usage:
    source /workspace/config/probing_env.sh
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python probing/scripts/activation_patching.py \
        --model-path /workspace/models/deepseek-v3-awq \
        --dataset probing/data/1hop_addition_dataset.json \
        --categories probing/probe_results/categories.json \
        --output-dir probing/patching_results \
        --max-pairs 60
"""

import argparse
import json
import pickle
import random
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from tqdm import tqdm

# Patch: autoawq imports PytorchGELUTanh which was renamed
import transformers.activations as _act
if not hasattr(_act, "PytorchGELUTanh"):
    _act.PytorchGELUTanh = _act.GELUTanh

# Reuse infrastructure from extraction script
from extract_hidden_states import (
    build_messages_for_condition,
    find_filler_boundaries,
    load_model,
    load_tokenizer,
)
from generate_1hop_dataset import build_system_message, build_user_turn


# ==============================================================================
# Pair selection
# ==============================================================================

def select_pairs(
    categories_path: Path,
    max_pairs: int = 107,
    min_delta_a: int = 10,
) -> List[dict]:
    """
    Select (source, target) pairs for patching.

    Target: filler_helped examples (wrong baseline, correct with filler).
    Source: filler_helped OR both_correct (correct with filler).
    Criteria: exact Y match, |A_source - A_target| >= min_delta_a.
    Per target: select source with largest |delta_A|.
    """
    with open(categories_path) as f:
        data = json.load(f)

    examples = data["examples"]
    fh = [e for e in examples if e["category"] == "filler_helped"]
    bc = [e for e in examples if e["category"] == "both_correct"]
    source_pool = fh + bc

    src_by_y = defaultdict(list)
    for e in source_pool:
        src_by_y[e["x"]].append(e)

    fh_by_y = defaultdict(list)
    for e in fh:
        fh_by_y[e["x"]].append(e)

    pairs = []
    for y, targets in fh_by_y.items():
        sources = src_by_y[y]
        for t in targets:
            best_s = None
            best_da = 0
            for s in sources:
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
                    "Y": y,
                    "delta_A": best_da,
                    "source_answer": best_s["fact_value"] + y,
                    "target_answer": t["fact_value"] + y,
                    "target_natural_answer": t["dots_250_model_answer"],
                })

    # Sort by delta_A descending, take max_pairs
    pairs.sort(key=lambda p: p["delta_A"], reverse=True)
    pairs = pairs[:max_pairs]

    print(f"Selected {len(pairs)} pairs "
          f"({len(set(p['source_idx'] for p in pairs))} unique sources, "
          f"{len(set(p['Y'] for p in pairs))} unique Y values)")
    print(f"  |delta_A|: min={min(p['delta_A'] for p in pairs)}, "
          f"max={max(p['delta_A'] for p in pairs)}, "
          f"mean={sum(p['delta_A'] for p in pairs) / len(pairs):.0f}")

    return pairs


# ==============================================================================
# Dense source state extraction
# ==============================================================================

class DenseStateCapture:
    """
    Capture hidden states at ALL filler positions + answer_prompt in one
    forward pass. States stored on CPU as float16 tensors.
    """

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
            # Store full sequence on CPU (will index later)
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
        """
        Run forward pass and extract states at given positions.

        Returns:
            {layer_idx: tensor of shape (n_positions, hidden_dim)} on CPU
        """
        self._captured = {}
        with torch.no_grad():
            self.model(input_ids=input_ids, attention_mask=attention_mask)

        result = {}
        pos_tensor = torch.tensor(positions)
        for layer_idx in self.layer_indices:
            full_seq = self._captured[layer_idx]  # (seq_len, hidden_dim)
            result[layer_idx] = full_seq[pos_tensor].clone()  # (n_positions, hidden_dim)

        self._captured = {}
        torch.cuda.empty_cache()
        return result


def build_dense_positions(filler_start: int, filler_end: int, seq_len: int) -> List[int]:
    """
    Build list of ALL positions from filler_start through answer_prompt.
    Includes: every filler token + any tokens between filler_end and seq_len-1
    (typically newlines and "Answer:" tokens) + answer_prompt.
    """
    positions = list(range(filler_start, seq_len))
    return positions


def extract_dense_source(
    model,
    tokenizer,
    problem: dict,
    few_shot: list,
    layer_indices: List[int],
    capturer: DenseStateCapture,
    filler_type: str = "dots",
    k: int = 250,
) -> dict:
    """
    Extract dense states for one source problem.

    Returns dict with:
        states: {layer_idx: tensor (n_positions, hidden_dim)} on CPU
        positions: list of absolute token indices
        filler_start, filler_end, seq_len
    """
    example_rng = random.Random(problem["idx"])
    messages = build_messages_for_condition(
        few_shot[:5], problem, filler_type, k, rng=example_rng
    )
    full_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(full_text, return_tensors="pt")

    first_param = next(model.parameters())
    input_ids = inputs["input_ids"].to(first_param.device)
    attention_mask = inputs["attention_mask"].to(first_param.device)
    seq_len = input_ids.shape[1]

    filler_start, filler_end = find_filler_boundaries(tokenizer, input_ids, k)
    dense_positions = build_dense_positions(filler_start, filler_end, seq_len)

    states = capturer.capture(input_ids, attention_mask, dense_positions)

    del input_ids, attention_mask
    torch.cuda.empty_cache()

    return {
        "states": states,
        "positions": dense_positions,
        "filler_start": filler_start,
        "filler_end": filler_end,
        "seq_len": seq_len,
    }


# ==============================================================================
# Contiguous state injection
# ==============================================================================

class ContiguousInjector:
    """
    Register hooks that replace hidden states at a contiguous range of
    token positions during the prefill forward pass.
    """

    def __init__(
        self,
        model,
        layer_indices: List[int],
        patch_positions: List[int],
        source_states: Dict[int, torch.Tensor],
    ):
        """
        Args:
            layer_indices: which layers to patch
            patch_positions: absolute token positions to replace (in target)
            source_states: {layer_idx: tensor (n_source_positions, hidden_dim)}
                The source positions are indexed by position within the
                contiguous range (0 = first patched position).
        """
        self.model = model
        self.layer_indices = layer_indices
        self.patch_positions = patch_positions
        self.source_states = source_states
        self._hooks = []

    def _make_hook(self, layer_idx: int, position_indices: torch.Tensor,
                   source_block: torch.Tensor):
        """
        Create a hook that replaces hidden states at specified positions.

        Args:
            position_indices: tensor of absolute token positions to patch
            source_block: tensor of shape (n_positions, hidden_dim) for this layer
        """
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output

            # Only patch during prefill (full sequence), not generation (1 token)
            if hidden.shape[1] <= 1:
                return output

            hidden = hidden.clone()
            device, dtype = hidden.device, hidden.dtype
            src = source_block.to(device=device, dtype=dtype)

            # Replace all patch positions at once
            hidden[0, position_indices.to(device)] = src

            if isinstance(output, tuple):
                return (hidden,) + output[1:]
            return hidden

        return hook_fn

    def register_hooks(self):
        self.remove_hooks()
        pos_tensor = torch.tensor(self.patch_positions, dtype=torch.long)

        for layer_idx in self.layer_indices:
            source_block = self.source_states[layer_idx]
            module = self.model.model.layers[layer_idx]
            hook = module.register_forward_hook(
                self._make_hook(layer_idx, pos_tensor, source_block)
            )
            self._hooks.append(hook)

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []


# ==============================================================================
# Patched generation
# ==============================================================================

def parse_answer(raw: str) -> Optional[int]:
    """Parse numeric answer from model output."""
    m = re.search(r"Answer:\s*(-?\d+)", raw)
    if m:
        return int(m.group(1))
    m = re.search(r"(-?\d+)", raw)
    if m:
        return int(m.group(1))
    return None


def run_patched_generation(
    model,
    tokenizer,
    target_input_ids: torch.Tensor,
    target_attention_mask: torch.Tensor,
    injector: ContiguousInjector,
) -> Tuple[str, Optional[int]]:
    """
    Run generation with injection hooks active.
    Hooks fire during prefill, become no-ops during autoregressive steps.
    """
    injector.register_hooks()
    try:
        with torch.no_grad():
            gen_output = model.generate(
                target_input_ids,
                attention_mask=target_attention_mask,
                max_new_tokens=20,
                do_sample=False,
                use_cache=True,
            )
    finally:
        injector.remove_hooks()

    new_tokens = gen_output[0][target_input_ids.shape[1]:]
    raw = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    answer = parse_answer(raw)
    return raw, answer


# ==============================================================================
# Position mapping between source and target
# ==============================================================================

def compute_patch_mapping(
    source_info: dict,
    target_filler_start: int,
    target_filler_end: int,
    target_seq_len: int,
    intervention_start_frac: float,
) -> Tuple[List[int], List[int]]:
    """
    Compute which target positions to patch and which source indices to use.

    Uses filler-relative offset alignment: source filler token i maps to
    target filler token i.

    Args:
        source_info: dict from extract_dense_source (has positions, filler_start, etc.)
        target_filler_start/end/seq_len: target's filler boundaries
        intervention_start_frac: fraction of filler where intervention begins
            (0.0 = start of filler, 0.5 = midpoint, 1.0 = end of filler,
             -1.0 = question_end, 2.0 = answer_prompt only)

    Returns:
        target_positions: absolute token positions in target to patch
        source_indices: corresponding indices into source's dense position list
    """
    src_filler_start = source_info["filler_start"]
    src_filler_end = source_info["filler_end"]
    src_positions = source_info["positions"]  # list of absolute positions
    src_seq_len = source_info["seq_len"]

    # Build source position -> index lookup
    src_pos_to_idx = {pos: i for i, pos in enumerate(src_positions)}

    target_filler_len = target_filler_end - target_filler_start + 1
    src_filler_len = src_filler_end - src_filler_start + 1

    if intervention_start_frac == 2.0:
        # Special: patch answer_prompt only (single token)
        target_ap = target_seq_len - 1
        src_ap = src_seq_len - 1
        if src_ap in src_pos_to_idx:
            return [target_ap], [src_pos_to_idx[src_ap]]
        return [], []

    if intervention_start_frac < 0:
        # question_end: patch from question_end through everything
        target_start = target_filler_start - 1
        src_start = src_filler_start - 1
    else:
        # Filler fraction
        target_start = target_filler_start + int(intervention_start_frac * (target_filler_len - 1))
        src_start = src_filler_start + int(intervention_start_frac * (src_filler_len - 1))

    target_positions = []
    source_indices = []

    # Map contiguous range from intervention start through end of sequence
    # Use filler-relative offset: target pos = target_start + offset,
    # source pos = src_start + offset
    max_offset = max(target_seq_len - target_start, src_seq_len - src_start)
    for offset in range(max_offset):
        tgt_pos = target_start + offset
        src_pos = src_start + offset

        if tgt_pos >= target_seq_len:
            break
        if src_pos not in src_pos_to_idx:
            continue

        target_positions.append(tgt_pos)
        source_indices.append(src_pos_to_idx[src_pos])

    return target_positions, source_indices


# ==============================================================================
# Metrics
# ==============================================================================

def compute_shift_metrics(
    patched_answer: Optional[int],
    target_natural: int,
    source_answer: int,
    target_answer: int,
) -> dict:
    """Compute metrics for one patched generation."""
    if patched_answer is None:
        return {
            "patched_answer": None,
            "parse_failure": True,
            "shift_ratio": None,
            "exact_match_donor": False,
            "exact_match_original": False,
            "novel_answer": True,
        }

    delta_a = source_answer - target_answer  # A_source - A_target (= A_source + Y - A_target - Y)
    if delta_a == 0:
        shift_ratio = 0.0 if patched_answer == target_natural else None
    else:
        shift_ratio = (patched_answer - target_natural) / delta_a

    return {
        "patched_answer": patched_answer,
        "parse_failure": False,
        "shift_ratio": shift_ratio,
        "exact_match_donor": patched_answer == source_answer,
        "exact_match_original": patched_answer == target_answer,
        "novel_answer": (patched_answer != source_answer and
                         patched_answer != target_answer),
    }


# ==============================================================================
# Null validation
# ==============================================================================

def run_null_validation(
    model,
    tokenizer,
    problems: list,
    few_shot: list,
    layer_indices: List[int],
    capturer: DenseStateCapture,
    n_examples: int = 10,
    filler_type: str = "dots",
    k: int = 250,
) -> bool:
    """
    Inject target's own states back into itself. Must preserve original answer.
    Returns True if all pass.
    """
    print("\n" + "=" * 60)
    print("PHASE 0: Null validation")
    print("=" * 60)

    first_param = next(model.parameters())
    input_device = first_param.device

    # Use first n filler_helped examples (or any that we'll actually patch)
    test_problems = problems[:n_examples]
    all_pass = True

    for i, problem in enumerate(test_problems):
        example_rng = random.Random(problem["idx"])
        messages = build_messages_for_condition(
            few_shot[:5], problem, filler_type, k, rng=example_rng
        )
        full_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(full_text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(input_device)
        attention_mask = inputs["attention_mask"].to(input_device)
        seq_len = input_ids.shape[1]

        filler_start, filler_end = find_filler_boundaries(tokenizer, input_ids, k)
        dense_positions = build_dense_positions(filler_start, filler_end, seq_len)

        # Capture own states
        own_states = capturer.capture(input_ids, attention_mask, dense_positions)

        # Create self-injection: patch from answer_prompt only (single token)
        target_ap = seq_len - 1
        src_ap_idx = dense_positions.index(target_ap)

        self_states = {}
        for layer_idx in layer_indices:
            self_states[layer_idx] = own_states[layer_idx][src_ap_idx:src_ap_idx + 1]

        injector = ContiguousInjector(model, layer_indices, [target_ap], self_states)

        raw, patched_answer = run_patched_generation(
            model, tokenizer, input_ids, attention_mask, injector
        )

        # Also get natural answer
        with torch.no_grad():
            gen_out = model.generate(
                input_ids, attention_mask=attention_mask,
                max_new_tokens=20, do_sample=False, use_cache=True,
            )
        natural_raw = tokenizer.decode(
            gen_out[0][seq_len:], skip_special_tokens=True
        ).strip()
        natural_answer = parse_answer(natural_raw)

        passed = (patched_answer == natural_answer)
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] prob_{problem['idx']:04d}: "
              f"natural={natural_answer}, patched={patched_answer}")

        if not passed:
            all_pass = False
            print(f"    Natural raw: {natural_raw!r}")
            print(f"    Patched raw: {raw!r}")

        del input_ids, attention_mask, own_states, self_states
        torch.cuda.empty_cache()

    if all_pass:
        print(f"\nNull validation PASSED ({len(test_problems)}/{len(test_problems)})")
    else:
        print(f"\nNull validation FAILED — hook implementation has a bug!")

    return all_pass


# ==============================================================================
# Main experiment
# ==============================================================================

INTERVENTION_POSITIONS = {
    "question_end": -1.0,
    "filler_0.00": 0.0,
    "filler_0.25": 0.25,
    "filler_0.50": 0.50,
    "filler_0.75": 0.75,
    "filler_1.00": 1.0,
    "answer_prompt": 2.0,
}


def run_experiment(
    model,
    tokenizer,
    pairs: List[dict],
    problems: list,
    few_shot: list,
    layer_indices: List[int],
    output_dir: Path,
    filler_type: str = "dots",
    k: int = 250,
    positions_to_test: Optional[List[str]] = None,
):
    """
    Run the full activation patching experiment.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if positions_to_test is None:
        positions_to_test = list(INTERVENTION_POSITIONS.keys())

    first_param = next(model.parameters())
    input_device = first_param.device

    # Phase 1: Dense source extraction
    print("\n" + "=" * 60)
    print("Dense source state extraction")
    print("=" * 60)

    capturer = DenseStateCapture(model, layer_indices)
    capturer.register_hooks()

    unique_source_idxs = sorted(set(p["source_idx"] for p in pairs))
    print(f"Extracting dense states for {len(unique_source_idxs)} unique sources...")

    source_cache = {}  # source_idx -> dense info dict
    prob_by_idx = {p["idx"]: p for p in problems}

    t0 = time.time()
    for src_idx in tqdm(unique_source_idxs, desc="Sources"):
        source_cache[src_idx] = extract_dense_source(
            model, tokenizer, prob_by_idx[src_idx], few_shot,
            layer_indices, capturer, filler_type, k,
        )
    capturer.remove_hooks()

    elapsed = time.time() - t0
    print(f"Source extraction: {elapsed:.0f}s "
          f"({elapsed / len(unique_source_idxs):.1f}s per source)")

    # Phase 2: Patched generations
    print("\n" + "=" * 60)
    print(f"Patched generation: {len(pairs)} pairs x "
          f"{len(positions_to_test)} positions")
    print("=" * 60)

    # Load checkpoint if exists
    checkpoint_path = output_dir / "checkpoint.pkl"
    if checkpoint_path.exists():
        with open(checkpoint_path, "rb") as f:
            results = pickle.load(f)
        print(f"Loaded checkpoint with {len(results)} results")
    else:
        results = []

    completed_keys = set()
    for r in results:
        completed_keys.add((r["source_idx"], r["target_idx"], r["position_name"]))

    total = len(pairs) * len(positions_to_test)
    done = len(completed_keys)
    t0 = time.time()

    for pair_i, pair in enumerate(pairs):
        target_problem = prob_by_idx[pair["target_idx"]]
        source_info = source_cache[pair["source_idx"]]

        # Build target prompt
        example_rng = random.Random(target_problem["idx"])
        messages = build_messages_for_condition(
            few_shot[:5], target_problem, filler_type, k, rng=example_rng
        )
        full_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(full_text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(input_device)
        attention_mask = inputs["attention_mask"].to(input_device)
        seq_len = input_ids.shape[1]

        target_filler_start, target_filler_end = find_filler_boundaries(
            tokenizer, input_ids, k
        )

        for pos_name in positions_to_test:
            key = (pair["source_idx"], pair["target_idx"], pos_name)
            if key in completed_keys:
                continue

            frac = INTERVENTION_POSITIONS[pos_name]
            target_positions, source_indices = compute_patch_mapping(
                source_info,
                target_filler_start, target_filler_end, seq_len,
                frac,
            )

            if not target_positions:
                print(f"  WARNING: no valid positions for {pos_name} "
                      f"(pair {pair['source_idx']}->{pair['target_idx']})")
                continue

            # Build source state subset for this intervention range
            patch_states = {}
            for layer_idx in layer_indices:
                full_layer = source_info["states"][layer_idx]
                idx_tensor = torch.tensor(source_indices, dtype=torch.long)
                patch_states[layer_idx] = full_layer[idx_tensor]

            injector = ContiguousInjector(
                model, layer_indices, target_positions, patch_states
            )

            raw, patched_answer = run_patched_generation(
                model, tokenizer, input_ids, attention_mask, injector
            )

            metrics = compute_shift_metrics(
                patched_answer,
                pair["target_natural_answer"],
                pair["source_answer"],
                pair["target_answer"],
            )

            result = {
                "source_idx": pair["source_idx"],
                "target_idx": pair["target_idx"],
                "position_name": pos_name,
                "source_A": pair["source_A"],
                "target_A": pair["target_A"],
                "Y": pair["Y"],
                "delta_A": pair["delta_A"],
                "source_answer": pair["source_answer"],
                "target_answer": pair["target_answer"],
                "target_natural_answer": pair["target_natural_answer"],
                "n_positions_patched": len(target_positions),
                "raw_response": raw,
                **metrics,
            }
            results.append(result)
            completed_keys.add(key)
            done += 1

            del patch_states
            torch.cuda.empty_cache()

        # Checkpoint after each pair
        if (pair_i + 1) % 5 == 0 or pair_i == len(pairs) - 1:
            with open(checkpoint_path, "wb") as f:
                pickle.dump(results, f)

            elapsed = time.time() - t0
            rate = done / elapsed if elapsed > 0 else 0
            remaining = (total - done) / rate if rate > 0 else 0
            print(f"  [{done}/{total}] {elapsed:.0f}s elapsed, "
                  f"~{remaining:.0f}s remaining")

        del input_ids, attention_mask
        torch.cuda.empty_cache()

    # Save final results
    with open(output_dir / "patching_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nDone: {len(results)} results saved to {output_dir}/")
    return results


# ==============================================================================
# Aggregation and reporting
# ==============================================================================

def aggregate_results(results: List[dict], output_dir: Path):
    """Compute and print aggregate metrics per intervention position."""
    by_position = defaultdict(list)
    for r in results:
        by_position[r["position_name"]].append(r)

    print("\n" + "=" * 80)
    print("AGGREGATE RESULTS")
    print("=" * 80)

    header = (f"{'Position':<16} {'N':>4} {'Parse%':>7} {'Novel%':>7} "
              f"{'Shift':>7} {'Shift*':>7} {'Donor%':>7} {'Orig%':>7}")
    print(header)
    print("-" * 80)

    summary = {}
    position_order = ["question_end", "filler_0.00", "filler_0.25",
                      "filler_0.50", "filler_0.75", "filler_1.00",
                      "answer_prompt"]

    for pos_name in position_order:
        if pos_name not in by_position:
            continue
        items = by_position[pos_name]
        n = len(items)
        n_parsed = sum(1 for r in items if not r["parse_failure"])
        n_novel = sum(1 for r in items if r["novel_answer"] and not r["parse_failure"])
        shifts = [r["shift_ratio"] for r in items
                  if r["shift_ratio"] is not None]
        # Shift excluding novel answers
        shifts_non_novel = [r["shift_ratio"] for r in items
                           if r["shift_ratio"] is not None and not r["novel_answer"]]
        n_donor = sum(1 for r in items if r["exact_match_donor"])
        n_orig = sum(1 for r in items if r["exact_match_original"])

        mean_shift = sum(shifts) / len(shifts) if shifts else float("nan")
        mean_shift_nn = (sum(shifts_non_novel) / len(shifts_non_novel)
                         if shifts_non_novel else float("nan"))

        print(f"{pos_name:<16} {n:>4} {n_parsed / n:>7.0%} {n_novel / max(n_parsed, 1):>7.0%} "
              f"{mean_shift:>7.2f} {mean_shift_nn:>7.2f} "
              f"{n_donor / max(n_parsed, 1):>7.0%} {n_orig / max(n_parsed, 1):>7.0%}")

        summary[pos_name] = {
            "n": n,
            "n_parsed": n_parsed,
            "n_novel": n_novel,
            "mean_shift_ratio": mean_shift,
            "mean_shift_ratio_non_novel": mean_shift_nn,
            "donor_match_rate": n_donor / max(n_parsed, 1),
            "original_match_rate": n_orig / max(n_parsed, 1),
        }

    print()
    print("Shift  = mean shift ratio (all parsed answers)")
    print("Shift* = mean shift ratio (excluding novel answers)")
    print("Novel% = fraction that matched neither donor nor original answer")

    with open(output_dir / "aggregate_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


# ==============================================================================
# Plotting
# ==============================================================================

def plot_results(results: List[dict], output_dir: Path):
    """Generate visualization plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plots")
        return

    by_position = defaultdict(list)
    for r in results:
        by_position[r["position_name"]].append(r)

    position_order = ["question_end", "filler_0.00", "filler_0.25",
                      "filler_0.50", "filler_0.75", "filler_1.00",
                      "answer_prompt"]
    positions_present = [p for p in position_order if p in by_position]
    short_labels = [p.replace("filler_", "f").replace("question_end", "q_end")
                    .replace("answer_prompt", "ans") for p in positions_present]

    # Plot 1: Shift ratio vs position (box plot)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Box plot of shift ratios
    ax = axes[0]
    box_data = []
    for pos in positions_present:
        shifts = [r["shift_ratio"] for r in by_position[pos]
                  if r["shift_ratio"] is not None]
        box_data.append(shifts)

    bp = ax.boxplot(box_data, tick_labels=short_labels, patch_artist=True,
                    widths=0.6, showfliers=True,
                    flierprops=dict(markersize=3, alpha=0.5))
    for patch in bp["boxes"]:
        patch.set_facecolor("#4CAF50")
        patch.set_alpha(0.5)

    ax.axhline(y=0, color="gray", linestyle=":", alpha=0.5, label="no effect")
    ax.axhline(y=1, color="red", linestyle="--", alpha=0.5, label="full adoption")
    ax.set_ylabel("Answer shift ratio")
    ax.set_title("Shift ratio by intervention start position")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # Mean shift + novel rate
    ax = axes[1]
    mean_shifts = []
    novel_rates = []
    n_patches = []
    for pos in positions_present:
        items = by_position[pos]
        shifts = [r["shift_ratio"] for r in items if r["shift_ratio"] is not None]
        mean_shifts.append(sum(shifts) / len(shifts) if shifts else 0)
        n_parsed = sum(1 for r in items if not r["parse_failure"])
        n_novel = sum(1 for r in items if r["novel_answer"] and not r["parse_failure"])
        novel_rates.append(n_novel / max(n_parsed, 1))
        n_patches.append(
            sum(r["n_positions_patched"] for r in items) / len(items)
            if items else 0
        )

    x = range(len(positions_present))
    ax.plot(x, mean_shifts, "o-", color="#4CAF50", linewidth=2,
            markersize=8, label="Mean shift ratio")
    ax2 = ax.twinx()
    ax2.bar(x, novel_rates, alpha=0.3, color="#FF9800", label="Novel answer rate")
    ax2.set_ylabel("Novel answer rate", color="#FF9800")
    ax2.set_ylim(0, 1)

    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, rotation=45, ha="right")
    ax.set_ylabel("Mean shift ratio")
    ax.set_title("Shift ratio and novel answer rate")
    ax.legend(loc="upper left", fontsize=9)
    ax2.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_dir / "patching_results.png", dpi=150, bbox_inches="tight")
    plt.savefig(output_dir / "patching_results.pdf", bbox_inches="tight")
    plt.close()
    print(f"Saved plots to {output_dir}/")


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Activation patching experiment")
    parser.add_argument("--model-path", type=str,
                        default="/workspace/models/deepseek-v3-awq")
    parser.add_argument("--dataset", type=Path,
                        default=Path("probing/data/1hop_addition_dataset.json"))
    parser.add_argument("--categories", type=Path,
                        default=Path("probing/probe_results/categories.json"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("probing/patching_results"))
    parser.add_argument("--max-pairs", type=int, default=107)
    parser.add_argument("--min-delta-a", type=int, default=10)
    parser.add_argument("--filler-type", type=str, default="dots")
    parser.add_argument("--filler-k", type=int, default=250)
    parser.add_argument("--skip-null-validation", action="store_true")
    parser.add_argument("--positions", nargs="+", default=None,
                        help="Positions to test (default: all)")
    parser.add_argument("--layer-start", type=int, default=0,
                        help="First layer to patch (for layer-band experiments)")
    parser.add_argument("--layer-end", type=int, default=61,
                        help="Last layer (exclusive) to patch")
    args = parser.parse_args()

    # Load dataset
    with open(args.dataset) as f:
        dataset = json.load(f)
    problems = dataset["examples"]
    few_shot = dataset["few_shot_facts"]
    print(f"Loaded {len(problems)} problems, {len(few_shot)} few-shot examples")

    # Add idx field if missing
    for i, p in enumerate(problems):
        if "idx" not in p:
            p["idx"] = i

    # Select pairs
    pairs = select_pairs(args.categories, args.max_pairs, args.min_delta_a)
    if not pairs:
        print("No valid pairs found!")
        return

    # Load model
    model, tokenizer = load_model(args.model_path)

    # Layer range
    layer_indices = list(range(args.layer_start, args.layer_end))
    print(f"Patching layers {args.layer_start}-{args.layer_end - 1} "
          f"({len(layer_indices)} layers)")

    # Null validation (Phase 0)
    if not args.skip_null_validation:
        capturer = DenseStateCapture(model, layer_indices)
        capturer.register_hooks()

        # Get some filler_helped problems for null test
        with open(args.categories) as f:
            cat_data = json.load(f)
        fh_idxs = [e["problem_idx"] for e in cat_data["examples"]
                    if e["category"] == "filler_helped"][:10]
        null_problems = [problems[i] for i in fh_idxs]

        passed = run_null_validation(
            model, tokenizer, null_problems, few_shot,
            layer_indices, capturer,
            n_examples=10, filler_type=args.filler_type, k=args.filler_k,
        )
        capturer.remove_hooks()

        if not passed:
            print("\nABORTING: Null validation failed. Fix hooks before proceeding.")
            return

    # Main experiment (Phase 1)
    positions_to_test = args.positions or list(INTERVENTION_POSITIONS.keys())
    results = run_experiment(
        model, tokenizer, pairs, problems, few_shot,
        layer_indices, args.output_dir,
        filler_type=args.filler_type, k=args.filler_k,
        positions_to_test=positions_to_test,
    )

    # Aggregate and plot
    aggregate_results(results, args.output_dir)
    plot_results(results, args.output_dir)


if __name__ == "__main__":
    main()
