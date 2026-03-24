"""
baseline_patching.py

Cross-condition patching: transplant dots_50 pre_answer states into baseline
(no filler) targets. Tests whether filler's contribution can be bypassed by
injecting the final hidden state.

If L60 pre_answer patching makes baseline targets produce the correct (donor)
answer, filler's entire role is building that one hidden state.
"""

import argparse
import json
import random
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

import transformers.activations as _act
if not hasattr(_act, "PytorchGELUTanh"):
    _act.PytorchGELUTanh = _act.GELUTanh

from extract_hidden_states import (
    build_messages_for_condition,
    find_filler_boundaries,
    load_model,
    load_tokenizer,
)
from filler_patching import (
    DenseStateCapture,
    ContiguousInjector,
    parse_answer,
    generate_answer,
    run_patched_generation,
    select_pairs,
    PRE_ANSWER_PREFIX,
)


# (position, layer_set) conditions to test
# position: "pre_answer" patches last token of baseline+"Answer: "
#           "question_end" patches baseline's question_end token
#           "answer_prompt" patches last token before generation
PATCH_CONDITIONS = {
    # Pre-answer position (right before the number)
    "pre_answer_all":    ("pre_answer", list(range(61))),
    "pre_answer_L60":    ("pre_answer", [60]),
    "pre_answer_L59_60": ("pre_answer", [59, 60]),
    "pre_answer_L55_60": ("pre_answer", list(range(55, 61))),
    # Question end position (last question token, before filler)
    "qend_all":          ("question_end", list(range(61))),
    "qend_L60":          ("question_end", [60]),
    "qend_L55_60":       ("question_end", list(range(55, 61))),
    # Answer prompt (generation prompt token, before "Answer: ")
    "ans_prompt_all":    ("answer_prompt", list(range(61))),
    "ans_prompt_L60":    ("answer_prompt", [60]),
}


def extract_source_states(
    model, tokenizer, problems, source_indices, few_shot,
    layer_indices, capturer,
):
    """Extract dots_50 pre_answer and question_end states for sources."""
    source_data = {}
    first_param = next(model.parameters())
    input_device = first_param.device

    prefix_ids = tokenizer.encode(PRE_ANSWER_PREFIX, add_special_tokens=False)

    print(f"Extracting dots_50 states for {len(source_indices)} sources...")
    t0 = time.time()

    for idx in tqdm(sorted(source_indices), desc="Source extraction"):
        problem = {**problems[idx], "idx": idx}
        example_rng = random.Random(idx)
        messages = build_messages_for_condition(
            few_shot[:5], problem, "dots", 50, rng=example_rng
        )
        full_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(full_text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(input_device)
        attention_mask = inputs["attention_mask"].to(input_device)
        seq_len = input_ids.shape[1]

        # Find question_end position
        question_end_pos, filler_start, filler_end = find_filler_boundaries(
            tokenizer, input_ids, 50
        )

        # Capture question_end and answer_prompt from standard forward pass
        answer_prompt_pos = seq_len - 1
        question_end_states = capturer.capture(
            input_ids, attention_mask, [question_end_pos, answer_prompt_pos]
        )
        # Split: index 0 = question_end, index 1 = answer_prompt
        answer_prompt_states = {l: question_end_states[l][1:2] for l in layer_indices}
        question_end_states = {l: question_end_states[l][0:1] for l in layer_indices}

        # Capture pre_answer state from extended forward pass
        ext_ids = torch.cat([
            input_ids,
            torch.tensor([prefix_ids], device=input_device),
        ], dim=1)
        ext_mask = torch.ones_like(ext_ids)
        pre_answer_pos = ext_ids.shape[1] - 1

        pre_answer_states = capturer.capture(ext_ids, ext_mask, [pre_answer_pos])

        source_data[idx] = {
            "pre_answer_states": pre_answer_states,
            "question_end_states": question_end_states,
            "answer_prompt_states": answer_prompt_states,
        }

        del input_ids, attention_mask, ext_ids, ext_mask
        torch.cuda.empty_cache()

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.0f}s ({elapsed / len(source_indices):.1f}s each)")
    return source_data


def run_baseline_patching(
    model, tokenizer, problems, few_shot, pairs, source_data,
    layer_indices, output_dir,
):
    """Patch dots_50 pre_answer states into baseline targets."""
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "checkpoint.json"

    first_param = next(model.parameters())
    input_device = first_param.device
    prefix_ids = tokenizer.encode(PRE_ANSWER_PREFIX, add_special_tokens=False)

    completed = {}
    results = []
    if checkpoint.exists():
        with open(checkpoint) as f:
            results = json.load(f).get("results", [])
        for r in results:
            completed[(r["target_idx"], r["source_idx"], r["layer_cond"])] = True
        print(f"  Resuming: {len(completed)} done")

    # Get natural baseline answers (no filler)
    print("Getting natural baseline answers...")
    natural_answers = {}
    for pair in pairs:
        tgt_idx = pair["target_idx"]
        if tgt_idx in natural_answers:
            continue
        problem = {**problems[tgt_idx], "idx": tgt_idx}
        example_rng = random.Random(tgt_idx)
        messages = build_messages_for_condition(
            few_shot[:5], problem, "dots", 0, rng=example_rng
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
    print(f"  {len(natural_answers)} targets, baseline accuracy: "
          f"{sum(1 for p in pairs if natural_answers[p['target_idx']] == p['target_answer']) / len(pairs):.1%}")

    # Also get natural dots_50 answers for comparison
    print("Getting natural dots_50 answers...")
    dots50_answers = {}
    for pair in pairs:
        tgt_idx = pair["target_idx"]
        if tgt_idx in dots50_answers:
            continue
        problem = {**problems[tgt_idx], "idx": tgt_idx}
        example_rng = random.Random(tgt_idx)
        messages = build_messages_for_condition(
            few_shot[:5], problem, "dots", 50, rng=example_rng
        )
        full_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(full_text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(input_device)
        attention_mask = inputs["attention_mask"].to(input_device)
        _, ans = generate_answer(model, tokenizer, input_ids, attention_mask)
        dots50_answers[tgt_idx] = ans
        del input_ids, attention_mask
        torch.cuda.empty_cache()
    print(f"  dots_50 accuracy: "
          f"{sum(1 for p in pairs if dots50_answers[p['target_idx']] == p['target_answer']) / len(pairs):.1%}")

    # Run interventions
    total = len(pairs) * len(PATCH_CONDITIONS)
    print(f"\nRunning {len(pairs)} pairs × {len(PATCH_CONDITIONS)} conditions = {total}")
    t0 = time.time()

    for pair_i, pair in enumerate(tqdm(pairs, desc="Baseline patching")):
        tgt_idx = pair["target_idx"]
        src_idx = pair["source_idx"]
        src_data = source_data[src_idx]

        # Build baseline target prompt
        problem = {**problems[tgt_idx], "idx": tgt_idx}
        example_rng = random.Random(tgt_idx)
        messages = build_messages_for_condition(
            few_shot[:5], problem, "dots", 0, rng=example_rng
        )
        full_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(full_text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(input_device)
        attention_mask = inputs["attention_mask"].to(input_device)
        seq_len = input_ids.shape[1]

        # Extended input with "Answer: " for pre_answer conditions
        ext_ids = torch.cat([
            input_ids,
            torch.tensor([prefix_ids], device=input_device),
        ], dim=1)
        ext_mask = torch.ones_like(ext_ids)
        pre_ans_pos = ext_ids.shape[1] - 1

        # Baseline positions
        tgt_answer_prompt_pos = seq_len - 1
        tgt_question_end_pos = seq_len - 2  # baseline: token before generation prompt

        for cond_name, (pos_type, cond_layers) in PATCH_CONDITIONS.items():
            key = (tgt_idx, src_idx, cond_name)
            if key in completed:
                continue

            # Select source states and target position based on position type
            if pos_type == "pre_answer":
                src_states = src_data["pre_answer_states"]
                patch_pos = [pre_ans_pos]
                run_ids, run_mask = ext_ids, ext_mask
            elif pos_type == "question_end":
                src_states = src_data["question_end_states"]
                patch_pos = [tgt_question_end_pos]
                run_ids, run_mask = input_ids, attention_mask
            elif pos_type == "answer_prompt":
                src_states = src_data["answer_prompt_states"]
                patch_pos = [tgt_answer_prompt_pos]
                run_ids, run_mask = input_ids, attention_mask

            patch_states = {l: src_states[l] for l in cond_layers}
            injector = ContiguousInjector(model, cond_layers, patch_pos, patch_states)
            raw, patched_ans = run_patched_generation(
                model, tokenizer, run_ids, run_mask, injector
            )

            natural_ans = natural_answers[tgt_idx]
            results.append({
                "target_idx": tgt_idx,
                "source_idx": src_idx,
                "cond_name": cond_name,
                "pos_type": pos_type,
                "A_target": pair["target_A"],
                "A_source": pair["source_A"],
                "Y": pair["Y"],
                "delta_A": pair["delta_A"],
                "baseline_natural": natural_ans,
                "dots50_natural": dots50_answers[tgt_idx],
                "expected_correct": pair["target_answer"],
                "expected_donor": pair["source_answer"],
                "patched_answer": patched_ans,
                "raw_response": raw,
                "matches_correct": patched_ans == pair["target_answer"],
                "matches_donor": patched_ans == pair["source_answer"],
                "matches_baseline": patched_ans == natural_ans,
                "parse_failure": patched_ans is None,
            })
            completed[key] = True

            del patch_states
            torch.cuda.empty_cache()

        if (pair_i + 1) % 5 == 0 or pair_i == len(pairs) - 1:
            with open(checkpoint, "w") as f:
                json.dump({"results": results}, f)

        del input_ids, attention_mask, ext_ids, ext_mask
        torch.cuda.empty_cache()

    elapsed = time.time() - t0
    print(f"Done: {len(results)} results in {elapsed:.0f}s")
    return results


def print_results(results, pairs):
    by_cond = defaultdict(list)
    for r in results:
        by_cond[r["cond_name"]].append(r)

    print(f"\n{'='*80}")
    print("Baseline target + dots_50 source patching")
    print(f"{'='*80}")
    print(f"{'Condition':<20} {'N':>4} {'Correct%':>9} {'Donor%':>8} {'Baseline%':>10} {'Novel%':>8}")
    print("-" * 65)

    for cond in PATCH_CONDITIONS:
        if cond not in by_cond:
            continue
        items = by_cond[cond]
        n = len(items)
        n_parsed = sum(1 for r in items if not r["parse_failure"])
        correct = sum(1 for r in items if r["matches_correct"]) / max(n_parsed, 1)
        donor = sum(1 for r in items if r["matches_donor"]) / max(n_parsed, 1)
        baseline = sum(1 for r in items if r["matches_baseline"]) / max(n_parsed, 1)
        novel = sum(1 for r in items if not r["matches_correct"] and
                    not r["matches_donor"] and not r["matches_baseline"] and
                    not r["parse_failure"]) / max(n_parsed, 1)
        print(f"{cond:<20} {n:>4} {correct:>9.1%} {donor:>8.1%} {baseline:>10.1%} {novel:>8.1%}")

    # Reference accuracies
    first_per_target = {}
    for r in results:
        if r["target_idx"] not in first_per_target:
            first_per_target[r["target_idx"]] = r
    baseline_acc = sum(1 for r in first_per_target.values()
                       if r["baseline_natural"] == r["expected_correct"]) / max(len(first_per_target), 1)
    print(f"\nReference: baseline natural accuracy on these targets: {baseline_acc:.1%}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/workspace/models/deepseek-v3-awq")
    parser.add_argument("--dataset", type=Path,
                        default=Path("probing/data/1hop_addition_dataset.json"))
    parser.add_argument("--categories", type=Path,
                        default=Path("probing/probe_results/categories.json"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("probing/patching_results_v2/baseline_cross"))
    parser.add_argument("--max-pairs", type=int, default=200)
    args = parser.parse_args()

    with open(args.dataset) as f:
        dataset = json.load(f)
    few_shot = dataset["few_shot_facts"]
    problems = dataset["examples"]
    for i, p in enumerate(problems):
        if "idx" not in p:
            p["idx"] = i

    pairs = select_pairs(args.categories, args.max_pairs)

    model, tokenizer = load_model(args.model_path)
    layer_indices = list(range(61))

    capturer = DenseStateCapture(model, layer_indices)
    capturer.register_hooks()
    source_indices = set(p["source_idx"] for p in pairs)
    source_data = extract_source_states(
        model, tokenizer, problems, source_indices, few_shot,
        layer_indices, capturer,
    )
    capturer.remove_hooks()

    results = run_baseline_patching(
        model, tokenizer, problems, few_shot, pairs, source_data,
        layer_indices, args.output_dir,
    )

    print_results(results, pairs)
    with open(args.output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nDone!")


if __name__ == "__main__":
    main()
