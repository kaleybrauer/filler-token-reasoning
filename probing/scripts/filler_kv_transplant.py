"""
filler_kv_transplant.py

Test the relay hypothesis: do filler KV entries carry example-specific
information about A?

For Y-matched pairs (same Y, different A):
1. Run both examples through prefill with filler
2. Replace filler positions in the KV cache of example B with those from A
3. Generate from the modified cache
4. Check if the answer shifts toward A's answer

If filler KV entries are relays carrying A-specific information,
the answer should shift toward the donor's A+Y.

Usage:
    source /workspace/config/probing_env.sh
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python probing/scripts/filler_kv_transplant.py \
        --model-path /workspace/models/deepseek-v3-awq \
        --max-pairs 100 \
        --output-dir probing/results/kv_transplant
"""

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

import transformers.activations as _act
if not hasattr(_act, "PytorchGELUTanh"):
    _act.PytorchGELUTanh = _act.GELUTanh

sys.path.insert(0, str(Path(__file__).parent))
from extract_hidden_states import find_filler_boundaries, load_model
from prompt_utils import build_messages, extract_answer


def find_y_matched_pairs(problems, min_a_diff=10, max_pairs=100):
    """Find pairs with same Y but different A (|ΔA| >= min_a_diff)."""
    from collections import defaultdict
    by_y = defaultdict(list)
    for i, p in enumerate(problems):
        by_y[p["x"]].append(i)

    pairs = []
    for y, indices in by_y.items():
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                a = problems[indices[i]]
                b = problems[indices[j]]
                if abs(a["fact_value"] - b["fact_value"]) >= min_a_diff:
                    pairs.append((indices[i], indices[j]))

    random.seed(42)
    random.shuffle(pairs)
    return pairs[:max_pairs]


def prefill_and_cache(model, tokenizer, input_ids, attention_mask):
    """Run prefill and return the KV cache."""
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
        )
    return outputs.past_key_values, outputs.logits


def transplant_filler_kv(target_cache, donor_cache,
                         target_filler_start, target_filler_end,
                         donor_filler_start, donor_filler_end,
                         truncate_last=True):
    """Replace filler KV entries in target cache with donor's.

    Supports different filler positions (different sequence lengths).
    Copies donor filler[i] → target filler[i] for matching offsets.
    """
    n_filler_target = target_filler_end - target_filler_start + 1
    n_filler_donor = donor_filler_end - donor_filler_start + 1
    n_copy = min(n_filler_target, n_filler_donor)

    new_cache = []
    for layer_idx in range(len(target_cache)):
        target_k, target_v = target_cache[layer_idx]
        donor_k, donor_v = donor_cache[layer_idx]

        new_k = target_k.clone()
        new_v = target_v.clone()
        ts = target_filler_start
        ds = donor_filler_start
        new_k[:, :, ts:ts + n_copy, :] = donor_k[:, :, ds:ds + n_copy, :]
        new_v[:, :, ts:ts + n_copy, :] = donor_v[:, :, ds:ds + n_copy, :]

        if truncate_last:
            new_k = new_k[:, :, :-1, :]
            new_v = new_v[:, :, :-1, :]

        new_cache.append((new_k, new_v))

    from transformers.cache_utils import DynamicCache
    cache = DynamicCache()
    for k, v in new_cache:
        cache.update(k, v, layer_idx=len(cache))
    return cache


def generate_from_cache(model, tokenizer, last_token_id, past_key_values, max_new_tokens=5):
    """Re-run the last token with a (possibly modified) KV cache, then generate.

    By feeding the last input token with the KV cache, we re-compute its
    attention over all cached positions. This is how we test what happens
    when filler KV entries are swapped — the answer position re-attends
    to the modified cache.

    Args:
        last_token_id: token id of the last input token (int)
        past_key_values: KV cache (may have transplanted filler entries)
    """
    device = next(model.parameters()).device
    input_id = torch.tensor([[last_token_id]], device=device)

    with torch.no_grad():
        outputs = model(
            input_ids=input_id,
            past_key_values=past_key_values,
            use_cache=True,
        )
    logits = outputs.logits[:, -1, :]
    past = outputs.past_key_values

    generated = []
    for step in range(max_new_tokens):
        next_token = logits.argmax(dim=-1, keepdim=True)
        if next_token.item() == tokenizer.eos_token_id:
            break
        generated.append(next_token.item())

        with torch.no_grad():
            outputs = model(
                input_ids=next_token,
                past_key_values=past,
                use_cache=True,
            )
        logits = outputs.logits[:, -1, :]
        past = outputs.past_key_values

    raw = tokenizer.decode(generated, skip_special_tokens=True).strip()
    answer = extract_answer(raw)
    return raw, answer


def main():
    parser = argparse.ArgumentParser(description="Filler KV transplant test")
    parser.add_argument("--model-path", type=str,
                        default="/workspace/models/deepseek-v3-awq")
    parser.add_argument("--dataset", type=Path,
                        default=Path("probing/data/1hop_addition_dataset.json"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("probing/results/kv_transplant"))
    parser.add_argument("--filler-k", type=int, default=100)
    parser.add_argument("--max-pairs", type=int, default=100)
    parser.add_argument("--min-a-diff", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=5)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.dataset) as f:
        dataset = json.load(f)
    few_shot = dataset["few_shot_facts"]
    problems = dataset["examples"]

    model, tokenizer = load_model(args.model_path)
    device = next(model.parameters()).device

    # Find Y-matched pairs
    pairs = find_y_matched_pairs(problems, args.min_a_diff, args.max_pairs)
    print(f"Found {len(pairs)} Y-matched pairs (|ΔA| >= {args.min_a_diff})")

    results = []
    donor_adopted = 0
    target_kept = 0
    novel = 0
    n_valid = 0

    for pair_idx, (idx_a, idx_b) in enumerate(tqdm(pairs, desc="transplant")):
        prob_a = problems[idx_a]
        prob_b = problems[idx_b]

        # Build prompts
        msgs_a = build_messages(few_shot[:5], prob_a, "dots", args.filler_k)
        msgs_b = build_messages(few_shot[:5], prob_b, "dots", args.filler_k)

        text_a = tokenizer.apply_chat_template(msgs_a, tokenize=False, add_generation_prompt=True)
        text_b = tokenizer.apply_chat_template(msgs_b, tokenize=False, add_generation_prompt=True)

        inputs_a = tokenizer(text_a, return_tensors="pt")
        inputs_b = tokenizer(text_b, return_tensors="pt")

        ids_a = inputs_a["input_ids"].to(device)
        mask_a = inputs_a["attention_mask"].to(device)
        ids_b = inputs_b["input_ids"].to(device)
        mask_b = inputs_b["attention_mask"].to(device)

        # Find filler boundaries for each
        _, filler_start_a, filler_end_a = find_filler_boundaries(tokenizer, ids_a, args.filler_k)
        _, filler_start_b, filler_end_b = find_filler_boundaries(tokenizer, ids_b, args.filler_k)

        # Prefill both separately (different seq lengths)
        cache_a, _ = prefill_and_cache(model, tokenizer, ids_a, mask_a)
        cache_b, _ = prefill_and_cache(model, tokenizer, ids_b, mask_b)

        # Get last token id from B (the answer position token)
        last_token_b = ids_b[0, -1].item()

        # Normal generation for B (control): re-run last token with B's cache
        raw_b_normal, ans_b_normal = generate_from_cache(
            model, tokenizer, last_token_b, cache_b, args.max_new_tokens
        )

        # Transplant: replace B's filler KV with A's filler KV
        cache_transplant = transplant_filler_kv(
            cache_b, cache_a,
            filler_start_b, filler_end_b,
            filler_start_a, filler_end_a,
        )

        # Generate from transplanted cache: re-run last token with modified KV
        raw_transplant, ans_transplant = generate_from_cache(
            model, tokenizer, last_token_b, cache_transplant, args.max_new_tokens
        )

        # Classify result
        n_valid += 1
        if ans_transplant is not None:
            if ans_transplant == prob_a["answer"]:
                outcome = "donor_adopted"
                donor_adopted += 1
            elif ans_transplant == prob_b["answer"]:
                outcome = "target_kept"
                target_kept += 1
            else:
                outcome = "novel"
                novel += 1
        else:
            outcome = "parse_failed"

        result = {
            "pair_idx": pair_idx,
            "idx_a": idx_a,
            "idx_b": idx_b,
            "A_donor": prob_a["fact_value"],
            "A_target": prob_b["fact_value"],
            "Y": prob_a["x"],
            "answer_donor": prob_a["answer"],
            "answer_target": prob_b["answer"],
            "normal_answer": ans_b_normal,
            "transplant_answer": ans_transplant,
            "transplant_raw": raw_transplant,
            "outcome": outcome,
        }
        results.append(result)

        del cache_a, cache_b, cache_transplant
        torch.cuda.empty_cache()

        # Print periodic summary
        if (pair_idx + 1) % 20 == 0:
            print(f"  [{pair_idx+1}/{len(pairs)}] donor={donor_adopted} "
                  f"target={target_kept} novel={novel}")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  Valid pairs: {n_valid}")
    print(f"  Donor adopted: {donor_adopted} ({donor_adopted/max(n_valid,1):.1%})")
    print(f"  Target kept:   {target_kept} ({target_kept/max(n_valid,1):.1%})")
    print(f"  Novel:         {novel} ({novel/max(n_valid,1):.1%})")

    # Save
    with open(args.output_dir / "transplant_results.json", "w") as f:
        json.dump({
            "summary": {
                "n_valid": n_valid,
                "donor_adopted": donor_adopted,
                "target_kept": target_kept,
                "novel": novel,
            },
            "results": results,
        }, f, indent=2)

    print(f"\nResults saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
