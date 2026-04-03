"""
Run KV transplant for all 3 conditions in one process (one model load).
"""

import sys
import time
from pathlib import Path

import torch

import transformers.activations as _act
if not hasattr(_act, "PytorchGELUTanh"):
    _act.PytorchGELUTanh = _act.GELUTanh

sys.path.insert(0, str(Path(__file__).parent))

# Monkey-patch sys.argv before importing main
import filler_kv_transplant
from extract_hidden_states import load_model
import json

def run_condition(model, tokenizer, filler_k, filler_type, output_dir, max_pairs=500):
    """Run one transplant condition using already-loaded model."""
    import argparse
    args = argparse.Namespace(
        model_path=None,
        dataset=Path("probing/data/1hop_addition_dataset.json"),
        output_dir=Path(output_dir),
        filler_k=filler_k,
        max_pairs=max_pairs,
        min_a_diff=10,
        max_new_tokens=5,
        mode="filler",
        filler_type=filler_type,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.dataset) as f:
        dataset = json.load(f)
    few_shot = dataset["few_shot_facts"]
    problems = dataset["examples"]

    # Override the model loading in the main loop — use our model
    import random
    import numpy as np
    from tqdm import tqdm
    from filler_kv_transplant import (
        find_y_matched_pairs, prefill_and_cache, generate_from_cache,
        transplant_kv, find_question_boundaries, find_filler_boundaries,
        extract_answer,
    )
    from prompt_utils import build_messages
    from transformers.cache_utils import DynamicCache

    device = next(model.parameters()).device
    pairs = find_y_matched_pairs(problems, args.min_a_diff, args.max_pairs)
    print(f"\nFound {len(pairs)} Y-matched pairs (|ΔA| >= {args.min_a_diff})")

    results = []
    donor_adopted = 0
    target_kept = 0
    novel = 0
    n_valid = 0

    filler_mode = "counting" if filler_type == "counting" else "dots"

    for pair_idx, (idx_a, idx_b) in enumerate(tqdm(pairs, desc=f"{filler_type}_k{filler_k}")):
        prob_a = problems[idx_a]
        prob_b = problems[idx_b]

        msgs_a = build_messages(few_shot[:5], prob_a, filler_mode, filler_k)
        msgs_b = build_messages(few_shot[:5], prob_b, filler_mode, filler_k)

        from extract_hidden_states import load_tokenizer
        tokenizer_local = tokenizer

        text_a = tokenizer_local.apply_chat_template(msgs_a, tokenize=False, add_generation_prompt=True)
        text_b = tokenizer_local.apply_chat_template(msgs_b, tokenize=False, add_generation_prompt=True)

        inputs_a = tokenizer_local(text_a, return_tensors="pt")
        inputs_b = tokenizer_local(text_b, return_tensors="pt")

        ids_a = inputs_a["input_ids"].to(device)
        mask_a = inputs_a["attention_mask"].to(device)
        ids_b = inputs_b["input_ids"].to(device)
        mask_b = inputs_b["attention_mask"].to(device)

        qend_a, filler_start_a, filler_end_a = find_filler_boundaries(tokenizer_local, ids_a, filler_k)
        qend_b, filler_start_b, filler_end_b = find_filler_boundaries(tokenizer_local, ids_b, filler_k)

        cache_a, _ = prefill_and_cache(model, tokenizer_local, ids_a, mask_a)
        cache_b, _ = prefill_and_cache(model, tokenizer_local, ids_b, mask_b)

        last_token_b = ids_b[0, -1].item()

        # Token IDs
        donor_ay_tok = tokenizer_local.encode(str(prob_a["answer"]), add_special_tokens=False)
        target_ay_tok = tokenizer_local.encode(str(prob_b["answer"]), add_special_tokens=False)
        donor_tok_id = donor_ay_tok[0] if len(donor_ay_tok) == 1 else None
        target_tok_id = target_ay_tok[0] if len(target_ay_tok) == 1 else None
        donor_a_tok = tokenizer_local.encode(str(prob_a["fact_value"]), add_special_tokens=False)
        target_a_tok = tokenizer_local.encode(str(prob_b["fact_value"]), add_special_tokens=False)
        donor_a_tok_id = donor_a_tok[0] if len(donor_a_tok) == 1 else None
        target_a_tok_id = target_a_tok[0] if len(target_a_tok) == 1 else None

        # Control: truncate last entry, re-feed last token
        cache_b_trunc = DynamicCache()
        for li in range(len(cache_b)):
            k, v = cache_b[li]
            cache_b_trunc.update(k[:, :, :-1, :], v[:, :, :-1, :], layer_idx=li)
        raw_b_normal, ans_b_normal, logits_normal = generate_from_cache(
            model, tokenizer_local, last_token_b, cache_b_trunc, args.max_new_tokens
        )

        # Transplant
        cache_transplant = transplant_kv(
            cache_b, cache_a,
            filler_start_b, filler_end_b,
            filler_start_a, filler_end_a,
        )
        raw_transplant, ans_transplant, logits_transplant = generate_from_cache(
            model, tokenizer_local, last_token_b, cache_transplant, args.max_new_tokens
        )

        # Ranks
        def get_rank(logits_arr, tok_id):
            if tok_id is None:
                return None
            logit = float(logits_arr[tok_id])
            return int((logits_arr > logit).sum()) + 1

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

        results.append({
            "pair_idx": pair_idx,
            "idx_a": idx_a, "idx_b": idx_b,
            "A_donor": prob_a["fact_value"], "A_target": prob_b["fact_value"],
            "Y": prob_a["x"],
            "answer_donor": prob_a["answer"], "answer_target": prob_b["answer"],
            "normal_answer": ans_b_normal, "transplant_answer": ans_transplant,
            "outcome": outcome,
            "donor_ay_rank_normal": get_rank(logits_normal, donor_tok_id),
            "donor_ay_rank_transplant": get_rank(logits_transplant, donor_tok_id),
            "target_ay_rank_normal": get_rank(logits_normal, target_tok_id),
            "target_ay_rank_transplant": get_rank(logits_transplant, target_tok_id),
            "donor_a_rank_normal": get_rank(logits_normal, donor_a_tok_id),
            "donor_a_rank_transplant": get_rank(logits_transplant, donor_a_tok_id),
            "target_a_rank_normal": get_rank(logits_normal, target_a_tok_id),
            "target_a_rank_transplant": get_rank(logits_transplant, target_a_tok_id),
        })

        del cache_a, cache_b, cache_b_trunc, cache_transplant
        torch.cuda.empty_cache()

        if (pair_idx + 1) % 50 == 0:
            valid_shifts = [r["donor_ay_rank_normal"] - r["donor_ay_rank_transplant"]
                           for r in results if r["donor_ay_rank_normal"] is not None]
            med = np.median(valid_shifts) if valid_shifts else 0
            print(f"  [{pair_idx+1}/{len(pairs)}] donor={donor_adopted} target={target_kept} "
                  f"novel={novel} med_shift={med:+.0f}")

    # Summary
    valid = [r for r in results if r["donor_ay_rank_normal"] is not None]
    def pct(vals): return np.percentile(vals, 16), np.median(vals), np.percentile(vals, 84)

    print(f"\n{'='*60}")
    print(f"SUMMARY: {filler_type} k={filler_k} ({n_valid} pairs)")
    print(f"{'='*60}")
    print(f"  Donor adopted: {donor_adopted} ({donor_adopted/max(n_valid,1):.1%})")
    print(f"  Target kept:   {target_kept} ({target_kept/max(n_valid,1):.1%})")
    print(f"  Novel:         {novel} ({novel/max(n_valid,1):.1%})")

    for label, key_n, key_t in [
        ("Donor A+Y", "donor_ay_rank_normal", "donor_ay_rank_transplant"),
        ("Target A+Y", "target_ay_rank_normal", "target_ay_rank_transplant"),
        ("Donor A", "donor_a_rank_normal", "donor_a_rank_transplant"),
        ("Target A", "target_a_rank_normal", "target_a_rank_transplant"),
    ]:
        vals_n = [r[key_n] for r in valid if r[key_n] is not None]
        vals_t = [r[key_t] for r in valid if r[key_t] is not None]
        if vals_n:
            p = pct(vals_n)
            pt = pct(vals_t)
            print(f"\n  {label} rank:  p16  median  p84")
            print(f"    Normal:     {p[0]:>6.0f}  {p[1]:>6.0f}  {p[2]:>6.0f}")
            print(f"    Transplant: {pt[0]:>6.0f}  {pt[1]:>6.0f}  {pt[2]:>6.0f}")

    with open(args.output_dir / "transplant_results.json", "w") as f:
        json.dump({"summary": {"n_valid": n_valid, "donor_adopted": donor_adopted,
                                "target_kept": target_kept, "novel": novel},
                   "results": results}, f, indent=2)
    print(f"\n  Saved to {args.output_dir}/")


def main():
    model, tokenizer = load_model("/workspace/models/deepseek-v3-awq")

    conditions = [
        (100, "dots", "probing/results/kv_transplant_v3/dots_100"),
        (10, "dots", "probing/results/kv_transplant_v3/dots_10"),
        (25, "counting", "probing/results/kv_transplant_v3/counting_25"),
    ]

    for filler_k, filler_type, output_dir in conditions:
        run_condition(model, tokenizer, filler_k, filler_type, output_dir, max_pairs=500)

    print("\nAll conditions complete.")


if __name__ == "__main__":
    main()
