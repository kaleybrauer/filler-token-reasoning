"""
extract_filler_attention.py

What do filler tokens attend to during prefill?

For each filler position, extract the attention distribution over all
previous tokens, aggregated by position group (system, few_shot, question,
earlier_filler). This reveals whether filler tokens attend to the fact
phrase, the number, or just generic context.

Usage:
    source /workspace/config/probing_env.sh
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python scripts/extract_filler_attention.py \
        --model-path /workspace/models/deepseek-v3-awq \
        --max-examples 100 \
        --output-dir results/filler_attention
"""

import argparse
import json
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from tqdm import tqdm

import transformers.activations as _act
if not hasattr(_act, "PytorchGELUTanh"):
    _act.PytorchGELUTanh = _act.GELUTanh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from extract.extract_hidden_states import find_filler_boundaries, load_model
from extract.extract_answer_attention import (
    AttentionExtractor,
    classify_positions,
    aggregate_attention_by_group,
)
from prompt_utils import build_messages


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/workspace/models/deepseek-v3-awq")
    parser.add_argument("--dataset", type=Path,
                        default=Path("data/1hop_addition_dataset.json"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/filler_attention"))
    parser.add_argument("--layers", type=str,
                        default="0,10,20,30,40,50,55,58,60")
    parser.add_argument("--max-examples", type=int, default=100)
    parser.add_argument("--ks", type=str, default="0,100",
                        help="Comma-separated filler k values (0=baseline)")
    parser.add_argument("--filler-query-positions", type=str, default="1,5,10,25,50,100",
                        help="Which filler tokens to use as queries (1-indexed)")
    args = parser.parse_args()

    layer_indices = [int(x) for x in args.layers.split(",")]
    filler_query_ks = [int(x) for x in args.filler_query_positions.split(",")]
    k_values = [int(x) for x in args.ks.split(",")]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.dataset) as f:
        dataset = json.load(f)
    few_shot = dataset["few_shot_facts"]
    problems = dataset["examples"][:args.max_examples]

    model, tokenizer = load_model(args.model_path)
    input_device = next(model.parameters()).device

    extractor = AttentionExtractor(model, layer_indices)
    extractor.register_hooks()

    groups_order = ["system", "few_shot", "question", "filler", "filler_label",
                    "answer_label", "assistant", "format", "other"]

    for k in k_values:
        query_ks = [fk for fk in filler_query_ks if fk <= k]

        print(f"\n{'#'*70}")
        print(f"  dots k={k}, querying filler positions {query_ks}")
        print(f"{'#'*70}")

        # Query labels: filler positions + special positions
        # "filler_label" = the Filler: format tokens before dots
        # "answer_label" = the Answer: tokens after dots
        # "answer_pos" = the <assistant> token (last input token)
        special_queries = ["filler_label", "answer_label", "answer_pos"]
        all_query_keys = list(query_ks) + special_queries

        all_group_attn = {li: {qk: defaultdict(list) for qk in all_query_keys}
                          for li in layer_indices}

        t0 = time.time()
        for prob_idx, problem in enumerate(tqdm(problems, desc=f"k{k}")):
            mode = "baseline" if k == 0 else "dots"
            messages = build_messages(few_shot[:5], problem, mode, k)
            full_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(full_text, return_tensors="pt")
            input_ids = inputs["input_ids"].to(input_device)
            attention_mask = inputs["attention_mask"].to(input_device)
            seq_len = input_ids.shape[1]

            filler_start, filler_end = -1, -1
            if k > 0:
                _, filler_start, filler_end = find_filler_boundaries(tokenizer, input_ids, k)

            position_groups = classify_positions(
                tokenizer, input_ids, filler_start, filler_end, seq_len
            )

            extractor._captured = {}
            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attention_mask,
                      output_attentions=True)

            # Query from filler positions
            for fk in query_ks:
                if k == 0:
                    continue  # no filler in baseline
                filler_pos = filler_start + fk - 1
                if filler_pos > filler_end:
                    continue

                for li in layer_indices:
                    if li not in extractor._captured:
                        continue
                    attn_w = extractor._captured[li]
                    agg = aggregate_attention_by_group(
                        attn_w, position_groups, [filler_pos]
                    )
                    for g, v in agg.items():
                        all_group_attn[li][fk][g].append(v[:, 0])

            # Query from special positions
            for sq in special_queries:
                if sq == "answer_pos":
                    q_positions = [seq_len - 1]
                elif sq in position_groups and position_groups[sq]:
                    q_positions = position_groups[sq]
                else:
                    continue

                for li in layer_indices:
                    if li not in extractor._captured:
                        continue
                    attn_w = extractor._captured[li]
                    agg = aggregate_attention_by_group(
                        attn_w, position_groups, q_positions
                    )
                    # Average across query positions in the group
                    for g, v in agg.items():
                        all_group_attn[li][sq][g].append(v.mean(axis=1))  # mean over queries

            extractor._captured = {}
            del input_ids, attention_mask
            torch.cuda.empty_cache()

        elapsed = time.time() - t0
        print(f"\n{len(problems)} examples in {elapsed:.0f}s")

        for fk in all_query_keys:
            if not any(all_group_attn[li][fk] for li in layer_indices):
                continue
            label = f"filler k={fk}" if isinstance(fk, int) else fk
            print(f"\n{'='*70}")
            print(f"  k={k}: {label} attending to:")
            print(f"{'='*70}")

            present_groups = set()
            for li in layer_indices:
                present_groups.update(all_group_attn[li][fk].keys())
            groups_sorted = [g for g in groups_order if g in present_groups]
            if not groups_sorted:
                groups_sorted = sorted(present_groups)

            header = f"  {'Layer':<8}" + "".join(f"{g[:10]:>12}" for g in groups_sorted)
            print(header)
            print(f"  {'-' * (len(header) - 2)}")

            for li in layer_indices:
                row = f"  L{li:<6}"
                for g in groups_sorted:
                    if g in all_group_attn[li][fk] and all_group_attn[li][fk][g]:
                        vals = np.stack(all_group_attn[li][fk][g])
                        row += f"{vals.mean():>12.4f}"
                    else:
                        row += f"{'—':>12}"
                print(row)

        summary = {}
        for fk in all_query_keys:
            key = str(fk)
            summary[key] = {}
            for li in layer_indices:
                summary[key][li] = {}
                for g in all_group_attn[li][fk]:
                    if all_group_attn[li][fk][g]:
                        vals = np.stack(all_group_attn[li][fk][g])
                        summary[key][li][g] = {
                            "mean": float(vals.mean()),
                            "std": float(vals.std()),
                        }

        label = "baseline" if k == 0 else f"dots_k{k}"
        with open(args.output_dir / f"{label}_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

    extractor.remove_hooks()
    print(f"\nResults saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
