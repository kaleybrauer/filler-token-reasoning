#!/usr/bin/env python3
"""Inspect tokenized examples from a 2-hop dataset to verify correctness."""
import argparse
import json
import random

from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="Path to train.jsonl or test.jsonl")
    parser.add_argument("--tokenizer", required=True, help="Tokenizer name")
    parser.add_argument("-n", type=int, default=5, help="Number of examples to show")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)

    with open(args.file) as f:
        rows = [json.loads(line) for line in f]

    random.seed(args.seed)
    sample = random.sample(rows, min(args.n, len(rows)))

    for i, row in enumerate(sample):
        print(f"{'='*70}")
        print(f"Example {i+1}: {row['id']}  |  N={row['filler_len']}  |  "
              f"a1={row['a1']} + a2={row['a2']} = {row['answer']}")
        print(f"{'='*70}")

        input_ids = row["input_ids"]
        labels = row["labels"]

        # Decode the full sequence
        full_text = tok.decode(input_ids, skip_special_tokens=False)

        # Find where labels start (first non -100)
        label_start = next(j for j, l in enumerate(labels) if l != -100)
        prompt_ids = input_ids[:label_start]
        answer_ids = input_ids[label_start:]

        prompt_text = tok.decode(prompt_ids, skip_special_tokens=False)
        answer_text = tok.decode(answer_ids, skip_special_tokens=False)

        # Show the last 200 chars of prompt (fewshot prefix is long)
        print(f"\nPrompt (last 200 chars):")
        print(f"  ...{prompt_text[-200:]}")
        print(f"\nSupervised answer tokens ({len(answer_ids)} tokens):")
        print(f"  text: {answer_text!r}")
        print(f"  ids:  {answer_ids}")
        print(f"  label ids: {labels[label_start:]}")

        # Verify labels match input_ids where not -100
        mismatches = []
        for j, (inp, lab) in enumerate(zip(input_ids, labels)):
            if lab != -100 and inp != lab:
                mismatches.append((j, inp, lab))
        if mismatches:
            print(f"\n  WARNING: LABEL MISMATCH at {len(mismatches)} positions!")
            for pos, inp, lab in mismatches[:5]:
                print(f"    pos {pos}: input_id={inp} label={lab}")
        else:
            print(f"  GOOD: Labels match input_ids in supervised region")

        # Verify attention mask
        attn = row["attention_mask"]
        if all(a == 1 for a in attn) and len(attn) == len(input_ids):
            print(f"  GOOD: Attention mask: all 1s, length matches ({len(attn)})")
        else:
            print(f"  WARNING: Attention mask issue: len={len(attn)} vs input={len(input_ids)}")

        # Check sequence length
        print(f"\n  Total tokens: {len(input_ids)} (prompt: {len(prompt_ids)}, answer: {len(answer_ids)})")
        print()


if __name__ == "__main__":
    main()