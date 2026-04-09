"""
Quick test: generate one prompt, tokenize it, and print the boundaries
that filler_with_label mode would transplant.

Usage:
    python scripts/test_filler_label_boundaries.py \
        --model-path /workspace/models/deepseek-v3-awq \
        --dataset data/1hop_addition_dataset.json \
        --filler-k 10
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from extract_hidden_states import find_filler_boundaries
from filler_kv_transplant import find_filler_label_start
from prompt_utils import build_messages


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str,
                        default="/workspace/models/deepseek-v3-awq")
    parser.add_argument("--dataset", type=Path,
                        default=Path("data/1hop_addition_dataset.json"))
    parser.add_argument("--filler-k", type=int, default=10)
    parser.add_argument("--filler-type", choices=["dots", "counting", "alphabet"],
                        default="dots")
    args = parser.parse_args()

    with open(args.dataset) as f:
        dataset = json.load(f)
    few_shot = dataset["few_shot_facts"][:5]
    prob = dataset["examples"][0]

    print(f"Problem: {prob}")
    print(f"Filler k={args.filler_k}, type={args.filler_type}\n")

    print(f"Loading tokenizer from {args.model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    msgs = build_messages(few_shot, prob, args.filler_type, args.filler_k)
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt")
    ids = inputs["input_ids"]

    qend, filler_start, filler_end = find_filler_boundaries(tokenizer, ids, args.filler_k)

    # filler_with_label range
    label_start = find_filler_label_start(tokenizer, ids, filler_start)
    label_end = filler_end
    seq_len = ids.shape[1]

    id_list = ids[0].tolist()

    def tok(pos):
        return repr(tokenizer.decode([id_list[pos]]))

    print(f"Sequence length : {seq_len} tokens")
    print(f"qend            : {qend:4d}  → {tok(qend)}")
    print(f"filler_label    : {label_start:4d}  → {tok(label_start)}  (first token transplanted, start of 'Filler:')")
    print(f"filler_start    : {filler_start:4d}  → {tok(filler_start)}")
    print(f"filler_end      : {filler_end:4d}  → {tok(filler_end)}  (last token transplanted)")
    print(f"filler_end+1    : {filler_end+1:4d}  → {tok(filler_end+1)}  (first token NOT transplanted)")
    print(f"last token      : {seq_len-1:4d}  → {tok(seq_len-1)}")
    print()

    print("── filler_with_label span (label_start … filler_end) ──")
    for i in range(label_start, label_end + 1):
        print(f"  [{i:4d}] {tok(i)}")

    print()
    print("── tokens just after span (not transplanted) ──")
    for i in range(label_end + 1, min(label_end + 6, seq_len)):
        print(f"  [{i:4d}] {tok(i)}")


if __name__ == "__main__":
    main()
