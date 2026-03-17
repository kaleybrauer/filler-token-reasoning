#!/usr/bin/env python3
"""Sample random outputs from eval_detailed.jsonl for inspection."""
import argparse
import json
import random

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="Path to eval_detailed.jsonl")
    parser.add_argument("-n", type=int, default=30, help="Number of samples")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wrong-only", action="store_true", help="Only show wrong answers")
    parser.add_argument("--correct-only", action="store_true", help="Only show correct answers")
    parser.add_argument("--filler-len", type=int, default=None, help="Filter to specific filler length")
    args = parser.parse_args()

    with open(args.file) as f:
        rows = [json.loads(line) for line in f]

    if args.wrong_only:
        rows = [r for r in rows if not r["correct"]]
    if args.correct_only:
        rows = [r for r in rows if r["correct"]]
    if args.filler_len is not None:
        rows = [r for r in rows if r["filler_len"] == args.filler_len]

    random.seed(args.seed)
    sample = random.sample(rows, min(args.n, len(rows)))

    for i, r in enumerate(sample):
        status = "✓" if r["correct"] else "✗"
        print(f"[{i+1}] {status}  N={r['filler_len']:3d}  expected={r['expected']:3d}  predicted={r['predicted']}  raw={r['generated_text']!r}")
        print(f"     Q1: {r['fact1']}")
        print(f"     Q2: {r['fact2']}")
        print(f"     a1={r['a1']}  a2={r['a2']}")
        print()

    total = len(rows)
    correct = sum(1 for r in rows if r["correct"])
    print(f"--- Pool: {correct}/{total} correct ({correct/total*100:.1f}%) | Showing {len(sample)} samples ---")

if __name__ == "__main__":
    main()