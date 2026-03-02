#!/usr/bin/env python3
"""
Generate additional test data from unused pairs within an existing fact pool.

Not all valid (fact1, fact2) pairs in a split's fact pool are necessarily
used when generating the original dataset. This script identifies which pairs
have already been generated (by scanning existing JSONL files), then generates
new test examples exclusively from the remaining unused pairs. 
Facts never leak across pools.

Usage:
    # Generate all available unused pairs (default)
    python dev/generate_extra_test.py \
        --data-dir data/datasets/2hop_add \
        --outfile extra_test.jsonl

    # Limit to a specific number of pairs
    python dev/generate_extra_test.py \
        --data-dir data/datasets/2hop_add \
        --n-test 200 \
        --outfile extra_test.jsonl
"""
import argparse
import json
import pathlib
import random
import sys
from typing import FrozenSet, List, Set, Tuple

# Import shared utilities from generate_2hop_dataset (lives in scripts/)
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "scripts"))
from generate_2hop_dataset import DatasetBuilder, write_jsonl

from transformers import AutoTokenizer


def collect_used_pairs(data_dir: pathlib.Path) -> Set[FrozenSet[str]]:
    """Scan all existing JSONL splits and return the set of used (fact1, fact2) pairs.

    Pairs are stored as frozensets so {A, B} == {B, A}.
    """
    used: Set[FrozenSet[str]] = set()
    for fname in ("train.jsonl", "val.jsonl", "test.jsonl"):
        p = data_dir / fname
        if not p.exists():
            continue
        with p.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                used.add(frozenset([row["fact1"], row["fact2"]]))
    return used


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate extra test data from unused pairs in an existing fact pool."
    )
    ap.add_argument(
        "--data-dir", type=str, required=True,
        help="Existing dataset directory (must contain manifest.json, fact_pools.json, "
             "and the existing *.jsonl splits)",
    )
    ap.add_argument(
        "--n-test", type=int, default=None,
        help="Number of unique unused pairs to use (default: all available unused pairs)",
    )
    ap.add_argument(
        "--pools", type=str, default="test,val",
        help="Comma-separated fact pools to draw unused pairs from: test, val, train "
             "(default: test,val). Avoid 'train' for clean held-out evaluation.",
    )
    ap.add_argument(
        "--eval-filler-lengths", type=str, default=None,
        help="Override eval filler lengths as comma-separated ints (default: from manifest)",
    )
    ap.add_argument(
        "--outfile", type=str, default="extra_test.jsonl",
        help="Output filename (default: extra_test.jsonl, written into --data-dir)",
    )
    ap.add_argument(
        "--outdir", type=str, default=None,
        help="Output directory (default: same as --data-dir)",
    )
    ap.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    args = ap.parse_args()

    data_dir = pathlib.Path(args.data_dir)
    outdir = pathlib.Path(args.outdir) if args.outdir else data_dir
    outdir.mkdir(parents=True, exist_ok=True)

    pools = [p.strip() for p in args.pools.split(",")]
    valid_pool_names = {"train", "val", "test"}
    for p in pools:
        if p not in valid_pool_names:
            raise SystemExit(f"Unknown pool '{p}'. Must be one of: {valid_pool_names}")

    # ── Load manifest ─────────────────────────────────────────────────────────
    manifest_path = data_dir / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"manifest.json not found in {data_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    if "fewshot_examples" not in manifest:
        raise SystemExit(
            "manifest.json does not contain 'fewshot_examples'. This dataset was "
            "generated with an older script version and cannot be used here."
        )

    # ── Load fact pools ───────────────────────────────────────────────────────
    fact_pools_path = data_dir / "fact_pools.json"
    if not fact_pools_path.exists():
        raise SystemExit(f"fact_pools.json not found in {data_dir}")
    fact_pools = json.loads(fact_pools_path.read_text(encoding="utf-8"))

    # Combine facts from the requested pools
    combined_facts: List[Tuple[str, int, str]] = []
    for pool_name in pools:
        pool_facts = fact_pools.get(pool_name, [])
        combined_facts.extend(
            (f["question"], f["answer"], f["type"]) for f in pool_facts
        )
        print(f"Pool '{pool_name}': {len(pool_facts)} facts")

    max_answer = manifest.get("max_answer", 1000)
    combined_facts = [(q, a, k) for (q, a, k) in combined_facts if 0 <= a < max_answer]

    # ── Find already-used pairs ───────────────────────────────────────────────
    used_pairs = collect_used_pairs(data_dir)
    print(f"Pairs already generated across all splits: {len(used_pairs)}")

    # ── Reconstruct fewshot examples from manifest ────────────────────────────
    fewshot_raw = manifest["fewshot_examples"]
    fewshot_examples: List[Tuple[str, str, int, int, int]] = [
        (ex["q1"], ex["q2"], ex["a1"], ex["a2"], ex["sum"])
        for ex in fewshot_raw
    ]

    # ── Resolve settings ──────────────────────────────────────────────────────
    if args.eval_filler_lengths:
        eval_filler_lengths = [int(x.strip()) for x in args.eval_filler_lengths.split(",")]
    else:
        eval_filler_lengths = manifest["eval_filler_lengths"]

    filler_type = manifest.get("filler_type", "dots")
    tokenizer_name = manifest["tokenizer"]

    print(f"Tokenizer:      {tokenizer_name}")
    print(f"Filler type:    {filler_type}")
    print(f"Filler lengths: {eval_filler_lengths}")

    tok = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

    # ── Build DatasetBuilder, then filter to unused pairs ─────────────────────
    rng = random.Random(args.seed)

    builder = DatasetBuilder(
        facts=combined_facts,
        tok=tok,
        fewshot_examples=fewshot_examples,
        max_answer=max_answer,
        filler_mode="eval",
        filler_min=manifest.get("filler_min", 0),
        filler_max=manifest.get("filler_max", 600),
        eval_filler_lengths=eval_filler_lengths,
        rng=rng,
        filler_type=filler_type,
    )

    # Filter builder.valid_pairs to only pairs not already generated
    all_valid = builder.valid_pairs
    unused_pairs = [
        (i, j) for (i, j) in all_valid
        if frozenset([combined_facts[i][0], combined_facts[j][0]]) not in used_pairs
    ]
    builder.valid_pairs = unused_pairs

    print(f"Valid pairs in pool:  {len(all_valid)}")
    print(f"Already used:         {len(all_valid) - len(unused_pairs)}")
    print(f"Unused pairs available: {len(unused_pairs)}")

    n_test = args.n_test if args.n_test is not None else len(unused_pairs)
    if n_test > len(unused_pairs):
        raise SystemExit(
            f"Requested {n_test} pairs but only {len(unused_pairs)} unused pairs "
            f"available in pool(s) {pools}. Reduce --n-test or add more pools with --pools."
        )

    print(f"\nGenerating {n_test} pairs × {len(eval_filler_lengths)} filler lengths "
          f"= {n_test * len(eval_filler_lengths)} examples...")

    rows = builder.build_test_split(
        n_pairs=n_test,
        filler_lengths=eval_filler_lengths,
        split="extra_test",
    )

    # ── Write output ──────────────────────────────────────────────────────────
    outpath = outdir / args.outfile
    write_jsonl(rows, outpath)
    print(f"\nWrote {len(rows)} examples → {outpath}")

    stem = pathlib.Path(args.outfile).stem
    meta = {
        "source_dataset": str(data_dir),
        "pools_used": pools,
        "outfile": args.outfile,
        "tokenizer": tokenizer_name,
        "filler_type": filler_type,
        "fewshot_examples": fewshot_raw,
        "eval_filler_lengths": eval_filler_lengths,
        "seed": args.seed,
        "max_answer": max_answer,
        "n_pairs": n_test,
        "total_examples": len(rows),
        "pair_counts": {
            "valid_in_pool": len(all_valid),
            "already_used": len(all_valid) - len(unused_pairs),
            "unused_available": len(unused_pairs),
            "pairs_generated": builder.used_pair_count,
        },
    }
    meta_path = outdir / f"{stem}_manifest.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote manifest  → {meta_path}")


if __name__ == "__main__":
    main()
