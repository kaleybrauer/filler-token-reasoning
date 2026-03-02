#!/usr/bin/env python3
"""
Generate additional test data using only facts not present in any existing split.

Reads an existing dataset's manifest.json and fact_pools.json to determine
which facts have already been used (train + val + test + fewshot), then
generates new test examples from the remaining unused facts in known_facts.json.

The same fewshot examples and prompt format are taken from the existing
manifest, so the new test data is compatible with the original dataset.

Usage:
    python dev/generate_extra_test.py \
        --data-dir data/datasets/2hop_add \
        --known-facts data/known_facts/known_facts.json \
        --n-test 200 \
        --outfile extra_test.jsonl
"""
import argparse
import json
import pathlib
import random
import sys
from typing import List, Set, Tuple

# Import shared utilities from generate_2hop_dataset (lives in scripts/)
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "scripts"))
from generate_2hop_dataset import DatasetBuilder, load_known_facts, write_jsonl

from transformers import AutoTokenizer


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate extra test data from facts not used in any existing split."
    )
    ap.add_argument(
        "--data-dir", type=str, required=True,
        help="Existing dataset directory (must contain manifest.json and fact_pools.json)",
    )
    ap.add_argument(
        "--known-facts", type=str, required=True,
        help="Path to known_facts.json (source of all candidate facts)",
    )
    ap.add_argument(
        "--n-test", type=int, default=200,
        help="Number of unique test pairs to generate (each pair × every filler length = total examples)",
    )
    ap.add_argument(
        "--eval-filler-lengths", type=str, default=None,
        help="Override eval filler lengths as comma-separated ints (default: taken from manifest)",
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
        help="Random seed for shuffling unused facts (default: 42)",
    )
    args = ap.parse_args()

    data_dir = pathlib.Path(args.data_dir)
    outdir = pathlib.Path(args.outdir) if args.outdir else data_dir
    outdir.mkdir(parents=True, exist_ok=True)

    # ── Load existing manifest ────────────────────────────────────────────────
    manifest_path = data_dir / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"manifest.json not found in {data_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    if "fewshot_examples" not in manifest:
        raise SystemExit(
            "manifest.json does not contain 'fewshot_examples'. This dataset was "
            "generated with an older script version. Cannot guarantee fact isolation "
            "without knowing which facts were reserved for few-shot."
        )

    # ── Collect all already-used fact questions ───────────────────────────────
    fact_pools_path = data_dir / "fact_pools.json"
    if not fact_pools_path.exists():
        raise SystemExit(f"fact_pools.json not found in {data_dir}")
    fact_pools = json.loads(fact_pools_path.read_text(encoding="utf-8"))

    used_questions: Set[str] = set()
    split_counts = {}
    for split in ("train", "val", "test"):
        facts_in_split = fact_pools.get(split, [])
        split_counts[split] = len(facts_in_split)
        for fact in facts_in_split:
            used_questions.add(fact["question"])

    # Exclude fewshot facts (each example uses q1 + q2)
    fewshot_raw = manifest["fewshot_examples"]
    n_fewshot_facts_recorded = 0
    for ex in fewshot_raw:
        used_questions.add(ex["q1"])
        used_questions.add(ex["q2"])
        n_fewshot_facts_recorded += 2

    # Warn if the manifest reserved more facts than are recorded in fewshot_examples
    n_fewshot_facts_reserved = manifest.get("n_fewshot_facts", n_fewshot_facts_recorded)
    if n_fewshot_facts_reserved > n_fewshot_facts_recorded:
        print(
            f"WARNING: manifest.n_fewshot_facts={n_fewshot_facts_reserved} but only "
            f"{n_fewshot_facts_recorded} fact questions appear in fewshot_examples. "
            f"Up to {n_fewshot_facts_reserved - n_fewshot_facts_recorded} reserved fewshot "
            f"facts are unaccounted for and may slip into the extra test set."
        )

    print(
        f"Facts already used: {len(used_questions)} total "
        f"({split_counts.get('train',0)} train + {split_counts.get('val',0)} val + "
        f"{split_counts.get('test',0)} test + {n_fewshot_facts_recorded} fewshot)"
    )

    # ── Load all known facts and filter to unused ─────────────────────────────
    known_path = pathlib.Path(args.known_facts)
    if not known_path.exists():
        raise SystemExit(f"--known-facts file not found: {known_path}")
    max_answer = manifest.get("max_answer", 1000)
    all_facts = load_known_facts(known_path)
    all_facts = [(q, a, k) for (q, a, k) in all_facts if 0 <= a < max_answer]

    unused_facts = [(q, a, k) for (q, a, k) in all_facts if q not in used_questions]

    print(f"Known facts total:     {len(all_facts)}")
    print(f"Unused facts available: {len(unused_facts)}")

    if len(unused_facts) < 4:
        raise SystemExit(
            f"Only {len(unused_facts)} unused facts — too few to form valid test pairs. "
            "Consider running evaluate_individual_facts.py on additional facts."
        )

    # ── Reconstruct fewshot examples from manifest ────────────────────────────
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
    print(f"Total examples: {args.n_test} pairs × {len(eval_filler_lengths)} lengths "
          f"= {args.n_test * len(eval_filler_lengths)}")

    tok = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

    # ── Shuffle unused facts and build test split ─────────────────────────────
    rng = random.Random(args.seed)
    rng.shuffle(unused_facts)

    builder = DatasetBuilder(
        facts=unused_facts,
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

    if args.n_test > builder.max_possible_pairs():
        raise SystemExit(
            f"Requested {args.n_test} test pairs but only "
            f"{builder.max_possible_pairs()} valid unique pairs available from "
            f"{len(unused_facts)} unused facts. Reduce --n-test."
        )

    print(f"\nGenerating examples...")
    rows = builder.build_test_split(
        n_pairs=args.n_test,
        filler_lengths=eval_filler_lengths,
        split="extra_test",
    )

    # ── Write output ──────────────────────────────────────────────────────────
    outpath = outdir / args.outfile
    write_jsonl(rows, outpath)
    print(f"\nWrote {len(rows)} examples → {outpath}")

    # Sidecar manifest for the extra test file
    stem = pathlib.Path(args.outfile).stem
    meta = {
        "source_dataset": str(data_dir),
        "known_facts_source": str(known_path),
        "outfile": args.outfile,
        "tokenizer": tokenizer_name,
        "filler_type": filler_type,
        "fewshot_examples": fewshot_raw,
        "eval_filler_lengths": eval_filler_lengths,
        "seed": args.seed,
        "max_answer": max_answer,
        "n_pairs": args.n_test,
        "total_examples": len(rows),
        "fact_counts": {
            "known_total": len(all_facts),
            "used_in_original": len(used_questions),
            "unused_available": len(unused_facts),
            "valid_pairs": builder.max_possible_pairs(),
            "pairs_used": builder.used_pair_count,
        },
    }
    meta_path = outdir / f"{stem}_manifest.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote manifest  → {meta_path}")


if __name__ == "__main__":
    main()
