#!/usr/bin/env python3
"""
Generate addition dataset with mixed-case alphabet filler (a-zA-Z cycling).

Tests whether filler token diversity explains the counting > dots performance gap.
Alphabet filler cycles through a-z A-Z (52 letters), producing 53 unique token
types in Qwen's tokenizer when space-separated (52 letter tokens + space token),
vs. 2 unique tokens for dots and ~11 for counting up to 300.

Eval filler lengths: 0, 250, 500, 1000

Usage:
    python dev/generate_alphabet_filler_dataset.py \\
        --tokenizer Qwen/Qwen2.5-7B-Instruct \\
        --outdir data/datasets/addition_alphabet_filler

Known facts default: results/fact_knowledge_base/known_facts.json
"""
import argparse
import json
import pathlib
import random
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

_SCRIPTS_DIR = pathlib.Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))
import generate_addition_dataset as gen  # noqa: E402

# ─────────────────────────────────────────────────────────────────────────────
# Default values for this experiment
# ─────────────────────────────────────────────────────────────────────────────

_REPO_ROOT = pathlib.Path(__file__).parent.parent
_DEFAULT_KNOWN_FACTS = str(_REPO_ROOT / "results" / "fact_knowledge_base" / "known_facts.json")
_DEFAULT_OUTDIR = str(_REPO_ROOT / "data" / "datasets" / "addition_alphabet_filler")
_DEFAULT_EVAL_LENGTHS = "0,250,500,1000"
_FILLER_TYPE = "alphabet"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate addition dataset with alphabet filler (a-zA-Z cycling).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--tokenizer", type=str, required=True,
                    help="Tokenizer name, e.g. Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--known-facts", type=str, default=_DEFAULT_KNOWN_FACTS,
                    help="Path to known_facts.json")
    ap.add_argument("--outdir", type=str, default=_DEFAULT_OUTDIR,
                    help="Output directory for generated JSONL files")

    ap.add_argument("--n-train", type=int, default=20000,
                    help="Training examples. At default cot-fraction=0.5 this yields "
                         "10k CoT + 10k filler examples.")
    ap.add_argument("--n-val", type=int, default=600)
    ap.add_argument("--n-val-eval", type=int, default=600,
                    help="Unique pairs for val_eval.jsonl (accuracy eval, each × all filler lengths)")
    ap.add_argument("--n-test", type=int, default=200,
                    help="Unique test pairs (each repeated × all filler lengths)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-answer", type=int, default=1000)

    ap.add_argument("--eval-filler-lengths", type=str, default=_DEFAULT_EVAL_LENGTHS,
                    help="Comma-separated filler lengths for eval and test set generation")
    ap.add_argument("--filler-mode", type=str, default="eval", choices=["uniform", "eval"],
                    help="How to sample filler lengths during training")
    ap.add_argument("--filler-min", type=int, default=0, help="Min filler length (uniform mode)")
    ap.add_argument("--filler-max", type=int, default=1000, help="Max filler length (uniform mode)")

    ap.add_argument("--fact-split-train", type=float, default=0.75)
    ap.add_argument("--fact-split-val", type=float, default=0.125)

    ap.add_argument("--n-fewshot", type=int, default=5)
    ap.add_argument("--n-fewshot-facts", type=int, default=10)

    ap.add_argument("--cot-mixture", action="store_true", default=True,
                    help="Enable CoT + alphabet-filler mixture (default: on)")
    ap.add_argument("--no-cot-mixture", action="store_false", dest="cot_mixture",
                    help="Disable CoT mixture (filler-only training)")
    ap.add_argument("--cot-fraction", type=float, default=0.5)

    args = ap.parse_args()

    if args.cot_mixture and not (0.0 < args.cot_fraction <= 0.5):
        raise SystemExit(
            f"--cot-fraction must be in (0, 0.5], got {args.cot_fraction}."
        )

    rng = random.Random(args.seed)
    eval_filler_lengths = [int(x.strip()) for x in args.eval_filler_lengths.split(",")]

    print(f"Filler type: {_FILLER_TYPE} (a-z A-Z cycling, 53 unique token types)")
    print(f"Eval filler lengths: {eval_filler_lengths}")
    print(f"CoT mixture: {'ENABLED' if args.cot_mixture else 'disabled'}")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    print(f"BOS token: {tok.bos_token!r} (id={tok.bos_token_id})")
    print(f"EOS token: {tok.eos_token!r} (id={tok.eos_token_id})")

    # Verify the tokenizer sees the expected number of unique tokens
    sample_filler = _build_alphabet_filler(52)
    sample_ids = tok.encode(sample_filler, add_special_tokens=False)
    unique_ids = len(set(sample_ids))
    print(f"\nAlphabet filler uniqueness check (one full cycle, 52 letters):")
    print(f"  Text:        {sample_filler[:40]}...")
    print(f"  Token IDs:   {sample_ids[:10]}...")
    print(f"  Unique token IDs: {unique_ids}")

    # Load known facts
    known_path = pathlib.Path(args.known_facts)
    if not known_path.exists():
        raise SystemExit(f"Error: --known-facts file not found: {known_path}")

    all_facts = gen.load_known_facts(known_path)
    all_facts = [(q, a, k) for (q, a, k) in all_facts if 0 <= a < args.max_answer]

    from collections import Counter
    kind_counts = Counter(k for _, _, k in all_facts)
    print(f"\nLoaded {len(all_facts)} known facts from {known_path}")
    for kind in sorted(kind_counts):
        print(f"  {kind}: {kind_counts[kind]}")

    if len(all_facts) < 50:
        raise SystemExit(f"Too few usable facts ({len(all_facts)}). Need at least 50.")

    rng.shuffle(all_facts)

    print(f"\nSelecting {args.n_fewshot} few-shot examples from "
          f"{args.n_fewshot_facts} reserved facts...")
    fewshot_examples, remaining_facts = gen.select_fewshot_examples(
        all_facts,
        n_examples=args.n_fewshot,
        max_answer=args.max_answer,
        rng=rng,
        n_fewshot_facts=args.n_fewshot_facts,
    )
    print(f"  Reserved {args.n_fewshot_facts} facts for few-shot; "
          f"{len(remaining_facts)} facts available for train/val/test")
    print("\nFew-shot examples:")
    for idx, (q1, q2, a1, a2, s) in enumerate(fewshot_examples):
        print(f"  [{idx+1}] {gen.compose_question(q1, q2)} -> {s} ({a1}+{a2})")

    train_facts, val_facts, test_facts = gen.split_facts(
        remaining_facts, rng,
        train_frac=args.fact_split_train,
        val_frac=args.fact_split_val,
    )

    print(f"\nFact pool sizes:")
    print(f"  Train: {len(train_facts)} facts")
    print(f"  Val:   {len(val_facts)} facts")
    print(f"  Test:  {len(test_facts)} facts")

    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    def make_builder(facts, filler_mode_override=None):
        return gen.DatasetBuilder(
            facts=facts,
            tok=tok,
            fewshot_examples=fewshot_examples,
            max_answer=args.max_answer,
            filler_mode=filler_mode_override or args.filler_mode,
            filler_min=args.filler_min,
            filler_max=args.filler_max,
            eval_filler_lengths=eval_filler_lengths,
            rng=rng,
            filler_type=_FILLER_TYPE,
        )

    # Train
    print("\nGenerating training data...")
    train_builder = make_builder(train_facts)
    if args.cot_mixture:
        n_cot = int(args.n_train * args.cot_fraction)
        n_filler_only = args.n_train - (n_cot * 2)
        if n_filler_only < 0:
            n_cot = args.n_train // 2
            n_filler_only = 0
        needed = n_cot + n_filler_only
        if needed > train_builder.max_possible_pairs():
            raise SystemExit(
                f"CoT mixture needs {needed} pairs but only "
                f"{train_builder.max_possible_pairs()} available."
            )
    else:
        if args.n_train > train_builder.max_possible_pairs():
            raise SystemExit(
                f"Cannot generate {args.n_train} train examples: only "
                f"{train_builder.max_possible_pairs()} valid pairs available."
            )
    train_rows = train_builder.build_split(
        args.n_train, "train",
        cot_mixture=args.cot_mixture,
        cot_fraction=args.cot_fraction,
    )

    # Val
    print("\nGenerating validation data...")
    val_builder = make_builder(val_facts)
    val_rows = val_builder.build_split(
        args.n_val, "val",
        cot_mixture=args.cot_mixture,
        cot_fraction=args.cot_fraction,
    )

    # Val eval (accuracy eval split)
    print("\nGenerating val_eval data...")
    print(f"  {args.n_val_eval} pairs × {len(eval_filler_lengths)} filler lengths "
          f"= {args.n_val_eval * len(eval_filler_lengths)} examples")
    val_eval_builder = make_builder(val_facts, filler_mode_override="eval")
    if args.n_val_eval > val_eval_builder.max_possible_pairs():
        raise SystemExit(
            f"Cannot generate {args.n_val_eval} val_eval pairs: only "
            f"{val_eval_builder.max_possible_pairs()} available."
        )
    val_eval_rows = val_eval_builder.build_test_split(
        n_pairs=args.n_val_eval,
        filler_lengths=eval_filler_lengths,
        split="val_eval",
    )

    # Test
    print("\nGenerating test data...")
    print(f"  {args.n_test} pairs × {len(eval_filler_lengths)} filler lengths "
          f"= {args.n_test * len(eval_filler_lengths)} examples")
    test_builder = make_builder(test_facts, filler_mode_override="eval")
    if args.n_test > test_builder.max_possible_pairs():
        raise SystemExit(
            f"Cannot generate {args.n_test} test pairs: only "
            f"{test_builder.max_possible_pairs()} available."
        )
    test_rows = test_builder.build_test_split(
        n_pairs=args.n_test,
        filler_lengths=eval_filler_lengths,
    )

    # Write
    gen.write_jsonl(train_rows, outdir / "train.jsonl")
    gen.write_jsonl(val_rows, outdir / "val.jsonl")
    gen.write_jsonl(val_eval_rows, outdir / "val_eval.jsonl")
    gen.write_jsonl(test_rows, outdir / "test.jsonl")

    fact_pools = {
        "train": [{"question": q, "answer": a, "type": t} for q, a, t in train_facts],
        "val":   [{"question": q, "answer": a, "type": t} for q, a, t in val_facts],
        "test":  [{"question": q, "answer": a, "type": t} for q, a, t in test_facts],
    }
    (outdir / "fact_pools.json").write_text(json.dumps(fact_pools, indent=2), encoding="utf-8")

    manifest = {
        "tokenizer": args.tokenizer,
        "known_facts_source": str(args.known_facts),
        "prompt_format": "system_prompt_with_examples",
        "filler_type": _FILLER_TYPE,
        "filler_description": "Mixed-case a-zA-Z cycling (52 unique letters, 53 unique token types)",
        "unique_token_types_in_filler": unique_ids,
        "cot_mixture": args.cot_mixture,
        "cot_fraction": args.cot_fraction if args.cot_mixture else None,
        "seed": args.seed,
        "max_answer": args.max_answer,
        "fewshot_examples": [
            {"q1": q1, "q2": q2, "a1": a1, "a2": a2, "sum": s}
            for (q1, q2, a1, a2, s) in fewshot_examples
        ],
        "n_fewshot": len(fewshot_examples),
        "n_fewshot_facts": args.n_fewshot_facts,
        "filler_mode": args.filler_mode,
        "filler_min": args.filler_min,
        "filler_max": args.filler_max,
        "eval_filler_lengths": eval_filler_lengths,
        "fact_split": {
            "train_frac": args.fact_split_train,
            "val_frac": args.fact_split_val,
            "test_frac": 1.0 - args.fact_split_train - args.fact_split_val,
        },
        "fact_counts": {
            "train": len(train_facts),
            "val": len(val_facts),
            "test": len(test_facts),
        },
        "example_counts": {
            "train": len(train_rows),
            "val": len(val_rows),
            "val_eval": len(val_eval_rows),
            "test": len(test_rows),
        },
        "test_design": {
            "n_pairs": args.n_test,
            "filler_lengths": eval_filler_lengths,
            "total_examples": len(test_rows),
        },
        "val_eval_design": {
            "n_pairs": args.n_val_eval,
            "filler_lengths": eval_filler_lengths,
            "total_examples": len(val_eval_rows),
        },
        "valid_pairs": {
            "train": train_builder.max_possible_pairs(),
            "val": val_builder.max_possible_pairs(),
            "val_eval": val_eval_builder.max_possible_pairs(),
            "test": test_builder.max_possible_pairs(),
        },
        "unique_pairs_used": {
            "train": train_builder.used_pair_count,
            "val": val_builder.used_pair_count,
            "val_eval": val_eval_builder.used_pair_count,
            "test": test_builder.used_pair_count,
        },
        "bos_token_id": tok.bos_token_id,
        "eos_token_id": tok.eos_token_id,
    }
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\n{'='*60}")
    print(f"Done. Wrote dataset to: {outdir}")
    print(f"{'='*60}")
    print(f"  train.jsonl:    {len(train_rows)} examples")
    print(f"  val.jsonl:      {len(val_rows)} examples")
    print(f"  val_eval.jsonl: {len(val_eval_rows)} examples "
          f"({args.n_val_eval} pairs × {len(eval_filler_lengths)} lengths)")
    print(f"  test.jsonl:     {len(test_rows)} examples "
          f"({args.n_test} pairs × {len(eval_filler_lengths)} lengths)")


if __name__ == "__main__":
    main()
