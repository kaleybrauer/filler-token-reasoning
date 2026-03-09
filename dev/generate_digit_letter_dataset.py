#!/usr/bin/env python3
"""
Generate addition dataset with digit-letter filler.

Tests whether digit tokens specifically drive the counting filler uplift, vs.
the positional structure of counting.

Construction: Generate "1 2 3 ... N" as normal, tokenize it, then replace each
token whose decoded text is purely numeric with the corresponding letter token
(digit chars substituted 0→a, 1→b, ..., 9→j). Decode back to a string. The
result has the same number of tokens and the same positional structure as
counting filler, but uses letter tokens instead of digit tokens.

Any token whose letter-substituted form does not encode back to a single token
is left unchanged and reported as a diagnostic warning.

Eval filler lengths: 0, 64, 128, 256  (matching 2hop_add counting for direct comparison)

Usage:
    python dev/generate_digit_letter_dataset.py \\
        --tokenizer Qwen/Qwen2.5-7B-Instruct \\
        --known-facts results/fact_knowledge_base/known_facts.json

Known facts default: results/fact_knowledge_base/known_facts.json
"""
import argparse
import json
import pathlib
import random
import sys
from typing import Dict

_SCRIPTS_DIR = pathlib.Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))
import generate_addition_dataset as gen  # noqa: E402

# ─────────────────────────────────────────────────────────────────────────────
# Defaults
# ─────────────────────────────────────────────────────────────────────────────

_REPO_ROOT = pathlib.Path(__file__).parent.parent
_DEFAULT_KNOWN_FACTS = str(_REPO_ROOT / "results" / "fact_knowledge_base" / "known_facts.json")
_DEFAULT_OUTDIR = str(_REPO_ROOT / "data" / "datasets" / "addition_digit_letter_filler")
_DEFAULT_EVAL_LENGTHS = "0,64,128,256"  # matches 2hop_add counting for direct comparison
_FILLER_TYPE = "digit_letter"
# In Qwen, every number is a sequence of individual digit tokens (0-9).
# The stable token-level substitution is: <digit> → <" letter"> (space-prefixed letter),
# i.e. the token for "0" maps to the token for " a", "1" → " b", ..., "9" → " j".
_DIGITS = "0123456789"
_LETTERS = "abcdefghij"  # space-prefixed when looked up below


def _build_substitution_map(tok) -> Dict[int, int]:
    """Build the 10-entry digit→letter token-ID map for Qwen.

    Maps each bare digit token (the token for "0", "1", ..., "9") to the
    corresponding space-prefixed letter token (" a", " b", ..., " j").
    """
    subst_map: Dict[int, int] = {}
    for digit, letter in zip(_DIGITS, _LETTERS):
        digit_ids = tok.encode(digit, add_special_tokens=False)
        letter_ids = tok.encode(" " + letter, add_special_tokens=False)
        if len(digit_ids) != 1:
            raise RuntimeError(f"Digit {digit!r} tokenizes to {len(digit_ids)} tokens; expected 1")
        if len(letter_ids) != 1:
            raise RuntimeError(f"Letter {' '+letter!r} tokenizes to {len(letter_ids)} tokens; expected 1")
        subst_map[digit_ids[0]] = letter_ids[0]
    return subst_map


def _make_digit_letter_builder(tok, subst_map: Dict[int, int]):
    """Return a filler builder function that applies the digit→letter substitution."""
    def _build_digit_letter_filler(filler_len: int) -> str:
        counting_text = " ".join(str(i) for i in range(1, filler_len + 1))
        token_ids = tok.encode(counting_text, add_special_tokens=False)
        substituted_ids = [subst_map.get(tid, tid) for tid in token_ids]
        return tok.decode(substituted_ids)
    return _build_digit_letter_filler


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate addition dataset with digit-letter filler.",
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
    ap.add_argument("--filler-mode", type=str, default="eval", choices=["uniform", "eval"])
    ap.add_argument("--filler-min", type=int, default=0)
    ap.add_argument("--filler-max", type=int, default=256)

    ap.add_argument("--fact-split-train", type=float, default=0.75)
    ap.add_argument("--fact-split-val", type=float, default=0.125)

    ap.add_argument("--n-fewshot", type=int, default=5)
    ap.add_argument("--n-fewshot-facts", type=int, default=10)

    ap.add_argument("--cot-mixture", action="store_true", default=True,
                    help="Enable CoT + digit-letter-filler mixture (default: on)")
    ap.add_argument("--no-cot-mixture", action="store_false", dest="cot_mixture",
                    help="Disable CoT mixture (filler-only training)")
    ap.add_argument("--cot-fraction", type=float, default=0.5)

    args = ap.parse_args()

    if args.cot_mixture and not (0.0 < args.cot_fraction <= 0.5):
        raise SystemExit(
            f"--cot-fraction must be in (0, 0.5], got {args.cot_fraction}."
        )

    eval_filler_lengths = [int(x.strip()) for x in args.eval_filler_lengths.split(",")]

    # ── Load tokenizer first (needed to build the substitution map) ──────────
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    print(f"BOS token: {tok.bos_token!r} (id={tok.bos_token_id})")
    print(f"EOS token: {tok.eos_token!r} (id={tok.eos_token_id})")

    # ── Build digit → letter token substitution map ──────────────────────────
    print("\nBuilding digit→letter token substitution map...")
    subst_map = _build_substitution_map(tok)
    print(f"  Substitution map (digit token → space-letter token):")
    for digit, letter in zip(_DIGITS, _LETTERS):
        d_id = tok.encode(digit, add_special_tokens=False)[0]
        l_id = tok.encode(" " + letter, add_special_tokens=False)[0]
        print(f"    {digit!r} (id={d_id}) → {' '+letter!r} (id={l_id})")

    # ── Monkey-patch gen._build_filler to handle "digit_letter" type ─────────
    _digit_letter_fn = _make_digit_letter_builder(tok, subst_map)
    _original_build_filler = gen._build_filler

    def _patched_build_filler(filler_len: int, filler_type: str = "dots") -> str:
        if filler_type == _FILLER_TYPE:
            return _digit_letter_fn(filler_len)
        return _original_build_filler(filler_len, filler_type)

    gen._build_filler = _patched_build_filler

    # ── Verify token count matches counting at all eval filler lengths ────────
    print(f"\nToken count verification (digit_letter vs counting):")
    token_counts_match = True
    for n in [fl for fl in eval_filler_lengths if fl > 0]:
        counting_text = " ".join(str(i) for i in range(1, n + 1))
        dl_text = _digit_letter_fn(n)
        n_counting = len(tok.encode(counting_text, add_special_tokens=False))
        n_dl = len(tok.encode(dl_text, add_special_tokens=False))
        match = "✓" if n_counting == n_dl else "✗ MISMATCH"
        print(f"  N={n:4d}: counting={n_counting} tokens, digit_letter={n_dl} tokens  {match}")
        if n_counting != n_dl:
            token_counts_match = False
    if not token_counts_match:
        print("  WARNING: Token counts differ for some filler lengths due to unmapped tokens.")

    # ── Print sample filler ───────────────────────────────────────────────────
    print(f"\nFiller type: {_FILLER_TYPE}")
    print(f"  Counting  N=8:  {' '.join(str(i) for i in range(1, 9))!r}")
    print(f"  Digit-ltr N=8:  {_digit_letter_fn(8)!r}")
    print(f"  Counting  N=16: {' '.join(str(i) for i in range(1, 17))!r}")
    print(f"  Digit-ltr N=16: {_digit_letter_fn(16)!r}")
    print(f"Eval filler lengths: {eval_filler_lengths}")
    print(f"CoT mixture: {'ENABLED' if args.cot_mixture else 'disabled'}")

    # ── Load known facts ──────────────────────────────────────────────────────
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

    rng = random.Random(args.seed)
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

    # Val eval
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
        "filler_description": (
            "Counting sequence 1..N tokenized, then each digit token replaced by its "
            "letter-substituted equivalent (0→a, 1→b, ..., 9→j) at the token ID level, "
            "then decoded. Same token count and positional structure as counting filler."
        ),
        "digit_to_letter": "0→a, 1→b, 2→c, 3→d, 4→e, 5→f, 6→g, 7→h, 8→i, 9→j",
        "substitution_map_size": len(subst_map),
        "token_counts_match": token_counts_match,
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
    if not token_counts_match:
        print("\n  WARNING: Token counts did not perfectly match counting for all filler lengths.")


if __name__ == "__main__":
    main()
