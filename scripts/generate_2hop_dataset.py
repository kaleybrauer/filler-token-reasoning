#!/usr/bin/env python3
"""
Generate 2-hop arithmetic dataset for opaque reasoner experiments.

- Facts are split into train/val/test pools first, then pairs are composed
  only from facts within each pool. This prevents any fact leakage.
- Pair deduplication within each split prevents duplicate questions.
- Filler lengths can be sampled uniformly or from a fixed set of eval values.
"""
import argparse
import json
import pathlib
import random
from typing import Any, Dict, List, Optional, Set, Tuple

from transformers import AutoTokenizer

INT_KEYS = ["answer", "value", "number", "n", "age", "atomic_number"]

# Default filler lengths used for evaluation (also used during training if --filler-mode=eval)
DEFAULT_EVAL_FILLER_LENGTHS = [0, 32, 128, 300, 600, 1000]


def _is_int(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _extract_int_from_obj(obj: Any) -> Optional[int]:
    if _is_int(obj):
        return obj
    if isinstance(obj, dict):
        for k in INT_KEYS:
            if k in obj and _is_int(obj[k]):
                return obj[k]
    return None


def load_facts(path: pathlib.Path, kind: str) -> List[Tuple[str, int, str]]:
    """
    Returns list of (question_text, answer_int, kind).
    Handles a few common JSON shapes:
      - dict[str, int]           (entity -> int OR question -> int)
      - dict[str, dict]          (entity -> {age: int} etc.)
      - list[dict]               ({question: str, answer: int} etc.)
    """
    raw = json.loads(path.read_text(encoding="utf-8"))

    out: List[Tuple[str, int, str]] = []

    def make_q(key_or_q: str) -> str:
        # If it already looks like a question, keep it.
        if "?" in key_or_q:
            return key_or_q.strip()
        if kind == "age":
            return f"At what age did {key_or_q} die?"
        if kind == "atomic":
            return f"What is the atomic number of {key_or_q}?"
        # static: treat key as question if not obviously an entity
        return key_or_q.strip()

    if isinstance(raw, dict):
        for k, v in raw.items():
            ans = _extract_int_from_obj(v)
            if ans is None:
                continue
            q = make_q(k)
            out.append((q, int(ans), kind))

    elif isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            ans = _extract_int_from_obj(item)
            if ans is None:
                continue

            q = None
            for qk in ["question", "q", "prompt", "text"]:
                if qk in item and isinstance(item[qk], str):
                    q = item[qk].strip()
                    break

            if q is None:
                # maybe it's entity-style
                for nk in ["name", "entity", "person", "element"]:
                    if nk in item and isinstance(item[nk], str):
                        q = make_q(item[nk])
                        break

            if q is None:
                continue

            out.append((q, int(ans), kind))

    else:
        raise ValueError(f"Unsupported JSON root type in {path}: {type(raw)}")

    return out


def split_facts(
    facts: List[Tuple[str, int, str]],
    rng: random.Random,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
) -> Tuple[List[Tuple[str, int, str]], List[Tuple[str, int, str]], List[Tuple[str, int, str]]]:
    """
    Split facts into train/val/test pools.
    
    Returns (train_facts, val_facts, test_facts).
    """
    facts = facts.copy()
    rng.shuffle(facts)
    
    n = len(facts)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    
    train_facts = facts[:n_train]
    val_facts = facts[n_train:n_train + n_val]
    test_facts = facts[n_train + n_val:]
    
    return train_facts, val_facts, test_facts


def sample_filler_len(
    rng: random.Random,
    mode: str,
    lo: int,
    hi: int,
    eval_lengths: List[int],
) -> int:
    """Sample filler length based on mode."""
    if mode == "uniform":
        if hi < lo:
            hi = lo
        return rng.randint(lo, hi)
    elif mode == "eval":
        # Sample from the discrete set of eval lengths
        return rng.choice(eval_lengths)
    else:
        raise ValueError(f"Unknown filler mode: {mode}")


def write_jsonl(rows: List[Dict[str, Any]], outpath: pathlib.Path) -> None:
    outpath.parent.mkdir(parents=True, exist_ok=True)
    with outpath.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


class DatasetBuilder:
    """Builds examples from a pool of facts with pair deduplication."""
    
    def __init__(
        self,
        facts: List[Tuple[str, int, str]],
        tok: AutoTokenizer,
        filler_id: int,
        filler_token: str,
        max_answer: int,
        filler_mode: str,
        filler_min: int,
        filler_max: int,
        eval_filler_lengths: List[int],
        rng: random.Random,
        max_tries: int = 2000,
    ):
        self.facts = facts
        self.tok = tok
        self.filler_id = filler_id
        self.filler_token = filler_token
        self.max_answer = max_answer
        self.filler_mode = filler_mode
        self.filler_min = filler_min
        self.filler_max = filler_max
        self.eval_filler_lengths = eval_filler_lengths
        self.rng = rng
        self.max_tries = max_tries
        
        # Track used pairs to prevent duplicates (order-invariant)
        self.used_pairs: Set[Tuple[str, str]] = set()
        
        # Get BOS/EOS tokens
        self.bos_ids = [tok.bos_token_id] if tok.bos_token_id else []
        self.eos_ids = [tok.eos_token_id] if tok.eos_token_id else []
    
    def _pair_key(self, q1: str, q2: str) -> Tuple[str, str]:
        """Create order-invariant key for a pair of questions."""
        return (min(q1, q2), max(q1, q2))
    
    def make_example(self, example_id: int, split: str) -> Dict[str, Any]:
        """Generate a single example, ensuring no duplicate pairs."""
        for _ in range(self.max_tries):
            (q1, a1, t1) = self.rng.choice(self.facts)
            (q2, a2, t2) = self.rng.choice(self.facts)
            
            # Skip if same question
            if q1 == q2:
                continue
            
            # Skip if pair already used
            pair_key = self._pair_key(q1, q2)
            if pair_key in self.used_pairs:
                continue
            
            # Check answer constraint
            s = a1 + a2
            if s < 0 or s >= self.max_answer:
                continue
            
            # Valid pair found
            self.used_pairs.add(pair_key)
            
            prompt = (
                "Answer with a single integer (no extra text).\n\n"
                f"{q1} + {q2}\n"
                "Answer:"
            )
            
            prompt_ids = self.tok.encode(prompt, add_special_tokens=False)
            
            # Sample filler length
            nfill = sample_filler_len(
                self.rng,
                self.filler_mode,
                self.filler_min,
                self.filler_max,
                self.eval_filler_lengths,
            )
            filler_seq = [self.filler_id] * nfill
            
            # Answer with leading space, terminated by EOS
            answer_text = " " + str(s)
            answer_ids = self.tok.encode(answer_text, add_special_tokens=False) + self.eos_ids
            
            # Assemble full sequence with BOS
            prefix_ids = self.bos_ids + prompt_ids + filler_seq
            input_ids = prefix_ids + answer_ids
            
            labels = [-100] * len(prefix_ids) + answer_ids
            attn = [1] * len(input_ids)
            
            return {
                "id": f"{split}-{example_id}",
                "split": split,
                "prompt": prompt,
                "fact1": q1,
                "fact2": q2,
                "a1": a1,
                "a2": a2,
                "answer": s,
                "type1": t1,
                "type2": t2,
                "filler_len": nfill,
                "filler_token": self.filler_token,
                "filler_token_id": self.filler_id,
                "prompt_ids": prompt_ids,
                "answer_ids": answer_ids,
                "input_ids": input_ids,
                "labels": labels,
                "attention_mask": attn,
            }
        
        raise RuntimeError(
            f"Failed to sample a valid example after {self.max_tries} tries. "
            f"Facts pool size: {len(self.facts)}, used pairs: {len(self.used_pairs)}. "
            "Try increasing --max-tries, adding more facts, or reducing dataset size."
        )
    
    def build_split(self, n: int, split: str) -> List[Dict[str, Any]]:
        """Build n examples for a split."""
        rows = []
        for i in range(n):
            rows.append(self.make_example(i, split))
            if (i + 1) % 10000 == 0:
                print(f"  Generated {i + 1}/{n} {split} examples...")
        return rows
    
    def max_possible_pairs(self) -> int:
        """Return maximum possible unique pairs from this fact pool."""
        n = len(self.facts)
        return n * (n - 1) // 2


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate 2-hop arithmetic dataset with fact-level train/val/test isolation."
    )
    ap.add_argument("--tokenizer", type=str, required=True, 
                    help="Tokenizer name, e.g. Qwen/Qwen2.5-7B")
    ap.add_argument("--sources", type=str, required=True, 
                    help="Directory containing age_facts.json, atomic_facts.json, static_facts.json")
    ap.add_argument("--outdir", type=str, required=True, 
                    help="Output directory for generated JSONL files")

    ap.add_argument("--n-train", type=int, default=50000)
    ap.add_argument("--n-val", type=int, default=2000)
    ap.add_argument("--n-test", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--max-answer", type=int, default=1000, 
                    help="Keep final answers < max-answer")

    ap.add_argument("--filler-token", type=str, default="<|fim_pad|>", 
                    help="Single-token filler string")
    ap.add_argument("--filler-mode", type=str, default="uniform", choices=["uniform", "eval"],
                    help="How to sample filler lengths: 'uniform' samples from [min,max], "
                         "'eval' samples from --eval-filler-lengths")
    ap.add_argument("--filler-min", type=int, default=0,
                    help="Min filler length (for uniform mode)")
    ap.add_argument("--filler-max", type=int, default=1000,
                    help="Max filler length (for uniform mode)")
    ap.add_argument("--eval-filler-lengths", type=str, default="0,32,128,300,600,1000",
                    help="Comma-separated filler lengths for eval mode and test set generation")

    ap.add_argument("--fact-split-train", type=float, default=0.8,
                    help="Fraction of facts for training pool")
    ap.add_argument("--fact-split-val", type=float, default=0.1,
                    help="Fraction of facts for validation pool")

    ap.add_argument("--max-tries", type=int, default=2000, 
                    help="Resample limit when enforcing constraints")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    
    # Parse eval filler lengths
    eval_filler_lengths = [int(x.strip()) for x in args.eval_filler_lengths.split(",")]
    print(f"Eval filler lengths: {eval_filler_lengths}")

    # Load tokenizer and validate filler token
    tok = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    filler_ids = tok.encode(args.filler_token, add_special_tokens=False)
    if len(filler_ids) != 1:
        raise SystemExit(
            f"filler-token {args.filler_token!r} does not map to exactly 1 token (got {filler_ids}). "
            "Pick another filler token."
        )
    filler_id = filler_ids[0]
    
    # Report BOS/EOS tokens
    print(f"BOS token: {tok.bos_token!r} (id={tok.bos_token_id})")
    print(f"EOS token: {tok.eos_token!r} (id={tok.eos_token_id})")

    # Load facts
    srcdir = pathlib.Path(args.sources)
    age_path = srcdir / "age_facts.json"
    atomic_path = srcdir / "atomic_facts.json"
    static_path = srcdir / "static_facts.json"

    age = load_facts(age_path, "age") if age_path.exists() else []
    atomic = load_facts(atomic_path, "atomic") if atomic_path.exists() else []
    static = load_facts(static_path, "static") if static_path.exists() else []

    all_facts = [(q, a, k) for (q, a, k) in (age + atomic + static) if 0 <= a < args.max_answer]
    
    print(f"\nLoaded facts: age={len(age)} atomic={len(atomic)} static={len(static)}")
    print(f"Usable facts (answer < {args.max_answer}): {len(all_facts)}")
    
    if len(all_facts) < 100:
        raise SystemExit(f"Too few usable facts ({len(all_facts)}). Check your source JSON files in {srcdir}.")

    # Split facts into train/val/test pools
    train_facts, val_facts, test_facts = split_facts(
        all_facts, rng, 
        train_frac=args.fact_split_train,
        val_frac=args.fact_split_val,
    )
    
    print(f"\nFact pool sizes:")
    print(f"  Train: {len(train_facts)} facts -> max {len(train_facts) * (len(train_facts)-1) // 2} pairs")
    print(f"  Val:   {len(val_facts)} facts -> max {len(val_facts) * (len(val_facts)-1) // 2} pairs")
    print(f"  Test:  {len(test_facts)} facts -> max {len(test_facts) * (len(test_facts)-1) // 2} pairs")
    
    # Check if we have enough pairs
    def check_capacity(facts, n_needed, split_name):
        max_pairs = len(facts) * (len(facts) - 1) // 2
        if n_needed > max_pairs:
            raise SystemExit(
                f"Cannot generate {n_needed} {split_name} examples: only {max_pairs} unique pairs possible "
                f"from {len(facts)} facts. Reduce --n-{split_name} or add more facts."
            )
    
    check_capacity(train_facts, args.n_train, "train")
    check_capacity(val_facts, args.n_val, "val")
    check_capacity(test_facts, args.n_test, "test")

    print(f"\nUsing filler_token_id={filler_id} for filler_token={args.filler_token!r}")
    print(f"Filler mode: {args.filler_mode}")
    if args.filler_mode == "uniform":
        print(f"  Range: [{args.filler_min}, {args.filler_max}]")
    else:
        print(f"  Values: {eval_filler_lengths}")

    # Create output directory
    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Build datasets
    print("\nGenerating training data...")
    train_builder = DatasetBuilder(
        facts=train_facts,
        tok=tok,
        filler_id=filler_id,
        filler_token=args.filler_token,
        max_answer=args.max_answer,
        filler_mode=args.filler_mode,
        filler_min=args.filler_min,
        filler_max=args.filler_max,
        eval_filler_lengths=eval_filler_lengths,
        rng=rng,
        max_tries=args.max_tries,
    )
    train_rows = train_builder.build_split(args.n_train, "train")
    
    print("\nGenerating validation data...")
    val_builder = DatasetBuilder(
        facts=val_facts,
        tok=tok,
        filler_id=filler_id,
        filler_token=args.filler_token,
        max_answer=args.max_answer,
        filler_mode=args.filler_mode,
        filler_min=args.filler_min,
        filler_max=args.filler_max,
        eval_filler_lengths=eval_filler_lengths,
        rng=rng,
        max_tries=args.max_tries,
    )
    val_rows = val_builder.build_split(args.n_val, "val")
    
    print("\nGenerating test data...")
    # Test set always uses eval filler lengths for clean evaluation
    test_builder = DatasetBuilder(
        facts=test_facts,
        tok=tok,
        filler_id=filler_id,
        filler_token=args.filler_token,
        max_answer=args.max_answer,
        filler_mode="eval",  # Always use eval mode for test
        filler_min=args.filler_min,
        filler_max=args.filler_max,
        eval_filler_lengths=eval_filler_lengths,
        rng=rng,
        max_tries=args.max_tries,
    )
    test_rows = test_builder.build_split(args.n_test, "test")

    # Write datasets
    write_jsonl(train_rows, outdir / "train.jsonl")
    write_jsonl(val_rows, outdir / "val.jsonl")
    write_jsonl(test_rows, outdir / "test.jsonl")

    # Save fact pools for reproducibility and analysis
    fact_pools = {
        "train": [{"question": q, "answer": a, "type": t} for q, a, t in train_facts],
        "val": [{"question": q, "answer": a, "type": t} for q, a, t in val_facts],
        "test": [{"question": q, "answer": a, "type": t} for q, a, t in test_facts],
    }
    (outdir / "fact_pools.json").write_text(json.dumps(fact_pools, indent=2), encoding="utf-8")

    # Save manifest
    manifest = {
        "tokenizer": args.tokenizer,
        "seed": args.seed,
        "max_answer": args.max_answer,
        "filler_token": args.filler_token,
        "filler_token_id": filler_id,
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
            "test": len(test_rows),
        },
        "unique_pairs_used": {
            "train": len(train_builder.used_pairs),
            "val": len(val_builder.used_pairs),
            "test": len(test_builder.used_pairs),
        },
        "bos_token_id": tok.bos_token_id,
        "eos_token_id": tok.eos_token_id,
    }
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Print summary
    print(f"\n{'='*60}")
    print(f"Done. Wrote dataset to: {outdir}")
    print(f"{'='*60}")
    print(f"  train.jsonl: {len(train_rows)} examples")
    print(f"  val.jsonl:   {len(val_rows)} examples")
    print(f"  test.jsonl:  {len(test_rows)} examples")
    print(f"\nFact isolation verified:")
    print(f"  - Train/val/test use completely separate fact pools")
    print(f"  - No duplicate pairs within any split")
    print(f"\nFiller length distribution in test set:")
    from collections import Counter
    test_filler_dist = Counter(r["filler_len"] for r in test_rows)
    for fl in sorted(test_filler_dist.keys()):
        print(f"  N={fl}: {test_filler_dist[fl]} examples")


if __name__ == "__main__":
    main()