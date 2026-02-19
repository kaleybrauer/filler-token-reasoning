#!/usr/bin/env python3
"""
Generate 2-hop arithmetic dataset for opaque reasoner experiments.

- Loads only model-verified known facts (from evaluate_individual_facts.py)
- Facts are split into train/val/test pools first, then pairs are composed
  only from facts within each pool. This prevents any fact leakage.
- Uses a 5-shot pure-pattern prompt format (no instructions).
- Filler tokens are counting sequences ("1 2 3 ... N") on a "Think:" line,
  with "Answer:" as the distinct output label. Each filler position gets a
  unique pretrained embedding.
- Pair deduplication within each split prevents duplicate questions.
- Filler lengths can be sampled uniformly or from a fixed set of eval values.
"""
import argparse
import json
import pathlib
import random
from typing import Any, Dict, List, Optional, Set, Tuple

from transformers import AutoTokenizer

# Default filler lengths used for evaluation (also used during training if --filler-mode=eval)
DEFAULT_EVAL_FILLER_LENGTHS = [0, 32, 128, 300, 600, 1000]

# ─────────────────────────────────────────────────────────────────────────────
# Few-shot examples for the prompt.
# These must NOT overlap with any facts in the evaluation pool.
# Few-shot examples never include a Think: line — only the training/test
# examples get filler tokens. The model learns Q1/Q2/Answer from few-shot,
# and learns Think→Answer from fine-tuning.
# ─────────────────────────────────────────────────────────────────────────────
FEWSHOT_EXAMPLES = [
    ("What is the number of legs of a cat?", "What is the number of days in a week?", 4, 7, 11),
    ("At what age did Abraham Lincoln die?", "What is the number of judges on the Supreme Court?", 56, 9, 65),
    ("What is the number of legs of a spider?", "At what age did Leonardo da Vinci die?", 8, 67, 75),
    ("What is the number of sides of a hexagon?", "What is the number of disciples that Jesus had?", 6, 12, 18),
    ("What is the number of continents on Earth?", "What is the number of sides of a triangle?", 7, 3, 10),
]


def build_fewshot_prefix() -> str:
    """Build the few-shot prefix string (constant across all examples).
    
    Each example is separated by a blank line. The prefix ends with a
    trailing blank line so the test question is separated consistently.
    """
    blocks = []
    for q1, q2, _a1, _a2, s in FEWSHOT_EXAMPLES:
        blocks.append(f"Q1: {q1}\nQ2: {q2}\nAnswer: {s}")
    return "\n\n".join(blocks) + "\n\n"


# Pre-build once so we don't re-create it for every example
FEWSHOT_PREFIX = build_fewshot_prefix()


def load_known_facts(path: pathlib.Path) -> List[Tuple[str, int, str]]:
    """Load pre-filtered known facts from known_facts.json.

    Expected format: list of {"question": str, "answer": int, "kind": str}
    (as produced by evaluate_individual_facts.py)
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    out: List[Tuple[str, int, str]] = []
    for item in raw:
        q = item["question"]
        a = int(item["answer"])
        k = item.get("kind", "unknown")
        out.append((q, a, k))
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


def _generate_valid_pairs(
    facts: List[Tuple[str, int, str]],
    max_answer: int,
    rng: random.Random,
) -> List[Tuple[int, int]]:
    """
    Pre-generate all valid (i, j) pairs where i < j and a_i + a_j < max_answer
    and a_i + a_j >= 0, then shuffle them.
    
    Returns shuffled list of (index_i, index_j) into the facts list.
    """
    pairs = []
    for i in range(len(facts)):
        for j in range(i + 1, len(facts)):
            s = facts[i][1] + facts[j][1]
            if 0 <= s < max_answer:
                pairs.append((i, j))
    rng.shuffle(pairs)
    return pairs


class DatasetBuilder:
    """Builds examples from a pool of facts using pre-generated pairs."""
    
    def __init__(
        self,
        facts: List[Tuple[str, int, str]],
        tok: AutoTokenizer,
        max_answer: int,
        filler_mode: str,
        filler_min: int,
        filler_max: int,
        eval_filler_lengths: List[int],
        rng: random.Random,
    ):
        self.facts = facts
        self.tok = tok
        self.max_answer = max_answer
        self.filler_mode = filler_mode
        self.filler_min = filler_min
        self.filler_max = filler_max
        self.eval_filler_lengths = eval_filler_lengths
        self.rng = rng
        
        # Pre-generate all valid pairs
        print(f"  Pre-generating valid pairs from {len(facts)} facts...")
        self.valid_pairs = _generate_valid_pairs(facts, max_answer, rng)
        print(f"  Found {len(self.valid_pairs)} valid pairs")
        
        # Pointer into the valid_pairs list
        self._pair_idx = 0
        
        # Track how many pairs were used (for reporting)
        self.used_pair_count = 0
        
        # Get BOS/EOS tokens
        self.bos_ids = [tok.bos_token_id] if tok.bos_token_id else []
        self.eos_ids = [tok.eos_token_id] if tok.eos_token_id else []
    
    def make_example(self, example_id: int, split: str) -> Dict[str, Any]:
        """Generate a single example from the next pre-generated pair."""
        if self._pair_idx >= len(self.valid_pairs):
            raise RuntimeError(
                f"Exhausted all {len(self.valid_pairs)} valid pairs from {len(self.facts)} facts. "
                f"Generated {self._pair_idx} examples so far. "
                "Reduce dataset size or add more facts."
            )
        
        i, j = self.valid_pairs[self._pair_idx]
        self._pair_idx += 1
        self.used_pair_count += 1
        
        # Randomly assign which fact is q1 vs q2
        if self.rng.random() < 0.5:
            (q1, a1, t1) = self.facts[i]
            (q2, a2, t2) = self.facts[j]
        else:
            (q1, a1, t1) = self.facts[j]
            (q2, a2, t2) = self.facts[i]
        
        s = a1 + a2
        
        # Build prompt_base: fewshot prefix + Q1/Q2 (no Think, no Answer)
        prompt_base = (
            f"{FEWSHOT_PREFIX}"
            f"Q1: {q1}\n"
            f"Q2: {q2}"
        )
        
        # Sample filler length (number of counting steps)
        nfill = sample_filler_len(
            self.rng,
            self.filler_mode,
            self.filler_min,
            self.filler_max,
            self.eval_filler_lengths,
        )
        
        # Build full prompt: prompt_base + optional Think line + Answer:
        if nfill == 0:
            prompt = prompt_base + "\nAnswer:"
        else:
            think_text = " ".join(str(i) for i in range(1, nfill + 1))
            prompt = prompt_base + f"\nThink: {think_text}\nAnswer:"
        
        prompt_ids = self.tok.encode(prompt, add_special_tokens=False)
        
        # Answer with leading space, terminated by EOS
        answer_text = " " + str(s)
        answer_ids = self.tok.encode(answer_text, add_special_tokens=False) + self.eos_ids
        
        # Assemble full sequence with BOS
        prefix_ids = self.bos_ids + prompt_ids
        input_ids = prefix_ids + answer_ids
        
        labels = [-100] * len(prefix_ids) + answer_ids
        attn = [1] * len(input_ids)
        
        return {
            "id": f"{split}-{example_id}",
            "split": split,
            "prompt": prompt,
            "prompt_base": prompt_base,
            "fact1": q1,
            "fact2": q2,
            "a1": a1,
            "a2": a2,
            "answer": s,
            "type1": t1,
            "type2": t2,
            "filler_len": nfill,
            "filler_type": "counting",
            "prompt_ids": prompt_ids,
            "answer_ids": answer_ids,
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attn,
        }
    
    def build_split(self, n: int, split: str) -> List[Dict[str, Any]]:
        """Build n examples for a split."""
        rows = []
        for i in range(n):
            rows.append(self.make_example(i, split))
            if (i + 1) % 10000 == 0:
                print(f"  Generated {i + 1}/{n} {split} examples...")
        return rows
    
    def max_possible_pairs(self) -> int:
        """Return maximum possible unique valid pairs from this fact pool."""
        return len(self.valid_pairs)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate 2-hop arithmetic dataset with fact-level train/val/test isolation."
    )
    ap.add_argument("--tokenizer", type=str, required=True, 
                    help="Tokenizer name, e.g. Qwen/Qwen2.5-7B")
    ap.add_argument("--known-facts", type=str, required=True,
                    help="Path to known_facts.json from evaluate_individual_facts.py")
    ap.add_argument("--outdir", type=str, required=True, 
                    help="Output directory for generated JSONL files")

    ap.add_argument("--n-train", type=int, default=28000)
    ap.add_argument("--n-val", type=int, default=600)
    ap.add_argument("--n-test", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--max-answer", type=int, default=1000, 
                    help="Keep final answers < max-answer")

    ap.add_argument("--filler-mode", type=str, default="eval", choices=["uniform", "eval"],
                    help="How to sample filler lengths: 'uniform' samples from [min,max], "
                         "'eval' samples from --eval-filler-lengths")
    ap.add_argument("--filler-min", type=int, default=0,
                    help="Min filler length (for uniform mode)")
    ap.add_argument("--filler-max", type=int, default=600,
                    help="Max filler length (for uniform mode)")
    ap.add_argument("--eval-filler-lengths", type=str, default="0,32,128,300,600",
                    help="Comma-separated filler lengths for eval mode and test set generation")

    ap.add_argument("--fact-split-train", type=float, default=0.75,
                    help="Fraction of facts for training pool (default: 0.75)")
    ap.add_argument("--fact-split-val", type=float, default=0.125,
                    help="Fraction of facts for validation pool (default: 0.125)")

    args = ap.parse_args()

    rng = random.Random(args.seed)
    
    # Parse eval filler lengths
    eval_filler_lengths = [int(x.strip()) for x in args.eval_filler_lengths.split(",")]
    print(f"Eval filler lengths: {eval_filler_lengths}")

    # Load tokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    
    # Report BOS/EOS tokens
    print(f"BOS token: {tok.bos_token!r} (id={tok.bos_token_id})")
    print(f"EOS token: {tok.eos_token!r} (id={tok.eos_token_id})")
    print(f"Filler type: counting tokens with Think:/Answer: format")

    # Load known facts
    known_path = pathlib.Path(args.known_facts)
    if not known_path.exists():
        raise SystemExit(f"Error: --known-facts file not found: {known_path}")

    all_facts = load_known_facts(known_path)
    all_facts = [(q, a, k) for (q, a, k) in all_facts if 0 <= a < args.max_answer]

    from collections import Counter
    kind_counts = Counter(k for _, _, k in all_facts)
    print(f"\nLoaded {len(all_facts)} known facts from {known_path}")
    for kind in sorted(kind_counts):
        print(f"  {kind}: {kind_counts[kind]}")
    
    if len(all_facts) < 50:
        raise SystemExit(f"Too few usable facts ({len(all_facts)}). Need at least 50.")

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

    print(f"\nFiller mode: {args.filler_mode}")
    if args.filler_mode == "uniform":
        print(f"  Range: [{args.filler_min}, {args.filler_max}] counting steps")
    else:
        print(f"  Values: {eval_filler_lengths} counting steps")

    # Create output directory
    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Build datasets
    print("\nGenerating training data...")
    train_builder = DatasetBuilder(
        facts=train_facts,
        tok=tok,
        max_answer=args.max_answer,
        filler_mode=args.filler_mode,
        filler_min=args.filler_min,
        filler_max=args.filler_max,
        eval_filler_lengths=eval_filler_lengths,
        rng=rng,
    )
    
    # Check capacity before generating
    if args.n_train > train_builder.max_possible_pairs():
        raise SystemExit(
            f"Cannot generate {args.n_train} train examples: only "
            f"{train_builder.max_possible_pairs()} valid unique pairs possible "
            f"from {len(train_facts)} facts. Reduce --n-train or add more facts."
        )
    
    train_rows = train_builder.build_split(args.n_train, "train")
    
    print("\nGenerating validation data...")
    val_builder = DatasetBuilder(
        facts=val_facts,
        tok=tok,
        max_answer=args.max_answer,
        filler_mode=args.filler_mode,
        filler_min=args.filler_min,
        filler_max=args.filler_max,
        eval_filler_lengths=eval_filler_lengths,
        rng=rng,
    )
    
    if args.n_val > val_builder.max_possible_pairs():
        raise SystemExit(
            f"Cannot generate {args.n_val} val examples: only "
            f"{val_builder.max_possible_pairs()} valid unique pairs possible "
            f"from {len(val_facts)} facts. Reduce --n-val or add more facts."
        )
    
    val_rows = val_builder.build_split(args.n_val, "val")
    
    print("\nGenerating test data...")
    # Test set always uses eval filler lengths for clean evaluation
    test_builder = DatasetBuilder(
        facts=test_facts,
        tok=tok,
        max_answer=args.max_answer,
        filler_mode="eval",  # Always use eval mode for test
        filler_min=args.filler_min,
        filler_max=args.filler_max,
        eval_filler_lengths=eval_filler_lengths,
        rng=rng,
    )
    
    if args.n_test > test_builder.max_possible_pairs():
        raise SystemExit(
            f"Cannot generate {args.n_test} test examples: only "
            f"{test_builder.max_possible_pairs()} valid unique pairs possible "
            f"from {len(test_facts)} facts. Reduce --n-test or add more facts."
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
        "known_facts_source": str(args.known_facts),
        "prompt_format": "fewshot_5shot_think_answer",
        "filler_type": "counting",
        "seed": args.seed,
        "max_answer": args.max_answer,
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
        "valid_pairs": {
            "train": train_builder.max_possible_pairs(),
            "val": val_builder.max_possible_pairs(),
            "test": test_builder.max_possible_pairs(),
        },
        "unique_pairs_used": {
            "train": train_builder.used_pair_count,
            "val": val_builder.used_pair_count,
            "test": test_builder.used_pair_count,
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
        seq_lens = [len(r["input_ids"]) for r in test_rows if r["filler_len"] == fl]
        avg_seq = sum(seq_lens) / len(seq_lens) if seq_lens else 0
        print(f"  N={fl}: {test_filler_dist[fl]} examples (avg {avg_seq:.0f} tokens)")
    
    # Report overall sequence length stats
    all_seq_lens = [len(r["input_ids"]) for r in train_rows]
    print(f"\nTraining sequence lengths: min={min(all_seq_lens)}, max={max(all_seq_lens)}, "
          f"avg={sum(all_seq_lens)/len(all_seq_lens):.0f}")


if __name__ == "__main__":
    main()