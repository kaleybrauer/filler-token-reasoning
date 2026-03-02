#!/usr/bin/env python3
"""
Evaluate a model on a random sample of unique test pairs drawn from
test.jsonl and extra_test.jsonl (if it exists).

By default samples all available unique pairs. Pass --n-pairs to limit.
Pass --seed to make sampling deterministic and comparable across runs.

NOTE on robustness: sampling test pairs with different seeds measures
estimation variance (how stable is accuracy with N pairs?). If you want
to check whether the *training effect* is robust, train multiple models
with different seeds and evaluate each on the full test set instead.

Usage:
    # All pairs, default seed
    python dev/sample_and_evaluate.py \
        --data-dir data/datasets/2hop_add \
        --model Qwen/Qwen2.5-72B-Instruct \
        --adapter outputs/qwen2p5-72b-qlora \
        --outdir results/eval_seed42 \
        --load-in-4bit --batch-size 8

    # Subsample 600 pairs, different seeds for variance estimate
    python dev/sample_and_evaluate.py \
        --data-dir data/datasets/2hop_add \
        --model Qwen/Qwen2.5-72B-Instruct \
        --n-pairs 600 --seed 0 \
        --outdir results/eval_seed0 \
        --load-in-4bit --batch-size 8
"""
import argparse
import json
import pathlib
import random
import subprocess
import sys
import tempfile
from collections import defaultdict
from typing import Dict, FrozenSet, List


def load_jsonl(path: pathlib.Path) -> List[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Evaluate a model on a random sample of test pairs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Sampling
    ap.add_argument("--data-dir", required=True,
                    help="Dataset directory (contains test.jsonl and manifest.json)")
    ap.add_argument("--extra-test", default=None,
                    help="Extra test JSONL (default: <data-dir>/extra_test.jsonl if it exists)")
    ap.add_argument("--n-pairs", type=int, default=None,
                    help="Unique pairs to sample (default: all available)")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed for pair sampling")

    # Model
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--load-in-8bit", action="store_true")
    ap.add_argument("--no-flash-attn", action="store_true")
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--cache-dir", default=None)

    # Evaluation settings
    ap.add_argument("--filler-lengths", default=None,
                    help="Comma-separated filler lengths (default: use stored lengths)")
    ap.add_argument("--filler-type", default=None, choices=["dots", "counting"])
    ap.add_argument("--max-new-tokens", type=int, default=10)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--batch-size", type=int, default=8)

    # Output
    ap.add_argument("--outdir", default=None,
                    help="Directory to save results (enables resume)")
    ap.add_argument("--report-every", type=int, default=100)
    ap.add_argument("--show-errors", action="store_true")

    # W&B
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb-project", default="opaque-reasoner")
    ap.add_argument("--wandb-run-name", default=None)

    args = ap.parse_args()

    data_dir = pathlib.Path(args.data_dir)
    extra_path = pathlib.Path(args.extra_test) if args.extra_test else data_dir / "extra_test.jsonl"

    # ── Load rows ─────────────────────────────────────────────────────────────
    test_path = data_dir / "test.jsonl"
    if not test_path.exists():
        raise SystemExit(f"test.jsonl not found in {data_dir}")

    all_rows = load_jsonl(test_path)
    print(f"Loaded {len(all_rows)} rows from test.jsonl")

    if extra_path.exists():
        extra_rows = load_jsonl(extra_path)
        print(f"Loaded {len(extra_rows)} rows from {extra_path.name}")
        all_rows.extend(extra_rows)
    else:
        print(f"No extra test file at {extra_path} — using test.jsonl only")

    # Filter CoT examples (same as evaluate.py)
    all_rows = [r for r in all_rows if r.get("sequence_type", "filler") != "cot"]

    # ── Group by unique pair ──────────────────────────────────────────────────
    pair_to_rows: Dict[FrozenSet, List[dict]] = defaultdict(list)
    for row in all_rows:
        key = frozenset([row["fact1"], row["fact2"]])
        pair_to_rows[key].append(row)

    unique_pairs = list(pair_to_rows.keys())
    print(f"Unique pairs available: {len(unique_pairs)}")

    # ── Sample ────────────────────────────────────────────────────────────────
    rng = random.Random(args.seed)
    rng.shuffle(unique_pairs)

    n_pairs = args.n_pairs if args.n_pairs is not None else len(unique_pairs)
    if n_pairs > len(unique_pairs):
        raise SystemExit(
            f"Requested {n_pairs} pairs but only {len(unique_pairs)} unique pairs available. "
            "Reduce --n-pairs."
        )

    sampled_rows = []
    for pair in unique_pairs[:n_pairs]:
        sampled_rows.extend(pair_to_rows[pair])

    print(f"Sampled {n_pairs} pairs → {len(sampled_rows)} examples (seed={args.seed})")

    # ── Write temp dir and call evaluate.py ───────────────────────────────────
    eval_script = pathlib.Path(__file__).parent.parent / "scripts" / "evaluate.py"

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = pathlib.Path(tmpdir)

        # Write sampled examples as test.jsonl
        out_jsonl = tmpdir_path / "test.jsonl"
        with out_jsonl.open("w", encoding="utf-8") as f:
            for row in sampled_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        # evaluate.py requires manifest.json for fewshot_examples
        manifest_src = data_dir / "manifest.json"
        if not manifest_src.exists():
            raise SystemExit(f"manifest.json not found in {data_dir}")
        (tmpdir_path / "manifest.json").write_bytes(manifest_src.read_bytes())

        # Build command
        cmd = [
            sys.executable, str(eval_script),
            "--model", args.model,
            "--data-dir", str(tmpdir_path),
            "--split", "test",
            "--batch-size", str(args.batch_size),
            "--max-new-tokens", str(args.max_new_tokens),
            "--temperature", str(args.temperature),
            "--report-every", str(args.report_every),
        ]
        if args.adapter:          cmd += ["--adapter", args.adapter]
        if args.load_in_4bit:     cmd.append("--load-in-4bit")
        if args.load_in_8bit:     cmd.append("--load-in-8bit")
        if args.no_flash_attn:    cmd.append("--no-flash-attn")
        if args.trust_remote_code: cmd.append("--trust-remote-code")
        if args.filler_lengths:   cmd += ["--filler-lengths", args.filler_lengths]
        if args.filler_type:      cmd += ["--filler-type", args.filler_type]
        if args.outdir:           cmd += ["--outdir", args.outdir]
        if args.show_errors:      cmd.append("--show-errors")
        if args.cache_dir:        cmd += ["--cache-dir", args.cache_dir]
        if args.wandb:            cmd.append("--wandb")
        if args.wandb_project:    cmd += ["--wandb-project", args.wandb_project]
        if args.wandb_run_name:   cmd += ["--wandb-run-name", args.wandb_run_name]

        print(f"\nRunning evaluate.py on {n_pairs} pairs (seed={args.seed})...\n")
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
