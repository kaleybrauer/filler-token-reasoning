#!/usr/bin/env python3
"""
Evaluate a model on the alphabet filler addition dataset.

Wrapper around scripts/evaluate.py that adds "alphabet" as a valid filler type.
Reads filler type from the dataset manifest by default (no --filler-type needed).

Default data dir:  data/datasets/addition_alphabet_filler
Default outdir:    results/alphabet_filler

Example:
    # Evaluate fine-tuned model
    python dev/evaluate_alphabet_filler.py \\
        --model Qwen/Qwen2.5-72B-Instruct \\
        --adapter outputs/alphabet_filler \\
        --load-in-4bit --batch-size 4

    # Evaluate base model (baseline)
    python dev/evaluate_alphabet_filler.py \\
        --model Qwen/Qwen2.5-72B-Instruct \\
        --outdir results/alphabet_filler_baseline \\
        --load-in-4bit --batch-size 4

    # Evaluate on val_eval split
    python dev/evaluate_alphabet_filler.py \\
        --model Qwen/Qwen2.5-72B-Instruct \\
        --adapter outputs/alphabet_filler \\
        --split val_eval --load-in-4bit
"""
import json
import pathlib
import sys
from typing import Any, Dict, List, Optional

# ─────────────────────────────────────────────────────────────────────────────
# Add scripts/ to path and patch _build_filler to support "alphabet" type.
# All evaluation logic is imported from scripts/evaluate.py.
# ─────────────────────────────────────────────────────────────────────────────

_REPO_ROOT = pathlib.Path(__file__).parent.parent
_SCRIPTS_DIR = _REPO_ROOT / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))

import generate_addition_dataset as _gen  # noqa: E402

_ALPHABET = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")

_orig_build_filler = _gen._build_filler


def _patched_build_filler(filler_len: int, filler_type: str = "dots") -> str:
    if filler_type == "alphabet":
        return " ".join(_ALPHABET[i % len(_ALPHABET)] for i in range(filler_len))
    return _orig_build_filler(filler_len, filler_type)


_gen._build_filler = _patched_build_filler

# Import evaluate functions AFTER patching so build_filler_prefill picks up the patch
import evaluate as _eval  # noqa: E402

# ─────────────────────────────────────────────────────────────────────────────
# Defaults for this experiment
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULT_DATA_DIR = str(_REPO_ROOT / "data" / "datasets" / "addition_alphabet_filler")
_DEFAULT_OUTDIR = str(_REPO_ROOT / "results" / "alphabet_filler")


def main() -> None:
    import argparse
    import time
    import torch
    from datasets import load_dataset

    parser = argparse.ArgumentParser(
        description="Evaluate model on alphabet filler addition dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model
    parser.add_argument("--model", type=str, required=True,
                        help="Base model name or path")
    parser.add_argument("--adapter", type=str, default=None,
                        help="Path to LoRA adapter (optional)")
    parser.add_argument("--load-in-4bit", action="store_true",
                        help="Use 4-bit quantization (recommended for 72B)")
    parser.add_argument("--load-in-8bit", action="store_true",
                        help="Use 8-bit quantization")
    parser.add_argument("--no-flash-attn", action="store_true",
                        help="Disable Flash Attention")
    parser.add_argument("--trust-remote-code", action="store_true")

    # Data
    parser.add_argument("--data-dir", type=str, default=_DEFAULT_DATA_DIR,
                        help="Directory containing test.jsonl and manifest.json")
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "val", "val_eval", "test"],
                        help="Which split to evaluate")
    parser.add_argument("--max-examples", type=int, default=None,
                        help="Evaluate only this many examples (for quick testing)")

    # Evaluation settings
    parser.add_argument("--filler-lengths", type=str, default=None,
                        help="Override filler lengths, e.g. '0,250,500,1000'. "
                             "Default: use stored filler_len from dataset.")
    parser.add_argument("--filler-type", type=str, default=None,
                        choices=["dots", "counting", "alphabet"],
                        help="Override filler type. Default: read from manifest.")
    parser.add_argument("--max-new-tokens", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=8)

    # Output
    parser.add_argument("--outdir", type=str, default=_DEFAULT_OUTDIR,
                        help="Directory to save detailed results (enables resume)")
    parser.add_argument("--report-every", type=int, default=100)
    parser.add_argument("--show-errors", action="store_true")

    # Cache
    parser.add_argument("--cache-dir", type=str, default=None)

    # Wandb
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="filler-token-reasoning")
    parser.add_argument("--wandb-run-name", type=str, default=None)

    args = parser.parse_args()

    filler_lengths = _eval.parse_int_list(args.filler_lengths)
    if filler_lengths:
        print(f"Evaluating at filler lengths: {filler_lengths}")
    else:
        print("Using stored filler lengths from dataset")

    # Load model
    print(f"\nLoading model: {args.model}")
    if args.adapter:
        print(f"With adapter: {args.adapter}")

    model, tokenizer = _eval.load_model_and_tokenizer(
        args.model,
        adapter_path=args.adapter,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        use_flash_attn=not args.no_flash_attn,
        trust_remote_code=args.trust_remote_code,
        cache_dir=args.cache_dir,
    )

    # Load dataset
    data_dir = pathlib.Path(args.data_dir)
    data_file = data_dir / f"{args.split}.jsonl"
    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")

    print(f"\nLoading data from {data_file}")
    dataset = load_dataset("json", data_files=str(data_file), split="train")
    print(f"Loaded {len(dataset)} examples")

    if "sequence_type" in dataset.column_names:
        n_before = len(dataset)
        dataset = dataset.filter(lambda ex: ex["sequence_type"] != "cot")
        n_cot = n_before - len(dataset)
        if n_cot > 0:
            print(f"Filtered out {n_cot} CoT examples, {len(dataset)} filler examples remaining")

    # Load manifest
    manifest_file = data_dir / "manifest.json"
    if not manifest_file.exists():
        raise FileNotFoundError(f"manifest.json not found in {data_dir}.")
    manifest = json.loads(manifest_file.read_text())

    manifest_filler_type = manifest.get("filler_type", "alphabet")
    print(f"Filler type from manifest: {manifest_filler_type}")
    filler_type = args.filler_type or manifest_filler_type
    print(f"Using filler type: {filler_type}")

    raw_fewshot = manifest.get("fewshot_examples")
    if not raw_fewshot:
        raise KeyError("manifest.json does not contain 'fewshot_examples'.")
    fewshot_examples = [(e["q1"], e["q2"], e["a1"], e["a2"], e["sum"]) for e in raw_fewshot]
    print(f"Loaded {len(fewshot_examples)} few-shot examples from manifest")

    # Initialize tracker
    outdir = pathlib.Path(args.outdir)
    tracker = _eval.ResultsTracker(outdir=outdir, report_every=args.report_every)

    # Wandb
    try:
        import wandb
        WANDB_AVAILABLE = True
    except ImportError:
        WANDB_AVAILABLE = False

    if args.wandb:
        if not WANDB_AVAILABLE:
            print("WARNING: wandb not installed")
        else:
            run_name = args.wandb_run_name or (
                f"alphabet-eval_{args.model.split('/')[-1]}"
                + ("_lora" if args.adapter else "_base")
            )
            wandb.init(
                project=args.wandb_project,
                name=run_name,
                config={
                    "model": args.model,
                    "adapter": args.adapter,
                    "filler_type": filler_type,
                    "filler_lengths": filler_lengths,
                    "split": args.split,
                    "max_examples": args.max_examples,
                    "temperature": args.temperature,
                    "load_in_4bit": args.load_in_4bit,
                    "batch_size": args.batch_size,
                },
            )

    # Evaluate
    print("\nStarting evaluation...")
    results = _eval.evaluate_dataset(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        tracker=tracker,
        fewshot_examples=fewshot_examples,
        filler_lengths=filler_lengths,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        max_examples=args.max_examples,
        batch_size=args.batch_size,
        filler_type=filler_type,
        dataset_type="addition",
    )

    tracker.finalize()

    if args.show_errors:
        _eval.analyze_errors(results)

    if args.wandb and WANDB_AVAILABLE:
        log_dict = {
            "overall_accuracy": results["overall_accuracy"],
            "overall_correct": results["overall_correct"],
            "overall_total": results["overall_total"],
        }
        for n, stats in results["accuracy_by_filler_len"].items():
            log_dict[f"accuracy_N{n}"] = stats["accuracy"]
            log_dict[f"correct_N{n}"] = stats["correct"]
            log_dict[f"total_N{n}"] = stats["total"]

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(10, 6))
            ns = sorted(results["accuracy_by_filler_len"].keys())
            accs = [results["accuracy_by_filler_len"][n]["accuracy"] * 100 for n in ns]
            ax.plot(ns, accs, "o-", linewidth=2, markersize=8)
            ax.set_xlabel("Filler Length (N)", fontsize=12)
            ax.set_ylabel("Accuracy (%)", fontsize=12)
            ax.set_title("Alphabet Filler: Accuracy vs N", fontsize=14)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 100)
            log_dict["accuracy_vs_N"] = wandb.Image(fig)
            plt.close(fig)
        except ImportError:
            pass

        wandb.log(log_dict)
        wandb.finish()
        print("Logged results to Weights & Biases")

    print("\nDone!")
    return results


if __name__ == "__main__":
    main()
