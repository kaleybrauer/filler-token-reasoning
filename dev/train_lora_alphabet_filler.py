#!/usr/bin/env python3
"""
Fine-tune on the alphabet filler addition dataset.

Launcher wrapper around scripts/train_lora.py with defaults pre-set for the
alphabet filler diversity experiment. Any argument accepted by train_lora.py
can be appended and will override these defaults.

Default data dir:  data/datasets/addition_alphabet_filler
Default output:    outputs/alphabet_filler

Example:
    # Single GPU (7B)
    python dev/train_lora_alphabet_filler.py \\
        --model Qwen/Qwen2.5-7B-Instruct \\
        --load-in-4bit --batch-size 4 --grad-accum 8 --wandb

    # Multi-GPU (72B)
    torchrun --nproc_per_node=8 dev/train_lora_alphabet_filler.py \\
        --model Qwen/Qwen2.5-72B-Instruct \\
        --load-in-4bit --batch-size 2 --grad-accum 16 --wandb

    # Override data or output dir
    python dev/train_lora_alphabet_filler.py \\
        --model Qwen/Qwen2.5-7B-Instruct \\
        --data-dir /path/to/custom/dataset \\
        --outdir outputs/my_run
"""
import argparse
import pathlib
import subprocess
import sys

_REPO_ROOT = pathlib.Path(__file__).parent.parent
_TRAIN_SCRIPT = _REPO_ROOT / "scripts" / "train_lora.py"

_DEFAULT_DATA_DIR = str(_REPO_ROOT / "data" / "datasets" / "addition_alphabet_filler")
_DEFAULT_OUTDIR = str(_REPO_ROOT / "outputs" / "alphabet_filler")


def main() -> None:
    # Parse only the args we want to set defaults for; pass everything else through.
    parser = argparse.ArgumentParser(
        description="Train LoRA on alphabet filler addition dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        # Allow unknown args so anything train_lora.py accepts passes through.
        add_help=False,
    )
    parser.add_argument("--model", type=str, required=True,
                        help="Base model name or path (e.g. Qwen/Qwen2.5-7B-Instruct)")
    parser.add_argument("--data-dir", type=str, default=_DEFAULT_DATA_DIR,
                        help="Directory with train.jsonl / val.jsonl")
    parser.add_argument("--outdir", type=str, default=_DEFAULT_OUTDIR,
                        help="Output directory for checkpoints and final adapter")
    parser.add_argument("-h", "--help", action="store_true",
                        help="Show this help and pass --help to train_lora.py")

    args, passthrough = parser.parse_known_args()

    if args.help:
        parser.print_help()
        print("\n--- train_lora.py help ---")
        subprocess.run([sys.executable, str(_TRAIN_SCRIPT), "--help"])
        return

    cmd = [
        sys.executable, str(_TRAIN_SCRIPT),
        "--model", args.model,
        "--data-dir", args.data_dir,
        "--outdir", args.outdir,
        # Default wandb run name identifies this as an alphabet filler run
        "--wandb-run-name", f"alphabet-filler_{pathlib.Path(args.model).name}",
    ] + passthrough

    print("Running:")
    print("  " + " \\\n    ".join(cmd))
    print()

    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
