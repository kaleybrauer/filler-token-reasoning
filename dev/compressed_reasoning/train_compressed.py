#!/usr/bin/env python3
"""
Train compressed reasoning model (curriculum stages).

Standard SFT with loss on all assistant tokens (thinking + answer).
Supports curriculum by initializing from a previous stage checkpoint.

Usage:
    # Stage 0: Initialize from base model
    python scripts/train_compressed_reasoning.py \
        --model Qwen/Qwen2.5-14B-Instruct \
        --data-dir data/datasets/compressed_s0 \
        --outdir outputs/compressed_s0 \
        --lr 1e-5 --epochs 1

    # Stage 1: Initialize from Stage 0
    python scripts/train_compressed_reasoning.py \
        --model outputs/compressed_s0 \
        --data-dir data/datasets/compressed_s1 \
        --outdir outputs/compressed_s1 \
        --lr 5e-6 --epochs 1

    # Stage 2: Initialize from Stage 1
    python scripts/train_compressed_reasoning.py \
        --model outputs/compressed_s1 \
        --data-dir data/datasets/compressed_s2 \
        --outdir outputs/compressed_s2 \
        --lr 5e-6 --epochs 1
"""
import argparse
import inspect
import json
import os
import pathlib
from typing import Any, Dict, List

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

USE_BF16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
DTYPE = torch.bfloat16 if USE_BF16 else torch.float16


def is_main_process() -> bool:
    return int(os.environ.get("RANK", 0)) == 0


# ─────────────────────────────────────────────────────────────────────────────
# Data Collator
# ─────────────────────────────────────────────────────────────────────────────

class DataCollatorForCausalLM:
    def __init__(self, pad_token_id: int, padding_side: str = "right"):
        self.pad_token_id = pad_token_id
        self.padding_side = padding_side

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)
        batch = {"input_ids": [], "attention_mask": [], "labels": []}

        for f in features:
            pad_len = max_len - len(f["input_ids"])
            if self.padding_side == "right":
                batch["input_ids"].append(f["input_ids"] + [self.pad_token_id] * pad_len)
                batch["attention_mask"].append(f["attention_mask"] + [0] * pad_len)
                batch["labels"].append(f["labels"] + [-100] * pad_len)
            else:
                batch["input_ids"].append([self.pad_token_id] * pad_len + f["input_ids"])
                batch["attention_mask"].append([0] * pad_len + f["attention_mask"])
                batch["labels"].append([-100] * pad_len + f["labels"])

        return {k: torch.tensor(v, dtype=torch.long) for k, v in batch.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset_split(data_dir: pathlib.Path, split: str):
    data_file = data_dir / f"{split}.jsonl"
    if not data_file.exists():
        return None

    dataset = load_dataset("json", data_files=str(data_file), split="train")
    n_total = len(dataset)

    keep_cols = {"input_ids", "attention_mask", "labels"}
    remove_cols = [c for c in dataset.column_names if c not in keep_cols]
    if remove_cols:
        dataset = dataset.remove_columns(remove_cols)

    print(f"  {split}: {n_total} examples")
    return dataset


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train compressed reasoning (curriculum SFT)"
    )

    # Model
    parser.add_argument("--model", type=str, required=True,
                        help="Base model or previous stage checkpoint")
    parser.add_argument("--no-flash-attn", action="store_true")
    parser.add_argument("--no-grad-checkpoint", action="store_true")

    # Data
    parser.add_argument("--data-dir", type=str, required=True)

    # Output
    parser.add_argument("--outdir", type=str, required=True)

    # Training
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--eval-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume-from", type=str, default=None)

    # Wandb
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="opaque-reasoner")
    parser.add_argument("--wandb-run-name", type=str, default=None)

    args = parser.parse_args()

    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    data_dir = pathlib.Path(args.data_dir)

    # Load manifest for logging
    manifest_path = data_dir / "manifest.json"
    manifest = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        print(f"Stage: {manifest.get('stage', '?')}, "
              f"n_think: {manifest.get('n_think_tokens', '?')}")

    # ── Load tokenizer ──
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Load model ──
    print(f"\nLoading model: {args.model}")
    model_kwargs = dict(
        torch_dtype=DTYPE,
        low_cpu_mem_usage=True,
    )

    if not args.no_flash_attn:
        try:
            import flash_attn
            model_kwargs["attn_implementation"] = "flash_attention_2"
            print("Using Flash Attention 2")
        except ImportError:
            model_kwargs["attn_implementation"] = "sdpa"
            print("Using SDPA attention")

    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)

    if not args.no_grad_checkpoint:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        print("Gradient checkpointing enabled")

    if torch.cuda.is_available():
        mem_gb = torch.cuda.memory_allocated() / 1e9
        print(f"GPU memory after model load: {mem_gb:.2f} GB")

    # ── Load data ──
    print(f"\nLoading data from {data_dir}")
    train_dataset = load_dataset_split(data_dir, "train")
    eval_dataset = load_dataset_split(data_dir, "val")

    if train_dataset is None:
        raise FileNotFoundError(f"No train.jsonl found in {data_dir}")

    # ── Training arguments ──
    ta_kwargs = dict(
        output_dir=str(outdir),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_strategy="steps",
        save_total_limit=3,
        bf16=USE_BF16,
        fp16=not USE_BF16,
        gradient_checkpointing=not args.no_grad_checkpoint,
        optim="adamw_bnb_8bit",
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        max_grad_norm=1.0,
        weight_decay=0.01,
        seed=args.seed,
    )

    if not args.no_grad_checkpoint:
        ta_kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}

    # Eval config
    ta_sig = inspect.signature(TrainingArguments.__init__).parameters
    if eval_dataset is not None:
        ta_kwargs["eval_steps"] = args.eval_steps
        if "eval_strategy" in ta_sig:
            ta_kwargs["eval_strategy"] = "steps"
        else:
            ta_kwargs["evaluation_strategy"] = "steps"

    # Wandb
    if args.wandb and WANDB_AVAILABLE:
        ta_kwargs["report_to"] = "wandb"
        stage = manifest.get("stage", "?")
        run_name = args.wandb_run_name or f"compressed_s{stage}_lr{args.lr}"
        ta_kwargs["run_name"] = run_name
        if is_main_process():
            wandb.init(
                project=args.wandb_project,
                name=run_name,
                config={
                    "phase": "compressed_reasoning",
                    "stage": stage,
                    "n_think_tokens": manifest.get("n_think_tokens"),
                    "model": args.model,
                    "lr": args.lr,
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "grad_accum": args.grad_accum,
                    "effective_batch_size": args.batch_size * args.grad_accum,
                    "train_examples": len(train_dataset),
                },
            )
    else:
        ta_kwargs["report_to"] = "none"

    training_args = TrainingArguments(**ta_kwargs)

    # ── Trainer ──
    data_collator = DataCollatorForCausalLM(
        pad_token_id=tokenizer.pad_token_id,
        padding_side=tokenizer.padding_side,
    )

    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    tr_sig = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in tr_sig:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = Trainer(**trainer_kwargs)

    # Resume
    resume_checkpoint = None
    if args.resume_from:
        if args.resume_from.lower() == "latest":
            import glob
            checkpoints = sorted(
                glob.glob(str(outdir / "checkpoint-*")),
                key=lambda x: int(x.rsplit("-", 1)[-1]),
            )
            if checkpoints:
                resume_checkpoint = checkpoints[-1]
                print(f"Resuming from: {resume_checkpoint}")
        else:
            resume_checkpoint = args.resume_from

    # Train
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\nStarting training...")
    print(f"  Stage: {manifest.get('stage', '?')}")
    print(f"  n_think: {manifest.get('n_think_tokens', '?')}")
    print(f"  Train: {len(train_dataset)} examples")
    print(f"  LR: {args.lr}")
    trainer.train(resume_from_checkpoint=resume_checkpoint)

    # Save
    print(f"\nSaving model to {outdir}")
    trainer.save_model(str(outdir))
    tokenizer.save_pretrained(str(outdir))

    config = {
        "phase": "compressed_reasoning",
        "stage": manifest.get("stage"),
        "n_think_tokens": manifest.get("n_think_tokens"),
        "model_init": args.model,
        "lr": args.lr,
        "epochs": args.epochs,
        "train_examples": len(train_dataset),
    }
    with open(outdir / "training_config.json", "w") as f:
        json.dump(config, f, indent=2)

    if args.wandb and WANDB_AVAILABLE and is_main_process():
        wandb.finish()

    print("Training complete!")


if __name__ == "__main__":
    main()