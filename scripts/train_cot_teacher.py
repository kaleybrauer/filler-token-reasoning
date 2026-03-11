#!/usr/bin/env python3
"""
Train a CoT teacher model (full fine-tuning) and cache hidden states.

Two modes:
  --phase train   Full fine-tune on CoT-only examples
  --phase cache   Load trained model, run forward passes, save hidden states

The cached hidden states are used as distillation targets for filler student training.

Hardware requirements (Qwen2.5-14B, bf16):
  Model weights: ~28 GB
  Optimizer (8-bit Adam): ~56 GB
  Gradients: ~28 GB
  Total: ~112 GB + activations
  → Fits on 1x H200 (141 GB) or 2x H100 (160 GB with DeepSpeed ZeRO-2)

Usage:
  # Phase 1: Train
  accelerate launch --config_file ds_config.yaml scripts/train_cot_teacher.py \\
      --phase train \\
      --model Qwen/Qwen2.5-14B-Instruct \\
      --data-dir data/datasets/2hop_14b \\
      --outdir outputs/cot_teacher_14b \\
      --lr 1e-5 --epochs 1 --batch-size 1 --grad-accum 32

  # Phase 2: Cache hidden states
  python scripts/train_cot_teacher.py \\
      --phase cache \\
      --model outputs/cot_teacher_14b \\
      --data-dir data/datasets/2hop_14b \\
      --cache-outdir outputs/cot_teacher_14b/cached_hidden_states \\
      --cache-layer -1 --batch-size 4
"""
import argparse
import inspect
import json
import os
import pathlib
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

# Optional imports
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

USE_BF16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
DTYPE = torch.bfloat16 if USE_BF16 else torch.float16


# ─────────────────────────────────────────────────────────────────────────────
# Data Collator (same as train_lora.py)
# ─────────────────────────────────────────────────────────────────────────────

class DataCollatorForCausalLM:
    """Pads pre-tokenized sequences for batching."""

    def __init__(self, pad_token_id: int, padding_side: str = "right"):
        self.pad_token_id = pad_token_id
        self.padding_side = padding_side

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)
        batch = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
        }
        for f in features:
            seq_len = len(f["input_ids"])
            pad_len = max_len - seq_len
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

def load_cot_dataset(data_dir: pathlib.Path, split: str = "train"):
    """Load dataset and filter to CoT examples only."""
    data_file = data_dir / f"{split}.jsonl"
    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")

    dataset = load_dataset("json", data_files=str(data_file), split="train")
    n_before = len(dataset)

    # Filter to CoT examples only
    if "sequence_type" in dataset.column_names:
        dataset = dataset.filter(lambda ex: ex["sequence_type"] == "cot")
        print(f"Filtered {split}: {n_before} → {len(dataset)} CoT examples")
    else:
        print(f"Loaded {split}: {len(dataset)} examples (no sequence_type column)")

    return dataset


def load_training_dataset(data_dir: pathlib.Path):
    """Load and prepare CoT training data."""
    dataset = load_cot_dataset(data_dir, "train")

    keep_cols = {"input_ids", "attention_mask", "labels"}
    remove_cols = [c for c in dataset.column_names if c not in keep_cols]
    if remove_cols:
        dataset = dataset.remove_columns(remove_cols)

    return dataset


def load_eval_dataset(data_dir: pathlib.Path):
    """Load and prepare CoT eval data."""
    dataset = load_cot_dataset(data_dir, "val")

    keep_cols = {"input_ids", "attention_mask", "labels"}
    remove_cols = [c for c in dataset.column_names if c not in keep_cols]
    if remove_cols:
        dataset = dataset.remove_columns(remove_cols)

    return dataset


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: Train CoT Teacher
# ─────────────────────────────────────────────────────────────────────────────

def train_teacher(args):
    """Full fine-tune on CoT examples."""

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model — full precision bf16, no quantization
    print(f"\nLoading model: {args.model}")
    model_kwargs = dict(
        torch_dtype=DTYPE,
        low_cpu_mem_usage=True,
    )

    # Attention implementation
    if not args.no_flash_attn:
        try:
            import flash_attn
            model_kwargs["attn_implementation"] = "flash_attention_2"
            print("Using Flash Attention 2")
        except ImportError:
            model_kwargs["attn_implementation"] = "sdpa"
            print("Using SDPA attention")

    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)

    # Enable gradient checkpointing to save memory
    if not args.no_grad_checkpoint:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        print("Gradient checkpointing enabled")

    if torch.cuda.is_available():
        mem_gb = torch.cuda.memory_allocated() / 1e9
        print(f"GPU memory after model load: {mem_gb:.2f} GB")

    # Load data
    data_dir = pathlib.Path(args.data_dir)
    train_dataset = load_training_dataset(data_dir)
    eval_dataset = None
    if (data_dir / "val.jsonl").exists():
        eval_dataset = load_eval_dataset(data_dir)

    print(f"\nTraining: {len(train_dataset)} CoT examples")
    if eval_dataset:
        print(f"Eval: {len(eval_dataset)} CoT examples")

    # Training arguments
    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

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
        optim="adamw_torch",
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
        run_name = args.wandb_run_name or f"cot_teacher_{args.model.split('/')[-1]}"
        ta_kwargs["run_name"] = run_name
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={
                "phase": "cot_teacher",
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

    # Data collator
    data_collator = DataCollatorForCausalLM(
        pad_token_id=tokenizer.pad_token_id,
        padding_side=tokenizer.padding_side,
    )

    # Trainer
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

    # Resume support
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
    print("\nStarting CoT teacher training...")
    trainer.train(resume_from_checkpoint=resume_checkpoint)

    # Save
    print(f"\nSaving model to {outdir}")
    trainer.save_model(str(outdir))
    tokenizer.save_pretrained(str(outdir))

    config = {
        "phase": "cot_teacher",
        "model": args.model,
        "lr": args.lr,
        "epochs": args.epochs,
        "train_examples": len(train_dataset),
    }
    with open(outdir / "training_config.json", "w") as f:
        json.dump(config, f, indent=2)

    if args.wandb and WANDB_AVAILABLE:
        wandb.finish()

    print("CoT teacher training complete!")


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: Cache Hidden States
# ─────────────────────────────────────────────────────────────────────────────

def find_answer_colon_position(input_ids: List[int], tokenizer) -> int:
    """Find the token position of the last "Answer:" in the sequence.

    This is the position just before the answer digits — where the model
    has finished its CoT reasoning and is about to output the result.
    We extract the hidden state here as the distillation target.

    Returns the index of the ":" token in the last "Answer:" occurrence.
    """
    # Tokenize "Answer:" to get the pattern
    answer_tokens = tokenizer.encode("Answer:", add_special_tokens=False)
    # Usually this is ["Answer", ":"] — we want the position of ":"

    # Search backwards for the pattern
    ids = list(input_ids)
    pattern_len = len(answer_tokens)

    for start in range(len(ids) - pattern_len, -1, -1):
        if ids[start:start + pattern_len] == answer_tokens:
            # Return position of the last token in the pattern (the ":")
            return start + pattern_len - 1

    # Fallback: search for just ":" near the end
    colon_id = tokenizer.encode(":", add_special_tokens=False)
    if colon_id:
        for pos in range(len(ids) - 1, max(0, len(ids) - 50), -1):
            if ids[pos] == colon_id[0]:
                return pos

    raise ValueError(
        f"Could not find 'Answer:' pattern in sequence of length {len(ids)}. "
        f"Last 20 tokens: {tokenizer.decode(ids[-20:])}"
    )


def cache_hidden_states(args):
    """Load trained CoT teacher and cache hidden states at Answer: position."""

    cache_outdir = pathlib.Path(args.cache_outdir)
    cache_outdir.mkdir(parents=True, exist_ok=True)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load trained model
    print(f"\nLoading trained model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=DTYPE,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()

    if torch.cuda.is_available():
        mem_gb = torch.cuda.memory_allocated() / 1e9
        print(f"GPU memory: {mem_gb:.2f} GB")

    # Load CoT training data (with all columns — we need metadata)
    data_dir = pathlib.Path(args.data_dir)
    dataset = load_cot_dataset(data_dir, "train")
    print(f"Loaded {len(dataset)} CoT examples for caching")

    # Determine which layer(s) to cache
    # -1 = last layer, -2 = second to last, etc.
    cache_layer = args.cache_layer
    print(f"Caching hidden states from layer: {cache_layer}")

    # Get model device
    if hasattr(model, "hf_device_map") and model.hf_device_map:
        first_device = next(iter(model.hf_device_map.values()))
        device = torch.device(
            f"cuda:{first_device}" if isinstance(first_device, int) else first_device
        )
    else:
        device = next(model.parameters()).device

    # Verify answer position detection on first example
    sample = dataset[0]
    sample_pos = find_answer_colon_position(sample["input_ids"], tokenizer)
    context = tokenizer.decode(sample["input_ids"][max(0, sample_pos - 5):sample_pos + 3])
    print(f"Sample Answer: position = {sample_pos}, context: '...{context}...'")
    print(f"  a1={sample.get('a1')}, a2={sample.get('a2')}, answer={sample.get('answer')}")

    # Cache hidden states
    all_hidden_states = []
    all_metadata = []
    errors = 0

    print(f"\nCaching hidden states for {len(dataset)} examples...")
    for i in tqdm(range(len(dataset)), desc="Caching"):
        ex = dataset[i]
        input_ids = ex["input_ids"]

        # Find the Answer: position
        try:
            answer_pos = find_answer_colon_position(input_ids, tokenizer)
        except ValueError as e:
            print(f"  Warning: {e}")
            errors += 1
            continue

        # Forward pass with hidden states
        ids_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

        with torch.no_grad():
            outputs = model(
                input_ids=ids_tensor,
                output_hidden_states=True,
                use_cache=False,
            )

        # Extract hidden state at the Answer: position from specified layer
        # outputs.hidden_states is tuple of (n_layers+1,) tensors [batch, seq, hidden]
        # Index 0 = embedding, 1 = after layer 0, ..., -1 = after last layer
        h = outputs.hidden_states[cache_layer][0, answer_pos].float().cpu().numpy()

        all_hidden_states.append(h)
        all_metadata.append({
            "id": ex.get("id", f"train-{i}"),
            "pair_id": ex.get("pair_id", i),
            "fact1": ex.get("fact1", ""),
            "fact2": ex.get("fact2", ""),
            "a1": ex.get("a1", 0),
            "a2": ex.get("a2", 0),
            "answer": ex.get("answer", 0),
            "answer_pos": answer_pos,
            "seq_len": len(input_ids),
        })

        # Free memory
        del outputs
        if i % 100 == 0:
            torch.cuda.empty_cache()

    print(f"\nCached {len(all_hidden_states)} hidden states ({errors} errors)")

    # Save as single .npz file
    hidden_states_array = np.stack(all_hidden_states)  # [n_examples, hidden_dim]
    print(f"Hidden states shape: {hidden_states_array.shape}")
    print(f"Hidden states size: {hidden_states_array.nbytes / 1e6:.1f} MB")

    np.savez_compressed(
        cache_outdir / "teacher_hidden_states.npz",
        hidden_states=hidden_states_array,
    )

    # Save metadata
    with open(cache_outdir / "teacher_metadata.json", "w") as f:
        json.dump(all_metadata, f, indent=2)

    # Save cache config
    cache_config = {
        "model": args.model,
        "cache_layer": cache_layer,
        "n_examples": len(all_hidden_states),
        "hidden_dim": hidden_states_array.shape[1],
        "dtype": "float32",
    }
    with open(cache_outdir / "cache_config.json", "w") as f:
        json.dump(cache_config, f, indent=2)

    print(f"Saved to {cache_outdir}/")
    print(f"  teacher_hidden_states.npz: {hidden_states_array.shape}")
    print(f"  teacher_metadata.json: {len(all_metadata)} entries")
    print("Done!")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train CoT teacher and cache hidden states for distillation"
    )

    parser.add_argument("--phase", type=str, required=True,
                        choices=["train", "cache"],
                        help="'train' = full fine-tune CoT teacher; "
                             "'cache' = extract and save hidden states")

    # Model
    parser.add_argument("--model", type=str, required=True,
                        help="Model name/path (base model for train, trained model for cache)")
    parser.add_argument("--no-flash-attn", action="store_true")
    parser.add_argument("--no-grad-checkpoint", action="store_true")

    # Data
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Directory containing train.jsonl (with CoT examples)")

    # Training args (phase=train)
    parser.add_argument("--outdir", type=str, default="outputs/cot_teacher",
                        help="Output directory for trained model")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume-from", type=str, default=None)

    # Caching args (phase=cache)
    parser.add_argument("--cache-outdir", type=str, default=None,
                        help="Output directory for cached hidden states "
                             "(default: {outdir}/cached_hidden_states)")
    parser.add_argument("--cache-layer", type=int, default=-1,
                        help="Which layer's hidden state to cache. "
                             "-1 = last layer, -2 = second to last, etc.")

    # Wandb
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="opaque-reasoner")
    parser.add_argument("--wandb-run-name", type=str, default=None)

    args = parser.parse_args()

    # Default cache outdir
    if args.cache_outdir is None:
        args.cache_outdir = str(pathlib.Path(args.outdir) / "cached_hidden_states")

    if args.phase == "train":
        train_teacher(args)
    elif args.phase == "cache":
        cache_hidden_states(args)


if __name__ == "__main__":
    main()