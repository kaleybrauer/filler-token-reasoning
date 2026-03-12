#!/usr/bin/env python3
"""
Train a filler student model with hidden state distillation from CoT teacher.

Initializes from the CoT teacher checkpoint and trains on filler (dot) examples
with a combined loss:
    L_total = L_answer + λ * MSE(student_h, teacher_h)

where student_h is the hidden state at the "Answer:" position in the filler
example, and teacher_h is the cached hidden state from the CoT teacher at the
same position in the CoT version of the same question.

Usage:
    python scripts/train_filler_student.py \
        --model outputs/cot_teacher_14b \
        --data-dir data/datasets/2hop_14b \
        --cache-dir outputs/cot_teacher_14b/cached_hidden_states \
        --outdir outputs/filler_student_14b \
        --lr 1e-5 --epochs 1 \
        --batch-size 1 --grad-accum 32 \
        --lambda-distill 1.0 \
        --wandb --wandb-run-name "filler_student_dots_lambda1"
"""
import argparse
import inspect
import json
import os
import pathlib
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
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
# Answer position detection (shared with train_cot_teacher.py)
# ─────────────────────────────────────────────────────────────────────────────

def find_answer_colon_position(input_ids: List[int], tokenizer) -> int:
    """Find the token position of the last "Answer:" in the sequence.

    For filler examples, this is the position of ":" in "Filler: ... \\nAnswer:"
    — the last token before the model generates the answer digits.

    Returns the index of the ":" token.
    """
    ids = list(input_ids)

    variants = set()
    for text in ["Answer:", " Answer:", "\nAnswer:"]:
        toks = tokenizer.encode(text, add_special_tokens=False)
        if text.startswith((" ", "\n")):
            toks = toks[1:]
        variants.add(tuple(toks))

    for pattern in variants:
        pattern_len = len(pattern)
        pattern_list = list(pattern)
        for start in range(len(ids) - pattern_len, -1, -1):
            if ids[start:start + pattern_len] == pattern_list:
                return start + pattern_len - 1

    raise ValueError(
        f"Could not find 'Answer:' pattern in sequence of length {len(ids)}. "
        f"Last 20 tokens: {tokenizer.decode(ids[-20:])}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────────────────────────────────────

def load_filler_dataset_with_teacher(
    data_dir: pathlib.Path,
    cache_dir: pathlib.Path,
    tokenizer: Any,
    split: str = "train",
) -> Any:
    """Load filler examples and match with cached teacher hidden states.

    Returns a dataset with extra columns:
        teacher_hidden_state: list of floats (the cached vector)
        answer_pos: int (position of "Answer:" in this filler example)
    """
    # Load filler examples
    data_file = data_dir / f"{split}.jsonl"
    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")

    dataset = load_dataset("json", data_files=str(data_file), split="train")
    n_before = len(dataset)

    # Filter to filler examples only
    if "sequence_type" in dataset.column_names:
        dataset = dataset.filter(lambda ex: ex["sequence_type"] != "cot")
        print(f"Filtered {split}: {n_before} → {len(dataset)} filler examples")

    # Load cached teacher hidden states and metadata
    hs_data = np.load(cache_dir / "teacher_hidden_states.npz")
    teacher_hs = hs_data["hidden_states"]  # [n_cot_examples, hidden_dim]
    print(f"Loaded teacher hidden states: {teacher_hs.shape}")

    with open(cache_dir / "teacher_metadata.json") as f:
        teacher_meta = json.load(f)

    # Build pair_id → teacher hidden state mapping
    pair_to_hs = {}
    for i, meta in enumerate(teacher_meta):
        pid = meta["pair_id"]
        pair_to_hs[pid] = teacher_hs[i]

    print(f"Teacher hidden states cover {len(pair_to_hs)} unique pair_ids")

    # Find answer positions for each filler example
    print("Finding Answer: positions for filler examples...")
    answer_positions = []
    errors = 0
    for i in range(len(dataset)):
        try:
            pos = find_answer_colon_position(
                dataset[i]["input_ids"], tokenizer
            )
            answer_positions.append(pos)
        except ValueError as e:
            print(f"  Warning example {i}: {e}")
            answer_positions.append(-1)
            errors += 1
    if errors:
        print(f"  {errors} examples had position detection errors")

    # Match filler examples to teacher hidden states by pair_id
    matched_hs = []
    matched_pos = []
    keep_indices = []

    has_pair_id = "pair_id" in dataset.column_names

    for i in range(len(dataset)):
        if answer_positions[i] < 0:
            continue  # skip examples with position detection errors

        if has_pair_id:
            pid = dataset[i]["pair_id"]
        else:
            # Fallback: try to match by index
            pid = i

        if pid in pair_to_hs:
            matched_hs.append(pair_to_hs[pid].tolist())
            matched_pos.append(answer_positions[i])
            keep_indices.append(i)

    print(f"Matched {len(keep_indices)} filler examples to teacher hidden states "
          f"(out of {len(dataset)} filler examples)")

    if len(keep_indices) == 0:
        raise RuntimeError(
            "No filler examples matched to teacher hidden states! "
            "Check that pair_ids are consistent between CoT and filler data."
        )

    # Filter dataset to matched examples only
    dataset = dataset.select(keep_indices)

    # Add teacher hidden states and answer positions as columns
    dataset = dataset.add_column("teacher_hidden_state", matched_hs)
    dataset = dataset.add_column("answer_pos", matched_pos)

    return dataset


# ─────────────────────────────────────────────────────────────────────────────
# Custom Data Collator
# ─────────────────────────────────────────────────────────────────────────────

class DistillationDataCollator:
    """Pads sequences and batches teacher hidden states + answer positions."""

    def __init__(self, pad_token_id: int, padding_side: str = "right"):
        self.pad_token_id = pad_token_id
        self.padding_side = padding_side

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)

        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []
        batch_teacher_hs = []
        batch_answer_pos = []

        for f in features:
            seq_len = len(f["input_ids"])
            pad_len = max_len - seq_len

            if self.padding_side == "right":
                batch_input_ids.append(f["input_ids"] + [self.pad_token_id] * pad_len)
                batch_attention_mask.append(f["attention_mask"] + [0] * pad_len)
                batch_labels.append(f["labels"] + [-100] * pad_len)
            else:
                batch_input_ids.append([self.pad_token_id] * pad_len + f["input_ids"])
                batch_attention_mask.append([0] * pad_len + f["attention_mask"])
                batch_labels.append([-100] * pad_len + f["labels"])

            batch_teacher_hs.append(f["teacher_hidden_state"])
            # Adjust answer_pos for left-padding
            pos = f["answer_pos"]
            if self.padding_side == "left":
                pos += pad_len
            batch_answer_pos.append(pos)

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
            "teacher_hidden_states": torch.tensor(batch_teacher_hs, dtype=torch.float32),
            "answer_positions": torch.tensor(batch_answer_pos, dtype=torch.long),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Custom Trainer with Distillation Loss
# ─────────────────────────────────────────────────────────────────────────────

class DistillationTrainer(Trainer):
    """HuggingFace Trainer with added hidden state distillation loss.

    Uses a forward hook to capture a single layer's hidden state, avoiding the
    memory cost of output_hidden_states=True (which stores all layers).
    """

    def __init__(self, *args, lambda_distill: float = 1.0,
                 distill_layer: int = -1, **kwargs):
        super().__init__(*args, **kwargs)
        self.lambda_distill = lambda_distill
        self.distill_layer = distill_layer
        self._captured_hidden = None
        self._hook_handle = None
        self._last_logged_step = -1

    def _get_target_module(self, model):
        """Get the transformer layer module to hook into."""
        # Unwrap DDP / FSDP wrappers
        base = model.module if hasattr(model, "module") else model
        # Qwen2.5 uses model.layers; adjust if architecture differs
        layers = base.model.layers
        # Convert negative index to positive
        idx = self.distill_layer if self.distill_layer >= 0 else len(layers) + self.distill_layer
        return layers[idx]

    def _register_hook(self, model):
        """Register a forward hook on the target layer to capture its output."""
        if self._hook_handle is not None:
            return  # Already registered

        target = self._get_target_module(model)

        def hook_fn(module, input, output):
            # Qwen2 layers return (hidden_states, ...) tuple
            hidden = output[0] if isinstance(output, tuple) else output
            self._captured_hidden = hidden

        self._hook_handle = target.register_forward_hook(hook_fn)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Pop distillation-specific inputs before passing to model
        teacher_hs = inputs.pop("teacher_hidden_states")  # [batch, hidden_dim]
        answer_pos = inputs.pop("answer_positions")  # [batch]

        # Register hook on first call
        self._register_hook(model)

        # Forward pass — no output_hidden_states needed
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=inputs["labels"],
        )

        # Standard language modeling loss (on answer tokens only, per labels mask)
        loss_answer = outputs.loss

        # Skip distillation during eval
        if not model.training:
            if return_outputs:
                return loss_answer, outputs
            return loss_answer

        # Extract student hidden states at answer positions from the hook
        hidden = self._captured_hidden  # [batch, seq, hidden]
        self._captured_hidden = None  # Free reference

        # Gather hidden states at each example's answer position
        # answer_pos: [batch] → need to index hidden[b, answer_pos[b], :]
        idx = answer_pos.unsqueeze(-1).unsqueeze(-1)  # [batch, 1, 1]
        idx = idx.expand(-1, -1, hidden.shape[-1])  # [batch, 1, hidden_dim]
        student_hs = hidden.gather(1, idx).squeeze(1)  # [batch, hidden_dim]

        # Distillation loss: MSE between student and teacher hidden states
        teacher_hs = teacher_hs.to(student_hs.device, dtype=student_hs.dtype)
        loss_distill = F.mse_loss(student_hs, teacher_hs)

        # Combined loss
        loss_total = loss_answer + self.lambda_distill * loss_distill

        # Log individual losses (once per global step, not per micro-batch)
        if (self.state.global_step > self._last_logged_step
                and self.state.global_step % self.args.logging_steps == 0):
            self._last_logged_step = self.state.global_step
            self.log({
                "loss_answer": loss_answer.item(),
                "loss_distill": loss_distill.item(),
                "loss_total": loss_total.item(),
            })

        if return_outputs:
            return loss_total, outputs
        return loss_total


# ─────────────────────────────────────────────────────────────────────────────
# Main Training Function
# ─────────────────────────────────────────────────────────────────────────────

def train_student(args):
    """Train filler student with hidden state distillation."""

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model from CoT teacher checkpoint — full fine-tuning continues
    print(f"\nLoading model from CoT teacher: {args.model}")
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

    # Gradient checkpointing
    if not args.no_grad_checkpoint:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        print("Gradient checkpointing enabled")

    if torch.cuda.is_available():
        mem_gb = torch.cuda.memory_allocated() / 1e9
        print(f"GPU memory after model load: {mem_gb:.2f} GB")

    # Load data with teacher hidden states
    data_dir = pathlib.Path(args.data_dir)
    cache_dir = pathlib.Path(args.cache_dir)

    train_dataset = load_filler_dataset_with_teacher(
        data_dir, cache_dir, tokenizer, split="train"
    )

    # Load eval data (filler only, no teacher matching needed for eval loss)
    eval_dataset = None
    if (data_dir / "val.jsonl").exists():
        val_data = load_dataset(
            "json", data_files=str(data_dir / "val.jsonl"), split="train"
        )
        if "sequence_type" in val_data.column_names:
            val_data = val_data.filter(lambda ex: ex["sequence_type"] != "cot")

        # Eval needs the same columns — add dummy teacher_hs and answer_pos
        # (they won't be used since eval only computes LM loss)
        hidden_dim = len(train_dataset[0]["teacher_hidden_state"])
        dummy_hs = [[0.0] * hidden_dim] * len(val_data)
        dummy_pos = [0] * len(val_data)

        # Find real answer positions for val
        for i in range(len(val_data)):
            try:
                pos = find_answer_colon_position(
                    val_data[i]["input_ids"], tokenizer
                )
                dummy_pos[i] = pos
            except ValueError:
                dummy_pos[i] = 0

        val_data = val_data.add_column("teacher_hidden_state", dummy_hs)
        val_data = val_data.add_column("answer_pos", dummy_pos)

        keep_cols = {"input_ids", "attention_mask", "labels",
                     "teacher_hidden_state", "answer_pos"}
        remove_cols = [c for c in val_data.column_names if c not in keep_cols]
        if remove_cols:
            val_data = val_data.remove_columns(remove_cols)
        eval_dataset = val_data

    # Strip metadata columns from training set (keep only what's needed)
    keep_cols = {"input_ids", "attention_mask", "labels",
                 "teacher_hidden_state", "answer_pos"}
    remove_cols = [c for c in train_dataset.column_names if c not in keep_cols]
    if remove_cols:
        train_dataset = train_dataset.remove_columns(remove_cols)

    print(f"\nTraining: {len(train_dataset)} filler examples with distillation")
    if eval_dataset:
        print(f"Eval: {len(eval_dataset)} filler examples")

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
        run_name = args.wandb_run_name or f"filler_student_{args.lr}"
        ta_kwargs["run_name"] = run_name
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={
                "phase": "filler_student",
                "model": args.model,
                "lr": args.lr,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "grad_accum": args.grad_accum,
                "effective_batch_size": args.batch_size * args.grad_accum,
                "lambda_distill": args.lambda_distill,
                "distill_layer": args.distill_layer,
                "train_examples": len(train_dataset),
                "filler_type": "dots",
            },
        )
    else:
        ta_kwargs["report_to"] = "none"

    training_args = TrainingArguments(**ta_kwargs)

    # Data collator
    data_collator = DistillationDataCollator(
        pad_token_id=tokenizer.pad_token_id,
        padding_side=tokenizer.padding_side,
    )

    # Custom trainer with distillation
    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        lambda_distill=args.lambda_distill,
        distill_layer=args.distill_layer,
    )
    tr_sig = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in tr_sig:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = DistillationTrainer(**trainer_kwargs)

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
        mem_alloc = torch.cuda.memory_allocated() / 1e9
        mem_reserved = torch.cuda.memory_reserved() / 1e9
        print(f"\nMemory before training: {mem_alloc:.2f} GB allocated, "
              f"{mem_reserved:.2f} GB reserved")

    print(f"\nStarting filler student training...")
    print(f"  λ_distill = {args.lambda_distill}")
    print(f"  Distill layer = {args.distill_layer}")
    trainer.train(resume_from_checkpoint=resume_checkpoint)

    # Save
    print(f"\nSaving model to {outdir}")
    trainer.save_model(str(outdir))
    tokenizer.save_pretrained(str(outdir))

    config = {
        "phase": "filler_student",
        "teacher_model": args.model,
        "lr": args.lr,
        "epochs": args.epochs,
        "lambda_distill": args.lambda_distill,
        "distill_layer": args.distill_layer,
        "train_examples": len(train_dataset),
    }
    with open(outdir / "training_config.json", "w") as f:
        json.dump(config, f, indent=2)

    if args.wandb and WANDB_AVAILABLE:
        wandb.finish()

    print("Filler student training complete!")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train filler student with hidden state distillation"
    )

    # Model
    parser.add_argument("--model", type=str, required=True,
                        help="Path to trained CoT teacher checkpoint")
    parser.add_argument("--no-flash-attn", action="store_true")
    parser.add_argument("--no-grad-checkpoint", action="store_true")

    # Data
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Directory containing train.jsonl with filler examples")
    parser.add_argument("--cache-dir", type=str, required=True,
                        help="Directory containing cached teacher hidden states "
                             "(teacher_hidden_states.npz + teacher_metadata.json)")

    # Output
    parser.add_argument("--outdir", type=str, default="outputs/filler_student",
                        help="Output directory for trained student model")

    # Training hyperparameters
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

    # Distillation
    parser.add_argument("--lambda-distill", type=float, default=1.0,
                        help="Weight of the distillation loss (default: 1.0)")
    parser.add_argument("--distill-layer", type=int, default=-1,
                        help="Which layer to extract student hidden states from. "
                             "Must match --cache-layer used during teacher caching. "
                             "-1 = last layer.")

    # Wandb
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="opaque-reasoner")
    parser.add_argument("--wandb-run-name", type=str, default=None)

    args = parser.parse_args()
    train_student(args)


if __name__ == "__main__":
    main()