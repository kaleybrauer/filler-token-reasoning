#!/usr/bin/env python3
"""
LoRA fine-tuning script for opaque reasoner experiments.
Supports pre-tokenized JSONL with mixed filler lengths.
"""
import argparse
import inspect
import json
import pathlib
from typing import Any, Dict, List, Optional
import random
import numpy as np
import torch
from datasets import load_dataset, Dataset, concatenate_datasets
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Optional wandb import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

USE_BF16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
DTYPE = torch.bfloat16 if USE_BF16 else torch.float16


# ─────────────────────────────────────────────────────────────────────────────
# Data Collator
# ─────────────────────────────────────────────────────────────────────────────

class DataCollatorForCausalLM:
    """Pads pre-tokenized sequences for batching."""
    
    def __init__(self, pad_token_id: int, padding_side: str = "right"):
        self.pad_token_id = pad_token_id
        self.padding_side = padding_side

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)
        batch_size = len(features)

        # Pre-allocate tensors
        batch_input_ids = torch.full((batch_size, max_len), self.pad_token_id, dtype=torch.long)
        batch_attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
        batch_labels = torch.full((batch_size, max_len), -100, dtype=torch.long)

        for i, f in enumerate(features):
            ids = torch.tensor(f["input_ids"], dtype=torch.long)
            mask = torch.tensor(f["attention_mask"], dtype=torch.long)
            labels = torch.tensor(f["labels"], dtype=torch.long)
            seq_len = ids.size(0)

            if self.padding_side == "right":
                batch_input_ids[i, :seq_len] = ids
                batch_attention_mask[i, :seq_len] = mask
                batch_labels[i, :seq_len] = labels
            else:
                offset = max_len - seq_len
                batch_input_ids[i, offset:] = ids
                batch_attention_mask[i, offset:] = mask
                batch_labels[i, offset:] = labels

        return {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask,
            "labels": batch_labels,
        }

# ─────────────────────────────────────────────────────────────────────────────
# Dataset Loading with Mixed N Support
# ─────────────────────────────────────────────────────────────────────────────

def load_pretokenized_dataset(
    data_dir: pathlib.Path,
    filler_lengths: Optional[List[int]] = None,
    split: str = "train",
) -> Dataset:
    """
    Load pre-tokenized dataset, optionally filtering/mixing filler lengths.
    
    Args:
        data_dir: Directory containing train.jsonl / val.jsonl or 
                  subdirectories like filler_0/, filler_32/, etc.
        filler_lengths: If provided, only include examples with these filler_len values.
                       If data is in subdirectories, load from those.
        split: "train" or "val"
    
    Returns:
        Dataset with input_ids, attention_mask, labels columns
    """
    filename = f"{split}.jsonl"
    main_file = data_dir / filename
    
    # Check if data is organized by filler length in subdirectories
    subdirs = [d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("filler_")]
    
    if subdirs and filler_lengths is not None:
        # Load from subdirectories
        datasets_to_concat = []
        for n in filler_lengths:
            subdir = data_dir / f"filler_{n}"
            file_path = subdir / filename
            if file_path.exists():
                ds = load_dataset("json", data_files=str(file_path), split="train")
                datasets_to_concat.append(ds)
                print(f"  Loaded {len(ds)} examples from {file_path}")
            else:
                print(f"  Warning: {file_path} not found, skipping")
        
        if not datasets_to_concat:
            raise FileNotFoundError(f"No data files found for filler lengths {filler_lengths}")
        
        dataset = concatenate_datasets(datasets_to_concat)
    
    elif main_file.exists():
        # Load from single file, optionally filter by filler_len
        dataset = load_dataset("json", data_files=str(main_file), split="train")
        
        if filler_lengths is not None:
            filler_set = set(filler_lengths)
            dataset = dataset.filter(
                lambda x: x.get("filler_len", 0) in filler_set,
                desc=f"Filtering to filler_len in {filler_lengths}"
            )
    else:
        raise FileNotFoundError(f"Data file not found: {main_file}")
    
    # Keep only the columns needed for training
    keep_cols = {"input_ids", "attention_mask", "labels"}
    remove_cols = [c for c in dataset.column_names if c not in keep_cols]
    if remove_cols:
        dataset = dataset.remove_columns(remove_cols)
    
    return dataset


# ─────────────────────────────────────────────────────────────────────────────
# Model Loading
# ─────────────────────────────────────────────────────────────────────────────

def load_model_and_tokenizer(
    model_name: str,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
    use_flash_attn: bool = True,
    use_gradient_checkpointing: bool = True,
    trust_remote_code: bool = False,
):
    """Load model and tokenizer with optional quantization."""
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        use_fast=True, 
        trust_remote_code=trust_remote_code,
    )
    
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else "<|endoftext|>"
    
    # Quantization config
    quant_config = None
    if load_in_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=DTYPE,
        )
    elif load_in_8bit:
        quant_config = BitsAndBytesConfig(load_in_8bit=True)
    
    # Model loading kwargs
    model_kwargs = dict(
        torch_dtype=DTYPE,
        quantization_config=quant_config,
        trust_remote_code=trust_remote_code,
    )
    
    # Always use device_map for proper Flash Attention and multi-GPU support
    model_kwargs["device_map"] = "auto"
    
    # Determine attention implementation before loading (don't load multiple times)
    attn_impl = None
    if use_flash_attn:
        # Check if flash_attn is installed
        try:
            import flash_attn
            attn_impl = "flash_attention_2"
            print("Using Flash Attention 2")
        except ImportError:
            # Fall back to SDPA (available in PyTorch 2.0+)
            attn_impl = "sdpa"
            print("Using SDPA (PyTorch native scaled dot-product attention)")
    
    if attn_impl:
        model_kwargs["attn_implementation"] = attn_impl
    
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    
    # Print memory usage after loading
    if torch.cuda.is_available():
        mem_gb = torch.cuda.memory_allocated() / 1e9
        print(f"GPU memory after model load: {mem_gb:.2f} GB")
    
    # Prepare for k-bit training if quantized
    if load_in_4bit or load_in_8bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=use_gradient_checkpointing)
    elif use_gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    
    return model, tokenizer


def apply_lora(
    model,
    r: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
    target_modules: Optional[List[str]] = None,
):
    """Apply LoRA adapters to model."""
    
    if target_modules is None:
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    
    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_int_list(s: str) -> Optional[List[int]]:
    """Parse comma-separated integers: '0,32,128' -> [0, 32, 128]"""
    if not s:
        return None
    return [int(x.strip()) for x in s.split(",")]


def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for opaque reasoner")
    
    # Required
    parser.add_argument("--model", type=str, required=True, help="Base model name or path")
    parser.add_argument("--data-dir", type=str, required=True, help="Directory with train.jsonl/val.jsonl")
    parser.add_argument("--outdir", type=str, required=True, help="Output directory")
    
    # Filler length selection (for mixed N training)
    parser.add_argument(
        "--filler-lengths", type=str, default=None,
        help="Comma-separated filler lengths to include, e.g. '0,32,128,300'. "
             "If not specified, uses all data."
    )
    
    # Quantization
    parser.add_argument("--load-in-4bit", action="store_true", help="Use 4-bit QLoRA")
    parser.add_argument("--load-in-8bit", action="store_true", help="Use 8-bit quantization")
    parser.add_argument("--no-flash-attn", action="store_true", help="Disable Flash Attention 2")
    parser.add_argument("--no-grad-checkpoint", action="store_true", 
                        help="Disable gradient checkpointing (faster for small models)")
    parser.add_argument("--trust-remote-code", action="store_true",
                        help="Allow executing remote code from model repos")
    
    # Training hyperparameters
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Per-device batch size")
    parser.add_argument("--grad-accum", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--max-steps", type=int, default=-1, help="Max steps (overrides epochs if > 0)")
    parser.add_argument("--warmup-ratio", type=float, default=0.03, help="Warmup ratio")
    parser.add_argument("--group-by-length", action="store_true", dest="group_by_length", default=True,
                        help="Group sequences by length to minimize padding (default: True)")
    parser.add_argument("--no-group-by-length", action="store_false", dest="group_by_length",
                        help="Disable length grouping (use random batching)")
    
    # Eval and checkpointing
    parser.add_argument("--eval-steps", type=int, default=200, help="Evaluation frequency")
    parser.add_argument("--save-steps", type=int, default=200, help="Checkpoint frequency")
    parser.add_argument("--logging-steps", type=int, default=10, help="Logging frequency")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # LoRA
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout")
    
    # Performance
    parser.add_argument("--num-workers", type=int, default=0, help="Dataloader workers")
    
    # Weights & Biases
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="opaque-reasoner", help="W&B project name")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="W&B run name (auto-generated if not set)")
    parser.add_argument("--wandb-entity", type=str, default=None, help="W&B entity/team name")
    
    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Parse filler lengths
    filler_lengths = parse_int_list(args.filler_lengths)
    if filler_lengths:
        print(f"Training with filler lengths: {filler_lengths}")
    else:
        print("Training with all available filler lengths")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Load data
    # ─────────────────────────────────────────────────────────────────────────
    
    data_dir = pathlib.Path(args.data_dir)
    
    print("Loading training data...")
    train_dataset = load_pretokenized_dataset(data_dir, filler_lengths, split="train")
    print(f"Training examples: {len(train_dataset)}")
    
    eval_dataset = None
    val_file = data_dir / "val.jsonl"
    if val_file.exists():
        print("Loading validation data...")
        eval_dataset = load_pretokenized_dataset(data_dir, filler_lengths, split="val")
        print(f"Validation examples: {len(eval_dataset)}")
    
    # Compute sequence lengths for length-grouped sampling
    use_length_grouping = args.group_by_length
    
    if use_length_grouping:
        print("Computing sequence lengths for length-grouped batching...")
        # Use column access instead of per-element indexing for speed
        all_input_ids = train_dataset["input_ids"]
        train_lengths = [len(ids) for ids in all_input_ids]
        min_len, max_len = min(train_lengths), max(train_lengths)
        avg_len = sum(train_lengths) / len(train_lengths)
        print(f"  Sequence lengths: min={min_len}, max={max_len}, avg={avg_len:.0f}")
        
        # Add length column for HF Trainer's group_by_length feature
        train_dataset = train_dataset.add_column("length", train_lengths)
        
        if max_len > min_len * 2:
            print(f"  Length grouping enabled (high variance in lengths)")
        else:
            print(f"  Note: Low variance in lengths, grouping may not help much")
    else:
        # Shuffle training data if not using length grouping
        train_dataset = train_dataset.shuffle(seed=args.seed)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Load model
    # ─────────────────────────────────────────────────────────────────────────
    
    print(f"\nLoading model: {args.model}")
    use_grad_ckpt = not args.no_grad_checkpoint
    model, tokenizer = load_model_and_tokenizer(
        args.model,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        use_flash_attn=not args.no_flash_attn,
        use_gradient_checkpointing=use_grad_ckpt,
        trust_remote_code=args.trust_remote_code,
    )
    
    model = apply_lora(
        model,
        r=args.lora_r,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
    )
    
    # ─────────────────────────────────────────────────────────────────────────
    # Training arguments
    # ─────────────────────────────────────────────────────────────────────────
    
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
        gradient_checkpointing=use_grad_ckpt,
        optim="adamw_torch",  # Don't use fused - it pre-allocates more memory
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=False,  # Reduce memory pressure
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        max_grad_norm=1.0,
        group_by_length=use_length_grouping,  # Group similar lengths to minimize padding
        length_column_name="length" if use_length_grouping else None,
    )
    
    # Configure wandb or disable reporting
    if args.wandb:
        if not WANDB_AVAILABLE:
            print("WARNING: --wandb flag set but wandb not installed. Run: pip install wandb")
            ta_kwargs["report_to"] = "none"
        else:
            ta_kwargs["report_to"] = "wandb"
            
            # Generate run name if not provided
            if args.wandb_run_name:
                run_name = args.wandb_run_name
            else:
                # Auto-generate descriptive name
                model_short = args.model.split("/")[-1]
                filler_str = args.filler_lengths if args.filler_lengths else "all"
                run_name = f"{model_short}_r{args.lora_r}_filler{filler_str}"
            
            ta_kwargs["run_name"] = run_name
            
            # Initialize wandb
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=run_name,
                config={
                    "model": args.model,
                    "filler_lengths": filler_lengths,
                    "lora_r": args.lora_r,
                    "lora_alpha": args.lora_alpha,
                    "lora_dropout": args.lora_dropout,
                    "learning_rate": args.lr,
                    "batch_size": args.batch_size,
                    "grad_accum": args.grad_accum,
                    "effective_batch_size": args.batch_size * args.grad_accum,
                    "epochs": args.epochs,
                    "max_steps": args.max_steps,
                    "train_examples": len(train_dataset),
                    "eval_examples": len(eval_dataset) if eval_dataset else 0,
                    "gradient_checkpointing": use_grad_ckpt,
                    "quantization": "4bit" if args.load_in_4bit else ("8bit" if args.load_in_8bit else "none"),
                },
            )
            print(f"Weights & Biases run: {run_name}")
    else:
        ta_kwargs["report_to"] = "none"
    
    if use_grad_ckpt:
        ta_kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
    
    # Handle eval strategy naming
    ta_sig = inspect.signature(TrainingArguments.__init__).parameters
    if eval_dataset is not None:
        ta_kwargs["eval_steps"] = args.eval_steps
        if "eval_strategy" in ta_sig:
            ta_kwargs["eval_strategy"] = "steps"
        else:
            ta_kwargs["evaluation_strategy"] = "steps"
    
    training_args = TrainingArguments(**ta_kwargs)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Trainer
    # ─────────────────────────────────────────────────────────────────────────
    
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
    
    # ─────────────────────────────────────────────────────────────────────────
    # Train
    # ─────────────────────────────────────────────────────────────────────────
    
    # Clear cache and report memory before training
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        mem_alloc = torch.cuda.memory_allocated() / 1e9
        mem_reserved = torch.cuda.memory_reserved() / 1e9
        print(f"\nMemory before training: {mem_alloc:.2f} GB allocated, {mem_reserved:.2f} GB reserved")
    
    print("\nStarting training...")
    trainer.train()
    
    print(f"\nSaving model to {outdir}")
    trainer.save_model(str(outdir))
    tokenizer.save_pretrained(str(outdir))
    
    # Save config
    config = {
        **vars(args),
        "filler_lengths_used": filler_lengths,
        "train_examples": len(train_dataset),
        "eval_examples": len(eval_dataset) if eval_dataset else 0,
    }
    with open(outdir / "training_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # Log model artifact to wandb
    if args.wandb and WANDB_AVAILABLE:
        # Log the adapter as an artifact
        artifact = wandb.Artifact(
            name=f"lora-adapter-{wandb.run.id}",
            type="model",
            description=f"LoRA adapter for {args.model}",
            metadata=config,
        )
        artifact.add_dir(str(outdir))
        wandb.log_artifact(artifact)
        
        # Finish the run
        wandb.finish()
        print("Weights & Biases run finished and artifact uploaded.")
    
    print(f"Done! Adapter saved to: {outdir}")


if __name__ == "__main__":
    main()