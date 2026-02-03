#!/usr/bin/env python3
import argparse
import os
import pathlib
from typing import Any, Dict, List

import inspect
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

import platform

class DataCollator:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)

        def pad(seq, pad_value):
            return seq + [pad_value] * (max_len - len(seq))

        input_ids = torch.tensor([pad(f["input_ids"], self.pad_id) for f in features], dtype=torch.long)
        attention_mask = torch.tensor([pad(f["attention_mask"], 0) for f in features], dtype=torch.long)
        labels = torch.tensor([pad(f["labels"], -100) for f in features], dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True, help="Base model name (e.g. Qwen/Qwen2.5-7B)")
    ap.add_argument("--data-dir", type=str, required=True, help="Directory with train.jsonl/val.jsonl")
    ap.add_argument("--outdir", type=str, required=True, help="Output directory for adapter/checkpoints")

    ap.add_argument("--load-in-4bit", action="store_true", help="Use QLoRA (4-bit) via bitsandbytes")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--eval-steps", type=int, default=200)
    ap.add_argument("--save-steps", type=int, default=200)
    ap.add_argument("--logging-steps", type=int, default=20)
    ap.add_argument("--max-steps", type=int, default=-1, help="If >0, overrides epochs")

    # LoRA params
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)

    args = ap.parse_args()

    data_dir = pathlib.Path(args.data_dir)
    train_file = str(data_dir / "train.jsonl")
    val_file = str(data_dir / "val.jsonl")

    ds = load_dataset("json", data_files={"train": train_file, "eval": val_file})
    train_ds = ds["train"]
    eval_ds = ds["eval"]

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    # Qwen-family often has no pad token set; use eos for padding.
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token if tok.eos_token is not None else "<|endoftext|>"

    quant_cfg = None
    if args.load_in_4bit:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        torch_dtype=torch.bfloat16 if platform.system() == "Linux" else torch.float16,
        quantization_config=quant_cfg,
    )

    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # --- TrainingArguments ---
    ta_kwargs = dict(
        output_dir=str(outdir),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        bf16=True,
        fp16=False,
        report_to="none",
        save_total_limit=2,
        remove_unused_columns=False,
        dataloader_num_workers=2,
    )

    ta_sig = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" in ta_sig:
        ta_kwargs["eval_strategy"] = "steps"
    else:
        ta_kwargs["evaluation_strategy"] = "steps"

    targs = TrainingArguments(**ta_kwargs)

    # --- Trainer ---
    trainer_kwargs = dict(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=DataCollator(pad_id=tok.pad_token_id),
    )

    tr_sig = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in tr_sig:
        trainer_kwargs["processing_class"] = tok
    else:
        trainer_kwargs["tokenizer"] = tok

    trainer = Trainer(**trainer_kwargs)


    trainer.train()
    trainer.save_model(str(outdir))
    tok.save_pretrained(str(outdir))
    print(f"Saved adapter/checkpoint to: {outdir}")

if __name__ == "__main__":
    main()
