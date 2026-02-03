#!/usr/bin/env python3
import argparse
import json
import pathlib
import re
from typing import Any, Dict, List, Optional

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from peft import PeftModel

INT_RE = re.compile(r"-?\d+")

def parse_int(text: str) -> Optional[int]:
    m = INT_RE.search(text)
    if not m:
        return None
    try:
        return int(m.group(0))
    except Exception:
        return None

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True, help="Base model name (e.g. Qwen/Qwen2.5-7B)")
    ap.add_argument("--data-file", type=str, required=True, help="Path to test.jsonl")
    ap.add_argument("--adapter", type=str, default=None, help="Optional LoRA adapter directory")
    ap.add_argument("--load-in-4bit", action="store_true", help="Load base model in 4-bit")
    ap.add_argument("--filler-token", type=str, default="<|fim_pad|>")
    ap.add_argument("--sweep", type=str, default="0,8,32,128,512,1000", help="Comma-separated filler lengths")
    ap.add_argument("--max-new-tokens", type=int, default=8)
    ap.add_argument("--max-examples", type=int, default=1000)
    ap.add_argument("--out-json", type=str, default=None, help="Optional output JSON path for results summary")
    args = ap.parse_args()

    sweep = [int(x.strip()) for x in args.sweep.split(",") if x.strip()]

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token if tok.eos_token is not None else "<|endoftext|>"

    filler_ids = tok.encode(args.filler_token, add_special_tokens=False)
    if len(filler_ids) != 1:
        raise SystemExit(f"filler-token {args.filler_token!r} is not single-token (got {filler_ids})")
    filler_id = filler_ids[0]

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
        torch_dtype=torch.bfloat16,
        quantization_config=quant_cfg,
    )
    if args.adapter is not None:
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    ds = load_dataset("json", data_files={"test": args.data_file})["test"]
    n = min(args.max_examples, len(ds))
    ds = ds.select(range(n))

    device = next(iter(model.parameters())).device

    results: Dict[int, Dict[str, Any]] = {}

    for L in sweep:
        correct = 0
        valid = 0

        for ex in ds:
            prompt_ids: List[int] = ex["prompt_ids"]
            gold: int = ex["answer"]

            input_ids = torch.tensor([prompt_ids + [filler_id] * L], dtype=torch.long, device=device)
            attn = torch.ones_like(input_ids, dtype=torch.long, device=device)

            with torch.no_grad():
                out = model.generate(
                    input_ids=input_ids,
                    attention_mask=attn,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    temperature=0.0,
                    pad_token_id=tok.pad_token_id,
                    eos_token_id=tok.eos_token_id,
                )

            gen_ids = out[0, input_ids.shape[1]:].tolist()
            text = tok.decode(gen_ids, skip_special_tokens=True)
            pred = parse_int(text)

            if pred is not None:
                valid += 1
                if pred == gold:
                    correct += 1

        acc = correct / n
        valid_rate = valid / n
        results[L] = {"accuracy": acc, "valid_rate": valid_rate, "n": n}
        print(f"filler_len={L:4d}  acc={acc:.4f}  valid={valid_rate:.4f}  (n={n})")

    if args.out_json:
        outp = pathlib.Path(args.out_json)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(
            {
                "model": args.model,
                "adapter": args.adapter,
                "filler_token": args.filler_token,
                "filler_token_id": filler_id,
                "sweep": sweep,
                "results": results,
            },
            indent=2
        ), encoding="utf-8")
        print(f"Wrote results to: {outp}")

if __name__ == "__main__":
    main()
