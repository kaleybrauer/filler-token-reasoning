#!/usr/bin/env python3
import argparse
from transformers import AutoTokenizer

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True, help="Tokenizer model name (e.g. Qwen/Qwen2.5-7B)")
    ap.add_argument("--candidate", type=str, required=True, help='Candidate filler token text, e.g. "<|fim_pad|>"')
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    ids = tok.encode(args.candidate, add_special_tokens=False)
    print(f"candidate: {args.candidate!r}")
    print(f"token_ids: {ids}")
    print(f"num_tokens: {len(ids)}")

    if len(ids) != 1:
        raise SystemExit(
            "ERROR: candidate does not map to exactly 1 token. "
            "Try a different candidate (e.g. <|fim_pad|>, <|fim_middle|>, <|file_sep|>)."
        )

    print(f"OK: single-token filler. filler_token_id={ids[0]}")

if __name__ == "__main__":
    main()
