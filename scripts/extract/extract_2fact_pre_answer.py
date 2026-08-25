"""
extract_2fact_pre_answer.py

Capture the TRUE answer position ("pre_answer") for the 2-fact BASELINE (no filler):
the residual stream state whose NEXT token is the answer NUMBER.

Why this script exists: `answer_prompt` (the <|Assistant|> gen-prompt token, the last
INPUT token) is NOT the answer position. With the bare-assistant format the model emits
a whitespace token (or two) FIRST and only then the number, and space-prefixed integers
are not single tokens on V3. So the residual that actually predicts the number sits one
(or more) positions AFTER answer_prompt -- which no 2-fact extraction has captured.

Method per example (greedy, deterministic):
  1. Build the SAME baseline prompt as the original extraction (few_shot[:5], k=0,
     rng=Random(idx)) so pkls line up with data/extracted_states_2fact/baseline by idx.
  2. Greedily generate up to --max-gen tokens; find m = index of the FIRST generated
     token that is a single-token integer in [0, --max-num)  (= the answer's first token).
  3. Forward-pass over prompt + generated[:m] and capture the residual at the LAST
     position (= P+m-1, whose next token is that number) across all 61 layers. That is
     `pre_answer`. Also capture question_end (P-2) and answer_prompt (P-1) from the same
     pass -- their values should reproduce the original baseline extraction (cross-check).

Output schema matches scripts/analysis/decode_2fact_baseline_heatmap.py
(states[pos][layer] fp16, fact_value_1/2, answer, model_correct), with three positions:
question_end, answer_prompt, pre_answer. Writes a NEW dir (does not touch the existing
baseline extraction).

GPU only (DeepSeek V3 AWQ, 3xH200). ~45-60 min for 500 examples incl. model load.
RUN DETACHED -- see the command in the handoff / docstring at the bottom.
"""

import argparse
import json
import pickle
import random
import re
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm

# scripts/extract on path so `from extract_hidden_states import ...` resolves; that
# module in turn puts scripts/ on the path for its `from data.<gen> import ...`.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_hidden_states import (  # noqa: E402
    load_model,
    build_messages_for_condition,
    problem_metadata,
    HiddenStateExtractor,
)


def build_number_token_ids(tokenizer, max_num=300):
    """{token_id: int_value} for integers in [0, max_num) that are a SINGLE token."""
    out = {}
    for v in range(max_num):
        ids = tokenizer.encode(str(v), add_special_tokens=False)
        if len(ids) == 1:
            out[ids[0]] = v
    return out


def parse_answer(text):
    """Same precedence as extract_hidden_states.get_model_answer (numeric tasks)."""
    m = re.search(r"Answer:\s*(-?\d+)", text)
    if m:
        return int(m.group(1))
    m = re.search(r"(-?\d+)", text)
    return int(m.group(1)) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="/workspace/models/deepseek-v3-awq")
    ap.add_argument("--dataset", type=Path, default=Path("data/2fact_addition_dataset.json"))
    ap.add_argument("--output-dir", type=Path,
                    default=Path("data/extracted_states_2fact_preanswer"))
    ap.add_argument("--max-problems", type=int, default=500,
                    help="Match the existing baseline extraction (idx 0..499)")
    ap.add_argument("--max-gen", type=int, default=8,
                    help="Greedy tokens generated while locating the number (baseline answers "
                         "at token 0, so a small budget suffices)")
    ap.add_argument("--max-num", type=int, default=300)
    ap.add_argument("--no-skip-existing", action="store_true")
    args = ap.parse_args()

    cond = "baseline"
    cond_dir = args.output_dir / cond
    cond_dir.mkdir(parents=True, exist_ok=True)

    with open(args.dataset) as f:
        dataset = json.load(f)
    few_shot = dataset.get("few_shot_facts") or dataset["few_shot_examples"]
    problems = dataset["examples"][:args.max_problems]
    print(f"Loaded {len(problems)} problems, {len(few_shot)} few-shot facts")

    model, tokenizer = load_model(args.model_path)
    layer_indices = list(range(61))
    extractor = HiddenStateExtractor(model, layer_indices)
    extractor.register_hooks()
    device = next(model.parameters()).device

    num_map = build_number_token_ids(tokenizer, args.max_num)
    num_ids = set(num_map.keys())
    print(f"Number tokens (single-token ints in [0,{args.max_num})): {len(num_ids)}")

    # metadata.json (mirror the existing baseline extraction)
    meta = []
    for i, p in enumerate(problems):
        e = {"idx": i, **problem_metadata(p, "2fact")}
        e["fact_phrase_1"] = p["fact_phrase_1"]
        e["fact_phrase_2"] = p["fact_phrase_2"]
        meta.append(e)
    with open(args.output_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    correct = total = no_number = 0
    m_hist = {}
    t0 = time.time()

    for idx, problem in enumerate(tqdm(problems, desc="baseline pre_answer")):
        save_path = cond_dir / f"prob_{idx:04d}.pkl"
        if save_path.exists() and not args.no_skip_existing:
            continue

        rng = random.Random(idx)
        messages = build_messages_for_condition(
            few_shot[:5], problem, "dots", 0, rng=rng, dataset_type="2fact"
        )
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        enc = tokenizer(text, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        attn = enc["attention_mask"].to(device)
        P = input_ids.shape[1]

        # 1 greedy generation; reuse it for BOTH correctness and locating the number
        with torch.no_grad():
            gen = model.generate(
                input_ids, attention_mask=attn,
                max_new_tokens=args.max_gen, do_sample=False,
            )
        gen_ids = gen[0, P:].tolist()
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        model_answer = parse_answer(gen_text)
        is_correct = (model_answer == problem["answer"])
        correct += int(is_correct)
        total += 1

        # first single-token integer = the answer's first token
        m = next((j for j, tid in enumerate(gen_ids) if tid in num_ids), None)
        if m is None:
            no_number += 1
            m_eff = 0  # fall back: pre_answer == answer_prompt
        else:
            m_eff = m
            m_hist[m] = m_hist.get(m, 0) + 1

        # forward over prompt + any token(s) before the number; capture
        if m_eff > 0:
            full = torch.cat([input_ids[0], gen[0, P:P + m_eff]]).unsqueeze(0)
        else:
            full = input_ids
        full_mask = torch.ones_like(full)
        # True question end = the LAST token containing "?" (the target's "...?\n\n").
        # P-2 was the ":" of "Answer:" -- a bug. Baseline has no "Filler:" scaffold,
        # so the tail is  "?\n\n" "Answer" ":" "<|Assistant|>".
        ids_list = input_ids[0].tolist()
        q_end = next((i for i in range(len(ids_list) - 1, -1, -1)
                      if "?" in tokenizer.decode([ids_list[i]])), P - 2)
        positions = {
            "question_end": q_end,
            "pre_answer": full.shape[1] - 1,   # position whose next token is the answer number
        }
        states = extractor.extract(full, full_mask, positions)

        result = {
            "problem_idx": idx, "condition": cond, "k": 0,
            **problem_metadata(problem, "2fact"),
            "positions": positions, "states": states,
            "model_response": gen_text, "model_answer": model_answer,
            "model_correct": is_correct,
            "seq_len": P, "filler_start": None, "filler_end": None,
            "gen_prefix": gen_text,
            # diagnostics specific to this extraction:
            "n_tokens_before_number": m,  # None if no number found within max_gen
            "pre_answer_token_value": (num_map[gen_ids[m]] if m is not None else None),
        }
        with open(save_path, "wb") as fp:
            pickle.dump(result, fp)

        del input_ids, attn, gen, full
        torch.cuda.empty_cache()

    extractor.remove_hooks()
    dt = time.time() - t0
    if total:
        print(f"\nAccuracy: {correct}/{total} = {correct / total:.1%}")
    print(f"no-number examples (fell back to answer_prompt): {no_number}")
    print(f"whitespace-tokens-before-number histogram (m -> count): "
          f"{dict(sorted(m_hist.items()))}")
    print(f"Time: {dt:.0f}s ({dt / max(len(problems), 1):.1f}s/example)")
    print(f"Saved to {cond_dir}")


if __name__ == "__main__":
    main()
