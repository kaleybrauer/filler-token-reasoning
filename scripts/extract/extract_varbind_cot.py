"""
extract_varbind_cot.py

Hidden-state extraction for the two varbind chain-of-thought (think-out-loud)
conditions, used to test whether the internal depth ladder seen in the
filler/baseline conditions (x@L33 -> c1*x@L38 -> y@L44 -> c2*y@L51 -> answer@L60,
all in ONE forward pass) survives when the model VERBALISES the chain across
token positions.

Two modes:

  teacher_forced
      The canonical reasoning (build_cot_reasoning_varbind) is supplied as the
      assistant turn. We forward-pass the full sequence and extract every token
      in the assistant CoT region. Because the reasoning is deterministic, the
      token position of every intermediate (x, c1*x, y, c2*y, answer) is known
      exactly (value_positions in the saved pkl). This is the clean, controlled
      comparison to the filler ladder.

  free_gen
      The prompt stops at the target user turn (the few-shot demos prime the
      "Thinking: ...\nAnswer: N" format). The model generates its OWN reasoning;
      we forward-pass [prompt + generated] and extract every generated token.
      Intermediate positions are recovered by scanning the generated tokens for
      the true values; the model's emitted answer + correctness are recorded.

Why a separate script (not a branch in extract_hidden_states.py): the filler
path keys everything off find_filler_boundaries, which assumes the
"Filler: <content>\n\nAnswer:" scaffold. CoT has no such scaffold — the region
of interest is the assistant turn itself, located by chat-template alignment.

The pure helpers (cot_value_targets, locate_value_positions,
build_teacher_forced_ids, build_free_gen_prefix) deliberately avoid importing
torch so they can be unit-tested on a CPU-only box with the lite tokenizer.
torch / load_model / HiddenStateExtractor are imported lazily inside the GPU
functions.

Usage (GPU):
    python scripts/extract/extract_varbind_cot.py \
        --mode teacher_forced \
        --dataset data/chained_var_binding_dataset.json \
        --output-dir data/extracted_states_varbind_cot_tf \
        --max-problems 200

    python scripts/extract/extract_varbind_cot.py \
        --mode free_gen \
        --dataset data/chained_var_binding_dataset.json \
        --output-dir data/extracted_states_varbind_cot_free \
        --max-problems 200 --max-new-tokens 96
"""

import argparse
import json
import pickle
import re
import sys
import time
from pathlib import Path

# scripts/ on the path so `from data.* import ...` and `from extract.* import ...`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Torch-free prompt construction (safe to import on a CPU box).
from data.generate_varbind_dataset import (  # noqa: E402
    build_cot_messages_varbind,
    build_cot_reasoning_varbind,
    build_cot_assistant_turn_varbind,
    _resolve_chain_varbind,
)

# Canonical label order for the five decode targets (paper notation x/y).
TARGET_LABELS = ["x", "c1x", "y", "c2y", "ans"]
TARGET_DISPLAY = {
    "x": "x (base, visible)",
    "c1x": "c1*x (chain product, hidden)",
    "y": "y (queried value, hidden)",
    "c2y": "c2*y (question product, hidden)",
    "ans": "answer (final)",
}


# ---------------------------------------------------------------------------
# Pure helpers (no torch) — CPU-testable
# ---------------------------------------------------------------------------

def cot_value_targets(problem: dict) -> dict:
    """The five intermediate/visible values for one depth-1 example, as ints."""
    base_name, x, c1, op1, k1 = _resolve_chain_varbind(problem)
    y = problem["queried_value"]
    c2 = problem["coefficient"]
    return {"x": x, "c1x": c1 * x, "y": y, "c2y": c2 * y, "ans": problem["answer"]}


def locate_value_positions(token_ids, region_start, tokenizer, targets) -> dict:
    """Map each target label -> sorted list of ABSOLUTE token indices in
    token_ids[region_start:] whose single-token decode (stripped) equals the
    value. Numbers appear space-prefixed in running text ("Ġ93" -> " 93"), so we
    match on the stripped decode, not the no-space token id."""
    decoded = {j: tokenizer.decode([token_ids[j]]).strip()
               for j in range(region_start, len(token_ids))}
    out = {}
    for label in TARGET_LABELS:
        val = str(targets[label])
        out[label] = [j for j in range(region_start, len(token_ids)) if decoded[j] == val]
    return out


def _tok_ids(tokenizer, text):
    """Tokenize a chat-template-rendered string the same way the filler pipeline
    does (let the tokenizer apply its default special-token handling, matching
    extract_hidden_states.py)."""
    enc = tokenizer(text)
    return enc["input_ids"]


def build_teacher_forced_ids(tokenizer, few_shot, problem):
    """Render the teacher-forced sequence and locate the assistant CoT region.

    Returns (full_ids, assistant_start, aligned):
      full_ids        token ids for system + few-shot + user + assistant(reasoning)
      assistant_start index of the first assistant-content token (== len(prefix))
      aligned         True iff full_ids[:assistant_start] == prefix_ids exactly
                      (the standard add_generation_prompt alignment invariant;
                      holds because the assistant turn opens on an atomic special
                      token). Caller should assert this.
    """
    full_msgs = build_cot_messages_varbind(few_shot, problem, teacher_forced=True)
    prefix_msgs = build_cot_messages_varbind(few_shot, problem, teacher_forced=False)
    full_text = tokenizer.apply_chat_template(
        full_msgs, tokenize=False, add_generation_prompt=False)
    prefix_text = tokenizer.apply_chat_template(
        prefix_msgs, tokenize=False, add_generation_prompt=True)
    full_ids = _tok_ids(tokenizer, full_text)
    prefix_ids = _tok_ids(tokenizer, prefix_text)
    assistant_start = len(prefix_ids)
    aligned = full_ids[:assistant_start] == prefix_ids
    return full_ids, assistant_start, aligned


# Assistant turns in the CoT format open with this token. Priming the free-gen
# assistant with it makes the model adopt the compact "Thinking: ...\nAnswer: N"
# format demonstrated by the few-shot examples instead of drifting into a verbose
# markdown style that overflows the generation budget (and never writes the chain
# values). It mirrors the teacher-forced assistant turn, which is
# build_cot_assistant_turn_varbind -> "Thinking: {reasoning}\nAnswer: N".
FREE_GEN_PRIME = "Thinking:"


def build_free_gen_prefix(tokenizer, few_shot, problem, prime=FREE_GEN_PRIME):
    """Render the free-generation prompt (stops at the target user turn, with the
    assistant generation prompt appended) and prime the assistant turn with
    `prime` ("Thinking:").

    Returns (gen_input_text, gen_input_ids, assistant_start, aligned):
      gen_input_text   prompt + generation prompt + prime (fed to model.generate)
      gen_input_ids    its token ids
      assistant_start  index of the first assistant-content token == length of the
                       UNPRIMED generation prompt, so the located/extracted region
                       begins at the primed "Thinking" token, exactly as the
                       teacher-forced region does (offsets are directly comparable).
      aligned          True iff gen_input_ids[:assistant_start] == base_ids — i.e.
                       the prime appends cleanly after the assistant special token
                       with no boundary token merge. Caller should assert this.
    """
    msgs = build_cot_messages_varbind(few_shot, problem, teacher_forced=False)
    base_text = tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True)
    base_ids = _tok_ids(tokenizer, base_text)
    assistant_start = len(base_ids)
    gen_input_text = base_text + prime
    gen_input_ids = _tok_ids(tokenizer, gen_input_text)
    aligned = gen_input_ids[:assistant_start] == base_ids
    return gen_input_text, gen_input_ids, assistant_start, aligned


_ANSWER_RE = re.compile(r"Answer:\s*(-?\d+)")


def parse_generated_answer(generated_text: str):
    """Parse 'Answer: N' from generated CoT; fall back to the last integer."""
    m = _ANSWER_RE.search(generated_text)
    if m:
        return int(m.group(1))
    nums = re.findall(r"-?\d+", generated_text)
    return int(nums[-1]) if nums else None


# ---------------------------------------------------------------------------
# GPU extraction (torch imported lazily)
# ---------------------------------------------------------------------------

def _extract_one_teacher_forced(model, tokenizer, extractor, layer_indices,
                                few_shot, problem, prob_idx):
    import torch
    full_ids, assistant_start, aligned = build_teacher_forced_ids(
        tokenizer, few_shot[:5], problem)
    if not aligned:
        raise RuntimeError(
            f"prob {prob_idx}: teacher-forced prefix/full token mismatch — "
            "the chat template is not prefix-consistent; position indices would "
            "be wrong. Inspect apply_chat_template output before trusting this.")
    targets = cot_value_targets(problem)
    value_positions = locate_value_positions(full_ids, assistant_start, tokenizer, targets)

    # Extract every token in the assistant CoT region (+ the question_end anchor).
    seq_len = len(full_ids)
    positions = {"question_end": assistant_start - 1}
    for i in range(assistant_start, seq_len):
        positions[f"cot_{i - assistant_start:03d}"] = i

    device = next(model.parameters()).device
    input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
    attn_mask = torch.ones_like(input_ids)
    states = extractor.extract(input_ids, attn_mask, positions)

    return {
        "problem_idx": prob_idx,
        "mode": "teacher_forced",
        "answer": problem["answer"],
        "queried_value": problem["queried_value"],
        "coefficient": problem["coefficient"],
        "operation": problem["operation"],
        "constant": problem["constant"],
        "targets": targets,
        "value_positions": value_positions,   # label -> [absolute token idx]
        "assistant_start": assistant_start,
        "seq_len": seq_len,
        "positions": positions,                # name -> absolute token idx
        "states": states,                      # name -> {layer: vec}
        "reasoning": build_cot_reasoning_varbind(problem),
    }


def _extract_one_free_gen(model, tokenizer, extractor, layer_indices,
                          few_shot, problem, prob_idx, max_new_tokens):
    import torch
    gen_input_text, gen_input_ids, assistant_start, aligned = build_free_gen_prefix(
        tokenizer, few_shot[:5], problem)
    if not aligned:
        raise RuntimeError(
            f"prob {prob_idx}: free-gen prime ('{FREE_GEN_PRIME}') does not append "
            "cleanly after the assistant marker (boundary token merge) — position "
            "indices would be wrong. Inspect apply_chat_template output.")
    device = next(model.parameters()).device
    prefix_t = torch.tensor([gen_input_ids], dtype=torch.long, device=device)
    prefix_mask = torch.ones_like(prefix_t)

    with torch.no_grad():
        gen = model.generate(
            prefix_t, attention_mask=prefix_mask,
            max_new_tokens=max_new_tokens, do_sample=False,
        )
    full_ids = gen[0].tolist()
    gen_ids = full_ids[assistant_start:]
    generated_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

    targets = cot_value_targets(problem)
    value_positions = locate_value_positions(full_ids, assistant_start, tokenizer, targets)
    model_answer = parse_generated_answer(generated_text)

    # Forward pass over the full (prompt + generated) sequence, extract every
    # generated token (+ the question_end anchor).
    seq_len = len(full_ids)
    positions = {"question_end": assistant_start - 1}
    for i in range(assistant_start, seq_len):
        positions[f"gen_{i - assistant_start:03d}"] = i

    input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
    attn_mask = torch.ones_like(input_ids)
    states = extractor.extract(input_ids, attn_mask, positions)

    return {
        "problem_idx": prob_idx,
        "mode": "free_gen",
        "answer": problem["answer"],
        "queried_value": problem["queried_value"],
        "coefficient": problem["coefficient"],
        "operation": problem["operation"],
        "constant": problem["constant"],
        "targets": targets,
        "value_positions": value_positions,
        "assistant_start": assistant_start,
        "seq_len": seq_len,
        "positions": positions,
        "states": states,
        "generated_text": generated_text,
        "model_answer": model_answer,
        "model_correct": (model_answer == problem["answer"]),
    }


def run(args):
    import torch  # noqa: F401
    from extract.extract_hidden_states import load_model, HiddenStateExtractor

    with open(args.dataset) as f:
        dataset = json.load(f)
    few_shot = dataset.get("few_shot_examples") or dataset["few_shot_facts"]
    problems = dataset["examples"]
    if args.max_problems:
        problems = problems[:args.max_problems]
    print(f"Loaded {len(problems)} problems, mode={args.mode}")

    num_layers = 61
    if args.layers == "all":
        layer_indices = list(range(num_layers))
    elif args.layers == "every4":
        layer_indices = list(range(0, num_layers, 4))
    else:
        layer_indices = [int(x) for x in args.layers.split(",")]
    print(f"Extracting from {len(layer_indices)} layers")

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model(args.model_path)

    # Save lm_head + final-norm weights once (decode scripts need them); mirrors
    # extract_hidden_states.py so the same data/model_weights/deepseek_v3/ files
    # are reused if already present.
    weights_dir = Path("data/model_weights/deepseek_v3")
    lm_head_path = weights_dir / "lm_head_weight.npy"
    norm_path = weights_dir / "rms_norm_weight.npy"
    if not (lm_head_path.exists() and norm_path.exists()):
        import numpy as np
        weights_dir.mkdir(parents=True, exist_ok=True)
        np.save(lm_head_path,
                model.lm_head.weight.detach().cpu().to(torch.float16).numpy())
        np.save(norm_path,
                model.model.norm.weight.detach().cpu().to(torch.float32).numpy())
        print(f"Saved lm_head + rms_norm to {weights_dir}")

    extractor = HiddenStateExtractor(model, layer_indices)
    extractor.register_hooks()

    n_correct = n_total = 0
    n_aligned = 0
    t0 = time.time()
    for i, problem in enumerate(problems):
        prob_idx = problem.get("idx", i)
        save_path = out_dir / f"prob_{prob_idx:04d}.pkl"
        if save_path.exists() and not args.no_skip_existing:
            continue
        if args.mode == "teacher_forced":
            result = _extract_one_teacher_forced(
                model, tokenizer, extractor, layer_indices, few_shot, problem, prob_idx)
            n_aligned += 1
        else:
            result = _extract_one_free_gen(
                model, tokenizer, extractor, layer_indices, few_shot, problem,
                prob_idx, args.max_new_tokens)
            if result["model_answer"] is not None:
                n_total += 1
                n_correct += int(result["model_correct"])

        with open(save_path, "wb") as f:
            pickle.dump(result, f)

        if i < 3 or i % 50 == 0:
            miss = ", ".join(
                f"{lab}@{result['value_positions'][lab]}" for lab in TARGET_LABELS)
            extra = ""
            if args.mode == "free_gen":
                extra = (f"  gen_answer={result['model_answer']} "
                         f"({'ok' if result['model_correct'] else 'wrong'})")
            print(f"  [{i}] prob {prob_idx}: positions {miss}{extra}")

    extractor.remove_hooks()
    dt = time.time() - t0
    print(f"\nDone in {dt:.0f}s.")
    if args.mode == "free_gen" and n_total:
        print(f"Free-gen accuracy: {n_correct}/{n_total} = {n_correct/n_total:.1%}")
    print(f"Saved to {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["teacher_forced", "free_gen"], required=True)
    ap.add_argument("--model-path", default="/workspace/models/deepseek-v3-awq")
    ap.add_argument("--dataset", type=Path,
                    default=Path("data/chained_var_binding_dataset.json"))
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--layers", default="all", help="'all', 'every4', or comma-separated")
    ap.add_argument("--max-problems", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=96,
                    help="free_gen only: generation budget for the CoT")
    ap.add_argument("--no-skip-existing", action="store_true")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
