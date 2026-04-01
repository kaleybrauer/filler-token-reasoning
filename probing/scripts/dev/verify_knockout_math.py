"""
verify_knockout_math.py

Diagnostic: verify that format_only_k0 and dots_50_ko_corrpos produce
identical hidden states at the answer position. If they don't, find where
the difference enters.

Tests:
1. Print token sequences side by side
2. Compare position_ids
3. Compare hidden states at answer_prompt (last input token) per layer
4. Verify attention weights to filler are exactly 0

Usage:
    source /workspace/config/probing_env.sh
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python probing/scripts/verify_knockout_math.py \
        --model-path /workspace/models/deepseek-v3-awq \
        --n-examples 5
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

import transformers.activations as _act
if not hasattr(_act, "PytorchGELUTanh"):
    _act.PytorchGELUTanh = _act.GELUTanh

sys.path.insert(0, str(Path(__file__).parent))
from extract_hidden_states import (
    build_messages_for_condition,
    find_filler_boundaries,
    load_tokenizer,
)
from extract_attention import load_model_eager
from attention_knockout import (
    build_messages_format_only,
    build_corrected_position_ids,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/workspace/models/deepseek-v3-awq")
    parser.add_argument("--dataset", type=Path,
                        default=Path("probing/data/1hop_addition_dataset.json"))
    parser.add_argument("--n-examples", type=int, default=5)
    parser.add_argument("--filler-k", type=int, default=50)
    parser.add_argument("--check-layers", type=str, default="0,10,20,30,40,50,55,58,60")
    args = parser.parse_args()

    check_layers = [int(x) for x in args.check_layers.split(",")]

    with open(args.dataset) as f:
        dataset = json.load(f)
    few_shot = dataset["few_shot_facts"]
    problems = dataset["examples"]

    model, tokenizer = load_model_eager(args.model_path)
    first_param = next(model.parameters())
    input_device = first_param.device

    # =========================================================================
    # TEST 1: Compare token sequences
    # =========================================================================
    print("\n" + "="*70)
    print("TEST 1: Token sequence comparison")
    print("="*70)

    prob = problems[0]
    rng = random.Random(0)

    # format_only
    fmt_msgs = build_messages_format_only(few_shot[:5], prob, args.filler_k)
    fmt_text = tokenizer.apply_chat_template(fmt_msgs, tokenize=False, add_generation_prompt=True)
    fmt_ids = tokenizer(fmt_text, return_tensors="pt")["input_ids"][0]

    # dots_50
    d50_msgs = build_messages_for_condition(few_shot[:5], prob, "dots", args.filler_k, rng=rng)
    d50_text = tokenizer.apply_chat_template(d50_msgs, tokenize=False, add_generation_prompt=True)
    d50_ids = tokenizer(d50_text, return_tensors="pt")["input_ids"][0]

    print(f"format_only length: {len(fmt_ids)}")
    print(f"dots_50 length:     {len(d50_ids)}")
    print(f"difference:         {len(d50_ids) - len(fmt_ids)} tokens")

    # Find where they diverge
    min_len = min(len(fmt_ids), len(d50_ids))
    first_diff = None
    for i in range(min_len):
        if fmt_ids[i] != d50_ids[i]:
            first_diff = i
            break
    if first_diff is not None:
        print(f"\nFirst difference at position {first_diff}")
        # Show context around divergence
        start = max(0, first_diff - 3)
        end_fmt = min(len(fmt_ids), first_diff + 10)
        end_d50 = min(len(d50_ids), first_diff + 60)
        print(f"  format_only [{start}:{end_fmt}]: {tokenizer.decode(fmt_ids[start:end_fmt])!r}")
        print(f"  dots_50     [{start}:{end_d50}]: {tokenizer.decode(d50_ids[start:end_d50])!r}")

    # Show last 10 tokens of each (around answer area)
    print(f"\nformat_only last 10 tokens: {[tokenizer.decode([t]) for t in fmt_ids[-10:]]}")
    print(f"dots_50 last 10 tokens:     {[tokenizer.decode([t]) for t in d50_ids[-10:]]}")

    # Find filler boundaries in dots_50
    _, filler_start, filler_end = find_filler_boundaries(tokenizer, d50_ids.unsqueeze(0), args.filler_k)
    print(f"\ndots_50 filler boundaries: [{filler_start}, {filler_end}] ({filler_end - filler_start + 1} tokens)")

    # Show tokens around filler
    pre = max(0, filler_start - 3)
    post = min(len(d50_ids), filler_end + 5)
    print(f"dots_50 around filler: {[tokenizer.decode([t]) for t in d50_ids[pre:post]]}")

    # Check: do tokens AFTER filler in dots_50 match the corresponding tokens in format_only?
    n_filler = filler_end - filler_start + 1
    d50_post_filler = d50_ids[filler_end + 1:]
    # In format_only, the corresponding tokens start at filler_start (since there's no filler)
    fmt_post = fmt_ids[filler_start:]
    match_len = min(len(d50_post_filler), len(fmt_post))
    matches = (d50_post_filler[:match_len] == fmt_post[:match_len]).all().item()
    print(f"\nPost-filler tokens match format_only: {matches}")
    if not matches:
        for i in range(match_len):
            if d50_post_filler[i] != fmt_post[i]:
                print(f"  Mismatch at offset {i}: dots_50='{tokenizer.decode([d50_post_filler[i]])}' "
                      f"vs format_only='{tokenizer.decode([fmt_post[i]])}'")
                if i > 3:
                    break

    # =========================================================================
    # TEST 2: Position ID comparison
    # =========================================================================
    print("\n" + "="*70)
    print("TEST 2: Position ID comparison")
    print("="*70)

    corrected_pos = build_corrected_position_ids(
        len(d50_ids), filler_start, filler_end, torch.device("cpu")
    )
    fmt_pos = torch.arange(len(fmt_ids))

    # Post-filler positions in corrected should match format_only positions
    d50_post = corrected_pos[filler_end + 1:]
    fmt_post_pos = fmt_pos[filler_start:]
    pos_match_len = min(len(d50_post), len(fmt_post_pos))
    pos_match = (d50_post[:pos_match_len] == fmt_post_pos[:pos_match_len]).all().item()
    print(f"Post-filler corrected position IDs match format_only: {pos_match}")
    if not pos_match:
        print(f"  dots_50 corrected post-filler: {d50_post[:10].tolist()}")
        print(f"  format_only corresponding:     {fmt_post_pos[:10].tolist()}")

    # Pre-filler positions should be identical
    pre_match = (corrected_pos[:filler_start] == fmt_pos[:filler_start]).all().item()
    print(f"Pre-filler position IDs match: {pre_match}")

    print(f"\nformat_only positions (last 5): {fmt_pos[-5:].tolist()}")
    print(f"dots_50 corrected (last 5):     {corrected_pos[-5:].tolist()}")
    print(f"dots_50 normal (last 5):        {torch.arange(len(d50_ids))[-5:].tolist()}")

    # =========================================================================
    # TEST 3: Hidden state comparison at answer_prompt
    # =========================================================================
    print("\n" + "="*70)
    print("TEST 3: Hidden state comparison (per-layer at last input token)")
    print("="*70)

    # We need to capture hidden states at specific layers for both conditions
    # and compare them.

    captured_states = {"format_only": {}, "ko_corrpos": {}}

    def make_capture_hooks(model, layers, storage_key):
        hooks = []
        for li in layers:
            layer = model.model.layers[li]
            def post_hook(module, input, output, _li=li, _key=storage_key):
                if isinstance(output, tuple):
                    h = output[0]
                else:
                    h = output
                captured_states[_key][_li] = h[0, -1].detach().cpu().float().numpy()
            hooks.append(layer.register_forward_hook(post_hook))
        return hooks

    # Also need knockout + corrpos hooks for the dots_50 condition
    def make_knockout_hooks(model, filler_start, filler_end):
        hooks = []
        for layer in model.model.layers:
            def mask_hook(module, args, kwargs, _fs=filler_start, _fe=filler_end):
                if 'attention_mask' in kwargs and kwargs['attention_mask'] is not None:
                    mask = kwargs['attention_mask']
                    if mask.dim() == 4 and mask.shape[-1] > _fe:
                        mask = mask.clone()
                        neg_inf = torch.finfo(mask.dtype).min
                        mask[:, :, :, _fs:_fe + 1] = neg_inf
                        kwargs['attention_mask'] = mask
                return args, kwargs
            hooks.append(layer.self_attn.register_forward_pre_hook(mask_hook, with_kwargs=True))
        return hooks

    for prob_idx in range(min(args.n_examples, len(problems))):
        prob = problems[prob_idx]
        rng = random.Random(prob_idx)

        # --- format_only ---
        fmt_msgs = build_messages_format_only(few_shot[:5], prob, args.filler_k)
        fmt_text = tokenizer.apply_chat_template(fmt_msgs, tokenize=False, add_generation_prompt=True)
        fmt_inputs = tokenizer(fmt_text, return_tensors="pt")
        fmt_ids_t = fmt_inputs["input_ids"].to(input_device)
        fmt_mask = fmt_inputs["attention_mask"].to(input_device)

        captured_states["format_only"] = {}
        hooks = make_capture_hooks(model, check_layers, "format_only")
        with torch.no_grad():
            model(input_ids=fmt_ids_t, attention_mask=fmt_mask)
        for h in hooks:
            h.remove()

        # --- dots_50 ko_corrpos ---
        d50_msgs = build_messages_for_condition(few_shot[:5], prob, "dots", args.filler_k, rng=rng)
        d50_text = tokenizer.apply_chat_template(d50_msgs, tokenize=False, add_generation_prompt=True)
        d50_inputs = tokenizer(d50_text, return_tensors="pt")
        d50_ids_t = d50_inputs["input_ids"].to(input_device)
        d50_mask = d50_inputs["attention_mask"].to(input_device)

        _, fs, fe = find_filler_boundaries(tokenizer, d50_ids_t, args.filler_k)
        corr_pos = build_corrected_position_ids(d50_ids_t.shape[1], fs, fe, input_device).unsqueeze(0)

        captured_states["ko_corrpos"] = {}
        capture_hooks = make_capture_hooks(model, check_layers, "ko_corrpos")
        ko_hooks = make_knockout_hooks(model, fs, fe)
        with torch.no_grad():
            model(input_ids=d50_ids_t, attention_mask=d50_mask, position_ids=corr_pos)
        for h in capture_hooks + ko_hooks:
            h.remove()

        # Compare
        if prob_idx == 0:
            print(f"\n  {'Layer':<8} {'Cosine':>10} {'L2 dist':>12} {'Max diff':>12}")
            print(f"  {'-'*44}")

        for li in check_layers:
            if li in captured_states["format_only"] and li in captured_states["ko_corrpos"]:
                h_fmt = captured_states["format_only"][li]
                h_ko = captured_states["ko_corrpos"][li]
                cos = np.dot(h_fmt, h_ko) / (np.linalg.norm(h_fmt) * np.linalg.norm(h_ko) + 1e-12)
                l2 = np.linalg.norm(h_fmt - h_ko)
                maxd = np.max(np.abs(h_fmt - h_ko))
                if prob_idx == 0:
                    print(f"  L{li:<6} {cos:>10.6f} {l2:>12.4f} {maxd:>12.6f}")

        del fmt_ids_t, fmt_mask, d50_ids_t, d50_mask
        torch.cuda.empty_cache()

    # =========================================================================
    # TEST 4: Verify attention to filler is zero
    # =========================================================================
    print("\n" + "="*70)
    print("TEST 4: Attention weight verification (ko_corrpos, example 0)")
    print("="*70)

    prob = problems[0]
    rng = random.Random(0)
    d50_msgs = build_messages_for_condition(few_shot[:5], prob, "dots", args.filler_k, rng=rng)
    d50_text = tokenizer.apply_chat_template(d50_msgs, tokenize=False, add_generation_prompt=True)
    d50_inputs = tokenizer(d50_text, return_tensors="pt")
    d50_ids_t = d50_inputs["input_ids"].to(input_device)
    d50_mask = d50_inputs["attention_mask"].to(input_device)

    _, fs, fe = find_filler_boundaries(tokenizer, d50_ids_t, args.filler_k)
    corr_pos = build_corrected_position_ids(d50_ids_t.shape[1], fs, fe, input_device).unsqueeze(0)

    # Capture attention weights at a few layers
    attn_to_filler = {}
    sample_layers = [0, 30, 58, 60]

    def make_attn_capture_hook(layer_idx, fs, fe):
        def hook(module, args, kwargs):
            # First apply knockout
            if 'attention_mask' in kwargs and kwargs['attention_mask'] is not None:
                mask = kwargs['attention_mask']
                if mask.dim() == 4 and mask.shape[-1] > fe:
                    mask = mask.clone()
                    neg_inf = torch.finfo(mask.dtype).min
                    mask[:, :, :, fs:fe + 1] = neg_inf
                    kwargs['attention_mask'] = mask
            return args, kwargs
        return hook

    def make_attn_output_hook(layer_idx, fs, fe):
        def hook(module, input, output):
            # output = (attn_output, attn_weights, past_kv) if output_attentions
            # but by default attn_weights is None
            pass
        return hook

    # Use output_attentions=True for sample layers
    # Actually, let's just check the 4D mask after the hook
    # Simpler: verify that the mask is correctly set
    ko_hooks = make_knockout_hooks(model, fs, fe)

    # Check mask at a specific layer by adding a verification hook
    mask_checks = {}
    verify_hooks = []
    for li in sample_layers:
        layer = model.model.layers[li]
        def verify_hook(module, args, kwargs, _li=li):
            if 'attention_mask' in kwargs and kwargs['attention_mask'] is not None:
                mask = kwargs['attention_mask']
                if mask.dim() == 4:
                    # Check last query position attending to filler columns
                    filler_attn = mask[0, :, -1, fs:fe+1].detach().cpu().float()
                    max_val = filler_attn.max().item()
                    min_val = filler_attn.min().item()
                    mask_checks[_li] = {"max": max_val, "min": min_val,
                                        "all_neginf": (filler_attn < -1e30).all().item()}
            return args, kwargs
        verify_hooks.append(layer.self_attn.register_forward_pre_hook(verify_hook, with_kwargs=True))

    with torch.no_grad():
        model(input_ids=d50_ids_t, attention_mask=d50_mask, position_ids=corr_pos)

    for h in ko_hooks + verify_hooks:
        h.remove()

    for li in sample_layers:
        if li in mask_checks:
            mc = mask_checks[li]
            print(f"  L{li}: filler mask max={mc['max']:.2f}, min={mc['min']:.2f}, "
                  f"all_neginf={mc['all_neginf']}")

    del d50_ids_t, d50_mask
    torch.cuda.empty_cache()

    print("\nDone.")


if __name__ == "__main__":
    main()
