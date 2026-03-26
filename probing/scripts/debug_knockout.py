"""Quick diagnostic: verify the attention knockout hook fires correctly with model.generate + KV cache."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import json
import random
import torch

import transformers.activations as _act
if not hasattr(_act, "PytorchGELUTanh"):
    _act.PytorchGELUTanh = _act.GELUTanh

from extract_hidden_states import build_messages_for_condition, find_filler_boundaries, load_tokenizer
from extract_attention import load_model_eager

model_path = "/workspace/models/deepseek-v3-awq"
with open("probing/data/1hop_addition_dataset.json") as f:
    dataset = json.load(f)
few_shot = dataset["few_shot_facts"]
problem = dataset["examples"][0]

model, tokenizer = load_model_eager(model_path)
device = next(model.parameters()).device

rng = random.Random(0)
messages = build_messages_for_condition(few_shot[:5], problem, "dots", 50, rng=rng)
full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(full_text, return_tensors="pt")
input_ids = inputs["input_ids"].to(device)
attention_mask = inputs["attention_mask"].to(device)
seq_len = input_ids.shape[1]

_, filler_start, filler_end = find_filler_boundaries(tokenizer, input_ids, 50)
print(f"seq_len={seq_len}, filler_start={filler_start}, filler_end={filler_end}")
print(f"Expected answer: {problem['answer']}")

# Track hook calls
hook_log = []

def mask_hook(module, args, kwargs):
    if 'attention_mask' in kwargs and kwargs['attention_mask'] is not None:
        mask = kwargs['attention_mask']
        if mask.dim() == 4:
            q_len = mask.shape[2]
            kv_len = mask.shape[-1]

            # Check filler columns BEFORE modification
            filler_vals_before = mask[0, 0, -1, filler_start:filler_end+1]
            is_already_blocked = (filler_vals_before < -1e30).all().item()

            if kv_len > filler_end:
                mask = mask.clone()
                neg_inf = torch.finfo(mask.dtype).min
                mask[:, :, :, filler_start:filler_end + 1] = neg_inf
                kwargs['attention_mask'] = mask

                # Check AFTER
                filler_vals_after = mask[0, 0, -1, filler_start:filler_end+1]
                is_blocked_after = (filler_vals_after < -1e30).all().item()

                hook_log.append({
                    "q_len": q_len, "kv_len": kv_len,
                    "blocked_before": is_already_blocked,
                    "blocked_after": is_blocked_after,
                })
    return args, kwargs

# Register on layer 0 only (to keep output manageable)
h = model.model.layers[0].self_attn.register_forward_pre_hook(mask_hook, with_kwargs=True)

# Run with KV cache
print("\n--- model.generate with use_cache=True ---")
hook_log.clear()
with torch.no_grad():
    gen_output = model.generate(
        input_ids, attention_mask=attention_mask,
        max_new_tokens=5, do_sample=False, use_cache=True,
    )
h.remove()

new_tokens = gen_output[0, seq_len:]
raw = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
print(f"Generated: {repr(raw)}")
print(f"Hook fired {len(hook_log)} times at layer 0:")
for i, entry in enumerate(hook_log):
    print(f"  step {i}: q_len={entry['q_len']}, kv_len={entry['kv_len']}, "
          f"blocked_before={entry['blocked_before']}, blocked_after={entry['blocked_after']}")

# Now run WITHOUT KV cache for comparison
h = model.model.layers[0].self_attn.register_forward_pre_hook(mask_hook, with_kwargs=True)
print("\n--- manual loop with use_cache=False ---")
hook_log.clear()
current_ids = input_ids.clone()
current_mask = attention_mask.clone()

for step in range(5):
    hook_handles_all = []

    def mask_hook_all(module, args, kwargs):
        if 'attention_mask' in kwargs and kwargs['attention_mask'] is not None:
            mask = kwargs['attention_mask']
            if mask.dim() == 4 and mask.shape[-1] > filler_end:
                mask = mask.clone()
                neg_inf = torch.finfo(mask.dtype).min
                mask[:, :, :, filler_start:filler_end + 1] = neg_inf
                kwargs['attention_mask'] = mask
                if module is model.model.layers[0].self_attn:
                    hook_log.append({
                        "q_len": mask.shape[2], "kv_len": mask.shape[-1],
                        "step": step,
                    })
        return args, kwargs

    for layer in model.model.layers:
        hh = layer.self_attn.register_forward_pre_hook(mask_hook_all, with_kwargs=True)
        hook_handles_all.append(hh)

    with torch.no_grad():
        outputs = model(input_ids=current_ids, attention_mask=current_mask, use_cache=False)

    for hh in hook_handles_all:
        hh.remove()

    next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    if next_token.item() == tokenizer.eos_token_id:
        break

    current_ids = torch.cat([current_ids, next_token], dim=1)
    current_mask = torch.cat([current_mask, torch.ones((1,1), dtype=current_mask.dtype, device=device)], dim=1)

h.remove()
new_tokens = current_ids[0, seq_len:]
raw2 = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
print(f"Generated: {repr(raw2)}")
print(f"Hook fired {len(hook_log)} times at layer 0:")
for entry in hook_log:
    print(f"  step {entry['step']}: q_len={entry['q_len']}, kv_len={entry['kv_len']}")

print(f"\nMatch: {raw == raw2}")
