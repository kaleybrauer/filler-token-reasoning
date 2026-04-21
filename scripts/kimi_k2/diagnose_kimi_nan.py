"""
Diagnose Kimi K2 AWQ NaN via forward hooks on block 1.
"""
from __future__ import annotations

import torch

import transformers.activations as _act
if not hasattr(_act, "PytorchGELUTanh"):
    _act.PytorchGELUTanh = _act.GELUTanh

from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "/workspace/models/kimi-k2-awq"
N_GPUS = torch.cuda.device_count()


def stats(name, t):
    if isinstance(t, tuple):
        t = t[0]
    t32 = t.detach().to(torch.float32)
    n_nan = torch.isnan(t32).sum().item()
    n_inf = torch.isinf(t32).sum().item()
    finite = t32[torch.isfinite(t32)]
    if finite.numel() == 0:
        print(f"  {name}: ALL NON-FINITE nan={n_nan} inf={n_inf} shape={tuple(t.shape)} dtype={t.dtype}")
        return
    print(f"  {name}: nan={n_nan} inf={n_inf} "
          f"min={finite.min().item():.3g} max={finite.max().item():.3g} "
          f"absmax={finite.abs().max().item():.3g} "
          f"shape={tuple(t.shape)} dtype={t.dtype}")


def main():
    print(f"Loading on {N_GPUS} GPUs...")
    max_memory = {
        i: f"{int(torch.cuda.get_device_properties(i).total_memory * 0.85 / 1e9)}GiB"
        for i in range(N_GPUS)
    }
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        max_memory=max_memory,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()
    print("Loaded")

    text = "The capital of France is"
    ids = tokenizer(text, return_tensors="pt")["input_ids"].to(next(model.parameters()).device)

    # Hook block 1 submodules. We use forward hooks so the data is captured
    # during the actual model.forward() call (which provides valid causal mask).
    block1 = model.model.layers[1]
    captured = {}

    def make_hook(name):
        def hook(mod, inp, output):
            captured[name] = output
        return hook

    handles = []
    handles.append(block1.input_layernorm.register_forward_hook(make_hook("input_layernorm")))
    handles.append(block1.self_attn.register_forward_hook(make_hook("self_attn")))
    handles.append(block1.post_attention_layernorm.register_forward_hook(make_hook("post_attn_ln")))
    handles.append(block1.mlp.register_forward_hook(make_hook("mlp (moe)")))
    handles.append(block1.mlp.gate.register_forward_hook(make_hook("mlp.gate")))
    handles.append(block1.mlp.shared_experts.register_forward_hook(make_hook("mlp.shared_experts")))
    # Sample 3 expert outputs
    for eid in [0, 100, 200]:
        handles.append(
            block1.mlp.experts[eid].register_forward_hook(make_hook(f"expert[{eid}]"))
        )

    with torch.no_grad():
        out = model(ids, use_cache=False)

    for h in handles:
        h.remove()

    print("\n=== block 1 captured outputs (last-token slice where applicable) ===")
    for name, val in captured.items():
        if isinstance(val, tuple):
            val = val[0]
        # For most: [1, seq, hidden]; for gate: tuple of (topk_idx, topk_weight).
        if val.dim() >= 2 and val.shape[0] == 1:
            stats(name, val[0, -1] if val.dim() == 3 else val[-1])
        else:
            stats(name, val)

    print(f"\nArgmax next: {out.logits[0, -1].argmax().item()} "
          f"(logits all-NaN: {torch.isnan(out.logits[0, -1]).all().item()})")


if __name__ == "__main__":
    main()
