"""Check the first broken post_attention_layernorm weight in detail."""
import torch
from safetensors import safe_open
import json
from pathlib import Path

MODEL_PATH = Path("/workspace/models/kimi-k2-awq")
index = json.loads((MODEL_PATH / "model.safetensors.index.json").read_text())
wm = index["weight_map"]

k = "model.layers.1.post_attention_layernorm.weight"
path = MODEL_PATH / wm[k]
with safe_open(path, framework="pt") as f:
    t = f.get_tensor(k)
print(f"Tensor: {k}")
print(f"  shape: {tuple(t.shape)}, dtype: {t.dtype}")

t32 = t.to(torch.float32)
n_inf = torch.isinf(t32).sum().item()
n_nan = torch.isnan(t32).sum().item()
finite = t32[torch.isfinite(t32)]
print(f"  inf count: {n_inf}")
print(f"  nan count: {n_nan}")
print(f"  finite min: {finite.min().item():.4g}")
print(f"  finite max: {finite.max().item():.4g}")
print(f"  finite mean: {finite.mean().item():.4g}")
print(f"  finite absmax: {finite.abs().max().item():.4g}")

# Look at extreme finite values
sorted_abs = finite.abs().sort(descending=True).values
print(f"\n  Top 10 |w| among finite:")
for v in sorted_abs[:10].tolist():
    print(f"    {v:.4g}")

# Under bf16 representation, bf16_max = 3.3895313892515355e+38
# Inf in bf16 means the stored value was already inf when converted to bf16.
print(f"\n  bf16 max finite: ~3.39e+38")
print(f"  fp16 max finite: ~65504")
print(f"  fp32 max finite: ~3.40e+38")

# Check: would the finite values still overflow fp16?
over_fp16 = (finite.abs() > 65504).sum().item()
print(f"  Finite values > fp16 max: {over_fp16}")
