"""
fetch_qwen35_unembedding.py — the Qwen3.5 unembedding for CPU analysis before (and as a cross-check of)
the GPU export in extract_qwen35_states.py.

Reads only lm_head.weight and the text model's final norm weight out of the Hub safetensors shards with
HTTP range requests (~2 GB instead of ~20 GiB of shards). Writes, per model, the same files the GPU step
writes: lm_head_weight.npy (fp16) and rms_norm_weight.npy, the norm's EFFECTIVE multiplier. Qwen3.5's
RMSNorm computes (1 + weight) * norm(x) (modeling_qwen3_5_moe.Qwen3_5MoeRMSNorm); a stored weight centred
near 0 rather than 1 confirms the offset convention on the real checkpoint.

    python scripts/jlens/fetch_qwen35_unembedding.py
"""
from __future__ import annotations

import json, struct
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
MODELS = {"qwen35_122b": "Qwen/Qwen3.5-122B-A10B", "qwen35_397b_fp8": "Qwen/Qwen3.5-397B-A17B-FP8"}
TENSORS = {"lm_head": "lm_head.weight", "norm": "model.language_model.norm.weight"}
DTYPES = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}


def read_tensor(fs, repo, shard, name):
    with fs.open(f"{repo}/{shard}", "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        meta = json.loads(f.read(n))[name]
        a, b = meta["data_offsets"]
        f.seek(8 + n + a)
        buf = bytearray(f.read(b - a))
    return torch.frombuffer(buf, dtype=DTYPES[meta["dtype"]]).reshape(meta["shape"]), meta["dtype"]


def main():
    from huggingface_hub import HfFileSystem, hf_hub_download
    fs = HfFileSystem()
    for tag, repo in MODELS.items():
        out = REPO / f"outputs/jlens/qwen35/{tag}/unembed_hub"
        out.mkdir(parents=True, exist_ok=True)
        idx = json.loads(Path(hf_hub_download(repo, "model.safetensors.index.json")).read_text())["weight_map"]
        cfg = json.loads(Path(hf_hub_download(repo, "config.json")).read_text())
        W, wdt = read_tensor(fs, repo, idx[TENSORS["lm_head"]], TENSORS["lm_head"])
        g, gdt = read_tensor(fs, repo, idx[TENSORS["norm"]], TENSORS["norm"])
        g = g.float()
        np.save(out / "lm_head_weight.npy", W.to(torch.float16).numpy())
        np.save(out / "rms_norm_weight.npy", (1.0 + g).numpy().astype(np.float32))
        info = {"repo": repo, "lm_head": {"shape": list(W.shape), "stored_dtype": wdt},
                "norm": {"stored_dtype": gdt, "stored_weight_mean": float(g.mean()), "stored_weight_abs_mean": float(g.abs().mean()),
                         "effective_multiplier_mean": float((1 + g).mean())},
                "rms_norm_eps": cfg["text_config"]["rms_norm_eps"], "convention": "1+w",
                "n_layers": cfg["text_config"]["num_hidden_layers"], "d_model": cfg["text_config"]["hidden_size"]}
        (out / "info.json").write_text(json.dumps(info, indent=1))
        print(tag, info, flush=True)


if __name__ == "__main__":
    main()
