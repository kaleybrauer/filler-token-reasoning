"""
extract_eval_states.py — cache the residuals the paper's lens-eval sets need, so
lens quality can be scored offline (CPU only) at any fit checkpoint.

`eval_lens_quality.py` loads all 328 GiB of V3 in order to run a few hundred
forward passes. While the fit is running there is no room for a second copy, so
a quality-vs-n curve would mean stopping the fit and paying a 13-minute reload
every time it is measured. But the states the metric actually needs are tiny —
one residual vector per (item, layer) at a single readout position — so extract
them once here and the scorer downstream becomes a matmul.

    551 items x 61 layers x 7168 dims, fp16  ->  ~480 MB

Tokenization and the readout site are taken from jlens itself (HFLensModel.encode
with force_bos, ActivationRecorder on model.layers storing output[0]), so a
cached state is the same object `lens.apply(..., positions=[-1])` would have
selected. The readout position is the token immediately preceding `target`, i.e.
the last prompt token — the published metric's site (data/evaluations/README.md).

The offline readout also needs the model's final RMSNorm + unembedding. Those are
already on disk as .npy, and G-UNEMBED below checks they reproduce
`HFLensModel.unembed` on this model before the states are trusted — the same
discipline as the J=I check that validated the lens application path.

Usage:
    python scripts/jlens/extract_eval_states.py
    python scripts/jlens/extract_eval_states.py --sets multihop,order-ops
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

# NOTE: `jlens` (the library) must be imported BEFORE REPO_ROOT/scripts joins
# sys.path. This file lives in scripts/jlens/, and Python treats that directory
# as a namespace package named `jlens`, which would shadow the library.
import jlens  # noqa: E402
from jlens.hooks import ActivationRecorder  # noqa: E402

if "filler-token-reasoning" in (jlens.__file__ or "scripts/jlens namespace pkg"):
    raise SystemExit(f"scripts/jlens shadowed the jlens library: {jlens.__file__}")

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from v3_autograd import load_v3_awq  # noqa: E402

EVAL_DIR = Path("/workspace/jacobian-lens/data/evaluations")
ALL_SETS = ["multihop", "order-ops", "association", "multilingual", "poetry", "typo"]
WEIGHTS = REPO_ROOT / "data/model_weights/deepseek_v3"


def rms_norm(x: np.ndarray, w: np.ndarray, eps: float) -> np.ndarray:
    """Identical to scripts/decode/extract_residual_fingerprints.py:61."""
    rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
    return (x / rms) * w


def check_unembed(lens_model, H: torch.Tensor, eps: float, n_check: int = 8) -> dict:
    """G-UNEMBED: does the offline numpy readout reproduce model.unembed()?

    Compares on the FINAL layer's residuals, where `unembed` produces the model's
    own logits, so a mismatch here is a mismatch in the readout the whole offline
    metric rests on. Reports top-1 agreement and the correlation of the logit
    vectors; fp16 storage plus a 129k-wide matmul makes exact equality the wrong
    thing to ask for, but the argmax must not move.
    """
    rms_w = np.load(WEIGHTS / "rms_norm_weight.npy").astype(np.float32)
    lm_w = np.load(WEIGHTS / "lm_head_weight.npy").astype(np.float32)

    h = H[:n_check, -1].float().numpy()                       # [n, d] final layer
    offline = rms_norm(h, rms_w, eps) @ lm_w.T                # [n, vocab]
    with torch.no_grad():
        online = lens_model.unembed(H[:n_check, -1].float()).float().cpu().numpy()

    top1_agree = float((offline.argmax(-1) == online.argmax(-1)).mean())
    corr = float(np.mean([
        np.corrcoef(offline[i], online[i])[0, 1] for i in range(len(offline))
    ]))
    max_abs = float(np.abs(offline - online).max())
    rel = max_abs / float(np.abs(online).max())
    return {"n_checked": int(n_check), "top1_agreement": top1_agree,
            "logit_corr": round(corr, 6), "max_abs_diff": round(max_abs, 4),
            "max_abs_diff_relative": round(rel, 6),
            "passed": top1_agree == 1.0 and corr > 0.999}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="/workspace/models/deepseek-v3-awq")
    ap.add_argument("--eval-dir", type=Path, default=EVAL_DIR)
    ap.add_argument("--sets", default=",".join(ALL_SETS))
    ap.add_argument("--max-seq-len", type=int, default=512,
                    help="Matches eval_lens_quality.py's lens.apply call")
    ap.add_argument("--max-memory-gib", default="130,122,108")
    ap.add_argument("--device-map", default=None)
    ap.add_argument("--out", type=Path,
                    default=REPO_ROOT / "outputs/jlens/eval_states.pt")
    args = ap.parse_args()

    slugs = [s.strip() for s in args.sets.split(",") if s.strip()]
    sets = {}
    for slug in slugs:
        path = args.eval_dir / f"lens-eval-{slug}.json"
        sets[slug] = json.loads(path.read_text())["items"]
    total = sum(len(v) for v in sets.values())
    print(f"eval sets: " + ", ".join(f"{s}={len(v)}" for s, v in sets.items())
          + f"  (total {total} items)")

    model, tokenizer = load_v3_awq(args.model_path,
                                   max_memory_gib=args.max_memory_gib,
                                   device_map=args.device_map)
    # Forward-only: none of the autograd patches from v3_autograd are needed.
    lens_model = jlens.from_hf(model, tokenizer)
    n_layers, d_model = lens_model.n_layers, lens_model.d_model
    eps = float(model.config.rms_norm_eps)
    print(f"  {lens_model}  rms_norm_eps={eps}")

    layers = list(range(n_layers))
    out = {"model_path": args.model_path, "n_layers": n_layers, "d_model": d_model,
           "rms_norm_eps": eps, "max_seq_len": args.max_seq_len,
           "readout_position": -1, "force_bos": True, "sets": {}}

    t_start = time.perf_counter()
    done = 0
    for slug, items in sets.items():
        H = torch.empty(len(items), n_layers, d_model, dtype=torch.float16)
        seq_lens = []
        for i, item in enumerate(items):
            input_ids = lens_model.encode(item["prompt"], max_length=args.max_seq_len)
            seq_lens.append(int(input_ids.shape[-1]))
            with torch.no_grad(), ActivationRecorder(lens_model.layers, at=layers) as rec:
                lens_model.forward(input_ids)
                for L in layers:
                    H[i, L] = rec.activations[L][0][-1].detach().half().cpu()
            done += 1
            if done % 25 == 0:
                el = time.perf_counter() - t_start
                print(f"  {done}/{total}  {el / done:.2f}s/item  "
                      f"eta {(total - done) * el / done / 60:.1f} min", flush=True)

        out["sets"][slug] = {
            "names": [it["name"] for it in items],
            "prompts": [it["prompt"] for it in items],
            "intermediates": [it["intermediates"] for it in items],
            "targets": [it.get("target") for it in items],
            "seq_lens": seq_lens,
            "H": H,  # [n_items, n_layers, d_model] fp16, residual at position -1
        }
        print(f"  {slug}: {tuple(H.shape)} fp16 "
              f"({H.numel() * 2 / 2**20:.0f} MiB), seq_len "
              f"{min(seq_lens)}-{max(seq_lens)}")

    first = out["sets"][slugs[0]]["H"]
    out["unembed_check"] = check_unembed(lens_model, first, eps)
    print(f"  G-UNEMBED: {out['unembed_check']}")
    if not out["unembed_check"]["passed"]:
        print("  !! offline readout does NOT match model.unembed — "
              "states saved, but do not score with them until this is resolved")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(".tmp")
    torch.save(out, tmp)
    tmp.replace(args.out)
    print(f"\nwrote {args.out} ({args.out.stat().st_size / 2**20:.0f} MiB) "
          f"in {(time.perf_counter() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
