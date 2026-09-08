"""
fit_v3.py — fit a Jacobian lens for DeepSeek V3 (AWQ) with jlens.fit().

One matrix J_L (7168 x 7168) per source layer, J_L = E_{prompt,t}[d h_60 / d h_L],
so that

    lens_L(h) = softmax( W_U . RMSNorm( J_L . h ) )

and the logit lens we currently use is exactly the special case J_L = I.
Nothing is trained: J is a closed-form running mean of gradients.

jlens.fit is called UNMODIFIED. Everything V3-specific lives in v3_autograd.py
(AWQ autograd enablement, the moe_infer rewrite, the no-repeat backward).

Run the gates first (scripts/jlens/gates.py). Resumable: re-running with the
same --checkpoint continues where it stopped. Disjoint --prompt-start slices
can be fitted separately and combined with JacobianLens.merge().

Usage:
    python scripts/jlens/fit_v3.py --n-prompts 1000 --dim-batch 16

Defaults reproduce the paper's setting: every layer below the final block,
128-token prompts, skip_first=16, plain (unnormalised) mean over prompts.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch

# NOTE: `jlens` (the library) must be imported BEFORE REPO_ROOT/scripts joins
# sys.path. This file lives in scripts/jlens/, and Python treats that directory
# as a namespace package named `jlens`, which would shadow the library.
import jlens  # noqa: E402
if "filler-token-reasoning" in (jlens.__file__ or "scripts/jlens namespace pkg"):
    raise SystemExit(f"scripts/jlens shadowed the jlens library: {jlens.__file__}")

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from safe_fit import safe_fit  # noqa: E402
from v3_autograd import layer_devices, load_v3_awq, make_differentiable  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="/workspace/models/deepseek-v3-awq")
    ap.add_argument("--prompts", type=Path,
                    default=REPO_ROOT / "scripts/jlens/prompts_wikitext.json")
    ap.add_argument("--prompt-start", type=int, default=0,
                    help="Slice offset, for sharding a fit across runs")
    ap.add_argument("--n-prompts", type=int, default=1000,
                    help="The paper fits 1000 sequences of 128 tokens. Checkpointed "
                         "and resumable, so a run can be stopped at any n and the "
                         "checkpoint used as the lens for that n.")
    ap.add_argument("--min-layer", type=int, default=0,
                    help="Fit source layers min-layer..target-1. 0 (the default) "
                         "passes source_layers=None, i.e. jlens' own default of "
                         "every layer below the target — the paper's setting.")
    ap.add_argument("--source-layers", default=None,
                    help="Explicit comma list; overrides --min-layer")
    ap.add_argument("--target-layer", type=int, default=None,
                    help="Default: the last block (60)")
    ap.add_argument("--dim-batch", type=int, default=16)
    ap.add_argument("--max-seq-len", type=int, default=128)
    ap.add_argument("--max-memory-gib", default="130,122,108")
    ap.add_argument("--gradient-checkpointing", action="store_true",
                    help="Recompute block internals in the backward. Slower per "
                         "pass, but frees ~35x the activation memory, which buys a "
                         "much larger --dim-batch; throughput is linear in dim-batch.")
    ap.add_argument("--device-map", default=None,
                    help="Explicit layer split 'a,b' (bypasses accelerate's "
                         "balancing, which cannot be skewed). Falls back to auto.")
    ap.add_argument("--checkpoint-every", type=int, default=4)
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "outputs/jlens/lens_v3.pt")
    ap.add_argument("--checkpoint", type=Path, default=None,
                    help="Default: <out>.fitckpt")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--cotangent-scale", type=float, default=1.0,
                    help="Scale the one-hot cotangent by this and divide the "
                         "resulting J rows by it. Exact (the map is linear in the "
                         "cotangent); it only moves fp16 intermediates away from "
                         "the 65504 ceiling. 1.0 reproduces the reference exactly. "
                         "Set from scripts/jlens/safe_fit.py --probe.")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = args.checkpoint or args.out.with_suffix(".fitckpt")

    corpus = json.loads(args.prompts.read_text())
    prompts = corpus["prompts"][args.prompt_start: args.prompt_start + args.n_prompts]
    print(f"corpus: {corpus['source']}  slice [{args.prompt_start}:"
          f"{args.prompt_start + args.n_prompts}] -> {len(prompts)} prompts")

    model, tokenizer = load_v3_awq(args.model_path,
                                  max_memory_gib=args.max_memory_gib,
                                  device_map=args.device_map)

    # from_hf calls model.eval(); the AWQ training flags must be set after it.
    lens_model = jlens.from_hf(model, tokenizer)
    patches = make_differentiable(model)
    if args.gradient_checkpointing:
        from v3_autograd import enable_block_checkpointing
        patches["blocks_checkpointed"] = enable_block_checkpointing(model)
    print(f"  patches: {patches}")
    print(f"  {lens_model}")

    target = args.target_layer if args.target_layer is not None else lens_model.n_layers - 1
    if args.source_layers:
        source_layers = [int(x) for x in args.source_layers.split(",")]
    elif args.min_layer == 0:
        source_layers = None  # jlens' default: every layer below target
    else:
        source_layers = list(range(args.min_layer, target))
    fit_and_save(
        lens_model, model, prompts, source_layers, target,
        dim_batch=args.dim_batch, max_seq_len=args.max_seq_len,
        checkpoint=checkpoint, checkpoint_every=args.checkpoint_every,
        resume=not args.no_resume, out=args.out, corpus=corpus,
        cotangent_scale=args.cotangent_scale,
        prompt_slice=[args.prompt_start, args.prompt_start + args.n_prompts],
        model_path=args.model_path,
    )


def fit_and_save(lens_model, model, prompts, source_layers, target, *, dim_batch,
                 max_seq_len, checkpoint, checkpoint_every, resume, out, corpus,
                 prompt_slice, model_path, cotangent_scale=1.0):
    """Run jlens.fit and write the lens + a meta sidecar. Split out of main() so
    scripts/jlens/gates.py --then-fit can reuse an already-loaded model."""
    devs = layer_devices(model)
    out.parent.mkdir(parents=True, exist_ok=True)
    effective = list(range(target)) if source_layers is None else source_layers
    print(f"  source layers {effective[0]}..{effective[-1]} ({len(effective)}) "
          f"-> target L{target}"
          f"{'  [source_layers=None: jlens default]' if source_layers is None else ''}")
    print(f"  band devices: {sorted({str(devs[l]) for l in effective})}")
    print(f"  checkpoint: {checkpoint} (every {checkpoint_every} prompts, "
          f"{len(effective) * 7168 ** 2 * 4 / 2**30:.1f} GiB/write)")

    # safe_fit, NOT jlens.fit: the reference accumulates every per-prompt Jacobian
    # without checking it is finite, and NaN is absorbing in a running sum. That
    # cost the 2026-08-25 run all 48 of its prompts. Same estimator, same running
    # mean; it only refuses to add a non-finite prompt and refuses to overwrite a
    # good checkpoint with a poisoned one. See scripts/jlens/safe_fit.py.
    health = checkpoint.with_name("fit_health.json")
    print(f"  cotangent scale: {cotangent_scale:g}"
          f"{'  [reference default]' if cotangent_scale == 1.0 else '  [rescaled: exact by linearity]'}")
    print(f"  health file: {health}")
    t0 = time.perf_counter()
    lens = safe_fit(
        lens_model,
        prompts,
        source_layers=source_layers,
        target_layer=target,
        dim_batch=dim_batch,
        max_seq_len=max_seq_len,
        checkpoint_path=str(checkpoint),
        checkpoint_every=checkpoint_every,
        resume=resume,
        scale=cotangent_scale,
        health_path=str(health),
    )
    elapsed = time.perf_counter() - t0

    lens.save(str(out))
    diagnostics = {}
    for layer in lens.source_layers:
        J = lens.jacobians[layer]
        diag = J.diagonal()
        diagnostics[f"L{layer}"] = {
            "mean_diag": round(diag.mean().item(), 4),
            "mean_abs_offdiag": round((J - torch.diag(diag)).abs().mean().item(), 6),
            "frob": round(J.norm().item(), 2),
            "frob_minus_I": round((J - torch.eye(J.shape[0])).norm().item(), 2),
            "finite": bool(torch.isfinite(J).all()),
        }
    meta = {
        "model": model_path,
        "corpus": corpus["source"],
        "corpus_sampling": corpus.get("loader", "unspecified"),
        "corpus_note": corpus.get("note"),
        "corpus_deviation": "The reference helper takes the FIRST n records >=600 "
                            "chars, which is a prefix of one article at small n "
                            "(measured: n=8 -> 1 article, n=60 -> 8, n=1000 -> 72). "
                            "Prompts 9+ are sampled one per article instead.",
        "prompt_slice": prompt_slice,
        "n_prompts_fitted": lens.n_prompts,
        "source_layers": lens.source_layers,
        "target_layer": target,
        "dim_batch": dim_batch,
        "max_seq_len": max_seq_len,
        "skip_first": jlens.fitting.SKIP_FIRST_N_POSITIONS,
        "source_layers_arg": "None (jlens default: every layer below target)"
                             if source_layers is None else "explicit",
        "note_dim_batch": "memory/speed knob only; the estimator and total backward "
                          "FLOPs are independent of it",
        "cotangent_scale": cotangent_scale,
        "fit_loop": "scripts/jlens/safe_fit.safe_fit (guarded accumulation); "
                    "estimator is jlens's, with the cotangent scaled",
        "wall_s": round(elapsed, 1),
        "s_per_prompt": round(elapsed / max(lens.n_prompts, 1), 1),
        "layer_convention": "L = output of block L (hook on model.layers[L]), "
                            "identical to scripts/extract/extract_hidden_states.py",
        "diagnostics": diagnostics,
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(f"\nsaved lens -> {out}")
    print(f"saved meta -> {out.with_suffix('.json')}")
    print(f"{elapsed / 60:.1f} min total, {elapsed / max(lens.n_prompts, 1) / 60:.1f} min/prompt")
    for k, v in diagnostics.items():
        print(f"  {k}: {v}")
    return lens


if __name__ == "__main__":
    main()
