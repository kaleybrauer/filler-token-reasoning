"""
gates.py — pre-flight gates for the DeepSeek V3 Jacobian-lens fit.

Each gate is cheap and kills a different failure mode. Run this before
committing GPU time to the fit (scripts/jlens/FIT_RUNBOOK.md section 6).

  G1  AWQ's backward returns the exact activation gradient
      (checked against the FORWARD path's own dequantised weight, so the
      dequant-flag difference between the two paths is settled empirically,
      and against a finite difference of the fused forward). Also checks the
      no-repeat backward against the stock one.
  G2  MLA attention differentiates (gradient arrives at the attention output).
  G3  MoE gradient flows — routed experts AND the gate weights.
  G4  Memory + speed probe over the real band, plus fp16 finiteness of the
      backward. This is the unknown that decides the run shape.
  G5  End-to-end jlens.fit on a 1-block span: J_59 must look like I + small
      (the logit-lens limit), which also validates the layer-index convention.

Usage:
    python scripts/jlens/gates.py --dim-batch 4,16,32 --then-fit
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

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

from v3_autograd import (  # noqa: E402
    _awq_backward_no_repeat,
    layer_devices,
    load_v3_awq,
    make_differentiable,
)

RESULTS: dict[str, dict] = {}


def _report(name: str, passed: bool, **detail):
    RESULTS[name] = {"passed": bool(passed), **detail}
    status = "PASS" if passed else "FAIL"
    print(f"\n[{name}] {status}")
    for k, v in detail.items():
        print(f"    {k}: {v}")


def _mem_snapshot(tag: str, n_gpus: int) -> dict:
    return {
        f"gpu{i}_{tag}_GiB": round(torch.cuda.max_memory_allocated(i) / 2**30, 2)
        for i in range(n_gpus)
    }


# --------------------------------------------------------------------- G1


def gate_g1(model, *, verbose=True):
    """AWQ backward == exact activation gradient of its own fused forward."""
    from awq.modules.linear import gemm as G
    from awq.modules.linear.gemm import WQLinear_GEMM

    out = {}
    targets = {
        "q_a_proj": model.model.layers[59].self_attn.q_a_proj,
        "o_proj": model.model.layers[59].self_attn.o_proj,
        "expert0_gate_proj": model.model.layers[59].mlp.experts[0].gate_proj,
    }
    ok = True
    for name, module in targets.items():
        assert isinstance(module, WQLinear_GEMM), f"{name} is {type(module)}"
        dev = module.qweight.device
        torch.manual_seed(0)
        x = torch.randn(2, 8, module.in_features, dtype=torch.float16, device=dev)
        x.requires_grad_(True)
        y = module(x)
        v = torch.randn_like(y)
        (grad,) = torch.autograd.grad(y, x, v)

        # Reference: dequantise with the FORWARD path's flag and use the same
        # contraction the forward performs (out = x @ W0). Exact grad = v @ W0.T.
        W0 = G.awq_ext.dequantize_weights_cuda(
            module.qweight, module.scales, module.qzeros, 0, 0, 0, False
        )
        ref = torch.matmul(v.float(), W0.float().transpose(0, 1))
        rel = ((grad.float() - ref).norm() / ref.norm()).item()

        # Finite difference of the fused forward, directional (fp16-robust).
        u = torch.randn_like(x)
        fd = {}
        with torch.no_grad():
            for eps in (0.03, 0.1, 0.3):
                d = (module(x + eps * u) - module(x - eps * u)).float() / (2 * eps)
                lhs = (d * v.float()).sum().item()          # v . (J u)
                rhs = (grad.float() * u.float()).sum().item()  # (J^T v) . u
                fd[f"eps={eps}"] = round(abs(lhs - rhs) / max(abs(rhs), 1e-9), 5)

        # Determinism of the AWQ kernels (Route B / finite differences would
        # need this; also a cheap check that nothing is stochastic).
        x2 = x.detach().clone().requires_grad_(True)
        (grad2,) = torch.autograd.grad(module(x2), x2, v)
        backend_diff = (grad2.float() - grad.float()).abs().max().item()

        passed = rel < 2e-3 and max(fd.values()) < 5e-2
        ok &= passed
        out[name] = {
            "shape": f"{module.in_features}->{module.out_features}",
            "rel_err_vs_forward_dequant": round(rel, 6),
            "fd_directional_rel_err": fd,
            "determinism_max_abs": backend_diff,
            "passed": passed,
        }
        if verbose:
            print(f"    {name}: {out[name]}")
    _report("G1_awq_backward", ok, **out)
    return ok


def gate_g1b(model):
    """no-repeat backward is numerically identical to the stock backward."""
    from awq.modules.linear.gemm import WQLinearMMFunction

    from v3_autograd import _ORIGINAL_AWQ_BACKWARD

    module = model.model.layers[59].self_attn.o_proj
    dev = module.qweight.device
    torch.manual_seed(1)
    x = torch.randn(4, 8, module.in_features, dtype=torch.float16, device=dev)
    v = torch.randn(4, 8, module.out_features, dtype=torch.float16, device=dev)

    def run():
        xx = x.detach().clone().requires_grad_(True)
        (g,) = torch.autograd.grad(module(xx), xx, v)
        return g.float()

    WQLinearMMFunction.backward = staticmethod(_awq_backward_no_repeat)
    g_new = run()
    WQLinearMMFunction.backward = _ORIGINAL_AWQ_BACKWARD
    g_old = run()
    WQLinearMMFunction.backward = staticmethod(_awq_backward_no_repeat)

    diff = (g_new - g_old).abs().max().item()
    scale = g_old.abs().max().item()
    passed = diff <= 1e-3 * max(scale, 1e-6)
    _report(
        "G1b_no_repeat_backward",
        passed,
        max_abs_diff=diff,
        grad_scale=scale,
        rel=diff / max(scale, 1e-9),
    )
    return passed


# ------------------------------------------------------------------ G2/G3


def gate_g2_g3(lens_model, model, prompt: str):
    """Gradient reaches the attention output, the routed experts, and the gate."""
    layer = lens_model.n_layers - 2  # block 59: source 59 -> target 60
    block = model.model.layers[layer + 1]

    captured = {}
    grads = {}

    def make_tensor_hook(key):
        def hook(g):
            grads[key] = (
                float(g.float().abs().max()),
                bool(torch.isfinite(g).all()),
            )
        return hook

    def attn_fwd_hook(mod, inputs, output):
        t = output[0] if isinstance(output, tuple) else output
        captured["attn_out"] = t
        if t.requires_grad:
            t.register_hook(make_tensor_hook("attn_out"))

    def gate_fwd_hook(mod, inputs, output):
        # MoEGate returns (topk_idx, topk_weight); the weight is the
        # differentiable half of the routing.
        topk_weight = output[1]
        captured["topk_weight"] = topk_weight
        if topk_weight.requires_grad:
            topk_weight.register_hook(make_tensor_hook("topk_weight"))

    fired = []
    expert_grads = []

    def make_expert_hook(i):
        def hook(mod, inputs, output):
            fired.append(i)
            if output.requires_grad:
                output.register_hook(
                    lambda g: expert_grads.append(float(g.float().abs().max()))
                )
        return hook

    handles = [
        block.self_attn.register_forward_hook(attn_fwd_hook),
        block.mlp.gate.register_forward_hook(gate_fwd_hook),
    ] + [e.register_forward_hook(make_expert_hook(i))
         for i, e in enumerate(block.mlp.experts)]
    try:
        input_ids = lens_model.encode(prompt, max_length=48)
        with (
            ActivationRecorder(
                lens_model.layers, at=[layer, layer + 1], start_graph_at=layer
            ) as rec,
            torch.enable_grad(),
        ):
            lens_model.forward(input_ids)
            h_src, h_tgt = rec.activations[layer], rec.activations[layer + 1]
            cot = torch.zeros_like(h_tgt)
            cot[:, 16:-1, :] = 1.0
            (g,) = torch.autograd.grad(h_tgt, h_src, cot)
    finally:
        for h in handles:
            h.remove()

    g_ok = bool(torch.isfinite(g).all()) and g.float().abs().max().item() > 0
    attn_ok = grads.get("attn_out", (0.0, False))[0] > 0
    expert_ok = len(expert_grads) > 0 and max(expert_grads) > 0
    gate_ok = grads.get("topk_weight", (0.0, False))[0] > 0

    _report(
        "G2_attention_differentiates",
        attn_ok and g_ok,
        grad_at_h_src_max=round(g.float().abs().max().item(), 5),
        grad_at_h_src_finite=bool(torch.isfinite(g).all()),
        attn_out_grad=grads.get("attn_out"),
    )
    _report(
        "G3_moe_gradient",
        expert_ok and gate_ok,
        experts_fired=len(set(fired)),
        experts_with_grad=len(expert_grads),
        max_expert_grad=max(expert_grads) if expert_grads else None,
        gate_topk_weight_grad=grads.get("topk_weight"),
        note="gate grad != 0 confirms straight-through routing is differentiated",
    )
    return (attn_ok and g_ok), (expert_ok and gate_ok)


# --------------------------------------------------------------------- G4


def gate_g4(lens_model, prompt: str, *, min_layer: int, dim_batches, n_passes: int,
            max_seq_len: int = 128):
    """Peak memory + per-pass time for the real band. Mirrors the inner loop of
    jlens.fitting.jacobian_for_prompt for a handful of passes; the fit itself
    calls jlens.fit unmodified."""
    n_gpus = torch.cuda.device_count()
    d_model = lens_model.d_model
    target = lens_model.n_layers - 1
    source_layers = list(range(min_layer, target))
    results = {}
    ok = True

    for dim_batch in dim_batches:
        for i in range(n_gpus):
            torch.cuda.reset_peak_memory_stats(i)
        torch.cuda.empty_cache()
        try:
            input_ids = lens_model.encode(prompt, max_length=max_seq_len)
            seq_len = input_ids.shape[1]
            with (
                ActivationRecorder(
                    lens_model.layers,
                    at=[*source_layers, target],
                    start_graph_at=min(source_layers),
                ) as rec,
                torch.enable_grad(),
            ):
                t0 = time.perf_counter()
                lens_model.forward(input_ids.expand(dim_batch, -1))
                torch.cuda.synchronize()
                t_fwd = time.perf_counter() - t0
                mem_fwd = _mem_snapshot("fwd", n_gpus)

                tgt = rec.activations[target]
                srcs = [rec.activations[l] for l in source_layers]
                valid = torch.arange(16, seq_len - 1, device=tgt.device)
                batch_idx = torch.arange(dim_batch, device=tgt.device)
                cot = torch.zeros_like(tgt)

                stats = {"max_abs": 0.0, "nonfinite": 0, "zero_layers": 0}
                t_bwd = []
                for p in range(n_passes):
                    dim_start = p * dim_batch
                    cot.zero_()
                    cot[batch_idx[:, None], valid[None, :], dim_start + batch_idx[:, None]] = 1.0
                    t1 = time.perf_counter()
                    grads = torch.autograd.grad(tgt, srcs, cot, retain_graph=True)
                    torch.cuda.synchronize()
                    t_bwd.append(time.perf_counter() - t1)
                    for gr in grads:
                        stats["max_abs"] = max(stats["max_abs"], gr.float().abs().max().item())
                        stats["nonfinite"] += int((~torch.isfinite(gr)).sum().item())
                        stats["zero_layers"] += int(gr.float().abs().max().item() == 0)
                    del grads
                mem_bwd = _mem_snapshot("peak", n_gpus)
            # Drop every reference into the retained graph before the next
            # dim_batch, or its activations stay alive and inflate the peak.
            del tgt, srcs, cot, rec
            torch.cuda.empty_cache()

            n_total = math.ceil(d_model / dim_batch)
            est_s = t_fwd + n_total * (sum(t_bwd) / len(t_bwd))
            entry = {
                "seq_len": seq_len,
                "t_forward_s": round(t_fwd, 2),
                "t_backward_s": round(sum(t_bwd) / len(t_bwd), 3),
                "passes_per_prompt": n_total,
                "est_min_per_prompt": round(est_s / 60, 1),
                "grad_max_abs": round(stats["max_abs"], 4),
                "grad_nonfinite": stats["nonfinite"],
                "grad_all_zero_layers": stats["zero_layers"],
                **mem_fwd,
                **mem_bwd,
            }
            entry["passed"] = stats["nonfinite"] == 0 and stats["max_abs"] > 0
            ok &= entry["passed"]
        except torch.cuda.OutOfMemoryError as exc:
            entry = {"OOM": str(exc)[:200], "passed": False}
            ok = False
            # The `del` below the with-block never ran, so the half-built graph is
            # still referenced by locals; drop them or it starves whatever runs next.
            for name in ("tgt", "srcs", "cot", "rec", "grads"):
                if name in locals():
                    del locals()[name]
            locals().clear() if False else None
            torch.cuda.empty_cache()
        results[f"dim_batch={dim_batch}"] = entry
        print(f"    dim_batch={dim_batch}: {entry}")

    _report("G4_memory_and_speed", ok, band=f"{min_layer}..{target - 1} -> {target}",
            n_source_layers=len(source_layers), **results)
    return ok


# --------------------------------------------------------------------- G5


def gate_g5(lens_model, prompts, *, dim_batch: int, source_layers, out_dir: Path):
    """Real jlens.fit on a 1-2 block span: J should be I + small."""
    t0 = time.perf_counter()
    lens = jlens.fit(
        lens_model,
        prompts,
        source_layers=source_layers,
        dim_batch=dim_batch,
        max_seq_len=128,
        checkpoint_path=None,
    )
    elapsed = time.perf_counter() - t0

    detail = {"n_prompts": len(prompts), "wall_s": round(elapsed, 1),
              "s_per_prompt": round(elapsed / max(len(prompts), 1), 1)}
    ok = True
    for layer in lens.source_layers:
        J = lens.jacobians[layer]
        diag = J.diagonal()
        off = J - torch.diag(diag)
        d = {
            "mean_diag": round(diag.mean().item(), 4),
            "median_diag": round(diag.median().item(), 4),
            "mean_abs_offdiag": round(off.abs().mean().item(), 5),
            "frob_J_minus_I": round((J - torch.eye(J.shape[0])).norm().item(), 2),
            "frob_J": round(J.norm().item(), 2),
            "finite": bool(torch.isfinite(J).all()),
        }
        # A one-block span must be dominated by the residual identity path.
        d["passed"] = d["finite"] and d["mean_diag"] > 0.5 and d["frob_J_minus_I"] < d["frob_J"]
        ok &= d["passed"]
        detail[f"L{layer}"] = d

    out_dir.mkdir(parents=True, exist_ok=True)
    lens.save(str(out_dir / "lens_g5_smoke.pt"))
    _report("G5_fit_smoke", ok, **detail)
    return ok


# -------------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="/workspace/models/deepseek-v3-awq")
    ap.add_argument("--prompts", type=Path,
                    default=REPO_ROOT / "scripts/jlens/prompts_wikitext.json")
    ap.add_argument("--min-layer", type=int, default=0,
                    help="0 = every layer below target (jlens/paper default)")
    ap.add_argument("--dim-batch", default="1,4,16")
    ap.add_argument("--g4-passes", type=int, default=3)
    ap.add_argument("--g5-layers", default="59")
    ap.add_argument("--g5-prompts", type=int, default=2)
    ap.add_argument("--g5-dim-batch", type=int, default=16)
    ap.add_argument("--max-memory-gib", default="130,122,108",
                    help="uniform int, or per-GPU comma list")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "outputs/jlens/gates.json")
    ap.add_argument("--skip", default="", help="comma list, e.g. g4,g5")
    ap.add_argument("--then-fit", action="store_true",
                    help="If every gate passes, run the fit in this process, reusing "
                         "the loaded model (saves a ~13 min reload).")
    ap.add_argument("--fit-n-prompts", type=int, default=1000)
    ap.add_argument("--fit-dim-batch", type=int, default=0,
                    help="0 = largest dim_batch G4 measured with enough headroom left")
    ap.add_argument("--fit-min-free-gib", type=float, default=10.0)
    ap.add_argument("--fit-checkpoint-every", type=int, default=2)
    ap.add_argument("--fit-out", type=Path,
                    default=REPO_ROOT / "outputs/jlens/lens_v3.pt")
    args = ap.parse_args()

    skip = {s.strip().lower() for s in args.skip.split(",") if s.strip()}
    prompts = json.loads(args.prompts.read_text())["prompts"]

    model, tokenizer = load_v3_awq(args.model_path, max_memory_gib=args.max_memory_gib)
    devs = layer_devices(model)
    print("  block->device:", {i: str(d) for i, d in devs.items() if i % 10 == 0 or i == 60})

    # from_hf calls model.eval(), so patch afterwards.
    lens_model = jlens.from_hf(model, tokenizer)
    info = make_differentiable(model)
    print(f"  patches: {info}")
    print(f"  {lens_model}")

    ids = lens_model.encode(prompts[0], max_length=32)
    print(f"  encode check: first ids {ids[0, :4].tolist()} "
          f"(bos_token_id={tokenizer.bos_token_id})")

    if "g1" not in skip:
        gate_g1(model)
        gate_g1b(model)
    if "g2" not in skip or "g3" not in skip:
        gate_g2_g3(lens_model, model, prompts[0])
    if "g4" not in skip:
        gate_g4(
            lens_model,
            prompts[0],
            min_layer=args.min_layer,
            dim_batches=[int(x) for x in args.dim_batch.split(",")],
            n_passes=args.g4_passes,
        )
    if "g5" not in skip:
        gate_g5(
            lens_model,
            prompts[: args.g5_prompts],
            dim_batch=args.g5_dim_batch,
            source_layers=[int(x) for x in args.g5_layers.split(",")],
            out_dir=args.out.parent,
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(RESULTS, indent=2))
    print(f"\nwrote {args.out}")
    print("SUMMARY: " + ", ".join(
        f"{k}={'PASS' if v['passed'] else 'FAIL'}" for k, v in RESULTS.items()))

    if not args.then_fit:
        return
    if not all(v["passed"] for v in RESULTS.values()):
        print("\n--then-fit: NOT starting, a gate failed")
        return

    dim_batch = args.fit_dim_batch
    if dim_batch == 0:
        picked = choose_dim_batch(RESULTS.get("G4_memory_and_speed", {}),
                                  torch.cuda.device_count(), args.fit_min_free_gib)
        if picked is None:
            print("\n--then-fit: NOT starting, no dim_batch left "
                  f"{args.fit_min_free_gib} GiB free on every GPU")
            return
        dim_batch, free = picked
        print(f"\n--then-fit: dim_batch={dim_batch} (min free across GPUs {free:.1f} GiB)")

    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    from fit_v3 import fit_and_save

    target = lens_model.n_layers - 1
    fit_and_save(
        lens_model, model, prompts[: args.fit_n_prompts],
        None if args.min_layer == 0 else list(range(args.min_layer, target)), target,
        dim_batch=dim_batch, max_seq_len=128,
        checkpoint=args.fit_out.with_suffix(".fitckpt"),
        checkpoint_every=args.fit_checkpoint_every, resume=True,
        out=args.fit_out, corpus=json.loads(args.prompts.read_text()),
        prompt_slice=[0, args.fit_n_prompts], model_path=args.model_path,
    )


def choose_dim_batch(g4: dict, n_gpus: int, min_free_gib: float):
    """Largest dim_batch G4 measured that still leaves min_free_gib on every GPU."""
    best = None
    for key, entry in g4.items():
        if not key.startswith("dim_batch=") or not isinstance(entry, dict):
            continue
        if not entry.get("passed"):
            continue
        try:
            free = min(
                torch.cuda.get_device_properties(i).total_memory / 2**30
                - entry[f"gpu{i}_peak_GiB"]
                for i in range(n_gpus)
            )
        except KeyError:
            continue
        db = int(key.split("=")[1])
        if free >= min_free_gib and (best is None or db > best[0]):
            best = (db, free)
    return best


if __name__ == "__main__":
    main()
