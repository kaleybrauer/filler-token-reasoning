"""
safe_fit.py — the J-lens fit with fp16 overflow handled and NaN made non-absorbing.

WHY THIS EXISTS. The 2026-08-25 run accumulated 48 prompts and produced an
unusable lens. Six of them (12, 16, 17, 22, 25, 41) overflowed fp16 in the reverse
pass: the cotangent is 1.0 and the gradient grows as it propagates back through 60
blocks, past fp16's 65504 ceiling, giving inf and then NaN. `jlens.fit` adds every
per-prompt Jacobian to the running sum WITHOUT a finiteness check, and NaN is
absorbing in a sum — one bad prompt poisons that row of J forever, and every later
checkpoint inherits it (snap40 - snap30 is NaN even though prompts 31-40 were all
fine). Measured damage: L59 80 of 7168 output dims poisoned, L40 2463, L0 6387.
Deeper spans overflow more, which is the signature of fp16 depth, not of a bug in
the gradient path (that path was audited and is correct).

TWO INDEPENDENT DEFENCES, because either alone would have saved the run:

1. `jacobian_for_prompt_scaled` — scale the cotangent by `s` and divide the
   resulting rows by `s`. The whole map is linear in the cotangent, so this is
   mathematically identical to the reference; it only moves the intermediates away
   from the fp16 ceiling. This is the fix named in FIT_RUNBOOK.md §11. Retries a
   pass at a smaller `s` if it still comes back non-finite.

2. `safe_fit` — never accumulate a non-finite per-prompt Jacobian, and never write
   a checkpoint whose running sum is non-finite. This turns a run-destroying event
   into one skipped prompt, and is the guard that matters most: it holds even if
   the scaling is mis-tuned.

The estimator is otherwise the reference's, byte for byte: same cotangent
placement (every valid target position >= source), same mean over source
positions, same skip_first, same plain unnormalised running mean over prompts.

Skipping is NOT free of consequence: it excludes prompts non-randomly (the
heavy-gradient tail), so `safe_fit` records every rejection in the health file and
in the lens sidecar. Scaling exists so that number stays at zero.

Usage:
    # probe: which scale keeps prompt 12 finite, at full depth, in ~10 s of compute
    python scripts/jlens/safe_fit.py --probe --prompt-index 11 --scales 1,0.125,0.015625
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import torch

import jlens  # noqa: E402  (must precede REPO_ROOT/scripts on sys.path)
from jlens.fitting import _check_layer_indices, valid_position_mask
from jlens.hooks import ActivationRecorder

if "filler-token-reasoning" in (jlens.__file__ or "scripts/jlens namespace pkg"):
    raise SystemExit(f"scripts/jlens shadowed the jlens library: {jlens.__file__}")

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))


class NonFiniteGradient(RuntimeError):
    """A pass stayed non-finite even at the smallest permitted cotangent scale."""


def jacobian_for_prompt_scaled(
    model, prompt, source_layers, *, target_layer=None, dim_batch=8,
    max_seq_len=128, skip_first=16, scale=1.0, min_scale=2.0 ** -20,
    backoff=8.0, max_retries=8, position_stats=None,
):
    """`jlens.fitting.jacobian_for_prompt` with a scaled cotangent.

    Identical estimator: cotangent one-hot at output dim `dim_start + b` at every
    valid target position, gradient averaged over valid source positions. The only
    change is that the one-hot carries `scale` instead of `1.0` and the resulting
    rows are divided by `scale` in fp32 — exact by linearity of the derivative.

    `scale` is carried DOWN across passes once a backoff happens: at full depth
    most passes overflow together, so re-discovering that per pass would nearly
    double the cost. It never rises again within a prompt.

    `position_stats`: optional dict to fill with per-position diagnostics that
    cost nothing extra (the paper's within-prompt outlier criteria): for every
    source layer, `grad_sq[layer]` = sum over output dims and residual dims of the
    squared (rescaled) gradient at each VALID source position — i.e. the squared
    Frobenius norm of that position's own Jacobian, so sum over positions /
    n_valid**2 recovers ||J_p||_F**2 up to cross terms — and `resid_norm[layer]` =
    ||h_layer[t]|| at every position. Also `valid_positions` and `input_ids`.
    The estimator is untouched; the accumulation happens only on accepted passes.

    Returns (jacobians, seq_len, n_valid_positions, scale_used, n_retries).
    """
    n_layers, d_model = model.n_layers, model.d_model
    source_layers, target_layer = _check_layer_indices(
        source_layers, target_layer, n_layers
    )

    input_ids = model.encode(prompt, max_length=max_seq_len)
    seq_len = input_ids.shape[1]
    position_mask = valid_position_mask(seq_len, skip_first=skip_first)
    n_valid_positions = int(position_mask.sum())

    jacobians = {
        layer: torch.zeros(d_model, d_model, dtype=torch.float32)
        for layer in source_layers
    }
    n_retries = 0

    with (
        ActivationRecorder(
            model.layers,
            at=[*source_layers, target_layer],
            start_graph_at=min(source_layers),
        ) as recorder,
        torch.enable_grad(),
    ):
        model.forward(input_ids.expand(dim_batch, -1))
        target_activation = recorder.activations[target_layer]
        source_activations = [recorder.activations[layer] for layer in source_layers]

        valid_positions = position_mask.nonzero(as_tuple=True)[0].to(
            target_activation.device
        )
        if position_stats is not None:
            position_stats["input_ids"] = input_ids[0].cpu()
            position_stats["valid_positions"] = position_mask.nonzero(as_tuple=True)[0]
            position_stats["resid_norm"] = {
                layer: recorder.activations[layer][0].detach().float().norm(dim=-1).cpu()
                for layer in [*source_layers, target_layer]
            }
            position_stats["grad_sq"] = {
                layer: torch.zeros(n_valid_positions, dtype=torch.float64)
                for layer in source_layers
            }
        batch_indices = torch.arange(dim_batch, device=target_activation.device)
        cotangent = torch.zeros_like(target_activation)
        cur = float(scale)

        for dim_start in range(0, d_model, dim_batch):
            n_dims = min(dim_batch, d_model - dim_start)
            while True:
                cotangent.zero_()
                cotangent[
                    batch_indices[:n_dims, None],
                    valid_positions[None, :],
                    dim_start + batch_indices[:n_dims, None],
                ] = cur
                # retain_graph=True on every pass: a retry has to be able to run
                # against the same graph, and the graph is freed with the `with`.
                grads = torch.autograd.grad(
                    outputs=target_activation,
                    inputs=source_activations,
                    grad_outputs=cotangent,
                    retain_graph=True,
                )
                rows, sq, ok = {}, {}, True
                for layer, grad in zip(source_layers, grads, strict=True):
                    pos = valid_positions.to(grad.device, non_blocking=True)
                    # .float() BEFORE the divide: the mean and the rescale are both
                    # done in fp32, so only the backward itself runs at fp16 risk.
                    g = grad[:n_dims, pos, :].float() / cur
                    r = g.mean(dim=1)
                    if not torch.isfinite(r).all():
                        ok = False
                        break
                    rows[layer] = r.cpu()
                    if position_stats is not None:
                        sq[layer] = g.double().pow(2).sum(dim=(0, 2)).cpu()
                del grads
                if ok:
                    for layer, r in rows.items():
                        jacobians[layer][dim_start : dim_start + n_dims, :] = r
                    if position_stats is not None:
                        for layer, v in sq.items():
                            position_stats["grad_sq"][layer] += v
                    break
                n_retries += 1
                if n_retries > max_retries or cur / backoff < min_scale:
                    raise NonFiniteGradient(
                        f"dims {dim_start}..{dim_start + n_dims} still non-finite "
                        f"at scale={cur:g} after {n_retries} retries"
                    )
                cur /= backoff

    return jacobians, seq_len, n_valid_positions, cur, n_retries


def _atomic_save(obj, path):
    import os
    tmp = f"{path}.tmp.{os.getpid()}"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def safe_fit(
    model, prompts, *, source_layers=None, target_layer=None, dim_batch=8,
    max_seq_len=128, skip_first=16, checkpoint_path=None, checkpoint_every=2,
    resume=True, scale=1.0, health_path=None, logger=None,
):
    """`jlens.fit` with the accumulation guarded. Same running mean, same output.

    The guard is the whole point: a per-prompt Jacobian that is not entirely
    finite is REJECTED, never added. NaN is absorbing in a running sum, so
    admitting one is unrecoverable — it cannot be subtracted back out, and every
    later checkpoint inherits it. Rejecting costs that prompt only.

    A checkpoint is likewise never written unless the running sum is finite, so a
    good checkpoint can never be overwritten by a poisoned one.
    """
    log = logger or (lambda m: print(m, flush=True))
    n_layers, d_model = model.n_layers, model.d_model
    source_layers, target_layer = _check_layer_indices(
        source_layers, target_layer, n_layers
    )

    jacobian_sum = {
        l: torch.zeros(d_model, d_model, dtype=torch.float32) for l in source_layers
    }
    n_done, next_idx, rejected = 0, 0, []

    if resume and checkpoint_path and Path(checkpoint_path).exists():
        ck = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        bad = [l for l, J in ck["jacobian_sum"].items() if not torch.isfinite(J).all()]
        if bad:
            raise SystemExit(
                f"REFUSING TO RESUME: {checkpoint_path} has non-finite entries in "
                f"{len(bad)} layers (e.g. L{bad[0]}). A poisoned running sum cannot "
                f"be repaired by continuing — restart from a clean checkpoint."
            )
        jacobian_sum = {int(l): J.float() for l, J in ck["jacobian_sum"].items()}
        n_done, next_idx = int(ck["n_done"]), int(ck["next_idx"])
        rejected = list(ck.get("rejected_prompt_indices", []))
        log(f"  resuming from checkpoint: {n_done}/{len(prompts)} prompts processed"
            f"{f', {len(rejected)} previously rejected' if rejected else ''}")

    def health(extra=None):
        if not health_path:
            return
        h = {"n_done": n_done, "next_idx": next_idx, "n_prompts": len(prompts),
             "rejected_prompt_indices": rejected, "n_rejected": len(rejected),
             "cotangent_scale": scale, "updated": time.strftime("%F %T"),
             "running_sum_finite": True}
        if extra:
            h.update(extra)
        Path(health_path).write_text(json.dumps(h, indent=2))

    def write_checkpoint():
        bad = [l for l, J in jacobian_sum.items() if not torch.isfinite(J).all()]
        if bad:
            # Must never happen: nothing non-finite is ever accumulated. If it
            # does, the in-memory sum is already lost, but the checkpoint on disk
            # is still good, so stop rather than overwrite it.
            health({"running_sum_finite": False, "poisoned_layers": bad[:10]})
            raise SystemExit(
                f"ABORT: running sum went non-finite in {len(bad)} layers despite "
                f"the accumulation guard. On-disk checkpoint left intact."
            )
        if checkpoint_path:
            _atomic_save({"jacobian_sum": jacobian_sum, "n_done": n_done,
                          "next_idx": next_idx, "source_layers": source_layers,
                          "target_layer": target_layer, "skip_first": skip_first,
                          "rejected_prompt_indices": rejected,
                          "cotangent_scale": scale}, checkpoint_path)
        health()

    sqrt_d = math.sqrt(d_model)
    health()
    for prompt_idx, prompt in enumerate(prompts):
        if prompt_idx < next_idx:
            continue
        t0 = time.perf_counter()
        try:
            per_prompt_J, seq_len, n_valid, scale_used, n_retries = (
                jacobian_for_prompt_scaled(
                    model, prompt, source_layers, target_layer=target_layer,
                    dim_batch=dim_batch, max_seq_len=max_seq_len,
                    skip_first=skip_first, scale=scale,
                )
            )
        except NonFiniteGradient as exc:
            rejected.append(prompt_idx)
            next_idx = prompt_idx + 1
            log(f"  REJECTED prompt {prompt_idx + 1}/{len(prompts)}: {exc}")
            write_checkpoint()
            continue
        except ValueError as exc:
            next_idx = prompt_idx + 1
            log(f"  skipping prompt {prompt_idx + 1}: {exc}")
            continue

        # Belt and braces: the estimator already verified every pass, but the sum
        # is the thing that cannot be repaired, so check once more before it goes in.
        nonfinite = [l for l in source_layers if not torch.isfinite(per_prompt_J[l]).all()]
        if nonfinite:
            rejected.append(prompt_idx)
            next_idx = prompt_idx + 1
            log(f"  REJECTED prompt {prompt_idx + 1}/{len(prompts)}: non-finite in "
                f"{len(nonfinite)} layers (e.g. L{nonfinite[0]}) — NOT accumulated")
            write_checkpoint()
            continue

        prompt_norm = max(per_prompt_J[l].norm().item() for l in source_layers) / sqrt_d
        rel_change = float("nan")
        if n_done > 0:
            rel_change = max(
                ((per_prompt_J[l] - jacobian_sum[l] / n_done).norm()
                 / ((n_done + 1) * (jacobian_sum[l] / n_done).norm())).item()
                for l in source_layers
            )
        for layer in source_layers:
            jacobian_sum[layer] += per_prompt_J[layer]
        n_done += 1
        next_idx = prompt_idx + 1

        log(f"  prompt {prompt_idx + 1}/{len(prompts)}  seq_len={seq_len} "
            f"n_valid={n_valid}  {time.perf_counter() - t0:.0f}s  "
            f"max||J||/sqrt(d)={prompt_norm:.3f}  max_d_mean={rel_change:.2e}  "
            f"scale={scale_used:g}{f' retries={n_retries}' if n_retries else ''}")
        if checkpoint_every and next_idx % checkpoint_every == 0:
            write_checkpoint()

    write_checkpoint()
    if n_done == 0:
        raise ValueError("no prompts were accepted")
    from jlens.lens import JacobianLens
    log(f"safe_fit: done, {n_done} prompts accepted, {len(rejected)} rejected")
    return JacobianLens(
        jacobians={l: jacobian_sum[l] / n_done for l in source_layers},
        n_prompts=n_done, d_model=d_model,
    )


def probe_scales(model, prompt, source_layers, *, target_layer=None, dim_batch=64,
                 max_seq_len=128, skip_first=16, scales=(1.0,), n_passes=3):
    """How large can the cotangent be before the fp16 backward overflows?

    Runs the first `n_passes` dim-blocks at full depth for each scale against ONE
    retained graph, so a decisive answer costs one forward plus a few backwards
    rather than a 40-minute prompt. Reports, per scale, whether every layer stayed
    finite and how close the raw fp16 gradient came to the 65504 ceiling.

    Valid because overflow is prompt-specific, not dimension-specific: 174 of the
    first 192 output dims were poisoned in the failed run, so the earliest passes
    already exercise the failure.
    """
    n_layers, d_model = model.n_layers, model.d_model
    source_layers, target_layer = _check_layer_indices(
        source_layers, target_layer, n_layers
    )
    input_ids = model.encode(prompt, max_length=max_seq_len)
    mask = valid_position_mask(input_ids.shape[1], skip_first=skip_first)
    out = {}

    with (
        ActivationRecorder(model.layers, at=[*source_layers, target_layer],
                           start_graph_at=min(source_layers)) as rec,
        torch.enable_grad(),
    ):
        model.forward(input_ids.expand(dim_batch, -1))
        tgt = rec.activations[target_layer]
        srcs = [rec.activations[l] for l in source_layers]
        valid = mask.nonzero(as_tuple=True)[0].to(tgt.device)
        bidx = torch.arange(dim_batch, device=tgt.device)
        cot = torch.zeros_like(tgt)

        for s in scales:
            worst_layer, max_abs, bad_layers = None, 0.0, []
            for p in range(n_passes):
                d0 = p * dim_batch
                n_dims = min(dim_batch, d_model - d0)
                cot.zero_()
                cot[bidx[:n_dims, None], valid[None, :], d0 + bidx[:n_dims, None]] = s
                grads = torch.autograd.grad(tgt, srcs, cot, retain_graph=True)
                for layer, g in zip(source_layers, grads, strict=True):
                    finite = bool(torch.isfinite(g).all())
                    if not finite and layer not in bad_layers:
                        bad_layers.append(layer)
                    if finite:
                        m = float(g.abs().max())
                        if m > max_abs:
                            max_abs, worst_layer = m, layer
                del grads
            out[s] = {
                "all_finite": not bad_layers,
                "n_bad_layers": len(bad_layers),
                "first_bad_layer": bad_layers[0] if bad_layers else None,
                "max_abs_grad_fp16": round(max_abs, 1),
                "worst_layer": worst_layer,
                "headroom_vs_65504": round(65504.0 / max_abs, 1) if max_abs else None,
            }
            print(f"  scale={s:<12g} {out[s]}", flush=True)
    return out


def validate_scales(model, prompt, source_layers, *, target_layer=None, dim_batch=64,
                    max_seq_len=128, skip_first=16, scales=(1.0,), passes=(0,),
                    ref_scale=1.0):
    """Two questions on one retained graph: does rescaling distort J, and where does
    a prompt actually overflow?

    `probe_scales` only walks the first few dim-blocks, which is why prompts 16 and
    41 looked finite there despite having failed in the real run — their overflow is
    at dims it never reached. `passes` here is an explicit list of dim-block indices,
    so it can be spread across the full 0..7167 range.

    For every layer it also compares the rows produced at each scale against those at
    `ref_scale`. Rescaling is exact in real arithmetic, so any disagreement is fp16:
    too-small a scale pushes small gradients into subnormals and loses them. That
    error is what decides how far the scale can safely be lowered, and it has to be
    measured rather than assumed.
    """
    n_layers, d_model = model.n_layers, model.d_model
    source_layers, target_layer = _check_layer_indices(source_layers, target_layer, n_layers)
    input_ids = model.encode(prompt, max_length=max_seq_len)
    mask = valid_position_mask(input_ids.shape[1], skip_first=skip_first)
    out = {}

    with (
        ActivationRecorder(model.layers, at=[*source_layers, target_layer],
                           start_graph_at=min(source_layers)) as rec,
        torch.enable_grad(),
    ):
        model.forward(input_ids.expand(dim_batch, -1))
        tgt = rec.activations[target_layer]
        srcs = [rec.activations[l] for l in source_layers]
        valid = mask.nonzero(as_tuple=True)[0].to(tgt.device)
        bidx = torch.arange(dim_batch, device=tgt.device)
        cot = torch.zeros_like(tgt)

        def rows_at(s, p):
            d0 = p * dim_batch
            n_dims = min(dim_batch, d_model - d0)
            cot.zero_()
            cot[bidx[:n_dims, None], valid[None, :], d0 + bidx[:n_dims, None]] = s
            grads = torch.autograd.grad(tgt, srcs, cot, retain_graph=True)
            r = {}
            for layer, g in zip(source_layers, grads, strict=True):
                pos = valid.to(g.device, non_blocking=True)
                r[layer] = (g[:n_dims, pos, :].float().mean(dim=1) / s).cpu()
            del grads
            return r

        ref = {p: rows_at(ref_scale, p) for p in passes}
        for s in scales:
            worst_rel, worst_at, bad = 0.0, None, []
            maxabs = 0.0
            for p in passes:
                cur = rows_at(s, p) if s != ref_scale else ref[p]
                for layer in source_layers:
                    a, b = cur[layer], ref[p][layer]
                    if not torch.isfinite(a).all():
                        if (p, layer) not in bad:
                            bad.append((p, layer))
                        continue
                    maxabs = max(maxabs, float(a.abs().max()) * s)
                    if torch.isfinite(b).all() and float(b.norm()) > 0:
                        rel = float((a - b).norm() / b.norm())
                        if rel > worst_rel:
                            worst_rel, worst_at = rel, (p, layer)
            out[s] = {
                "all_finite": not bad,
                "n_bad_pass_layer": len(bad),
                "first_bad": bad[0] if bad else None,
                "max_rel_diff_vs_ref": round(worst_rel, 6),
                "worst_at_pass_layer": worst_at,
                "max_abs_grad_fp16_est": round(maxabs, 1),
            }
            print(f"  scale={s:<12g} {out[s]}", flush=True)
    return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--validate", action="store_true",
                    help="Spread passes across the full dim range and check "
                         "rescaled rows against scale=1")
    ap.add_argument("--passes", default="0,20,40,60,80,100")
    ap.add_argument("--then-fit", action="store_true",
                    help="After validating, HOLD the loaded model and wait for "
                         "logs/GO_JLENS_FIT (JSON: cotangent_scale, n_prompts). "
                         "A 13-min reload costs more than the GPUs cost idling "
                         "while the validation output is reviewed.")
    ap.add_argument("--go-timeout-min", type=int, default=240)
    ap.add_argument("--prompt-index", type=int, nargs="+", default=[11],
                    help="0-based. The failed run's bad prompts were 1-based "
                         "12,16,17,22,25,41 -> 0-based 11,15,16,21,24,40")
    ap.add_argument("--scales", default="1,0.125,0.015625,0.001953125")
    ap.add_argument("--dim-batch", type=int, default=64)
    ap.add_argument("--n-passes", type=int, default=3)
    ap.add_argument("--device-map", default="22,42")
    ap.add_argument("--max-memory-gib", default="130,122,108")
    ap.add_argument("--prompts", type=Path,
                    default=REPO_ROOT / "scripts/jlens/prompts_wikitext.json")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "outputs/jlens/scale_probe.json")
    args = ap.parse_args()
    if not (args.probe or args.validate):
        raise SystemExit("this module is a library; use --probe or --validate")

    from v3_autograd import enable_block_checkpointing, load_v3_awq, make_differentiable

    corpus = json.loads(args.prompts.read_text())["prompts"]
    scales = [float(x) for x in args.scales.split(",")]
    model, tokenizer = load_v3_awq("/workspace/models/deepseek-v3-awq",
                                   max_memory_gib=args.max_memory_gib,
                                   device_map=args.device_map)
    lens_model = jlens.from_hf(model, tokenizer)
    print(f"  patches: {make_differentiable(model)}")
    print(f"  blocks_checkpointed: {enable_block_checkpointing(model)}")

    report = {}
    passes = [int(x) for x in args.passes.split(",")]

    # Would a wider-exponent cotangent avoid the problem outright? fp16 tops out at
    # 65504; bf16 carries fp32's exponent range (~3.4e38), so an overflow simply
    # cannot occur there. The AWQ backward casts the dequantised weight to
    # grad_output.dtype, so a bf16 cotangent would make the whole reverse pass bf16.
    # This is almost certainly why the reference never needed scaling: its models are
    # bf16, and only our AWQ path is fp16-locked. Worth one second to find out whether
    # autograd will accept the dtype mismatch against fp16 activations.
    if args.validate:
        import torch as _t
        try:
            _ids = lens_model.encode(corpus[0], max_length=32)
            with ActivationRecorder(lens_model.layers, at=[0, lens_model.n_layers - 1],
                                    start_graph_at=0) as _r, _t.enable_grad():
                lens_model.forward(_ids.expand(2, -1))
                _tg = _r.activations[lens_model.n_layers - 1]
                _sr = _r.activations[0]
                _c = _t.zeros_like(_tg, dtype=_t.bfloat16)
                _c[:, -2, 0] = 1.0
                _g = _t.autograd.grad(_tg, _sr, _c, retain_graph=False)[0]
                print(f"\nbf16 cotangent: ACCEPTED, grad dtype={_g.dtype}, "
                      f"finite={bool(_t.isfinite(_g).all())}", flush=True)
                report["bf16_cotangent"] = {"accepted": True, "grad_dtype": str(_g.dtype)}
        except Exception as _e:
            print(f"\nbf16 cotangent: REJECTED -> {type(_e).__name__}: {str(_e)[:200]}", flush=True)
            report["bf16_cotangent"] = {"accepted": False, "error": f"{type(_e).__name__}: {str(_e)[:300]}"}

    for pi in args.prompt_index:
        print(f"\n=== prompt index {pi} (1-based {pi + 1}) ===", flush=True)
        if args.validate:
            report[str(pi)] = validate_scales(
                lens_model, corpus[pi], None, dim_batch=args.dim_batch,
                scales=scales, passes=passes,
            )
        else:
            report[str(pi)] = probe_scales(
                lens_model, corpus[pi], None, dim_batch=args.dim_batch,
                scales=scales, n_passes=args.n_passes,
            )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nwrote {args.out}")

    if args.then_fit:
        go = REPO_ROOT / "logs/GO_JLENS_FIT"
        abort = REPO_ROOT / "logs/ABORT_JLENS_FIT"
        ckpt = REPO_ROOT / "outputs/jlens/lens_v3.fitckpt"
        print(f"\n{'=' * 70}\nVALIDATION COMPLETE — model held in memory ({args.go_timeout_min} min max).\n"
              f"  to start the fit:  echo '{{\"cotangent_scale\": 0.015625}}' > {go}\n"
              f"  to abort:          touch {abort}\n{'=' * 70}", flush=True)
        waited = 0
        while waited < args.go_timeout_min * 60:
            if abort.exists():
                print("ABORT_JLENS_FIT present — exiting without fitting", flush=True)
                raise SystemExit(0)
            if go.exists():
                cfg = {}
                try:
                    cfg = json.loads(go.read_text() or "{}")
                except Exception as e:
                    print(f"  GO file unparseable ({e}); using defaults", flush=True)
                scale = float(cfg.get("cotangent_scale", 1.0))
                n_prompts = int(cfg.get("n_prompts", 1000))
                every = int(cfg.get("checkpoint_every", 2))
                dim_batch = int(cfg.get("dim_batch", args.dim_batch))
                print(f"\nGO: scale={scale:g} n_prompts={n_prompts} dim_batch={dim_batch}", flush=True)
                if not ckpt.exists():
                    raise SystemExit(f"{ckpt} missing — seed it from a CLEAN checkpoint first")
                sys.path.insert(0, str(REPO_ROOT / "scripts/jlens"))
                from fit_v3 import fit_and_save  # lazy: fit_v3 imports this module
                corpus_meta = json.loads(args.prompts.read_text())
                target = lens_model.n_layers - 1
                fit_and_save(
                    lens_model, model, corpus_meta["prompts"][:n_prompts], None, target,
                    dim_batch=dim_batch, max_seq_len=128, checkpoint=ckpt,
                    checkpoint_every=every, resume=True,
                    out=REPO_ROOT / "outputs/jlens/lens_v3.pt", corpus=corpus_meta,
                    prompt_slice=[0, n_prompts],
                    model_path="/workspace/models/deepseek-v3-awq",
                    cotangent_scale=scale,
                )
                raise SystemExit(0)
            time.sleep(15)
            waited += 15
        print(f"no GO within {args.go_timeout_min} min — exiting", flush=True)
