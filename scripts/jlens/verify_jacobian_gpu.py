"""
verify_jacobian_gpu.py — two GPU checks that ride one model load, answering the two
reviewer caveats that cannot be settled on CPU.

CHECK 1 — is the backward the true Jacobian of the QUANTIZED forward?
    We validated the AWQ backward against a finite difference of its own fused forward
    on single projections (agreement 2.07e-4), but that is a per-layer weight-level
    check. This is the end-to-end version the reviewer asked for: a directional
    derivative through the whole forward computation, compared against the per-prompt
    Jacobian actually stored on disk.

    The estimator is J_L = mean over valid source positions t of the sum over target
    positions t' >= t of dh_final,t'/dh_L,t. Perturb h_L by delta at EVERY valid source
    position at once and the response is, to first order,

        sum_{t' valid} delta_h_final,t'  =  n_valid * J_L delta

    because causality kills the t' < t terms, so the double sum over the (contiguous)
    valid set is exactly the estimator's masked one. A central difference in delta then
    gives J_L delta directly, with no backward involved -- an independent estimator, so
    agreement is evidence about the gradient path and not a restatement of it.

CHECK 2 — how local is the Jacobian to the current expert routing?
    The straight-through treatment of top-k means J holds routing fixed. The same
    perturbed forwards report what fraction of (MoE layer, position) routing decisions
    actually change at each step size. If routing is stable at the step where the
    derivative check is clean, the locality concern is mild; if it flips often, J is
    only valid in a very small neighbourhood and that has to be said.

Forward passes only: no autograd patches, no fitting, no backward.
Needs a per-prompt Jacobian on disk (outputs/jlens/per_prompt/J_p{idx:04d}.pt).

    python scripts/jlens/verify_jacobian_gpu.py --prompt-index 100 --layers 30,50,59
"""

from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch

import jlens  # noqa: E402  (must precede REPO_ROOT/scripts on sys.path)
from jlens.fitting import valid_position_mask  # noqa: E402

if "filler-token-reasoning" in (jlens.__file__ or "scripts/jlens namespace pkg"):
    raise SystemExit(f"scripts/jlens shadowed the jlens library: {jlens.__file__}")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from v3_autograd import load_v3_awq  # noqa: E402


class Perturb:
    """Add `delta` to a decoder block's output at the given positions."""

    def __init__(self, layer_module, positions):
        self.mod, self.pos, self.delta, self.h = layer_module, positions, None, None

    def __enter__(self):
        def hook(_m, _a, out):
            if self.delta is None:
                return None
            hs = out[0] if isinstance(out, tuple) else out
            hs = hs.clone()
            hs[:, self.pos, :] += self.delta.to(hs.dtype).to(hs.device)
            return (hs,) + tuple(out[1:]) if isinstance(out, tuple) else hs
        self.h = self.mod.register_forward_hook(hook)
        return self

    def __exit__(self, *a):
        self.h.remove()


class Routing:
    """Capture top-k expert indices from every MoE gate in the model."""

    def __init__(self, model):
        self.gates = [m for n, m in model.named_modules() if n.endswith("mlp.gate")]
        self.idx, self.hs = {}, []

    def __enter__(self):
        for i, g in enumerate(self.gates):
            def hook(_m, _a, out, i=i):
                t = out[0] if isinstance(out, tuple) else out
                if torch.is_tensor(t):
                    self.idx[i] = t.detach().flatten().cpu()
            self.hs.append(g.register_forward_hook(hook))
        return self

    def __exit__(self, *a):
        for h in self.hs:
            h.remove()

    def snapshot(self):
        return {k: v.clone() for k, v in self.idx.items()}


def compare_routing(a, b):
    keys = sorted(set(a) & set(b))
    if not keys:
        return None
    diff = tot = 0
    for k in keys:
        x, y = a[k], b[k]
        n = min(x.numel(), y.numel())
        diff += int((x[:n] != y[:n]).sum()); tot += n
    return dict(n_moe_layers=len(keys), n_decisions=tot,
                frac_changed=diff / tot if tot else float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="/workspace/models/deepseek-v3-awq")
    ap.add_argument("--prompts", type=Path, default=REPO / "scripts/jlens/prompts_wikitext.json")
    ap.add_argument("--per-prompt-dir", type=Path, default=REPO / "outputs/jlens/per_prompt")
    ap.add_argument("--prompt-index", type=int, default=100,
                    help="corpus index; needs J_p{idx:04d}.pt on disk")
    ap.add_argument("--layers", default="30,50,59")
    ap.add_argument("--rel-eps", default="0.3,0.1,0.03,0.01",
                    help="step sizes as a fraction of the mean residual norm at that layer")
    ap.add_argument("--n-directions", type=int, default=2)
    ap.add_argument("--max-seq-len", type=int, default=128)
    ap.add_argument("--skip-first", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-memory-gib", default="130,122,108")
    ap.add_argument("--device-map", default=None)
    ap.add_argument("--out", type=Path, default=REPO / "outputs/jlens/jacobian_verification.json")
    args = ap.parse_args()

    jp = args.per_prompt_dir / f"J_p{args.prompt_index:04d}.pt"
    if not jp.exists():
        raise SystemExit(f"{jp} not found — pick a prompt index that was refitted")
    meta = torch.load(jp, map_location="cpu", weights_only=False, mmap=True)
    print(f"per-prompt Jacobian {jp.name}: max||J||/sqrt(d)="
          f"{meta['max_norm_over_sqrt_d']:.3f}  n_valid={meta['n_valid']}  "
          f"settings={meta['settings']}", flush=True)

    prompt = json.loads(args.prompts.read_text())["prompts"][args.prompt_index]
    model, tokenizer = load_v3_awq(args.model_path, max_memory_gib=args.max_memory_gib,
                                   device_map=args.device_map)
    lens_model = jlens.from_hf(model, tokenizer)   # forward-only
    layers_mod = lens_model.layers
    target = lens_model.n_layers - 1
    d = lens_model.d_model

    ids = lens_model.encode(prompt, max_length=args.max_seq_len)
    seq = int(ids.shape[-1])
    mask = valid_position_mask(seq, skip_first=args.skip_first)
    pos = mask.nonzero(as_tuple=True)[0]
    n_valid = int(mask.sum())
    print(f"prompt {args.prompt_index}: seq_len={seq}, n_valid={n_valid} "
          f"(stored n_valid={meta['n_valid']})", flush=True)
    if n_valid != meta["n_valid"]:
        print("  !! valid-position count differs from the fit — check skip_first/max_seq_len")

    from jlens.hooks import ActivationRecorder

    def run(layer, delta):
        """(sum of h_target over valid positions, residual at `layer`, routing)."""
        with torch.no_grad(), ActivationRecorder(layers_mod, at=[layer, target]) as rec, \
             Routing(model) as rt, Perturb(layers_mod[layer], pos) as pt:
            pt.delta = delta
            lens_model.forward(ids)
            h_t = rec.activations[target][0][pos].float()
            h_l = rec.activations[layer][0][pos].float()
            return h_t.sum(0).cpu(), h_l.cpu(), rt.snapshot()

    g = torch.Generator().manual_seed(args.seed)
    rows = []
    for L in [int(x) for x in args.layers.split(",")]:
        Jl = meta["J"][L].float()                       # [d, d]
        _, h_l, route0 = run(L, None)
        scale = float(h_l.norm(dim=-1).mean())
        print(f"\nlayer {L}: mean ||h_L|| over valid positions = {scale:.2f}", flush=True)
        for di in range(args.n_directions):
            v = torch.randn(d, generator=g); v /= v.norm()
            pred_unit = (Jl @ v)                        # J_L v, the predicted response
            for rel in [float(x) for x in args.rel_eps.split(",")]:
                eps = rel * scale
                t0 = time.time()
                sp, _, rp = run(L, eps * v)
                sm, _, rm = run(L, -eps * v)
                meas = (sp - sm) / (2 * eps * n_valid)  # -> J_L v in the linear limit
                err = float((meas - pred_unit).norm() / (pred_unit.norm() + 1e-12))
                cos = float(torch.nn.functional.cosine_similarity(
                    meas[None, :], pred_unit[None, :]).item())
                ratio = float(meas.norm() / (pred_unit.norm() + 1e-12))
                rt = compare_routing(route0, rp)
                rows.append(dict(layer=L, direction=di, rel_eps=rel, eps=eps,
                                 rel_error=err, cosine=cos, norm_ratio=ratio,
                                 routing=rt, secs=round(time.time() - t0, 1)))
                print(f"  dir {di}  rel_eps {rel:<5} eps {eps:8.2f}  "
                      f"rel.err {err:8.4f}  cos {cos:7.4f}  |meas|/|pred| {ratio:6.3f}"
                      + (f"  routing changed {rt['frac_changed']*100:5.2f}%"
                         if rt else "  routing n/a"), flush=True)
        del Jl

    out = dict(prompt_index=args.prompt_index, per_prompt_file=str(jp),
               settings=meta["settings"], n_valid=n_valid, seq_len=seq,
               target_layer=target, d_model=d, results=rows)
    args.out.write_text(json.dumps(out, indent=1, default=str))
    print(f"\nwrote {args.out}")
    best = min(rows, key=lambda r: r["rel_error"])
    print(f"BEST agreement: layer {best['layer']} rel_eps {best['rel_eps']} "
          f"-> relative error {best['rel_error']:.4f}, cosine {best['cosine']:.4f}")
    print("Read it as: the error should fall as the step shrinks and then flatten at the "
          "fp16 floor. A flat-from-the-start curve, or a norm ratio far from 1, means the "
          "stored Jacobian is not the derivative of this forward computation.")


if __name__ == "__main__":
    main()
