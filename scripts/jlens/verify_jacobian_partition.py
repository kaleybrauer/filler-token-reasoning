"""
verify_jacobian_partition.py — is layer 30's flat finite-difference floor linearization
error, or does the stored J_L disagree with the derivative of this forward?

verify_jacobian_gpu.py perturbed h_L by eps*v at all n = 111 valid positions at once.
The objection: that coherent perturbation spans every position, so the sweep may never
have been linear even at step 0.01. Routing is frozen for every comparison against J
here (FreezeRouting), so expert flips cannot enter.

Partition test. Split the valid positions into m random groups and perturb one group per
run. By linearity the m responses, summed, reconstruct the full estimator exactly, with
no sampling error:

    sum_g sum_{t' valid} [h_t'(+eps v on g) - h_t'(-eps v on g)] / (2 eps n)  ->  J_L v

Each run now perturbs n/m positions. If the reconstruction's error falls as m grows, the
floor was linearization error; if it stays at the m = 1 value, the stored J_L is not the
derivative of this forward. The per-group ratio R_g / |g| is the random-subset estimator:
it targets the mean Jacobian over g's positions rather than over all n, so its error
against J_L v also carries a sampling spread that grows as |g| shrinks. It is reported,
but only the reconstruction isolates linearization.

Also recorded: steps 0.003 and 0.001 on every layer (routing free and frozen), and every
finite-difference vector, saved beside the JSON, so self-consistency between steps,
||D(eps_i) - D(eps_j)||, can be checked offline without any J.

Forward passes only: no autograd patches, no fitting, no backward.

    python scripts/jlens/verify_jacobian_partition.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

import jlens  # noqa: E402  (must precede REPO_ROOT/scripts on sys.path)
from jlens.fitting import valid_position_mask  # noqa: E402
from jlens.hooks import ActivationRecorder  # noqa: E402

if "filler-token-reasoning" in (jlens.__file__ or "scripts/jlens namespace pkg"):
    raise SystemExit(f"scripts/jlens shadowed the jlens library: {jlens.__file__}")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from v3_autograd import load_v3_awq  # noqa: E402
from verify_jacobian_gpu import (  # noqa: E402
    FreezeRouting, Perturb, Routing, compare_routing_reachable)


def floats(s):
    return [float(x) for x in s.split(",") if x]


def ints(s):
    return [int(x) for x in s.split(",") if x]


def agreement(meas, pred):
    return dict(rel_error=float((meas - pred).norm() / (pred.norm() + 1e-12)),
                cosine=float(torch.nn.functional.cosine_similarity(
                    meas[None, :], pred[None, :]).item()),
                norm_ratio=float(meas.norm() / (pred.norm() + 1e-12)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="/workspace/models/deepseek-v3-awq")
    ap.add_argument("--prompts", type=Path, default=REPO / "scripts/jlens/prompts_wikitext.json")
    ap.add_argument("--per-prompt-dir", type=Path, default=REPO / "outputs/jlens/per_prompt")
    ap.add_argument("--prompt-index", type=int, default=100)
    ap.add_argument("--layers", default="30,50,59",
                    help="same order and seed as verify_jacobian_gpu.py, so every "
                         "(layer, direction) draws the same v as that run")
    ap.add_argument("--n-directions", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sweep-layers", default="30",
                    help="layers that get the full routing-frozen step sweep")
    ap.add_argument("--sweep-eps", default="0.3,0.1,0.03,0.01,0.003,0.001")
    ap.add_argument("--small-eps", default="0.003,0.001",
                    help="steps run on every layer, with routing free and frozen")
    ap.add_argument("--partition-layers", default="30")
    ap.add_argument("--partition-eps", default="0.1,0.03")
    ap.add_argument("--partition-m", default="4,16")
    ap.add_argument("--partition-seed", type=int, default=1)
    ap.add_argument("--max-seq-len", type=int, default=128)
    ap.add_argument("--skip-first", type=int, default=16)
    ap.add_argument("--max-memory-gib", default="130,122,108")
    ap.add_argument("--device-map", default=None)
    ap.add_argument("--out", type=Path, default=REPO / "outputs/jlens/jacobian_partition.json")
    args = ap.parse_args()

    jp = args.per_prompt_dir / f"J_p{args.prompt_index:04d}.pt"
    if not jp.exists():
        raise SystemExit(f"{jp} not found — pick a prompt index that was refitted")
    meta = torch.load(jp, map_location="cpu", weights_only=False, mmap=True)
    prompt = json.loads(args.prompts.read_text())["prompts"][args.prompt_index]
    vec_path = args.out.with_name(args.out.stem + "_vectors.pt")

    model, tokenizer = load_v3_awq(args.model_path, max_memory_gib=args.max_memory_gib,
                                   device_map=args.device_map)
    lens_model = jlens.from_hf(model, tokenizer)   # forward-only
    layers_mod, target, d = lens_model.layers, lens_model.n_layers - 1, lens_model.d_model

    ids = lens_model.encode(prompt, max_length=args.max_seq_len)
    seq = int(ids.shape[-1])
    pos = valid_position_mask(seq, skip_first=args.skip_first).nonzero(as_tuple=True)[0]
    n_valid, first_pos = int(pos.numel()), int(pos.min())
    print(f"prompt {args.prompt_index}: seq_len={seq}, n_valid={n_valid} "
          f"(stored n_valid={meta['n_valid']})", flush=True)
    if n_valid != meta["n_valid"]:
        raise SystemExit("valid-position count differs from the fit")

    def run(layer, delta, where):
        """(sum of h_target over ALL valid positions, h_layer at valid positions, top-k),
        with h_layer perturbed by delta at positions `where`."""
        with torch.no_grad(), ActivationRecorder(layers_mod, at=[layer, target]) as rec, \
             Routing(model) as rt, Perturb(layers_mod[layer], where) as pt:
            pt.delta = delta
            lens_model.forward(ids)
            return (rec.activations[target][0][pos].float().sum(0).cpu(),
                    rec.activations[layer][0][pos].float().cpu(), rt.snapshot_topk())

    def central(layer, v, eps, where, frozen=None):
        """Central difference of the target sum; ~ |where| * J_where v in the linear limit."""
        if frozen is None:
            (sp, _, tp), (sm, _, tm) = run(layer, eps * v, where), run(layer, -eps * v, where)
        else:
            with FreezeRouting(model, frozen):
                (sp, _, tp), (sm, _, tm) = run(layer, eps * v, where), run(layer, -eps * v, where)
        return (sp - sm) / (2 * eps), tp, tm

    def save():
        out = dict(prompt_index=args.prompt_index, per_prompt_file=str(jp),
                   settings=meta["settings"], n_valid=n_valid, seq_len=seq,
                   target_layer=target, d_model=d, args=vars(args), vectors_file=str(vec_path),
                   elapsed_s=round(time.time() - t_start, 1), results=rows)
        args.out.write_text(json.dumps(out, indent=1, default=str))
        torch.save(vectors, vec_path)

    g = torch.Generator().manual_seed(args.seed)
    pg = torch.Generator().manual_seed(args.partition_seed)
    sweep_layers, part_layers = set(ints(args.sweep_layers)), set(ints(args.partition_layers))
    small = floats(args.small_eps)
    rows, vectors, t_start = [], {}, time.time()
    for L in ints(args.layers):
        Jl = meta["J"][L].float()                       # [d, d]
        base, h_l, topk0 = run(L, None, pos)
        scale = float(h_l.norm(dim=-1).mean())
        vectors[f"base_target_sum/L{L}"] = base
        print(f"\nlayer {L}: mean ||h_L|| over valid positions = {scale:.2f}", flush=True)
        for di in range(args.n_directions):
            v = torch.randn(d, generator=g); v /= v.norm()
            pred = Jl @ v                               # J_L v, the predicted response
            vectors[f"v/L{L}/d{di}"], vectors[f"pred/L{L}/d{di}"] = v, pred
            steps = floats(args.sweep_eps) if L in sweep_layers else small
            for rel in steps:
                eps = rel * scale
                R, _, _ = central(L, v, eps, pos, frozen=topk0)
                a = agreement(R / n_valid, pred)
                vectors[f"full_frozen/L{L}/d{di}/{rel}"] = R / n_valid
                rows.append(dict(kind="full", routing_frozen=True, layer=L, direction=di,
                                 rel_eps=rel, eps=eps, **a))
                print(f"  dir {di}  rel_eps {rel:<6} all {n_valid} positions  [frozen]  "
                      f"rel.err {a['rel_error']:8.4f}  cos {a['cosine']:7.4f}  "
                      f"|meas|/|pred| {a['norm_ratio']:6.3f}", flush=True)
                if rel in small:
                    R, tp, tm = central(L, v, eps, pos)
                    a = agreement(R / n_valid, pred)
                    rp = compare_routing_reachable(topk0, tp, L, first_pos)
                    rm = compare_routing_reachable(topk0, tm, L, first_pos)
                    vectors[f"full_free/L{L}/d{di}/{rel}"] = R / n_valid
                    rows.append(dict(kind="full", routing_frozen=False, layer=L, direction=di,
                                     rel_eps=rel, eps=eps, **a,
                                     routing_reachable_plus=rp, routing_reachable_minus=rm))
                    fmt = lambda r: (f"{r['frac_tokens_set_changed']*100:5.2f}% tok "
                                     f"{r['frac_experts_replaced']*100:5.2f}% exp") if r else "n/a"
                    print(f"  dir {di}  rel_eps {rel:<6} all {n_valid} positions  [free]    "
                          f"rel.err {a['rel_error']:8.4f}  cos {a['cosine']:7.4f}  "
                          f"|meas|/|pred| {a['norm_ratio']:6.3f}  | reachable +eps {fmt(rp)}"
                          f"  -eps {fmt(rm)}", flush=True)
            if L not in part_layers:
                continue
            for rel in floats(args.partition_eps):
                eps = rel * scale
                for m in ints(args.partition_m):
                    perm = pos[torch.randperm(n_valid, generator=pg)]
                    groups = [grp.sort().values for grp in torch.tensor_split(perm, m)]
                    total, per_group = torch.zeros(d), []
                    for gi, grp in enumerate(groups):
                        R, _, _ = central(L, v, eps, grp, frozen=topk0)
                        total += R
                        vectors[f"partition/L{L}/d{di}/{rel}/m{m}/g{gi}"] = R
                        per_group.append(dict(group=gi, positions=grp.tolist(),
                                              **agreement(R / grp.numel(), pred)))
                    a = agreement(total / n_valid, pred)
                    sub = [x["rel_error"] for x in per_group]
                    sizes = [len(x["positions"]) for x in per_group]
                    rows.append(dict(kind="partition", routing_frozen=True, layer=L,
                                     direction=di, rel_eps=rel, eps=eps, m=m, **a,
                                     subset_rel_error_mean=sum(sub) / len(sub),
                                     subset_rel_error_min=min(sub),
                                     subset_rel_error_max=max(sub), groups=per_group))
                    print(f"  dir {di}  rel_eps {rel:<6} m {m:>2} groups of {min(sizes)}-"
                          f"{max(sizes)}  [frozen]  reconstructed rel.err {a['rel_error']:8.4f}"
                          f"  cos {a['cosine']:7.4f}  |meas|/|pred| {a['norm_ratio']:6.3f}"
                          f"  | per-group R_g/|g| rel.err mean {sum(sub)/len(sub):.4f} "
                          f"[{min(sub):.4f}-{max(sub):.4f}]", flush=True)
        del Jl
        save()   # after every layer, so a late failure keeps what was measured

    save()
    print(f"\nwrote {args.out} and {vec_path} in {(time.time() - t_start)/60:.1f} min")


if __name__ == "__main__":
    main()
