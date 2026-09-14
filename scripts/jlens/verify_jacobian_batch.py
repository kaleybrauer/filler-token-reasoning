"""
verify_jacobian_batch.py — does the layer-30 miss close when the check runs in the
regime the lens was fitted in?

At batch 1 the routing-frozen finite difference missed the stored J_30 by 30-45% while
agreeing with itself across step sizes and partitions (jacobian_partition.json).
Underflow is ruled out: prompts 10-19, fitted at cotangent scale 1 and 1/64 with the
same batch, agree to 0.3% at every layer. Leading explanation: the fit replicated each
prompt 64 times, so AWQ took its dequantise + matmul kernel (batch x seq >= 1024),
while the batch-1 check took the fused gemm kernel. Row-0 residual norms differ by
0.16-0.84% between the fit's recorded position_stats and the check, and V3's routing
changes experts under 0.1% nudges, so each side froze a different expert set.

A. Regime identity on the J prompt, no perturbation. Batch 1, batch 8 and batch 64
   copies without a graph, and batch 64 with the fit's setup (make_differentiable,
   block checkpointing, graph rooted at layer 0). Row-0 residual norms are compared
   with the fit's position_stats (computed the same way, on device), routing across
   regimes and across the 64 copies (each copy computed different rows of J), and the
   batch-1 target sum against the stock-model partition run, which says whether the
   patches leave the forward arithmetic untouched.
B. The confirmation. Routing-frozen central difference at batch 64, in whichever
   batch-64 mode reproduced the fit, with the same directions, steps and eps as the
   batch-1 checks. If the explanation is right, layer 30 drops to the 1-4% seen at
   50 and 59. Plus a swap: batch-1 activations with the batch-64 expert selection
   frozen in, which says how much of the miss is the expert selection alone.
C. Held-out WikiText residuals (default prompts 200..231) in the fit's kernel regime,
   in extract_wikitext_states.py's format, so lens readouts can be compared against
   wikitext_states.pt (batch 1) on CPU.

Forward passes only, no backward. Results are saved after each part.

    python scripts/jlens/verify_jacobian_batch.py
"""

from __future__ import annotations

import argparse
import contextlib
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
from extract_eval_states import check_unembed  # noqa: E402
from v3_autograd import (  # noqa: E402
    enable_block_checkpointing, load_v3_awq, make_differentiable)
from verify_jacobian_gpu import (  # noqa: E402
    FreezeRouting, Perturb, Routing, compare_routing_reachable)

FIT_SPANS = [(0, 100), (100, 115)]


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
                    help="same order and seed as the batch-1 checks, so every "
                         "(layer, direction) draws the same v")
    ap.add_argument("--n-directions", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rel-eps", default="0.1,0.03,0.01")
    ap.add_argument("--swap-eps", default="0.03")
    ap.add_argument("--part-b-regime", choices=["auto", "graph", "nograph"], default="auto",
                    help="auto: whichever batch-copies mode reproduces the fit's norms")
    ap.add_argument("--batch1-vectors", type=Path,
                    default=REPO / "outputs/jlens/jacobian_partition_vectors.pt")
    ap.add_argument("--heldout-start", type=int, default=200)
    ap.add_argument("--heldout-n", type=int, default=32)
    ap.add_argument("--skip-part-c", action="store_true")
    ap.add_argument("--max-seq-len", type=int, default=128)
    ap.add_argument("--skip-first", type=int, default=16)
    ap.add_argument("--max-memory-gib", default="130,122,108")
    ap.add_argument("--device-map", default="22,42", help="the fit's placement")
    ap.add_argument("--out", type=Path, default=REPO / "outputs/jlens/jacobian_batch.json")
    ap.add_argument("--states-out", type=Path,
                    default=REPO / "outputs/jlens/wikitext_states_fitregime.pt")
    args = ap.parse_args()

    # Everything that can fail without the model, before the model.
    jp = args.per_prompt_dir / f"J_p{args.prompt_index:04d}.pt"
    if not jp.exists():
        raise SystemExit(f"{jp} not found")
    meta = torch.load(jp, map_location="cpu", weights_only=False, mmap=True)
    fit_batch = int(meta["settings"]["dim_batch"])
    fit_norm = meta["position_stats"]["resid_norm"]          # {layer: [seq]}, row 0 of the fit
    b1_vec = (torch.load(args.batch1_vectors, map_location="cpu")
              if args.batch1_vectors.exists() else {})
    corpus = json.loads(args.prompts.read_text())
    prompt = corpus["prompts"][args.prompt_index]
    lo, hi = args.heldout_start, args.heldout_start + args.heldout_n
    if not args.skip_part_c and any(lo < b and a < hi for a, b in FIT_SPANS):
        raise SystemExit(f"held-out prompts {lo}..{hi - 1} overlap the fitted span")
    layers = ints(args.layers)
    print(f"J prompt {args.prompt_index} (fitted at dim_batch={fit_batch}), layers {layers}, "
          f"batch-1 vectors: {len(b1_vec)} saved", flush=True)

    model, tokenizer = load_v3_awq(args.model_path, max_memory_gib=args.max_memory_gib,
                                   device_map=args.device_map)
    lens_model = jlens.from_hf(model, tokenizer)
    patches = make_differentiable(model)                     # the fit's setup, as in refit_prompts.py
    patches["blocks_checkpointed"] = enable_block_checkpointing(model)
    print(f"  patches: {patches}  {lens_model}", flush=True)
    layers_mod, n_layers, d = lens_model.layers, lens_model.n_layers, lens_model.d_model
    target, all_layers = n_layers - 1, list(range(n_layers))

    ids = lens_model.encode(prompt, max_length=args.max_seq_len)
    seq = int(ids.shape[-1])
    pos = valid_position_mask(seq, skip_first=args.skip_first).nonzero(as_tuple=True)[0]
    n_valid, first_pos = int(pos.numel()), int(pos.min())
    if n_valid != meta["n_valid"]:
        raise SystemExit("valid-position count differs from the fit")

    def run(input_ids, batch, *, record=(), grad=False, delta_layer=None, delta=None,
            frozen=None, keep_routing=False):
        """One forward of `batch` copies. Returns, for row 0: the target sum over valid
        positions, {layer: residual [seq, d] fp16 cpu}, {layer: per-position norm
        computed on device exactly as the fit's position_stats}, and routing
        {block: [batch, seq, k]} when asked."""
        at = sorted({*record, target, *([delta_layer] if delta_layer is not None else [])})
        with contextlib.ExitStack() as stack:
            stack.enter_context(torch.enable_grad() if grad else torch.no_grad())
            rec = stack.enter_context(ActivationRecorder(
                layers_mod, at=at, start_graph_at=0 if grad else None))
            if frozen is not None:
                stack.enter_context(FreezeRouting(model, frozen))
            rt = stack.enter_context(Routing(model)) if keep_routing else None
            if delta is not None:
                stack.enter_context(Perturb(layers_mod[delta_layer], pos)).delta = delta
            lens_model.forward(input_ids.expand(batch, -1))
            tsum = rec.activations[target][0][pos].detach().float().sum(0).cpu()
            hs = {L: rec.activations[L][0].detach().half().cpu() for L in record}
            norms = {L: rec.activations[L][0].detach().float().norm(dim=-1).cpu() for L in record}
            routing = ({b: t.view(batch, -1, t.shape[-1]) for b, t in rt.snapshot_topk().items()}
                       if rt is not None else None)
        return tsum, hs, norms, routing

    t_start = time.time()
    report = dict(prompt_index=args.prompt_index, per_prompt_file=str(jp),
                  settings=meta["settings"], patches=patches, args=vars(args),
                  n_valid=n_valid, seq_len=seq, target_layer=target)

    def save():
        report["elapsed_s"] = round(time.time() - t_start, 1)
        args.out.write_text(json.dumps(report, indent=1, default=str))

    # ------------------------------------------------------------ A. regime identity
    regimes = {"b1": (1, False), "b8": (8, False), f"b{fit_batch}": (fit_batch, False),
               f"b{fit_batch}_fitgraph": (fit_batch, True)}
    R = {}
    for name, (batch, grad) in regimes.items():
        t0 = time.time()
        R[name] = run(ids, batch, record=all_layers, grad=grad, keep_routing=True)
        print(f"  A: {name} forward {time.time() - t0:.1f}s", flush=True)
    bfit, bgraph = f"b{fit_batch}", f"b{fit_batch}_fitgraph"
    A = {}
    for name in regimes:
        norms = R[name][2]
        rel = {L: float(((norms[L] - fit_norm[L].float()).abs() / fit_norm[L].float()).max())
               for L in all_layers}
        A[f"{name}_vs_fit_norms"] = dict(
            bitwise=all(torch.equal(norms[L], fit_norm[L].float()) for L in all_layers),
            max_rel=max(rel.values()),
            max_rel_at={L: rel[L] for L in layers},
            mean_signed_valid={L: float(((norms[L] - fit_norm[L].float())
                                         / fit_norm[L].float())[pos].mean()) for L in layers})
    base_key = f"base_target_sum/L{layers[0]}"
    if base_key in b1_vec:
        saved = b1_vec[base_key]
        A["patched_b1_vs_stock_b1_target_sum"] = dict(
            bitwise=bool(torch.equal(R["b1"][0], saved)),
            rel_diff=float((R["b1"][0] - saved).norm() / saved.norm()))
    row0 = lambda r: {b: t[0] for b, t in r.items()}
    cmp = lambda x, y: {str(L): compare_routing_reachable(row0(R[x][3]), row0(R[y][3]), L, first_pos)
                        for L in [-1, *layers]}
    A["routing_b1_vs_fit"] = cmp("b1", bfit)
    A["routing_b8_vs_fit"] = cmp("b8", bfit)
    A["routing_fitgraph_vs_nograph"] = cmp(bgraph, bfit)
    for name in (bfit, bgraph):
        rout = R[name][3]
        mism = sum(int((t[1:].sort(-1).values != t[:1].sort(-1).values).any(-1).sum())
                   for t in rout.values())
        A[f"{name}_copies_routing_mismatch"] = mism / sum(t[1:].shape[0] * t.shape[1]
                                                        for t in rout.values())
    h1, hfit = R["b1"][1], R[bfit][1]
    A["activation_rel_diff_b1_vs_fit"] = {
        L: float(((h1[L].float() - hfit[L].float())[pos].norm(dim=-1)
                  / hfit[L].float()[pos].norm(dim=-1)).mean()) for L in all_layers}
    A["b8_equals_fit_batch"] = all(torch.equal(R["b8"][1][L], hfit[L]) for L in all_layers)
    # The batch-64 mode that reproduces the fit decides part B.
    use_graph = (A[f"{bgraph}_vs_fit_norms"]["max_rel"] < A[f"{bfit}_vs_fit_norms"]["max_rel"]
                 if args.part_b_regime == "auto" else args.part_b_regime == "graph")
    bmode = bgraph if use_graph else bfit
    A["part_b_regime"] = bmode
    report["A"] = A
    save()
    for name in regimes:
        x = A[f"{name}_vs_fit_norms"]
        print(f"  A: {name:>14} vs fit norms: bitwise {x['bitwise']}  max rel {x['max_rel']:.2e}  "
              f"mean signed at {layers}: "
              + ", ".join(f"{v * 100:+.3f}%" for v in x["mean_signed_valid"].values()), flush=True)
    for key in ("routing_b1_vs_fit", "routing_b8_vs_fit", "routing_fitgraph_vs_nograph"):
        print(f"  A: {key}: tokens with a changed expert set, all blocks / from {layers}: "
              + " / ".join(f"{A[key][str(L)]['frac_tokens_set_changed'] * 100:.2f}%"
                           if A[key][str(L)] else "n/a" for L in [-1, *layers]), flush=True)
    print(f"  A: copies disagreeing with copy 0: {A[f'{bfit}_copies_routing_mismatch']:.2e} (no graph), "
          f"{A[f'{bgraph}_copies_routing_mismatch']:.2e} (fit graph); batch 8 == batch {fit_batch}: "
          f"{A['b8_equals_fit_batch']}; part B runs in {bmode}", flush=True)
    if "patched_b1_vs_stock_b1_target_sum" in A:
        print(f"  A: patched batch-1 target sum vs the stock partition run: "
              f"{A['patched_b1_vs_stock_b1_target_sum']}", flush=True)

    # ------------------------------------------------------------ B. confirmation
    batch_b, graph_b = regimes[bmode]
    routing_b = R[bmode][3]
    frozen_b = {b: t.reshape(-1, t.shape[-1]) for b, t in routing_b.items()}   # [batch*seq, k]
    frozen_swap = {b: t[0] for b, t in routing_b.items()}                     # row 0, [seq, k]
    vectors = {}
    B = []
    g = torch.Generator().manual_seed(args.seed)
    for L in layers:
        Jl = meta["J"][L].float()
        scale = float(h1[L][pos].float().norm(dim=-1).mean())                # the batch-1 checks' scale
        print(f"\nlayer {L}: batch-1 mean ||h_L|| = {scale:.2f}", flush=True)
        for di in range(args.n_directions):
            v = torch.randn(d, generator=g); v /= v.norm()
            pred = Jl @ v
            for rel in floats(args.rel_eps):
                eps, t0 = rel * scale, time.time()
                sp = run(ids, batch_b, grad=graph_b, delta_layer=L, delta=eps * v, frozen=frozen_b)[0]
                sm = run(ids, batch_b, grad=graph_b, delta_layer=L, delta=-eps * v, frozen=frozen_b)[0]
                D = (sp - sm) / (2 * eps * n_valid)
                vectors[f"fitregime/L{L}/d{di}/{rel}"] = D
                row = dict(kind="fit_regime", regime=bmode, layer=L, direction=di, rel_eps=rel,
                           eps=eps, **agreement(D, pred), secs=round(time.time() - t0, 1))
                b1 = b1_vec.get(f"full_frozen/L{L}/d{di}/{rel}")
                if b1 is not None:
                    row["rel_diff_vs_batch1_same_step"] = float((D - b1).norm() / b1.norm())
                B.append(row)
                print(f"  dir {di}  rel_eps {rel:<5} [{bmode}, frozen]  rel.err {row['rel_error']:8.4f}  "
                      f"cos {row['cosine']:7.4f}  |meas|/|pred| {row['norm_ratio']:6.3f}"
                      + (f"  | vs batch-1 D {row['rel_diff_vs_batch1_same_step']:.4f}" if b1 is not None else ""),
                      flush=True)
            for rel in floats(args.swap_eps):
                eps = rel * scale
                sp = run(ids, 1, delta_layer=L, delta=eps * v, frozen=frozen_swap)[0]
                sm = run(ids, 1, delta_layer=L, delta=-eps * v, frozen=frozen_swap)[0]
                D = (sp - sm) / (2 * eps * n_valid)
                vectors[f"swap/L{L}/d{di}/{rel}"] = D
                row = dict(kind="swap_b1_activations_fit_routing", layer=L, direction=di,
                           rel_eps=rel, eps=eps, **agreement(D, pred))
                B.append(row)
                print(f"  dir {di}  rel_eps {rel:<5} [batch 1, {bmode} experts frozen]  "
                      f"rel.err {row['rel_error']:8.4f}  cos {row['cosine']:7.4f}  "
                      f"|meas|/|pred| {row['norm_ratio']:6.3f}", flush=True)
            vectors[f"pred/L{L}/d{di}"] = pred
        del Jl
        report["B"] = B
        save()
        torch.save(vectors, args.out.with_name(args.out.stem + "_vectors.pt"))

    # ------------------------------------------------------------ C. held-out states
    if not args.skip_part_c:
        batch_c = 8 if A["b8_equals_fit_batch"] else fit_batch
        heldout = corpus["prompts"][lo:hi]
        T = args.max_seq_len - args.skip_first
        H = torch.empty(len(heldout), T, n_layers, d, dtype=torch.float16)
        kept_ids, seq_lens, t0 = [], [], time.time()
        print(f"\nC: {len(heldout)} held-out prompts {lo}..{hi - 1} at {batch_c} copies, no graph", flush=True)
        for i, p in enumerate(heldout):
            pid = lens_model.encode(p, max_length=args.max_seq_len)
            if int(pid.shape[-1]) < args.max_seq_len:
                raise SystemExit(f"prompt {lo + i} tokenised to {int(pid.shape[-1])} < {args.max_seq_len}")
            seq_lens.append(int(pid.shape[-1]))
            kept_ids.append(pid[0, args.skip_first:].cpu())
            hs = run(pid, batch_c, record=all_layers)[1]
            for L in all_layers:
                H[i, :, L] = hs[L][args.skip_first:]
            if (i + 1) % 8 == 0:
                print(f"  {i + 1}/{len(heldout)}  {(time.time() - t0) / (i + 1):.1f}s/prompt", flush=True)
        rms_eps = float(model.config.rms_norm_eps)
        states = {"model_path": args.model_path, "corpus": corpus["source"],
                  "prompt_span": [lo, hi], "held_out_from": FIT_SPANS, "n_layers": n_layers,
                  "layers": all_layers, "d_model": d, "rms_norm_eps": rms_eps,
                  "max_seq_len": args.max_seq_len, "skip_first": args.skip_first, "force_bos": True,
                  "seq_lens": seq_lens, "input_ids": torch.stack(kept_ids), "H": H,
                  "unembed_check": check_unembed(lens_model, H[:8, -1, :, :], rms_eps),
                  "regime": {"copies_per_prompt": batch_c, "graph": False,
                             "awq_kernel": "dequantize+matmul (batch*seq >= 1024)",
                             "device_map": args.device_map}}
        print(f"  G-UNEMBED: {states['unembed_check']}", flush=True)
        tmp = args.states_out.with_suffix(".tmp")
        torch.save(states, tmp)
        tmp.replace(args.states_out)
        report["C"] = dict(states_file=str(args.states_out), copies_per_prompt=batch_c,
                           prompt_span=[lo, hi], unembed_check=states["unembed_check"])
        save()
    print(f"\nwrote {args.out} in {(time.time() - t_start) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
