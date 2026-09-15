"""
spike_source.py — which fit prompts carry the early-layer readout spikes?

bilingual_diagnostics.py spikes found the layer-5 and layer-15 p99 kurtosis spikes under the half-B
lens and not under half A. The lens is a plain mean of per-prompt Jacobians, so whatever some prompts
contribute along a fixed direction enters every lens containing them with weight 1/N, and adds up
linearly over prompt blocks. Per layer:

  1. the spike's own pair of directions, defined without the half split:
                v_in  = mean unit residual h/|h| over the spike activations (the inputs where it occurs)
                w_out = RMSNorm weight * (mean unembedding row of the tokens the spikes read out - the
                        vocabulary's mean row), unit length (the readout it produces)
              and each part's push w_out^T S v_in, beside the same at control inputs (w_out^T S v_ctrl).
  2. for context, the top singular pairs of J_B - J_A. The top one need not be the spike: at layer 5
     it is the Latin-punctuation-vs-Han script axis, and a direction chosen to maximise the half
     difference separates the halves by construction, so its per-part split is partly selection.
  Parts: the clean prompt sum S of every block the fit checkpoints resolve (0-9, 10-19, 20-29,
  30-39, 40-59, 60-99, giants removed), each replacement prompt 100-111 (its own refitted Jacobian),
  and each excluded giant (in no lens). A lens's component is the sum over its parts / its prompt count.

Half A = 0-19, 30-59, replacements 100-107; half B = 20-29, 60-99, replacements 108-111 (clean_floor.py).
Only linear algebra on the stored Jacobians plus one unembedding pass per direction.

    python scripts/jlens/spike_source.py --layers 5 15
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_filtered_lens import LayerReader, logged_norms   # noqa: E402
from clean_floor import BlockLens, in_spans                  # noqa: E402

J_DIR = REPO / "outputs/jlens"
PP = J_DIR / "per_prompt"
T_FILTER = 40.0
LENSES = {  # lens -> parts (clean_floor.SUBSETS, spelled out in parts)
    "n50 (half A)": ["0-9", "10-19", "30-39", "40-59"] + [f"r{i}" for i in range(100, 108)],
    "n50b (half B)": ["20-29", "60-99"] + [f"r{i}" for i in range(108, 112)],
    "n61": ["0-9", "10-19", "20-29", "30-39", "40-59"] + [f"r{i}" for i in range(100, 112)],
    "n93": ["0-9", "10-19", "30-39", "40-59", "60-99"] + [f"r{i}" for i in range(100, 112)],
    "shipped (100)": ["0-9", "10-19", "20-29", "30-39", "40-59", "60-99"] + [f"r{i}" for i in range(100, 112)],
}


def parts():
    """name -> (half, n_prompts, reader(L) -> float64 clean prompt SUM at layer L)"""
    ck = {n: J_DIR / f"lens_v3.snap{n}.fitckpt" for n in (10, 20, 30, 40, 60)}
    ck[100] = J_DIR / "lens_v3.fitckpt"
    blocks = {"0-9": (ck[10], None), "10-19": (ck[20], ck[10]), "20-29": (ck[30], ck[20]),
              "30-39": (ck[40], ck[30]), "40-59": (ck[60], ck[40]), "60-99": (ck[100], ck[60])}
    giants = {i: PP / f"J_p{i:04d}.pt" for i, v in logged_norms().items() if v > T_FILTER}
    out = {}
    for name, (hi, lo) in blocks.items():
        span = [(LayerReader(lo, "jacobian_sum").n_done if lo else 0, LayerReader(hi, "jacobian_sum").n_done)]
        bl = BlockLens([(hi, lo)], {i: p for i, p in giants.items() if in_spans(i, span)}, None, name)
        out[name] = ("B" if name in ("20-29", "60-99") else "A", bl.n,
                     lambda L, bl=bl: bl[L].astype(np.float64) * bl.n)
    for i in range(100, 112):
        r = LayerReader(PP / f"J_p{i:04d}.pt", "J")
        out[f"r{i}"] = ("A" if i < 108 else "B", 1, lambda L, r=r: r.read(L).astype(np.float64))
    excluded = {f"giant {i}": ("excluded", 1, lambda L, r=LayerReader(p, "J"): r.read(L).astype(np.float64))
                for i, p in sorted(giants.items())}
    return out, excluded


def top_singular(D, k=3, iters=60, seed=0):
    """Top-k singular triplets by subspace iteration (D is d x d, k tiny)."""
    Q = np.linalg.qr(np.random.default_rng(seed).standard_normal((D.shape[1], k)))[0]
    for _ in range(iters):
        Q = np.linalg.qr(D.T @ (D @ Q))[0]
    Ub, s, Wt = np.linalg.svd(D @ Q, full_matrices=False)
    return Ub, s, Q @ Wt.T


def unembedding():
    from score_eval_states import WEIGHTS
    W = np.load(WEIGHTS / "lm_head_weight.npy", mmap_mode="r")
    g = np.load(WEIGHTS / "rms_norm_weight.npy").astype(np.float64)
    mean_row = sum(np.asarray(W[s:s + 16384], np.float64).sum(0) for s in range(0, W.shape[0], 16384)) / W.shape[0]
    return W, g, mean_row


def unembed_direction(u, pieces, cls, classes, k=15):
    W, g, _ = UNEMBED
    gu = (g * u).astype(np.float32)
    z = np.concatenate([np.asarray(W[s:s + 16384], np.float32) @ gu for s in range(0, W.shape[0], 16384)])
    c = z - z.mean()
    kurt = float((c ** 4).mean() / (c ** 2).mean() ** 2 - 3)
    out = {"excess_kurtosis": kurt}
    for side, order in (("top", np.argsort(-z)), ("bottom", np.argsort(z))):
        ids = order[:50]
        out[side] = [pieces[i] for i in ids[:k]]
        out[f"{side}50_classes"] = {classes[j]: int((cls[ids] == j).sum()) for j in np.unique(cls[ids])}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", nargs="+", type=int, default=[5, 15])
    ap.add_argument("--spikes-json", type=Path, default=J_DIR / "bilingual_spikes_n61_n93.json")
    ap.add_argument("--n-control", type=int, default=2000)
    args = ap.parse_args()
    import torch
    global UNEMBED
    UNEMBED = unembedding()
    vocab = json.loads((J_DIR / "vocab_scripts.json").read_text())
    pieces, cls, classes = vocab["pieces"], np.array(vocab["cls"]), vocab["classes"]
    prts, excluded = parts()
    spikes = json.loads(args.spikes_json.read_text()) if args.spikes_json.exists() else None
    S_states = torch.load(J_DIR / "wikitext_states.pt", map_location="cpu", mmap=True, weights_only=False)
    H = S_states["H"]
    report = {}
    for L in args.layers:
        t0 = time.time()
        SA = SB = None
        for name, (half, n, read) in prts.items():
            S = read(L)
            if half == "A":
                SA = S if SA is None else SA + S
            else:
                SB = S if SB is None else SB + S
        nA = sum(n for h, n, _ in prts.values() if h == "A")
        nB = sum(n for h, n, _ in prts.values() if h == "B")
        J_ship = ((SA + SB) / (nA + nB)).astype(np.float32)
        D = SB / nB - SA / nA
        del SA, SB
        U, s, V = top_singular(D)
        fro2 = float(np.einsum("ij,ij->", D, D))
        u, v = U[:, 0], V[:, 0]
        lay = {"n_prompts": {"A": nA, "B": nB}, "sigma_top3": s.tolist(),
               "share_of_difference_top1": float(s[0] ** 2 / fro2),
               "share_of_difference_top3": float((s ** 2).sum() / fro2)}
        del D
        print(f"\n=== layer {L} ===  halves n={nA}/{nB};  J_B - J_A top singular values {np.round(s, 2).tolist()}; "
              f"top-1 holds {lay['share_of_difference_top1']:.1%} of ||J_B - J_A||^2, top-3 "
              f"{lay['share_of_difference_top3']:.1%}", flush=True)

        # singular pairs come oriented so u^T (J_B - J_A) v = sigma1 > 0: half B's excess is +u
        lay["u_unembedding"] = ue = unembed_direction(u, pieces, cls, classes)
        print(f"  u unembeds to (excess kurtosis {ue['excess_kurtosis']:.1f}):\n"
              f"    + side: {' | '.join(repr(x)[1:-1] for x in ue['top'])}   {ue['top50_classes']}\n"
              f"    - side: {' | '.join(repr(x)[1:-1] for x in ue['bottom'])}   {ue['bottom50_classes']}")

        spike_dirs = None
        if spikes and str(L) in spikes["layers"]:
            sl = spikes["layers"][str(L)]
            sp, T = np.array(sl["spike_flat_indices"]), sl["n_positions"]
            ctrl = np.random.default_rng(L).choice(np.setdiff1d(np.arange(H.shape[0] * T), sp),
                                                   args.n_control, replace=False)

            def unit_mean(idx):
                p, t = np.divmod(idx, T)
                h = H[torch.from_numpy(p), torch.from_numpy(t), L].float().numpy().astype(np.float64)
                m = (h / np.linalg.norm(h, axis=1, keepdims=True)).mean(0)
                return m / np.linalg.norm(m)

            v_in, v_ctrl = unit_mean(sp), unit_mean(ctrl)
            first_id = {}
            for i, pc in enumerate(pieces):
                first_id.setdefault(pc, i)
            toks = [pc for pc, _ in sl["readout_top10_frequent"][:20]]
            W, g, mean_row = UNEMBED
            ids = [first_id[pc] for pc in toks]
            w_out = g * (np.asarray(W[sorted(ids)], np.float64).mean(0) - mean_row)
            w_out /= np.linalg.norm(w_out)
            spike_dirs = (w_out, v_in, v_ctrl)
            lay["spike_directions"] = {"readout_tokens": toks, "cos_v_in_v_ctrl": float(v_in @ v_ctrl),
                                       "cos_w_out_u_svd": float(w_out @ (g * u) / np.linalg.norm(g * u)),
                                       "cos_v_in_v_svd": float(v_in @ v)}
            print(f"  spike directions: readout tokens {' | '.join(repr(x)[1:-1] for x in toks[:12])};  "
                  f"cos(v_in, v_ctrl) {v_in @ v_ctrl:.2f}; overlap with the top singular pair: "
                  f"|cos| out {abs(lay['spike_directions']['cos_w_out_u_svd']):.2f}, in {abs(v_in @ v):.2f}")

        rows = {}
        for name, (half, n, read) in {**prts, **excluded}.items():
            S = read(L)
            r = {"half": half, "n": n, "c_svd": float(u @ S @ v)}
            if spike_dirs:
                w_out, v_in, v_ctrl = spike_dirs
                r["c_spike"], r["c_control"] = float(w_out @ S @ v_in), float(w_out @ S @ v_ctrl)
            rows[name] = r
        cB_minus_cA = (sum(r["c_svd"] for r in rows.values() if r["half"] == "B") / nB
                       - sum(r["c_svd"] for r in rows.values() if r["half"] == "A") / nA)
        lay["parts"] = rows
        lay["check_sigma1_equals_uDv"] = [float(s[0]), float(cB_minus_cA)]
        keys = ["c_spike", "c_control", "c_svd"] if spike_dirs else ["c_svd"]
        lay["lens_component"] = {k: {key: sum(rows[p][key] for p in ps) / sum(rows[p]["n"] for p in ps)
                                     for key in keys} for k, ps in LENSES.items()}
        print(f"  per part, per prompt (sum / n): spike = w_out^T S v_in, control = w_out^T S v_ctrl, "
              f"svd = u^T S v   [check sigma1 {s[0]:.3f} = {cB_minus_cA:.3f}]")
        print(f"  {'part':>12} {'half':>9} {'n':>3} " + " ".join(f"{key[2:]:>10}" for key in keys))
        for name, r in rows.items():
            print(f"  {name:>12} {r['half']:>9} {r['n']:3d} " + " ".join(f"{r[key] / r['n']:10.3f}" for key in keys))
        print("  component in each lens: " + ";  ".join(
            f"{k} " + "/".join(f"{val:.3f}" for val in comp.values()) for k, comp in lay["lens_component"].items())
            + f"   ({' / '.join(key[2:] for key in keys)})")
        lay["secs"] = round(time.time() - t0, 1)
        report[L] = lay
        del J_ship
    out = J_DIR / "spike_source.json"
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
