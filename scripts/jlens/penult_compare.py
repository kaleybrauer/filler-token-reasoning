"""
penult_compare.py — does V3's script offset survive a penultimate-layer target?  (PENULT_RUNBOOK.md)

Reads what the runbook's CPU steps wrote and prints one table:
  - bilingual_depth_{target59,target60_same,shipped,logit}.json  (bilingual_diagnostics.py depth):
    script-class eta^2, Han-minus-Latin offset and full-vocabulary kurtosis, per layer;
  - the two mean lenses and the shipped lens: share of |J|_F^2 in the top output direction, whether
    that direction is still the shipped lens's script axis, and the gain along the shipped axis
    relative to a typical direction (|u^T J| / (|J|_F / sqrt d));
  - per-prompt max||J||_F/sqrt(d) under both targets (does the target change the heavy tail?).

The two mean lenses must be built from the SAME prompts (same indices, same prompt hashes): the
comparison is paired, and a few-prompt lens is only comparable with a lens of the same few prompts.

Pre-specified reading (fixed in PENULT_RUNBOOK.md before the GPU run), at layers 20, 30 and 45
(33, 50 and 75% depth; a 5-prompt lens is spiky below ~30% depth, so L18 and earlier are shown but
not used):
  gone      eta^2(target 59) <= 0.03 at a layer -- the Qwen3.5-122B lens's level (<= 0.02); the logit lens is 0.003
  survives  eta^2(target 59) >= half of the same prompts' target-60 value
  reduced   anything in between

    python scripts/jlens/penult_compare.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import models  # noqa: E402
from score_lazy import LazyLens  # noqa: E402

GONE = 0.03


def top_pair(J, iters=80, seed=0):
    v = np.random.default_rng(seed).standard_normal(J.shape[1]).astype(np.float32)
    for _ in range(iters):
        v = J.T @ (J @ v)
        v /= np.linalg.norm(v)
    u = J @ v
    s = float(np.linalg.norm(u))
    return s, u / s


def depth_rows(path: Path) -> dict[int, dict]:
    return {r["layer"]: r for r in json.loads(path.read_text())["per_layer"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="target59", help="models.py alt-lens name of the new-target lens")
    ap.add_argument("--b", default="target60_same", help="alt-lens name of the same prompts' target-60 lens")
    ap.add_argument("--depth-dir", type=Path, default=models.J)
    ap.add_argument("--tag", default="", help="--tag given to bilingual_diagnostics.py depth, if any")
    ap.add_argument("--verdict-layers", nargs="+", type=int, default=[20, 30, 45])
    ap.add_argument("--share-layers", nargs="+", type=int, default=[0, 20, 30, 45, 58])
    ap.add_argument("--out", type=Path, default=models.J / "penult_compare.json")
    args = ap.parse_args()

    m = models.get("v3")
    paths = {args.a: m["alt_lenses"][args.a], args.b: m["alt_lenses"][args.b], "shipped": m["lens"]}
    meta = {k: json.loads(paths[k].with_suffix(".meta.json").read_text()) for k in (args.a, args.b)}
    ia, ib = meta[args.a]["indices"], meta[args.b]["indices"]
    sha = lambda k: {i: p["prompt_sha1"] for i, p in meta[k]["prompts"].items()}  # noqa: E731
    if ia != ib or sha(args.a) != sha(args.b):
        raise SystemExit(f"the two lenses are not built from the same prompts: {args.a} {ia} vs {args.b} {ib}. "
                         f"Rebuild both with build_mean_lens.py --indices {','.join(map(str, sorted(set(ia) & set(ib))))}")
    print(f"{args.a}: target layer {meta[args.a]['target_layer']}, n={len(ia)} prompts {ia}\n"
          f"{args.b}: target layer {meta[args.b]['target_layer']}, same prompts\n")

    # per-prompt norms under the two targets
    print("per-prompt max||J||_F/sqrt(d)   (shipped-lens pre-filter: 40)")
    norms = {}
    for i in map(str, ia):
        na, nb = (meta[k]["prompts"][i]["max_norm_over_sqrt_d"] for k in (args.a, args.b))
        norms[i] = {args.a: na, args.b: nb}
        print(f"  idx {i:>4}: target {meta[args.a]['target_layer']} {na:9.3f}   target {meta[args.b]['target_layer']} {nb:9.3f}")
    for k in (args.a, args.b):
        if meta[k]["left_out_above_max_norm"]:
            print(f"  {k}: left out above {meta[k]['max_norm']:g}: {meta[k]['left_out_above_max_norm']}")

    # readout statistics
    names = [args.a, args.b, "shipped", "logit"]
    rows = {k: depth_rows(args.depth_dir / f"bilingual_depth_{k}{args.tag if k in (args.a, args.b) else ''}.json")
            for k in names}
    layers = sorted(set.intersection(*(set(r) for r in rows.values())))
    target_a = meta[args.a]["target_layer"]
    print(f"\nmedians over held-out WikiText activations. At layers >= {target_a} the {args.a} lens is the identity "
          f"(layer {target_a}) or undefined ({target_a + 1}+): those rows repeat the logit lens.")
    head = " ".join(f"{k[:13]:>13}" for k in names)
    print(f"{'L':>3} {'depth':>5} | eta2 (3 script classes)          {head}\n{'':>9} | Han-Latin offset, sd\n"
          f"{'':>9} | full-vocab excess kurtosis p50")
    table = {}
    for L in layers:
        e = {k: rows[k][L]["eta2_script3"]["p50"] for k in names}
        o = {k: rows[k][L]["han_minus_latin_mean_logit_sd"]["p50"] for k in names}
        q = {k: rows[k][L]["kurtosis"]["full"]["p50"] for k in names}
        table[L] = {"depth_0_100": rows["logit"][L]["depth_0_100"], "eta2": e, "han_minus_latin_sd": o, "kurt_full_p50": q}
        print(f"{L:3d} {table[L]['depth_0_100']:5.1f} | eta2 " + " ".join(f"{e[k]:13.4f}" for k in names))
        print(f"{'':>9} | off  " + " ".join(f"{o[k]:+13.2f}" for k in names))
        print(f"{'':>9} | kurt " + " ".join(f"{q[k]:+13.2f}" for k in names))

    # the lenses themselves
    print(f"\ntop output direction u1 of each lens   (share of |J|_F^2;  |cos| with the shipped lens's u1;  "
          f"gain along the shipped u1 / typical gain)")
    lenses = {k: LazyLens(p) for k, p in paths.items()}
    lens_rows = {}
    for L in args.share_layers:
        Js = {k: np.asarray(lz[L], np.float32) for k, lz in lenses.items() if L in lz.layers}
        if "shipped" not in Js:
            continue
        _, u_ship = top_pair(Js["shipped"])
        lens_rows[L] = {}
        for k, J in Js.items():
            s, u = top_pair(J)
            fro = float(np.linalg.norm(J.astype(np.float64)))
            lens_rows[L][k] = {"top1_share": s * s / fro ** 2, "abs_cos_u1_vs_shipped_u1": abs(float(u @ u_ship)),
                               "gain_along_shipped_u1_over_typical": float(np.linalg.norm(u_ship @ J)) / (fro / np.sqrt(J.shape[0]))}
        print(f"  L{L:02d} " + "   ".join(
            f"{k[:13]}: {r['top1_share']:.3f} / {r['abs_cos_u1_vs_shipped_u1']:.2f} / {r['gain_along_shipped_u1_over_typical']:.1f}x"
            for k, r in lens_rows[L].items()))

    # pre-specified reading
    print(f"\npre-specified reading (gone: eta2 <= {GONE}; survives: >= half the same prompts' target-{meta[args.b]['target_layer']} value)")
    verdict = {}
    for L in args.verdict_layers:
        a, b = table[L]["eta2"][args.a], table[L]["eta2"][args.b]
        verdict[L] = "gone" if a <= GONE else "survives" if a >= 0.5 * b else "reduced"
        print(f"  L{L}: eta2 {b:.3f} -> {a:.3f}  (x{a / b:.2f}; logit lens {table[L]['eta2']['logit']:.4f})  {verdict[L]}")
    k18 = table[min(args.verdict_layers)]["kurt_full_p50"]
    print(f"  full-vocab median kurtosis at L{min(args.verdict_layers)}: {k18[args.b]:+.2f} -> {k18[args.a]:+.2f}")
    args.out.write_text(json.dumps({"lenses": {k: str(p) for k, p in paths.items()}, "indices": ia, "per_prompt_norms": norms,
                                    "per_layer": table, "lens": lens_rows, "gone_threshold": GONE, "verdict": verdict}, indent=1))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
