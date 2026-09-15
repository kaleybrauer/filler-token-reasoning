"""
compare_autocorr_null.py — Figure 28 panel (c) for V3 under two nulls, against Sonnet 4.5.

The paper describes its null only as "position-shuffled". The figure's run (workspace_readouts_
shipped100.json) draws the null position from the same prompt, which subtracts document-level topic
persistence; the rerun with --null both (workspace_readouts_shipped100_xnull.json) adds a null drawn
from another prompt, which keeps it. This checks that the rerun reproduces the original (panels (a)
and (b) exactly, the within-prompt null to float rounding), then prints per token offset the peak and
its depth under each null, each curve's correlation with Sonnet's digitised curve over depth (all
depths and inside Sonnet's band), the peak's fall per doubling of the offset, and the mean gap
between the two nulls inside the band.

    python scripts/jlens/compare_autocorr_null.py
"""
import json, sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
J = REPO / "outputs/jlens"


def main():
    old = json.load(open(J / "workspace_readouts_shipped100.json"))
    new = json.load(open(J / "workspace_readouts_shipped100_xnull.json"))
    ref = json.load(open(J / "reference_fig28_sonnet45.json"))
    a, b = old["per_layer"], new["per_layer"]
    if [r["layer"] for r in a] != [r["layer"] for r in b]:
        raise SystemExit("layer grids differ between the original run and the rerun")

    exact = within = 0.0
    for ra, rb in zip(a, b):
        for k, v in ra.items():
            if k == "secs" or not isinstance(v, (int, float)):
                continue
            if k.startswith("autocorr_d"):
                within = max(within, abs(v - rb[k]))
            else:
                exact = max(exact, abs(v - rb[k]))
    ok = exact == 0.0 and within < 1e-6
    print(f"rerun reproduces the original: panels (a)(b) max diff {exact:.1e}, within-prompt null max diff "
          f"{within:.1e} -> {'PASS' if ok else 'FAIL'}")

    offs = new["offsets"]
    depth = np.array([r["depth_0_100"] for r in b])
    sonnet = {s["label"]: np.array(s["y"]) for s in ref["panels"][2]["series"]}
    n_ref = len(sonnet[str(offs[0])])
    if n_ref != len(depth):
        raise SystemExit(f"Sonnet reference has {n_ref} layers, the rerun {len(depth)}")
    s_depth = np.arange(n_ref) / (n_ref - 1) * 100
    band = (depth >= 37) & (depth <= 92)

    def stats(v, s):
        i = int(np.argmax(v))
        return v[i], depth[i], np.corrcoef(v, s)[0, 1], np.corrcoef(v[band], s[band])[0, 1]

    print(f"\n{'offset':>6} | {'within-prompt null':^27} | {'cross-prompt null':^27} | {'Sonnet 4.5':^12} | x-w gap")
    print(f"{'':>6} | {'peak':>6} {'at %':>5} {'r all':>6} {'r band':>7} | {'peak':>6} {'at %':>5} {'r all':>6} "
          f"{'r band':>7} | {'peak':>6} {'at %':>5} | band mean")
    peaks = {"within": [], "cross": [], "sonnet": []}
    for o in offs:
        s = sonnet[str(o)]
        w = np.array([r[f"autocorr_d{o}"] for r in b])
        x = np.array([r[f"autocorr_xprompt_d{o}"] for r in b])
        ws, xs, j = stats(w, s), stats(x, s), int(np.argmax(s))
        peaks["within"].append(ws[0]); peaks["cross"].append(xs[0]); peaks["sonnet"].append(s[j])
        print(f"{o:>6} | {ws[0]:6.2f} {ws[1]:5.0f} {ws[2]:+6.2f} {ws[3]:+7.2f} | {xs[0]:6.2f} {xs[1]:5.0f} {xs[2]:+6.2f} "
              f"{xs[3]:+7.2f} | {s[j]:6.2f} {s_depth[j]:5.0f} | {np.mean((x - w)[band]):+.2f}")

    print("\npeak ratio per doubling of the offset (offset k -> 2k):")
    for name, v in peaks.items():
        ratios = [f"{v[i + 1] / v[i]:.2f}" if v[i] > 0 else "  - " for i in range(len(v) - 1)]
        print(f"  {name:>6}: " + " ".join(ratios))

    print("\ncross-prompt null, per layer:")
    print("  depth " + " ".join(f"{d:5.0f}" for d in depth))
    for o in offs:
        print(f"  d{o:<4} " + " ".join(f"{r[f'autocorr_xprompt_d{o}']:5.2f}" for r in b))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
