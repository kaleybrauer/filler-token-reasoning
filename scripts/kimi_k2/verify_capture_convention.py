"""Gate: does an extraction hold the RESIDUAL STREAM or per-layer MLP WRITES?

Run this on the FIRST condition of a re-extraction, before spending GPU-hours on the
rest. It reads saved pkls only — no model, no GPU.

Discriminator: the residual stream is a running sum of ~2L component writes, so its norm
grows by ~2 orders of magnitude from layer 0 to three-quarters depth. Independent
per-layer MLP writes do not accumulate at all, so that ratio sits near 1.

Measured separation on this repo's extractions (2026-08-18, 57 conditions, no overlap):
    residual stream (DeepSeek V3 x26, Kimi K2.5 x13):  mid-growth  56x - 100x
    per-layer writes (Kimi K2 x18):                    mid-growth  ~1.0x
A monotonicity test is NOT used as the gate: it false-positives on real streams that dip
in the last few layers (extracted_states_varbind_cot_free reads mono=0.63 but grows 73x).

Usage:
    python scripts/kimi_k2/verify_capture_convention.py <states_dir> [<states_dir> ...]

Exit code 0 if every condition passes as residual stream, 1 otherwise.
"""
from __future__ import annotations

import glob
import pickle
import sys
from pathlib import Path

import numpy as np

# Mid-growth below this => not accumulating => per-layer writes.
# Observed streams are >=56x and observed writes are ~1.0x, so anything in between is
# ambiguous and should be investigated rather than trusted.
GROWTH_PASS = 20.0


def profile(cond_dir: Path, n: int = 8):
    files = sorted(glob.glob(str(cond_dir / "*.pkl")))[:n]
    if not files:
        return None
    norms, conventions = [], set()
    for f in files:
        try:
            d = pickle.load(open(f, "rb"))
        except Exception:
            continue
        st = d.get("states")
        if not st:
            continue
        conventions.add(d.get("capture_convention"))
        pos = list(st.keys())[-1]              # last position = answer_prompt
        layers = sorted(st[pos].keys())
        norms.append([float(np.linalg.norm(np.asarray(st[pos][l], dtype=np.float32)))
                      for l in layers])
    if not norms:
        return None
    return np.mean(np.array(norms), axis=0), conventions


def main(dirs: list[str]) -> int:
    print(f"{'condition':<52} {'L0':>7} {'L~0.75':>8} {'Llast':>9} {'growth':>8}  verdict")
    print("-" * 104)
    failures = 0
    for d in dirs:
        root = Path(d)
        conds = sorted([p for p in root.iterdir() if p.is_dir()]) or [root]
        for c in conds:
            r = profile(c)
            if r is None:
                continue
            m, conventions = r
            L = len(m) - 1
            mid = int(round(0.75 * L))
            growth = m[mid] / max(m[0], 1e-9)
            ok = growth >= GROWTH_PASS
            failures += (not ok)
            verdict = "RESIDUAL STREAM ok" if ok else "*** per-layer WRITES - FAIL ***"
            print(f"{root.name + '/' + c.name:<52} {m[0]:7.1f} {m[mid]:8.1f} {m[L]:9.1f} "
                  f"{growth:8.1f}  {verdict}")
            stamped = {x for x in conventions if x}
            if stamped:
                print(f"{'':<52} stamped convention: {', '.join(sorted(stamped))}")
    print()
    print("PASS - all conditions hold the residual stream" if not failures
          else f"FAIL - {failures} condition(s) hold per-layer writes; do NOT proceed")
    return 1 if failures else 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1:]))
