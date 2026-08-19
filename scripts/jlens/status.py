"""
status.py — where is the J-lens fit up to?

Reads the fit checkpoint and the run logs and prints: prompts done, wall-clock
per prompt, the convergence diagnostic, and an ETA. Safe to run while the fit is
going (torch.load of the checkpoint is a read; _atomic_save never leaves a
partial file in place).

Usage:
    python scripts/jlens/status.py
    python scripts/jlens/status.py --checkpoint outputs/jlens/lens_v3.fitckpt
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# "  prompt 7/1000  seq_len=128 n_valid=111  1183s  max||J||/sqrt(d)=1.234  max_d_mean=1.20e-02"
PROMPT_RE = re.compile(
    r"prompt (\d+)/(\d+)\s+seq_len=(\d+) n_valid=(\d+)\s+([\d.]+)s\s+"
    r"max\|\|J\|\|/sqrt\(d\)=([\d.eE+-]+)\s+max_d_mean=([\d.eEnan+-]+)"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path,
                    default=REPO_ROOT / "outputs/jlens/lens_v3.fitckpt")
    ap.add_argument("--logs", nargs="+", type=Path,
                    default=[REPO_ROOT / "logs/jlens_gates.log",
                             REPO_ROOT / "logs/jlens_fit.log"])
    ap.add_argument("--tail", type=int, default=8, help="Recent prompts to show")
    args = ap.parse_args()

    rows = []
    for log in args.logs:
        if log.exists():
            for line in log.read_text(errors="replace").splitlines():
                m = PROMPT_RE.search(line)
                if m:
                    rows.append((log.name, *m.groups()))

    print("=== J-lens fit status ===")
    if args.checkpoint.exists():
        import torch
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        layers = state["source_layers"]
        size_gib = args.checkpoint.stat().st_size / 2**30
        print(f"checkpoint : {args.checkpoint}")
        print(f"  prompts fitted : {state['n_done']}  (next index {state['next_idx']})")
        print(f"  layers         : {len(layers)} [{layers[0]}..{layers[-1]}] "
              f"-> target L{state['target_layer']}, skip_first={state['skip_first']}")
        print(f"  file           : {size_gib:.1f} GiB")
        print("  -> usable as a lens right now: "
              "scripts/jlens/apply_lens.py divides the running sum by n_done")
    else:
        print(f"checkpoint : {args.checkpoint} does not exist yet")

    if not rows:
        print("\nno per-prompt lines in the logs yet (still loading the model?)")
        return

    times = [float(r[5]) for r in rows]
    print(f"\nper-prompt lines: {len(rows)}")
    print(f"  wall/prompt : median {sorted(times)[len(times) // 2] / 60:.1f} min, "
          f"last {times[-1] / 60:.1f} min")
    n_done, n_total = int(rows[-1][1]), int(rows[-1][2])
    remaining = n_total - n_done
    print(f"  ETA         : {remaining} prompts left x median "
          f"= {remaining * sorted(times)[len(times) // 2] / 3600:.1f} h "
          f"(the run is meant to be stopped early, not to reach {n_total})")

    print(f"\nlast {args.tail} prompts (max_d_mean = relative change of the running "
          f"mean; falls ~1/n once converged):")
    print(f"  {'prompt':>10} {'min':>6} {'||J||/sqrt(d)':>14} {'max_d_mean':>12}")
    for _, idx, tot, _, _, secs, norm, dmean in rows[-args.tail:]:
        print(f"  {idx + '/' + tot:>10} {float(secs) / 60:6.1f} {float(norm):14.3f} "
              f"{dmean:>12}")

    norms = [float(r[6]) for r in rows]
    hi, lo = max(norms), min(norms)
    print(f"\nper-prompt ||J||/sqrt(d): min {lo:.3f}  max {hi:.3f}  ratio {hi / max(lo, 1e-9):.1f}x")
    print("  (a very heavy tail here would be the argument for per-prompt normalisation,"
          "\n   which the Anthropic reference does NOT do - flag it, do not just add it)")


if __name__ == "__main__":
    main()
