"""
fit_progress.py — how far the fit has got, without reading the 11.5 GiB checkpoint.

`n_done` lives inside the checkpoint, but `torch.load` materialises all 60 Jacobians
to read it — 11.5 GiB off the network mount per poll, competing with the fit's own
checkpoint writes. The fit log already carries the same information, so pollers use
`last_prompt_from_logs()` as a cheap monotone trigger and only pay `checkpoint_n()`
once they have decided to act.

The log number is `prompt_idx + 1`, i.e. the position in the prompt list, which
equals `n_done` only when no prompt has been skipped (`fit` skips prompts too short
to yield valid positions without incrementing `n_done`). That is why it is a trigger
and not the reported value: anything written down comes from `checkpoint_n()`.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# jlens_validate_fit.log is where the current run writes (the fit is running inside
# safe_fit.py --then-fit). jlens_fit.log is used when the watchdog relaunches
# fit_v3.py directly. The DEAD run's log was archived to jlens_fit_POISONED_run.log
# and is deliberately NOT read: its prompt numbers ran to 48, which made this
# trigger fire constantly and forced a 11.5 GiB authoritative read every poll.
DEFAULT_LOGS = [REPO_ROOT / "logs/jlens_gates.log",
                REPO_ROOT / "logs/jlens_fit.log",
                REPO_ROOT / "logs/jlens_validate_fit.log"]

# "  prompt 7/1000  seq_len=128 n_valid=111  1183s  max||J||/sqrt(d)=..."
PROMPT_RE = re.compile(r"prompt (\d+)/(\d+)\s+seq_len=")
SKIP_RE = re.compile(r"skipping prompt (\d+)")


def last_prompt_from_logs(logs=None) -> tuple[int, int]:
    """(last completed prompt number, number of skipped prompts) across `logs`."""
    last, skips = 0, 0
    for log in logs or DEFAULT_LOGS:
        log = Path(log)
        if not log.exists():
            continue
        text = log.read_text(errors="replace")
        for m in PROMPT_RE.finditer(text):
            last = max(last, int(m.group(1)))
        skips += len(SKIP_RE.findall(text))
    return last, skips


def checkpoint_n(path) -> int:
    """Authoritative `n_done` from a fit checkpoint. Reads the whole file.

    Returns -1 if the file is missing or momentarily unreadable — `_atomic_save`
    renames a .tmp into place, so a read can land on neither inode.
    """
    import torch

    try:
        return int(torch.load(path, map_location="cpu", weights_only=True)["n_done"])
    except Exception:
        return -1


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None,
                    help="Also read the authoritative n_done (reads 11.5 GiB)")
    args = ap.parse_args()
    last, skips = last_prompt_from_logs()
    print(f"last_prompt={last} skipped={skips}")
    if args.checkpoint:
        print(f"checkpoint_n={checkpoint_n(args.checkpoint)}")
