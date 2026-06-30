"""
Figure for the position-resolved 2-fact KV transplant (fact-matched, dots_50).

3 rows x 2 columns:
  cols = causal channel: A1 (vary_a1) | A2 (vary_a2)
  row 0: donor-rank shift   (bars: whole / <a>_pos@0.15 / no_<a>_pos@0.15, 95% CI)
  row 1: donor adoption %   (same bars)
  row 2: theta-robustness   (pos-shift & no_pos-shift vs theta in {0.15,0.20,0.30})

Message: in every panel pos ~= whole and no_pos ~= 0 -> the decoded positions carry
the addend; A1 bars tower over A2 (stronger / more concentrated).

Run:
    /root/.venvs/filler-probing/bin/python plotting/plot_twofact_transplant.py
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("results/twofact_kv_transplant")
THETAS = ["0.15", "0.2", "0.3"]
THETA_F = [0.15, 0.20, 0.30]


def boot_ci(vals, fn, n=10000, seed=0):
    v = np.asarray(vals, float)
    if len(v) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n, len(v)))
    s = fn(v[idx], axis=1)
    return float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def load(direction):
    d = json.load(open(ROOT / f"dots_50_{direction}" / "twofact_transplant_results.json"))
    return [r for r in d["results"] if r["T_correct"] and r["D_correct"]]


def stat(res, label, metric):
    """(value, lo, hi) for metric in {'shift','adoption'} on config `label`."""
    if metric == "shift":
        vals = [r["rank_norm"]["donor"] - r["configs"][label]["rank_transplant"]["donor"]
                for r in res if r["rank_norm"]["donor"] is not None
                and r["configs"][label]["rank_transplant"]["donor"] is not None]
        lo, hi = boot_ci(vals, np.median)
        return (float(np.median(vals)) if vals else float("nan")), lo, hi
    vals = [1.0 if r["configs"][label]["outcome"] == "donor" else 0.0 for r in res]
    lo, hi = boot_ci(vals, np.mean)
    return float(np.mean(vals)) * 100, lo * 100, hi * 100


def lighten(c, amt=0.55):
    rgb = np.array(matplotlib.colors.to_rgb(c))
    return tuple(rgb + (1 - rgb) * amt)


CHANNELS = [("vary_a1", "a1", "A1", "#1f77b4"), ("vary_a2", "a2", "A2", "#d62728")]

fig = plt.figure(figsize=(11, 12.5))
gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.95], hspace=0.5, wspace=0.28)

for col, (direction, tag, name, color) in enumerate(CHANNELS):
    res = load(direction)
    n = len(res)
    scopes = [("whole", "whole", "0.6"),
              (f"{tag}_pos@0.15", f"{name}\npositions", color),
              (f"no_{tag}_pos@0.15", f"non-{name}", lighten(color))]
    whole_shift = stat(res, "whole", "shift")[0]

    # --- row 0: rank shift ---
    ax = fig.add_subplot(gs[0, col])
    for i, (lab, _, c) in enumerate(scopes):
        v, lo, hi = stat(res, lab, "shift")
        ax.bar(i, v, color=c, edgecolor="k", linewidth=0.6,
               yerr=[[max(v - lo, 0)], [max(hi - v, 0)]], capsize=5, error_kw={"lw": 1.1})
    ax.axhline(whole_shift, ls="--", color="0.5", alpha=0.8, lw=1)
    pv = stat(res, scopes[1][0], "shift")[0]
    ax.annotate(f"{pv/whole_shift:.0%}\nof whole", (1, pv), textcoords="offset points",
                xytext=(0, 8), ha="center", va="bottom", fontsize=9, color=color, fontweight="bold")
    ax.set_xticks(range(3)); ax.set_xticklabels([s[1] for s in scopes])
    ax.set_title(f"{name} channel — vary {name}  (n={n})", fontweight="bold")
    if col == 0:
        ax.set_ylabel("donor-rank shift\n(norm → transplant)")
    ax.margins(y=0.18)

    # --- row 1: adoption ---
    ax2 = fig.add_subplot(gs[1, col])
    for i, (lab, _, c) in enumerate(scopes):
        v, lo, hi = stat(res, lab, "adoption")
        ax2.bar(i, v, color=c, edgecolor="k", linewidth=0.6,
                yerr=[[max(v - lo, 0)], [max(hi - v, 0)]], capsize=5, error_kw={"lw": 1.1})
        if v < 0.05:
            ax2.annotate("0.0%", (i, 0), textcoords="offset points", xytext=(0, 3),
                         ha="center", fontsize=9, color="k", fontweight="bold")
    ax2.set_xticks(range(3)); ax2.set_xticklabels([s[1] for s in scopes])
    if col == 0:
        ax2.set_ylabel("donor adoption (%)\n(answer flips to donor)")
    ax2.margins(y=0.18)

    # --- row 2: theta robustness ---
    ax3 = fig.add_subplot(gs[2, col])
    pos_sh = [stat(res, f"{tag}_pos@{t}", "shift")[0] for t in THETAS]
    no_sh = [stat(res, f"no_{tag}_pos@{t}", "shift")[0] for t in THETAS]
    ax3.plot(THETA_F, pos_sh, "o-", color=color, lw=2, label=f"{name} positions")
    ax3.plot(THETA_F, no_sh, "s--", color="0.45", lw=2, label=f"non-{name}")
    ax3.axvline(0.15, color="green", ls=":", alpha=0.6, lw=1)
    ax3.set_xticks(THETA_F); ax3.set_xlabel("decode threshold θ")
    if col == 0:
        ax3.set_ylabel("donor-rank shift")
    ax3.set_title("robustness: addend is distributed", fontsize=10)
    ax3.legend(fontsize=8, loc="center right")

fig.suptitle("Filler positions that decode an addend causally carry it\n"
             "(2-fact, DeepSeek V3, dots_50, fact-matched transplant; θ=0.15 in rows 1–2)",
             fontsize=13, fontweight="bold")
fig.subplots_adjust(top=0.91)
for ext in ("png", "pdf"):
    fig.savefig(ROOT / f"twofact_transplant_figure.{ext}", dpi=150, bbox_inches="tight")
print(f"saved {ROOT}/twofact_transplant_figure.png and .pdf")
