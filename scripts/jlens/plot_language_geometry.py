"""
plot_language_geometry.py — does V3 hold a concept in a language-neutral, language-conditioned or
hybrid form? One figure from language_geometry.py (matched en/zh/ja concept states, raw residual
unless noted; 581 concepts, 2 templates per language).

  A  variance shares over depth: concept, language offset, language x concept, template variation
  B  en->zh top-1 retrieval: raw centred, raw uncentred, J-lens centred with the half-fit range
  C  the concept-token zh-en direction vs the Wikipedia zh-en direction; stability across templates
  D  interaction / template variation (F) for the same token in zh vs ja contexts, different tokens,
     and en-zh; layers where template variation is <3% of variance are not drawn (F unstable)
  E  same-token control at the layers where context distances overlap: concept-specific difference
     against context distance, within vs across languages

Writes results/jlens_workspace/language_geometry.{png,pdf} and the plotted values as a CSV.

    python scripts/jlens/plot_language_geometry.py
"""
from __future__ import annotations

import csv, json, sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_fig28 import AXIS, INK, MUTED, OUT, style   # noqa: E402

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "outputs/jlens/language_geometry_v2.json"
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#a3acb5"   # categorical slots 1-3 + context gray
NO_BAND = (0, 0)
TABLE = []


def series(rows, get):
    Ls = sorted(rows, key=int)
    return np.array([int(L) / 60 * 100 for L in Ls]), np.array([get(rows[L]) for L in Ls], float)


def line(ax, x, y, color, label, panel, lw=2.0, z=3):
    ax.plot(x, y, color=color, lw=lw, solid_capstyle="round", solid_joinstyle="round", zorder=z, label=label)
    TABLE.extend({"panel": panel, "series": label, "depth": round(float(a), 2), "value": float(b)} for a, b in zip(x, y))


def end_labels(ax, items, min_gap, x=100, log=False):
    """Direct labels at the line ends: a colored key beside ink text, nudged apart vertically
    (min_gap in data units, or in decades on a log axis)."""
    f, finv = (np.log10, lambda v: 10 ** v) if log else ((lambda v: v), (lambda v: v))
    items = sorted(((lab, f(y), col) for lab, y, col in items), key=lambda t: t[1])
    ys = []
    for _, y, _ in items:
        ys.append(max(y, ys[-1] + min_gap) if ys else y)
    shift = (ys[-1] - items[-1][1]) / 2 if len(ys) > 1 else 0
    for (label, _, color), y in zip(items, ys):
        yv = finv(y - shift)
        ax.plot([x + 2.0, x + 5.0], [yv, yv], color=color, lw=2.0, clip_on=False, solid_capstyle="round")
        ax.text(x + 6.5, yv, label, color=INK, fontsize=8, va="center", ha="left", clip_on=False)


def main():
    R = json.loads(SRC.read_text())
    raw, ship = R["per_lens"]["raw"], R["per_lens"].get("shipped")
    halves = [R["per_lens"][k] for k in ("n50", "n50b") if k in R["per_lens"] and len(R["per_lens"][k]) == len(raw)]
    fig = plt.figure(figsize=(19.5, 10.4))
    gs = fig.add_gridspec(2, 3, left=0.04, right=0.925, top=0.93, bottom=0.13, wspace=0.48, hspace=0.45)

    # A: variance shares
    a = fig.add_subplot(gs[0, 0])
    style(a, "A  Concept identity dominates; language is small until the output", "Share of variance →", NO_BAND)
    parts = [("concept", BLUE, lambda o: o["anova"]["share"]["concept"]),
             ("language offset", ORANGE, lambda o: o["anova"]["share"]["lang"]),
             ("language × concept", AQUA, lambda o: o["anova"]["share"]["interaction"]),
             ("template variation", GRAY, lambda o: o["anova"]["share"]["template"] + o["anova"]["share"]["replicate"])]
    ends = []
    for label, color, get in parts:
        x, y = series(raw, get)
        line(a, x, y, color, label, "A")
        ends.append((label, y[-1], color))
    a.set_ylim(0, 0.75)
    end_labels(a, ends, min_gap=0.045)
    a.legend(fontsize=7.5, frameon=False, loc="upper center", ncol=2, handlelength=1.6)

    # B: retrieval
    b = fig.add_subplot(gs[0, 1])
    style(b, "B  Translations stay matched; the offset hides it at the output", "en→zh top-1 retrieval →", NO_BAND)
    x, yc = series(raw, lambda o: 100 * o["retrieval"]["en-zh"]["centred"]["top1"])
    _, yr = series(raw, lambda o: 100 * o["retrieval"]["en-zh"]["raw"]["top1"])
    ends = []
    if ship:
        xs, ys = series(ship, lambda o: 100 * o["retrieval"]["en-zh"]["centred"]["top1"])
        if halves:
            hs = np.array([series(h, lambda o: 100 * o["retrieval"]["en-zh"]["centred"]["top1"])[1] for h in halves])
            b.fill_between(xs, hs.min(0), hs.max(0), color=AQUA, alpha=0.18, lw=0, zorder=2)
        line(b, xs, ys, AQUA, "J-lens, centred", "B")
        ends.append(("J-lens, centred", ys[-1], AQUA))
    line(b, x, yr, GRAY, "raw, uncentred", "B")
    line(b, x, yc, BLUE, "raw, centred", "B", z=4)
    ends += [("raw, uncentred", yr[-1], GRAY), ("raw, centred", yc[-1], BLUE)]
    b.axhline(100 / R["n_concepts"], color=AXIS, lw=1, zorder=1)
    b.text(1.5, 100 / R["n_concepts"] + 1.8, f"chance {100 / R['n_concepts']:.1f}%", fontsize=7.5, color=MUTED)
    b.set_ylim(0, 100)
    b.yaxis.set_major_formatter(lambda v, p: f"{v:.0f}%")
    b.annotate(f"{yc[0]:.0f}% at layer 0: the embeddings\nalready place translations together",
               xy=(x[0], yc[0]), xytext=(9, 55), fontsize=7.5, color=MUTED,
               arrowprops=dict(arrowstyle="-", color=AXIS, lw=1))
    end_labels(b, ends, min_gap=6.5)
    b.legend(fontsize=7.5, frameon=False, loc="lower left", bbox_to_anchor=(0.30, 0.03), handlelength=1.6)

    # C: language direction
    c = fig.add_subplot(gs[0, 2])
    style(c, "C  Mid-depth, the language direction isn't the running-text one", "|cosine| →", NO_BAND)
    x, yw = series(raw, lambda o: o["language_directions"]["zh-en"]["abs_cos_wiki_zh_minus_en"])
    _, yt = series(raw, lambda o: o["language_directions"]["zh-en"]["cos_template1_template2"])
    line(c, x, yt, GRAY, "stability: template 1 vs template 2", "C")
    line(c, x, yw, ORANGE, "vs Wikipedia zh−en direction", "C", z=4)
    c.set_ylim(0, 1.05)
    c.text(0.5, 0.43, "zh vs en separable along it at every layer\n(cross-validated AUC 1.00)",
           transform=c.transAxes, fontsize=7.5, color=MUTED, ha="center")
    end_labels(c, [("stability", yt[-1], GRAY), ("vs Wikipedia", yw[-1], ORANGE)], min_gap=0.07)
    c.legend(fontsize=7.5, frameon=False, loc="upper center", handlelength=1.6)

    # D: F for word identity vs language context
    d = fig.add_subplot(gs[1, 0])
    style(d, "D  Most of the language difference is the word itself", "Interaction ÷ template variation (F) →", NO_BAND)
    cells = [("zh–ja, different tokens", BLUE, "zh-ja different tokens"), ("en–zh", AQUA, "en-zh"),
             ("zh–ja, same token", ORANGE, "zh-ja same token")]
    x_all, rep = series(raw, lambda o: o["pairwise_anova"]["zh-ja same token"]["share"]["replicate"])
    ok = rep >= 0.03
    start = x_all[np.argmax(ok)]
    d.axvspan(0, start, color="#eef1f4", lw=0, zorder=0)
    d.text(start / 2, 1.08, "not drawn:\ntemplate variation\n<3% of variance", fontsize=7, color=MUTED, ha="center", va="bottom")
    ends = []
    for label, color, key in cells:
        x, y = series(raw, lambda o, key=key: o["pairwise_anova"][key]["F_interaction_vs_replicate"])
        line(d, x[ok], y[ok], color, label, "D", z=4 if "same" in label else 3)
        ends.append((label, y[-1], color))
    d.set_yscale("log")
    d.set_ylim(1, 40)
    d.set_yticks([1, 2, 4, 8, 16, 32])
    d.yaxis.set_major_formatter(lambda v, p: f"{v:g}")
    d.axhline(1, color=AXIS, lw=1, zorder=1)
    d.text(start + 1.5, 1.07, "F = 1: interaction no larger than template noise", fontsize=7.5, color=MUTED)
    d.legend(fontsize=7.5, frameon=False, loc="upper right", handlelength=1.6)
    end_labels(d, ends, min_gap=0.1, log=True)

    # E: same-token context-distance control, small multiples
    sub = gs[1, 1:].subgridspec(1, 3, wspace=0.12)
    layers = [25, 40, 50]
    rows = {L: raw[str(L)]["context_distance_same_token"] for L in layers}
    allx = [r["context_distance"] for L in layers for r in rows[L]]
    ally = [r["D"] for L in layers for r in rows[L]]
    xlim = (min(allx) - 0.04, max(allx) + 0.04)
    ylim = (0, max(ally) * 1.18)
    for i, L in enumerate(layers):
        e = fig.add_subplot(sub[0, i])
        title = (f"E  The same-token language effect isn't just context distance\n{L / 60 * 100:.0f}% depth (layer {L})"
                 if i == 0 else f"\n{L / 60 * 100:.0f}% depth (layer {L})")
        style(e, title, "Concept-specific difference D →" if i == 0 else "", NO_BAND)
        e.set_xlim(*xlim)
        e.set_ylim(*ylim)
        e.set_xlabel("Context distance (1 − cos) →", fontsize=9, color=MUTED)
        if i:
            e.tick_params(labelleft=False)
        within = sorted((r for r in rows[L] if not r["cross_language"]), key=lambda r: r["context_distance"])
        above = {within[1]["pair"]} if len(within) == 2 and within[1]["context_distance"] - within[0]["context_distance"] < 0.1 else set()
        for r in rows[L]:
            cross = r["cross_language"]
            e.scatter(r["context_distance"], r["D"], s=64, color=ORANGE if cross else GRAY, edgecolor="white",
                      linewidth=2, zorder=4, label=("across languages" if cross else "within a language"))
            TABLE.append({"panel": f"E layer {L}", "series": r["pair"], "depth": r["context_distance"], "value": r["D"]})
            if not cross:
                name = r["pair"].replace(":", "").replace("-", "–")
                e.annotate(name, (r["context_distance"], r["D"]), xytext=(0, 9 if r["pair"] in above else -13),
                           textcoords="offset points", fontsize=7.5, color=MUTED, ha="center")
        if i == 0:
            h, lab = e.get_legend_handles_labels()
            uniq = dict(zip(lab, h))
            e.legend(uniq.values(), uniq.keys(), fontsize=7.5, frameon=False, loc="upper left", handletextpad=0.3)

    fig.text(0.04, 0.03,
             f"DeepSeek-V3: {R['n_concepts']} concepts that are single tokens in English, Chinese and Japanese "
             f"({len(R['dropped_shared_word'])} dropped for sharing a word with another concept), two neutral templates per "
             "language, state at the concept token; raw residual unless noted.\n"
             "F uses per-dimension degrees of freedom: a descriptive ratio, not a significance test. "
             "E uses the 288 concepts whose Chinese and Japanese strings are identical (same token, different language context); "
             "J-lens ribbon: range of the two half-fit lenses.", fontsize=8, color=MUTED)
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"language_geometry.{ext}", dpi=170)
    with open(OUT / "language_geometry_figure.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["panel", "series", "depth", "value"])
        w.writeheader(); w.writerows(TABLE)
    print(f"wrote {OUT / 'language_geometry.png'} (+pdf, csv); J-lens halves in ribbon: {len(halves)}")


if __name__ == "__main__":
    main()
