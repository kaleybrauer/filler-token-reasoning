"""
plot_offset_input.py — one panel: the Chinese-minus-Latin readout offset across depth on English and on Chinese text,
released J-lens and logit lens, with the model's own output at 100 % depth (the offset does not depend on the input).
Same data as plot_hero_offset.py panel (a).

    python scripts/jlens/plot_offset_input.py
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_fig28 import AXIS, MUTED, OUT
from plot_hero_offset import GRAY, DOT, frame, series, endlabel

LBL = 12
EN, ZH = "#1b2733", "#b02a86"   # English text, Chinese text (blue/orange mean final/penultimate elsewhere)


def main():
    fig, a = plt.subplots(1, 1, figsize=(8.0, 4.8))
    frame(a, "", "Chinese minus Latin mean logit, in sd\n(above zero: Chinese tokens score higher)")
    a.axhline(0, color=AXIS, lw=0.9, zorder=1)
    # the J-lens lines stop below the target, where the lens stops being the identity; the logit lens runs all
    # the way to the output, where it is the model's own logits
    for name, col, ls, to_output in (("shipped", EN, "-", False), ("shipped_wikizh", ZH, "-", False),
                                     ("logit", EN, DOT, True), ("logit_wikizh", ZH, DOT, True)):
        x, y = series(name, "han_minus_latin_mean_logit_sd", below_output=not to_output)
        a.plot(x, y, color=col, ls=ls, lw=2.0 if ls == "-" else 1.4, zorder=3)
    # each dotted line ends at the model's own logits for that text, labelled at the edge
    for name, col, lang in (("logit_wikizh", ZH, "Chinese"), ("logit", EN, "English")):
        x, y = series(name, "han_minus_latin_mean_logit_sd", below_output=False)
        a.text(x[-1] + 1.5, y[-1], f"model's output, {lang}", fontsize=LBL, color=col, ha="left", va="center",
               clip_on=False)
    xe, ye = series("shipped", "han_minus_latin_mean_logit_sd"); xz, yz = series("shipped_wikizh", "han_minus_latin_mean_logit_sd")
    a.text(xe[-1] - 1, ye[-1] - 0.12, "J-lens, English text", fontsize=LBL, color=EN, ha="right", va="top")
    a.text(xz[-1] - 1, yz[-1] + 0.12, "J-lens, Chinese text", fontsize=LBL, color=ZH, ha="right", va="bottom")
    a.text(22, -0.22, "logit lens, both texts", fontsize=LBL, color=GRAY, ha="center", va="top")
    a.xaxis.label.set_size(13); a.yaxis.label.set_size(13); a.tick_params(labelsize=12)
    a.set_ylim(-1.9, 1.9); a.set_xlim(0, 100)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"offset_input.{ext}", dpi=170, facecolor="white")
    print(f"wrote {OUT / 'offset_input.png'}")


if __name__ == "__main__":
    main()
