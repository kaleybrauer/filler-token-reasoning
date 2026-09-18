"""
plot_hero_main.py — the post's primary figure (outline_v3.md, section 3): a final-layer-target J-lens carries the
last block's Han-vs-Latin script shift into every layer's readout as a component the state doesn't contain and that
doesn't follow the input; a penultimate target leaves it out.

  (a) how many of a readout's top 12 tokens are Han, at layer L (default 30, 50 % depth), over the depth
      diagnostic's English activations (24 positions x 100 paragraphs): histogram for the logit lens, the released
      J-lens (final target) and the penultimate-target J-lens. --panel-a example instead draws one rule-picked
      activation's top-K tokens under the three lenses (the median Han share under the released lens among
      complete-word positions).
  (b) script share of readout variance (eta^2, median over activations), log axis: released lens, N prompts at the
      final target, the same N at the penultimate target, the logit lens drawn through to the output.
  (c) Han - Latin readout offset (sd units, median) on English and Chinese text, J-lens and logit lens, with the
      model's own output at 100 % depth.
No title and no annotation text beyond line labels; the caption carries the numbers.
Writes results/jlens_workspace/hero_main.{png,pdf} and hero_main_panel_a.json (the counts, or the example).

    python scripts/jlens/plot_hero_main.py [--layer 30] [--k 20] [--panel-a hist|example]
"""
from __future__ import annotations

import argparse, json, os, sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import models                                                  # noqa: E402
from bilingual_diagnostics import token_class, CLASSES, HAN    # noqa: E402
from plot_bilingual import load                                 # noqa: E402
from plot_fig28 import AXIS, INK, MUTED, OUT, style             # noqa: E402
from plot_hero_offset import BLUE, ORANGE, GRAY, DASH, DOT, frame, series, endlabel   # noqa: E402
from workspace_readouts import logits_of                        # noqa: E402

REPO = Path(__file__).resolve().parents[2]
HAN_TINT = "#c8571b"
CJK_FONT = Path("/workspace/fonts/wqy-microhei.ttc")      # DejaVu has no Han glyphs; registered below if present
CJK_FAMILY = "DejaVu Sans"
if CJK_FONT.exists():
    from matplotlib import font_manager as _fm
    _fm.fontManager.addfont(str(CJK_FONT))
    CJK_FAMILY = _fm.FontProperties(fname=str(CJK_FONT)).get_name()
PENULT = "target59"          # alt-lens key of the penultimate lens (models.py); its n is read from the file


def pick_activation(H, keep, U, eps, lens, L, pos, cls, input_ids, tok, chunk=64):
    """Index (prompt, position) of the activation with the median Han share of the top 100 under `lens` at L,
    among the diagnostic's positions whose input token is a complete Latin-alphabet word: it starts with a space, is
    alphabetic, and the next token does not continue it (raw WikiText is full of markup, and a readout on a bracket or on half a word, e.g.
    " kings" of "kingship", is not the picture a reader needs)."""
    P, T = H.shape[0], H.shape[1]
    def whole_word(p, t):
        i = int(input_ids[p, t])
        piece = tok.decode([i])
        if t + 1 >= T or cls[i] != CLASSES.index("Latin") or not piece.startswith(" ") or not piece.strip().isalpha():
            return False
        return not tok.decode([int(input_ids[p, t + 1])])[:1].isalpha()
    idx = [(p, t) for p in range(P) for t in pos if keep[p, t] and whole_word(p, t)]
    X = np.stack([H[p, t, L].float().numpy() for p, t in idx])
    J = lens[L]
    share = np.empty(len(idx))
    for s in range(0, len(idx), chunk):
        z = logits_of(X[s:s + chunk] @ J.T, U, eps)
        top = np.argpartition(-z, 99, axis=1)[:, :100]
        share[s:s + chunk] = (cls[top] == HAN).mean(1)
    order = np.argsort(share, kind="stable")
    med = order[len(order) // 2]
    return idx[med], float(share[med]), float(np.median(share))


def han_counts(H, keep, U, eps, lens, L, pos, cls, k, chunk=64):
    """Number of Han tokens among the top k of every diagnostic activation at L (lens None = logit lens)."""
    X = np.stack([H[p, t, L].float().numpy() for p in range(H.shape[0]) for t in pos if keep[p, t]])
    if lens is not None:
        X = X @ lens[L].T
    out = []
    for s in range(0, len(X), chunk):
        z = logits_of(X[s:s + chunk], U, eps)
        top = np.argpartition(-z, k - 1, axis=1)[:, :k]
        out.append((cls[top] == HAN).sum(1))
    return np.concatenate(out)


def top_tokens(x, J, U, eps, tok, k):
    z = logits_of((x @ J.T if J is not None else x)[None], U, eps)[0]
    ids = np.argsort(-z)[:k]
    return [(int(i), tok.decode([int(i)]), float(z[i])) for i in ids]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=30)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--panel-a", choices=("hist", "example"), default="hist")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = str(args.threads)
    import torch
    from extract.extract_hidden_states import load_tokenizer
    torch.set_num_threads(args.threads)
    m = models.get("v3")
    S = torch.load(m["states"]["wikitext"], map_location="cpu", weights_only=False, mmap=True)
    H, eps, L = S["H"], float(S["rms_norm_eps"]), args.layer
    P, T = H.shape[0], H.shape[1]
    tok = load_tokenizer(S["model_path"])
    U = (np.load(m["unembed_dir"] / "lm_head_weight.npy").astype(np.float32)
         * np.load(m["unembed_dir"] / "rms_norm_weight.npy").astype(np.float32)[None, :])
    cls = np.array([CLASSES.index(token_class(tok.decode([i]))) for i in range(U.shape[0])])
    pos = np.unique(np.round(np.linspace(0, T - 1, 24)).astype(int))          # the depth diagnostic's grid
    keep = np.ones((P, T), bool)
    for p, t in m["saturated"]:
        keep[p - S["prompt_span"][0], t] = False
    released, penult = models.open_lens("v3", "shipped"), models.open_lens("v3", PENULT)
    OUT.mkdir(parents=True, exist_ok=True)
    info = {"layer": L, "depth_pct": round(100 * L / 60, 1), "k": args.k, "panel_a": args.panel_a,
            "released_lens_n": getattr(released, "n", None), "penultimate_lens_n": getattr(penult, "n", None),
            "vocab_han_share": float((cls == HAN).mean())}
    if args.panel_a == "hist":
        counts = {name: han_counts(H, keep, U, eps, lz, L, pos, cls, args.k)
                  for name, lz in (("logit lens", None), ("J-lens, final-layer target", released), ("J-lens, penultimate target", penult))}
        info["n_activations"] = int(len(next(iter(counts.values()))))
        info["hist"] = {name: np.bincount(c, minlength=args.k + 1).tolist() for name, c in counts.items()}
        info["summary"] = {name: {"median": float(np.median(c)), "q25": float(np.percentile(c, 25)), "q75": float(np.percentile(c, 75)),
                                  "frac_majority_han": float(np.mean(c > args.k / 2))} for name, c in counts.items()}
        for name, v in info["summary"].items():
            print(f"  {name:28s} Han in top {args.k}: median {v['median']:.0f} (IQR {v['q25']:.0f}-{v['q75']:.0f}); "
                  f"majority Han in {v['frac_majority_han']:.0%} of {info['n_activations']}")
    else:
        (p, t), share, med = pick_activation(H, keep, U, eps, released, L, pos, cls, S["input_ids"], tok)
        x = H[p, t, L].float().numpy()
        lists = {"logit lens": top_tokens(x, None, U, eps, tok, args.k),
                 "J-lens, final-layer target": top_tokens(x, released[L], U, eps, tok, args.k),
                 "J-lens, penultimate target": top_tokens(x, penult[L], U, eps, tok, args.k)}
        ids = S["input_ids"][p].tolist()
        info.update({"prompt": int(p + S["prompt_span"][0]), "position": int(t), "input_token": tok.decode([ids[t]]),
                     "context": tok.decode(ids[max(0, t - 12):t + 1]), "han_share_top100_released": share,
                     "median_han_share_top100_released": med,
                     "lists": {name: [{"id": i, "token": s_, "logit": z, "han": bool(cls[i] == HAN)} for i, s_, z in v]
                               for name, v in lists.items()}})
        print(f"activation: paragraph {info['prompt']} position {t} ({info['context']!r}); Han share of top 100 under "
              f"the released lens {share:.2f} (median {med:.2f})")
    (OUT / "hero_main_panel_a.json").write_text(json.dumps(info, indent=1, ensure_ascii=False))

    n_pen = info["penultimate_lens_n"]
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(15.5, 4.9) if args.panel_a == "hist" else (15.5, 5.6),
                                  gridspec_kw={"width_ratios": [1, 1, 1] if args.panel_a == "hist" else [1.15, 1, 1]})

    if args.panel_a == "hist":
        # (a) Han tokens among the top k, histogram per lens
        style(a, f"(a) Han tokens among a readout's top {args.k}, at {info['depth_pct']:.0f}% depth", "share of activations →", (0, 0))
        a.set_xlabel(f"Han tokens among the top {args.k} →", fontsize=9, color=MUTED)
        xs = np.arange(args.k + 1)
        for (name, col, ls, lw) in (("logit lens", GRAY, DOT, 1.6), ("J-lens, final-layer target", BLUE, "-", 2.0),
                                    ("J-lens, penultimate target", ORANGE, "-", 2.2)):
            h = np.array(info["hist"][name], float) / info["n_activations"]
            a.plot(xs, h, color=col, ls=ls, lw=lw, drawstyle="steps-mid", zorder=3)
        a.set_xlim(-0.5, args.k + 0.5)
        a.set_xticks(range(0, args.k + 1, 5 if args.k >= 15 else 2))
        a.set_ylim(0, None)
        a.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
        hist = {n: np.array(v, float) / info["n_activations"] for n, v in info["hist"].items()}
        h_pen, h_log = hist["J-lens, penultimate target"], hist["logit lens"]
        a.text(0.9, h_pen[0], "J-lens, penultimate target", fontsize=7.8, color=ORANGE, va="center")
        xl = int(np.argmax(h_log)) + 1.4
        a.text(xl, h_log[int(np.argmax(h_log))] + 0.004, "logit lens", fontsize=7.8, color=GRAY, va="bottom")
        h_fin = np.array(info["hist"]["J-lens, final-layer target"], float) / info["n_activations"]
        a.text(args.k - 0.3, h_fin[-1] + 0.012, "J-lens, final-layer target", fontsize=7.8, color=BLUE, ha="right", va="bottom")
    else:
        # (a) one rule-picked readout, three lenses
        a.set_axis_off()
        a.set_title(f"(a) One readout at {info['depth_pct']:.0f}% depth, three lenses", fontsize=10.5, color=INK, loc="left")
        lists = {n: [(d["id"], d["token"], d["logit"]) for d in v] for n, v in info["lists"].items()}
        cols = [("logit lens", "logit lens", GRAY), ("J-lens, final-layer target", "J-lens, final target", BLUE),
                ("J-lens, penultimate target", "J-lens, penultimate", ORANGE)]
        top = 0.85
        for j_, (name, head, col) in enumerate(cols):
            xj = 0.02 + j_ * 0.335
            a.text(xj, top, head, transform=a.transAxes, fontsize=8.6, color=col, va="top", fontweight="bold")
            for r_, (i_, s_, z) in enumerate(lists[name]):
                han = cls[i_] == HAN
                a.text(xj, top - 0.07 - r_ * 0.058, s_.replace("\n", "\\n"), transform=a.transAxes, fontsize=9.0,
                       color=HAN_TINT if han else INK, va="top", family=CJK_FAMILY if han else "DejaVu Sans")

    # (b) script share, two targets, log axis
    frame(b, "(b) Script share of readout variance (η²)", "median over activations, log scale →")
    for name, col, ls, lw in (("shipped", BLUE, "-", 2.0), ("target60_same", BLUE, DASH, 1.4), ("target59", ORANGE, "-", 2.2)):
        xs, ys = series(name, "eta2_script3")
        b.plot(xs, ys, color=col, ls=ls, lw=lw, zorder=3)
    xs, ys = series("logit", "eta2_script3", below_output=False)
    b.plot(xs, ys, color=GRAY, ls=DOT, lw=1.6, zorder=3)
    b.plot([xs[-1]], [ys[-1]], marker="D", ms=6, color=GRAY, mec="white", mew=0.8, zorder=5)
    b.set_yscale("log"); b.set_ylim(1e-3, 1.0); b.set_xlim(0, 100)
    b.text(2, 0.62, f"final-layer target: released lens (solid), {n_pen} prompts (dashed)", fontsize=7.8, color=BLUE, va="bottom")
    b.text(2, 0.036, f"penultimate-layer target, the same {n_pen} prompts", fontsize=7.8, color=ORANGE, va="bottom")
    b.text(2, 0.0034, "logit lens", fontsize=7.8, color=GRAY, va="bottom")

    # (c) the offset does not follow the input
    frame(c, "(c) Han − Latin readout offset", "median over activations, sd units →")
    c.axhline(0, color=AXIS, lw=0.9, zorder=1)
    for name, col, ls in (("shipped", BLUE, "-"), ("shipped_wikizh", ORANGE, "-"), ("logit", BLUE, DOT), ("logit_wikizh", ORANGE, DOT)):
        xs, ys = series(name, "han_minus_latin_mean_logit_sd")
        c.plot(xs, ys, color=col, ls=ls, lw=2.0 if ls == "-" else 1.4, zorder=3)
    for name, col, dy in (("shipped", BLUE, -7), ("shipped_wikizh", ORANGE, 7)):
        xs, ys = series(name, "han_minus_latin_mean_logit_sd", below_output=False)
        c.plot([xs[-1]], [ys[-1]], marker="D", ms=6, color=col, mec="white", mew=0.8, zorder=5)
        endlabel(c, xs[-1], ys[-1], f"model's output, {'English' if name == 'shipped' else 'Chinese'}: {ys[-1]:+.1f}", col, dy)
    xe, ye = series("shipped", "han_minus_latin_mean_logit_sd"); xz, yz = series("shipped_wikizh", "han_minus_latin_mean_logit_sd")
    c.text(xe[-1] - 1, ye[-1] - 0.12, "J-lens, English text", fontsize=7.8, color=BLUE, ha="right", va="top")
    c.text(xz[-1] - 1, yz[-1] + 0.12, "J-lens, Chinese text", fontsize=7.8, color=ORANGE, ha="right", va="bottom")
    c.text(50, 0.12, "logit lens, both", fontsize=7.8, color=GRAY, ha="center", va="bottom")
    c.set_ylim(-1.9, 1.9); c.set_xlim(0, 100)

    fig.tight_layout(w_pad=2.2)
    if args.panel_a == "example":
        # (a) the input on one line: "Input:", the preceding words (italic, muted), the read-out token (bold), packed on a
        # shared baseline at draw time. Leading words are dropped until the line fits inside the panel.
        from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea
        r = fig.canvas.get_renderer()
        for n_before in range(12, 3, -1):
            before = "…" + tok.decode(ids[max(0, t - n_before):t]).replace("\n", " ")
            box = AnchoredOffsetbox(
                loc="upper left", frameon=False, pad=0, borderpad=0, bbox_to_anchor=(0.02, 0.975),
                bbox_transform=a.transAxes,
                child=HPacker(align="baseline", pad=0, sep=0, children=[
                    TextArea("Input:  ", textprops=dict(color=MUTED, fontsize=9.0)),
                    TextArea(before, textprops=dict(color=MUTED, fontsize=9.0, style="italic")),
                    TextArea(info["input_token"], textprops=dict(color=INK, fontsize=9.4, fontweight="bold", style="italic"))]))
            a.add_artist(box)
            fig.canvas.draw()
            if box.get_window_extent(r).x1 <= a.get_window_extent(r).x1 - 4:
                break
            box.remove()
        print(f"input line: {n_before} preceding tokens")
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"hero_main.{ext}", dpi=170, facecolor="white")
    print(f"wrote {OUT / 'hero_main.png'}")


if __name__ == "__main__":
    main()
