"""
plot_readout_hist.py — why the kurtosis panel goes negative: one readout's token scores, stacked by language.

One typical held-out English activation (WikiText, prompt set of wikitext_states.pt), read several ways. Each panel
shows how the readout's 129,280 token scores are spread, in standard deviations from the readout's mean, with the
vocabulary stacked by language (Latin-alphabet, Chinese-character, everything else) and each group scaled by its share
of the vocabulary, so the outline is the whole readout. "Typical" is the activation with the median kurtosis, under
the first lens listed, among the first 40 prompts at one position. Histograms and summary statistics are cached, so
re-plotting takes seconds.

The layers are illustrative: 18 is where the released lens's median kurtosis bottoms out (-0.92, 30% depth), and 20
is the first grid layer where the ten-prompt final-target lens has left its noisy early range. The released lens's
median kurtosis is negative from 17% to 63% of depth, so any layer in that band shows the split to some degree.

    python scripts/jlens/plot_readout_hist.py                # main: layer 18, released final-target lens vs logit lens
    python scripts/jlens/plot_readout_hist.py --which paired # layer 20: final vs penultimate target on the same ten
                                                             #   prompts, beside the logit lens
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts")); sys.path.insert(0, str(REPO / "scripts/jlens"))
OUT = REPO / "results/jlens_workspace"
POS, N = 50, 40
BINS = np.linspace(-4, 6, 141)
INK, MUTED, AXIS = "#1b2733", "#5f6f80", "#b9c4cf"
LAT_FILL, ZH_FILL, OTHER_FILL = "#566372", "#b02a86", "#dfe4ea"   # Latin, Chinese, everything else

CONFIGS = {
    "main": {"layer": 18, "cache": "readout_hist.json", "out": "readout_hist",
             "arms": [("final", "lens_v3_filtered40.pt", "(a) Final-layer target"),
                      ("logit", None, "(b) Logit lens, same activation")]},
    "paired": {"layer": 20, "cache": "readout_hist_paired.json", "out": "readout_hist_paired",
               "arms": [("final10", "lens_v3_target60_same_prompts.pt", "(a) Final-layer target"),
                        ("penult10", "lens_v3_target59.pt", "(b) Penultimate target"),
                        ("logit", None, "(c) Logit lens, same activation")]},
}


def compute(cfg):
    import torch
    from score_eval_states import WEIGHTS, rms_norm
    from score_lazy import LazyLens
    from bilingual_diagnostics import token_class, CLASSES
    from extract.extract_hidden_states import load_tokenizer
    torch.set_num_threads(4)
    layer = cfg["layer"]
    st = torch.load(REPO / "outputs/jlens/wikitext_states.pt", map_location="cpu", weights_only=False, mmap=True)
    tok = load_tokenizer(st["model_path"])
    rms_w = np.load(WEIGHTS / "rms_norm_weight.npy").astype(np.float32)
    W = np.load(WEIGHTS / "lm_head_weight.npy", mmap_mode="r")                 # fp16; read in chunks
    V, eps = W.shape[0], st["rms_norm_eps"]
    cls = np.array([CLASSES.index(token_class(tok.decode([i]))) for i in range(V)])
    han, lat = cls == CLASSES.index("Han"), cls == CLASSES.index("Latin")
    h = st["H"][:N, POS, layer].float().numpy()

    def scores(y):
        yn = rms_norm(y, rms_w, eps).astype(np.float32)
        out = np.empty((y.shape[0], V), np.float32)
        for a in range(0, V, 16384):
            out[:, a:a + 16384] = yn @ W[a:a + 16384].astype(np.float32).T
        return (out - out.mean(1, keepdims=True)) / out.std(1, keepdims=True)

    z = {}
    for key, lens_file, _ in cfg["arms"]:
        y = h if lens_file is None else h @ np.array(LazyLens(REPO / "outputs/jlens" / lens_file)[layer]).astype(np.float32).T
        z[key] = scores(y)
    kurt = {k: (v ** 4).mean(1) - 3 for k, v in z.items()}
    i0 = int(np.argsort(kurt[cfg["arms"][0][0]])[N // 2])
    res = {"layer": layer, "position": POS, "n_activations": N, "activation_index": i0, "bins": BINS.tolist(),
           "shares": {"latin": float(lat.mean()), "chinese": float(han.mean()), "other": float(1 - lat.mean() - han.mean())}}
    for k, v in z.items():
        gap = (v[:, han].mean(1) - v[:, lat].mean(1)) / np.sqrt(0.5 * (v[:, han].var(1) + v[:, lat].var(1)))
        res[k] = {"gap_sd_p10_50_90": np.percentile(gap, [10, 50, 90]).tolist(),
                  "kurtosis_median": float(np.median(kurt[k])), "max_score": float(v[i0].max()),
                  "hist_latin": np.histogram(v[i0, lat], BINS, density=True)[0].tolist(),
                  "hist_chinese": np.histogram(v[i0, han], BINS, density=True)[0].tolist(),
                  "hist_all": np.histogram(v[i0], BINS, density=True)[0].tolist()}
    (OUT / cfg["cache"]).write_text(json.dumps(res, indent=1))
    return res


def smooth(y, sd_bins=1.6):
    k = np.arange(-6, 7); g = np.exp(-0.5 * (k / sd_bins) ** 2); g /= g.sum()
    return np.convolve(y, g, mode="same")


def plot(cfg, res):
    sh = res["shares"]
    edges = np.array(res["bins"]); x = 0.5 * (edges[1:] + edges[:-1])
    plt.rcParams.update({"font.size": 15, "axes.labelsize": 15, "xtick.labelsize": 14})
    n = len(cfg["arms"])
    fig, axes = plt.subplots(1, n, figsize=(5.9 * n, 4.3), sharey=True)
    for ax, (key, _, title) in zip(axes, cfg["arms"]):
        r = res[key]
        lat = smooth(np.array(r["hist_latin"])) * sh["latin"]
        zh = smooth(np.array(r["hist_chinese"])) * sh["chinese"]
        tot = smooth(np.array(r["hist_all"]))
        other = np.clip(tot - lat - zh, 0, None)
        # stacked: Latin at the bottom, Chinese on top of it, everything else above; the top edge is the whole readout
        ax.fill_between(x, 0, lat, color=LAT_FILL, lw=0)
        ax.fill_between(x, lat, lat + zh, color=ZH_FILL, lw=0)
        ax.fill_between(x, lat + zh, lat + zh + other, color=OTHER_FILL, lw=0)
        ax.plot(x, lat + zh + other, color=INK, lw=1.3)
        ax.set_title(title, loc="left", color=INK, fontsize=16.5, fontweight="bold", pad=12)
        ax.set_xlim(-3.6, 5.2); ax.set_ylim(0, None)
        ax.set_yticks([])
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(AXIS)
        ax.tick_params(colors=MUTED)
    handles = [Patch(color=ZH_FILL, label="Chinese tokens"), Patch(color=LAT_FILL, label="Latin tokens"),
               Patch(color=OTHER_FILL, label="everything else")]
    axes[-1].legend(handles=handles, frameon=False, fontsize=14.5, loc="upper right", handlelength=1.4)
    fig.supxlabel("token score, in standard deviations from the readout's mean", color=MUTED, fontsize=15, y=0.06)
    fig.tight_layout(w_pad=3, rect=(0, 0.1, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{cfg['out']}.{ext}", dpi=170, facecolor="white")
    print(f"wrote {OUT / (cfg['out'] + '.png')}")
    for key, _, _ in cfg["arms"]:
        r = res[key]
        print(f"  {key:9s} gap {r['gap_sd_p10_50_90'][1]:+.2f} sd (10-90 {r['gap_sd_p10_50_90'][0]:+.2f}..{r['gap_sd_p10_50_90'][2]:+.2f}),"
              f" median kurtosis {r['kurtosis_median']:+.2f}, top score in the plotted readout {r['max_score']:.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=sorted(CONFIGS), default="main")
    cfg = CONFIGS[ap.parse_args().which]
    cache = OUT / cfg["cache"]
    res = json.loads(cache.read_text()) if cache.exists() else compute(cfg)
    plot(cfg, res)


if __name__ == "__main__":
    main()
