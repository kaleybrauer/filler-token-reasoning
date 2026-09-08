"""heatmap_peaks.py — best (position, layer) exact-match cell per target in one or
more decode_2fact_*.json heatmaps (logit lens vs J-lens arms), plus each arm's
accuracy at the reference cells. Accuracies only (no deltas).

    python scripts/jlens/heatmap_peaks.py LABEL=path.json [LABEL=path.json ...]
"""
import json, sys

REF = {"A1": ("pos_001", 50), "A2": ("pos_005", 51), "A1A2": ("pos_016", 60)}  # logit-lens peaks (sum = A1A2)


def load(path):
    d = json.load(open(path))
    keys = [k for k in d if k.startswith("pos_")]
    targets = sorted({k.split("_")[1] for k in d[keys[0]]["0"] if k.endswith("_exact")})
    return d, keys, targets


def main():
    arms = [a.split("=", 1) for a in sys.argv[1:]]
    for label, path in arms:
        d, keys, targets = load(path)
        n = d["_n"]
        print(f"\n{label}  ({path}, n={n}, layers {d['_layers'][0]}..{d['_layers'][-1]})")
        for t in targets:
            best = max(((d[p][L][f"frac_{t}_exact"], p, int(L)) for p in keys for L in d[p]))
            ref = REF.get(t)
            ref_s = ""
            if ref and ref[0] in d and str(ref[1]) in d[ref[0]]:
                ref_s = f"   at logit-lens peak {ref[0]} L{ref[1]}: {100 * d[ref[0]][str(ref[1])][f'frac_{t}_exact']:5.1f}%"
            print(f"  {t:>5}: best {100 * best[0]:5.1f}% @ {best[1]} L{best[2]}{ref_s}")


if __name__ == "__main__":
    main()
