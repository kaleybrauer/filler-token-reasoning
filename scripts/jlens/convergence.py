"""
convergence.py — when is the J-lens converged *for our use*?

Prompt counts borrowed from other people's models (praxagent n=24,
xiangchensong flat from n=60, the paper's n=1000) are not a stopping rule for
ours. `jlens.fit`'s own `max_d_mean` diagnostic tracks the matrix, which is
necessary but not what we consume. This tracks the READOUT instead: decode a
fixed probe set of extracted states through the lens at successive n and measure
how much the decoded tokens still move.

    agreement(n) = fraction of (example, layer, position) probes whose
                   number-token argmax is unchanged from the previous snapshot

When that saturates (~>=0.99 and flat), the lens has stopped changing in the
only way that affects our heatmaps — whatever n that happens at.

CPU only, reads the fit checkpoint (running sum / n_done), never touches the
GPUs, and is safe to run against a live fit: jlens writes checkpoints atomically
(tmp + os.replace), so a read always sees a complete file.

Usage:
    python scripts/jlens/convergence.py --once
    python scripts/jlens/convergence.py --watch     # re-runs on each new checkpoint
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]


def rms_norm(x, w, eps=1e-6):
    return (x / np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)) * w


def load_probe_states(extraction_dir: Path, n_examples: int, positions, layers):
    """{(pos, layer): (n_ex, d) float32} for correct examples only."""
    files = sorted(extraction_dir.glob("prob_*.pkl"))
    picked, data = [], []
    for f in files:
        with open(f, "rb") as fh:
            d = pickle.load(fh)
        if not d.get("model_correct", False):
            continue
        data.append(d)
        picked.append(f.name)
        if len(data) >= n_examples:
            break
    if not data:
        raise SystemExit(f"no correct examples in {extraction_dir}")

    avail_pos = [p for p in data[0]["states"] if p.startswith("pos_")]
    positions = [p for p in positions if p in avail_pos] or sorted(avail_pos)[:5]
    avail_layers = set(data[0]["states"][positions[0]])
    layers = [l for l in layers if l in avail_layers]

    probes = {}
    for pos in positions:
        for layer in layers:
            probes[(pos, layer)] = np.stack(
                [d["states"][pos][layer].astype(np.float32) for d in data]
            )
    truth = {
        "fact_value_1": np.array([d["fact_value_1"] for d in data]),
        "fact_value_2": np.array([d["fact_value_2"] for d in data]),
        "answer": np.array([d["answer"] for d in data]),
    }
    return probes, truth, len(data), positions, layers


def decode(probes, jac, lm_head_num, num_vals, norm_w):
    """{(pos,layer): predicted number per example} through the lens."""
    preds = {}
    for (pos, layer), vecs in probes.items():
        J = jac.get(layer)
        if J is None:
            continue
        H = rms_norm(vecs @ J.T, norm_w)
        preds[(pos, layer)] = num_vals[np.argmax(H @ lm_head_num.T, axis=1)]
    return preds


def cell_accuracy(preds, truth):
    """Per-(pos, layer) hit rate for each ground-truth quantity.

    This is what the heatmaps report -- an aggregate over examples -- and it is
    the statistic the stopping decision should be made on.
    """
    return {name: {f"{p}/L{l}": float((preds[(p, l)] == vals).mean())
                   for (p, l) in preds}
            for name, vals in truth.items()}


def compare_accuracy(cur, prev):
    """How much the reported heatmap moved, per quantity.

    `rel_change` is smooth and extrapolatable; `peak` is the discrete conclusion
    ("which cell wins"), which is what actually has to stop moving.
    """
    out = {}
    for name in cur:
        keys = sorted(cur[name])
        a = np.array([cur[name][k] for k in keys])
        b = np.array([prev[name][k] for k in keys])
        peak_a, peak_b = keys[int(a.argmax())], keys[int(b.argmax())]
        out[name] = {
            "rel_change": round(float(np.linalg.norm(a - b) / (np.linalg.norm(b) or 1.0)), 4),
            "max_abs_change": round(float(np.abs(a - b).max()), 4),
            "peak": peak_a,
            "peak_acc": round(float(a.max()), 3),
            "peak_moved": peak_a != peak_b,
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path,
                    default=REPO_ROOT / "outputs/jlens/lens_v3.fitckpt")
    ap.add_argument("--extraction-dir", type=Path,
                    default=REPO_ROOT / "data/extracted_states_2fact_allpos/dots_10")
    ap.add_argument("--lm-head", type=Path,
                    default=REPO_ROOT / "data/model_weights/deepseek_v3/lm_head_weight.npy")
    ap.add_argument("--rms-norm", type=Path,
                    default=REPO_ROOT / "data/model_weights/deepseek_v3/rms_norm_weight.npy")
    ap.add_argument("--model-path", default="/workspace/models/deepseek-v3-awq")
    ap.add_argument("--n-examples", type=int, default=50)
    ap.add_argument("--positions", default="pos_001,pos_005,pos_009,pos_013,pos_016")
    ap.add_argument("--layers", default="35,40,43,46,49,52,55,59")
    ap.add_argument("--threads", type=int, default=8,
                    help="Keep low: a GPU fit is usually running alongside")
    ap.add_argument("--out", type=Path,
                    default=REPO_ROOT / "outputs/jlens/convergence.jsonl")
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--poll-s", type=int, default=300)
    args = ap.parse_args()

    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["MKL_NUM_THREADS"] = str(args.threads)
    import torch
    torch.set_num_threads(args.threads)

    import sys
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from extract.extract_hidden_states import load_tokenizer
    tokenizer = load_tokenizer(args.model_path)
    number_tokens = {}
    for val in range(300):
        ids = tokenizer.encode(str(val), add_special_tokens=False)
        if len(ids) == 1:
            number_tokens[ids[0]] = val
    num_ids = sorted(number_tokens)
    num_vals = np.array([number_tokens[t] for t in num_ids])
    lm_head_num = np.load(args.lm_head).astype(np.float32)[num_ids]
    norm_w = np.load(args.rms_norm).astype(np.float32)

    probes, truth, n_ex, positions, layers = load_probe_states(
        args.extraction_dir, args.n_examples,
        args.positions.split(","), [int(x) for x in args.layers.split(",")])
    print(f"probe set: {n_ex} correct examples x {len(positions)} positions "
          f"x {len(layers)} layers = {len(probes) * n_ex} decode points")

    # Resume the comparison across restarts. The probe set must be identical or
    # the stored predictions are not comparable, so gate on a signature.
    sig = [n_ex, positions, layers]
    state_path = args.out.with_suffix(".state.pkl")
    prev_preds = prev_acc = prev_n = None
    if state_path.exists():
        try:
            with open(state_path, "rb") as fh:
                st = pickle.load(fh)
            if st.get("sig") == sig:
                prev_preds, prev_acc, prev_n = st["preds"], st["acc"], st["n_done"]
                print(f"resuming comparison from stored n_done={prev_n}")
            else:
                print("stored state has a different probe set; ignoring it")
        except Exception as exc:
            print(f"could not read {state_path}: {exc}")
    seen_mtime = None
    while True:
        if not args.checkpoint.exists():
            print("waiting for a checkpoint...")
        else:
            mtime = args.checkpoint.stat().st_mtime
            if mtime != seen_mtime:
                seen_mtime = mtime
                t0 = time.perf_counter()
                state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
                n_done = state["n_done"]
                jac = {int(l): (J.float() / n_done).numpy()
                       for l, J in state["jacobian_sum"].items()
                       if int(l) in layers}
                del state
                preds = decode(probes, jac, lm_head_num, num_vals, norm_w)
                del jac
                acc = cell_accuracy(preds, truth)

                row = {"n_done": n_done, "n_examples": n_ex,
                       "load_decode_s": round(time.perf_counter() - t0, 1)}
                if prev_preds is not None and prev_n != n_done:
                    same = np.concatenate([(preds[k] == prev_preds[k]) for k in preds])
                    shift = np.concatenate(
                        [np.abs(preds[k].astype(int) - prev_preds[k].astype(int))
                         for k in preds])
                    row["prev_n"] = prev_n
                    row["agreement_vs_prev"] = round(float(same.mean()), 4)
                    row["median_abs_shift"] = float(np.median(shift))
                    # Per-cell agreement, so a stable band is distinguishable
                    # from a stable average over unstable cells.
                    row["worst_cells"] = sorted(
                        ({"cell": f"{p}/L{l}",
                          "agree": round(float((preds[(p, l)] == prev_preds[(p, l)]).mean()), 3)}
                         for (p, l) in preds),
                        key=lambda d: d["agree"])[:5]
                    # The statistic we actually report. Per-probe argmax agreement
                    # above is a much stricter functional than any conclusion drawn
                    # from it -- cells on an argmax boundary flip at every n, so that
                    # series asymptotes below 1.0. Heatmap accuracy is an aggregate
                    # over examples and settles sooner: stop when THIS stops moving.
                    row["heatmap"] = compare_accuracy(acc, prev_acc)
                # How often the lens lands on a true operand, for context only
                row["hit_A1"] = round(float(np.mean([
                    (preds[k] == truth["fact_value_1"]).mean() for k in preds])), 4)
                row["hit_A2"] = round(float(np.mean([
                    (preds[k] == truth["fact_value_2"]).mean() for k in preds])), 4)

                args.out.parent.mkdir(parents=True, exist_ok=True)
                with open(args.out, "a") as f:
                    f.write(json.dumps(row) + "\n")
                print(json.dumps(row), flush=True)
                prev_preds, prev_acc, prev_n = preds, acc, n_done
                with open(state_path, "wb") as fh:
                    pickle.dump({"sig": sig, "preds": preds, "acc": acc,
                                 "n_done": n_done}, fh)

        if args.once or not args.watch:
            break
        time.sleep(args.poll_s)


if __name__ == "__main__":
    main()
