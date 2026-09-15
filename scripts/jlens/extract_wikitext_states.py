"""
extract_wikitext_states.py — V3 residuals on HELD-OUT pretraining-like text, at
every position, so Figure 28's panels (a), (b) and (c) can be computed on the same
kind of distribution the paper uses.

Why this exists. Panel (d) is a property of W_U J and needs no activations, but
(a) next-token accuracy, (b) readout kurtosis and (c) top-1 autocorrelation all
depend on what the model is being asked. The states we already cache are the
551 lens-eval items at ONE readout position each: short synthetic probes with a
forced answer slot, about as far from prose as a prompt gets, and single-position,
so (c) is not even defined on them. Computing the panels there and comparing
against the paper would confound the model with the prompt format.

Held out by construction. The lens was fitted on corpus prompts 0..99 with 100..114
refitted for the pre-filter, so this defaults to 200..299 — disjoint from both, and
from the 115..199 band left as a buffer.

Conventions are taken from jlens itself (encode with force_bos, ActivationRecorder
on model.layers storing output[0]), identical to extract_eval_states.py, so a state
here is the same object the lens would transport. Positions below --skip-first are
dropped: they are attention sinks with atypical residual statistics, and the fit
excludes them for the same reason.

Forward passes only — no backward, none of the autograd patches, no fitting.
Cost is dominated by the one-time model load (~13 min on 3xH200).

    100 prompts x 128 positions x 61 layers x 7168 dims, fp16  ->  ~11 GB

    setsid nohup bash logs/extract_wikitext.sh >/dev/null 2>&1 </dev/null &
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

import jlens  # noqa: E402  (must precede REPO_ROOT/scripts on sys.path)
from jlens.hooks import ActivationRecorder  # noqa: E402

if "filler-token-reasoning" in (jlens.__file__ or "scripts/jlens namespace pkg"):
    raise SystemExit(f"scripts/jlens shadowed the jlens library: {jlens.__file__}")

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from extract_eval_states import check_unembed  # noqa: E402
from v3_autograd import load_v3_awq            # noqa: E402

FIT_SPANS = [(0, 100), (100, 115)]   # fitted prompts, and the individually refitted ones


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="/workspace/models/deepseek-v3-awq")
    ap.add_argument("--prompts", type=Path, nargs="+",
                    default=[REPO_ROOT / "scripts/jlens/prompts_wikitext.json"],
                    help="one or more corpora, all extracted after a single model load")
    ap.add_argument("--start", type=int, default=200, help="first corpus index (held out)")
    ap.add_argument("--n-prompts", type=int, default=100)
    ap.add_argument("--max-seq-len", type=int, default=128,
                    help="Matches the sequence length the lens was fitted at")
    ap.add_argument("--skip-first", type=int, default=16,
                    help="Drop leading attention-sink positions, as the fit does")
    ap.add_argument("--max-memory-gib", default="130,122,108")
    ap.add_argument("--device-map", default=None)
    ap.add_argument("--layer-stride", type=int, default=1,
                    help="Keep every Nth layer (1 = all 61, ~11 GB)")
    ap.add_argument("--out", type=Path, nargs="+",
                    default=[REPO_ROOT / "outputs/jlens/wikitext_states.pt"],
                    help="one output per --prompts corpus")
    ap.add_argument("--wait-for-prompts", type=int, default=0, metavar="MIN",
                    help="load the model first, then wait up to MIN minutes for the corpus "
                         "files (still being built) to appear")
    args = ap.parse_args()
    if len(args.prompts) != len(args.out):
        raise SystemExit("--prompts and --out must pair up")

    # Index-based holding out applies to the corpus the lens was fitted on; any other corpus
    # (build_wiki_corpus.py's Wikipedia paragraphs) is held out by construction.
    fit_source = json.loads((REPO_ROOT / "scripts/jlens/prompts_wikitext.json").read_text())["source"]
    lo, hi = args.start, args.start + args.n_prompts

    def load_job(path, out_path):
        corpus = json.loads(path.read_text())
        if corpus["source"] == fit_source:
            for a, b in FIT_SPANS:
                if lo < b and a < hi:
                    raise SystemExit(f"prompts {lo}..{hi-1} overlap the fitted span {a}..{b-1}; "
                                     f"the panels must be measured on held-out text")
        prompts = corpus["prompts"][lo:hi]
        if len(prompts) < args.n_prompts:
            raise SystemExit(f"corpus has {len(corpus['prompts'])} prompts, need index {hi}")
        print(f"corpus {corpus['source']}  prompts {lo}..{hi-1} (held out) -> {out_path}", flush=True)
        return corpus, prompts, out_path

    def wait_for_job(path, out_path):
        deadline = time.time() + 60 * args.wait_for_prompts
        while True:
            try:
                return load_job(path, out_path)     # a file mid-write fails to parse; try again
            except (FileNotFoundError, json.JSONDecodeError):
                if time.time() > deadline:
                    raise SystemExit(f"{path} not ready after {args.wait_for_prompts} min")
                print(f"  waiting for {path}", flush=True)
                time.sleep(60)

    if not args.wait_for_prompts:     # fail fast, before the model load
        jobs = [load_job(p, o) for p, o in zip(args.prompts, args.out)]

    model, tokenizer = load_v3_awq(args.model_path,
                                   max_memory_gib=args.max_memory_gib,
                                   device_map=args.device_map)
    lens_model = jlens.from_hf(model, tokenizer)   # forward-only: no autograd patches
    n_layers, d_model = lens_model.n_layers, lens_model.d_model
    eps = float(model.config.rms_norm_eps)
    layers = list(range(0, n_layers, args.layer_stride))
    print(f"  {lens_model}  rms_norm_eps={eps}  keeping {len(layers)} layers", flush=True)

    T = args.max_seq_len - args.skip_first
    for k, (path, out_path) in enumerate(zip(args.prompts, args.out)):
        # waiting per corpus: one still being built doesn't hold up one that is ready
        corpus, prompts, out_path = wait_for_job(path, out_path) if args.wait_for_prompts else jobs[k]
        gb = len(prompts) * T * len(layers) * d_model * 2 / 1e9
        print(f"\n{corpus['source']}: will store {len(prompts)} x {T} positions x {len(layers)} layers "
              f"x {d_model}  =  {gb:.1f} GB fp16", flush=True)

        H = torch.empty(len(prompts), T, len(layers), d_model, dtype=torch.float16)
        kept_ids, seq_lens = [], []
        t0 = time.perf_counter()
        for i, p in enumerate(prompts):
            input_ids = lens_model.encode(p, max_length=args.max_seq_len)
            n = int(input_ids.shape[-1])
            if n < args.max_seq_len:
                raise SystemExit(f"prompt {lo+i} tokenised to {n} < {args.max_seq_len}; "
                                 f"positions would not align across prompts")
            seq_lens.append(n)
            kept_ids.append(input_ids[0, args.skip_first:].cpu())
            with torch.no_grad(), ActivationRecorder(lens_model.layers, at=layers) as rec:
                lens_model.forward(input_ids)
                for j, L in enumerate(layers):
                    H[i, :, j] = rec.activations[L][0][args.skip_first:].detach().half().cpu()
            if (i + 1) % 10 == 0:
                el = time.perf_counter() - t0
                print(f"  {i+1}/{len(prompts)}  {el/(i+1):.1f}s/prompt  "
                      f"eta {(len(prompts)-i-1)*el/(i+1)/60:.1f} min", flush=True)

        # Same gate as the eval-state extraction: the offline readout must reproduce the
        # model's own unembedding before any of these states are scored.
        probe = H[:8, -1, :, :].reshape(-1, len(layers), d_model)[:8]
        out = {"model_path": args.model_path, "corpus": corpus["source"],
               "prompt_span": [lo, hi], "held_out_from": FIT_SPANS,
               "n_layers": n_layers, "layers": layers, "d_model": d_model,
               "rms_norm_eps": eps, "max_seq_len": args.max_seq_len,
               "skip_first": args.skip_first, "force_bos": True,
               "seq_lens": seq_lens, "input_ids": torch.stack(kept_ids),
               "H": H,   # [n_prompts, n_positions, n_layers, d_model] fp16
               "unembed_check": check_unembed(lens_model, probe, eps)}
        print(f"  G-UNEMBED: {out['unembed_check']}", flush=True)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_path.with_suffix(".tmp")
        torch.save(out, tmp)
        tmp.replace(out_path)
        print(f"wrote {out_path} ({out_path.stat().st_size/2**30:.1f} GiB) "
              f"in {(time.perf_counter()-t0)/60:.1f} min", flush=True)
        del H, out


if __name__ == "__main__":
    main()
