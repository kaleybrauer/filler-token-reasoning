"""
check_qwen35_transfer.py — checks on the Qwen3.5 activations copied back from the GPU box, before any analysis.

For each model tag under outputs/jlens_qwen35/:
  hard (exit non-zero)  every expected file present and no half-written .tmp; the run's G-UNEMBED passed; the
                        exported unembedding equals the Hub range-read copy used for the lens-only measures
                        (so the norm convention agrees); layer count, d_model and vocabulary match models.py;
                        paragraph spans are the held-out ones, 100 paragraphs x 112 positions; the concept spec
                        is the committed concept_prompts_qwen35.json with 300 duplicates mapped; no non-finite
                        values; duplicate concept prompts hold identical states
  reported              fp16 clamping (from the run's log and from a scan for +-65504), paragraphs whose ids
                        differ from a local re-tokenisation, the FP8-vs-bf16 regime check, provenance

Writes outputs/jlens_qwen35/<tag>/transfer_check.json.

    python scripts/jlens/check_qwen35_transfer.py qwen35_122b qwen35_397b_fp8
"""
from __future__ import annotations

import argparse, hashlib, json, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import models

REPO = models.REPO
CONCEPTS = REPO / "scripts/jlens/concept_prompts_qwen35.json"
CORPORA = {"wikitext": ("scripts/jlens/prompts_wikitext.json", 200),     # as in extract_qwen35_states.py
           "wiki_zh": ("scripts/jlens/prompts_wiki_zh.json", 0),
           "wiki_en": ("scripts/jlens/prompts_wiki_en.json", 0)}
MAX_SEQ, SKIP_FIRST, FP16_MAX, VOCAB = 128, 16, 65504.0, 248320
TEMPLATE_KEYS = {"en:1", "en:2", "zh:1", "zh:2", "ja:1", "ja:2"}


class Report:
    def __init__(self):
        self.failures, self.notes = [], {}

    def require(self, ok, what):
        print(f"  {'ok  ' if ok else 'FAIL'} {what}", flush=True)
        if not ok:
            self.failures.append(what)
        return ok


def compare_unembed(d, hub, r):
    W, Wh = np.load(d / "lm_head_weight.npy", mmap_mode="r"), np.load(hub / "lm_head_weight.npy", mmap_mode="r")
    if not r.require(W.shape == Wh.shape, f"lm_head shape {W.shape} equals the Hub copy {Wh.shape}"):
        return
    diff = max(float(np.abs(W[i:i + 16384].astype(np.float32) - Wh[i:i + 16384].astype(np.float32)).max())
               for i in range(0, W.shape[0], 16384))
    g, gh = np.load(d / "rms_norm_weight.npy").astype(np.float64), np.load(hub / "rms_norm_weight.npy").astype(np.float64)
    gdiff = float(np.abs(g - gh).max())
    r.notes["unembed_vs_hub"] = {"lm_head_max_abs_diff": diff, "norm_multiplier_max_abs_diff": gdiff,
                                 "norm_multiplier_mean": float(g.mean())}
    r.require(diff == 0.0, f"lm_head equals the Hub copy exactly (max |diff| {diff:.3g})")
    r.require(gdiff < 1e-5, f"final-norm multiplier equals the Hub copy's 1 + w (max |diff| {gdiff:.3g}, mean {g.mean():.3f})")


def scan(H, chunk):
    """non-finite count, states with a component at the fp16 limit ([item, (position,) layer]) and, for
    paragraph states, each chunk's median state norm by layer; streams over the first axis"""
    nonfinite, sat, norms = 0, [], []
    for i in range(0, H.shape[0], chunk):
        x = H[i:i + chunk].float()
        nonfinite += int((~torch.isfinite(x)).sum())
        for idx in torch.nonzero((x.abs() >= FP16_MAX).any(-1)).tolist():
            sat.append([idx[0] + i] + idx[1:])
        if x.dim() == 4:
            norms.append(x.norm(dim=-1).flatten(0, 1).median(0).values)
    return nonfinite, sat, norms


def check_paragraphs(tag, m, out, r, tok, n_expected):
    for name, (src, lo) in CORPORA.items():
        path = out / f"{name}_states.pt"
        print(f" {name}", flush=True)
        if not r.require(path.exists(), f"{path.name} present"):
            continue
        S = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
        H = S["H"]
        r.require(tuple(H.shape) == (n_expected, MAX_SEQ - SKIP_FIRST, m["n_layers"], m["d_model"]),
                  f"H shape {tuple(H.shape)} = ({n_expected}, {MAX_SEQ - SKIP_FIRST}, {m['n_layers']}, {m['d_model']})")
        r.require(S["prompt_span"] == [lo, lo + n_expected], f"prompt span {S['prompt_span']} = [{lo}, {lo + n_expected}]")
        r.require(S["unembed_check"]["passed"], "states carry a passed G-UNEMBED")
        r.require((S["max_seq_len"], S["skip_first"]) == (MAX_SEQ, SKIP_FIRST), "max_seq_len 128, skip_first 16")
        prompts = json.loads((REPO / src).read_text())["prompts"][lo:lo + n_expected]
        mism = [lo + i for i, p in enumerate(prompts)
                if tok(p)["input_ids"][SKIP_FIRST:MAX_SEQ] != S["input_ids"][i].tolist()]
        nonfinite, sat, norms = scan(H, 4)
        r.require(nonfinite == 0, f"no non-finite values ({nonfinite})")
        med = torch.stack(norms).median(0).values.tolist()
        r.notes[name] = {"fp16_clamped_logged": S["fp16_clamped"], "saturated_states_prompt_pos_layer": sat[:200],
                         "n_saturated_states": len(sat), "ids_differ_from_local_tokenisation": mism,
                         "median_norm_by_layer": [round(v, 2) for v in med]}
        print(f"  fp16 clamping logged in {len(S['fp16_clamped'])} (prompt, layer) cells; {len(sat)} saturated states"
              f"{': ' + str(sat[:6]) if sat else ''}", flush=True)
        print(f"  {len(mism)} paragraphs whose ids differ from a local re-tokenisation{': ' + str(mism[:6]) if mism else ''}", flush=True)
        print(f"  median state norm by layer: first {med[0]:.1f}, middle {med[len(med) // 2]:.1f}, last {med[-1]:.1f}", flush=True)


def check_concepts(tag, m, out, r, max_concepts):
    path = out / "concept_states.pt"
    print(" concepts", flush=True)
    if not r.require(path.exists(), f"{path.name} present"):
        return
    S = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    spec = json.loads(CONCEPTS.read_text())
    prompts = spec["prompts"] if not max_concepts else [p for p in spec["prompts"] if p["concept"] < max_concepts]
    H = S["H"]
    r.require(tuple(H.shape) == (len(prompts), m["n_layers"], m["d_model"]),
              f"H shape {tuple(H.shape)} = ({len(prompts)}, {m['n_layers']}, {m['d_model']})")
    r.require(S["spec"]["prompts"] == prompts, "spec prompts equal the committed concept_prompts_qwen35.json")
    r.require(S["provenance"]["concept_prompts_sha256"] == hashlib.sha256(CONCEPTS.read_bytes()).hexdigest(),
              "the run's concept_prompts sha256 matches the committed file")
    r.require(S["unembed_check"]["passed"], "states carry a passed G-UNEMBED")
    r.require(set(S["template_states"]) == TEMPLATE_KEYS, f"template states {sorted(S['template_states'])}")
    dups = [(i, j) for i, j in enumerate(S["dup_of"]) if j != i]
    if not max_concepts:
        r.require(len(dups) == 300, f"{len(dups)} duplicate prompts mapped (expected 300)")
    sample = dups[:: max(1, len(dups) // 50)]
    r.require(all(torch.equal(H[i], H[j]) for i, j in sample), f"duplicates hold identical states ({len(sample)} checked)")
    nonfinite, sat, _ = scan(H, 256)
    r.require(nonfinite == 0, f"no non-finite values ({nonfinite})")
    r.notes["concepts"] = {"fp16_clamped_logged": S["fp16_clamped"], "saturated_states_prompt_layer": sat[:200],
                           "n_saturated_states": len(sat), "n_duplicates": len(dups)}
    print(f"  fp16 clamping logged in {len(S['fp16_clamped'])} states; {len(sat)} saturated (prompt, layer) states", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tags", nargs="+", help="models.MODELS keys: qwen35_122b, qwen35_397b_fp8")
    ap.add_argument("--root", type=Path, default=models.QS, help="directory holding <tag>/")
    ap.add_argument("--hub-dir", default=None, help="Hub range-read unembedding; default outputs/jlens/qwen35/<tag>/unembed_hub, 'none' skips")
    ap.add_argument("--n-paragraphs", type=int, default=100, help="dry runs only")
    ap.add_argument("--max-concepts", type=int, default=None, help="dry runs only")
    args = ap.parse_args()
    failed = []
    for tag in args.tags:
        m, out, r = models.get(tag), args.root / tag, Report()
        print(f"\n=== {tag}  ({out})", flush=True)
        tmp = sorted(p.name for p in out.rglob("*.tmp"))
        r.require(not tmp, f"no half-written .tmp files {tmp if tmp else ''}")
        d = out / "unembed"
        if r.require((d / "check.json").exists() and (d / "lm_head_weight.npy").exists(), "unembedding export present"):
            chk = json.loads((d / "check.json").read_text())
            r.require(chk["passed"], f"G-UNEMBED passed on the model (top-1 {chk['top1_agreement']}, corr {chk['logit_corr']}, "
                                     f"norm convention {chk['norm_convention']})")
            r.require(np.load(d / "lm_head_weight.npy", mmap_mode="r").shape[0] == VOCAB, f"vocabulary {VOCAB}")
            r.notes["g_unembed"] = chk
            hub = None if args.hub_dir == "none" else Path(args.hub_dir) if args.hub_dir else models.Q / tag / "unembed_hub"
            if hub is not None:
                compare_unembed(d, hub, r)
        run = out / "run.json"
        if r.require(run.exists(), "run.json present (the extraction finished)"):
            prov = json.loads(run.read_text())["provenance"]
            r.notes["provenance"] = prov
            print(f"  provenance: torch {prov['torch']}, transformers {prov['transformers']}, {len(prov['gpus'])}x "
                  f"{prov['gpus'][0] if prov['gpus'] else '?'}, model {prov['model']}", flush=True)
        if (out / "regime.pt").exists():
            st = torch.load(out / "regime.pt", map_location="cpu", mmap=True, weights_only=False)["stats"]
            r.notes["regime"] = st
            print(f"  regime (FP8 kernels vs bf16 compute): final top-1 agreement {st['final_top1_agreement']:.3f}; "
                  f"min mean cos {min(st['cos_mean']):.4f}; max mean rel diff {max(st['rel_diff_mean']):.4f}", flush=True)
        elif (out / "regime_error.json").exists():
            r.notes["regime"] = json.loads((out / "regime_error.json").read_text())
            print(f"  regime check errored on the box: {r.notes['regime']}", flush=True)
        else:
            r.notes["regime"] = "not run (no FP8 modules, or step skipped)"
            print("  regime check: not present (expected for the bf16 122B)", flush=True)
        check_paragraphs(tag, m, out, r, models.load_tokenizer_for(tag), args.n_paragraphs)
        check_concepts(tag, m, out, r, args.max_concepts)
        (out / "transfer_check.json").write_text(json.dumps({"failures": r.failures, **r.notes}, indent=1, default=str))
        print(f"{tag}: {'PASSED' if not r.failures else 'FAILED: ' + '; '.join(r.failures)}  -> {out / 'transfer_check.json'}", flush=True)
        failed += [tag] if r.failures else []
    if failed:
        raise SystemExit(f"transfer check failed for {failed}")


if __name__ == "__main__":
    main()
