"""
extract_concept_states.py — V3 residuals for the matched concept prompts (build_concept_prompts.py),
for the residual-space language tests. Forward passes only, batch 1 (the regime of every other
cached state), all 61 layers, at the concept token (the last token of every prompt).

The state before the concept (the template's last token) does not depend on the concept in a
causal model, so it is stored once per (language, template), from a forward of the template alone.
Prompts whose token sequences are identical (bare zh and ja prompts with the same string) are run
once and mapped.

Same conventions as extract_wikitext_states.py (jlens encode with force_bos, ActivationRecorder on
model.layers storing output[0]); every prompt is asserted to end on the expected concept token.
G-UNEMBED is recorded in the file. A failed check still saves the states (so the forwards can be
inspected rather than rerun) but exits non-zero, and language_geometry.py refuses such a file.
5,400 prompts x 61 layers x 7168 fp16 ~ 4.7 GB.

    setsid nohup bash logs/extract_concepts.sh >/dev/null 2>&1 </dev/null &
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch

import jlens  # noqa: E402  (must precede REPO_ROOT/scripts on sys.path)
from jlens.hooks import ActivationRecorder  # noqa: E402
if not hasattr(jlens, "from_hf"):
    raise SystemExit(f"scripts/jlens shadowed the jlens library: {jlens.__file__}")

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts")); sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_eval_states import check_unembed  # noqa: E402
from v3_autograd import load_v3_awq            # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="/workspace/models/deepseek-v3-awq")
    ap.add_argument("--prompts", type=Path, default=REPO_ROOT / "scripts/jlens/concept_prompts.json")
    ap.add_argument("--max-memory-gib", default="130,122,108")
    ap.add_argument("--device-map", default=None)
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "outputs/jlens/concept_states.pt")
    args = ap.parse_args()
    spec = json.loads(args.prompts.read_text())
    prompts = spec["prompts"]
    print(f"{len(prompts)} prompts, {len(spec['concepts'])} concepts, languages "
          f"{sorted({p['lang'] for p in prompts})}", flush=True)

    # fail fast, before the 13-minute load: the prompt file was built on another machine
    from extract.extract_hidden_states import load_tokenizer
    tok = load_tokenizer(args.model_path)
    bad = [i for i, p in enumerate(prompts)
           if (lambda ids: ids[-1] != p["expect_last_id"] or len(ids) != p["n_tokens"])(tok(p["text"])["input_ids"])]
    if bad:
        raise SystemExit(f"{len(bad)} prompts tokenise differently here than when built, e.g. {prompts[bad[0]]}")
    print("tokenizer pre-check passed for every prompt", flush=True)

    model, tokenizer = load_v3_awq(args.model_path, max_memory_gib=args.max_memory_gib, device_map=args.device_map)
    lm = jlens.from_hf(model, tokenizer)
    n_layers, d = lm.n_layers, lm.d_model
    eps = float(model.config.rms_norm_eps)
    layers = list(range(n_layers))
    uniq, dup_of = {}, []
    for i, p in enumerate(prompts):
        ids = lm.encode(p["text"])
        if int(ids[0, -1]) != p["expect_last_id"] or int(ids.shape[-1]) != p["n_tokens"]:
            raise SystemExit(f"prompt {i} tokenised differently on the GPU box: {ids[0].tolist()} vs expected "
                             f"last {p['expect_last_id']}, n {p['n_tokens']}")
        dup_of.append(uniq.setdefault(tuple(ids[0].tolist()), i))
    run = sorted(set(dup_of))
    print(f"{len(run)} distinct token sequences to run ({len(prompts) - len(run)} duplicates mapped)", flush=True)

    def states_at_last(text, pos=-1):
        ids = lm.encode(text)
        with torch.no_grad(), ActivationRecorder(lm.layers, at=layers) as rec:
            lm.forward(ids)
            return torch.stack([rec.activations[L][0][pos].detach().half().cpu() for L in layers]), ids

    H = torch.empty(len(prompts), n_layers, d, dtype=torch.float16)
    t0 = time.perf_counter()
    for k, i in enumerate(run):
        H[i], _ = states_at_last(prompts[i]["text"])
        if (k + 1) % 200 == 0:
            el = time.perf_counter() - t0
            print(f"  {k+1}/{len(run)}  {el/(k+1):.2f}s/prompt  eta {(len(run)-k-1)*el/(k+1)/60:.1f} min", flush=True)
    for i, j in enumerate(dup_of):
        if j != i:
            H[i] = H[j]

    # the pre-concept state of a bare prompt is the BOS state (same for every language): position 0
    st, ids = states_at_last(prompts[0]["text"], pos=0)
    template_states = {"bos": {"text": "", "ids": ids[0, :1].tolist(), "H": st}}
    for lang, tpls in spec["templates"].items():
        for ti, tpl in enumerate(tpls, start=1):            # template index 0 is bare
            st, ids = states_at_last(tpl)
            template_states[f"{lang}:{ti}"] = {"text": tpl, "ids": ids[0].tolist(), "H": st}
    print(f"template states: {sorted(template_states)}", flush=True)

    probe = H[:8]                                        # [8, n_layers, d] at the concept token
    out = {"model_path": args.model_path, "prompts_file": str(args.prompts), "spec": spec,
           "n_layers": n_layers, "layers": layers, "d_model": d, "rms_norm_eps": eps, "force_bos": True,
           "dup_of": dup_of, "template_states": template_states,
           "H": H,   # [n_prompts, n_layers, d] fp16 at the concept token
           "unembed_check": check_unembed(lm, probe, eps)}
    print(f"  G-UNEMBED: {out['unembed_check']}", flush=True)
    sat = int((H[:, -1].float().abs() >= 65504).any(-1).sum())
    print(f"  final-layer fp16-saturated states: {sat}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(".tmp"); torch.save(out, tmp); tmp.replace(args.out)
    print(f"wrote {args.out} ({args.out.stat().st_size/2**30:.2f} GiB) in {(time.perf_counter()-t0)/60:.1f} min")
    if not out["unembed_check"]["passed"]:
        raise SystemExit("G-UNEMBED FAILED: states saved for inspection only; the analysis will refuse them")


if __name__ == "__main__":
    main()
