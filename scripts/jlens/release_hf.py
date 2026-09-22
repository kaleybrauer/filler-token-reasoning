"""
release_hf.py — package the DeepSeek-V3 J-lenses for the Hugging Face Hub (kbrauer/deepseek-v3-jacobian-lens).

    python scripts/jlens/release_hf.py halves     # write the two disjoint 50-prompt lenses as JacobianLens files
    python scripts/jlens/release_hf.py check      # the halves' prompt-weighted mean must equal the released lens
    python scripts/jlens/release_hf.py meta       # sidecars, per-prompt norms and fit prompts into the staging dir
    python scripts/jlens/release_hf.py upload     # create the repo (private) and upload; then compare sha256s

Big files are written to and uploaded from outputs/jlens/ on /workspace; the staging dir holds only small files.
HF_HOME is pointed at /workspace so the hub client's chunk cache does not land on the container's 5 GB root disk.
The half lenses are written layer by layer through memory-mapped fp16 arrays, so building one never holds more than
a layer or two in RAM (the sandbox has a 16 GB cgroup).
"""
from __future__ import annotations

import argparse, hashlib, json, os, re, shutil, sys
from pathlib import Path

os.environ.setdefault("HF_HOME", "/workspace/.hf_home")
import numpy as np

REPO = Path(__file__).resolve().parents[2]
J = REPO / "outputs/jlens"
STAGE = Path("/workspace/jlens_release")
REPO_ID = "kbrauer/deepseek-v3-jacobian-lens"
sys.path.insert(0, str(REPO / "scripts")); sys.path.insert(0, str(Path(__file__).resolve().parent))

HALVES = {"half_a": "n50", "half_b": "n50b"}      # file name -> clean_floor.SUBSETS name
FILES = {                                        # path in repo -> local path
    "lens.pt": J / "lens_v3_filtered40.pt",
    "halves/half_a.pt": J / "lens_v3_half_a.pt",
    "halves/half_b.pt": J / "lens_v3_half_b.pt",
    "penultimate/target59_n10.pt": J / "lens_v3_target59.pt",
    "penultimate/target60_n10.pt": J / "lens_v3_target60_same_prompts.pt",
}


def build_halves():
    import torch
    from clean_floor import build_subset
    for out_name, subset in HALVES.items():
        out = J / f"lens_v3_{out_name}.pt"
        if out.exists():
            print(f"{out} exists; skipping"); continue
        lz = build_subset(J, J / "per_prompt", 40.0, subset)
        tmp = J / f"_tmp_{out_name}"; tmp.mkdir(exist_ok=True)
        for L in lz.layers:
            np.save(tmp / f"L{L:02d}.npy", lz[L].astype(np.float16))
            print(f"  {out_name} L{L:02d}", flush=True) if L % 10 == 0 else None
        Jd = {L: torch.from_numpy(np.load(tmp / f"L{L:02d}.npy", mmap_mode="r")) for L in lz.layers}
        d = int(next(iter(Jd.values())).shape[0])
        part = out.with_suffix(".pt.part")
        torch.save({"J": Jd, "n_prompts": int(lz.n), "source_layers": list(lz.layers), "d_model": d}, part)
        part.replace(out)
        spans = [list(s) for s in lz.spans]
        idx = sorted({i for lo, hi in lz.spans for i in range(lo, hi)} - set(lz.giants) | set(lz.added))
        (J / f"lens_v3_{out_name}.meta.json").write_text(json.dumps({
            "built_from": "outputs/jlens/lens_v3*.fitckpt running sums, differenced by snapshot, minus the pre-filtered "
                          "prompts' own Jacobians, plus the replacement prompts (scripts/jlens/clean_floor.py)",
            "subset": subset, "n_prompts": int(lz.n), "prompt_indices": idx, "original_spans": spans,
            "excluded_by_prefilter": sorted(lz.giants), "replacements": sorted(lz.added),
            "target_layer": 60, "dtype": "fp16"}, indent=1))
        shutil.rmtree(tmp)
        print(f"wrote {out} (n={lz.n}) and its meta", flush=True)


def check():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from score_lazy import LazyLens
    a, b, full = (LazyLens(FILES[k]) for k in ("halves/half_a.pt", "halves/half_b.pt", "lens.pt"))
    print(f"n: half_a {a.n}, half_b {b.n}, released {full.n}")
    assert a.n + b.n == full.n, "halves do not add up to the released lens"
    ia = json.loads((J / "lens_v3_half_a.meta.json").read_text())["prompt_indices"]
    ib = json.loads((J / "lens_v3_half_b.meta.json").read_text())["prompt_indices"]
    assert not set(ia) & set(ib), "halves overlap"
    worst = 0.0
    for L in (0, 15, 30, 45, 59):
        m = (a.n * a[L].astype(np.float64) + b.n * b[L]) / (a.n + b.n)
        f = full[L].astype(np.float64)
        rel = float(np.abs(m - f).max() / np.abs(f).max())
        worst = max(worst, rel)
        print(f"  L{L:2d}: max |mean of halves - released| / max|released| = {rel:.2e}")
    assert worst < 5e-3, "halves do not reproduce the released lens"
    print("CHECK_OK: the halves are disjoint and their prompt-weighted mean is the released lens (to fp16 rounding)")


def logged(log, rx=re.compile(r"\(idx (\d+)\).*?max\|\|J\|\|/sqrt\(d\)=([0-9.]+)")):
    out = {}
    for line in (REPO / log).read_text(errors="replace").splitlines():
        m = rx.search(line)
        if m:
            out[int(m.group(1))] = float(m.group(2))
    return out


def meta():
    from build_filtered_lens import logged_norms
    (STAGE / "meta").mkdir(parents=True, exist_ok=True)
    rel = json.loads((J / "lens_v3_filtered40.meta.json").read_text())
    for src, dst in (("lens_v3_filtered40.meta.json", "lens.meta.json"), ("lens_v3_half_a.meta.json", "half_a.meta.json"),
                     ("lens_v3_half_b.meta.json", "half_b.meta.json"), ("lens_v3_target59.meta.json", "target59_n10.meta.json"),
                     ("lens_v3_target60_same_prompts.meta.json", "target60_n10.meta.json")):
        shutil.copy(J / src, STAGE / "meta" / dst)
    t60 = {**logged_norms(), **{int(k): v["refit_norm"] for k, v in rel["added"].items()}}
    t59 = logged("logs/jlens_penult.log")
    excluded = sorted(int(k) for k in rel["excluded"])
    used = sorted(set(range(100)) - set(excluded) | {int(k) for k in rel["added"]})
    (STAGE / "meta/per_prompt_norms.json").write_text(json.dumps({
        "statistic": "max over source layers of ||J_l||_F / sqrt(d_model), per fit prompt",
        "target_layer_60": {str(k): round(v, 4) for k, v in sorted(t60.items())},
        "target_layer_59": {str(k): round(v, 4) for k, v in sorted(t59.items())},
        "released_lens_prompts": used, "excluded_by_prefilter_cutoff_40": excluded}, indent=1))
    corpus = json.loads((REPO / "scripts/jlens/prompts_wikitext.json").read_text())
    need = sorted(set(used) | set(excluded) | set(range(100, 110)))
    (STAGE / "meta/fit_prompts.json").write_text(json.dumps({
        "source": corpus["source"], "loader": corpus["loader"], "min_chars": corpus["min_chars"],
        "license_note": "Text excerpts from WikiText-103 (Merity et al., 2016), CC BY-SA 3.0.",
        "prompts": {str(i): corpus["prompts"][i] for i in need}}, indent=1, ensure_ascii=False))
    shutil.copy(REPO / "scripts/jlens/HF_README.md", STAGE / "README.md")
    print(f"staged {sorted(p.name for p in STAGE.rglob('*') if p.is_file())}")


def sha256(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def upload():
    from huggingface_hub import HfApi
    # the HF_TOKEN in probing_env.sh is read-only; a write token lives in /workspace/keys/hf_token
    tok_file = Path("/workspace/keys/hf_token")
    token = tok_file.read_text().strip() if tok_file.exists() else os.environ["HF_TOKEN"]
    api = HfApi(token=token)
    print(f"uploading as {api.whoami()['name']} with a token of role "
          f"{api.whoami().get('auth', {}).get('accessToken', {}).get('role', '?')}", flush=True)
    api.create_repo(REPO_ID, repo_type="model", private=True, exist_ok=True)
    api.upload_folder(repo_id=REPO_ID, folder_path=str(STAGE), commit_message="model card and metadata")
    for path_in_repo, local in FILES.items():
        print(f"uploading {local.name} -> {path_in_repo} ({local.stat().st_size / 1e9:.2f} GB)", flush=True)
        api.upload_file(path_or_fileobj=str(local), path_in_repo=path_in_repo, repo_id=REPO_ID,
                        commit_message=f"add {path_in_repo}")
    remote = {f.path: f for f in api.list_repo_tree(REPO_ID, recursive=True, expand=True) if hasattr(f, "lfs") and f.lfs}
    ok = True
    for path_in_repo, local in FILES.items():
        r = remote.get(path_in_repo)
        mine = sha256(local)
        match = r is not None and r.lfs.sha256 == mine
        ok &= match
        print(f"  {path_in_repo}: {'OK' if match else 'MISMATCH'} (sha256 {mine[:12]}…)", flush=True)
    print("UPLOAD_OK" if ok else "UPLOAD_MISMATCH")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["halves", "check", "meta", "upload"])
    {"halves": build_halves, "check": check, "meta": meta, "upload": upload}[ap.parse_args().step]()
