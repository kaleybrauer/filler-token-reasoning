"""
run_qwen35_analysis.py — every CPU analysis of the Qwen3.5 activations once they are back from the GPU box
(QWEN35_GPU_RUNBOOK.md step 6), one job at a time on the 16 GB box.

Per model, gated by check_qwen35_transfer.py:
  readouts   Figure 28 panels (a)-(c) on held-out WikiText (workspace_readouts.py)
  depth      the panel (b) decomposition -- full-vocabulary, Latin-only and Han-only kurtosis, script offset,
             top-k script make-up -- on WikiText, Chinese and English Wikipedia, lens and logit lens
             (bilingual_diagnostics.py depth)
  pairs      translation pairs between the Latin and Han top readouts (translation_pairs.py)
  geometry   residual-space language geometry on the concept states (language_geometry.py)
The 397B also runs its 24-prompt lens (praxagent) through readouts, WikiText depth and geometry, the
fit-robustness role V3's half-fit lenses played. A step whose output exists is skipped, so an interrupted run
is simply relaunched. Each step logs to logs/qwen35_<step>.log and prints <step>_OK / _FAILED / _SKIPPED.

    nohup python scripts/jlens/run_qwen35_analysis.py qwen35_122b qwen35_397b_fp8 > logs/qwen35_analysis.out 2>&1 &
"""
from __future__ import annotations

import argparse, os, shlex, subprocess, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import models

REPO = models.REPO
HEAVY = ("cka_layers", "survey_published", "workspace_readouts", "bilingual_diagnostics", "language_geometry")


def heavy_job_running():
    me = os.getpid()
    for d in Path("/proc").iterdir():
        if d.name.isdigit() and int(d.name) != me:
            try:
                args = (d / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
            except OSError:
                continue
            if any(f"scripts/jlens/{h}.py" in args for h in HEAVY):
                return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+", help="qwen35_122b, qwen35_397b_fp8")
    ap.add_argument("--python", default=sys.executable, help="interpreter command for each step")
    ap.add_argument("--logs", type=Path, default=REPO / "logs")
    ap.add_argument("--check-args", default="", help="extra check_qwen35_transfer.py arguments (dry runs only)")
    ap.add_argument("--no-wait", action="store_true", help="do not wait for other jlens jobs to finish")
    args = ap.parse_args()
    py = shlex.split(args.python)
    args.logs.mkdir(parents=True, exist_ok=True)
    while not args.no_wait and heavy_job_running():
        print(f"waiting for a running jlens job to finish ({time.strftime('%H:%M')})", flush=True)
        time.sleep(120)

    def run(name, done, cmd, env=None):
        if done is not None and done.exists():
            print(f"{name}_SKIPPED (exists)", flush=True)
            return True
        t0 = time.time()
        with open(args.logs / f"qwen35_{name}.log", "w") as log:
            ok = subprocess.run(py + cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
                                env={**os.environ, **(env or {})}).returncode == 0
        print(f"{name}_{'OK' if ok else 'FAILED'} ({(time.time() - t0) / 60:.0f} min)", flush=True)
        return ok

    S = "scripts/jlens/"
    for M in args.models:
        m, A = models.get(M), models.QS / M / "analysis"
        if not run(f"check_{M}", None, [S + "check_qwen35_transfer.py", M] + shlex.split(args.check_args)):
            print(f"nothing run for {M}: see {args.logs / f'qwen35_check_{M}.log'}", flush=True)
            continue
        alt = sorted(m["alt_lenses"])                       # the 397B's 24-prompt lens
        run(f"readouts_{M}_shipped", A / "workspace_readouts_shipped.json",
            [S + "workspace_readouts.py", "--model", M, "--label", "shipped"])
        for key in alt:
            run(f"readouts_{M}_{key}", A / f"workspace_readouts_{key}.json",
                [S + "workspace_readouts.py", "--model", M, "--lens", str(m["alt_lenses"][key]), "--label", key])
        run(f"depth_{M}_wikitext", A / "bilingual_depth_logit_topk.npz",
            [S + "bilingual_diagnostics.py", "depth", "--model", M, "--lenses", "shipped", "logit"])
        for lang in ("zh", "en"):
            run(f"depth_{M}_wiki{lang}", A / f"bilingual_depth_logit_wiki{lang}_topk.npz",
                [S + "bilingual_diagnostics.py", "depth", "--model", M, "--lenses", "shipped", "logit",
                 "--states", str(m["states"][f"wiki_{lang}"]), "--tag", f"_wiki{lang}"])
        for key in alt:
            run(f"depth_{M}_wikitext_{key}", A / f"bilingual_depth_{key}_topk.npz",
                [S + "bilingual_diagnostics.py", "depth", "--model", M, "--lenses", key])
        topk = [str(p) for lens in ("shipped", "logit") for tag in ("", "_wikizh", "_wikien")
                if (p := A / f"bilingual_depth_{lens}{tag}_topk.npz").exists()]
        if topk:
            run(f"pairs_{M}", A / "translation_pairs.json", [S + "translation_pairs.py", "--model", M] + topk,
                env={"OMP_NUM_THREADS": "1"})
        run(f"geometry_{M}", A / "language_geometry.json",
            [S + "language_geometry.py", "--model", M, "--lenses", "raw", "shipped"] + alt)
    print("QWEN35_ANALYSIS_DONE", flush=True)


if __name__ == "__main__":
    main()
