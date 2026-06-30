"""
Generate release/tables/table_shuffled_control.tex — paired rows comparing
original-decode top-K accuracy vs the shuffled-tokens control.

Reads:
  release/judge_outputs/<model>_<task>_dots_10_<judge>_<prompt>.json
  release/judge_outputs_shuffled/<model>_<task>_dots_10_<judge>_<prompt>.json

Run from repo root:
    python scripts/release/generate_shuffled_control_table.py
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ORIG = REPO / "release" / "judge_outputs"
SHUF = REPO / "release" / "judge_outputs_shuffled"
OUT = REPO / "release" / "tables" / "table_shuffled_control.tex"

K_VALUES = [1, 2, 3, 5, 10]


def hits_per_K(records, kind):
    """Count top-K hits. kind='flat' for letterpos/capitalpos/1fact (bool per K),
    'both' for 2fact (dict per K with 'both' subkey)."""
    out = {K: 0 for K in K_VALUES}
    n = 0
    for r in records:
        n += 1
        tk = r.get("top_k", {})
        for K in K_VALUES:
            v = tk.get(str(K), tk.get(K))
            if kind == "both" and isinstance(v, dict):
                if v.get("both"):
                    out[K] += 1
            elif v is True or v == 1:
                out[K] += 1
    return n, out


def fmt(hits, n):
    if n == 0:
        return "--"
    return f"{hits / n * 100:.1f}"


# (display_label, model, task, prompt_kind, top_k_kind, model_label_for_display)
SPECS = [
    ("1-fact",       "deepseek_v3", "1fact",      "neutral",   "flat", "DeepSeek V3"),
    ("2-fact",       "deepseek_v3", "2fact",      "neutral",   "both", "DeepSeek V3"),
    ("Letter-pos.",  "deepseek_v3", "letterpos",  "neutral",   "flat", "DeepSeek V3"),
    ("Letter-pos.",  "deepseek_v3", "letterpos",  "chemistry", "flat", "DeepSeek V3"),
    ("Capital-pos.", "deepseek_v3", "capitalpos", "neutral",   "flat", "DeepSeek V3"),
    ("Capital-pos.", "deepseek_v3", "capitalpos", "geography", "flat", "DeepSeek V3"),
    ("2-fact",       "kimi_k2",     "2fact",      "neutral",   "both", "Kimi K2"),
    ("Letter-pos.",  "kimi_k2",     "letterpos",  "neutral",   "flat", "Kimi K2"),
    ("Letter-pos.",  "kimi_k2",     "letterpos",  "chemistry", "flat", "Kimi K2"),
    ("Capital-pos.", "kimi_k2",     "capitalpos", "neutral",   "flat", "Kimi K2"),
    ("Capital-pos.", "kimi_k2",     "capitalpos", "geography", "flat", "Kimi K2"),
]
JUDGES = ["haiku", "sonnet"]
COND = "dots_10"


def main():
    rows = []
    for task_disp, model, task, prompt, kind, model_disp in SPECS:
        for judge in JUDGES:
            orig_p = ORIG / f"{model}_{task}_{COND}_{judge}_{prompt}.json"
            shuf_p = SHUF / f"{model}_{task}_{COND}_{judge}_{prompt}.json"
            if not orig_p.exists() or not shuf_p.exists():
                print(f"  skip (missing): {model} {task} {judge} {prompt}; "
                      f"orig={orig_p.exists()} shuf={shuf_p.exists()}")
                continue
            n_o, h_o = hits_per_K(json.load(open(orig_p)), kind)
            n_s, h_s = hits_per_K(json.load(open(shuf_p)), kind)
            rows.append({
                "task_disp": task_disp, "model_disp": model_disp,
                "judge": judge.capitalize(), "prompt": prompt,
                "task_kind": kind,
                "n_o": n_o, "h_o": h_o, "n_s": n_s, "h_s": h_s,
            })

    lines = []
    lines.append(r"\begin{table*}[h]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\caption{Shuffled-tokens control. For each example, the "
                 r"top-50 token list is replaced with a different example's "
                 r"top-50 from the same (model, task, condition); the judge "
                 r"is then scored against the original example's ground "
                 r"truth. The control isolates whether the judge is reading "
                 r"example-specific content vs.\ relying on a task-level "
                 r"prior. Rows show the original (residual-decode) "
                 r"top-$K$ accuracy followed by the shuffled-control "
                 r"accuracy, on dots\_10. Cost in accuracy from shuffling "
                 r"is reported as $\Delta$.}")
    lines.append(r"\label{tab:shuffled_control}")
    lines.append(r"\begin{tabular}{l l l l l r r r r r r}")
    lines.append(r"\toprule")
    lines.append(r"Task & Model & Judge & Prompt & Method & $n$ & "
                 r"$K{=}1$ & $K{=}2$ & $K{=}3$ & $K{=}5$ & $K{=}10$ \\")
    lines.append(r"\midrule")
    for i, r in enumerate(rows):
        if i > 0:
            lines.append(r"\addlinespace[3pt]")
        common = (f"{r['task_disp']} & {r['model_disp']} & "
                  f"{r['judge']} & {r['prompt']}")
        cells_o = [fmt(r['h_o'][K], r['n_o']) for K in K_VALUES]
        cells_s = [fmt(r['h_s'][K], r['n_s']) for K in K_VALUES]
        deltas = []
        for K in K_VALUES:
            if r['n_o'] > 0 and r['n_s'] > 0:
                d = r['h_s'][K] / r['n_s'] * 100 - r['h_o'][K] / r['n_o'] * 100
                deltas.append(f"{d:+.1f}")
            else:
                deltas.append("--")
        # 2fact has no top-1
        if r["task_kind"] == "both":
            cells_o[0] = "--"; cells_s[0] = "--"; deltas[0] = "--"
        lines.append(f"{common} & original & {r['n_o']:,} & "
                     + " & ".join(cells_o) + r" \\")
        lines.append(r"  &     &     &     & shuffled & "
                     + f"{r['n_s']:,} & " + " & ".join(cells_s) + r" \\")
        lines.append(r"  &     &     &     & $\Delta$ & "
                     + " & " + " & ".join(deltas) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n")
    print(f"\nWrote {OUT}")
    print(f"  {len(rows)} (model, task, judge, prompt) paired rows")


if __name__ == "__main__":
    main()
