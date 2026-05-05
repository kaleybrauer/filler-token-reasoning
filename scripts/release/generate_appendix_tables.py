"""
Generate LaTeX tables for the paper appendix from release/ data.

Produces four tables in release/tables/:
  table_1fact.tex          (DeepSeek V3 only)
  table_2fact.tex          (DeepSeek V3 + Kimi K2)
  table_letterpos_neutral.tex
  table_letterpos_leading.tex   (chemistry / geography prompts)

Each row: one (model, condition[, domain]) combination.
Each cell: top-K hit fraction at K = 1, 2, 3, 5, 10.
Includes a rule-based ("direct match") column that scores top tokens against
ground truth without an LLM judge: numeric-token equality for addition tasks,
EN/ZH alias substring match for letter-position. Both this and the LLM-judge
columns use ground truth, so neither is more "supervised" than the other —
they differ only in matching method.

Run from repo root:
    python scripts/release/generate_appendix_tables.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
REL = REPO / "release"
OUT = REL / "tables"

sys.path.insert(0, str(REPO / "scripts" / "analysis"))
from multilingual_coverage import token_matches  # for letterpos / capitalpos

K_VALUES = [1, 2, 3, 5, 10]
CONDS_2FACT = ["dots_10", "dots_25", "dots_50", "counting_10", "counting_25", "counting_50"]
CONDS_1FACT = ["dots_10", "dots_25", "dots_50", "dots_100",
               "counting_5", "counting_25", "counting_50",
               "alphabet_10", "alphabet_25"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def numeric_token_seq(top_tokens):
    out = []
    for tok in top_tokens:
        s = tok["str"].strip()
        if s and all("0" <= c <= "9" for c in s):
            out.append(int(s))
    return out


def supervised_1fact(records):
    """For 1-fact: hit if `fact_value` (A) is in top-K numeric tokens."""
    rows_per_K = {K: [0, 0] for K in K_VALUES}  # K -> [hits, n]
    for r in records:
        a = r.get("fact_value")
        if a is None:
            continue
        nums = numeric_token_seq(r["top_tokens"])
        for K in K_VALUES:
            rows_per_K[K][1] += 1
            if a in nums[:K]:
                rows_per_K[K][0] += 1
    return rows_per_K


def supervised_2fact(records):
    """For 2-fact: hit if BOTH A1 and A2 are in top-K numeric tokens."""
    rows_per_K = {K: [0, 0] for K in K_VALUES}
    for r in records:
        a1, a2 = r.get("fact_value_1"), r.get("fact_value_2")
        if a1 is None or a2 is None:
            continue
        nums = numeric_token_seq(r["top_tokens"])
        for K in K_VALUES:
            rows_per_K[K][1] += 1
            top = set(nums[:K])
            if a1 in top and a2 in top:
                rows_per_K[K][0] += 1
    return rows_per_K


def supervised_lp(records, aliases, truth_field):
    """For letterpos/capitalpos: hit if any of the top-K tokens contains a
    substring matching the entity in EN or ZH (or chemical symbol)."""
    rows_per_K = {K: [0, 0] for K in K_VALUES}
    for r in records:
        truth = r.get(truth_field)
        if not truth or truth not in aliases:
            continue
        ref = aliases[truth]
        for K in K_VALUES:
            rows_per_K[K][1] += 1
            for tok in r["top_tokens"][:K]:
                if token_matches(tok["str"], ref)[0]:
                    rows_per_K[K][0] += 1
                    break
    return rows_per_K


def fmt_pct(hits, n):
    if n == 0:
        return "--"
    return f"{hits / n * 100:.1f}"


def collect_judge(path):
    """Return {(cond, judge): {K: hits, "n": n}}."""
    d = json.load(open(path))
    out = {}
    for k, v in d.items():
        cond, judge = k.rsplit("_", 1)
        out[(cond, judge)] = {**{int(K): hits for K, hits in v["summary"].items()},
                              "n": v["n"]}
    return out


# ---------------------------------------------------------------------------
# Per-task table builders
# ---------------------------------------------------------------------------

def build_table(rows, caption, label):
    """rows: list of dicts with keys: model, domain (optional), prompt (optional),
       cond, n, sup_hits {K: hits}, haiku_hits {K: hits}, sonnet_hits {K: hits},
       group_label (str shown in left col), is_pooled (bool: emphasize)."""
    K = K_VALUES
    # Column count: 2 (group, n) + 5 supervised + 5 haiku + 5 sonnet
    body = []
    for r in rows:
        cells = [r["group_label"], f"{r['n']:,}"]
        for stem in ("sup_hits", "haiku_hits", "sonnet_hits"):
            for k in K:
                hits = r[stem].get(k)
                cells.append(fmt_pct(hits, r["n"]) if hits is not None else "--")
        line = " & ".join(cells) + r" \\"
        if r.get("is_pooled"):
            line = r"\midrule " + line.replace(r"\\", r" \\")
        body.append(line)

    header = (
        r"\begin{table*}[h]" + "\n"
        r"\centering" + "\n"
        r"\small" + "\n"
        r"\caption{" + caption + r"}" + "\n"
        r"\label{tab:" + label + r"}" + "\n"
        r"\begin{tabular}{l r " + " ".join(["r"] * (3 * len(K))) + r"}" + "\n"
        r"\toprule" + "\n"
        r"& & \multicolumn{" + str(len(K)) + r"}{c}{Direct match top-$K$ (\%)}"
        r" & \multicolumn{" + str(len(K)) + r"}{c}{Haiku judge top-$K$ (\%)}"
        r" & \multicolumn{" + str(len(K)) + r"}{c}{Sonnet judge top-$K$ (\%)} \\" + "\n"
        r"\cmidrule(lr){3-" + str(2 + len(K)) + r"}"
        r" \cmidrule(lr){" + str(3 + len(K)) + r"-" + str(2 + 2 * len(K)) + r"}"
        r" \cmidrule(lr){" + str(3 + 2 * len(K)) + r"-" + str(2 + 3 * len(K)) + r"}" + "\n"
        r"Group & $n$ & " + " & ".join([f"$K{{=}}{k}$" for k in K] * 3) + r" \\" + "\n"
        r"\midrule"
    )
    footer = r"\bottomrule" + "\n" + r"\end{tabular}" + "\n" + r"\end{table*}"
    return header + "\n" + "\n".join(body) + "\n" + footer


def pool(rows_per_K_list):
    """Sum hits and n across multiple rows_per_K dicts."""
    pooled = {K: [0, 0] for K in K_VALUES}
    for rpk in rows_per_K_list:
        for K, (h, n) in rpk.items():
            pooled[K][0] += h
            pooled[K][1] += n
    return pooled


# ---------------------------------------------------------------------------
# Build each task's data
# ---------------------------------------------------------------------------

def build_1fact():
    judge = collect_judge(REL / "accuracy_tables/llm_judge_deepseek_v3_1fact_neutral.json")
    rows = []
    sup_rpks = []
    haiku_pool = {K: [0, 0] for K in K_VALUES}
    sonnet_pool = {K: [0, 0] for K in K_VALUES}
    for cond in CONDS_1FACT:
        f = REL / f"top_tokens/deepseek_v3_1fact_{cond}.json"
        if not f.exists():
            continue
        records = json.load(open(f))
        sup = supervised_1fact(records)
        sup_rpks.append(sup)
        n_sup = sup[1][1]
        sup_hits = {K: sup[K][0] for K in K_VALUES}

        h = judge.get((cond, "haiku"), {"n": 0})
        s = judge.get((cond, "sonnet"), {"n": 0})
        haiku_hits = {K: h.get(K) for K in K_VALUES}
        sonnet_hits = {K: s.get(K) for K in K_VALUES}
        for K in K_VALUES:
            if h.get(K) is not None:
                haiku_pool[K][0] += h[K]; haiku_pool[K][1] += h["n"]
                sonnet_pool[K][0] += s[K]; sonnet_pool[K][1] += s["n"]

        rows.append({
            "group_label": cond.replace("_", r"\_"), "n": n_sup,
            "sup_hits": sup_hits, "haiku_hits": haiku_hits, "sonnet_hits": sonnet_hits,
        })

    pooled_sup = pool(sup_rpks)
    rows.append({
        "group_label": r"\textbf{Pooled}", "n": pooled_sup[1][1],
        "sup_hits": {K: pooled_sup[K][0] for K in K_VALUES},
        "haiku_hits": {K: haiku_pool[K][0] for K in K_VALUES},
        "sonnet_hits": {K: sonnet_pool[K][0] for K in K_VALUES},
        "is_pooled": True,
    })
    # The pooled n for haiku/sonnet may differ from supervised n — annotate using
    # haiku_pool for the n column? They're typically equal per condition. Use sup n.
    return rows


def build_2fact():
    rows = []
    section_pool_haiku = {K: [0, 0] for K in K_VALUES}
    section_pool_sonnet = {K: [0, 0] for K in K_VALUES}
    for model, model_label, judge_path in [
        ("deepseek_v3", "DeepSeek V3",
         REL / "accuracy_tables/llm_judge_deepseek_v3_2fact_neutral.json"),
        ("kimi_k2", "Kimi K2",
         REL / "accuracy_tables/llm_judge_kimi_k2_2fact_neutral.json"),
    ]:
        judge = collect_judge(judge_path)
        sup_rpks = []
        haiku_pool = {K: [0, 0] for K in K_VALUES}
        sonnet_pool = {K: [0, 0] for K in K_VALUES}
        # subheader as a row
        rows.append({
            "group_label": r"\multicolumn{17}{l}{\textit{" + model_label + r"}}",
            "n": "", "sup_hits": {}, "haiku_hits": {}, "sonnet_hits": {},
            "is_subheader": True,
        })
        for cond in CONDS_2FACT:
            f = REL / f"top_tokens/{model}_2fact_{cond}.json"
            if not f.exists():
                continue
            records = json.load(open(f))
            sup = supervised_2fact(records)
            sup_rpks.append(sup)
            n_sup = sup[1][1]
            sup_hits = {K: sup[K][0] for K in K_VALUES}
            h = judge.get((cond, "haiku"), {"n": 0})
            s = judge.get((cond, "sonnet"), {"n": 0})
            haiku_hits = {K: h.get(K) for K in K_VALUES}
            sonnet_hits = {K: s.get(K) for K in K_VALUES}
            for K in K_VALUES:
                if h.get(K) is not None:
                    haiku_pool[K][0] += h[K]; haiku_pool[K][1] += h["n"]
                    sonnet_pool[K][0] += s[K]; sonnet_pool[K][1] += s["n"]
            rows.append({
                "group_label": cond.replace("_", r"\_"), "n": n_sup,
                "sup_hits": sup_hits, "haiku_hits": haiku_hits, "sonnet_hits": sonnet_hits,
            })
        pooled_sup = pool(sup_rpks)
        rows.append({
            "group_label": r"\textbf{" + model_label + r" pooled}", "n": pooled_sup[1][1],
            "sup_hits": {K: pooled_sup[K][0] for K in K_VALUES},
            "haiku_hits": {K: haiku_pool[K][0] for K in K_VALUES},
            "sonnet_hits": {K: sonnet_pool[K][0] for K in K_VALUES},
            "is_pooled": True,
        })
    return rows


def build_letterpos(prompt_kind):
    """prompt_kind: 'neutral' or 'leading' (chemistry/geography)."""
    element_aliases = json.load(open(REPO / "data" / "element_aliases.json"))
    capital_aliases = json.load(open(REPO / "data" / "capital_aliases.json"))

    rows = []
    for model, model_label in [("deepseek_v3", "DeepSeek V3"), ("kimi_k2", "Kimi K2")]:
        for domain, task_key, judge_prompt, aliases, truth_field in [
            ("Elements", "letterpos",
             "neutral" if prompt_kind == "neutral" else "chemistry",
             element_aliases, "element"),
            ("Capitals", "capitalpos",
             "neutral" if prompt_kind == "neutral" else "geography",
             capital_aliases, "intermediate"),
        ]:
            judge_path = (REL / "accuracy_tables"
                          / f"llm_judge_{model}_{task_key}_{judge_prompt}.json")
            judge = collect_judge(judge_path)

            rows.append({
                "group_label": (r"\multicolumn{17}{l}{\textit{" + model_label
                                + ", " + domain + r"}}"),
                "n": "", "sup_hits": {}, "haiku_hits": {}, "sonnet_hits": {},
                "is_subheader": True,
            })
            sup_rpks = []
            haiku_pool = {K: [0, 0] for K in K_VALUES}
            sonnet_pool = {K: [0, 0] for K in K_VALUES}
            for cond in CONDS_2FACT:
                f = REL / f"top_tokens/{model}_{task_key}_{cond}.json"
                if not f.exists():
                    continue
                records = json.load(open(f))
                sup = supervised_lp(records, aliases, truth_field)
                sup_rpks.append(sup)
                n_sup = sup[1][1]
                sup_hits = {K: sup[K][0] for K in K_VALUES}
                h = judge.get((cond, "haiku"), {"n": 0})
                s = judge.get((cond, "sonnet"), {"n": 0})
                haiku_hits = {K: h.get(K) for K in K_VALUES}
                sonnet_hits = {K: s.get(K) for K in K_VALUES}
                for K in K_VALUES:
                    if h.get(K) is not None:
                        haiku_pool[K][0] += h[K]; haiku_pool[K][1] += h["n"]
                        sonnet_pool[K][0] += s[K]; sonnet_pool[K][1] += s["n"]
                rows.append({
                    "group_label": cond.replace("_", r"\_"), "n": n_sup,
                    "sup_hits": sup_hits,
                    "haiku_hits": haiku_hits, "sonnet_hits": sonnet_hits,
                })
            pooled_sup = pool(sup_rpks)
            rows.append({
                "group_label": (r"\textbf{" + model_label + ", " + domain
                                + r" pooled}"),
                "n": pooled_sup[1][1],
                "sup_hits": {K: pooled_sup[K][0] for K in K_VALUES},
                "haiku_hits": {K: haiku_pool[K][0] for K in K_VALUES},
                "sonnet_hits": {K: sonnet_pool[K][0] for K in K_VALUES},
                "is_pooled": True,
            })
    return rows


# ---------------------------------------------------------------------------
# Render tables (slightly different writer to handle subheaders + pooled rows)
# ---------------------------------------------------------------------------

def render_table(rows, caption, label):
    K = K_VALUES
    n_metric_cols = 3 * len(K)
    total_cols = 2 + n_metric_cols
    lines = []
    lines.append(r"\begin{table*}[h]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\caption{" + caption + r"}")
    lines.append(r"\label{tab:" + label + r"}")
    lines.append(r"\begin{tabular}{l r " + " ".join(["r"] * n_metric_cols) + r"}")
    lines.append(r"\toprule")
    lines.append(
        r"& & \multicolumn{" + str(len(K)) + r"}{c}{Direct match top-$K$ (\%)}"
        r" & \multicolumn{" + str(len(K)) + r"}{c}{Haiku judge top-$K$ (\%)}"
        r" & \multicolumn{" + str(len(K)) + r"}{c}{Sonnet judge top-$K$ (\%)} \\"
    )
    lines.append(
        r"\cmidrule(lr){3-" + str(2 + len(K)) + r"}"
        r" \cmidrule(lr){" + str(3 + len(K)) + r"-" + str(2 + 2 * len(K)) + r"}"
        r" \cmidrule(lr){" + str(3 + 2 * len(K)) + r"-" + str(2 + 3 * len(K)) + r"}"
    )
    header_cells = ["Group", "$n$"] + [f"$K{{=}}{k}$" for k in K] * 3
    lines.append(" & ".join(header_cells) + r" \\")
    lines.append(r"\midrule")
    for r in rows:
        if r.get("is_subheader"):
            lines.append(r["group_label"] + r" \\")
            continue
        if r.get("is_pooled"):
            lines.append(r"\addlinespace[2pt]")
        cells = [r["group_label"]]
        cells.append(f"{r['n']:,}" if r["n"] != "" else "")
        for stem in ("sup_hits", "haiku_hits", "sonnet_hits"):
            for k in K:
                hits = r[stem].get(k)
                cells.append(fmt_pct(hits, r["n"]) if hits is not None and r["n"] else "--")
        lines.append(" & ".join(cells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUT.mkdir(parents=True, exist_ok=True)

    rows = build_1fact()
    (OUT / "table_1fact.tex").write_text(render_table(
        rows,
        caption=("Decoding accuracy on 1-fact addition (DeepSeek V3, neutral judge "
                 "prompt). Each cell is the top-$K$ hit rate as a percentage of $n$. "
                 "Direct match: target value $A$ in top-$K$ numeric tokens of the "
                 "aggregated residuals. Haiku/Sonnet: judge's primary guess plus "
                 "first $K{-}1$ backups contains the target."),
        label="decode_1fact"))

    rows = build_2fact()
    (OUT / "table_2fact.tex").write_text(render_table(
        rows,
        caption=("Decoding accuracy on 2-fact addition (neutral judge prompt). "
                 "Direct match: BOTH addends $A_1$ and $A_2$ in top-$K$ numeric "
                 "tokens. Haiku/Sonnet: $\\{n_1, n_2, \\textrm{backups}\\}$ "
                 "contains both $A_1$ and $A_2$ within top-$K$."),
        label="decode_2fact"))

    rows = build_letterpos("neutral")
    (OUT / "table_letterpos_neutral.tex").write_text(render_table(
        rows,
        caption=("Decoding accuracy on letter-position task with the "
                 "\\emph{neutral} (task-agnostic) judge prompt. Elements: "
                 "atomic-number-to-element entity. Capitals: country/state-to-"
                 "capital entity. Direct match: substring of the entity name "
                 "(EN or ZH alias, or chemical symbol for elements) appears in "
                 "any top-$K$ token."),
        label="decode_letterpos_neutral"))

    rows = build_letterpos("leading")
    (OUT / "table_letterpos_leading.tex").write_text(render_table(
        rows,
        caption=("Decoding accuracy on letter-position task with the leading "
                 "judge prompts (\\emph{chemistry} for elements, "
                 "\\emph{geography} for capitals). Direct-match column is "
                 "identical to Table~\\ref{tab:decode_letterpos_neutral} "
                 "since the residual tokens are the same; only the judge "
                 "framing differs."),
        label="decode_letterpos_leading"))

    print(f"Wrote 4 tables to {OUT}/")
    for f in sorted(OUT.glob("*.tex")):
        n_lines = len(f.read_text().splitlines())
        print(f"  {f.name}: {n_lines} lines")


if __name__ == "__main__":
    main()
