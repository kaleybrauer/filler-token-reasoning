"""
llm_decode_batch.py

Batch-decode aggregated-residual top-50 JSONs (from aggregate_residuals_all_settings.py)
across multiple conditions × LLMs with threaded Anthropic API calls.

Two prompt variants are saved below; --prompt selects which one (default: neutral).
The filler description (dots vs counting sequence) is picked per-condition from
the condition name, so counting_* runs get the right warning.

Output files: <outdir>/llm_decode_<cond>_<model>_<prompt>.json
Summary: <outdir>/llm_decode_summary_<prompt>.json

Usage:
    python scripts/decode/llm_decode_batch.py \
        --aggregated-dir outputs/deepseek_aggregated \
        --outdir outputs/deepseek_aggregated \
        --conditions dots_10 dots_25 dots_50 counting_10 counting_25 \
        --prompt neutral \
        --models haiku sonnet
"""

from __future__ import annotations

import argparse
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

# ----------------------------------------------------------------------------
# Prompts
# ----------------------------------------------------------------------------

PROMPT_NEUTRAL = (
    "These are potentially important tokens that were taken from a model's "
    "internal state during a task. The tokens in this list are hints about "
    "what specific numbers the model was 'thinking about' in its internal "
    "state while doing the calculation, when it has not yet gotten to "
    "outputting the final answer. The original prompt contained formatting "
    "words like 'Filler' and 'Answer' that are not relevant to the task. "
    "The model is Chinese, so it may use Chinese tokens as well as English.\n\n"
    "What specific two numbers was the model thinking about?\n\n"
    "Think briefly, then end your response with a JSON object on its own line: "
    "{\"n1\": <int>, \"confidence_n1\": <float 0-1>, "
    "\"n2\": <int>, \"confidence_n2\": <float 0-1>, "
    "\"backups\": [<int>, <int>, ... up to 10 ranked alternatives]}. "
    "If you cannot determine n1 or n2, output null for that field."
)

PROMPT_ADDITION = (
    "These are potentially important tokens that were taken from a model doing "
    "an addition task. The model was computing the sum of two operands, and "
    "the tokens in this list are hints about what the two operands were — the "
    "numbers the model was 'thinking about' in its internal state while doing "
    "the calculation, when it has not yet gotten to outputting the final sum. "
    "The original prompt contained formatting words like 'Filler' and 'Answer' "
    "that are not relevant to the task. The model is Chinese, so it may use "
    "Chinese tokens as well as English.\n\n"
    "What two numbers are being added together in the task?\n\n"
    "Think briefly, then end your response with a JSON object on its own line: "
    "{\"n1\": <int>, \"confidence_n1\": <float 0-1>, "
    "\"n2\": <int>, \"confidence_n2\": <float 0-1>, "
    "\"backups\": [<int>, <int>, ... up to 10 ranked alternatives]}. "
    "If you cannot determine n1 or n2, output null for that field."
)

PROMPTS = {"neutral": PROMPT_NEUTRAL, "addition": PROMPT_ADDITION}

MODEL_IDS = {
    "haiku":  "claude-haiku-4-5-20251001",
    "sonnet": "claude-sonnet-4-6",
}


def build_prompt(mode: str, cond: str) -> str:
    """Return the system prompt for a given mode and condition.

    Currently the prompt is condition-agnostic — the filler type (dots/counting)
    isn't surfaced because the residual already suppresses it and explicit
    mentions were found to make decoders overreact.
    """
    return PROMPTS[mode]


# ----------------------------------------------------------------------------
# JSON parsing + formatting
# ----------------------------------------------------------------------------

JSON_RE = re.compile(r'\{[^{}]*"n1"[^{}]*"backups"[^{}]*\[[^\]]*\][^{}]*\}', re.DOTALL)
JSON_ANY = re.compile(r'\{[^{}]*"n1"[^{}]*\}', re.DOTALL)


def parse(text: str):
    m = JSON_RE.findall(text) or JSON_ANY.findall(text)
    if not m:
        return None
    try:
        return json.loads(m[-1])
    except Exception:
        return None


def format_tokens(top):
    return "\n".join(
        f"{i:2d}. {t['prob']:.3f}  {t['str']!r}" for i, t in enumerate(top, 1)
    )


# ----------------------------------------------------------------------------
# Per-run driver
# ----------------------------------------------------------------------------

def run_one(client, label, model_id, prompt, samples, max_workers):
    n = len(samples)
    print(f"\n=== {label} (n={n}) ===", flush=True)
    results = [None] * n

    def worker(i):
        user = f"Top tokens (rank. score  'string'):\n{format_tokens(samples[i]['top_tokens'])}"
        try:
            resp = client.messages.create(
                model=model_id, max_tokens=1000,
                system=prompt,
                messages=[{"role": "user", "content": user}],
            )
            return i, resp.content[0].text, None
        except Exception as e:
            return i, None, str(e)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(worker, i) for i in range(n)]
        for f in tqdm(as_completed(futures), total=n, desc=label):
            i, text, err = f.result()
            results[i] = {"text": text, "error": err}

    scored = []
    for i, s in enumerate(samples):
        parsed = parse(results[i]["text"] or "") or {}
        n1, n2 = parsed.get("n1"), parsed.get("n2")
        backups = parsed.get("backups", []) or []
        A1 = s["fact_value_1"]
        A2 = s["fact_value_2"]
        primary = {n1, n2}
        top_k = {}
        for K in [2, 3, 5, 10]:
            cand = [n1, n2] + backups[:K - 2]
            top_k[K] = {"a1": A1 in cand, "a2": A2 in cand,
                        "both": (A1 in cand) and (A2 in cand)}
        scored.append({
            "idx": s.get("idx"), "A1": A1, "A2": A2,
            "pred": [n1, n2], "backups": backups,
            "got_a1": A1 in primary, "got_a2": A2 in primary,
            "both": (A1 in primary) and (A2 in primary),
            "top_k": top_k, "raw": results[i]["text"],
        })

    summary = {}
    for K in [2, 3, 5, 10]:
        a1 = sum(r["top_k"][K]["a1"] for r in scored)
        a2 = sum(r["top_k"][K]["a2"] for r in scored)
        b  = sum(r["top_k"][K]["both"] for r in scored)
        summary[K] = (a1, a2, b)
        print(f"  top-{K:>2}: A1={a1}/{n}, A2={a2}/{n}, Both={b}/{n}", flush=True)
    return scored, summary


# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--aggregated-dir", type=Path, required=True,
                    help="Directory containing aggregated_<cond>.json files")
    ap.add_argument("--outdir", type=Path, default=None,
                    help="Output dir for llm_decode_* files (default: same as --aggregated-dir)")
    ap.add_argument("--conditions", nargs="+", required=True,
                    help="Condition names; filler type (dots/counting) is inferred from the name")
    ap.add_argument("--prompt", choices=list(PROMPTS), default="neutral",
                    help="Prompt variant (default: neutral)")
    ap.add_argument("--models", nargs="+", default=["haiku", "sonnet"],
                    choices=list(MODEL_IDS))
    ap.add_argument("--threads", type=int, default=8,
                    help="ThreadPoolExecutor max_workers per run")
    ap.add_argument("--key-path", type=Path,
                    default=Path("/workspace/keys/anthropic_api_key"),
                    help="File with Anthropic API key (ignored if $ANTHROPIC_API_KEY already set)")
    ap.add_argument("--print-prompts", action="store_true",
                    help="Print the resolved prompt for each condition and exit")
    args = ap.parse_args()

    outdir = args.outdir or args.aggregated_dir
    outdir.mkdir(parents=True, exist_ok=True)

    if args.print_prompts:
        for cond in args.conditions:
            print(f"\n{'=' * 72}\n== {cond}  (prompt={args.prompt})\n{'=' * 72}")
            print(build_prompt(args.prompt, cond))
        return

    if "ANTHROPIC_API_KEY" not in os.environ:
        os.environ["ANTHROPIC_API_KEY"] = args.key_path.read_text().strip()
    import anthropic
    client = anthropic.Anthropic()

    all_summaries = {}
    for cond in args.conditions:
        agg_path = args.aggregated_dir / f"aggregated_{cond}.json"
        if not agg_path.exists():
            print(f"MISSING: {agg_path}; skipping {cond}")
            continue
        with open(agg_path) as f:
            samples = json.load(f)
        prompt = build_prompt(args.prompt, cond)
        print(f"\n{'#' * 72}\n# {cond}  (n={len(samples)}, prompt={args.prompt})"
              f"\n{'#' * 72}", flush=True)
        for mlabel in args.models:
            model_id = MODEL_IDS[mlabel]
            scored, summary = run_one(client, f"{cond}/{mlabel}", model_id,
                                      prompt, samples, args.threads)
            out_path = outdir / f"llm_decode_{cond}_{mlabel}_{args.prompt}.json"
            with open(out_path, "w") as f:
                json.dump(scored, f, indent=2, ensure_ascii=False)
            print(f"  Saved → {out_path}")
            all_summaries[(cond, mlabel)] = (len(samples), summary)

    print(f"\n\n=== SUMMARY ({args.prompt} prompt, primary + top-K Both) ===")
    print(f"{'Condition':<14} {'Model':<7} {'N':>4}  {'prim-Both':>10} {'top-3-Both':>10} "
          f"{'top-5-Both':>10} {'top-10-Both':>12}")
    for (cond, mlabel), (n, summary) in all_summaries.items():
        prim, t3, t5, t10 = summary[2][2], summary[3][2], summary[5][2], summary[10][2]
        print(f"  {cond:<12} {mlabel:<7} {n:>4}   {prim:>4}/{n}     "
              f"{t3:>4}/{n}     {t5:>4}/{n}     {t10:>4}/{n}")

    summary_path = outdir / f"llm_decode_summary_{args.prompt}.json"
    with open(summary_path, "w") as f:
        json.dump({f"{c}_{m}": {"n": n, "summary": s}
                   for (c, m), (n, s) in all_summaries.items()}, f, indent=2)
    print(f"\nSaved summary → {summary_path}")


if __name__ == "__main__":
    main()
