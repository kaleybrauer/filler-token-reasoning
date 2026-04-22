"""
llm_decode_pool.py

Send pooled top-K tokens (from pool_top_tokens.py) to Claude Haiku and ask it
to identify the two operands being added. Two modes:

    --mode fullvocab    Input contains arbitrary tokens (may include element
                        names, Chinese numerals, etc.). Prompt is minimal.

    --mode numeric      Input contains only numeric tokens. Prompt clarifies
                        the top-probability tokens ARE operands (not sums),
                        and requests confidence + 10 ranked backup candidates.

Expects $ANTHROPIC_API_KEY or a key path via --key-path.

Usage:
    python scripts/dev/llm_decode_pool.py \
        --input /tmp/pooled_top50.json \
        --mode fullvocab \
        --output /tmp/llm_decode_results.json
"""

import argparse
import json
import os
import re
from pathlib import Path


SYSTEM_FULLVOCAB = (
    "These are potentially important tokens that were taken from a model doing "
    "an addition task. The model was computing the sum of two operands, and "
    "the tokens in this list are hints about what the two operands were — the "
    "numbers the model was 'thinking about' in its internal state while doing "
    "the calculation, when it has not yet gotten to outputting the final sum. "
    "The original prompt contained formatting words like 'Filler' and 'Answer' "
    "that are not relevant to the task. There are also dots serving as filler "
    "tokens in the original prompt that are not relevant to the task. The model "
    "is Chinese, so it may use Chinese tokens as well as English.\n\n"
    "What two numbers are being added together in the task?\n\n"
    "Think briefly, then end your response with a JSON object on its own line: "
    "{\"n1\": <int>, \"confidence_n1\": <float 0-1>, "
    "\"n2\": <int>, \"confidence_n2\": <float 0-1>, "
    "\"backups\": [<int>, <int>, ... up to 10 ranked alternatives]}. "
    "If you cannot determine n1 or n2, output null for that field."
)

SYSTEM_NUMERIC = (
    "These are potentially important numeric tokens that were taken from a "
    "model doing an addition task. The model was computing the sum of two "
    "operands, and the numbers in this list are hints about what the two "
    "operands were — the numbers the model was 'thinking about' in its "
    "internal state while doing the calculation, when it has not yet gotten "
    "to outputting the final sum. The original prompt contained formatting "
    "words like 'Filler' and 'Answer' that are not relevant to the task. "
    "There are also dots serving as filler tokens in the original prompt "
    "that are not relevant to the task.\n\n"
    "What two numbers are being added together in the task?\n\n"
    "Think briefly, then end your response with a JSON object on its own line: "
    "{\"n1\": <int>, \"confidence_n1\": <float 0-1>, "
    "\"n2\": <int>, \"confidence_n2\": <float 0-1>, "
    "\"backups\": [<int>, <int>, ... up to 10 ranked alternatives]}. "
    "If you cannot determine n1 or n2, output null for that field."
)


def format_fullvocab(top):
    return "\n".join(
        f"{i:2d}. {t['prob']:.4f}  {t['str']!r}" for i, t in enumerate(top, 1)
    )


def format_numeric(top):
    return "\n".join(
        f"{i:2d}. {t['prob']:.4f}  {t['value']}" for i, t in enumerate(top, 1)
    )


JSON_RE_NUM = re.compile(
    r'\{[^{}]*"n1"[^{}]*"backups"[^{}]*\[[^\]]*\][^{}]*\}', re.DOTALL
)
JSON_RE_ANY = re.compile(r'\{[^{}]*"n1"[^{}]*\}', re.DOTALL)


def parse_numeric(text):
    m = JSON_RE_NUM.findall(text) or JSON_RE_ANY.findall(text)
    if not m:
        return None
    try:
        return json.loads(m[-1])
    except Exception:
        return None


def parse_fullvocab(text):
    # New schema includes "backups"; fall back to the minimal n1/n2 form.
    m = JSON_RE_NUM.findall(text) or JSON_RE_ANY.findall(text)
    if not m:
        return None
    try:
        return json.loads(m[-1])
    except Exception:
        return None


def ground_truth(sample):
    """Return (A1, A2) tuple — for 2-fact samples, else (A, None) for 1-hop."""
    if "fact_value_1" in sample and "fact_value_2" in sample:
        return sample["fact_value_1"], sample["fact_value_2"]
    if "A1" in sample and "A2" in sample:
        return sample["A1"], sample["A2"]
    if "fact_value" in sample:
        return sample["fact_value"], None
    if "A" in sample:
        return sample["A"], None
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True,
                    help="JSON from pool_top_tokens.py")
    ap.add_argument("--mode", choices=["fullvocab", "numeric"], required=True)
    ap.add_argument("--model", default="claude-haiku-4-5-20251001")
    ap.add_argument("--max-tokens", type=int, default=1000)
    ap.add_argument("--key-path", type=Path, default=None,
                    help="File containing the Anthropic API key (otherwise uses env var)")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--print-prompt", action="store_true")
    args = ap.parse_args()

    if args.key_path:
        os.environ["ANTHROPIC_API_KEY"] = args.key_path.read_text().strip()
    import anthropic
    client = anthropic.Anthropic()

    if args.mode == "numeric":
        SYSTEM = SYSTEM_NUMERIC
        formatter = format_numeric
        token_key = "top_numeric"
        parser = parse_numeric
    else:
        SYSTEM = SYSTEM_FULLVOCAB
        formatter = format_fullvocab
        token_key = "top_tokens"
        parser = parse_fullvocab

    if args.print_prompt:
        print("=" * 70)
        print("SYSTEM PROMPT")
        print("=" * 70)
        print(SYSTEM)
        print("=" * 70)

    with open(args.input) as f:
        samples = json.load(f)

    results = []
    for s in samples:
        user = f"Top tokens (rank. prob  value):\n{formatter(s[token_key])}"
        resp = client.messages.create(
            model=args.model, max_tokens=args.max_tokens,
            system=SYSTEM, messages=[{"role": "user", "content": user}],
        )
        text = resp.content[0].text
        parsed = parser(text) or {}
        n1 = parsed.get("n1")
        n2 = parsed.get("n2")
        # Two schemas exist: numeric has single "confidence"; fullvocab has
        # "confidence_n1" / "confidence_n2".
        conf = parsed.get("confidence")
        conf_n1 = parsed.get("confidence_n1")
        conf_n2 = parsed.get("confidence_n2")
        backups = parsed.get("backups", []) or []

        A1, A2 = ground_truth(s)
        primary = {n1, n2}
        got_a1 = A1 in primary
        got_a2 = (A2 is None) or (A2 in primary)
        both = got_a1 and got_a2

        top_k_stats = {}
        for K in [2, 5, 10]:
            cand = [n1, n2] + (backups[:K - 2] if backups else [])
            top_k_stats[f"top_{K}"] = {
                "a1": A1 in cand,
                "a2": (A2 is None) or (A2 in cand),
                "both": (A1 in cand) and ((A2 is None) or (A2 in cand)),
            }

        results.append({
            "idx": s.get("idx"),
            "A1": A1, "A2": A2,
            "pred": [n1, n2],
            "conf": conf, "conf_n1": conf_n1, "conf_n2": conf_n2,
            "backups": backups,
            "got_a1": got_a1, "got_a2": got_a2, "both": both,
            "top_k": top_k_stats, "raw": text,
        })
        mark = "✓" if both else "✗"
        conf_str = (f"conf={conf}" if conf is not None
                    else f"conf=({conf_n1},{conf_n2})")
        print(f"{mark} truth=({A1},{A2})  pred=({n1},{n2}) {conf_str}")

    n = len(results)
    print(f"\n=== Primary (n={n}) ===")
    print(f"A1: {sum(r['got_a1'] for r in results)}/{n}   "
          f"A2: {sum(r['got_a2'] for r in results)}/{n}   "
          f"Both: {sum(r['both'] for r in results)}/{n}")
    for K in [2, 5, 10]:
        a1 = sum(r["top_k"][f"top_{K}"]["a1"] for r in results)
        a2 = sum(r["top_k"][f"top_{K}"]["a2"] for r in results)
        b = sum(r["top_k"][f"top_{K}"]["both"] for r in results)
        print(f"Top-{K}: A1={a1}/{n}, A2={a2}/{n}, Both={b}/{n}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
