"""
parse_outputs.py

Reads the raw CSV from run_experiment_grid.py. For each generated completion:
  - Extracts all integers (word-boundary matched)
  - Extracts element names (mapped to atomic numbers)
  - Checks which ground truth quantities appear: A1, A2, sum, A1-A2, A2-A1, etc.
  - Records the first number generated

Outputs a scored CSV with additional columns.

Usage:
    python parse_outputs.py
    python parse_outputs.py --input results/patchscope_grid.csv --output results/patchscope_scored.csv
"""

import argparse
import csv
import re
import os
import sys

# ============================================================
# ELEMENT LOOKUP
# ============================================================

ELEMENT_TO_Z = {
    "hydrogen": 1, "helium": 2, "lithium": 3, "beryllium": 4, "boron": 5,
    "carbon": 6, "nitrogen": 7, "oxygen": 8, "fluorine": 9, "neon": 10,
    "sodium": 11, "magnesium": 12, "aluminum": 13, "aluminium": 13,
    "silicon": 14, "phosphorus": 15, "sulfur": 16, "sulphur": 16,
    "chlorine": 17, "argon": 18, "potassium": 19, "calcium": 20,
    "scandium": 21, "titanium": 22, "vanadium": 23, "chromium": 24,
    "manganese": 25, "iron": 26, "cobalt": 27, "nickel": 28,
    "copper": 29, "zinc": 30, "gallium": 31, "germanium": 32,
    "arsenic": 33, "selenium": 34, "bromine": 35, "krypton": 36,
    "rubidium": 37, "strontium": 38, "yttrium": 39, "zirconium": 40,
    "niobium": 41, "molybdenum": 42, "technetium": 43, "ruthenium": 44,
    "rhodium": 45, "palladium": 46, "silver": 47, "cadmium": 48,
    "indium": 49, "tin": 50, "antimony": 51, "tellurium": 52,
    "iodine": 53, "xenon": 54, "cesium": 55, "caesium": 55,
    "barium": 56, "lanthanum": 57, "cerium": 58, "praseodymium": 59,
    "neodymium": 60, "promethium": 61, "samarium": 62, "europium": 63,
    "gadolinium": 64, "terbium": 65, "dysprosium": 66, "holmium": 67,
    "erbium": 68, "thulium": 69, "ytterbium": 70, "lutetium": 71,
    "hafnium": 72, "tantalum": 73, "tungsten": 74, "rhenium": 75,
    "osmium": 76, "iridium": 77, "platinum": 78, "gold": 79,
    "mercury": 80, "thallium": 81, "lead": 82, "bismuth": 83,
    "polonium": 84, "astatine": 85, "radon": 86, "francium": 87,
    "radium": 88, "actinium": 89, "thorium": 90, "protactinium": 91,
    "uranium": 92, "neptunium": 93, "plutonium": 94, "americium": 95,
    "curium": 96, "berkelium": 97, "californium": 98, "einsteinium": 99,
}

Z_TO_ELEMENT = {v: k for k, v in ELEMENT_TO_Z.items() if k != "aluminium" and k != "sulphur" and k != "caesium"}

# Build regex for element names (case insensitive, longest match first)
_element_names_sorted = sorted(ELEMENT_TO_Z.keys(), key=len, reverse=True)
ELEMENT_PATTERN = re.compile(
    r'\b(' + '|'.join(_element_names_sorted) + r')\b',
    re.IGNORECASE
)

# Number words for small integers
WORD_TO_NUM = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
    "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
    "eighty": 80, "ninety": 90, "hundred": 100,
}

WORD_NUM_PATTERN = re.compile(
    r'\b(' + '|'.join(WORD_TO_NUM.keys()) + r')\b',
    re.IGNORECASE
)


# ============================================================
# EXTRACTION
# ============================================================

def extract_integers(text):
    """Extract all integers from text using word boundaries.
    Returns list of (int_value, start_position) tuples, in order of appearance.
    """
    results = []
    # Match digit sequences on word boundaries
    for m in re.finditer(r'\b(\d+)\b', text):
        results.append((int(m.group(1)), m.start()))
    return results


def extract_number_words(text):
    """Extract spelled-out numbers. Returns list of (int_value, start_position)."""
    results = []
    for m in WORD_NUM_PATTERN.finditer(text):
        word = m.group(1).lower()
        results.append((WORD_TO_NUM[word], m.start()))
    return results


def extract_element_names(text):
    """Extract element names and return their atomic numbers.
    Returns list of (atomic_number, element_name, start_position).
    """
    results = []
    for m in ELEMENT_PATTERN.finditer(text):
        name = m.group(1).lower()
        z = ELEMENT_TO_Z[name]
        results.append((z, name, m.start()))
    return results


def score_row(row):
    """
    Score a single row. Adds new fields to the row dict.

    Checks for: A1, A2, sum, |A1-A2|, A1*A2, and element names
    corresponding to A1 or A2.
    """
    text = row.get("generated_text", "")
    a1 = int(row["A1"])
    a2 = int(row["A2"])
    s = int(row["sum"])

    # Derived quantities to check
    diff_pos = abs(a1 - a2)
    diff_a1_a2 = a1 - a2
    diff_a2_a1 = a2 - a1
    product = a1 * a2

    # Extract numbers
    digit_nums = extract_integers(text)
    word_nums = extract_number_words(text)
    all_nums = sorted(digit_nums + word_nums, key=lambda x: x[1])
    num_values = [n[0] for n in all_nums]

    # Extract element names
    elements = extract_element_names(text)
    element_zs = [e[0] for e in elements]

    # First number generated
    first_num = all_nums[0][0] if all_nums else None

    # --- Check matches ---
    # A number "matches" if it appears as a word-boundary integer in the text
    contains_a1 = a1 in num_values
    contains_a2 = a2 in num_values
    contains_sum = s in num_values

    # Also check if element names for A1/A2 appear
    # (A1 and A2 are atomic numbers, so check if any extracted element has that Z)
    element_matches_a1 = a1 in element_zs
    element_matches_a2 = a2 in element_zs

    # Check derived quantities
    contains_diff = diff_pos in num_values and diff_pos != 0
    contains_product = product in num_values and product not in (a1, a2, s)

    # Classify the first number
    if first_num is not None:
        if first_num == a1 and first_num != a2 and first_num != s:
            first_num_identity = "A1"
        elif first_num == a2 and first_num != a1 and first_num != s:
            first_num_identity = "A2"
        elif first_num == s and first_num != a1 and first_num != a2:
            first_num_identity = "SUM"
        elif first_num == a1 and first_num == a2:
            first_num_identity = "A1=A2"
        elif first_num == a1 and first_num == s:
            first_num_identity = "A1=SUM"
        elif first_num == a2 and first_num == s:
            first_num_identity = "A2=SUM"
        elif first_num == diff_pos:
            first_num_identity = "DIFF"
        elif first_num == product:
            first_num_identity = "PRODUCT"
        else:
            first_num_identity = "OTHER"
    else:
        first_num_identity = "NONE"

    # Collect all unrecognized numbers
    known_values = {a1, a2, s, diff_pos, diff_a1_a2, diff_a2_a1, product, 0, 1}
    unknown_nums = [n for n in num_values if n not in known_values]

    # --- Write scored fields ---
    row["first_number"] = first_num if first_num is not None else ""
    row["first_num_identity"] = first_num_identity
    row["contains_A1"] = int(contains_a1)
    row["contains_A2"] = int(contains_a2)
    row["contains_sum"] = int(contains_sum)
    row["element_matches_A1"] = int(element_matches_a1)
    row["element_matches_A2"] = int(element_matches_a2)
    row["contains_diff"] = int(contains_diff)
    row["contains_product"] = int(contains_product)
    row["all_numbers"] = ";".join(str(n) for n in num_values)
    row["all_elements"] = ";".join(e[1] for e in elements)
    row["unknown_numbers"] = ";".join(str(n) for n in unknown_nums)
    row["is_error"] = int(text.startswith("ERROR:"))
    row["is_empty"] = int(len(text.strip()) == 0)

    return row


SCORED_COLUMNS = [
    "example_idx", "problem_idx", "A1", "A2", "sum",
    "source_layer", "source_pos", "source_desc",
    "template", "target_layer",
    "generated_text",
    # Scored fields
    "first_number", "first_num_identity",
    "contains_A1", "contains_A2", "contains_sum",
    "element_matches_A1", "element_matches_A2",
    "contains_diff", "contains_product",
    "all_numbers", "all_elements", "unknown_numbers",
    "is_error", "is_empty",
]


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Parse and score patchscope outputs")
    parser.add_argument("--input", default="results/patchscope_grid.csv")
    parser.add_argument("--output", default="results/patchscope_scored.csv")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Input file not found: {args.input}")
        sys.exit(1)

    # Read input
    with open(args.input, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"Loaded {len(rows)} rows from {args.input}")

    # Score each row
    n_errors = 0
    n_empty = 0
    for row in rows:
        score_row(row)
        if row["is_error"]:
            n_errors += 1
        if row["is_empty"]:
            n_empty += 1

    print(f"Errors: {n_errors}, Empty: {n_empty}")

    # Quick summary
    has_any_num = sum(1 for r in rows if r["all_numbers"])
    has_a1 = sum(1 for r in rows if r["contains_A1"] == 1)
    has_a2 = sum(1 for r in rows if r["contains_A2"] == 1)
    has_sum = sum(1 for r in rows if r["contains_sum"] == 1)

    print(f"\nQuick summary:")
    print(f"  Rows with any number:  {has_any_num}/{len(rows)} ({100*has_any_num/len(rows):.1f}%)")
    print(f"  Contains A1:           {has_a1}/{len(rows)} ({100*has_a1/len(rows):.1f}%)")
    print(f"  Contains A2:           {has_a2}/{len(rows)} ({100*has_a2/len(rows):.1f}%)")
    print(f"  Contains sum:          {has_sum}/{len(rows)} ({100*has_sum/len(rows):.1f}%)")

    # First number identity distribution
    from collections import Counter
    identity_counts = Counter(r["first_num_identity"] for r in rows)
    print(f"\n  First number identity distribution:")
    for identity, count in identity_counts.most_common():
        print(f"    {identity:>10}: {count:>5} ({100*count/len(rows):.1f}%)")

    # Write output
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SCORED_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"\nScored CSV saved to {args.output}")


if __name__ == "__main__":
    main()