"""Tests for extract_hidden_states.py and generate_1hop_dataset.py.

Verifies:
    1. Filler token generation (dots, counting, counting-scrambled)
    2. Prompt construction (system messages, user turns, full message lists)
    3. Tokenizer loading (PreTrainedTokenizerFast vs broken LlamaTokenizer)
    4. Filler boundary detection for all filler types
    5. Extraction position computation

Requires:
    - Model tokenizer files at /workspace/models/deepseek-v3-awq/
    - Dataset at data/1hop_addition_dataset.json

Usage:
    source /workspace/config/probing_env.sh
    cd scripts
    uv run --project /workspace/filler-token-reasoning/probing python -m pytest test_extraction.py -v
"""

import json
import random
from pathlib import Path

import pytest
import torch

from generate_1hop_dataset import (
    build_system_message,
    build_user_turn,
    filler_description,
    make_filler_tokens,
)
from extract_hidden_states import (
    CONDITIONS,
    FILLER_FRACTIONS,
    build_messages_for_condition,
    compute_baseline_positions,
    compute_extraction_positions,
    find_filler_boundaries,
    load_tokenizer,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MODEL_PATH = "/workspace/models/deepseek-v3-awq"
DATASET_PATH = Path(__file__).resolve().parent.parent / "data" / "1hop_addition_dataset.json"

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tokenizer():
    """Load the correct fast tokenizer once for all tests."""
    return load_tokenizer(MODEL_PATH)


@pytest.fixture(scope="module")
def dataset():
    with open(DATASET_PATH) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def few_shot(dataset):
    return dataset["few_shot_facts"][:5]


@pytest.fixture(scope="module")
def target(dataset):
    return dataset["examples"][0]


# ===========================================================================
# 1. Filler token generation
# ===========================================================================

class TestMakeFillerTokens:
    def test_dots_returns_spaced_dots(self):
        result = make_filler_tokens("dots", 5)
        assert result == ". . . . ."

    def test_dots_k0_returns_empty(self):
        assert make_filler_tokens("dots", 0) == ""

    def test_counting_returns_sequential(self):
        result = make_filler_tokens("counting", 10)
        assert result == "1 2 3 4 5 6 7 8 9 10"

    def test_counting_k1(self):
        assert make_filler_tokens("counting", 1) == "1"

    def test_counting_scrambled_has_all_numbers(self):
        rng = random.Random(42)
        result = make_filler_tokens("counting-scrambled", 20, rng=rng)
        nums = [int(x) for x in result.split()]
        assert sorted(nums) == list(range(1, 21))

    def test_counting_scrambled_is_not_sequential(self):
        """Scrambled should differ from sequential (with overwhelming probability)."""
        rng = random.Random(42)
        result = make_filler_tokens("counting-scrambled", 50, rng=rng)
        sequential = make_filler_tokens("counting", 50)
        assert result != sequential

    def test_counting_scrambled_deterministic(self):
        """Same seed produces same shuffle."""
        r1 = make_filler_tokens("counting-scrambled", 100, rng=random.Random(7))
        r2 = make_filler_tokens("counting-scrambled", 100, rng=random.Random(7))
        assert r1 == r2

    def test_unknown_filler_type_raises(self):
        with pytest.raises(ValueError, match="Unknown filler type"):
            make_filler_tokens("unknown", 5)


class TestFillerDescription:
    def test_dots(self):
        assert filler_description("dots") == "dots"

    def test_counting(self):
        assert filler_description("counting") == "sequential integers"

    def test_scrambled(self):
        assert filler_description("counting-scrambled") == "shuffled integers"


# ===========================================================================
# 2. Prompt construction
# ===========================================================================

class TestBuildSystemMessage:
    def test_baseline_no_filler_mention(self):
        msg = build_system_message("dots", 0)
        assert "filler" not in msg.lower()
        assert "Answer: [ANSWER]" in msg

    def test_dots_mentions_dots(self):
        msg = build_system_message("dots", 250)
        assert "250 filler tokens" in msg
        assert "dots" in msg

    def test_counting_mentions_sequential(self):
        msg = build_system_message("counting", 150)
        assert "150 filler tokens" in msg
        assert "sequential integers" in msg

    def test_scrambled_mentions_shuffled(self):
        msg = build_system_message("counting-scrambled", 150)
        assert "shuffled integers" in msg


class TestBuildUserTurn:
    def test_baseline_has_no_filler_section(self):
        turn = build_user_turn("the atomic number of oxygen", 25, "dots", 0)
        assert "Question: What is the atomic number of oxygen plus 25?" in turn
        assert "Filler:" not in turn
        assert turn.endswith("Answer:")

    def test_dots_has_filler_section(self):
        turn = build_user_turn("the atomic number of oxygen", 25, "dots", 5)
        assert "Filler: . . . . ." in turn
        assert turn.endswith("Answer:")

    def test_counting_has_numbers(self):
        turn = build_user_turn("the atomic number of oxygen", 25, "counting", 5)
        assert "Filler: 1 2 3 4 5" in turn

    def test_scrambled_has_shuffled_numbers(self):
        rng = random.Random(42)
        turn = build_user_turn("the atomic number of oxygen", 25, "counting-scrambled", 10, rng=rng)
        assert "Filler:" in turn
        # All numbers 1-10 should be present
        filler_part = turn.split("Filler: ")[1].split("\n")[0]
        nums = [int(x) for x in filler_part.split()]
        assert sorted(nums) == list(range(1, 11))


class TestBuildMessagesForCondition:
    def test_baseline_message_count(self, few_shot, target):
        msgs = build_messages_for_condition(few_shot, target, "dots", 0)
        # 1 system + 5*(user+assistant) + 1 target user = 12
        assert len(msgs) == 12

    def test_system_message_is_first(self, few_shot, target):
        msgs = build_messages_for_condition(few_shot, target, "dots", 250)
        assert msgs[0]["role"] == "system"
        assert "250 filler tokens" in msgs[0]["content"]

    def test_assistant_turns_have_answer_prefix(self, few_shot, target):
        msgs = build_messages_for_condition(few_shot, target, "counting", 150)
        for msg in msgs:
            if msg["role"] == "assistant":
                assert msg["content"].startswith("Answer: ")

    def test_last_message_is_user(self, few_shot, target):
        msgs = build_messages_for_condition(few_shot, target, "dots", 250)
        assert msgs[-1]["role"] == "user"

    def test_counting_filler_in_user_turns(self, few_shot, target):
        rng = random.Random(0)
        msgs = build_messages_for_condition(few_shot, target, "counting", 10, rng=rng)
        user_msgs = [m for m in msgs if m["role"] == "user"]
        for um in user_msgs:
            assert "Filler: " in um["content"]
            # Each filler should contain numbers 1-10
            filler_part = um["content"].split("Filler: ")[1].split("\n")[0]
            nums = sorted(int(x) for x in filler_part.split())
            assert nums == list(range(1, 11))


# ===========================================================================
# 3. Tokenizer loading
# ===========================================================================

class TestLoadTokenizer:
    def test_returns_fast_tokenizer(self, tokenizer):
        # Should NOT be LlamaTokenizer (sentencepiece)
        cls_name = type(tokenizer).__name__
        assert "Llama" not in cls_name or "Fast" in cls_name

    def test_dots_tokenize_individually(self, tokenizer):
        dots = " ".join(["."] * 20)
        ids = tokenizer(dots, add_special_tokens=False)["input_ids"]
        # Each ". " should be ~1 token, so 20 dots should give ~20 tokens
        assert len(ids) >= 19, f"Expected ~20 tokens for 20 dots, got {len(ids)}"

    def test_counting_numbers_tokenize_individually(self, tokenizer):
        text = " ".join(str(i) for i in range(1, 21))
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        # 20 numbers with spaces → each number should be ≥1 token
        # At minimum 20 tokens (numbers) + some space tokens
        assert len(ids) >= 20, f"Expected ≥20 tokens for counting 1-20, got {len(ids)}"

    def test_chat_template_works(self, tokenizer):
        msgs = [{"role": "user", "content": "Hello"}]
        result = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        assert "Hello" in result
        assert "Assistant" in result

    def test_dots250_matches_existing_extraction(self, tokenizer, few_shot, target):
        """The tokenizer should produce the same seq_len as the existing dots_250 extraction."""
        rng = random.Random(0)
        msgs = build_messages_for_condition(few_shot, target, "dots", 250, rng=rng)
        full = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = tokenizer(full)["input_ids"]
        # Existing extraction recorded seq_len=1732 for prob_0000
        assert len(ids) == 1732, f"Expected 1732 tokens (matching existing extraction), got {len(ids)}"


# ===========================================================================
# 4. Filler boundary detection
# ===========================================================================

class TestFindFillerBoundaries:
    def test_baseline_returns_neg1(self, tokenizer):
        ids = torch.tensor([[1, 2, 3]])
        start, end = find_filler_boundaries(tokenizer, ids, k=0)
        assert start == -1
        assert end == -1

    def _build_and_tokenize(self, tokenizer, few_shot, target, filler_type, k):
        rng = random.Random(0)
        msgs = build_messages_for_condition(few_shot, target, filler_type, k, rng=rng)
        full = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(full, return_tensors="pt")
        return inputs["input_ids"]

    def test_dots250_filler_len(self, tokenizer, few_shot, target):
        ids = self._build_and_tokenize(tokenizer, few_shot, target, "dots", 250)
        start, end = find_filler_boundaries(tokenizer, ids, 250)
        filler_len = end - start + 1
        assert filler_len == 250, f"Expected filler_len=250 for dots_250, got {filler_len}"

    def test_dots250_matches_existing(self, tokenizer, few_shot, target):
        """Boundary positions should match what the existing extraction recorded."""
        ids = self._build_and_tokenize(tokenizer, few_shot, target, "dots", 250)
        start, end = find_filler_boundaries(tokenizer, ids, 250)
        assert start == 1479, f"Expected filler_start=1479, got {start}"
        assert end == 1728, f"Expected filler_end=1728, got {end}"

    def test_counting150_filler_has_correct_length(self, tokenizer, few_shot, target):
        ids = self._build_and_tokenize(tokenizer, few_shot, target, "counting", 150)
        start, end = find_filler_boundaries(tokenizer, ids, 150)
        filler_len = end - start + 1
        # 150 numbers + 149 inter-number spaces + 1 leading space = 300
        assert filler_len == 300, f"Expected filler_len=300 for counting_150, got {filler_len}"

    def test_scrambled150_filler_same_length_as_counting(self, tokenizer, few_shot, target):
        ids_c = self._build_and_tokenize(tokenizer, few_shot, target, "counting", 150)
        ids_s = self._build_and_tokenize(tokenizer, few_shot, target, "counting-scrambled", 150)
        start_c, end_c = find_filler_boundaries(tokenizer, ids_c, 150)
        start_s, end_s = find_filler_boundaries(tokenizer, ids_s, 150)
        assert (end_c - start_c) == (end_s - start_s)

    def test_filler_start_after_colon(self, tokenizer, few_shot, target):
        """filler_start - 1 should be the ':' of 'Filler:'."""
        ids = self._build_and_tokenize(tokenizer, few_shot, target, "counting", 150)
        start, end = find_filler_boundaries(tokenizer, ids, 150)
        colon_tok = tokenizer.decode([ids[0][start - 1].item()])
        assert colon_tok.strip() == ":", f"Expected ':' before filler_start, got {colon_tok!r}"

    def test_filler_end_before_newline_or_answer(self, tokenizer, few_shot, target):
        """Token after filler_end should be newline or 'Answer'."""
        ids = self._build_and_tokenize(tokenizer, few_shot, target, "counting", 150)
        start, end = find_filler_boundaries(tokenizer, ids, 150)
        next_tok = tokenizer.decode([ids[0][end + 1].item()])
        assert next_tok.strip() in ("", "Answer"), f"Expected newline or 'Answer' after filler_end, got {next_tok!r}"

    def test_filler_content_is_numbers_for_counting(self, tokenizer, few_shot, target):
        """Decoded filler region should contain all numbers 1-150."""
        ids = self._build_and_tokenize(tokenizer, few_shot, target, "counting", 150)
        start, end = find_filler_boundaries(tokenizer, ids, 150)
        filler_text = tokenizer.decode(ids[0][start:end + 1].tolist())
        nums = [int(x) for x in filler_text.split()]
        assert sorted(nums) == list(range(1, 151))

    def test_filler_content_is_dots_for_dots(self, tokenizer, few_shot, target):
        """Decoded filler region should be dot tokens."""
        ids = self._build_and_tokenize(tokenizer, few_shot, target, "dots", 250)
        start, end = find_filler_boundaries(tokenizer, ids, 250)
        filler_text = tokenizer.decode(ids[0][start:end + 1].tolist()).strip()
        # Should be all dots and spaces/newlines
        non_dot = filler_text.replace(".", "").replace(" ", "").replace("\n", "")
        assert non_dot == "", f"Expected only dots/spaces, found extra: {non_dot!r}"

    def test_multiple_examples_consistent(self, tokenizer, dataset):
        """Boundary detection should work across different examples."""
        few_shot = dataset["few_shot_facts"][:5]
        for i in [0, 10, 50, 100]:
            target = dataset["examples"][i]
            rng = random.Random(i)
            msgs = build_messages_for_condition(few_shot, target, "counting", 150, rng=rng)
            full = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(full, return_tensors="pt")
            ids = inputs["input_ids"]
            start, end = find_filler_boundaries(tokenizer, ids, 150)
            filler_len = end - start + 1
            assert filler_len == 300, f"Example {i}: expected filler_len=300, got {filler_len}"


# ===========================================================================
# 5. Position computation
# ===========================================================================

class TestComputeExtractionPositions:
    def test_position_count(self):
        positions = compute_extraction_positions(100, 349, 360)
        # question_end + 7 filler fractions + answer_prompt = 9
        assert len(positions) == 9

    def test_question_end_before_filler(self):
        positions = compute_extraction_positions(100, 349, 360)
        assert positions["question_end"] == 99

    def test_answer_prompt_is_last_token(self):
        positions = compute_extraction_positions(100, 349, 360)
        assert positions["answer_prompt"] == 359

    def test_filler_000_is_filler_start(self):
        positions = compute_extraction_positions(100, 349, 360)
        assert positions["filler_0.00"] == 100

    def test_filler_100_is_filler_end(self):
        positions = compute_extraction_positions(100, 349, 360)
        assert positions["filler_1.00"] == 349

    def test_filler_050_is_midpoint(self):
        positions = compute_extraction_positions(100, 349, 360)
        expected = 100 + int(0.5 * 249)  # 224
        assert positions["filler_0.50"] == expected

    def test_all_positions_in_bounds(self):
        positions = compute_extraction_positions(100, 349, 360)
        for name, idx in positions.items():
            assert 0 <= idx < 360, f"{name}={idx} out of bounds for seq_len=360"

    def test_positions_are_monotonically_nondecreasing(self):
        positions = compute_extraction_positions(100, 349, 360)
        filler_keys = [k for k in sorted(positions.keys()) if k.startswith("filler_")]
        filler_vals = [positions[k] for k in filler_keys]
        for i in range(1, len(filler_vals)):
            assert filler_vals[i] >= filler_vals[i - 1]


class TestComputeBaselinePositions:
    def test_has_two_positions(self):
        positions = compute_baseline_positions(100)
        assert len(positions) == 2

    def test_question_end(self):
        positions = compute_baseline_positions(100)
        assert positions["question_end"] == 98

    def test_answer_prompt(self):
        positions = compute_baseline_positions(100)
        assert positions["answer_prompt"] == 99


# ===========================================================================
# 6. CONDITIONS config
# ===========================================================================

class TestConditionsConfig:
    def test_all_conditions_are_tuples(self):
        for name, val in CONDITIONS.items():
            assert isinstance(val, tuple) and len(val) == 2, f"{name}: expected (k, filler_type) tuple"

    def test_baseline_k_is_zero(self):
        assert CONDITIONS["baseline"][0] == 0

    def test_counting_conditions_exist(self):
        assert "counting_150" in CONDITIONS
        assert "counting_scrambled_150" in CONDITIONS

    def test_counting_conditions_have_correct_k(self):
        assert CONDITIONS["counting_150"] == (150, "counting")
        assert CONDITIONS["counting_scrambled_150"] == (150, "counting-scrambled")
