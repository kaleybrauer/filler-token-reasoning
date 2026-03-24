"""
Unit tests for filler_patching.py logic functions.
Run without GPU: uv run --project /workspace/filler-token-reasoning/probing \
    python -m pytest probing/scripts/test_filler_patching.py -v
"""

import sys
import json
import tempfile
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import pytest

# Add scripts dir to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from filler_patching import (
    START_CONDITIONS,
    compute_patch_mapping,
    compute_metrics,
    parse_answer,
    select_pairs,
    ContiguousInjector,
    build_dense_positions,
)


# ==============================================================================
# parse_answer tests
# ==============================================================================

class TestParseAnswer:
    def test_answer_prefix(self):
        assert parse_answer("Answer: 42") == 42

    def test_answer_prefix_no_space(self):
        assert parse_answer("Answer:42") == 42

    def test_bare_number(self):
        assert parse_answer("42") == 42

    def test_negative(self):
        assert parse_answer("Answer: -17") == -17

    def test_with_extra_text(self):
        assert parse_answer("The answer is Answer: 150 done") == 150

    def test_no_number(self):
        assert parse_answer("I don't know") is None

    def test_empty_string(self):
        assert parse_answer("") is None

    def test_number_in_noise(self):
        assert parse_answer("blah 99 blah") == 99


# ==============================================================================
# compute_metrics tests
# ==============================================================================

class TestComputeMetrics:
    def test_exact_donor_match(self):
        # Target: A=10, Source: A=50, Y=5 → target_ans=15, source_ans=55
        m = compute_metrics(
            patched_answer=55, natural_answer=15,
            source_answer=55, target_answer=15, delta_a=40
        )
        assert m["shift"] == 40
        assert m["shift_ratio"] == 1.0
        assert m["exact_match_donor"] is True
        assert m["exact_match_original"] is False
        assert m["novel"] is False
        assert m["parse_failure"] is False

    def test_no_shift(self):
        m = compute_metrics(
            patched_answer=15, natural_answer=15,
            source_answer=55, target_answer=15, delta_a=40
        )
        assert m["shift"] == 0
        assert m["shift_ratio"] == 0.0
        assert m["exact_match_original"] is True
        assert m["exact_match_donor"] is False

    def test_partial_shift(self):
        m = compute_metrics(
            patched_answer=35, natural_answer=15,
            source_answer=55, target_answer=15, delta_a=40
        )
        assert m["shift"] == 20
        assert m["shift_ratio"] == 0.5
        assert m["novel"] is True

    def test_negative_delta(self):
        # Source A < Target A
        m = compute_metrics(
            patched_answer=5, natural_answer=55,
            source_answer=15, target_answer=55, delta_a=-40
        )
        assert m["shift"] == -50
        assert m["shift_ratio"] == pytest.approx(1.25)

    def test_parse_failure(self):
        m = compute_metrics(
            patched_answer=None, natural_answer=15,
            source_answer=55, target_answer=15, delta_a=40
        )
        assert m["parse_failure"] is True
        assert m["shift"] is None

    def test_natural_parse_failure(self):
        m = compute_metrics(
            patched_answer=55, natural_answer=None,
            source_answer=55, target_answer=15, delta_a=40
        )
        assert m["parse_failure"] is False
        assert m["shift"] is None  # can't compute shift
        assert m["exact_match_donor"] is True

    def test_zero_delta(self):
        m = compute_metrics(
            patched_answer=15, natural_answer=15,
            source_answer=15, target_answer=15, delta_a=0
        )
        assert m["shift_ratio"] is None  # division by zero guarded


# ==============================================================================
# compute_patch_mapping tests
# ==============================================================================

def make_source_info(question_end=100, filler_start=105, filler_end=154, seq_len=160):
    """Create a mock source_info dict."""
    positions = list(range(question_end, seq_len))
    return {
        "positions": positions,
        "question_end_pos": question_end,
        "filler_start": filler_start,
        "filler_end": filler_end,
        "seq_len": seq_len,
    }


class TestComputePatchMapping:
    def test_answer_prompt_only(self):
        src = make_source_info()
        tgt_pos, src_idx = compute_patch_mapping(
            "answer_prompt", src,
            target_question_end=200, target_filler_start=205,
            target_seq_len=260,
        )
        assert len(tgt_pos) == 1
        assert tgt_pos[0] == 259  # target_seq_len - 1
        assert len(src_idx) == 1

    def test_pre_answer_returns_none(self):
        src = make_source_info()
        tgt_pos, src_idx = compute_patch_mapping(
            "pre_answer", src,
            target_question_end=200, target_filler_start=205,
            target_seq_len=260,
        )
        assert tgt_pos is None
        assert src_idx is None

    def test_k1_patches_all_filler_and_beyond(self):
        src = make_source_info(question_end=100, filler_start=105,
                               filler_end=154, seq_len=160)
        tgt_pos, src_idx = compute_patch_mapping(
            "k1", src,
            target_question_end=200, target_filler_start=205,
            target_seq_len=260,
        )
        # k1 starts at filler_start + 0 = 205 in target, 105 in source
        assert tgt_pos[0] == 205
        # Should go through answer_prompt (259)
        assert tgt_pos[-1] == 259
        # Source should start at filler_start (105)
        src_positions = src["positions"]
        assert src_positions[src_idx[0]] == 105

    def test_k50_patches_last_filler_and_beyond(self):
        src = make_source_info(question_end=100, filler_start=105,
                               filler_end=154, seq_len=160)
        tgt_pos, src_idx = compute_patch_mapping(
            "k50", src,
            target_question_end=200, target_filler_start=205,
            target_seq_len=260,
        )
        # k50 starts at filler_start + 49 = 254 in target
        assert tgt_pos[0] == 254
        # Source starts at 105 + 49 = 154
        src_positions = src["positions"]
        assert src_positions[src_idx[0]] == 154

    def test_question_end_includes_question_end(self):
        src = make_source_info()
        tgt_pos, src_idx = compute_patch_mapping(
            "question_end", src,
            target_question_end=200, target_filler_start=205,
            target_seq_len=260,
        )
        assert tgt_pos[0] == 200  # starts at question_end
        assert tgt_pos[-1] == 259  # includes answer_prompt

    def test_position_count_k1_vs_question_end(self):
        src = make_source_info()
        tgt_qe, _ = compute_patch_mapping(
            "question_end", src,
            target_question_end=200, target_filler_start=205,
            target_seq_len=260,
        )
        tgt_k1, _ = compute_patch_mapping(
            "k1", src,
            target_question_end=200, target_filler_start=205,
            target_seq_len=260,
        )
        # question_end patches more positions (includes question_end through filler_start-1)
        assert len(tgt_qe) > len(tgt_k1)

    def test_k25_patches_fewer_than_k1(self):
        src = make_source_info()
        tgt_k1, _ = compute_patch_mapping(
            "k1", src,
            target_question_end=200, target_filler_start=205,
            target_seq_len=260,
        )
        tgt_k25, _ = compute_patch_mapping(
            "k25", src,
            target_question_end=200, target_filler_start=205,
            target_seq_len=260,
        )
        assert len(tgt_k25) < len(tgt_k1)


# ==============================================================================
# build_dense_positions tests
# ==============================================================================

class TestBuildDensePositions:
    def test_basic(self):
        pos = build_dense_positions(question_end=100, filler_start=105, seq_len=160)
        assert pos[0] == 100
        assert pos[-1] == 159
        assert len(pos) == 60  # 100..159

    def test_contiguous(self):
        pos = build_dense_positions(50, 55, 70)
        assert pos == list(range(50, 70))


# ==============================================================================
# select_pairs tests
# ==============================================================================

class TestSelectPairs:
    def _make_categories_file(self, tmp_path):
        examples = [
            # filler_helped targets
            {"problem_idx": 0, "fact_value": 10, "x": 5, "category_dots_50": "filler_helped"},
            {"problem_idx": 1, "fact_value": 20, "x": 5, "category_dots_50": "filler_helped"},
            # both_correct sources
            {"problem_idx": 2, "fact_value": 80, "x": 5, "category_dots_50": "both_correct"},
            {"problem_idx": 3, "fact_value": 30, "x": 5, "category_dots_50": "both_correct"},
            # different Y
            {"problem_idx": 4, "fact_value": 90, "x": 10, "category_dots_50": "both_correct"},
            # both_wrong (not in source pool)
            {"problem_idx": 5, "fact_value": 50, "x": 5, "category_dots_50": "both_wrong"},
        ]
        path = tmp_path / "categories.json"
        with open(path, "w") as f:
            json.dump({"examples": examples}, f)
        return path

    def test_basic_pairing(self, tmp_path):
        path = self._make_categories_file(tmp_path)
        pairs = select_pairs(path, max_pairs=200, min_delta_a=10)
        assert len(pairs) >= 1
        # All targets should be filler_helped
        target_idxs = {p["target_idx"] for p in pairs}
        assert target_idxs <= {0, 1}

    def test_y_matching(self, tmp_path):
        path = self._make_categories_file(tmp_path)
        pairs = select_pairs(path, max_pairs=200, min_delta_a=10)
        for p in pairs:
            assert p["Y"] == 5  # all filler_helped have x=5

    def test_min_delta_a(self, tmp_path):
        path = self._make_categories_file(tmp_path)
        pairs = select_pairs(path, max_pairs=200, min_delta_a=50)
        for p in pairs:
            assert p["abs_delta_A"] >= 50

    def test_no_self_pairing(self, tmp_path):
        path = self._make_categories_file(tmp_path)
        pairs = select_pairs(path, max_pairs=200, min_delta_a=10)
        for p in pairs:
            assert p["source_idx"] != p["target_idx"]

    def test_largest_delta_selected(self, tmp_path):
        path = self._make_categories_file(tmp_path)
        pairs = select_pairs(path, max_pairs=200, min_delta_a=10)
        # Target 0 (A=10) should pair with source 2 (A=80, delta=70)
        # not source 3 (A=30, delta=20)
        t0_pair = [p for p in pairs if p["target_idx"] == 0]
        if t0_pair:
            assert t0_pair[0]["source_idx"] == 2
            assert t0_pair[0]["abs_delta_A"] == 70

    def test_both_wrong_excluded_from_sources(self, tmp_path):
        path = self._make_categories_file(tmp_path)
        pairs = select_pairs(path, max_pairs=200, min_delta_a=10)
        source_idxs = {p["source_idx"] for p in pairs}
        assert 5 not in source_idxs  # both_wrong should not be a source


# ==============================================================================
# ContiguousInjector tests (mock model)
# ==============================================================================

class MockModule(torch.nn.Module):
    """Minimal module to register hooks on."""
    def forward(self, x):
        return x


class MockModel:
    """Minimal model-like object with layers."""
    def __init__(self, n_layers=3):
        self.model = type("M", (), {"layers": [MockModule() for _ in range(n_layers)]})()


class TestContiguousInjector:
    def test_hook_replaces_positions(self):
        model = MockModel(n_layers=2)
        hidden_dim = 8
        # Source states: 2 positions, 2 layers
        source_states = {
            0: torch.ones(2, hidden_dim) * 99,
            1: torch.ones(2, hidden_dim) * 88,
        }
        patch_positions = [3, 5]

        injector = ContiguousInjector(model, [0, 1], patch_positions, source_states)
        injector.register_hooks()

        # Simulate prefill: seq_len > 1
        fake_hidden = torch.zeros(1, 10, hidden_dim)
        # Manually trigger hooks
        for i, layer in enumerate(model.model.layers):
            for hook in layer._forward_hooks.values():
                result = hook(layer, None, (fake_hidden,))
                if result is not None:
                    fake_hidden = result[0] if isinstance(result, tuple) else result

        # Positions 3 and 5 should be replaced
        assert torch.all(fake_hidden[0, 3] == 88)  # layer 1 was last
        assert torch.all(fake_hidden[0, 5] == 88)
        # Other positions should be zero
        assert torch.all(fake_hidden[0, 0] == 0)
        assert torch.all(fake_hidden[0, 4] == 0)

        injector.remove_hooks()

    def test_hook_skips_generation(self):
        model = MockModel(n_layers=1)
        hidden_dim = 8
        source_states = {0: torch.ones(1, hidden_dim) * 99}

        injector = ContiguousInjector(model, [0], [0], source_states)
        injector.register_hooks()

        # Simulate generation step: seq_len = 1
        fake_hidden = torch.zeros(1, 1, hidden_dim)
        for hook in model.model.layers[0]._forward_hooks.values():
            result = hook(model.model.layers[0], None, (fake_hidden,))

        # Should return original output (no modification)
        assert result is None or torch.all(fake_hidden == 0)

        injector.remove_hooks()

    def test_remove_hooks_cleans_up(self):
        model = MockModel(n_layers=2)
        source_states = {0: torch.ones(1, 4), 1: torch.ones(1, 4)}
        injector = ContiguousInjector(model, [0, 1], [0], source_states)
        injector.register_hooks()
        assert sum(len(l._forward_hooks) for l in model.model.layers) == 2
        injector.remove_hooks()
        assert sum(len(l._forward_hooks) for l in model.model.layers) == 0

    def test_layer_filter(self):
        """Only layers in source_states should get hooks."""
        model = MockModel(n_layers=3)
        # Only provide states for layer 1
        source_states = {1: torch.ones(1, 4)}
        injector = ContiguousInjector(model, [0, 1, 2], [0], source_states)
        injector.register_hooks()
        # Only layer 1 should have a hook
        assert len(model.model.layers[0]._forward_hooks) == 0
        assert len(model.model.layers[1]._forward_hooks) == 1
        assert len(model.model.layers[2]._forward_hooks) == 0
        injector.remove_hooks()


# ==============================================================================
# Integration: START_CONDITIONS consistency
# ==============================================================================

class TestStartConditions:
    def test_all_conditions_handled(self):
        """Every condition in START_CONDITIONS should be handled by compute_patch_mapping."""
        src = make_source_info()
        for cond_name in START_CONDITIONS:
            # Should not raise
            tgt_pos, src_idx = compute_patch_mapping(
                cond_name, src,
                target_question_end=200, target_filler_start=205,
                target_seq_len=260,
            )
            if cond_name == "pre_answer":
                assert tgt_pos is None
            else:
                assert isinstance(tgt_pos, list)

    def test_conditions_are_ordered_by_patch_count(self):
        """Earlier start conditions should patch more positions."""
        src = make_source_info()
        counts = {}
        for cond_name in START_CONDITIONS:
            tgt_pos, _ = compute_patch_mapping(
                cond_name, src,
                target_question_end=200, target_filler_start=205,
                target_seq_len=260,
            )
            if tgt_pos is not None:
                counts[cond_name] = len(tgt_pos)

        assert counts["question_end"] > counts["k1"]
        assert counts["k1"] > counts["k5"]
        assert counts["k5"] > counts["k10"]
        assert counts["k10"] > counts["k25"]
        assert counts["k25"] > counts["k50"]
        assert counts["k50"] > counts["answer_prompt"]
        assert counts["answer_prompt"] == 1
