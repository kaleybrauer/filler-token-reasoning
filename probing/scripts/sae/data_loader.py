"""Load paired views from extracted hidden states for multi-view SAE training."""

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class ExampleMetadata:
    problem_idx: int
    fact_value: int   # A
    x: int            # Y
    answer: int       # A+Y
    model_correct: Dict[str, bool]  # {condition_name: bool}
    category: str


def load_view(
    extraction_dir: Path,
    condition: str,
    position: str,
    layer: int,
    max_examples: Optional[int] = None,
) -> Tuple[np.ndarray, List[dict]]:
    """Load hidden states for one (condition, position, layer).

    Returns:
        vectors: (N, 7168) float32
        metadata: list of dicts with problem info
    """
    cond_dir = extraction_dir / condition
    files = sorted(cond_dir.glob("prob_*.pkl"))
    if max_examples:
        files = files[:max_examples]

    vectors = []
    metadata = []
    for f in files:
        with open(f, "rb") as fp:
            data = pickle.load(fp)

        if position not in data["states"]:
            raise KeyError(f"Position '{position}' not in {f.name}. "
                           f"Available: {list(data['states'].keys())}")
        if layer not in data["states"][position]:
            raise KeyError(f"Layer {layer} not in {f.name}")

        vec = data["states"][position][layer].astype(np.float32)
        vectors.append(vec)
        metadata.append({
            "problem_idx": data["problem_idx"],
            "fact_value": data["fact_value"],
            "x": data["x"],
            "answer": data["answer"],
            "model_correct": data.get("model_correct"),
            "model_answer": data.get("model_answer"),
        })

    return np.stack(vectors, axis=0), metadata


def load_categories(categories_file: Path, condition: str = "dots_50") -> Dict[int, str]:
    """Load per-example category labels."""
    with open(categories_file) as f:
        cat_data = json.load(f)

    cat_field = f"category_{condition}"
    categories = {}
    for ex in cat_data["examples"]:
        field = cat_field if cat_field in ex else "category"
        categories[ex["problem_idx"]] = ex.get(field, "unknown")
    return categories


def load_paired_views(
    extraction_dir: Path,
    conditions: Tuple[str, str],
    position: str,
    layer: int,
    categories_file: Optional[Path] = None,
) -> Tuple[np.ndarray, np.ndarray, List[ExampleMetadata]]:
    """Load paired views for all examples.

    Returns:
        view_a: (N, D) float32 — first condition
        view_b: (N, D) float32 — second condition
        metadata: list of ExampleMetadata
    """
    cond_a, cond_b = conditions

    vecs_a, meta_a = load_view(extraction_dir, cond_a, position, layer)
    vecs_b, meta_b = load_view(extraction_dir, cond_b, position, layer)

    assert len(meta_a) == len(meta_b), (
        f"Mismatched example counts: {cond_a}={len(meta_a)}, {cond_b}={len(meta_b)}"
    )

    # Verify alignment
    for i, (ma, mb) in enumerate(zip(meta_a, meta_b)):
        assert ma["problem_idx"] == mb["problem_idx"], (
            f"Mismatched problem_idx at position {i}: {ma['problem_idx']} vs {mb['problem_idx']}"
        )

    # Load categories
    categories = {}
    if categories_file and categories_file.exists():
        categories = load_categories(categories_file, conditions[1])

    # Build metadata
    metadata = []
    for ma, mb in zip(meta_a, meta_b):
        metadata.append(ExampleMetadata(
            problem_idx=ma["problem_idx"],
            fact_value=ma["fact_value"],
            x=ma["x"],
            answer=ma["answer"],
            model_correct={
                cond_a: ma["model_correct"],
                cond_b: mb["model_correct"],
            },
            category=categories.get(ma["problem_idx"], "unknown"),
        ))

    return vecs_a, vecs_b, metadata


class PairedViewDataset(Dataset):
    """PyTorch dataset for paired-view SAE training.

    Each item returns (view_a, view_b, correct_in_both, problem_idx).
    Views are standardized using training-set statistics.
    """

    def __init__(
        self,
        view_a: np.ndarray,
        view_b: np.ndarray,
        metadata: List[ExampleMetadata],
        conditions: Tuple[str, str],
        normalize: bool = True,
        stats: Optional[Dict] = None,
        include_categories: Optional[Tuple[str, ...]] = None,
    ):
        # Filter to included categories
        if include_categories is not None:
            keep = [i for i, m in enumerate(metadata) if m.category in include_categories]
            view_a = view_a[keep]
            view_b = view_b[keep]
            metadata = [metadata[i] for i in keep]

        self.n = len(metadata)
        self.metadata = metadata
        self.conditions = conditions
        self.include_categories = include_categories

        # Correctness mask: True if model correct in both conditions
        self.correct_mask = np.array([
            all(m.model_correct.get(c, False) for c in conditions)
            for m in metadata
        ], dtype=bool)

        # Normalize
        if normalize:
            if stats is None:
                self.mean_a = view_a.mean(axis=0)
                self.std_a = view_a.std(axis=0)
                self.std_a[self.std_a < 1e-8] = 1.0
                self.mean_b = view_b.mean(axis=0)
                self.std_b = view_b.std(axis=0)
                self.std_b[self.std_b < 1e-8] = 1.0
            else:
                self.mean_a = stats["mean_a"]
                self.std_a = stats["std_a"]
                self.mean_b = stats["mean_b"]
                self.std_b = stats["std_b"]

            view_a = (view_a - self.mean_a) / self.std_a
            view_b = (view_b - self.mean_b) / self.std_b
        else:
            self.mean_a = self.mean_b = np.zeros(view_a.shape[1])
            self.std_a = self.std_b = np.ones(view_a.shape[1])

        self.view_a = torch.from_numpy(view_a).float()
        self.view_b = torch.from_numpy(view_b).float()
        self.correct_mask_t = torch.from_numpy(self.correct_mask)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return (
            self.view_a[idx],
            self.view_b[idx],
            self.correct_mask_t[idx],
            idx,
        )

    def get_stats(self) -> Dict:
        return {
            "mean_a": self.mean_a,
            "std_a": self.std_a,
            "mean_b": self.mean_b,
            "std_b": self.std_b,
        }

    def get_ground_truth(self) -> Dict[str, np.ndarray]:
        """Return ground truth arrays (for post-hoc evaluation only)."""
        return {
            "A": np.array([m.fact_value for m in self.metadata], dtype=np.float32),
            "Y": np.array([m.x for m in self.metadata], dtype=np.float32),
            "A+Y": np.array([m.answer for m in self.metadata], dtype=np.float32),
            "category": np.array([m.category for m in self.metadata]),
            "correct_mask": self.correct_mask,
        }
