from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class SAEConfig:
    # Data
    extraction_dir: str = "data/extracted_states"
    categories_file: str = "results/probe_results/categories.json"
    conditions: Tuple[str, str] = ("baseline", "dots_50")
    position: str = "answer_prompt"
    layer: int = 58
    delta: bool = False  # If True, train on h(cond_b) - h(cond_a) at two positions
    delta_positions: Tuple[str, str] = ("answer_prompt", "pre_answer")

    # Model
    input_dim: int = 7168
    encoder_hidden: int = 512
    dim_shared: int = 64
    dim_view: int = 32
    topk_shared: int = 0   # 0 = no topk (use L1 only), >0 = keep top K active
    topk_view: int = 0

    # Training
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 64
    n_epochs: int = 500
    seed: int = 42

    # Loss weights
    lambda_shared: float = 0.01
    lambda_view: float = 0.01
    lambda_consist: float = 1.0
    lambda_separate: float = 0.1

    # Consistency filter: only enforce consistency on examples where
    # the model answers correctly in both conditions
    consistency_filter: str = "both_correct"

    # Example filter: which categories to include in training
    # Default: all categories (consistency loss already filters to correct examples)
    include_categories: Optional[Tuple[str, ...]] = None

    # Output
    output_dir: str = "sae_results"
