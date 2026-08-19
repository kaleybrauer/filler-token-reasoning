"""
apply_lens.py — load a fitted Jacobian lens and transport residuals with it.

Applying a lens is forward-only: J_L acts on the RAW residual, before the final
RMSNorm, so a fitted lens drops straight onto the already-extracted numpy states
with no re-extraction and no model. Same states in both arms = a clean
head-to-head against the logit lens (which is the J_L = I special case).

    lens_L(h) = softmax( W_U . RMSNorm( J_L . h ) )

Consumers: scripts/decode/extract_residual_fingerprints.py --jlens,
           scripts/analysis/decode_2fact_heatmap.py --jlens
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_jlens(path: str | Path) -> dict[int, np.ndarray]:
    """Load {layer -> J [d,d] float32} from a lens file, a fit checkpoint, or an .npz.

    A fit checkpoint (written every --checkpoint-every prompts, holding a running
    SUM plus n_done) is accepted so a partial fit can be inspected while the run
    is still going; the sum is divided by n_done to give the same mean the
    finished lens would have.
    """
    path = Path(path)
    if path.suffix == ".npz":
        z = np.load(path)
        return {int(k[1:]): z[k].astype(np.float32) for k in z.files if k.startswith("L")}

    import torch

    ck = torch.load(path, map_location="cpu", weights_only=True)
    if "J" in ck:  # JacobianLens.save()
        return {int(l): J.float().numpy() for l, J in ck["J"].items()}
    if "jacobian_sum" in ck:  # jlens.fit() checkpoint: running sum
        n = ck["n_done"]
        if n == 0:
            raise ValueError(f"{path} has n_done=0 — no prompts fitted yet")
        return {int(l): (J.float() / n).numpy() for l, J in ck["jacobian_sum"].items()}
    raise ValueError(f"{path}: not a lens or a fit checkpoint (keys {sorted(ck)})")


def transport(vecs: np.ndarray, J: np.ndarray) -> np.ndarray:
    """h -> J h, batched over rows: vecs [n, d] @ J.T -> [n, d]."""
    return vecs @ J.T.astype(vecs.dtype)


def describe(lens: dict[int, np.ndarray]) -> str:
    layers = sorted(lens)
    d = lens[layers[0]].shape[0]
    return f"J-lens: {len(layers)} layers [{layers[0]}..{layers[-1]}], d={d}"
