"""
v3_autograd.py — make DeepSeek V3 AWQ differentiable w.r.t. activations.

The Jacobian lens (Anthropic, "Verbalizable Representations Form a Global
Workspace in Language Models") needs d h_final / d h_L: gradients with respect
to *activations*, never weights. For a linear layer that is grad_x = grad_y @ W,
which needs W's values, not its dtype and not its gradient — so a 4-bit AWQ
checkpoint can supply it exactly, and no bf16 copy of V3 is required.

Three runtime changes, no files edited:

1. enable_awq_autograd(model)
     AutoAWQ already ships the exact activation gradient: WQLinearMMFunction
     .backward dequantises the weight and returns grad_output @ W.T. But
     WQLinear_GEMM.forward only builds the graph when `self.training` is True;
     in eval it wraps the same call in torch.no_grad(). Flipping .training on
     the WQLinear leaves is all that is needed. Must run AFTER jlens.from_hf(),
     which calls model.eval() and would undo it.

2. patch_moe_infer(model)
     DeepseekV3MoE.moe_infer carries modeling_deepseek.py's only
     @torch.no_grad(), plus two in-place ops autograd's version counter
     rejects. The replacement is the same computation written out-of-place.
     Discrete top-k routing stays constant (correct straight-through);
     gradient flows through the selected experts and the gate weights.

3. patch_awq_backward()
     The stock backward materialises batch_size copies of the dequantised
     weight (`.repeat(B,1,1)`) before torch.bmm. Broadcasting matmul computes
     the identical product without the copy — for V3's o_proj that avoids
     235 MB x dim_batch of transient allocation per call. Verified equal in
     gate G1.

Nothing here changes the model's forward numerics: (1) and (2) only decide
whether a graph is recorded, (3) only touches the backward.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

# Applies the PytorchGELUTanh shim autoawq needs, and gives us the one correct
# tokenizer loader (AutoTokenizer silently merges filler tokens on this repo).
from extract.extract_hidden_states import load_tokenizer  # noqa: E402

V3_AWQ_PATH = "/workspace/models/deepseek-v3-awq"


# ---------------------------------------------------------------- loading


def build_device_map(splits: str, n_layers: int = 61, n_gpus: int = 3) -> dict:
    """Explicit per-module device map: 'a,b' puts layers [0,a) on GPU0,
    [a,b) on GPU1, [b,n_layers) on GPU2, embeddings on GPU0, norm+lm_head last.

    Why not device_map="auto": it routes through accelerate's
    get_balanced_memory, which can only LOWER each card's cap toward a uniform
    share. So a deliberate skew is impossible, and lowering one card below that
    share drops the total under the 327.7 GiB of weights and triggers CPU
    offload — which the AWQ quantizer rejects outright. Measured, "auto" splits
    the layers 21/19/21 and lands 119.6 GiB on GPU2, which then OOMs first
    because with source_layers=None every card also retains activations.

    Cost per card is weights + activations, and activations scale with the
    layer count on that card (~1.1 GiB/layer at dim_batch=8, ~0.55 at 4):

        auto (21/19/21):  ~126 / ~127 / ~142 GiB   -> GPU2 over the 139.8 limit
        22/20/19:         ~133 / ~133 / ~129 GiB   -> fits

    The 3 dense layers (0-2) are ~16x smaller than a MoE layer, which is why
    GPU0 can carry more blocks than the others.
    """
    a, b = (int(x) for x in splits.split(","))
    device_map = {"model.embed_tokens": 0, "model.norm": n_gpus - 1,
                  "lm_head": n_gpus - 1}
    for i in range(n_layers):
        device_map[f"model.layers.{i}"] = 0 if i < a else (1 if i < b else 2)
    return device_map


def load_v3_awq(
    model_path: str = V3_AWQ_PATH,
    *,
    max_memory_gib: str | int = "130,122,108",
    device_map: str | None = None,
    attn_implementation: str = "eager",
    verbose: bool = True,
):
    """Load V3 AWQ across the visible GPUs, front-loaded for backward headroom.

    327.7 GiB of weights + 11.2 GiB of buffers must fit without CPU offload (the
    AWQ quantizer rejects a device_map containing cpu/disk). Accelerate also
    keeps per-device slack for the largest atomic block, so the caps have to sum
    to ~360 GiB: uniform 112 (336) and skewed 130/108/108 (346) both fell back
    to CPU and were rejected.

    A uniform 120 GiB cap (360 total) loads, but places weights unevenly —
    103.8 / 106.7 / 119.6 GiB — and with source_layers=None EVERY card retains
    activations, so the fit OOMs on GPU2 first (measured: dim_batch=8 died with
    71 MiB free on GPU2 while GPU0/GPU1 still had 25/22 GiB spare). The default
    below keeps the same 360 GiB total but caps GPU2 lower so the three cards end
    up with comparable free memory, which is what sets the largest usable
    dim_batch — and dim_batch is the whole ballgame for throughput, since ~97% of
    a backward pass is fixed per-node overhead rather than FLOPs.

    max_memory_gib: int for a uniform cap, or a per-GPU comma list.
    """
    from transformers import AutoModelForCausalLM

    tokenizer = load_tokenizer(model_path)

    n_gpus = torch.cuda.device_count()
    if isinstance(max_memory_gib, str) and "," in max_memory_gib:
        caps = [int(x) for x in max_memory_gib.split(",")]
        if len(caps) != n_gpus:
            raise ValueError(f"{len(caps)} caps for {n_gpus} GPUs")
    else:
        caps = [int(max_memory_gib)] * n_gpus
    max_memory = {i: f"{caps[i]}GiB" for i in range(n_gpus)}
    if verbose:
        print(f"Loading {model_path} on {n_gpus} GPUs, max_memory={max_memory}")

    t0 = time.time()

    def _load(mm, dmap="auto"):
        return AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=dmap,
            max_memory=None if dmap != "auto" else mm,
            torch_dtype=torch.float16,  # AWQ CUDA kernels are fp16-only
            trust_remote_code=True,
            attn_implementation=attn_implementation,
        )

    if device_map:
        explicit = build_device_map(device_map, n_gpus=n_gpus)
        if verbose:
            print(f"  explicit device_map splits={device_map}: "
                  f"{sum(1 for v in explicit.values() if v == 0)}/"
                  f"{sum(1 for v in explicit.values() if v == 1)}/"
                  f"{sum(1 for v in explicit.values() if v == 2)} modules per GPU")
        try:
            model = _load(None, explicit)
            model.eval()
            if verbose:
                print(f"Model loaded in {time.time() - t0:.0f}s")
                for i in range(n_gpus):
                    print(f"  GPU {i}: "
                          f"{torch.cuda.memory_allocated(i) / 2**30:.1f} GiB weights")
            return model, tokenizer
        except Exception as exc:
            # Never let a device-map experiment cost a whole unattended night:
            # fall through to the auto path, which is known to load and run.
            print(f"  explicit device_map failed ({type(exc).__name__}: "
                  f"{str(exc)[:160]}); falling back to auto")
            torch.cuda.empty_cache()

    try:
        model = _load(max_memory)
    except ValueError as exc:
        if "CPU or disk device" not in str(exc):
            raise
        # Accelerate wanted to offload: fall back to the extraction path's
        # known-good uniform cap (fails in seconds, before weights are read).
        fallback = {i: "120GiB" for i in range(n_gpus)}
        print(f"  caps {max_memory} forced CPU offload; retrying with {fallback}")
        model = _load(fallback)
    model.eval()
    if verbose:
        print(f"Model loaded in {time.time() - t0:.0f}s")
        for i in range(n_gpus):
            print(f"  GPU {i}: {torch.cuda.memory_allocated(i) / 2**30:.1f} GiB weights")
    return model, tokenizer


def layer_devices(model) -> dict[int, torch.device]:
    """Block index -> device, for reading the device_map after load."""
    return {
        i: next(block.parameters()).device
        for i, block in enumerate(model.model.layers)
    }


# ------------------------------------------------------- 1. AWQ autograd


def enable_awq_autograd(model) -> int:
    """Set .training=True on every WQLinear leaf so the graph is recorded.

    Returns the number of modules flipped. Call AFTER jlens.from_hf(), which
    calls model.eval(). The MoE container and MoEGate must stay in eval:
    DeepseekV3MoE.forward has `if not self.training: y = moe_infer` with no
    else branch, and MoEGate.forward asserts `not self.training`.
    """
    from awq.modules.linear.gemm import WQLinear_GEMM

    n = 0
    for module in model.modules():
        if isinstance(module, WQLinear_GEMM):
            module.training = True
            n += 1
    if n == 0:
        raise RuntimeError("no WQLinear_GEMM modules found — is this an AWQ model?")

    for module in model.modules():
        name = type(module).__name__
        if name in ("DeepseekV3MoE", "MoEGate") and module.training:
            raise RuntimeError(f"{name} must stay in eval mode; see docstring")
    return n


# ---------------------------------------------------------- 2. MoE patch


def _moe_infer_autograd(self, x, topk_ids, topk_weight):
    """moe_infer without @torch.no_grad() and without in-place ops.

    Identical arithmetic to the stock implementation for ep_size == 1; the two
    rewrites are `new_x[idxs] = outs` -> index_copy (out-of-place; idxs is a
    permutation so every row is written exactly once) and `.mul_()` -> `*`.
    """
    if self.ep_size > 1:
        raise NotImplementedError("expert parallelism is not supported here")

    cnts = topk_ids.new_zeros((topk_ids.shape[0], len(self.experts)))
    cnts.scatter_(1, topk_ids, 1)  # integer bookkeeping, no grad
    tokens_per_expert = cnts.sum(dim=0)
    idxs = topk_ids.view(-1).argsort()
    sorted_tokens = x[idxs // topk_ids.shape[1]]
    tokens_per_expert = tokens_per_expert.cpu().numpy()

    outputs = []
    start_idx = 0
    for i, num_tokens in enumerate(tokens_per_expert):
        end_idx = start_idx + num_tokens
        if num_tokens == 0:
            continue
        expert = self.experts[i + self.ep_rank * self.experts_per_rank]
        outputs.append(expert(sorted_tokens[start_idx:end_idx]))
        start_idx = end_idx
    outs = torch.cat(outputs, dim=0) if len(outputs) else sorted_tokens.new_empty(0)

    new_x = torch.zeros_like(outs).index_copy(0, idxs, outs)
    return (
        (
            new_x.view(*topk_ids.shape, -1).type(topk_weight.dtype)
            * topk_weight.unsqueeze(dim=-1)
        )
        .sum(dim=1)
        .type(new_x.dtype)
    )


def patch_moe_infer(model) -> int:
    """Swap in the autograd-safe moe_infer on the loaded DeepseekV3MoE class."""
    moe_cls = None
    n = 0
    for module in model.modules():
        if type(module).__name__ == "DeepseekV3MoE":
            moe_cls = type(module)
            n += 1
    if moe_cls is None:
        raise RuntimeError("no DeepseekV3MoE modules found")
    moe_cls.moe_infer = _moe_infer_autograd
    return n


# ------------------------------------------------------ 3. AWQ backward


_ORIGINAL_AWQ_BACKWARD = None


def _awq_backward_no_repeat(ctx, grad_output):
    """Same value as the stock backward, without replicating the weight.

    Stock: grad_output.bmm(W.T.unsqueeze(0).repeat(B,1,1)).
    Here:  grad_output @ W.T  — matmul broadcasts a 2-D operand over the batch.
    """
    from awq.modules.linear import gemm as G

    _, qweight, qzeros, scales, _ = ctx.saved_tensors
    if G.awq_ext is not None:
        weights = G.awq_ext.dequantize_weights_cuda(
            qweight, scales, qzeros, 1, 0, 0, False
        ).to(grad_output.dtype)
    else:
        weights = G.awq_dequantize_triton(qweight, scales, qzeros).to(grad_output.dtype)

    grad_input = None
    if ctx.needs_input_grad[0]:
        grad_input = torch.matmul(grad_output, weights.transpose(0, 1))
    return grad_input, None, None, None, None, None, None, None


def patch_awq_backward() -> None:
    """Install the no-repeat backward (idempotent)."""
    global _ORIGINAL_AWQ_BACKWARD
    from awq.modules.linear.gemm import WQLinearMMFunction

    if _ORIGINAL_AWQ_BACKWARD is None:
        _ORIGINAL_AWQ_BACKWARD = WQLinearMMFunction.backward
    WQLinearMMFunction.backward = staticmethod(_awq_backward_no_repeat)


def unpatch_awq_backward() -> None:
    from awq.modules.linear.gemm import WQLinearMMFunction

    if _ORIGINAL_AWQ_BACKWARD is not None:
        WQLinearMMFunction.backward = _ORIGINAL_AWQ_BACKWARD


# ---------------------------------------------------------------- all-in-one


def make_differentiable(model, *, no_repeat_backward: bool = True) -> dict:
    """Apply every patch. Call after jlens.from_hf()."""
    n_awq = enable_awq_autograd(model)
    n_moe = patch_moe_infer(model)
    if no_repeat_backward:
        patch_awq_backward()
    return {
        "awq_leaves_in_train_mode": n_awq,
        "moe_layers_patched": n_moe,
        "no_repeat_backward": no_repeat_backward,
    }


# ------------------------------------------- 4. gradient checkpointing

_CKPT_ORIGINALS: dict[int, object] = {}


def enable_block_checkpointing(model) -> int:
    """Recompute each block's internals during the backward instead of keeping them.

    modeling_deepseek.py advertises `supports_gradient_checkpointing = True` but
    its forward never calls checkpoint, so each block's forward is wrapped by
    hand. Wrap AFTER loading, so `orig` is accelerate's device-map-hooked
    forward and the cross-device moves are replayed on recompute too.

    Why this is a *speed* change here, when checkpointing is normally a pure
    compute cost: peak activation memory stops scaling with depth and scales
    only with dim_batch (61 block-boundary tensors + ONE block's internals,
    ~2 GiB at B=8, against ~73 GiB retained today). That lifts the ceiling on
    dim_batch, and throughput is linear in dim_batch because ~82% of a pass is
    fixed per-node overhead — ~45k quantised expert matmuls, each dequantising
    its weight — rather than FLOPs.

    Numerically transparent: eval mode, dropout 0, and routing is argsort/topk
    over identical inputs, so the recompute reproduces the same experts and the
    same activations. Verified rather than assumed — scripts/jlens/ckpt_verify.py
    refits J_59 and compares against the non-checkpointed value.
    """
    import torch.utils.checkpoint as ckpt

    def _make(orig_fwd):
        def fwd(*args, **kwargs):
            # No graph is being built (blocks below jlens' start_graph_at, or
            # any plain no-grad pass) -> nothing to recompute, and checkpoint
            # would only add a warning and overhead.
            if not torch.is_grad_enabled() or not any(
                torch.is_tensor(a) and a.requires_grad
                for a in (*args, *kwargs.values())
            ):
                return orig_fwd(*args, **kwargs)
            return ckpt.checkpoint(orig_fwd, *args, use_reentrant=False, **kwargs)

        return fwd

    n = 0
    for block in model.model.layers:
        if id(block) in _CKPT_ORIGINALS:
            continue
        _CKPT_ORIGINALS[id(block)] = block.forward
        block.forward = _make(block.forward)
        n += 1
    return n


def disable_block_checkpointing(model) -> int:
    n = 0
    for block in model.model.layers:
        orig = _CKPT_ORIGINALS.pop(id(block), None)
        if orig is not None:
            block.forward = orig
            n += 1
    return n
