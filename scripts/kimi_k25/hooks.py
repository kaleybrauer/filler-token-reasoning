"""
hooks.py — worker-side hidden-state capture for vLLM V1.

vLLM >= 0.15 (needed for Kimi K2.5) has no V0 engine, so the path the older
extractors use —

    llm.llm_engine.model_executor.driver_worker.model_runner.model

— no longer exists: with TP the workers are separate processes and the driver
process holds no model. The supported route is a worker extension class, which
vLLM instantiates inside every worker and exposes through collective_rpc:

    llm = LLM(..., worker_extension_cls="hooks.HiddenStateCaptureExt")
    llm.collective_rpc("hs_install")
    ...
    llm.generate(...)
    per_worker = llm.collective_rpc("hs_fetch", args=(positions,))
    states = per_worker[0]["states"]      # pure TP: every worker holds all layers

This module must be importable inside the worker processes, so it lives on its
own (no repo-relative imports) and the caller puts its directory on PYTHONPATH.

CAPTURE CONVENTION. vLLM's DeepseekV2DecoderLayer.forward returns
`(hidden_states, residual)` where `hidden_states` is the layer's MLP write and
the residual stream after layer L is the SUM of the two. Hooking output[0], as
the Kimi K2 extractor does, therefore captures layer writes: verified on the
saved states, where the answer is undecodable at every layer until a 100% jump at
the final one, versus the smooth emergence seen in the transformers-path V3
states. We capture the sum, and keep the raw write for the last layer so the
caller can check both against what the model actually generated.
"""
from __future__ import annotations

import torch

# Attribute paths tried in order when locating the decoder-layer list. K2.5 wraps
# the text model (KimiK25ForConditionalGeneration -> language_model), the older
# Kimi/DeepSeek models do not.
LAYER_PATHS = (
    "language_model.model.layers",
    "model.language_model.model.layers",
    "model.layers",
    "model.model.layers",
)


class HiddenStateCaptureExt:
    """Mixed into each vLLM worker; every method here is callable via collective_rpc."""

    # -- internals ---------------------------------------------------------

    def _hs_model(self):
        runner = getattr(self, "model_runner", None)
        if runner is None:
            raise RuntimeError("worker has no model_runner; vLLM internals changed")
        model = getattr(runner, "model", None)
        if model is None and hasattr(runner, "get_model"):
            model = runner.get_model()
        if model is None:
            raise RuntimeError("could not reach the model on this worker")
        return model

    def _hs_layers(self, model):
        """Return (layer_list, owning_module, path_used)."""
        for path in LAYER_PATHS:
            obj = model
            ok = True
            for part in path.split("."):
                if not hasattr(obj, part):
                    ok = False
                    break
                obj = getattr(obj, part)
            if ok and isinstance(obj, torch.nn.ModuleList) and len(obj) > 0:
                owner = model
                for part in path.split(".")[:-1]:
                    owner = getattr(owner, part)
                return obj, owner, path
        raise RuntimeError(f"no decoder-layer ModuleList found; tried {LAYER_PATHS}")

    def _hs_hook(self, global_idx: int):
        def hook(module, inputs, output):
            # Only the first forward after hs_clear() is the prefill we want; vLLM
            # calls the layer again for every decode step, with shape (1, hidden).
            if global_idx in self._hs_captured:
                return
            if (isinstance(output, tuple) and len(output) >= 2
                    and torch.is_tensor(output[0]) and torch.is_tensor(output[1])
                    and output[0].shape == output[1].shape):
                # split-residual convention: stream = write + carried residual
                hidden = output[0] + output[1]
                self._hs_convention = "sum(output[0], output[1])"
                if global_idx == self._hs_last_layer:
                    self._hs_raw_last = output[0].detach()
            elif isinstance(output, tuple):
                hidden = output[0]
                self._hs_convention = "output[0] (no residual in return)"
            else:
                hidden = output
                self._hs_convention = "output (single tensor)"
            self._hs_captured[global_idx] = hidden.detach()
        return hook

    # -- rpc surface -------------------------------------------------------

    def hs_install(self) -> dict:
        """Register forward hooks on every decoder layer this worker owns."""
        model = self._hs_model()
        layers, owner, path = self._hs_layers(model)

        # Under pipeline parallel a worker owns a slice of the layers and the rest
        # are PPMissingLayer placeholders; start_layer maps local -> global index.
        start = int(getattr(owner, "start_layer", 0))

        self._hs_captured = {}
        self._hs_raw_last = None
        self._hs_convention = None
        self._hs_handles = []
        self._hs_last_layer = start + len(layers) - 1

        for i, layer in enumerate(layers):
            if type(layer).__name__ == "PPMissingLayer":
                continue
            self._hs_handles.append(
                layer.register_forward_hook(self._hs_hook(start + i))
            )
        return {
            "path": path,
            "start_layer": start,
            "n_hooked": len(self._hs_handles),
            "n_layers_local": len(layers),
        }

    def hs_clear(self) -> None:
        self._hs_captured = {}
        self._hs_raw_last = None

    def hs_fetch(self, positions: list[int], want_raw_last: bool = False) -> dict:
        """Slice the captured states to `positions` and return them as fp16 arrays.

        Slicing happens here on purpose: a full capture is 61 x seq x 7168, ~786 MB
        per request, which must never cross the RPC boundary.
        """
        states = {}
        for layer_idx, tensor in self._hs_captured.items():
            flat = tensor if tensor.dim() == 2 else tensor[0]
            states[layer_idx] = flat[positions].to(torch.float16).cpu().numpy()

        out = {
            "states": states,
            "convention": self._hs_convention,
            "n_layers": len(states),
            "seq_len": int(next(iter(self._hs_captured.values())).shape[-2])
            if self._hs_captured else 0,
        }
        if want_raw_last and self._hs_raw_last is not None:
            flat = self._hs_raw_last
            flat = flat if flat.dim() == 2 else flat[0]
            out["raw_last"] = flat[positions].to(torch.float16).cpu().numpy()
            out["last_layer"] = self._hs_last_layer
        return out

    def hs_remove(self) -> None:
        for handle in getattr(self, "_hs_handles", []):
            handle.remove()
        self._hs_handles = []
