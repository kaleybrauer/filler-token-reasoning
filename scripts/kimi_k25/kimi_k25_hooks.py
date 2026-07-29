"""vLLM V1 worker extension that captures Kimi K2.5's residual stream.

Injected via `LLM(..., worker_extension_cls="kimi_k25_hooks.K25CaptureExt")` and
driven with `llm.collective_rpc("k25_*")`. vLLM adds this class to the Worker's
__bases__ and asserts that no attribute name collides with the Worker's own, so
every name here is prefixed `k25_` / `_k25_`.

Capture convention
------------------
vLLM's `DeepseekV2DecoderLayer.forward` returns `(hidden_states, residual)` in the
*split* convention: `input_layernorm`/`post_attention_layernorm` are fused add+norm,
so on return `residual` is the stream after attention and `hidden_states` is only
the MLP write. The residual stream after layer L is therefore

    output[0] + output[1]

which is exactly what `DeepseekV2Model.forward` then feeds to `self.norm(...)`.
Capturing `output[0]` alone yields per-layer MLP writes — the bug the A6 audit found
in the existing Kimi K2 extractions. Verified against vLLM 0.25.1 source.

The whole point of this file is that the slice happens *inside* the worker: a full
61 x seq x 7168 bf16 capture is ~786 MB per request, versus ~50 MB once sliced.
"""
from __future__ import annotations


class K25CaptureExt:
    # No class-level attributes: vLLM asserts every non-dunder attribute of this
    # class is absent from the Worker class, and instance attrs are invisible to
    # that check.

    def k25_install_hooks(self) -> dict:
        """Register a forward hook on every text-model decoder layer."""
        model = self.model_runner.get_model()
        lm = getattr(model, "language_model", None)
        if lm is None:
            raise RuntimeError(
                f"model {type(model).__name__} has no .language_model — "
                "K2.5 wraps a DeepseekV2ForCausalLM under that attribute"
            )
        layers = lm.model.layers

        self._k25_captured = {}
        self._k25_errors = []
        self._k25_positions = None
        self._k25_seq_len = None
        self._k25_index = None
        self._k25_handles = []

        for i, layer in enumerate(layers):
            self._k25_handles.append(
                layer.register_forward_hook(self._k25_make_hook(i))
            )
        return {
            "rank": getattr(self, "rank", None),
            "n_layers": len(layers),
            "layer_cls": type(layers[0]).__name__,
            "model_cls": type(model).__name__,
        }

    def _k25_make_hook(self, idx: int):
        import torch

        def hook(_mod, _inp, output):
            # Armed only between set_positions() and pop().
            if self._k25_positions is None:
                return
            # First forward per layer after arming == the prefill. Later calls are
            # single-token decode steps and would overwrite it.
            if idx in self._k25_captured:
                return

            if not (isinstance(output, tuple) and len(output) >= 2):
                self._k25_errors.append(
                    f"L{idx}: output is {type(output).__name__}, expected the "
                    "(hidden_states, residual) 2-tuple"
                )
                return
            hidden, residual = output[0], output[1]
            if hidden is None or residual is None:
                self._k25_errors.append(f"L{idx}: None in (hidden, residual)")
                return

            n_tok = hidden.shape[0]
            if n_tok < self._k25_seq_len:
                # A short forward means the prefill was split (chunked prefill) or
                # partly served from cache (prefix caching). Refuse to capture:
                # silently keeping it is exactly the failure mode that produces a
                # complete-looking run with wrong data.
                self._k25_errors.append(
                    f"L{idx}: forward saw {n_tok} tokens but the prompt is "
                    f"{self._k25_seq_len} — prefill was split or cached"
                )
                return

            if self._k25_index is None or self._k25_index.device != hidden.device:
                self._k25_index = torch.tensor(
                    self._k25_positions, dtype=torch.long, device=hidden.device
                )

            sel = self._k25_index
            # fp32 for the add so the bf16 mantissa isn't the limiting factor,
            # then fp16 to match data/extracted_states_varbind_allpos/.
            vec = (hidden.index_select(0, sel).to(torch.float32)
                   + residual.index_select(0, sel).to(torch.float32))
            self._k25_captured[idx] = vec.to(torch.float16).cpu().numpy()

        return hook

    def k25_set_positions(self, positions, seq_len: int) -> int:
        """Arm the hooks for one request. Also clears any previous capture."""
        self._k25_positions = [int(p) for p in positions]
        self._k25_seq_len = int(seq_len)
        self._k25_index = None
        self._k25_captured = {}
        self._k25_errors = []
        return len(self._k25_positions)

    def k25_clear(self) -> None:
        self._k25_captured = {}
        self._k25_errors = []
        self._k25_positions = None
        self._k25_index = None

    def k25_pop(self) -> dict:
        """Return this request's capture. Only rank 0 ships the arrays — with TP=8
        the hidden states are replicated, so returning all 8 would move ~400 MB per
        example for nothing. Every rank still reports its layer count and errors so
        a partial capture on a non-zero rank is still visible.

        The payload carries ONE contiguous fp16 buffer plus its shape, not a dict of
        61 small arrays. vLLM's MessageQueue pickles with out-of-band PickleBuffers
        and inlines anything under 1 MiB; a per-layer dict lands 61 separate ~315 KB
        buffers in that path and came back with the rows mangled (the arrays arrived
        as short lists). Raw bytes + an explicit shape round-trip through any
        serializer unchanged, and it is a single blob rather than 61.
        """
        import numpy as np

        rank = getattr(self, "rank", 0)
        layers = sorted(self._k25_captured)
        payload = {
            "rank": rank,
            "n_captured": len(self._k25_captured),
            "errors": list(self._k25_errors),
            "layers": layers,
            "shape": None,
            "data": None,
        }
        if rank == 0 and layers:
            stacked = np.ascontiguousarray(
                np.stack([self._k25_captured[li] for li in layers]), dtype=np.float16
            )
            payload["shape"] = list(stacked.shape)   # (n_layers, n_pos, hidden)
            payload["data"] = stacked.tobytes()
        self._k25_captured = {}
        self._k25_errors = []
        self._k25_positions = None
        return payload

    def k25_remove_hooks(self) -> int:
        n = len(getattr(self, "_k25_handles", []))
        for h in getattr(self, "_k25_handles", []):
            h.remove()
        self._k25_handles = []
        return n
