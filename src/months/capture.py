"""Reading hidden states out of the model at the last prompt position.

Two capture sites are used:
    resid       the output of a transformer block (the residual stream)
    mlp_input   the input to a block's MLP (after its layer norm)

Prompts are left padded, so the last position is the answer position for every
prompt in a batch. The forward pass is cut off right after the deepest captured
layer, since nothing later is needed.
"""
from __future__ import annotations

from typing import Any

import torch

from common.model import decoder_layers

SITES = ("resid", "mlp_input")


class StopForward(Exception):
    """Raised inside a hook to end the forward pass early."""


class ActivationCapture:
    """Context manager that records last-token activations at (layer, site) pairs."""

    def __init__(self, model: Any, pairs: list[tuple[int, str]], keep_graph: bool = False):
        self.model = model
        self.pairs = sorted(set(pairs))
        self.keep_graph = keep_graph
        self.store: dict[tuple[int, str], torch.Tensor] = {}
        self._handles: list = []

    def _record(self, key: tuple[int, str], hidden: torch.Tensor) -> None:
        last = hidden[:, -1, :]
        self.store[key] = last if self.keep_graph else last.detach()

    def __enter__(self):
        layers = decoder_layers(self.model)
        for layer, site in self.pairs:
            key = (layer, site)
            if site == "mlp_input":
                def pre_hook(module, args, key=key):
                    self._record(key, args[0])
                self._handles.append(layers[layer].mlp.register_forward_pre_hook(pre_hook))
            elif site == "resid":
                def hook(module, inputs, output, key=key):
                    hidden = output[0] if isinstance(output, tuple) else output
                    self._record(key, hidden)
                self._handles.append(layers[layer].register_forward_hook(hook))
            else:
                raise ValueError(f"unknown site {site!r}; expected one of {SITES}")
        return self

    def __exit__(self, *exc_info):
        for handle in self._handles:
            handle.remove()
        self._handles = []
        return False


def _run_truncated(model: Any, encoded, deepest_layer: int) -> None:
    """Forward pass that stops after `deepest_layer`."""
    layers = decoder_layers(model)

    def stop(module, inputs, output):
        raise StopForward()

    handle = layers[deepest_layer].register_forward_hook(stop)
    try:
        model(input_ids=encoded.input_ids, attention_mask=encoded.attention_mask)
    except StopForward:
        pass
    finally:
        handle.remove()


def capture_with_grad(model: Any, tokenizer: Any, prompts: list[str], layer: int,
                      site: str) -> torch.Tensor:
    """Last-token activations for one batch of prompts, keeping the autograd graph.

    Used inside the geometry training step, where gradients must flow from the
    captured activations back to the adapter.
    """
    encoded = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    with ActivationCapture(model, [(layer, site)], keep_graph=True) as capture:
        _run_truncated(model, encoded, layer)
    return capture.store[(layer, site)]


@torch.no_grad()
def capture_many(model: Any, tokenizer: Any, prompts: list[str],
                 pairs: list[tuple[int, str]], batch_size: int = 64,
                 dtype: torch.dtype = torch.float32) -> dict[tuple[int, str], torch.Tensor]:
    """Last-token activations at every (layer, site) pair, for all prompts, on CPU.

    One pass over the prompts records every pair at once. Returns tensors of
    shape [n_prompts, hidden_size] in `dtype` (float32 by default; bfloat16 is
    lossless for storage because the model runs in bfloat16).
    """
    deepest = max(layer for layer, _ in pairs)
    chunks: dict[tuple[int, str], list[torch.Tensor]] = {pair: [] for pair in set(pairs)}
    with ActivationCapture(model, pairs) as capture:
        for start in range(0, len(prompts), batch_size):
            encoded = tokenizer(prompts[start:start + batch_size], return_tensors="pt",
                                padding=True).to(model.device)
            capture.store.clear()
            _run_truncated(model, encoded, deepest)
            missing = set(chunks) - set(capture.store)
            if missing:
                raise RuntimeError(f"capture missed {sorted(missing)}")
            for key, activation in capture.store.items():
                chunks[key].append(activation.to(dtype).cpu())
    return {key: torch.cat(parts, dim=0) for key, parts in chunks.items()}
