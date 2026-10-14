"""Activation patching between a trained model and the base model.

The question these tools answer: if the base model's activations are replaced,
inside a chosen subspace, by a trained model's activations on the same prompt,
does the base model inherit the trained model's accuracy?

    single-layer patch    at one layer, at the answer position, replace the
                          component of the hidden state inside subspace Q:
                              h <- h - Q Q^T h + Q Q^T h_donor
    live multi-layer patch the same replacement applied at several layers (and
                          optionally every position), on the live hidden state,
                          so patches at earlier layers propagate forward

Subspaces: the span of a set of probe planes, a random subspace of the same
rank (control), or the top directions along which the two models differ.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .data import band_summary
from common.model import decoder_layers
from common.lora import attach_lora

BATCH = 16


def random_subspace(hidden: int, rank: int, seed: int) -> torch.Tensor:
    """Orthonormal basis [hidden, rank] of a random subspace."""
    generator = torch.Generator().manual_seed(seed)
    gaussian = torch.randn(hidden, rank, generator=generator, dtype=torch.float32)
    q, _ = torch.linalg.qr(gaussian)
    return q[:, :rank]


def difference_subspace(base_states: torch.Tensor, donor_states: torch.Tensor, rank: int,
                        centered: bool = True) -> torch.Tensor:
    """Top-`rank` right singular directions of donor - base over prompts.

    Centered (default): the mean difference is removed first, so the directions
    describe how the change varies across prompts. Uncentered: the constant
    shift can appear as the first direction.
    """
    difference = (donor_states - base_states).double()
    if centered:
        difference = difference - difference.mean(dim=0, keepdim=True)
    _, _, vt = torch.linalg.svd(difference, full_matrices=False)
    return vt[:rank].T.float()


class SitePatch:
    """Read, and optionally overwrite, the answer-position activation at one (layer, site)."""

    def __init__(self, model: Any, layer: int, site: str):
        self.blocks = decoder_layers(model)
        self.layer, self.site = layer, site
        self.captured: torch.Tensor | None = None
        self.replacement: torch.Tensor | None = None
        self._handle = None

    def __enter__(self):
        block = self.blocks[self.layer]
        if self.site == "mlp_input":
            def pre_hook(module, args):
                hidden = args[0]
                self.captured = hidden[:, -1, :].detach().float().cpu()
                if self.replacement is None:
                    return None
                hidden = hidden.clone()
                hidden[:, -1, :] = self.replacement.to(hidden.device, hidden.dtype)
                return (hidden,) + tuple(args[1:])
            self._handle = block.mlp.register_forward_pre_hook(pre_hook)
        elif self.site == "resid":
            def hook(module, inputs, output):
                hidden = output[0] if isinstance(output, tuple) else output
                self.captured = hidden[:, -1, :].detach().float().cpu()
                if self.replacement is None:
                    return None
                hidden = hidden.clone()
                hidden[:, -1, :] = self.replacement.to(hidden.device, hidden.dtype)
                return (hidden,) + tuple(output[1:]) if isinstance(output, tuple) else hidden
            self._handle = block.register_forward_hook(hook)
        else:
            raise ValueError(f"unknown site {self.site!r}")
        return self

    def __exit__(self, *exc_info):
        self._handle.remove()


@torch.no_grad()
def score_with_single_layer_patch(model: Any, tokenizer: Any, prompts: list[str],
                                  answers: np.ndarray, sum_values: np.ndarray,
                                  month_ids: list[int], layer: int, site: str,
                                  adapter: tuple[dict, float] | None = None,
                                  donor_coordinates: torch.Tensor | None = None,
                                  basis: torch.Tensor | None = None) -> tuple[dict, torch.Tensor, dict]:
    """Score the model (plus adapter) with an optional subspace patch at one layer.

    If `donor_coordinates` [n_prompts, rank] and `basis` [hidden, rank] are given,
    the component of the answer-position activation inside the basis is replaced
    by those coordinates:  h <- h - Q Q^T h + coordinates @ Q^T.
    Returns (band accuracies, captured activations [n_prompts, hidden], per-sum accuracy).
    """
    month_ids_tensor = torch.tensor(month_ids, device=model.device)
    predictions, captured = [], []
    remove = attach_lora(model, adapter[0], adapter[1]) if adapter else None
    try:
        with SitePatch(model, layer, site) as patch:
            for start in range(0, len(prompts), BATCH):
                encoded = tokenizer(prompts[start:start + BATCH], return_tensors="pt",
                                    padding=True).to(model.device)
                n = encoded.input_ids.shape[0]
                if donor_coordinates is not None:
                    patch.replacement = None
                    model(**encoded)                       # read this model's own activation
                    own = patch.captured
                    coordinates = donor_coordinates[start:start + n]
                    patch.replacement = own - (own @ basis) @ basis.T + coordinates @ basis.T
                logits = model(**encoded).logits[:, -1, :].float()
                patch.replacement = None
                captured.append(patch.captured.clone())
                predictions.append(logits.index_select(1, month_ids_tensor).argmax(dim=1).cpu().numpy())
    finally:
        if remove is not None:
            remove()
    predictions = np.concatenate(predictions)
    bands = band_summary(predictions, answers, sum_values)["bands"]
    hit = predictions == answers
    per_sum = {int(value): float(hit[sum_values == value].mean()) for value in np.unique(sum_values)}
    accuracies = {name: bands[name]["accuracy"] for name in ("2-96", "2-24", "25-48", "49-96") if name in bands}
    return accuracies, torch.cat(captured), per_sum


class LivePatch:
    """Replace the in-subspace part of the live hidden state at several layers."""

    def __init__(self, model: Any, layers: list[int], positions: str = "all"):
        if positions not in ("all", "last"):
            raise ValueError("positions must be 'all' or 'last'")
        self.blocks = decoder_layers(model)
        self.layers = layers
        self.positions = positions
        self.captured: dict[int, torch.Tensor] = {}
        self.donor: dict[int, torch.Tensor] = {}
        self.bases: dict[int, torch.Tensor] = {}
        self.active = False
        self._handles: list = []

    def __enter__(self):
        for layer in self.layers:
            def hook(module, inputs, output, layer=layer):
                hidden = output[0] if isinstance(output, tuple) else output
                kept = hidden if self.positions == "all" else hidden[:, -1:, :]
                self.captured[layer] = kept.detach().float().cpu()
                if not (self.active and layer in self.bases):
                    return None
                basis = self.bases[layer].to(hidden.device)
                donor = self.donor[layer].to(hidden.device)
                if self.positions == "all":
                    current = hidden.float()
                    patched = current - (current @ basis) @ basis.T + (donor @ basis) @ basis.T
                    hidden = patched.to(hidden.dtype)
                else:
                    current = hidden[:, -1, :].float()
                    patched = current - (current @ basis) @ basis.T + (donor[:, 0, :] @ basis) @ basis.T
                    hidden = hidden.clone()
                    hidden[:, -1, :] = patched.to(hidden.dtype)
                return (hidden,) + tuple(output[1:]) if isinstance(output, tuple) else hidden
            self._handles.append(self.blocks[layer].register_forward_hook(hook))
        return self

    def __exit__(self, *exc_info):
        for handle in self._handles:
            handle.remove()


@torch.no_grad()
def score_with_live_patch(model: Any, tokenizer: Any, prompts: list[str], answers: np.ndarray,
                          sum_values: np.ndarray, month_ids: list[int], layers: list[int],
                          positions: str = "all", adapter: tuple[dict, float] | None = None,
                          bases: dict[int, torch.Tensor] | None = None,
                          donor_adapter: tuple[dict, float] | None = None,
                          collect_last: bool = False) -> tuple[dict, dict | None, dict]:
    """Score model (+adapter); with `bases`, patch each layer's subspace from a donor pass.

    For each batch the donor (model + donor_adapter, or the base model when
    donor_adapter is None) is run first to record its hidden states, then the
    scored model is run with the live patch active.
    """
    month_ids_tensor = torch.tensor(month_ids, device=model.device)
    predictions = []
    last_states = {layer: [] for layer in layers}
    with LivePatch(model, layers, positions) as patch:
        if bases is not None:
            patch.bases = {layer: bases[layer].float() for layer in layers}
        for start in range(0, len(prompts), BATCH):
            encoded = tokenizer(prompts[start:start + BATCH], return_tensors="pt",
                                padding=True).to(model.device)
            patch.active = False
            if bases is not None:
                remove = attach_lora(model, donor_adapter[0], donor_adapter[1]) if donor_adapter else None
                try:
                    model(**encoded)
                finally:
                    if remove is not None:
                        remove()
                patch.donor = {layer: patch.captured[layer].clone() for layer in layers}
                patch.active = True
            remove = attach_lora(model, adapter[0], adapter[1]) if adapter else None
            try:
                logits = model(**encoded).logits[:, -1, :].float()
            finally:
                if remove is not None:
                    remove()
            patch.active = False
            if collect_last:
                for layer in layers:
                    last_states[layer].append(patch.captured[layer][:, -1, :].clone())
            predictions.append(logits.index_select(1, month_ids_tensor).argmax(dim=1).cpu().numpy())
    predictions = np.concatenate(predictions)
    bands = band_summary(predictions, answers, sum_values)["bands"]
    hit = predictions == answers
    accuracies = {name: bands[name]["accuracy"] for name in ("2-96", "2-24", "25-48", "49-96")}
    per_sum = {int(value): float(hit[sum_values == value].mean()) for value in np.unique(sum_values)}
    collected = {layer: torch.cat(states) for layer, states in last_states.items()} if collect_last else None
    return accuracies, collected, per_sum
