"""The geometry objective for the linebreak task, built from ``common.geometry``.

Both terms are computed on the 3-D helix readout of the supervised layer, over the
tokens of the current batch, grouped by character count:

    L_mean   = mean_v || mu_hat_v - mu_v ||^2 / r0^2
    L_spread = mean_v ( sigma_v / sigma0 - tau )^2         (one spread per count)
    L        = L_mean + lambda * L_spread

The targets ``mu_v`` are fixed once, before training (``freeze_targets``): the ideal
helix is rotated onto the base model's per-count centroids by orthogonal Procrustes and
scaled to RMS radius ``alpha_r * r0``, where ``r0`` is the base centroids' RMS radius.
``sigma0`` is the base model's pooled within-count spread, measured at the same time.
"""
from __future__ import annotations

import numpy as np
import torch

from common.geometry import (centroid_term, class_geometry, frozen_targets, procrustes_frame,
                             spread_term, unit_pattern)


def freeze_targets(points: np.ndarray, counts: np.ndarray, lookup: torch.Tensor, alpha_r: float,
                   literal: bool = False) -> dict:
    """Targets and base statistics from the base model's readout of some tokens.

    ``points [N, k]`` and ``counts [N]`` are the base model's projected tokens;
    ``lookup [max_count + 1, k]`` holds the ideal point of every count. Returns the
    per-count ``targets`` (same shape as ``lookup``), a ``has_target`` mask over counts,
    and the scalars ``r0`` and ``sigma0``. With ``literal`` the targets are the lookup itself
    (no Procrustes, centering or rescaling) and the loss is per token (direct regression).
    """
    # drop counts whose ideal point is undefined (the empirical target has NaN rows)
    defined = ~torch.isnan(lookup).any(dim=1).cpu().numpy()
    keep = defined[np.asarray(counts)]
    points, counts = points[keep], counts[keep]
    geometry = class_geometry(points, counts)
    classes = geometry["classes"]
    ideal = lookup[torch.as_tensor(classes)].cpu()
    frame = procrustes_frame(geometry["centroids"], ideal.double().numpy())
    like = dict(device=lookup.device, dtype=torch.float32)
    r0 = torch.tensor(geometry["centroid_radius"], **like)
    if literal:
        rows = ideal.to(**like)
    else:
        rows = frozen_targets(torch.tensor(frame["center"], **like), torch.tensor(frame["orientation"], **like),
                              alpha_r * r0, unit_pattern(ideal.to(**like)))
    targets = torch.zeros_like(lookup, dtype=torch.float32)
    has_target = torch.zeros(len(lookup), dtype=torch.bool, device=lookup.device)
    targets[torch.as_tensor(classes)] = rows
    has_target[torch.as_tensor(classes)] = True
    return dict(targets=targets, has_target=has_target, r0=r0, per_token=literal,
                sigma0=torch.tensor(geometry["within_rms"], **like))


def geometry_loss(points: torch.Tensor, counts: torch.Tensor, frozen: dict, tau: float,
                  spread_weight: float) -> torch.Tensor:
    """L_mean + spread_weight * L_spread over the batch's tokens (``points [M, k]``, ``counts [M]``)."""
    keep = frozen["has_target"][counts]
    points, counts = points[keep], counts[keep]
    if frozen.get("per_token"):   # direct regression: every token to its own target
        return (points - frozen["targets"][counts]).square().sum(dim=1).mean() / frozen["r0"].square()
    classes, class_index = torch.unique(counts, return_inverse=True)
    sizes = points.new_zeros(len(classes)).index_add_(0, class_index, points.new_ones(len(points)))
    centroids = points.new_zeros((len(classes), points.shape[1])).index_add_(0, class_index, points)
    centroids = centroids / sizes[:, None]
    loss = centroid_term(centroids, frozen["targets"][classes], frozen["r0"])
    if spread_weight > 0:
        penalty, _ = spread_term(points, class_index, centroids, frozen["sigma0"], tau, "per_value")
        loss = loss + spread_weight * penalty
    return loss
