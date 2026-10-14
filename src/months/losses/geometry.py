"""The geometry loss: pull per-value centroids to an ideal ring, keep the spread.

Notation used throughout (see the paper's method section):

    points          projected activations on one 2-D probe plane, shape [N, 2]
    values          the task variable's value for each point, shape [N]
    centroids       the mean point of each value
    ideal ring      value v sits at angle 2*pi*v/period on a circle
    plane record    the base model's geometry on this plane: its centroids,
                    centroid radius r0, within-value spread sigma0, and the
                    Procrustes frame (center, orientation) that places the
                    ideal ring over the base model's centroids

The loss has two terms:

    mean term       mean over values of ||centroid_v - target_v||^2 / r0^2
    spread term     (sigma / sigma0 - spread_ratio)^2

    loss = mean term + spread_weight * spread term

With target_frame="fixed" the targets are computed once from the base model's
frame and never move. With target_frame="dynamic" the ideal ring is re-aligned
to the current centroids at every step by orthogonal Procrustes (rotation or
reflection only; no scaling). Targets are always detached.
"""
from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor

from common.geometry import (centroid_term, class_centroids, frozen_targets, spread_term,
                             unit_pattern, within_class_spread)


def _tensor_field(record: dict, name: str, like: Tensor) -> Tensor:
    return torch.as_tensor(record[name], device=like.device, dtype=like.dtype)


def ideal_ring(classes: Any, period: int, device=None, dtype=torch.float64) -> Tensor:
    """Unit-circle coordinates of each class value at angle 2*pi*(value mod period)/period."""
    class_values = torch.as_tensor(classes, device=device, dtype=torch.float64)
    angle = 2.0 * math.pi * torch.remainder(class_values, period) / period
    return torch.stack((torch.cos(angle), torch.sin(angle)), dim=1).to(dtype)


def centered_unit_ring(classes: Any, period: int, device=None, dtype=torch.float64) -> Tensor:
    """The ideal ring, centered and scaled to RMS radius 1 over the given classes.

    The angles are computed in float64 and the ring is cast to `dtype` before centering.
    """
    return unit_pattern(ideal_ring(classes, period, device=device, dtype=dtype))


def procrustes_targets(centroids: Tensor, classes: Any, period: int,
                       radius: Tensor) -> tuple[Tensor, Tensor]:
    """Place the ideal ring over the current centroids by orthogonal Procrustes.

    Fits a rotation or reflection (no scaling) of the centered unit ring onto
    the centered centroids, then scales the ring to `radius` and re-centers it
    at the centroids' mean. Computed without gradients; targets are detached.
    Returns (targets [K, 2], orientation [2, 2]).
    """
    with torch.no_grad():
        observed = centroids.detach().to(torch.float64)
        pattern = centered_unit_ring(classes, period, device=observed.device)
        observed_centered = observed - observed.mean(dim=0, keepdim=True)
        u, _, vh = torch.linalg.svd(pattern.T @ observed_centered)
        orientation = u @ vh
        targets = observed.mean(dim=0, keepdim=True) + radius.to(torch.float64) * (pattern @ orientation)
    return targets.to(centroids.dtype), orientation.to(centroids.dtype)


def fixed_targets(record: dict, classes: Any, period: int, radius: Tensor,
                  like: Tensor) -> Tensor:
    """Targets in the base model's frozen frame: center + (radius * unit ring) @ orientation."""
    center = _tensor_field(record, "base_center", like)
    if "target_unit" in record:   # a stored target shape instead of the ring (Appendix A.3.3)
        return center[None, :] + radius * _tensor_field(record, "target_unit", like)
    orientation = _tensor_field(record, "base_orientation", like)
    pattern = centered_unit_ring(classes, period, device=like.device, dtype=like.dtype)
    return frozen_targets(center, orientation, radius, pattern)


def geometry_loss(points: Tensor, values: Any, period: int, record: dict,
                  radius_scale: float, spread_ratio: float, spread_weight: float,
                  target_frame: str = "fixed", shuffled_targets: bool = False
                  ) -> tuple[Tensor, dict[str, float]]:
    """Per-value centroid matching plus the spread term (the paper's Eq. for L_geometry).

    radius_scale     target centroid radius as a multiple of the base model's r0
    spread_ratio     target within-value spread as a multiple of the base model's sigma0
    spread_weight    lambda, the weight of the spread term
    target_frame     "fixed" (base model's frame) or "dynamic" (re-fit each step)
    shuffled_targets control: permute which value goes to which ring position,
                     using the permutation stored in the record
    """
    classes = record["classes"]
    centroids, class_index, _ = class_centroids(points, values, classes)
    base_radius = _tensor_field(record, "base_centroid_radius", points).reshape(())
    base_spread = _tensor_field(record, "base_within_rms", points).reshape(())
    target_radius = base_radius * radius_scale

    if target_frame == "dynamic":
        targets, orientation = procrustes_targets(centroids, classes, period, target_radius)
    elif target_frame == "fixed":
        targets = fixed_targets(record, classes, period, target_radius, points)
        orientation = _tensor_field(record, "base_orientation", points)
    else:
        raise ValueError("target_frame must be 'fixed' or 'dynamic'")

    if shuffled_targets:
        permutation = torch.as_tensor(record["shuffle_mapping"], device=points.device,
                                      dtype=torch.int64)
        targets = targets[permutation]

    mean_term = centroid_term(centroids, targets, base_radius)
    spread_penalty, spread = spread_term(points, class_index, centroids, base_spread, spread_ratio, "pooled")
    loss = mean_term + spread_weight * spread_penalty

    with torch.no_grad():
        centered = centroids - centroids.mean(dim=0, keepdim=True)
        current_radius = torch.sqrt(centered.square().sum(dim=1).mean())
    diagnostics = {
        "mean_term": float(mean_term.detach()),
        "spread_term": float(spread_penalty.detach()),
        "centroid_radius_ratio": float(current_radius / base_radius),
        "spread_ratio": float(spread.detach() / base_spread),
        "orientation_determinant": float(torch.linalg.det(orientation.detach().to(torch.float64))),
    }
    if not all(math.isfinite(value) for value in diagnostics.values()):
        raise FloatingPointError("geometry loss diagnostics are not finite")
    return loss, diagnostics
