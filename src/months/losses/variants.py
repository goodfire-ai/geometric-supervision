"""Alternatives to the per-value geometry loss, used in the appendix ablation.

    per_value               the paper's loss: centroids pulled to their targets
                            (this is `geometry_loss`)
    per_prompt              every projected point pulled to its value's target,
                            with no spread term
    per_prompt_spread       per_prompt plus the same spread term as per_value

The per-prompt loss decomposes exactly into a size-weighted centroid term plus
the within-value variance, so on its own it drives the spread to zero. The
spread term counteracts that, which is what the ablation shows.

Both per-prompt variants use the fixed frame regardless of `target_frame`.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from .geometry import (class_centroids, fixed_targets, geometry_loss,
                       within_class_spread, _tensor_field)


def _per_prompt_terms(points: Tensor, values: Any, period: int, record: dict,
                      radius_scale: float, shuffled_targets: bool):
    classes = record["classes"]
    centroids, class_index, _ = class_centroids(points, values, classes)
    base_radius = _tensor_field(record, "base_centroid_radius", points).reshape(())
    base_spread = _tensor_field(record, "base_within_rms", points).reshape(())
    targets = fixed_targets(record, classes, period, base_radius * radius_scale, points)
    if shuffled_targets:
        permutation = torch.as_tensor(record["shuffle_mapping"], device=points.device,
                                      dtype=torch.int64)
        targets = targets[permutation]
    residual = points - targets.detach()[class_index]
    point_term = residual.square().sum(dim=1).mean() / base_radius.square()
    spread = within_class_spread(points, class_index, centroids)
    return point_term, spread, base_spread, centroids, base_radius


def _diagnostics(point_term, spread, base_spread, centroids, base_radius, spread_term=None):
    with torch.no_grad():
        centered = centroids - centroids.mean(dim=0, keepdim=True)
        current_radius = torch.sqrt(centered.square().sum(dim=1).mean())
    out = {
        "mean_term": float(point_term.detach()),
        "centroid_radius_ratio": float(current_radius / base_radius),
        "spread_ratio": float(spread.detach() / base_spread),
    }
    if spread_term is not None:
        out["spread_term"] = float(spread_term.detach())
    return out


def per_prompt_loss(points: Tensor, values: Any, period: int, record: dict,
                    radius_scale: float, spread_ratio: float, spread_weight: float,
                    target_frame: str = "fixed", shuffled_targets: bool = False
                    ) -> tuple[Tensor, dict[str, float]]:
    """Mean over prompts of ||point - target_of_its_value||^2 / r0^2. No spread term."""
    point_term, spread, base_spread, centroids, base_radius = _per_prompt_terms(
        points, values, period, record, radius_scale, shuffled_targets)
    return point_term, _diagnostics(point_term, spread, base_spread, centroids, base_radius)


def per_prompt_loss_with_spread(points: Tensor, values: Any, period: int, record: dict,
                                radius_scale: float, spread_ratio: float, spread_weight: float,
                                target_frame: str = "fixed", shuffled_targets: bool = False
                                ) -> tuple[Tensor, dict[str, float]]:
    """per_prompt_loss plus spread_weight * (sigma / sigma0 - spread_ratio)^2."""
    point_term, spread, base_spread, centroids, base_radius = _per_prompt_terms(
        points, values, period, record, radius_scale, shuffled_targets)
    spread_term = (spread - base_spread * spread_ratio).square() / base_spread.square()
    loss = point_term + spread_weight * spread_term
    return loss, _diagnostics(point_term, spread, base_spread, centroids, base_radius, spread_term)


LOSS_FUNCTIONS = {
    "per_value": geometry_loss,
    "per_prompt": per_prompt_loss,
    "per_prompt_spread": per_prompt_loss_with_spread,
}
