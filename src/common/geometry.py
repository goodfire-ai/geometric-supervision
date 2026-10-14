"""The geometry objective shared by both tasks, and the Procrustes R^2 used to measure it.

Notation (paper Section 2): a readout maps each activation to a point in R^k; points
are grouped by the value v of the task variable; mu_hat_v are the per-value centroids;
mu_v are the fixed targets.

Fitting the targets, once, on the base model:

    class_geometry      centroids, their RMS radius r0 around their mean, and the
                        pooled within-value RMS spread sigma0
    procrustes_frame    the rotation or reflection that best aligns the (centered)
                        ideal shape with the (centered) base-model centroids
    unit_pattern        the ideal shape centered and scaled to RMS radius 1
    frozen_targets      mu_v = center + (alpha_r * r0 * pattern_v) @ orientation

The loss, at every training step:

    centroid_term       mean_v || mu_hat_v - mu_v ||^2 / r0^2
    spread_term         "pooled":    (sigma / sigma0 - tau)^2, sigma pooled over all points
                        "per_value": mean_v (sigma_v / sigma0 - tau)^2, one sigma per value
    loss                centroid_term + lambda * spread_term

The months task uses the pooled spread (paper Eq. 5-6); the linebreak task holds each
character count's spread separately.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# fitting the targets on the base model (numpy, float64)
# ---------------------------------------------------------------------------

def class_geometry(points: np.ndarray, values: np.ndarray) -> dict:
    """Per-value centroids, their RMS radius r0, and the pooled within-value RMS sigma0."""
    points = np.asarray(points, dtype=np.float64)
    values = np.asarray(values)
    if points.ndim != 2 or values.shape != (len(points),):
        raise ValueError("expected points [N, k] and values [N]")
    classes = np.unique(values)
    if len(classes) < 3:
        raise ValueError("need at least three values")
    centroids = np.stack([points[values == value].mean(axis=0) for value in classes])
    center = centroids.mean(axis=0)
    centroid_radius = math.sqrt(float(np.mean(np.sum((centroids - center) ** 2, axis=1))))
    within_sum_of_squares = sum(
        float(np.sum((points[values == value] - centroids[i]) ** 2))
        for i, value in enumerate(classes))
    return {
        "classes": classes,
        "centroids": centroids,
        "center": center,
        "centroid_radius": centroid_radius,
        "within_rms": math.sqrt(within_sum_of_squares / len(points)),
    }


def procrustes_frame(centroids: np.ndarray, ideal: np.ndarray) -> dict:
    """Rotation or reflection (no scaling) that best maps the centered ideal onto the
    centered centroids: minimizes ||(ideal - mean) @ Q - (centroids - center)||."""
    centroids = np.asarray(centroids, dtype=np.float64)
    ideal = np.asarray(ideal, dtype=np.float64)
    center = centroids.mean(axis=0)
    ideal = ideal - ideal.mean(axis=0)
    observed = centroids - center
    u, _, vt = np.linalg.svd(ideal.T @ observed)
    orientation = u @ vt
    return {"center": center, "orientation": orientation,
            "determinant": float(np.linalg.det(orientation))}


def unit_pattern(ideal: Tensor) -> Tensor:
    """The ideal points centered and scaled to RMS radius 1."""
    centered = ideal - ideal.mean(dim=0, keepdim=True)
    rms_radius = torch.sqrt(centered.square().sum(dim=1).mean())
    return centered / rms_radius


def frozen_targets(center: Tensor, orientation: Tensor, radius: Tensor, pattern: Tensor) -> Tensor:
    """Targets in the base model's frame: center + (radius * unit pattern) @ orientation."""
    return center[None, :] + (radius * pattern) @ orientation


# ---------------------------------------------------------------------------
# the loss (torch, differentiable)
# ---------------------------------------------------------------------------

def class_centroids(points: Tensor, values: Any, classes: Any) -> tuple[Tensor, Tensor, Tensor]:
    """Differentiable per-class means.

    Returns (centroids [K, k], class index of each point [N], counts [K]).
    `classes` fixes the row order and must contain exactly the values present.
    """
    if points.ndim != 2:
        raise ValueError("points must have shape [N, k]")
    labels = torch.as_tensor(values, device=points.device)
    class_values = torch.as_tensor(classes, device=points.device)
    matches = labels[:, None] == class_values[None, :]
    if not bool(matches.any(dim=1).all()):
        raise ValueError("every point's value must be one of the classes")
    class_index = matches.to(torch.int64).argmax(dim=1)
    counts = points.new_zeros(len(class_values)).index_add_(
        0, class_index, points.new_ones(len(class_index)))
    if not bool((counts > 0).all()):
        raise ValueError("every class must occur among the points")
    sums = points.new_zeros((len(class_values), points.shape[1])).index_add_(0, class_index, points)
    return sums / counts[:, None], class_index, counts


def within_class_spread(points: Tensor, class_index: Tensor, centroids: Tensor) -> Tensor:
    """Pooled RMS distance of each point to its own class centroid (sigma in the paper)."""
    residual = points - centroids[class_index]
    return torch.sqrt(residual.square().sum(dim=1).mean().clamp_min(0))


def per_class_spread(points: Tensor, class_index: Tensor, centroids: Tensor) -> Tensor:
    """RMS distance of each class's points to its centroid, one value per class [K]."""
    squared = (points - centroids[class_index]).square().sum(dim=1)
    totals = points.new_zeros(len(centroids)).index_add_(0, class_index, squared)
    counts = points.new_zeros(len(centroids)).index_add_(0, class_index, torch.ones_like(squared))
    return torch.sqrt((totals / counts).clamp_min(1e-12))


def centroid_term(centroids: Tensor, targets: Tensor, r0: Tensor) -> Tensor:
    """mean_v ||mu_hat_v - mu_v||^2 / r0^2 (targets are detached)."""
    return (centroids - targets.detach()).square().sum(dim=1).mean() / r0.square()


def spread_term(points: Tensor, class_index: Tensor, centroids: Tensor, sigma0: Tensor,
                tau: float, mode: str = "pooled") -> tuple[Tensor, Tensor]:
    """The spread penalty and the spread it measured (pooled sigma, or mean sigma_v)."""
    if mode == "pooled":
        spread = within_class_spread(points, class_index, centroids)
        return (spread - sigma0 * tau).square() / sigma0.square(), spread
    if mode == "per_value":
        spreads = per_class_spread(points, class_index, centroids)
        return (spreads / sigma0 - tau).square().mean(), spreads.mean()
    raise ValueError("spread mode must be 'pooled' or 'per_value'")


# ---------------------------------------------------------------------------
# measuring alignment (numpy)
# ---------------------------------------------------------------------------

def procrustes_r2(points: np.ndarray, ideal: np.ndarray) -> float | None:
    """Fraction of the points' variance explained by the ideal shape after the best
    similarity transform (rotation or reflection, one uniform scale, translation).

    Scale and translation do not matter, so a readout that only inflates the shape
    gains nothing; the points have to sit at the right positions in the right order.
    """
    observed = np.asarray(points, dtype=np.float64)
    observed = observed - observed.mean(axis=0)
    total = (observed ** 2).sum()
    if total <= 0:
        return None
    ideal = np.asarray(ideal, dtype=np.float64)
    ideal = ideal - ideal.mean(axis=0)
    ideal_energy = (ideal ** 2).sum()
    if ideal_energy <= 0:
        return None
    u, singular_values, vt = np.linalg.svd(ideal.T @ observed)
    scale = singular_values.sum() / ideal_energy
    fit = scale * (ideal @ (u @ vt))
    return float(1.0 - ((observed - fit) ** 2).sum() / total)
