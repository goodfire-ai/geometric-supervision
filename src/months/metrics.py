"""Geometry summaries of projected activations."""
from __future__ import annotations

import numpy as np

from common.geometry import procrustes_r2


def ring_r2(points: np.ndarray, values: np.ndarray, period: int) -> float | None:
    """Fraction of the points' variance explained by the ideal ring (Procrustes R^2).

    The ideal ring places value v at angle 2*pi*v/period; see
    `common.geometry.procrustes_r2` for the fit.
    """
    angle = 2.0 * np.pi * np.asarray(values, dtype=np.float64) / period
    return procrustes_r2(points, np.stack([np.cos(angle), np.sin(angle)], axis=1))


def class_separation(points: np.ndarray, values: np.ndarray, period: int,
                     min_classes: int = 2) -> float | None:
    """Between-class over within-class variance of value mod period (count weighted)."""
    labels = np.remainder(np.asarray(values, dtype=np.int64), period)
    classes = [label for label in np.unique(labels) if (labels == label).sum() >= 2]
    if len(classes) < min_classes:
        return None
    points = np.asarray(points, dtype=np.float64)
    grand_mean = points.mean(axis=0)
    between = within = 0.0
    n_total = 0
    for label in classes:
        subset = points[labels == label]
        centroid = subset.mean(axis=0)
        between += len(subset) * ((centroid - grand_mean) ** 2).sum()
        within += ((subset - centroid) ** 2).sum()
        n_total += len(subset)
    if within <= 0:
        return None
    return float((between / n_total) / (within / n_total))


def within_value_spread(points: np.ndarray, values: np.ndarray, period: int | None = None) -> float:
    """RMS distance of points to their value's centroid (sigma in the paper).

    With `period` given, values are taken modulo the period first.
    """
    labels = np.asarray(values)
    if period is not None:
        labels = np.remainder(labels, period)
    points = np.asarray(points, dtype=np.float64)
    squared = []
    for label in np.unique(labels):
        subset = points[labels == label]
        squared.append(((subset - subset.mean(axis=0)) ** 2).sum(axis=1))
    return float(np.sqrt(np.concatenate(squared).mean()))
