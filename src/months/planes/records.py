"""Building plane records.

A plane record is what the geometry loss needs to know about one probe plane:

    weight, bias            the affine probe, so points = activations @ weight.T + bias
    period                  the Fourier period T of the plane
    classes                 the task-variable values present in the training prompts
    class_centroids         the base model's mean point for each class
    base_centroid_radius    RMS distance of those centroids from their mean (r0)
    base_within_rms         RMS distance of base-model points to their centroid (sigma0)
    base_center             the centroids' mean
    base_orientation        the 2x2 rotation/reflection that places the ideal ring
                            over the base model's centroids (orthogonal Procrustes)
    orientation_determinant +1 for a rotation, -1 for a reflection
    shuffle_mapping         a fixed permutation of class rows, for the shuffled control

Everything about the base model is measured once, on the 210 training prompts,
and frozen. Statistics are computed in float64 from float32 projections.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch

from common.geometry import class_geometry
from common.geometry import procrustes_frame as common_procrustes_frame

# Seed of the permutation used by the shuffled-targets control. Fixed so the
# control is the same permutation everywhere.
SHUFFLE_SEED = 167000


def procrustes_frame(centroids: np.ndarray, classes: np.ndarray, period: int) -> dict:
    """Rotation or reflection that best aligns the ideal ring with the centroids.

    The ideal ring puts class value v at angle 2*pi*(v mod period)/period. Both
    the ring and the centroids are centered; the fit has no scaling.
    """
    centroids = np.asarray(centroids, dtype=np.float64)
    classes = np.asarray(classes, dtype=np.float64)
    angle = 2 * np.pi * (classes % period) / period
    ideal = np.stack((np.cos(angle), np.sin(angle)), axis=1)
    if (period > 2 and np.linalg.matrix_rank(ideal - ideal.mean(axis=0)) < 2) or \
            np.linalg.matrix_rank(centroids - centroids.mean(axis=0)) < 1:
        raise ValueError("Procrustes inputs have insufficient rank")
    return common_procrustes_frame(centroids, ideal)


def shuffled_class_rows(classes: np.ndarray, seed: int) -> np.ndarray:
    """Row index each class is sent to under a fixed random permutation of classes."""
    values = np.asarray(sorted(int(value) for value in classes), dtype=np.int64)
    shuffled = np.random.default_rng(seed).permutation(values)
    row_of_class = {int(value): index for index, value in enumerate(values)}
    return np.asarray([row_of_class[int(target)] for target in shuffled], dtype=np.int64)


def build_plane_record(weight: np.ndarray, bias: np.ndarray, points: np.ndarray,
                       values: np.ndarray, period: int,
                       shuffle_seed: int = SHUFFLE_SEED) -> dict:
    """Assemble one plane record (CPU tensors) from a probe and the base model's points."""
    weight = np.asarray(weight, dtype=np.float32)
    bias = np.asarray(bias, dtype=np.float32)
    if weight.ndim != 2 or weight.shape[0] != 2 or bias.shape != (2,):
        raise ValueError("expected weight [2, hidden] and bias [2]")
    if not (np.isfinite(weight).all() and np.isfinite(bias).all()):
        raise ValueError("probe weights must be finite")
    geometry = class_geometry(points, values)
    frame = procrustes_frame(geometry["centroids"], geometry["classes"], period)
    return {
        "weight": torch.from_numpy(weight.copy()),
        "bias": torch.from_numpy(bias.copy()),
        "period": int(period),
        "classes": torch.from_numpy(geometry["classes"].astype(np.int64)),
        "class_centroids": torch.from_numpy(geometry["centroids"].astype(np.float32)),
        "base_centroid_radius": float(geometry["centroid_radius"]),
        "base_within_rms": float(geometry["within_rms"]),
        "base_center": torch.from_numpy(frame["center"].astype(np.float32)),
        "base_orientation": torch.from_numpy(frame["orientation"].astype(np.float32)),
        "orientation_determinant": frame["determinant"],
        "shuffle_mapping": torch.from_numpy(shuffled_class_rows(geometry["classes"], shuffle_seed)),
    }


def save_plane_records(path: Path, records: dict) -> str:
    """Write a dict of records with torch.save and a .sha256 sidecar. Returns the digest."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(records, temporary)
    digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
    temporary.replace(path)
    path.with_suffix(path.suffix + ".sha256").write_text(f"{digest}  {path.name}\n")
    return digest
