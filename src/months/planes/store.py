"""Reading plane records back, and taking the joint subspace of several planes."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

_CACHE: dict[tuple[str, int, str], dict] = {}


def record_key(carrier: str, period: int) -> str:
    return f"{carrier}__p{period}"


def plane_records_path(root: Path, layer: int, site: str) -> Path:
    return Path(root) / f"L{layer}__{site}.pt"


def load_plane_records(root: Path, layer: int, site: str) -> dict:
    """All plane records at one (layer, site): {"<carrier>__p<period>": record}.

    Cached in memory, since a training worker reads the same file for every seed.
    Extra keys written by older versions of the pipeline are ignored by callers.
    """
    key = (str(root), layer, site)
    if key not in _CACHE:
        path = plane_records_path(root, layer, site)
        if not path.is_file():
            raise FileNotFoundError(f"no plane records at {path}; run the build-planes script")
        _CACHE[key] = torch.load(path, map_location="cpu", weights_only=False)
    return _CACHE[key]


def plane_subspace(records: dict, carriers: list[str], periods: list[int]
                   ) -> tuple[torch.Tensor, list[str]]:
    """Orthonormal basis Q [hidden, rank] of the span of the requested planes.

    The planes share directions, so the union is taken with one SVD of the
    stacked probe weights rather than plane by plane. Returns Q and the record
    keys that were found.
    """
    rows, used = [], []
    for carrier in carriers:
        for period in periods:
            key = record_key(carrier, period)
            if key in records:
                rows.append(np.asarray(records[key]["weight"], dtype=np.float64))
                used.append(key)
    if not rows:
        raise KeyError(f"no plane records for carriers {carriers} and periods {periods}")
    stacked = np.concatenate(rows, axis=0)
    _, singular_values, vt = np.linalg.svd(stacked, full_matrices=False)
    rank = int((singular_values > singular_values[0] * 1e-6).sum())
    basis = torch.as_tensor(vt[:rank].T.copy(), dtype=torch.float32)
    return basis, used
