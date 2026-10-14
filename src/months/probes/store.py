"""Reading and writing Fourier probes.

Probes written by `probes/fit.py` are stored one file per (layer, site):

    <root>/L<layer>__<site>.pt
        {"<carrier>__p<period>": {"weight": [2, hidden] (cos row, sin row),
                                  "bias": [2], "r2": {"cos": float, "sin": float}}}

Callers ask `store.plane(carrier, site, layer, period)` and get back
(weight [2, hidden], bias [2]) as float32 numpy arrays, or None if that probe
does not exist.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch


class ProbeStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(f"probe root {self.root} does not exist; run scripts/months/02_fit_probes.py")
        self._layer_cache: dict[tuple[int, str], dict] = {}

    def _layer_probes(self, layer: int, site: str) -> dict:
        key = (layer, site)
        if key not in self._layer_cache:
            path = self.root / f"L{layer}__{site}.pt"
            if path.is_file():
                self._layer_cache[key] = torch.load(path, map_location="cpu", weights_only=False)
            else:
                self._layer_cache[key] = {}
        return self._layer_cache[key]

    def plane(self, carrier: str, site: str, layer: int, period: int):
        """(weight [2, hidden], bias [2]) with the cosine row first, or None."""
        entry = self._layer_probes(layer, site).get(f"{carrier}__p{period}")
        if entry is None:
            return None
        return (np.asarray(entry["weight"], dtype=np.float32),
                np.asarray(entry["bias"], dtype=np.float32))

    def available_periods(self, carrier: str, site: str, layer: int) -> list[int]:
        prefix = f"{carrier}__p"
        return sorted(int(key[len(prefix):]) for key in self._layer_probes(layer, site)
                      if key.startswith(prefix))

    def fingerprint(self) -> str:
        """SHA-256 over (file name, file hash) of every probe file, for provenance."""
        digest = hashlib.sha256()
        for path in sorted(self.root.glob("*.pt")):
            digest.update(path.name.encode())
            digest.update(hashlib.sha256(path.read_bytes()).hexdigest().encode())
        return digest.hexdigest()

    def describe(self) -> dict:
        return {"fingerprint": self.fingerprint()}


def save_layer_probes(root: Path, layer: int, site: str, planes: dict, report: dict | None = None) -> Path:
    """Write repo-layout probes for one (layer, site).

    `planes` maps "<carrier>__p<period>" to {"weight": [2, hidden], "bias": [2], "r2": {...}}.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"L{layer}__{site}.pt"
    payload = {key: {"weight": torch.as_tensor(np.asarray(entry["weight"], dtype=np.float32)),
                     "bias": torch.as_tensor(np.asarray(entry["bias"], dtype=np.float32)),
                     "r2": entry.get("r2")}
               for key, entry in planes.items()}
    torch.save(payload, path)
    if report is not None:
        (root / f"L{layer}__{site}.report.json").write_text(json.dumps(report, indent=1))
    return path
