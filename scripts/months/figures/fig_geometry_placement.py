#!/usr/bin/env python3
"""Where to put the geometry LoRA: two heatmaps of overall accuracy (sums 2-96).

    --panel supervise   rank-1 adapters, x = edit layer 0..16, y = supervision layer 16..31
                        (only cells with supervision layer > edit layer exist)
    --panel rank        x = edit layer 0..16, y = rank 1, 2, 4, 8, 16, supervision layer 18

The canonical cell (edit layer 4, supervision layer 18, rank 1) is outlined in black.

Reads results/family_bands.json with ids geometry_e<edit>_s<supervise> (rank 1, family
"geometry_edit_supervise") and rank_r<rank>_e<edit>_s18 (ranks 2-16, family "geometry_rank").
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _heatmap import draw_accuracy_heatmap  # noqa: E402
from common import figstyle  # noqa: E402
from months.paths import RESULTS_DIR, FIGURES_DIR  # noqa: E402

EDIT_LAYERS = list(range(17))
SUPERVISE_LAYERS = list(range(16, 32))
RANKS = [1, 2, 4, 8, 16]
CANONICAL_EDIT, CANONICAL_SUPERVISE = 4, 18


def mean_accuracy(bands: dict, config_id: str, band: str) -> float:
    entry = bands.get(config_id)
    if entry and band in entry["bands"]:
        return entry["bands"][band]["mean"]
    return np.nan


def supervise_grid(bands: dict, band: str) -> np.ndarray:
    return np.array([[mean_accuracy(bands, f"geometry_e{edit:02d}_s{supervise:02d}", band)
                      for edit in EDIT_LAYERS] for supervise in SUPERVISE_LAYERS])


def rank_grid(bands: dict, band: str) -> np.ndarray:
    matrix = np.full((len(RANKS), len(EDIT_LAYERS)), np.nan)
    for i, rank in enumerate(RANKS):
        for j, edit in enumerate(EDIT_LAYERS):
            config_id = f"geometry_e{edit:02d}_s18" if rank == 1 else f"rank_r{rank:02d}_e{edit:02d}_s18"
            matrix[i, j] = mean_accuracy(bands, config_id, band)
    return matrix


def make(panel: str, bands_path: Path = RESULTS_DIR / "family_bands.json", out_dir: Path = FIGURES_DIR,
         band: str = "2-96", name: str | None = None) -> None:
    bands = json.loads(Path(bands_path).read_text())
    figstyle.apply()
    if panel == "supervise":
        figure, ax = plt.subplots(figsize=(7.0, 5.2))
        draw_accuracy_heatmap(ax, figure, supervise_grid(bands, band),
                              [str(e) for e in EDIT_LAYERS], [str(s) for s in SUPERVISE_LAYERS],
                              "Edit Layer", "Supervision Layer", band,
                              box=(EDIT_LAYERS.index(CANONICAL_EDIT), SUPERVISE_LAYERS.index(CANONICAL_SUPERVISE)),
                              cell_font_size=6.6)
        figure.subplots_adjust(left=0.1, right=0.86, top=0.97, bottom=0.11)
        name = name or "fig_geometry_edit_supervise"
        what = "rank-1 geometry LoRAs, edit layer x supervision layer"
    elif panel == "rank":
        figure, ax = plt.subplots(figsize=(7.0, 2.9))
        draw_accuracy_heatmap(ax, figure, rank_grid(bands, band),
                              [str(e) for e in EDIT_LAYERS], [str(r) for r in RANKS],
                              "Edit Layer", "LoRA Rank", band,
                              box=(EDIT_LAYERS.index(CANONICAL_EDIT), RANKS.index(1)),
                              cell_font_size=6.3, colorbar_aspect=14)
        figure.subplots_adjust(left=0.1, right=0.84, top=0.96, bottom=0.19)
        name = name or "fig_geometry_rank_layer"
        what = "geometry LoRAs supervised at layer 18, rank x edit layer"
    else:
        raise ValueError("panel must be 'supervise' or 'rank'")
    figstyle.save(figure, name, out_dir, note=(
        f"{what}; canonical recipe otherwise (fixed frame, residual stream, input/offset planes at periods 2, 5, 10, 20, "
        f"radius scale 1.45, spread ratio 1.25, spread weight 0.3, 200 steps, sums <= 24). Accuracy on sums {band}, "
        f"mean over seeds. Black box = edit layer 4, supervision layer 18, rank 1. Data: {Path(bands_path)}."))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", choices=["supervise", "rank"], required=True)
    parser.add_argument("--bands", default=str(RESULTS_DIR / "family_bands.json"))
    parser.add_argument("--band", default="2-96")
    parser.add_argument("--out", default=str(FIGURES_DIR))
    parser.add_argument("--name", default=None)
    args = parser.parse_args()
    make(args.panel, Path(args.bands), Path(args.out), args.band, args.name)


if __name__ == "__main__":
    main()
