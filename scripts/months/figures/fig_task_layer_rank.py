#!/usr/bin/env python3
"""Task-LoRA accuracy by edit layer (0..31) and rank (1, 2, 4, 8, 16), two stacked heatmaps.

Top panel: accuracy on the first held-out cycle (sums 25-36). Bottom panel:
accuracy on the whole grid (sums 2-96). The cell used in the main text (rank 8,
layer 3) is outlined in black; the base model's accuracy is a colorbar tick.

Reads results/family_bands.json: {config_id: {"family", "config", "bands": {band: {"mean", "sd", "n_seeds"}}}}
with ids task_L<layer>_r<rank> in the family "task_layer_rank".
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _heatmap import base_accuracies, draw_accuracy_heatmap  # noqa: E402
from common import figstyle  # noqa: E402
from months.paths import RESULTS_DIR, FIGURES_DIR  # noqa: E402

RANKS = [1, 2, 4, 8, 16]
LAYERS = list(range(32))


def accuracy_grid(bands: dict, band: str) -> np.ndarray:
    matrix = np.full((len(RANKS), len(LAYERS)), np.nan)
    for i, rank in enumerate(RANKS):
        for j, layer in enumerate(LAYERS):
            entry = bands.get(f"task_L{layer:02d}_r{rank:02d}")
            if entry and band in entry["bands"]:
                matrix[i, j] = entry["bands"][band]["mean"]
    return matrix


def make(bands_path: Path = RESULTS_DIR / "family_bands.json", out_dir: Path = FIGURES_DIR,
         top_band: str = "25-36", bottom_band: str = "2-96", box: tuple[int, int] = (8, 3),
         base_values: dict | None = None, width: float = 7.0, name: str = "fig_task_layer_rank") -> None:
    bands = json.loads(Path(bands_path).read_text())
    base_values = {**base_accuracies(), **(base_values or {})}
    figstyle.apply()
    panel_height = 0.38 * width
    figure, (top, bottom) = plt.subplots(2, 1, figsize=(width, 1.85 * panel_height), gridspec_kw={"hspace": 0.5})
    box_position = (box[1], RANKS.index(box[0])) if box[0] > 0 else None
    for ax, band in ((top, top_band), (bottom, bottom_band)):
        draw_accuracy_heatmap(ax, figure, accuracy_grid(bands, band),
                              [str(layer) for layer in LAYERS], [str(rank) for rank in RANKS],
                              "Layer of Task LoRA", "LoRA Rank", band, box=box_position, annotate=True,
                              cell_font_size=5.5, colorbar_fraction=0.03, base_accuracy=base_values[band])
        ax.set_xticks(range(0, 32, 2))
        ax.set_xticklabels([str(layer) for layer in range(0, 32, 2)], fontsize=9.5)
    figure.subplots_adjust(left=0.09, right=0.83, top=0.97, bottom=0.12)
    figstyle.save(figure, name, out_dir, note=(
        f"Task LoRAs (cross-entropy on the answer, trained on sums <= 24) at every layer 0-31 and rank 1-16; "
        f"accuracy on sums {top_band} (top) and {bottom_band} (bottom), mean over seeds. Black box = rank {box[0]}, "
        f"layer {box[1]}. Base model marked on each colorbar. Data: {Path(bands_path)}."))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bands", default=str(RESULTS_DIR / "family_bands.json"))
    parser.add_argument("--out", default=str(FIGURES_DIR))
    parser.add_argument("--box", nargs=2, type=int, default=[8, 3], metavar=("RANK", "LAYER"))
    parser.add_argument("--base-values", default=None, help='JSON like {"25-36": 0.278, "2-96": 0.363}')
    parser.add_argument("--name", default="fig_task_layer_rank")
    args = parser.parse_args()
    make(Path(args.bands), Path(args.out), box=tuple(args.box),
         base_values=json.loads(args.base_values) if args.base_values else None, name=args.name)


if __name__ == "__main__":
    main()
