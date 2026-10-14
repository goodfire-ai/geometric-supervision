#!/usr/bin/env python3
"""Geometry-LoRA accuracy over the radius scale x spread ratio grid at the canonical placement.

Each cell is the mean (and standard deviation) over seeds; the canonical cell
(radius scale 1.45, spread ratio 1.25) is outlined in black and the base model
is marked on the colorbar.

Reads results/family_bands.json with ids gamtau_a<radius code>_t<spread code>, where the
code is the value with two decimals and no point (1.45 -> 145), family "radius_spread".
The canonical cell is the config geometry_e04_s18.
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
from months.configs import CANONICAL_GEOMETRY_ID, RADIUS_SCALES, SPREAD_RATIOS  # noqa: E402
from months.paths import RESULTS_DIR, FIGURES_DIR  # noqa: E402

CANONICAL_RADIUS, CANONICAL_SPREAD = 1.45, 1.25


def value_code(x: float) -> str:
    return f"{x:.2f}".replace(".", "")


def config_id_for(radius_scale: float, spread_ratio: float) -> str:
    if (radius_scale, spread_ratio) == (CANONICAL_RADIUS, CANONICAL_SPREAD):
        return CANONICAL_GEOMETRY_ID
    return f"gamtau_a{value_code(radius_scale)}_t{value_code(spread_ratio)}"


def grids(bands: dict, band: str) -> tuple[np.ndarray, np.ndarray]:
    means = np.full((len(SPREAD_RATIOS), len(RADIUS_SCALES)), np.nan)
    sds = np.full_like(means, np.nan)
    for i, spread_ratio in enumerate(SPREAD_RATIOS):
        for j, radius_scale in enumerate(RADIUS_SCALES):
            entry = bands.get(config_id_for(radius_scale, spread_ratio))
            if entry and band in entry["bands"]:
                means[i, j] = entry["bands"][band]["mean"]
                sds[i, j] = entry["bands"][band]["sd"]
    return means, sds


def make(band: str = "2-96", bands_path: Path = RESULTS_DIR / "family_bands.json",
         out_dir: Path = FIGURES_DIR, font_scale: float = 1.35, name: str | None = None) -> None:
    bands = json.loads(Path(bands_path).read_text())
    means, sds = grids(bands, band)
    figstyle.apply()
    plt.rcParams["mathtext.fontset"] = "stix"
    figure, ax = plt.subplots(figsize=(5.6 + 0.5 * (len(RADIUS_SCALES) - 5), 4.0 + 0.45 * (len(SPREAD_RATIOS) - 4)))
    draw_accuracy_heatmap(ax, figure, means, [f"{r:g}" for r in RADIUS_SCALES], [f"{s:g}" for s in SPREAD_RATIOS],
                          r"Target Radius Scale $\alpha_r$", r"Spread Target $\tau$", band,
                          box=(RADIUS_SCALES.index(CANONICAL_RADIUS), SPREAD_RATIOS.index(CANONICAL_SPREAD)),
                          sd_matrix=sds, cell_font_size=7.5 * font_scale, tick_font_size=10 * font_scale,
                          label_font_size=11 * font_scale, colorbar_font_size=10.5 * font_scale)
    figure.subplots_adjust(left=0.13, right=0.86, top=0.97, bottom=0.14)
    name = name or {"2-96": "fig_radius_spread_overall", "49-96": "fig_radius_spread_far"}.get(
        band, f"fig_radius_spread_{band.replace('-', '_')}")
    figstyle.save(figure, name, out_dir, note=(
        f"Rank-1 geometry LoRA at edit layer 4, supervised at layer 18 (spread weight 0.3, 200 steps, sums <= 24): "
        f"target radius scale x spread ratio grid, accuracy on sums {band}, mean ± sd over seeds. Black box = the "
        f"main-text setting (1.45, 1.25). Base model marked on the colorbar. Data: {Path(bands_path)}."))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", default="2-96", dest="band")
    parser.add_argument("--bands", default=str(RESULTS_DIR / "family_bands.json"))
    parser.add_argument("--out", default=str(FIGURES_DIR))
    parser.add_argument("--font-scale", type=float, default=1.35)
    parser.add_argument("--name", default=None)
    args = parser.parse_args()
    make(args.band, Path(args.bands), Path(args.out), args.font_scale, args.name)


if __name__ == "__main__":
    main()
