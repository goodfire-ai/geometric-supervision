#!/usr/bin/env python3
"""Accuracy per sum as a heatmap, one row per model, with each adapter's training range boxed.

Reads
    cache/ladder.json                results["base"] and results["ladder_p0NN"], each with
                                     "per_sum": {"<sum>": {"n", "correct"}} and "train_max_sum"
    runs/<config_id>/seed-*/result.json   for any other adapter; seeds are pooled by
                                     summing the per-sum counts

Presets
    main     base model, the main-text task LoRA (rank 8, layer 3) and the geometry LoRA,
             both trained on sums <= 24; view clipped at sum 96
    ladder   base model and the training-range ladder of task LoRAs (<= 24 .. <= 96)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba
from matplotlib.patches import Patch


from common import figstyle  # noqa: E402
from months.evaluate import pooled_per_sum, seed_dirs  # noqa: E402
from months.paths import CACHE_DIR, FIGURES_DIR, RUNS_DIR  # noqa: E402

TASK_BOX_COLOR = figstyle.ROLE["task"]
# Blue (the geometry role color) would vanish into the blue accuracy ramp, so the
# geometry box borrows green.
GEOMETRY_BOX_COLOR = figstyle.AQUA
MISSING_CELL_COLOR = "#e7e3dc"
BOX_LINE_WIDTH = 1.6
BOX_FILL_ALPHA = 0.08
BOTTOM_MARGIN_INCHES = 0.80


def ladder_row(ladder: dict, key: str) -> dict:
    return ladder["results"][key]["per_sum"]


def runs_row(runs_root: Path, config_id: str) -> dict:
    run_dirs = seed_dirs(runs_root / config_id)
    if not run_dirs:
        raise FileNotFoundError(f"no completed runs under {runs_root / config_id}")
    return pooled_per_sum(run_dirs)


def preset_rows(preset: str, ladder: dict, runs_root: Path) -> tuple[list[dict], int]:
    """Row specs [{label, per_sum, train_max_sum or None, box_color}] and the view's max sum."""
    rows = [{"label": "Base Model", "per_sum": ladder_row(ladder, "base"), "train_max_sum": None, "box_color": None}]
    if preset == "main":
        rows.append({"label": "Task LoRA (≤ 24)", "per_sum": runs_row(runs_root, "task_L03_r08"),
                     "train_max_sum": 24, "box_color": TASK_BOX_COLOR})
        rows.append({"label": "Geometry LoRA (≤ 24)", "per_sum": runs_row(runs_root, "geometry_e04_s18"),
                     "train_max_sum": 24, "box_color": GEOMETRY_BOX_COLOR})
        return rows, 96
    if preset == "ladder":
        for cap in (24, 36, 48, 60, 72, 84, 96):
            key = f"ladder_p{cap:03d}"
            if key not in ladder["results"]:
                continue
            rows.append({"label": f"Task LoRA (≤ {cap})", "per_sum": ladder_row(ladder, key),
                         "train_max_sum": cap, "box_color": TASK_BOX_COLOR})
        return rows, 120
    raise ValueError(f"unknown preset {preset!r}")


def accuracy_matrix(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    sums = np.array(sorted(int(s) for s in rows[0]["per_sum"]))
    matrix = np.full((len(rows), len(sums)), np.nan)
    for i, row in enumerate(rows):
        for j, total in enumerate(sums):
            entry = row["per_sum"].get(str(int(total)))
            if entry and entry["n"]:
                matrix[i, j] = entry["correct"] / entry["n"]
    return sums, matrix


def make(preset: str = "main", ladder_path: Path = CACHE_DIR / "ladder.json", runs_root: Path = RUNS_DIR,
         out_dir: Path = FIGURES_DIR, font_scale: float | None = None, name: str | None = None) -> None:
    ladder = json.loads(Path(ladder_path).read_text())
    rows, max_sum = preset_rows(preset, ladder, Path(runs_root))
    font_scale = font_scale if font_scale is not None else (1.25 if preset == "main" else 1.0)
    name = name or {"main": "fig_behavioral_main", "ladder": "fig_training_range_ladder"}[preset]

    figstyle.apply()
    sums, matrix = accuracy_matrix(rows)
    visible = sums <= max_sum
    missing = bool(np.isnan(matrix[:, visible]).any())

    figure_height = 0.46 * len(rows) + 1.45
    figure, ax = plt.subplots(figsize=(10.4, figure_height))
    colormap = figstyle.CMAP.copy()
    colormap.set_bad(MISSING_CELL_COLOR)
    image = ax.imshow(matrix, aspect="auto", cmap=colormap, vmin=0, vmax=1, interpolation="nearest",
                      extent=(sums[0] - 0.5, sums[-1] + 0.5, len(rows) - 0.5, -0.5))

    for i, row in enumerate(rows):
        if row["train_max_sum"] is None:
            continue
        color = row["box_color"]
        left = sums[0] - 0.5
        width = row["train_max_sum"] + 0.5 - left
        # full row height, so consecutive boxes share an edge and read as one staircase
        ax.add_patch(plt.Rectangle((left, i - 0.5), width, 1.0, lw=0, facecolor=color,
                                   alpha=BOX_FILL_ALPHA, zorder=5))
        # clip_on=False: the left and bottom edges sit exactly on the axes boundary
        ax.add_patch(plt.Rectangle((left, i - 0.5), width, 1.0, fill=False, edgecolor=color,
                                   lw=BOX_LINE_WIDTH, zorder=6, clip_on=False))

    ax.set_yticks(range(len(rows)), [row["label"] for row in rows])
    ax.set_xlabel("Sum = Input + Offset", fontsize=11 * font_scale)
    ax.tick_params(axis="both", labelsize=10 * font_scale)
    ax.grid(False)
    ax.set_xticks([t for t in range(12, 133, 12) if t <= max_sum])
    ax.set_xlim(sums[0] - 0.5, max_sum + 0.5)

    colorbar = figure.colorbar(image, ax=ax, fraction=0.028, pad=0.014)
    colorbar.set_label("Accuracy", fontsize=11 * font_scale)
    colorbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    colorbar.set_ticklabels(["0", "0.25", "0.5", "0.75", "1.0"])
    colorbar.ax.tick_params(labelsize=10 * font_scale)
    colorbar.outline.set_visible(False)

    figure.subplots_adjust(bottom=BOTTOM_MARGIN_INCHES / figure_height)

    def swatch(color, label):
        return Patch(facecolor=to_rgba(color, BOX_FILL_ALPHA), edgecolor=color, linewidth=BOX_LINE_WIDTH, label=label)

    box_colors = {row["box_color"] for row in rows if row["box_color"]}
    if GEOMETRY_BOX_COLOR in box_colors:
        handles = [swatch(GEOMETRY_BOX_COLOR, "Geometry LoRA Training Range"),
                   swatch(TASK_BOX_COLOR, "Task LoRA Training Range")]
    else:
        handles = [swatch(TASK_BOX_COLOR, "LoRA Training Range")]
    if missing:
        handles.append(Patch(facecolor=MISSING_CELL_COLOR, edgecolor=figstyle.AXIS, linewidth=0.8, label="Not evaluated"))
    figure.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=len(handles),
                  fontsize=9.5 * font_scale, handlelength=2.1, handleheight=1.3, columnspacing=2.2,
                  borderaxespad=0.0, frameon=False)

    figstyle.save(figure, name, out_dir, note=(
        f"Accuracy per sum (argmax over the twelve month tokens), one row per model, view clipped at sum {max_sum}. "
        f"Boxes mark each adapter's training range (sums 2..cap); the base model has none. Rows: "
        f"{', '.join(row['label'] for row in rows)}. Adapter rows pool their seeds by summing per-sum counts. "
        f"Data: {Path(ladder_path)} and {Path(runs_root)}."))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=["main", "ladder"], default="main")
    parser.add_argument("--ladder", default=str(CACHE_DIR / "ladder.json"))
    parser.add_argument("--runs", default=str(RUNS_DIR))
    parser.add_argument("--out", default=str(FIGURES_DIR))
    parser.add_argument("--font-scale", type=float, default=None)
    parser.add_argument("--name", default=None)
    args = parser.parse_args()
    make(args.preset, Path(args.ladder), Path(args.runs), Path(args.out), args.font_scale, args.name)


if __name__ == "__main__":
    main()
