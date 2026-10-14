"""Shared drawing helpers for the accuracy heatmaps (layer x rank, edit x supervise, radius x spread)."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


from common import figstyle  # noqa: E402

BAND_LABEL = {"2-24": "Sum ≤ 24", "2-36": "Sum ≤ 36", "25-36": "Sum 25–36", "25-48": "Sum 25–48",
              "49-96": "Sum 49–96", "25-96": "Sum 25–96", "2-96": "Sum 2–96"}

# The base model's accuracy on each band, marked on every colorbar.
# Measured on the evaluation grid; used only when cache/ladder.json has not been built yet.
BASE_ACCURACY_FALLBACK = {"2-24": 0.790, "2-36": 0.582, "25-36": 0.278, "25-48": 0.229,
                          "49-96": 0.274, "25-96": 0.259, "2-96": 0.363}


def base_accuracies(ladder_path=None) -> dict[str, float]:
    """The base model's band accuracies, read from the ladder file the pipeline produces."""
    from months.evaluate import base_model_bands
    from months.paths import CACHE_DIR
    measured = base_model_bands(ladder_path or CACHE_DIR / "ladder.json")
    return measured if measured is not None else dict(BASE_ACCURACY_FALLBACK)

MISSING_CELL_COLOR = "#e9ecef"


def cell_label(value: float, sd: float | None = None) -> str:
    """'.73' for 0.73, '1.0' at the top, optionally with a second line '±0.04'."""
    text = "1.0" if value >= 0.995 else f"{value:.2f}"[1:]
    if sd is not None:
        text += f"\n±{sd:.2f}"
    return text


def draw_accuracy_heatmap(ax, figure, matrix: np.ndarray, x_labels: list[str], y_labels: list[str],
                          x_title: str, y_title: str, band: str, box: tuple[int, int] | None = None,
                          annotate: bool = True, sd_matrix: np.ndarray | None = None,
                          cell_font_size: float = 7.0, tick_font_size: float = 9.5,
                          label_font_size: float = 11.0, colorbar_font_size: float = 10.5,
                          colorbar_fraction: float = 0.05, colorbar_aspect: float = 20,
                          base_accuracy: float | None = None) -> None:
    """Viridis heatmap of accuracies in [0, 1] with annotated cells and a base-model colorbar tick.

    `box` is (column, row) of the cell to outline in black. Missing cells (NaN)
    show as light gray.
    """
    ax.set_facecolor(MISSING_CELL_COLOR)
    image = ax.imshow(matrix, cmap="viridis", vmin=0, vmax=1, origin="lower", aspect="auto")
    if annotate:
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                value = matrix[row, column]
                if not np.isfinite(value):
                    continue
                sd = sd_matrix[row, column] if sd_matrix is not None else None
                ax.text(column, row, cell_label(value, sd), ha="center", va="center",
                        fontsize=cell_font_size, color="white" if value < 0.55 else figstyle.INK)
    if box is not None:
        ax.add_patch(plt.Rectangle((box[0] - 0.5, box[1] - 0.5), 1, 1, fill=False, ec="black", lw=1.8))
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, fontsize=tick_font_size)
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels, fontsize=tick_font_size)
    ax.set_xlabel(x_title, fontsize=label_font_size)
    ax.set_ylabel(y_title, fontsize=label_font_size)
    ax.grid(False)

    base = base_accuracies()[band] if base_accuracy is None else base_accuracy
    ticks = sorted([t for t in (0, 0.25, 0.5, 0.75, 1.0) if abs(t - base) > 0.07] + [base])
    colorbar = figure.colorbar(image, ax=ax, fraction=colorbar_fraction, pad=0.03, ticks=ticks,
                               aspect=colorbar_aspect)
    colorbar.set_label(f"Accuracy ({BAND_LABEL[band]})", fontsize=colorbar_font_size)
    colorbar.ax.set_yticklabels(["Base Model" if t == base else f"{t:.2f}" for t in ticks],
                                fontsize=colorbar_font_size * 0.86)
    colorbar.ax.axhline(base, color="white", lw=1.4)
    for label in colorbar.ax.get_yticklabels():
        if label.get_text() == "Base Model":
            label.set_color(figstyle.BASE_GRAY)
