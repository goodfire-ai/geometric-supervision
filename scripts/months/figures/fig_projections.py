#!/usr/bin/env python3
"""Fourier-plane projections: every evaluation prompt scattered on a probe plane.

One panel per (period, model condition): the 1074 prompts projected through
the plane's cos/sin probes, colored by the task variable's value modulo the
period. A perfectly sharpened representation puts each residue class in a
tight cluster at its own angle. Each panel carries a dashed unit circle and the
plane's own axes for reference, a colorbar labeling the residues, and
optionally the ring R^2 of the points against the ideal ring.

Data: cache/manifolds/<condition>__grid1074.npz written by 06_capture_manifolds.py,
with arrays "values_<carrier>" and "pts_L<layer>_<site>_<carrier>_p<period>".

Usage:
    scripts/months/figures/fig_projections.py --preset main
    scripts/months/figures/fig_projections.py --carrier sum --layer 18 --site resid --periods 2 5 10 \
        --conditions base task_L03_r08 geometry_e04_s18 --r2 --name my_figure
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


from common import figstyle  # noqa: E402
from months.metrics import ring_r2  # noqa: E402
from months.paths import CACHE_DIR, FIGURES_DIR  # noqa: E402

LABELS = {
    "base": "Base Model",
    "task_L03_r08": "Task LoRA",
    "geometry_e04_s18": "Geometry LoRA",
    "perprompt_pervalue_lam000": "Per-value, λ = 0",
    "perprompt_pervalue_lam100": "Per-value, λ = 1",
    "perprompt_pervalue_lam300": "Per-value, λ = 3",
    "perprompt_perprompt_lam000": "Per-prompt, λ = 0",
    "perprompt_perprompt_lam030": "Per-prompt, λ = 0.3",
    "perprompt_perprompt_lam100": "Per-prompt, λ = 1",
    "perprompt_perprompt_lam300": "Per-prompt, λ = 3",
}

APPENDIX_CONDITIONS = ["base", "ladder_p024", "ladder_p048", "ladder_p096", "geometry_e04_s18"]

# Font and margin settings. "paper" is what the main-text figures use; "compact"
# is the smaller original setting kept for dense appendix grids.
SIZES = {
    "paper": dict(tick_size=13, colorbar_tick_size=10, r2_size=14, label_size=18, title_size=17,
                  row_size=18, left_in=1.3, bottom_in=0.75, wspace=0.5),
    "compact": dict(tick_size=7, colorbar_tick_size=6.5, r2_size=9, label_size=11.5, title_size=14,
                    row_size=14, left_in=0.86, bottom_in=0.56, wspace=0.42),
}

PRESETS = {
    "main": dict(name="fig_projections_main", carrier="sum", layer=18, site="resid", periods=[2, 5, 10],
                 conditions=["base", "task_L03_r08", "geometry_e04_s18"], r2=True, size="paper"),
    "appendix_sum_L18": dict(name="fig_projections_sum_L18", carrier="sum", layer=18, site="resid",
                             periods=[2, 5, 10, 20, 50], conditions=APPENDIX_CONDITIONS, r2=True, size="compact"),
    "appendix_sum_L20": dict(name="fig_projections_sum_L20", carrier="sum", layer=20, site="resid",
                             periods=[2, 5, 10, 20, 50], conditions=APPENDIX_CONDITIONS, r2=True, size="compact"),
    "appendix_sum_L22": dict(name="fig_projections_sum_L22", carrier="sum", layer=22, site="resid",
                             periods=[2, 5, 10, 20, 50], conditions=APPENDIX_CONDITIONS, r2=True, size="compact"),
    "appendix_input_L18": dict(name="fig_projections_input_L18", carrier="input-month", layer=18, site="mlp_input",
                               periods=[2, 5, 10, 20], conditions=APPENDIX_CONDITIONS, r2=True, size="compact"),
    "appendix_offset_L18": dict(name="fig_projections_offset_L18", carrier="offset", layer=18, site="mlp_input",
                                periods=[2, 5, 10, 20, 50], conditions=APPENDIX_CONDITIONS, r2=True, size="compact"),
    "perprompt_offset": dict(name="fig_perprompt_offset", carrier="offset", layer=18, site="resid",
                             periods=[2, 5, 10, 20],
                             conditions=["base", "geometry_e04_s18", "perprompt_perprompt_lam000",
                                         "perprompt_perprompt_lam030"],
                             labels={"geometry_e04_s18": "Per-value, λ = 0.3"}, r2=True, size="paper"),
}


def condition_label(condition: str, overrides: dict | None = None) -> str:
    if overrides and condition in overrides:
        return overrides[condition]
    if condition in LABELS:
        return LABELS[condition]
    match = re.fullmatch(r"ladder_p(\d+)", condition)
    return f"Task LoRA (≤ {int(match.group(1))})" if match else condition


def residue_colors(values: np.ndarray, period: int) -> np.ndarray:
    """Rainbow colors for value mod period; the whole ramp is spent on 0 .. T-1."""
    residues = np.remainder(np.asarray(values, dtype=float), period)
    return plt.get_cmap("rainbow")(residues / max(period - 1, 1))[:, :3]


def residue_ticks(period: int, max_ticks: int = 6) -> list[int]:
    """Which residues to label on the colorbar: all of them up to 10, else a round step."""
    if period <= 10:
        return list(range(period))
    step = int(np.ceil((period - 1) / (max_ticks - 1)))
    step = next((candidate for candidate in (5, 10, 20, 25) if candidate >= step), step)
    ticks = list(range(0, period, step))
    if ticks[-1] != period - 1:
        ticks.append(period - 1)
    return ticks


def draw(carrier: str, layer: int, site: str, periods: list[int], conditions: list[str],
         cache_dir: Path, r2: bool = True, limit: float = 4.0, labels: dict | None = None,
         size: str = "paper", colorbar_last_only: bool = False, **size_overrides):
    """Build the figure: conditions across, periods down."""
    settings = {**SIZES[size], **size_overrides}
    figstyle.apply()
    plt.rcParams["mathtext.fontset"] = "stix"
    data = {condition: np.load(Path(cache_dir) / f"{condition}__grid1074.npz") for condition in conditions}

    n_rows, n_cols = len(periods), len(conditions)
    width = 2.06 * n_cols + 0.85
    height = 1.74 * n_rows + 0.75
    figure, axes = plt.subplots(n_rows, n_cols, figsize=(width, height), squeeze=False,
                                sharex=True, sharey=True,
                                gridspec_kw=dict(wspace=settings["wspace"], hspace=0.14))
    step = 1 if limit <= 3 else 2
    ticks = [t for t in range(-int(limit), int(limit) + 1) if t % step == 0]

    for row, period in enumerate(periods):
        for column, condition in enumerate(conditions):
            ax = axes[row][column]
            archive = data[condition]
            values = archive[f"values_{carrier}"]
            ax.grid(False)
            ax.set_aspect("equal")
            ax.set_xlim(-limit, limit)
            ax.set_ylim(-limit, limit)
            ax.axhline(0, color=figstyle.AXIS, lw=0.7, zorder=0)
            ax.axvline(0, color=figstyle.AXIS, lw=0.7, zorder=0)
            ax.add_patch(plt.Circle((0, 0), 1.0, fill=False, lw=0.8, ls=(0, (3, 2.6)),
                                    color=figstyle.MUTED, alpha=0.8, zorder=1))
            key = f"pts_L{layer}_{site}_{carrier}_p{period}"
            if key in archive.files:
                points = archive[key]
                ax.scatter(points[:, 0], points[:, 1], s=3.4, c=residue_colors(values, period),
                           lw=0, alpha=0.85, zorder=3)
                if r2:
                    value = ring_r2(points, values, period)
                    if value is not None:
                        ax.text(0.035, 0.965, f"$R^2$ = {value:.2f}", transform=ax.transAxes,
                                ha="left", va="top", fontsize=settings["r2_size"], color=figstyle.INK,
                                family=figstyle.SERIF_STACK)
            else:
                ax.text(0.5, 0.5, "—", ha="center", va="center", transform=ax.transAxes, color=figstyle.MUTED)
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
            ax.tick_params(labelsize=settings["tick_size"], length=2,
                           labelbottom=(row == n_rows - 1), labelleft=(column == 0))
            for spine in ax.spines.values():
                spine.set_color(figstyle.AXIS)
                spine.set_linewidth(0.7)
            if row == 0:
                ax.set_title(condition_label(condition, labels), fontsize=settings["title_size"], pad=5,
                             family=figstyle.SERIF_STACK)
            if row == n_rows - 1:
                ax.set_xlabel("cos", fontsize=settings["label_size"], family=figstyle.SERIF_STACK)
            if column == 0:
                ax.set_ylabel("sin", fontsize=settings["label_size"], family=figstyle.SERIF_STACK)

            # colorbar: the same rainbow ramp in every panel, ticked at the residues of this period
            if colorbar_last_only and column != n_cols - 1:
                continue
            colorbar_ax = ax.inset_axes([1.06, 0.09, 0.048, 0.82])
            ramp = residue_colors(np.linspace(0, max(period - 1, 1), 256), period)
            colorbar_ax.imshow(ramp[:, None, :], aspect="auto", origin="lower", extent=(0, 1, 0, 1))
            colorbar_ax.set_xticks([])
            residues = residue_ticks(period)
            colorbar_ax.set_yticks([r / max(period - 1, 1) for r in residues])
            colorbar_ax.set_yticklabels([str(r) for r in residues])
            colorbar_ax.yaxis.tick_right()
            colorbar_ax.grid(False)
            colorbar_ax.tick_params(labelsize=settings["colorbar_tick_size"], length=2.6, width=0.7,
                                    color=figstyle.INK, direction="out", pad=1.6)
            for spine in colorbar_ax.spines.values():
                spine.set_color(figstyle.INK)
                spine.set_linewidth(0.7)

    figure.subplots_adjust(left=settings["left_in"] / width, right=1 - 0.14 / width,
                           top=1 - 0.36 / height, bottom=settings["bottom_in"] / height)
    figure.canvas.draw()
    for row, period in enumerate(periods):
        box = axes[row][0].get_position()
        figure.text(0.06 / width, (box.y0 + box.y1) / 2, f"Period {period}", rotation=90,
                    va="center", ha="left", fontsize=settings["row_size"], family=figstyle.SERIF_STACK)
    return figure


def make(preset: str, cache_dir: Path = CACHE_DIR / "manifolds", out_dir: Path = FIGURES_DIR,
         name: str | None = None) -> None:
    settings = dict(PRESETS[preset])
    figure_name = name or settings.pop("name")
    settings.pop("name", None)
    figure = draw(cache_dir=cache_dir, **settings)
    note = (f"Projections of the 1074 evaluation prompts onto the {settings['carrier']} Fourier planes at "
            f"layer {settings['layer']} ({settings['site']}), periods {settings['periods']}, colored by value mod period. "
            f"Conditions: {settings['conditions']}. Dashed circle = unit circle; R^2 = ring R^2 over all points. "
            f"Data: {cache_dir}.")
    figstyle.save(figure, figure_name, out_dir, note)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=sorted(PRESETS), default=None)
    parser.add_argument("--carrier", default="sum")
    parser.add_argument("--layer", type=int, default=18)
    parser.add_argument("--site", default="resid")
    parser.add_argument("--periods", nargs="+", type=int, default=[2, 5, 10])
    parser.add_argument("--conditions", nargs="+", default=["base", "task_L03_r08", "geometry_e04_s18"])
    parser.add_argument("--labels", nargs="*", default=[], help="condition=Label overrides")
    parser.add_argument("--r2", action="store_true")
    parser.add_argument("--size", choices=sorted(SIZES), default="paper")
    parser.add_argument("--limit", type=float, default=4.0)
    parser.add_argument("--cache", default=str(CACHE_DIR / "manifolds"))
    parser.add_argument("--out", default=str(FIGURES_DIR))
    parser.add_argument("--name", default=None)
    args = parser.parse_args()
    if args.preset:
        make(args.preset, Path(args.cache), Path(args.out), args.name)
        return
    labels = dict(item.split("=", 1) for item in args.labels)
    figure = draw(args.carrier, args.layer, args.site, args.periods, args.conditions, Path(args.cache),
                  r2=args.r2, limit=args.limit, labels=labels, size=args.size)
    name = args.name or f"fig_projections_{args.carrier}_L{args.layer}_{args.site}"
    figstyle.save(figure, name, Path(args.out),
                  f"Projections of the evaluation prompts onto the {args.carrier} planes at layer {args.layer} "
                  f"({args.site}); conditions {args.conditions}; data {args.cache}.")


if __name__ == "__main__":
    main()
