#!/usr/bin/env python3
"""Ring R^2 of the Fourier planes across periods: which periods training sharpens.

Two views of the same numbers:
    lines     x = Fourier period, y = ring R^2, one line per model condition,
              one panel per layer
    heatmap   x = period, y = layer, one panel per condition

Ring R^2 measures how well the projected points sit on an ideal ring at the
right angles (best rotation/reflection, scale and translation of the ideal
ring). Past period ~45 each residue class holds only one or two sums on the
2..96 grid, so R^2 there asks whether the sum is encoded as an angle at all
rather than whether the residues separate; the paper's main figure therefore
stops at period 60.

Data: cache/period_sweep_r2_resid.json written by 07_period_sweep.py:
    {"conditions": {<condition>: {"r2__L<layer>__<carrier>": {"<period>": r2}}}}

Usage:
    scripts/months/figures/fig_r2_spectrum.py --preset main_small
    scripts/months/figures/fig_r2_spectrum.py --mode lines --layers 18 --pmax 60 --bare --no-title ...
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


from common import figstyle  # noqa: E402
from months.paths import CACHE_DIR, FIGURES_DIR  # noqa: E402

LABELS = {"base": "Base Model", "task_L03_r08": "Task LoRA", "geometry_e04_s18": "Geometry LoRA"}
APPENDIX_CONDITIONS = ["base", "ladder_p024", "ladder_p048", "ladder_p096", "geometry_e04_s18"]
BASE_TEN_PERIODS = [2, 5, 10, 20, 50]
MULTIPLES_OF_THREE = list(range(3, 31, 3))
NEAR_SINGLETON_FROM = 45     # past this period the residue classes are near singletons on the grid

PRESETS = {
    "main_small": dict(name="fig_r2_lines_main", mode="lines", layers=[18], carrier="sum", pmax=60, bare=True,
                       title=False, conditions=["base", "task_L03_r08", "geometry_e04_s18"],
                       xticks=[2, 5, 10, 20, 30, 40, 50, 60], width=6.6, panel_height=4.2,
                       font_scale=1.3, line_width_scale=1.2),
    "lines_L18": dict(name="fig_r2_lines_L18", mode="lines", layers=[18], carrier="sum", pmax=100, bare=True,
                      title=False, conditions=APPENDIX_CONDITIONS,
                      xticks=[2, 5, 10, 12, 20, 25, 30, 33, 40, 50, 60, 70, 80, 90, 100]),
    "lines_L22": dict(name="fig_r2_lines_L22", mode="lines", layers=[22], carrier="sum", pmax=100, bare=True,
                      title=False, conditions=APPENDIX_CONDITIONS,
                      xticks=[2, 5, 10, 12, 20, 25, 30, 33, 40, 50, 60, 70, 80, 90, 100]),
    "heatmap_T100": dict(name="fig_r2_heatmap_T100", mode="heatmap", carrier="sum", pmax=100,
                         conditions=APPENDIX_CONDITIONS, xticks=[2, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]),
}


def condition_label(condition: str) -> str:
    if condition in LABELS:
        return LABELS[condition]
    match = re.fullmatch(r"ladder_p(\d+)", condition)
    return f"Task LoRA (≤ {int(match.group(1))})" if match else condition


def condition_role(condition: str) -> str:
    if condition == "base":
        return "base"
    if condition.startswith("geometry") or condition.startswith("perprompt") or condition.startswith("gamtau"):
        return "geometry"
    return "task"


def series(conditions_data: dict, condition: str, layer: int, carrier: str) -> tuple[np.ndarray, np.ndarray]:
    values = conditions_data[condition].get(f"r2__L{layer}__{carrier}", {})
    periods = sorted(int(period) for period in values)
    return np.asarray(periods), np.asarray([values[str(period)] for period in periods])


def line_style(condition: str, task_index: int, line_width_scale: float) -> dict:
    role = condition_role(condition)
    if role == "base":
        style = dict(figstyle.BASE_LINE)
        style["lw"] = style["lw"] * line_width_scale
        return style
    if role == "geometry":
        return dict(color=figstyle.ROLE["geometry"], lw=1.9 * line_width_scale)
    ramp = figstyle.RAMP["task"]
    return dict(color=ramp[task_index % len(ramp)], lw=1.5 * line_width_scale)


def draw_lines(conditions_data: dict, conditions: list[str], layers: list[int], carrier: str, pmax: int,
               bare: bool = False, title: bool = True, xticks: list[int] | None = None, width: float = 10.6,
               panel_height: float = 3.5, font_scale: float = 1.0, line_width_scale: float = 1.0):
    figstyle.apply()
    height = panel_height * len(layers)
    figure, axes = plt.subplots(len(layers), 1, figsize=(width, height), sharex=True, sharey=True,
                                gridspec_kw=dict(hspace=0.34))
    axes = np.atleast_1d(axes)
    styles = {}
    task_index = 0
    for condition in conditions:
        styles[condition] = line_style(condition, task_index, line_width_scale)
        if condition_role(condition) == "task":
            task_index += 1

    for ax, layer in zip(axes, layers):
        if not bare:
            if pmax > NEAR_SINGLETON_FROM:
                ax.axvspan(NEAR_SINGLETON_FROM, pmax, color="#f0eeea", zorder=0, lw=0)
                ax.annotate("classes become near-singletons:\n$R^2$ measures a different thing here",
                            ((NEAR_SINGLETON_FROM + pmax) / 2, 0.02), ha="center", va="bottom",
                            fontsize=7.5, color=figstyle.MUTED, style="italic")
            for period in BASE_TEN_PERIODS:
                if period <= pmax:
                    ax.axvline(period, color=figstyle.AQUA, lw=0.9, alpha=0.35, zorder=1)
            for period in MULTIPLES_OF_THREE:
                if period <= pmax:
                    ax.axvline(period, color=figstyle.RED, lw=0.9, alpha=0.22, zorder=1)
        for condition in conditions:
            if condition not in conditions_data:
                continue
            periods, r2 = series(conditions_data, condition, layer, carrier)
            keep = periods <= pmax
            zorder = {"base": 6, "geometry": 7, "task": 4}[condition_role(condition)]
            ax.plot(periods[keep], r2[keep], zorder=zorder, **styles[condition])
        ax.set_ylim(-0.03, 1.04)
        ax.set_ylabel("Circle $R^2$", fontsize=11 * font_scale)
        ax.tick_params(axis="y", labelsize=9.5 * font_scale)
        if title:
            ax.set_title(f"Layer {layer}", loc="left", fontsize=13 * font_scale)

    ticks = [t for t in (xticks or (2, 5, 10, 12, 20, 24, 30, 40, 50, 60)) if t <= pmax]
    for ax in axes:
        ax.set_xticks(ticks)
        ax.set_xlim(2, pmax)
        ax.tick_params(axis="x", labelsize=9.5 * font_scale, labelbottom=True)
    axes[-1].set_xlabel("Fourier period $T$", fontsize=11 * font_scale)

    handles = [plt.Line2D([], [], label=condition_label(condition), **styles[condition])
               for condition in conditions if condition in conditions_data]
    if not bare:
        handles += [plt.Line2D([], [], color=figstyle.AQUA, lw=2.4, alpha=0.45, label="periods {2, 5, 10, 20, 50}"),
                    plt.Line2D([], [], color=figstyle.RED, lw=2.4, alpha=0.35, label="multiples of 3 (T ≤ 30)")]
    figure.subplots_adjust(left=(0.80 * font_scale) / width, right=1 - 0.16 / width,
                           top=1 - (0.42 if title else 0.16) / height,
                           bottom=(0.92 * (0.5 + 0.5 * font_scale)) / height)
    figure.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.06 / height), ncol=len(handles),
                  fontsize=9.5 * font_scale, handlelength=1.9, columnspacing=1.6, borderaxespad=0.0, frameon=False)
    return figure


def draw_heatmap(conditions_data: dict, conditions: list[str], carrier: str, pmax: int,
                 xticks: list[int] | None = None, only: list[int] | None = None, stack: bool = False):
    figstyle.apply()
    present = [condition for condition in conditions if condition in conditions_data]
    layers = sorted({int(key.split("__")[1][1:]) for key in conditions_data[present[0]] if key.startswith("r2__")})

    def matrix(condition: str) -> tuple[np.ndarray, np.ndarray]:
        periods, _ = series(conditions_data, condition, layers[0], carrier)
        if only:
            index = [int(np.where(periods == period)[0][0]) for period in only]
        else:
            index = np.where(periods <= pmax)[0]
        rows = np.stack([series(conditions_data, condition, layer, carrier)[1][index] for layer in layers])
        return rows, periods[index]

    if stack:
        figure, axes = plt.subplots(len(present), 1, figsize=(10.6, 1.85 * len(present) + 1.0), sharex=True,
                                    gridspec_kw=dict(hspace=0.42))
    else:
        panel_width = 2.15 if only else 3.5
        figure, axes = plt.subplots(1, len(present), figsize=(panel_width * len(present) + 1.2, 4.4), sharey=True,
                                    gridspec_kw=dict(wspace=0.10))
    axes = np.atleast_1d(axes)
    for ax, condition in zip(axes, present):
        values, periods_used = matrix(condition)
        if only:
            image = ax.imshow(values, aspect="auto", cmap=figstyle.CMAP, vmin=0, vmax=1, interpolation="nearest",
                              extent=(-0.5, len(periods_used) - 0.5, layers[-1] + 0.5, layers[0] - 0.5))
            ax.set_xticks(range(len(periods_used)), [str(period) for period in periods_used])
        else:
            image = ax.imshow(values, aspect="auto", cmap=figstyle.CMAP, vmin=0, vmax=1, interpolation="nearest",
                              extent=(periods_used[0] - 0.5, periods_used[-1] + 0.5, layers[-1] + 0.5, layers[0] - 0.5))
            if xticks:
                ax.set_xticks([t for t in xticks if t <= pmax])
        ax.set_title(condition_label(condition), loc="left", fontsize=14, pad=7)
        ax.set_xlabel("Fourier period $T$", fontsize=12.5)
        ax.grid(False)
        ax.tick_params(labelsize=10)
        ax.set_yticks(list(range(0, layers[-1] + 1, 10)))
        if stack and ax is not axes[-1]:
            ax.set_xlabel("")
    for ax in (axes if stack else axes[:1]):
        ax.set_ylabel("Layer", fontsize=12.5)
    colorbar = figure.colorbar(image, ax=axes, fraction=0.020, pad=0.012)
    colorbar.set_label("Circle $R^2$", fontsize=12.5)
    colorbar.ax.tick_params(labelsize=10)
    colorbar.outline.set_visible(False)
    return figure


def load_conditions(data_path: Path) -> dict:
    return json.loads(Path(data_path).read_text())["conditions"]


def make(preset: str, data_path: Path = CACHE_DIR / "period_sweep_r2_resid.json",
         out_dir: Path = FIGURES_DIR, name: str | None = None) -> None:
    settings = dict(PRESETS[preset])
    figure_name = name or settings.pop("name")
    settings.pop("name", None)
    mode = settings.pop("mode")
    conditions_data = load_conditions(data_path)
    if mode == "lines":
        figure = draw_lines(conditions_data, **settings)
    else:
        figure = draw_heatmap(conditions_data, **settings)
    note = (f"Ring R^2 of the {settings['carrier']} Fourier planes vs period ({mode}), conditions "
            f"{settings['conditions']}, periods up to {settings['pmax']}. Ring R^2: best similarity fit of the ideal "
            f"ring (rotation/reflection, scale, translation) to the projected evaluation prompts, 1 - SSE/SST. "
            f"Data: {data_path}.")
    figstyle.save(figure, figure_name, out_dir, note)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=sorted(PRESETS), default=None)
    parser.add_argument("--mode", choices=["lines", "heatmap"], default="lines")
    parser.add_argument("--carrier", default="sum")
    parser.add_argument("--layers", nargs="+", type=int, default=[18])
    parser.add_argument("--pmax", type=int, default=60)
    parser.add_argument("--conditions", nargs="+", default=["base", "task_L03_r08", "geometry_e04_s18"])
    parser.add_argument("--bare", action="store_true", help="lines: no guide lines or shading")
    parser.add_argument("--no-title", dest="title", action="store_false")
    parser.add_argument("--xticks", nargs="*", type=int, default=None)
    parser.add_argument("--width", type=float, default=10.6)
    parser.add_argument("--panel-h", type=float, default=3.5)
    parser.add_argument("--font-scale", type=float, default=1.0)
    parser.add_argument("--lw-scale", type=float, default=1.0)
    parser.add_argument("--only", nargs="*", type=int, default=None, help="heatmap: only these periods, as columns")
    parser.add_argument("--stack", action="store_true", help="heatmap: one condition per row")
    parser.add_argument("--data", default=str(CACHE_DIR / "period_sweep_r2_resid.json"))
    parser.add_argument("--out", default=str(FIGURES_DIR))
    parser.add_argument("--name", default=None)
    args = parser.parse_args()
    if args.preset:
        make(args.preset, Path(args.data), Path(args.out), args.name)
        return
    conditions_data = load_conditions(Path(args.data))
    if args.mode == "lines":
        figure = draw_lines(conditions_data, args.conditions, args.layers, args.carrier, args.pmax, bare=args.bare,
                            title=args.title, xticks=args.xticks, width=args.width, panel_height=args.panel_h,
                            font_scale=args.font_scale, line_width_scale=args.lw_scale)
        name = args.name or f"fig_r2_lines_{args.carrier}" + "".join(f"_L{layer}" for layer in args.layers)
    else:
        figure = draw_heatmap(conditions_data, args.conditions, args.carrier, args.pmax, xticks=args.xticks,
                              only=args.only, stack=args.stack)
        name = args.name or f"fig_r2_heatmap_{args.carrier}"
    figstyle.save(figure, name, Path(args.out),
                  f"Ring R^2 vs Fourier period ({args.mode}); conditions {args.conditions}; data {args.data}.")


if __name__ == "__main__":
    main()
