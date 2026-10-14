#!/usr/bin/env python3
"""Overall accuracy of the geometry LoRA as more Fourier periods are supervised.

Periods are added in a fixed order (2, 5, 10, 20, 50, then the rest by the base
model's ring R^2); x is the number of periods per variable (input and offset each
get one plane per period), on a log axis. Error bars are the standard deviation
over seeds where several exist. The period added at each of the first six steps
is written above its bar; the base model is a dashed line.

Reads results/family_bands.json (ids planes_ladder_k<k>, family "planes_ladder")
and configs/months/ladder_order.json ("period_order").
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


from common import figstyle  # noqa: E402
from months.paths import RESULTS_DIR, CONFIGS_DIR, FIGURES_DIR  # noqa: E402

X_TICKS = [1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 50, 98]
BASE_OVERALL_ACCURACY_FALLBACK = 36.3   # used only when cache/ladder.json is absent


def ladder_points(bands: dict, band: str) -> list[tuple[int, float, float, int]]:
    """(k, mean %, sd %, seeds) for every planes_ladder_k<k> config present, sorted by k."""
    points = []
    for config_id, entry in bands.items():
        if entry.get("family") != "planes_ladder" or band not in entry["bands"]:
            continue
        k = int(config_id.rsplit("_k", 1)[1])
        stats = entry["bands"][band]
        points.append((k, 100 * stats["mean"], 100 * stats["sd"], stats["n_seeds"]))
    return sorted(points)


def make(bands_path: Path = RESULTS_DIR / "family_bands.json",
         order_path: Path = CONFIGS_DIR / "ladder_order.json", out_dir: Path = FIGURES_DIR,
         band: str = "2-96", base_accuracy: float | None = None,
         name: str = "fig_planes_ladder") -> None:
    bands = json.loads(Path(bands_path).read_text())
    period_order = json.loads(Path(order_path).read_text())["period_order"]
    points = ladder_points(bands, band)
    if base_accuracy is None:
        from _heatmap import base_accuracies
        base_accuracy = 100 * base_accuracies().get(band, BASE_OVERALL_ACCURACY_FALLBACK / 100)
    if not points:
        raise SystemExit("no planes_ladder configs in the bands file")

    figstyle.apply()
    figure, ax = plt.subplots(figsize=(6.6, 3.4))
    ks = [p[0] for p in points]
    means = [p[1] for p in points]
    sds = [p[2] for p in points]
    ax.errorbar(ks, means, yerr=sds, fmt="-o", ms=5, lw=1.9, color=figstyle.ROLE["geometry"], capsize=2.5, zorder=3)
    ax.axhline(base_accuracy, ls="--", lw=1.4, color=figstyle.INK, alpha=0.6, zorder=2, label="Base Model")
    ax.set_xscale("log")
    ax.set_xticks(X_TICKS)
    ax.set_xticklabels([str(t) for t in X_TICKS])
    ax.minorticks_off()
    ax.set_xlim(0.9, 105)
    for k, mean, sd in zip(ks, means, sds):
        if k <= 6:
            ax.annotate(f"$+{period_order[k - 1]}$", (k, mean + sd), textcoords="offset points", xytext=(0, 4),
                        ha="center", va="bottom", fontsize=10, color=figstyle.INK)
    ax.set_xlabel("Number of Supervised Periods per Variable", fontsize=12.5)
    ax.set_ylabel("Accuracy (Sum 2–96)", fontsize=12.5)
    ax.set_ylim(30, 82)
    ax.legend(frameon=False, loc="upper right")
    ax.set_axisbelow(True)
    figure.tight_layout()
    figstyle.save(figure, name, out_dir, note=(
        "Canonical geometry LoRA (rank 1, edit layer 4, supervised at layer 18, fixed frame) supervising the first k "
        f"periods of the order {period_order[:6]}... on the input and offset planes; accuracy on sums {band}, "
        "mean over seeds (error bars = sd where more than one seed; one seed beyond k = 6). Labels give the period "
        f"added at each of the first six steps. Data: {Path(bands_path)}, {Path(order_path)}."))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bands", default=str(RESULTS_DIR / "family_bands.json"))
    parser.add_argument("--order", default=str(CONFIGS_DIR / "ladder_order.json"))
    parser.add_argument("--out", default=str(FIGURES_DIR))
    parser.add_argument("--base-accuracy", type=float, default=None, help="percent; default: from cache/ladder.json")
    parser.add_argument("--name", default="fig_planes_ladder")
    args = parser.parse_args()
    make(Path(args.bands), Path(args.order), Path(args.out), base_accuracy=args.base_accuracy, name=args.name)


if __name__ == "__main__":
    main()
