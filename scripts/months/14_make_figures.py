#!/usr/bin/env python3
"""Render every paper figure from the pipeline outputs into outputs/months/figures/.

Each figure lives in scripts/months/figures/ with its own CLI; this driver calls their
`make` functions with the settled presets. Use --only to render a subset, and
--out to write somewhere else.
"""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "figures"))

from months.paths import FIGURES_DIR  # noqa: E402

# name -> (module in scripts/figures, keyword arguments for its make())
FIGURES = {
    # main text
    "behavioral_main": ("fig_behavioral", {"preset": "main"}),
    "projections_main": ("fig_projections", {"preset": "main"}),
    "r2_lines_main": ("fig_r2_spectrum", {"preset": "main_small"}),
    # appendix: task LoRAs
    "task_layer_rank": ("fig_task_layer_rank", {}),
    "training_range_ladder": ("fig_behavioral", {"preset": "ladder"}),
    # appendix: Fourier probes
    "r2_heatmap": ("fig_r2_spectrum", {"preset": "heatmap_T100"}),
    "r2_lines_L18": ("fig_r2_spectrum", {"preset": "lines_L18"}),
    "r2_lines_L22": ("fig_r2_spectrum", {"preset": "lines_L22"}),
    "projections_input_L18": ("fig_projections", {"preset": "appendix_input_L18"}),
    "projections_offset_L18": ("fig_projections", {"preset": "appendix_offset_L18"}),
    "projections_sum_L18": ("fig_projections", {"preset": "appendix_sum_L18"}),
    "projections_sum_L20": ("fig_projections", {"preset": "appendix_sum_L20"}),
    "projections_sum_L22": ("fig_projections", {"preset": "appendix_sum_L22"}),
    # appendix: geometry LoRA placement and ablations
    "geometry_edit_supervise": ("fig_geometry_placement", {"panel": "supervise"}),
    "geometry_rank_layer": ("fig_geometry_placement", {"panel": "rank"}),
    "perprompt_projections": ("fig_projections", {"preset": "perprompt_offset"}),
    "radius_spread_overall": ("fig_radius_spread", {"band": "2-96"}),
    "radius_spread_far": ("fig_radius_spread", {"band": "49-96"}),
    "planes_ladder": ("fig_planes_ladder", {}),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", default=None, help="names from the FIGURES table")
    parser.add_argument("--out", default=str(FIGURES_DIR))
    args = parser.parse_args()
    names = args.only or list(FIGURES)
    unknown = [name for name in names if name not in FIGURES]
    if unknown:
        raise SystemExit(f"unknown figure names {unknown}; choose from {sorted(FIGURES)}")
    failures = []
    for name in names:
        module_name, kwargs = FIGURES[name]
        print(f"== {name}")
        try:
            make = importlib.import_module(module_name).make
            make(out_dir=Path(args.out), **kwargs)
        except Exception as error:  # keep going so one missing input does not hide the rest
            failures.append((name, error))
            print(f"   failed: {error}")
    if failures:
        raise SystemExit(f"{len(failures)} figure(s) failed: {[name for name, _ in failures]}")


if __name__ == "__main__":
    main()
