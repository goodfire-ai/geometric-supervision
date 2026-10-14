#!/usr/bin/env python3
"""Summarize every trained run, and print the numbers quoted in the main text.

Writes results/family_bands.json:
    {<config_id>: {"family": ..., "config": {...}, "bands": {<band>: {"mean", "sd", "n_seeds"}}}}

Bands are the standard ones plus "25-36" (the first held-out cycle) and
"2-36", which the task-LoRA heatmap uses. Accuracy per seed is recomputed from
the per-sum counts so the extra bands exist for every run.

Then prints the base model's, the task LoRA's and the geometry LoRA's accuracy, and the
effect of patching each LoRA's activations into the base model at layer 18 (Section 3.3).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from _common import log

import numpy as np

from months.configs import CANONICAL_GEOMETRY_ID, CANONICAL_TASK_ID, load_configs
from months.data import BANDS
from months.evaluate import band_accuracy, seed_band_statistics
from months.paths import CACHE_DIR, CONFIGS_DIR, RESULTS_DIR, RUNS_DIR

EXTENDED_BANDS = {**BANDS, "25-36": (25, 36), "2-36": (2, 36)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", default=str(CONFIGS_DIR))
    parser.add_argument("--runs", default=str(RUNS_DIR))
    parser.add_argument("--out", default=str(RESULTS_DIR / "family_bands.json"))
    args = parser.parse_args()

    configs = load_configs(Path(args.configs))
    summary = {}
    missing = []
    for config_id, config in configs.items():
        stats = seed_band_statistics(Path(args.runs) / config_id, EXTENDED_BANDS)
        if not stats:
            missing.append(config_id)
            continue
        summary[config_id] = {"family": config["family"], "config": config, "bands": stats}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=1, sort_keys=True))
    log(f"{len(summary)} configs summarized, {len(missing)} without runs -> {out}")
    if missing:
        log(f"missing: {missing[:8]}{' ...' if len(missing) > 8 else ''}")
    print_main_numbers(summary, Path(args.runs).parent)


def print_main_numbers(summary: dict, root: Path) -> None:
    """Overall accuracy (sums 2-96) of the models and patching conditions in Section 3."""
    print("\nmain text (accuracy on sums 2-96)")
    ladder = root / "cache" / "ladder.json"
    if ladder.exists():
        per_sum = json.loads(ladder.read_text())["results"]["base"]["per_sum"]
        print(f"  base model                      {100 * band_accuracy(per_sum, 2, 96):5.1f}%")
    for label, config_id in (("task LoRA", CANONICAL_TASK_ID), ("geometry LoRA", CANONICAL_GEOMETRY_ID)):
        if config_id in summary:
            band = summary[config_id]["bands"]["2-96"]
            print(f"  {label:31s} {100 * band['mean']:5.1f}% (sd {100 * band['sd']:.1f}, {band['n_seeds']} seeds)")
    for path in sorted((root / "results").glob("patch_planes_L18__*__input_offset_2_5_10_20.json")):
        payload = json.loads(path.read_text())
        results, donor = payload["results"], payload["donor"]
        random = np.mean([v["2-96"] for k, v in results.items() if k.startswith("base_from_donor_random")])
        diff = next((v for k, v in results.items() if k.startswith("base_from_donor_diff")), None)
        donor = {CANONICAL_TASK_ID: "task LoRA", CANONICAL_GEOMETRY_ID: "geometry LoRA"}.get(donor, donor)
        print(f"  patching from {donor}: Fourier planes {100 * results['base_from_donor_planes']['2-96']:.1f}%, "
              f"random subspace {100 * random:.1f}%"
              + (f", top-16 difference directions {100 * diff['2-96']:.1f}%" if diff else ""))


if __name__ == "__main__":
    main()
