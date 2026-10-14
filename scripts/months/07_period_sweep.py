#!/usr/bin/env python3
"""Ring R^2 of every Fourier plane, for every period, layer and model condition.

For each condition the 1074 evaluation prompts are run once, capturing the
residual stream at every requested layer. Each activation set is projected
through the probe plane of every (carrier, period) available at that layer
and the Procrustes ring R^2 against the ideal ring is recorded, along with the
between/within class separation. Output:

    cache/period_sweep_r2_resid.json
        {"conditions": {<name>: {"r2__L<layer>__<carrier>": {"<period>": r2, ...},
                                  "sep__L<layer>__<carrier>": {...}}}, ...}
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from _common import log

from months import data
from months.adapters import load_condition
from months.capture import capture_many
from months.metrics import class_separation, ring_r2
from common.model import enable_determinism, load_base_model, load_tokenizer, runtime_fingerprint
from months.paths import CACHE_DIR, RUNS_DIR, PROBES_DIR
from months.probes import ProbeStore

DEFAULT_CONDITIONS = ["base", "task_L03_r08", "geometry_e04_s18"] + [
    f"ladder_p{cap:03d}" for cap in (12, 24, 36, 48, 60, 72, 84, 96)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--conditions", nargs="+", default=DEFAULT_CONDITIONS)
    parser.add_argument("--layers", nargs="+", type=int, default=list(range(32)))
    parser.add_argument("--site", default="resid")
    parser.add_argument("--period-range", nargs=2, type=int, default=[2, 151], metavar=("MIN", "MAX_EXCLUSIVE"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--probe-root", default=str(PROBES_DIR))
    parser.add_argument("--runs", default=str(RUNS_DIR))
    parser.add_argument("--out", default=str(CACHE_DIR / "period_sweep_r2_resid.json"))
    args = parser.parse_args()

    periods = list(range(args.period_range[0], args.period_range[1]))
    probes = ProbeStore(Path(args.probe_root))
    planes = {}
    for layer in args.layers:
        for carrier in data.CARRIERS:
            for period in periods:
                plane = probes.plane(carrier, args.site, layer, period)
                if plane is not None:
                    planes[(layer, carrier, period)] = plane
    log(f"{len(planes)} planes loaded from {probes.root}")

    enable_determinism()
    tokenizer = load_tokenizer()
    model = load_base_model("cuda")
    rows, prompts = data.evaluation_grid()
    values = {carrier: data.carrier_values(carrier, rows) for carrier in data.CARRIERS}
    pairs = [(layer, args.site) for layer in args.layers]

    output = {"site": args.site, "layers": args.layers, "periods": periods, "n_prompts": len(prompts),
              "probe_root": probes.describe(), "conditions": {}, "runtime": runtime_fingerprint()}
    for name in args.conditions:
        condition = load_condition(name, seed=args.seed, runs_root=Path(args.runs))
        with condition.applied(model):
            activations = capture_many(model, tokenizer, prompts, pairs, batch_size=32)
        record = {}
        for layer in args.layers:
            hidden = activations[(layer, args.site)].numpy().astype(np.float32)
            for carrier in data.CARRIERS:
                r2_by_period, separation_by_period = {}, {}
                for period in periods:
                    plane = planes.get((layer, carrier, period))
                    if plane is None:
                        continue
                    weight, bias = plane
                    points = hidden @ weight.T + bias[None, :]
                    r2 = ring_r2(points, values[carrier], period)
                    if r2 is not None:
                        r2_by_period[str(period)] = r2
                    separation = class_separation(points, values[carrier], period)
                    if separation is not None:
                        separation_by_period[str(period)] = separation
                record[f"r2__L{layer}__{carrier}"] = r2_by_period
                record[f"sep__L{layer}__{carrier}"] = separation_by_period
        output["conditions"][name] = record
        peak = record.get("r2__L18__sum", {})
        best = max(peak, key=peak.get) if peak else "-"
        log(f"{name:24s} L18 sum: peak period {best}, R^2(10) = {peak.get('10', float('nan')):.3f}")
        del condition
        torch.cuda.empty_cache()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=1))
    log(f"-> {out}")


if __name__ == "__main__":
    main()
