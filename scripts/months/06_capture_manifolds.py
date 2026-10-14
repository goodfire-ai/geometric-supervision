#!/usr/bin/env python3
"""Project every prompt's activation onto the probe planes, for each model condition.

For each condition (the base model and any trained adapters) and each prompt
set (the 210 training prompts and the 1074 evaluation prompts), the last-token
activation at every requested (layer, site) is projected through every
requested (carrier, period) probe. One file per condition and prompt set:

    cache/manifolds/<condition>__<prompt_set>.npz
        sum, values_<carrier>                the task variables per prompt
        pts_L<layer>_<site>_<carrier>_p<period>   projected points [n, 2]

Files are merged with what is already on disk, so captures can be added a
few planes at a time without losing earlier ones.
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
from common.model import enable_determinism, load_base_model, load_tokenizer, runtime_fingerprint
from months.paths import CACHE_DIR, RUNS_DIR, PROBES_DIR
from months.probes import ProbeStore

DEFAULT_CONDITIONS = ["base", "task_L03_r08", "ladder_p024", "ladder_p048", "ladder_p096", "geometry_e04_s18"]
DEFAULT_PAIRS = ["18,resid", "18,mlp_input", "20,resid", "22,resid"]
DEFAULT_PERIODS = [2, 5, 10, 20, 50]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--conditions", nargs="+", default=DEFAULT_CONDITIONS)
    parser.add_argument("--pairs", nargs="+", default=DEFAULT_PAIRS, help="'layer,site' items")
    parser.add_argument("--carriers", nargs="+", default=list(data.CARRIERS))
    parser.add_argument("--periods", nargs="+", type=int, default=DEFAULT_PERIODS)
    parser.add_argument("--seed", type=int, default=42, help="which seed's adapter to load")
    parser.add_argument("--probe-root", default=str(PROBES_DIR))
    parser.add_argument("--runs", default=str(RUNS_DIR))
    parser.add_argument("--out", default=str(CACHE_DIR / "manifolds"))
    args = parser.parse_args()

    pairs = [(int(layer), site) for layer, site in (item.split(",") for item in args.pairs)]
    probes = ProbeStore(Path(args.probe_root))
    planes = {}
    for layer, site in pairs:
        for carrier in args.carriers:
            for period in args.periods:
                plane = probes.plane(carrier, site, layer, period)
                if plane is not None:
                    planes[(layer, site, carrier, period)] = plane
    log(f"{len(planes)} planes over {len(pairs)} (layer, site) pairs from {probes.root}")

    enable_determinism()
    tokenizer = load_tokenizer()
    model = load_base_model("cuda")
    prompt_sets = {"train210": data.training_grid(), "grid1074": data.evaluation_grid()}
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for name in args.conditions:
        condition = load_condition(name, seed=args.seed, runs_root=Path(args.runs))
        for set_name, (rows, prompts) in prompt_sets.items():
            with condition.applied(model):
                activations = capture_many(model, tokenizer, prompts, pairs, batch_size=32)
            arrays = {"sum": data.sums(rows)}
            for carrier in args.carriers:
                arrays[f"values_{carrier}"] = data.carrier_values(carrier, rows)
            for (layer, site, carrier, period), (weight, bias) in planes.items():
                hidden = activations[(layer, site)].numpy().astype(np.float32)
                arrays[f"pts_L{layer}_{site}_{carrier}_p{period}"] = (hidden @ weight.T + bias[None, :]).astype(np.float32)
            path = out_dir / f"{name}__{set_name}.npz"
            if path.is_file():
                with np.load(path) as existing:
                    merged = {key: existing[key] for key in existing.files}
                merged.update(arrays)
                arrays = merged
            np.savez_compressed(path, **arrays)
            log(f"{name:24s} {set_name}: {len([k for k in arrays if k.startswith('pts_')])} planes -> {path.name}")
        del condition
        torch.cuda.empty_cache()

    index_path = out_dir / "index.json"
    index = json.loads(index_path.read_text()) if index_path.is_file() else {"conditions": {}}
    for name in args.conditions:
        index["conditions"][name] = {"seed": args.seed, "pairs": [list(pair) for pair in pairs],
                                     "periods": args.periods, "carriers": args.carriers}
    index["probe_root"] = probes.describe()
    index["runtime"] = runtime_fingerprint()
    index_path.write_text(json.dumps(index, indent=1))


if __name__ == "__main__":
    main()
