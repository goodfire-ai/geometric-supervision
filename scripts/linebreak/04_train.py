#!/usr/bin/env python3
"""Train every (config, seed) adapter in configs/linebreak/adapters.jsonl (1 GPU per shard).

Units are round-robined over --of shards so several GPUs can split them, and a unit
whose adapter already exists is skipped, so an interrupted run can simply be restarted.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from linebreak import paths, train


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", type=Path, default=paths.CONFIGS_DIR / "adapters.jsonl")
    parser.add_argument("--out", type=Path, default=paths.adapters_dir())
    parser.add_argument("--config-ids", nargs="*", default=None)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--of", type=int, default=1)
    args = parser.parse_args()

    configs = [json.loads(line) for line in args.configs.read_text().splitlines() if line.strip()]
    if args.config_ids:
        configs = [c for c in configs if c["config_id"] in set(args.config_ids)]
    for config in configs:
        train.check_config(config)
    units = [(c, s) for c in configs if c["method"] != "base" for s in c["seeds"]]
    units = units[args.shard::args.of]
    todo = [(c, s) for c, s in units
            if not (args.out / c["config_id"] / f"seed-{s}" / "adapter.safetensors").exists()]
    print(f"shard {args.shard}/{args.of}: {len(units)} units, {len(todo)} to run", flush=True)
    for config, seed in todo:
        train.run(config, seed, args.out / config["config_id"] / f"seed-{seed}")


if __name__ == "__main__":
    main()
