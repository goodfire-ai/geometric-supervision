#!/usr/bin/env python3
"""Train adapters for a list of (config, seed) tasks with one resident model.

Tasks are every seed of the --config-ids, or of every committed config with --all;
--shard i --of n keeps every n-th task, so n processes can split the work. Completed runs (result.json present) are
skipped unless --force is given. Plane records are read from planes/<set> where
<set> is the config's "planes" field; the control sets take the seed's own
directory (planes/random_seed42, ...).

    scripts/months/04_train.py --config-ids geometry_e04_s18 task_L03_r08
    scripts/months/04_train.py --all --shard 3 --of 8
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from _common import log, shard

from months.configs import load_configs
from common.model import enable_determinism, load_base_model, load_tokenizer
from months.data import month_token_ids
from months.paths import CONFIGS_DIR, PLANES_DIR, RUNS_DIR
from months.train import run_is_complete, run_training


def planes_root_for(config: dict, seed: int, planes_dir: Path) -> Path | None:
    if config["method"] != "geometry":
        return None
    plane_set = config["planes"]
    if plane_set in ("random", "span"):
        return planes_dir / f"{plane_set}_seed{seed}"
    return planes_dir / plane_set


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-ids", nargs="*", default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=None, help="override the configs' seeds")
    parser.add_argument("--all", action="store_true", help="every config in configs/months")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--of", type=int, default=1)
    parser.add_argument("--configs", default=str(CONFIGS_DIR))
    parser.add_argument("--planes", default=str(PLANES_DIR))
    parser.add_argument("--runs", default=str(RUNS_DIR))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--snapshot-steps", nargs="*", type=int, default=[])
    args = parser.parse_args()

    configs = load_configs(Path(args.configs))
    tasks: list[tuple[str, int]] = []
    for config_id in (list(configs) if args.all else args.config_ids or []):
        seeds = args.seeds or configs[config_id]["seeds"]
        tasks.extend((config_id, int(seed)) for seed in seeds)
    if not tasks:
        raise SystemExit("nothing to do: pass --config-ids or --all")
    tasks = shard(tasks, args.shard, args.of)
    pending = [(cid, seed) for cid, seed in tasks
               if args.force or not run_is_complete(Path(args.runs) / cid / f"seed-{seed}")]
    log(f"{len(tasks)} tasks in this shard, {len(pending)} to run")
    if not pending:
        return

    enable_determinism()
    tokenizer = load_tokenizer()
    model = load_base_model("cuda")
    month_ids = month_token_ids(tokenizer)
    for index, (config_id, seed) in enumerate(pending, 1):
        config = configs[config_id]
        out_dir = Path(args.runs) / config_id / f"seed-{seed}"
        started = time.time()
        result = run_training(config, seed, model, tokenizer, month_ids,
                              planes_root_for(config, seed, Path(args.planes)), out_dir,
                              snapshot_steps=tuple(args.snapshot_steps))
        bands = result["bands"]
        summary = "  ".join(f"{name}={bands[name]['accuracy']:.4f}"
                            for name in ("2-24", "25-48", "49-96", "2-96") if name in bands)
        log(f"[{index}/{len(pending)}] {config_id} seed {seed}  {summary}  ({time.time() - started:.0f}s)")
    log("done")


if __name__ == "__main__":
    main()
