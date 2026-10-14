#!/usr/bin/env python3
"""Per-token layer-5 activations on the eval pool for the base model and the seed-1
task and geometry adapters, for the helix-fit figures (1 GPU)."""
from __future__ import annotations

import argparse

from linebreak import harvest, paths

MODELS = {"base": None, "task_lora": "task_lora/seed-1", "geometry_lora": "geometry_lora/seed-1"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    parser.add_argument("--num-docs", type=int, default=500)
    parser.add_argument("--max-count", type=int, default=300)
    parser.add_argument("--cap", type=int, default=150, help="at most this many tokens per count")
    args = parser.parse_args()
    for tag in args.models:
        adapter = None if MODELS[tag] is None else str(paths.adapters_dir() / MODELS[tag])
        harvest.run_cloud(adapter, tag, args.num_docs, args.max_count, args.cap)


if __name__ == "__main__":
    main()
