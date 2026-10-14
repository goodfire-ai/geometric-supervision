#!/usr/bin/env python3
"""Linebreaking error of the base model and the LoRAs (Fig. 4).

By default seed 1 of each LoRA at every 5th width from 20 to 300; --paper scores all three
seeds at every width from 1 to 300, as in the paper.

Resumable: widths already in outputs/linebreak/eval/errors_<model>.csv are skipped. To
spread the work over several GPUs, run one process per shard with --shard i --of n
(each writes eval/_parts/part_<i>.csv), then fold the parts in with --merge.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from linebreak import evaluate, paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", type=Path, default=paths.CONFIGS_DIR / "adapters.jsonl")
    parser.add_argument("--adapters", type=Path, default=paths.adapters_dir())
    parser.add_argument("--num-docs", type=int, default=300)
    parser.add_argument("--widths", type=int, nargs="+", default=evaluate.WIDTHS)
    parser.add_argument("--seeds", type=int, nargs="+", default=evaluate.EVAL_SEEDS,
                        help="which seeds of each LoRA to score")
    parser.add_argument("--paper", action="store_true", help="all seeds and every width, as in the paper")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--of", type=int, default=1)
    parser.add_argument("--merge", action="store_true", help="fold eval/_parts/*.csv into the per-model CSVs")
    args = parser.parse_args()
    if args.paper:
        args.seeds, args.widths = evaluate.PAPER_SEEDS, evaluate.PAPER_WIDTHS
    if args.merge:
        evaluate.merge_shards(args.configs, args.adapters)
    elif args.of > 1:
        evaluate.run_shard(args.configs, args.adapters, args.num_docs, args.widths,
                           evaluate.MAX_SEQ_LEN, args.shard, args.of, args.seeds)
    else:
        evaluate.evaluate_all(args.configs, args.adapters, args.num_docs, args.widths, evaluate.MAX_SEQ_LEN,
                              args.seeds)


if __name__ == "__main__":
    main()
