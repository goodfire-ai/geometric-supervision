#!/usr/bin/env python3
"""Train and evaluate the geometry-LoRA hyperparameter sweep and the lambda = 0 ablation (Appendix B.3.1).

    configs/linebreak/hparam_sweep.jsonl   rank {1,2,4,8,16}; supervision layer 3-7 (editing blocks
                                           0..layer); alpha_r x tau grid (6 x 6); 3 seeds each
    configs/linebreak/nospread.jsonl       the canonical geometry LoRA with spread_weight 0; 3 seeds

The alternate supervision targets of Appendix B.3.2 use the same runner but are not part of the
default sweep (see the README):

    configs/linebreak/alternate_targets.jsonl            each target at the paper's configuration
    configs/linebreak/alternate_targets_pca_grid.jsonl   the PCA-subspace targets over alpha_r x tau

Each (config, seed) unit is trained (skipped if its adapter exists) and then evaluated at every
10th width from 20 to 300, writing outputs/linebreak/eval/errors_<config>_seed<s>.csv; both steps
resume. Units are round-robined over --of shards (one GPU each). --summarize prints each
configuration's linebreaking error, averaged over seeds (including lambda = 0, which has no figure).
"""
from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

from linebreak import corpus, evaluate, paths, train
from linebreak.figures import cell_mean_l1

SWEEP_WIDTHS = list(range(20, 301, 10))
CONFIGS = [paths.CONFIGS_DIR / "hparam_sweep.jsonl", paths.CONFIGS_DIR / "nospread.jsonl"]


def load(config_files) -> list[dict]:
    return [json.loads(line) for f in config_files for line in Path(f).read_text().splitlines() if line.strip()]


def summarize(configs) -> None:
    import numpy as np
    import pandas as pd
    for config in configs:
        files = [paths.eval_dir() / f"errors_{config['config_id']}_seed{seed}.csv" for seed in config["seeds"]]
        frames = [pd.read_csv(f) for f in files if f.exists()]
        if frames:
            values = np.array([[cell_mean_l1(d), cell_mean_l1(d[d.width <= 150]), cell_mean_l1(d[d.width > 150])]
                               for d in frames])
            print(f"{config['config_id']:32s} {values[:, 0].mean():6.2f} ± {values[:, 0].std():4.2f}   "
                  f"<=150 {values[:, 1].mean():6.2f}   >150 {values[:, 2].mean():6.2f}  ({len(frames)} seeds)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--configs", type=Path, nargs="+", default=CONFIGS)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--of", type=int, default=1)
    parser.add_argument("--num-docs", type=int, default=300)
    parser.add_argument("--summarize", action="store_true", help="print the error of each configuration")
    args = parser.parse_args()

    configs = load(args.configs)
    for config in configs:
        train.check_config(config)
    if args.summarize:
        summarize(configs)
        return 0

    units = [(c, s) for c in configs for s in c["seeds"]][args.shard::args.of]
    print(f"shard {args.shard}/{args.of}: {len(units)} (config, seed) units", flush=True)
    docs = corpus.load_docs(paths.corpus_dir() / "eval_docs.txt")[:args.num_docs]
    failed = 0
    for config, seed in units:
        out = paths.adapters_dir() / config["config_id"] / f"seed-{seed}"
        try:
            if not (out / "adapter.safetensors").exists():
                train.run(config, seed, out)
            evaluate.evaluate_model(str(out), f"{config['config_id']}_seed{seed}", docs, SWEEP_WIDTHS,
                                    evaluate.MAX_SEQ_LEN)
        except Exception:
            failed += 1
            print(f"[FAIL] {config['config_id']} seed {seed}", flush=True)
            traceback.print_exc()
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
