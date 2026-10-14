#!/usr/bin/env python3
"""Table for the alternate forms of months supervision (Appendix A.3.3).

For each target (ring, centroids, line, regression), probe source and set of supervised
variables, prints the best configuration of its grid by accuracy on sums 2-96 (mean +/- sd
over seeds), with its accuracy on the held-out sums 37-96.

    16_summarize_alternate.py
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from months.paths import RUNS_DIR

ID = re.compile(r"^alt_(ring|centroids|line|regression)_(?:(months|addition)_)?(in|off|inoff)_")
VARIABLES = {"in": "input", "off": "offset", "inoff": "input + offset"}


def accuracy(per_sum: dict, lo: int, hi: int) -> float:
    rows = [v for k, v in per_sum.items() if lo <= int(k) <= hi]
    return 100 * sum(r["correct"] for r in rows) / sum(r["n"] for r in rows)


def main() -> None:
    best: dict[tuple, tuple] = defaultdict(lambda: (-1.0,))
    for run in sorted(Path(RUNS_DIR).glob("alt_*")):
        match = ID.match(run.name)
        seeds = sorted(run.glob("seed-*/result.json"))
        if not match or not seeds:
            continue
        per_sum = [json.loads(f.read_text())["per_sum"] for f in seeds]
        overall = [accuracy(p, 2, 96) for p in per_sum]
        held_out = [accuracy(p, 37, 96) for p in per_sum]
        target, source, variables = match.groups()
        key = (target, source or "", variables)
        if np.mean(overall) > best[key][0]:
            best[key] = (np.mean(overall), np.std(overall), np.mean(held_out), len(seeds), run.name)
    if not best:
        raise SystemExit(f"no alt_* runs under {RUNS_DIR}")
    print(f"{'target':12s} {'probes':9s} {'variables':15s} {'2-96':>13s} {'37-96':>7s}  seeds  config")
    for (target, source, variables), (mean, sd, held_out, n, name) in sorted(best.items()):
        print(f"{target:12s} {source or '-':9s} {VARIABLES[variables]:15s} {mean:6.1f} ± {sd:4.1f} "
              f"{held_out:7.1f}  {n:5d}  {name}")


if __name__ == "__main__":
    main()
