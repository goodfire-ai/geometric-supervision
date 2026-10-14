#!/usr/bin/env python3
"""Per-character-count mean activations at layer 5 on the diag pool (1 GPU).

Counts up to 150 (documents wrapped to width 150) feed the PCA figure; counts up to
300 (width 300) define the ablated subspaces.
"""
from __future__ import annotations

import argparse

from linebreak import harvest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-docs", type=int, default=1000)
    parser.add_argument("--max-counts", type=int, nargs="+", default=[150, 300])
    args = parser.parse_args()
    for max_count in args.max_counts:
        harvest.run_means(args.num_docs, max_count)


if __name__ == "__main__":
    main()
