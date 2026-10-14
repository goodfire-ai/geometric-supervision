#!/usr/bin/env python3
"""Character-count probe R^2 at every layer of the base model (Fig. 5a; 1 GPU)."""
from __future__ import annotations

import argparse

from linebreak import harvest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-docs", type=int, default=400)
    args = parser.parse_args()
    harvest.run_probe_r2_by_layer(args.num_docs)


if __name__ == "__main__":
    main()
