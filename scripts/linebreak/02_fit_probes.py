#!/usr/bin/env python3
"""Fit the helix readout at layer 5: cos/sin Fourier probes at period T and a linear
character-count probe, by ridge regression on diag documents wrapped to width 150 (1 GPU)."""
from __future__ import annotations

import argparse

from linebreak import probes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-docs", type=int, default=250)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    probes.fit(num_docs=args.num_docs, seed=args.seed)


if __name__ == "__main__":
    main()
