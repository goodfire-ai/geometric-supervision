#!/usr/bin/env python3
"""Zero-ablate the character-count subspace at layer 5 and re-measure linebreaking error (Fig. 5c; 1 GPU)."""
from __future__ import annotations

import argparse

from linebreak import manifold


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-docs", type=int, default=250)
    parser.add_argument("--widths", type=int, nargs="+", default=manifold.ABLATION_WIDTHS)
    args = parser.parse_args()
    manifold.run_ablation(args.num_docs, args.widths)


if __name__ == "__main__":
    main()
