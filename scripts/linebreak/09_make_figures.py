#!/usr/bin/env python3
"""Draw every linebreak figure into outputs/linebreak/figures (CPU)."""
from __future__ import annotations

import argparse

from linebreak import figures, paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="*", default=None, choices=list(figures.FIGURES))
    args = parser.parse_args()
    for name in args.only or list(figures.FIGURES):
        figures.FIGURES[name](paths.OUTPUTS_DIR)


if __name__ == "__main__":
    main()
