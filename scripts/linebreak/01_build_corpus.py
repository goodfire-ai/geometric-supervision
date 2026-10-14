#!/usr/bin/env python3
"""Stream FineWeb once and write the eval / train / diag document pools (CPU; network the first time)."""
from __future__ import annotations

import argparse

from linebreak import corpus, paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="only verify an existing corpus")
    args = parser.parse_args()
    if not args.check:
        corpus.build(paths.corpus_dir())
    corpus.verify(paths.corpus_dir())


if __name__ == "__main__":
    main()
