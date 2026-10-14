"""The FineWeb document pools the whole linebreak pipeline draws from.

A single filtered stream of FineWeb is sliced into three disjoint pools written to
disk once, so every later step loads a fast, stable, non-overlapping split:

    eval   linebreaking error (Fig. 4) and the ablations (Fig. 5c)
    train  fine-tuning the adapters
    diag   fitting the probes and the per-count means

Documents are kept only if they contain no newline and are longer than ``MIN_LEN``
characters. Order is the stream order (no shuffle), so the split is reproducible
given the pinned dataset revision.
"""
from __future__ import annotations

import json
from itertools import islice
from pathlib import Path

DATASET = "HuggingFaceFW/fineweb"
DATASET_REVISION = "9bb295ddab0e05d785b879661af7260fed5140fc"
MIN_LEN = 2500
N_EVAL, N_TRAIN, N_DIAG = 500, 4000, 1200
SPLITS = ("eval", "train", "diag")


def build(out_dir: Path, n_eval: int = N_EVAL, n_train: int = N_TRAIN, n_diag: int = N_DIAG,
          min_len: int = MIN_LEN) -> None:
    """Stream, filter and slice FineWeb into the three disjoint pools."""
    from datasets import load_dataset

    stream = load_dataset(DATASET, split="train", streaming=True, revision=DATASET_REVISION)
    stream = stream.filter(lambda x: ("\n" not in x["text"]) and (len(x["text"]) > min_len))
    total = n_eval + n_train + n_diag
    docs = [x["text"] for x in islice(stream, total)]
    if len(docs) != total:
        raise RuntimeError(f"stream yielded {len(docs)} docs, expected {total}")

    pools = {"eval": docs[:n_eval],
             "train": docs[n_eval:n_eval + n_train],
             "diag": docs[n_eval + n_train:]}
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, pool in pools.items():
        with open(out_dir / f"{name}_docs.txt", "w") as f:
            for doc in pool:
                f.write(doc + "\n")
        print(f"  {name}: {len(pool)} docs")
    manifest = dict(dataset=DATASET, revision=DATASET_REVISION, min_len=min_len,
                    n_eval=n_eval, n_train=n_train, n_diag=n_diag,
                    order="eval|train|diag from one filtered stream (disjoint)")
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))


def load_docs(path: str | Path) -> list[str]:
    """Read one pool file into a list of newline-free documents."""
    with open(path) as f:
        return [line.rstrip("\n") for line in f]


def verify(out_dir: Path) -> None:
    """Check an on-disk corpus: present, newline-free, disjoint, and the sizes in its manifest."""
    manifest = json.loads((out_dir / "manifest.json").read_text())
    pools = {s: load_docs(out_dir / f"{s}_docs.txt") for s in SPLITS}
    for s in SPLITS:
        if len(pools[s]) != manifest[f"n_{s}"]:
            raise ValueError(f"{s}: {len(pools[s])} docs, manifest says {manifest[f'n_{s}']}")
    sets = {s: set(pools[s]) for s in SPLITS}
    for a, b in (("eval", "train"), ("eval", "diag"), ("train", "diag")):
        if sets[a] & sets[b]:
            raise ValueError(f"{a} and {b} pools overlap")
    print(f"  corpus OK: {', '.join(f'{s}={len(pools[s])}' for s in SPLITS)}, disjoint")
