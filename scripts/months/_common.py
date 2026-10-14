"""Small helpers shared by the numbered scripts (logging, sharding)."""
from __future__ import annotations

import time


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def shard(items: list, index: int, count: int) -> list:
    """Every `count`-th item starting at `index` (one shard per GPU process)."""
    return items[index::count]
