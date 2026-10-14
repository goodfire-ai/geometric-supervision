"""Where things live.

Committed inputs (experiment configs) are under ``<repo>/configs``. Everything the
pipeline generates goes under ``<repo>/outputs/<task>/``; set ``GEOSUP_OUTPUTS`` to
put it somewhere else. The base model is found through the usual Hugging Face
environment variables (``HF_HOME``, ``HF_HUB_CACHE``, ``HF_HUB_OFFLINE``).
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_ROOT = REPO_ROOT / "configs"


def outputs_root() -> Path:
    return Path(os.environ.get("GEOSUP_OUTPUTS", REPO_ROOT / "outputs"))
