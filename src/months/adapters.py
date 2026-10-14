"""Loading a trained adapter back onto the frozen model.

A "condition" is a named model to evaluate: "base" (no adapter) or a config id
whose trained adapter is read from runs/<config_id>/seed-<seed>/adapter.safetensors.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .configs import load_configs
from common.lora import attach_lora, load_adapter
from .paths import CONFIGS_DIR, RUNS_DIR


class Condition:
    """A model condition: the base model, or the base model plus one trained adapter."""

    def __init__(self, name: str, params: dict | None = None, scaling: float = 0.0):
        self.name = name
        self.params = params
        self.scaling = scaling

    @property
    def is_base(self) -> bool:
        return self.params is None

    @contextmanager
    def applied(self, model: Any):
        """Context manager that attaches the adapter (if any) for the duration."""
        remove = attach_lora(model, self.params, self.scaling) if self.params else None
        try:
            yield
        finally:
            if remove is not None:
                remove()


def load_condition(name: str, seed: int = 42, runs_root: Path = RUNS_DIR,
                   configs_dir: Path = CONFIGS_DIR, device: str = "cuda") -> Condition:
    if name.lower() == "base":
        return Condition("base")
    adapter_path = Path(runs_root) / name / f"seed-{seed}" / "adapter.safetensors"
    if not adapter_path.is_file():
        raise FileNotFoundError(f"no trained adapter at {adapter_path}")
    params, scaling = load_adapter(adapter_path, device=device)
    # the saved scaling is authoritative; the config is consulted only as a check
    configs = load_configs(configs_dir)
    if name in configs and abs(float(configs[name]["scaling"]) - scaling) > 1e-6:
        raise ValueError(f"{name}: adapter scaling {scaling} differs from config")
    return Condition(name, params, scaling)
