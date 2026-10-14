"""Scoring a model on the months grid.

A prediction is the argmax over the twelve month tokens at the answer position
(restricted argmax). The model's top token is a month for over 99% of prompts,
so this simplification changes little.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .data import BANDS, band_summary

EVAL_BATCH = 16


@torch.no_grad()
def evaluate(model: Any, tokenizer: Any, prompts: list[str], answer_indices: np.ndarray,
             month_ids: list[int]) -> dict[str, np.ndarray]:
    """Month logits, restricted-argmax predictions, and NLL of the correct month."""
    month_ids_tensor = torch.tensor(month_ids, device=model.device)
    month_logits, predictions, nll = [], [], []
    for start in range(0, len(prompts), EVAL_BATCH):
        encoded = tokenizer(prompts[start:start + EVAL_BATCH], return_tensors="pt",
                            padding=True).to(model.device)
        logits = model(input_ids=encoded.input_ids,
                       attention_mask=encoded.attention_mask).logits[:, -1, :].float()
        candidates = logits.index_select(1, month_ids_tensor)
        answers = torch.as_tensor(answer_indices[start:start + candidates.shape[0]],
                                  device=model.device)
        target_ids = month_ids_tensor[answers]
        log_probs = torch.log_softmax(logits, dim=-1)
        nll.append((-log_probs.gather(1, target_ids[:, None])[:, 0]).cpu().numpy())
        month_logits.append(candidates.cpu().numpy())
        predictions.append(candidates.argmax(dim=1).cpu().numpy())
    return {
        "month_logits": np.concatenate(month_logits).astype(np.float32),
        "predictions": np.concatenate(predictions).astype(np.int64),
        "nll": np.concatenate(nll).astype(np.float32),
    }


def summarize(evaluation: dict, answer_indices: np.ndarray, sum_values: np.ndarray) -> dict:
    return band_summary(evaluation["predictions"], answer_indices, sum_values)


# ---------------------------------------------------------------------------
# reading saved runs
# ---------------------------------------------------------------------------

def read_result(run_dir: Path) -> dict:
    return json.loads((Path(run_dir) / "result.json").read_text())


def seed_dirs(config_dir: Path) -> list[Path]:
    return sorted(path for path in Path(config_dir).glob("seed-*") if (path / "result.json").is_file())


def pooled_per_sum(run_dirs: list[Path]) -> dict[str, dict[str, int]]:
    """Per-sum counts pooled over runs (sum the counts, do not average rates)."""
    pooled: dict[str, dict[str, int]] = defaultdict(lambda: {"n": 0, "correct": 0})
    for run_dir in run_dirs:
        for value, entry in read_result(run_dir)["per_sum"].items():
            pooled[value]["n"] += int(entry["n"])
            pooled[value]["correct"] += int(entry["correct"])
    return {value: dict(entry) for value, entry in sorted(pooled.items(), key=lambda item: int(item[0]))}


def band_accuracy(per_sum: dict, low: int, high: int) -> float:
    n = sum(entry["n"] for value, entry in per_sum.items() if low <= int(value) <= high)
    n_correct = sum(entry["correct"] for value, entry in per_sum.items() if low <= int(value) <= high)
    return n_correct / n if n else float("nan")


def seed_band_statistics(config_dir: Path, bands: dict | None = None) -> dict[str, dict]:
    """Per-band accuracy mean, standard deviation (ddof=1), and seed count for one config."""
    bands = bands or BANDS
    runs = seed_dirs(config_dir)
    out = {}
    for name, (low, high) in bands.items():
        values = []
        for run_dir in runs:
            per_sum = read_result(run_dir)["per_sum"]
            value = band_accuracy(per_sum, low, high)
            if not np.isnan(value):
                values.append(value)
        if values:
            array = np.asarray(values)
            out[name] = {"mean": float(array.mean()),
                         "sd": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
                         "n_seeds": len(array)}
    return out


def base_model_bands(ladder_path: Path, bands: dict | None = None) -> dict[str, float] | None:
    """The base model's accuracy per band, from the per-sum counts in cache/ladder.json.

    Returns None when the file does not exist yet (the ladder stage has not run).
    """
    ladder_path = Path(ladder_path)
    if not ladder_path.is_file():
        return None
    per_sum = json.loads(ladder_path.read_text())["results"]["base"]["per_sum"]
    bands = bands or {**BANDS, "25-36": (25, 36), "2-36": (2, 36)}
    return {name: band_accuracy(per_sum, low, high) for name, (low, high) in bands.items()}
