#!/usr/bin/env python3
"""Assemble cache/ladder.json: per-sum accuracy of the base model and of each
training-range ladder adapter on the wide grid (sums 2..132).

Ladder rows pool the seeds by count (sum n and correct across seeds, divide
once). The base model is evaluated here on the same grid, so every number in
the file comes from this repo.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from _common import log

from months import data
from months.configs import load_configs
from months.evaluate import band_accuracy, evaluate, pooled_per_sum, seed_dirs
from common.model import enable_determinism, load_base_model, load_tokenizer, runtime_fingerprint
from months.data import month_token_ids
from months.paths import CACHE_DIR, CONFIGS_DIR, RUNS_DIR


def rates(per_sum: dict, cutoff: int, max_sum: int) -> dict:
    return {"in_train_range": band_accuracy(per_sum, 2, cutoff),
            "beyond_train_range": band_accuracy(per_sum, cutoff + 1, max_sum),
            "overall": band_accuracy(per_sum, 2, max_sum)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", default=str(CONFIGS_DIR))
    parser.add_argument("--runs", default=str(RUNS_DIR))
    parser.add_argument("--out", default=str(CACHE_DIR / "ladder.json"))
    args = parser.parse_args()

    configs = load_configs(Path(args.configs))
    ladder = sorted((c for c in configs.values() if c["family"] == "training_range_ladder"),
                    key=lambda c: c["train_max_sum"])
    max_sum = max(int(c["eval_max_sum"]) for c in ladder)

    enable_determinism()
    tokenizer = load_tokenizer()
    model = load_base_model("cuda")
    month_ids = month_token_ids(tokenizer)
    rows, prompts = data.evaluation_grid(max_sum)
    log(f"evaluating the base model on sums 2..{max_sum} ({len(prompts)} prompts)")
    with torch.no_grad():
        evaluation = evaluate(model, tokenizer, prompts, data.answers(rows), month_ids)
    base_summary = data.band_summary(evaluation["predictions"], data.answers(rows), data.sums(rows))
    results = {"base": {"per_sum": base_summary["per_sum"], **rates(base_summary["per_sum"], max_sum, max_sum),
                        "config_id": None}}
    log(f"base overall {results['base']['overall']:.4f}")

    for config in ladder:
        runs = seed_dirs(Path(args.runs) / config["config_id"])
        if not runs:
            log(f"skip {config['config_id']}: no runs")
            continue
        per_sum = pooled_per_sum(runs)
        cutoff = int(config["train_max_sum"])
        results[config["config_id"]] = {"per_sum": per_sum, **rates(per_sum, cutoff, max_sum),
                                        "config_id": config["config_id"], "train_max_sum": cutoff,
                                        "n_seeds": len(runs)}
        entry = results[config["config_id"]]
        log(f"{config['config_id']}: {len(runs)} seeds, in range {entry['in_train_range']:.4f}, "
            f"beyond {entry['beyond_train_range']:.4f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"results": results, "max_sum": max_sum, "n_prompts": len(prompts),
                               "scoring": "argmax over the twelve month tokens",
                               "runtime": runtime_fingerprint()}, indent=1))
    log(f"-> {out}")


if __name__ == "__main__":
    main()
