"""One training run: a config and a seed in, a trained adapter and its scores out.

Both objectives share the same recipe: a LoRA on one layer's MLP projections,
AdamW with a constant learning rate, 200 full-batch steps over the 210 training
prompts, then evaluation on the 1074-prompt grid. They differ only in the loss:

    task        cross-entropy on the correct month token at the answer position
    geometry    the geometry loss on the supervised Fourier planes, averaged over
                planes, with no term on the model's output

Outputs written to <out_dir>:
    result.json        config, bands, per-sum counts, loss history, runtime
    predictions.npz    month logits, predictions, NLL and sum for every eval prompt
    adapter.safetensors        the trained adapter (A and B for each module)
"""
from __future__ import annotations

import json
import math
import os
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from . import data
from .capture import capture_with_grad
from .evaluate import evaluate
from common.lora import attach_lora, init_lora, save_adapter, trainable_tensors
from .losses import LOSS_FUNCTIONS, cross_entropy_on_answer
from common.model import runtime_fingerprint
from .paths import OUTPUTS_DIR
from .planes import load_plane_records, record_key

LOG_EVERY = 25


def geometry_loss_for_batch(model: Any, tokenizer: Any, prompts: list[str], config: dict,
                            plane_records: dict, values_by_carrier: dict[str, np.ndarray]
                            ) -> tuple[torch.Tensor, dict]:
    """Mean over the config's planes of the chosen geometry loss."""
    activations = capture_with_grad(model, tokenizer, prompts,
                                    int(config["supervise_layer"]), config["site"])
    loss_function = LOSS_FUNCTIONS[config["loss"]]
    device = activations.device
    total = None
    diagnostics = {}
    for carrier in config["carriers"]:
        for period in config["periods"]:
            record = plane_records[record_key(carrier, period)]
            weight = record["weight"].to(device, torch.float32)
            bias = record["bias"].to(device, torch.float32)
            points = activations.float() @ weight.t() + bias[None, :]
            loss, info = loss_function(
                points, values_by_carrier[carrier], int(period), record,
                float(config["radius_scale"]), float(config["spread_ratio"]),
                float(config["spread_weight"]), config["target_frame"],
                bool(config["shuffled_targets"]))
            total = loss if total is None else total + loss
            diagnostics[record_key(carrier, period)] = info
    n_planes = len(config["carriers"]) * len(config["periods"])
    return total / n_planes, diagnostics


def run_training(config: dict, seed: int, model: Any, tokenizer: Any, month_ids: list[int],
                 planes_root: Path | None, out_dir: Path, snapshot_steps: tuple[int, ...] = ()
                 ) -> dict:
    """Train one adapter, evaluate it, and write the outputs. Returns the result dict."""
    device = model.device
    method = config["method"]
    train_rows, train_prompts = data.training_grid(int(config["train_max_sum"]))
    eval_rows, eval_prompts = data.evaluation_grid(int(config["eval_max_sum"]))
    train_answers = data.answers(train_rows)
    eval_answers = data.answers(eval_rows)
    eval_sums = data.sums(eval_rows)

    plane_records = values_by_carrier = None
    if method == "geometry":
        if planes_root is None:
            raise ValueError("geometry configs need a planes root")
        plane_records = load_plane_records(planes_root, int(config["supervise_layer"]), config["site"])
        values_by_carrier = {carrier: data.carrier_values(carrier, train_rows)
                             for carrier in config["carriers"]}
    elif method != "task":
        raise ValueError(f"unknown method {config['method']!r}")

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    params = init_lora(model, [int(config["edit_layer"])], list(config["lora_modules"]),
                       int(config["rank"]), seed)
    tensors = trainable_tensors(params)
    optimizer = torch.optim.AdamW(tensors, lr=float(config["lr"]),
                                  weight_decay=float(config["weight_decay"]))
    scaling = float(config["scaling"])
    steps = int(config["steps"])

    history = []
    snapshots = {}
    started = time.time()
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        remove = attach_lora(model, params, scaling)
        try:
            if method == "task":
                loss_value = cross_entropy_on_answer(model, tokenizer, train_prompts,
                                                     train_answers, month_ids)
            else:
                loss, _ = geometry_loss_for_batch(model, tokenizer, train_prompts, config,
                                                  plane_records, values_by_carrier)
                loss.backward()
                loss_value = float(loss.detach())
        finally:
            remove()
        if not math.isfinite(loss_value):
            raise FloatingPointError(f"non-finite loss at step {step}: {loss_value}")
        grad_norm = float(torch.nn.utils.clip_grad_norm_(tensors, float(config["grad_clip"])))
        optimizer.step()
        if step % LOG_EVERY == 0 or step == 1 or step == steps:
            history.append({"step": step, "loss": loss_value, "grad_norm": grad_norm})
        if step in snapshot_steps and step != steps:
            snapshots[step] = {key: (a.detach().clone(), b.detach().clone())
                               for key, (a, b) in params.items()}
    train_seconds = time.time() - started

    started = time.time()
    remove = attach_lora(model, params, scaling)
    try:
        evaluation = evaluate(model, tokenizer, eval_prompts, eval_answers, month_ids)
    finally:
        remove()
    eval_seconds = time.time() - started
    summary = data.band_summary(evaluation["predictions"], eval_answers, eval_sums)

    result = {
        "config_id": config["config_id"],
        "config_hash": config["config_hash"],
        "seed": seed,
        "config": config,
        "bands": summary["bands"],
        "per_sum": summary["per_sum"],
        "loss_history": history,
        "wall_time_s": {"train": round(train_seconds, 1), "eval": round(eval_seconds, 1)},
        "planes_root": _relative(planes_root) if planes_root else None,
        "runtime": runtime_fingerprint(),
    }
    _write_outputs(Path(out_dir), result, evaluation, eval_sums, params, scaling, snapshots)
    return result


def _write_outputs(out_dir: Path, result: dict, evaluation: dict, eval_sums: np.ndarray,
                   params: dict, scaling: float, snapshots: dict) -> None:
    """Write into a temporary directory and rename it, so a run is complete iff result.json exists."""
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = out_dir.with_name(out_dir.name + ".tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    np.savez_compressed(temporary / "predictions.npz",
                        month_logits=evaluation["month_logits"],
                        predictions=evaluation["predictions"],
                        nll=evaluation["nll"], sum=eval_sums)
    save_adapter(temporary / "adapter.safetensors", params, scaling)
    for step, snapshot in snapshots.items():
        save_adapter(temporary / f"adapter_step{step}.safetensors", snapshot, scaling)
    (temporary / "result.json").write_text(json.dumps(result, indent=1, sort_keys=True))
    if out_dir.exists():
        shutil.rmtree(out_dir)
    os.replace(temporary, out_dir)


def _relative(path: Path) -> str:
    """The path relative to outputs/months when it is inside it, so results carry no machine paths."""
    try:
        return str(Path(path).resolve().relative_to(OUTPUTS_DIR.resolve()))
    except ValueError:
        return Path(path).name


def run_is_complete(out_dir: Path) -> bool:
    return (Path(out_dir) / "result.json").is_file()
