#!/usr/bin/env python3
"""Adapters on many layers at once, for the single-layer vs multi-layer comparison.

Same recipe as the single-layer runs (rank 1, 200 steps, AdamW) but the LoRA
spans layers 0..upto on the MLP projections and, with --attention, the
attention projections as well. Geometry runs are supervised at layer 18 with
the canonical planes. Output: results/multilayer_<method>_<tag>.json with
one row per (learning rate, scaling, seed).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from _common import log

from months import data
from months.configs import GEOMETRY_DEFAULTS
from months.evaluate import evaluate
from common.lora import ATTENTION_MODULES, MLP_MODULES, attach_lora, init_lora, parameter_count, trainable_tensors
from months.losses import cross_entropy_on_answer
from common.model import enable_determinism, load_base_model, load_tokenizer, runtime_fingerprint
from months.data import month_token_ids
from months.paths import RESULTS_DIR, PLANES_DIR
from months.planes import load_plane_records
from months.train import geometry_loss_for_batch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["task", "geometry"], required=True)
    parser.add_argument("--upto", type=int, required=True, help="adapt layers 0..upto inclusive")
    parser.add_argument("--attention", action="store_true", help="also adapt q/k/v/o projections")
    parser.add_argument("--rank", type=int, default=1)
    parser.add_argument("--lrs", nargs="+", type=float, default=[5e-4])
    parser.add_argument("--scalings", nargs="+", type=float, default=[2.0])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--planes", default=str(PLANES_DIR / "canonical"))
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    tag = f"L0-{args.upto}_{'mlp_attn' if args.attention else 'mlp'}"
    out = Path(args.out) if args.out else RESULTS_DIR / f"multilayer_{args.method}_{tag}.json"

    enable_determinism()
    tokenizer = load_tokenizer()
    model = load_base_model("cuda")
    month_ids = month_token_ids(tokenizer)
    train_rows, train_prompts = data.training_grid()
    eval_rows, eval_prompts = data.evaluation_grid()
    train_answers, eval_answers, eval_sums = data.answers(train_rows), data.answers(eval_rows), data.sums(eval_rows)
    modules = list(MLP_MODULES) + (list(ATTENTION_MODULES) if args.attention else [])
    layers = list(range(0, args.upto + 1))

    geometry_config = plane_records = values_by_carrier = None
    if args.method == "geometry":
        geometry_config = {**GEOMETRY_DEFAULTS}
        plane_records = load_plane_records(Path(args.planes), geometry_config["supervise_layer"], geometry_config["site"])
        values_by_carrier = {carrier: data.carrier_values(carrier, train_rows) for carrier in geometry_config["carriers"]}

    runs = []
    for lr in args.lrs:
        for scaling in args.scalings:
            for seed in args.seeds:
                started = time.time()
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                np.random.seed(seed)
                params = init_lora(model, layers, modules, args.rank, seed)
                tensors = trainable_tensors(params)
                optimizer = torch.optim.AdamW(tensors, lr=lr, weight_decay=0.01)
                remove = attach_lora(model, params, scaling)
                try:
                    for _ in range(args.steps):
                        optimizer.zero_grad(set_to_none=True)
                        if args.method == "task":
                            cross_entropy_on_answer(model, tokenizer, train_prompts, train_answers, month_ids)
                        else:
                            loss, _ = geometry_loss_for_batch(model, tokenizer, train_prompts, geometry_config,
                                                              plane_records, values_by_carrier)
                            loss.backward()
                        torch.nn.utils.clip_grad_norm_(tensors, 1.0)
                        optimizer.step()
                    with torch.no_grad():
                        evaluation = evaluate(model, tokenizer, eval_prompts, eval_answers, month_ids)
                finally:
                    remove()
                bands = data.band_summary(evaluation["predictions"], eval_answers, eval_sums)["bands"]
                runs.append({"method": args.method, "layers": layers, "modules": modules, "rank": args.rank,
                             "lr": lr, "scaling": scaling, "seed": seed, "n_params": parameter_count(params),
                             "bands": bands, "seconds": round(time.time() - started, 1)})
                log(f"lr {lr:g} scaling {scaling:g} seed {seed}: " + "  ".join(
                    f"{b}={bands[b]['accuracy']:.4f}" for b in ("2-24", "25-48", "49-96", "2-96")))
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(json.dumps({"runs": runs, "runtime": runtime_fingerprint()}, indent=1))
    log(f"{len(runs)} runs -> {out}")


if __name__ == "__main__":
    main()
