#!/usr/bin/env python3
"""Fit the Fourier probes (Feucht et al.'s recipe) from the base model.

Two stages, so the expensive forward pass is done once:

    --stage harvest   run the 39,601 addition prompts through the model once and
                      save the last-token activation at every requested
                      (layer, site) to <out>/activations/L<layer>__<site>.pt
    --stage fit       for each (layer, site), fit all probes at once from the
                      saved activations and write <out>/L<layer>__<site>.pt

Default pairs: the residual stream at every layer (for the R^2 heatmap and the
period sweeps) and the MLP input at layer 18 (for the appendix projections).
Use --layers / --sites to shard the fit stage across GPUs.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from _common import log

from months.capture import capture_many
from common.model import enable_determinism, load_base_model, load_tokenizer, runtime_fingerprint
from months.paths import PROBES_DIR
from months.probes import fit
from months.probes.store import save_layer_probes


def default_pairs() -> list[tuple[int, str]]:
    return [(layer, "resid") for layer in range(32)] + [(18, "mlp_input")]


def parse_pairs(layers: list[int] | None, sites: list[str] | None) -> list[tuple[int, str]]:
    if layers is None and sites is None:
        return default_pairs()
    layers = layers if layers is not None else list(range(32))
    sites = sites if sites is not None else ["resid"]
    return [(layer, site) for layer in layers for site in sites]


def harvest(pairs: list[tuple[int, str]], out_dir: Path, batch_size: int) -> None:
    enable_determinism()
    tokenizer = load_tokenizer()
    model = load_base_model("cuda")
    prompts, first, second = fit.addition_prompts()
    lengths = {len(tokenizer(prompt).input_ids) for prompt in prompts[:2000]}
    if lengths != {5}:
        raise RuntimeError(f"expected every prompt to be 5 tokens, saw lengths {sorted(lengths)}")
    log(f"harvesting {len(prompts)} prompts at {len(pairs)} (layer, site) pairs, batch {batch_size}")
    started = time.time()
    activations = capture_many(model, tokenizer, prompts, pairs, batch_size=batch_size,
                               dtype=torch.bfloat16)
    activations_dir = out_dir / "activations"
    activations_dir.mkdir(parents=True, exist_ok=True)
    for (layer, site), tensor in activations.items():
        torch.save(tensor, activations_dir / f"L{layer}__{site}.pt")
    (activations_dir / "meta.json").write_text(json.dumps({
        "prompt_template": fit.PROMPT_TEMPLATE, "operand_range": [fit.OPERAND_MIN, fit.OPERAND_MAX],
        "n_prompts": len(prompts), "first_operand": first.tolist()[:5], "batch_size": batch_size,
        "dtype": "bfloat16", "pairs": [list(pair) for pair in pairs],
        "runtime": runtime_fingerprint(), "seconds": round(time.time() - started, 1)}, indent=1))
    log(f"harvest done in {time.time() - started:.0f}s -> {activations_dir}")


def fit_pairs(pairs: list[tuple[int, str]], out_dir: Path, device: str) -> None:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    _, first, second = fit.addition_prompts()
    values = fit.variable_values(first, second)
    slots = fit.probe_slots(values)
    log(f"{len(slots)} probes per (layer, site)")
    for layer, site in pairs:
        path = out_dir / "activations" / f"L{layer}__{site}.pt"
        if not path.is_file():
            raise SystemExit(f"missing {path}; run --stage harvest first")
        started = time.time()
        activations = torch.load(path).to(torch.float32)
        result = fit.fit_probes_batched(activations, slots, values, device=device)
        planes = fit.planes_from_fit(slots, result)
        report = {
            "layer": layer, "site": site, "n_probes": len(slots),
            "recipe": {"operands": [fit.OPERAND_MIN, fit.OPERAND_MAX], "lr": fit.LEARNING_RATE,
                       "epochs": fit.EPOCHS, "test_fraction": fit.TEST_FRACTION,
                       "split_seed": fit.SPLIT_SEED, "init_seed": fit.INIT_SEED,
                       "fit": "all probes of a layer trained jointly; equivalent to independent per-probe Adam"},
            "r2_by_plane": {key: entry["r2"] for key, entry in planes.items()},
            "untrained_slots": [slot.__dict__ for slot, trained in zip(slots, result["trained"].tolist()) if not trained],
            "runtime": runtime_fingerprint(), "seconds": round(time.time() - started, 1),
        }
        save_layer_probes(out_dir, layer, site, planes, report)
        median_r2 = torch.nanmedian(result["r2"]).item()
        log(f"L{layer:>2} {site:<9} {len(planes)} planes, median test R^2 {median_r2:.3f}, {time.time() - started:.0f}s")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["harvest", "fit", "both"], default="both")
    parser.add_argument("--layers", type=int, nargs="*", default=None)
    parser.add_argument("--sites", nargs="*", default=None, choices=["resid", "mlp_input"])
    parser.add_argument("--out", default=str(PROBES_DIR))
    parser.add_argument("--batch-size", type=int, default=16,
                        help="harvest batch size; every prompt is 5 tokens so batching never pads")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    out_dir = Path(args.out)
    if args.stage in ("harvest", "both"):
        harvest(default_pairs() if args.stage == "both" else parse_pairs(args.layers, args.sites),
                out_dir, args.batch_size)
    if args.stage in ("fit", "both"):
        fit_pairs(parse_pairs(args.layers, args.sites), out_dir, args.device)


if __name__ == "__main__":
    main()
