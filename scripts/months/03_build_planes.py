#!/usr/bin/env python3
"""Build plane records: probe directions plus the base model's geometry on them.

The base model is run once on the 210 training prompts; for every requested
(layer, site, carrier, period) the activations are projected through the probe
and the record (centroids, radius, spread, Procrustes frame, shuffle
permutation) is built. Records are written to <out>/L<layer>__<site>.pt.

Plane sets:
    --set canonical   every (layer, site, carrier, period) the configs need,
                      plus the sum planes used for patching (default)
    --set all         input-month and offset planes for every period 2..99 at
                      layer 18 (the planes-ladder experiment)
    --set random      eight random orthonormal 2-D planes per seed (control)
    --set span        eight random orthonormal pairs inside the span of the real
                      eight planes per seed (control)

--probe-root selects where the probes come from (default: outputs/months/probes).
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
from months.capture import capture_many
from months.configs import load_configs
from common.model import enable_determinism, load_base_model, load_tokenizer, runtime_fingerprint
from months.paths import CONFIGS_DIR, PLANES_DIR, PROBES_DIR
from months.planes import build_plane_record, save_plane_records
from months.planes.store import load_plane_records
from months.probes import ProbeStore

CANONICAL_CARRIERS = ["input-month", "offset"]
CANONICAL_PERIODS = [2, 5, 10, 20]
# Planes the patching and projection scripts read beyond what training needs: every
# layer's own input/offset/sum planes at the base-10 periods (multi-layer patching
# walks layers 4..31), the longer periods at layers 18/20/22, and the MLP-input site
# at layer 18 for the appendix projections.
EXTRA_PLANES = {(layer, "resid"): [(carrier, period) for carrier in data.CARRIERS for period in (2, 5, 10, 20)]
                for layer in range(4, 32)}
EXTRA_PLANES[(18, "resid")] += [("sum", 50), ("sum", 99), ("input-month", 50), ("offset", 50),
                                ("input-month", 99), ("offset", 99)]
EXTRA_PLANES[(20, "resid")] += [(carrier, 50) for carrier in data.CARRIERS]
EXTRA_PLANES[(22, "resid")] += [(carrier, 50) for carrier in data.CARRIERS]
EXTRA_PLANES[(18, "mlp_input")] = [(carrier, period) for carrier in data.CARRIERS for period in (2, 5, 10, 20, 50)]


def required_planes(configs: dict) -> dict[tuple[int, str], set[tuple[str, int]]]:
    """(layer, site) -> {(carrier, period)} over every geometry config trained on the canonical planes."""
    needed: dict[tuple[int, str], set] = {}
    for config in configs.values():
        if config["method"] != "geometry" or config["planes"] != "canonical":
            continue
        key = (int(config["supervise_layer"]), config["site"])
        for carrier in config["carriers"]:
            for period in config["periods"]:
                needed.setdefault(key, set()).add((carrier, int(period)))
    for key, extras in EXTRA_PLANES.items():
        needed.setdefault(key, set()).update(extras)
    return needed


def capture_training_prompts(pairs: list[tuple[int, str]]) -> tuple[list[dict], dict]:
    enable_determinism()
    tokenizer = load_tokenizer()
    model = load_base_model("cuda")
    rows, prompts = data.training_grid()
    started = time.time()
    activations = capture_many(model, tokenizer, prompts, pairs, batch_size=64)
    log(f"captured {len(prompts)} training prompts at {len(pairs)} pairs in {time.time() - started:.0f}s")
    del model
    torch.cuda.empty_cache()
    return rows, {key: tensor.numpy() for key, tensor in activations.items()}


def records_from_probes(probes: ProbeStore, activations: np.ndarray, rows: list[dict], layer: int,
                        site: str, planes: list[tuple[str, int]], report: dict) -> dict:
    records = {}
    for carrier, period in sorted(planes):
        plane = probes.plane(carrier, site, layer, period)
        if plane is None:
            report["missing"].append({"layer": layer, "site": site, "carrier": carrier, "period": period})
            continue
        weight, bias = plane
        points = activations.astype(np.float32) @ weight.T + bias[None, :]
        record = build_plane_record(weight, bias, points, data.carrier_values(carrier, rows), period)
        record["probe_source"] = "fit"
        records[f"{carrier}__p{period}"] = record
        report["planes"][f"L{layer}__{site}__{carrier}__p{period}"] = {
            "centroid_radius": record["base_centroid_radius"], "within_rms": record["base_within_rms"],
            "n_classes": int(record["classes"].numel()), "determinant": record["orientation_determinant"]}
    return records


def random_or_span_planes(mode: str, seed: int, canonical_root: Path, activations: np.ndarray,
                          rows: list[dict], probes_reference: ProbeStore, report: dict) -> dict:
    """Eight control planes standing in for input/offset x {2, 5, 10, 20} at layer 18.

    random: fresh orthonormal Gaussian pairs in the full hidden space, zero bias.
    span:   a random orthonormal basis of the real eight planes' 16-dim span,
            split into eight pairs, so the subspace is the true one but the
            cos/sin pairing and orientation are scrambled.
    Directions are scaled to the norms of the real probes so the projected
    coordinates have comparable magnitude.
    """
    rng = np.random.default_rng(seed)
    hidden = activations.shape[1]
    real = {}
    for carrier in CANONICAL_CARRIERS:
        for period in CANONICAL_PERIODS:
            real[(carrier, period)] = probes_reference.plane(carrier, "resid", 18, period)
    if mode == "span":
        stacked = np.concatenate([weight.astype(np.float64) for weight, _ in real.values()], axis=0)
        _, singular, vt = np.linalg.svd(stacked, full_matrices=False)
        rank = int((singular > singular[0] * 1e-6).sum())
        basis = vt[:rank]
        rotation, _ = np.linalg.qr(rng.normal(size=(rank, rank)))
        new_directions = rotation.T @ basis
    records = {}
    index = 0
    for carrier in CANONICAL_CARRIERS:
        for period in CANONICAL_PERIODS:
            real_weight, _ = real[(carrier, period)]
            if mode == "random":
                gaussian = rng.normal(size=(hidden, 2))
                directions, _ = np.linalg.qr(gaussian)
                weight = directions.T
            else:
                weight = new_directions[2 * index: 2 * index + 2]
            norms = np.linalg.norm(real_weight, axis=1, keepdims=True)
            weight = (weight / np.linalg.norm(weight, axis=1, keepdims=True) * norms).astype(np.float32)
            bias = np.zeros(2, dtype=np.float32)
            points = activations.astype(np.float32) @ weight.T + bias[None, :]
            record = build_plane_record(weight, bias, points, data.carrier_values(carrier, rows), period)
            record["probe_source"] = f"{mode}:seed{seed}"
            records[f"{carrier}__p{period}"] = record
            report["planes"][f"L18__resid__{carrier}__p{period}"] = {
                "centroid_radius": record["base_centroid_radius"], "within_rms": record["base_within_rms"]}
            index += 1
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", choices=["canonical", "all", "random", "span"], default="canonical")
    parser.add_argument("--seed", type=int, default=42, help="direction seed for --set random/span")
    parser.add_argument("--probe-root", default=str(PROBES_DIR))
    parser.add_argument("--configs", default=str(CONFIGS_DIR))
    parser.add_argument("--out", default=None, help="default: planes/<set> (planes/<set>_seed<seed> for controls)")
    args = parser.parse_args()

    probes = ProbeStore(Path(args.probe_root))
    if args.out:
        out_dir = Path(args.out)
    elif args.set in ("random", "span"):
        out_dir = PLANES_DIR / f"{args.set}_seed{args.seed}"
    else:
        out_dir = PLANES_DIR / args.set
    report = {"set": args.set, "probe_root": probes.describe(), "planes": {}, "missing": [],
              "runtime": runtime_fingerprint()}

    if args.set == "canonical":
        needed = required_planes(load_configs(Path(args.configs)))
    elif args.set == "all":
        needed = {(18, "resid"): {(carrier, period) for carrier in CANONICAL_CARRIERS for period in range(2, 100)}}
    else:
        needed = {(18, "resid"): set()}
    pairs = sorted(needed)
    log(f"set {args.set}: {len(pairs)} (layer, site) pairs, {sum(len(v) for v in needed.values())} planes")
    rows, activations = capture_training_prompts(pairs)

    for layer, site in pairs:
        if args.set in ("random", "span"):
            records = random_or_span_planes(args.set, args.seed, PLANES_DIR / "canonical",
                                            activations[(layer, site)], rows, probes, report)
        else:
            records = records_from_probes(probes, activations[(layer, site)], rows, layer, site,
                                          sorted(needed[(layer, site)]), report)
        if records:
            path = out_dir / f"L{layer}__{site}.pt"
            save_plane_records(path, records)
            log(f"L{layer:>2} {site:<9} {len(records):>3} planes -> {path}")
    if report["missing"]:
        log(f"WARNING: {len(report['missing'])} planes had no probe in {probes.root}")
    (out_dir / "report.json").write_text(json.dumps(report, indent=1))
    log(f"report -> {out_dir / 'report.json'}")


if __name__ == "__main__":
    main()
