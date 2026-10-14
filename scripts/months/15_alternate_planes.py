#!/usr/bin/env python3
"""Readouts and targets for the alternate forms of months supervision (Appendix A.3.3).

Each set keeps the geometry LoRA's recipe and changes only the readout or the target shape.
Written to outputs/months/planes/<set>/L18__resid.pt:

    centroids                  the same eight Fourier planes as the geometry LoRA, but the target
                               is the base model's own per-value centroids, scaled by alpha_r
                               about their mean (no ideal ring)
    line_months                one linear probe per variable (input, offset), fit by ridge
                               regression on the months training prompts (penalty by 5-fold
                               cross-validation); the target is an evenly spaced line, centered
                               and scaled like the ring (alpha_r * r0)
    regression_months          the same probes; the target is the literal value (direct
                               regression when trained with the per-prompt loss and alpha_r = 1)
    line_addition,             the same, with the probes fit on the a+b= prompts like the
    regression_addition        Fourier probes (Appendix A.2)

Needs the canonical planes (03_build_planes.py --set canonical) and, for the *_addition sets,
the layer-18 a+b= activations (02_fit_probes.py --stage harvest).

    15_alternate_planes.py
"""
from __future__ import annotations

import copy
import json

import numpy as np
import torch

from _common import log

from common.geometry import class_geometry, procrustes_frame
from common.model import enable_determinism, load_base_model, load_tokenizer
from months import data
from months.capture import capture_many
from months.paths import PLANES_DIR, PROBES_DIR
from months.planes import save_plane_records
from months.planes.store import load_plane_records
from months.probes import fit

LAYER, SITE = 18, "resid"
CARRIERS = ["input-month", "offset"]
PERIODS = [2, 5, 10, 20]
LAMBDAS = np.logspace(-4, 2, 13)   # ridge penalty relative to the mean squared singular value
ADDITION_RIDGE = 1e-3              # fixed penalty for the a+b= probes, as for the Fourier probes
FOLDS = 5


def centroid_records() -> dict:
    """The canonical planes with the base model's own centroids as the target shape."""
    canonical = load_plane_records(PLANES_DIR / "canonical", LAYER, SITE)
    records = {}
    for carrier in CARRIERS:
        for period in PERIODS:
            key = f"{carrier}__p{period}"
            record = copy.deepcopy(canonical[key])
            centroids = record["class_centroids"].double()
            center = centroids.mean(dim=0)
            r0 = float(record["base_centroid_radius"])
            record["base_center"] = center.float()
            record["target_unit"] = ((centroids - center) / r0).float()
            record["probe_source"] = "base-model centroids"
            records[key] = record
    return records


def _ridge(X: np.ndarray, y: np.ndarray, lam: float) -> tuple[np.ndarray, float]:
    mu, ym = X.mean(0), y.mean()
    U, s, Vt = np.linalg.svd(X - mu, full_matrices=False)
    w = Vt.T @ ((s / (s ** 2 + lam * (s ** 2).mean())) * (U.T @ (y - ym)))
    return w, float(ym - mu @ w)


def months_value_probes(activations: np.ndarray, rows: list[dict], report: dict) -> dict:
    """Ridge probes for each variable's value on the months training prompts (penalty by CV)."""
    X = activations.astype(np.float64)
    folds = np.array_split(np.random.default_rng(0).permutation(len(X)), FOLDS)
    probes = {}
    for carrier in CARRIERS:
        y = np.asarray(data.carrier_values(carrier, rows), dtype=np.float64)
        scores = {}
        for lam in LAMBDAS:
            sse = sst = 0.0
            for k in range(FOLDS):
                test = folds[k]
                train = np.concatenate([folds[j] for j in range(FOLDS) if j != k])
                w, b = _ridge(X[train], y[train], lam)
                sse += ((X[test] @ w + b - y[test]) ** 2).sum()
                sst += ((y[test] - y[train].mean()) ** 2).sum()
            scores[lam] = 1 - sse / sst
        lam = max(scores, key=scores.get)
        w, b = _ridge(X, y, lam)
        probes[carrier] = (w.astype(np.float32)[None, :], np.asarray([b], np.float32))
        report["months"][carrier] = {"cv_r2": float(scores[lam]), "lambda": float(lam)}
        log(f"value probe (months prompts) {carrier}: cv R^2 {scores[lam]:.4f} (lambda {lam:.0e})")
    return probes


def addition_value_probes(report: dict) -> dict:
    """Ridge probes for a and b on the layer-18 a+b= activations (same split as the Fourier probes)."""
    activations = torch.load(PROBES_DIR / "activations" / f"L{LAYER}__{SITE}.pt").double().cuda()
    _, first, second = fit.addition_prompts()
    train, test = fit.train_test_indices(activations.shape[0])
    x_mean = activations[train].mean(dim=0)
    xc = activations - x_mean
    gram = xc[train].T @ xc[train]
    gram += ADDITION_RIDGE * torch.trace(gram) / gram.shape[0] * torch.eye(
        gram.shape[0], device=gram.device, dtype=gram.dtype)
    probes = {}
    for carrier, values in (("input-month", first), ("offset", second)):
        y = torch.as_tensor(values, dtype=torch.float64, device=activations.device)
        y_mean = y[train].mean()
        w = torch.linalg.solve(gram, xc[train].T @ (y[train] - y_mean))
        prediction = xc[test] @ w + y_mean
        r2 = 1 - ((prediction - y[test]) ** 2).sum() / ((y[test] - y[test].mean()) ** 2).sum()
        probes[carrier] = (w.float().cpu().numpy()[None, :], np.asarray([float(y_mean - x_mean @ w)], np.float32))
        report["addition"][carrier] = {"test_r2": float(r2)}
        log(f"value probe (a+b= prompts) {carrier}: test R^2 {float(r2):.4f}")
    del activations
    torch.cuda.empty_cache()
    return probes


def line_records(probes: dict, activations: np.ndarray, rows: list[dict], literal: bool) -> dict:
    """1-D readouts: an evenly spaced line (Procrustes sign, unit RMS) or the literal values."""
    records = {}
    for carrier in CARRIERS:
        weight, bias = probes[carrier]
        points = activations.astype(np.float64) @ weight.T.astype(np.float64) + bias[None, :]
        geometry = class_geometry(points, data.carrier_values(carrier, rows))
        classes = geometry["classes"].astype(np.float64)
        r0 = geometry["centroid_radius"]
        if literal:
            center = geometry["center"]
            unit = (classes[:, None] - center[None, :]) / r0
        else:
            line = classes[:, None]
            frame = procrustes_frame(geometry["centroids"], line)
            centered = line - line.mean(axis=0)
            center = frame["center"]
            unit = centered / np.sqrt((centered ** 2).sum(axis=1).mean()) @ frame["orientation"]
        records[f"{carrier}__p0"] = {
            "weight": torch.from_numpy(weight.astype(np.float32)),
            "bias": torch.from_numpy(bias.astype(np.float32)),
            "period": 0,
            "classes": torch.from_numpy(geometry["classes"].astype(np.int64)),
            "class_centroids": torch.from_numpy(geometry["centroids"].astype(np.float32)),
            "base_centroid_radius": float(r0),
            "base_within_rms": float(geometry["within_rms"]),
            "base_center": torch.from_numpy(np.asarray(center, np.float32)),
            "target_unit": torch.from_numpy(unit.astype(np.float32)),
            "base_orientation": torch.eye(1),
            "probe_source": "ridge value probe" + (" (literal)" if literal else ""),
        }
    return records


def main() -> None:
    report = {"layer": LAYER, "site": SITE, "months": {}, "addition": {}}
    save_plane_records(PLANES_DIR / "centroids" / f"L{LAYER}__{SITE}.pt", centroid_records())
    log("centroids: written")

    addition = addition_value_probes(report)
    enable_determinism()
    tokenizer = load_tokenizer()
    model = load_base_model("cuda")
    rows, prompts = data.training_grid()
    activations = capture_many(model, tokenizer, prompts, [(LAYER, SITE)], batch_size=64)[(LAYER, SITE)].numpy()
    months = months_value_probes(activations, rows, report)
    for source, probes in (("months", months), ("addition", addition)):
        for kind, literal in (("line", False), ("regression", True)):
            name = f"{kind}_{source}"
            save_plane_records(PLANES_DIR / name / f"L{LAYER}__{SITE}.pt",
                               line_records(probes, activations, rows, literal))
            log(f"{name}: written")
    (PLANES_DIR / "alternate_probes_report.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
