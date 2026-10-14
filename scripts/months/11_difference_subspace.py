#!/usr/bin/env python3
"""What is in the top-16 directions along which a trained model differs from the base? (Appendix A.4.1)

At the answer position of layer L, D = H_donor - H_base over the 1074 prompts, and V_16
holds its top-16 right singular vectors (uncentered, as in the patching experiment).
For each direction and for the subspace as a whole this script asks:

  1. variance decomposition: how much of the coordinate D v_k varies with the sum, the
     input month, the offset, the sum mod 10, the sum mod 12 and the tens digit; and the
     Fourier power of its mean profile over the sum
  2. alignment with the probe planes at the same layer (sum, input-month and offset planes)
  3. synthetic patches (Table tab:diff_synthetic): replace the base activation inside V_16 not
     by the donor's coordinates but by a function of the prompt (per-sum mean, per-offset and
     per-input means, coordinates borrowed from another prompt with the same sum / input /
     offset, decimal or mod-12 features, ...). If a function of the sum recovers the donor
     patch's accuracy, the difference the adapter adds is a representation of the sum.

Output: results/diff_anatomy_L<layer>_<donor>[_centered][_seed<s>].json.
The paper uses layer 18, donors task_L03_r08 and geometry_e04_s18, seeds 42-44.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from _common import log
from common.model import enable_determinism, load_base_model, load_tokenizer
from months import data
from months.adapters import load_condition
from months.data import month_token_ids
from months.patching import score_with_single_layer_patch
from months.paths import RESULTS_DIR, PLANES_DIR, RUNS_DIR
from months.planes.store import load_plane_records, plane_subspace

GEOMETRY, TASK = "geometry_e04_s18", "task_L03_r08"
RANK = 16
PERIODS = list(range(2, 61))


def setup():
    """Deterministic model, tokenizer, and the evaluation grid with its task variables."""
    enable_determinism()
    tokenizer = load_tokenizer()
    model = load_base_model("cuda")
    rows, prompts = data.evaluation_grid()
    grid = {"rows": rows, "prompts": prompts, "answers": data.answers(rows), "sums": data.sums(rows),
            "inputs": data.carrier_values("input-month", rows), "offsets": data.carrier_values("offset", rows),
            "month_ids": month_token_ids(tokenizer)}
    return tokenizer, model, grid


def variance_explained(values: np.ndarray, labels: np.ndarray) -> float:
    """Fraction of the variance of `values` (n or n x d) explained by the class means of `labels`."""
    values = np.asarray(values, np.float64)
    if values.ndim == 1:
        values = values[:, None]
    total = ((values - values.mean(0)) ** 2).sum()
    if total == 0:
        return 0.0
    between = 0.0
    for label in np.unique(labels):
        members = values[labels == label]
        between += len(members) * ((members.mean(0) - values.mean(0)) ** 2).sum()
    return float(between / total)


def save_json(path: Path, payload) -> None:
    def default(value):
        if isinstance(value, (np.floating, np.integer)):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        raise TypeError(type(value))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, default=default))
    log(f"-> {path}")


def fourier_power(profile_sums: np.ndarray, profile: np.ndarray) -> dict[int, float]:
    """Fraction of the (centered) profile's energy at each period, by least squares on cos/sin."""
    centered = profile - profile.mean()
    total = float((centered ** 2).sum()) or 1.0
    out = {}
    for period in PERIODS:
        angle = 2 * np.pi * profile_sums / period
        design = np.c_[np.cos(angle), np.sin(angle)]
        coefficients, *_ = np.linalg.lstsq(design, centered, rcond=None)
        out[period] = float(((design @ coefficients) ** 2).sum() / total)
    return out


def describe_directions(D: np.ndarray, basis: np.ndarray, grid: dict) -> list[dict]:
    sums, inputs, offsets = grid["sums"], grid["inputs"], grid["offsets"]
    coordinates = D @ basis
    unique_sums = np.unique(sums)
    rows = []
    for k in range(basis.shape[1]):
        c = coordinates[:, k]
        profile = np.array([c[sums == s].mean() for s in unique_sums])
        power = fourier_power(unique_sums.astype(float), profile)
        top = sorted(power.items(), key=lambda item: -item[1])[:4]
        rows.append({
            "direction": k + 1,
            "share_of_diff_energy": float((c ** 2).sum() / (D ** 2).sum()),
            "mean": float(c.mean()), "sd": float(c.std()),
            "explained_by": {
                "sum": variance_explained(c, sums),
                "input_month": variance_explained(c, inputs),
                "offset": variance_explained(c, offsets),
                "sum_mod_10": variance_explained(c, sums % 10),
                "sum_mod_12": variance_explained(c, sums % 12),
                "tens_digit": variance_explained(c, sums // 10),
                "in_range_flag": variance_explained(c, (sums <= 24).astype(int)),
                "offset_mod_10": variance_explained(c, offsets % 10),
                "offset_tens": variance_explained(c, offsets // 10),
                "offset_and_input_additive_r2": float(1 - ((c - regression_fit(c[:, None], np.c_[one_hot(offsets), one_hot(inputs)])[:, 0]) ** 2).sum()
                                                       / ((c - c.mean()) ** 2).sum()),
            },
            "spearman_with_sum": float(np.corrcoef(np.argsort(np.argsort(c)), np.argsort(np.argsort(sums)))[0, 1]),
            "linear_r2_with_sum": float(np.corrcoef(c, sums)[0, 1] ** 2),
            "top_periods": [{"period": p, "power": round(v, 3)} for p, v in top],
            "profile_by_sum": {int(s): float(v) for s, v in zip(unique_sums, profile)},
        })
    return rows


def plane_alignment(basis: np.ndarray, layer: int) -> dict:
    """Fraction of the difference subspace lying inside each family of probe planes."""
    records = load_plane_records(PLANES_DIR / "canonical", layer, "resid")
    out = {}
    families = {
        "sum_planes_2_5_10_20": (["sum"], [2, 5, 10, 20]),
        "sum_planes_all": (["sum"], sorted({int(k.split("__p")[1]) for k in records if k.startswith("sum__")})),
        "input_planes_2_5_10_20": (["input-month"], [2, 5, 10, 20]),
        "offset_planes_2_5_10_20": (["offset"], [2, 5, 10, 20]),
        "input_and_offset_planes": (["input-month", "offset"], [2, 5, 10, 20]),
    }
    for name, (carriers, periods) in families.items():
        try:
            Q, used = plane_subspace(records, carriers, periods)
        except KeyError:
            continue
        Q = Q.numpy()
        overlap = np.linalg.norm(Q.T @ basis) ** 2 / basis.shape[1]
        out[name] = {"planes": len(used), "rank": int(Q.shape[1]), "fraction_of_diff_subspace_inside": float(overlap),
                     "fraction_expected_at_random": float(Q.shape[1] / basis.shape[0])}
    for key in [k for k in records if k.startswith("sum__")]:
        carrier, period = key.split("__p")
        Q, _ = plane_subspace(records, [carrier], [int(period)])
        out[f"single::{key}"] = float(np.linalg.norm(Q.numpy().T @ basis) ** 2 / basis.shape[1])
    return out


def group_means(values: np.ndarray, labels: np.ndarray) -> np.ndarray:
    out = np.zeros_like(values)
    for label in np.unique(labels):
        members = labels == label
        out[members] = values[members].mean(0)
    return out


def regression_fit(values: np.ndarray, design: np.ndarray) -> np.ndarray:
    design = np.c_[design, np.ones(len(design))]
    coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
    return design @ coefficients


def one_hot(labels: np.ndarray) -> np.ndarray:
    classes = np.unique(labels)
    return (labels[:, None] == classes[None]).astype(float)


def synthetic_coordinates(donor_coordinates: np.ndarray, base_coordinates: np.ndarray, grid: dict,
                          rng: np.random.Generator) -> dict[str, np.ndarray]:
    sums, inputs, offsets = grid["sums"], grid["inputs"], grid["offsets"]
    C = donor_coordinates
    out = {
        "donor_exact": C,
        "per_sum_mean": group_means(C, sums),
        "linear_in_sum": regression_fit(C, sums[:, None].astype(float)),
        "quadratic_in_sum": regression_fit(C, np.c_[sums, sums ** 2].astype(float)),
        "decimal_features": regression_fit(C, np.c_[one_hot(sums % 10), one_hot(sums // 10)]),
        "mod12_features": regression_fit(C, one_hot(sums % 12)),
        "mod10_features": regression_fit(C, one_hot(sums % 10)),
        "tens_digit_features": regression_fit(C, one_hot(sums // 10)),
        "per_input_mean": group_means(C, inputs),
        "per_offset_mean": group_means(C, offsets),
        "base_plus_mean_shift": base_coordinates + (C - base_coordinates).mean(0, keepdims=True),
    }
    # borrow the coordinates of another prompt with the same sum / input / offset
    for name, labels in (("swap_within_sum", sums), ("swap_within_input", inputs), ("swap_within_offset", offsets)):
        swapped = C.copy()
        for label in np.unique(labels):
            members = np.where(labels == label)[0]
            swapped[members] = C[rng.permutation(members)]
        out[name] = swapped
    # the lcm(10,12) shift: far prompts get the per-sum mean of sum - 60 (in range) where that exists
    shifted = group_means(C, sums).copy()
    per_sum = {int(s): group_means(C, sums)[sums == s][0] for s in np.unique(sums)}
    for i, s in enumerate(sums):
        if s - 60 >= 2:
            shifted[i] = per_sum[int(s - 60)]
    out["per_sum_mean_of_sum_minus_60"] = shifted
    shifted12 = group_means(C, sums).copy()
    for i, s in enumerate(sums):
        if s - 12 >= 2:
            shifted12[i] = per_sum[int(s - 12)]
    out["per_sum_mean_of_sum_minus_12"] = shifted12
    # every out-of-range sum mapped to the in-range sum 13..24 with the same residue mod 12
    def in_range_twin(s: int) -> int:
        return s if s <= 24 else 13 + (s - 13) % 12
    out["per_sum_mean_of_in_range_twin"] = np.stack([per_sum[in_range_twin(int(s))] for s in sums])
    # the same borrowing, but from the BASE model's own coordinates: no adapter involved at all
    base_per_sum = {int(s): group_means(base_coordinates, sums)[sums == s][0] for s in np.unique(sums)}
    out["base_coordinates_of_sum_minus_60"] = np.stack([base_per_sum[int(s - 60)] if s - 60 >= 2 else base_coordinates[i] for i, s in enumerate(sums)])
    out["base_coordinates_of_in_range_twin"] = np.stack([base_per_sum[in_range_twin(int(s))] if s > 24 else base_coordinates[i] for i, s in enumerate(sums)])
    out["decimal_features_of_offset"] = regression_fit(C, np.c_[one_hot(offsets % 10), one_hot(offsets // 10)])
    out["decimal_features_plus_input"] = regression_fit(C, np.c_[one_hot(sums % 10), one_hot(sums // 10), one_hot(inputs)])
    out["offset_and_input_additive"] = regression_fit(C, np.c_[one_hot(offsets), one_hot(inputs)])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--layers", nargs="+", type=int, default=[18])
    parser.add_argument("--donors", nargs="+", default=[GEOMETRY, TASK])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-patches", action="store_true")
    parser.add_argument("--centered", action="store_true",
                        help="center the differences over prompts before the SVD (the paper's convention); "
                             "the mean shift is then not part of the subspace")
    args = parser.parse_args()

    tokenizer, model, grid = setup()
    rng = np.random.default_rng(0)
    for layer in args.layers:
        base_acc, H_base, _ = score_with_single_layer_patch(model, tokenizer, grid["prompts"], grid["answers"],
                                                            grid["sums"], grid["month_ids"], layer, "resid")
        log(f"L{layer} base {base_acc}")
        for donor in args.donors:
            cond = load_condition(donor, seed=args.seed, runs_root=RUNS_DIR)
            adapter = (cond.params, cond.scaling)
            donor_acc, H_donor, _ = score_with_single_layer_patch(model, tokenizer, grid["prompts"], grid["answers"],
                                                                  grid["sums"], grid["month_ids"], layer, "resid",
                                                                  adapter=adapter)
            log(f"L{layer} {donor} {donor_acc}")
            D = (H_donor - H_base).double().numpy()
            D_for_svd = D - D.mean(0, keepdims=True) if args.centered else D
            _, singular, vt = np.linalg.svd(D_for_svd, full_matrices=False)
            basis = vt[:RANK].T                       # [hidden, 16]
            energy = singular ** 2 / (singular ** 2).sum()
            suffix = ("_centered" if args.centered else "") + (f"_seed{args.seed}" if args.seed != 42 else "")
            report = {
                "layer": layer, "donor": donor, "seed": args.seed, "centered": args.centered, "base_acc": base_acc, "donor_acc": donor_acc,
                "energy_share_top16": float(energy[:RANK].sum()), "energy_share_by_direction": energy[:RANK].tolist(),
                "mean_diff_norm": float(np.linalg.norm(D.mean(0))), "rms_diff_norm": float(np.sqrt((D ** 2).sum(1).mean())),
                "directions": describe_directions(D, basis, grid),
                "plane_alignment": plane_alignment(basis, layer),
                "patches": {},
            }
            # how much of the whole difference (not just the top 16) is a function of the sum
            report["whole_diff_explained_by"] = {
                "sum": variance_explained(D, grid["sums"]), "input_month": variance_explained(D, grid["inputs"]),
                "offset": variance_explained(D, grid["offsets"]), "sum_mod_12": variance_explained(D, grid["sums"] % 12),
                "sum_mod_10": variance_explained(D, grid["sums"] % 10)}
            if not args.no_patches:
                basis_t = torch.as_tensor(basis, dtype=torch.float32)
                donor_coordinates = (H_donor.double().numpy() @ basis).astype(np.float32)
                base_coordinates = (H_base.double().numpy() @ basis).astype(np.float32)
                for name, coordinates in synthetic_coordinates(donor_coordinates, base_coordinates, grid, rng).items():
                    acc, _, per_sum = score_with_single_layer_patch(
                        model, tokenizer, grid["prompts"], grid["answers"], grid["sums"], grid["month_ids"], layer, "resid",
                        donor_coordinates=torch.as_tensor(coordinates, dtype=torch.float32), basis=basis_t)
                    acc["61-72"] = round(float(np.mean([per_sum[s] for s in range(61, 73)])), 4)
                    report["patches"][name] = acc
                    log(f"  patch {name:32} {acc}")
            save_json(RESULTS_DIR / f"diff_anatomy_L{layer}_{donor}{suffix}.json", report)


if __name__ == "__main__":
    main()
