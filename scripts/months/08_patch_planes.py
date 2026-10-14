#!/usr/bin/env python3
"""Single-layer activation patching between a trained model (the donor) and the base model.

At one layer and site, for every evaluation prompt, the answer-position
activation's component inside a subspace is swapped between the two models:

    base <- donor   ("sufficiency": does the base model inherit the donor's accuracy?)
    donor <- base   ("necessity":   does the donor lose its accuracy?)

Subspaces: the span of the chosen probe planes; rank-matched random subspaces
(three seeds); and the top-k directions along which the two models' activations
differ (k from --diff-ranks). Output: results/patch_planes_L<layer>__<donor>__<set>.json
with band accuracies and per-sum accuracy for every condition.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from _common import log

from months import data
from months.adapters import load_condition
from common.model import enable_determinism, load_base_model, load_tokenizer, runtime_fingerprint
from months.data import month_token_ids
from months.paths import RESULTS_DIR, PLANES_DIR, RUNS_DIR
from months.patching import difference_subspace, random_subspace, score_with_single_layer_patch
from months.planes import load_plane_records, plane_subspace


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--donor", required=True, help="config id of the trained adapter")
    parser.add_argument("--layer", type=int, default=18)
    parser.add_argument("--site", default="resid", choices=["resid", "mlp_input"])
    parser.add_argument("--carriers", nargs="+", default=["input-month", "offset"])
    parser.add_argument("--periods", nargs="+", type=int, default=[2, 5, 10, 20])
    parser.add_argument("--set-name", default=None, help="label for the output file (default from carriers/periods)")
    parser.add_argument("--random-seeds", nargs="*", type=int, default=[0, 1, 2])
    parser.add_argument("--diff-ranks", nargs="*", type=int, default=[16])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--planes", default=str(PLANES_DIR / "canonical"))
    parser.add_argument("--runs", default=str(RUNS_DIR))
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    set_name = args.set_name or ("_".join(c.replace("input-month", "input") for c in args.carriers)
                                  + "_" + "_".join(str(p) for p in args.periods))
    out = Path(args.out) if args.out else RESULTS_DIR / f"patch_planes_L{args.layer}__{args.donor}__{set_name}.json"

    enable_determinism()
    tokenizer = load_tokenizer()
    model = load_base_model("cuda")
    month_ids = month_token_ids(tokenizer)
    rows, prompts = data.evaluation_grid()
    answers, sums = data.answers(rows), data.sums(rows)

    records = load_plane_records(Path(args.planes), args.layer, args.site)
    basis, used = plane_subspace(records, args.carriers, args.periods)
    hidden, rank = basis.shape
    log(f"L{args.layer}/{args.site}: {len(used)} planes span rank {rank} of {hidden}")
    donor = load_condition(args.donor, seed=args.seed, runs_root=Path(args.runs))
    adapter = (donor.params, donor.scaling)

    def score(**kwargs):
        return score_with_single_layer_patch(model, tokenizer, prompts, answers, sums, month_ids,
                                             args.layer, args.site, **kwargs)

    results, per_sum = {}, {}
    results["base"], base_states, per_sum["base"] = score()
    results["donor"], donor_states, per_sum["donor"] = score(adapter=adapter)
    log(f"base {results['base']}  donor {results['donor']}")

    base_coordinates = base_states @ basis
    donor_coordinates = donor_states @ basis
    results["base_from_donor_planes"], _, per_sum["base_from_donor_planes"] = score(
        donor_coordinates=donor_coordinates, basis=basis)
    results["donor_from_base_planes"], _, per_sum["donor_from_base_planes"] = score(
        adapter=adapter, donor_coordinates=base_coordinates, basis=basis)
    log(f"base <- donor planes {results['base_from_donor_planes']}")

    for random_seed in args.random_seeds:
        random_basis = random_subspace(hidden, rank, random_seed)
        key = f"base_from_donor_random{random_seed}"
        results[key], _, per_sum[key] = score(donor_coordinates=donor_states @ random_basis, basis=random_basis)
        key = f"donor_from_base_random{random_seed}"
        results[key], _, per_sum[key] = score(adapter=adapter, donor_coordinates=base_states @ random_basis,
                                              basis=random_basis)
    for k in args.diff_ranks:
        k = min(k, hidden)
        diff_basis = difference_subspace(base_states, donor_states, k)
        key = f"base_from_donor_diff{k}"
        results[key], _, per_sum[key] = score(donor_coordinates=donor_states @ diff_basis, basis=diff_basis)
        key = f"donor_from_base_diff{k}"
        results[key], _, per_sum[key] = score(adapter=adapter, donor_coordinates=base_states @ diff_basis,
                                              basis=diff_basis)
        log(f"diff rank {k}: base <- donor {results[f'base_from_donor_diff{k}']}")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "donor": args.donor, "seed": args.seed, "layer": args.layer, "site": args.site,
        "carriers": args.carriers, "periods": args.periods, "planes_used": used, "subspace_rank": rank,
        "in_plane_relative_change": float((donor_coordinates - base_coordinates).norm() / base_coordinates.norm()),
        "results": results, "per_sum": per_sum, "runtime": runtime_fingerprint()}, indent=1))
    log(f"-> {out}")


if __name__ == "__main__":
    main()
