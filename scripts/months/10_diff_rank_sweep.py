#!/usr/bin/env python3
"""How many directions carry a trained adapter's effect at one layer?

At a single layer and the answer position, patch base <- donor along the
top-k directions of the donor-minus-base activation difference, for k in a
sweep, both centered (mean difference removed) and uncentered (so the mean
shift can enter at k = 1). Also: the mean shift alone, the full swap, and
rank-matched random controls. Output: results/diff_rank_L<layer>__<donor>[_seed<s>].json
(Appendix A.4, Table tab:diff_rank.)
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
from months.paths import RESULTS_DIR, RUNS_DIR
from months.patching import difference_subspace, random_subspace, score_with_single_layer_patch

RANKS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--donor", required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--site", default="resid")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--runs", default=str(RUNS_DIR))
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    suffix = "" if args.seed == 42 else f"_seed{args.seed}"
    out = Path(args.out) if args.out else RESULTS_DIR / f"diff_rank_L{args.layer}__{args.donor}{suffix}.json"

    enable_determinism()
    tokenizer = load_tokenizer()
    model = load_base_model("cuda")
    month_ids = month_token_ids(tokenizer)
    rows, prompts = data.evaluation_grid()
    answers, sums = data.answers(rows), data.sums(rows)
    donor = load_condition(args.donor, seed=args.seed, runs_root=Path(args.runs))
    adapter = (donor.params, donor.scaling)

    def score(**kwargs):
        return score_with_single_layer_patch(model, tokenizer, prompts, answers, sums, month_ids,
                                             args.layer, args.site, **kwargs)

    results, per_sum = {}, {}
    results["base"], base_states, per_sum["base"] = score()
    results["donor"], donor_states, per_sum["donor"] = score(adapter=adapter)
    hidden = base_states.shape[1]
    identity = torch.eye(hidden)
    difference = donor_states - base_states
    mean_shift = difference.mean(dim=0, keepdim=True)
    centered_sv = torch.linalg.svdvals((difference - mean_shift).double())
    uncentered_sv = torch.linalg.svdvals(difference.double())
    spectrum = {"centered_variance_share": {k: float((centered_sv[:k] ** 2).sum() / (centered_sv ** 2).sum()) for k in RANKS},
                "uncentered_variance_share": {k: float((uncentered_sv[:k] ** 2).sum() / (uncentered_sv ** 2).sum()) for k in RANKS},
                "mean_shift_norm_over_rms_difference": float(mean_shift.norm() / difference.norm(dim=1).pow(2).mean().sqrt())}
    results["base_plus_mean_shift"], _, per_sum["base_plus_mean_shift"] = score(donor_coordinates=base_states + mean_shift, basis=identity)
    results["base_full_swap"], _, per_sum["base_full_swap"] = score(donor_coordinates=donor_states, basis=identity)
    log(f"base {results['base']['49-96']:.3f} donor {results['donor']['49-96']:.3f} "
        f"mean shift {results['base_plus_mean_shift']['49-96']:.3f} full {results['base_full_swap']['49-96']:.3f} (far band)")
    for k in RANKS:
        for centered in (True, False):
            basis = difference_subspace(base_states, donor_states, k, centered=centered)
            key = f"base_from_donor_diff{k}_{'centered' if centered else 'uncentered'}"
            results[key], _, per_sum[key] = score(donor_coordinates=donor_states @ basis, basis=basis)
        log(f"k={k:5d} centered {results[f'base_from_donor_diff{k}_centered']['49-96']:.3f}  "
            f"uncentered {results[f'base_from_donor_diff{k}_uncentered']['49-96']:.3f}")
        if k in (1, 16, 128):
            for random_seed in (0, 1, 2):
                basis = random_subspace(hidden, k, random_seed)
                key = f"base_from_donor_random{k}_seed{random_seed}"
                results[key], _, _ = score(donor_coordinates=donor_states @ basis, basis=basis)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"donor": args.donor, "seed": args.seed, "layer": args.layer, "site": args.site,
                               "ranks": RANKS, "spectrum": spectrum, "results": results, "per_sum": per_sum,
                               "runtime": runtime_fingerprint()}, indent=1))
    log(f"-> {out}")


if __name__ == "__main__":
    main()
