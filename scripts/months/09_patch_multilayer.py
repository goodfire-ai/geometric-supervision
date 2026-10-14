#!/usr/bin/env python3
"""Live multi-layer plane patching: base <- donor at several layers at once.

At every layer in --layers, the in-subspace component of the live hidden state
(all positions by default) is replaced by the donor's, so earlier patches
propagate through the network. The subspace at each layer is the span of that
layer's own probe planes (or, with --fixed-layer, one layer's planes reused
everywhere). Controls: rank-matched random subspaces and the top difference
directions per layer; the reverse patch (donor <- base) tests necessity.

Output: results/patch_multilayer_L<first>-<last>_<positions>__<donor>.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from _common import log

from months import data
from months.adapters import load_condition
from common.model import enable_determinism, load_base_model, load_tokenizer, runtime_fingerprint
from months.data import month_token_ids
from months.paths import RESULTS_DIR, PLANES_DIR, RUNS_DIR
from months.patching import difference_subspace, random_subspace, score_with_live_patch
from months.planes import load_plane_records, plane_subspace


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--donor", required=True)
    parser.add_argument("--layers", nargs="+", type=int, required=True)
    parser.add_argument("--carriers", nargs="+", default=["input-month", "offset", "sum"])
    parser.add_argument("--periods", nargs="+", type=int, default=[2, 5, 10, 20])
    parser.add_argument("--positions", choices=["all", "last"], default="all")
    parser.add_argument("--fixed-layer", type=int, default=None, help="use this layer's planes at every patched layer")
    parser.add_argument("--random-seeds", nargs="*", type=int, default=[0, 1, 2])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--planes", default=str(PLANES_DIR / "canonical"))
    parser.add_argument("--runs", default=str(RUNS_DIR))
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    name = f"L{args.layers[0]}-{args.layers[-1]}_{args.positions}" + (f"_fixed{args.fixed_layer}" if args.fixed_layer is not None else "")
    out = Path(args.out) if args.out else RESULTS_DIR / f"patch_multilayer_{name}__{args.donor}.json"

    enable_determinism()
    tokenizer = load_tokenizer()
    model = load_base_model("cuda")
    month_ids = month_token_ids(tokenizer)
    rows, prompts = data.evaluation_grid()
    answers, sums = data.answers(rows), data.sums(rows)

    bases, used = {}, {}
    for layer in args.layers:
        source_layer = args.fixed_layer if args.fixed_layer is not None else layer
        records = load_plane_records(Path(args.planes), source_layer, "resid")
        bases[layer], used[layer] = plane_subspace(records, args.carriers, args.periods)
    ranks = {layer: int(basis.shape[1]) for layer, basis in bases.items()}
    hidden = next(iter(bases.values())).shape[0]
    log(f"layers {args.layers}, subspace ranks {ranks}")
    donor = load_condition(args.donor, seed=args.seed, runs_root=Path(args.runs))
    adapter = (donor.params, donor.scaling)

    def score(**kwargs):
        return score_with_live_patch(model, tokenizer, prompts, answers, sums, month_ids,
                                     args.layers, args.positions, **kwargs)

    results, per_sum = {}, {}
    results["base"], base_last, per_sum["base"] = score(collect_last=True)
    results["donor"], donor_last, per_sum["donor"] = score(adapter=adapter, collect_last=True)
    results["base_from_donor_planes"], _, per_sum["base_from_donor_planes"] = score(bases=bases, donor_adapter=adapter)
    log(f"base {results['base']['2-96']:.4f}  donor {results['donor']['2-96']:.4f}  "
        f"base <- donor planes {results['base_from_donor_planes']}")
    for random_seed in args.random_seeds:
        random_bases = {layer: random_subspace(hidden, ranks[layer], 1000 * random_seed + layer) for layer in args.layers}
        key = f"base_from_donor_random{random_seed}"
        results[key], _, per_sum[key] = score(bases=random_bases, donor_adapter=adapter)
    diff_bases = {layer: difference_subspace(base_last[layer], donor_last[layer], ranks[layer]) for layer in args.layers}
    results["base_from_donor_diff"], _, per_sum["base_from_donor_diff"] = score(bases=diff_bases, donor_adapter=adapter)
    results["donor_from_base_planes"], _, per_sum["donor_from_base_planes"] = score(adapter=adapter, bases=bases, donor_adapter=None)
    log(f"diff directions {results['base_from_donor_diff']}  donor <- base {results['donor_from_base_planes']}")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "donor": args.donor, "seed": args.seed, "layers": args.layers, "positions": args.positions,
        "fixed_layer": args.fixed_layer, "carriers": args.carriers, "periods": args.periods,
        "planes_used": {str(k): v for k, v in used.items()}, "ranks": {str(k): v for k, v in ranks.items()},
        "results": results, "per_sum": per_sum, "runtime": runtime_fingerprint()}, indent=1))
    log(f"-> {out}")


if __name__ == "__main__":
    main()
