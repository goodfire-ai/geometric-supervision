#!/usr/bin/env python3
"""Write every experiment family to configs/months/<family>.jsonl, and the alternate-target
configs (Appendix A.3.3) to configs/months_alternate/alternate_targets.jsonl. The latter live in
their own directory so that 04_train.py --all (the paper's sweeps) does not include them.

Run with --check to verify that the committed files match what the code
generates (so a config cannot drift from the generator that documents it).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from months.configs import all_families, alternate_targets, write_jsonl
from months.paths import CONFIGS_DIR

EXPECTED_COUNTS = {
    "task_layer_rank": 160,
    "geometry_edit_supervise": 271,
    "geometry_rank": 68,
    "radius_spread": 47,
    "training_range_ladder": 8,
    "per_prompt_ablation": 7,
    "supervision_controls": 3,
    "planes_ladder": 20,
}
EXPECTED_ALTERNATE = 504


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="compare with the committed files instead of writing")
    parser.add_argument("--out", default=str(CONFIGS_DIR))
    args = parser.parse_args()
    out_dir = Path(args.out)

    period_order = json.loads((out_dir / "ladder_order.json").read_text())["period_order"]
    families = all_families(period_order)

    for name, rows in families.items():
        if len(rows) != EXPECTED_COUNTS[name]:
            raise SystemExit(f"{name}: generated {len(rows)} configs, expected {EXPECTED_COUNTS[name]}")
    all_ids = [row["config_id"] for rows in families.values() for row in rows]
    if len(set(all_ids)) != len(all_ids):
        raise SystemExit("duplicate config ids across families")

    n_runs = sum(len(row["seeds"]) for rows in families.values() for row in rows)
    for name, rows in families.items():
        path = out_dir / f"{name}.jsonl"
        text = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
        if args.check:
            if not path.is_file() or path.read_text() != text:
                raise SystemExit(f"{path.name} differs from the generator output")
        else:
            write_jsonl(path, rows)
        print(f"  {name:28s} {len(rows):4d} configs  {sum(len(r['seeds']) for r in rows):5d} runs")
    print(f"{'checked' if args.check else 'wrote'} {len(all_ids)} configs, {n_runs} runs -> {out_dir}")

    rows = alternate_targets()
    if len(rows) != EXPECTED_ALTERNATE:
        raise SystemExit(f"alternate_targets: generated {len(rows)} configs, expected {EXPECTED_ALTERNATE}")
    path = out_dir.parent / "months_alternate" / "alternate_targets.jsonl"
    text = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    if args.check:
        if not path.is_file() or path.read_text() != text:
            raise SystemExit(f"{path.name} differs from the generator output")
    else:
        write_jsonl(path, rows)
    print(f"  {'alternate_targets':28s} {len(rows):4d} configs  {sum(len(r['seeds']) for r in rows):5d} runs -> {path.parent}")


if __name__ == "__main__":
    main()
