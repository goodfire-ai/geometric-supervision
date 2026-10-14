"""Training configurations: one row per (adapter placement, objective, hyperparameters).

Every experiment family in the paper is generated here, so `configs/*.jsonl`
can always be rebuilt and checked. A config row has these fields:

    config_id        short readable id, e.g. "geometry_e04_s18"
    family           which experiment the row belongs to
    method           "task" (cross-entropy on the answer) or "geometry"
    edit_layer       transformer block whose MLP gets the adapter
    lora_modules     the linear layers adapted inside that block
    rank             adapter rank
    scaling          multiplier on the adapter output (2.0 throughout)
    steps            optimizer steps (200)
    lr, weight_decay, grad_clip    AdamW settings (2e-3, 0.01, 1.0)
    train_max_sum    training prompts are all sums 2..train_max_sum (24)
    eval_max_sum     evaluation grid is all sums 2..eval_max_sum (96)
    seeds            training seeds

  and, for geometry configs only:

    supervise_layer  block whose residual stream is read for the loss
    site             "resid" or "mlp_input"
    target_frame     "fixed" (base-model frame) or "dynamic" (Procrustes each step)
    carriers         which task variables' planes are supervised
    periods          which Fourier periods
    radius_scale     target centroid radius as a multiple of the base model's (alpha_r)
    spread_ratio     target within-value spread as a multiple of the base model's (tau)
    spread_weight    weight of the spread term (lambda)
    loss             "per_value", "per_prompt" or "per_prompt_spread"
    shuffled_targets whether the value-to-ring-position assignment is permuted
    planes           name of the plane set to train against ("canonical", "all",
                     "random", "span", or an alternate set from 15_alternate_planes.py)

`config_hash` is a content hash over every field except the id, recomputed on
load so that a row cannot be edited silently.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from common.lora import MLP_MODULES

SEEDS = [42, 43, 44]

DEFAULTS = {
    "lora_modules": list(MLP_MODULES),
    "rank": 1,
    "scaling": 2.0,
    "steps": 200,
    "lr": 2e-3,
    "weight_decay": 0.01,
    "grad_clip": 1.0,
    "train_max_sum": 24,
    "eval_max_sum": 96,
}

GEOMETRY_DEFAULTS = {
    "supervise_layer": 18,
    "site": "resid",
    "target_frame": "fixed",
    "carriers": ["input-month", "offset"],
    "periods": [2, 5, 10, 20],
    "radius_scale": 1.45,
    "spread_ratio": 1.25,
    "spread_weight": 0.3,
    "loss": "per_value",
    "shuffled_targets": False,
    "planes": "canonical",
}

CANONICAL_GEOMETRY_ID = "geometry_e04_s18"
CANONICAL_TASK_ID = "task_L03_r08"

RADIUS_SCALES = [0.8, 1.0, 1.2, 1.45, 1.7, 2.0, 2.5, 3.0]
SPREAD_RATIOS = [0.5, 0.8, 1.0, 1.25, 1.5, 2.0]
LADDER_CAPS = [12, 24, 36, 48, 60, 72, 84, 96]
PLANES_LADDER_RUNGS = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 98]


def config_hash(row: dict) -> str:
    payload = json.dumps({key: value for key, value in row.items()
                          if key not in ("config_id", "config_hash")}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def _code(x: float) -> str:
    """1.45 -> '145', 0.8 -> '080', 3.0 -> '300' (stable two-decimal code for ids)."""
    return f"{x:.2f}".replace(".", "")


def make_task_config(config_id: str, family: str, edit_layer: int, rank: int = 1,
                     seeds: list[int] = SEEDS, **overrides) -> dict:
    row = {"config_id": config_id, "family": family, "method": "task", **DEFAULTS,
           "edit_layer": edit_layer, "rank": rank, "seeds": list(seeds)}
    row.update(overrides)
    row["config_hash"] = config_hash(row)
    return row


def make_geometry_config(config_id: str, family: str, edit_layer: int = 4,
                         seeds: list[int] = SEEDS, **overrides) -> dict:
    row = {"config_id": config_id, "family": family, "method": "geometry", **DEFAULTS,
           **GEOMETRY_DEFAULTS, "edit_layer": edit_layer, "seeds": list(seeds)}
    row.update(overrides)
    row["config_hash"] = config_hash(row)
    return row


# ---------------------------------------------------------------------------
# experiment families
# ---------------------------------------------------------------------------

def task_layer_rank_grid() -> list[dict]:
    """Task LoRAs at every layer 0..31 and rank 1, 2, 4, 8, 16 (appendix heatmap)."""
    return [make_task_config(f"task_L{layer:02d}_r{rank:02d}", "task_layer_rank", layer, rank)
            for layer in range(32) for rank in (1, 2, 4, 8, 16)]


def geometry_edit_supervise_grid() -> list[dict]:
    """Rank-1 geometry LoRAs: edit layer 0..16 x supervise layer 16..31 (supervise > edit)."""
    return [make_geometry_config(f"geometry_e{edit:02d}_s{supervise:02d}", "geometry_edit_supervise",
                                 edit, supervise_layer=supervise)
            for edit in range(17) for supervise in range(16, 32) if supervise > edit]


def geometry_rank_grid() -> list[dict]:
    """Geometry LoRAs of rank 2, 4, 8, 16 at edit layers 0..16, supervised at layer 18.

    Rank 1 comes from `geometry_edit_supervise_grid` (the s18 column).
    """
    return [make_geometry_config(f"rank_r{rank:02d}_e{edit:02d}_s18", "geometry_rank", edit, rank=rank)
            for rank in (2, 4, 8, 16) for edit in range(17)]


def radius_spread_grid() -> list[dict]:
    """The alpha_r x tau grid at the canonical placement (the canonical cell is geometry_e04_s18)."""
    rows = []
    for radius_scale in RADIUS_SCALES:
        for spread_ratio in SPREAD_RATIOS:
            if (radius_scale, spread_ratio) == (1.45, 1.25):
                continue
            rows.append(make_geometry_config(
                f"gamtau_a{_code(radius_scale)}_t{_code(spread_ratio)}", "radius_spread",
                radius_scale=radius_scale, spread_ratio=spread_ratio))
    return rows


def training_range_ladder() -> list[dict]:
    """Task LoRAs (rank 1, layer 4) trained on sums up to 12, 24, ..., 96; evaluated to 132."""
    return [make_task_config(f"ladder_p{cap:03d}", "training_range_ladder", 4, 1,
                             train_max_sum=cap, eval_max_sum=132)
            for cap in LADDER_CAPS]


def per_prompt_ablation() -> list[dict]:
    """Per-value vs per-prompt loss, each at spread weights 0, 0.3, 1, 3.

    The per-value cell at weight 0.3 is the canonical config and is not repeated.
    """
    rows = []
    for loss, name in (("per_value", "pervalue"), ("per_prompt_spread", "perprompt")):
        for weight in (0.0, 0.3, 1.0, 3.0):
            if loss == "per_value" and weight == 0.3:
                continue
            rows.append(make_geometry_config(f"perprompt_{name}_lam{_code(weight)}", "per_prompt_ablation",
                                             loss=loss, spread_weight=weight))
    return rows


def supervision_site_controls() -> list[dict]:
    """Controls for the choice of planes: random directions, random pairs within the
    Fourier span, and the real planes with a permuted value-to-angle assignment."""
    return [
        make_geometry_config("ctl_randomplanes", "supervision_controls", planes="random"),
        make_geometry_config("ctl_spanplanes", "supervision_controls", planes="span"),
        make_geometry_config("ctl_shuffled", "supervision_controls", shuffled_targets=True),
    ]


def planes_ladder(period_order: list[int], rungs: list[int] = PLANES_LADDER_RUNGS) -> list[dict]:
    """Supervise the first k periods of a fixed ordering, for increasing k.

    Three seeds up to six periods, one seed beyond (the long rungs are slow
    and the trend is already clear).
    """
    rows = []
    for k in rungs:
        periods = sorted(period_order[:k])
        seeds = SEEDS if k <= 6 else SEEDS[:1]
        rows.append(make_geometry_config(f"planes_ladder_k{k:02d}", "planes_ladder",
                                         periods=periods, seeds=seeds, planes="all"))
    return rows


ALTERNATE_LRS = [2.5e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2]
ALTERNATE_RADIUS_SCALES = [1.0, 1.45, 2.0, 3.0]
ALTERNATE_SPREAD_WEIGHTS = [0.0, 0.3]
ALTERNATE_VARIABLES = {"in": ["input-month"], "off": ["offset"], "inoff": ["input-month", "offset"]}


def alternate_targets() -> list[dict]:
    """Alternate forms of variable supervision (Appendix A.3.3), each tuned over the same grid.

    ring        the geometry LoRA's ideal rings on the canonical Fourier planes
    centroids   the same planes, with the base model's own centroids as the target
    line        an evenly spaced line under a linear value probe (per-value loss)
    regression  direct regression of the value through the same probe (per-prompt loss)
    For line and regression, the probes are fit on the months prompts ("months") or on the
    a+b= prompts ("addition"; input + offset only). Plane sets: 15_alternate_planes.py.
    """
    family, rows = "alternate_targets", []
    for lr in ALTERNATE_LRS:
        for vname, carriers in ALTERNATE_VARIABLES.items():
            sources = ("months", "addition") if vname == "inoff" else ("months",)
            for source in sources:
                rows.append(make_geometry_config(
                    f"alt_regression_{source}_{vname}_lr{lr:g}", family, carriers=carriers, periods=[0],
                    planes=f"regression_{source}", loss="per_prompt", radius_scale=1.0,
                    spread_weight=0.0, lr=lr))
            for alpha in ALTERNATE_RADIUS_SCALES:
                for weight in ALTERNATE_SPREAD_WEIGHTS:
                    suffix = f"{vname}_lr{lr:g}_a{_code(alpha)}_sw{_code(weight)}"
                    for source in sources:
                        rows.append(make_geometry_config(
                            f"alt_line_{source}_{suffix}", family, carriers=carriers, periods=[0],
                            planes=f"line_{source}", radius_scale=alpha, spread_weight=weight, lr=lr))
                    for target, planes in (("ring", "canonical"), ("centroids", "centroids")):
                        rows.append(make_geometry_config(
                            f"alt_{target}_{suffix}", family, carriers=carriers, planes=planes,
                            radius_scale=alpha, spread_weight=weight, lr=lr))
    return rows


def all_families(period_order: list[int]) -> dict[str, list[dict]]:
    return {
        "task_layer_rank": task_layer_rank_grid(),
        "geometry_edit_supervise": geometry_edit_supervise_grid(),
        "geometry_rank": geometry_rank_grid(),
        "radius_spread": radius_spread_grid(),
        "training_range_ladder": training_range_ladder(),
        "per_prompt_ablation": per_prompt_ablation(),
        "supervision_controls": supervision_site_controls(),
        "planes_ladder": planes_ladder(period_order),
    }


# ---------------------------------------------------------------------------
# reading configs back
# ---------------------------------------------------------------------------

def write_jsonl(path: Path, rows: list[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def load_configs(configs_dir: Path) -> dict[str, dict]:
    """Every config in configs/*.jsonl keyed by id, with hashes verified."""
    configs = {}
    for path in sorted(Path(configs_dir).glob("*.jsonl")):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            expected = config_hash(row)
            if row.get("config_hash") != expected:
                raise ValueError(f"{row.get('config_id')} in {path.name}: config_hash "
                                 f"{row.get('config_hash')} does not match its contents ({expected})")
            if row["config_id"] in configs:
                raise ValueError(f"duplicate config id {row['config_id']}")
            configs[row["config_id"]] = row
    return configs


def resolve_alias(config_id: str) -> str:
    """Grid cells that coincide with the canonical config point at it."""
    aliases = {
        "gamtau_a145_t125": CANONICAL_GEOMETRY_ID,
        "perprompt_pervalue_lam030": CANONICAL_GEOMETRY_ID,
        "rank_r01_e04_s18": CANONICAL_GEOMETRY_ID,
    }
    return aliases.get(config_id, config_id)
