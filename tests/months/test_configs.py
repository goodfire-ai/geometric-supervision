import json

from months import configs
from months.paths import CONFIGS_DIR


def period_order():
    return json.loads((CONFIGS_DIR / "ladder_order.json").read_text())["period_order"]


def test_family_counts():
    families = configs.all_families(period_order())
    counts = {name: len(rows) for name, rows in families.items()}
    assert counts == {
        "task_layer_rank": 160, "geometry_edit_supervise": 271, "geometry_rank": 68,
        "radius_spread": 47, "training_range_ladder": 8, "per_prompt_ablation": 7,
        "supervision_controls": 3, "planes_ladder": 20,
    }
    ids = [row["config_id"] for rows in families.values() for row in rows]
    assert len(set(ids)) == len(ids)


def test_canonical_config_contents():
    rows = {row["config_id"]: row for row in configs.geometry_edit_supervise_grid()}
    canonical = rows[configs.CANONICAL_GEOMETRY_ID]
    assert canonical["edit_layer"] == 4 and canonical["supervise_layer"] == 18
    assert canonical["periods"] == [2, 5, 10, 20] and canonical["carriers"] == ["input-month", "offset"]
    assert (canonical["radius_scale"], canonical["spread_ratio"], canonical["spread_weight"]) == (1.45, 1.25, 0.3)
    assert canonical["loss"] == "per_value" and canonical["target_frame"] == "fixed"


def test_hash_detects_edits():
    row = configs.make_task_config("task_L03_r08", "task_layer_rank", 3, 8)
    assert configs.config_hash(row) == row["config_hash"]
    row["rank"] = 4
    assert configs.config_hash(row) != row["config_hash"]


def test_committed_configs_load_and_verify():
    loaded = configs.load_configs(CONFIGS_DIR)
    assert configs.CANONICAL_GEOMETRY_ID in loaded
    assert configs.CANONICAL_TASK_ID in loaded
    assert loaded["planes_ladder_k04"]["periods"] == [2, 5, 10, 20]
    assert loaded["planes_ladder_k98"]["seeds"] == [42]
