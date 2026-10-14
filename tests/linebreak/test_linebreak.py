import json
import math

import numpy as np
import pandas as pd
import torch
from torch import nn

from linebreak import evaluate, harvest, losses, manifold, paths, targets, train, wrapping
from linebreak.probes import HelixFrame, fit_period

T = 154.77


def identity_frame():
    return HelixFrame(u_cos=np.eye(3)[0], u_sin=np.eye(3)[1], w_count=np.eye(3)[2] * T,
                      b_cos=0, b_sin=0, b_count=0, period=T)


def test_wrapping_and_counts():
    text = "hello world\nfoo bar"
    prev_nl = wrapping.previous_newline_index(text)
    assert prev_nl[0] == -1 and prev_nl[12] == 11
    # token 'foo' occupies [12, 15); its count is (15 - 1) - prev_nl[14] = 3
    assert wrapping.token_character_counts([(12, 15)], prev_nl, max_count=150)[0] == 3
    wrapped = wrapping.wrap("the quick brown fox jumps over the lazy dog", width=15)
    assert all(len(line) <= 15 for line in wrapped.split("\n"))
    # words longer than the width are kept whole unless split_long_words is set
    assert wrapping.wrap("a supercalifragilistic word", width=8).split("\n")[1] == "supercalifragilistic"
    assert max(len(line) for line in wrapping.wrap("a supercalifragilistic word", 8, True).split("\n")) <= 8


def test_period_fit_and_frame_r2():
    counts = np.arange(1, 151)
    frame = identity_frame()
    points = frame.ideal(counts)
    period, _, _ = fit_period(points[:, 0], points[:, 1])
    assert abs(period - T) < 1.5
    assert frame.r2(points, counts) > 0.999
    rotated = points @ np.linalg.qr(np.random.default_rng(0).normal(size=(3, 3)))[0] * 2 + 1
    assert frame.procrustes_r2(rotated, counts) > 0.999


def test_targets_match_ideal_helix():
    frame = identity_frame()
    _, _, lookup = targets.build("helix", frame, max_count=150, device="cpu")
    assert np.allclose(lookup.numpy(), frame.ideal(np.arange(151)), atol=1e-5)
    assert targets.build("circle", frame, 150, "cpu")[2].shape == (151, 2)
    assert targets.build("none", frame, 150, "cpu") == (None, None, None)


def test_geometry_loss_is_zero_at_target():
    counts = np.repeat(np.arange(1, 31), 5)
    c = torch.arange(31, dtype=torch.float32)
    lookup = torch.stack([torch.cos(2 * math.pi * c / 20), torch.sin(2 * math.pi * c / 20)], 1)
    angle = 0.7
    rotation = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    points = lookup[counts].numpy() @ rotation.T * 2.0 + np.array([1.0, -3.0])
    points = points + np.random.default_rng(0).normal(scale=0.1, size=points.shape)
    frozen = losses.freeze_targets(points, counts, lookup, alpha_r=1.0)
    assert frozen["has_target"][1:].all() and not frozen["has_target"][0]
    # points sitting exactly on their targets: zero centroid term and zero spread (tau = 0)
    at_target = frozen["targets"][torch.as_tensor(counts)]
    loss = losses.geometry_loss(at_target, torch.as_tensor(counts), frozen, tau=0.0, spread_weight=1.0)
    assert loss.item() < 1e-6
    # the base model's own points: centroids near the targets, spread ratio 1 so the loss is ~(1 - tau)^2
    loss = losses.geometry_loss(torch.tensor(points, dtype=torch.float32), torch.as_tensor(counts), frozen,
                                tau=0.0, spread_weight=0.0)
    assert loss.item() < 0.05


def test_per_count_means():
    hiddens = np.array([[1.0, 0], [3.0, 0], [5.0, 0]], np.float32)
    means = harvest.per_count_means(hiddens, np.array([2, 2, 7]), max_count=10)
    assert means[2, 0] == 2.0 and means[7, 0] == 5.0 and np.isnan(means[3, 0])


def test_pca_of_a_helix_is_three_dimensional():
    counts = np.arange(1, 151, dtype=float)
    means = np.full((151, 8), np.nan)
    means[1:] = np.column_stack([np.cos(2 * np.pi * counts / T), np.sin(2 * np.pi * counts / T),
                                 counts / T, np.zeros((150, 5))])
    _, explained, _, _ = manifold.pca_variance(means, 1, 150)
    assert explained[:3].sum() > 0.99


def test_linebreak_error_summary():
    # width 20: a break at 15 chars before an 8-char word is valid (+3); before a 3-char word it is early (-2)
    df = pd.DataFrame([dict(width=20, line_index=1, over_word=15 + 8 - 20),
                       dict(width=20, line_index=2, over_word=15 + 3 - 20)])
    summary = evaluate.summarize(df)
    assert abs(summary["l1"] - 2.5) < 1e-9 and abs(summary["pct_early"] - 50.0) < 1e-9


def test_committed_configs_validate():
    lines = (paths.CONFIGS_DIR / "adapters.jsonl").read_text().splitlines()
    configs = [json.loads(line) for line in lines if line.strip()]
    assert {c["config_id"] for c in configs} == {"base", "task_lora", "geometry_lora"}
    for config in configs:
        train.check_config(config)
        full = train._defaults(config)
        assert full["edit_layers"] == [0, 5] and full["rank"] == 1 and full["geom_layer"] == paths.COUNT_LAYER


def test_sweep_configs_validate():
    configs = [json.loads(line) for name in ("hparam_sweep.jsonl", "nospread.jsonl")
               for line in (paths.CONFIGS_DIR / name).read_text().splitlines() if line.strip()]
    assert len(configs) == 47 and len({c["config_id"] for c in configs}) == 47
    for config in configs:
        train.check_config(config)
        full = train._defaults(config)
        assert full["method"] == "geometry" and full["seeds"] == [1, 2, 3]
        assert full["edit_layers"] == [0, full["geom_layer"]]  # edit every block up to the supervised layer


def test_count_regression_targets_are_the_counts():
    # direct regression: targets are the literal lookup, and the loss is zero at the targets
    frame = identity_frame()
    P, b, lookup = targets.build("count_regression", frame, 150, "cpu")
    assert P.shape == (1, 3)
    rng = np.random.default_rng(0)
    counts = rng.integers(1, 151, 2000)
    points = (counts / T + 0.1 * rng.standard_normal(2000))[:, None]
    frozen = losses.freeze_targets(points, counts, lookup, alpha_r=1.0, literal=True)
    assert torch.allclose(frozen["targets"][frozen["has_target"]], lookup[frozen["has_target"]])
    c = torch.as_tensor(counts)
    assert float(losses.geometry_loss(lookup[c].float(), c, frozen, 0.5, 1.0)) == 0.0
    assert float(losses.geometry_loss(torch.tensor(points, dtype=torch.float32), c, frozen, 0.5, 1.0)) > 0.0


def test_count_line_shares_the_helix_count_axis():
    frame = identity_frame()
    P_line, b_line, lookup_line = targets.build("linear", frame, 150, "cpu")
    P_helix, b_helix, lookup_helix = targets.build("helix", frame, 150, "cpu")
    assert torch.allclose(P_line[0], P_helix[2]) and torch.allclose(lookup_line[:, 0], lookup_helix[:, 2])
