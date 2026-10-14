import numpy as np
import torch

from months.losses import geometry_loss, per_prompt_loss, per_prompt_loss_with_spread
from months.losses.geometry import ideal_ring, within_class_spread, class_centroids
from months.metrics import ring_r2
from months.planes.records import build_plane_record


def synthetic_plane(period=5, points_per_class=8, noise=0.15, seed=0):
    """A noisy ring in a random 2-D probe plane of a 16-dim space, with its plane record."""
    rng = np.random.default_rng(seed)
    hidden = 16
    weight = np.linalg.qr(rng.normal(size=(hidden, 2)))[0].T.astype(np.float32)  # [2, hidden]
    bias = np.zeros(2, dtype=np.float32)
    values = np.repeat(np.arange(2, 2 + 10), points_per_class)          # values 2..11
    angle = 2 * np.pi * (values % period) / period
    ring = 0.8 * np.stack([np.cos(angle), np.sin(angle)], axis=1)
    points = ring + noise * rng.normal(size=ring.shape)
    record = build_plane_record(weight, bias, points, values, period)
    return record, torch.tensor(points, dtype=torch.float64), values


def test_loss_is_zero_at_fixed_targets():
    record, points, values = synthetic_plane()
    classes = record["classes"]
    # move every point exactly onto its fixed target at the base radius and base spread
    from months.losses.geometry import fixed_targets
    radius = torch.tensor(record["base_centroid_radius"], dtype=torch.float64)
    targets = fixed_targets(record, classes, record["period"], radius, points)
    centroids, class_index, _ = class_centroids(points, values, classes)
    moved = points - centroids[class_index] + targets[class_index]
    loss, info = geometry_loss(moved, values, record["period"], record, 1.0, 1.0, 0.3, "fixed")
    assert abs(info["mean_term"]) < 1e-8
    assert abs(info["spread_ratio"] - 1.0) < 1e-6
    assert float(loss) < 1e-8


def test_canonical_loss_equals_mean_plus_weighted_spread():
    record, points, values = synthetic_plane()
    loss, info = geometry_loss(points, values, record["period"], record, 1.45, 1.25, 0.3, "fixed")
    spread_term = (info["spread_ratio"] - 1.25) ** 2
    assert abs(float(loss) - (info["mean_term"] + 0.3 * spread_term)) < 1e-9


def test_per_prompt_decomposition():
    """per-prompt loss == size-weighted centroid term + sigma^2 / r0^2 (exact identity)."""
    record, points, values = synthetic_plane()
    from months.losses.geometry import fixed_targets
    classes = record["classes"]
    radius = torch.tensor(record["base_centroid_radius"] * 1.45, dtype=torch.float64)
    targets = fixed_targets(record, classes, record["period"], radius, points)
    centroids, class_index, counts = class_centroids(points, values, classes)
    weighted_centroid_term = ((counts / counts.sum()) * (centroids - targets).square().sum(dim=1)).sum()
    sigma = within_class_spread(points, class_index, centroids)
    expected = (weighted_centroid_term + sigma ** 2) / record["base_centroid_radius"] ** 2
    loss, _ = per_prompt_loss(points, values, record["period"], record, 1.45, 1.25, 0.0)
    assert abs(float(loss) - float(expected)) < 1e-9


def test_per_prompt_spread_reduces_to_per_prompt_at_zero_weight():
    record, points, values = synthetic_plane()
    a, _ = per_prompt_loss(points, values, record["period"], record, 1.45, 1.25, 0.0)
    b, _ = per_prompt_loss_with_spread(points, values, record["period"], record, 1.45, 1.25, 0.0)
    assert abs(float(a) - float(b)) < 1e-12


def test_shuffled_targets_change_the_loss():
    record, points, values = synthetic_plane()
    plain, _ = geometry_loss(points, values, record["period"], record, 1.45, 1.25, 0.3, "fixed")
    shuffled, _ = geometry_loss(points, values, record["period"], record, 1.45, 1.25, 0.3, "fixed",
                                shuffled_targets=True)
    assert float(shuffled) > float(plain)
    mapping = record["shuffle_mapping"].numpy()
    assert sorted(mapping.tolist()) == list(range(len(mapping)))


def test_dynamic_frame_matches_fixed_frame_on_base_points():
    """On the base model's own points the Procrustes fit reproduces the stored frame."""
    record, points, values = synthetic_plane()
    fixed, info_fixed = geometry_loss(points, values, record["period"], record, 1.0, 1.0, 0.3, "fixed")
    dynamic, info_dynamic = geometry_loss(points, values, record["period"], record, 1.0, 1.0, 0.3, "dynamic")
    assert abs(float(fixed) - float(dynamic)) < 1e-6
    assert abs(abs(info_dynamic["orientation_determinant"]) - 1.0) < 1e-6


def test_plane_record_fields_and_frame():
    record, _, _ = synthetic_plane()
    for key in ("weight", "bias", "period", "classes", "class_centroids", "base_centroid_radius",
                "base_within_rms", "base_center", "base_orientation", "orientation_determinant",
                "shuffle_mapping"):
        assert key in record
    orientation = record["base_orientation"].double()
    assert torch.allclose(orientation.T @ orientation, torch.eye(2, dtype=torch.float64), atol=1e-5)
    assert abs(abs(record["orientation_determinant"]) - 1.0) < 1e-5


def test_ring_r2_on_ideal_and_permuted_ring():
    values = np.arange(2, 97)
    angle = 2 * np.pi * values / 10
    points = 2.0 * np.stack([np.cos(angle), np.sin(angle)], axis=1) + 0.3
    assert abs(ring_r2(points, values, 10) - 1.0) < 1e-9
    permuted = np.random.default_rng(1).permutation(values)
    assert ring_r2(points, permuted, 10) < 0.1


def test_stored_target_shape_replaces_the_ring():
    # Appendix A.3.3: a record with target_unit is pulled to center + radius * target_unit
    import torch
    from months.losses.geometry import fixed_targets
    unit = torch.tensor([[1.0, 0.0], [0.0, 2.0], [-1.0, -2.0]])
    record = {"base_center": torch.tensor([0.5, -0.5]), "target_unit": unit}
    out = fixed_targets(record, [1, 2, 3], 3, torch.tensor(2.0), torch.zeros(1))
    assert torch.allclose(out, torch.tensor([0.5, -0.5]) + 2.0 * unit)
