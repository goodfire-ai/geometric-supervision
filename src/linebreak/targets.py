"""Idealised target geometries the LoRA can be supervised toward.

Given the frozen readout frame (``probes.HelixFrame``), a target consists of a readout
matrix ``P`` (rows are the probe directions), a bias ``b``, and a per-count ``lookup``
of ideal points in that readout:

    none    : no geometry (the task-LoRA baseline; supervision is off)
    circle  : the Fourier pair only; target ``(cos 2piv/T, sin 2piv/T)``
    helix   : the Fourier pair plus the count axis; target ``(cos, sin, v/T)``

``helix`` is the one used for the paper's geometry LoRA: the count axis separates
high counts that a bare circle would fold back onto low ones.

Alternate targets (Appendix B.3.2; same loss, LoRA and data, only the readout or the target changes):
    linear    : the count axis only; per-count means pulled to evenly spaced ``v/T`` (the helix's
                count coordinate, with the same loss)
    count_regression : the count axis only; every token's readout pulled to its own count
                ``v/T`` (per-token squared error; no centering, alpha_r rescaling or spread term),
                i.e. direct regression of the count through the frozen count probe
    empirical : the helix readout, but the target is the base model's own per-count means
                (scaled by alpha_r about their center) instead of the ideal helix
    pca3      : the ideal helix read out of the top-3 PCA directions of the per-count means
    pca3_empirical : the same PCA readout, with the base model's own per-count means as the target
    random3   : the ideal helix read out of a fixed random 3-dimensional subspace
"""
from __future__ import annotations

MODES = ("none", "circle", "helix", "linear", "empirical", "pca3", "random3", "pca3_empirical",
         "count_regression")


def build(mode: str, frame, max_count: int, device):
    """Return ``(P, b, lookup)`` torch tensors on ``device`` (``(None, None, None)`` for 'none')."""
    if mode == "none":
        return None, None, None
    if mode not in MODES:
        raise ValueError(f"unknown target mode {mode!r}; choose from {MODES}")
    import math

    import torch

    T = frame.period
    u_cos = torch.tensor(frame.u_cos, dtype=torch.float32)
    u_sin = torch.tensor(frame.u_sin, dtype=torch.float32)
    w_count = torch.tensor(frame.w_count, dtype=torch.float32)
    c = torch.arange(max_count + 1, dtype=torch.float32)
    phase = 2 * math.pi * c / T

    helix_lookup = torch.stack([torch.cos(phase), torch.sin(phase), c / T], dim=1)
    helix_P = torch.stack([u_cos, u_sin, w_count / T])
    helix_b = torch.tensor([frame.b_cos, frame.b_sin, frame.b_count / T])
    if mode == "circle":
        P = torch.stack([u_cos, u_sin])
        b = torch.tensor([frame.b_cos, frame.b_sin])
        lookup = torch.stack([torch.cos(phase), torch.sin(phase)], dim=1)
    elif mode == "helix":
        P, b, lookup = helix_P, helix_b, helix_lookup
    elif mode in ("linear", "count_regression"):
        P = (w_count / T)[None, :]
        b = torch.tensor([frame.b_count / T])
        lookup = (c / T)[:, None]
    elif mode == "empirical":
        # base model's per-count means (03_count_means.py) read through the helix frame;
        # counts without a mean get NaN and are dropped by freeze_targets
        import numpy as np
        from . import paths
        means = np.load(paths.manifold_dir() / f"mean_hiddens_layer{paths.COUNT_LAYER}_to300.npy")
        means = torch.tensor(means[:max_count + 1], dtype=torch.float32)
        P, b = helix_P, helix_b
        lookup = means @ P.T + b
    elif mode == "random3":
        import numpy as np
        Q = np.linalg.qr(np.random.default_rng(0).standard_normal((u_cos.shape[0], 3)))[0]
        P, b, lookup = torch.tensor(Q.T.copy(), dtype=torch.float32), torch.zeros(3), helix_lookup
    else:  # pca3, pca3_empirical
        import numpy as np
        from . import manifold, paths
        means = np.load(paths.manifold_dir() / f"mean_hiddens_layer{paths.COUNT_LAYER}_to300.npy")
        Vt, _, _, _ = manifold.pca_variance(means, manifold.SUBSPACE_CMIN, manifold.SUBSPACE_CMAX)
        P, b = torch.tensor(Vt[:3].copy(), dtype=torch.float32), torch.zeros(3)
        if mode == "pca3":
            lookup = helix_lookup
        else:   # counts without a mean get NaN and are dropped by freeze_targets
            lookup = torch.tensor(means[:max_count + 1], dtype=torch.float32) @ P.T
    return P.to(device), b.to(device), lookup.to(device)
