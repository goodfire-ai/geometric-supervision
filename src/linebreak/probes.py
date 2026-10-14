"""The frozen readout of the character-count helix, and how it is fit.

Three linear directions read the count geometry out of the layer-``COUNT_LAYER``
residual stream:

    x = u_cos . h + b_cos      y = u_sin . h + b_sin      z = (w_count . h + b_count) / T

``u_cos, u_sin`` are a Fourier pair fit to ``(cos, sin)`` of the count phase at period
``T``; ``w_count`` is a linear count probe. Together they map an activation to a point
that, for an ideal helix, sits at ``(cos 2piv/T, sin 2piv/T, v/T)``. This is the frame
the geometry LoRA supervises toward and that every count-manifold figure is read in, so
it is fit once and then frozen.

The period ``T`` is the model's own: the count at which the representation completes one
turn in the Fourier plane, recovered from the phase of the per-count means (``fit_period``),
which gives ``T ~ 154.77`` characters.
"""
from __future__ import annotations

import numpy as np

from . import paths

RIDGE_LAMBDA = 10.0
FIT_WIDTH = 150            # documents wrapped to this width when fitting the probes
FIT_MIN_COUNT = 20        # ignore the special low counts right after a newline

# The helix period is kept fixed at the value in the paper, found by a phase fit to the
# base model's per-count means (Appendix B.3). We don't re-fit it on each run because the
# fit depends on which counts are included. `fit` still stores a fresh fit for reference.
PERIOD = 154.77


# --------------------------------------------------------------------------- #
#  Pure geometry: period fit and the frozen frame (no model, no GPU)
# --------------------------------------------------------------------------- #
def fit_period(pc1: np.ndarray, pc2: np.ndarray) -> tuple[float, float, float]:
    """Recover the helix period from the phase of ordered per-count points.

    The angle ``theta_v = atan2(pc2, pc1)`` advances by ``2pi/T`` per count. Unwrapping
    it across counts and fitting ``theta_v = omega*v + b`` by least squares gives the
    period ``T = 2pi/|omega|``. Returns ``(period, phase R^2, degrees per step)``.
    """
    theta = np.arctan2(pc2, pc1)
    theta_u = np.unwrap(theta)
    v = np.arange(len(theta_u))
    A = np.vstack([v, np.ones_like(v)]).T
    (omega, b), *_ = np.linalg.lstsq(A, theta_u, rcond=None)
    pred = A @ np.array([omega, b])
    ss_res = float(np.sum((theta_u - pred) ** 2))
    ss_tot = float(np.sum((theta_u - theta_u.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    period = 2 * np.pi / abs(omega) if omega != 0 else float("inf")
    return period, r2, float(np.degrees(omega))


class HelixFrame:
    """The frozen ``(u_cos, u_sin, w_count/T)`` readout and the ideal helix it targets."""

    def __init__(self, u_cos, u_sin, w_count, b_cos, b_sin, b_count, period):
        self.u_cos = np.asarray(u_cos, np.float64)
        self.u_sin = np.asarray(u_sin, np.float64)
        self.w_count = np.asarray(w_count, np.float64)
        self.b_cos, self.b_sin, self.b_count = float(b_cos), float(b_sin), float(b_count)
        self.period = float(period)

    @classmethod
    def load(cls, path=None) -> "HelixFrame":
        p = np.load(path or (paths.probes_dir() / f"probes_layer{paths.COUNT_LAYER}.npz"))
        return cls(p["u_cos"], p["u_sin"], p["w_count"],
                   p["b_cos"], p["b_sin"], p["b_count"], p["period"])

    def project(self, H: np.ndarray) -> np.ndarray:
        """``H[...,4096] -> [...,3]`` in the fixed frame (count axis divided by ``T``)."""
        H = np.asarray(H, np.float64)
        return np.stack([H @ self.u_cos + self.b_cos,
                         H @ self.u_sin + self.b_sin,
                         (H @ self.w_count + self.b_count) / self.period], axis=-1)

    def ideal(self, counts) -> np.ndarray:
        """The ideal helix point ``(cos, sin, v/T)`` at each count."""
        c = np.asarray(counts, np.float64)
        th = 2 * np.pi * c / self.period
        return np.stack([np.cos(th), np.sin(th), c / self.period], axis=-1)

    def residual(self, projected: np.ndarray, counts) -> np.ndarray:
        return np.sqrt(((projected - self.ideal(counts)) ** 2).sum(-1))

    def r2(self, projected: np.ndarray, counts) -> float:
        """Fixed-frame helix ``R^2 = 1 - SS(P - ideal) / SS(P - mean P)`` (no alignment)."""
        ideal = self.ideal(counts)
        return float(1 - ((projected - ideal) ** 2).sum() / ((projected - projected.mean(0)) ** 2).sum())

    def procrustes_r2(self, projected: np.ndarray, counts) -> float:
        """Helix ``R^2`` after aligning the ideal helix to the points (rotation+scale+translation).

        This is the paper's metric: an idealised period-``T`` helix is fit to the cloud by
        orthogonal Procrustes (with uniform scale and translation), and ``R^2`` is the fraction
        of the cloud's variance the aligned helix captures. Unlike the fixed-frame ``r2`` it does
        not assume the probe readout already lands at unit radius, so it isolates *shape*.
        """
        P = np.asarray(projected, np.float64)
        ideal = self.ideal(counts)
        muP, muI = P.mean(0), ideal.mean(0)
        Pc, Ic = P - muP, ideal - muI
        U, S, Vt = np.linalg.svd(Pc.T @ Ic)
        rot = U @ Vt
        scale = S.sum() / (Ic ** 2).sum()
        fit = scale * (Ic @ rot.T) + muP
        return float(1 - ((P - fit) ** 2).sum() / ((P - muP) ** 2).sum())

    def per_count_means(self, H, counts, cmin=15, cmax=150, min_tokens=40):
        """Per-count mean projections over ``[cmin, cmax]`` (counts with enough tokens)."""
        counts = np.asarray(counts)
        present = np.array([j for j in range(cmin, cmax + 1) if (counts == j).sum() >= min_tokens])
        means = np.stack([self.project(H[counts == j]).mean(0) for j in present])
        return present, means


# --------------------------------------------------------------------------- #
#  Fitting (GPU): ridge probes on layer-COUNT_LAYER activations
# --------------------------------------------------------------------------- #
def ridge_fit(X, Y, lam: float):
    """Normal-equations ridge with an unpenalised bias column. ``X:[N,D] Y:[N,K] -> [D+1,K]``."""
    import torch

    n, d = X.shape
    Xb = torch.cat([X, torch.ones(n, 1, device=X.device, dtype=X.dtype)], dim=1)
    A = Xb.T @ Xb
    reg = lam * torch.eye(d + 1, device=X.device, dtype=X.dtype)
    reg[-1, -1] = 0.0
    return torch.linalg.solve(A + reg, Xb.T @ Y)


def fit(num_docs: int = 250, seed: int = 1) -> None:
    """Fit and save the frozen frame from layer-``COUNT_LAYER`` activations (1 GPU).

    Documents (the disjoint ``diag`` pool) are wrapped to width ``FIT_WIDTH``; the count
    period is read from the per-count means, then the Fourier and count probes are fit at
    that period by ridge regression. Saves ``probes_layer{COUNT_LAYER}.npz``.
    """
    import torch

    from . import corpus, harvest

    layer = paths.COUNT_LAYER
    docs = corpus.load_docs(paths.corpus_dir() / "diag_docs.txt")[:num_docs]
    model, tok = harvest.load_model()
    H, counts = harvest.layer_counts(model, tok, docs, layer=layer, width=FIT_WIDTH, seed=seed)
    keep = counts >= FIT_MIN_COUNT
    H, counts = H[keep], counts[keep]
    print(f"  {H.shape[0]} tokens (count>={FIT_MIN_COUNT}) at layer {layer}", flush=True)

    # The period is fixed (see PERIOD); the fresh phase fit is only stored for reference.
    order = np.array(sorted(np.unique(counts)))
    means = np.stack([H[counts == c].mean(0) for c in order])
    means_centered = means - means.mean(0)
    _, _, vt = np.linalg.svd(means_centered, full_matrices=False)
    pcs = means_centered @ vt[:2].T
    fitted, phase_r2, deg = fit_period(pcs[:, 0], pcs[:, 1])
    period = PERIOD
    print(f"  period T={period:.2f} (fixed); fresh phase fit over these counts = "
          f"{fitted:.2f} (R^2={phase_r2:.3f}, {deg:+.3f} deg/count)", flush=True)

    Ht = torch.as_tensor(H, dtype=torch.float32, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    ct = torch.as_tensor(counts, dtype=torch.float32, device=Ht.device)
    phi = 2 * np.pi * ct / period
    beta_count = ridge_fit(Ht, ct[:, None], RIDGE_LAMBDA)
    beta_fourier = ridge_fit(Ht, torch.stack([torch.cos(phi), torch.sin(phi)], 1), RIDGE_LAMBDA)

    out = paths.probes_dir() / f"probes_layer{layer}.npz"
    np.savez(out,
             u_cos=beta_fourier[:-1, 0].cpu().numpy(), b_cos=float(beta_fourier[-1, 0]),
             u_sin=beta_fourier[:-1, 1].cpu().numpy(), b_sin=float(beta_fourier[-1, 1]),
             w_count=beta_count[:-1, 0].cpu().numpy(), b_count=float(beta_count[-1, 0]),
             period=period, fitted_period=fitted, min_count=FIT_MIN_COUNT, lam=RIDGE_LAMBDA, layer=layer)
    print(f"  saved {out}", flush=True)
