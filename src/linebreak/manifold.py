"""Characterising the character-count manifold: its PCA structure and its causal role.

Two analyses on the layer-``COUNT_LAYER`` per-count means:

* PCA (Fig. 5b): the top three principal components of the per-count means (counts
  20-150) capture 96.8% of their variance and trace a helix.
* Zero-ablation (Fig. 5c and Appendix B.4): projecting the residual stream off the
  count subspace at block ``COUNT_LAYER`` and re-measuring linebreaking error. Ablating
  the top-3 (or top-6) PCA directions inflates the error sharply and increasingly with
  width, while ablating a random subspace of the same size has no effect.
"""
from __future__ import annotations

import numpy as np

from . import paths

ABLATION_WIDTHS = list(range(20, 301, 10))  # every 10 widths, 20-300
SUBSPACE_CMIN, SUBSPACE_CMAX = 20, 150

# The subspace to project out at block COUNT_LAYER; set by run_ablation, read by the hook.
_ablation = {"V": None}


def pca_variance(means: np.ndarray, cmin: int, cmax: int):
    """Return ``(components[k,4096], explained_variance_ratio, projected)`` for present counts."""
    counts = np.arange(len(means))
    ok = ~np.isnan(means[:, 0]) & (counts >= cmin) & (counts <= cmax)
    M = means[ok].astype(np.float64)
    U, S, Vt = np.linalg.svd(M - M.mean(0), full_matrices=False)
    evr = (S ** 2) / (S ** 2).sum()
    projected = (M - M.mean(0)) @ Vt.T
    return Vt, evr, projected, counts[ok]


def build_subspaces(means_to300: np.ndarray, frame, rng) -> dict[str, np.ndarray]:
    """The subspaces to ablate: top-6/3 PCA of the clean means, the helix frame, and matched random."""
    Vt, evr, _, _ = pca_variance(means_to300, SUBSPACE_CMIN, SUBSPACE_CMAX)
    helix = np.linalg.qr(np.stack([frame.u_cos, frame.u_sin, frame.w_count]).T)[0].T
    return {
        "pca6": Vt[:6], "pca3": Vt[:3], "helix": helix,
        "rand6": np.linalg.qr(rng.standard_normal((4096, 6)))[0].T,
        "rand3": np.linalg.qr(rng.standard_normal((4096, 3)))[0].T,
        "_var6": float(evr[:6].sum()), "_var3": float(evr[:3].sum()),
    }


def _hook(module, inp, out):
    V = _ablation["V"]
    if V is None:
        return out
    h = out[0] if isinstance(out, tuple) else out
    B, L, D = h.shape
    flat = h.reshape(-1, D)
    flat = flat - (flat @ V.T) @ V                       # project the subspace out
    h = flat.reshape(B, L, D)
    return (h,) + out[1:] if isinstance(out, tuple) else h


def _doc_errors(model, tok, doc, width, newline_ids, max_seq_len, device):
    from . import evaluate
    rows = evaluate.collect_breaks(model, tok, [doc], width, newline_ids, max_seq_len, device)
    return np.array([L + wl - width for (li, L, wl, tl, _) in rows if li >= 1], float)


def _condition_stats(model, tok, docs, width, newline_ids, V, max_seq_len, device, rng, n_boot=300):
    import torch

    _ablation["V"] = V
    per_doc = [_doc_errors(model, tok, d, width, newline_ids, max_seq_len, device) for d in docs]
    _ablation["V"] = None
    per_doc = [a for a in per_doc if len(a)]

    def stats(arrs):
        v = np.concatenate(arrs) if arrs else np.array([0.0])
        return np.abs(v).mean(), np.clip(-v, 0, None).mean(), 100 * (v < 0).mean()

    l1, pl1, prem = stats(per_doc)
    boot = [stats([per_doc[j] for j in rng.integers(0, len(per_doc), len(per_doc))]) for _ in range(n_boot)]
    boot = np.array(boot)
    return dict(l1=float(l1), l1_se=float(boot[:, 0].std()),
                prem_l1=float(pl1), prem_pct=float(prem),
                n=int(sum(len(a) for a in per_doc)))


def run_ablation(num_docs: int, widths: list[int]) -> None:
    """Fig. 5c: linebreaking error under zero-ablation of each subspace, per width."""
    import torch

    from . import corpus, harvest, wrapping
    from .probes import HelixFrame

    layer = paths.COUNT_LAYER
    means = np.load(paths.manifold_dir() / f"mean_hiddens_layer{layer}_to300.npy")
    frame = HelixFrame.load()
    rng = np.random.default_rng(0)
    subs = build_subspaces(means, frame, rng)
    print(f"  PCA of clean means: top-3={subs['_var3']*100:.1f}% top-6={subs['_var6']*100:.1f}% var", flush=True)

    model, tok = harvest.load_model()
    device = next(model.parameters()).device
    newline_ids = wrapping.newline_token_ids(tok)
    to_bf16 = lambda A: torch.tensor(A, dtype=torch.bfloat16, device=device)
    conditions = [("base", None), ("pca6", to_bf16(subs["pca6"])), ("pca6_rand", to_bf16(subs["rand6"])),
                  ("pca3", to_bf16(subs["pca3"])), ("pca3_rand", to_bf16(subs["rand3"])),
                  ("helix", to_bf16(subs["helix"]))]

    docs = corpus.load_docs(paths.corpus_dir() / "eval_docs.txt")[:num_docs]
    handle = model.model.layers[layer].register_forward_hook(_hook)
    result = dict(widths=widths, conditions=[c for c, _ in conditions],
                  var6=subs["_var6"], var3=subs["_var3"])
    for w in widths:
        print(f"  width {w}:", flush=True)
        for name, V in conditions:
            r = _condition_stats(model, tok, docs, w, newline_ids, V, 1024, device, np.random.default_rng(1))
            result[f"W{w}__{name}"] = r["l1"]
            result[f"W{w}__{name}__se"] = r["l1_se"]
            print(f"    {name:12s} L1={r['l1']:6.2f}±{r['l1_se']:.2f}  %prem={r['prem_pct']:.1f}", flush=True)
    handle.remove()
    out = paths.probes_dir() / "intervene_ablate_lberror.npz"
    np.savez(out, **result)
    print(f"  saved {out}", flush=True)
