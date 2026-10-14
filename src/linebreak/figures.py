"""The linebreak figures and the headline numbers they report.

Each figure reads pipeline outputs under ``root`` (the evaluation CSVs, activation clouds,
per-count means, ablation results and sweep evaluations) and writes a PDF and PNG to
``outputs/linebreak/figures``. A figure whose inputs are missing is skipped.
``compute_numbers`` collects the headline numbers, which ``scripts/linebreak/10_summarize.py``
prints.
"""
from __future__ import annotations

import numpy as np

from . import paths, style

MODELS = [("base", "Base Model"), ("task_lora", "Task LoRA"), ("geometry_lora", "Geometry LoRA")]
IN_RANGE = (15, 150)
FULL_RANGE = (15, 300)
# Widths below 20 are excluded from the figures and numbers: there almost every word
# overflows the line, so the task is degenerate.
WMIN = 20
MAX_CONTEXT = 13   # in-context examples shown in Fig. 4
MIN_BREAKS = 15    # a (context, width) cell needs this many breaks to count


def _plt():
    """Headless matplotlib with the paper style applied."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    style.setup()
    return plt


def _load_errors(root, tag):
    """Evaluation CSVs for one model: ``errors_base.csv``, or one per seed of a LoRA."""
    import pandas as pd
    eval_dir = root / "eval"
    if tag == "base":
        files = [eval_dir / "errors_base.csv"] if (eval_dir / "errors_base.csv").exists() else []
    else:
        files = sorted(eval_dir.glob(f"errors_{tag}_seed*.csv"))
    return [pd.read_csv(f) for f in files]


def cell_mean_l1(df) -> float:
    """The headline linebreaking error: the mean |error| of each (in-context examples, width)
    cell of Fig. 4, averaged over cells. Sparse cells (hard, wide-line, short-context breaks)
    therefore count as much as the many easy ones."""
    d = df[(df.line_index >= 1) & (df.line_index <= MAX_CONTEXT) & (df.width >= WMIN)]
    cells = [g.over_word.abs().mean() for _, g in d.groupby(["line_index", "width"]) if len(g) >= MIN_BREAKS]
    return float(np.mean(cells)) if cells else float("nan")


def _style_axes(ax, grid=True):
    ax.tick_params(labelsize=13, colors="black")
    if grid:
        ax.grid(True, color=style.C["grid"], alpha=0.8)
        ax.set_axisbelow(True)


# --------------------------------------------------------------------------- #
#  Figure 4: linebreaking error
# --------------------------------------------------------------------------- #
def fig_l1_heatmap(root) -> None:
    """Fig. 4a-f: linebreaking error by line width (x) and in-context examples (y).

    Top row: each model's error (seed 1 for the LoRAs), smoothed between cells. Bottom row:
    differences between models, cell by cell, green where the first model has lower error.
    """
    from matplotlib.colors import LinearSegmentedColormap, LogNorm, TwoSlopeNorm
    frames = {tag: _load_errors(root, tag) for tag, _ in MODELS}
    if not all(frames.values()):
        print("  (skip l1_heatmap: need evaluation CSVs for all three models)")
        return
    plt = _plt()

    contexts = np.arange(1, MAX_CONTEXT + 1)
    widths = sorted({int(w) for f in frames.values() for w in f[0].width.unique() if w >= WMIN})
    column = {w: i for i, w in enumerate(widths)}

    def grid(df):
        values = np.full((len(contexts), len(widths)), np.nan)
        d = df[(df.line_index >= 1) & (df.line_index <= MAX_CONTEXT)]
        for (context, width), g in d.groupby(["line_index", "width"]):
            if len(g) >= MIN_BREAKS and int(width) in column:
                values[int(context) - 1, column[int(width)]] = g.over_word.abs().mean()
        return values

    grids = {tag: grid(f[0]) for tag, f in frames.items()}
    extent = [widths[0], widths[-1], contexts[0], contexts[-1]]
    values = np.concatenate([g.ravel() for g in grids.values()])
    values = values[~np.isnan(values)]
    top_norm = LogNorm(vmin=max(0.5, float(values[values > 0].min())), vmax=float(values.max()))
    diff_norm = TwoSlopeNorm(vmin=-20.0, vcenter=0.0, vmax=20.0)
    green_red = LinearSegmentedColormap.from_list("gwr", ["#1a9850", "#f7f7f7", "#b2182b"])

    fig = plt.figure(figsize=(13.8, 7.4))
    gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 0.055], height_ratios=[1, 1], hspace=0.32, wspace=0.17)

    def panel(row, col, values, title, cmap, norm, interpolation):
        ax = fig.add_subplot(gs[row, col])
        image = ax.imshow(values, aspect="auto", origin="lower", extent=extent, cmap=cmap, norm=norm,
                          interpolation=interpolation)
        ax.axvline(150, color="0.3", ls=":", lw=1.1)   # the widest width seen in training
        ax.set_title(title, fontsize=17, color="black")
        ax.grid(False)
        ax.set_xlim(widths[0], widths[-1])
        ax.set_xticks([100, 200, 300])
        ax.tick_params(labelsize=14)
        if col == 0:
            ax.set_ylabel("In-Context Examples", fontsize=16)
        else:
            ax.set_yticklabels([])
        if row == 0:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("Line Width", fontsize=16)
        return image

    for col, (tag, label) in enumerate(MODELS):
        title = "Geometry LoRA (Helix)" if tag == "geometry_lora" else label
        image = panel(0, col, grids[tag], title, "Blues", top_norm, "bilinear")
    colorbar = fig.colorbar(image, cax=fig.add_subplot(gs[0, 3]))
    colorbar.set_label("Linebreaking Error (L1)", fontsize=13)
    colorbar.ax.tick_params(labelsize=13, length=5, width=0.9, direction="out")

    differences = [(grids["task_lora"] - grids["base"], "Task LoRA $-$ Base"),
                   (grids["geometry_lora"] - grids["base"], "Geometry LoRA $-$ Base"),
                   (grids["geometry_lora"] - grids["task_lora"], "Geometry $-$ Task LoRA")]
    for col, (values, title) in enumerate(differences):
        image = panel(1, col, values, title, green_red, diff_norm, None)
    colorbar = fig.colorbar(image, cax=fig.add_subplot(gs[1, 3]))
    colorbar.set_label("$\\Delta$ L1 (chars; green: lower error)", fontsize=13)
    colorbar.ax.tick_params(labelsize=13, length=5, width=0.9, direction="out")
    style.save(fig, "fig4_l1_error_heatmap")


def _error_distribution(root, name, bins, xlim, size, log_scale, markers):
    """Histogram of the signed linebreaking error per model, seeds pooled."""
    import pandas as pd
    if not any(_load_errors(root, tag) for tag, _ in MODELS):
        print(f"  (skip {name}: no evaluation CSVs)")
        return
    plt = _plt()
    centers = 0.5 * (bins[:-1] + bins[1:])
    fig, ax = plt.subplots(figsize=size)
    ax.axvline(0, color=style.C["muted"], ls=":", lw=1.4, zorder=1)
    for tag, _ in MODELS:
        frames = _load_errors(root, tag)
        if not frames:
            continue
        df = pd.concat(frames, ignore_index=True)
        errors = df[(df.line_index >= 1) & (df.width >= WMIN)].over_word.values
        density = np.histogram(errors, bins=bins, density=True)[0]
        marker_style = dict(marker="o", ms=4.5, mec="white", mew=0.6) if markers else {}
        ax.plot(centers, density, color=style.MODEL_COLORS[tag], lw=2.2 if markers else 2.4,
                label=style.MODEL_LABELS[tag], **marker_style)
    if log_scale:
        ax.set_yscale("log")
    ax.set_xlabel("Linebreaking Error (Offset from Target)", fontsize=16)
    ax.set_ylabel("Density (log scale)" if log_scale else "Density", fontsize=16)
    ax.set_xlim(*xlim)
    ax.tick_params(labelsize=14)
    ax.legend(fontsize=14, frameon=False, loc="upper left")
    ax.grid(False)
    fig.tight_layout()
    style.save(fig, name)


def fig_error_distribution(root) -> None:
    """Fig. 4g: distribution of the linebreaking error near the target width."""
    _error_distribution(root, "fig4g_overshoot_distribution", np.arange(-25.5, 12.5, 1.0), (-25, 12),
                        (6.0, 5.8), log_scale=False, markers=True)


def fig_error_distribution_log(root) -> None:
    """Appendix B.5: the same distribution on a log scale, showing the base model's heavy
    tail of breaks that come far too early."""
    _error_distribution(root, "figB_overshoot_logtail", np.arange(-110.5, 12.5, 2.0), (-110, 12),
                        (7.2, 5.0), log_scale=True, markers=False)


# --------------------------------------------------------------------------- #
#  Figure 5: the character-count representation
# --------------------------------------------------------------------------- #
def fig_count_probe_by_layer(root) -> None:
    """Fig. 5a: R^2 of a linear character-count probe at the output of each layer."""
    import pandas as pd
    path = root / "probes" / "probe_r2_by_layer.csv"
    if not path.exists():
        print("  (skip count_probe_by_layer: no probe_r2_by_layer.csv)")
        return
    plt = _plt()
    df = pd.read_csv(path).sort_values("layer")
    df = df[df.layer >= 0]   # layer -1 is the embeddings
    fig, ax = plt.subplots(figsize=(5.3, 4.9))
    ax.plot(df.layer, df.r2, "-o", color=style.C["ink"], lw=2.4, ms=5)
    ax.axvline(paths.COUNT_LAYER, color="0.6", ls=":", lw=1.2)   # the supervised layer
    ax.set_xlabel("Layer", fontsize=17)
    ax.set_ylabel(r"Character-Count Probe $R^2$", fontsize=17)
    ax.set_xlim(-0.5, int(df.layer.max()) + 0.5)
    ax.set_ylim(0, 1)
    _style_axes(ax)
    ax.tick_params(labelsize=15)
    fig.tight_layout()
    style.save(fig, "fig5a_count_probe_by_layer")


def fig_count_manifold_pca(root) -> None:
    """Fig. 5b: the per-count mean activations at layer 5 projected onto their top 3 PCs."""
    from . import manifold
    path = root / "manifold" / f"mean_hiddens_layer{paths.COUNT_LAYER}.npy"
    if not path.exists():
        print("  (skip count_manifold_pca: no per-count means)")
        return
    plt = _plt()
    _, _, proj, counts = manifold.pca_variance(np.load(path), 20, 150)

    fig = plt.figure(figsize=(6.4, 5.0))
    ax = fig.add_axes([0.0, 0.05, 0.70, 0.92], projection="3d")
    ax.plot(proj[:, 0], proj[:, 1], proj[:, 2], color="0.7", lw=0.8, alpha=0.5, zorder=1)
    points = ax.scatter(proj[:, 0], proj[:, 1], proj[:, 2], c=counts, cmap=style.CMAP_COUNT,
                        s=16, vmin=int(counts.min()), vmax=int(counts.max()), zorder=3)
    for axis, set_lim in enumerate((ax.set_xlim, ax.set_ylim, ax.set_zlim)):
        lo, hi = proj[:, axis].min(), proj[:, axis].max()
        set_lim(lo - 0.1 * (hi - lo), hi + 0.1 * (hi - lo))
    ax.set_xlabel("PC1", fontsize=16, labelpad=10)
    ax.set_ylabel("PC2", fontsize=16, labelpad=14)
    ax.set_zlabel("PC3", fontsize=16, labelpad=6)
    ax.tick_params(labelsize=13, colors="black", pad=3)
    ax.view_init(elev=18, azim=-60)
    colorbar = fig.colorbar(points, cax=fig.add_axes([0.805, 0.22, 0.032, 0.56]))
    colorbar.set_label("Character Count", fontsize=16)
    colorbar.ax.tick_params(labelsize=13, colors="black")
    style.save(fig, "fig5b_count_manifold_pca")


def _load_ablation(root, name):
    path = root / "probes" / "intervene_ablate_lberror.npz"
    if not path.exists():
        print(f"  (skip {name}: no ablation results)")
        return None
    return np.load(path, allow_pickle=True)


def fig_ablation(root) -> None:
    """Appendix B.4: linebreaking error by width when the count subspace at layer 5 is
    zero-ablated, against random subspaces of the same size (dashed)."""
    d = _load_ablation(root, "ablation")
    if d is None:
        return
    plt = _plt()
    widths = np.array([int(w) for w in d["widths"]])

    def series(condition, suffix=""):
        keys = [f"W{w}__{condition}{suffix}" for w in widths]
        return np.array([float(d[k]) if k in d else np.nan for k in keys])

    curves = [  # condition, label, color, line style
        ("pca6", "Ablate count 6-D (PCA)", "#b2182b", "-"),
        ("pca3", "Ablate count 3-D (PCA)", "#e08214", "-"),
        ("helix", "Ablate helix 3-D", "#2166ac", "-"),
        ("base", "Base (intact)", style.C["ink"], "-"),
        ("pca6_rand", "Ablate random 6-D", "#6a3d9a", "--"),
        ("pca3_rand", "Ablate random 3-D", "#33a02c", ":"),
    ]
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    ax.axvspan(150, widths.max(), color="#00000006", lw=0, zorder=0)   # widths not seen in training
    ax.axvline(150, color="0.5", ls=":", lw=1.4, zorder=1)
    for condition, label, color, line in curves:
        y, se = series(condition), series(condition, "__se")
        if np.all(np.isnan(y)):
            continue
        solid = line == "-"
        z = 5 if solid else 6
        if condition == "base":
            z = 4
        ax.plot(widths, y, color=color, ls=line, lw=2.4 if solid else 1.8, marker="o",
                ms=4.5 if solid else 3, mec="white", mew=0.5, label=label, zorder=z)
        if solid:
            ax.fill_between(widths, y - se, y + se, color=color, alpha=0.15, lw=0, zorder=z - 1)
    ax.set_yscale("log")
    ax.set_xlim(widths.min(), widths.max())
    ax.set_xlabel("Line Width", fontsize=15)
    ax.set_ylabel("Linebreaking Error (L1, chars)", fontsize=15)
    ax.tick_params(labelsize=13, colors="black")
    ax.grid(True, which="both", color=style.C["grid"], alpha=0.6)
    ax.set_axisbelow(True)
    ax.legend(fontsize=11.5, frameon=False, loc="upper left", ncol=1)
    fig.tight_layout()
    style.save(fig, "fig5c_ablation_lberror")


def fig_ablation_bars(root, bar_widths=(60, 100, 150, 250)) -> None:
    """Fig. 5c: the ablation at a few widths: intact model, a random 3-D subspace ablated,
    and the top 3 PCs of the count representation ablated (bars show bootstrap SEs)."""
    d = _load_ablation(root, "ablation_bars")
    if d is None:
        return
    plt = _plt()
    widths = [w for w in bar_widths if w in {int(x) for x in d["widths"]}]

    def value(width, condition, suffix=""):
        key = f"W{width}__{condition}{suffix}"
        return float(d[key]) if key in d else np.nan

    bars = [("base", "Base Model", style.C["ink"]),
            ("pca3_rand", "Ablate random 3-D", "0.68"),
            ("pca3", "Ablate count 3-D", "#e08214")]
    x = np.arange(len(widths))
    bar_width = 0.205
    fig, ax = plt.subplots(figsize=(4.15, 5.0))
    for i, (condition, label, color) in enumerate(bars):
        heights = [value(w, condition) for w in widths]
        errors = [value(w, condition, "__se") for w in widths]
        ax.bar(x + (i - (len(bars) - 1) / 2) * bar_width, heights, bar_width, yerr=errors, color=color,
               label=label, error_kw=dict(capsize=2, elinewidth=0.9, ecolor="0.3"),
               edgecolor="white", linewidth=0.4, zorder=3)
    ax.set_yscale("log")
    ax.set_ylim(2.5, 200)
    ax.set_xticks(x)
    ax.set_xticklabels([str(w) for w in widths])
    ax.set_xlabel("Line Width", fontsize=15)
    ax.set_ylabel("Linebreaking Error (L1, chars)", fontsize=15)
    ax.tick_params(labelsize=12, colors="black")
    ax.grid(True, axis="y", which="both", color=style.C["grid"], alpha=0.6)
    ax.set_axisbelow(True)
    ax.legend(fontsize=11.5, frameon=False, loc="upper left", ncol=1)
    fig.tight_layout()
    style.save(fig, "fig5c_ablation_lberror_bars")


# --------------------------------------------------------------------------- #
#  Figure 6: fit to the ideal helix
# --------------------------------------------------------------------------- #
def _cloud(root, tag, lo, hi):
    """One model's layer-5 activations for counts lo..hi, projected onto the helix
    readout, and their R^2 against the ideal helix. None if the cloud is missing."""
    from .probes import HelixFrame
    path = root / "probes" / f"cloud_layer{paths.COUNT_LAYER}_{tag}.npz"
    if not path.exists():
        return None, None, None
    d = np.load(path)
    activations, counts = d["H"].astype(np.float32), d["count"].astype(int)
    keep = (counts >= lo) & (counts <= hi)
    frame = HelixFrame.load(root / "probes" / f"probes_layer{paths.COUNT_LAYER}.npz")
    proj = frame.project(activations[keep])
    return proj, counts[keep], frame.procrustes_r2(proj, counts[keep])


def _procrustes_map(points, ideal):
    """The rotation, uniform scale and shift that best map ``ideal`` onto ``points``, as a
    function, so the aligned ideal helix can be drawn as a smooth curve."""
    mean_points, mean_ideal = points.mean(0), ideal.mean(0)
    x, y = ideal - mean_ideal, points - mean_points
    u, s, vt = np.linalg.svd(x.T @ y)
    rotation = u @ vt
    scale = s.sum() / (x ** 2).sum()
    return lambda q: scale * ((q - mean_ideal) @ rotation) + mean_points


def _fig_helix(root, lo, hi, name, panel_letters=False) -> None:
    """Each model's activations (colored by count), its per-count mean (solid) and the ideal
    helix (dashed). The ideal helix is aligned to each model's counts up to 150, where the
    representation is helical, and drawn over the whole plotted range. R^2 is over all
    plotted tokens."""
    from matplotlib.lines import Line2D

    from .probes import HelixFrame
    if not any((root / "probes" / f"cloud_layer{paths.COUNT_LAYER}_{tag}.npz").exists() for tag, _ in MODELS):
        print(f"  (skip {name}: no activation clouds)")
        return
    plt = _plt()
    frame = HelixFrame.load(root / "probes" / f"probes_layer{paths.COUNT_LAYER}.npz")
    rng = np.random.RandomState(0)

    panels = []
    for tag, label in MODELS:
        proj, counts, r2 = _cloud(root, tag, lo, hi)
        if proj is None:
            panels.append((tag, label, None))
            continue
        in_range = counts <= 150
        align = _procrustes_map(proj[in_range], frame.ideal(counts[in_range]))
        mean_counts = np.array([c for c in range(lo, hi + 1) if (counts == c).sum() >= 20])
        means = np.stack([proj[counts == c].mean(0) for c in mean_counts]) if len(mean_counts) else None
        panels.append((tag, label, dict(proj=proj, counts=counts, align=align, r2=r2,
                                        mean_counts=mean_counts, means=means)))

    clouds = [d["proj"][:, :2] for _, _, d in panels if d is not None]
    half_width = np.ceil(np.percentile(np.abs(np.vstack(clouds)), 99.5) * 2) / 2
    z_lo, z_hi = 0.0, 1.0   # the count axis (count / T) is shown from 0 to 1

    def visible(p):
        return ((np.abs(p[:, 0]) <= half_width) & (np.abs(p[:, 1]) <= half_width)
                & (p[:, 2] >= z_lo) & (p[:, 2] <= z_hi))

    fig = plt.figure(figsize=(17, 6.2))
    count_grid = np.linspace(lo, hi, 400)
    for i, (tag, label, d) in enumerate(panels, 1):
        ax = fig.add_subplot(1, 3, i, projection="3d")
        if d is None:
            ax.text2D(0.5, 0.5, f"{label}\n(no cloud)", transform=ax.transAxes, ha="center", va="center")
            continue
        sample = rng.choice(len(d["counts"]), size=min(3000, len(d["counts"])), replace=False)
        p, c = d["proj"][sample], d["counts"][sample]
        shown = visible(p)
        points = ax.scatter(p[shown, 0], p[shown, 1], p[shown, 2], c=c[shown], cmap=style.CMAP_COUNT,
                            s=5, alpha=0.42, vmin=lo, vmax=hi, linewidths=0, zorder=2)
        ideal = d["align"](frame.ideal(count_grid))
        shown = visible(ideal)
        ax.plot(ideal[shown, 0], ideal[shown, 1], ideal[shown, 2], ls="--", color="0.12", lw=2.6, zorder=6)
        if d["means"] is not None:
            means = d["means"]
            ax.plot(means[:, 0], means[:, 1], np.full(len(means), z_lo), color="0.7", lw=1.6, alpha=0.8, zorder=1)
            ax.plot(means[:, 0], means[:, 1], means[:, 2], color=style.MODEL_COLORS[tag], lw=3.0, zorder=4)
        ax.text2D(0.5, 0.93, label, transform=ax.transAxes, ha="center", va="top", fontsize=19, color="black")
        ax.text2D(0.06, 0.72, f"$R^2$ = {d['r2']:.2f}", transform=ax.transAxes, ha="left", va="top",
                  fontsize=15, color="0.30")
        if panel_letters:
            ax.text2D(0.0, 0.99, f"({'abc'[i - 1]})", transform=ax.transAxes, ha="left", va="top",
                      fontsize=18, fontweight="bold", color="black")
        ax.set_xlim(-half_width, half_width)
        ax.set_ylim(-half_width, half_width)
        ax.set_zlim(z_lo, z_hi)
        ticks = np.arange(-np.floor(half_width), np.floor(half_width) + 1, 1)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xlabel(r"$u_{\cos}\!\cdot h$", fontsize=17, labelpad=9)
        ax.set_ylabel(r"$u_{\sin}\!\cdot h$", fontsize=17, labelpad=9)
        ax.set_zlabel(r"$w_{\mathrm{count}}\!\cdot h\,/\,T$", fontsize=15, labelpad=2)
        ax.tick_params(labelsize=13)
        ax.view_init(elev=14, azim=-60)
        colorbar = fig.colorbar(points, ax=ax, shrink=0.46, pad=0.14, aspect=16)
        colorbar.set_label("Character Count", fontsize=14)
        colorbar.ax.tick_params(labelsize=12)
        colorbar.solids.set_alpha(1)
    handles = [Line2D([0], [0], ls="--", color="0.12", lw=2.6), Line2D([0], [0], color="0.4", lw=3.0)]
    fig.legend(handles, ["Ideal Helix (Aligned)", "Empirical Count Mean"], loc="lower center",
               ncol=2, fontsize=15, frameon=False, bbox_to_anchor=(0.5, 0.02))
    fig.subplots_adjust(left=0.02, right=0.98, wspace=0.22, bottom=0.12)
    style.save(fig, name)


def fig_helix_fit(root) -> None:
    """Fig. 6: counts 15-150."""
    _fig_helix(root, *IN_RANGE, "fig6_helix_fit")


def fig_helix_fit_to300(root) -> None:
    """Appendix B.4: counts 15-300."""
    _fig_helix(root, *FULL_RANGE, "figB_helix_fit_to300", panel_letters=True)


# --------------------------------------------------------------------------- #
#  Appendix B.3.1: hyperparameter sweeps
# --------------------------------------------------------------------------- #
SWEEP_RANKS = [1, 2, 4, 8, 16]
SWEEP_LAYERS = [3, 4, 5, 6, 7]
SWEEP_RADII = [0.8, 1.0, 1.2, 1.5, 1.8, 2.4]
SWEEP_TAUS = [0.3, 0.5, 0.8, 1.0, 1.4, 2.0]


def _sweep_l1(root, config_id):
    """Headline error of one sweep configuration, averaged over its seeds (NaN if missing)."""
    import pandas as pd
    files = sorted((root / "eval").glob(f"errors_{config_id}_seed*.csv"))
    return float(np.mean([cell_mean_l1(pd.read_csv(f)) for f in files])) if files else float("nan")


def fig_hparam_sweeps(root) -> None:
    """Appendix B.3.1: error of the geometry LoRA against (a) rank, (b) supervised layer and
    (c) target radius alpha_r x spread ratio tau. The paper's configuration is outlined."""
    from matplotlib.patches import Rectangle
    by_rank = [_sweep_l1(root, f"sweep_rank{r}") for r in SWEEP_RANKS]
    by_layer = [_sweep_l1(root, f"sweep_sup{s}") for s in SWEEP_LAYERS]
    grid = np.array([[_sweep_l1(root, f"sweep_ar{a}_tau{t}") for t in SWEEP_TAUS] for a in SWEEP_RADII])
    if np.isnan(by_rank).all() and np.isnan(by_layer).all() and np.isnan(grid).all():
        print("  (skip hparam_sweeps: no sweep evaluations)")
        return
    plt = _plt()
    color = style.MODEL_COLORS["geometry_lora"]
    fig = plt.figure(figsize=(15.0, 4.3))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.28], wspace=0.34)
    ax_rank, ax_layer, ax_grid = (fig.add_subplot(gs[0, i]) for i in range(3))

    ax_rank.plot(range(len(SWEEP_RANKS)), by_rank, "-o", color=color, lw=2.4, ms=7)
    ax_rank.set_xticks(range(len(SWEEP_RANKS)))
    ax_rank.set_xticklabels([str(r) for r in SWEEP_RANKS])
    ax_rank.set_ylim(7, 10)
    ax_rank.set_xlabel("LoRA Rank", fontsize=15)

    ax_layer.plot(SWEEP_LAYERS, by_layer, "-o", color=color, lw=2.4, ms=7)
    ax_layer.axvline(paths.COUNT_LAYER, color="0.5", ls="--", lw=1.1)
    ax_layer.set_xlabel("Supervision Layer", fontsize=15)

    for ax, letter in ((ax_rank, "(a)"), (ax_layer, "(b)"), (ax_grid, "(c)")):
        ax.text(-0.14, 1.03, letter, transform=ax.transAxes, ha="left", va="bottom",
                fontsize=17, fontweight="bold", color="black")
    for ax in (ax_rank, ax_layer):
        ax.set_ylabel("Linebreaking Error (L1, chars)", fontsize=14)
        _style_axes(ax)

    image = ax_grid.imshow(grid, origin="lower", aspect="auto", cmap="viridis_r")   # yellow = lower error
    lo, hi = np.nanmin(grid), np.nanmax(grid)
    for i, j in np.ndindex(grid.shape):
        if not np.isnan(grid[i, j]):
            text_color = "white" if (grid[i, j] - lo) / (hi - lo) > 0.55 else "black"
            ax_grid.text(j, i, f"{grid[i, j]:.1f}", ha="center", va="center", fontsize=9.5, color=text_color)
    ax_grid.set_xticks(range(len(SWEEP_TAUS)))
    ax_grid.set_xticklabels([str(t) for t in SWEEP_TAUS])
    ax_grid.set_yticks(range(len(SWEEP_RADII)))
    ax_grid.set_yticklabels([str(a) for a in SWEEP_RADII])
    ax_grid.set_xlabel(r"Spread Ratio $\tau$", fontsize=15)
    ax_grid.set_ylabel(r"Target Radius $\alpha_r$", fontsize=15)
    ax_grid.tick_params(labelsize=12, colors="black")
    ax_grid.grid(False)
    row, col = SWEEP_RADII.index(1.2), SWEEP_TAUS.index(0.5)
    ax_grid.add_patch(Rectangle((col - 0.5, row - 0.5), 1, 1, fill=False, edgecolor="black", lw=2.2))
    colorbar = fig.colorbar(image, ax=ax_grid, fraction=0.046, pad=0.02)
    colorbar.set_label("Linebreaking Error (L1, chars)", fontsize=12)
    colorbar.ax.tick_params(labelsize=11)
    style.save(fig, "figB_hparam_sweeps")


def fig_dataset_examples(root) -> None:
    """Appendix B.1: the first evaluation document wrapped to widths 20, 40 and 80 (text)."""
    from . import corpus, wrapping
    docs_path = paths.corpus_dir() / "eval_docs.txt"
    if not docs_path.exists():
        print("  (skip dataset_examples: no corpus)")
        return
    doc = corpus.load_docs(docs_path)[0]
    lines = []
    for width in (20, 40, 80):
        lines.append(f"--- wrapped to width {width} ---")
        lines.extend(wrapping.wrap(doc[:600], width).split("\n")[:8])
        lines.append("")
    out = paths.figures_dir() / "figB_dataset_examples.txt"
    out.write_text("\n".join(lines))
    print(f"  wrote {out}")


FIGURES = {
    "l1_heatmap": fig_l1_heatmap,
    "error_distribution": fig_error_distribution,
    "error_distribution_log": fig_error_distribution_log,
    "count_probe_by_layer": fig_count_probe_by_layer,
    "count_manifold_pca": fig_count_manifold_pca,
    "helix_fit": fig_helix_fit,
    "helix_fit_to300": fig_helix_fit_to300,
    "ablation": fig_ablation,
    "ablation_bars": fig_ablation_bars,
    "hparam_sweeps": fig_hparam_sweeps,
    "dataset_examples": fig_dataset_examples,
}


# --------------------------------------------------------------------------- #
#  Headline numbers
# --------------------------------------------------------------------------- #
def compute_numbers(root) -> dict:
    """The headline numbers, from whatever outputs exist (missing ones are left out)."""
    import pandas as pd

    from . import evaluate, manifold

    numbers: dict[str, float] = {}
    for tag, _ in MODELS:
        frames = _load_errors(root, tag)
        if frames:
            per_seed = [cell_mean_l1(df) for df in frames]
            numbers[f"l1_{tag}"] = float(np.mean(per_seed))
            if len(per_seed) > 1:
                numbers[f"l1_{tag}_sd"] = float(np.std(per_seed, ddof=1))
            pooled = pd.concat(frames, ignore_index=True)
            numbers[f"pct_early_{tag}"] = evaluate.summarize(pooled[pooled.width >= WMIN])["pct_early"]
        for key, count_range in ((f"helix_r2_{tag}", IN_RANGE), (f"helix_r2_to300_{tag}", FULL_RANGE)):
            r2 = _cloud(root, tag, *count_range)[2]
            if r2 is not None:
                numbers[key] = r2

    means_path = root / "manifold" / f"mean_hiddens_layer{paths.COUNT_LAYER}.npy"
    if means_path.exists():   # variance of the per-count means (counts 20-150) in the top 3 PCs
        _, explained, _, _ = manifold.pca_variance(np.load(means_path), 20, 150)
        numbers["pca_top3_var"] = float(explained[:3].sum())

    r2_path = root / "probes" / "probe_r2_by_layer.csv"
    if r2_path.exists():
        df = pd.read_csv(r2_path)
        numbers["probe_r2_peak"] = float(df.r2.max())
        numbers["probe_r2_peak_layer"] = int(df.loc[df.r2.idxmax(), "layer"])

    ablation_path = root / "probes" / "intervene_ablate_lberror.npz"
    if ablation_path.exists():
        d = np.load(ablation_path, allow_pickle=True)
        for key, name in (("W150__base", "ablate_intact_w150"), ("W150__pca3", "ablate_pca3_w150")):
            if key in d:
                numbers[name] = float(d[key])
    return numbers
