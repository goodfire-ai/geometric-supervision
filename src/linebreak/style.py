"""Figure style for the linebreak figures: serif text, a fixed palette, OOD bands.

Fonts fall back to matplotlib's bundled STIX (Times-metric) if Times New Roman is
not installed.
"""
from __future__ import annotations

#: Entity -> colour (base is neutral ink; count uses viridis).
C = {
    "base": "#0b0b0b", "task": "#eb6834", "geom": "#2a78d6",
    "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
    "grid": "#e1e0d9", "axis": "#c3c2b7", "surface": "#fcfcfb", "mean": "#d1105a",
}
#: Out-of-distribution bands (in-dist 5-150, near-OOD 151-200, far-OOD 201-300).
BANDS = [("in-distribution", 5, 150, "#00000008"),
         ("near-OOD", 151, 200, "#00000012"),
         ("far-OOD", 201, 300, "#0000001f")]
CMAP_COUNT = "viridis"

#: Per-model display names used across figures.
MODEL_LABELS = {"base": "Base Model", "task_lora": "Task LoRA",
                "geometry_lora": "Geometry LoRA (Helix)"}
MODEL_COLORS = {"base": C["base"], "task_lora": C["task"], "geometry_lora": C["geom"]}


def setup() -> None:
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.dpi": 130, "savefig.dpi": 300, "savefig.bbox": "tight",
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif", "serif"],
        "mathtext.fontset": "stix",
        "font.size": 15, "axes.titlesize": 18, "axes.labelsize": 17,
        "xtick.labelsize": 14.5, "ytick.labelsize": 14.5,
        "axes.edgecolor": C["ink2"], "axes.linewidth": 1.0, "axes.labelcolor": C["ink"],
        "axes.titlecolor": C["ink"], "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": C["grid"], "grid.linewidth": 0.7, "grid.alpha": 0.8,
        "xtick.color": C["ink2"], "ytick.color": C["ink2"],
        "xtick.labelcolor": C["ink2"], "ytick.labelcolor": C["ink2"],
        "xtick.major.size": 5, "ytick.major.size": 5,
        "xtick.major.width": 1.0, "ytick.major.width": 1.0,
        "xtick.direction": "out", "ytick.direction": "out",
        "legend.frameon": False, "legend.fontsize": 14,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "lines.linewidth": 2.2, "lines.solid_capstyle": "round",
    })


def bands(ax, label: bool = True) -> None:
    """Shade the in-dist / near-OOD / far-OOD width bands behind a width-axis plot."""
    for name, a, b, col in BANDS:
        ax.axvspan(a, b, color=col, lw=0, zorder=0)
        if label:
            ax.text((a + b) / 2, 1.0, name, ha="center", va="bottom", fontsize=9,
                    color=C["muted"], transform=ax.get_xaxis_transform())


def save(fig, name: str) -> None:
    from . import paths
    out = paths.figures_dir()
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}")
    print(f"  wrote {out / name}.pdf", flush=True)
