"""Shared matplotlib style for the paper figures.

Every figure is written twice: a PDF with real text (Type 42 fonts, which the
conference camera-ready checkers require) and a 300 dpi PNG. A short provenance
note is written next to each figure as <name>.provenance.txt.

Color names the objective in every figure:
    base       gray, dashed       no adapter
    task       orange             cross-entropy on the answer
    geometry   blue               geometry supervision
Annotations (boxes, cutoff rules) are black, never a role color.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap

# TeX Gyre Termes is the free Times clone TeX uses; vendored so figures match a
# Times-set paper without a system font install.
FONT_DIR = Path(__file__).resolve().parent / "fonts"
for font_path in sorted(FONT_DIR.glob("*.otf")):
    try:
        font_manager.fontManager.addfont(str(font_path))
    except Exception:
        pass

SERIF_STACK = ["Times New Roman", "TeX Gyre Termes", "STIXGeneral", "DejaVu Serif"]

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
VIOLET, RED, YELLOW = "#4a3aa7", "#e34948", "#eda100"
MAGENTA = "#c2278f"
INK, INK2, MUTED = "#0c0e12", "#4c525c", "#858b95"
GRID, AXIS = "#e2e5ea", "#c2c7d0"
SURFACE, PANEL = "#ffffff", "#f2f4f7"
BASE_GRAY = "#858b95"

BASE_LINE = dict(color=BASE_GRAY, lw=1.8, ls=(0, (3.5, 2.2)))
RULE_LINE = dict(color=INK, lw=1.0, ls=(0, (5, 4)))
CHANCE_LINE = dict(color=INK, lw=1.0, ls=(0, (1, 2.2)))

SEQUENTIAL_BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
CMAP = LinearSegmentedColormap.from_list("seqblue", SEQUENTIAL_BLUES)
ORDINAL_BLUES = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]
CMAP_ORDINAL = LinearSegmentedColormap.from_list("seqblue_ord", ORDINAL_BLUES)

ROLE = {
    "base": BASE_GRAY,
    "task": ORANGE,
    "geometry": BLUE,
    "control": MUTED,
    "external": MAGENTA,
}

# Several curves of the same role (a ladder of task LoRAs): ordered steps of one hue.
RAMP = {
    "geometry": ORDINAL_BLUES,
    "task": ["#f9a06a", "#ef7434", "#c94f14", "#8c3210"],
    "base": ["#c3c8d0", BASE_GRAY, "#5b616b"],
}

# Shading for the sum bands on any plot with sum on the x axis.
BAND_SHADING = [
    ("train", 2, 24, "#e6eaf0"),
    ("near", 25, 48, "#eef1f5"),
    ("far", 49, 96, "#f6f8fa"),
]


def apply(scale: float = 1.0) -> None:
    """Install the shared rcParams. `scale` enlarges type for slide versions."""
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.facecolor": SURFACE,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "serif",
        "font.serif": SERIF_STACK,
        "mathtext.fontset": "stix",
        "font.size": 9 * scale,
        "axes.titlesize": 10.5 * scale,
        "axes.labelsize": 12.5 * scale,
        "xtick.labelsize": 11.0 * scale,
        "ytick.labelsize": 11.0 * scale,
        "legend.fontsize": 8.5 * scale,
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": INK,
        "axes.linewidth": 1.0,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.labelcolor": INK,
        "ytick.labelcolor": INK,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.7,
        "axes.axisbelow": True,
        "lines.linewidth": 1.8,
        "lines.solid_capstyle": "round",
        "legend.frameon": False,
        "figure.constrained_layout.use": False,
    })


def save(figure, name: str, out_dir: Path, note: str | None = None) -> list[Path]:
    """Write <name>.pdf and <name>.png into out_dir, plus a provenance note."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if note:
        (out_dir / f"{name}.provenance.txt").write_text(note.strip() + "\n")
        index_path = out_dir / "provenance.json"
        index = json.loads(index_path.read_text()) if index_path.is_file() else {}
        index[name] = note.strip()
        index_path.write_text(json.dumps(index, indent=1, sort_keys=True))
    paths = []
    for extension in ("pdf", "png"):
        path = out_dir / f"{name}.{extension}"
        figure.savefig(path)
        paths.append(path)
    plt.close(figure)
    for path in paths:
        print(f"  {path.name:48s} {path.stat().st_size / 1024:7.0f} KB")
    return paths


def shade_bands(ax, alpha: float = 1.0, label: bool = True, y: float = 1.0) -> None:
    for name, low, high, color in BAND_SHADING:
        ax.axvspan(low, high, color=color, alpha=alpha, zorder=0, lw=0)
        if label:
            ax.annotate(name, ((low + high) / 2, y), xycoords=("data", "axes fraction"),
                        ha="center", va="bottom", fontsize=7.5, color=MUTED)


def chance_line(ax, n_classes: int = 12, **kwargs) -> float:
    kwargs = {**CHANCE_LINE, "zorder": 1, **kwargs}
    ax.axhline(1.0 / n_classes, **kwargs)
    return 1.0 / n_classes
