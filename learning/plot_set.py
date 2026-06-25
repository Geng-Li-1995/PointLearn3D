"""Shared plot aesthetics: figure sizes, fonts, colors, and save helpers."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

FIG_STANDARD = (8, 6)
FIG_WIDE = (13, 8)
FIG_METRICS = (13, 5)
FIG_OVERVIEW = (13, 10)

FS_LABEL = 20
FS_TITLE = 18
FS_TICK = 16
FS_LEGEND = 18
FS_SUPTITLE = 20

COLORS = [
    "blue", "red", "green", "purple", "orange",
    "black", "cyan", "navy", "yellow", "brown",
]
MARKERS = ["o", "x", "s", "^", "v", "<", ">", "*", "D", "p"]
LINESTYLES = [
    "-", "--", ":", "-.",
    (0, (1, 1)), (0, (5, 2)), (0, (3, 1, 1, 1)),
    (0, (5, 5)), (0, (2, 2)), (0, (1, 2)),
]

RC_PARAMS = {
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": FS_TICK,
    "axes.labelsize": FS_LABEL,
    "axes.titlesize": FS_TITLE,
    "xtick.labelsize": FS_TICK,
    "ytick.labelsize": FS_TICK,
    "legend.fontsize": FS_LEGEND,
}

PLOT_DPI = 150
ZORDER_DATA = 5


def apply_plot_style() -> None:
    """Apply global matplotlib style (serif font, unified sizes)."""
    plt.rcParams.update(RC_PARAMS)


def palette_color_at(index: int) -> str:
    return COLORS[index % len(COLORS)]


def style_axes(ax, *, xlabel: str, ylabel: str, title: str) -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.tick_params(direction="in", top=True, right=True)


def plot_series(ax, x, y, *, color_idx: int = 0, label: str | None = None) -> None:
    ax.plot(
        x,
        y,
        color=palette_color_at(color_idx),
        linestyle=LINESTYLES[color_idx % len(LINESTYLES)],
        marker=MARKERS[color_idx % len(MARKERS)],
        linewidth=1.5,
        markersize=4,
        markeredgecolor="black",
        markerfacecolor="white",
        label=label,
        zorder=ZORDER_DATA,
    )


def save_figure(fig, path: str | Path, dpi: int = PLOT_DPI) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    return path
