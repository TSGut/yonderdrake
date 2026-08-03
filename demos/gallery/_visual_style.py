"""Shared visual language for Yonderdrake gallery demos."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

INK = "#15253d"
MUTED = "#617084"
PAPER = "#f7f4ed"
WHITE = "#fffdf8"
CORAL = "#e55f4f"
TEAL = "#168c8c"
GOLD = "#e1a72b"
BLUE = "#3567b7"


def configure_matplotlib(plt: Any) -> None:
    """Apply a warm, high-contrast style suitable for screens and papers."""
    plt.rcParams.update(
        {
            "figure.facecolor": PAPER,
            "axes.facecolor": WHITE,
            "axes.edgecolor": "#d7d0c3",
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#d9d5cc",
            "grid.alpha": 0.48,
            "grid.linewidth": 0.7,
            "font.family": "DejaVu Sans",
            "font.size": 24,
            "axes.labelsize": 24,
            "axes.titlesize": 28,
            "legend.fontsize": 20,
            "xtick.labelsize": 22,
            "ytick.labelsize": 22,
            "legend.frameon": False,
            "legend.labelcolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "savefig.facecolor": PAPER,
            "savefig.bbox": "tight",
        }
    )


def output_directory(path: str) -> Path:
    destination = Path(path)
    destination.mkdir(parents=True, exist_ok=True)
    return destination


def signature(figure: Any) -> None:
    figure.text(
        0.985,
        0.975,
        "YONDERDRAKE",
        ha="right",
        va="top",
        color=MUTED,
        fontsize=20,
        fontweight="bold",
        alpha=0.88,
    )


def time_counter(figure: Any) -> Any:
    """Create the standard animation time readout below the signature."""
    return figure.text(
        0.985,
        0.938,
        "",
        ha="right",
        va="top",
        color=INK,
        fontsize=22,
        fontweight="bold",
    )


def material_parameter_table(
    figure: Any,
    materials: Sequence[str],
    damping: Sequence[str],
    orders: Sequence[str],
    *,
    bbox: tuple[float, float, float, float],
) -> Any:
    """Add a compact material, damping, and fractional-order table."""
    if not (len(materials) == len(damping) == len(orders)):
        raise ValueError("materials, damping, and orders must have equal lengths")
    axis = figure.add_axes(bbox)
    axis.set_axis_off()
    material_width = 0.86 / len(materials)
    table = axis.table(
        cellText=((r"$b$", *damping), (r"$\alpha$", *orders)),
        colLabels=("material", *materials),
        cellLoc="left",
        colLoc="left",
        colWidths=(0.14, *(material_width for _ in materials)),
        bbox=(0.0, 0.0, 1.0, 1.0),
        edges="horizontal",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(18)
    for (row, _column), cell in table.get_celld().items():
        cell.set_facecolor(PAPER)
        cell.set_edgecolor("#d7d0c3")
        cell.set_linewidth(0.8)
        cell.PAD = 0.14
        cell.set_text_props(
            color=INK,
            fontweight="bold" if row == 0 or _column == 0 else "normal",
        )
    return table
