"""Turn benchmark CSV files into publication-quality scaling plots."""

from __future__ import annotations

import argparse
import csv
import os
import tempfile
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "yonderdrake-matplotlib"),
)

import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt  # noqa: E402

INK = "#15253d"
MUTED = "#617084"
PAPER = "#f7f4ed"
WHITE = "#fffdf8"
TEAL = "#168c8c"
GOLD = "#d59618"
CORAL = "#d95f4f"
BLUE = "#3567b7"

REPRESENTATION_STYLE = {
    "birk-song": ("Birk–Song", TEAL, "o"),
    "diethelm": ("Diethelm", GOLD, "s"),
}
OPERATOR_STYLE = {
    "spectral": ("Spectral", TEAL, "o"),
    "riesz": ("Riesz/restricted", CORAL, "s"),
}


def _configure_matplotlib() -> None:
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
            "grid.alpha": 0.55,
            "grid.linewidth": 0.7,
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "legend.frameon": False,
            "legend.labelcolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "savefig.facecolor": PAPER,
            "savefig.bbox": "tight",
        }
    )


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"{path} contains no benchmark rows")
    return rows


def _ordered_present(
    rows: Sequence[dict[str, str]],
    field: str,
    preferred: Iterable[str],
) -> list[str]:
    present = {row[field] for row in rows}
    ordered = [value for value in preferred if value in present]
    return ordered + sorted(present - set(ordered))


def _pairs(
    rows: Sequence[dict[str, str]],
    x_field: str,
    y_field: str,
) -> tuple[list[float], list[float]]:
    values = sorted(
        (float(row[x_field]), float(row[y_field]))
        for row in rows
        if row[x_field] and row[y_field]
    )
    return [value[0] for value in values], [value[1] for value in values]


def _signature(figure: Any) -> None:
    figure.text(
        0.985,
        0.975,
        "YONDERDRAKE",
        ha="right",
        va="top",
        color=MUTED,
        fontsize=8,
        fontweight="bold",
        alpha=0.9,
    )


def _save(figure: Any, path: Path, dpi: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        path,
        dpi=dpi,
        metadata={
            "Creator": "Yonderdrake",
            "Description": "Replottable benchmark result",
        },
    )
    plt.close(figure)
    return path


def _time_figure(
    rows: Sequence[dict[str, str]],
    metric: str,
    title: str,
    ylabel: str,
    sampling_note: str,
) -> Any:
    applications = _ordered_present(
        rows,
        "application",
        ("thermal", "skullball"),
    )
    representations = _ordered_present(
        rows,
        "representation",
        ("birk-song", "diethelm"),
    )
    controls = (
        ("h", "h_characteristic", r"Characteristic mesh size $h$", True),
        ("dt", "dt", r"Timestep $\Delta t$", False),
        ("L", "L", r"Diffusive nodes $L$", False),
    )
    figure, axes = plt.subplots(
        len(applications),
        len(controls),
        figsize=(14.2, 3.55 * len(applications)),
        squeeze=False,
    )
    for row_index, application in enumerate(applications):
        for column_index, (
            parameter,
            x_field,
            xlabel,
            invert_x,
        ) in enumerate(controls):
            axis = axes[row_index, column_index]
            for representation in representations:
                selected = [
                    row
                    for row in rows
                    if row["application"] == application
                    and row["representation"] == representation
                    and row["varied_parameter"] in {parameter, "baseline"}
                ]
                x_values, y_values = _pairs(selected, x_field, metric)
                if not x_values:
                    continue
                label, color, marker = REPRESENTATION_STYLE.get(
                    representation,
                    (representation, BLUE, "o"),
                )
                axis.plot(
                    x_values,
                    y_values,
                    color=color,
                    marker=marker,
                    linewidth=2.0,
                    markersize=6,
                    label=label,
                )
            application_name = application.replace("-", " ").title()
            control_name = {
                "h": "mesh refinement",
                "dt": "timestep refinement",
                "L": "diffusive quadrature",
            }[parameter]
            axis.set_title(f"{application_name} · {control_name}")
            axis.set_xlabel(xlabel)
            axis.set_ylabel(ylabel if column_index == 0 else "")
            axis.set_xscale("log", base=2)
            axis.set_yscale("log")
            if invert_x:
                axis.invert_xaxis()
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        figure.legend(
            handles,
            labels,
            loc="upper center",
            ncol=len(handles),
            bbox_to_anchor=(0.5, 0.885),
        )
    ranks = rows[0].get("mpi_ranks", "unknown")
    figure.suptitle(title, color=INK, fontsize=18, fontweight="bold", y=0.99)
    figure.text(
        0.5,
        0.935,
        f"{sampling_note} · MPI ranks: {ranks}",
        ha="center",
        color=MUTED,
        fontsize=10,
    )
    _signature(figure)
    figure.subplots_adjust(
        left=0.075,
        right=0.985,
        bottom=0.085,
        top=0.79,
        hspace=0.52,
        wspace=0.26,
    )
    return figure


def plot_time_fractional(
    csv_path: Path,
    *,
    output_directory: Path | None = None,
    dpi: int = 220,
) -> list[Path]:
    """Plot steady-step and setup scaling from a time benchmark CSV."""
    _configure_matplotlib()
    rows = _read_rows(csv_path)
    required = {
        "application",
        "representation",
        "varied_parameter",
        "h_characteristic",
        "dt",
        "L",
        "setup_seconds",
        "seconds_per_step",
    }
    missing = required - rows[0].keys()
    if missing:
        raise ValueError(f"{csv_path} is missing columns: {', '.join(sorted(missing))}")
    destination = output_directory or csv_path.parent
    stem = csv_path.stem
    steady = _time_figure(
        rows,
        "seconds_per_step",
        "Fractional time-stepping · steady cost",
        "Seconds per timestep",
        f"Median of {rows[0].get('repeats', 'unknown')} timed intervals",
    )
    setup = _time_figure(
        rows,
        "setup_seconds",
        "Fractional time-stepping · setup cost",
        "Setup and first-step seconds",
        "One construction and first step per case",
    )
    return [
        _save(steady, destination / f"{stem}-steady.png", dpi),
        _save(setup, destination / f"{stem}-setup.png", dpi),
    ]


def plot_spatial_operators(
    csv_path: Path,
    *,
    output_directory: Path | None = None,
    dpi: int = 220,
) -> list[Path]:
    """Plot mesh and numerical-control scaling from a spatial CSV."""
    _configure_matplotlib()
    rows = _read_rows(csv_path)
    required = {
        "operator",
        "varied_parameter",
        "h_characteristic",
        "sinc_truncation_target",
        "target_quadrature_degree",
        "setup_seconds",
        "application_seconds",
    }
    missing = required - rows[0].keys()
    if missing:
        raise ValueError(f"{csv_path} is missing columns: {', '.join(sorted(missing))}")
    operators = _ordered_present(rows, "operator", ("spectral", "riesz"))
    figure, axes = plt.subplots(1, 3, figsize=(15.1, 4.7))

    mesh_axis = axes[0]
    for operator in operators:
        selected = [
            row
            for row in rows
            if row["operator"] == operator
            and row["varied_parameter"] in {"h", "baseline"}
        ]
        x_values, y_values = _pairs(
            selected,
            "h_characteristic",
            "application_seconds",
        )
        label, color, marker = OPERATOR_STYLE.get(
            operator,
            (operator, BLUE, "o"),
        )
        mesh_axis.plot(
            x_values,
            y_values,
            color=color,
            marker=marker,
            linewidth=2.1,
            markersize=6,
            label=label,
        )
    mesh_axis.set_title("Mesh refinement")
    mesh_axis.set_xlabel(r"Characteristic mesh size $h$")
    mesh_axis.set_ylabel("Seconds per application")
    mesh_axis.set_xscale("log", base=2)
    mesh_axis.set_yscale("log")
    mesh_axis.invert_xaxis()
    mesh_axis.legend(loc="best")

    control_specs = (
        (
            axes[1],
            "spectral",
            "sinc_truncation_target",
            "sinc_truncation_target",
            "Spectral sinc target",
            "Truncation target",
            True,
        ),
        (
            axes[2],
            "riesz",
            "target_quadrature_degree",
            "target_quadrature_degree",
            "Riesz target quadrature",
            "Quadrature degree",
            False,
        ),
    )
    metric_styles = (
        ("application_seconds", "Repeated application", TEAL, "o", "-"),
        ("setup_seconds", "Setup + first application", GOLD, "s", "--"),
    )
    for (
        axis,
        operator,
        varied_parameter,
        x_field,
        panel_title,
        xlabel,
        invert_x,
    ) in control_specs:
        selected = [
            row
            for row in rows
            if row["operator"] == operator
            and row["varied_parameter"] in {varied_parameter, "baseline"}
        ]
        for metric, label, color, marker, linestyle in metric_styles:
            x_values, y_values = _pairs(selected, x_field, metric)
            axis.plot(
                x_values,
                y_values,
                color=color,
                marker=marker,
                linestyle=linestyle,
                linewidth=2.0,
                markersize=6,
                label=label,
            )
        axis.set_title(panel_title)
        axis.set_xlabel(xlabel)
        axis.set_ylabel("Seconds")
        axis.set_yscale("log")
        if operator == "spectral":
            axis.set_xscale("log")
        if invert_x:
            axis.invert_xaxis()
        axis.legend(loc="best")

    ranks = rows[0].get("mpi_ranks", "unknown")
    repeats = rows[0].get("repeats", "unknown")
    figure.suptitle(
        "Spatial fractional-operator scaling",
        color=INK,
        fontsize=18,
        fontweight="bold",
        y=0.985,
    )
    figure.text(
        0.5,
        0.905,
        "Habitat geometry · one setup per case; "
        f"median of {repeats} repeated applications · MPI ranks: {ranks}",
        ha="center",
        color=MUTED,
        fontsize=10,
    )
    _signature(figure)
    figure.subplots_adjust(
        left=0.065,
        right=0.985,
        bottom=0.16,
        top=0.77,
        wspace=0.26,
    )
    destination = output_directory or csv_path.parent
    path = destination / f"{csv_path.stem}.png"
    return [_save(figure, path, dpi)]


def main() -> None:
    output = Path(__file__).resolve().parent / "benchmarks-output"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--time",
        type=Path,
        default=output / "time-fractional-scaling.csv",
    )
    parser.add_argument(
        "--spatial",
        type=Path,
        default=output / "spatial-operator-scaling.csv",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dpi", type=int, default=220)
    args = parser.parse_args()
    if args.dpi < 72:
        parser.error("dpi must be at least 72")
    for csv_path in (args.time, args.spatial):
        if not csv_path.is_file():
            parser.error(f"benchmark CSV does not exist: {csv_path}")
    generated = [
        *plot_time_fractional(
            args.time,
            output_directory=args.output_dir,
            dpi=args.dpi,
        ),
        *plot_spatial_operators(
            args.spatial,
            output_directory=args.output_dir,
            dpi=args.dpi,
        ),
    ]
    for path in generated:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
