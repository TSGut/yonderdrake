"""Compare dense, matrix-free, and hierarchical Riesz backends."""

from __future__ import annotations

import argparse
import csv
import os
import platform
import tempfile
from importlib.util import find_spec
from pathlib import Path
from time import perf_counter

import numpy as np

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "yonderdrake-matplotlib"),
)

from yonderdrake.riesz.dense import DenseRieszBackend, RieszMeshData
from yonderdrake.riesz.hmatrix import HierarchicalRieszBackend
from yonderdrake.riesz.matfree import MatrixFreeRieszBackend
from yonderdrake.riesz.outer_quadrature import triangle_quadrature


def _integers(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def _separated_squares(
    subdivisions: int,
) -> tuple[np.ndarray, np.ndarray]:
    coordinates = np.asarray(
        [
            (column / subdivisions, row / subdivisions)
            for row in range(subdivisions + 1)
            for column in range(subdivisions + 1)
        ],
        dtype=np.float64,
    )
    cells = []
    width = subdivisions + 1
    for row in range(subdivisions):
        for column in range(subdivisions):
            lower_left = row * width + column
            lower_right = lower_left + 1
            upper_left = lower_left + width
            upper_right = upper_left + 1
            cells.extend(
                [
                    (lower_left, lower_right, upper_left),
                    (lower_right, upper_right, upper_left),
                ]
            )
    cell_array = np.asarray(cells, dtype=np.int64)
    dimension = coordinates.shape[0]
    return (
        np.vstack((coordinates, coordinates + np.array([4.0, 0.0]))),
        np.vstack((cell_array, cell_array + dimension)),
    )


def _median_apply(backend: object, vector: np.ndarray, repeats: int) -> float:
    samples = []
    for _ in range(repeats):
        started = perf_counter()
        backend.apply(vector)  # type: ignore[attr-defined]
        samples.append(perf_counter() - started)
    return float(np.median(samples))


def _relative_error(actual: np.ndarray, expected: np.ndarray) -> float:
    return float(
        np.linalg.norm(actual - expected)
        / max(np.linalg.norm(expected), np.finfo(np.float64).tiny)
    )


def _plot(rows: list[dict[str, object]], output: Path, dpi: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    degrees = np.asarray([row["dofs"] for row in rows], dtype=np.float64)
    figure, axes = plt.subplots(2, 2, figsize=(12.5, 8.2))
    for key, label, marker in (
        ("dense_apply_seconds", "dense", "o"),
        ("matfree_apply_seconds", "matrix-free", "s"),
        ("hmatrix_apply_seconds", "hmatrix", "^"),
    ):
        axes[0, 0].loglog(
            degrees,
            [row[key] for row in rows],
            marker=marker,
            linewidth=1.8,
            label=label,
        )
    axes[0, 0].set_title("Application")
    axes[0, 0].set_xlabel("Global degrees of freedom")
    axes[0, 0].set_ylabel("Median apply time [s]")
    axes[0, 0].grid(alpha=0.25)
    axes[0, 0].legend(frameon=False)

    for key, label, marker in (
        ("dense_build_seconds", "dense assembly", "o"),
        ("hmatrix_build_seconds", "hmatrix construction", "^"),
    ):
        axes[0, 1].loglog(
            degrees,
            [row[key] for row in rows],
            marker=marker,
            linewidth=1.8,
            label=label,
        )
    axes[0, 1].set_title("Construction")
    axes[0, 1].set_xlabel("Global degrees of freedom")
    axes[0, 1].set_ylabel("Construction time [s]")
    axes[0, 1].grid(alpha=0.25)
    axes[0, 1].legend(frameon=False)

    for key, label, marker in (
        ("matfree_relative_error", "matrix-free", "s"),
        ("hmatrix_relative_error", "hmatrix", "^"),
    ):
        axes[1, 0].semilogy(
            degrees,
            [row[key] for row in rows],
            marker=marker,
            linewidth=1.8,
            label=label,
        )
    axes[1, 0].set_title("Agreement with dense assembly")
    axes[1, 0].set_xlabel("Global degrees of freedom")
    axes[1, 0].set_ylabel("Relative action error")
    axes[1, 0].grid(alpha=0.25)
    axes[1, 0].legend(frameon=False)

    axes[1, 1].plot(
        degrees,
        [row["compression_ratio"] for row in rows],
        marker="o",
        linewidth=1.8,
        label="stored / dense entries",
    )
    axes[1, 1].plot(
        degrees,
        [row["sample_fraction"] for row in rows],
        marker="s",
        linewidth=1.8,
        label="sampled / dense entries",
    )
    axes[1, 1].set_title("Hierarchical work and storage")
    axes[1, 1].set_xlabel("Global degrees of freedom")
    axes[1, 1].set_ylabel("Fraction of dense matrix")
    axes[1, 1].set_ylim(bottom=0.0)
    axes[1, 1].grid(alpha=0.25)
    axes[1, 1].legend(frameon=False)
    figure.suptitle("Riesz backend comparison", fontweight="bold")
    figure.tight_layout()
    figure.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subdivisions", default="2,3,4,5,6,7,8")
    parser.add_argument("--order", type=float, default=0.4)
    parser.add_argument("--quadrature-degree", type=int, default=2)
    parser.add_argument("--compression-tolerance", type=float, default=1.0e-3)
    parser.add_argument("--admissibility", type=float, default=1.0)
    parser.add_argument("--leaf-size", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            Path(__file__).resolve().parent
            / "benchmarks-output"
            / "riesz-backend-comparison.csv"
        ),
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument(
        "--plot-output",
        type=Path,
        help="write the figure somewhere other than beside the CSV",
    )
    parser.add_argument("--plot-dpi", type=int, default=220)
    args = parser.parse_args()
    if not args.no_plots and find_spec("matplotlib") is None:
        parser.error("plot output requires `python -m pip install -e '.[visual]'`")
    subdivisions = _integers(args.subdivisions)
    if args.smoke:
        subdivisions = [2]
        args.repeats = 1
    if not subdivisions or min(subdivisions) < 1:
        parser.error("subdivisions must be positive")
    if not 0.0 < args.order < 1.0:
        parser.error("order must satisfy 0 < order < 1")
    if args.quadrature_degree < 1 or args.repeats < 1:
        parser.error("quadrature degree and repeats must be positive")
    if not 0.0 < args.compression_tolerance < 1.0:
        parser.error("compression tolerance must lie in (0, 1)")
    if args.admissibility <= 0.0 or args.leaf_size < 1:
        parser.error("admissibility and leaf size must be positive")

    rows: list[dict[str, object]] = []
    rule = triangle_quadrature(args.quadrature_degree)
    for subdivision in subdivisions:
        coordinates, cells = _separated_squares(subdivision)
        mesh = RieszMeshData.build(coordinates, cells)
        vector = np.random.default_rng(2468).standard_normal(coordinates.shape[0])

        dense = DenseRieszBackend(mesh, args.order, rule)
        started = perf_counter()
        dense.assemble()
        dense_build_seconds = perf_counter() - started
        reference = dense.apply(vector)
        dense_apply_seconds = _median_apply(dense, vector, args.repeats)

        matrix_free = MatrixFreeRieszBackend(mesh, args.order, rule)
        matrix_free_action = matrix_free.apply(vector)
        matfree_apply_seconds = _median_apply(
            matrix_free,
            vector,
            args.repeats,
        )

        hierarchical = HierarchicalRieszBackend(
            mesh,
            args.order,
            rule,
            compression_tolerance=args.compression_tolerance,
            admissibility=args.admissibility,
            leaf_size=args.leaf_size,
        )
        started = perf_counter()
        hierarchical.build()
        hmatrix_build_seconds = perf_counter() - started
        hierarchical_action = hierarchical.apply(vector)
        hmatrix_apply_seconds = _median_apply(
            hierarchical,
            vector,
            args.repeats,
        )
        diagnostics = hierarchical.diagnostics()
        rows.append(
            {
                "subdivisions": subdivision,
                "dofs": coordinates.shape[0],
                "cells": cells.shape[0],
                "order": args.order,
                "quadrature_degree": args.quadrature_degree,
                "compression_tolerance": args.compression_tolerance,
                "admissibility": args.admissibility,
                "leaf_size": args.leaf_size,
                "dense_build_seconds": dense_build_seconds,
                "dense_apply_seconds": dense_apply_seconds,
                "matfree_apply_seconds": matfree_apply_seconds,
                "hmatrix_build_seconds": hmatrix_build_seconds,
                "hmatrix_apply_seconds": hmatrix_apply_seconds,
                "compression_ratio": diagnostics["compression_ratio"],
                "sample_fraction": (
                    diagnostics["exact_entry_evaluations"]
                    / diagnostics["dense_entries"]
                ),
                "average_far_field_rank": diagnostics["average_far_field_rank"],
                "maximum_far_field_rank": diagnostics["maximum_far_field_rank"],
                "hmatrix_relative_error": _relative_error(
                    hierarchical_action,
                    reference,
                ),
                "matfree_relative_error": _relative_error(
                    matrix_free_action,
                    reference,
                ),
                "python": platform.python_version(),
                "platform": platform.platform(),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(args.output)
    if not args.no_plots:
        plot_path = args.plot_output or args.output.with_suffix(".png")
        _plot(rows, plot_path, args.plot_dpi)
        print(plot_path)


if __name__ == "__main__":
    main()
