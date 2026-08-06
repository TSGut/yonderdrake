"""Measure spatial numerical-control error against fixed references."""

from __future__ import annotations

import argparse
import csv
import os
import platform
import tempfile
from importlib.util import find_spec
from pathlib import Path

import numpy as np

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "yonderdrake-matplotlib"),
)

from yonderdrake.riesz.dense import DenseRieszBackend, RieszMeshData
from yonderdrake.riesz.outer_quadrature import triangle_quadrature
from yonderdrake.spectral.sinc import positive_power_sinc


def _integers(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def _floats(value: str) -> list[float]:
    return [float(item) for item in value.split(",") if item.strip()]


def _spectral_action(rule: object, eigenvalues: np.ndarray) -> np.ndarray:
    negative_mask = rule.log_shifts <= 0.0
    negative_shifts = np.exp(rule.log_shifts[negative_mask])
    positive_inverse_shifts = np.exp(-rule.log_shifts[~negative_mask])
    negative = np.sum(
        rule.weights[negative_mask, None]
        * eigenvalues[None, :]
        / (negative_shifts[:, None] + eigenvalues[None, :]),
        axis=0,
    )
    positive = np.sum(
        rule.weights[~negative_mask, None]
        * eigenvalues[None, :]
        / (1.0 + positive_inverse_shifts[:, None] * eigenvalues[None, :]),
        axis=0,
    )
    return negative + positive


def _spectral_rows(targets: list[float], order: float) -> list[dict[str, object]]:
    eigenvalues = np.logspace(-2.0, 2.0, 161)
    exact = eigenvalues**order
    rows = []
    for target in targets:
        rule = positive_power_sinc(order, target)
        approximation = _spectral_action(rule, eigenvalues)
        rows.append(
            {
                "operator": "spectral",
                "control": "sinc_truncation_target",
                "control_value": target,
                "relative_error": float(np.max(np.abs(approximation / exact - 1.0))),
                "reference": "analytic scalar eigenvalue power",
                "reference_control": "",
                "order": order,
                "nodes": rule.num_nodes,
            }
        )
    return rows


def _unit_square_mesh() -> RieszMeshData:
    coordinates = np.asarray(
        [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
        dtype=np.float64,
    )
    cells = np.asarray([(0, 1, 3), (1, 2, 3)], dtype=np.int64)
    return RieszMeshData.build(coordinates, cells)


def _riesz_rows(
    degrees: list[int],
    reference_degree: int,
    order: float,
) -> list[dict[str, object]]:
    mesh = _unit_square_mesh()
    reference = DenseRieszBackend(
        mesh,
        order,
        triangle_quadrature(reference_degree),
    ).assemble()
    reference_norm = np.linalg.norm(reference)
    return [
        {
            "operator": "riesz",
            "control": "target_quadrature_degree",
            "control_value": degree,
            "relative_error": float(
                np.linalg.norm(
                    DenseRieszBackend(
                        mesh,
                        order,
                        triangle_quadrature(degree),
                    ).assemble()
                    - reference
                )
                / reference_norm
            ),
            "reference": "over-resolved dense Galerkin action",
            "reference_control": reference_degree,
            "order": order,
            "nodes": "",
        }
        for degree in degrees
    ]


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    metadata = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
    }
    enriched = [{**row, **metadata} for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(enriched[0]))
        writer.writeheader()
        writer.writerows(enriched)


def _plot(path: Path, rows: list[dict[str, object]], dpi: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    specifications = (
        (
            axes[0],
            "spectral",
            "Sinc representation",
            "Truncation target",
            True,
        ),
        (
            axes[1],
            "riesz",
            "Riesz target quadrature",
            "Quadrature degree",
            False,
        ),
    )
    for axis, operator, title, xlabel, invert in specifications:
        selected = [row for row in rows if row["operator"] == operator]
        axis.loglog(
            [float(row["control_value"]) for row in selected],
            [float(row["relative_error"]) for row in selected],
            marker="o",
            linewidth=2.0,
        )
        axis.set_title(title)
        axis.set_xlabel(xlabel)
        axis.set_ylabel("Relative error")
        axis.grid(alpha=0.3)
        if invert:
            axis.invert_xaxis()
    figure.suptitle("Spatial numerical-control accuracy", fontweight="bold")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sinc-targets",
        default="1e-2,5e-3,2e-3,1e-3,5e-4,2e-4,1e-4",
    )
    parser.add_argument("--target-quadrature-degrees", default="2,4,6,8,10,12")
    parser.add_argument("--reference-target-quadrature-degree", type=int, default=18)
    parser.add_argument("--order", type=float, default=0.58)
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            Path(__file__).resolve().parent
            / "benchmarks-output"
            / "spatial-operator-accuracy.csv"
        ),
    )
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument(
        "--plot-output",
        type=Path,
        help="write the figure somewhere other than beside the CSV",
    )
    parser.add_argument("--plot-dpi", type=int, default=220)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if not args.no_plots and find_spec("matplotlib") is None:
        parser.error("plot output requires `python -m pip install -e '.[visual]'`")
    targets = _floats(args.sinc_targets)
    degrees = _integers(args.target_quadrature_degrees)
    if args.smoke:
        targets = [1.0e-2]
        degrees = [2]
        args.reference_target_quadrature_degree = 4
    if not targets or min(targets) <= 0.0:
        parser.error("sinc targets must be positive")
    if not degrees or min(degrees) < 1:
        parser.error("quadrature degrees must be positive")
    if args.reference_target_quadrature_degree <= max(degrees):
        parser.error("reference quadrature degree must exceed measured degrees")
    if not 0.0 < args.order < 1.0:
        parser.error("order must lie in (0, 1)")
    if args.plot_dpi < 72:
        parser.error("plot-dpi must be at least 72")

    rows = [
        *_spectral_rows(targets, args.order),
        *_riesz_rows(
            degrees,
            args.reference_target_quadrature_degree,
            args.order,
        ),
    ]
    _write_csv(args.output, rows)
    print(args.output)
    if not args.no_plots:
        plot_path = args.plot_output or args.output.with_suffix(".png")
        _plot(plot_path, rows, args.plot_dpi)
        print(plot_path)


if __name__ == "__main__":
    main()
