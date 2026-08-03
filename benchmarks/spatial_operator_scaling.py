"""Benchmark the habitat demo's spectral and Riesz operators over h and quadrature."""

from __future__ import annotations

import argparse
import csv
import gc
import platform
import sys
from importlib.util import find_spec
from math import ceil
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from mpi4py import MPI


def _integers(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def _floats(value: str) -> list[float]:
    return [float(item) for item in value.split(",") if item.strip()]


def _names(value: str) -> list[str]:
    return [item.strip().lower() for item in value.split(",") if item.strip()]


def _runtime_metadata(fd: Any) -> dict[str, str]:
    from petsc4py import PETSc

    return {
        "python": platform.python_version(),
        "firedrake": getattr(fd, "__version__", "unknown"),
        "petsc": ".".join(str(item) for item in PETSc.Sys.getVersion()),
        "platform": platform.platform(),
    }


def _mesh_and_population(
    fd: Any,
    mesh_resolution: int,
    field_degree: int,
) -> tuple[Any, Any, Any]:
    """Build the irregular-island geometry and field from the habitat demo."""
    mesh = fd.UnitSquareMesh(mesh_resolution, mesh_resolution)
    coordinates = mesh.coordinates.dat.data
    square = 2.0 * coordinates - 1.0
    horizontal = square[:, 0].copy()
    vertical = square[:, 1].copy()
    coordinates[:, 0] = horizontal * np.sqrt(1.0 - 0.5 * vertical**2)
    coordinates[:, 1] = vertical * np.sqrt(1.0 - 0.5 * horizontal**2)
    angles = np.arctan2(coordinates[:, 1], coordinates[:, 0])
    shoreline = (
        1.0
        + 0.20 * np.cos(5.0 * angles + 0.25)
        + 0.08 * np.sin(3.0 * angles - 0.4)
    )
    coordinates[:] *= shoreline[:, None]
    space = fd.FunctionSpace(mesh, "CG", field_degree)
    x, y = fd.SpatialCoordinate(mesh)
    population = fd.Function(space, name="habitat_population").interpolate(
        1.05 * fd.exp(-13.0 * ((x + 0.38) ** 2 + (y - 0.38) ** 2))
        + 0.82 * fd.exp(-17.0 * ((x - 0.42) ** 2 + (y + 0.28) ** 2))
        + 0.62 * fd.exp(-24.0 * ((x + 0.78) ** 2 + (y + 0.18) ** 2))
    )
    boundary = fd.DirichletBC(space, 0.0, "on_boundary")
    boundary.apply(population)
    return mesh, population, boundary


def _shift_solver_parameters(size: int) -> dict[str, Any]:
    if size == 1:
        return {"ksp_type": "preonly", "pc_type": "lu"}
    return {
        "ksp_type": "cg",
        "ksp_rtol": 1.0e-9,
        "pc_type": "gamg",
    }


def _measure(
    fd: Any,
    operator_name: str,
    mesh_resolution: int,
    sinc_truncation_target: float,
    quadrature_degree: int,
    order: float,
    repeats: int,
    minimum_sample_seconds: float,
    varied_parameter: str,
    field_degree: int,
    riesz_assembly: str,
    riesz_quadrature_rule: str,
) -> dict[str, Any]:
    from yonderdrake import (
        RieszFractionalLaplacian,
        SpectralFractionalLaplacian,
    )

    communicator = fd.COMM_WORLD
    communicator.barrier()
    start = perf_counter()
    mesh, population, boundary = _mesh_and_population(
        fd,
        mesh_resolution,
        field_degree,
    )
    if operator_name == "spectral":
        operator = SpectralFractionalLaplacian(
            population,
            order,
            bcs=boundary,
            sinc_truncation_target=sinc_truncation_target,
            shift_cache="all",
            shift_solver_parameters=_shift_solver_parameters(
                communicator.size
            ),
        )
    else:
        operator = RieszFractionalLaplacian(
            population,
            order,
            bcs=boundary if order >= 0.5 else None,
            extension="zero",
            quadrature_degree=quadrature_degree,
            quadrature_rule=riesz_quadrature_rule,
            assembly=riesz_assembly,
        )
    image = fd.assemble(operator)
    with image.dat.vec_ro as vector:
        vector.norm()
    setup_seconds = communicator.allreduce(
        perf_counter() - start,
        op=MPI.MAX,
    )
    samples = []
    application_counts = []
    for _ in range(repeats):
        communicator.barrier()
        start = perf_counter()
        applications = 0
        target_applications = 1
        while True:
            for _ in range(target_applications - applications):
                image = fd.assemble(operator)
                with image.dat.vec_ro as vector:
                    vector.norm()
                applications += 1
            elapsed = communicator.allreduce(
                perf_counter() - start,
                op=MPI.MAX,
            )
            if elapsed >= minimum_sample_seconds:
                break
            target_applications = max(
                target_applications + 1,
                ceil(
                    target_applications
                    * minimum_sample_seconds
                    / max(elapsed, np.finfo(np.float64).tiny)
                ),
            )
        samples.append(elapsed / applications)
        application_counts.append(applications)
    local_cells = mesh.cell_set.size
    cells = communicator.allreduce(local_cells, op=MPI.SUM)
    area = float(fd.assemble(fd.Constant(1.0) * fd.dx(domain=mesh)))
    diagnostics = operator.diagnostics()
    return {
        "operator": operator_name,
        "varied_parameter": varied_parameter,
        "mesh_resolution": mesh_resolution,
        "h_characteristic": (area / cells) ** 0.5,
        "cells": cells,
        "dofs": population.function_space().dim(),
        "order": order,
        "field_degree": field_degree,
        "riesz_assembly": (
            riesz_assembly if operator_name == "riesz" else ""
        ),
        "sinc_truncation_target": (
            sinc_truncation_target if operator_name == "spectral" else ""
        ),
        "sinc_nodes": diagnostics.get("num_nodes", ""),
        "quadrature_degree": (
            quadrature_degree if operator_name == "riesz" else ""
        ),
        "quadrature_rule": diagnostics.get("quadrature_rule", ""),
        "quadrature_points": diagnostics.get(
            "quadrature_points_per_cell",
            "",
        ),
        "setup_seconds": setup_seconds,
        "application_seconds": float(np.median(samples)),
        "minimum_sample_seconds": minimum_sample_seconds,
        "median_applications_per_sample": float(np.median(application_counts)),
        "repeats": repeats,
        "mpi_ranks": communicator.size,
        **_runtime_metadata(fd),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operators", default="spectral,riesz")
    parser.add_argument("--mesh-resolutions", default="4,5,6,7,8,10")
    parser.add_argument(
        "--sinc-truncation-targets",
        default="1e-2,5e-3,2e-3,1e-3,5e-4,1e-4",
    )
    parser.add_argument("--quadrature-degrees", default="2,4,6,8,10,12")
    parser.add_argument("--order", type=float, default=0.58)
    parser.add_argument("--field-degree", type=int, choices=(1, 2), default=1)
    parser.add_argument(
        "--riesz-assembly",
        choices=("matfree", "hmatrix"),
        default="matfree",
    )
    parser.add_argument(
        "--riesz-quadrature-rule",
        choices=("boundary", "ordinary"),
        default="boundary",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--minimum-sample-seconds", type=float, default=0.1)
    parser.add_argument(
        "--cold-start",
        action="store_true",
        help="include first-use compilation instead of warming each operator",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            Path(__file__).resolve().parent
            / "benchmarks-output"
            / "spatial-operator-scaling.csv"
        ),
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="run one small spectral and Riesz case",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="write CSV only",
    )
    parser.add_argument(
        "--plot-output-directory",
        type=Path,
        help="write figures somewhere other than beside the CSV",
    )
    parser.add_argument("--plot-dpi", type=int, default=220)
    args = parser.parse_args()
    if not args.no_plots and find_spec("matplotlib") is None:
        parser.error("plot output requires `python -m pip install -e '.[visual]'`")
    if args.plot_dpi < 72:
        parser.error("plot-dpi must be at least 72")

    command_line = sys.argv
    try:
        sys.argv = [command_line[0]]
        import firedrake as fd
    finally:
        sys.argv = command_line

    operators = _names(args.operators)
    resolutions = _integers(args.mesh_resolutions)
    truncation_targets = _floats(args.sinc_truncation_targets)
    degrees = _integers(args.quadrature_degrees)
    if args.smoke:
        resolutions = [3]
        truncation_targets = [1.0e-2]
        degrees = [2]
        args.repeats = 1
        args.minimum_sample_seconds = 0.0
    if not operators:
        parser.error("at least one operator is required")
    if set(operators) - {"spectral", "riesz"}:
        parser.error("operators must be spectral and/or riesz")
    if not resolutions or min(resolutions) < 2:
        parser.error("mesh resolutions must be at least 2")
    if not truncation_targets or min(truncation_targets) <= 0.0:
        parser.error("sinc truncation targets must be positive")
    if not degrees or min(degrees) < 1:
        parser.error("quadrature degrees must be positive")
    if not 0.0 < args.order < 1.0 or args.repeats < 1:
        parser.error("order must lie in (0, 1) and repeats must be positive")
    if args.minimum_sample_seconds < 0.0:
        parser.error("minimum sample seconds must be non-negative")

    baseline_resolution = resolutions[(len(resolutions) - 1) // 2]
    baseline_target = truncation_targets[len(truncation_targets) // 2]
    baseline_degree = degrees[len(degrees) // 2]
    rows = []
    for operator_name in operators:
        if not args.cold_start:
            _measure(
                fd,
                operator_name,
                baseline_resolution,
                baseline_target,
                baseline_degree,
                args.order,
                1,
                0.0,
                "warmup",
                args.field_degree,
                args.riesz_assembly,
                args.riesz_quadrature_rule,
            )
            gc.collect()
            fd.COMM_WORLD.barrier()
        cases: list[tuple[int, float, int, str]] = [
            (
                resolution,
                baseline_target,
                baseline_degree,
                "baseline" if resolution == baseline_resolution else "h",
            )
            for resolution in resolutions
        ]
        controls = (
            [
                (
                    baseline_resolution,
                    target,
                    baseline_degree,
                    "sinc_truncation_target",
                )
                for target in truncation_targets
            ]
            if operator_name == "spectral"
            else [
                (
                    baseline_resolution,
                    baseline_target,
                    degree,
                    "quadrature_degree",
                )
                for degree in degrees
            ]
        )
        cases.extend(controls)
        unique: dict[tuple[int, float, int], str] = {}
        for resolution, target, degree, parameter in cases:
            key = (resolution, target, degree)
            is_baseline = key == (
                baseline_resolution,
                baseline_target,
                baseline_degree,
            )
            unique[key] = "baseline" if is_baseline else parameter
        for (resolution, target, degree), parameter in unique.items():
            row = _measure(
                fd,
                operator_name,
                resolution,
                target,
                degree,
                args.order,
                args.repeats,
                args.minimum_sample_seconds,
                parameter,
                args.field_degree,
                args.riesz_assembly,
                args.riesz_quadrature_rule,
            )
            rows.append(row)
            if fd.COMM_WORLD.rank == 0:
                print(
                    f"{operator_name:8s} resolution={resolution} "
                    f"target={target:g} q={degree}: "
                    f"{row['application_seconds']:.4g} s/apply",
                    flush=True,
                )
    if fd.COMM_WORLD.rank == 0:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {args.output}")
        if not args.no_plots:
            from benchmarks.plot_results import plot_spatial_operators

            for path in plot_spatial_operators(
                args.output,
                output_directory=args.plot_output_directory,
                dpi=args.plot_dpi,
            ):
                print(f"wrote {path}")
    fd.COMM_WORLD.barrier()


if __name__ == "__main__":
    main()
