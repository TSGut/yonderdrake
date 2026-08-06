"""Benchmark demo-based fractional time stepping over h, dt, and L."""

from __future__ import annotations

import argparse
import csv
import gc
import itertools
import platform
import sys
from collections.abc import Callable
from dataclasses import dataclass
from importlib.util import find_spec
from math import ceil, cos, pi, sin
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from mpi4py import MPI


@dataclass
class Problem:
    """A constructed demo problem and the operation being timed."""

    mesh: Any
    space: Any
    stepper: Any
    advance: Callable[[float], None]
    fractional_terms: int


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


def _solver_parameters(size: int) -> dict[str, Any]:
    if size == 1:
        return {
            "snes_type": "ksponly",
            "ksp_type": "preonly",
            "pc_type": "lu",
        }
    return {
        "snes_type": "ksponly",
        "ksp_type": "cg",
        "ksp_rtol": 1.0e-9,
        "pc_type": "gamg",
    }


def _representation(name: str, nodes: int) -> Any:
    from yonderdrake import BirkSong, Diethelm2008

    if name == "birk-song":
        return BirkSong(nodes)
    if name == "diethelm":
        return Diethelm2008(nodes)
    raise ValueError(f"unknown representation {name!r}")


def _source_position(
    time: float,
    duration: float,
    major_radius: float,
    minor_radius: float,
) -> tuple[float, float, float]:
    phase = 2.0 * pi * time / duration
    minor_phase = 0.75 * sin(2.0 * phase + 0.2)
    return (
        (major_radius + minor_radius * cos(minor_phase)) * cos(phase),
        (major_radius + minor_radius * cos(minor_phase)) * sin(phase),
        minor_radius * sin(minor_phase),
    )


def _thermal_problem(
    fd: Any,
    mesh_level: int,
    step_size: float,
    duration: float,
    representation_name: str,
    nodes: int,
) -> Problem:
    """Build the moving-source torus problem used by the thermal demo."""
    major_segments = 24 * 2**mesh_level
    minor_segments = 12 * 2**mesh_level
    major_radius = 2.0
    minor_radius = 0.68
    mesh = fd.TorusMesh(
        major_segments,
        minor_segments,
        major_radius,
        minor_radius,
    )
    space = fd.FunctionSpace(mesh, "CG", 1)
    x, y, z = fd.SpatialCoordinate(mesh)
    field = fd.Function(space, name="thermal_scanner_temperature").interpolate(
        0.08 + 0.018 * x / major_radius + 0.025 * z / minor_radius
    )
    source = fd.Function(space, name="moving_thermal_source")
    source_center = tuple(fd.Constant(0.0) for _ in range(3))
    source_expression = 9.0 * fd.exp(
        -sum(
            (coordinate - center_coordinate) ** 2
            for coordinate, center_coordinate in zip(
                (x, y, z),
                source_center,
                strict=True,
            )
        )
        / 0.30**2
    )
    test = fd.TestFunction(space)
    time = fd.Constant(0.0)
    dt = fd.Constant(step_size)

    from yonderdrake import CaputoDerivative, FractionalTimeStepper

    residual = (
        fd.inner(CaputoDerivative(field, 0.64), test)
        + 0.018 * fd.inner(fd.grad(field), fd.grad(test))
        + 0.16 * fd.inner(field, test)
        - fd.inner(source, test)
    ) * fd.dx
    stepper = FractionalTimeStepper(
        residual,
        _representation(representation_name, nodes),
        time,
        dt,
        field,
        solver_parameters=_solver_parameters(mesh.comm.size),
    )

    def advance(target_time: float) -> None:
        center = _source_position(
            target_time,
            duration,
            major_radius,
            minor_radius,
        )
        for coordinate, value in zip(source_center, center, strict=True):
            coordinate.assign(value)
        source.interpolate(source_expression)
        stepper.advance()
        time.assign(target_time)

    return Problem(mesh, space, stepper, advance, 1)


def _skullball_problem(
    fd: Any,
    mesh_level: int,
    step_size: float,
    duration: float,
    representation_name: str,
    nodes: int,
    interpolant: str = "quadratic",
) -> Problem:
    """Build the layered pulse problem used by the skullball demo."""
    del duration
    mesh = fd.UnitDiskMesh(refinement_level=mesh_level + 2)
    mesh.coordinates.dat.data[:] *= 1.30
    space = fd.FunctionSpace(mesh, "CG", 1)
    x, y = fd.SpatialCoordinate(mesh)
    test = fd.TestFunction(space)
    time = fd.Constant(0.0)
    dt = fd.Constant(step_size)

    head_level = (x / 0.98) ** 2 + ((y + 0.02) / 1.06) ** 2
    skull_level = (x / 0.88) ** 2 + ((y + 0.01) / 0.96) ** 2
    brain_level = (x / 0.73) ** 2 + ((y - 0.01) / 0.82) ** 2
    brain = fd.conditional(brain_level <= 1.0, 1.0, 0.0)
    skull = fd.conditional(
        brain_level <= 1.0,
        0.0,
        fd.conditional(skull_level <= 1.0, 1.0, 0.0),
    )
    skin = fd.conditional(
        skull_level <= 1.0,
        0.0,
        fd.conditional(head_level <= 1.0, 1.0, 0.0),
    )
    bath = fd.conditional(head_level <= 1.0, 0.0, 1.0)
    materials = (
        ("bath", bath, 0.96, 0.003, 0.10),
        ("skin", skin, 0.94, 0.018, 0.16),
        ("skull", skull, 1.95, 0.110, 0.41),
        ("brain", brain, 1.00, 0.014, 0.58),
    )
    speed_squared = sum(speed**2 * mask for _, mask, speed, _, _ in materials)
    radius_squared = (x + 0.34) ** 2 + (y - 0.08) ** 2
    field = fd.Function(space, name="skullball_pressure").interpolate(
        1.30 * fd.exp(-72.0 * radius_squared) - 0.52 * fd.exp(-18.0 * radius_squared)
    )
    previous = field.copy(deepcopy=True)
    older = field.copy(deepcopy=True)

    from yonderdrake import (
        CaputoDerivative,
        FractionalTimeStepper,
        Recurrence,
    )

    acceleration = (field - 2.0 * previous + older) / dt**2
    velocity = (field - previous) / dt
    fractional_damping = sum(
        fd.inner(
            damping * mask * fd.grad(CaputoDerivative(field, order)),
            fd.grad(test),
        )
        * fd.dx
        for _, mask, _, damping, order in materials
    )
    residual = (
        fd.inner(acceleration, test) * fd.dx
        + fd.inner(speed_squared * fd.grad(field), fd.grad(test)) * fd.dx
        + fractional_damping
        + 0.96 * fd.inner(velocity, test) * fd.ds
    )
    stepper = FractionalTimeStepper(
        residual,
        _representation(representation_name, nodes),
        time,
        dt,
        field,
        formulation=Recurrence(interpolant=interpolant),
        solver_parameters=_solver_parameters(mesh.comm.size),
    )

    def advance(target_time: float) -> None:
        stepper.advance()
        older.assign(previous)
        previous.assign(field)
        time.assign(target_time)

    return Problem(mesh, space, stepper, advance, len(materials))


PROBLEMS = {
    "thermal": _thermal_problem,
    "skullball": _skullball_problem,
}


def _cases(
    levels: list[int],
    timesteps: list[float],
    nodes: list[int],
    sweep: str,
) -> list[tuple[int, float, int, str]]:
    if sweep == "cartesian":
        return [
            (*case, "cartesian") for case in itertools.product(levels, timesteps, nodes)
        ]
    baseline = (
        levels[len(levels) // 2],
        timesteps[len(timesteps) // 2],
        nodes[len(nodes) // 2],
    )
    cases: list[tuple[int, float, int, str]] = []
    for level in levels:
        cases.append((level, baseline[1], baseline[2], "h"))
    for timestep in timesteps:
        cases.append((baseline[0], timestep, baseline[2], "dt"))
    for count in nodes:
        cases.append((baseline[0], baseline[1], count, "L"))
    unique: dict[tuple[int, float, int], str] = {}
    for level, timestep, count, parameter in cases:
        key = (level, timestep, count)
        unique[key] = "baseline" if key == baseline else parameter
    return [(*key, parameter) for key, parameter in unique.items()]


def _run_case(
    fd: Any,
    application: str,
    representation: str,
    mesh_level: int,
    step_size: float,
    nodes: int,
    duration: float,
    repeats: int,
    minimum_sample_seconds: float,
    varied_parameter: str,
) -> dict[str, Any]:
    communicator = fd.COMM_WORLD
    communicator.barrier()
    start = perf_counter()
    problem = PROBLEMS[application](
        fd,
        mesh_level,
        step_size,
        duration,
        representation,
        nodes,
    )
    problem.advance(step_size)
    setup_seconds = communicator.allreduce(
        perf_counter() - start,
        op=MPI.MAX,
    )
    initial_stats = problem.stepper.solver_stats()
    steps = max(1, round(duration / step_size))
    samples = []
    timed_step_counts = []
    next_step = 2
    for _ in range(repeats):
        communicator.barrier()
        start = perf_counter()
        completed_steps = 0
        target_steps = steps
        while True:
            for _ in range(target_steps - completed_steps):
                problem.advance(next_step * step_size)
                next_step += 1
                completed_steps += 1
            elapsed = communicator.allreduce(
                perf_counter() - start,
                op=MPI.MAX,
            )
            if elapsed >= minimum_sample_seconds:
                break
            target_steps = max(
                target_steps + 1,
                ceil(
                    target_steps
                    * minimum_sample_seconds
                    / max(elapsed, np.finfo(np.float64).tiny)
                ),
            )
        samples.append(elapsed / completed_steps)
        timed_step_counts.append(completed_steps)
    stats = problem.stepper.solver_stats()
    local_cells = problem.mesh.cell_set.size
    cells = communicator.allreduce(local_cells, op=MPI.SUM)
    area = float(fd.assemble(fd.Constant(1.0) * fd.dx(domain=problem.mesh)))
    seconds_per_step = float(np.median(samples))
    total_timed_steps = sum(timed_step_counts)
    return {
        "application": application,
        "representation": representation,
        "varied_parameter": varied_parameter,
        "mesh_level": mesh_level,
        "h_characteristic": (area / cells) ** 0.5,
        "cells": cells,
        "dofs": problem.space.dim(),
        "dt": step_size,
        "L": nodes,
        "fractional_terms": problem.fractional_terms,
        "history_fields": problem.fractional_terms * nodes,
        "history_bytes": problem.fractional_terms * nodes * problem.space.dim() * 8,
        "minimum_steps_per_sample": steps,
        "minimum_simulated_duration": steps * step_size,
        "minimum_sample_seconds": minimum_sample_seconds,
        "median_timed_steps_per_sample": float(np.median(timed_step_counts)),
        "total_timed_steps": total_timed_steps,
        "repeats": repeats,
        "setup_seconds": setup_seconds,
        "seconds_per_step": seconds_per_step,
        "linear_iterations_per_step": (
            (stats["linear_iterations"] - initial_stats["linear_iterations"])
            / total_timed_steps
        ),
        "mpi_ranks": communicator.size,
        **_runtime_metadata(fd),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--applications",
        default="thermal,skullball",
        help="comma-separated subset of thermal,skullball",
    )
    parser.add_argument(
        "--representations",
        default="birk-song,diethelm",
        help="comma-separated subset of birk-song,diethelm",
    )
    parser.add_argument("--mesh-levels", default="0,1,2")
    parser.add_argument("--timesteps", default="0.04,0.02,0.01")
    parser.add_argument("--nodes", default="8,16,32")
    parser.add_argument("--duration", type=float, default=0.08)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--minimum-sample-seconds", type=float, default=0.1)
    parser.add_argument(
        "--cold-start",
        action="store_true",
        help="include first-use compilation instead of warming each application",
    )
    parser.add_argument(
        "--sweep",
        choices=("one-at-a-time", "cartesian"),
        default="one-at-a-time",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            Path(__file__).resolve().parent
            / "benchmarks-output"
            / "time-fractional-scaling.csv"
        ),
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="run one small thermal/Diethelm case",
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

    applications = _names(args.applications)
    representations = _names(args.representations)
    levels = _integers(args.mesh_levels)
    timesteps = _floats(args.timesteps)
    nodes = _integers(args.nodes)
    if args.smoke:
        applications = ["thermal"]
        representations = ["diethelm"]
        levels = [0]
        timesteps = [0.04]
        nodes = [4]
        args.duration = 0.04
        args.repeats = 1
        args.minimum_sample_seconds = 0.0
    unknown = set(applications) - PROBLEMS.keys()
    if not applications:
        parser.error("at least one application is required")
    if unknown:
        parser.error(f"unknown application(s): {', '.join(sorted(unknown))}")
    if not representations:
        parser.error("at least one representation is required")
    if set(representations) - {"birk-song", "diethelm"}:
        parser.error("representations must be birk-song and/or diethelm")
    if not levels or min(levels) < 0:
        parser.error("mesh levels must be non-negative")
    if not timesteps or min(timesteps) <= 0.0:
        parser.error("timesteps must be positive")
    if not nodes or min(nodes) < 1 or max(nodes) > 256:
        parser.error("nodes must lie between 1 and 256")
    if args.duration <= 0.0 or args.repeats < 1:
        parser.error("duration and repeats must be positive")
    if args.minimum_sample_seconds < 0.0:
        parser.error("minimum sample seconds must be non-negative")

    cases = _cases(levels, timesteps, nodes, args.sweep)
    rows = []
    for application, representation in itertools.product(
        applications,
        representations,
    ):
        if not args.cold_start:
            warm_problem = PROBLEMS[application](
                fd,
                levels[len(levels) // 2],
                timesteps[len(timesteps) // 2],
                args.duration,
                representation,
                nodes[len(nodes) // 2],
            )
            warm_problem.advance(timesteps[len(timesteps) // 2])
            del warm_problem
            gc.collect()
            fd.COMM_WORLD.barrier()
        for level, timestep, count, varied_parameter in cases:
            row = _run_case(
                fd,
                application,
                representation,
                level,
                timestep,
                count,
                args.duration,
                args.repeats,
                args.minimum_sample_seconds,
                varied_parameter,
            )
            rows.append(row)
            if fd.COMM_WORLD.rank == 0:
                print(
                    f"{application:9s} {representation:9s} "
                    f"h-level={level} dt={timestep:g} L={count:3d}: "
                    f"{row['seconds_per_step']:.4g} s/step",
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
            from benchmarks.plot_results import plot_time_fractional

            for path in plot_time_fractional(
                args.output,
                output_directory=args.plot_output_directory,
                dpi=args.plot_dpi,
            ):
                print(f"wrote {path}")
    fd.COMM_WORLD.barrier()


if __name__ == "__main__":
    main()
