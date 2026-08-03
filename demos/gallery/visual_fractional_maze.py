"""Compare classical, spectral, and Riesz potentials in a maze."""

from __future__ import annotations

import argparse
import os
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "yonderdrake-matplotlib"),
)
import matplotlib
import numpy as np

matplotlib.use("Agg")

from _maze_geometry import MazeGeometry, square_cell_maze  # noqa: E402
from _visual_data import load_plot_csv, save_plot_csv  # noqa: E402
from _visual_mpi import gather_p1_animation_data  # noqa: E402
from _visual_style import (  # noqa: E402
    BLUE,
    INK,
    MUTED,
    PAPER,
    WHITE,
    configure_matplotlib,
    output_directory,
    signature,
)
from matplotlib import animation, tri  # noqa: E402
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from PIL import Image  # noqa: E402

MODEL_NAMES = ("Classical", "Spectral", "Riesz")


@dataclass(frozen=True)
class MazeData:
    coordinates: np.ndarray
    cells: np.ndarray
    times: np.ndarray
    fields: dict[str, np.ndarray]
    steady_fields: dict[str, np.ndarray]
    diagnostics: dict[str, Any]


def _collective_relative_steady_error(
    values: dict[str, np.ndarray],
    steady_values: dict[str, np.ndarray],
    communicator: Any,
) -> float:
    """Return the largest global error relative to each steady peak."""
    local_extrema = np.asarray(
        [
            (
                np.max(np.abs(values[name] - steady_values[name])),
                np.max(np.abs(steady_values[name])),
            )
            for name in MODEL_NAMES
        ],
        dtype=float,
    )
    global_extrema = np.max(
        np.asarray(communicator.allgather(local_extrema)),
        axis=0,
    )
    scales = np.maximum(global_extrema[:, 1], np.finfo(float).tiny)
    return float(np.max(global_extrema[:, 0] / scales))


def _sample_history(
    times: list[float],
    snapshots: dict[str, list[np.ndarray]],
    number_of_frames: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Sample a uniformly stepped history across its full time interval."""
    sample_count = min(number_of_frames, len(times))
    indices = np.unique(
        np.rint(np.linspace(0, len(times) - 1, sample_count)).astype(int)
    )
    sampled_times = np.asarray(times, dtype=float)[indices]
    sampled_values = {
        name: np.asarray(values)[indices] for name, values in snapshots.items()
    }
    return sampled_times, sampled_values


def _geometric_time_grid(
    initial_step: float,
    final_time: float,
    maximum_steps: int,
) -> np.ndarray:
    """Return increasing timesteps that land exactly on ``final_time``."""
    number_of_steps = min(
        maximum_steps,
        max(1, int(np.ceil(final_time / initial_step - 1.0e-12))),
    )
    if number_of_steps == 1:
        return np.asarray((0.0, final_time))
    if initial_step * number_of_steps >= final_time:
        return np.linspace(0.0, final_time, number_of_steps + 1)

    lower = 1.0
    upper = 2.0
    exponents = np.arange(number_of_steps, dtype=float)
    while initial_step * float(np.sum(upper**exponents)) < final_time:
        upper *= 2.0
    for _ in range(80):
        growth = 0.5 * (lower + upper)
        duration = initial_step * float(np.sum(growth**exponents))
        if duration < final_time:
            lower = growth
        else:
            upper = growth
    growth = 0.5 * (lower + upper)
    increments = initial_step * growth**exponents
    increments[-1] += final_time - float(np.sum(increments))
    return np.concatenate(([0.0], np.cumsum(increments)))


def _solve(
    args: argparse.Namespace,
    destination: Path,
) -> tuple[MazeData, Path] | None:
    import firedrake as fd

    from yonderdrake import (
        RieszFractionalLaplacian,
        SpectralFractionalLaplacian,
    )

    communicator = fd.COMM_WORLD
    mesh = fd.Mesh(str(args.mesh), comm=communicator)
    space = fd.FunctionSpace(mesh, "CG", 1)
    test = fd.TestFunction(space)
    x, y = fd.SpatialCoordinate(mesh)
    geometry = square_cell_maze(args.columns, args.rows)
    start_x, start_y = geometry.start
    goal_x, goal_y = geometry.goal

    def normalized_terminal(
        name: str,
        centre_x: float,
        centre_y: float,
    ) -> Any:
        field = fd.Function(space, name=name).interpolate(
            fd.exp(
                -(
                    (x - float(centre_x)) ** 2
                    + (y - float(centre_y)) ** 2
                )
                / (2.0 * args.source_width**2)
            )
        )
        field /= float(fd.assemble(field * fd.dx))
        return field

    start_source = normalized_terminal(
        "start source",
        float(start_x),
        float(start_y),
    )
    goal_sink = normalized_terminal(
        "goal sink",
        float(goal_x),
        float(goal_y),
    )
    forcing = fd.Function(space, name="balanced start-to-goal forcing")
    forcing.assign(start_source - goal_sink)
    forcing -= float(fd.assemble(forcing * fd.dx)) / float(
        fd.assemble(fd.Constant(1.0) * fd.dx(domain=mesh))
    )
    riesz_boundary = fd.DirichletBC(space, 0.0, "on_boundary")
    neumann_nullspace = fd.VectorSpaceBasis(constant=True, comm=mesh.comm)

    direct_parameters = {
        "snes_type": "ksponly",
        "ksp_type": "preonly",
        "pc_type": "lu",
    }
    neumann_steady_parameters = {
        "snes_type": "ksponly",
        "ksp_type": "cg",
        "ksp_rtol": 2.0e-10,
        "ksp_atol": 1.0e-12,
        "ksp_max_it": 2000,
        "pc_type": "gamg",
    }
    external_parameters = {
        "snes_type": "ksponly",
        "mat_type": "matfree",
        "ksp_type": "gmres",
        "ksp_rtol": 2.0e-7,
        "ksp_atol": 1.0e-11,
        "ksp_max_it": 400,
        "pc_type": "python",
        "pc_python_type": "firedrake.MassInvPC",
        "Mp_pc_type": "lu",
    }
    unknowns = {
        name: fd.Function(space, name=f"{name.lower()} maze potential")
        for name in MODEL_NAMES
    }
    previous = {
        name: fd.Function(space, name=f"previous {name.lower()} potential")
        for name in MODEL_NAMES
    }
    spectral = unknowns["Spectral"]
    spectral_operator = SpectralFractionalLaplacian(
        spectral,
        args.order,
        sinc_truncation_target=args.sinc_truncation_target,
        shift_cache="all",
        shift_solver_parameters={
            "ksp_type": "preonly",
            "pc_type": "lu",
        },
    )
    riesz = unknowns["Riesz"]
    riesz_operator = RieszFractionalLaplacian(
        riesz,
        args.order,
        bcs=riesz_boundary,
        extension="zero",
        quadrature_degree=args.quadrature_degree,
        quadrature_rule=args.quadrature_rule,
        assembly="hmatrix",
        compression_tolerance=args.compression_tolerance,
        admissibility=args.admissibility,
        leaf_size=args.leaf_size,
    )
    classical = unknowns["Classical"]
    classical_steady_residual = (
        fd.inner(fd.grad(classical), fd.grad(test)) - fd.inner(forcing, test)
    ) * fd.dx
    spectral_steady_residual = (
        fd.inner(spectral_operator, test) - fd.inner(forcing, test)
    ) * fd.dx
    riesz_steady_residual = (
        fd.inner(riesz_operator, test) - fd.inner(forcing, test)
    ) * fd.dx
    fd.solve(
        classical_steady_residual == 0,
        classical,
        nullspace=neumann_nullspace,
        transpose_nullspace=neumann_nullspace,
        solver_parameters=neumann_steady_parameters,
    )
    fd.solve(
        spectral_steady_residual == 0,
        spectral,
        nullspace=neumann_nullspace,
        transpose_nullspace=neumann_nullspace,
        solver_parameters=external_parameters,
    )
    fd.solve(
        riesz_steady_residual == 0,
        riesz,
        bcs=riesz_boundary,
        solver_parameters=external_parameters,
    )
    domain_volume = float(
        fd.assemble(fd.Constant(1.0) * fd.dx(domain=mesh))
    )
    for field in (classical, spectral):
        field -= float(fd.assemble(field * fd.dx)) / domain_volume
    steady_values = {
        name: np.asarray(field.dat.data_ro).copy()
        for name, field in unknowns.items()
    }
    for field in (*unknowns.values(), *previous.values()):
        field.assign(0.0)

    time_grid = _geometric_time_grid(
        args.initial_dt,
        args.final_time,
        args.time_steps,
    )
    step_size = fd.Constant(time_grid[1])
    transient_residuals = {
        "Classical": (
            fd.inner(
                (classical - previous["Classical"]) / step_size,
                test,
            )
            + fd.inner(fd.grad(classical), fd.grad(test))
            - fd.inner(forcing, test)
        )
        * fd.dx,
        "Spectral": (
            fd.inner(
                (spectral - previous["Spectral"]) / step_size,
                test,
            )
            + fd.inner(spectral_operator, test)
            - fd.inner(forcing, test)
        )
        * fd.dx,
        "Riesz": (
            fd.inner(
                (riesz - previous["Riesz"]) / step_size,
                test,
            )
            + fd.inner(riesz_operator, test)
            - fd.inner(forcing, test)
        )
        * fd.dx,
    }
    transient_solvers = {
        name: fd.NonlinearVariationalSolver(
            fd.NonlinearVariationalProblem(
                transient_residuals[name],
                unknowns[name],
                bcs=riesz_boundary if name == "Riesz" else None,
            ),
            solver_parameters=(
                direct_parameters if name == "Classical" else external_parameters
            ),
        )
        for name in MODEL_NAMES
    }
    times = [0.0]
    snapshots = {
        name: [np.asarray(field.dat.data_ro).copy()]
        for name, field in unknowns.items()
    }
    maximum_steps = time_grid.size - 1
    history_stride = max(
        1,
        maximum_steps // max(1, 4 * (args.frames - 1)),
    )
    minimum_steps = min(args.frames - 1, maximum_steps)
    convergence_error = 1.0
    converged = False
    previous_time = 0.0
    for step, scheduled_time in enumerate(time_grid[1:], start=1):
        current_time = float(scheduled_time)
        step_size.assign(current_time - previous_time)
        for solver in transient_solvers.values():
            solver.solve()
        for name in ("Classical", "Spectral"):
            field = unknowns[name]
            field -= float(fd.assemble(field * fd.dx)) / domain_volume
        for name, field in unknowns.items():
            previous[name].assign(field)
        previous_time = current_time
        convergence_error = _collective_relative_steady_error(
            {
                name: np.asarray(field.dat.data_ro)
                for name, field in unknowns.items()
            },
            steady_values,
            communicator,
        )
        converged = (
            step >= minimum_steps
            and convergence_error <= args.convergence_tolerance
        )
        if step % history_stride == 0 or converged or step == maximum_steps:
            times.append(current_time)
            for name, field in unknowns.items():
                snapshots[name].append(np.asarray(field.dat.data_ro).copy())
        if converged:
            break

    if not converged and communicator.rank == 0:
        warnings.warn(
            "transient did not reach the requested steady-field tolerance "
            f"{args.convergence_tolerance:.3g} by t={previous_time:.3g}; "
            f"final relative error is {convergence_error:.3g}",
            stacklevel=2,
        )
    integration_steps = step
    times_array, values = _sample_history(times, snapshots, args.frames)

    local_fields = dict(values)
    local_fields.update(
        {
            f"{name} steady": np.repeat(
                steady_values[name][np.newaxis, :],
                times_array.size,
                axis=0,
            )
            for name in MODEL_NAMES
        }
    )
    gathered = gather_p1_animation_data(
        mesh,
        local_fields,
        communicator,
    )
    if communicator.rank != 0:
        return None
    assert gathered is not None
    coordinates, cells, saved_fields = gathered
    diagnostics = {
        "spectral": spectral_operator.diagnostics(),
        "riesz": riesz_operator.diagnostics(),
        "transient": {
            "converged": converged,
            "convergence_error": convergence_error,
            "convergence_tolerance": args.convergence_tolerance,
            "integration_steps": integration_steps,
            "final_time": previous_time,
        },
    }
    data_path = destination / "fractional-maze-data.csv.gz"
    save_plot_csv(
        data_path,
        times=times_array,
        coordinates=coordinates,
        cells=cells,
        fields=saved_fields,
        metadata={
            "demo": "fractional-maze",
            "models": MODEL_NAMES,
            "order": args.order,
            "columns": args.columns,
            "rows": args.rows,
            "source_width": args.source_width,
            "mesh": args.mesh.name,
            "quadrature_degree": args.quadrature_degree,
            "quadrature_rule": args.quadrature_rule,
            "compression_tolerance": args.compression_tolerance,
            "admissibility": args.admissibility,
            "leaf_size": args.leaf_size,
            "sinc_truncation_target": args.sinc_truncation_target,
            "time_discretization": "geometric backward Euler",
            "initial_dt": float(time_grid[1]),
            "final_dt": float(time_grid[-1] - time_grid[-2]),
            "maximum_time_steps": args.time_steps,
            "maximum_time": args.final_time,
            "convergence_tolerance": args.convergence_tolerance,
            "converged": converged,
            "convergence_error": convergence_error,
            "integration_steps": integration_steps,
            "final_time": previous_time,
            "frames": times_array.size,
        },
    )
    return MazeData(
        coordinates,
        cells,
        times_array,
        {name: saved_fields[name] for name in MODEL_NAMES},
        {
            name: saved_fields[f"{name} steady"][0]
            for name in MODEL_NAMES
        },
        diagnostics,
    ), data_path


def _load(args: argparse.Namespace, path: Path) -> MazeData:
    saved = load_plot_csv(path)
    if saved.metadata.get("demo") != "fractional-maze":
        raise ValueError(f"{path} is not fractional-maze data")
    args.order = float(saved.metadata["order"])
    args.columns = int(saved.metadata["columns"])
    args.rows = int(saved.metadata["rows"])
    args.source_width = float(saved.metadata["source_width"])
    args.quadrature_rule = str(saved.metadata.get("quadrature_rule", "boundary"))
    transient_diagnostics = {
        key: saved.metadata[key]
        for key in (
            "converged",
            "convergence_error",
            "convergence_tolerance",
            "integration_steps",
            "final_time",
        )
        if key in saved.metadata
    }
    if "converged" in transient_diagnostics:
        transient_diagnostics["converged"] = (
            str(transient_diagnostics["converged"]).lower() == "true"
        )
    return MazeData(
        coordinates=saved.coordinates,
        cells=saved.cells,
        times=saved.times,
        fields={name: saved.fields[name] for name in MODEL_NAMES},
        steady_fields={
            name: saved.fields[f"{name} steady"][0] for name in MODEL_NAMES
        },
        diagnostics=(
            {"transient": transient_diagnostics}
            if transient_diagnostics
            else {}
        ),
    )


def _boundary_segments(coordinates: np.ndarray, cells: np.ndarray) -> np.ndarray:
    edges = np.sort(
        np.concatenate(
            (cells[:, (0, 1)], cells[:, (1, 2)], cells[:, (2, 0)]),
            axis=0,
        ),
        axis=1,
    )
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    return coordinates[unique[counts == 1]]


def _cell_gradients(
    coordinates: np.ndarray,
    cells: np.ndarray,
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    points = coordinates[cells]
    differences = values[cells]
    twice_area = (
        (points[:, 1, 0] - points[:, 0, 0])
        * (points[:, 2, 1] - points[:, 0, 1])
        - (points[:, 2, 0] - points[:, 0, 0])
        * (points[:, 1, 1] - points[:, 0, 1])
    )
    gradient_x = (
        differences[:, 0] * (points[:, 1, 1] - points[:, 2, 1])
        + differences[:, 1] * (points[:, 2, 1] - points[:, 0, 1])
        + differences[:, 2] * (points[:, 0, 1] - points[:, 1, 1])
    ) / twice_area
    gradient_y = (
        differences[:, 0] * (points[:, 2, 0] - points[:, 1, 0])
        + differences[:, 1] * (points[:, 0, 0] - points[:, 2, 0])
        + differences[:, 2] * (points[:, 1, 0] - points[:, 0, 0])
    ) / twice_area
    return gradient_x, gradient_y


def _cell_gradient_magnitude(
    coordinates: np.ndarray,
    cells: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    gradient_x, gradient_y = _cell_gradients(coordinates, cells, values)
    return np.hypot(gradient_x, gradient_y)


def _log_scaled(
    values: np.ndarray,
    floor: float,
    ceiling: float,
) -> np.ndarray:
    if floor <= 0.0 or ceiling <= floor:
        return np.zeros_like(values)
    clipped = np.clip(values, floor, ceiling)
    scaled = (np.log(clipped) - np.log(floor)) / (
        np.log(ceiling) - np.log(floor)
    )
    return np.where(values > floor, scaled, 0.0)


def _decorate_maze_axis(
    axis: Any,
    geometry: MazeGeometry,
    boundary_segments: np.ndarray,
) -> None:
    axis.add_collection(
        LineCollection(
            boundary_segments,
            colors=INK,
            linewidths=0.9,
            capstyle="round",
            joinstyle="round",
            zorder=5,
        )
    )
    axis.scatter(
        *geometry.start,
        s=92,
        color=BLUE,
        edgecolor=WHITE,
        linewidth=1.5,
        zorder=8,
    )
    axis.scatter(
        *geometry.goal,
        s=120,
        marker="*",
        color="#000000",
        edgecolor=WHITE,
        linewidth=1.2,
        zorder=8,
    )
    axis.set_aspect("equal")
    axis.set_axis_off()


def _render(
    args: argparse.Namespace,
    data: MazeData,
    destination: Path,
    data_path: Path,
) -> Path:
    geometry = square_cell_maze(args.columns, args.rows)
    coordinates = data.coordinates
    cells = data.cells
    triangulation = tri.Triangulation(coordinates[:, 0], coordinates[:, 1], cells)
    boundary_segments = _boundary_segments(coordinates, cells)
    gradient_magnitudes = {
        name: _cell_gradient_magnitude(coordinates, cells, values)
        for name, values in data.steady_fields.items()
    }
    gradient_limits = {
        name: (
            max(float(np.max(magnitude)) * 1.0e-3, np.finfo(float).tiny),
            float(np.max(magnitude)),
        )
        for name, magnitude in gradient_magnitudes.items()
    }
    clipped_gradients = {
        name: _log_scaled(magnitude, *gradient_limits[name])
        for name, magnitude in gradient_magnitudes.items()
    }
    configure_matplotlib(plt)
    figure = plt.figure(figsize=(17.0, 7.3), facecolor=PAPER)
    grid = figure.add_gridspec(
        1,
        3,
        left=0.035,
        right=0.972,
        bottom=0.045,
        top=0.64,
        wspace=0.035,
    )
    figure.text(
        0.038,
        0.965,
        "A maze seen by three Laplacians",
        ha="left",
        va="top",
        color=INK,
        fontsize=31,
        fontweight="bold",
    )
    figure.text(
        0.039,
        0.875,
        (
            r"$\mathcal{A}u=q_{\rm start}-q_{\rm goal}$"
            rf"  •  fractional order $s={args.order:.2f}$"
            rf"  •  {coordinates.shape[0]:,} vertices"
        ),
        color=INK,
        fontsize=22,
    )
    figure.text(
        0.039,
        0.785,
        "The blue circle is the source and the black star is the goal sink.",
        color=MUTED,
        fontsize=19,
    )
    subtitles = (
        r"$\mathcal{A}=-\Delta_N$",
        rf"$\mathcal{{A}}=(-\Delta_N)^{{{args.order:.2f}}}$",
        rf"$\mathcal{{A}}=(-\Delta)^{{{args.order:.2f}}}_R$",
    )
    for column, (name, subtitle) in enumerate(zip(MODEL_NAMES, subtitles, strict=True)):
        gradient_axis = figure.add_subplot(grid[column])
        gradient_axis.tripcolor(
            triangulation,
            facecolors=clipped_gradients[name],
            shading="flat",
            cmap="magma",
            vmin=0.0,
            vmax=1.0,
        )
        _decorate_maze_axis(gradient_axis, geometry, boundary_segments)
        gradient_axis.set_title(
            f"{name}\n{subtitle}",
            loc="left",
            color=INK,
            fontsize=23,
            fontweight="bold",
            pad=8,
            linespacing=1.18,
        )
    figure.text(
        0.012,
        0.365,
        r"equilibrium $|\nabla u|$",
        rotation=90,
        va="center",
        color=INK,
        fontsize=18,
        fontweight="bold",
    )
    signature(figure)

    output_path = destination / "fractional-maze.png"
    figure.savefig(output_path, dpi=190, facecolor=PAPER)
    plt.close(figure)
    print(output_path)
    print(data_path)
    print(f"mesh: {coordinates.shape[0]} vertices, {cells.shape[0]} triangles")
    for name, values in data.steady_fields.items():
        print(f"{name}: range [{values.min():.6e}, {values.max():.6e}]")
    for name, diagnostics in data.diagnostics.items():
        print(f"{name}: {diagnostics}")
    return output_path


def _render_animation(
    args: argparse.Namespace,
    data: MazeData,
    destination: Path,
) -> Path:
    geometry = square_cell_maze(args.columns, args.rows)
    coordinates = data.coordinates
    cells = data.cells
    triangulation = tri.Triangulation(coordinates[:, 0], coordinates[:, 1], cells)
    boundary_segments = _boundary_segments(coordinates, cells)
    gradients = {
        name: np.asarray(
            [
                _cell_gradient_magnitude(coordinates, cells, frame)
                for frame in values
            ]
        )
        for name, values in data.fields.items()
    }
    steady_gradients = {
        name: _cell_gradient_magnitude(
            coordinates,
            cells,
            data.steady_fields[name],
        )
        for name in MODEL_NAMES
    }
    gradient_limits = {
        name: (
            max(float(np.max(values)) * 1.0e-3, np.finfo(float).tiny),
            float(np.max(values)),
        )
        for name, values in steady_gradients.items()
    }
    displayed_gradients = {
        name: np.asarray(
            [_log_scaled(frame, *gradient_limits[name]) for frame in values]
        )
        for name, values in gradients.items()
    }

    configure_matplotlib(plt)
    figure = plt.figure(figsize=(17.0, 7.3), facecolor=PAPER)
    grid = figure.add_gridspec(
        1,
        3,
        left=0.035,
        right=0.972,
        bottom=0.045,
        top=0.64,
        wspace=0.035,
    )
    figure.text(
        0.038,
        0.965,
        "Diffusion finds the maze",
        ha="left",
        va="top",
        color=INK,
        fontsize=31,
        fontweight="bold",
    )
    figure.text(
        0.039,
        0.875,
        (
            r"$\partial_tu+\mathcal{A}u=q_{\rm start}-q_{\rm goal}$"
            rf"  •  fractional order $s={args.order:.2f}$"
        ),
        color=INK,
        fontsize=22,
    )
    time_label = figure.text(
        0.039,
        0.785,
        "",
        color=MUTED,
        fontsize=19,
    )
    subtitles = (
        r"$\mathcal{A}=-\Delta_N$",
        rf"$\mathcal{{A}}=(-\Delta_N)^{{{args.order:.2f}}}$",
        rf"$\mathcal{{A}}=(-\Delta)^{{{args.order:.2f}}}_R$",
    )
    gradient_images = []
    for column, (name, subtitle) in enumerate(zip(MODEL_NAMES, subtitles, strict=True)):
        gradient_axis = figure.add_subplot(grid[column])
        gradient_image = gradient_axis.tripcolor(
            triangulation,
            facecolors=displayed_gradients[name][0],
            shading="flat",
            cmap="magma",
            vmin=0.0,
            vmax=1.0,
        )
        _decorate_maze_axis(gradient_axis, geometry, boundary_segments)
        gradient_axis.set_title(
            f"{name}\n{subtitle}",
            loc="left",
            color=INK,
            fontsize=23,
            fontweight="bold",
            pad=8,
            linespacing=1.18,
        )
        gradient_images.append(gradient_image)
    figure.text(
        0.012,
        0.365,
        r"fixed-scale $|\nabla u|$",
        rotation=90,
        va="center",
        color=INK,
        fontsize=18,
        fontweight="bold",
    )
    signature(figure)

    def update(frame: int) -> tuple[Any, ...]:
        for name, gradient_image in zip(
            MODEL_NAMES,
            gradient_images,
            strict=True,
        ):
            gradient_image.set_array(displayed_gradients[name][frame])
        if frame == 0:
            label = "initial state"
        elif frame == len(data.times) - 1 and data.diagnostics.get(
            "transient", {}
        ).get("converged", False):
            label = f"path established at $t={data.times[frame]:.3g}$"
        else:
            label = f"time $t={data.times[frame]:.3g}$"
        time_label.set_text(label)
        return (*gradient_images, time_label)

    initial_hold = max(2, round(0.5 * args.fps))
    frame_indices = (
        (0,) * initial_hold
        + tuple(range(1, len(data.times)))
        + (len(data.times) - 1,) * max(1, round(args.fps))
    )
    movie = animation.FuncAnimation(
        figure,
        update,
        frames=frame_indices,
        interval=1000 / args.fps,
        blit=False,
    )
    output_path = destination / "fractional-maze.gif"
    render_path = output_path.with_name(f".{output_path.stem}-render.gif")
    movie.save(
        render_path,
        writer=animation.PillowWriter(fps=args.fps),
        dpi=1350.0 / figure.get_size_inches()[0],
    )
    plt.close(figure)
    rendered = Image.open(render_path)
    target_width = 900
    target_height = round(target_width * rendered.height / rendered.width)
    frames = []
    durations = []
    for frame in range(rendered.n_frames):
        rendered.seek(frame)
        durations.append(
            int(rendered.info.get("duration", round(1000 / args.fps)))
        )
        frames.append(
            rendered.convert("RGB").resize(
                (target_width, target_height),
                Image.Resampling.LANCZOS,
            )
        )
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
    )
    render_path.unlink()
    print(output_path)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parents[1] / "demo-output"),
    )
    parser.add_argument("--mesh", type=Path)
    parser.add_argument("--columns", type=int, default=7)
    parser.add_argument("--rows", type=int, default=5)
    parser.add_argument("--order", type=float, default=0.58)
    parser.add_argument("--source-width", type=float, default=0.24)
    parser.add_argument("--quadrature-degree", type=int, default=2)
    parser.add_argument(
        "--quadrature-rule",
        choices=("boundary", "ordinary"),
        default="ordinary",
    )
    parser.add_argument("--compression-tolerance", type=float, default=2.0e-3)
    parser.add_argument("--admissibility", type=float, default=1.0)
    parser.add_argument("--leaf-size", type=int, default=12)
    parser.add_argument("--sinc-truncation-target", type=float, default=2.0e-3)
    parser.add_argument("--initial-dt", type=float, default=0.05)
    parser.add_argument("--time-steps", type=int, default=160)
    parser.add_argument("--final-time", type=float, default=120.0)
    parser.add_argument("--convergence-tolerance", type=float, default=5.0e-2)
    parser.add_argument("--frames", type=int, default=96)
    parser.add_argument("--fps", type=float, default=12.0)
    parser.add_argument("--plot-data", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    meshes = Path(__file__).resolve().parent / "meshes"
    if args.smoke:
        args.columns = 5
        args.rows = 4
        args.mesh = args.mesh or meshes / "fractional-maze-smoke.msh"
        args.quadrature_degree = 2
        args.leaf_size = 8
        args.sinc_truncation_target = 1.0e-2
        args.frames = 7
        args.initial_dt = 0.03
        args.time_steps = 6
        args.final_time = 0.18
        args.convergence_tolerance = 0.999
    else:
        args.mesh = args.mesh or meshes / "fractional-maze.msh"
    if not 0.0 < args.order < 1.0:
        parser.error("order must lie in (0, 1)")
    if args.source_width <= 0.0:
        parser.error("source width must be positive")
    if args.initial_dt <= 0.0 or args.final_time <= 0.0:
        parser.error("initial dt and final time must be positive")
    if args.time_steps < 1:
        parser.error("time steps must be positive")
    if not 0.0 < args.convergence_tolerance < 1.0:
        parser.error("convergence tolerance must lie in (0, 1)")
    if args.frames < 3:
        parser.error("frames must be at least 3")
    if args.fps <= 0.0:
        parser.error("fps must be positive")
    destination = output_directory(args.output)
    if args.plot_data is None:
        result = _solve(args, destination)
        if result is None:
            return
        data, data_path = result
    else:
        from mpi4py import MPI

        if MPI.COMM_WORLD.rank != 0:
            return
        data_path = args.plot_data
        data = _load(args, data_path)
    _render(args, data, destination, data_path)
    _render_animation(args, data, destination)


if __name__ == "__main__":
    main()
