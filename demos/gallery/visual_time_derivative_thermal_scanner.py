"""Compare three time derivatives for a moving heat source on a torus."""

from __future__ import annotations

import argparse
import os
import tempfile
from math import cos, pi, sin
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "yonderdrake-matplotlib"),
)
import matplotlib
import numpy as np

matplotlib.use("Agg")

from _visual_data import save_plot_csv  # noqa: E402
from _visual_style import (  # noqa: E402
    CORAL,
    GOLD,
    INK,
    PAPER,
    TEAL,
    configure_matplotlib,
    output_directory,
    signature,
    time_counter,
)
from matplotlib import animation  # noqa: E402
from matplotlib import pyplot as plt  # noqa: E402


def source_position(
    time: float,
    final_time: float,
    major_radius: float,
    minor_radius: float,
) -> tuple[float, float, float]:
    phase = 2.0 * pi * time / final_time
    minor_phase = 0.75 * sin(2.0 * phase + 0.2)
    return (
        (major_radius + minor_radius * cos(minor_phase)) * cos(phase),
        (major_radius + minor_radius * cos(minor_phase)) * sin(phase),
        minor_radius * sin(minor_phase),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parents[1] / "demo-output"),
    )
    parser.add_argument("--major-segments", type=int, default=96)
    parser.add_argument("--minor-segments", type=int, default=48)
    parser.add_argument("--modes", type=int, default=32)
    parser.add_argument("--dt", type=float, default=0.015)
    parser.add_argument("--final-time", type=float, default=1.5)
    args = parser.parse_args()

    import firedrake as fd

    from yonderdrake import (
        CaputoDerivative,
        Diethelm2008,
        FractionalTimeStepper,
        RiemannLiouvilleDerivative,
    )

    configure_matplotlib(plt)
    destination = output_directory(args.output)
    alpha = 0.64
    diffusivity = 0.018
    cooling = 0.16
    beam_width = 0.30
    beam_power = 9.0
    major_radius = 2.0
    minor_radius = 0.68
    mesh = fd.TorusMesh(
        args.major_segments,
        args.minor_segments,
        major_radius,
        minor_radius,
    )
    space = fd.FunctionSpace(mesh, "CG", 1)
    x, y, z = fd.SpatialCoordinate(mesh)
    preload_expression = (
        0.08
        + 0.018 * x / major_radius
        + 0.025 * z / minor_radius
    )
    model_names = ("Classical", "Caputo", "Riemann–Liouville")
    fields = {
        name: fd.Function(space, name=name).interpolate(preload_expression)
        for name in model_names
    }
    source = fd.Function(space, name="moving_laser")
    time = fd.Constant(0.0)
    step_size = fd.Constant(args.dt)

    def spatial_residual(field: Any, test: Any) -> Any:
        return (
            diffusivity * fd.inner(fd.grad(field), fd.grad(test))
            + cooling * fd.inner(field, test)
            - fd.inner(source, test)
        ) * fd.dx

    ordinary_previous = fields["Classical"].copy(deepcopy=True)
    ordinary_test = fd.TestFunction(space)
    ordinary_residual = (
        fd.inner(
            (fields["Classical"] - ordinary_previous) / step_size,
            ordinary_test,
        )
        * fd.dx
        + spatial_residual(fields["Classical"], ordinary_test)
    )
    solver_parameters = {"ksp_type": "preonly", "pc_type": "lu"}
    fractional_residuals = {}
    for name, marker in (
        ("Caputo", CaputoDerivative),
        ("Riemann–Liouville", RiemannLiouvilleDerivative),
    ):
        test = fd.TestFunction(space)
        fractional_residuals[name] = (
            fd.inner(marker(fields[name], alpha), test) * fd.dx
            + spatial_residual(fields[name], test)
        )
    fractional_steppers = {}
    for name in ("Caputo", "Riemann–Liouville"):
        fractional_steppers[name] = FractionalTimeStepper(
            fractional_residuals[name],
            Diethelm2008(args.modes),
            time,
            step_size,
            fields[name],
            solver_parameters=solver_parameters,
        )
    ordinary_solver = fd.NonlinearVariationalSolver(
        fd.NonlinearVariationalProblem(
            ordinary_residual,
            fields["Classical"],
        ),
        solver_parameters=solver_parameters,
    )

    coordinates = np.asarray(mesh.coordinates.dat.data_ro).copy()
    cells = np.asarray(
        mesh.coordinates.function_space().cell_node_map().values,
        dtype=np.int32,
    ).copy()
    sensor_locations = np.asarray(
        [
            source_position(
                0.12 * args.final_time,
                args.final_time,
                major_radius,
                minor_radius,
            ),
            source_position(
                0.47 * args.final_time,
                args.final_time,
                major_radius,
                minor_radius,
            ),
            source_position(
                0.78 * args.final_time,
                args.final_time,
                major_radius,
                minor_radius,
            ),
        ]
    )
    sensor_indices = [
        int(np.argmin(np.sum((coordinates - location) ** 2, axis=1)))
        for location in sensor_locations
    ]

    times = [0.0]
    histories = {
        name: [np.asarray(field.dat.data_ro).copy()]
        for name, field in fields.items()
    }
    sensor_histories = {
        name: [[histories[name][0][index] for index in sensor_indices]]
        for name in model_names
    }
    centers = [
        source_position(
            0.0,
            args.final_time,
            major_radius,
            minor_radius,
        )
    ]
    number_of_steps = round(args.final_time / args.dt)
    for index in range(1, number_of_steps + 1):
        target = index * args.dt
        center = source_position(
            target,
            args.final_time,
            major_radius,
            minor_radius,
        )
        laser = beam_power * fd.exp(
            -(
                (x - center[0]) ** 2
                + (y - center[1]) ** 2
                + (z - center[2]) ** 2
            )
            / beam_width**2
        )
        source.interpolate(laser)
        ordinary_solver.solve()
        ordinary_previous.assign(fields["Classical"])
        fractional_steppers["Caputo"].advance()
        fractional_steppers["Riemann–Liouville"].advance()
        time.assign(time + step_size)
        times.append(float(time))
        centers.append(center)
        for name, field in fields.items():
            values = np.asarray(field.dat.data_ro).copy()
            histories[name].append(values)
            sensor_histories[name].append(
                [values[item] for item in sensor_indices]
            )

    history_arrays = {
        name: np.asarray(values)
        for name, values in histories.items()
    }
    sensor_arrays = {
        name: np.asarray(values)
        for name, values in sensor_histories.items()
    }
    centers_array = np.asarray(centers)
    display_centers = 1.012 * centers_array
    all_values = np.concatenate(
        [values.reshape(-1) for values in history_arrays.values()]
    )
    value_max = float(np.max(all_values))
    sensor_limit = 1.08 * value_max
    data_path = destination / "time-derivative-thermal-scanner-data.csv.gz"
    diagnostic_series = {
        "source:x": (np.asarray(times), centers_array[:, 0]),
        "source:y": (np.asarray(times), centers_array[:, 1]),
        "source:z": (np.asarray(times), centers_array[:, 2]),
    }
    for name in model_names:
        for sensor_index in range(len(sensor_indices)):
            diagnostic_series[f"{name}:sensor_{sensor_index + 1}"] = (
                np.asarray(times),
                sensor_arrays[name][:, sensor_index],
            )
    save_plot_csv(
        data_path,
        times=times,
        coordinates=coordinates,
        cells=cells,
        fields=history_arrays,
        metadata={
            "demo": "time-derivative-thermal-scanner",
            "equation": "D_t T-kappa Delta_Gamma T+lambda T=q(x,t)",
            "operators": model_names,
            "alpha": alpha,
            "diffusivity": diffusivity,
            "cooling": cooling,
            "beam_width": beam_width,
            "beam_power": beam_power,
            "major_radius": major_radius,
            "minor_radius": minor_radius,
            "sensor_indices": sensor_indices,
            "sensor_locations": sensor_locations.tolist(),
            "dt": args.dt,
            "final_time": args.final_time,
            "fps": 10,
            "value_max": value_max,
            "sensor_limit": sensor_limit,
        },
        series=diagnostic_series,
    )

    movie_figure = plt.figure(figsize=(18.0, 22.0), facecolor=PAPER)
    movie_grid = movie_figure.add_gridspec(
        3,
        2,
        left=0.045,
        right=0.955,
        bottom=0.045,
        top=0.775,
        width_ratios=(1.00, 1.35),
        hspace=0.24,
        wspace=0.14,
    )
    surfaces = []
    surface_axes = []
    path_lines = []
    laser_markers = []
    sensor_lines = []
    sensor_colors = (TEAL, CORAL, GOLD)
    row_colors = (INK, TEAL, CORAL)
    for row, (name, row_color) in enumerate(
        zip(model_names, row_colors, strict=True)
    ):
        surface_axis = movie_figure.add_subplot(
            movie_grid[row, 0],
            projection="3d",
        )
        sensor_axis = movie_figure.add_subplot(movie_grid[row, 1])
        surface = surface_axis.plot_trisurf(
            coordinates[:, 0],
            coordinates[:, 1],
            coordinates[:, 2],
            triangles=cells,
            shade=False,
            antialiased=False,
            cmap="viridis",
            vmin=0.0,
            vmax=value_max,
        )
        surface.set_array(
            np.mean(history_arrays[name][0][cells], axis=1)
        )
        path_line, = surface_axis.plot(
            [],
            [],
            [],
            color="white",
            linewidth=1.0,
            alpha=0.7,
        )
        laser_marker = surface_axis.scatter(
            [display_centers[0, 0]],
            [display_centers[0, 1]],
            [display_centers[0, 2]],
            color=GOLD,
            edgecolor="white",
            linewidth=0.7,
            s=32,
            zorder=5,
        )
        for location, color in zip(
            sensor_locations,
            sensor_colors,
            strict=True,
        ):
            surface_axis.scatter(
                [1.012 * location[0]],
                [1.012 * location[1]],
                [1.012 * location[2]],
                marker="x",
                color=color,
                s=26,
                linewidth=1.3,
            )
        surface_axis.set_title(name, loc="left", color=row_color)
        surface_axis.view_init(elev=58, azim=-58)
        surface_axis.set_box_aspect((2.7, 2.7, 1.0), zoom=1.35)
        surface_axis.set_axis_off()
        surfaces.append(surface)
        surface_axes.append(surface_axis)
        path_lines.append(path_line)
        laser_markers.append(laser_marker)

        row_sensor_lines = []
        for sensor_index, color in enumerate(sensor_colors):
            line, = sensor_axis.plot(
                [],
                [],
                color=color,
                linewidth=1.9,
                label=f"sensor {sensor_index + 1}",
            )
            row_sensor_lines.append(line)
        sensor_axis.set_xlim(0.0, args.final_time)
        sensor_axis.set_ylim(0.0, sensor_limit)
        sensor_axis.set_ylabel("temperature")
        if row == 2:
            sensor_axis.set_xlabel("time")
        sensor_lines.append(row_sensor_lines)
    movie_figure.suptitle(
        "Three thermal memories",
        x=0.055,
        y=0.970,
        ha="left",
        color=INK,
        fontsize=24,
        fontweight="bold",
    )
    movie_figure.text(
        0.056,
        0.930,
        (
            rf"$\Delta t={args.dt:g}$  •  "
            rf"$\alpha={alpha:.2f}$  •  "
            rf"$\kappa={diffusivity:.3f}$  •  "
            rf"$\lambda={cooling:.2f}$  •  "
            f"{coordinates.shape[0]:,} vertices"
        ),
        color=INK,
        fontsize=24,
    )
    movie_figure.text(
        0.056,
        0.885,
        (
            r"$\mathcal{D}_t T-\kappa\Delta_{\Gamma}T+\lambda T"
            r"=q(\mathbf{x},t),\quad \mathcal{D}_t\in"
            r"\{\partial_t,D_C^{0.64},D_{RL}^{0.64}\}$"
        ),
        color=INK,
        fontsize=24,
    )
    movie_figure.text(
        0.056,
        0.840,
        (
            r"$T(\mathbf{x},0)=0.08+0.009x+0.025z/0.68$"
            "\n"
            r"$q=9\exp(-\|\mathbf{x}-\mathbf{x}_b(t)\|^2/0.30^2)$"
        ),
        color=INK,
        fontsize=20,
    )
    movie_figure.legend(
        sensor_lines[0],
        ("sensor 1", "sensor 2", "sensor 3"),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.815),
        ncol=3,
        fontsize=20,
        handlelength=2.0,
        columnspacing=1.8,
    )
    signature(movie_figure)
    time_label = time_counter(movie_figure)

    def update(frame: int):
        artists = []
        rotation = 360.0 * frame / max(1, len(times) - 1)
        for row, name in enumerate(model_names):
            surfaces[row].set_array(
                np.mean(history_arrays[name][frame][cells], axis=1)
            )
            surfaces[row].set_clim(0.0, value_max)
            surface_axes[row].view_init(
                elev=58 + rotation,
                azim=-58,
            )
            path_lines[row].set_data_3d(
                display_centers[: frame + 1, 0],
                display_centers[: frame + 1, 1],
                display_centers[: frame + 1, 2],
            )
            laser_markers[row]._offsets3d = (
                [display_centers[frame, 0]],
                [display_centers[frame, 1]],
                [display_centers[frame, 2]],
            )
            for sensor_index, line in enumerate(sensor_lines[row]):
                line.set_data(
                    times[: frame + 1],
                    sensor_arrays[name][: frame + 1, sensor_index],
                )
                artists.append(line)
            artists.extend(
                (
                    surfaces[row],
                    path_lines[row],
                    laser_markers[row],
                )
            )
        time_label.set_text(f"t = {times[frame]:.2f}")
        artists.append(time_label)
        return artists

    movie = animation.FuncAnimation(
        movie_figure,
        update,
        frames=len(times),
        interval=100,
        blit=False,
    )
    gif_path = destination / "time-derivative-thermal-scanner.gif"
    movie.save(
        gif_path,
        writer=animation.PillowWriter(fps=10),
        dpi=120,
    )
    plt.close(movie_figure)

    print(gif_path)
    print(data_path)
    for name in model_names:
        print(f"{name}: peak temperature {history_arrays[name].max():.6e}")
    print("caputo:", fractional_steppers["Caputo"].solver_stats())
    print(
        "riemann_liouville:",
        fractional_steppers["Riemann–Liouville"].solver_stats(),
    )


if __name__ == "__main__":
    main()
