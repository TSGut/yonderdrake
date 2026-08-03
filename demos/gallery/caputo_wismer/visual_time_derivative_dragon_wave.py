"""Compare three Wismer-type damping laws on a dragon-shaped domain."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "yonderdrake-matplotlib"),
)
import matplotlib
import numpy as np

matplotlib.use("Agg")

try:
    from ._gallery_path import add_gallery_path  # type: ignore[import-not-found]
except ImportError:
    from _gallery_path import add_gallery_path  # type: ignore[no-redef]

add_gallery_path()

from _visual_data import save_plot_csv  # noqa: E402
from _visual_style import (  # noqa: E402
    BLUE,
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
from matplotlib import animation, colors, ticker, tri  # noqa: E402
from matplotlib import pyplot as plt  # noqa: E402


def dragon_outline() -> np.ndarray:
    """Return a detailed side-profile dragon silhouette."""
    return np.asarray(
        [
            # Tapered tail and a ridge of dorsal spines.
            (-3.92, 0.04),
            (-3.73, 0.17),
            (-3.55, 0.31),
            (-3.33, 0.44),
            (-3.08, 0.55),
            (-2.87, 0.60),
            (-2.72, 0.83),
            (-2.60, 0.62),
            (-2.40, 0.91),
            (-2.26, 0.66),
            (-2.03, 0.96),
            (-1.88, 0.72),
            (-1.66, 0.91),
            (-1.48, 0.96),
            # A bat-like wing with several articulated fingers.
            (-1.91, 2.49),
            (-1.59, 2.30),
            (-1.29, 2.08),
            (-1.00, 1.84),
            (-0.76, 1.66),
            (-0.43, 2.89),
            (-0.30, 2.58),
            (-0.23, 2.25),
            (-0.17, 1.88),
            (-0.11, 1.59),
            (0.70, 2.49),
            (0.61, 2.18),
            (0.54, 1.89),
            (0.44, 1.61),
            (0.31, 1.37),
            (0.47, 1.10),
            (0.72, 1.13),
            # Neck crest, swept horns, brow, and muzzle.
            (0.91, 1.35),
            (1.08, 1.62),
            (1.11, 2.20),
            (1.31, 1.94),
            (1.57, 2.50),
            (1.55, 1.96),
            (1.91, 2.25),
            (1.78, 1.78),
            (2.03, 1.74),
            (2.18, 1.58),
            (2.45, 1.55),
            (2.60, 1.70),
            (2.69, 1.48),
            (2.96, 1.42),
            (3.27, 1.36),
            (3.56, 1.28),
            (3.84, 1.13),
            (3.66, 1.02),
            # Deep mouth notch followed by the lower jaw.
            (3.32, 1.00),
            (3.01, 0.96),
            (2.76, 0.88),
            (3.04, 0.84),
            (3.34, 0.75),
            (3.70, 0.63),
            (3.91, 0.49),
            (3.68, 0.41),
            (3.37, 0.43),
            (3.06, 0.48),
            (2.79, 0.56),
            (2.51, 0.53),
            (2.28, 0.39),
            (2.08, 0.20),
            # Chest, foreleg, four small claws, and wrist spur.
            (1.90, 0.05),
            (1.72, -0.12),
            (1.64, -0.42),
            (1.72, -0.77),
            (1.94, -1.05),
            (2.23, -1.14),
            (2.04, -1.23),
            (2.30, -1.34),
            (2.01, -1.35),
            (1.82, -1.50),
            (1.70, -1.32),
            (1.46, -1.40),
            (1.61, -1.18),
            (1.39, -0.98),
            (1.22, -0.67),
            (0.91, -0.57),
            (0.64, -0.55),
            # Belly, powerful rear leg, ankle spur, and hind claws.
            (0.39, -0.67),
            (0.25, -0.96),
            (0.34, -1.24),
            (0.58, -1.50),
            (0.37, -1.47),
            (0.20, -1.63),
            (0.05, -1.47),
            (-0.18, -1.55),
            (-0.01, -1.31),
            (-0.21, -1.08),
            (-0.37, -0.76),
            (-0.67, -0.66),
            (-0.95, -0.61),
            (-1.11, -0.84),
            (-1.03, -1.10),
            (-0.83, -1.36),
            (-1.06, -1.31),
            (-1.24, -1.45),
            (-1.36, -1.27),
            (-1.60, -1.32),
            (-1.42, -1.11),
            (-1.55, -0.88),
            (-1.67, -0.56),
            # Underside of the body and tail.
            (-1.94, -0.45),
            (-2.22, -0.35),
            (-2.50, -0.24),
            (-2.79, -0.12),
            (-3.07, -0.02),
            (-3.34, 0.03),
            (-3.61, 0.03),
        ],
        dtype=np.float64,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parents[2] / "demo-output"),
    )
    parser.add_argument(
        "--mesh",
        type=Path,
        help="Gmsh mesh; defaults to the committed dragon mesh",
    )
    parser.add_argument("--modes", type=int, default=32)
    parser.add_argument("--dt", type=float, default=0.0125)
    parser.add_argument("--final-time", type=float, default=2.0)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument(
        "--degree",
        type=int,
        choices=(1, 2),
        default=2,
        help="CG degree for the solve; output is sampled at mesh vertices",
    )
    args = parser.parse_args()

    import firedrake as fd

    from yonderdrake import (
        CaputoDerivative,
        Diethelm2008,
        FractionalTimeStepper,
    )

    configure_matplotlib(plt)
    destination = output_directory(args.output)
    caputo_orders = (0.58, 0.82)
    wave_speed = 1.85
    damping = 0.055
    mesh_path = args.mesh or (
        Path(__file__).resolve().parent.parent / "meshes" / "dragon.msh"
    )
    outline_points = dragon_outline()
    outline = np.vstack((outline_points, outline_points[0]))
    mesh = fd.Mesh(str(mesh_path), comm=fd.COMM_SELF)
    space = fd.FunctionSpace(mesh, "CG", args.degree)
    output_view = (
        None
        if args.degree == 1
        else fd.Function(fd.FunctionSpace(mesh, "CG", 1), name="p1_output_view")
    )

    def p1_snapshot(field: Any) -> np.ndarray:
        if output_view is None:
            return np.asarray(field.dat.data_ro).copy()
        return np.asarray(output_view.interpolate(field).dat.data_ro).copy()

    x, y = fd.SpatialCoordinate(mesh)
    throat_radius_squared = (x - 1.82) ** 2 + (y - 0.72) ** 2
    initial_expression = 1.45 * fd.exp(-17.0 * throat_radius_squared) - 0.72 * fd.exp(
        -4.8 * throat_radius_squared
    )
    model_names = (
        "Classical",
        *(rf"Caputo  $D_C^{{{order:.2f}}}$" for order in caputo_orders),
    )
    fields = {
        name: fd.Function(space, name=name).interpolate(initial_expression)
        for name in model_names
    }
    boundary_conditions = {
        name: fd.DirichletBC(space, 0.0, "on_boundary") for name in model_names
    }
    for name, field in fields.items():
        boundary_conditions[name].apply(field)
    previous = {name: field.copy(deepcopy=True) for name, field in fields.items()}
    older = {name: field.copy(deepcopy=True) for name, field in fields.items()}
    test = fd.TestFunction(space)
    time = fd.Constant(0.0)
    step_size = fd.Constant(args.dt)

    def inertia_and_stiffness(name: str) -> Any:
        field = fields[name]
        acceleration = (field - 2.0 * previous[name] + older[name]) / step_size**2
        return (
            fd.inner(acceleration, test) * fd.dx
            + wave_speed**2 * fd.inner(fd.grad(field), fd.grad(test)) * fd.dx
        )

    ordinary_name = model_names[0]
    fractional_orders = dict(zip(model_names[1:], caputo_orders, strict=True))
    fractional_markers = {
        name: CaputoDerivative(fields[name], fractional_orders[name])
        for name in model_names[1:]
    }
    ordinary_residual = (
        inertia_and_stiffness(ordinary_name)
        + damping
        * fd.inner(
            fd.grad((fields[ordinary_name] - previous[ordinary_name]) / step_size),
            fd.grad(test),
        )
        * fd.dx
    )
    solver_parameters = {
        "snes_type": "ksponly",
        "ksp_type": "preonly",
        "pc_type": "lu",
    }
    ordinary_solver = fd.NonlinearVariationalSolver(
        fd.NonlinearVariationalProblem(
            ordinary_residual,
            fields[ordinary_name],
            bcs=boundary_conditions[ordinary_name],
        ),
        solver_parameters=solver_parameters,
    )
    fractional_steppers = {}
    for name in model_names[1:]:
        residual = (
            inertia_and_stiffness(name)
            + damping
            * fd.inner(
                fd.grad(fractional_markers[name]),
                fd.grad(test),
            )
            * fd.dx
        )
        fractional_steppers[name] = FractionalTimeStepper(
            residual,
            Diethelm2008(args.modes),
            time,
            step_size,
            fields[name],
            bcs=boundary_conditions[name],
            solver_parameters=solver_parameters,
        )

    times = [0.0]
    histories = {
        name: [p1_snapshot(field)] for name, field in fields.items()
    }
    number_of_steps = round(args.final_time / args.dt)
    for index in range(1, number_of_steps + 1):
        ordinary_solver.solve()
        for name in model_names[1:]:
            fractional_steppers[name].advance()
        for name in model_names:
            older[name].assign(previous[name])
            previous[name].assign(fields[name])
            histories[name].append(p1_snapshot(fields[name]))
        time.assign(index * args.dt)
        times.append(float(time))

    history_arrays = {name: np.asarray(values) for name, values in histories.items()}
    coordinates = np.asarray(mesh.coordinates.dat.data_ro).copy()
    cells = np.asarray(
        mesh.coordinates.function_space().cell_node_map().values,
        dtype=np.int32,
    ).copy()
    triangulation = tri.Triangulation(
        coordinates[:, 0],
        coordinates[:, 1],
        cells,
    )
    all_values = np.concatenate(
        [values.reshape(-1) for values in history_arrays.values()]
    )
    value_limit = max(float(np.max(np.abs(all_values))), 1.0e-8)
    pressure_norm = colors.SymLogNorm(
        linthresh=0.03 * value_limit,
        vmin=-value_limit,
        vmax=value_limit,
        base=10,
    )
    x_min, y_min = outline.min(axis=0)
    x_max, y_max = outline.max(axis=0)
    x_padding = 0.04 * (x_max - x_min)
    y_padding = 0.05 * (y_max - y_min)
    frequencies = np.geomspace(0.04, 8.0, 180)
    simulation_orders = (1.0, *caputo_orders)
    dispersion_by_order = {}
    for order in simulation_orders:
        wave_numbers = frequencies / np.sqrt(
            wave_speed**2 + damping * (-1j * frequencies) ** order
        )
        attenuation = np.imag(wave_numbers)
        fit_region = frequencies <= 0.8
        fitted_order = float(
            np.polyfit(
                np.log(frequencies[fit_region]),
                np.log(attenuation[fit_region]),
                deg=1,
            )[0]
        )
        dispersion_by_order[order] = (
            attenuation,
            fitted_order,
        )
    attenuation_curve_specs = (
        ("Classical", 1.0, INK, "-", 2.6),
        (
            rf"Caputo  $\alpha={caputo_orders[0]:.2f}$",
            caputo_orders[0],
            CORAL,
            "-",
            3.8,
        ),
        (
            rf"Riemann–Liouville  $\alpha={caputo_orders[0]:.2f}$",
            caputo_orders[0],
            GOLD,
            (0, (4, 2)),
            1.8,
        ),
        (
            rf"Caputo  $\alpha={caputo_orders[1]:.2f}$",
            caputo_orders[1],
            BLUE,
            "-",
            3.8,
        ),
        (
            rf"Riemann–Liouville  $\alpha={caputo_orders[1]:.2f}$",
            caputo_orders[1],
            TEAL,
            (0, (4, 2)),
            1.8,
        ),
    )
    attenuation_curves = tuple(
        (
            label,
            color,
            linestyle,
            linewidth,
            *dispersion_by_order[order],
        )
        for label, order, color, linestyle, linewidth in attenuation_curve_specs
    )
    data_path = destination / "time-derivative-dragon-wave-data.csv.gz"
    diagnostic_series = {}
    for (
        label,
        _color,
        _linestyle,
        _linewidth,
        attenuation,
        _fitted_order,
    ) in attenuation_curves:
        diagnostic_series[f"{label}:attenuation"] = (
            frequencies,
            attenuation,
        )
    save_plot_csv(
        data_path,
        times=times,
        coordinates=coordinates,
        cells=cells,
        fields=history_arrays,
        metadata={
            "demo": "time-derivative-dragon-wave",
            "equation": ("u_tt-c^2 Delta u-b Delta D_t^alpha u=0"),
            "operators": model_names,
            "caputo_orders": caputo_orders,
            "wave_speed": wave_speed,
            "damping": damping,
            "dt": args.dt,
            "final_time": args.final_time,
            "fps": args.fps,
            "element_degree": args.degree,
            "mesh": mesh_path.name,
            "pressure_limit": value_limit,
            "pressure_linthresh": pressure_norm.linthresh,
            "fitted_orders": {
                label: fitted_order
                for (
                    label,
                    _color,
                    _linestyle,
                    _linewidth,
                    _attenuation,
                    fitted_order,
                ) in attenuation_curves
            },
        },
        anatomy=(
            outline,
            np.ones(outline.shape[0], dtype=np.uint8),
        ),
        series=diagnostic_series,
    )

    figure = plt.figure(figsize=(17.0, 14.0), facecolor=PAPER)
    grid = figure.add_gridspec(
        3,
        3,
        left=0.065,
        right=0.885,
        bottom=0.085,
        top=0.765,
        width_ratios=(0.84, 1.00, 1.16),
        hspace=0.23,
        wspace=0.20,
    )
    figure.suptitle(
        "Attenuation on the dragon",
        x=0.040,
        y=0.973,
        ha="left",
        color=INK,
        fontsize=25,
        fontweight="bold",
    )
    figure.text(
        0.041,
        0.923,
        (
            rf"$\Delta t={args.dt:g}$  •  "
            rf"$\alpha\in\{{{caputo_orders[0]:.2f},"
            rf"{caputo_orders[1]:.2f}\}}$  •  "
            f"{coordinates.shape[0]:,} vertices"
        ),
        color=INK,
        fontsize=24,
    )
    figure.text(
        0.041,
        0.875,
        (
            r"$u_{tt}-c^2\Delta u-b\Delta\mathcal{D}_t u=0,"
            r"\qquad \mathcal{D}_t\in"
            rf"\{{\partial_t,D_C^{{{caputo_orders[0]:.2f}}},"
            rf"D_C^{{{caputo_orders[1]:.2f}}}\}},"
            r"\qquad u|_{\partial\Omega}=0$"
        ),
        color=INK,
        fontsize=26,
    )
    figure.text(
        0.041,
        0.833,
        (
            r"$u(\mathbf{x},0)=1.45e^{-17r^2}-0.72e^{-4.8r^2},"
            r"\quad u_t(\mathbf{x},0)=0,"
            r"\quad r^2=(x-1.82)^2+(y-0.72)^2$"
        ),
        color=INK,
        fontsize=22,
    )
    surfaces = []
    surface_axes = []
    heatmaps = []
    attenuation_markers = []
    diagnostic_axis = figure.add_subplot(grid[:, 2])
    diagnostic_axis.text(
        0.5,
        1.015,
        "attenuation laws",
        transform=diagnostic_axis.transAxes,
        ha="center",
        color=INK,
        fontsize=24,
    )
    diagnostic_axis.set_xlabel(r"angular frequency  $\omega$")
    diagnostic_axis.set_ylabel(
        r"attenuation  $\operatorname{Im}k$",
        labelpad=12,
    )
    diagnostic_axis.yaxis.set_label_position("right")
    diagnostic_axis.yaxis.tick_right()
    diagnostic_axis.grid(True, which="both", alpha=0.18)
    frequency_probe = diagnostic_axis.axvline(
        frequencies[0],
        color=INK,
        linewidth=0.9,
        alpha=0.35,
    )
    frequency_label = diagnostic_axis.text(
        0.96,
        0.36,
        "",
        transform=diagnostic_axis.transAxes,
        ha="right",
        color=INK,
        fontsize=22,
        bbox={
            "facecolor": PAPER,
            "edgecolor": "none",
            "alpha": 0.84,
            "pad": 1.5,
        },
    )
    initial_values = history_arrays[model_names[0]][0]
    for row, _name in enumerate(model_names):
        surface_axis = figure.add_subplot(grid[row, 0], projection="3d")
        heatmap_axis = figure.add_subplot(grid[row, 1])
        surface = surface_axis.plot_trisurf(
            coordinates[:, 0],
            coordinates[:, 1],
            initial_values,
            triangles=cells,
            cmap="viridis",
            norm=pressure_norm,
            shade=False,
            antialiased=False,
            linewidth=0.0,
        )
        heatmap = heatmap_axis.tripcolor(
            triangulation,
            initial_values,
            shading="gouraud",
            cmap="viridis",
            norm=pressure_norm,
        )
        heatmap_axis.plot(
            outline[:, 0],
            outline[:, 1],
            color=INK,
            linewidth=0.8,
            alpha=0.80,
        )
        surface_axis.set_xlim(x_min - x_padding, x_max + x_padding)
        surface_axis.set_ylim(y_min - y_padding, y_max + y_padding)
        surface_axis.set_zlim(-value_limit, value_limit)
        surface_axis.set_box_aspect(
            (x_max - x_min, 0.88 * (y_max - y_min), 3.65),
            zoom=1.27,
        )
        surface_axis.view_init(elev=31, azim=-66)
        surface_axis.set_axis_off()
        heatmap_axis.set_xlim(x_min - x_padding, x_max + x_padding)
        heatmap_axis.set_ylim(y_min - y_padding, y_max + y_padding)
        heatmap_axis.set_aspect("equal")
        heatmap_axis.set_axis_off()
        if row == 0:
            surface_axis.text2D(
                0.5,
                1.02,
                "wave surface",
                transform=surface_axis.transAxes,
                ha="center",
                color=INK,
                fontsize=24,
            )
            heatmap_axis.text(
                0.5,
                1.02,
                "pressure heatmap",
                transform=heatmap_axis.transAxes,
                ha="center",
                color=INK,
                fontsize=24,
            )
        surface_axes.append(surface_axis)
        surfaces.append(surface)
        heatmaps.append(heatmap)

    for axis, name in zip(surface_axes, model_names, strict=True):
        position = axis.get_position()
        figure.text(
            0.027,
            0.5 * (position.y0 + position.y1),
            name,
            rotation=90,
            ha="center",
            va="center",
            color=INK,
            fontsize=24,
        )

    for (
        label,
        color,
        linestyle,
        linewidth,
        attenuation,
        fitted_order,
    ) in attenuation_curves:
        legend_label = label.replace("Riemann–Liouville", "RL")
        diagnostic_axis.loglog(
            frequencies,
            attenuation,
            color=color,
            linewidth=linewidth,
            linestyle=linestyle,
            alpha=0.94,
            label=rf"{legend_label}  ($m={fitted_order:.2f}$)",
        )
        (attenuation_marker,) = diagnostic_axis.plot(
            [frequencies[0]],
            [attenuation[0]],
            marker="o",
            markersize=5,
            color=color,
            linestyle="none",
            zorder=6,
        )
        attenuation_markers.append(attenuation_marker)

    diagnostic_axis.legend(
        loc="upper left",
        fontsize=14,
        frameon=False,
        handlelength=2.1,
        handletextpad=0.5,
        labelspacing=0.35,
        borderaxespad=0.35,
    )
    colorbar_axis = figure.add_axes((0.270, 0.040, 0.300, 0.012))
    pressure_colorbar = figure.colorbar(
        heatmaps[0],
        cax=colorbar_axis,
        orientation="horizontal",
        label="acoustic pressure",
    )
    pressure_colorbar.set_ticks((-value_limit, 0.0, value_limit))
    pressure_colorbar.formatter = ticker.FuncFormatter(
        lambda value, _position: f"{value:.2g}"
    )
    pressure_colorbar.update_ticks()
    pressure_colorbar.ax.xaxis.set_label_position("top")
    signature(figure)
    time_label = time_counter(figure)
    times_array = np.asarray(times)

    def update(frame: int):
        artists = []
        frequency_index = round(frame * (len(frequencies) - 1) / max(1, len(times) - 1))
        probe_frequency = frequencies[frequency_index]
        for row, name in enumerate(model_names):
            values = history_arrays[name][frame]
            triangle_values = values[cells]
            vertices = np.concatenate(
                (
                    coordinates[cells],
                    triangle_values[:, :, None],
                ),
                axis=2,
            )
            surfaces[row].set_verts(vertices)
            surfaces[row].set_array(triangle_values.mean(axis=1))
            surfaces[row].set_clim(pressure_norm.vmin, pressure_norm.vmax)
            heatmaps[row].set_array(values)
            heatmaps[row].set_clim(pressure_norm.vmin, pressure_norm.vmax)
            artists.extend((surfaces[row], heatmaps[row]))
        for marker, curve in zip(
            attenuation_markers,
            attenuation_curves,
            strict=True,
        ):
            attenuation = curve[4]
            marker.set_data(
                [probe_frequency],
                [attenuation[frequency_index]],
            )
            artists.append(marker)
        frequency_probe.set_xdata([probe_frequency, probe_frequency])
        frequency_label.set_text(rf"$\omega={probe_frequency:.2f}$")
        time_label.set_text(f"t = {times_array[frame]:.3f}")
        artists.extend((frequency_probe, frequency_label, time_label))
        return artists

    movie = animation.FuncAnimation(
        figure,
        update,
        frames=len(times),
        interval=1000 / args.fps,
        blit=False,
    )
    gif_path = destination / "time-derivative-dragon-wave.gif"
    movie.save(
        gif_path,
        writer=animation.PillowWriter(fps=args.fps),
        dpi=120,
    )
    plt.close(figure)

    print(gif_path)
    print(data_path)
    print(f"mesh: {coordinates.shape[0]} vertices, {cells.shape[0]} triangles")
    for name in model_names:
        print(f"{name}: final peak {np.max(np.abs(history_arrays[name][-1])):.6e}")
    for name, stepper in fractional_steppers.items():
        print(name, stepper.solver_stats())


if __name__ == "__main__":
    main()
