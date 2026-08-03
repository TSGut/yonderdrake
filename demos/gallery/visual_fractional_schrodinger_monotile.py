"""Compare four Schrödinger models on the hat aperiodic monotile."""

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
from matplotlib import animation, tri  # noqa: E402
from matplotlib import pyplot as plt  # noqa: E402


def hat_outline() -> np.ndarray:
    """Return the Smith-Myers-Kaplan-Goodman-Strauss hat polykite."""
    root_three_over_two = np.sqrt(3.0) / 2.0
    lattice = np.asarray(
        [
            (0, 0),
            (-1, -1),
            (0, -2),
            (2, -2),
            (2, -1),
            (4, -2),
            (5, -1),
            (4, 0),
            (3, 0),
            (2, 2),
            (0, 3),
            (0, 2),
            (-1, 2),
        ],
        dtype=np.float64,
    )
    outline = np.column_stack(
        (
            lattice[:, 0] + 0.5 * lattice[:, 1],
            root_three_over_two * lattice[:, 1],
        )
    )
    return outline - 0.5 * (outline.min(axis=0) + outline.max(axis=0))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parents[1] / "demo-output"),
    )
    parser.add_argument(
        "--mesh",
        type=Path,
        help="compatible Gmsh mesh; defaults to the committed demo mesh",
    )
    parser.add_argument("--modes", type=int, default=24)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--final-time", type=float, default=0.80)
    parser.add_argument("--time-order", type=float, default=0.92)
    parser.add_argument("--space-order", type=float, default=0.72)
    parser.add_argument(
        "--sinc-truncation-target",
        type=float,
        default=5.0e-3,
    )
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument(
        "--degree",
        type=int,
        choices=(1, 2),
        default=2,
        help="CG degree for the solve; output is sampled at mesh vertices",
    )
    args = parser.parse_args()

    import firedrake as fd

    try:
        import irksome
    except ImportError as error:
        raise SystemExit(
            "This optional demo requires Irksome: "
            "https://github.com/firedrakeproject/Irksome"
        ) from error

    from yonderdrake import (
        CaputoDerivative,
        Diethelm2008,
        FractionalTimeStepper,
        SpectralFractionalLaplacian,
    )

    configure_matplotlib(plt)
    destination = output_directory(args.output)
    mesh_path = args.mesh or (
        Path(__file__).resolve().parent / "meshes" / "aperiodic-monotile.msh"
    )
    outline_points = hat_outline()
    outline = np.vstack((outline_points, outline_points[0]))
    mesh = fd.Mesh(str(mesh_path), comm=fd.COMM_SELF)
    space = fd.VectorFunctionSpace(mesh, "CG", args.degree, dim=2)
    # Sample higher-degree output onto animation vertices.
    output_view = (
        None
        if args.degree == 1
        else fd.Function(
            fd.VectorFunctionSpace(mesh, "CG", 1, dim=2), name="p1_output_view"
        )
    )

    def p1_snapshot(state: Any) -> np.ndarray:
        if output_view is None:
            return np.asarray(state.dat.data_ro).copy()
        return np.asarray(output_view.interpolate(state).dat.data_ro).copy()

    x, y = fd.SpatialCoordinate(mesh)
    envelope = fd.exp(-((x + 0.10) ** 2 / 0.70 + (y - 0.05) ** 2 / 0.58))
    phase = 2.2 * (x + 0.10) + 0.45 * (y - 0.05)
    initial_expression = fd.as_vector(
        (envelope * fd.cos(phase), envelope * fd.sin(phase))
    )

    model_names = (
        "Classical",
        "Space fractional",
        "Time fractional",
        "Fully fractional",
    )
    states = {
        name: fd.Function(space, name=name).interpolate(initial_expression)
        for name in model_names
    }
    boundaries = {
        name: fd.DirichletBC(
            space,
            fd.as_vector((0.0, 0.0)),
            "on_boundary",
        )
        for name in model_names
    }
    for name, state in states.items():
        boundaries[name].apply(state)

    time = fd.Constant(0.0)
    step_size = fd.Constant(args.dt)
    shift_solver_parameters = {
        "ksp_type": "preonly",
        "pc_type": "lu",
    }
    direct_parameters = {
        "snes_type": "ksponly",
        "ksp_type": "preonly",
        "pc_type": "lu",
    }
    external_parameters = {
        "mat_type": "matfree",
        "snes_type": "ksponly",
        "ksp_type": "gmres",
        "ksp_gmres_restart": 2000 if args.degree == 2 else 30,
        "ksp_rtol": 2.0e-7,
        "ksp_atol": 1.0e-10,
        "ksp_max_it": 2500 if args.degree == 2 else 500,
        "pc_type": "python",
        "pc_python_type": "firedrake.MassInvPC",
        "Mp_pc_type": "lu",
    }

    def local_hamiltonian_form(state: Any, test: Any) -> Any:
        return (
            -fd.inner(fd.grad(state[1]), fd.grad(test[0]))
            + fd.inner(fd.grad(state[0]), fd.grad(test[1]))
        ) * fd.dx

    def rotate(field: Any) -> Any:
        return fd.as_vector((-field[1], field[0]))

    classical_state = states["Classical"]
    classical_test = fd.TestFunction(space)
    classical_form = fd.inner(
        irksome.Dt(classical_state), classical_test
    ) * fd.dx + local_hamiltonian_form(classical_state, classical_test)
    classical_stepper = irksome.TimeStepper(
        classical_form,
        irksome.GaussLegendre(1),
        time,
        step_size,
        classical_state,
        bcs=boundaries["Classical"],
        solver_parameters=direct_parameters,
    )

    space_state = states["Space fractional"]
    space_test = fd.TestFunction(space)
    space_hamiltonian = SpectralFractionalLaplacian(
        space_state,
        args.space_order,
        bcs=boundaries["Space fractional"],
        sinc_truncation_target=args.sinc_truncation_target,
        shift_cache="all",
        shift_solver_parameters=shift_solver_parameters,
    )
    space_form = (
        fd.inner(
            irksome.Dt(space_state) + rotate(space_hamiltonian),
            space_test,
        )
        * fd.dx
    )
    space_stepper = irksome.TimeStepper(
        space_form,
        irksome.GaussLegendre(1),
        time,
        step_size,
        space_state,
        bcs=boundaries["Space fractional"],
        solver_parameters=external_parameters,
    )

    time_state = states["Time fractional"]
    time_test = fd.TestFunction(space)
    time_form = fd.inner(
        CaputoDerivative(time_state, args.time_order),
        time_test,
    ) * fd.dx + local_hamiltonian_form(time_state, time_test)
    time_stepper = FractionalTimeStepper(
        time_form,
        Diethelm2008(args.modes),
        time,
        step_size,
        time_state,
        bcs=boundaries["Time fractional"],
        solver_parameters=direct_parameters,
    )

    fully_state = states["Fully fractional"]
    fully_test = fd.TestFunction(space)
    fully_hamiltonian = SpectralFractionalLaplacian(
        fully_state,
        args.space_order,
        bcs=boundaries["Fully fractional"],
        sinc_truncation_target=args.sinc_truncation_target,
        shift_cache="all",
        shift_solver_parameters=shift_solver_parameters,
    )
    fully_form = (
        fd.inner(
            CaputoDerivative(fully_state, args.time_order) + rotate(fully_hamiltonian),
            fully_test,
        )
        * fd.dx
    )
    fully_stepper = FractionalTimeStepper(
        fully_form,
        Diethelm2008(args.modes),
        time,
        step_size,
        fully_state,
        bcs=boundaries["Fully fractional"],
        solver_parameters=external_parameters,
    )

    times = [0.0]
    component_histories = {name: [p1_snapshot(state)] for name, state in states.items()}
    number_of_steps = round(args.final_time / args.dt)
    for index in range(1, number_of_steps + 1):
        classical_stepper.advance()
        space_stepper.advance()
        time_stepper.advance()
        fully_stepper.advance()
        time.assign(index * args.dt)
        times.append(float(time))
        for name, state in states.items():
            component_histories[name].append(p1_snapshot(state))

    components = {
        name: np.asarray(history) for name, history in component_histories.items()
    }
    densities = {
        name: np.sum(np.square(history), axis=2) for name, history in components.items()
    }
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
    density_limit = 1.0
    data_fields = {}
    for name in model_names:
        data_fields[f"{name}: real"] = components[name][:, :, 0]
        data_fields[f"{name}: imaginary"] = components[name][:, :, 1]
        data_fields[f"{name}: probability density"] = densities[name]

    data_path = destination / "fractional-schrodinger-aperiodic-monotile-data.csv.gz"
    save_plot_csv(
        data_path,
        times=times,
        coordinates=coordinates,
        cells=cells,
        fields=data_fields,
        metadata={
            "demo": "fractional-schrodinger-aperiodic-monotile",
            "equation": "i D_C^alpha psi = (-Delta_D)^s psi",
            "models": model_names,
            "time_order": args.time_order,
            "space_order": args.space_order,
            "dt": args.dt,
            "final_time": args.final_time,
            "modes": args.modes,
            "mesh": mesh_path.name,
            "element_degree": args.degree,
            "sinc_truncation_target": args.sinc_truncation_target,
            "density_limit": density_limit,
            "monotile_reference": "https://doi.org/10.5070/C64163843",
            "irksome": "https://github.com/firedrakeproject/Irksome",
        },
    )

    figure = plt.figure(figsize=(21.0, 9.0), facecolor=PAPER)
    grid = figure.add_gridspec(
        1,
        4,
        left=0.06,
        right=0.975,
        bottom=0.120,
        top=0.590,
        wspace=0.08,
    )
    figure.suptitle(
        "Fractional Schrödinger on the Aperiodic Monotile",
        x=0.045,
        y=0.965,
        ha="left",
        color=INK,
        fontsize=27,
        fontweight="bold",
    )
    figure.text(
        0.046,
        0.885,
        (
            rf"$\Delta t={args.dt:g}$  •  "
            rf"$\alpha={args.time_order:.2f}$  •  "
            rf"$s={args.space_order:.2f}$  •  "
            f"{coordinates.shape[0]:,} vertices"
        ),
        color=INK,
        fontsize=24,
    )
    figure.text(
        0.046,
        0.735,
        (
            r"$\psi_0=e^{-[(x+0.10)^2/0.70+(y-0.05)^2/0.58]}"
            r"e^{\,i[2.2(x+0.10)+0.45(y-0.05)]},"
            r"\qquad \psi|_{\partial\Omega}=0$"
        ),
        color=INK,
        fontsize=26,
    )
    equations = (
        r"$i\partial_t\psi=(-\Delta_D)\psi$",
        rf"$i\partial_t\psi=(-\Delta_D)^{{{args.space_order:.2f}}}\psi$",
        rf"$iD_C^{{{args.time_order:.2f}}}\psi=(-\Delta_D)\psi$",
        rf"$iD_C^{{{args.time_order:.2f}}}\psi"
        rf"=(-\Delta_D)^{{{args.space_order:.2f}}}\psi$",
    )
    panel_names = (
        "Classical",
        "Space fractional",
        "Time fractional",
        "Fully fractional",
    )
    model_colors = (INK, TEAL, GOLD, CORAL)
    x_min, y_min = outline.min(axis=0)
    x_max, y_max = outline.max(axis=0)
    x_padding = 0.03 * (x_max - x_min)
    y_range = y_max - y_min
    images = []
    for column, (name, panel_name, equation, model_color) in enumerate(
        zip(model_names, panel_names, equations, model_colors, strict=True)
    ):
        axis = figure.add_subplot(grid[0, column])
        image = axis.tripcolor(
            triangulation,
            densities[name][0],
            shading="gouraud",
            cmap="viridis",
            vmin=0.0,
            vmax=density_limit,
        )
        axis.plot(
            outline[:, 0],
            outline[:, 1],
            color=INK,
            linewidth=1.5,
            solid_joinstyle="round",
        )
        axis.set_xlim(x_min - x_padding, x_max + x_padding)
        axis.set_ylim(y_min - 0.03 * y_range, y_max + 0.12 * y_range)
        axis.set_aspect("equal")
        axis.set_axis_off()
        axis.text(
            0.0,
            1.18,
            panel_name,
            transform=axis.transAxes,
            color=model_color,
            fontsize=30,
            fontweight="bold",
            va="top",
        )
        axis.text(
            0.0,
            1.045,
            equation,
            transform=axis.transAxes,
            color=INK,
            fontsize=18,
            va="top",
        )
        images.append(image)
    colorbar_axis = figure.add_axes((0.39, 0.045, 0.22, 0.018))
    colorbar = figure.colorbar(
        images[0],
        cax=colorbar_axis,
        orientation="horizontal",
        label=r"probability density  $|\psi|^2$",
        ticks=(0.0, 0.5, 1.0),
    )
    colorbar.ax.xaxis.set_label_position("top")
    colorbar.set_ticklabels(("0", "0.5", "1"))
    signature(figure)
    time_label = time_counter(figure)

    def update(frame: int) -> tuple[Any, ...]:
        for image, name in zip(images, model_names, strict=True):
            image.set_array(densities[name][frame])
        time_label.set_text(f"t = {times[frame]:.3f}")
        return (*images, time_label)

    movie = animation.FuncAnimation(
        figure,
        update,
        frames=len(times),
        interval=1000 / args.fps,
        blit=False,
    )
    output_path = destination / "fractional-schrodinger-aperiodic-monotile.gif"
    movie.save(
        output_path,
        writer=animation.PillowWriter(fps=args.fps),
        dpi=110,
    )
    plt.close(figure)

    print(output_path)
    print(data_path)
    print(f"mesh: {coordinates.shape[0]} vertices, {cells.shape[0]} triangles")
    print("Irksome classical:", classical_stepper.solver_stats())
    print("Irksome space fractional:", space_stepper.solver_stats())
    print("Yonderdrake time fractional:", time_stepper.solver_stats())
    print("Yonderdrake fully fractional:", fully_stepper.solver_stats())


if __name__ == "__main__":
    main()
