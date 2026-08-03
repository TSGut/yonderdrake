"""Compare classical, Caputo, and Riemann-Liouville phase separation on a sphere."""

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parents[1] / "demo-output"),
    )
    parser.add_argument("--refinement", type=int, default=5)
    parser.add_argument("--modes", type=int, default=32)
    parser.add_argument("--dt", type=float, default=0.015)
    parser.add_argument("--final-time", type=float, default=1.2)
    args = parser.parse_args()

    import firedrake as fd

    from yonderdrake import (
        BirkSong,
        CaputoDerivative,
        FractionalTimeStepper,
        RiemannLiouvilleDerivative,
    )

    configure_matplotlib(plt)
    destination = output_directory(args.output)
    alpha = 0.70
    interface_width = 0.04
    mesh = fd.IcosahedralSphereMesh(
        radius=1.0,
        refinement_level=args.refinement,
    )
    space = fd.FunctionSpace(mesh, "CG", 1)
    x, y, z = fd.SpatialCoordinate(mesh)
    initial_pattern = 0.46 * (
        fd.sin(2.2 * fd.pi * x) * fd.sin(2.8 * fd.pi * y)
        + 0.62
        * fd.sin(3.4 * fd.pi * y + 0.35)
        * fd.sin(2.1 * fd.pi * z)
        + 0.38 * fd.sin(2.7 * fd.pi * (z + x))
    )
    time = fd.Constant(0.0)
    step_size = fd.Constant(args.dt)
    model_names = ("Classical", "Caputo", "Riemann–Liouville")
    fields = {
        name: fd.Function(space, name=name).interpolate(initial_pattern)
        for name in model_names
    }

    def spatial_residual(field: Any, test: Any) -> Any:
        return (
            interface_width**2 * fd.inner(fd.grad(field), fd.grad(test))
            + fd.inner(field**3 - field, test)
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
    solver_parameters = {
        "snes_type": "newtonls",
        "snes_rtol": 1.0e-9,
        "snes_max_it": 20,
        "ksp_type": "preonly",
        "pc_type": "lu",
    }
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
            BirkSong(args.modes),
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

    def free_energy(field: Any) -> float:
        return float(
            fd.assemble(
                (
                    0.5
                    * interface_width**2
                    * fd.inner(fd.grad(field), fd.grad(field))
                    + 0.25 * (field**2 - 1.0) ** 2
                )
                * fd.dx
            )
        )

    times = [0.0]
    histories = {
        name: [np.asarray(field.dat.data_ro).copy()]
        for name, field in fields.items()
    }
    energies = {
        name: [free_energy(field)]
        for name, field in fields.items()
    }
    contrasts = {
        name: [float(fd.assemble(field**2 * fd.dx))]
        for name, field in fields.items()
    }
    ordinary_iterations = 0
    number_of_steps = round(args.final_time / args.dt)
    for _ in range(number_of_steps):
        ordinary_solver.solve()
        ordinary_iterations += ordinary_solver.snes.getIterationNumber()
        ordinary_previous.assign(fields["Classical"])
        fractional_steppers["Caputo"].advance()
        fractional_steppers["Riemann–Liouville"].advance()
        time.assign(time + step_size)
        times.append(float(time))
        for name, field in fields.items():
            histories[name].append(np.asarray(field.dat.data_ro).copy())
            energies[name].append(free_energy(field))
            contrasts[name].append(float(fd.assemble(field**2 * fd.dx)))

    history_arrays = {
        name: np.asarray(values)
        for name, values in histories.items()
    }
    data_path = destination / "time-derivative-phase-separation-data.csv.gz"
    diagnostic_series = {}
    for name in model_names:
        diagnostic_series[f"{name}:free_energy"] = (
            np.asarray(times),
            np.asarray(energies[name]),
        )
        diagnostic_series[f"{name}:phase_contrast"] = (
            np.asarray(times),
            np.asarray(contrasts[name]),
        )
    save_plot_csv(
        data_path,
        times=times,
        coordinates=coordinates,
        cells=cells,
        fields=history_arrays,
        metadata={
            "demo": "time-derivative-phase-separation",
            "equation": "D_t u-epsilon^2 Delta_S2 u+u^3-u=0",
            "operators": model_names,
            "alpha": alpha,
            "interface_width": interface_width,
            "dt": args.dt,
            "final_time": args.final_time,
            "fps": 9,
            "field_limits": (-1.0, 1.0),
        },
        series=diagnostic_series,
    )

    movie_figure = plt.figure(figsize=(16.0, 14.2), facecolor=PAPER)
    movie_grid = movie_figure.add_gridspec(
        3,
        2,
        left=0.075,
        right=0.94,
        bottom=0.06,
        top=0.570,
        width_ratios=(0.9, 1.12),
        hspace=0.34,
        wspace=0.24,
    )
    surfaces = []
    energy_markers = []
    contrast_markers = []
    surface_axes = []
    model_colors = (INK, TEAL, CORAL)
    for row, (name, model_color) in enumerate(
        zip(model_names, model_colors, strict=True)
    ):
        surface_axis = movie_figure.add_subplot(
            movie_grid[row, 0],
            projection="3d",
        )
        metric_axis = movie_figure.add_subplot(movie_grid[row, 1])
        surface = surface_axis.plot_trisurf(
            coordinates[:, 0],
            coordinates[:, 1],
            coordinates[:, 2],
            triangles=cells,
            shade=False,
            antialiased=False,
            cmap="viridis",
            vmin=-1.0,
            vmax=1.0,
        )
        surface.set_array(
            np.mean(history_arrays[name][0][cells], axis=1)
        )
        surface_axis.set_title(name, loc="left", color=model_color)
        surface_axis.view_init(elev=22, azim=-58)
        surface_axis.set_box_aspect((1.0, 1.0, 1.0))
        surface_axis.set_axis_off()
        surface_axes.append(surface_axis)
        surfaces.append(surface)

        metric_axis.plot(
            times,
            energies[name],
            color=TEAL,
            linewidth=2.2,
        )
        metric_axis.tick_params(axis="y", labelcolor=TEAL)
        contrast_axis = metric_axis.twinx()
        contrast_axis.plot(
            times,
            contrasts[name],
            color=CORAL,
            linewidth=1.7,
        )
        contrast_axis.tick_params(axis="y", labelcolor=CORAL)
        if row == 2:
            metric_axis.set_xlabel("time")
        energy_markers.append(
            metric_axis.scatter(
                [times[0]],
                [energies[name][0]],
                s=38,
                color=INK,
                zorder=5,
            )
        )
        contrast_markers.append(
            contrast_axis.scatter(
                [times[0]],
                [contrasts[name][0]],
                s=28,
                color=CORAL,
                zorder=5,
            )
        )
    movie_figure.suptitle(
        "Three clocks on a sphere",
        x=0.075,
        y=0.965,
        ha="left",
        color=INK,
        fontsize=24,
        fontweight="bold",
    )
    movie_figure.text(
        0.076,
        0.895,
        (
            rf"$\Delta t={args.dt:g}$  •  "
            rf"$\alpha={alpha:.2f}$  •  "
            rf"$\varepsilon={interface_width:.2f}$  •  "
            f"{coordinates.shape[0]:,} vertices"
        ),
        color=INK,
        fontsize=24,
    )
    movie_figure.text(
        0.076,
        0.825,
        (
            r"$\mathcal{D}_t u-\varepsilon^2\Delta_{S^2}u+u^3-u=0,"
            r"\quad \mathcal{D}_t\in"
            r"\{\partial_t,D_C^{0.70},D_{RL}^{0.70}\}$"
        ),
        color=INK,
        fontsize=24,
    )
    movie_figure.text(
        0.076,
        0.735,
        (
            r"$u(\mathbf{x},0)=0.46["
            r"\sin(2.2\pi x)\sin(2.8\pi y)"
            r"]$"
            "\n"
            r"$+0.46[0.62\sin(3.4\pi y+0.35)\sin(2.1\pi z)"
            r"+0.38\sin(2.7\pi(z+x))]$"
        ),
        color=INK,
        fontsize=19,
    )
    movie_figure.text(
        0.076,
        0.635,
        (
            r"$\mathcal{F}[u]=\int_{S^2}"
            r"[\frac{1}{2}\varepsilon^2|\nabla_\Gamma u|^2"
            r"+\frac{1}{4}(u^2-1)^2]\,dS,"
            r"\qquad C[u]=\int_{S^2}u^2\,dS$"
        ),
        color=INK,
        fontsize=20,
    )
    signature(movie_figure)
    time_label = time_counter(movie_figure)

    def update(frame: int):
        artists = []
        for row, name in enumerate(model_names):
            surfaces[row].set_array(
                np.mean(history_arrays[name][frame][cells], axis=1)
            )
            surfaces[row].set_clim(-1.0, 1.0)
            surface_axes[row].view_init(
                elev=22,
                azim=-58 + 0.7 * frame,
            )
            energy_markers[row].set_offsets(
                [[times[frame], energies[name][frame]]]
            )
            contrast_markers[row].set_offsets(
                [[times[frame], contrasts[name][frame]]]
            )
            artists.extend(
                (
                    surfaces[row],
                    energy_markers[row],
                    contrast_markers[row],
                )
            )
        time_label.set_text(f"t = {times[frame]:.2f}")
        artists.append(time_label)
        return artists

    movie = animation.FuncAnimation(
        movie_figure,
        update,
        frames=len(times),
        interval=110,
        blit=False,
    )
    gif_path = destination / "time-derivative-phase-separation.gif"
    movie.save(gif_path, writer=animation.PillowWriter(fps=9), dpi=120)
    plt.close(movie_figure)

    print(gif_path)
    print(data_path)
    for name in model_names:
        print(
            f"{name}: energy {energies[name][0]:.6e} -> "
            f"{energies[name][-1]:.6e}"
        )
    print(f"ordinary nonlinear iterations: {ordinary_iterations}")
    print("caputo:", fractional_steppers["Caputo"].solver_stats())
    print(
        "riemann_liouville:",
        fractional_steppers["Riemann–Liouville"].solver_stats(),
    )


if __name__ == "__main__":
    main()
