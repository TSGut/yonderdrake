"""Compare classical, spectral, and Riesz heat flow on a Koch snowflake."""

from __future__ import annotations

import argparse
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "yonderdrake-matplotlib"),
)
import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")

from _visual_data import load_plot_csv, save_plot_csv  # noqa: E402
from _visual_style import (  # noqa: E402
    INK,
    PAPER,
    configure_matplotlib,
    output_directory,
    signature,
    time_counter,
)
from matplotlib import animation, tri  # noqa: E402
from matplotlib import pyplot as plt  # noqa: E402

MODEL_NAMES = ("Classical", "Spectral", "Riesz")


@dataclass(frozen=True)
class HeatData:
    coordinates: np.ndarray
    cells: np.ndarray
    times: np.ndarray
    fields: dict[str, np.ndarray]
    outline: np.ndarray
    diagnostics: dict[str, Any]


def koch_outline(iterations: int, radius: float = 1.62) -> np.ndarray:
    """Return a counterclockwise Koch-snowflake polygon."""
    angles = np.deg2rad((90.0, 210.0, 330.0))
    points = radius * np.column_stack((np.cos(angles), np.sin(angles)))
    rotation = np.asarray(
        (
            (0.5, np.sqrt(3.0) / 2.0),
            (-np.sqrt(3.0) / 2.0, 0.5),
        )
    )
    for _ in range(iterations):
        refined = []
        for start, end in zip(points, np.roll(points, -1, axis=0), strict=True):
            third = (end - start) / 3.0
            first = start + third
            second = start + 2.0 * third
            peak = first + rotation @ third
            refined.extend((start, first, peak, second))
        points = np.asarray(refined)
    return points


def _simulate(args: argparse.Namespace, destination: Path) -> tuple[HeatData, Path]:
    import firedrake as fd
    import irksome

    from yonderdrake import (
        RieszFractionalLaplacian,
        SpectralFractionalLaplacian,
    )

    outline_points = koch_outline(args.snowflake_iterations)
    outline = np.vstack((outline_points, outline_points[0]))
    mesh = fd.Mesh(
        str(args.mesh),
        comm=fd.COMM_SELF,
    )
    space = fd.FunctionSpace(mesh, "CG", 1)
    x, y = fd.SpatialCoordinate(mesh)
    angle = np.deg2rad(-18.0)
    rotated_x = np.cos(angle) * (x + 0.18) - np.sin(angle) * (y - 0.04)
    rotated_y = np.sin(angle) * (x + 0.18) + np.cos(angle) * (y - 0.04)
    radius_squared = (rotated_x / 0.92) ** 2 + (rotated_y / 0.60) ** 2
    initial_expression = fd.conditional(
        radius_squared < 0.27,
        1.0,
        fd.conditional(radius_squared < 1.0, 0.56, 0.0),
    )
    states = {
        name: fd.Function(space, name=name).interpolate(initial_expression)
        for name in MODEL_NAMES
    }
    boundaries = {
        name: fd.DirichletBC(space, 0.0, "on_boundary") for name in MODEL_NAMES
    }
    for name, state in states.items():
        boundaries[name].apply(state)

    time = fd.Constant(0.0)
    step_size = fd.Constant(args.dt)
    direct_parameters = {
        "snes_type": "ksponly",
        "ksp_type": "preonly",
        "pc_type": "lu",
    }
    external_parameters = {
        "mat_type": "matfree",
        "snes_type": "ksponly",
        "ksp_type": "gmres",
        "ksp_rtol": 2.0e-7,
        "ksp_atol": 1.0e-10,
        "ksp_max_it": 300,
        "pc_type": "python",
        "pc_python_type": "firedrake.MassInvPC",
        "Mp_pc_type": "lu",
    }
    table = irksome.RadauIIA(1)

    classical_state = states["Classical"]
    classical_test = fd.TestFunction(space)
    classical_form = (
        fd.inner(irksome.Dt(classical_state), classical_test)
        + args.diffusivity * fd.inner(fd.grad(classical_state), fd.grad(classical_test))
    ) * fd.dx
    steppers: dict[str, Any] = {
        "Classical": irksome.TimeStepper(
            classical_form,
            table,
            time,
            step_size,
            classical_state,
            bcs=boundaries["Classical"],
            solver_parameters=direct_parameters,
        )
    }

    spectral_state = states["Spectral"]
    spectral_test = fd.TestFunction(space)
    spectral_operator = SpectralFractionalLaplacian(
        spectral_state,
        args.order,
        bcs=boundaries["Spectral"],
        sinc_truncation_target=args.sinc_truncation_target,
        shift_cache="all",
        shift_solver_parameters={
            "ksp_type": "preonly",
            "pc_type": "lu",
        },
    )
    spectral_form = (
        fd.inner(
            irksome.Dt(spectral_state) + args.diffusivity * spectral_operator,
            spectral_test,
        )
        * fd.dx
    )
    steppers["Spectral"] = irksome.TimeStepper(
        spectral_form,
        table,
        time,
        step_size,
        spectral_state,
        bcs=boundaries["Spectral"],
        solver_parameters=external_parameters,
    )

    riesz_state = states["Riesz"]
    riesz_test = fd.TestFunction(space)
    riesz_operator = RieszFractionalLaplacian(
        riesz_state,
        args.order,
        bcs=boundaries["Riesz"] if args.order >= 0.5 else None,
        extension="zero",
        target_quadrature_degree=args.target_quadrature_degree,
        target_quadrature_rule=args.target_quadrature_rule,
        assembly="hmatrix",
        compression_tolerance=args.compression_tolerance,
        admissibility=args.admissibility,
        leaf_size=args.leaf_size,
    )
    riesz_form = (
        fd.inner(
            irksome.Dt(riesz_state) + args.diffusivity * riesz_operator,
            riesz_test,
        )
        * fd.dx
    )
    steppers["Riesz"] = irksome.TimeStepper(
        riesz_form,
        table,
        time,
        step_size,
        riesz_state,
        bcs=boundaries["Riesz"],
        solver_parameters=external_parameters,
    )

    histories = {
        name: [np.asarray(state.dat.data_ro).copy()] for name, state in states.items()
    }
    times = [0.0]
    number_of_steps = round(args.final_time / args.dt)
    for index in range(1, number_of_steps + 1):
        for stepper in steppers.values():
            stepper.advance()
        time.assign(index * args.dt)
        times.append(float(time))
        for name, state in states.items():
            histories[name].append(np.asarray(state.dat.data_ro).copy())

    coordinates = np.asarray(mesh.coordinates.dat.data_ro).copy()
    cells = np.asarray(
        mesh.coordinates.function_space().cell_node_map().values,
        dtype=np.int32,
    ).copy()
    fields = {name: np.asarray(history) for name, history in histories.items()}
    diagnostics = {
        "classical": steppers["Classical"].solver_stats(),
        "spectral_stepper": steppers["Spectral"].solver_stats(),
        "riesz_stepper": steppers["Riesz"].solver_stats(),
        "spectral_operator": spectral_operator.diagnostics(),
        "riesz_operator": riesz_operator.diagnostics(),
    }
    data_path = destination / "fractional-heat-koch-snowflake-data.csv.gz"
    save_plot_csv(
        data_path,
        times=times,
        coordinates=coordinates,
        cells=cells,
        fields=fields,
        metadata={
            "demo": "fractional-heat-koch-snowflake",
            "models": MODEL_NAMES,
            "order": args.order,
            "diffusivity": args.diffusivity,
            "dt": args.dt,
            "final_time": args.final_time,
            "mesh": args.mesh.name,
            "snowflake_iterations": args.snowflake_iterations,
            "target_quadrature_degree": args.target_quadrature_degree,
            "target_quadrature_rule": args.target_quadrature_rule,
            "compression_tolerance": args.compression_tolerance,
            "admissibility": args.admissibility,
            "leaf_size": args.leaf_size,
            "sinc_truncation_target": args.sinc_truncation_target,
            "irksome": "https://github.com/firedrakeproject/Irksome",
        },
    )
    return (
        HeatData(
            coordinates=coordinates,
            cells=cells,
            times=np.asarray(times),
            fields=fields,
            outline=outline,
            diagnostics=diagnostics,
        ),
        data_path,
    )


def _load(args: argparse.Namespace, path: Path) -> HeatData:
    saved = load_plot_csv(path)
    if saved.metadata.get("demo") != "fractional-heat-koch-snowflake":
        raise ValueError(f"{path} is not Koch-snowflake heat-demo data")
    args.order = float(saved.metadata["order"])
    args.diffusivity = float(saved.metadata["diffusivity"])
    args.dt = float(saved.metadata["dt"])
    args.final_time = float(saved.metadata["final_time"])
    args.snowflake_iterations = int(saved.metadata["snowflake_iterations"])
    args.target_quadrature_degree = int(saved.metadata["target_quadrature_degree"])
    args.target_quadrature_rule = str(
        saved.metadata.get("target_quadrature_rule", "ordinary")
    )
    args.compression_tolerance = float(saved.metadata["compression_tolerance"])
    args.admissibility = float(saved.metadata["admissibility"])
    args.leaf_size = int(saved.metadata["leaf_size"])
    args.sinc_truncation_target = float(saved.metadata["sinc_truncation_target"])
    return HeatData(
        coordinates=saved.coordinates,
        cells=saved.cells,
        times=saved.times,
        fields={name: saved.fields[name] for name in MODEL_NAMES},
        outline=np.vstack(
            (
                koch_outline(args.snowflake_iterations),
                koch_outline(args.snowflake_iterations)[0],
            )
        ),
        diagnostics={},
    )


def _render(
    args: argparse.Namespace,
    data: HeatData,
    destination: Path,
    data_path: Path,
) -> Path:
    coordinates = data.coordinates
    cells = data.cells
    triangulation = tri.Triangulation(
        coordinates[:, 0],
        coordinates[:, 1],
        cells,
    )
    x_min, y_min = coordinates.min(axis=0)
    x_max, y_max = coordinates.max(axis=0)
    x_padding = 0.04 * (x_max - x_min)
    y_padding = 0.04 * (y_max - y_min)

    figure = plt.figure(figsize=(17.0, 12.0), facecolor=PAPER)
    grid = figure.add_gridspec(
        2,
        3,
        left=0.045,
        right=0.965,
        bottom=0.115,
        top=0.705,
        height_ratios=(1.0, 0.52),
        hspace=0.015,
        wspace=0.08,
    )
    figure.suptitle(
        "Fractional heat on the Koch snowflake",
        x=0.045,
        y=0.965,
        ha="left",
        color=INK,
        fontsize=27,
        fontweight="bold",
    )
    figure.text(
        0.046,
        0.910,
        (
            rf"$\Delta t={args.dt:g}$  •  "
            rf"$s={args.order:.2f}$  •  "
            rf"$\kappa={args.diffusivity:.2f}$  •  "
            f"{coordinates.shape[0]:,} vertices"
        ),
        color=INK,
        fontsize=24,
    )
    figure.text(
        0.046,
        0.840,
        (
            r"$\partial_tu+\kappa\mathcal{A}_su=0,\qquad "
            r"\mathcal{A}_s\in"
            rf"\{{-\Delta_D,(-\Delta_D)^{{{args.order:.2f}}},"
            rf"(-\Delta)^{{{args.order:.2f}}}_R\}},\qquad "
            r"u|_{\partial\Omega}=0$"
        ),
        color=INK,
        fontsize=25,
    )
    figure.text(
        0.046,
        0.765,
        (
            r"$u(\mathbf{x},0)=1$ on the inner plateau, "
            r"$0.56$ on the outer plateau, and $0$ elsewhere"
        ),
        color=INK,
        fontsize=23,
    )
    subtitles = (
        r"$\mathcal{A}_1=-\Delta_D$",
        rf"$\mathcal{{A}}_s=(-\Delta_D)^{{{args.order:.2f}}}$",
        rf"$\mathcal{{A}}_s=(-\Delta)^{{{args.order:.2f}}}_R$",
    )
    surfaces = []
    surface_axes = []
    heatmaps = []
    for column, (name, subtitle) in enumerate(zip(MODEL_NAMES, subtitles, strict=True)):
        values = data.fields[name][0]
        surface_axis = figure.add_subplot(grid[0, column], projection="3d")
        heatmap_axis = figure.add_subplot(grid[1, column])
        surface = surface_axis.plot_trisurf(
            coordinates[:, 0],
            coordinates[:, 1],
            values,
            triangles=cells,
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            shade=False,
            antialiased=True,
            linewidth=0.0,
        )
        surface.set_array(values[cells].mean(axis=1))
        surface_axis.set_xlim(x_min - x_padding, x_max + x_padding)
        surface_axis.set_ylim(y_min - y_padding, y_max + y_padding)
        surface_axis.set_zlim(0.0, 1.02)
        surface_axis.set_box_aspect(
            (x_max - x_min, y_max - y_min, 1.45),
            zoom=1.12,
        )
        surface_axis.view_init(elev=30.0, azim=-62.0)
        surface_axis.set_axis_off()
        surface_axis.set_title(
            name,
            loc="left",
            color=INK,
            fontsize=30,
            fontweight="bold",
            pad=3,
        )
        surface_axis.text2D(
            0.0,
            0.94,
            subtitle,
            transform=surface_axis.transAxes,
            color=INK,
            fontsize=21,
        )
        heatmap = heatmap_axis.tripcolor(
            triangulation,
            values,
            shading="gouraud",
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
        )
        heatmap_axis.plot(
            data.outline[:, 0],
            data.outline[:, 1],
            color=INK,
            linewidth=1.25,
            solid_joinstyle="round",
        )
        heatmap_axis.set_xlim(x_min - x_padding, x_max + x_padding)
        heatmap_axis.set_ylim(y_min - y_padding, y_max + y_padding)
        heatmap_axis.set_aspect("equal")
        heatmap_axis.set_axis_off()
        surfaces.append(surface)
        surface_axes.append(surface_axis)
        heatmaps.append(heatmap)

    figure.text(
        0.018,
        0.505,
        "surface",
        rotation=90,
        va="center",
        color=INK,
        fontsize=22,
    )
    figure.text(
        0.018,
        0.205,
        "heatmap",
        rotation=90,
        va="center",
        color=INK,
        fontsize=22,
    )
    colorbar_axis = figure.add_axes((0.37, 0.055, 0.26, 0.016))
    colorbar = figure.colorbar(
        heatmaps[0],
        cax=colorbar_axis,
        orientation="horizontal",
        label="temperature",
    )
    colorbar.ax.xaxis.set_label_position("top")
    signature(figure)
    time_label = time_counter(figure)

    def update(frame: int) -> tuple[Any, ...]:
        fraction = frame / max(1, len(data.times) - 1)
        azimuth = -62.0 + 90.0 * fraction
        elevation = 30.0 + 4.0 * np.sin(2.0 * np.pi * fraction)
        for surface, surface_axis, heatmap, name in zip(
            surfaces,
            surface_axes,
            heatmaps,
            MODEL_NAMES,
            strict=True,
        ):
            values = data.fields[name][frame]
            triangle_values = values[cells]
            vertices = np.concatenate(
                (coordinates[cells], triangle_values[:, :, None]),
                axis=2,
            )
            surface.set_verts(vertices)
            surface.set_array(triangle_values.mean(axis=1))
            surface.set_clim(0.0, 1.0)
            surface_axis.view_init(elev=elevation, azim=azimuth)
            heatmap.set_array(values)
            heatmap.set_clim(0.0, 1.0)
        time_label.set_text(f"t = {data.times[frame]:.2f}")
        return (*surfaces, *heatmaps, time_label)

    movie = animation.FuncAnimation(
        figure,
        update,
        frames=len(data.times),
        interval=1000 / args.fps,
        blit=False,
    )
    output_path = destination / "fractional-heat-koch-snowflake.gif"
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
    for frame in range(rendered.n_frames):
        rendered.seek(frame)
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
        duration=round(1000 / args.fps),
        loop=0,
        optimize=True,
    )
    rendered.close()
    render_path.unlink()
    print(output_path)
    print(data_path)
    print(f"mesh: {coordinates.shape[0]} vertices, {cells.shape[0]} triangles")
    for name in MODEL_NAMES:
        print(
            f"{name}: final range "
            f"[{data.fields[name][-1].min():.6f}, "
            f"{data.fields[name][-1].max():.6f}]"
        )
    for name, diagnostics in data.diagnostics.items():
        print(f"{name}: {diagnostics}")
    return output_path


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
    parser.add_argument("--snowflake-iterations", type=int, default=3)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--final-time", type=float, default=0.90)
    parser.add_argument("--order", type=float, default=0.72)
    parser.add_argument("--diffusivity", type=float, default=0.32)
    parser.add_argument("--target-quadrature-degree", type=int, default=2)
    parser.add_argument(
        "--target-quadrature-rule",
        choices=("boundary", "ordinary"),
        default="boundary",
    )
    parser.add_argument("--compression-tolerance", type=float, default=2.0e-3)
    parser.add_argument("--admissibility", type=float, default=1.0)
    parser.add_argument("--leaf-size", type=int, default=12)
    parser.add_argument(
        "--sinc-truncation-target",
        type=float,
        default=2.0e-3,
    )
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--plot-data", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    meshes = Path(__file__).resolve().parent / "meshes"
    if args.smoke:
        args.snowflake_iterations = 1
        if args.mesh is None:
            args.mesh = meshes / "koch-snowflake-smoke.msh"
        args.final_time = 0.04
        args.target_quadrature_degree = 2
        args.leaf_size = 6
    elif args.mesh is None:
        args.mesh = meshes / "koch-snowflake.msh"
    try:
        import irksome  # noqa: F401
    except ImportError as error:
        raise SystemExit(
            "This optional demo requires Irksome: "
            "https://github.com/firedrakeproject/Irksome"
        ) from error
    configure_matplotlib(plt)
    destination = output_directory(args.output)
    if args.plot_data is None:
        data, data_path = _simulate(args, destination)
    else:
        data_path = args.plot_data
        data = _load(args, data_path)
    _render(args, data, destination, data_path)


if __name__ == "__main__":
    main()
