"""Race two 3D periodic fractional heat flows through a multiscale gyroid."""

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

matplotlib.use("Agg")

from _visual_data import load_plot_csv, save_plot_csv  # noqa: E402
from _visual_style import (  # noqa: E402
    INK,
    MUTED,
    PAPER,
    WHITE,
    configure_matplotlib,
    output_directory,
    signature,
    time_counter,
)
from matplotlib import animation  # noqa: E402
from matplotlib import pyplot as plt  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402


@dataclass(frozen=True)
class GyroidData:
    """Saved logical-grid values for the animation."""

    times: np.ndarray
    coordinates: np.ndarray
    fields: dict[str, np.ndarray]
    shape: tuple[int, int, int]
    lengths: tuple[float, float, float]
    orders: tuple[float, float]
    metadata: dict[str, Any]
    diagnostics: dict[str, dict[str, Any]]


_CUBE_CORNERS = np.asarray(
    (
        (0, 0, 0),
        (1, 0, 0),
        (0, 1, 0),
        (1, 1, 0),
        (0, 0, 1),
        (1, 0, 1),
        (0, 1, 1),
        (1, 1, 1),
    ),
    dtype=np.int64,
)
_TETRAHEDRA = (
    (0, 1, 3, 7),
    (0, 3, 2, 7),
    (0, 2, 6, 7),
    (0, 6, 4, 7),
    (0, 4, 5, 7),
    (0, 5, 1, 7),
)
_TETRAHEDRON_EDGES = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))


def _isosurface(
    values: np.ndarray,
    lengths: tuple[float, float, float],
) -> np.ndarray:
    """Triangulate the zero level with periodic marching tetrahedra."""
    shape = values.shape
    extended = np.pad(values, ((0, 1), (0, 1), (0, 1)), mode="wrap")
    spacing = np.asarray(lengths) / np.asarray(shape)
    triangles: list[np.ndarray] = []
    for lower in np.ndindex(shape):
        corner_indices = _CUBE_CORNERS + np.asarray(lower)
        corner_values = extended[
            corner_indices[:, 0],
            corner_indices[:, 1],
            corner_indices[:, 2],
        ]
        corner_points = corner_indices * spacing
        for tetrahedron in _TETRAHEDRA:
            indices = np.asarray(tetrahedron)
            tetrahedron_values = corner_values[indices]
            if np.all(tetrahedron_values >= 0.0) or np.all(
                tetrahedron_values < 0.0
            ):
                continue
            tetrahedron_points = corner_points[indices]
            crossings = []
            for start, end in _TETRAHEDRON_EDGES:
                start_value = tetrahedron_values[start]
                end_value = tetrahedron_values[end]
                if (start_value < 0.0) == (end_value < 0.0):
                    continue
                fraction = start_value / (start_value - end_value)
                crossings.append(
                    tetrahedron_points[start]
                    + fraction
                    * (tetrahedron_points[end] - tetrahedron_points[start])
                )
            if len(crossings) == 3:
                triangles.append(np.asarray(crossings))
            elif len(crossings) == 4:
                polygon = np.asarray(crossings)
                triangles.extend((polygon[[0, 1, 2]], polygon[[0, 2, 3]]))
    return np.asarray(triangles, dtype=np.float64).reshape((-1, 3, 3))


def _logical_values(state: Any, grid: Any) -> np.ndarray:
    with state.dat.vec_ro as vector:
        coefficients = np.asarray(vector.array_r).copy()
    values = np.empty(grid.size, dtype=np.float64)
    values[grid.global_to_flat] = coefficients
    return values.reshape(grid.shape)


def _simulate(args: argparse.Namespace, destination: Path) -> tuple[GyroidData, Path]:
    import firedrake as fd

    from yonderdrake import PeriodicFractionalLaplacian

    if fd.COMM_WORLD.size != 1:
        raise ValueError("this rendering demo runs on one MPI rank")
    if args.cells < 3:
        raise ValueError("--cells must be at least 3 in every periodic direction")
    if args.dt <= 0.0 or args.final_time <= 0.0:
        raise ValueError("--dt and --final-time must be positive")
    if args.frames < 2:
        raise ValueError("--frames must be at least 2")
    if not 0.0 < args.orders[0] < args.orders[1] < 1.0:
        raise ValueError("--orders must satisfy 0 < LOW < HIGH < 1")

    lengths = (2.0 * np.pi,) * 3
    shape = (args.cells,) * 3
    mesh = fd.PeriodicBoxMesh(
        *shape,
        *lengths,
        hexahedral=True,
        reorder=False,
    )
    space = fd.FunctionSpace(mesh, "Q", 1)
    x, y, z = fd.SpatialCoordinate(mesh)
    base = (
        fd.sin(x) * fd.cos(y)
        + fd.sin(y) * fd.cos(z)
        + fd.sin(z) * fd.cos(x)
    )
    middle = (
        fd.sin(2.0 * x + 0.35) * fd.cos(y)
        + fd.sin(2.0 * y - 0.45) * fd.cos(z)
        + fd.sin(2.0 * z + 0.65) * fd.cos(x)
    )
    fine = (
        fd.cos(3.0 * x - 0.2)
        * fd.sin(2.0 * y + 0.4)
        * fd.cos(z + 0.3)
    )
    initial_expression = base + 0.48 * middle + 0.30 * fine
    orders = tuple(float(order) for order in args.orders)
    model_names = tuple(f"s={order:.2f}" for order in orders)
    states = {
        name: fd.Function(space, name=f"periodic_fractional_heat_{name}").interpolate(
            initial_expression
        )
        for name in model_names
    }
    operators = {
        name: PeriodicFractionalLaplacian(states[name], order)
        for name, order in zip(model_names, orders, strict=True)
    }
    grid = operators[model_names[0]].operator_data["grid"]

    maximum_wave_number_squared = 0.0
    for count, length in zip(grid.shape, grid.lengths, strict=True):
        wave_numbers = 2.0 * np.pi * np.fft.fftfreq(count, d=length / count)
        maximum_wave_number_squared += float(np.max(wave_numbers**2))
    stability_number = args.dt * args.diffusivity * max(
        maximum_wave_number_squared**order for order in orders
    )
    if stability_number > 1.0:
        maximum_dt = 1.0 / (
            args.diffusivity
            * max(maximum_wave_number_squared**order for order in orders)
        )
        raise ValueError(
            "explicit gallery timestep is too large; choose --dt no larger than "
            f"{maximum_dt:.5g}"
        )

    number_of_steps = round(args.final_time / args.dt)
    if number_of_steps < 1:
        raise ValueError("--final-time must span at least one timestep")
    saved_steps = set(
        np.linspace(
            0,
            number_of_steps,
            min(args.frames, number_of_steps + 1),
            dtype=int,
        ).tolist()
    )
    times = [0.0]
    histories = {
        name: [_logical_values(state, grid)] for name, state in states.items()
    }
    for step in range(1, number_of_steps + 1):
        for name, state in states.items():
            action = fd.assemble(operators[name])
            state.dat.data[:] -= args.dt * args.diffusivity * action.dat.data_ro
        if step in saved_steps:
            times.append(step * args.dt)
            for name, state in states.items():
                histories[name].append(_logical_values(state, grid))

    axes = [
        np.arange(count) * length / count
        for count, length in zip(grid.shape, grid.lengths, strict=True)
    ]
    logical_coordinates = np.column_stack(
        [coordinate.reshape(-1) for coordinate in np.meshgrid(*axes, indexing="ij")]
    )
    history_arrays = {
        name: np.asarray(history) for name, history in histories.items()
    }
    metadata = {
        "demo": "periodic-fractional-gyroid",
        "equation": "partial_t u + kappa (-Delta_periodic)^s u = 0",
        "shape": grid.shape,
        "lengths": grid.lengths,
        "orders": orders,
        "models": model_names,
        "diffusivity": args.diffusivity,
        "dt": args.dt,
        "final_time": number_of_steps * args.dt,
        "stability_number": stability_number,
        "fps": args.fps,
    }
    data_path = destination / "periodic-fractional-gyroid-data.csv.gz"
    rms_series = {
        f"{name}:rms": (
            np.asarray(times),
            np.sqrt(np.mean(history**2, axis=(1, 2, 3))),
        )
        for name, history in history_arrays.items()
    }
    save_plot_csv(
        data_path,
        times=times,
        coordinates=logical_coordinates,
        cells=np.empty((0, 3), dtype=np.int64),
        fields={
            name: history.reshape((len(times), -1))
            for name, history in history_arrays.items()
        },
        metadata=metadata,
        series=rms_series,
    )
    return (
        GyroidData(
            times=np.asarray(times),
            coordinates=logical_coordinates,
            fields=history_arrays,
            shape=grid.shape,
            lengths=grid.lengths,
            orders=orders,
            metadata=metadata,
            diagnostics={
                name: operator.diagnostics()
                for name, operator in operators.items()
            },
        ),
        data_path,
    )


def _load(path: Path) -> GyroidData:
    saved = load_plot_csv(path)
    if saved.metadata.get("demo") != "periodic-fractional-gyroid":
        raise ValueError(f"{path} is not periodic-fractional-gyroid data")
    shape = tuple(int(value) for value in saved.metadata["shape"])
    lengths = tuple(float(value) for value in saved.metadata["lengths"])
    orders = tuple(float(value) for value in saved.metadata["orders"])
    model_names = tuple(str(value) for value in saved.metadata["models"])
    if len(shape) != 3 or len(lengths) != 3 or len(orders) != 2:
        raise ValueError(f"{path} does not contain a three-dimensional grid")
    return GyroidData(
        times=saved.times,
        coordinates=saved.coordinates,
        fields={
            name: saved.fields[name].reshape((-1, *shape))
            for name in model_names
        },
        shape=shape,
        lengths=lengths,
        orders=orders,
        metadata=saved.metadata,
        diagnostics={},
    )


def _surface_colors(
    triangles: np.ndarray,
    length: float,
    color_map: str,
) -> np.ndarray:
    heights = np.mean(triangles[:, :, 2], axis=1) / length
    return plt.get_cmap(color_map)(0.08 + 0.84 * heights)


def _draw_periodic_box(axis: Any, lengths: tuple[float, float, float]) -> None:
    corners = np.asarray(list(np.ndindex(2, 2, 2)), dtype=float) * lengths
    for start in range(8):
        for direction in range(3):
            end = start ^ (1 << (2 - direction))
            if start < end:
                axis.plot(
                    *zip(corners[start], corners[end], strict=True),
                    color=MUTED,
                    linewidth=0.8,
                    alpha=0.38,
                )


def _render(
    args: argparse.Namespace,
    data: GyroidData,
    destination: Path,
    data_path: Path,
) -> Path:
    configure_matplotlib(plt)
    diffusivity = float(data.metadata["diffusivity"])
    model_names = tuple(data.fields)
    initial_limit = max(
        float(np.max(np.abs(fields[0]))) for fields in data.fields.values()
    )

    figure = plt.figure(figsize=(16.0, 10.0), facecolor=PAPER)
    figure.suptitle(
        "A fractional race through a periodic gyroid",
        x=0.045,
        y=0.960,
        ha="left",
        color=INK,
        fontsize=28,
        fontweight="bold",
    )
    figure.text(
        0.046,
        0.875,
        rf"$\partial_tu+{diffusivity:.2f}(-\Delta_{{\rm per}})^s u=0$"
        "   •   same initial field   •   opposite faces are identified",
        color=INK,
        fontsize=21,
    )
    grid = figure.add_gridspec(
        2,
        2,
        left=0.035,
        right=0.965,
        bottom=0.065,
        top=0.815,
        height_ratios=(1.0, 0.36),
        hspace=0.08,
        wspace=0.055,
    )
    surface_axes = []
    surfaces = []
    slice_axes = []
    slice_images = []
    color_maps = ("viridis", "magma")
    title_colors = ("#168c8c", "#d04f55")
    descriptions = ("more nonlocal · fine modes linger", "near-local · fine modes fade")
    middle_z = data.shape[2] // 2
    extent = (0.0, data.lengths[0], 0.0, data.lengths[1])
    for column, (name, order, color_map, title_color, description) in enumerate(
        zip(
            model_names,
            data.orders,
            color_maps,
            title_colors,
            descriptions,
            strict=True,
        )
    ):
        surface_axis = figure.add_subplot(grid[0, column], projection="3d")
        surface_axis.set_facecolor(WHITE)
        surface_axis.set_box_aspect((1.0, 1.0, 1.0), zoom=1.02)
        surface_axis.set_xlim(0.0, data.lengths[0])
        surface_axis.set_ylim(0.0, data.lengths[1])
        surface_axis.set_zlim(0.0, data.lengths[2])
        surface_axis.set_axis_off()
        surface_axis.set_title(
            rf"$s={order:.2f}$",
            loc="left",
            color=title_color,
            fontsize=27,
            pad=-4,
        )
        surface_axis.text2D(
            0.0,
            0.92,
            description,
            transform=surface_axis.transAxes,
            color=MUTED,
            fontsize=17,
        )
        _draw_periodic_box(surface_axis, data.lengths)
        triangles = _isosurface(data.fields[name][0], data.lengths)
        surface = Poly3DCollection(
            triangles,
            facecolors=_surface_colors(
                triangles,
                data.lengths[2],
                color_map,
            ),
            edgecolors="none",
            alpha=0.95,
        )
        surface_axis.add_collection3d(surface)

        slice_axis = figure.add_subplot(grid[1, column])
        sliced = data.fields[name][0][:, :, middle_z].T
        slice_image = slice_axis.imshow(
            sliced,
            origin="lower",
            cmap="RdBu_r",
            vmin=-initial_limit,
            vmax=initial_limit,
            interpolation="bilinear",
            extent=extent,
        )
        slice_axis.contour(
            sliced,
            levels=(0.0,),
            colors=(INK,),
            linewidths=1.15,
            origin="lower",
            extent=extent,
        )
        slice_axis.set_title(r"midplane $z=\pi$", loc="left", fontsize=17)
        slice_axis.set_xticks(())
        slice_axis.set_yticks(())
        slice_axis.set_aspect("equal")
        surface_axes.append(surface_axis)
        surfaces.append(surface)
        slice_axes.append(slice_axis)
        slice_images.append(slice_image)

    figure.text(
        0.5,
        0.037,
        "synchronized camera and colour scale · black contour is the zero level",
        ha="center",
        color=MUTED,
        fontsize=16,
    )
    signature(figure)
    time_label = time_counter(figure)

    def update(frame: int) -> tuple[Any, ...]:
        fraction = frame / max(1, len(data.times) - 1)
        elevation = 23.0 + 3.0 * np.sin(2.0 * np.pi * fraction)
        azimuth = -52.0 + 62.0 * fraction
        for name, color_map, surface_axis, surface, axis, image in zip(
            model_names,
            color_maps,
            surface_axes,
            surfaces,
            slice_axes,
            slice_images,
            strict=True,
        ):
            values = data.fields[name][frame]
            current_triangles = _isosurface(values, data.lengths)
            surface.set_verts(current_triangles)
            surface.set_facecolor(
                _surface_colors(
                    current_triangles,
                    data.lengths[2],
                    color_map,
                )
            )
            surface_axis.view_init(elev=elevation, azim=azimuth)
            sliced = values[:, :, middle_z].T
            image.set_data(sliced)
            for collection in tuple(axis.collections):
                collection.remove()
            axis.contour(
                sliced,
                levels=(0.0,),
                colors=(INK,),
                linewidths=1.0,
                origin="lower",
                extent=extent,
            )
        time_label.set_text(f"t = {data.times[frame]:.2f}")
        return (*surfaces, *slice_images, time_label)

    movie = animation.FuncAnimation(
        figure,
        update,
        frames=len(data.times),
        interval=1000 / args.fps,
        blit=False,
    )
    output_path = destination / "periodic-fractional-gyroid.gif"
    movie.save(
        output_path,
        writer=animation.PillowWriter(fps=args.fps),
        dpi=900.0 / figure.get_size_inches()[0],
    )
    plt.close(figure)
    print(output_path)
    print(data_path)
    print(f"grid: {data.shape[0]} x {data.shape[1]} x {data.shape[2]}")
    for name, fields in data.fields.items():
        print(f"{name} range: [{fields[-1].min():.6f}, {fields[-1].max():.6f}]")
    if data.diagnostics:
        for name, diagnostics in data.diagnostics.items():
            print(f"{name} operator: {diagnostics}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parents[1] / "demo-output"),
    )
    parser.add_argument("--plot-data", type=Path)
    parser.add_argument("--cells", type=int, default=24)
    parser.add_argument(
        "--orders",
        type=float,
        nargs=2,
        metavar=("LOW", "HIGH"),
        default=(0.42, 0.88),
    )
    parser.add_argument("--diffusivity", type=float, default=0.18)
    parser.add_argument("--dt", type=float, default=0.015)
    parser.add_argument("--final-time", type=float, default=2.4)
    parser.add_argument("--frames", type=int, default=36)
    parser.add_argument("--fps", type=int, default=9)
    args = parser.parse_args()

    destination = output_directory(args.output)
    if args.plot_data is None:
        data, data_path = _simulate(args, destination)
    else:
        data_path = args.plot_data
        data = _load(data_path)
    _render(args, data, destination, data_path)


if __name__ == "__main__":
    main()
