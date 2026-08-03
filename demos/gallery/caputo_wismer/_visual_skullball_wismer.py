"""Stylized concentric-head ("skullball") Caputo-Wismer animations."""

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
from _visual_mpi import gather_p1_animation_data  # noqa: E402
from _visual_style import (  # noqa: E402
    BLUE,
    CORAL,
    INK,
    PAPER,
    configure_matplotlib,
    material_parameter_table,
    output_directory,
    signature,
    time_counter,
)
from matplotlib import animation, colors, ticker, tri  # noqa: E402
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

MATERIAL_ORDERS = {
    "bath": 0.10,
    "skin": 0.16,
    "skull": 0.41,
    "brain": 0.58,
}
SOURCE_SEED = 20260724


def ellipse(
    x_radius: float,
    y_radius: float,
    *,
    y_center: float,
) -> np.ndarray:
    """Return a smooth closed ellipse for the interface overlays."""
    angles = np.linspace(0.0, 2.0 * np.pi, 361)
    return np.column_stack(
        (
            x_radius * np.cos(angles),
            y_center + y_radius * np.sin(angles),
        )
    )


def anatomical_outlines() -> tuple[np.ndarray, ...]:
    """Return the stylized skin, skull, and brain interfaces."""
    return (
        ellipse(0.98, 1.06, y_center=-0.02),
        ellipse(0.88, 0.96, y_center=-0.01),
        ellipse(0.73, 0.82, y_center=0.01),
    )


def source_parameters() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return reproducible irregular source locations, frequencies, and phases."""
    generator = np.random.default_rng(SOURCE_SEED)
    angles = generator.uniform(0.0, 2.0 * np.pi, 3)
    radii = generator.uniform(1.11, 1.22, 3)
    locations = radii[:, None] * np.column_stack(
        (np.cos(angles), np.sin(angles))
    )
    frequencies = generator.uniform(9.7, 12.1, 3)
    phases = generator.uniform(0.0, 2.0 * np.pi, 3)
    return locations, frequencies, phases


def main(scenario: str) -> None:
    """Solve and render one skullball experiment."""
    if scenario not in {"pulse", "sources"}:
        raise ValueError("scenario must be 'pulse' or 'sources'")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parents[2] / "demo-output"),
    )
    parser.add_argument("--refinement", type=int, default=4)
    parser.add_argument("--modes", type=int, default=28)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--final-time", type=float, default=4.8)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument(
        "--degree",
        type=int,
        choices=(1, 2),
        default=2,
        help="CG degree for the solve; output is sampled at mesh vertices",
    )
    args = parser.parse_args()

    import firedrake as fd

    from yonderdrake import CaputoDerivative, Diethelm2008, FractionalTimeStepper

    configure_matplotlib(plt)
    communicator = fd.COMM_WORLD
    destination = output_directory(args.output)
    mesh = fd.UnitDiskMesh(refinement_level=args.refinement)
    mesh.coordinates.dat.data[:] *= 1.30
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

    head_level = (x / 0.98) ** 2 + ((y + 0.02) / 1.06) ** 2
    skull_level = (x / 0.88) ** 2 + ((y + 0.01) / 0.96) ** 2
    brain_level = (x / 0.73) ** 2 + ((y - 0.01) / 0.82) ** 2

    bath_speed = 0.96
    skin_speed = 0.94
    skull_speed = 1.95
    brain_speed = 1.00
    bath_damping = 0.003
    skin_damping = 0.018
    skull_damping = 0.110
    brain_damping = 0.014
    brain_mask = fd.conditional(brain_level <= 1.0, 1.0, 0.0)
    skull_mask = fd.conditional(
        brain_level <= 1.0,
        0.0,
        fd.conditional(skull_level <= 1.0, 1.0, 0.0),
    )
    skin_mask = fd.conditional(
        skull_level <= 1.0,
        0.0,
        fd.conditional(head_level <= 1.0, 1.0, 0.0),
    )
    bath_mask = fd.conditional(head_level <= 1.0, 0.0, 1.0)
    layered_speed_squared = (
        bath_speed**2 * bath_mask
        + skin_speed**2 * skin_mask
        + skull_speed**2 * skull_mask
        + brain_speed**2 * brain_mask
    )
    layered_damping = (
        bath_damping * bath_mask
        + skin_damping * skin_mask
        + skull_damping * skull_mask
        + brain_damping * brain_mask
    )
    layered_order = (
        MATERIAL_ORDERS["bath"] * bath_mask
        + MATERIAL_ORDERS["skin"] * skin_mask
        + MATERIAL_ORDERS["skull"] * skull_mask
        + MATERIAL_ORDERS["brain"] * brain_mask
    )
    volume = float(fd.assemble(fd.Constant(1.0) * fd.dx(domain=mesh)))
    average_speed_squared = float(
        fd.assemble(layered_speed_squared * fd.dx) / volume
    )
    average_damping = float(
        fd.assemble(layered_damping * fd.dx) / volume
    )
    average_order = float(fd.assemble(layered_order * fd.dx) / volume)
    homogeneous_speed_squared = fd.Constant(average_speed_squared)
    homogeneous_damping = fd.Constant(average_damping)
    damping_by_material = {
        "bath": bath_damping * bath_mask,
        "skin": skin_damping * skin_mask,
        "skull": skull_damping * skull_mask,
        "brain": brain_damping * brain_mask,
    }

    pulse_radius_squared = (x + 0.34) ** 2 + (y - 0.08) ** 2
    pulse_expression = (
        1.30 * fd.exp(-72.0 * pulse_radius_squared)
        - 0.52 * fd.exp(-18.0 * pulse_radius_squared)
    )
    forcing_time = fd.Constant(0.0)
    source_expression = 0.0
    locations, source_frequencies, source_phases = source_parameters()
    for (source_x, source_y), frequency, phase in zip(
        locations,
        source_frequencies,
        source_phases,
        strict=True,
    ):
        radius = float(np.hypot(source_x, source_y))
        inward_x = -source_x / radius
        inward_y = -source_y / radius
        offset_x = x - float(source_x)
        offset_y = y - float(source_y)
        axial = inward_x * offset_x + inward_y * offset_y
        transverse = -inward_y * offset_x + inward_x * offset_y
        beam = (axial / 0.075) * fd.exp(
            -(axial / 0.095) ** 2 - (transverse / 0.115) ** 2
        )
        source_expression += (
            0.18
            * beam
            * fd.sin(float(frequency) * forcing_time + float(phase))
            * (1.0 - fd.exp(-3.5 * forcing_time))
        )
    if scenario == "pulse":
        source_expression = fd.Constant(0.0)

    test = fd.TestFunction(space)
    time = fd.Constant(0.0)
    step_size = fd.Constant(args.dt)
    model_specs = (
        ("Layered tissues", layered_speed_squared),
        ("Homogenized", homogeneous_speed_squared),
    )
    fields: dict[str, Any] = {}
    previous: dict[str, Any] = {}
    older: dict[str, Any] = {}
    steppers: dict[str, Any] = {}
    solver_parameters = (
        {
            "snes_type": "ksponly",
            "ksp_type": "cg",
            "ksp_rtol": 1.0e-9,
            "pc_type": "gamg",
        }
        if communicator.size > 1
        else {
            "snes_type": "ksponly",
            "ksp_type": "preonly",
            "pc_type": "lu",
        }
    )
    for material_name, speed_squared in model_specs:
        field = fd.Function(space, name=material_name.replace(" ", "_"))
        if scenario == "pulse":
            field.interpolate(pulse_expression)
        else:
            field.assign(0.0)
        fields[material_name] = field
        previous[material_name] = field.copy(deepcopy=True)
        older[material_name] = field.copy(deepcopy=True)
        acceleration = (
            field - 2.0 * previous[material_name] + older[material_name]
        ) / step_size**2
        velocity = (field - previous[material_name]) / step_size
        if material_name == "Layered tissues":
            fractional_form = sum(
                fd.inner(
                    damping_by_material[region]
                    * fd.grad(
                        CaputoDerivative(
                            field,
                            MATERIAL_ORDERS[region],
                        )
                    ),
                    fd.grad(test),
                )
                * fd.dx
                for region in ("bath", "skin", "skull", "brain")
            )
        else:
            fractional_form = (
                fd.inner(
                    homogeneous_damping
                    * fd.grad(CaputoDerivative(field, average_order)),
                    fd.grad(test),
                )
                * fd.dx
            )
        residual = (
            fd.inner(acceleration, test) * fd.dx
            + fd.inner(speed_squared * fd.grad(field), fd.grad(test)) * fd.dx
            + fractional_form
            + bath_speed * fd.inner(velocity, test) * fd.ds
            - fd.inner(source_expression, test) * fd.dx
        )
        steppers[material_name] = FractionalTimeStepper(
            residual,
            Diethelm2008(args.modes),
            time,
            step_size,
            field,
            solver_parameters=solver_parameters,
        )

    times = [0.0]
    histories = {
        key: [p1_snapshot(field)]
        for key, field in fields.items()
    }
    number_of_steps = round(args.final_time / args.dt)
    for index in range(1, number_of_steps + 1):
        next_time = index * args.dt
        forcing_time.assign(next_time)
        for key in fields:
            steppers[key].advance()
        for key, field in fields.items():
            older[key].assign(previous[key])
            previous[key].assign(field)
            histories[key].append(p1_snapshot(field))
        time.assign(next_time)
        times.append(next_time)

    history_arrays = {
        key: np.asarray(values) for key, values in histories.items()
    }
    history_arrays["Absolute difference"] = np.abs(
        history_arrays["Layered tissues"] - history_arrays["Homogenized"]
    )
    gathered = gather_p1_animation_data(
        mesh,
        history_arrays,
        communicator,
    )
    if communicator.rank != 0:
        communicator.barrier()
        return
    assert gathered is not None
    coordinates, cells, history_arrays = gathered
    triangulation = tri.Triangulation(
        coordinates[:, 0],
        coordinates[:, 1],
        cells,
    )
    pressure_values = np.concatenate(
        (
            history_arrays["Layered tissues"].reshape(-1),
            history_arrays["Homogenized"].reshape(-1),
        )
    )
    pressure_limit = max(float(np.max(np.abs(pressure_values))), 1.0e-8)
    pressure_norm = colors.SymLogNorm(
        linthresh=0.025 * pressure_limit,
        vmin=-pressure_limit,
        vmax=pressure_limit,
        base=10,
    )
    difference_limit = max(
        float(np.max(history_arrays["Absolute difference"])),
        1.0e-10,
    )
    difference_norm = colors.Normalize(vmin=0.0, vmax=difference_limit)
    output_name = (
        "caputo-wismer-skullball-pulse.gif"
        if scenario == "pulse"
        else "caputo-wismer-skullball-sources.gif"
    )
    output_path = destination / output_name
    data_path = destination / output_name.replace(".gif", "-data.csv.gz")
    coordinate_head_level = (
        (coordinates[:, 0] / 0.98) ** 2
        + ((coordinates[:, 1] + 0.02) / 1.06) ** 2
    )
    coordinate_skull_level = (
        (coordinates[:, 0] / 0.88) ** 2
        + ((coordinates[:, 1] + 0.01) / 0.96) ** 2
    )
    coordinate_brain_level = (
        (coordinates[:, 0] / 0.73) ** 2
        + ((coordinates[:, 1] - 0.01) / 0.82) ** 2
    )
    vertex_labels = np.zeros(coordinates.shape[0], dtype=np.uint8)
    vertex_labels[coordinate_head_level <= 1.0] = 1
    vertex_labels[coordinate_skull_level <= 1.0] = 2
    vertex_labels[coordinate_brain_level <= 1.0] = 3
    outlines = anatomical_outlines()
    outline_coordinates = np.concatenate(outlines)
    outline_labels = np.concatenate(
        tuple(
            np.full(outline.shape[0], index, dtype=np.uint8)
            for index, outline in enumerate(outlines, start=1)
        )
    )
    source_rows = (
        np.column_stack(
            (
                locations,
                source_frequencies,
                source_phases,
            )
        )
        if scenario == "sources"
        else None
    )
    save_plot_csv(
        data_path,
        times=times,
        coordinates=coordinates,
        cells=cells,
        fields=history_arrays,
        metadata={
            "demo": "caputo-wismer-skullball",
            "scenario": scenario,
            "equation": (
                "u_tt-div(c^2(x) grad u)-sum_m div("
                "b_m chi_m(x) grad D_C^alpha_m u)=f"
            ),
            "boundary_condition": "c_b^2 partial_n u + c_b u_t = 0",
            "dt": args.dt,
            "final_time": args.final_time,
            "fps": args.fps,
            "element_degree": args.degree,
            "material_labels": {
                "0": "bath",
                "1": "skin",
                "2": "skull",
                "3": "brain",
            },
            "material_orders": MATERIAL_ORDERS,
            "material_damping": {
                "bath": bath_damping,
                "skin": skin_damping,
                "skull": skull_damping,
                "brain": brain_damping,
            },
            "average_speed_squared": average_speed_squared,
            "average_damping": average_damping,
            "average_order": average_order,
            "pressure_limit": pressure_limit,
            "pressure_linthresh": pressure_norm.linthresh,
            "difference_limit": difference_limit,
            "source_seed": SOURCE_SEED,
        },
        vertex_labels=vertex_labels,
        anatomy=(outline_coordinates, outline_labels),
        sources=source_rows,
    )

    figure = plt.figure(figsize=(16.0, 12.2), facecolor=PAPER)
    grid = figure.add_gridspec(
        1,
        3,
        left=0.035,
        right=0.965,
        bottom=0.190,
        top=0.535,
        wspace=0.055,
    )
    title_text = (
        "One pulse crosses the skullball"
        if scenario == "pulse"
        else "Three sources converge on the skullball"
    )
    figure.suptitle(
        title_text,
        x=0.045,
        y=0.965,
        ha="left",
        color=INK,
        fontsize=28,
        fontweight="bold",
    )
    figure.text(
        0.046,
        0.897,
        (
            rf"$\Delta t={args.dt:g}$  •  "
            f"{coordinates.shape[0]:,} vertices"
        ),
        color=INK,
        fontsize=24,
    )
    figure.text(
        0.046,
        0.835,
        (
            r"$u_{tt}-\nabla\!\cdot(c^2(\mathbf{x})\nabla u)"
            r"-\sum_m\nabla\!\cdot\!\left(b_m\chi_m(\mathbf{x})"
            r"\nabla D_C^{\alpha_m}u\right)=f(\mathbf{x},t),"
            r"\qquad c_b^2\partial_nu+c_bu_t=0"
            r"\ \mathrm{on}\ \partial\Omega$"
        ),
        color=INK,
        fontsize=25,
    )
    condition_text = (
        r"$u(\mathbf{x},0)=1.30e^{-72r^2}-0.52e^{-18r^2},"
        r"\quad u_t(\mathbf{x},0)=0,\quad f=0$"
        if scenario == "pulse"
        else (
            r"$u(\mathbf{x},0)=u_t(\mathbf{x},0)=0;"
            r"\quad f=\sum_{j=1}^3$ seeded asynchronous inward dipoles"
            rf"  $(\mathrm{{seed}}={SOURCE_SEED})$"
        )
    )
    figure.text(0.046, 0.775, condition_text, color=INK, fontsize=21.6)
    material_parameter_table(
        figure,
        (
            "bath",
            "skin",
            "skull",
            "brain",
            "homogenized",
        ),
        (
            f"{bath_damping:.3f}",
            f"{skin_damping:.3f}",
            f"{skull_damping:.3f}",
            f"{brain_damping:.3f}",
            f"{average_damping:.3f}",
        ),
        (
            f"{MATERIAL_ORDERS['bath']:.2f}",
            f"{MATERIAL_ORDERS['skin']:.2f}",
            f"{MATERIAL_ORDERS['skull']:.2f}",
            f"{MATERIAL_ORDERS['brain']:.2f}",
            f"{average_order:.3f}",
        ),
        bbox=(0.046, 0.625, 0.90, 0.095),
    )

    model_names = (
        "Layered tissues",
        "Homogenized",
        "Absolute difference",
    )
    model_titles = ("Layered", "Homogenized", "Difference")
    model_norms = (pressure_norm, pressure_norm, difference_norm)
    outline_colors = (CORAL, INK, BLUE)
    outline_widths = (1.55, 2.20, 1.55)
    heatmaps = []
    for column, (name, panel_title, norm) in enumerate(
        zip(model_names, model_titles, model_norms, strict=True)
    ):
        axis = figure.add_subplot(grid[0, column])
        image = axis.tripcolor(
            triangulation,
            history_arrays[name][0],
            shading="gouraud",
            cmap="viridis",
            norm=norm,
        )
        for outline, outline_color, linewidth in zip(
            outlines,
            outline_colors,
            outline_widths,
            strict=True,
        ):
            axis.plot(
                outline[:, 0],
                outline[:, 1],
                color=outline_color,
                linewidth=linewidth,
                alpha=0.96,
            )
        if scenario == "sources":
            for source_x, source_y in locations:
                axis.plot(
                    source_x,
                    source_y,
                    marker="o",
                    markersize=5,
                    markerfacecolor="#fff3a6",
                    markeredgecolor=INK,
                    markeredgewidth=0.7,
                )
                axis.annotate(
                    "",
                    xy=(0.25 * source_x, 0.25 * source_y),
                    xytext=(source_x, source_y),
                    arrowprops={
                        "arrowstyle": "-|>",
                        "color": INK,
                        "lw": 0.9,
                        "alpha": 0.72,
                    },
                )
        else:
            axis.plot(
                -0.34,
                0.08,
                marker="*",
                markersize=7,
                color="#fff3a6",
                markeredgecolor=INK,
                markeredgewidth=0.6,
            )
        axis.set_xlim(-1.34, 1.34)
        axis.set_ylim(-1.34, 1.34)
        axis.set_aspect("equal")
        axis.set_axis_off()
        axis.set_title(
            panel_title,
            loc="left",
            color=INK,
            fontsize=26,
            pad=4,
        )
        heatmaps.append(image)

    interface_handles = tuple(
        Line2D((0.0, 1.0), (0.0, 0.0), color=color, linewidth=3.1)
        for color in outline_colors
    )
    figure.legend(
        interface_handles,
        ("skin interface", "skull interface", "brain interface"),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.012),
        ncol=3,
        frameon=False,
        fontsize=20,
    )
    pressure_colorbar_axis = figure.add_axes((0.12, 0.112, 0.49, 0.014))
    pressure_colorbar = figure.colorbar(
        heatmaps[0],
        cax=pressure_colorbar_axis,
        orientation="horizontal",
        label="acoustic pressure",
    )
    pressure_colorbar.set_ticks(
        (
            -pressure_limit,
            -0.1 * pressure_limit,
            0.0,
            0.1 * pressure_limit,
            pressure_limit,
        )
    )
    pressure_colorbar.formatter = ticker.FuncFormatter(
        lambda value, _position: f"{value:.2g}"
    )
    pressure_colorbar.update_ticks()
    pressure_colorbar.ax.xaxis.set_label_position("top")
    difference_colorbar_axis = figure.add_axes((0.70, 0.112, 0.22, 0.014))
    difference_colorbar = figure.colorbar(
        heatmaps[2],
        cax=difference_colorbar_axis,
        orientation="horizontal",
        label="absolute difference",
    )
    difference_colorbar.set_ticks(
        (0.0, 0.5 * difference_limit, difference_limit)
    )
    difference_formatter = ticker.ScalarFormatter(useMathText=True)
    difference_formatter.set_powerlimits((-3, 3))
    difference_formatter.set_useOffset(False)
    difference_colorbar.formatter = difference_formatter
    difference_colorbar.update_ticks()
    difference_colorbar.ax.xaxis.set_label_position("top")
    signature(figure)
    time_label = time_counter(figure)
    times_array = np.asarray(times)

    def update(frame: int) -> tuple[Any, ...]:
        for image, key, norm in zip(
            heatmaps,
            model_names,
            model_norms,
            strict=True,
        ):
            image.set_array(history_arrays[key][frame])
            image.set_clim(norm.vmin, norm.vmax)
        time_label.set_text(f"t = {times_array[frame]:.3f}")
        return (*heatmaps, time_label)

    movie = animation.FuncAnimation(
        figure,
        update,
        frames=len(times),
        interval=1000 / args.fps,
        blit=False,
    )
    movie.save(
        output_path,
        writer=animation.PillowWriter(fps=args.fps),
        dpi=120,
    )
    plt.close(figure)

    print(output_path)
    print(data_path)
    print(
        f"mesh: {coordinates.shape[0]} vertices, {cells.shape[0]} triangles"
    )
    print(
        f"homogenized: c²={average_speed_squared:.6f}, "
        f"b={average_damping:.6f}, alpha={average_order:.6f}"
    )
    for key, stepper in steppers.items():
        print(key, stepper.solver_stats())
    communicator.barrier()
