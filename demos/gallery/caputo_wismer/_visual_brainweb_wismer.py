"""BrainWeb Caputo-Wismer animations on a nonconforming demo mesh."""

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
    from ._brainweb import (  # type: ignore[import-not-found]
        BRAINWEB_PAGE,
        BRAINWEB_SLICE_INDEX,
        MM_PER_MODEL_UNIT,
        load_brainweb_slice,
        sample_labels,
        slice_coordinates,
    )
    from ._gallery_path import add_gallery_path  # type: ignore[import-not-found]
except ImportError:
    from _brainweb import (  # type: ignore[no-redef]
        BRAINWEB_PAGE,
        BRAINWEB_SLICE_INDEX,
        MM_PER_MODEL_UNIT,
        load_brainweb_slice,
        sample_labels,
        slice_coordinates,
    )
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
from scipy import ndimage  # noqa: E402

SOURCE_SEED = 20260724
X_LIMITS = (-1.12, 1.12)
Y_LIMITS = (-1.50, 1.15)

# Illustrative nondimensional parameters; these are not BrainWeb data.
REGION_PARAMETERS = {
    "bath": {
        "labels": (0,),
        "density": 1.00,
        "speed": 0.96,
        "damping": 0.003,
        "order": 0.10,
    },
    "scalp": {
        "labels": (4, 5, 6, 9),
        "density": 1.05,
        "speed": 0.94,
        "damping": 0.018,
        "order": 0.16,
    },
    "skull": {
        "labels": (7,),
        "density": 1.85,
        "speed": 1.95,
        "damping": 0.110,
        "order": 0.41,
    },
    "CSF": {
        "labels": (1,),
        "density": 1.00,
        "speed": 1.00,
        "damping": 0.004,
        "order": 0.12,
    },
    "grey": {
        "labels": (2, 8),
        "density": 1.04,
        "speed": 1.00,
        "damping": 0.015,
        "order": 0.58,
    },
    "white": {
        "labels": (3,),
        "density": 1.04,
        "speed": 1.03,
        "damping": 0.017,
        "order": 0.62,
    },
}


def source_parameters(
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return reproducible irregular source positions aimed at the brain."""
    generator = np.random.default_rng(SOURCE_SEED)
    background = labels == 0
    distance_from_head = ndimage.distance_transform_edt(background)
    candidates = np.argwhere(
        background & (distance_from_head >= 7.0) & (distance_from_head <= 15.0)
    )
    brain_pixels = np.argwhere(np.isin(labels, (1, 2, 3, 8)))
    target_yx = brain_pixels.mean(axis=0)
    shuffled = candidates[generator.permutation(candidates.shape[0])]
    selected: list[np.ndarray] = []
    selected_angles: list[float] = []
    for candidate in shuffled:
        displacement = candidate - target_yx
        angle = float(np.arctan2(displacement[0], displacement[1]))
        if all(
            abs(np.angle(np.exp(1j * (angle - other)))) > 0.72
            for other in selected_angles
        ):
            selected.append(candidate)
            selected_angles.append(angle)
        if len(selected) == 3:
            break
    if len(selected) != 3:
        raise RuntimeError("Could not place three separated BrainWeb sources")
    selected_yx = np.asarray(selected)
    locations = np.column_stack(
        (
            (selected_yx[:, 1] - 90.0) / MM_PER_MODEL_UNIT,
            (selected_yx[:, 0] - 126.0) / MM_PER_MODEL_UNIT,
        )
    )
    target = np.array(
        (
            (target_yx[1] - 90.0) / MM_PER_MODEL_UNIT,
            (target_yx[0] - 126.0) / MM_PER_MODEL_UNIT,
        )
    )
    frequencies = generator.uniform(9.7, 12.1, 3)
    phases = generator.uniform(0.0, 2.0 * np.pi, 3)
    return locations, target, frequencies, phases


def _material_masks(fd: Any, mesh: Any, labels: np.ndarray) -> dict[str, Any]:
    """Create one cellwise indicator function per anatomical material."""
    material_space = fd.FunctionSpace(mesh, "DG", 0)
    coordinate_space = fd.VectorFunctionSpace(mesh, "DG", 0)
    cell_centres = fd.Function(coordinate_space).interpolate(fd.SpatialCoordinate(mesh))
    dof_coordinates = np.asarray(
        cell_centres.dat.data_ro,
        dtype=float,
    ).reshape(-1, 2)
    cell_labels = sample_labels(labels, dof_coordinates)
    masks: dict[str, Any] = {}
    for region, parameters in REGION_PARAMETERS.items():
        mask = fd.Function(material_space, name=f"{region}_mask")
        mask.dat.data[:] = np.isin(
            cell_labels,
            parameters["labels"],
        ).astype(float)
        masks[region] = mask
    coverage = sum(np.asarray(mask.dat.data_ro) for mask in masks.values())
    if not np.allclose(coverage, 1.0):
        raise RuntimeError("BrainWeb material masks do not cover the mesh")
    return masks


def _plot_anatomical_interfaces(
    axis: Any,
    labels: np.ndarray,
    x_voxels: np.ndarray,
    y_voxels: np.ndarray,
) -> None:
    """Overlay detailed, non-colormap anatomical interface curves."""
    interfaces = (
        (labels > 0, CORAL, 1.65),
        (labels == 7, INK, 2.15),
        (np.isin(labels, (2, 3, 8)), BLUE, 1.55),
        (labels == 3, "#e8a43c", 0.85),
    )
    for mask, color, linewidth in interfaces:
        axis.contour(
            x_voxels,
            y_voxels,
            mask.astype(float),
            levels=(0.5,),
            colors=(color,),
            linewidths=(linewidth,),
            alpha=0.96,
        )


def main(scenario: str) -> None:
    """Solve and render a BrainWeb pulse or asynchronous-source experiment."""
    if scenario not in {"pulse", "sources"}:
        raise ValueError("scenario must be 'pulse' or 'sources'")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parents[2] / "demo-output"),
    )
    parser.add_argument(
        "--refinement",
        type=int,
        default=4,
        help="structured-grid refinement; 4 gives 96 by 112 cells",
    )
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
    parser.add_argument(
        "--slice-index",
        type=int,
        default=BRAINWEB_SLICE_INDEX,
        help="BrainWeb axial voxel index (default 80, z=8 mm)",
    )
    args = parser.parse_args()
    if args.refinement < 2:
        parser.error("--refinement must be at least 2")

    import firedrake as fd

    from yonderdrake import Diethelm2008
    from yonderdrake.applications import (
        CaputoWismerImpedanceBoundary,
        CaputoWismerMaterial,
        CaputoWismerStepper,
    )

    configure_matplotlib(plt)
    communicator = fd.COMM_WORLD
    destination = output_directory(args.output)
    labels = None
    if communicator.rank == 0:
        # Download and cache the phantom when absent.
        labels = load_brainweb_slice(
            destination / ".brainweb",
            slice_index=args.slice_index,
        )
    labels = communicator.bcast(labels, root=0)
    x_voxels, y_voxels = slice_coordinates()
    slice_z_mm = -72 + args.slice_index

    refinement_factor = 2 ** (args.refinement - 2)
    nx = 24 * refinement_factor
    ny = 28 * refinement_factor
    mesh = fd.RectangleMesh(
        nx,
        ny,
        X_LIMITS[1] - X_LIMITS[0],
        Y_LIMITS[1] - Y_LIMITS[0],
    )
    mesh.coordinates.dat.data[:, 0] += X_LIMITS[0]
    mesh.coordinates.dat.data[:, 1] += Y_LIMITS[0]
    space = fd.FunctionSpace(mesh, "CG", args.degree)
    # Sample higher-degree output onto animation vertices.
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
    masks = _material_masks(fd, mesh, labels)

    heterogeneous_materials = tuple(
        CaputoWismerMaterial(
            indicator=masks[region],
            density=parameters["density"],
            wave_speed=parameters["speed"],
            damping=parameters["damping"],
            alpha=parameters["order"],
        )
        for region, parameters in REGION_PARAMETERS.items()
    )
    layered_speed_squared = sum(
        parameters["speed"] ** 2 * masks[region]
        for region, parameters in REGION_PARAMETERS.items()
    )
    layered_damping = sum(
        parameters["damping"] * masks[region]
        for region, parameters in REGION_PARAMETERS.items()
    )
    layered_order = sum(
        parameters["order"] * masks[region]
        for region, parameters in REGION_PARAMETERS.items()
    )
    layered_density = sum(
        parameters["density"] * masks[region]
        for region, parameters in REGION_PARAMETERS.items()
    )
    volume = float(fd.assemble(fd.Constant(1.0) * fd.dx(domain=mesh)))
    average_speed_squared = float(fd.assemble(layered_speed_squared * fd.dx) / volume)
    average_damping = float(fd.assemble(layered_damping * fd.dx) / volume)
    average_order = float(fd.assemble(layered_order * fd.dx) / volume)
    average_density = float(fd.assemble(layered_density * fd.dx) / volume)
    homogeneous_materials = (
        CaputoWismerMaterial(
            indicator=fd.Constant(1.0),
            density=average_density,
            wave_speed=np.sqrt(average_speed_squared),
            damping=average_damping,
            alpha=average_order,
        ),
    )

    pulse_x = -30.0 / MM_PER_MODEL_UNIT
    pulse_y = 10.0 / MM_PER_MODEL_UNIT
    pulse_radius_squared = (x - pulse_x) ** 2 + (y - pulse_y) ** 2
    pulse_expression = 1.30 * fd.exp(-72.0 * pulse_radius_squared) - 0.52 * fd.exp(
        -18.0 * pulse_radius_squared
    )
    forcing_time = fd.Constant(0.0)
    source_expression = 0.0
    locations, target, source_frequencies, source_phases = source_parameters(labels)
    for (source_x, source_y), frequency, phase in zip(
        locations,
        source_frequencies,
        source_phases,
        strict=True,
    ):
        direction = target - np.array((source_x, source_y))
        direction /= np.linalg.norm(direction)
        offset_x = x - float(source_x)
        offset_y = y - float(source_y)
        axial = float(direction[0]) * offset_x + float(direction[1]) * offset_y
        transverse = -float(direction[1]) * offset_x + float(direction[0]) * offset_y
        beam = (axial / 0.075) * fd.exp(
            -((axial / 0.095) ** 2) - (transverse / 0.115) ** 2
        )
        source_expression += (
            0.18
            * beam
            * fd.sin(float(frequency) * forcing_time + float(phase))
            * (1.0 - fd.exp(-3.5 * forcing_time))
        )
    if scenario == "pulse":
        source_expression = fd.Constant(0.0)

    time = fd.Constant(0.0)
    step_size = fd.Constant(args.dt)
    model_specs = (
        ("Layered tissues", heterogeneous_materials),
        ("Homogenized", homogeneous_materials),
    )
    fields: dict[str, Any] = {}
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
    boundary = CaputoWismerImpedanceBoundary(
        coefficient=1.0
        / (
            REGION_PARAMETERS["bath"]["density"]
            * REGION_PARAMETERS["bath"]["speed"]
        )
    )
    for material_name, materials in model_specs:
        field = fd.Function(space, name=material_name.replace(" ", "_"))
        if scenario == "pulse":
            field.interpolate(pulse_expression)
        else:
            field.assign(0.0)
        fields[material_name] = field
        steppers[material_name] = CaputoWismerStepper(
            field,
            time,
            step_size,
            materials=materials,
            representation=Diethelm2008(args.modes),
            volume_source=source_expression,
            boundaries=(boundary,),
            solver_parameters=solver_parameters,
        )

    times = [0.0]
    histories = {key: [p1_snapshot(field)] for key, field in fields.items()}
    number_of_steps = round(args.final_time / args.dt)
    for index in range(1, number_of_steps + 1):
        next_time = index * args.dt
        forcing_time.assign(next_time)
        for key in fields:
            steppers[key].advance()
        for key, field in fields.items():
            histories[key].append(p1_snapshot(field))
        time.assign(next_time)
        times.append(next_time)

    history_arrays = {key: np.asarray(values) for key, values in histories.items()}
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
        "caputo-wismer-brainweb-pulse.gif"
        if scenario == "pulse"
        else "caputo-wismer-brainweb-sources.gif"
    )
    output_path = destination / output_name
    data_path = destination / output_name.replace(".gif", "-data.csv.gz")
    anatomy_x, anatomy_y = np.meshgrid(x_voxels, y_voxels)
    anatomy_coordinates = np.column_stack(
        (anatomy_x.reshape(-1), anatomy_y.reshape(-1))
    )
    vertex_labels = sample_labels(labels, coordinates)
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
            "demo": "caputo-wismer-brainweb",
            "scenario": scenario,
            "equation": (
                "sum_m chi_m/(rho_m c_m^2) u_tt"
                "-div(sum_m chi_m/rho_m grad u)"
                "-sum_m div(b_m chi_m/rho_m grad D_C^alpha_m u)=f"
            ),
            "boundary_condition": (
                "partial_n u/rho_b + u_t/(rho_b c_b) = 0"
            ),
            "dt": args.dt,
            "final_time": args.final_time,
            "fps": args.fps,
            "element_degree": args.degree,
            "brainweb_slice_index": args.slice_index,
            "brainweb_slice_z_mm": slice_z_mm,
            "brainweb_source": BRAINWEB_PAGE,
            "brainweb_citation": (
                "Collins et al., IEEE TMI 17(3), 463-468 (1998), doi:10.1109/42.712135"
            ),
            "material_parameters": REGION_PARAMETERS,
            "average_speed_squared": average_speed_squared,
            "average_density": average_density,
            "average_damping": average_damping,
            "average_order": average_order,
            "pressure_limit": pressure_limit,
            "pressure_linthresh": pressure_norm.linthresh,
            "difference_limit": difference_limit,
            "source_seed": SOURCE_SEED,
            "source_target": target.tolist(),
        },
        vertex_labels=vertex_labels,
        anatomy=(anatomy_coordinates, labels.reshape(-1)),
        sources=source_rows,
    )

    figure = plt.figure(figsize=(16.0, 12.5), facecolor=PAPER)
    grid = figure.add_gridspec(
        1,
        3,
        left=0.035,
        right=0.965,
        bottom=0.225,
        top=0.615 if scenario == "sources" else 0.565,
        wspace=0.055,
    )
    title_text = (
        "A pulse crosses a BrainWeb anatomical head"
        if scenario == "pulse"
        else "Three sources cross a BrainWeb anatomical head"
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
        0.900,
        (
            rf"$\Delta t={args.dt:g}$  •  "
            rf"$z={slice_z_mm:g}\,\mathrm{{mm}}$  •  "
            f"{coordinates.shape[0]:,} vertices"
        ),
        color=INK,
        fontsize=23,
    )
    figure.text(
        0.046,
        0.835,
        (
            r"$\sum_m\frac{\chi_m}{\rho_m c_m^2}u_{tt}"
            r"-\nabla\!\cdot\!\sum_m\frac{\chi_m}{\rho_m}\nabla u"
            r"-\nabla\!\cdot\!\sum_m\frac{b_m\chi_m}{\rho_m}"
            r"\nabla D_C^{\alpha_m}u=f(\mathbf{x},t),"
            r"\qquad \rho_b^{-1}\partial_nu+(\rho_bc_b)^{-1}u_t=0"
            r"\ \mathrm{on}\ \partial\Omega$"
        ),
        color=INK,
        fontsize=25,
    )
    if scenario == "pulse":
        figure.text(
            0.046,
            0.775,
            (
                r"$u(\mathbf{x},0)=1.30e^{-72r^2}-0.52e^{-18r^2},"
                r"\quad u_t(\mathbf{x},0)=0,\quad f=0$"
            ),
            color=INK,
            fontsize=21,
        )
    material_parameter_table(
        figure,
        ("bath", "scalp", "skull", "CSF", "grey", "white", "homogenized"),
        tuple(
            f"{REGION_PARAMETERS[region]['damping']:.3f}"
            for region in ("bath", "scalp", "skull", "CSF", "grey", "white")
        )
        + (f"{average_damping:.3f}",),
        tuple(
            f"{REGION_PARAMETERS[region]['order']:.2f}"
            for region in ("bath", "scalp", "skull", "CSF", "grey", "white")
        )
        + (f"{average_order:.3f}",),
        bbox=(
            (0.046, 0.655, 0.90, 0.100)
            if scenario == "sources"
            else (0.046, 0.625, 0.90, 0.095)
        ),
    )

    model_names = (
        "Layered tissues",
        "Homogenized",
        "Absolute difference",
    )
    model_titles = ("Layered", "Homogenized", "Difference")
    model_norms = (pressure_norm, pressure_norm, difference_norm)
    heatmaps = []
    for column, (name, row_title, norm) in enumerate(
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
        _plot_anatomical_interfaces(axis, labels, x_voxels, y_voxels)
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
                arrow_target = (
                    source_x + 0.58 * (target[0] - source_x),
                    source_y + 0.58 * (target[1] - source_y),
                )
                axis.annotate(
                    "",
                    xy=arrow_target,
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
                pulse_x,
                pulse_y,
                marker="*",
                markersize=7,
                color="#fff3a6",
                markeredgecolor=INK,
                markeredgewidth=0.6,
            )
        axis.set_xlim(*X_LIMITS)
        axis.set_ylim(*Y_LIMITS)
        axis.set_aspect("equal")
        axis.set_axis_off()
        axis.set_title(
            row_title,
            loc="left",
            color=INK,
            fontsize=26,
            pad=4,
        )
        heatmaps.append(image)

    interface_colors = (CORAL, INK, BLUE, "#e8a43c")
    interface_handles = tuple(
        Line2D((0.0, 1.0), (0.0, 0.0), color=color, linewidth=3.0)
        for color in interface_colors
    )
    figure.legend(
        interface_handles,
        ("skin edge", "skull", "brain / CSF", "white matter"),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.065),
        ncol=4,
        frameon=False,
        fontsize=18,
    )
    pressure_colorbar_axis = figure.add_axes((0.12, 0.145, 0.49, 0.014))
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
    difference_colorbar_axis = figure.add_axes((0.70, 0.145, 0.22, 0.014))
    difference_colorbar = figure.colorbar(
        heatmaps[2],
        cax=difference_colorbar_axis,
        orientation="horizontal",
        label="absolute difference",
    )
    difference_colorbar.set_ticks((0.0, 0.5 * difference_limit, difference_limit))
    difference_formatter = ticker.ScalarFormatter(useMathText=True)
    difference_formatter.set_powerlimits((-3, 3))
    difference_formatter.set_useOffset(False)
    difference_colorbar.formatter = difference_formatter
    difference_colorbar.update_ticks()
    difference_colorbar.ax.xaxis.set_label_position("top")
    figure.text(
        0.5,
        0.008,
        (
            "Anatomy: BrainWeb normal phantom • Collins et al., IEEE TMI "
            "17(3), 463–468 (1998)\n"
            f"{BRAINWEB_PAGE}"
        ),
        ha="center",
        va="bottom",
        color=INK,
        fontsize=17,
        linespacing=1.1,
    )
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
    if scenario == "sources":
        readme_movie = animation.FuncAnimation(
            figure,
            update,
            frames=range(0, len(times), 2),
            interval=2000 / args.fps,
            blit=False,
        )
        readme_movie.save(
            destination / "caputo-wismer-brainweb-sources-readme.gif",
            writer=animation.PillowWriter(fps=args.fps / 2.0),
            dpi=900.0 / figure.get_size_inches()[0],
        )
    plt.close(figure)

    print(output_path)
    print(data_path)
    print(
        f"BrainWeb slice: index={args.slice_index}, z={slice_z_mm:g} mm, "
        f"labels={sorted(np.unique(labels).tolist())}"
    )
    print(f"mesh: {coordinates.shape[0]} vertices, {cells.shape[0]} triangles")
    print(
        f"homogenized: c²={average_speed_squared:.6f}, "
        f"b={average_damping:.6f}, alpha={average_order:.6f}"
    )
    for key, stepper in steppers.items():
        print(key, stepper.solver_stats())
    communicator.barrier()
