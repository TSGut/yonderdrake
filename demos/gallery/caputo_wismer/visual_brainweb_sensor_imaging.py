"""Compare attenuation-aware vessel reconstructions through a BrainWeb slice."""

from __future__ import annotations

import argparse
import os
import sys
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
    from ._brainweb import (
        BRAINWEB_PAGE,
        BRAINWEB_SLICE_INDEX,
        load_brainweb_slice,
        sample_labels,
        slice_coordinates,
    )
    from ._gallery_path import add_gallery_path
    from ._imaging import (
        _solver_parameters,
        reconstruct_sensor_data,
        vessel_values,
    )
    from ._visual_brainweb_wismer import (
        REGION_PARAMETERS,
        X_LIMITS,
        Y_LIMITS,
        _material_masks,
        _plot_anatomical_interfaces,
    )
except ImportError:
    from _brainweb import (
        BRAINWEB_PAGE,
        BRAINWEB_SLICE_INDEX,
        load_brainweb_slice,
        sample_labels,
        slice_coordinates,
    )
    from _gallery_path import add_gallery_path
    from _imaging import (
        _solver_parameters,
        reconstruct_sensor_data,
        vessel_values,
    )
    from _visual_brainweb_wismer import (
        REGION_PARAMETERS,
        X_LIMITS,
        Y_LIMITS,
        _material_masks,
        _plot_anatomical_interfaces,
    )

add_gallery_path()

from _visual_mpi import gather_p1_animation_data  # noqa: E402
from _visual_style import (  # noqa: E402
    INK,
    configure_matplotlib,
    output_directory,
    signature,
)
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib import tri  # noqa: E402


def _materials(
    fd: Any,
    mesh: Any,
    labels: np.ndarray,
) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    from yonderdrake.applications import CaputoWismerMaterial

    masks = _material_masks(fd, mesh, labels)
    heterogeneous = tuple(
        CaputoWismerMaterial(
            indicator=masks[region],
            density=parameters["density"],
            wave_speed=parameters["speed"],
            damping=parameters["damping"],
            alpha=parameters["order"],
        )
        for region, parameters in REGION_PARAMETERS.items()
    )
    volume = float(fd.assemble(fd.Constant(1.0) * fd.dx(domain=mesh)))
    fractions = {
        region: float(fd.assemble(mask * fd.dx)) / volume
        for region, mask in masks.items()
    }
    speed = float(
        np.sqrt(
            sum(
                fractions[region] * parameters["speed"] ** 2
                for region, parameters in REGION_PARAMETERS.items()
            )
        )
    )
    damping = float(
        sum(
            fractions[region] * parameters["damping"]
            for region, parameters in REGION_PARAMETERS.items()
        )
    )
    alpha = float(
        sum(
            fractions[region] * parameters["order"]
            for region, parameters in REGION_PARAMETERS.items()
        )
    )
    density = float(
        sum(
            fractions[region] * parameters["density"]
            for region, parameters in REGION_PARAMETERS.items()
        )
    )
    homogeneous = (
        CaputoWismerMaterial(
            indicator=fd.Constant(1.0),
            density=density,
            wave_speed=speed,
            damping=damping,
            alpha=alpha,
        ),
    )
    return heterogeneous, homogeneous


def _relative_error(
    truth: np.ndarray,
    image: np.ndarray,
    inside_brain: np.ndarray,
) -> float:
    difference = image[inside_brain] - truth[inside_brain]
    return float(np.linalg.norm(difference) / np.linalg.norm(truth[inside_brain]))


def _plot_sensors(axis: Any, locations: np.ndarray) -> None:
    axis.scatter(
        locations[:, 0],
        locations[:, 1],
        s=18,
        marker="o",
        facecolors="white",
        edgecolors=INK,
        linewidths=0.8,
        zorder=20,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refinement", type=int, default=3)
    parser.add_argument("--degree", type=int, choices=(1, 2), default=2)
    parser.add_argument("--sensors", type=int, default=128)
    parser.add_argument("--sensor-width", type=float, default=0.03)
    parser.add_argument("--modes", type=int, default=16)
    parser.add_argument("--dt", type=float, default=0.003)
    parser.add_argument("--final-time", type=float, default=3.0)
    parser.add_argument(
        "--inverse-iterations",
        type=int,
        default=100,
        help="optimizer iteration cap (default: 100)",
    )
    parser.add_argument("--inverse-tolerance", type=float, default=1.0e-5)
    parser.add_argument("--regularization", type=float, default=1.0e-7)
    parser.add_argument("--attenuation-filter-length", type=float)
    parser.add_argument("--attenuation-filter-order", type=int, default=2)
    parser.add_argument(
        "--slice-index",
        type=int,
        default=BRAINWEB_SLICE_INDEX,
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parents[2] / "demo-output"),
    )
    args = parser.parse_args()
    if args.refinement < 2:
        parser.error("--refinement must be at least 2")
    if args.sensors < 1:
        parser.error("--sensors must be positive")
    if args.smoke:
        args.refinement = 2
        args.sensors = 24
        args.modes = 2
        args.dt = 0.04
        args.final_time = 0.24
        args.inverse_iterations = 1
    sys.argv[1:] = []

    import firedrake as fd

    from yonderdrake.applications import (
        CaputoWismerImpedanceBoundary,
        CaputoWismerModel,
        SensorArray,
    )

    configure_matplotlib(plt)
    communicator = fd.COMM_WORLD
    destination = output_directory(args.output)
    labels = None
    if communicator.rank == 0:
        labels = load_brainweb_slice(
            destination / ".brainweb",
            slice_index=args.slice_index,
        )
    labels = communicator.bcast(labels, root=0)
    refinement_factor = 2 ** (args.refinement - 2)
    mesh = fd.RectangleMesh(
        24 * refinement_factor,
        28 * refinement_factor,
        X_LIMITS[1] - X_LIMITS[0],
        Y_LIMITS[1] - Y_LIMITS[0],
        comm=communicator,
    )
    mesh.coordinates.dat.data[:, 0] += X_LIMITS[0]
    mesh.coordinates.dat.data[:, 1] += Y_LIMITS[0]
    space = fd.FunctionSpace(mesh, "CG", args.degree)
    plot_space = fd.FunctionSpace(mesh, "CG", 1)
    coordinate_field = fd.Function(
        fd.VectorFunctionSpace(mesh, "CG", args.degree)
    ).interpolate(fd.SpatialCoordinate(mesh))
    dof_coordinates = np.asarray(coordinate_field.dat.data_ro, dtype=float)
    point_labels = sample_labels(labels, dof_coordinates)
    inside_brain = np.isin(point_labels, (1, 2, 3, 8))

    initial_pressure = fd.Function(space, name="vascular_initial_pressure")
    values = vessel_values(dof_coordinates, dimension=2, width=0.028)
    values[~inside_brain] = 0.0
    initial_pressure.dat.data[:] = values

    center = (
        0.5 * (X_LIMITS[0] + X_LIMITS[1]),
        0.5 * (Y_LIMITS[0] + Y_LIMITS[1]),
    )
    angles = 2.0 * np.pi * np.arange(args.sensors, dtype=np.float64) / args.sensors
    sensor_locations = np.column_stack(
        (
            center[0] + 0.96 * 0.5 * (X_LIMITS[1] - X_LIMITS[0]) * np.cos(angles),
            center[1] + 0.96 * 0.5 * (Y_LIMITS[1] - Y_LIMITS[0]) * np.sin(angles),
        )
    )
    sensors = SensorArray(
        space,
        sensor_locations,
        width=args.sensor_width,
    )
    heterogeneous, homogeneous = _materials(fd, mesh, labels)
    num_steps = round(args.final_time / args.dt)
    solver_parameters = _solver_parameters(communicator.size > 1)
    bath = REGION_PARAMETERS["bath"]
    boundary_impedance = 1.0 / (bath["density"] * bath["speed"])
    forward_model = CaputoWismerModel(
        space,
        materials=heterogeneous,
        sensors=sensors,
        dt=args.dt,
        num_steps=num_steps,
        num_modes=args.modes,
        boundaries=(
            CaputoWismerImpedanceBoundary(
                coefficient=boundary_impedance,
            ),
        ),
        solver_parameters=solver_parameters,
    )
    data = forward_model.propagate(initial_pressure).sensor_data
    assert data is not None

    homogeneous_model = CaputoWismerModel(
        space,
        materials=homogeneous,
        sensors=sensors,
        dt=args.dt,
        num_steps=num_steps,
        num_modes=args.modes,
        boundaries=(
            CaputoWismerImpedanceBoundary(
                coefficient=boundary_impedance,
            ),
        ),
        solver_parameters=solver_parameters,
    )

    reconstruction_specs = (
        (
            "Dissipative inversion · heterogeneous",
            "kaltenbacher",
            forward_model,
            False,
        ),
        (
            "Dissipative inversion · homogenised",
            "kaltenbacher",
            homogeneous_model,
            False,
        ),
        (
            "Lossless backpropagation · heterogeneous",
            "time_reversal",
            forward_model,
            False,
        ),
        (
            "Reverse attenuation · heterogeneous",
            "time_reversal",
            forward_model,
            True,
        ),
    )
    reconstructions = []
    for _, method, model, compensation in reconstruction_specs:
        reconstructions.append(
            reconstruct_sensor_data(
                model,
                data,
                method=method,
                regularization=args.regularization,
                max_iterations=args.inverse_iterations,
                tolerance=args.inverse_tolerance,
                positivity=True,
                compensate_attenuation=compensation,
                filter_length=(
                    args.attenuation_filter_length if compensation else None
                ),
                filter_order=args.attenuation_filter_order,
            )
        )

    plot_view = fd.Function(plot_space, name="p1_reconstruction_view")

    def local_plot_values(field: Any) -> np.ndarray:
        return np.asarray(
            plot_view.interpolate(field).dat.data_ro,
            dtype=np.float64,
        ).copy()

    field_names = tuple(
        f"reconstruction_{index}" for index in range(len(reconstruction_specs))
    )
    local_fields = {
        "truth": local_plot_values(initial_pressure)[None, :],
        **{
            name: local_plot_values(image)[None, :]
            for name, image in zip(
                field_names,
                reconstructions,
                strict=True,
            )
        },
    }
    gathered = gather_p1_animation_data(
        mesh,
        local_fields,
        communicator,
    )
    if communicator.rank != 0:
        communicator.barrier()
        return
    assert gathered is not None
    coordinates, cells, gathered_fields = gathered
    plot_inside_brain = np.isin(
        sample_labels(labels, coordinates),
        (1, 2, 3, 8),
    )
    truth = gathered_fields["truth"][0]
    images = tuple(gathered_fields[name][0] for name in field_names)
    relative_errors = tuple(
        _relative_error(truth, image, plot_inside_brain) for image in images
    )
    truth_scale = float(np.max(np.abs(truth[plot_inside_brain])))
    image_scales = tuple(
        float(np.max(np.abs(image[plot_inside_brain]))) for image in images
    )
    peak_ratios = tuple(scale / truth_scale for scale in image_scales)
    truth_display = np.clip(truth / truth_scale, 0.0, 1.0)
    truth_display[~plot_inside_brain] = 0.0
    displays = tuple(
        np.clip(image / scale, 0.0, 1.0)
        for image, scale in zip(images, image_scales, strict=True)
    )
    for image in displays:
        image[~plot_inside_brain] = 0.0

    triangulation = tri.Triangulation(
        coordinates[:, 0],
        coordinates[:, 1],
        cells,
    )
    x_voxels, y_voxels = slice_coordinates()

    figure, axes = plt.subplots(3, 2, figsize=(18, 22))
    figure.suptitle(
        (
            "Caputo-Wismer initial-pressure imaging"
            "\nin the BrainWeb phantom with heterogeneous density, speed,"
            " and attenuation"
            f"\n{args.sensors} exterior elliptical-array sensors"
        ),
        color=INK,
        fontweight="bold",
        fontsize=24,
    )

    spatial_axes = [axes[0, 0], *axes[1:, :].flat]
    spatial_values = [
        truth_display,
        *displays,
    ]
    spatial_titles = [
        "True vessel and sensor positions",
        *(
            f"{label} · error {relative_errors[index]:.2f} · "
            f"peak {peak_ratios[index]:.2f}×"
            for index, (label, *_rest) in enumerate(reconstruction_specs)
        ),
    ]
    for axis, image, title in zip(
        spatial_axes,
        spatial_values,
        spatial_titles,
        strict=True,
    ):
        axis.tricontourf(
            triangulation,
            image,
            levels=np.linspace(0.0, 1.0, 31),
            cmap="magma",
            extend="max",
        )
        _plot_anatomical_interfaces(axis, labels, x_voxels, y_voxels)
        _plot_sensors(axis, sensors.locations)
        axis.set_title(title, fontsize=15)
        axis.set_aspect("equal")
        axis.set_xlim(*X_LIMITS)
        axis.set_ylim(*Y_LIMITS)

    times = np.linspace(0.0, num_steps * args.dt, num_steps + 1)
    trace_artist = axes[0, 1].imshow(
        data.T,
        origin="lower",
        aspect="auto",
        extent=(times[0], times[-1], 1, sensors.num_sensors),
        cmap="coolwarm",
    )
    axes[0, 1].set_title("Recorded pressure")
    axes[0, 1].set_xlabel("time")
    axes[0, 1].set_ylabel("sensor")
    figure.colorbar(
        trace_artist,
        ax=axes[0, 1],
        fraction=0.046,
        pad=0.04,
    )
    signature(figure)
    figure.subplots_adjust(
        left=0.07,
        right=0.96,
        bottom=0.04,
        top=0.85,
        hspace=0.34,
        wspace=0.22,
    )
    if args.smoke:
        plt.close(figure)
        print("completed smoke run")
        communicator.barrier()
        return

    readme_figure, readme_axes = plt.subplots(1, 3, figsize=(18, 7.5))
    readme_figure.suptitle(
        "BrainWeb initial-pressure reconstruction",
        color=INK,
        fontweight="bold",
        fontsize=23,
    )
    for axis, image, title in zip(
        readme_axes[:2],
        (truth_display, displays[0]),
        (
            "True vessel excitation",
            "Kaltenbacher reconstruction · heterogeneous",
        ),
        strict=True,
    ):
        axis.tricontourf(
            triangulation,
            image,
            levels=np.linspace(0.0, 1.0, 31),
            cmap="magma",
            extend="max",
        )
        _plot_anatomical_interfaces(axis, labels, x_voxels, y_voxels)
        _plot_sensors(axis, sensors.locations)
        axis.set_title(title, fontsize=16)
        axis.set_aspect("equal")
        axis.set_xlim(*X_LIMITS)
        axis.set_ylim(*Y_LIMITS)
        axis.set_xticks([])
        axis.set_yticks([])

    readme_trace_artist = readme_axes[2].imshow(
        data.T,
        origin="lower",
        aspect="auto",
        extent=(times[0], times[-1], 1, sensors.num_sensors),
        cmap="coolwarm",
    )
    readme_axes[2].set_title("Recorded pressure", fontsize=16)
    readme_axes[2].set_xlabel("time")
    readme_axes[2].set_ylabel("sensor")
    readme_figure.colorbar(
        readme_trace_artist,
        ax=readme_axes[2],
        fraction=0.046,
        pad=0.04,
    )
    signature(readme_figure)
    readme_figure.text(
        0.5,
        0.025,
        (
            "Anatomy: BrainWeb normal phantom • Collins et al., IEEE TMI "
            "17(3), 463–468 (1998)\n"
            f"{BRAINWEB_PAGE}"
        ),
        ha="center",
        va="bottom",
        color=INK,
        fontsize=16,
        linespacing=1.1,
    )
    readme_figure.subplots_adjust(
        left=0.04,
        right=0.91,
        bottom=0.24,
        top=0.82,
        wspace=0.20,
    )
    readme_output = destination / "caputo-wismer-brainweb-imaging-readme.png"
    with plt.rc_context({"savefig.bbox": None}):
        readme_figure.savefig(readme_output, dpi=50, bbox_inches=None)
    plt.close(readme_figure)

    output = destination / "caputo-wismer-brainweb-imaging-comparison.png"
    figure.savefig(output, dpi=150)
    np.savetxt(
        destination / "caputo-wismer-brainweb-imaging-sensors.csv",
        np.column_stack((times, data)),
        delimiter=",",
        header="time,"
        + ",".join(f"sensor_{index + 1}" for index in range(sensors.num_sensors)),
        comments="",
    )
    plt.close(figure)
    print(f"wrote {readme_output}")
    print(f"wrote {output}")
    communicator.barrier()


if __name__ == "__main__":
    main()
