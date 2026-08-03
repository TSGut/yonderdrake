"""Compare attenuation-aware vessel reconstructions in a layered skull-ball."""

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
    from ._gallery_path import add_gallery_path
    from ._imaging import (
        _solver_parameters,
        reconstruct_sensor_data,
        vessel_values,
    )
except ImportError:
    from _gallery_path import add_gallery_path
    from _imaging import (
        _solver_parameters,
        reconstruct_sensor_data,
        vessel_values,
    )

add_gallery_path()

from _visual_mpi import gather_p1_animation_data  # noqa: E402
from _visual_style import (  # noqa: E402
    CORAL,
    INK,
    TEAL,
    configure_matplotlib,
    output_directory,
    signature,
)
from matplotlib import pyplot as plt  # noqa: E402


def _materials(
    fd: Any,
    mesh: Any,
) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    from yonderdrake.applications import CaputoWismerMaterial

    coordinates = fd.SpatialCoordinate(mesh)
    radius_squared = sum(coordinate**2 for coordinate in coordinates)
    tissue = fd.conditional(radius_squared <= 0.58**2, 1.0, 0.0)
    skull = fd.conditional(
        radius_squared <= 0.58**2,
        0.0,
        fd.conditional(radius_squared <= 0.74**2, 1.0, 0.0),
    )
    bath = fd.conditional(radius_squared <= 0.74**2, 0.0, 1.0)
    specifications = (
        (bath, 1.00, 0.96, 0.003, 0.12),
        (skull, 1.85, 1.62, 0.065, 0.42),
        (tissue, 1.04, 1.00, 0.018, 0.60),
    )
    heterogeneous = tuple(
        CaputoWismerMaterial(
            indicator=indicator,
            density=density,
            wave_speed=speed,
            damping=damping,
            alpha=alpha,
        )
        for indicator, density, speed, damping, alpha in specifications
    )
    volume = float(fd.assemble(fd.Constant(1.0) * fd.dx(domain=mesh)))
    fractions = np.asarray(
        [
            float(fd.assemble(indicator * fd.dx(domain=mesh))) / volume
            for indicator, *_ in specifications
        ]
    )
    speed = float(
        np.sqrt(
            sum(
                fraction * specification[2] ** 2
                for fraction, specification in zip(
                    fractions,
                    specifications,
                    strict=True,
                )
            )
        )
    )
    damping = float(
        sum(
            fraction * specification[3]
            for fraction, specification in zip(
                fractions,
                specifications,
                strict=True,
            )
        )
    )
    alpha = float(
        sum(
            fraction * specification[4]
            for fraction, specification in zip(
                fractions,
                specifications,
                strict=True,
            )
        )
    )
    density = float(
        sum(
            fraction * specification[1]
            for fraction, specification in zip(
                fractions,
                specifications,
                strict=True,
            )
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


def _relative_error(
    truth: np.ndarray,
    image: np.ndarray,
    reconstruction_region: np.ndarray,
) -> float:
    difference = image[reconstruction_region] - truth[reconstruction_region]
    return float(
        np.linalg.norm(difference) / np.linalg.norm(truth[reconstruction_region])
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refinement", type=int, default=4)
    parser.add_argument("--degree", type=int, choices=(1, 2), default=2)
    parser.add_argument("--sensors", type=int, default=128)
    parser.add_argument("--sensor-width", type=float, default=0.025)
    parser.add_argument("--modes", type=int, default=12)
    parser.add_argument("--dt", type=float, default=0.003)
    parser.add_argument("--final-time", type=float, default=2.4)
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
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parents[2] / "demo-output"),
    )
    args = parser.parse_args()
    if args.refinement < 1:
        parser.error("--refinement must be at least 1")
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
    mesh = fd.UnitDiskMesh(
        refinement_level=args.refinement,
        comm=communicator,
    )
    space = fd.FunctionSpace(mesh, "CG", args.degree)
    plot_space = fd.FunctionSpace(mesh, "CG", 1)
    coordinate_field = fd.Function(
        fd.VectorFunctionSpace(mesh, "CG", args.degree)
    ).interpolate(fd.SpatialCoordinate(mesh))
    dof_coordinates = np.asarray(coordinate_field.dat.data_ro, dtype=float)
    initial_pressure = fd.Function(space, name="vascular_initial_pressure")
    initial_values = vessel_values(
        dof_coordinates,
        dimension=2,
        width=0.035,
    )
    initial_values[np.linalg.norm(dof_coordinates, axis=1) > 0.56] = 0.0
    initial_pressure.dat.data[:] = initial_values

    sensors = SensorArray.ring(
        space,
        args.sensors,
        0.90,
        width=args.sensor_width,
    )
    heterogeneous, homogeneous = _materials(fd, mesh)
    num_steps = round(args.final_time / args.dt)
    solver_parameters = _solver_parameters(communicator.size > 1)
    forward_model = CaputoWismerModel(
        space,
        materials=heterogeneous,
        sensors=sensors,
        dt=args.dt,
        num_steps=num_steps,
        num_modes=args.modes,
        boundaries=(CaputoWismerImpedanceBoundary(coefficient=1.0 / 0.96),),
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
        boundaries=(CaputoWismerImpedanceBoundary(coefficient=1.0 / 0.96),),
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
    coordinates, _, gathered_fields = gathered
    reconstruction_region = np.linalg.norm(coordinates, axis=1) <= 0.56
    truth = gathered_fields["truth"][0]
    images = tuple(gathered_fields[name][0] for name in field_names)
    relative_errors = tuple(
        _relative_error(truth, image, reconstruction_region) for image in images
    )
    truth_scale = float(np.max(np.abs(truth[reconstruction_region])))
    image_scales = tuple(
        float(np.max(np.abs(image[reconstruction_region]))) for image in images
    )
    peak_ratios = tuple(scale / truth_scale for scale in image_scales)
    truth_display = np.clip(truth / truth_scale, 0.0, 1.0)
    displays = tuple(
        np.clip(image / scale, 0.0, 1.0)
        for image, scale in zip(images, image_scales, strict=True)
    )
    for display in displays:
        display[~reconstruction_region] = 0.0

    figure, axes = plt.subplots(3, 2, figsize=(18, 21))
    figure.suptitle(
        (
            "Caputo-Wismer initial-pressure imaging"
            "\nin a layered skull-ball with heterogeneous density, speed,"
            " and attenuation"
            f"\n{args.sensors} uniformly spaced exterior ring sensors"
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
    for axis, display, title in zip(
        spatial_axes,
        spatial_values,
        spatial_titles,
        strict=True,
    ):
        axis.tricontourf(
            coordinates[:, 0],
            coordinates[:, 1],
            display,
            levels=np.linspace(0.0, 1.0, 31),
            cmap="magma",
            extend="max",
        )
        for radius, color in ((0.58, TEAL), (0.74, CORAL)):
            axis.add_patch(
                plt.Circle(
                    (0.0, 0.0),
                    radius,
                    fill=False,
                    color=color,
                    linewidth=1.5,
                )
            )
        _plot_sensors(axis, sensors.locations)
        axis.set_title(title, fontsize=15)
        axis.set_aspect("equal")
        axis.set_xlim(-1.0, 1.0)
        axis.set_ylim(-1.0, 1.0)

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

    output = destination / "caputo-wismer-ball-imaging-comparison.png"
    figure.savefig(output, dpi=150)
    np.savetxt(
        destination / "caputo-wismer-ball-imaging-sensors.csv",
        np.column_stack((times, data)),
        delimiter=",",
        header="time,"
        + ",".join(f"sensor_{index + 1}" for index in range(sensors.num_sensors)),
        comments="",
    )
    plt.close(figure)
    print(f"wrote {output}")
    communicator.barrier()


if __name__ == "__main__":
    main()
