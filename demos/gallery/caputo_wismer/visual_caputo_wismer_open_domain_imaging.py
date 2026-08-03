"""Compare three one-pass images from a heterogeneous open-domain model."""

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
    from ._imaging import normalized_array, vessel_values
except ImportError:
    from _gallery_path import add_gallery_path
    from _imaging import normalized_array, vessel_values

add_gallery_path()

from _visual_mpi import gather_p1_animation_data  # noqa: E402
from _visual_style import (  # noqa: E402
    CORAL,
    INK,
    MUTED,
    PAPER,
    configure_matplotlib,
    output_directory,
    signature,
)
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib import tri  # noqa: E402


def _solver_parameters(parallel: bool) -> dict[str, Any]:
    if parallel:
        return {
            "snes_type": "ksponly",
            "ksp_type": "gmres",
            "ksp_rtol": 1.0e-9,
            "ksp_max_it": 500,
            "ksp_error_if_not_converged": True,
            "pc_type": "bjacobi",
            "sub_pc_type": "ilu",
        }
    return {
        "snes_type": "ksponly",
        "ksp_type": "preonly",
        "pc_type": "lu",
    }


def _correlation(truth: np.ndarray, image: np.ndarray) -> float:
    return float(np.corrcoef(truth, image)[0, 1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=int, default=44)
    parser.add_argument("--degree", type=int, choices=(1, 2), default=1)
    parser.add_argument("--sensors", type=int, default=48)
    parser.add_argument("--modes", type=int, default=8)
    parser.add_argument("--dt", type=float, default=0.006)
    parser.add_argument("--final-time", type=float, default=1.5)
    parser.add_argument("--filter-length", type=float, default=0.055)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parents[2] / "demo-output"),
    )
    args = parser.parse_args()
    if args.cells < 8:
        parser.error("--cells must be at least 8")
    if args.sensors < 4:
        parser.error("--sensors must be at least 4")
    if args.smoke:
        args.cells = 12
        args.degree = 1
        args.sensors = 12
        args.modes = 2
        args.dt = 0.02
        args.final_time = 0.18
        args.filter_length = 0.12
    sys.argv[1:] = []

    import firedrake as fd

    from yonderdrake.applications import (
        CaputoWismerMaterial,
        CaputoWismerModel,
        CaputoWismerPML,
        SensorArray,
        time_reverse_sensor_data,
    )

    configure_matplotlib(plt)
    communicator = fd.COMM_WORLD
    destination = output_directory(args.output)
    mesh = fd.RectangleMesh(
        args.cells,
        args.cells,
        2.6,
        2.6,
        comm=communicator,
    )
    mesh.coordinates.dat.data[:, :] -= 1.3
    space = fd.FunctionSpace(mesh, "CG", args.degree)
    plot_space = fd.FunctionSpace(mesh, "CG", 1)
    x, y = fd.SpatialCoordinate(mesh)

    inclusion_left = fd.conditional(
        ((x + 0.34) / 0.31) ** 2 + ((y - 0.08) / 0.52) ** 2 <= 1.0,
        1.0,
        0.0,
    )
    inclusion_right = fd.conditional(
        ((x - 0.37) / 0.25) ** 2 + ((y + 0.13) / 0.43) ** 2 <= 1.0,
        1.0,
        0.0,
    )
    inclusion = fd.max_value(inclusion_left, inclusion_right)
    background = 1.0 - inclusion
    materials = (
        CaputoWismerMaterial(
            indicator=background,
            density=1.0,
            wave_speed=1.0,
            damping=0.006,
            alpha=0.28,
        ),
        CaputoWismerMaterial(
            indicator=inclusion,
            density=1.38,
            wave_speed=1.22,
            damping=0.035,
            alpha=0.62,
        ),
    )
    pml = CaputoWismerPML.box(
        mesh,
        ((-1.0, 1.0), (-1.0, 1.0)),
        reference_speed=1.0,
        reflection=1.0e-5,
        polynomial_order=3,
    )
    sensors = SensorArray.ring(
        space,
        args.sensors,
        0.88,
        width=0.045 if not args.smoke else 0.12,
    )
    coordinate_space = fd.VectorFunctionSpace(mesh, "CG", args.degree)
    dof_coordinates = np.asarray(
        fd.Function(coordinate_space)
        .interpolate(fd.SpatialCoordinate(mesh))
        .dat.data_ro,
        dtype=np.float64,
    )
    initial = fd.Function(space, name="branching initial pressure")
    initial_values = vessel_values(
        dof_coordinates,
        dimension=2,
        width=0.035 if not args.smoke else 0.08,
    )
    initial_values[
        (np.abs(dof_coordinates[:, 0]) >= 0.78)
        | (np.abs(dof_coordinates[:, 1]) >= 0.78)
    ] = 0.0
    initial.dat.data[:] = initial_values

    num_steps = round(args.final_time / args.dt)
    model = CaputoWismerModel(
        space,
        materials=materials,
        sensors=sensors,
        pml=pml,
        dt=args.dt,
        num_steps=num_steps,
        num_modes=args.modes,
        stiffness_theta=1.0,
        solver_parameters=_solver_parameters(communicator.size > 1),
    )
    propagation = model.propagate(initial, record_history=True)
    data = propagation.sensor_data
    assert data is not None
    snapshot_index = min(
        num_steps,
        max(1, round(0.55 * num_steps)),
    )
    images = {
        "Exact adjoint": model.adjoint(args.dt * data),
        "Lossless time reversal": time_reverse_sensor_data(
            model,
            data,
            compensate_attenuation=False,
        ),
        "Regularized reverse attenuation": time_reverse_sensor_data(
            model,
            data,
            compensate_attenuation=True,
            filter_length=args.filter_length,
            filter_order=2,
        ),
    }

    plot_view = fd.Function(plot_space, name="plot view")

    def local_values(field: Any) -> np.ndarray:
        return np.asarray(
            plot_view.interpolate(field).dat.data_ro,
            dtype=np.float64,
        ).copy()

    local_fields = {
        "truth": local_values(initial)[None, :],
        "snapshot": local_values(propagation.field_history[snapshot_index])[None, :],
        "density": local_values(
            fd.Function(space).interpolate(background + 1.38 * inclusion)
        )[None, :],
        **{name: local_values(field)[None, :] for name, field in images.items()},
    }
    gathered = gather_p1_animation_data(mesh, local_fields, communicator)
    if communicator.rank != 0:
        communicator.barrier()
        return
    assert gathered is not None
    coordinates, cells, fields = gathered
    triangulation = tri.Triangulation(
        coordinates[:, 0],
        coordinates[:, 1],
        cells,
    )
    truth = normalized_array(fields["truth"][0])
    snapshot = normalized_array(fields["snapshot"][0])
    displays = {name: normalized_array(fields[name][0]) for name in images}

    figure = plt.figure(figsize=(21, 15), facecolor=PAPER)
    grid = figure.add_gridspec(
        2,
        3,
        left=0.055,
        right=0.97,
        bottom=0.07,
        top=0.82,
        hspace=0.28,
        wspace=0.24,
    )
    figure.suptitle(
        "Open-domain Caputo-Wismer imaging",
        x=0.055,
        y=0.955,
        ha="left",
        color=INK,
        fontsize=32,
        fontweight="bold",
    )
    figure.text(
        0.055,
        0.895,
        (
            "Heterogeneous density and attenuation, exterior sensor ring, "
            "box PML, and three one-pass reconstructions"
        ),
        color=MUTED,
        fontsize=22,
    )
    signature(figure)

    def spatial_axis(axis: Any) -> None:
        axis.set_aspect("equal")
        axis.set_xlim(-1.31, 1.31)
        axis.set_ylim(-1.31, 1.31)
        axis.set_xticks((-1.0, 0.0, 1.0))
        axis.set_yticks((-1.0, 0.0, 1.0))
        axis.grid(False)
        axis.plot(
            (-1.0, 1.0, 1.0, -1.0, -1.0),
            (-1.0, -1.0, 1.0, 1.0, -1.0),
            color=INK,
            linestyle="--",
            linewidth=1.5,
            alpha=0.72,
        )

    setup_axis = figure.add_subplot(grid[0, 0])
    setup_axis.tripcolor(
        triangulation,
        fields["density"][0],
        shading="gouraud",
        cmap="viridis",
        vmin=1.0,
        vmax=1.38,
    )
    setup_axis.tricontour(
        triangulation,
        truth,
        levels=(0.25, 0.55),
        colors=(CORAL, CORAL),
        linewidths=(1.5, 2.4),
    )
    setup_axis.scatter(
        sensors.locations[:, 0],
        sensors.locations[:, 1],
        s=16,
        facecolors=PAPER,
        edgecolors=INK,
        linewidths=0.7,
        zorder=20,
    )
    setup_axis.set_title("Model, source, sensors, and PML", fontsize=21)
    spatial_axis(setup_axis)

    snapshot_axis = figure.add_subplot(grid[0, 1])
    wave_limit = max(0.25, float(np.max(np.abs(snapshot))))
    snapshot_axis.tripcolor(
        triangulation,
        snapshot,
        shading="gouraud",
        cmap="RdBu_r",
        vmin=-wave_limit,
        vmax=wave_limit,
    )
    snapshot_axis.set_title(
        rf"Outgoing pressure at $t={snapshot_index * args.dt:.2f}$",
        fontsize=21,
    )
    spatial_axis(snapshot_axis)

    trace_axis = figure.add_subplot(grid[0, 2])
    times = args.dt * np.arange(num_steps + 1)
    selected = np.linspace(
        0,
        args.sensors - 1,
        min(9, args.sensors),
        dtype=int,
    )
    colors = plt.get_cmap("viridis")(np.linspace(0.08, 0.92, selected.size))
    for sensor_index, color in zip(selected, colors, strict=True):
        trace_axis.plot(
            times,
            data[:, sensor_index],
            color=color,
            linewidth=1.3,
            alpha=0.9,
        )
    trace_axis.axhline(0.0, color=INK, linewidth=0.8, alpha=0.55)
    trace_axis.set_title("Recorded pressure", fontsize=21)
    trace_axis.set_xlabel("time")
    trace_axis.set_ylabel("pressure", fontsize=18)

    for column, (name, values) in enumerate(displays.items()):
        axis = figure.add_subplot(grid[1, column])
        limit = max(float(np.max(np.abs(values))), 1.0e-12)
        axis.tripcolor(
            triangulation,
            values,
            shading="gouraud",
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
        )
        axis.set_title(
            f"{name}\ncorrelation {_correlation(truth, values):.2f}",
            fontsize=20,
        )
        spatial_axis(axis)

    figure.text(
        0.055,
        0.025,
        (
            "Dashed square: physical region. Exterior strip: PML. "
            "Each reconstruction uses the same forward model and sensor map."
        ),
        color=MUTED,
        fontsize=18,
    )
    output_path = destination / "caputo-wismer-open-domain-imaging.png"
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    print(output_path)


if __name__ == "__main__":
    main()
