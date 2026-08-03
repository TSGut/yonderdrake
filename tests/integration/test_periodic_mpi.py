"""MPI checks for the slab-distributed periodic Fourier backend."""

from __future__ import annotations

import pytest

fd = pytest.importorskip("firedrake")

from yonderdrake import PeriodicFractionalLaplacian  # noqa: E402


def distributed_periodic_check(expected_size: int) -> None:
    order = 0.4
    if expected_size == 2:
        interval = fd.PeriodicIntervalMesh(7, 2.0)
        interval_space = fd.FunctionSpace(interval, "CG", 1)
        (interval_x,) = fd.SpatialCoordinate(interval)
        interval_u = fd.Function(interval_space).interpolate(
            fd.sin(fd.pi * interval_x)
            + 0.25 * fd.cos(2.0 * fd.pi * interval_x)
        )
        interval_operator = PeriodicFractionalLaplacian(interval_u, order)
        interval_result = fd.assemble(interval_operator)
        interval_expected = fd.Function(interval_space).interpolate(
            fd.pi ** (2.0 * order) * fd.sin(fd.pi * interval_x)
            + 0.25
            * (2.0 * fd.pi) ** (2.0 * order)
            * fd.cos(2.0 * fd.pi * interval_x)
        )
        assert fd.norm(interval_result - interval_expected) < 2.0e-12
        assert (
            interval_operator.diagnostics()["fft_backend"]
            == "numpy-mpi-slab"
        )

    rectangle = fd.PeriodicRectangleMesh(
        6,
        4,
        2.0,
        3.0,
        quadrilateral=True,
        reorder=False,
    )
    assert rectangle.comm.size == expected_size
    space = fd.FunctionSpace(rectangle, "Q", 1)
    x, y = fd.SpatialCoordinate(rectangle)
    u = fd.Function(space).interpolate(
        fd.sin(fd.pi * x) * fd.cos(2.0 * fd.pi * y / 3.0)
    )
    eigenvalue = fd.pi**2 + (2.0 * fd.pi / 3.0) ** 2
    operator = PeriodicFractionalLaplacian(u, order)

    result = fd.assemble(operator)
    expected = fd.Function(space).interpolate(eigenvalue**order * u)

    assert fd.norm(result - expected) < 2.0e-12
    diagnostics = operator.diagnostics()
    assert diagnostics["ranks"] == expected_size
    assert diagnostics["fft_backend"] == "numpy-mpi-slab"
    assert diagnostics["decomposition"] == "slab-alltoallv"
    assert diagnostics["replicated_grid_map"] is False
    assert diagnostics["local_grid_map_values"] < space.dim()
    assert diagnostics["local_real_values"] < space.dim()

    box = fd.PeriodicBoxMesh(
        6,
        4,
        3,
        2.0,
        3.0,
        4.0,
        hexahedral=True,
        reorder=False,
    )
    space_3d = fd.FunctionSpace(box, "Q", 1)
    x, y, z = fd.SpatialCoordinate(box)
    u_3d = fd.Function(space_3d).interpolate(
        fd.sin(fd.pi * x)
        * fd.cos(2.0 * fd.pi * y / 3.0)
        * fd.cos(2.0 * fd.pi * z / 4.0)
    )
    eigenvalue_3d = fd.pi**2
    eigenvalue_3d += (2.0 * fd.pi / 3.0) ** 2
    eigenvalue_3d += (2.0 * fd.pi / 4.0) ** 2
    operator_3d = PeriodicFractionalLaplacian(u_3d, order)

    result_3d = fd.assemble(operator_3d)
    expected_3d = fd.Function(space_3d).interpolate(
        eigenvalue_3d**order * u_3d
    )

    assert fd.norm(result_3d - expected_3d) < 3.0e-12
    assert operator_3d.diagnostics()["shape"] == (6, 4, 3)
    assert operator_3d.diagnostics()["ranks"] == expected_size
    assert operator_3d.diagnostics()["fft_backend"] == "numpy-mpi-slab"
    assert operator_3d.diagnostics()["local_real_values"] < space_3d.dim()


@pytest.mark.parallel(nprocs=2)
@pytest.mark.unit
def test_distributed_periodic_two_ranks() -> None:
    distributed_periodic_check(2)


@pytest.mark.parallel(nprocs=4)
@pytest.mark.unit
def test_distributed_periodic_four_ranks() -> None:
    distributed_periodic_check(4)
