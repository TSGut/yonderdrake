"""Two-rank ownership check for the non-stored Riesz backend."""

from __future__ import annotations

import numpy as np
import pytest

fd = pytest.importorskip("firedrake")

from yonderdrake import RieszFractionalLaplacian  # noqa: E402


def assert_distributed_linearization(
    space,
    u,
    operator,
    direction,
) -> None:
    jacobian = fd.derivative(operator, u, fd.TrialFunction(space))
    action = fd.assemble(fd.action(jacobian, direction))
    matrix = fd.assemble(jacobian)
    matrix_action = fd.Function(space)
    with direction.dat.vec_ro as source, matrix_action.dat.vec as target:
        matrix.petscmat.mult(source, target)
    assert fd.norm(action - matrix_action) < 3.0e-10

    test = fd.TestFunction(space)
    covector = fd.assemble(fd.inner(direction, test) * fd.dx)
    adjoint = fd.assemble(fd.action(fd.adjoint(jacobian), covector))
    reference = fd.Cofunction(space.dual())
    with covector.dat.vec_ro as source, reference.dat.vec as target:
        matrix.petscmat.multTranspose(source, target)
    with adjoint.dat.vec_ro as actual, reference.dat.vec_ro as expected:
        difference = actual.copy()
        difference.axpy(-1.0, expected)
        assert difference.norm() < 3.0e-10


def distributed_matfree_check(expected_size: int) -> None:
    mesh = fd.UnitSquareMesh(2, 2)
    assert mesh.comm.size == expected_size
    space = fd.FunctionSpace(mesh, "CG", 1)
    x, y = fd.SpatialCoordinate(mesh)
    u = fd.Function(space).interpolate(0.2 + x * (1.0 - y))
    operator = RieszFractionalLaplacian(
        u,
        0.3,
        assembly="matfree",
        source_evaluation="hybrid",
        source_quadrature_degree=8,
        target_quadrature_degree=3,
    )
    action = fd.assemble(operator)
    test = fd.TestFunction(space)
    distributed_weak = fd.assemble(fd.inner(action, test) * fd.dx)

    manager = operator.operator_data["manager"]
    with u.dat.vec_ro as vector:
        start, end = vector.getOwnershipRange()
        local_values = np.asarray(vector.array_r).copy()
    parts = mesh.comm.allgather((start, end, local_values))
    coefficients = np.empty(space.dim())
    for part_start, part_end, values in parts:
        coefficients[part_start:part_end] = values
    serial_reference = manager.backend.apply(coefficients)

    with distributed_weak.dat.vec_ro as vector:
        local_start, local_end = vector.getOwnershipRange()
        np.testing.assert_allclose(
            vector.array_r,
            serial_reference[local_start:local_end],
            rtol=3.0e-11,
            atol=3.0e-11,
        )
    direction = fd.Function(space).interpolate(0.1 + x * y)
    assert_distributed_linearization(space, u, operator, direction)


def distributed_hmatrix_check(expected_size: int, degree: int) -> None:
    mesh = fd.UnitSquareMesh(4, 4)
    assert mesh.comm.size == expected_size
    space = fd.FunctionSpace(mesh, "CG", degree)
    x, y = fd.SpatialCoordinate(mesh)
    u = fd.Function(space).interpolate(0.2 + x * (1.0 - y) + 0.1 * x * y)
    reference = fd.assemble(
        RieszFractionalLaplacian(
            u,
            0.3,
            assembly="matfree",
            source_evaluation="hybrid",
            source_quadrature_degree=8,
            target_quadrature_degree=3,
        )
    )
    operator = RieszFractionalLaplacian(
        u,
        0.3,
        assembly="hmatrix",
        source_evaluation="hybrid",
        source_quadrature_degree=8,
        target_quadrature_degree=3,
        compression_tolerance=1.0e-8,
        leaf_size=2,
    )
    action = fd.assemble(operator)
    assert fd.norm(action - reference) < 2.0e-7 * max(
        1.0,
        fd.norm(reference),
    )
    diagnostics = operator.diagnostics()
    assert diagnostics["distribution"] == "rank_block"
    assert diagnostics["replicated_source_dofs"] == 0
    assert diagnostics["admissible_blocks"] > 0
    if expected_size == 2 and degree == 1:
        direction = fd.Function(space).interpolate(0.1 + x * y)
        assert_distributed_linearization(
            space,
            u,
            operator,
            direction,
        )


def distributed_tetrahedral_hmatrix_check(expected_size: int, degree: int) -> None:
    mesh = fd.UnitCubeMesh(1, 1, 1)
    assert mesh.comm.size == expected_size
    space = fd.FunctionSpace(mesh, "CG", degree)
    x, y, z = fd.SpatialCoordinate(mesh)
    source = fd.Function(space).interpolate(
        0.2 + x * (1.0 - y) + 0.1 * z + 0.05 * x * z
    )
    reference = fd.assemble(
        RieszFractionalLaplacian(
            source,
            0.3,
            assembly="matfree",
            source_evaluation="hybrid",
            source_quadrature_degree=8,
            target_quadrature_degree=1,
        )
    )
    operator = RieszFractionalLaplacian(
        source,
        0.3,
        assembly="hmatrix",
        source_evaluation="hybrid",
        source_quadrature_degree=8,
        target_quadrature_degree=1,
        compression_tolerance=1.0e-8,
        leaf_size=2,
    )
    action = fd.assemble(operator)
    assert fd.norm(action - reference) < 3.0e-10 * max(
        1.0,
        fd.norm(reference),
    )
    diagnostics = operator.diagnostics()
    assert diagnostics["distribution"] == "rank_block"
    assert diagnostics["replicated_source_dofs"] == 0


@pytest.mark.parallel(nprocs=2)
@pytest.mark.verification
def test_distributed_matfree_two_ranks() -> None:
    distributed_matfree_check(2)


@pytest.mark.parallel(nprocs=4)
@pytest.mark.verification
def test_distributed_matfree_four_ranks() -> None:
    distributed_matfree_check(4)


@pytest.mark.parallel(nprocs=2)
@pytest.mark.verification
@pytest.mark.parametrize("degree", [1, 2])
def test_distributed_hmatrix_two_ranks(degree: int) -> None:
    distributed_hmatrix_check(2, degree)


@pytest.mark.parallel(nprocs=2)
@pytest.mark.verification
@pytest.mark.parametrize("degree", [1, 2])
def test_distributed_tetrahedral_hmatrix_two_ranks(degree: int) -> None:
    distributed_tetrahedral_hmatrix_check(2, degree)


@pytest.mark.parallel(nprocs=2)
@pytest.mark.verification
def test_dense_backend_rejects_distributed_mesh() -> None:
    mesh = fd.UnitSquareMesh(1, 1)
    space = fd.FunctionSpace(mesh, "CG", 1)
    with pytest.raises(NotImplementedError, match="serial execution"):
        RieszFractionalLaplacian(
            fd.Function(space),
            0.3,
            assembly="dense",
        )


@pytest.mark.parallel(nprocs=4)
@pytest.mark.verification
def test_distributed_hmatrix_four_ranks() -> None:
    distributed_hmatrix_check(4, 1)
