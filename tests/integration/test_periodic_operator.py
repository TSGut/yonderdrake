"""Firedrake integration for the periodic Fourier fractional Laplacian."""

from __future__ import annotations

import numpy as np
import pytest

fd = pytest.importorskip("firedrake")
PETSc = pytest.importorskip("petsc4py").PETSc

from yonderdrake import PeriodicFractionalLaplacian  # noqa: E402


@pytest.mark.unit
def test_periodic_action_matches_one_dimensional_fourier_modes() -> None:
    length = 2.0 * np.pi
    mesh = fd.PeriodicIntervalMesh(12, length)
    space = fd.FunctionSpace(mesh, "CG", 1)
    x = fd.SpatialCoordinate(mesh)[0]
    u = fd.Function(space).interpolate(fd.sin(2.0 * x) + 0.3 * fd.cos(3.0 * x))
    order = 0.41
    operator = PeriodicFractionalLaplacian(u, order)

    result = fd.assemble(operator)
    expected = fd.Function(space).interpolate(
        2.0 ** (2.0 * order) * fd.sin(2.0 * x)
        + 0.3 * 3.0 ** (2.0 * order) * fd.cos(3.0 * x)
    )

    assert fd.norm(result - expected) < 2.0e-12
    assert operator.diagnostics() == {
        "shape": (12,),
        "lengths": pytest.approx((length,)),
        "spacing": pytest.approx((length / 12.0,)),
        "fft_backend": "numpy-serial",
        "applications": 1,
        "ranks": 1,
    }


@pytest.mark.unit
def test_periodic_action_matches_rectangular_mode_and_weak_form() -> None:
    lengths = (3.0, 5.0)
    shape = (6, 5)
    mesh = fd.PeriodicRectangleMesh(
        *shape,
        *lengths,
        quadrilateral=True,
        reorder=False,
    )
    space = fd.FunctionSpace(mesh, "Q", 1)
    x, y = fd.SpatialCoordinate(mesh)
    u = fd.Function(space).interpolate(
        fd.sin(2.0 * fd.pi * x / lengths[0])
        * fd.cos(4.0 * fd.pi * y / lengths[1])
    )
    order = 0.6
    eigenvalue = (2.0 * np.pi / lengths[0]) ** 2
    eigenvalue += (4.0 * np.pi / lengths[1]) ** 2
    operator = PeriodicFractionalLaplacian(u, order)

    result = fd.assemble(operator)
    expected = fd.Function(space).interpolate(eigenvalue**order * u)
    test = fd.TestFunction(space)
    weak_result = fd.assemble(fd.inner(operator, test) * fd.dx)
    weak_expected = fd.assemble(fd.inner(expected, test) * fd.dx)

    assert fd.norm(result - expected) < 2.0e-12
    with weak_result.dat.vec_ro as actual, weak_expected.dat.vec_ro as reference:
        difference = actual.copy()
        difference.axpy(-1.0, reference)
        assert difference.norm() < 2.0e-12
    diagnostics = operator.diagnostics()
    assert diagnostics["shape"] == shape
    assert diagnostics["applications"] == 2


@pytest.mark.unit
def test_periodic_action_matches_three_dimensional_mode_and_weak_form() -> None:
    lengths = (2.0, 3.0, 4.0)
    shape = (6, 5, 4)
    mesh = fd.PeriodicBoxMesh(
        *shape,
        *lengths,
        hexahedral=True,
        reorder=False,
    )
    space = fd.FunctionSpace(mesh, "Q", 1)
    x, y, z = fd.SpatialCoordinate(mesh)
    u = fd.Function(space).interpolate(
        fd.sin(2.0 * fd.pi * x / lengths[0])
        * fd.cos(2.0 * fd.pi * y / lengths[1])
        * fd.cos(4.0 * fd.pi * z / lengths[2])
    )
    order = 0.43
    eigenvalue = (2.0 * np.pi / lengths[0]) ** 2
    eigenvalue += (2.0 * np.pi / lengths[1]) ** 2
    eigenvalue += (4.0 * np.pi / lengths[2]) ** 2
    operator = PeriodicFractionalLaplacian(u, order)

    result = fd.assemble(operator)
    expected = fd.Function(space).interpolate(eigenvalue**order * u)
    test = fd.TestFunction(space)
    weak_result = fd.assemble(fd.inner(operator, test) * fd.dx)
    weak_expected = fd.assemble(fd.inner(expected, test) * fd.dx)

    assert fd.norm(result - expected) < 3.0e-12
    with weak_result.dat.vec_ro as actual, weak_expected.dat.vec_ro as reference:
        difference = actual.copy()
        difference.axpy(-1.0, reference)
        assert difference.norm() < 3.0e-12
    assert operator.diagnostics()["shape"] == shape


@pytest.mark.unit
def test_periodic_linearization_matrix_and_adjoint() -> None:
    mesh = fd.PeriodicIntervalMesh(6, 1.0)
    space = fd.FunctionSpace(mesh, "CG", 1)
    x = fd.SpatialCoordinate(mesh)[0]
    u = fd.Function(space).interpolate(fd.sin(2.0 * fd.pi * x))
    direction = fd.Function(space).interpolate(
        fd.cos(2.0 * fd.pi * x) + 0.2 * fd.sin(4.0 * fd.pi * x)
    )
    operator = PeriodicFractionalLaplacian(u, 0.35)
    direction_operator = PeriodicFractionalLaplacian(direction, 0.35)
    expected = fd.assemble(direction_operator)
    jacobian = fd.derivative(operator, u, fd.TrialFunction(space))

    action = fd.assemble(fd.action(jacobian, direction))
    matrix = fd.assemble(jacobian)
    matrix_action = fd.Function(space)
    with direction.dat.vec_ro as source, matrix_action.dat.vec as target:
        matrix.petscmat.mult(source, target)

    assert fd.norm(action - expected) < 2.0e-12
    assert fd.norm(matrix_action - expected) < 2.0e-12

    test = fd.TestFunction(space)
    covector = fd.assemble(fd.inner(direction, test) * fd.dx)
    adjoint = fd.assemble(fd.action(fd.adjoint(jacobian), covector))
    reference = fd.Cofunction(space.dual())
    with covector.dat.vec_ro as source, reference.dat.vec as target:
        matrix.petscmat.multTranspose(source, target)
    with adjoint.dat.vec_ro as actual, reference.dat.vec_ro as expected_adjoint:
        difference = actual.copy()
        difference.axpy(-1.0, expected_adjoint)
        assert difference.norm() < 2.0e-12


@pytest.mark.unit
def test_periodic_operator_solves_in_a_variational_residual() -> None:
    mesh = fd.PeriodicIntervalMesh(8, 1.0)
    space = fd.FunctionSpace(mesh, "CG", 1)
    x = fd.SpatialCoordinate(mesh)[0]
    u = fd.Function(space, name="u")
    test = fd.TestFunction(space)
    order = 0.45
    eigenvalue = (2.0 * np.pi) ** 2
    forcing = fd.Function(space).interpolate(fd.sin(2.0 * fd.pi * x))
    operator = PeriodicFractionalLaplacian(u, order)
    residual = (
        fd.inner(u, test)
        + fd.inner(operator, test)
        - fd.inner(forcing, test)
    ) * fd.dx

    fd.solve(
        residual == 0,
        u,
        solver_parameters={
            "snes_type": "ksponly",
            "mat_type": "matfree",
            "ksp_type": "gmres",
            "pc_type": "none",
            "ksp_rtol": 1.0e-12,
        },
    )
    expected = fd.Function(space).interpolate(
        forcing / (1.0 + eigenvalue**order)
    )

    assert fd.norm(u - expected) < 2.0e-11


@pytest.mark.unit
def test_periodic_validation_rejects_wrong_spaces_and_meshes() -> None:
    interval = fd.UnitIntervalMesh(4)
    interval_space = fd.FunctionSpace(interval, "CG", 1)
    with pytest.raises(ValueError, match="fully periodic"):
        PeriodicFractionalLaplacian(fd.Function(interval_space), 0.5)

    triangular = fd.PeriodicRectangleMesh(3, 2, 1.0, 1.0)
    triangular_space = fd.FunctionSpace(triangular, "CG", 1)
    with pytest.raises(NotImplementedError, match="quadrilateral"):
        PeriodicFractionalLaplacian(fd.Function(triangular_space), 0.5)

    partial = fd.PeriodicRectangleMesh(
        3,
        2,
        1.0,
        1.0,
        direction="x",
        quadrilateral=True,
    )
    partial_space = fd.FunctionSpace(partial, "Q", 1)
    with pytest.raises(ValueError, match="fully periodic"):
        PeriodicFractionalLaplacian(fd.Function(partial_space), 0.5)

    periodic = fd.PeriodicRectangleMesh(
        3,
        2,
        1.0,
        1.0,
        quadrilateral=True,
    )
    quadratic = fd.FunctionSpace(periodic, "Q", 2)
    with pytest.raises(NotImplementedError, match="degree-one"):
        PeriodicFractionalLaplacian(fd.Function(quadratic), 0.5)
    vector = fd.VectorFunctionSpace(periodic, "Q", 1)
    with pytest.raises(NotImplementedError, match="scalar"):
        PeriodicFractionalLaplacian(fd.Function(vector), 0.5)

    tetrahedral = fd.PeriodicBoxMesh(3, 3, 3, 1.0, 1.0, 1.0)
    tetrahedral_space = fd.FunctionSpace(tetrahedral, "CG", 1)
    with pytest.raises(NotImplementedError, match="hexahedral"):
        PeriodicFractionalLaplacian(fd.Function(tetrahedral_space), 0.5)

    too_small = fd.PeriodicBoxMesh(
        3,
        2,
        3,
        1.0,
        1.0,
        1.0,
        hexahedral=True,
    )
    too_small_space = fd.FunctionSpace(too_small, "Q", 1)
    with pytest.raises(ValueError, match="at least three input cells"):
        PeriodicFractionalLaplacian(fd.Function(too_small_space), 0.5)


@pytest.mark.unit
def test_periodic_validation_rejects_nonuniform_coordinates() -> None:
    mesh = fd.PeriodicRectangleMesh(
        4,
        3,
        1.0,
        1.0,
        quadrilateral=True,
    )
    mesh.coordinates.dat.data[0, 0] += 0.03
    space = fd.FunctionSpace(mesh, "Q", 1)
    with pytest.raises(ValueError, match="uniform|Cartesian|wrapped"):
        PeriodicFractionalLaplacian(fd.Function(space), 0.5)


@pytest.mark.unit
def test_periodic_order_cache_and_scalar_validation(monkeypatch) -> None:
    with pytest.raises(TypeError, match="real scalar"):
        PeriodicFractionalLaplacian(object(), object())
    with pytest.raises(ValueError, match="0 < s < 1"):
        PeriodicFractionalLaplacian(object(), 0.0)
    with pytest.raises(TypeError, match="Firedrake Function"):
        PeriodicFractionalLaplacian(object(), 0.5)

    mesh = fd.PeriodicIntervalMesh(4, 1.0)
    space = fd.FunctionSpace(mesh, "CG", 1)
    u = fd.Function(space).assign(1.0)
    monkeypatch.setattr(PETSc, "ScalarType", np.complex128)
    with pytest.raises(NotImplementedError, match="real binary64"):
        PeriodicFractionalLaplacian(u, 0.5)


@pytest.mark.unit
def test_periodic_order_is_immutable_and_cache_can_be_rebuilt() -> None:
    mesh = fd.PeriodicIntervalMesh(4, 1.0)
    space = fd.FunctionSpace(mesh, "CG", 1)
    u = fd.Function(space).assign(1.0)
    order = fd.Constant(0.5)
    operator = PeriodicFractionalLaplacian(u, order)
    assert operator.diagnostics()["applications"] == 0
    fd.assemble(operator)
    manager = operator.operator_data["manager"]
    interpolated = manager.apply(2.0 * u)
    assert fd.norm(interpolated) < 2.0e-12
    operator.invalidate_cache()
    assert operator.diagnostics()["applications"] == 0
    fd.assemble(operator)
    assert operator.operator_data["manager"] is not manager
    order.assign(0.6)
    with pytest.raises(RuntimeError, match="changing s"):
        fd.assemble(operator)
