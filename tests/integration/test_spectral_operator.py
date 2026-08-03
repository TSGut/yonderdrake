"""Verification of the Firedrake spectral external operator."""

from __future__ import annotations

import numpy as np
import pytest

fd = pytest.importorskip("firedrake")
PETSc = pytest.importorskip("petsc4py").PETSc
scipy_linalg = pytest.importorskip("scipy.linalg")

from yonderdrake import SpectralFractionalLaplacian  # noqa: E402


def dense_reference(space, bc, u, order: float) -> np.ndarray:
    trial = fd.TrialFunction(space)
    test = fd.TestFunction(space)
    mass = fd.assemble(fd.inner(trial, test) * fd.dx, bcs=bc)
    stiffness = fd.assemble(
        fd.inner(fd.grad(trial), fd.grad(test)) * fd.dx,
        bcs=bc,
    )
    mass_array = mass.petscmat.convert("dense").getDenseArray()
    stiffness_array = stiffness.petscmat.convert("dense").getDenseArray()
    boundary = np.asarray(bc.nodes, dtype=np.int64)
    interior = np.setdiff1d(np.arange(space.dim()), boundary)
    mass_interior = mass_array[np.ix_(interior, interior)]
    stiffness_interior = stiffness_array[np.ix_(interior, interior)]
    eigenvalues, eigenvectors = scipy_linalg.eigh(
        stiffness_interior,
        mass_interior,
    )
    coefficients = (
        eigenvectors.T @ mass_interior @ u.dat.data_ro[interior]
    )
    result = np.zeros(space.dim())
    result[interior] = eigenvectors @ (eigenvalues**order * coefficients)
    return result


@pytest.mark.verification
def test_spectral_action_matches_generalized_eigendecomposition() -> None:
    mesh = fd.UnitIntervalMesh(6)
    space = fd.FunctionSpace(mesh, "CG", 1)
    x = fd.SpatialCoordinate(mesh)
    u = fd.Function(space).interpolate(fd.sin(fd.pi * x[0]))
    bc = fd.DirichletBC(space, 0.0, "on_boundary")
    order = 0.6
    operator = SpectralFractionalLaplacian(
        u,
        order,
        bcs=bc,
        sinc_truncation_target=1.0e-7,
        shift_cache="all",
        shift_solver_parameters={"ksp_type": "preonly", "pc_type": "lu"},
    )
    result = fd.assemble(operator)
    reference = dense_reference(space, bc, u, order)
    np.testing.assert_allclose(result.dat.data_ro, reference, rtol=2.0e-5)
    test = fd.TestFunction(space)
    weak_result = fd.assemble(fd.inner(operator, test) * fd.dx)
    weak_reference = fd.assemble(fd.inner(result, test) * fd.dx)
    with weak_result.dat.vec_ro as actual, weak_reference.dat.vec_ro as expected:
        difference = actual.copy()
        difference.axpy(-1.0, expected)
        assert difference.norm() < 1.0e-11

    first = operator.diagnostics()
    assert first["applications"] == 2
    fd.assemble(operator)
    second = operator.diagnostics()
    assert second["matrix_setups"] == first["matrix_setups"]
    assert second["cache_reuses"] == 2


@pytest.mark.verification
def test_spectral_symmetry_positivity_and_jacobian_action() -> None:
    mesh = fd.UnitIntervalMesh(5)
    space = fd.FunctionSpace(mesh, "CG", 1)
    x = fd.SpatialCoordinate(mesh)
    u = fd.Function(space).interpolate(x[0] * (1.0 - x[0]))
    direction = fd.Function(space).interpolate(
        x[0] * (1.0 - x[0]) * (1.0 + x[0])
    )
    bc = fd.DirichletBC(space, 0.0, "on_boundary")
    operator = SpectralFractionalLaplacian(
        u,
        0.4,
        bcs=bc,
        sinc_truncation_target=1.0e-6,
        shift_cache="all",
        shift_solver_parameters={"ksp_type": "preonly", "pc_type": "lu"},
    )
    lu = fd.assemble(operator)
    direction_operator = SpectralFractionalLaplacian(
        direction,
        0.4,
        bcs=bc,
        sinc_truncation_target=1.0e-6,
        shift_cache="all",
        shift_solver_parameters={"ksp_type": "preonly", "pc_type": "lu"},
    )
    lv = fd.assemble(direction_operator)
    symmetry_left = fd.assemble(fd.inner(u, lv) * fd.dx)
    symmetry_right = fd.assemble(fd.inner(lu, direction) * fd.dx)
    assert symmetry_left == pytest.approx(symmetry_right, rel=2.0e-8)
    assert fd.assemble(fd.inner(u, lu) * fd.dx) > 0.0

    jacobian = fd.derivative(operator, u, fd.TrialFunction(space))
    applied = fd.assemble(fd.action(jacobian, direction))
    np.testing.assert_allclose(applied.dat.data_ro, lv.dat.data_ro, rtol=2.0e-8)

    assembled_jacobian = fd.assemble(jacobian)
    matrix_applied = fd.Function(space)
    with direction.dat.vec_ro as source, matrix_applied.dat.vec as target:
        assembled_jacobian.petscmat.mult(source, target)
    np.testing.assert_allclose(
        matrix_applied.dat.data_ro,
        lv.dat.data_ro,
        rtol=2.0e-8,
        atol=2.0e-10,
    )

    test = fd.TestFunction(space)
    covector = fd.assemble(fd.inner(direction, test) * fd.dx)
    adjoint_applied = fd.assemble(fd.action(fd.adjoint(jacobian), covector))
    expected_adjoint = fd.Cofunction(space.dual())
    with covector.dat.vec_ro as source, expected_adjoint.dat.vec as target:
        assembled_jacobian.petscmat.multTranspose(source, target)
    with (
        adjoint_applied.dat.vec_ro as actual,
        expected_adjoint.dat.vec_ro as expected,
    ):
        difference = actual.copy()
        difference.axpy(-1.0, expected)
        assert difference.norm() < 2.0e-8


@pytest.mark.unit
def test_spectral_validation_is_immediate() -> None:
    with pytest.raises(TypeError, match="real scalar"):
        SpectralFractionalLaplacian(object(), object(), bcs=object())
    with pytest.raises(ValueError, match="0 < s < 1"):
        SpectralFractionalLaplacian(object(), 1.0, bcs=object())
    with pytest.raises(ValueError, match="shift_cache"):
        SpectralFractionalLaplacian(
            object(),
            0.5,
            bcs=object(),
            shift_cache="bounded",
        )
    with pytest.raises(TypeError, match="Firedrake Function"):
        SpectralFractionalLaplacian(object(), 0.5, bcs=object())

    mesh = fd.UnitSquareMesh(2, 2)
    space = fd.FunctionSpace(mesh, "CG", 1)
    u = fd.Function(space)
    with pytest.raises(ValueError, match="at least one"):
        SpectralFractionalLaplacian(u, 0.5, bcs=[])
    with pytest.raises(TypeError, match="DirichletBC"):
        SpectralFractionalLaplacian(u, 0.5, bcs=[object()])
    other_space = fd.FunctionSpace(mesh, "CG", 2)
    wrong_space = fd.DirichletBC(other_space, 0.0, "on_boundary")
    with pytest.raises(ValueError, match="u's space"):
        SpectralFractionalLaplacian(u, 0.5, bcs=wrong_space)
    nonzero_field = fd.Function(space).assign(1.0)
    nonzero = fd.DirichletBC(space, nonzero_field, "on_boundary")
    with pytest.raises(ValueError, match="homogeneous"):
        SpectralFractionalLaplacian(u, 0.5, bcs=nonzero)
    partial = fd.DirichletBC(space, 0.0, 1)
    with pytest.raises(ValueError, match="complete exterior boundary"):
        SpectralFractionalLaplacian(u, 0.5, bcs=partial)

    complete = [
        fd.DirichletBC(space, 0.0, boundary)
        for boundary in (1, 2, 3, 4)
    ]
    SpectralFractionalLaplacian(u, 0.5, bcs=complete)
    SpectralFractionalLaplacian(u, 0.5)
    with pytest.raises(TypeError, match="solver parameters"):
        SpectralFractionalLaplacian(
            u,
            0.5,
            bcs=complete,
            shift_solver_parameters=object(),
        )


@pytest.mark.verification
def test_spectral_neumann_eigenfunction_and_constant_nullspace() -> None:
    mesh = fd.UnitIntervalMesh(16)
    space = fd.FunctionSpace(mesh, "CG", 1)
    x = fd.SpatialCoordinate(mesh)
    order = 0.6

    eigenfunction = fd.Function(space).interpolate(fd.cos(fd.pi * x[0]))
    action = fd.assemble(
        SpectralFractionalLaplacian(
            eigenfunction,
            order,
            sinc_truncation_target=1.0e-8,
            shift_cache="all",
            shift_solver_parameters={
                "ksp_type": "preonly",
                "pc_type": "lu",
            },
        )
    )
    expected = fd.Function(space).interpolate(
        (fd.pi**2) ** order * fd.cos(fd.pi * x[0])
    )
    relative_error = fd.errornorm(expected, action) / fd.norm(expected)
    assert relative_error < 8.0e-3

    constant = fd.Function(space).assign(1.0)
    constant_action = fd.assemble(
        SpectralFractionalLaplacian(
            constant,
            order,
            sinc_truncation_target=1.0e-8,
            shift_cache="all",
            shift_solver_parameters={
                "ksp_type": "preonly",
                "pc_type": "lu",
            },
        )
    )
    assert fd.norm(constant_action) < 2.0e-10


@pytest.mark.verification
def test_spectral_neumann_balanced_source_sink() -> None:
    mesh = fd.UnitIntervalMesh(24)
    space = fd.FunctionSpace(mesh, "CG", 1)
    x = fd.SpatialCoordinate(mesh)[0]

    start = fd.Function(space).interpolate(
        fd.exp(-((x - 0.15) / 0.06) ** 2)
    )
    goal = fd.Function(space).interpolate(
        fd.exp(-((x - 0.85) / 0.06) ** 2)
    )
    start /= float(fd.assemble(start * fd.dx))
    goal /= float(fd.assemble(goal * fd.dx))
    forcing = fd.Function(space).assign(start - goal)
    assert abs(float(fd.assemble(forcing * fd.dx))) < 2.0e-14

    potential = fd.Function(space)
    test = fd.TestFunction(space)
    operator = SpectralFractionalLaplacian(
        potential,
        0.6,
        sinc_truncation_target=1.0e-5,
        shift_cache="all",
        shift_solver_parameters={"ksp_type": "preonly", "pc_type": "lu"},
    )
    residual = (fd.inner(operator, test) - fd.inner(forcing, test)) * fd.dx
    nullspace = fd.VectorSpaceBasis(constant=True, comm=mesh.comm)
    fd.solve(
        residual == 0,
        potential,
        nullspace=nullspace,
        transpose_nullspace=nullspace,
        solver_parameters={
            "snes_type": "ksponly",
            "mat_type": "matfree",
            "ksp_type": "gmres",
            "ksp_rtol": 1.0e-9,
            "pc_type": "python",
            "pc_python_type": "firedrake.MassInvPC",
            "Mp_pc_type": "lu",
        },
    )
    potential -= float(fd.assemble(potential * fd.dx))

    assert abs(float(fd.assemble(potential * fd.dx))) < 2.0e-12
    coordinates = np.asarray(mesh.coordinates.dat.data_ro).reshape(-1)
    start_node = int(np.argmin(np.abs(coordinates - 0.15)))
    goal_node = int(np.argmin(np.abs(coordinates - 0.85)))
    assert potential.dat.data_ro[start_node] > potential.dat.data_ro[goal_node]


@pytest.mark.unit
def test_spectral_operator_data_has_identity_semantics() -> None:
    mesh = fd.UnitIntervalMesh(2)
    space = fd.FunctionSpace(mesh, "CG", 1)
    u = fd.Function(space)
    bc = fd.DirichletBC(space, 0.0, "on_boundary")
    left = SpectralFractionalLaplacian(u, 0.5, bcs=bc)
    right = SpectralFractionalLaplacian(u, 0.5, bcs=bc)
    assert left != right
    assert len({left, right}) == 2


@pytest.mark.verification
def test_spectral_cache_invalidation_and_order_immutability() -> None:
    mesh = fd.UnitIntervalMesh(2)
    space = fd.FunctionSpace(mesh, "CG", 1)
    u = fd.Function(space).assign(1.0)
    bc = fd.DirichletBC(space, 0.0, "on_boundary")
    order = fd.Constant(0.5)
    operator = SpectralFractionalLaplacian(
        u,
        order,
        bcs=bc,
        sinc_truncation_target=0.1,
        shift_solver_parameters={"ksp_type": "preonly", "pc_type": "lu"},
    )
    assert operator.diagnostics()["applications"] == 0
    fd.assemble(operator)
    first_manager = operator.operator_data["manager"]
    operator.invalidate_cache()
    assert operator.diagnostics()["applications"] == 0
    fd.assemble(operator)
    assert operator.operator_data["manager"] is not first_manager

    manager = operator.operator_data["manager"]
    primal = fd.Function(space).interpolate(
        fd.SpatialCoordinate(mesh)[0]
        * (1.0 - fd.SpatialCoordinate(mesh)[0])
    )
    dual = manager.primal_to_dual(primal)
    recovered = manager.dual_to_primal(dual)
    assert fd.norm(recovered - primal) < 2.0e-12

    order.assign(0.6)
    with pytest.raises(RuntimeError, match="changing s"):
        fd.assemble(operator)


@pytest.mark.unit
def test_spectral_rejects_complex_petsc(monkeypatch) -> None:
    mesh = fd.UnitSquareMesh(1, 1)
    space = fd.FunctionSpace(mesh, "CG", 1)
    u = fd.Function(space)
    bc = fd.DirichletBC(space, 0.0, "on_boundary")
    monkeypatch.setattr(PETSc, "ScalarType", np.complex128)
    with pytest.raises(NotImplementedError, match="real binary64"):
        SpectralFractionalLaplacian(u, 0.5, bcs=bc)


@pytest.mark.verification
def test_spectral_stream_cache_has_bounded_setups() -> None:
    mesh = fd.UnitIntervalMesh(3)
    space = fd.FunctionSpace(mesh, "CG", 1)
    x = fd.SpatialCoordinate(mesh)
    u = fd.Function(space).interpolate(x[0] * (1.0 - x[0]))
    bc = fd.DirichletBC(space, 0.0, "on_boundary")
    operator = SpectralFractionalLaplacian(
        u,
        0.5,
        bcs=bc,
        sinc_truncation_target=1.0e-4,
        shift_solver_parameters={"ksp_type": "preonly", "pc_type": "lu"},
        mass_solver_parameters={"ksp_type": "preonly", "pc_type": "lu"},
    )
    first = fd.assemble(operator)
    reference = dense_reference(space, bc, u, 0.5)
    np.testing.assert_allclose(first.dat.data_ro, reference, rtol=2.0e-3)
    diagnostics = operator.diagnostics()
    assert diagnostics["shift_cache"] == "stream"
    assert diagnostics["cached_shift_solvers"] == 0
    assert diagnostics["matrix_setups"] == 3
    assert diagnostics["mass_solver_parameters"]["pc_type"] == "lu"
    second = fd.assemble(operator)
    np.testing.assert_allclose(second.dat.data_ro, first.dat.data_ro)
