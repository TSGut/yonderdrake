"""Two- and four-rank checks for distributed spectral shifted solves."""

from __future__ import annotations

import pytest

fd = pytest.importorskip("firedrake")

from yonderdrake import SpectralFractionalLaplacian  # noqa: E402


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
    assert fd.norm(action - matrix_action) < 2.0e-8

    test = fd.TestFunction(space)
    covector = fd.assemble(fd.inner(direction, test) * fd.dx)
    adjoint = fd.assemble(fd.action(fd.adjoint(jacobian), covector))
    reference = fd.Cofunction(space.dual())
    with covector.dat.vec_ro as source, reference.dat.vec as target:
        matrix.petscmat.multTranspose(source, target)
    with adjoint.dat.vec_ro as actual, reference.dat.vec_ro as expected:
        difference = actual.copy()
        difference.axpy(-1.0, expected)
        assert difference.norm() < 2.0e-8


def distributed_spectral_check(expected_size: int) -> None:
    mesh = fd.UnitSquareMesh(4, 4)
    assert mesh.comm.size == expected_size
    space = fd.FunctionSpace(mesh, "CG", 1)
    x, y = fd.SpatialCoordinate(mesh)
    u = fd.Function(space).interpolate(fd.sin(fd.pi * x) * fd.sin(fd.pi * y))
    bc = fd.DirichletBC(space, 0.0, "on_boundary")
    order = 0.4
    operator = SpectralFractionalLaplacian(
        u,
        order,
        bcs=bc,
        sinc_truncation_target=1.0e-4,
        shift_cache="all",
        shift_solver_parameters={"ksp_type": "preonly", "pc_type": "lu"},
    )
    action = fd.assemble(operator)
    continuum = fd.Function(space).interpolate((2.0 * fd.pi**2) ** order * u)
    relative_error = fd.norm(action - continuum) / fd.norm(continuum)
    assert relative_error < 0.1
    fd.assemble(operator)
    assert operator.diagnostics()["cache_reuses"] == 1
    direction = fd.Function(space).interpolate(
        (1.0 + x) * fd.sin(fd.pi * x) * fd.sin(fd.pi * y)
    )
    assert_distributed_linearization(space, u, operator, direction)

    neumann_mode = fd.Function(space).interpolate(fd.cos(fd.pi * x))
    neumann_operator = SpectralFractionalLaplacian(
        neumann_mode,
        order,
        sinc_truncation_target=1.0e-4,
        shift_cache="all",
        shift_solver_parameters={"ksp_type": "preonly", "pc_type": "lu"},
    )
    neumann_action = fd.assemble(neumann_operator)
    neumann_continuum = fd.Function(space).interpolate(
        (fd.pi**2) ** order * neumann_mode
    )
    neumann_error = fd.norm(neumann_action - neumann_continuum) / fd.norm(
        neumann_continuum
    )
    assert neumann_error < 0.1

    constant = fd.Function(space).assign(1.0)
    constant_action = fd.assemble(
        SpectralFractionalLaplacian(
            constant,
            order,
            sinc_truncation_target=1.0e-4,
            shift_cache="all",
            shift_solver_parameters={
                "ksp_type": "preonly",
                "pc_type": "lu",
            },
        )
    )
    assert fd.norm(constant_action) < 2.0e-9


@pytest.mark.parallel(nprocs=2)
@pytest.mark.verification
def test_distributed_spectral_two_ranks() -> None:
    distributed_spectral_check(2)


@pytest.mark.parallel(nprocs=4)
@pytest.mark.verification
def test_distributed_spectral_four_ranks() -> None:
    distributed_spectral_check(4)
