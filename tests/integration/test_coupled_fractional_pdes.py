"""End-to-end Caputo plus spatial fractional-operator checks."""

from __future__ import annotations

from math import gamma

import numpy as np
import pytest

fd = pytest.importorskip("firedrake")

from yonderdrake import (  # noqa: E402
    BirkSong,
    CaputoDerivative,
    FractionalTimeStepper,
    RieszFractionalLaplacian,
    SpectralFractionalLaplacian,
)

MATRIX_FREE_SOLVER = {
    "snes_type": "ksponly",
    "mat_type": "matfree",
    "ksp_type": "gmres",
    "pc_type": "none",
    "ksp_rtol": 1.0e-10,
}


@pytest.mark.verification
@pytest.mark.slow
def test_caputo_spectral_manufactured_eigenfunction() -> None:
    mesh = fd.UnitIntervalMesh(6)
    space = fd.FunctionSpace(mesh, "CG", 1)
    x = fd.SpatialCoordinate(mesh)
    u = fd.Function(space).assign(0.0)
    v = fd.TestFunction(space)
    t = fd.Constant(0.0)
    dt = fd.Constant(0.05)
    alpha = 0.5
    order = 0.4
    power = 2.0
    bc = fd.DirichletBC(space, 0.0, "on_boundary")
    source = fd.Function(space)
    spatial = SpectralFractionalLaplacian(
        u,
        order,
        bcs=bc,
        sinc_truncation_target=1.0e-4,
        shift_cache="all",
        shift_solver_parameters={"ksp_type": "preonly", "pc_type": "lu"},
    )
    residual = (
        fd.inner(CaputoDerivative(u, alpha), v)
        + fd.inner(spatial, v)
        - fd.inner(source, v)
    ) * fd.dx
    stepper = FractionalTimeStepper(
        residual,
        BirkSong(16),
        t,
        dt,
        u,
        bcs=bc,
        solver_parameters=MATRIX_FREE_SOLVER,
    )
    caputo_coefficient = gamma(power + 1.0) / gamma(power + 1.0 - alpha)
    for index in range(1, 5):
        target = index * float(dt)
        source.interpolate(
            (
                caputo_coefficient * target ** (power - alpha)
                + (fd.pi**2) ** order * target**power
            )
            * fd.sin(fd.pi * x[0])
        )
        stepper.advance()
        t.assign(t + dt)
    exact = fd.Function(space).interpolate(float(t) ** power * fd.sin(fd.pi * x[0]))
    relative_error = fd.norm(u - exact) / fd.norm(exact)
    assert relative_error < 8.0e-2


@pytest.mark.verification
@pytest.mark.slow
def test_caputo_riesz_zero_exterior_decay() -> None:
    mesh = fd.UnitSquareMesh(1, 1)
    space = fd.FunctionSpace(mesh, "CG", 1)
    x, y = fd.SpatialCoordinate(mesh)
    u = fd.Function(space).interpolate(1.0 + 0.1 * x - 0.05 * y)
    initial = u.copy(deepcopy=True)
    v = fd.TestFunction(space)
    t = fd.Constant(0.0)
    dt = fd.Constant(0.05)
    spatial = RieszFractionalLaplacian(
        u,
        0.3,
        target_quadrature_degree=3,
        assembly="matfree",
    )
    residual = (
        fd.inner(CaputoDerivative(u, 0.55), v) + fd.inner(spatial, v)
    ) * fd.dx
    stepper = FractionalTimeStepper(
        residual,
        BirkSong(8),
        t,
        dt,
        u,
        solver_parameters=MATRIX_FREE_SOLVER,
    )
    initial_energy = fd.assemble(fd.inner(initial, initial) * fd.dx)
    for _ in range(2):
        stepper.advance()
        t.assign(t + dt)
    final_energy = fd.assemble(fd.inner(u, u) * fd.dx)
    assert np.isfinite(final_energy)
    assert 0.0 < final_energy < initial_energy


def unit_disk_riesz_error(refinement_level: int) -> float:
    order = 0.3
    mesh = fd.UnitDiskMesh(refinement_level)
    space = fd.FunctionSpace(mesh, "CG", 1)
    u = fd.Function(space)
    operator = RieszFractionalLaplacian(
        u,
        order,
        target_quadrature_degree=5,
        assembly="dense",
    )
    fd.assemble(operator)
    matrix = operator.operator_data["manager"].backend.assemble()
    test = fd.TestFunction(space)
    right_hand_side = fd.assemble(fd.inner(1.0, test) * fd.dx)
    coefficients = np.linalg.solve(matrix, right_hand_side.dat.data_ro)
    u.dat.data[:] = coefficients
    x, y = fd.SpatialCoordinate(mesh)
    prefactor = 1.0 / (
        2.0 ** (2.0 * order) * gamma(1.0 + order) ** 2
    )
    exact = fd.Function(space).interpolate(
        prefactor * fd.max_value(1.0 - x * x - y * y, 0.0) ** order
    )
    return fd.norm(u - exact) / fd.norm(exact)


@pytest.mark.verification
@pytest.mark.slow
def test_zero_exterior_unit_disk_benchmark_refines() -> None:
    coarse = unit_disk_riesz_error(0)
    fine = unit_disk_riesz_error(1)
    assert fine < coarse
    assert fine < 0.5
