"""Optional composition with Irksome's classical time steppers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

fd = pytest.importorskip("firedrake")
irksome = pytest.importorskip("irksome")

from yonderdrake import (  # noqa: E402
    RieszFractionalLaplacian,
    SpectralFractionalLaplacian,
)


@pytest.mark.verification
def test_spectral_fractional_laplacian_inside_irksome_dt_form() -> None:
    mesh = fd.UnitIntervalMesh(8)
    space = fd.FunctionSpace(mesh, "CG", 1)
    x = fd.SpatialCoordinate(mesh)[0]
    u = fd.Function(space, name="u").interpolate(fd.sin(fd.pi * x))
    initial = u.copy(deepcopy=True)
    test = fd.TestFunction(space)
    boundary = fd.DirichletBC(space, 0.0, "on_boundary")
    time = fd.Constant(0.0)
    dt = fd.Constant(0.02)

    spatial = SpectralFractionalLaplacian(
        u,
        0.6,
        bcs=boundary,
        sinc_truncation_target=1.0e-8,
        shift_cache="all",
        shift_solver_parameters={"ksp_type": "preonly", "pc_type": "lu"},
    )
    spatial_initial = fd.assemble(spatial)
    eigenvalue = float(
        fd.assemble(fd.inner(spatial_initial, initial) * fd.dx)
        / fd.assemble(fd.inner(initial, initial) * fd.dx)
    )
    residual = (fd.inner(irksome.Dt(u), test) + fd.inner(spatial, test)) * fd.dx
    stepper = irksome.TimeStepper(
        residual,
        irksome.GaussLegendre(1),
        time,
        dt,
        u,
        bcs=boundary,
        solver_parameters={
            "mat_type": "matfree",
            "snes_type": "ksponly",
            "ksp_type": "gmres",
            "ksp_rtol": 1.0e-11,
            "pc_type": "none",
        },
    )
    stepper.advance()

    step_size = float(dt)
    amplification = (1.0 - 0.5 * step_size * eigenvalue) / (
        1.0 + 0.5 * step_size * eigenvalue
    )
    expected = initial.copy(deepcopy=True)
    expected *= amplification
    assert fd.norm(u - expected) < 2.0e-7
    assert np.allclose(u.dat.data_ro[boundary.nodes], 0.0)


@pytest.mark.verification
def test_vector_cg2_spectral_operator_inside_irksome_dt_form() -> None:
    mesh = fd.UnitIntervalMesh(4)
    space = fd.VectorFunctionSpace(mesh, "CG", 2, dim=2)
    x = fd.SpatialCoordinate(mesh)[0]
    u = fd.Function(space, name="u").interpolate(
        fd.as_vector(
            (
                fd.sin(fd.pi * x),
                0.4 * fd.sin(fd.pi * x),
            )
        )
    )
    test = fd.TestFunction(space)
    boundary = fd.DirichletBC(
        space,
        fd.as_vector((0.0, 0.0)),
        "on_boundary",
    )
    time = fd.Constant(0.0)
    dt = fd.Constant(0.01)
    spatial = SpectralFractionalLaplacian(
        u,
        0.7,
        bcs=boundary,
        sinc_truncation_target=1.0e-7,
        shift_cache="all",
        shift_solver_parameters={"ksp_type": "preonly", "pc_type": "lu"},
    )
    rotated = fd.as_vector((-spatial[1], spatial[0]))
    residual = fd.inner(irksome.Dt(u) + rotated, test) * fd.dx
    stepper = irksome.TimeStepper(
        residual,
        irksome.GaussLegendre(1),
        time,
        dt,
        u,
        bcs=boundary,
        solver_parameters={
            "mat_type": "matfree",
            "snes_type": "ksponly",
            "ksp_type": "gmres",
            "ksp_rtol": 1.0e-10,
            "pc_type": "python",
            "pc_python_type": "firedrake.MassInvPC",
            "Mp_pc_type": "lu",
        },
    )
    stepper.advance()

    assert np.all(np.isfinite(u.dat.data_ro))
    assert np.allclose(u.dat.data_ro[boundary.nodes], 0.0)
    assert spatial.diagnostics()["applications"] > 0


@pytest.mark.verification
def test_monotile_classical_schrodinger_preserves_mass() -> None:
    mesh_path = (
        Path(__file__).resolve().parents[2]
        / "demos"
        / "gallery"
        / "meshes"
        / "aperiodic-monotile.msh"
    )
    mesh = fd.Mesh(str(mesh_path), comm=fd.COMM_SELF)
    space = fd.VectorFunctionSpace(mesh, "CG", 1, dim=2)
    x, y = fd.SpatialCoordinate(mesh)
    envelope = fd.exp(-((x + 0.10) ** 2 / 0.70 + (y - 0.05) ** 2 / 0.58))
    phase = 2.2 * (x + 0.10) + 0.45 * (y - 0.05)
    state = fd.Function(space).interpolate(
        fd.as_vector((envelope * fd.cos(phase), envelope * fd.sin(phase)))
    )
    boundary = fd.DirichletBC(
        space,
        fd.as_vector((0.0, 0.0)),
        "on_boundary",
    )
    boundary.apply(state)
    test = fd.TestFunction(space)
    time = fd.Constant(0.0)
    dt = fd.Constant(0.01)
    residual = (
        fd.inner(irksome.Dt(state), test) * fd.dx
        + (
            -fd.inner(fd.grad(state[1]), fd.grad(test[0]))
            + fd.inner(fd.grad(state[0]), fd.grad(test[1]))
        )
        * fd.dx
    )
    stepper = irksome.TimeStepper(
        residual,
        irksome.GaussLegendre(1),
        time,
        dt,
        state,
        bcs=boundary,
        solver_parameters={
            "snes_type": "ksponly",
            "ksp_type": "preonly",
            "pc_type": "lu",
        },
    )
    initial_mass = fd.assemble(fd.inner(state, state) * fd.dx)
    for index in range(80):
        stepper.advance()
        time.assign((index + 1) * float(dt))
    final_mass = fd.assemble(fd.inner(state, state) * fd.dx)

    assert np.isclose(final_mass, initial_mass, rtol=2.0e-10)


@pytest.mark.verification
def test_riesz_fractional_laplacian_inside_irksome_dt_form() -> None:
    mesh = fd.UnitSquareMesh(2, 2)
    space = fd.FunctionSpace(mesh, "CG", 1)
    x, y = fd.SpatialCoordinate(mesh)
    u = fd.Function(space, name="u").interpolate(fd.sin(fd.pi * x) * fd.sin(fd.pi * y))
    initial_norm = fd.norm(u)
    test = fd.TestFunction(space)
    boundary = fd.DirichletBC(space, 0.0, "on_boundary")
    time = fd.Constant(0.0)
    dt = fd.Constant(0.01)
    spatial = RieszFractionalLaplacian(
        u,
        0.6,
        bcs=boundary,
        assembly="hmatrix",
        quadrature_degree=2,
        compression_tolerance=1.0e-3,
        leaf_size=4,
    )
    residual = (fd.inner(irksome.Dt(u), test) + fd.inner(spatial, test)) * fd.dx
    stepper = irksome.TimeStepper(
        residual,
        irksome.RadauIIA(1),
        time,
        dt,
        u,
        bcs=boundary,
        solver_parameters={
            "mat_type": "matfree",
            "snes_type": "ksponly",
            "ksp_type": "gmres",
            "ksp_rtol": 1.0e-10,
            "pc_type": "none",
        },
    )
    stepper.advance()

    assert fd.norm(u) < initial_norm
    assert np.allclose(u.dat.data_ro[boundary.nodes], 0.0)
    assert spatial.diagnostics()["applications"] > 0
