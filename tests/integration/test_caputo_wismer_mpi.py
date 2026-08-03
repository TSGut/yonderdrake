"""Distributed Caputo-Wismer reconstruction checks."""

from __future__ import annotations

import numpy as np
import pytest

fd = pytest.importorskip("firedrake")

from yonderdrake.applications import (  # noqa: E402
    CaputoWismerInverseProblem,
    CaputoWismerMaterial,
    CaputoWismerModel,
    CaputoWismerPML,
    SensorArray,
    time_reverse_sensor_data,
)


@pytest.mark.parallel(nprocs=2)
@pytest.mark.verification
def test_distributed_caputo_wismer_reconstruction_two_ranks() -> None:
    mesh = fd.UnitSquareMesh(2, 2)
    assert mesh.comm.size == 2
    space = fd.FunctionSpace(mesh, "CG", 2)
    x, y = fd.SpatialCoordinate(mesh)
    truth = fd.Function(space).interpolate(
        fd.exp(-20.0 * ((x - 0.4) ** 2 + (y - 0.55) ** 2))
    )
    sensors = SensorArray.ring(
        space,
        5,
        0.32,
        width=0.18,
        center=(0.5, 0.5),
    )
    material = CaputoWismerMaterial(
        indicator=fd.Constant(1.0),
        density=1.0,
        wave_speed=1.0,
        damping=0.01,
        alpha=0.5,
    )
    solver_parameters = {
        "ksp_type": "cg",
        "ksp_rtol": 1.0e-10,
        "pc_type": "gamg",
    }
    model = CaputoWismerModel(
        space,
        materials=(material,),
        dt=0.02,
        num_steps=4,
        num_modes=3,
        sensors=sensors,
        solver_parameters=solver_parameters,
    )
    data = model.propagate(truth).sensor_data
    assert data is not None
    problem = CaputoWismerInverseProblem(model, data, regularization=1.0e-8)
    zero = fd.Function(space)
    initial_objective, _ = problem.objective_gradient(zero)
    result = problem.solve(
        max_iterations=50,
        tolerance=1.0e-6,
        positivity=True,
    )
    assert result.converged
    assert result.objective < initial_objective
    assert np.all(np.isfinite(result.pressure.dat.data_ro))
    lossless = time_reverse_sensor_data(
        model,
        data,
        compensate_attenuation=False,
        positivity=True,
    )
    assert np.linalg.norm(lossless.dat.data_ro) > 0.0
    assert np.all(np.isfinite(lossless.dat.data_ro))
    compensated = time_reverse_sensor_data(
        model,
        data,
        compensate_attenuation=True,
        filter_length=0.1,
        positivity=True,
    )
    assert np.linalg.norm(compensated.dat.data_ro) > 0.0
    assert np.all(np.isfinite(compensated.dat.data_ro))


@pytest.mark.parallel(nprocs=2)
@pytest.mark.verification
def test_distributed_caputo_wismer_pml_adjoint_two_ranks() -> None:
    mesh = fd.UnitSquareMesh(3, 3)
    assert mesh.comm.size == 2
    space = fd.FunctionSpace(mesh, "CG", 2)
    x, y = fd.SpatialCoordinate(mesh)
    truth = fd.Function(space).interpolate(x * (1.0 - x) * y * (1.0 - y))
    sensors = SensorArray.ring(
        space,
        4,
        0.22,
        width=0.16,
        center=(0.5, 0.5),
    )
    material = CaputoWismerMaterial(
        indicator=fd.Constant(1.0),
        density=1.2,
        wave_speed=1.0,
        damping=0.01,
        alpha=0.45,
    )
    pml = CaputoWismerPML.box(
        mesh,
        ((0.2, 0.8), (0.2, 0.8)),
        reference_speed=1.0,
        reflection=1.0e-3,
    )
    model = CaputoWismerModel(
        space,
        materials=(material,),
        dt=0.02,
        num_steps=2,
        num_modes=2,
        sensors=sensors,
        pml=pml,
    )
    values = np.arange(12, dtype=np.float64).reshape(3, 4) / 11.0
    data = model.propagate(truth).sensor_data
    assert data is not None
    observed = float(np.sum(data * values))
    expected = float(fd.assemble(truth * model.adjoint(values) * fd.dx))
    assert observed == pytest.approx(expected, rel=2.0e-9, abs=2.0e-9)
    compensated = time_reverse_sensor_data(
        model,
        data,
        filter_length=0.08,
    )
    assert np.all(np.isfinite(compensated.dat.data_ro))
