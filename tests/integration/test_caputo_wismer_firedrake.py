"""Firedrake tests for the Caputo-Wismer acoustic application."""

from __future__ import annotations

import numpy as np
import pytest

fd = pytest.importorskip("firedrake")

from yonderdrake import SineDiffusive, SumOfExponentials  # noqa: E402
from yonderdrake.applications import (  # noqa: E402
    CaputoWismerArraySource,
    CaputoWismerImpedanceBoundary,
    CaputoWismerInverseProblem,
    CaputoWismerMaterial,
    CaputoWismerModel,
    CaputoWismerPML,
    CaputoWismerSource,
    CaputoWismerStepper,
    SensorArray,
    reconstruct_initial_pressure,
    time_reverse_sensor_data,
)


def _material(
    *,
    indicator: object | None = None,
    density: object = 1.0,
    wave_speed: object = 1.0,
    damping: object = 0.01,
    alpha: float = 0.5,
) -> CaputoWismerMaterial:
    return CaputoWismerMaterial(
        indicator=fd.Constant(1.0) if indicator is None else indicator,
        density=density,
        wave_speed=wave_speed,
        damping=damping,
        alpha=alpha,
    )


@pytest.mark.parametrize("dimension", [2, 3])
def test_sensor_sampling_and_lifting_are_adjoint(dimension: int) -> None:
    mesh = fd.UnitSquareMesh(3, 3) if dimension == 2 else fd.UnitCubeMesh(2, 2, 2)
    space = fd.FunctionSpace(mesh, "CG", 1)
    coordinates = fd.SpatialCoordinate(mesh)
    field = fd.Function(space).interpolate(sum(coordinates) + 0.25)
    sensors = (
        SensorArray.ring(space, 2, 0.2, width=0.2, center=(0.5, 0.5))
        if dimension == 2
        else SensorArray.sphere(
            space,
            2,
            0.2,
            width=0.2,
            center=(0.5, 0.5, 0.5),
        )
    )
    coefficients = np.asarray((0.7, -1.2))
    observed = float(np.dot(sensors.sample(field), coefficients))
    backprojected = sensors.adjoint_field(coefficients)
    expected = float(fd.assemble(field * backprojected * fd.dx))
    assert observed == pytest.approx(expected, rel=2.0e-13, abs=2.0e-13)


def test_sensor_array_rejects_incompatible_spaces_and_unresolved_sensors() -> None:
    mesh = fd.UnitSquareMesh(1, 1)
    scalar = fd.FunctionSpace(mesh, "CG", 1)
    vector = fd.VectorFunctionSpace(mesh, "CG", 1)
    discontinuous = fd.FunctionSpace(mesh, "DG", 0)
    with pytest.raises(NotImplementedError, match="continuous Lagrange"):
        SensorArray(discontinuous, ((0.5, 0.5),), width=0.2)
    with pytest.raises(NotImplementedError, match="scalar"):
        SensorArray(vector, ((0.5, 0.5),), width=0.2)
    for locations in ((), ((0.5,),), ((np.nan, 0.5),)):
        with pytest.raises(ValueError, match="locations"):
            SensorArray(scalar, locations, width=0.2)
    with pytest.raises(TypeError, match="real scalar"):
        SensorArray(scalar, ((0.5, 0.5),), width=object())
    with pytest.raises(ValueError, match="positive"):
        SensorArray(scalar, ((0.5, 0.5),), width=0.0)
    with pytest.raises(ValueError, match="no resolvable support"):
        SensorArray(scalar, ((1.0e6, 1.0e6),), width=1.0e-3)


def test_sensor_array_validates_sampling_and_adjoint_buffers() -> None:
    mesh = fd.UnitSquareMesh(1, 1)
    space = fd.FunctionSpace(mesh, "CG", 1)
    other = fd.FunctionSpace(mesh, "CG", 2)
    sensors = SensorArray(space, ((0.4, 0.5), (0.6, 0.5)), width=0.2)
    with pytest.raises(ValueError, match="sensor function space"):
        sensors.sample(fd.Function(other))
    for values, error in (
        ((1.0,), "shape"),
        ((1.0, np.nan), "finite"),
    ):
        with pytest.raises(ValueError, match=error):
            sensors.adjoint_field(values)
        with pytest.raises(ValueError, match=error):
            sensors.adjoint_covector(values)
    with pytest.raises(ValueError, match="sensor function space"):
        sensors.adjoint_field((1.0, -1.0), out=fd.Function(other))
    with pytest.raises(ValueError, match="dual sensor space"):
        sensors.adjoint_covector(
            (1.0, -1.0),
            out=fd.Cofunction(other.dual()),
        )
    field_out = fd.Function(space)
    covector_out = fd.Cofunction(space.dual())
    assert sensors.adjoint_field((1.0, -1.0), out=field_out) is field_out
    assert sensors.adjoint_covector((1.0, -1.0), out=covector_out) is covector_out


def test_caputo_wismer_stepper_advances_conservative_density_model() -> None:
    mesh = fd.UnitSquareMesh(2, 2)
    space = fd.FunctionSpace(mesh, "CG", 1)
    x, _ = fd.SpatialCoordinate(mesh)
    left = fd.conditional(x <= 0.5, 1.0, 0.0)
    right = 1.0 - left
    field = fd.Function(space).interpolate(x * (1.0 - x))
    initial = field.copy(deepcopy=True)
    stepper = CaputoWismerStepper(
        field,
        fd.Constant(0.0),
        fd.Constant(0.02),
        materials=(
            _material(indicator=left, density=1.0, alpha=0.3),
            _material(
                indicator=right,
                density=1.4,
                wave_speed=1.2,
                damping=0.02,
                alpha=0.6,
            ),
        ),
        num_modes=4,
        boundaries=(CaputoWismerImpedanceBoundary(coefficient=1.0),),
        solver_parameters={
            "snes_type": "ksponly",
            "ksp_type": "preonly",
            "pc_type": "lu",
        },
    )
    stepper.advance()
    assert np.all(np.isfinite(field.dat.data_ro))
    assert stepper.solver_stats()["num_fractional_terms"] == 2
    stepper.reset(initial, t0=0.0)
    np.testing.assert_allclose(field.dat.data_ro, initial.dat.data_ro)


def test_caputo_wismer_enforces_sum_of_exponentials_interval() -> None:
    mesh = fd.UnitSquareMesh(1, 1)
    space = fd.FunctionSpace(mesh, "CG", 1)
    material = (_material(),)
    with pytest.raises(ValueError, match="below.*min_step"):
        CaputoWismerModel(
            space,
            materials=material,
            dt=0.025,
            num_steps=2,
            representation=SumOfExponentials(
                target_error=0.1,
                t_final=0.1,
                min_step=0.05,
            ),
        )
    with pytest.raises(ValueError, match="exceeds.*t_final"):
        CaputoWismerModel(
            space,
            materials=material,
            dt=0.05,
            num_steps=2,
            representation=SumOfExponentials(
                target_error=0.1,
                t_final=0.075,
                min_step=0.05,
            ),
        )


def test_caputo_wismer_rejects_sine_diffusive_representation() -> None:
    mesh = fd.UnitSquareMesh(1, 1)
    space = fd.FunctionSpace(mesh, "CG", 1)
    material = (_material(),)
    with pytest.raises(NotImplementedError, match="time steppers only"):
        CaputoWismerModel(
            space,
            materials=material,
            dt=0.05,
            num_steps=2,
            representation=SineDiffusive(4),
        )
    field = fd.Function(space)
    with pytest.raises(NotImplementedError, match="time steppers only"):
        CaputoWismerStepper(
            field,
            fd.Constant(0.0),
            fd.Constant(0.05),
            materials=material,
            representation=SineDiffusive(4),
        )


def test_caputo_wismer_accepts_distinct_sum_of_exponential_mode_counts() -> None:
    mesh = fd.UnitSquareMesh(1, 1)
    space = fd.FunctionSpace(mesh, "CG", 1)
    x, _ = fd.SpatialCoordinate(mesh)
    left = fd.conditional(x <= 0.5, 1.0, 0.0)
    model = CaputoWismerModel(
        space,
        materials=(
            _material(indicator=left, alpha=0.2),
            _material(indicator=1.0 - left, alpha=0.8),
        ),
        dt=0.1,
        num_steps=2,
        representation=SumOfExponentials(
            target_error=0.1,
            t_final=0.2,
            min_step=0.1,
        ),
    )
    mode_counts = tuple(len(decay) for decay, _, _ in model._recurrence)
    assert mode_counts[0] != mode_counts[1]


@pytest.mark.parametrize("degree", [1, 2])
def test_pressure_model_and_adjoint_are_exact_with_density(degree: int) -> None:
    mesh = fd.UnitSquareMesh(3, 3)
    space = fd.FunctionSpace(mesh, "CG", degree)
    x, y = fd.SpatialCoordinate(mesh)
    left = fd.conditional(x <= 0.5, 1.0, 0.0)
    initial = fd.Function(space).interpolate(
        fd.sin(fd.pi * x) * fd.sin(2.0 * fd.pi * y)
    )
    sensors = SensorArray.ring(space, 4, 0.3, width=0.16, center=(0.5, 0.5))
    model = CaputoWismerModel(
        space,
        materials=(
            _material(indicator=left, density=0.9, alpha=0.4),
            _material(
                indicator=1.0 - left,
                density=1.3,
                wave_speed=1.15,
                damping=0.02,
                alpha=0.6,
            ),
        ),
        dt=0.025,
        num_steps=5,
        num_modes=4,
        sensors=sensors,
        boundaries=(CaputoWismerImpedanceBoundary(coefficient=1.0),),
    )
    rng = np.random.default_rng(3412)
    sensor_values = rng.standard_normal((6, sensors.num_sensors))
    observed_pairing = float(
        np.sum(model.propagate(initial).sensor_data * sensor_values)
    )
    adjoint_pairing = float(fd.assemble(initial * model.adjoint(sensor_values) * fd.dx))
    assert observed_pairing == pytest.approx(adjoint_pairing, rel=3.0e-10, abs=3.0e-10)


@pytest.mark.parametrize("dimension", [2, 3])
def test_pml_model_and_adjoint_are_exact(dimension: int) -> None:
    if dimension == 2:
        mesh = fd.UnitSquareMesh(4, 4)
        center = (0.5, 0.5)
        interior = ((0.2, 0.8),) * 2
    else:
        mesh = fd.UnitCubeMesh(2, 2, 2)
        center = (0.5, 0.5, 0.5)
        interior = ((0.2, 0.8),) * 3
    space = fd.FunctionSpace(mesh, "CG", 1)
    coordinates = fd.SpatialCoordinate(mesh)
    profile = coordinates[0] * (1.0 - coordinates[0])
    for axis in range(1, dimension):
        profile *= coordinates[axis] * (1.0 - coordinates[axis])
    initial = fd.Function(space).interpolate(profile)
    sensors = (
        SensorArray.ring(space, 3, 0.18, width=0.16, center=center)
        if dimension == 2
        else SensorArray.sphere(space, 3, 0.18, width=0.18, center=center)
    )
    pml = CaputoWismerPML.box(
        mesh,
        interior,
        reference_speed=1.0,
        reflection=1.0e-3,
    )
    model = CaputoWismerModel(
        space,
        materials=(_material(density=1.1),),
        dt=0.01,
        num_steps=2,
        num_modes=2,
        sensors=sensors,
        pml=pml,
    )
    rng = np.random.default_rng(781 + dimension)
    values = rng.standard_normal((3, sensors.num_sensors))
    observed = float(np.sum(model.propagate(initial).sensor_data * values))
    expected = float(fd.assemble(initial * model.adjoint(values) * fd.dx))
    assert observed == pytest.approx(expected, rel=3.0e-10, abs=3.0e-10)


def test_pml_box_profiles_and_validation() -> None:
    mesh = fd.UnitSquareMesh(2, 2)
    pml = CaputoWismerPML.box(
        mesh,
        ((0.2, 0.8), (0.25, 0.75)),
        outer_bounds=((0.0, 1.0), (0.0, 1.0)),
        reference_speed=1.5,
        reflection=1.0e-4,
        polynomial_order=2,
        outer_boundary=False,
    )
    space = fd.FunctionSpace(mesh, "CG", 1)
    profiles = [fd.Function(space).interpolate(value) for value in pml.damping]
    assert pml.interior_bounds == ((0.2, 0.8), (0.25, 0.75))
    assert pml.reflection == pytest.approx(1.0e-4)
    assert pml.polynomial_order == 2
    assert not pml.outer_boundary
    for profile in profiles:
        assert np.all(profile.dat.data_ro >= 0.0)
        assert np.max(profile.dat.data_ro) > 0.0

    invalid = (
        ({"interior_bounds": ((0.2, 0.8),), "reference_speed": 1.0}, "shape"),
        (
            {"interior_bounds": ((0.8, 0.2), (0.2, 0.8)), "reference_speed": 1.0},
            "positive width",
        ),
        (
            {"interior_bounds": ((0.2, 0.8),) * 2, "reference_speed": object()},
            "real scalar",
        ),
        (
            {"interior_bounds": ((0.2, 0.8),) * 2, "reference_speed": 0.0},
            "positive",
        ),
        (
            {
                "interior_bounds": ((0.2, 0.8),) * 2,
                "reference_speed": 1.0,
                "reflection": object(),
            },
            "real scalar",
        ),
        (
            {
                "interior_bounds": ((0.2, 0.8),) * 2,
                "reference_speed": 1.0,
                "reflection": 1.0,
            },
            "0 < reflection < 1",
        ),
        (
            {
                "interior_bounds": ((0.2, 0.8),) * 2,
                "reference_speed": 1.0,
                "polynomial_order": True,
            },
            "positive integer",
        ),
        (
            {
                "interior_bounds": ((0.0, 0.8), (0.2, 0.8)),
                "reference_speed": 1.0,
            },
            "positive PML width",
        ),
    )
    for arguments, error in invalid:
        with pytest.raises((TypeError, ValueError), match=error):
            CaputoWismerPML.box(mesh, **arguments)

    interval = fd.UnitIntervalMesh(2)
    with pytest.raises(NotImplementedError, match="2D or 3D"):
        CaputoWismerPML.box(
            interval,
            ((0.2, 0.8),),
            reference_speed=1.0,
        )


def test_pml_suppresses_the_returning_boundary_wave() -> None:
    mesh = fd.UnitSquareMesh(24, 24)
    space = fd.FunctionSpace(mesh, "CG", 1)
    x, y = fd.SpatialCoordinate(mesh)
    initial = fd.Function(space).interpolate(
        fd.exp(-180.0 * ((x - 0.5) ** 2 + (y - 0.5) ** 2))
    )
    sensor = SensorArray(space, ((0.5, 0.5),), width=0.07)
    parameters = {
        "materials": (_material(damping=0.0),),
        "dt": 0.008,
        "num_steps": 220,
        "num_modes": 1,
        "sensors": sensor,
    }
    reflecting = CaputoWismerModel(space, **parameters)
    pml = CaputoWismerPML.box(
        mesh,
        ((0.2, 0.8), (0.2, 0.8)),
        reference_speed=1.0,
        reflection=1.0e-4,
    )
    absorbing = CaputoWismerModel(space, pml=pml, **parameters)
    reflecting_trace = reflecting.propagate(initial).sensor_data[:, 0]
    absorbing_trace = absorbing.propagate(initial).sensor_data[:, 0]
    returning = slice(175, None)
    reflecting_return = np.linalg.norm(reflecting_trace[returning])
    absorbing_return = np.linalg.norm(absorbing_trace[returning])
    assert np.all(np.isfinite(absorbing_trace))
    assert np.max(np.abs(absorbing_trace)) < 2.0 * np.max(np.abs(initial.dat.data_ro))
    assert absorbing_return < 0.35 * reflecting_return


def test_volume_boundary_and_array_sources_drive_the_model() -> None:
    mesh = fd.UnitSquareMesh(3, 3)
    space = fd.FunctionSpace(mesh, "CG", 1)
    x, y = fd.SpatialCoordinate(mesh)
    array = SensorArray(
        space,
        ((0.35, 0.5), (0.65, 0.5)),
        width=0.14,
    )
    signal = np.asarray((0.0, 1.0, -0.3, 0.1))
    array_signals = np.column_stack((signal, -0.4 * signal))
    model = CaputoWismerModel(
        space,
        materials=(_material(damping=0.0),),
        dt=0.02,
        num_steps=3,
        sources=(
            CaputoWismerSource.volume(
                fd.exp(-30.0 * ((x - 0.5) ** 2 + (y - 0.5) ** 2)),
                signal,
            ),
            CaputoWismerSource.boundary(
                fd.Constant(0.2),
                signal,
                boundary_id=1,
            ),
            CaputoWismerArraySource(array=array, signals=array_signals),
        ),
    )
    result = model.propagate(fd.Function(space), record_history=True)
    assert len(result.field_history) == 4
    assert np.linalg.norm(result.final_pressure.dat.data_ro) > 0.0
    assert np.all(np.isfinite(result.final_pressure.dat.data_ro))


def test_affine_source_model_retains_exact_initial_pressure_adjoint() -> None:
    mesh = fd.UnitSquareMesh(2, 2)
    space = fd.FunctionSpace(mesh, "CG", 1)
    x, y = fd.SpatialCoordinate(mesh)
    initial = fd.Function(space).interpolate(x * (1.0 - y))
    sensors = SensorArray.ring(space, 3, 0.25, width=0.2, center=(0.5, 0.5))
    model = CaputoWismerModel(
        space,
        materials=(_material(),),
        dt=0.03,
        num_steps=3,
        sensors=sensors,
        sources=(CaputoWismerSource.volume(x * (1.0 - x), (0.0, 0.4, 0.2, 0.0)),),
    )
    zero_data = model.propagate(fd.Function(space)).sensor_data
    data = model.propagate(initial).sensor_data
    assert zero_data is not None and data is not None
    rng = np.random.default_rng(92)
    values = rng.standard_normal(data.shape)
    observed = float(np.sum((data - zero_data) * values))
    expected = float(fd.assemble(initial * model.adjoint(values) * fd.dx))
    assert observed == pytest.approx(expected, rel=3.0e-10, abs=3.0e-10)


def test_reversed_attenuation_is_filtered_and_has_exact_adjoint() -> None:
    mesh = fd.UnitSquareMesh(3, 3)
    space = fd.FunctionSpace(mesh, "CG", 1)
    x, y = fd.SpatialCoordinate(mesh)
    sensors = SensorArray.ring(space, 3, 0.25, width=0.18, center=(0.5, 0.5))
    initial = fd.Function(space).interpolate(
        fd.exp(-20.0 * ((x - 0.4) ** 2 + (y - 0.55) ** 2))
    )
    with pytest.raises(ValueError, match="filter"):
        CaputoWismerModel(
            space,
            materials=(_material(damping=0.03),),
            dt=0.02,
            num_steps=3,
            sensors=sensors,
            attenuation="reversed",
        )
    model = CaputoWismerModel(
        space,
        materials=(_material(damping=0.03),),
        dt=0.02,
        num_steps=3,
        num_modes=3,
        sensors=sensors,
        attenuation="reversed",
        attenuation_filter_length=0.08,
        attenuation_filter_order=2,
        pml=CaputoWismerPML.box(
            mesh,
            ((0.15, 0.85), (0.15, 0.85)),
            reference_speed=1.0,
            reflection=1.0e-2,
        ),
    )
    rng = np.random.default_rng(18)
    values = rng.standard_normal((4, sensors.num_sensors))
    observed = float(np.sum(model.propagate(initial).sensor_data * values))
    expected = float(fd.assemble(initial * model.adjoint(values) * fd.dx))
    assert observed == pytest.approx(expected, rel=3.0e-10, abs=3.0e-10)


def test_time_reversal_returns_finite_nonzero_images() -> None:
    mesh = fd.UnitSquareMesh(4, 4)
    space = fd.FunctionSpace(mesh, "CG", 1)
    x, y = fd.SpatialCoordinate(mesh)
    sensors = SensorArray.ring(space, 8, 0.38, width=0.13, center=(0.5, 0.5))
    truth = fd.Function(space).interpolate(
        fd.exp(-35.0 * ((x - 0.4) ** 2 + (y - 0.55) ** 2))
    )
    model = CaputoWismerModel(
        space,
        materials=(_material(damping=0.02),),
        dt=0.025,
        num_steps=8,
        num_modes=3,
        sensors=sensors,
        boundaries=(CaputoWismerImpedanceBoundary(coefficient=1.0),),
    )
    data = model.propagate(truth).sensor_data
    lossless = time_reverse_sensor_data(model, data, compensate_attenuation=False)
    compensated = reconstruct_initial_pressure(
        model,
        data,
        method="time_reversal",
        filter_length=0.1,
        positivity=False,
    )
    for image in (lossless, compensated):
        assert np.all(np.isfinite(image.dat.data_ro))
        assert np.linalg.norm(image.dat.data_ro) > 0.0
    assert not np.allclose(lossless.dat.data_ro, compensated.dat.data_ro)


def test_iterative_objective_gradient_passes_taylor_test_with_pml() -> None:
    mesh = fd.UnitSquareMesh(3, 3)
    space = fd.FunctionSpace(mesh, "CG", 1)
    x, y = fd.SpatialCoordinate(mesh)
    truth = fd.Function(space).interpolate(x * (1.0 - y))
    direction = fd.Function(space).interpolate(fd.sin(fd.pi * x) * fd.sin(fd.pi * y))
    sensors = SensorArray.ring(space, 3, 0.24, width=0.18, center=(0.5, 0.5))
    pml = CaputoWismerPML.box(
        mesh,
        ((0.15, 0.85), (0.15, 0.85)),
        reference_speed=1.0,
        reflection=1.0e-2,
    )
    model = CaputoWismerModel(
        space,
        materials=(_material(alpha=0.45),),
        dt=0.03,
        num_steps=3,
        num_modes=3,
        sensors=sensors,
        pml=pml,
    )
    problem = CaputoWismerInverseProblem(
        model,
        model.propagate(truth).sensor_data,
        regularization=2.0e-4,
    )
    candidate = fd.Function(space).interpolate(0.3 + 0.1 * x)
    objective, gradient = problem.objective_gradient(candidate)
    with (
        gradient.dat.vec_ro as gradient_vector,
        direction.dat.vec_ro as direction_vector,
    ):
        derivative = float(gradient_vector.dot(direction_vector))
    errors = []
    for epsilon in (1.0e-3, 5.0e-4, 2.5e-4):
        perturbed = candidate.copy(deepcopy=True)
        perturbed += epsilon * direction
        perturbed_objective, _ = problem.objective_gradient(perturbed)
        errors.append(abs(perturbed_objective - objective - epsilon * derivative))
    assert errors[1] < 0.3 * errors[0]
    assert errors[2] < 0.3 * errors[1]


def test_iterative_solver_reduces_sensor_misfit() -> None:
    mesh = fd.UnitSquareMesh(2, 2)
    space = fd.FunctionSpace(mesh, "CG", 1)
    x, y = fd.SpatialCoordinate(mesh)
    truth = fd.Function(space).interpolate(
        fd.exp(-18.0 * ((x - 0.4) ** 2 + (y - 0.55) ** 2))
    )
    sensors = SensorArray.ring(space, 5, 0.32, width=0.18, center=(0.5, 0.5))
    model = CaputoWismerModel(
        space,
        materials=(_material(),),
        dt=0.04,
        num_steps=5,
        num_modes=3,
        sensors=sensors,
    )
    problem = CaputoWismerInverseProblem(
        model,
        model.propagate(truth).sensor_data,
        regularization=1.0e-8,
    )
    initial_objective, _ = problem.objective_gradient(fd.Function(space))
    result = problem.solve(
        max_iterations=8,
        tolerance=1.0e-10,
        positivity=False,
        warm_start=False,
    )
    assert result.objective < 1.0e-3 * initial_objective
    assert result.iterations <= 8
    assert np.all(np.isfinite(result.pressure.dat.data_ro))


def test_inverse_problem_validates_data_and_solver_controls() -> None:
    mesh = fd.UnitSquareMesh(1, 1)
    space = fd.FunctionSpace(mesh, "CG", 1)
    sensors = SensorArray(space, ((0.5, 0.5),), width=0.25)
    model = CaputoWismerModel(
        space,
        materials=(_material(),),
        dt=0.05,
        num_steps=2,
        num_modes=2,
        sensors=sensors,
    )
    data = np.zeros((3, 1))
    with pytest.raises(TypeError, match="CaputoWismerModel"):
        CaputoWismerInverseProblem(object(), data)
    no_sensors = CaputoWismerModel(
        space,
        materials=(_material(),),
        dt=0.05,
        num_steps=2,
        num_modes=2,
    )
    with pytest.raises(ValueError, match="sensor array"):
        CaputoWismerInverseProblem(no_sensors, data)
    with pytest.raises(ValueError, match="shape"):
        CaputoWismerInverseProblem(model, np.zeros((2, 1)))
    invalid_data = data.copy()
    invalid_data[1, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        CaputoWismerInverseProblem(model, invalid_data)
    with pytest.raises(ValueError, match="nonnegative"):
        CaputoWismerInverseProblem(model, data, regularization=-1.0)

    problem = CaputoWismerInverseProblem(model, data)
    other = fd.FunctionSpace(mesh, "CG", 2)
    with pytest.raises(ValueError, match="inversion space"):
        problem.objective_gradient(fd.Function(other))
    with pytest.raises(ValueError, match="positive integer"):
        problem.solve(max_iterations=0)
    with pytest.raises(ValueError, match="positive"):
        problem.solve(tolerance=0.0)
    with pytest.raises(ValueError, match="initial_guess"):
        problem.solve(initial_guess=fd.Function(other))

    result = problem.solve(
        max_iterations=1,
        tolerance=1.0e-6,
        positivity=True,
        warm_start=True,
    )
    assert result.objective == pytest.approx(0.0)
    assert np.all(result.pressure.dat.data_ro >= 0.0)


def test_reconstruction_entry_points_cover_adjoint_and_time_reversal_controls() -> None:
    mesh = fd.UnitSquareMesh(2, 2)
    space = fd.FunctionSpace(mesh, "CG", 1)
    x, y = fd.SpatialCoordinate(mesh)
    sensors = SensorArray.ring(space, 3, 0.25, width=0.2, center=(0.5, 0.5))
    truth = fd.Function(space).interpolate(x * (1.0 - x) * y * (1.0 - y))
    model = CaputoWismerModel(
        space,
        materials=(_material(damping=0.015),),
        dt=0.03,
        num_steps=3,
        num_modes=2,
        sensors=sensors,
    )
    data = model.propagate(truth).sensor_data
    adjoint = reconstruct_initial_pressure(model, data, method="adjoint")
    default_filtered = time_reverse_sensor_data(
        model,
        data,
        compensate_attenuation=True,
        positivity=True,
    )
    assert np.linalg.norm(adjoint.dat.data_ro) > 0.0
    assert np.all(default_filtered.dat.data_ro >= 0.0)
    with pytest.raises(ValueError, match="only used"):
        time_reverse_sensor_data(
            model,
            data,
            compensate_attenuation=False,
            filter_length=0.1,
        )


def test_filtered_reverse_model_propagates_without_a_pml() -> None:
    mesh = fd.UnitSquareMesh(2, 2)
    space = fd.FunctionSpace(mesh, "CG", 1)
    x, y = fd.SpatialCoordinate(mesh)
    initial = fd.Function(space).interpolate(x * (1.0 - x) * y * (1.0 - y))
    model = CaputoWismerModel(
        space,
        materials=(_material(damping=0.01),),
        dt=0.02,
        num_steps=2,
        num_modes=2,
        attenuation="reversed",
        attenuation_filter_length=0.1,
        attenuation_filter_order=2,
        stiffness_theta=1.0,
    )
    result = model.propagate(initial)
    assert np.all(np.isfinite(result.final_pressure.dat.data_ro))
    assert np.linalg.norm(result.final_pressure.dat.data_ro) > 0.0


def test_source_validation_rejects_inconsistent_signals_and_spaces() -> None:
    mesh = fd.UnitSquareMesh(1, 1)
    space = fd.FunctionSpace(mesh, "CG", 1)
    profile = fd.Constant(1.0)
    base = {
        "space": space,
        "materials": (_material(),),
        "dt": 0.1,
        "num_steps": 2,
        "num_modes": 1,
    }
    invalid_sources = (
        (CaputoWismerSource.volume(profile, (0.0, 1.0)), "shape"),
        (CaputoWismerSource.volume(profile, (0.0, np.nan, 0.0)), "finite"),
        (
            CaputoWismerSource(
                profile=profile,
                signal=(0.0, 1.0, 0.0),
                region="volume",
                boundary_id=1,
            ),
            "boundary_id",
        ),
        (
            CaputoWismerSource(
                profile=profile,
                signal=(0.0, 1.0, 0.0),
                region="invalid",  # type: ignore[arg-type]
            ),
            "region",
        ),
        (object(), "sources must contain"),
    )
    for source, error in invalid_sources:
        with pytest.raises((TypeError, ValueError), match=error):
            CaputoWismerModel(**base, sources=(source,))

    array = SensorArray(space, ((0.5, 0.5),), width=0.2)
    array_cases = (
        (np.zeros((2, 1)), "shape"),
        (np.array([[0.0], [np.nan], [0.0]]), "finite"),
    )
    for signals, error in array_cases:
        with pytest.raises(ValueError, match=error):
            CaputoWismerModel(
                **base,
                sources=(CaputoWismerArraySource(array=array, signals=signals),),
            )

    other_mesh = fd.UnitSquareMesh(1, 1)
    other_space = fd.FunctionSpace(other_mesh, "CG", 1)
    other_array = SensorArray(other_space, ((0.5, 0.5),), width=0.2)
    with pytest.raises(ValueError, match="model space"):
        CaputoWismerModel(
            **base,
            sources=(
                CaputoWismerArraySource(
                    array=other_array,
                    signals=np.zeros((3, 1)),
                ),
            ),
        )


def test_caputo_wismer_stepper_validation_and_public_stepper() -> None:
    mesh = fd.UnitSquareMesh(1, 1)
    space = fd.FunctionSpace(mesh, "CG", 1)
    field = fd.Function(space)
    base = {"u": field, "t": fd.Constant(0.0), "dt": fd.Constant(0.1)}
    for changes, error in (
        ({"materials": ()}, "at least one"),
        ({"materials": (object(),)}, "CaputoWismerMaterial"),
        ({"materials": (_material(),), "num_modes": True}, "positive integer"),
        ({"materials": (_material(),), "stiffness_theta": object()}, "real scalar"),
        ({"materials": (_material(),), "stiffness_theta": 1.1}, "between"),
        ({"materials": (_material(),), "boundaries": (object(),)}, "ImpedanceBoundary"),
    ):
        with pytest.raises((TypeError, ValueError), match=error):
            CaputoWismerStepper(**base, **changes)

    vector = fd.Function(fd.VectorFunctionSpace(mesh, "CG", 1))
    with pytest.raises(NotImplementedError, match="scalar"):
        CaputoWismerStepper(
            vector,
            fd.Constant(0.0),
            fd.Constant(0.1),
            materials=(_material(),),
        )
    discontinuous = fd.Function(fd.FunctionSpace(mesh, "DG", 0))
    with pytest.raises(NotImplementedError, match="continuous Lagrange"):
        CaputoWismerStepper(
            discontinuous,
            fd.Constant(0.0),
            fd.Constant(0.1),
            materials=(_material(),),
        )
    stepper = CaputoWismerStepper(**base, materials=(_material(),), num_modes=1)
    assert stepper.fractional_stepper is not None


def test_model_validation() -> None:
    mesh = fd.UnitSquareMesh(1, 1)
    space = fd.FunctionSpace(mesh, "CG", 1)
    sensors = SensorArray(space, ((0.5, 0.5),), width=0.25)
    material = _material()
    base = {
        "materials": (material,),
        "dt": 0.1,
        "num_steps": 1,
        "num_modes": 1,
        "sensors": sensors,
    }
    for changes, error in (
        ({"dt": object()}, "real scalar"),
        ({"dt": -0.1}, "nonnegative"),
        ({"dt": 0.0}, "positive"),
        ({"num_steps": 0}, "positive integer"),
        ({"num_modes": 0}, "positive integer"),
        ({"stiffness_theta": -0.1}, "between"),
        ({"stiffness_theta": object()}, "real scalar"),
        ({"materials": ()}, "at least one"),
        ({"materials": (object(),)}, "CaputoWismerMaterial"),
        ({"attenuation": "invalid"}, "attenuation"),
        ({"attenuation_filter_length": 0.0}, "positive"),
        ({"attenuation_filter_length": 0.1}, "only used"),
        ({"pml": object()}, "CaputoWismerPML"),
        (
            {"pml": CaputoWismerPML(damping=(fd.Constant(0.0),))},
            "directional fields",
        ),
        ({"boundaries": (object(),)}, "ImpedanceBoundary"),
    ):
        arguments = dict(base)
        arguments.update(changes)
        with pytest.raises((TypeError, ValueError), match=error):
            CaputoWismerModel(space, **arguments)
    model = CaputoWismerModel(space, **base)
    other_mesh = fd.UnitSquareMesh(1, 1)
    other_space = fd.FunctionSpace(other_mesh, "CG", 1)
    other_sensors = SensorArray(other_space, ((0.5, 0.5),), width=0.2)
    with pytest.raises(ValueError, match="model space"):
        CaputoWismerModel(space, **{**base, "sensors": other_sensors})
    with pytest.raises(NotImplementedError, match="continuous Lagrange"):
        CaputoWismerModel(
            fd.FunctionSpace(mesh, "DG", 0),
            materials=(material,),
            dt=0.1,
            num_steps=1,
        )
    with pytest.raises(NotImplementedError, match="scalar pressure"):
        CaputoWismerModel(
            fd.VectorFunctionSpace(mesh, "CG", 1),
            materials=(material,),
            dt=0.1,
            num_steps=1,
        )
    with pytest.raises(NotImplementedError, match="2D or 3D"):
        CaputoWismerModel(
            fd.FunctionSpace(fd.UnitIntervalMesh(1), "CG", 1),
            materials=(material,),
            dt=0.1,
            num_steps=1,
        )
    with pytest.raises(ValueError, match="model space"):
        model.propagate(fd.Function(fd.FunctionSpace(mesh, "CG", 2)))
    with pytest.raises(ValueError, match="shape"):
        model.adjoint(np.zeros((1, 1)))
    invalid_values = np.zeros((2, 1))
    invalid_values[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        model.adjoint_covector(invalid_values)
    model_without_sensors = CaputoWismerModel(
        space,
        materials=(material,),
        dt=0.1,
        num_steps=1,
        num_modes=1,
    )
    with pytest.raises(ValueError, match="no sensor array"):
        model_without_sensors.adjoint_covector(np.zeros((2, 1)))
    with pytest.raises(ValueError, match="method"):
        reconstruct_initial_pressure(
            model,
            np.zeros((2, 1)),
            method="invalid",  # type: ignore[arg-type]
        )
