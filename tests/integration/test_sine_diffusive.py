"""Integration tests for sine diffusive oscillator stepping."""

from __future__ import annotations

import copy

import numpy as np
import pytest

fd = pytest.importorskip("firedrake")

from yonderdrake import (  # noqa: E402
    BirkSong,
    CaputoDerivative,
    FractionalTimeStepper,
    Oscillator,
    Recurrence,
    SineDiffusive,
)
from yonderdrake.time.coefficients import oscillator_coefficients  # noqa: E402


def _make_stepper(*, num_modes: int = 12, step_size: float = 0.1):
    mesh = fd.UnitIntervalMesh(2)
    space = fd.FunctionSpace(mesh, "CG", 1)
    solution = fd.Function(space).assign(1.0)
    test = fd.TestFunction(space)
    time = fd.Constant(0.0)
    dt = fd.Constant(step_size)
    alpha = 0.6
    reaction = 0.8
    residual = (
        fd.inner(CaputoDerivative(solution, alpha), test)
        + reaction * fd.inner(solution, test)
    ) * fd.dx
    representation = SineDiffusive(num_modes)
    stepper = FractionalTimeStepper(
        residual,
        representation,
        time,
        dt,
        solution,
    )
    return solution, time, dt, stepper, representation, alpha, reaction


@pytest.mark.verification
def test_variable_steps_match_independent_oscillator_update() -> None:
    solution, _, dt, stepper, representation, alpha, reaction = _make_stepper()
    spectrum = representation.spectrum(alpha)
    scalar_value = 1.0
    positions = np.zeros(representation.num_modes)
    velocities = np.zeros(representation.num_modes)
    for step_size in (0.1, 0.025, 0.2, 0.075):
        dt.assign(step_size)
        (
            cosine,
            sine_over_frequency,
            negative_frequency_sine,
            position_forcing,
            velocity_forcing,
            implicit_weight,
        ) = oscillator_coefficients(spectrum, alpha, step_size)
        history = np.dot(
            spectrum.weights,
            cosine * positions + sine_over_frequency * velocities,
        )
        updated = (
            implicit_weight * scalar_value - history
        ) / (implicit_weight + reaction)
        increment = updated - scalar_value
        old_positions = positions.copy()
        positions = (
            cosine * old_positions
            + sine_over_frequency * velocities
            + position_forcing * increment
        )
        velocities = (
            negative_frequency_sine * old_positions
            + cosine * velocities
            + velocity_forcing * increment
        )
        scalar_value = updated
        stepper.advance()

    assert fd.norm(solution - scalar_value) < 2.0e-11
    for field, expected in zip(stepper.history, positions, strict=True):
        assert fd.norm(field - float(expected)) < 2.0e-11
    for field, expected in zip(
        stepper.oscillator_velocities,
        velocities,
        strict=True,
    ):
        assert fd.norm(field - float(expected)) < 2.0e-11
    assert stepper.solver_stats()["fields_per_mode"] == 2


@pytest.mark.verification
def test_checkpoint_and_reset_cover_both_oscillator_fields() -> None:
    solution, time, dt, source, *_ = _make_stepper(num_modes=6)
    for _ in range(3):
        source.advance()
        time.assign(time + dt)
    state = copy.deepcopy(source.checkpoint_state())

    target_solution, target_time, _, target, *_ = _make_stepper(num_modes=6)
    target.restore_checkpoint(state)
    np.testing.assert_allclose(
        target_solution.dat.data_ro,
        solution.dat.data_ro,
    )
    assert float(target_time) == float(time)
    for left, right in zip(target.history, source.history, strict=True):
        np.testing.assert_allclose(left.dat.data_ro, right.dat.data_ro)
    for left, right in zip(
        target.oscillator_velocities,
        source.oscillator_velocities,
        strict=True,
    ):
        np.testing.assert_allclose(left.dat.data_ro, right.dat.data_ro)

    target.reset(2.0, 0.0)
    assert fd.norm(target_solution - 2.0) < 1.0e-14
    assert all(fd.norm(field) == 0.0 for field in target.history)
    assert all(
        fd.norm(field) == 0.0 for field in target.oscillator_velocities
    )


@pytest.mark.verification
def test_dispatch_rejects_incompatible_formulations() -> None:
    solution, time, dt, _, representation, alpha, _ = _make_stepper()
    test = fd.TestFunction(solution.function_space())
    residual = fd.inner(CaputoDerivative(solution, alpha), test) * fd.dx
    with pytest.raises(NotImplementedError, match="does not support Recurrence"):
        FractionalTimeStepper(
            residual,
            representation,
            time,
            dt,
            solution,
            formulation=Recurrence(),
        )
    with pytest.raises(NotImplementedError, match="does not support Oscillator"):
        FractionalTimeStepper(
            residual,
            BirkSong(4),
            time,
            dt,
            solution,
            formulation=Oscillator(),
        )
