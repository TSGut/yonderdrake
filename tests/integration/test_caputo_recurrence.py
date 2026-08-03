"""Integration tests for the eliminated Caputo recurrence."""

from __future__ import annotations

import copy
from math import gamma

import mpmath as mp
import numpy as np
import pytest

fd = pytest.importorskip("firedrake")

from yonderdrake import (  # noqa: E402
    BirkSong,
    CaputoDerivative,
    Diethelm2008,
    FractionalTimeStepper,
    FullHistory,
    Recurrence,
    SumOfExponentials,
)
from yonderdrake.time.coefficients import (  # noqa: E402
    quadratic_recurrence_coefficients,
    recurrence_coefficients,
)


def make_scalar_problem(
    representation,
    *,
    initial_value: float = 2.0,
    dt_value: float = 0.05,
    solver_parameters=None,
    formulation=None,
):
    mesh = fd.UnitIntervalMesh(4)
    space = fd.FunctionSpace(mesh, "CG", 1)
    u = fd.Function(space, name="u").assign(initial_value)
    v = fd.TestFunction(space)
    t = fd.Constant(0.0)
    dt = fd.Constant(dt_value)
    residual = fd.inner(CaputoDerivative(u, 0.6), v) * fd.dx
    options = {}
    if formulation is not None:
        options["formulation"] = formulation
    stepper = FractionalTimeStepper(
        residual,
        representation,
        t,
        dt,
        u,
        solver_parameters=solver_parameters,
        **options,
    )
    return u, t, dt, stepper


@pytest.mark.verification
@pytest.mark.parametrize("representation", [BirkSong(24), Diethelm2008(24)])
def test_constant_has_zero_caputo_derivative(representation) -> None:
    u, t, _, stepper = make_scalar_problem(representation)
    original_time = float(t)
    for _ in range(3):
        stepper.advance()
    assert fd.norm(u - 2.0) < 1.0e-11
    assert float(t) == original_time
    assert all(fd.norm(mode) < 1.0e-11 for mode in stepper.history)
    assert stepper.solver_stats()["solves"] == 3
    assert len(stepper.history) == representation.num_modes


@pytest.mark.verification
@pytest.mark.parametrize("interpolant", ["linear", "quadratic"])
def test_variable_step_matches_scalar_recurrence(interpolant: str) -> None:
    alpha = 0.45
    representation = BirkSong(48)
    mesh = fd.UnitIntervalMesh(2)
    space = fd.FunctionSpace(mesh, "CG", 1)
    u = fd.Function(space).assign(1.0)
    v = fd.TestFunction(space)
    t = fd.Constant(0.0)
    dt = fd.Constant(0.1)
    reaction = 0.8
    residual = (
        fd.inner(CaputoDerivative(u, alpha), v)
        + reaction * fd.inner(u, v)
    ) * fd.dx
    stepper = FractionalTimeStepper(
        residual,
        representation,
        t,
        dt,
        u,
        formulation=Recurrence(interpolant=interpolant),
    )

    spectrum = representation.spectrum(alpha)
    scalar_u = 1.0
    scalar_penultimate = scalar_u
    scalar_modes = np.zeros(representation.num_modes)
    previous_step_size = None
    for step_size in [0.1, 0.025, 0.2, 0.075]:
        dt.assign(step_size)
        if interpolant == "quadratic":
            decay, interpolation, old_interpolation, implicit, old_implicit = (
                quadratic_recurrence_coefficients(
                    spectrum,
                    step_size,
                    previous_step_size=previous_step_size,
                )
            )
        else:
            decay, interpolation, implicit = recurrence_coefficients(
                spectrum,
                step_size,
            )
            old_interpolation = np.zeros_like(interpolation)
            old_implicit = 0.0
        history = np.dot(spectrum.weights * decay, scalar_modes) + (
            old_implicit * (scalar_u - scalar_penultimate)
        )
        new_scalar_u = (implicit * scalar_u - history) / (
            implicit + reaction
        )
        scalar_modes = (
            decay * scalar_modes
            + interpolation * (new_scalar_u - scalar_u)
            + old_interpolation * (scalar_u - scalar_penultimate)
        )
        scalar_penultimate = scalar_u
        scalar_u = new_scalar_u
        previous_step_size = step_size
        stepper.advance()

    assert fd.norm(u - scalar_u) < 2.0e-11
    for field, value in zip(stepper.history, scalar_modes, strict=True):
        assert fd.norm(field - float(value)) < 2.0e-11
    assert stepper.solver_stats()["last_step_size"] == 0.075


@pytest.mark.verification
def test_sum_of_exponentials_guards_variable_steps_and_final_time() -> None:
    representation = SumOfExponentials(
        target_error=1.0e-4,
        t_final=0.15,
        min_step=0.05,
    )
    _, time, dt, stepper = make_scalar_problem(
        representation,
        dt_value=0.05,
    )
    for _ in range(3):
        stepper.advance()
        time.assign(time + dt)
    with pytest.raises(ValueError, match="exceeds.*t_final"):
        stepper.advance()

    _, _, short_dt, short_stepper = make_scalar_problem(
        representation,
        dt_value=0.025,
    )
    with pytest.raises(ValueError, match="below.*min_step"):
        short_stepper.advance()
    assert float(short_dt) == 0.025


@pytest.mark.verification
def test_sum_of_exponentials_agrees_with_full_history_relaxation() -> None:
    step_size = 0.05
    num_steps = 10
    representation = SumOfExponentials(
        target_error=1.0e-6,
        t_final=num_steps * step_size,
        min_step=step_size,
    )
    compressed_u, compressed_time, compressed_dt, compressed = (
        make_scalar_problem(representation, dt_value=step_size)
    )
    history_u, history_time, history_dt, history = make_scalar_problem(
        FullHistory(),
        dt_value=step_size,
    )
    for _ in range(num_steps):
        compressed.advance()
        history.advance()
        compressed_time.assign(compressed_time + compressed_dt)
        history_time.assign(history_time + history_dt)
    np.testing.assert_allclose(
        compressed_u.dat.data_ro,
        history_u.dat.data_ro,
        atol=2.0e-5,
        rtol=0.0,
    )


@pytest.mark.verification
def test_residual_time_is_evaluated_at_end_of_step() -> None:
    alpha = 0.5
    representation = BirkSong(16)
    mesh = fd.UnitIntervalMesh(1)
    space = fd.FunctionSpace(mesh, "CG", 1)
    u = fd.Function(space).assign(0.0)
    v = fd.TestFunction(space)
    t = fd.Constant(0.0)
    dt = fd.Constant(0.1)
    residual = (
        fd.inner(CaputoDerivative(u, alpha), v)
        - fd.inner(t, v)
    ) * fd.dx
    stepper = FractionalTimeStepper(
        residual,
        representation,
        t,
        dt,
        u,
        solver_parameters={"ksp_type": "preonly", "pc_type": "lu"},
    )
    spectrum = representation.spectrum(alpha)
    interpolation = -np.expm1(-spectrum.rates * float(dt)) / (
        spectrum.rates * float(dt)
    )
    implicit = float(np.dot(spectrum.weights, interpolation))

    stepper.advance()

    assert float(t) == 0.0
    assert fd.norm(u - float(dt) / implicit) < 2.0e-11


@pytest.mark.verification
def test_vector_field_uses_one_recurrence_per_component() -> None:
    alpha = 0.6
    step_size = 0.05
    reaction = 0.8
    representation = BirkSong(24)
    mesh = fd.UnitIntervalMesh(3)
    space = fd.VectorFunctionSpace(mesh, "CG", 1, dim=2)
    u = fd.Function(space).interpolate(fd.as_vector((1.0, -0.4)))
    initial = u.copy(deepcopy=True)
    test = fd.TestFunction(space)
    time = fd.Constant(0.0)
    dt = fd.Constant(step_size)
    residual = (
        fd.inner(CaputoDerivative(u, alpha), test)
        + reaction * fd.inner(u, test)
    ) * fd.dx
    stepper = FractionalTimeStepper(
        residual,
        representation,
        time,
        dt,
        u,
        solver_parameters={"ksp_type": "preonly", "pc_type": "lu"},
    )
    stepper.advance()

    spectrum = representation.spectrum(alpha)
    arguments = spectrum.rates * step_size
    interpolation = -np.expm1(-arguments) / arguments
    implicit = float(np.dot(spectrum.weights, interpolation))
    expected = initial.copy(deepcopy=True)
    expected *= implicit / (implicit + reaction)
    assert fd.norm(u - expected) < 2.0e-11


@pytest.mark.verification
def test_checkpoint_reset_and_restore() -> None:
    u, t, dt, stepper = make_scalar_problem(BirkSong(12))
    dt.assign(0.03)
    stepper.advance()
    t.assign(t + dt)
    checkpoint = copy.deepcopy(stepper.checkpoint_state())
    saved_u = u.copy(deepcopy=True)
    saved_history = stepper.history

    stepper.reset(7.0, t0=1.25)
    assert fd.norm(u - 7.0) < 1.0e-12
    assert all(fd.norm(mode) == 0.0 for mode in stepper.history)
    assert float(t) == 1.25

    stepper.restore_checkpoint(checkpoint)
    assert fd.norm(u - saved_u) < 1.0e-12
    assert float(t) == pytest.approx(0.03)
    for restored, saved in zip(stepper.history, saved_history, strict=True):
        assert fd.norm(restored - saved) < 1.0e-12


@pytest.mark.verification
def test_checkpoint_rejects_incompatible_spectrum_before_mutation() -> None:
    u, t, dt, stepper = make_scalar_problem(BirkSong(12))
    stepper.advance()
    t.assign(t + dt)
    checkpoint = copy.deepcopy(stepper.checkpoint_state())

    other_u, _, _, other = make_scalar_problem(Diethelm2008(12))
    committed = other_u.copy(deepcopy=True)
    with pytest.raises(ValueError, match="representation"):
        other.restore_checkpoint(checkpoint)
    assert fd.norm(other_u - committed) == 0.0

    scaled_u, _, _, scaled = make_scalar_problem(
        BirkSong(12, rate_scale=2.0)
    )
    scaled_committed = scaled_u.copy(deepcopy=True)
    with pytest.raises(ValueError, match="representation"):
        scaled.restore_checkpoint(checkpoint)
    assert fd.norm(scaled_u - scaled_committed) == 0.0

    linear_u, _, _, linear = make_scalar_problem(
        BirkSong(12),
        formulation=Recurrence(interpolant="linear"),
    )
    linear_committed = linear_u.copy(deepcopy=True)
    with pytest.raises(ValueError, match="formulation"):
        linear.restore_checkpoint(checkpoint)
    assert fd.norm(linear_u - linear_committed) == 0.0

    malformed = copy.deepcopy(checkpoint)
    malformed["lower_limit"] = float("nan")
    original = u.copy(deepcopy=True)
    with pytest.raises(ValueError, match="times must be finite"):
        stepper.restore_checkpoint(malformed)
    assert fd.norm(u - original) == 0.0

    malformed_stats = copy.deepcopy(checkpoint)
    malformed_stats["stats"]["solves"] = "not-an-integer"
    with pytest.raises(ValueError, match="solver statistics"):
        stepper.restore_checkpoint(malformed_stats)
    assert fd.norm(u - original) == 0.0


@pytest.mark.verification
def test_failed_solve_does_not_commit_state() -> None:
    mesh = fd.UnitIntervalMesh(4)
    space = fd.FunctionSpace(mesh, "CG", 1)
    u = fd.Function(space).assign(0.25)
    v = fd.TestFunction(space)
    t = fd.Constant(0.0)
    dt = fd.Constant(0.1)
    residual = (
        fd.inner(CaputoDerivative(u, 0.5), v)
        + fd.inner(u**3 - 10.0, v)
    ) * fd.dx
    stepper = FractionalTimeStepper(
        residual,
        Diethelm2008(8),
        t,
        dt,
        u,
        solver_parameters={"snes_max_it": 0},
    )
    committed_u = u.copy(deepcopy=True)
    committed_modes = stepper.history
    with pytest.raises(fd.ConvergenceError):
        stepper.advance()
    assert fd.norm(u - committed_u) == 0.0
    for current, committed in zip(
        stepper.history,
        committed_modes,
        strict=True,
    ):
        assert fd.norm(current - committed) == 0.0
    assert stepper.solver_stats()["failures"] == 1


@pytest.mark.verification
def test_nonlinear_residual_and_strong_dirichlet_condition() -> None:
    mesh = fd.UnitIntervalMesh(6)
    space = fd.FunctionSpace(mesh, "CG", 1)
    u = fd.Function(space).assign(0.8)
    v = fd.TestFunction(space)
    t = fd.Constant(0.0)
    dt = fd.Constant(0.1)
    residual = (
        fd.inner(CaputoDerivative(u, 0.5), v)
        + fd.inner(u**3, v)
        - fd.inner(1.0, v)
    ) * fd.dx
    bc = fd.DirichletBC(space, 1.0, "on_boundary")
    stepper = FractionalTimeStepper(
        residual,
        BirkSong(16),
        t,
        dt,
        u,
        bcs=bc,
    )
    stepper.advance()
    assert np.allclose(u.dat.data_ro[bc.nodes], 1.0)


@pytest.mark.verification
def test_marker_and_configuration_validation() -> None:
    mesh = fd.UnitIntervalMesh(2)
    space = fd.FunctionSpace(mesh, "CG", 1)
    u = fd.Function(space)
    v = fd.TestFunction(space)
    t = fd.Constant(0.0)
    dt = fd.Constant(0.1)
    ordinary = fd.inner(u, v) * fd.dx
    with pytest.raises(ValueError, match="at least one"):
        FractionalTimeStepper(ordinary, BirkSong(4), t, dt, u)
    doubled = (
        fd.inner(CaputoDerivative(u, 0.4), v)
        + fd.inner(CaputoDerivative(u, 0.6), v)
    ) * fd.dx
    multi_term = FractionalTimeStepper(doubled, BirkSong(4), t, dt, u)
    assert multi_term.solver_stats()["num_fractional_terms"] == 2
    with pytest.raises(ValueError, match="linear.*quadratic"):
        Recurrence(interpolant="constant")
    with pytest.raises(NotImplementedError, match="own history rule"):
        FractionalTimeStepper(
            doubled,
            FullHistory(),
            t,
            dt,
            u,
            formulation=Recurrence(),
        )


@pytest.mark.verification
def test_multiple_caputo_orders_share_one_physical_field() -> None:
    """Each term owns its spectrum while all terms use the same increment."""
    mesh = fd.UnitIntervalMesh(2)
    space = fd.FunctionSpace(mesh, "CG", 1)
    initial_value = 2.0
    step_size = 0.08
    coefficients = (0.35, 0.65)
    alphas = (0.25, 0.75)
    representation = BirkSong(12)
    u = fd.Function(space).assign(initial_value)
    v = fd.TestFunction(space)
    t = fd.Constant(0.0)
    dt = fd.Constant(step_size)
    residual = (
        sum(
            coefficient * fd.inner(CaputoDerivative(u, alpha), v)
            for coefficient, alpha in zip(
                coefficients,
                alphas,
                strict=True,
            )
        )
        + fd.inner(u, v)
    ) * fd.dx
    stepper = FractionalTimeStepper(
        residual,
        representation,
        t,
        dt,
        u,
        solver_parameters={"ksp_type": "preonly", "pc_type": "lu"},
    )
    stepper.advance()

    implicit_sum = 0.0
    for coefficient, alpha in zip(coefficients, alphas, strict=True):
        spectrum = representation.spectrum(alpha)
        arguments = spectrum.rates * step_size
        interpolation = -np.expm1(-arguments) / arguments
        implicit_sum += coefficient * float(
            np.dot(spectrum.weights, interpolation)
        )
    expected = initial_value * implicit_sum / (1.0 + implicit_sum)

    assert np.allclose(u.dat.data_ro, expected)
    assert len(stepper.term_histories) == 2
    assert all(len(modes) == 12 for modes in stepper.term_histories)


@pytest.mark.verification
def test_alpha_is_immutable_and_step_size_is_validated() -> None:
    mesh = fd.UnitIntervalMesh(2)
    space = fd.FunctionSpace(mesh, "CG", 1)
    u = fd.Function(space).assign(1.0)
    v = fd.TestFunction(space)
    t = fd.Constant(0.0)
    dt = fd.Constant(0.1)
    alpha = fd.Constant(0.6)
    residual = fd.inner(CaputoDerivative(u, alpha), v) * fd.dx
    stepper = FractionalTimeStepper(residual, BirkSong(6), t, dt, u)
    alpha.assign(0.7)
    with pytest.raises(RuntimeError, match="changing alpha"):
        stepper.advance()

    alpha.assign(0.6)
    dt.assign(0.0)
    with pytest.raises(ValueError, match="finite and positive"):
        stepper.advance()


def power_solution_error(num_modes: int, step_size: float) -> float:
    alpha = 0.55
    power = 2.0
    mesh = fd.UnitIntervalMesh(1)
    space = fd.FunctionSpace(mesh, "CG", 1)
    u = fd.Function(space).assign(0.0)
    v = fd.TestFunction(space)
    t = fd.Constant(0.0)
    dt = fd.Constant(step_size)
    source = fd.Constant(0.0)
    residual = (
        fd.inner(CaputoDerivative(u, alpha), v) - fd.inner(source, v)
    ) * fd.dx
    stepper = FractionalTimeStepper(residual, BirkSong(num_modes), t, dt, u)
    coefficient = gamma(power + 1.0) / gamma(power + 1.0 - alpha)
    num_steps = round(1.0 / step_size)
    for index in range(1, num_steps + 1):
        target_time = index * step_size
        source.assign(coefficient * target_time ** (power - alpha))
        stepper.advance()
        t.assign(t + dt)
    return abs(float(u.dat.data_ro[0]) - 1.0)


@pytest.mark.verification
@pytest.mark.slow
def test_power_function_converges_in_modes_and_timestep() -> None:
    coarse_modes = power_solution_error(4, 0.05)
    fine_modes = power_solution_error(64, 0.05)
    assert fine_modes < coarse_modes

    coarse_step = power_solution_error(64, 0.05)
    fine_step = power_solution_error(64, 0.025)
    assert fine_step < coarse_step
    assert fine_step < 4.0e-3


def mittag_leffler(alpha: float, argument: float) -> float:
    with mp.workdps(80):
        order = mp.mpf(str(alpha))
        value = mp.mpf(str(argument))
        total = mp.mpf(0)
        term = mp.mpf(1)
        index = 0
        while abs(term) > mp.mpf("1e-70"):
            term = value**index / mp.gamma(order * index + 1)
            total += term
            index += 1
            if index > 10000:
                raise RuntimeError("Mittag-Leffler series did not converge")
        return float(total)


def relaxation_error(num_modes: int, step_size: float) -> float:
    alpha = 0.6
    mesh = fd.UnitIntervalMesh(1)
    space = fd.FunctionSpace(mesh, "CG", 1)
    u = fd.Function(space).assign(1.0)
    v = fd.TestFunction(space)
    t = fd.Constant(0.0)
    dt = fd.Constant(step_size)
    residual = (
        fd.inner(CaputoDerivative(u, alpha), v) + fd.inner(u, v)
    ) * fd.dx
    stepper = FractionalTimeStepper(residual, Diethelm2008(num_modes), t, dt, u)
    for _ in range(round(1.0 / step_size)):
        stepper.advance()
        t.assign(t + dt)
    exact = mittag_leffler(alpha, -1.0)
    return abs(float(u.dat.data_ro[0]) - exact)


@pytest.mark.verification
@pytest.mark.slow
def test_mittag_leffler_relaxation_converges() -> None:
    coarse_modes = relaxation_error(16, 0.05)
    fine_modes = relaxation_error(64, 0.05)
    assert fine_modes < coarse_modes

    coarse_step = relaxation_error(64, 0.1)
    fine_step = relaxation_error(64, 0.05)
    assert fine_step < coarse_step
    assert fine_step < 2.0e-2
