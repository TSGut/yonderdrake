"""Verification of the exact one-mode exponential-memory operator."""

from __future__ import annotations

import copy
from math import exp

import numpy as np
import pytest

fd = pytest.importorskip("firedrake")

from yonderdrake import (  # noqa: E402
    AuxiliaryODE,
    BirkSong,
    CaputoDerivative,
    CaputoFabrizioOperator,
    ExponentialMemory,
    ExponentialMemoryCompatibilityWarning,
    FractionalTimeStepper,
    TimeMemoryStepper,
)


def make_linear_response(
    *,
    decay_rate: float = 1.7,
    formulation=None,
):
    mesh = fd.UnitIntervalMesh(1)
    space = fd.FunctionSpace(mesh, "CG", 1)
    u = fd.Function(space, name="u").assign(0.0)
    test = fd.TestFunction(space)
    time = fd.Constant(0.0)
    dt = fd.Constant(0.1)
    source = fd.Constant(0.0)
    residual = (
        fd.inner(ExponentialMemory(u, decay_rate), test)
        - fd.inner(source, test)
    ) * fd.dx
    options = {}
    if formulation is not None:
        options["formulation"] = formulation
    with pytest.warns(ExponentialMemoryCompatibilityWarning):
        stepper = TimeMemoryStepper(
            residual,
            time,
            dt,
            u,
            solver_parameters={
                "mat_type": "aij",
                "ksp_type": "preonly",
                "pc_type": "lu",
            },
            **options,
        )
    return u, time, dt, source, stepper


@pytest.mark.verification
def test_variable_steps_are_exact_for_a_linear_history() -> None:
    decay_rate = 1.7
    u, time, dt, source, stepper = make_linear_response(
        decay_rate=decay_rate
    )
    elapsed = 0.0
    for step_size in (0.07, 0.13, 0.04, 0.21):
        dt.assign(step_size)
        elapsed += step_size
        expected_memory = (1.0 - exp(-decay_rate * elapsed)) / decay_rate
        source.assign(expected_memory)
        stepper.advance()
        time.assign(time + dt)
        assert fd.norm(u - elapsed) < 2.0e-12
        assert fd.norm(stepper.history[0] - expected_memory) < 2.0e-12

    stats = stepper.solver_stats()
    assert stats["num_modes"] == 1
    assert stats["num_fractional_terms"] == 0
    assert stats["num_exponential_memory_terms"] == 1


def quadratic_error(step_size: float) -> float:
    decay_rate = 0.8
    u, time, dt, source, stepper = make_linear_response(
        decay_rate=decay_rate
    )
    dt.assign(step_size)
    num_steps = round(1.0 / step_size)
    for index in range(1, num_steps + 1):
        target_time = index * step_size
        exact_memory = 2.0 * (
            target_time / decay_rate
            - (1.0 - exp(-decay_rate * target_time)) / decay_rate**2
        )
        source.assign(exact_memory)
        stepper.advance()
        time.assign(time + dt)
    return abs(float(u.dat.data_ro[0]) - 1.0)


@pytest.mark.verification
def test_quadratic_history_converges_under_time_refinement() -> None:
    coarse = quadratic_error(0.1)
    fine = quadratic_error(0.05)
    assert fine < 0.3 * coarse
    assert fine < 1.0e-3


@pytest.mark.verification
def test_caputo_fabrizio_wrapper_matches_its_exponential_definition() -> None:
    alpha = 0.4
    normalization = 1.3
    rate = alpha / (1.0 - alpha)
    gain = normalization / (1.0 - alpha)
    mesh = fd.UnitIntervalMesh(1)
    space = fd.FunctionSpace(mesh, "CG", 1)
    test = fd.TestFunction(space)
    time = fd.Constant(0.0)
    dt = fd.Constant(0.08)
    source = fd.Constant(0.2)

    fields = [fd.Function(space).assign(0.0) for _ in range(2)]
    residuals = [
        (
            fd.inner(
                CaputoFabrizioOperator(
                    fields[0],
                    alpha,
                    normalization=normalization,
                ),
                test,
            )
            - fd.inner(source, test)
        )
        * fd.dx,
        (
            gain
            * fd.inner(ExponentialMemory(fields[1], rate), test)
            - fd.inner(source, test)
        )
        * fd.dx,
    ]
    steppers = []
    for residual, field in zip(residuals, fields, strict=True):
        with pytest.warns(ExponentialMemoryCompatibilityWarning):
            steppers.append(
                TimeMemoryStepper(
                    residual,
                    time,
                    dt,
                    field,
                    solver_parameters={
                        "ksp_type": "preonly",
                        "pc_type": "lu",
                    },
                )
            )
    for stepper in steppers:
        stepper.advance()
    assert fd.norm(fields[0] - fields[1]) < 2.0e-12


@pytest.mark.verification
def test_mixed_caputo_and_exponential_markers_have_separate_modes() -> None:
    mesh = fd.UnitIntervalMesh(1)
    space = fd.FunctionSpace(mesh, "CG", 1)
    u = fd.Function(space).assign(1.0)
    test = fd.TestFunction(space)
    residual = (
        fd.inner(CaputoDerivative(u, 0.5), test)
        + fd.inner(ExponentialMemory(u, 2.0), test)
        + fd.inner(u, test)
    ) * fd.dx
    with pytest.warns(ExponentialMemoryCompatibilityWarning):
        stepper = TimeMemoryStepper(
            residual,
            fd.Constant(0.0),
            fd.Constant(0.1),
            u,
            representation=BirkSong(3),
            solver_parameters={"ksp_type": "preonly", "pc_type": "lu"},
        )
    stepper.advance()
    assert sorted(len(group) for group in stepper.term_histories) == [1, 3]
    assert stepper.solver_stats()["num_fractional_terms"] == 1
    assert stepper.solver_stats()["num_exponential_memory_terms"] == 1


@pytest.mark.verification
@pytest.mark.parametrize(
    "formulation",
    [None, AuxiliaryODE(scheme="backward_euler")],
)
def test_checkpoint_roundtrip(formulation) -> None:
    u, time, dt, source, stepper = make_linear_response(
        formulation=formulation
    )
    source.assign(0.0 if formulation is not None else 0.2)
    stepper.advance()
    time.assign(time + dt)
    state = copy.deepcopy(stepper.checkpoint_state())

    other_u, _, _, _, other = make_linear_response(
        formulation=formulation
    )
    other.restore_checkpoint(state)
    assert np.allclose(other_u.dat.data_ro, u.dat.data_ro)
    assert other.solver_stats()["solves"] == 1
    for observed, expected in zip(
        other.history,
        stepper.history,
        strict=True,
    ):
        assert np.allclose(observed.dat.data_ro, expected.dat.data_ro)


@pytest.mark.verification
def test_api_rejects_misrouting_and_mutable_rate() -> None:
    mesh = fd.UnitIntervalMesh(1)
    space = fd.FunctionSpace(mesh, "CG", 1)
    u = fd.Function(space)
    test = fd.TestFunction(space)
    time = fd.Constant(0.0)
    dt = fd.Constant(0.1)
    rate = fd.Constant(1.0)
    residual = fd.inner(ExponentialMemory(u, rate), test) * fd.dx

    with pytest.raises(ValueError, match="TimeMemoryStepper"):
        FractionalTimeStepper(residual, BirkSong(2), time, dt, u)
    with pytest.warns(ExponentialMemoryCompatibilityWarning):
        stepper = TimeMemoryStepper(residual, time, dt, u)
    rate.assign(2.0)
    with pytest.raises(RuntimeError, match="time-memory parameter"):
        stepper.advance()

    ordinary = fd.inner(u, test) * fd.dx
    with pytest.raises(ValueError, match="time-memory marker"):
        TimeMemoryStepper(ordinary, time, dt, u)


@pytest.mark.verification
def test_auxiliary_backward_euler_matches_scalar_mode_equation() -> None:
    decay_rate = 1.2
    reaction = 0.7
    step_size = 0.05
    mesh = fd.UnitIntervalMesh(1)
    space = fd.FunctionSpace(mesh, "CG", 1)
    u = fd.Function(space).assign(1.0)
    test = fd.TestFunction(space)
    residual = (
        fd.inner(ExponentialMemory(u, decay_rate), test)
        + reaction * fd.inner(u, test)
    ) * fd.dx
    with pytest.warns(ExponentialMemoryCompatibilityWarning):
        stepper = TimeMemoryStepper(
            residual,
            fd.Constant(0.0),
            fd.Constant(step_size),
            u,
            formulation=AuxiliaryODE(scheme="backward_euler"),
            solver_parameters={
                "mat_type": "aij",
                "ksp_type": "preonly",
                "pc_type": "lu",
            },
        )
    stepper.advance()

    implicit = 1.0 / (1.0 + decay_rate * step_size)
    expected = implicit / (implicit + reaction)
    assert np.allclose(u.dat.data_ro, expected)
