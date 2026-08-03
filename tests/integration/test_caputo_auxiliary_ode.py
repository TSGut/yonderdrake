"""Integration tests for the monolithic auxiliary-ODE formulation."""

from __future__ import annotations

import copy
from math import gamma

import numpy as np
import pytest

fd = pytest.importorskip("firedrake")

from yonderdrake import (  # noqa: E402
    AuxiliaryODE,
    BirkSong,
    CaputoDerivative,
    Diethelm2008,
    FractionalTimeStepper,
    Recurrence,
    SumOfExponentials,
)


def make_relaxation_stepper(
    representation,
    formulation,
    *,
    step_size: float = 0.1,
):
    mesh = fd.UnitIntervalMesh(1)
    space = fd.FunctionSpace(mesh, "CG", 1)
    u = fd.Function(space).assign(1.0)
    v = fd.TestFunction(space)
    t = fd.Constant(0.0)
    dt = fd.Constant(step_size)
    residual = (
        fd.inner(CaputoDerivative(u, 0.6), v) + fd.inner(u, v)
    ) * fd.dx
    stepper = FractionalTimeStepper(
        residual,
        representation,
        t,
        dt,
        u,
        formulation=formulation,
        solver_parameters={"ksp_type": "preonly", "pc_type": "lu"},
    )
    return u, t, dt, stepper


@pytest.mark.verification
@pytest.mark.parametrize("representation_type", [BirkSong, Diethelm2008])
@pytest.mark.parametrize("scheme", ["backward_euler", "trapezoidal"])
def test_constant_solution_and_mixed_memory_contract(
    representation_type,
    scheme: str,
) -> None:
    mesh = fd.UnitIntervalMesh(1)
    space = fd.FunctionSpace(mesh, "CG", 1)
    u = fd.Function(space).assign(2.0)
    v = fd.TestFunction(space)
    t = fd.Constant(0.0)
    dt = fd.Constant(0.1)
    residual = fd.inner(CaputoDerivative(u, 0.4), v) * fd.dx
    stepper = FractionalTimeStepper(
        residual,
        representation_type(4),
        t,
        dt,
        u,
        formulation=AuxiliaryODE(scheme=scheme),
        solver_parameters={"ksp_type": "preonly", "pc_type": "lu"},
    )
    stepper.advance()
    assert fd.norm(u - 2.0) < 1.0e-11
    assert all(fd.norm(mode) < 1.0e-11 for mode in stepper.history)
    assert stepper.solver_stats()["stored_fields"] == 5
    context = stepper.appctx["yonderdrake"]
    assert context["physical_field"] == 0
    assert context["mode_fields"] == (1, 2, 3, 4)
    assert len(context["field_names"]) == 5
    assert float(t) == 0.0


@pytest.mark.verification
@pytest.mark.parametrize("scheme", ["backward_euler", "trapezoidal"])
def test_one_step_matches_independent_scalar_mode_system(scheme: str) -> None:
    representation = BirkSong(6)
    u, _, dt, stepper = make_relaxation_stepper(
        representation,
        AuxiliaryODE(scheme=scheme),
    )
    spectrum = representation.spectrum(0.6)
    z = spectrum.rates * float(dt)
    if scheme == "backward_euler":
        response = 1.0 / (1.0 + z)
    else:
        response = 1.0 / (1.0 + 0.5 * z)
    coefficient = np.dot(spectrum.weights, response)
    expected_u = coefficient / (coefficient + 1.0)
    expected_modes = response * (expected_u - 1.0)

    stepper.advance()
    assert fd.norm(u - float(expected_u)) < 2.0e-11
    for mode, expected in zip(stepper.history, expected_modes, strict=True):
        assert fd.norm(mode - float(expected)) < 2.0e-11


@pytest.mark.verification
@pytest.mark.parametrize("scheme", ["backward_euler", "trapezoidal"])
def test_sum_of_exponentials_supports_auxiliary_ode(scheme: str) -> None:
    representation = SumOfExponentials(
        target_error=0.1,
        t_final=0.2,
        min_step=0.1,
    )
    u, _, dt, stepper = make_relaxation_stepper(
        representation,
        AuxiliaryODE(scheme=scheme),
    )
    exact_local = float(dt) ** (-0.6) / gamma(1.4)
    expected = exact_local / (exact_local + 1.0)
    stepper.advance()
    assert fd.norm(u - expected) < 2.0e-11


@pytest.mark.verification
@pytest.mark.parametrize("scheme", ["backward_euler", "trapezoidal"])
def test_auxiliary_residual_time_is_evaluated_at_end_of_step(
    scheme: str,
) -> None:
    representation = Diethelm2008(5)
    mesh = fd.UnitIntervalMesh(1)
    space = fd.FunctionSpace(mesh, "CG", 1)
    u = fd.Function(space).assign(0.0)
    v = fd.TestFunction(space)
    t = fd.Constant(0.0)
    dt = fd.Constant(0.1)
    residual = (
        fd.inner(CaputoDerivative(u, 0.6), v)
        + fd.inner(u, v)
        - fd.inner(t, v)
    ) * fd.dx
    stepper = FractionalTimeStepper(
        residual,
        representation,
        t,
        dt,
        u,
        formulation=AuxiliaryODE(scheme=scheme),
        solver_parameters={"ksp_type": "preonly", "pc_type": "lu"},
    )
    z = representation.spectrum(0.6).rates * float(dt)
    response = (
        1.0 / (1.0 + z)
        if scheme == "backward_euler"
        else 1.0 / (1.0 + 0.5 * z)
    )
    coefficient = float(
        np.dot(representation.spectrum(0.6).weights, response)
    )

    stepper.advance()

    assert float(t) == 0.0
    assert fd.norm(u - float(dt) / (coefficient + 1.0)) < 2.0e-11


@pytest.mark.verification
@pytest.mark.parametrize("scheme", ["backward_euler", "trapezoidal"])
def test_checkpoint_restart_matches_uninterrupted(scheme: str) -> None:
    formulation = AuxiliaryODE(scheme=scheme)
    u_a, t_a, dt_a, uninterrupted = make_relaxation_stepper(
        Diethelm2008(5),
        formulation,
        step_size=0.08,
    )
    uninterrupted.advance()
    t_a.assign(t_a + dt_a)
    checkpoint = copy.deepcopy(uninterrupted.checkpoint_state())

    u_b, _, _, restarted = make_relaxation_stepper(
        Diethelm2008(5),
        formulation,
        step_size=0.08,
    )
    restarted.restore_checkpoint(checkpoint)
    uninterrupted.advance()
    restarted.advance()

    np.testing.assert_allclose(u_a.dat.data_ro, u_b.dat.data_ro, atol=2.0e-11)
    for mode_a, mode_b in zip(
        uninterrupted.history,
        restarted.history,
        strict=True,
    ):
        np.testing.assert_allclose(
            mode_a.dat.data_ro,
            mode_b.dat.data_ro,
            atol=2.0e-11,
        )


@pytest.mark.verification
def test_auxiliary_checkpoint_rejects_incompatible_spectrum() -> None:
    _, _, _, source = make_relaxation_stepper(
        BirkSong(5),
        AuxiliaryODE(),
    )
    source.advance()
    checkpoint = copy.deepcopy(source.checkpoint_state())
    target_u, _, _, target = make_relaxation_stepper(
        Diethelm2008(5),
        AuxiliaryODE(),
    )
    committed = target_u.copy(deepcopy=True)

    with pytest.raises(ValueError, match="representation"):
        target.restore_checkpoint(checkpoint)

    assert fd.norm(target_u - committed) == 0.0


def relaxation_value(formulation, step_size: float) -> float:
    u, t, dt, stepper = make_relaxation_stepper(
        BirkSong(6),
        formulation,
        step_size=step_size,
    )
    for _ in range(round(1.0 / step_size)):
        stepper.advance()
        t.assign(t + dt)
    return float(u.dat.data_ro[0])


@pytest.mark.verification
@pytest.mark.slow
@pytest.mark.parametrize("scheme", ["backward_euler", "trapezoidal"])
def test_auxiliary_and_recurrence_paths_are_asymptotically_consistent(
    scheme: str,
) -> None:
    coarse_step = 0.1
    fine_step = 0.05
    coarse_difference = abs(
        relaxation_value(AuxiliaryODE(scheme=scheme), coarse_step)
        - relaxation_value(Recurrence(), coarse_step)
    )
    fine_difference = abs(
        relaxation_value(AuxiliaryODE(scheme=scheme), fine_step)
        - relaxation_value(Recurrence(), fine_step)
    )
    assert fine_difference < coarse_difference


def auxiliary_power_value(scheme: str, step_size: float) -> float:
    alpha = 0.55
    reaction = 1.0
    mesh = fd.UnitIntervalMesh(1)
    space = fd.FunctionSpace(mesh, "CG", 1)
    u = fd.Function(space).assign(0.0)
    v = fd.TestFunction(space)
    t = fd.Constant(0.0)
    dt = fd.Constant(step_size)
    source = fd.Constant(0.0)
    residual = (
        fd.inner(CaputoDerivative(u, alpha), v)
        + reaction * fd.inner(u, v)
        - fd.inner(source, v)
    ) * fd.dx
    stepper = FractionalTimeStepper(
        residual,
        BirkSong(6),
        t,
        dt,
        u,
        formulation=AuxiliaryODE(scheme=scheme),
        solver_parameters={"ksp_type": "preonly", "pc_type": "lu"},
    )
    coefficient = 2.0 / gamma(3.0 - alpha)
    for index in range(1, round(1.0 / step_size) + 1):
        target_time = index * step_size
        source.assign(
            coefficient * target_time ** (2.0 - alpha)
            + reaction * target_time**2
        )
        stepper.advance()
        t.assign(t + dt)
    return float(u.dat.data_ro[0])


@pytest.mark.verification
@pytest.mark.slow
@pytest.mark.parametrize(
    ("scheme", "minimum_reduction"),
    [("backward_euler", 1.7), ("trapezoidal", 2.0)],
)
def test_auxiliary_expected_temporal_order(
    scheme: str,
    minimum_reduction: float,
) -> None:
    reference = auxiliary_power_value(scheme, 0.0125)
    coarse_error = abs(auxiliary_power_value(scheme, 0.2) - reference)
    fine_error = abs(auxiliary_power_value(scheme, 0.1) - reference)
    assert coarse_error / fine_error > minimum_reduction


@pytest.mark.unit
def test_auxiliary_configuration_validation() -> None:
    with pytest.raises(ValueError, match="scheme must"):
        AuxiliaryODE(scheme="forward_euler")
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        AuxiliaryODE(coupling="split")
