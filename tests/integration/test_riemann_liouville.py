"""Verification of the left Riemann-Liouville time derivative."""

from __future__ import annotations

import copy
from math import gamma

import numpy as np
import pytest

fd = pytest.importorskip("firedrake")

from yonderdrake import (  # noqa: E402
    AuxiliaryODE,
    BirkSong,
    Diethelm2008,
    FractionalTimeStepper,
    Recurrence,
    RiemannLiouvilleDerivative,
)


def make_constant_problem(
    representation,
    formulation,
    *,
    initial_value: float = 2.0,
    lower_limit: float = 0.0,
    step_size: float = 0.1,
):
    alpha = 0.6
    mesh = fd.UnitIntervalMesh(1)
    space = fd.FunctionSpace(mesh, "CG", 1)
    u = fd.Function(space, name="u").assign(initial_value)
    v = fd.TestFunction(space)
    t = fd.Constant(lower_limit)
    dt = fd.Constant(step_size)
    source = fd.Constant(0.0)
    residual = (
        fd.inner(RiemannLiouvilleDerivative(u, alpha), v)
        + fd.inner(u, v)
        - fd.inner(source, v)
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
    return alpha, u, t, dt, source, stepper


@pytest.mark.verification
@pytest.mark.parametrize("representation", [BirkSong(8), Diethelm2008(8)])
@pytest.mark.parametrize(
    "formulation",
    [
        Recurrence(),
        AuxiliaryODE(scheme="backward_euler"),
        AuxiliaryODE(scheme="trapezoidal"),
    ],
)
def test_constant_has_exact_riemann_liouville_derivative(
    representation,
    formulation,
) -> None:
    initial_value = 2.0
    alpha, u, t, dt, source, stepper = make_constant_problem(
        representation,
        formulation,
        initial_value=initial_value,
    )
    for index in range(1, 4):
        elapsed = index * float(dt)
        source.assign(
            initial_value * elapsed ** (-alpha) / gamma(1.0 - alpha)
            + initial_value
        )
        stepper.advance()
        t.assign(t + dt)

    assert fd.norm(u - initial_value) < 2.0e-11
    assert all(fd.norm(mode) < 2.0e-11 for mode in stepper.history)


@pytest.mark.verification
def test_reset_changes_lower_limit_and_initial_trace() -> None:
    alpha, u, t, dt, source, stepper = make_constant_problem(
        BirkSong(8),
        Recurrence(),
    )
    stepper.reset(3.0, t0=1.25)
    elapsed = float(dt)
    source.assign(3.0 * elapsed ** (-alpha) / gamma(1.0 - alpha) + 3.0)
    stepper.advance()

    assert float(t) == pytest.approx(1.25)
    assert fd.norm(u - 3.0) < 2.0e-11


@pytest.mark.verification
@pytest.mark.parametrize(
    "formulation",
    [Recurrence(), AuxiliaryODE(scheme="backward_euler")],
)
def test_checkpoint_preserves_lower_limit_and_initial_trace(formulation) -> None:
    alpha, u_a, t_a, dt_a, source_a, uninterrupted = make_constant_problem(
        Diethelm2008(6),
        formulation,
        initial_value=1.5,
        lower_limit=0.4,
    )
    source_a.assign(
        1.5 * float(dt_a) ** (-alpha) / gamma(1.0 - alpha) + 1.5
    )
    uninterrupted.advance()
    t_a.assign(t_a + dt_a)
    state = copy.deepcopy(uninterrupted.checkpoint_state())

    _, u_b, t_b, dt_b, source_b, restarted = make_constant_problem(
        Diethelm2008(6),
        formulation,
        initial_value=99.0,
        lower_limit=0.4,
    )
    restarted.restore_checkpoint(state)
    elapsed = 2.0 * float(dt_a)
    exact_source = 1.5 * elapsed ** (-alpha) / gamma(1.0 - alpha) + 1.5
    source_a.assign(exact_source)
    source_b.assign(exact_source)
    uninterrupted.advance()
    restarted.advance()

    np.testing.assert_allclose(u_a.dat.data_ro, u_b.dat.data_ro, atol=2.0e-11)
    assert float(t_b) == pytest.approx(float(t_a))
    assert float(dt_b) == pytest.approx(float(dt_a))
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
def test_checkpoint_rejects_a_different_fractional_operator() -> None:
    _, _, _, _, _, stepper = make_constant_problem(
        BirkSong(4),
        Recurrence(),
    )
    state = stepper.checkpoint_state()
    state["operator_kinds"] = ("caputo",)
    with pytest.raises(ValueError, match="operators do not match"):
        stepper.restore_checkpoint(state)
