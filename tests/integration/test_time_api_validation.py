"""Fractional time-stepper API validation and lifecycle tests."""

from __future__ import annotations

import pytest

fd = pytest.importorskip("firedrake")

from yonderdrake import (  # noqa: E402
    AuxiliaryODE,
    BirkSong,
    CaputoDerivative,
    FractionalTimeStepper,
    FullHistory,
    Oscillator,
    Recurrence,
    RiemannLiouvilleDerivative,
    SineDiffusive,
)

TIME_VARIANTS = [
    pytest.param(BirkSong(2), Recurrence(), id="recurrence"),
    pytest.param(BirkSong(2), AuxiliaryODE(), id="auxiliary"),
    pytest.param(FullHistory(), None, id="full-history"),
    pytest.param(SineDiffusive(2), Oscillator(), id="oscillator"),
]


def make_stepper(
    mesh,
    representation,
    formulation,
    *,
    alpha=0.5,
    initial_value: float = 1.0,
    u0=None,
    solver_parameters=None,
    time_value: float = 0.0,
    dt_value: float = 0.1,
):
    space = fd.FunctionSpace(mesh, "CG", 1)
    u = fd.Function(space).assign(initial_value)
    test = fd.TestFunction(space)
    time = fd.Constant(time_value)
    dt = fd.Constant(dt_value)
    residual = (
        fd.inner(CaputoDerivative(u, alpha), test) + fd.inner(u, test)
    ) * fd.dx
    options = {}
    if formulation is not None:
        options["formulation"] = formulation
    stepper = FractionalTimeStepper(
        residual,
        representation,
        time,
        dt,
        u,
        u0=u0,
        solver_parameters=solver_parameters,
        **options,
    )
    return u, time, dt, stepper


@pytest.mark.verification
@pytest.mark.parametrize("representation,formulation", TIME_VARIANTS)
def test_marker_must_wrap_stepper_solution(
    representation,
    formulation,
) -> None:
    mesh = fd.UnitIntervalMesh(1)
    space = fd.FunctionSpace(mesh, "CG", 1)
    u = fd.Function(space)
    other = fd.Function(space)
    test = fd.TestFunction(space)
    residual = fd.inner(CaputoDerivative(other, 0.5), test) * fd.dx
    options = {}
    if formulation is not None:
        options["formulation"] = formulation
    with pytest.raises(ValueError, match="stepper solution"):
        FractionalTimeStepper(
            residual,
            representation,
            fd.Constant(0.0),
            fd.Constant(0.1),
            u,
            **options,
        )


@pytest.mark.verification
@pytest.mark.parametrize("representation,formulation", TIME_VARIANTS)
def test_time_stepping_requires_continuous_lagrange(
    representation,
    formulation,
) -> None:
    mesh = fd.UnitIntervalMesh(1)
    space = fd.FunctionSpace(mesh, "DG", 0)
    u = fd.Function(space)
    test = fd.TestFunction(space)
    residual = fd.inner(CaputoDerivative(u, 0.5), test) * fd.dx
    options = {}
    if formulation is not None:
        options["formulation"] = formulation
    with pytest.raises(NotImplementedError, match="continuous Lagrange"):
        FractionalTimeStepper(
            residual,
            representation,
            fd.Constant(0.0),
            fd.Constant(0.1),
            u,
            **options,
        )


@pytest.mark.verification
def test_auxiliary_requires_one_scalar_marker() -> None:
    mesh = fd.UnitIntervalMesh(1)
    space = fd.FunctionSpace(mesh, "CG", 1)
    u = fd.Function(space)
    test = fd.TestFunction(space)
    ordinary = fd.inner(u, test) * fd.dx
    multiple = (
        fd.inner(CaputoDerivative(u, 0.4), test)
        + fd.inner(CaputoDerivative(u, 0.6), test)
    ) * fd.dx
    for residual in (ordinary, multiple):
        with pytest.raises(ValueError, match="exactly one"):
            FractionalTimeStepper(
                residual,
                BirkSong(2),
                fd.Constant(0.0),
                fd.Constant(0.1),
                u,
                formulation=AuxiliaryODE(),
            )

    vector_space = fd.VectorFunctionSpace(mesh, "CG", 1)
    vector = fd.Function(vector_space)
    vector_test = fd.TestFunction(vector_space)
    residual = fd.inner(CaputoDerivative(vector, 0.5), vector_test) * fd.dx
    with pytest.raises(NotImplementedError, match="scalar"):
        FractionalTimeStepper(
            residual,
            BirkSong(2),
            fd.Constant(0.0),
            fd.Constant(0.1),
            vector,
            formulation=AuxiliaryODE(),
        )

    trial = fd.TrialFunction(space)
    residual = (
        fd.inner(CaputoDerivative(u, 0.5), test)
        + fd.inner(trial, test)
    ) * fd.dx
    with pytest.raises(ValueError, match="one test argument"):
        FractionalTimeStepper(
            residual,
            BirkSong(2),
            fd.Constant(0.0),
            fd.Constant(0.1),
            u,
            formulation=AuxiliaryODE(),
        )


@pytest.mark.verification
def test_formulation_and_representation_are_validated() -> None:
    mesh = fd.UnitIntervalMesh(1)
    space = fd.FunctionSpace(mesh, "CG", 1)
    u = fd.Function(space)
    test = fd.TestFunction(space)
    residual = fd.inner(CaputoDerivative(u, 0.5), test) * fd.dx
    time = fd.Constant(0.0)
    dt = fd.Constant(0.1)

    with pytest.raises(TypeError, match="formulation"):
        FractionalTimeStepper(
            residual,
            BirkSong(2),
            time,
            dt,
            u,
            formulation=object(),
        )
    with pytest.raises(NotImplementedError, match="eliminated"):
        FractionalTimeStepper(
            residual,
            FullHistory(),
            time,
            dt,
            u,
            formulation=AuxiliaryODE(),
        )
    with pytest.raises(TypeError, match="representation"):
        FractionalTimeStepper(residual, object(), time, dt, u)
    with pytest.raises(TypeError, match="representation"):
        FractionalTimeStepper(
            residual,
            object(),
            time,
            dt,
            u,
            formulation=AuxiliaryODE(),
        )
    ordinary = fd.inner(u, test) * fd.dx
    with pytest.raises(ValueError, match="at least one"):
        FractionalTimeStepper(
            ordinary,
            FullHistory(),
            time,
            dt,
            u,
        )


@pytest.mark.verification
@pytest.mark.parametrize("representation,formulation", TIME_VARIANTS)
def test_initial_state_reset_and_jacobian_invalidation(
    representation,
    formulation,
) -> None:
    mesh = fd.UnitIntervalMesh(1)
    space = fd.FunctionSpace(mesh, "CG", 1)
    initial = fd.Function(space).assign(2.0)
    u, time, dt, stepper = make_stepper(
        mesh,
        representation,
        formulation,
        initial_value=9.0,
        u0=initial,
    )
    assert fd.norm(u - initial) == 0.0
    assert stepper.transformed_residual is not None
    if isinstance(formulation, AuxiliaryODE):
        assert len(stepper.mode_residuals) == 2
        assert stepper.appctx["yonderdrake"]["scheme"] == "backward_euler"

    stepper.advance()
    time.assign(time + dt)
    old_solver = stepper._solver
    stepper.invalidate_jacobian()
    stepper.advance()
    assert stepper._solver is not old_solver

    current_time = float(time)
    stepper.reset(3.0)
    assert float(time) == current_time
    assert fd.norm(u - 3.0) < 1.0e-14
    assert all(fd.norm(memory) == 0.0 for memory in stepper.history)
    assert stepper.solver_stats()["solves"] == 0
    stepper.reset(4.0, t0=1.5)
    assert float(time) == 1.5


@pytest.mark.verification
@pytest.mark.parametrize(
    "representation,formulation",
    [
        pytest.param(BirkSong(2), AuxiliaryODE(), id="auxiliary"),
        pytest.param(FullHistory(), None, id="full-history"),
    ],
)
def test_order_is_immutable_and_step_size_is_validated(
    representation,
    formulation,
) -> None:
    alpha = fd.Constant(0.5)
    _, _, dt, stepper = make_stepper(
        fd.UnitIntervalMesh(1),
        representation,
        formulation,
        alpha=alpha,
    )
    alpha.assign(0.6)
    with pytest.raises(RuntimeError, match="changing alpha"):
        stepper.advance()
    alpha.assign(0.5)
    dt.assign(0.0)
    with pytest.raises(ValueError, match="finite and positive"):
        stepper.advance()


@pytest.mark.verification
@pytest.mark.parametrize("representation,formulation", TIME_VARIANTS)
@pytest.mark.parametrize("alpha", [0.0, float("nan")])
def test_fractional_order_is_validated(
    representation,
    formulation,
    alpha: float,
) -> None:
    with pytest.raises(ValueError, match="0 < alpha < 1"):
        make_stepper(
            fd.UnitIntervalMesh(1),
            representation,
            formulation,
            alpha=alpha,
        )


@pytest.mark.verification
@pytest.mark.parametrize("representation,formulation", TIME_VARIANTS)
def test_nonfinite_step_size_is_rejected(
    representation,
    formulation,
) -> None:
    _, _, dt, stepper = make_stepper(
        fd.UnitIntervalMesh(1),
        representation,
        formulation,
    )
    dt.assign(float("inf"))
    with pytest.raises(ValueError, match="dt must be finite and positive"):
        stepper.advance()


@pytest.mark.verification
@pytest.mark.parametrize(
    "time_value,error",
    [
        pytest.param(float("nan"), "t must be finite", id="nonfinite"),
        pytest.param(
            float.fromhex("0x1.fffffffffffffp+1023"),
            r"t \+ dt",
            id="nonincreasing-target",
        ),
    ],
)
def test_full_history_validates_time(time_value: float, error: str) -> None:
    with pytest.raises(ValueError, match=error):
        _, _, _, stepper = make_stepper(
            fd.UnitIntervalMesh(1),
            FullHistory(),
            None,
            time_value=time_value,
        )
        stepper.advance()


@pytest.mark.verification
@pytest.mark.parametrize("formulation", [Recurrence(), AuxiliaryODE()])
def test_riemann_liouville_time_must_exceed_lower_limit(formulation) -> None:
    mesh = fd.UnitIntervalMesh(1)
    space = fd.FunctionSpace(mesh, "CG", 1)
    u = fd.Function(space).assign(1.0)
    test = fd.TestFunction(space)
    time = fd.Constant(0.0)
    residual = fd.inner(
        RiemannLiouvilleDerivative(u, 0.5),
        test,
    ) * fd.dx
    stepper = FractionalTimeStepper(
        residual,
        BirkSong(2),
        time,
        fd.Constant(0.1),
        u,
        formulation=formulation,
    )
    time.assign(-1.0)
    with pytest.raises(ValueError, match="must exceed its lower limit"):
        stepper.advance()
    time.assign(float("nan"))
    with pytest.raises(ValueError, match="t must be finite"):
        stepper.advance()


@pytest.mark.verification
@pytest.mark.parametrize("representation,formulation", TIME_VARIANTS)
def test_initial_checkpoint_state_roundtrips(
    representation,
    formulation,
) -> None:
    mesh = fd.UnitIntervalMesh(1)
    source_u, _, _, source = make_stepper(
        mesh,
        representation,
        formulation,
        initial_value=2.0,
    )
    target_u, _, _, target = make_stepper(
        mesh,
        representation,
        formulation,
        initial_value=9.0,
    )
    target.restore_checkpoint(source.checkpoint_state())
    assert fd.norm(target_u - source_u) == 0.0
    assert target.solver_stats()["last_step_size"] is None


@pytest.mark.verification
def test_auxiliary_failed_solve_restores_physical_and_mode_state() -> None:
    mesh = fd.UnitIntervalMesh(1)
    space = fd.FunctionSpace(mesh, "CG", 1)
    u = fd.Function(space).assign(0.25)
    test = fd.TestFunction(space)
    residual = (
        fd.inner(CaputoDerivative(u, 0.5), test)
        + fd.inner(u**3 - 10.0, test)
    ) * fd.dx
    stepper = FractionalTimeStepper(
        residual,
        BirkSong(2),
        fd.Constant(0.0),
        fd.Constant(0.1),
        u,
        formulation=AuxiliaryODE(),
        solver_parameters={"snes_max_it": 0},
    )
    committed = u.copy(deepcopy=True)
    committed_history = stepper.history

    with pytest.raises(fd.ConvergenceError):
        stepper.advance()

    assert fd.norm(u - committed) == 0.0
    assert stepper.solver_stats()["failures"] == 1
    for observed, expected in zip(
        stepper.history,
        committed_history,
        strict=True,
    ):
        assert fd.norm(observed - expected) == 0.0
