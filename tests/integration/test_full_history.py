"""Integration tests for direct full-history Caputo stepping."""

from __future__ import annotations

import copy
from math import gamma
from pathlib import Path

import numpy as np
import pytest

fd = pytest.importorskip("firedrake")

from yonderdrake import (  # noqa: E402
    AlikhanovL21Sigma,
    CaputoDerivative,
    FastObliviousCQ,
    FractionalTimeStepper,
    FullHistory,
    LubichCQ,
    RiemannLiouvilleDerivative,
)


def make_relaxation(
    mesh,
    *,
    alpha: float = 0.6,
    initial_value: float = 1.0,
):
    space = fd.FunctionSpace(mesh, "CG", 1)
    u = fd.Function(space, name="u").assign(initial_value)
    test = fd.TestFunction(space)
    time = fd.Constant(0.0)
    dt = fd.Constant(0.1)
    residual = (fd.inner(CaputoDerivative(u, alpha), test) + fd.inner(u, test)) * fd.dx
    stepper = FractionalTimeStepper(
        residual,
        FullHistory(),
        time,
        dt,
        u,
        solver_parameters={"ksp_type": "preonly", "pc_type": "lu"},
    )
    return u, time, dt, stepper


def checkpoint_path(tmp_path: Path, comm, name: str) -> Path:
    path = str(tmp_path / name) if comm.rank == 0 else None
    return Path(comm.bcast(path, root=0))


def checkpoint_file_restart(path: Path, comm) -> None:
    mesh = fd.UnitIntervalMesh(3, comm=comm, name="full_history_mesh")
    source_u, source_time, source_dt, source = make_relaxation(mesh)
    for step_size in (0.1, 0.04):
        source_dt.assign(step_size)
        source.advance()
        source_time.assign(source_time + source_dt)
    with fd.CheckpointFile(str(path), "w", comm=comm) as checkpoint:
        source.save_checkpoint(checkpoint)

    with fd.CheckpointFile(str(path), "r", comm=comm) as checkpoint:
        restarted_mesh = checkpoint.load_mesh("full_history_mesh")
        target_u, target_time, target_dt, target = make_relaxation(
            restarted_mesh,
            initial_value=9.0,
        )
        target.load_checkpoint(checkpoint)

    np.testing.assert_allclose(target_u.dat.data_ro, source_u.dat.data_ro)
    assert float(target_time) == pytest.approx(float(source_time))
    assert float(target_dt) == pytest.approx(float(source_dt))
    for target_increment, source_increment in zip(
        target.history,
        source.history,
        strict=True,
    ):
        np.testing.assert_allclose(
            target_increment.dat.data_ro,
            source_increment.dat.data_ro,
        )

    source.advance()
    target.advance()
    np.testing.assert_allclose(
        target_u.dat.data_ro,
        source_u.dat.data_ro,
        atol=2.0e-11,
    )


def fast_cq_checkpoint_file_restart(path: Path, comm) -> None:
    mesh = fd.UnitIntervalMesh(1, comm=comm, name="fast_cq_checkpoint_mesh")
    representation = FastObliviousCQ(
        num_levels=5,
        nodes_per_level=4,
        direct_steps=6,
    )

    def build(active_mesh, initial: float):
        space = fd.FunctionSpace(active_mesh, "CG", 1)
        test = fd.TestFunction(space)
        u = fd.Function(space).assign(initial)
        time = fd.Constant(0.0)
        dt = fd.Constant(0.05)
        residual = (
            fd.inner(CaputoDerivative(u, 0.6), test) + fd.inner(u, test)
        ) * fd.dx
        return (
            u,
            time,
            dt,
            FractionalTimeStepper(
                residual,
                representation,
                time,
                dt,
                u,
            ),
        )

    source_u, source_time, source_dt, source = build(mesh, 1.0)
    for _ in range(5):
        source.advance()
        source_time.assign(source_time + source_dt)
    with fd.CheckpointFile(str(path), "w", comm=comm) as checkpoint:
        source.save_checkpoint(checkpoint)

    with fd.CheckpointFile(str(path), "r", comm=comm) as checkpoint:
        restarted_mesh = checkpoint.load_mesh("fast_cq_checkpoint_mesh")
        target_u, target_time, _, target = build(restarted_mesh, 9.0)
        target.load_checkpoint(checkpoint)
    np.testing.assert_allclose(target_u.dat.data_ro, source_u.dat.data_ro)
    assert float(target_time) == pytest.approx(float(source_time))
    source.advance()
    target.advance()
    np.testing.assert_allclose(
        target_u.dat.data_ro,
        source_u.dat.data_ro,
        atol=2.0e-9,
    )


@pytest.mark.verification
def test_variable_step_matches_independent_full_history() -> None:
    alpha = 0.45
    reaction = 0.8
    mesh = fd.UnitIntervalMesh(2)
    space = fd.FunctionSpace(mesh, "CG", 1)
    u = fd.Function(space).assign(1.0)
    test = fd.TestFunction(space)
    time = fd.Constant(0.0)
    dt = fd.Constant(0.1)
    residual = (
        fd.inner(CaputoDerivative(u, alpha), test) + reaction * fd.inner(u, test)
    ) * fd.dx
    stepper = FractionalTimeStepper(residual, FullHistory(), time, dt, u)

    times = [0.0]
    increments: list[float] = []
    expected = 1.0
    for step_size in (0.1, 0.025, 0.2, 0.075):
        dt.assign(step_size)
        target = times[-1] + step_size
        history = 0.0
        for left, right, increment in zip(
            times[:-1],
            times[1:],
            increments,
            strict=True,
        ):
            weight = (
                (target - left) ** (1.0 - alpha) - (target - right) ** (1.0 - alpha)
            ) / ((right - left) * gamma(2.0 - alpha))
            history += weight * increment
        implicit = step_size ** (-alpha) / gamma(2.0 - alpha)
        updated = (implicit * expected - history) / (implicit + reaction)

        stepper.advance()
        time.assign(time + dt)
        increments.append(updated - expected)
        times.append(target)
        expected = updated

    assert fd.norm(u - expected) < 2.0e-11
    assert stepper.solver_stats()["history_steps"] == 4
    for field, increment in zip(stepper.history, increments, strict=True):
        assert fd.norm(field - increment) < 2.0e-11


@pytest.mark.verification
def test_linear_history_is_integrated_exactly_on_variable_steps() -> None:
    alpha = 0.55
    mesh = fd.UnitIntervalMesh(1)
    space = fd.FunctionSpace(mesh, "CG", 1)
    u = fd.Function(space).assign(0.0)
    test = fd.TestFunction(space)
    time = fd.Constant(0.0)
    dt = fd.Constant(0.1)
    source = fd.Constant(0.0)
    residual = (
        fd.inner(CaputoDerivative(u, alpha), test) - fd.inner(source, test)
    ) * fd.dx
    stepper = FractionalTimeStepper(residual, FullHistory(), time, dt, u)

    target = 0.0
    for step_size in (0.1, 0.07, 0.2, 0.03):
        dt.assign(step_size)
        target += step_size
        source.assign(target ** (1.0 - alpha) / gamma(2.0 - alpha))
        stepper.advance()
        time.assign(time + dt)
        assert fd.norm(u - target) < 2.0e-11


@pytest.mark.verification
def test_multiple_orders_share_the_physical_history() -> None:
    alphas = (0.3, 0.7)
    coefficients = (0.4, 0.6)
    step_size = 0.08
    mesh = fd.UnitIntervalMesh(1)
    space = fd.FunctionSpace(mesh, "CG", 1)
    u = fd.Function(space).assign(2.0)
    test = fd.TestFunction(space)
    time = fd.Constant(0.0)
    dt = fd.Constant(step_size)
    residual = (
        sum(
            coefficient * fd.inner(CaputoDerivative(u, alpha), test)
            for coefficient, alpha in zip(
                coefficients,
                alphas,
                strict=True,
            )
        )
        + fd.inner(u, test)
    ) * fd.dx
    stepper = FractionalTimeStepper(residual, FullHistory(), time, dt, u)
    stepper.advance()

    implicit = sum(
        coefficient * step_size ** (-alpha) / gamma(2.0 - alpha)
        for coefficient, alpha in zip(coefficients, alphas, strict=True)
    )
    expected = 2.0 * implicit / (1.0 + implicit)
    assert fd.norm(u - expected) < 2.0e-11
    assert stepper.solver_stats()["num_fractional_terms"] == 2
    assert len(stepper.history) == 1


def power_solution_error(step_size: float) -> float:
    alpha = 0.55
    power = 2.0
    mesh = fd.UnitIntervalMesh(1)
    space = fd.FunctionSpace(mesh, "CG", 1)
    u = fd.Function(space).assign(0.0)
    test = fd.TestFunction(space)
    time = fd.Constant(0.0)
    dt = fd.Constant(step_size)
    source = fd.Constant(0.0)
    residual = (
        fd.inner(CaputoDerivative(u, alpha), test) - fd.inner(source, test)
    ) * fd.dx
    stepper = FractionalTimeStepper(residual, FullHistory(), time, dt, u)
    coefficient = gamma(power + 1.0) / gamma(power + 1.0 - alpha)
    for index in range(1, round(1.0 / step_size) + 1):
        target = index * step_size
        source.assign(coefficient * target ** (power - alpha))
        stepper.advance()
        time.assign(time + dt)
    return abs(float(u.dat.data_ro[0]) - 1.0)


def cq_power_solution_error(
    step_size: float,
    *,
    alpha: float,
    power: float,
    num_corrections: int,
) -> float:
    mesh = fd.UnitIntervalMesh(1)
    space = fd.FunctionSpace(mesh, "CG", 1)
    u = fd.Function(space).assign(0.0)
    test = fd.TestFunction(space)
    time = fd.Constant(0.0)
    dt = fd.Constant(step_size)
    source = fd.Constant(0.0)
    residual = (
        fd.inner(CaputoDerivative(u, alpha), test) - fd.inner(source, test)
    ) * fd.dx
    stepper = FractionalTimeStepper(
        residual,
        LubichCQ(
            order="bdf2",
            num_corrections=num_corrections,
        ),
        time,
        dt,
        u,
    )
    coefficient = gamma(power + 1.0) / gamma(power + 1.0 - alpha)
    for index in range(1, round(1.0 / step_size) + 1):
        target = index * step_size
        source.assign(coefficient * target ** (power - alpha))
        stepper.advance()
        time.assign(time + dt)
    return abs(float(u.dat.data_ro[0]) - 1.0)


def alikhanov_power_solution_error(
    step_size: float,
    *,
    alpha: float,
    power: float,
) -> float:
    mesh = fd.UnitIntervalMesh(1)
    space = fd.FunctionSpace(mesh, "CG", 1)
    u = fd.Function(space).assign(0.0)
    test = fd.TestFunction(space)
    time = fd.Constant(0.0)
    dt = fd.Constant(step_size)
    coefficient = gamma(power + 1.0) / gamma(power + 1.0 - alpha)
    source = coefficient * time ** (power - alpha) + time**power
    residual = (
        fd.inner(CaputoDerivative(u, alpha), test)
        + fd.inner(u, test)
        - fd.inner(source, test)
    ) * fd.dx
    stepper = FractionalTimeStepper(
        residual,
        AlikhanovL21Sigma(),
        time,
        dt,
        u,
    )
    for _ in range(round(1.0 / step_size)):
        stepper.advance()
        time.assign(time + dt)
    return abs(float(u.dat.data_ro[0]) - 1.0)


def fast_cq_power_solution_error(
    step_size: float,
    *,
    alpha: float,
    power: float,
) -> float:
    mesh = fd.UnitIntervalMesh(1)
    space = fd.FunctionSpace(mesh, "CG", 1)
    u = fd.Function(space).assign(0.0)
    test = fd.TestFunction(space)
    time = fd.Constant(0.0)
    dt = fd.Constant(step_size)
    coefficient = gamma(power + 1.0) / gamma(power + 1.0 - alpha)
    source = coefficient * time ** (power - alpha)
    residual = (
        fd.inner(CaputoDerivative(u, alpha), test) - fd.inner(source, test)
    ) * fd.dx
    steps = round(1.0 / step_size)
    levels = max(2, steps.bit_length())
    stepper = FractionalTimeStepper(
        residual,
        FastObliviousCQ(
            num_levels=levels,
            nodes_per_level=15,
            direct_steps=20,
        ),
        time,
        dt,
        u,
    )
    for _ in range(steps):
        stepper.advance()
        time.assign(time + dt)
    return abs(float(u.dat.data_ro[0]) - 1.0)


@pytest.mark.verification
def test_power_solution_converges_in_timestep() -> None:
    coarse = power_solution_error(0.1)
    fine = power_solution_error(0.05)
    assert fine < coarse
    assert fine < 1.0e-2


@pytest.mark.verification
def test_lubich_starting_correction_recovers_fractional_power() -> None:
    alpha = 0.55
    corrected = cq_power_solution_error(
        0.05,
        alpha=alpha,
        power=alpha,
        num_corrections=1,
    )
    uncorrected = cq_power_solution_error(
        0.05,
        alpha=alpha,
        power=alpha,
        num_corrections=0,
    )
    assert corrected < 2.0e-11
    assert uncorrected > 1.0e-4


@pytest.mark.verification
def test_lubich_bdf2_is_second_order_for_smooth_power() -> None:
    coarse = cq_power_solution_error(
        0.1,
        alpha=0.55,
        power=2.0,
        num_corrections=0,
    )
    fine = cq_power_solution_error(
        0.05,
        alpha=0.55,
        power=2.0,
        num_corrections=0,
    )
    assert coarse / fine > 3.5
    assert fine < 2.0e-3


@pytest.mark.verification
def test_lubich_cq_rejects_variable_steps_without_committing() -> None:
    mesh = fd.UnitIntervalMesh(1)
    u, time, dt, _ = make_relaxation(mesh)
    test = fd.TestFunction(u.function_space())
    residual = (fd.inner(CaputoDerivative(u, 0.6), test) + fd.inner(u, test)) * fd.dx
    stepper = FractionalTimeStepper(
        residual,
        LubichCQ(),
        time,
        dt,
        u,
    )
    stepper.advance()
    time.assign(time + dt)
    committed = u.copy(deepcopy=True)
    dt.assign(0.05)
    with pytest.raises(ValueError, match="uniform"):
        stepper.advance()
    assert fd.norm(u - committed) == 0.0
    assert len(stepper.history) == 1


@pytest.mark.verification
def test_lubich_cq_checkpoint_round_trip() -> None:
    mesh = fd.UnitIntervalMesh(2)
    space = fd.FunctionSpace(mesh, "CG", 1)
    test = fd.TestFunction(space)
    time = fd.Constant(0.0)
    dt = fd.Constant(0.05)

    def build(initial: float):
        u = fd.Function(space).assign(initial)
        residual = (
            fd.inner(CaputoDerivative(u, 0.6), test) + fd.inner(u, test)
        ) * fd.dx
        return u, FractionalTimeStepper(
            residual,
            LubichCQ(),
            time,
            dt,
            u,
        )

    source_u, source = build(1.0)
    for _ in range(3):
        source.advance()
        time.assign(time + dt)
    checkpoint = copy.deepcopy(source.checkpoint_state())
    target_u, target = build(9.0)
    target.restore_checkpoint(checkpoint)
    assert fd.norm(target_u - source_u) < 2.0e-12

    source.advance()
    target.advance()
    assert fd.norm(target_u - source_u) < 2.0e-11


@pytest.mark.verification
def test_alikhanov_is_second_order_at_the_offset_state_and_time() -> None:
    coarse = alikhanov_power_solution_error(0.1, alpha=0.55, power=2.0)
    fine = alikhanov_power_solution_error(0.05, alpha=0.55, power=2.0)
    assert coarse / fine > 3.5
    assert fine < 2.0e-3


@pytest.mark.verification
def test_alikhanov_weakly_singular_power_converges() -> None:
    coarse = alikhanov_power_solution_error(0.05, alpha=0.55, power=0.55)
    fine = alikhanov_power_solution_error(0.025, alpha=0.55, power=0.55)
    assert fine < coarse
    assert fine < 2.0e-2


@pytest.mark.verification
def test_alikhanov_rejects_incompatible_markers_and_variable_steps() -> None:
    mesh = fd.UnitIntervalMesh(1)
    space = fd.FunctionSpace(mesh, "CG", 1)
    u = fd.Function(space).assign(1.0)
    test = fd.TestFunction(space)
    time = fd.Constant(0.0)
    dt = fd.Constant(0.1)
    mixed_order = (
        fd.inner(CaputoDerivative(u, 0.4), test)
        + fd.inner(CaputoDerivative(u, 0.6), test)
    ) * fd.dx
    with pytest.raises(NotImplementedError, match="shared alpha"):
        FractionalTimeStepper(
            mixed_order,
            AlikhanovL21Sigma(),
            time,
            dt,
            u,
        )
    riemann_liouville = (
        fd.inner(
            RiemannLiouvilleDerivative(u, 0.5),
            test,
        )
        * fd.dx
    )
    with pytest.raises(NotImplementedError, match="Caputo"):
        FractionalTimeStepper(
            riemann_liouville,
            AlikhanovL21Sigma(),
            time,
            dt,
            u,
        )

    residual = (fd.inner(CaputoDerivative(u, 0.5), test) + fd.inner(u, test)) * fd.dx
    stepper = FractionalTimeStepper(
        residual,
        AlikhanovL21Sigma(),
        time,
        dt,
        u,
    )
    stepper.advance()
    time.assign(time + dt)
    dt.assign(0.05)
    with pytest.raises(ValueError, match="uniform"):
        stepper.advance()


@pytest.mark.verification
def test_alikhanov_checkpoint_round_trip() -> None:
    mesh = fd.UnitIntervalMesh(1)
    space = fd.FunctionSpace(mesh, "CG", 1)
    u = fd.Function(space).assign(1.0)
    test = fd.TestFunction(space)
    time = fd.Constant(0.0)
    dt = fd.Constant(0.05)
    residual = (fd.inner(CaputoDerivative(u, 0.6), test) + fd.inner(u, test)) * fd.dx
    stepper = FractionalTimeStepper(
        residual,
        AlikhanovL21Sigma(),
        time,
        dt,
        u,
    )
    for _ in range(3):
        stepper.advance()
        time.assign(time + dt)
    checkpoint = copy.deepcopy(stepper.checkpoint_state())
    saved = u.copy(deepcopy=True)
    stepper.reset(4.0, t0=0.0)
    stepper.restore_checkpoint(checkpoint)
    assert fd.norm(u - saved) < 2.0e-12
    stepper.advance()


@pytest.mark.verification
def test_fast_oblivious_cq_has_bdf1_smooth_convergence() -> None:
    coarse = fast_cq_power_solution_error(0.05, alpha=0.55, power=2.0)
    fine = fast_cq_power_solution_error(0.025, alpha=0.55, power=2.0)
    assert coarse / fine > 1.8
    assert fine < 3.0e-2


@pytest.mark.verification
def test_fast_oblivious_cq_weakly_singular_power_converges() -> None:
    coarse = fast_cq_power_solution_error(0.05, alpha=0.55, power=0.55)
    fine = fast_cq_power_solution_error(0.025, alpha=0.55, power=0.55)
    assert fine < coarse
    assert fine < 8.0e-2


@pytest.mark.verification
def test_fast_oblivious_cq_checkpoint_round_trip() -> None:
    mesh = fd.UnitIntervalMesh(1)
    space = fd.FunctionSpace(mesh, "CG", 1)
    test = fd.TestFunction(space)
    dt = fd.Constant(0.02)
    representation = FastObliviousCQ(
        num_levels=8,
        nodes_per_level=10,
        direct_steps=10,
    )

    def build(initial: float):
        time = fd.Constant(0.0)
        u = fd.Function(space).assign(initial)
        residual = (
            fd.inner(CaputoDerivative(u, 0.6), test) + fd.inner(u, test)
        ) * fd.dx
        stepper = FractionalTimeStepper(
            residual,
            representation,
            time,
            dt,
            u,
        )
        return u, time, stepper

    source_u, source_time, source = build(1.0)
    for _ in range(24):
        source.advance()
        source_time.assign(source_time + dt)
    state = copy.deepcopy(source.checkpoint_state())
    target_u, target_time, target = build(9.0)
    target.restore_checkpoint(state)
    assert fd.norm(target_u - source_u) < 2.0e-12
    assert float(target_time) == pytest.approx(float(source_time))

    source.advance()
    target.advance()
    assert fd.norm(target_u - source_u) < 2.0e-9


@pytest.mark.verification
def test_fast_oblivious_cq_lifecycle_and_limits() -> None:
    mesh = fd.UnitIntervalMesh(1)
    space = fd.FunctionSpace(mesh, "CG", 1)
    u = fd.Function(space).assign(1.0)
    test = fd.TestFunction(space)
    alpha = fd.Constant(0.6)
    time = fd.Constant(0.0)
    dt = fd.Constant(0.05)
    residual = (fd.inner(CaputoDerivative(u, alpha), test) + fd.inner(u, test)) * fd.dx
    stepper = FractionalTimeStepper(
        residual,
        FastObliviousCQ(
            num_levels=3,
            nodes_per_level=4,
            direct_steps=6,
        ),
        time,
        dt,
        u,
    )

    stepper.advance()
    assert len(stepper.history) == 1
    assert stepper.solver_stats()["history_steps"] == 1
    with pytest.raises(RuntimeError, match="t must be advanced"):
        stepper.advance()
    time.assign(time + dt)

    dt.assign(0.04)
    with pytest.raises(ValueError, match="uniform"):
        stepper.advance()
    dt.assign(0.05)
    alpha.assign(0.5)
    with pytest.raises(RuntimeError, match="changing alpha"):
        stepper.advance()
    alpha.assign(0.6)

    stepper.reset(2.0, t0=0.25)
    assert float(time) == pytest.approx(0.25)
    assert stepper.history == ()
    assert stepper.solver_stats()["solves"] == 0
    for _ in range(7):
        stepper.advance()
        time.assign(time + dt)
    with pytest.raises(RuntimeError, match="max_steps"):
        stepper.advance()


@pytest.mark.verification
def test_fast_oblivious_cq_riemann_liouville_trace() -> None:
    mesh = fd.UnitIntervalMesh(1)
    space = fd.FunctionSpace(mesh, "CG", 1)
    test = fd.TestFunction(space)
    time = fd.Constant(0.0)
    dt = fd.Constant(0.05)

    def build(marker):
        u = fd.Function(space).assign(1.0)
        residual = fd.inner(marker(u, 0.6), test) * fd.dx
        return u, FractionalTimeStepper(
            residual,
            FastObliviousCQ(
                num_levels=3,
                nodes_per_level=4,
                direct_steps=6,
            ),
            time,
            dt,
            u,
        )

    caputo_u, caputo = build(CaputoDerivative)
    riemann_u, riemann = build(RiemannLiouvilleDerivative)
    caputo.advance()
    riemann.advance()
    assert float(caputo_u.dat.data_ro[0]) == pytest.approx(1.0)
    assert float(riemann_u.dat.data_ro[0]) == pytest.approx(0.6)


@pytest.mark.verification
def test_direct_methods_reset_and_reject_nonuniform_checkpoints() -> None:
    mesh = fd.UnitIntervalMesh(1)
    space = fd.FunctionSpace(mesh, "CG", 1)
    test = fd.TestFunction(space)

    for representation in (
        LubichCQ(),
        AlikhanovL21Sigma(),
    ):
        time = fd.Constant(0.0)
        dt = fd.Constant(0.05)
        u = fd.Function(space).assign(1.0)
        residual = (
            fd.inner(CaputoDerivative(u, 0.6), test) + fd.inner(u, test)
        ) * fd.dx
        stepper = FractionalTimeStepper(
            residual,
            representation,
            time,
            dt,
            u,
        )
        for _ in range(2):
            stepper.advance()
            time.assign(time + dt)
        malformed = copy.deepcopy(stepper.checkpoint_state())
        malformed["times"][-1] = 0.11
        malformed["time"] = 0.11
        with pytest.raises(ValueError, match="not uniform"):
            stepper.restore_checkpoint(malformed)

        stepper.reset(2.0, t0=0.0)
        empty = copy.deepcopy(stepper.checkpoint_state())
        stepper.restore_checkpoint(empty)
        assert stepper.history == ()
        assert stepper.solver_stats()["solves"] == 0


@pytest.mark.verification
def test_lubich_cq_riemann_liouville_trace_and_fast_cq_rollback() -> None:
    mesh = fd.UnitIntervalMesh(1)
    space = fd.FunctionSpace(mesh, "CG", 1)
    test = fd.TestFunction(space)
    time = fd.Constant(0.0)
    dt = fd.Constant(0.05)
    u = fd.Function(space).assign(1.0)
    residual = fd.inner(RiemannLiouvilleDerivative(u, 0.6), test) * fd.dx
    lubich = FractionalTimeStepper(
        residual,
        LubichCQ(order="bdf1", num_corrections=0),
        time,
        dt,
        u,
    )
    lubich.advance()
    assert float(u.dat.data_ro[0]) == pytest.approx(0.6)

    fast_u = fd.Function(space).assign(1.0)
    fast_time = fd.Constant(0.0)
    fast_residual = (
        fd.inner(CaputoDerivative(fast_u, 0.6), test) + fd.inner(fast_u, test)
    ) * fd.dx
    fast = FractionalTimeStepper(
        fast_residual,
        FastObliviousCQ(
            num_levels=3,
            nodes_per_level=4,
            direct_steps=6,
        ),
        fast_time,
        dt,
        fast_u,
    )
    fast.advance()
    fast_time.assign(fast_time + dt)
    committed = fast_u.copy(deepcopy=True)

    class FailingSolver:
        def solve(self) -> None:
            raise RuntimeError("deliberate solver failure")

    fast._solver = FailingSolver()
    with pytest.raises(RuntimeError, match="deliberate"):
        fast.advance()
    assert fd.norm(fast_u - committed) == 0.0
    assert fast.solver_stats()["failures"] == 1


@pytest.mark.verification
def test_checkpoint_reset_and_restore() -> None:
    mesh = fd.UnitIntervalMesh(2)
    u, time, dt, stepper = make_relaxation(mesh)
    for step_size in (0.1, 0.04):
        dt.assign(step_size)
        stepper.advance()
        time.assign(time + dt)
    checkpoint = copy.deepcopy(stepper.checkpoint_state())
    saved_u = u.copy(deepcopy=True)
    saved_history = stepper.history

    stepper.reset(3.0, t0=1.0)
    assert fd.norm(u - 3.0) < 1.0e-12
    assert stepper.history == ()

    stepper.restore_checkpoint(checkpoint)
    assert fd.norm(u - saved_u) < 1.0e-12
    assert float(time) == pytest.approx(0.14)
    for restored, saved in zip(
        stepper.history,
        saved_history,
        strict=True,
    ):
        assert fd.norm(restored - saved) < 1.0e-12


@pytest.mark.verification
def test_checkpoint_file_restart(tmp_path: Path) -> None:
    checkpoint_file_restart(
        checkpoint_path(tmp_path, fd.COMM_WORLD, "full-history.h5"),
        fd.COMM_WORLD,
    )


@pytest.mark.verification
def test_fast_cq_checkpoint_file_restart(tmp_path: Path) -> None:
    fast_cq_checkpoint_file_restart(
        checkpoint_path(tmp_path, fd.COMM_WORLD, "fast-cq.h5"),
        fd.COMM_WORLD,
    )


@pytest.mark.parallel(nprocs=2)
@pytest.mark.verification
def test_checkpoint_file_restart_two_ranks(tmp_path: Path) -> None:
    checkpoint_file_restart(
        checkpoint_path(tmp_path, fd.COMM_WORLD, "full-history-two.h5"),
        fd.COMM_WORLD,
    )


@pytest.mark.parallel(nprocs=2)
@pytest.mark.verification
def test_fast_cq_checkpoint_file_restart_two_ranks(tmp_path: Path) -> None:
    fast_cq_checkpoint_file_restart(
        checkpoint_path(tmp_path, fd.COMM_WORLD, "fast-cq-two.h5"),
        fd.COMM_WORLD,
    )


@pytest.mark.parallel(nprocs=4)
@pytest.mark.verification
def test_checkpoint_file_restart_four_ranks(tmp_path: Path) -> None:
    checkpoint_file_restart(
        checkpoint_path(tmp_path, fd.COMM_WORLD, "full-history-four.h5"),
        fd.COMM_WORLD,
    )


@pytest.mark.verification
def test_failure_and_time_validation_do_not_append_history() -> None:
    mesh = fd.UnitIntervalMesh(2)
    space = fd.FunctionSpace(mesh, "CG", 1)
    u = fd.Function(space).assign(0.25)
    test = fd.TestFunction(space)
    time = fd.Constant(0.0)
    dt = fd.Constant(0.1)
    residual = (
        fd.inner(CaputoDerivative(u, 0.5), test) + fd.inner(u**3 - 10.0, test)
    ) * fd.dx
    stepper = FractionalTimeStepper(
        residual,
        FullHistory(),
        time,
        dt,
        u,
        solver_parameters={"snes_max_it": 0},
    )
    committed = u.copy(deepcopy=True)
    with pytest.raises(fd.ConvergenceError):
        stepper.advance()
    assert fd.norm(u - committed) == 0.0
    assert stepper.history == ()

    _, valid_time, _, valid = make_relaxation(fd.UnitIntervalMesh(1))
    valid.advance()
    with pytest.raises(RuntimeError, match="t must be advanced"):
        valid.advance()
    valid_time.assign(0.1)
    assert len(valid.history) == 1


@pytest.mark.verification
def test_riemann_liouville_constant_uses_exact_initial_trace() -> None:
    alpha = 0.4
    initial = 2.0
    mesh = fd.UnitIntervalMesh(1)
    space = fd.FunctionSpace(mesh, "CG", 1)
    u = fd.Function(space).assign(initial)
    test = fd.TestFunction(space)
    time = fd.Constant(0.0)
    dt = fd.Constant(0.1)
    source = fd.Constant(0.0)
    residual = (
        fd.inner(RiemannLiouvilleDerivative(u, alpha), test) - fd.inner(source, test)
    ) * fd.dx
    stepper = FractionalTimeStepper(residual, FullHistory(), time, dt, u)

    target = 0.0
    for step_size in (0.1, 0.04, 0.2):
        dt.assign(step_size)
        target += step_size
        source.assign(initial * target ** (-alpha) / gamma(1.0 - alpha))
        stepper.advance()
        time.assign(time + dt)
        assert fd.norm(u - initial) < 2.0e-11
    assert all(fd.norm(increment) < 2.0e-11 for increment in stepper.history)

    checkpoint = copy.deepcopy(stepper.checkpoint_state())
    stepper.reset(9.0, t0=1.0)
    stepper.restore_checkpoint(checkpoint)
    assert fd.norm(u - initial) < 2.0e-11
    assert float(time) == pytest.approx(target)
