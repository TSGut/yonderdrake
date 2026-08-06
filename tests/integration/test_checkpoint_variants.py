"""Checkpoint coverage across time and spatial operator variants."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

fd = pytest.importorskip("firedrake")

from yonderdrake import (  # noqa: E402
    AuxiliaryODE,
    BirkSong,
    CaputoDerivative,
    Diethelm2008,
    Diethelm2022,
    FastObliviousCQ,
    FractionalTimeStepper,
    FullHistory,
    Oscillator,
    Recurrence,
    RiemannLiouvilleDerivative,
    RieszFractionalLaplacian,
    SineDiffusive,
    SpectralFractionalLaplacian,
    SumOfExponentials,
    YuanAgrawal,
)

MATRIX_FREE_SOLVER = {
    "snes_type": "ksponly",
    "mat_type": "matfree",
    "ksp_type": "gmres",
    "pc_type": "none",
    "ksp_rtol": 1.0e-10,
}


def make_time_stepper(
    mesh,
    representation,
    formulation,
    marker,
    *,
    initial_value: float,
):
    space = fd.FunctionSpace(mesh, "CG", 1)
    u = fd.Function(space, name="u").assign(initial_value)
    test = fd.TestFunction(space)
    time = fd.Constant(0.0)
    dt = fd.Constant(0.1)
    residual = (fd.inner(marker(u, 0.4), test) + fd.inner(u, test)) * fd.dx
    options = {}
    if formulation is not None:
        options["formulation"] = formulation
    stepper = FractionalTimeStepper(
        residual,
        representation,
        time,
        dt,
        u,
        solver_parameters={"ksp_type": "preonly", "pc_type": "lu"},
        **options,
    )
    return u, time, dt, stepper


def checkpoint_path(tmp_path: Path, name: str, comm=fd.COMM_WORLD) -> Path:
    path = str(tmp_path / f"{name}.h5") if comm.rank == 0 else None
    return Path(comm.bcast(path, root=0))


def assert_same_state(left, right) -> None:
    np.testing.assert_allclose(left.u.dat.data_ro, right.u.dat.data_ro)
    assert float(left.t) == pytest.approx(float(right.t))
    assert float(left.dt) == pytest.approx(float(right.dt))
    assert len(left.history) == len(right.history)
    for left_memory, right_memory in zip(
        left.history,
        right.history,
        strict=True,
    ):
        np.testing.assert_allclose(
            left_memory.dat.data_ro,
            right_memory.dat.data_ro,
        )
    left_velocities = getattr(left, "oscillator_velocities", ())
    right_velocities = getattr(right, "oscillator_velocities", ())
    assert len(left_velocities) == len(right_velocities)
    for left_velocity, right_velocity in zip(
        left_velocities,
        right_velocities,
        strict=True,
    ):
        np.testing.assert_allclose(
            left_velocity.dat.data_ro,
            right_velocity.dat.data_ro,
        )


def replace_state_value(state, path: tuple[object, ...], value) -> None:
    target = state
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value


def assert_restore_rejected_without_mutation(
    stepper,
    state,
    cases,
) -> None:
    committed = stepper.u.copy(deepcopy=True)
    committed_history = stepper.history
    for path, value, error in cases:
        malformed = copy.deepcopy(state)
        replace_state_value(malformed, path, value)
        with pytest.raises(ValueError, match=error):
            stepper.restore_checkpoint(malformed)
        assert fd.norm(stepper.u - committed) == 0.0
        for observed, expected in zip(
            stepper.history,
            committed_history,
            strict=True,
        ):
            assert fd.norm(observed - expected) == 0.0


def formulation_variant(formulation) -> str:
    return getattr(
        formulation,
        "scheme",
        getattr(formulation, "interpolant", ""),
    )


TIME_VARIANTS = (
    [
        pytest.param(
            representation,
            formulation,
            marker,
            id=f"{representation.__class__.__name__}-"
            f"{formulation.__class__.__name__}-"
            f"{formulation_variant(formulation)}-"
            f"{marker.__name__}",
        )
        for representation in (
            BirkSong(2),
            Diethelm2008(2),
            Diethelm2022(
                3,
                truncation_radius=2.0,
            ),
            YuanAgrawal(2),
            SumOfExponentials(
                target_error=0.1,
                t_final=0.3,
                min_step=0.1,
            ),
        )
        for formulation in (
            Recurrence(),
            AuxiliaryODE(scheme="backward_euler"),
            AuxiliaryODE(scheme="trapezoidal"),
        )
        for marker in (CaputoDerivative, RiemannLiouvilleDerivative)
    ]
    + [
        pytest.param(
            SineDiffusive(2),
            Oscillator(),
            marker,
            id=f"SineDiffusive-Oscillator-{marker.__name__}",
        )
        for marker in (CaputoDerivative, RiemannLiouvilleDerivative)
    ]
    + [
        pytest.param(
            FullHistory(),
            None,
            marker,
            id=f"FullHistory-{marker.__name__}",
        )
        for marker in (CaputoDerivative, RiemannLiouvilleDerivative)
    ]
)


@pytest.mark.verification
@pytest.mark.parametrize(
    "representation,formulation,marker",
    TIME_VARIANTS,
)
def test_time_variant_checkpoint_file_restart(
    tmp_path: Path,
    representation,
    formulation,
    marker,
) -> None:
    path = checkpoint_path(tmp_path, "time-variant")
    mesh = fd.UnitIntervalMesh(1, name="time_variant_mesh")
    _, source_time, source_dt, source = make_time_stepper(
        mesh,
        representation,
        formulation,
        marker,
        initial_value=1.0,
    )
    source.advance()
    source_time.assign(source_time + source_dt)
    with fd.CheckpointFile(str(path), "w", comm=mesh.comm) as checkpoint:
        source.save_checkpoint(checkpoint)

    with fd.CheckpointFile(str(path), "r", comm=mesh.comm) as checkpoint:
        restarted_mesh = checkpoint.load_mesh("time_variant_mesh")
        _, _, _, restarted = make_time_stepper(
            restarted_mesh,
            representation,
            formulation,
            marker,
            initial_value=9.0,
        )
        restarted.load_checkpoint(checkpoint)

    assert_same_state(restarted, source)
    source.advance()
    restarted.advance()
    assert_same_state(restarted, source)


@pytest.mark.verification
@pytest.mark.parametrize(
    "source_formulation,target_representation,target_formulation,target_marker,error",
    [
        pytest.param(
            Recurrence(),
            BirkSong(2, rate_scale=2.0),
            Recurrence(),
            CaputoDerivative,
            "representation",
            id="representation",
        ),
        pytest.param(
            Recurrence(),
            BirkSong(2),
            Recurrence(),
            RiemannLiouvilleDerivative,
            "time-memory operator",
            id="operator",
        ),
        pytest.param(
            AuxiliaryODE(scheme="backward_euler"),
            BirkSong(2),
            AuxiliaryODE(scheme="trapezoidal"),
            CaputoDerivative,
            "scheme",
            id="auxiliary-scheme",
        ),
    ],
)
def test_checkpoint_file_rejects_incompatible_time_configuration(
    tmp_path: Path,
    source_formulation,
    target_representation,
    target_formulation,
    target_marker,
    error: str,
) -> None:
    path = checkpoint_path(tmp_path, "incompatible")
    mesh = fd.UnitIntervalMesh(1, name="validation_mesh")
    _, source_time, source_dt, source = make_time_stepper(
        mesh,
        BirkSong(2),
        source_formulation,
        CaputoDerivative,
        initial_value=1.0,
    )
    source.advance()
    source_time.assign(source_time + source_dt)
    with fd.CheckpointFile(str(path), "w", comm=mesh.comm) as checkpoint:
        source.save_checkpoint(checkpoint)

    with fd.CheckpointFile(str(path), "r", comm=mesh.comm) as checkpoint:
        restarted_mesh = checkpoint.load_mesh("validation_mesh")
        target_u, _, _, target = make_time_stepper(
            restarted_mesh,
            target_representation,
            target_formulation,
            target_marker,
            initial_value=9.0,
        )
        committed = target_u.copy(deepcopy=True)
        with pytest.raises(ValueError, match=error):
            target.load_checkpoint(checkpoint)
        assert fd.norm(target_u - committed) == 0.0


def make_multi_order_stepper(mesh, representation, *, initial_value: float):
    space = fd.FunctionSpace(mesh, "CG", 1)
    u = fd.Function(space, name="u").assign(initial_value)
    test = fd.TestFunction(space)
    time = fd.Constant(0.0)
    dt = fd.Constant(0.1)
    residual = (
        fd.inner(CaputoDerivative(u, 0.3), test)
        + fd.inner(RiemannLiouvilleDerivative(u, 0.6), test)
        + fd.inner(u, test)
    ) * fd.dx
    stepper = FractionalTimeStepper(
        residual,
        representation,
        time,
        dt,
        u,
        solver_parameters={"ksp_type": "preonly", "pc_type": "lu"},
    )
    return time, dt, stepper


@pytest.mark.verification
@pytest.mark.parametrize(
    "representation",
    [
        pytest.param(BirkSong(2), id="BirkSong"),
        pytest.param(Diethelm2008(2), id="Diethelm2008"),
        pytest.param(FullHistory(), id="FullHistory"),
    ],
)
def test_multi_order_checkpoint_file_restart(
    tmp_path: Path,
    representation,
) -> None:
    path = checkpoint_path(tmp_path, "multi-order")
    mesh = fd.UnitIntervalMesh(1, name="multi_order_mesh")
    source_time, source_dt, source = make_multi_order_stepper(
        mesh,
        representation,
        initial_value=1.0,
    )
    source.advance()
    source_time.assign(source_time + source_dt)
    with fd.CheckpointFile(str(path), "w", comm=mesh.comm) as checkpoint:
        source.save_checkpoint(checkpoint)

    with fd.CheckpointFile(str(path), "r", comm=mesh.comm) as checkpoint:
        restarted_mesh = checkpoint.load_mesh("multi_order_mesh")
        _, _, restarted = make_multi_order_stepper(
            restarted_mesh,
            representation,
            initial_value=9.0,
        )
        restarted.load_checkpoint(checkpoint)

    assert_same_state(restarted, source)
    source.advance()
    restarted.advance()
    assert_same_state(restarted, source)


@pytest.mark.verification
def test_recurrence_restore_rejects_corrupt_state_before_mutation() -> None:
    mesh = fd.UnitIntervalMesh(1)
    _, source_time, source_dt, source = make_time_stepper(
        mesh,
        BirkSong(2),
        Recurrence(),
        CaputoDerivative,
        initial_value=1.0,
    )
    source.advance()
    source_time.assign(source_time + source_dt)
    state = source.checkpoint_state()
    _, _, _, target = make_time_stepper(
        mesh,
        BirkSong(2),
        Recurrence(),
        CaputoDerivative,
        initial_value=9.0,
    )
    assert_restore_rejected_without_mutation(
        target,
        state,
        [
            (("version",), 2, "unsupported"),
            (("parameters",), (0.5,), "parameters"),
            (("mode_groups",), [], "mode count"),
            (("u",), [], "physical field"),
            (("penultimate",), [], "physical field"),
            (("mode_groups", 0, 0, 0), float("nan"), "finite"),
            (("time",), "bad", "time metadata"),
            (("lower_limit",), float("nan"), "times must be finite"),
            (("dt",), 0.0, "dt must be"),
            (("stats", "solves"), "bad", "solver statistics"),
            (("stats", "failures"), -1, "nonnegative"),
            (("stats", "last_step_size"), 0.0, "last step size"),
        ],
    )


@pytest.mark.verification
def test_multi_order_restore_rejects_incompatible_layout() -> None:
    mesh = fd.UnitIntervalMesh(1)
    _, _, source = make_multi_order_stepper(
        mesh,
        BirkSong(2),
        initial_value=1.0,
    )
    state = source.checkpoint_state()
    _, _, target = make_multi_order_stepper(
        mesh,
        BirkSong(2),
        initial_value=9.0,
    )
    assert_restore_rejected_without_mutation(
        target,
        state,
        [
            (("version",), 2, "unsupported"),
            (
                ("operator_kinds",),
                ("caputo", "caputo"),
                "time-memory operators",
            ),
            (("parameters",), (0.3, 0.5), "parameters"),
            (("representations",), [], "representation count"),
            (("mode_groups",), [], "mode count"),
            (("mode_groups", 0, 0), [], "mode field"),
        ],
    )


@pytest.mark.verification
def test_auxiliary_restore_rejects_corrupt_state_before_mutation() -> None:
    mesh = fd.UnitIntervalMesh(1)
    formulation = AuxiliaryODE(scheme="trapezoidal")
    _, source_time, source_dt, source = make_time_stepper(
        mesh,
        BirkSong(2),
        formulation,
        CaputoDerivative,
        initial_value=1.0,
    )
    source.advance()
    source_time.assign(source_time + source_dt)
    state = source.checkpoint_state()
    _, _, _, target = make_time_stepper(
        mesh,
        BirkSong(2),
        formulation,
        CaputoDerivative,
        initial_value=9.0,
    )
    assert_restore_rejected_without_mutation(
        target,
        state,
        [
            (("version",), 2, "unsupported"),
            (
                ("operator_kinds",),
                ("riemann_liouville",),
                "time-memory operator",
            ),
            (("formulation", "kind"), "recurrence", "formulation"),
            (("formulation", "scheme"), "backward_euler", "auxiliary scheme"),
            (("parameters",), (0.5,), "parameter"),
            (("representations",), [], "representation count"),
            (("modes",), [], "mode count"),
            (("physical",), [], "wrong local shape"),
            (("modes", 0, 0), float("nan"), "finite"),
            (("time",), "bad", "time metadata"),
            (("lower_limit",), float("nan"), "times must be finite"),
            (("dt",), 0.0, "dt must be"),
            (("stats", "solves"), "bad", "solver statistics"),
            (("stats", "failures"), -1, "nonnegative"),
            (("stats", "last_step_size"), 0.0, "last step size"),
        ],
    )


@pytest.mark.verification
def test_full_history_restore_rejects_corrupt_state_before_mutation() -> None:
    mesh = fd.UnitIntervalMesh(1)
    _, source_time, source_dt, source = make_time_stepper(
        mesh,
        FullHistory(),
        None,
        CaputoDerivative,
        initial_value=1.0,
    )
    source.advance()
    source_time.assign(source_time + source_dt)
    state = source.checkpoint_state()
    _, _, _, target = make_time_stepper(
        mesh,
        FullHistory(),
        None,
        CaputoDerivative,
        initial_value=9.0,
    )
    assert_restore_rejected_without_mutation(
        target,
        state,
        [
            (("version",), 2, "unsupported"),
            (
                ("representations", 0, "interpolant"),
                "constant",
                "representation",
            ),
            (("operator_kinds",), ("riemann_liouville",), "operators"),
            (("parameters",), (0.5,), "parameters"),
            (("increments",), None, "history is invalid"),
            (("history_count",), 2, "history times"),
            (("times",), [0.0, 0.0], "history times"),
            (("times",), ["bad", 0.1], "time metadata"),
            (("time",), 0.2, "history times"),
            (("dt",), 0.0, "dt must be"),
            (("u",), [], "wrong local shape"),
            (("increments", 0), [], "wrong local shape"),
            (("increments", 0, 0), float("nan"), "finite"),
            (("stats", "solves"), "bad", "solver statistics"),
            (("stats", "failures"), -1, "solver statistics"),
            (("stats", "last_step_size"), 0.0, "last step size"),
        ],
    )


@pytest.mark.verification
def test_fast_cq_restore_rejects_corrupt_state_before_mutation() -> None:
    mesh = fd.UnitIntervalMesh(1)
    representation = FastObliviousCQ(
        num_levels=5,
        nodes_per_level=4,
        direct_steps=6,
    )
    _, source_time, source_dt, source = make_time_stepper(
        mesh,
        representation,
        None,
        CaputoDerivative,
        initial_value=1.0,
    )
    for _ in range(8):
        source.advance()
        source_time.assign(source_time + source_dt)
    state = source.checkpoint_state()
    _, _, _, target = make_time_stepper(
        mesh,
        representation,
        None,
        CaputoDerivative,
        initial_value=9.0,
    )
    assert_restore_rejected_without_mutation(
        target,
        state,
        [
            (("formulation", "kind"), "direct-history", "formulation"),
            (("uniform_step_size",), "bad", "metadata is invalid"),
            (("u",), [], "wrong local shape"),
            (("histories",), [], "term count"),
            (("histories", 0, "accepted_steps"), 7, "step count"),
            (("stats", "solves"), "bad", "solver statistics"),
            (("stats", "failures"), -1, "solver statistics"),
        ],
    )


def make_spatial_stepper(
    mesh,
    kind: str,
    option: str,
    *,
    initial_value: float,
):
    degree = 2 if kind == "riesz" and option == "hmatrix" else 1
    space = fd.FunctionSpace(mesh, "CG", degree)
    u = fd.Function(space, name="u").assign(initial_value)
    test = fd.TestFunction(space)
    time = fd.Constant(0.0)
    dt = fd.Constant(0.05)
    if kind == "spectral":
        boundary = fd.DirichletBC(space, 0.0, "on_boundary")
        boundary.apply(u)
        spatial = SpectralFractionalLaplacian(
            u,
            0.4,
            bcs=boundary,
            sinc_truncation_target=0.1,
            shift_cache=option,
            shift_solver_parameters={
                "ksp_type": "preonly",
                "pc_type": "lu",
            },
        )
    else:
        boundary = None
        spatial = RieszFractionalLaplacian(
            u,
            0.3,
            assembly=option,
            target_quadrature_degree=2,
            compression_tolerance=1.0e-8,
            leaf_size=1,
        )
    residual = (
        fd.inner(CaputoDerivative(u, 0.5), test) + fd.inner(spatial, test)
    ) * fd.dx
    stepper = FractionalTimeStepper(
        residual,
        BirkSong(2),
        time,
        dt,
        u,
        bcs=boundary,
        solver_parameters=MATRIX_FREE_SOLVER,
    )
    return time, dt, stepper


SPATIAL_VARIANTS = [
    pytest.param("spectral", cache, id=f"spectral-{cache}")
    for cache in ("stream", "all")
] + [
    pytest.param("riesz", assembly, id=f"riesz-{assembly}")
    for assembly in ("dense", "matfree", "hmatrix")
]


@pytest.mark.verification
@pytest.mark.parametrize("kind,option", SPATIAL_VARIANTS)
def test_spatial_variant_rebuilds_across_checkpoint(
    tmp_path: Path,
    kind: str,
    option: str,
) -> None:
    spatial_variant_restart(tmp_path, kind, option)


def spatial_variant_restart(
    tmp_path: Path,
    kind: str,
    option: str,
) -> None:
    path = checkpoint_path(tmp_path, f"{kind}-{option}", fd.COMM_WORLD)
    mesh = (
        fd.UnitIntervalMesh(
            2,
            comm=fd.COMM_WORLD,
            name="spatial_variant_mesh",
        )
        if kind == "spectral"
        else fd.UnitSquareMesh(
            1,
            1,
            comm=fd.COMM_WORLD,
            name="spatial_variant_mesh",
        )
    )
    source_time, source_dt, source = make_spatial_stepper(
        mesh,
        kind,
        option,
        initial_value=1.0,
    )
    source.advance()
    source_time.assign(source_time + source_dt)
    with fd.CheckpointFile(str(path), "w", comm=mesh.comm) as checkpoint:
        source.save_checkpoint(checkpoint)

    with fd.CheckpointFile(str(path), "r", comm=mesh.comm) as checkpoint:
        restarted_mesh = checkpoint.load_mesh("spatial_variant_mesh")
        _, _, restarted = make_spatial_stepper(
            restarted_mesh,
            kind,
            option,
            initial_value=9.0,
        )
        restarted.load_checkpoint(checkpoint)

    assert_same_state(restarted, source)
    source.advance()
    restarted.advance()
    assert_same_state(restarted, source)


@pytest.mark.parallel(nprocs=2)
@pytest.mark.verification
@pytest.mark.parametrize(
    "kind,option",
    [
        pytest.param("spectral", "stream", id="spectral-stream"),
        pytest.param("spectral", "all", id="spectral-all"),
        pytest.param("riesz", "matfree", id="riesz-matfree"),
        pytest.param("riesz", "hmatrix", id="riesz-hmatrix"),
    ],
)
def test_distributed_spatial_variant_rebuilds_across_checkpoint(
    tmp_path: Path,
    kind: str,
    option: str,
) -> None:
    spatial_variant_restart(tmp_path, kind, option)
