"""Collective Firedrake checkpoint and restart tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

fd = pytest.importorskip("firedrake")

from yonderdrake import (  # noqa: E402
    AuxiliaryODE,
    BirkSong,
    CaputoDerivative,
    FractionalTimeStepper,
    Recurrence,
)


def make_stepper(mesh, formulation, *, initial_value: float):
    space = fd.FunctionSpace(mesh, "CG", 1)
    u = fd.Function(space, name="u").assign(initial_value)
    test = fd.TestFunction(space)
    time = fd.Constant(0.0)
    dt = fd.Constant(0.1)
    residual = (
        fd.inner(CaputoDerivative(u, 0.6), test) + fd.inner(u, test)
    ) * fd.dx
    stepper = FractionalTimeStepper(
        residual,
        BirkSong(4),
        time,
        dt,
        u,
        formulation=formulation,
        solver_parameters={"ksp_type": "preonly", "pc_type": "lu"},
    )
    return u, time, dt, stepper


def checkpoint_path(tmp_path: Path, comm, name: str) -> Path:
    path = str(tmp_path / name) if comm.rank == 0 else None
    return Path(comm.bcast(path, root=0))


def restart_check(formulation, path: Path, comm) -> None:
    mesh = fd.UnitIntervalMesh(4, comm=comm, name="checkpoint_mesh")
    source_u, source_t, source_dt, source = make_stepper(
        mesh,
        formulation,
        initial_value=1.0,
    )
    source.advance()
    source_t.assign(source_t + source_dt)
    with fd.CheckpointFile(str(path), "w", comm=comm) as checkpoint:
        source.save_checkpoint(checkpoint)

    with fd.CheckpointFile(str(path), "r", comm=comm) as checkpoint:
        restarted_mesh = checkpoint.load_mesh("checkpoint_mesh")
        target_u, target_t, target_dt, target = make_stepper(
            restarted_mesh,
            formulation,
            initial_value=9.0,
        )
        target.load_checkpoint(checkpoint)

    np.testing.assert_allclose(target_u.dat.data_ro, source_u.dat.data_ro)
    assert float(target_t) == pytest.approx(float(source_t))
    assert float(target_dt) == pytest.approx(float(source_dt))
    for target_mode, source_mode in zip(
        target.history,
        source.history,
        strict=True,
    ):
        np.testing.assert_allclose(
            target_mode.dat.data_ro,
            source_mode.dat.data_ro,
        )

    source.advance()
    target.advance()
    np.testing.assert_allclose(
        target_u.dat.data_ro,
        source_u.dat.data_ro,
        atol=2.0e-11,
    )


@pytest.mark.parametrize(
    "formulation",
    [Recurrence(), AuxiliaryODE(scheme="trapezoidal")],
)
def test_checkpoint_file_restart(tmp_path: Path, formulation) -> None:
    restart_check(
        formulation,
        checkpoint_path(tmp_path, fd.COMM_WORLD, "state.h5"),
        fd.COMM_WORLD,
    )


def test_checkpoint_file_rejects_duplicate_name(tmp_path: Path) -> None:
    mesh = fd.UnitIntervalMesh(1, name="checkpoint_mesh")
    _, _, _, stepper = make_stepper(
        mesh,
        Recurrence(),
        initial_value=1.0,
    )
    path = checkpoint_path(tmp_path, mesh.comm, "duplicate.h5")
    with fd.CheckpointFile(str(path), "w", comm=mesh.comm) as checkpoint:
        stepper.save_checkpoint(checkpoint, name="restart")
        with pytest.raises(ValueError, match="already exists"):
            stepper.save_checkpoint(checkpoint, name="restart")


@pytest.mark.parallel(nprocs=2)
@pytest.mark.verification
def test_checkpoint_file_restart_two_ranks(tmp_path: Path) -> None:
    restart_check(
        Recurrence(),
        checkpoint_path(tmp_path, fd.COMM_WORLD, "state-two.h5"),
        fd.COMM_WORLD,
    )


@pytest.mark.parallel(nprocs=4)
@pytest.mark.verification
def test_checkpoint_file_restart_four_ranks(tmp_path: Path) -> None:
    restart_check(
        Recurrence(),
        checkpoint_path(tmp_path, fd.COMM_WORLD, "state-four.h5"),
        fd.COMM_WORLD,
    )
