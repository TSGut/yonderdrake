"""Collective recurrence restart with Firedrake CheckpointFile."""

from __future__ import annotations

import argparse
from pathlib import Path

from firedrake import (
    COMM_WORLD,
    CheckpointFile,
    Constant,
    Function,
    FunctionSpace,
    TestFunction,
    UnitIntervalMesh,
    dx,
    inner,
    norm,
)

from yonderdrake import BirkSong, CaputoDerivative, FractionalTimeStepper


def make_stepper(mesh, initial_value: float):
    space = FunctionSpace(mesh, "CG", 1)
    u = Function(space, name="solution").assign(initial_value)
    test = TestFunction(space)
    time = Constant(0.0)
    dt = Constant(0.05)
    residual = (
        inner(CaputoDerivative(u, 0.6), test) + inner(u, test)
    ) * dx
    stepper = FractionalTimeStepper(
        residual,
        BirkSong(12),
        time,
        dt,
        u,
    )
    return u, time, dt, stepper


def main() -> None:
    """Save, reload, and continue one fractional-time state."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("fractional-time-restart.h5"),
    )
    args = parser.parse_args()

    mesh = UnitIntervalMesh(4, name="restart_mesh")
    u, t, dt, stepper = make_stepper(mesh, 1.0)
    stepper.advance()
    t.assign(t + dt)

    with CheckpointFile(str(args.checkpoint), "w", comm=mesh.comm) as checkpoint:
        stepper.save_checkpoint(checkpoint)

    with CheckpointFile(str(args.checkpoint), "r", comm=COMM_WORLD) as checkpoint:
        restarted_mesh = checkpoint.load_mesh("restart_mesh")
        u, t, dt, stepper = make_stepper(restarted_mesh, 0.0)
        stepper.load_checkpoint(checkpoint)

    stepper.advance()
    t.assign(t + dt)
    solution_norm = norm(u)
    if COMM_WORLD.rank == 0:
        print(f"restarted at t={float(t):.2f}, norm={solution_norm:.12f}")


if __name__ == "__main__":
    main()
