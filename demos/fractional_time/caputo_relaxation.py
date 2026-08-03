"""Mittag-Leffler relaxation with the default Caputo recurrence."""

from firedrake import (
    Constant,
    Function,
    FunctionSpace,
    TestFunction,
    UnitIntervalMesh,
    dx,
    inner,
)

from yonderdrake import BirkSong, CaputoDerivative, FractionalTimeStepper


def main() -> None:
    """Run the relaxation problem and print its final value."""
    mesh = UnitIntervalMesh(1)
    space = FunctionSpace(mesh, "CG", 1)
    u = Function(space, name="relaxation").assign(1.0)
    v = TestFunction(space)
    t = Constant(0.0)
    dt = Constant(0.02)

    residual = (inner(CaputoDerivative(u, 0.6), v) + inner(u, v)) * dx
    stepper = FractionalTimeStepper(
        residual,
        BirkSong(num_modes=48),
        t,
        dt,
        u,
    )

    while float(t) < 1.0 - 0.5 * float(dt):
        stepper.advance()
        t.assign(t + dt)

    print(f"u({float(t):.2f}) = {u.dat.data_ro[0]:.12f}")
    print(stepper.solver_stats())


if __name__ == "__main__":
    main()
