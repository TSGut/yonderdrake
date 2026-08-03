"""Mittag-Leffler relaxation with tolerance-driven exponential memory."""

from firedrake import (
    Constant,
    Function,
    FunctionSpace,
    TestFunction,
    UnitIntervalMesh,
    dx,
    inner,
)

from yonderdrake import (
    CaputoDerivative,
    FractionalTimeStepper,
    SumOfExponentials,
)


def main() -> None:
    """Run a relaxation problem over the representation's valid interval."""
    final_time = 1.0
    step_size = 0.02
    representation = SumOfExponentials(
        target_error=1.0e-6,
        t_final=final_time,
        min_step=step_size,
    )
    mesh = UnitIntervalMesh(1)
    space = FunctionSpace(mesh, "CG", 1)
    u = Function(space, name="relaxation").assign(1.0)
    test = TestFunction(space)
    time = Constant(0.0)
    dt = Constant(step_size)
    residual = (
        inner(CaputoDerivative(u, 0.6), test) + inner(u, test)
    ) * dx
    stepper = FractionalTimeStepper(
        residual,
        representation,
        time,
        dt,
        u,
    )

    while float(time) < final_time - 0.5 * step_size:
        stepper.advance()
        time.assign(time + dt)

    spectrum = representation.spectrum(0.6)
    print(f"u({float(time):.2f}) = {u.dat.data_ro[0]:.12f}")
    print(f"derived modes = {spectrum.rates.size}")
    print(stepper.solver_stats())


if __name__ == "__main__":
    main()
