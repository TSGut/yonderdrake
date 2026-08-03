"""A prescribed linear solution with sine diffusive memory."""

from math import gamma

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
    Oscillator,
    SineDiffusive,
)


def main() -> None:
    """Recover u=t from its analytic Caputo derivative."""
    alpha = 0.5
    final_time = 1.0
    step_size = 0.01
    representation = SineDiffusive(128)
    mesh = UnitIntervalMesh(1)
    space = FunctionSpace(mesh, "CG", 1)
    u = Function(space, name="linear_solution")
    test = TestFunction(space)
    time = Constant(0.0)
    dt = Constant(step_size)
    forcing = (time + dt) ** (1.0 - alpha) / gamma(2.0 - alpha)
    residual = (
        inner(CaputoDerivative(u, alpha), test) - inner(forcing, test)
    ) * dx
    stepper = FractionalTimeStepper(
        residual,
        representation,
        time,
        dt,
        u,
        formulation=Oscillator(),
    )

    while float(time) < final_time - 0.5 * step_size:
        stepper.advance()
        time.assign(time + dt)

    observed = float(u.dat.data_ro[0])
    print(f"u({float(time):.2f}) = {observed:.12f}")
    print(f"absolute error = {abs(observed - final_time):.6e}")
    print(stepper.solver_stats())


if __name__ == "__main__":
    main()
