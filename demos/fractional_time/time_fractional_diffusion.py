"""Homogeneous time-fractional diffusion on the unit square."""

from firedrake import (
    Constant,
    DirichletBC,
    Function,
    FunctionSpace,
    SpatialCoordinate,
    TestFunction,
    UnitSquareMesh,
    dx,
    grad,
    inner,
    pi,
    sin,
)

from yonderdrake import CaputoDerivative, Diethelm2008, FractionalTimeStepper


def main() -> None:
    """Run homogeneous time-fractional diffusion on the unit square."""
    mesh = UnitSquareMesh(16, 16)
    space = FunctionSpace(mesh, "CG", 1)
    x = SpatialCoordinate(mesh)
    u = Function(space, name="solution").interpolate(
        sin(pi * x[0]) * sin(pi * x[1])
    )
    v = TestFunction(space)
    t = Constant(0.0)
    dt = Constant(0.01)
    bc = DirichletBC(space, 0.0, "on_boundary")

    residual = (
        inner(CaputoDerivative(u, 0.7), v) + inner(grad(u), grad(v))
    ) * dx
    stepper = FractionalTimeStepper(
        residual,
        Diethelm2008(num_modes=64),
        t,
        dt,
        u,
        bcs=bc,
        solver_parameters={"ksp_type": "cg", "pc_type": "hypre"},
    )

    while float(t) < 0.1 - 0.5 * float(dt):
        stepper.advance()
        t.assign(t + dt)

    print(f"advanced to t={float(t):.3f}")
    print(stepper.solver_stats())


if __name__ == "__main__":
    main()
