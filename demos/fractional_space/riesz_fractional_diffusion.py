"""Caputo decay coupled to the distinct zero-exterior Riesz operator."""

from firedrake import (
    Constant,
    Function,
    FunctionSpace,
    SpatialCoordinate,
    TestFunction,
    UnitSquareMesh,
    assemble,
    dx,
    inner,
)

from yonderdrake import (
    BirkSong,
    CaputoDerivative,
    FractionalTimeStepper,
    RieszFractionalLaplacian,
)


def main() -> None:
    """Run time-fractional diffusion with the zero-exterior operator."""
    mesh = UnitSquareMesh(2, 2)
    space = FunctionSpace(mesh, "CG", 1)
    x, y = SpatialCoordinate(mesh)
    u = Function(space, name="riesz_fractional_solution").interpolate(
        x * (1.0 - x) * y * (1.0 - y)
    )
    v = TestFunction(space)
    t = Constant(0.0)
    dt = Constant(0.05)
    spatial = RieszFractionalLaplacian(
        u,
        0.3,
        extension="zero",
        quadrature_degree=3,
        assembly="matfree",
    )
    residual = (
        inner(CaputoDerivative(u, 0.55), v) + inner(spatial, v)
    ) * dx
    stepper = FractionalTimeStepper(
        residual,
        BirkSong(8),
        t,
        dt,
        u,
        solver_parameters={
            "snes_type": "ksponly",
            "mat_type": "matfree",
            "ksp_type": "gmres",
            "pc_type": "none",
        },
    )
    for _ in range(2):
        stepper.advance()
        t.assign(t + dt)

    print(f"completed zero-exterior Riesz diffusion at t={float(t):.2f}")
    print(f"L2 energy={assemble(inner(u, u) * dx):.12e}")
    print(stepper.solver_stats())
    print(spatial.diagnostics())


if __name__ == "__main__":
    main()
