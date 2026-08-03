"""Caputo evolution coupled to the spectral Dirichlet fractional Laplacian."""

from math import gamma

from firedrake import (
    Constant,
    DirichletBC,
    Function,
    FunctionSpace,
    SpatialCoordinate,
    TestFunction,
    UnitSquareMesh,
    dx,
    inner,
    pi,
    sin,
)

from yonderdrake import (
    BirkSong,
    CaputoDerivative,
    FractionalTimeStepper,
    SpectralFractionalLaplacian,
)


def main() -> None:
    """Run a manufactured time-and-space-fractional diffusion problem."""
    mesh = UnitSquareMesh(4, 4)
    space = FunctionSpace(mesh, "CG", 1)
    x, y = SpatialCoordinate(mesh)
    u = Function(space, name="spectral_fractional_solution").assign(0.0)
    source = Function(space)
    v = TestFunction(space)
    t = Constant(0.0)
    dt = Constant(0.05)
    alpha, order, power = 0.5, 0.4, 2.0
    bc = DirichletBC(space, 0.0, "on_boundary")
    spatial = SpectralFractionalLaplacian(
        u,
        order,
        bcs=bc,
        sinc_truncation_target=1.0e-4,
        shift_cache="all",
        shift_solver_parameters={"ksp_type": "preonly", "pc_type": "lu"},
    )
    residual = (
        inner(CaputoDerivative(u, alpha), v)
        + inner(spatial, v)
        - inner(source, v)
    ) * dx
    stepper = FractionalTimeStepper(
        residual,
        BirkSong(16),
        t,
        dt,
        u,
        bcs=bc,
        solver_parameters={
            "snes_type": "ksponly",
            "mat_type": "matfree",
            "ksp_type": "gmres",
            "pc_type": "none",
        },
    )
    caputo_coefficient = gamma(power + 1.0) / gamma(power + 1.0 - alpha)
    for index in range(1, 5):
        target = index * float(dt)
        source.interpolate(
            (
                caputo_coefficient * target ** (power - alpha)
                + (2.0 * pi**2) ** order * target**power
            )
            * sin(pi * x)
            * sin(pi * y)
        )
        stepper.advance()
        t.assign(t + dt)

    print(f"completed spectral fractional diffusion at t={float(t):.2f}")
    print(stepper.solver_stats())
    print(spatial.diagnostics())


if __name__ == "__main__":
    main()
