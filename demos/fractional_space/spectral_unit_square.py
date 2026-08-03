"""Spectral action on a unit-square Dirichlet eigenfunction."""

from firedrake import (
    DirichletBC,
    Function,
    FunctionSpace,
    SpatialCoordinate,
    UnitSquareMesh,
    assemble,
    dx,
    inner,
    norm,
    pi,
    sin,
)

from yonderdrake import SpectralFractionalLaplacian


def main() -> None:
    """Apply the spectral operator to a known Dirichlet eigenfunction."""
    mesh = UnitSquareMesh(12, 12)
    space = FunctionSpace(mesh, "CG", 1)
    x, y = SpatialCoordinate(mesh)
    u = Function(space, name="eigenfunction").interpolate(
        sin(pi * x) * sin(pi * y)
    )
    bc = DirichletBC(space, 0.0, "on_boundary")
    order = 0.6
    operator = SpectralFractionalLaplacian(
        u,
        order,
        bcs=bc,
        sinc_truncation_target=1.0e-6,
        shift_cache="all",
        shift_solver_parameters={"ksp_type": "preonly", "pc_type": "lu"},
    )
    action = assemble(operator)
    continuum = Function(space).interpolate((2.0 * pi**2) ** order * u)
    relative_error = norm(action - continuum) / norm(continuum)
    energy = assemble(inner(u, action) * dx)

    print(f"relative continuum-eigenfunction error: {relative_error:.6e}")
    print(f"positive quadratic form: {energy:.12e}")
    print(operator.diagnostics())


if __name__ == "__main__":
    main()
