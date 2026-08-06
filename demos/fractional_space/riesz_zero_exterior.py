"""Zero-exterior Riesz action on a small affine triangular mesh."""

from firedrake import (
    Function,
    FunctionSpace,
    SpatialCoordinate,
    UnitSquareMesh,
    assemble,
    dx,
    inner,
)

from yonderdrake import RieszFractionalLaplacian


def main() -> None:
    """Apply the zero-exterior operator and print its energy."""
    mesh = UnitSquareMesh(2, 2)
    space = FunctionSpace(mesh, "CG", 1)
    x, y = SpatialCoordinate(mesh)
    u = Function(space, name="compact_domain_field").interpolate(
        x * (1.0 - x) * y * (1.0 - y)
    )
    operator = RieszFractionalLaplacian(
        u,
        0.35,
        extension="zero",
        target_quadrature_degree=5,
        assembly="matfree",
    )
    action = assemble(operator)
    energy = assemble(inner(u, action) * dx)

    print(f"zero-exterior Riesz energy: {energy:.12e}")
    print(operator.diagnostics())


if __name__ == "__main__":
    main()
