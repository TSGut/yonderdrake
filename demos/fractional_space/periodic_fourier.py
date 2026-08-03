"""Apply the periodic Fourier fractional Laplacian to a known mode."""

from __future__ import annotations

from math import pi

import firedrake as fd

from yonderdrake import PeriodicFractionalLaplacian


def main() -> None:
    """Apply the periodic operator and print the known-mode error."""
    mesh = fd.PeriodicRectangleMesh(
        24,
        20,
        2.0 * pi,
        2.0 * pi,
        quadrilateral=True,
        reorder=False,
    )
    space = fd.FunctionSpace(mesh, "Q", 1)
    x, y = fd.SpatialCoordinate(mesh)
    u = fd.Function(space, name="periodic_mode").interpolate(
        fd.sin(2.0 * x) * fd.cos(3.0 * y)
    )
    s = 0.6

    action = fd.assemble(PeriodicFractionalLaplacian(u, s))
    exact = fd.Function(space).interpolate(13.0**s * u)
    relative_error = fd.norm(action - exact) / fd.norm(exact)

    if mesh.comm.rank == 0:
        print(f"relative Fourier-mode error: {relative_error:.3e}")


if __name__ == "__main__":
    main()
