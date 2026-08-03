"""Periodic Fourier fractional Laplacian."""

from __future__ import annotations

from math import isfinite
from typing import Any

from yonderdrake._firedrake import (
    ExternalOperatorData,
    require_real_float64_petsc,
)


def PeriodicFractionalLaplacian(u: Any, s: Any) -> Any:
    """Create the Fourier multiplier ``(-Delta_periodic)^s``."""
    try:
        order = float(s)
    except (TypeError, ValueError) as error:
        raise TypeError("s must be a real scalar") from error
    if not isfinite(order) or not 0.0 < order < 1.0:
        raise ValueError("s must satisfy 0 < s < 1")
    try:
        space = u.function_space()
    except AttributeError as error:
        raise TypeError(
            "u must be a Firedrake Function or symbolic coefficient"
        ) from error
    if u.ufl_shape != ():
        raise NotImplementedError(
            "PeriodicFractionalLaplacian supports scalar fields only"
        )
    element = space.ufl_element()
    degree = element.degree()
    degrees = degree if isinstance(degree, tuple) else (degree,)
    if element.family() not in {"Lagrange", "Q"} or any(
        value != 1 for value in degrees
    ):
        raise NotImplementedError(
            "PeriodicFractionalLaplacian supports continuous degree-one "
            "nodal fields only"
        )
    mesh = space.mesh()
    dimension = int(mesh.geometric_dimension)
    cell = mesh.ufl_cell().cellname
    supported_cell = {1: "interval", 2: "quadrilateral", 3: "hexahedron"}
    if dimension not in supported_cell or cell != supported_cell[dimension]:
        raise NotImplementedError(
            "PeriodicFractionalLaplacian currently supports 1D intervals, "
            "2D quadrilateral rectangles, and 3D hexahedral boxes"
        )
    require_real_float64_petsc()

    from yonderdrake.periodic._external import PeriodicExternalOperator
    from yonderdrake.periodic.grid import PeriodicGridMap

    grid = PeriodicGridMap.from_space(space)
    return PeriodicExternalOperator(
        u,
        function_space=space,
        operator_data=ExternalOperatorData(
            order=order,
            order_operand=s,
            grid=grid,
            manager=None,
        ),
    )
