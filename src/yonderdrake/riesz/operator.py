"""Zero-exterior Riesz/restricted fractional Laplacian."""

from __future__ import annotations

from math import isfinite
from typing import Any

from yonderdrake._firedrake import (
    ExternalOperatorData,
    require_real_float64_petsc,
    validate_exterior_bcs,
)


def RieszFractionalLaplacian(
    u: Any,
    s: Any,
    *,
    extension: str = "zero",
    quadrature_degree: int = 6,
    quadrature_rule: str = "boundary",
    assembly: str = "matfree",
    compression_tolerance: float = 1.0e-6,
    admissibility: float = 1.0,
    leaf_size: int = 16,
    bcs: Any = None,
    mass_solver_parameters: Any = None,
) -> Any:
    """Create the zero-exterior operator on affine triangles or tetrahedra."""
    try:
        order = float(s)
    except (TypeError, ValueError) as error:
        raise TypeError("s must be a real scalar") from error
    if not isfinite(order) or not 0.0 < order < 1.0:
        raise ValueError("s must satisfy 0 < s < 1")
    if extension != "zero":
        raise ValueError("extension must be 'zero'")
    if quadrature_rule not in {"boundary", "ordinary"}:
        raise ValueError("quadrature_rule must be 'boundary' or 'ordinary'")
    if assembly not in {"dense", "matfree", "hmatrix"}:
        raise ValueError("assembly must be 'dense', 'matfree', or 'hmatrix'")
    if (
        not isinstance(quadrature_degree, int)
        or isinstance(quadrature_degree, bool)
        or quadrature_degree < 1
    ):
        raise ValueError("quadrature_degree must be a positive integer")
    try:
        compression_tolerance = float(compression_tolerance)
    except (TypeError, ValueError) as error:
        raise TypeError("compression_tolerance must be a real scalar") from error
    if not isfinite(compression_tolerance) or not 0.0 < compression_tolerance < 1.0:
        raise ValueError("compression_tolerance must satisfy 0 < tolerance < 1")
    try:
        admissibility = float(admissibility)
    except (TypeError, ValueError) as error:
        raise TypeError("admissibility must be a real scalar") from error
    if not isfinite(admissibility) or admissibility <= 0.0:
        raise ValueError("admissibility must be finite and positive")
    if not isinstance(leaf_size, int) or isinstance(leaf_size, bool) or leaf_size < 1:
        raise ValueError("leaf_size must be a positive integer")
    try:
        mass_parameters = dict(
            mass_solver_parameters
            or {
                "ksp_type": "cg",
                "ksp_rtol": 1.0e-12,
                "ksp_atol": 1.0e-15,
                "pc_type": "jacobi",
            }
        )
    except (TypeError, ValueError) as error:
        raise TypeError("mass_solver_parameters must be a mapping") from error
    try:
        space = u.function_space()
    except AttributeError as error:
        raise TypeError(
            "u must be a Firedrake Function or symbolic coefficient"
        ) from error
    if u.ufl_shape != ():
        raise NotImplementedError(
            "RieszFractionalLaplacian supports scalar fields only"
        )
    element = space.ufl_element()
    if element.family() != "Lagrange" or element.degree() not in {1, 2}:
        raise NotImplementedError(
            "RieszFractionalLaplacian supports continuous Lagrange degree 1 or 2"
        )
    mesh = space.mesh()
    require_real_float64_petsc()
    boundary_conditions = validate_exterior_bcs(
        space,
        bcs,
        required=order >= 0.5,
        operator_name="the zero-exterior Riesz realization",
    )
    supported_cell = (
        mesh.geometric_dimension == 2
        and mesh.ufl_cell().cellname == "triangle"
    ) or (
        mesh.geometric_dimension == 3
        and mesh.ufl_cell().cellname == "tetrahedron"
    )
    if not supported_cell:
        raise NotImplementedError(
            "RieszFractionalLaplacian supports affine 2D triangle and "
            "3D tetrahedral meshes only"
        )
    coordinate_degree = (
        mesh.coordinates.function_space().ufl_element().degree()
    )
    degree_values = (
        coordinate_degree
        if isinstance(coordinate_degree, tuple)
        else (coordinate_degree,)
    )
    if any(degree != 1 for degree in degree_values):
        raise NotImplementedError(
            "RieszFractionalLaplacian requires degree-1 affine mesh coordinates"
        )
    if mesh.comm.size != 1 and assembly == "dense":
        raise NotImplementedError(
            "assembly='dense' currently supports serial execution only"
        )

    from yonderdrake.riesz._external import RieszExternalOperator

    return RieszExternalOperator(
        u,
        function_space=space,
        operator_data=ExternalOperatorData(
            order=order,
            order_operand=s,
            extension=extension,
            quadrature_degree=quadrature_degree,
            quadrature_rule=quadrature_rule,
            assembly=assembly,
            compression_tolerance=compression_tolerance,
            admissibility=admissibility,
            leaf_size=leaf_size,
            bcs=boundary_conditions,
            mass_solver_parameters=mass_parameters,
            manager=None,
        ),
    )
