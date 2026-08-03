"""Spectral fractional powers of Dirichlet and Neumann Laplacians."""

from __future__ import annotations

from math import isfinite
from typing import Any

from yonderdrake._firedrake import (
    ExternalOperatorData,
    require_real_float64_petsc,
    validate_exterior_bcs,
)
from yonderdrake.spectral.sinc import positive_power_sinc


def SpectralFractionalLaplacian(
    u: Any,
    s: Any,
    *,
    bcs: Any = None,
    sinc_truncation_target: float = 1.0e-10,
    shift_cache: str = "stream",
    shift_solver_parameters: Any = None,
    mass_solver_parameters: Any = None,
) -> Any:
    """Create ``(-Delta)^s`` with Dirichlet or natural Neumann boundaries.

    Omitting ``bcs`` uses the discrete Neumann Laplacian. Supplying ``bcs``
    selects the homogeneous-Dirichlet realization and must constrain the
    complete exterior boundary.
    """
    try:
        order = float(s)
    except (TypeError, ValueError) as error:
        raise TypeError("s must be a real scalar") from error
    if not isfinite(order) or not 0.0 < order < 1.0:
        raise ValueError("s must satisfy 0 < s < 1")
    quadrature = positive_power_sinc(order, sinc_truncation_target)
    if shift_cache not in {"all", "stream"}:
        raise ValueError("shift_cache must be 'all' or 'stream'")
    try:
        space = u.function_space()
    except AttributeError as error:
        raise TypeError(
            "u must be a Firedrake Function or symbolic coefficient"
        ) from error
    require_real_float64_petsc()
    boundary_conditions = (
        ()
        if bcs is None
        else validate_exterior_bcs(
            space,
            bcs,
            required=True,
            operator_name="the homogeneous-Dirichlet spectral realization",
        )
    )
    try:
        shift_parameters = dict(
            shift_solver_parameters
            or (
                {
                    "ksp_type": "preonly",
                    "pc_type": "lu",
                }
                if not boundary_conditions
                else {
                    "ksp_type": "cg",
                    "pc_type": "lu",
                    "ksp_rtol": 1.0e-12,
                }
            )
        )
        mass_parameters = dict(
            mass_solver_parameters
            or {"ksp_type": "preonly", "pc_type": "lu"}
        )
    except (TypeError, ValueError) as error:
        raise TypeError("solver parameters must be mappings") from error

    from yonderdrake.spectral._external import SpectralExternalOperator

    data = ExternalOperatorData(
        order=order,
        order_operand=s,
        quadrature=quadrature,
        bcs=boundary_conditions,
        shift_cache=shift_cache,
        shift_solver_parameters=shift_parameters,
        mass_solver_parameters=mass_parameters,
        manager=None,
    )
    return SpectralExternalOperator(
        u,
        function_space=space,
        operator_data=data,
    )
