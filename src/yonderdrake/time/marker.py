"""Lazy public constructors for method-neutral time-memory UFL markers."""

from __future__ import annotations

from math import isfinite
from typing import Any


def CaputoDerivative(u: Any, alpha: Any) -> Any:
    """Create a Caputo marker for ``FractionalTimeStepper``."""
    from yonderdrake.time._ufl_marker import CaputoDerivativeMarker

    return CaputoDerivativeMarker(u, alpha)


def RiemannLiouvilleDerivative(u: Any, alpha: Any) -> Any:
    """Create a left Riemann-Liouville marker."""
    from yonderdrake.time._ufl_marker import RiemannLiouvilleDerivativeMarker

    return RiemannLiouvilleDerivativeMarker(u, alpha)


def ExponentialMemory(u: Any, decay_rate: Any) -> Any:
    """Create a one-timescale fading-memory marker."""
    from yonderdrake.time._ufl_marker import ExponentialMemoryMarker

    return ExponentialMemoryMarker(u, decay_rate)


def CaputoFabrizioOperator(
    u: Any,
    alpha: Any,
    *,
    normalization: Any = 1.0,
) -> Any:
    """Return the published Caputo-Fabrizio exponential-memory operator."""
    try:
        order = float(alpha)
    except (TypeError, ValueError) as error:
        raise TypeError("alpha must be a real scalar") from error
    if not 0.0 < order < 1.0:
        raise ValueError("alpha must satisfy 0 < alpha < 1")
    try:
        scale = float(normalization)
    except (TypeError, ValueError) as error:
        raise TypeError("normalization must be a real scalar") from error
    if not isfinite(scale) or scale <= 0.0:
        raise ValueError("normalization must be finite and positive")
    return (scale / (1.0 - order)) * ExponentialMemory(
        u,
        order / (1.0 - order),
    )
