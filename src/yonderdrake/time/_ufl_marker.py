"""UFL implementation and traversal for time-memory markers."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import ufl
from ufl.constantvalue import as_ufl
from ufl.core.operator import Operator
from ufl.core.ufl_type import ufl_type
from ufl.corealg.multifunction import MultiFunction
from ufl.corealg.traversal import unique_pre_traversal


@ufl_type(
    inherit_shape_from_operand=0,
    inherit_indices_from_operand=0,
    num_ops=2,
)
class CaputoDerivativeMarker(Operator):
    """Shape-preserving data marker with no numerical-method policy."""

    __slots__ = ()

    def __init__(self, field: Any, alpha: Any) -> None:
        field = as_ufl(field)
        alpha = as_ufl(alpha)
        if alpha.ufl_shape != ():
            raise ValueError("alpha must be scalar")
        Operator.__init__(self, (field, alpha))

    @property
    def field(self) -> Any:
        return self.ufl_operands[0]

    @property
    def alpha(self) -> Any:
        return self.ufl_operands[1]

    def __str__(self) -> str:
        return f"CaputoDerivative({self.field}, {self.alpha})"


@ufl_type(
    inherit_shape_from_operand=0,
    inherit_indices_from_operand=0,
    num_ops=2,
)
class RiemannLiouvilleDerivativeMarker(Operator):
    """Shape-preserving marker for a left Riemann-Liouville derivative."""

    __slots__ = ()

    def __init__(self, field: Any, alpha: Any) -> None:
        field = as_ufl(field)
        alpha = as_ufl(alpha)
        if alpha.ufl_shape != ():
            raise ValueError("alpha must be scalar")
        Operator.__init__(self, (field, alpha))

    @property
    def field(self) -> Any:
        return self.ufl_operands[0]

    @property
    def alpha(self) -> Any:
        return self.ufl_operands[1]

    def __str__(self) -> str:
        return f"RiemannLiouvilleDerivative({self.field}, {self.alpha})"


@ufl_type(
    inherit_shape_from_operand=0,
    inherit_indices_from_operand=0,
    num_ops=2,
)
class ExponentialMemoryMarker(Operator):
    """Shape-preserving marker for one exponentially decaying memory state."""

    __slots__ = ()

    def __init__(self, field: Any, decay_rate: Any) -> None:
        field = as_ufl(field)
        decay_rate = as_ufl(decay_rate)
        if decay_rate.ufl_shape != ():
            raise ValueError("decay_rate must be scalar")
        Operator.__init__(self, (field, decay_rate))

    @property
    def field(self) -> Any:
        return self.ufl_operands[0]

    @property
    def decay_rate(self) -> Any:
        return self.ufl_operands[1]

    def __str__(self) -> str:
        return f"ExponentialMemory({self.field}, {self.decay_rate})"


def _clear_multifunction_handlers_cache() -> None:
    """Invalidate UFL's private dispatch cache or fail with a useful error."""
    try:
        cache = MultiFunction._handlers_cache
    except AttributeError as error:
        raise RuntimeError(
            "Unsupported UFL version: MultiFunction._handlers_cache is "
            "required to register Yonderdrake's fractional marker types"
        ) from error
    if not isinstance(cache, dict):
        raise RuntimeError(
            "Unsupported UFL version: MultiFunction._handlers_cache is no "
            "longer a dictionary"
        )
    cache.clear()


# Lazy import may leave stale UFL dispatch tables.
_clear_multifunction_handlers_cache()


FractionalDerivativeMarker = (
    CaputoDerivativeMarker | RiemannLiouvilleDerivativeMarker
)
TimeMemoryMarker = FractionalDerivativeMarker | ExponentialMemoryMarker


def find_fractional_derivative_markers(
    form: Any,
) -> tuple[FractionalDerivativeMarker, ...]:
    """Return all supported time-derivative markers in deterministic order."""
    found: list[FractionalDerivativeMarker] = []
    seen: set[FractionalDerivativeMarker] = set()
    marker_types = (CaputoDerivativeMarker, RiemannLiouvilleDerivativeMarker)
    for integral in form.integrals():
        for expression in unique_pre_traversal(integral.integrand()):
            if isinstance(expression, marker_types) and expression not in seen:
                found.append(expression)
                seen.add(expression)
    return tuple(found)


def find_time_memory_markers(
    form: Any,
) -> tuple[TimeMemoryMarker, ...]:
    """Return all supported time-memory markers in deterministic order."""
    found: list[TimeMemoryMarker] = []
    seen: set[TimeMemoryMarker] = set()
    marker_types = (
        CaputoDerivativeMarker,
        RiemannLiouvilleDerivativeMarker,
        ExponentialMemoryMarker,
    )
    for integral in form.integrals():
        for expression in unique_pre_traversal(integral.integrand()):
            if isinstance(expression, marker_types) and expression not in seen:
                found.append(expression)
                seen.add(expression)
    return tuple(found)


def replace_fractional_derivative_markers(
    form: Any,
    replacement: Callable[[FractionalDerivativeMarker], Any],
) -> Any:
    """Replace supported markers before Firedrake compiles a form."""
    markers = find_fractional_derivative_markers(form)
    return ufl.replace(form, {marker: replacement(marker) for marker in markers})


def replace_time_memory_markers(
    form: Any,
    replacement: Callable[[TimeMemoryMarker], Any],
) -> Any:
    """Replace supported time-memory markers before form compilation."""
    markers = find_time_memory_markers(form)
    return ufl.replace(form, {marker: replacement(marker) for marker in markers})


def evaluate_form_at_end_time(form: Any, t: Any, dt: Any) -> Any:
    """Evaluate a symbolic time coefficient at ``t + dt``."""
    if not isinstance(t, ufl.core.expr.Expr):
        return form
    return ufl.replace(form, {t: t + dt})
