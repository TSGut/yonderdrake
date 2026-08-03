"""Real-valued auxiliary-differential-equation acoustic PML."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, log
from typing import Any

import numpy as np


def _positive_real(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a real scalar") from error
    if not isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _bounds(values: Any, dimension: int, name: str) -> tuple[tuple[float, float], ...]:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (dimension, 2) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must have shape ({dimension}, 2)")
    if np.any(array[:, 0] >= array[:, 1]):
        raise ValueError(f"every interval in {name} must have positive width")
    return tuple((float(lower), float(upper)) for lower, upper in array)


@dataclass(frozen=True, slots=True, kw_only=True)
class CaputoWismerPML:
    """Directional damping fields for a time-domain acoustic PML."""

    damping: tuple[Any, ...]
    outer_boundary: bool = True
    interior_bounds: tuple[tuple[float, float], ...] | None = None
    reflection: float | None = None
    polynomial_order: int | None = None

    @classmethod
    def box(
        cls,
        mesh: Any,
        interior_bounds: Any,
        *,
        reference_speed: float,
        reflection: float = 1.0e-6,
        polynomial_order: int = 3,
        outer_bounds: Any = None,
        outer_boundary: bool = True,
    ) -> CaputoWismerPML:
        """Construct polynomial PML profiles around an interior box."""
        try:
            import firedrake as fd
            from mpi4py import MPI
        except ImportError as error:
            raise RuntimeError(
                "CaputoWismerPML.box requires an active Firedrake environment"
            ) from error
        dimension = int(mesh.geometric_dimension)
        if dimension not in {2, 3}:
            raise NotImplementedError("CaputoWismerPML supports 2D or 3D meshes")
        interior = _bounds(interior_bounds, dimension, "interior_bounds")
        if outer_bounds is None:
            coordinates = np.asarray(mesh.coordinates.dat.data_ro, dtype=np.float64)
            local_lower = np.min(coordinates, axis=0)
            local_upper = np.max(coordinates, axis=0)
            global_lower = np.empty_like(local_lower)
            global_upper = np.empty_like(local_upper)
            mesh.comm.Allreduce(local_lower, global_lower, op=MPI.MIN)
            mesh.comm.Allreduce(local_upper, global_upper, op=MPI.MAX)
            outer = tuple(
                (float(global_lower[index]), float(global_upper[index]))
                for index in range(dimension)
            )
        else:
            outer = _bounds(outer_bounds, dimension, "outer_bounds")
        speed = _positive_real(reference_speed, "reference_speed")
        try:
            target = float(reflection)
        except (TypeError, ValueError) as error:
            raise TypeError("reflection must be a real scalar") from error
        if not isfinite(target) or not 0.0 < target < 1.0:
            raise ValueError("reflection must satisfy 0 < reflection < 1")
        if (
            not isinstance(polynomial_order, int)
            or isinstance(polynomial_order, bool)
            or polynomial_order < 1
        ):
            raise ValueError("polynomial_order must be a positive integer")

        x = fd.SpatialCoordinate(mesh)
        profiles = []
        for axis, ((inner_lower, inner_upper), (outer_lower, outer_upper)) in enumerate(
            zip(interior, outer, strict=True)
        ):
            left_width = inner_lower - outer_lower
            right_width = outer_upper - inner_upper
            if left_width <= 0.0 or right_width <= 0.0:
                raise ValueError(
                    "interior_bounds must leave a positive PML width on every side"
                )
            left_max = -((polynomial_order + 1) * speed * log(target)) / (
                2.0 * left_width
            )
            right_max = -((polynomial_order + 1) * speed * log(target)) / (
                2.0 * right_width
            )
            left_distance = fd.max_value(inner_lower - x[axis], 0.0)
            right_distance = fd.max_value(x[axis] - inner_upper, 0.0)
            profiles.append(
                left_max * (left_distance / left_width) ** polynomial_order
                + right_max * (right_distance / right_width) ** polynomial_order
            )
        return cls(
            damping=tuple(profiles),
            outer_boundary=outer_boundary,
            interior_bounds=interior,
            reflection=target,
            polynomial_order=polynomial_order,
        )


__all__ = ["CaputoWismerPML"]
