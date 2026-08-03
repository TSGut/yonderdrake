"""Caputo-Wismer material and sensor models."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite, pi
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


def _centre(values: Sequence[float], dimension: int) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (dimension,) or not np.all(np.isfinite(result)):
        raise ValueError(f"center must contain {dimension} finite coordinates")
    return result


def ring_sensor_locations(
    num_sensors: int,
    radius: float,
    *,
    center: Sequence[float] = (0.0, 0.0),
) -> np.ndarray:
    """Return uniformly spaced sensor centres on a two-dimensional ring."""
    if (
        not isinstance(num_sensors, int)
        or isinstance(num_sensors, bool)
        or num_sensors < 1
    ):
        raise ValueError("num_sensors must be a positive integer")
    radius_value = _positive_real(radius, "radius")
    center_values = _centre(center, 2)
    angles = 2.0 * pi * np.arange(num_sensors, dtype=np.float64) / num_sensors
    return center_values + radius_value * np.column_stack(
        (np.cos(angles), np.sin(angles))
    )


def sphere_sensor_locations(
    num_sensors: int,
    radius: float,
    *,
    center: Sequence[float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    """Return approximately uniform sensor centres on a three-dimensional sphere."""
    if (
        not isinstance(num_sensors, int)
        or isinstance(num_sensors, bool)
        or num_sensors < 1
    ):
        raise ValueError("num_sensors must be a positive integer")
    radius_value = _positive_real(radius, "radius")
    center_values = _centre(center, 3)
    indices = np.arange(num_sensors, dtype=np.float64)
    z = 1.0 - 2.0 * (indices + 0.5) / num_sensors
    azimuth = pi * (3.0 - np.sqrt(5.0)) * indices
    radial = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    points = np.column_stack((radial * np.cos(azimuth), radial * np.sin(azimuth), z))
    return center_values + radius_value * points


@dataclass(frozen=True, slots=True, kw_only=True)
class CaputoWismerMaterial:
    """One piecewise acoustic material in a Caputo-Wismer model."""

    indicator: Any
    density: Any
    wave_speed: Any
    damping: Any
    alpha: Any


class SensorArray:
    """Gaussian volume sensors with a matching discrete adjoint."""

    def __init__(
        self,
        space: Any,
        locations: Any,
        *,
        width: float,
    ) -> None:
        try:
            import firedrake as fd
        except ImportError as error:
            raise RuntimeError(
                "SensorArray requires an active Firedrake environment"
            ) from error
        if space.ufl_element().family() != "Lagrange":
            raise NotImplementedError(
                "SensorArray supports continuous Lagrange spaces only"
            )
        if space.value_shape != ():
            raise NotImplementedError("SensorArray supports scalar fields only")
        mesh = space.mesh()
        dimension = int(mesh.geometric_dimension)
        if dimension not in {2, 3}:
            raise NotImplementedError("SensorArray supports 2D or 3D meshes")
        points = np.asarray(locations, dtype=np.float64)
        if (
            points.ndim != 2
            or points.shape[0] == 0
            or points.shape[1] != dimension
            or not np.all(np.isfinite(points))
        ):
            raise ValueError(f"locations must have shape (num_sensors, {dimension})")
        width_value = _positive_real(width, "width")

        self._fd = fd
        self.space = space
        self.locations = points.copy()
        self.locations.setflags(write=False)
        self.width = width_value
        self.dimension = dimension
        coordinates = fd.SpatialCoordinate(mesh)
        degree = int(space.ufl_element().degree())
        coordinate_space = fd.VectorFunctionSpace(mesh, "CG", degree)
        coordinate_values = np.asarray(
            fd.Function(coordinate_space).interpolate(coordinates).dat.data_ro,
            dtype=np.float64,
        )
        if coordinate_values.shape != (space.dof_dset.size, dimension):
            raise RuntimeError("sensor coordinate and field layouts do not match")
        offsets = coordinate_values[None, :, :] - points[:, None, :]
        kernel_values = np.exp(
            -0.5 * np.sum(offsets * offsets, axis=2) / width_value**2
        )

        trial = fd.TrialFunction(space)
        test = fd.TestFunction(space)
        mass_matrix = fd.assemble(
            fd.inner(trial, test) * fd.dx,
            mat_type="aij",
        )
        load = fd.assemble(test * fd.dx)
        local_masses = kernel_values @ np.asarray(
            load.dat.data_ro,
            dtype=np.float64,
        )
        self._comm = mesh.comm
        masses = np.asarray(
            self._comm.allreduce(local_masses),
            dtype=np.float64,
        )
        if np.any(~np.isfinite(masses)) or np.any(masses <= np.finfo(np.float64).tiny):
            index = int(
                np.flatnonzero(
                    (~np.isfinite(masses)) | (masses <= np.finfo(np.float64).tiny)
                )[0]
            )
            raise ValueError(f"sensor {index} has no resolvable support on the mesh")
        kernel_values /= masses[:, None]
        kernel_buffer = fd.Function(space, name="sensor_kernel")
        row_buffer = fd.Cofunction(space.dual(), name="sensor_row")
        row_values = np.empty_like(kernel_values)
        for index, values in enumerate(kernel_values):
            kernel_buffer.dat.data[:] = values
            with (
                kernel_buffer.dat.vec_ro as kernel_vector,
                row_buffer.dat.vec as row_vector,
            ):
                mass_matrix.petscmat.mult(kernel_vector, row_vector)
            row_values[index] = row_buffer.dat.data_ro
        self._kernel_values = kernel_values
        self._row_values = row_values
        self._adjoint_buffer = fd.Function(space, name="sensor_adjoint")

    @classmethod
    def ring(
        cls,
        space: Any,
        num_sensors: int,
        radius: float,
        *,
        width: float,
        center: Sequence[float] = (0.0, 0.0),
    ) -> SensorArray:
        """Construct a two-dimensional circular sensor array."""
        return cls(
            space,
            ring_sensor_locations(
                num_sensors,
                radius,
                center=center,
            ),
            width=width,
        )

    @classmethod
    def sphere(
        cls,
        space: Any,
        num_sensors: int,
        radius: float,
        *,
        width: float,
        center: Sequence[float] = (0.0, 0.0, 0.0),
    ) -> SensorArray:
        """Construct a three-dimensional spherical sensor array."""
        return cls(
            space,
            sphere_sensor_locations(
                num_sensors,
                radius,
                center=center,
            ),
            width=width,
        )

    @property
    def num_sensors(self) -> int:
        return int(self.locations.shape[0])

    def sample(self, field: Any) -> np.ndarray:
        """Return the spatially averaged value recorded by every sensor."""
        if field.function_space() != self.space:
            raise ValueError("field must belong to the sensor function space")
        local_values = self._row_values @ np.asarray(
            field.dat.data_ro,
            dtype=np.float64,
        )
        return np.asarray(
            self._comm.allreduce(local_values),
            dtype=np.float64,
        )

    def adjoint_field(
        self,
        values: Any,
        *,
        out: Any = None,
    ) -> Any:
        """Map sensor values back to their normalized spatial kernels."""
        coefficients = np.asarray(values, dtype=np.float64)
        if coefficients.shape != (self.num_sensors,):
            raise ValueError(f"values must have shape ({self.num_sensors},)")
        if not np.all(np.isfinite(coefficients)):
            raise ValueError("sensor values must be finite")
        result = self._adjoint_buffer if out is None else out
        if result.function_space() != self.space:
            raise ValueError("out must belong to the sensor function space")
        result.dat.data[:] = coefficients @ self._kernel_values
        return result

    def adjoint_covector(
        self,
        values: Any,
        *,
        out: Any = None,
    ) -> Any:
        """Apply the exact coefficient-space transpose of ``sample``."""
        coefficients = np.asarray(values, dtype=np.float64)
        if coefficients.shape != (self.num_sensors,):
            raise ValueError(f"values must have shape ({self.num_sensors},)")
        if not np.all(np.isfinite(coefficients)):
            raise ValueError("sensor values must be finite")
        result = (
            self._fd.Cofunction(self.space.dual(), name="sensor_covector")
            if out is None
            else out
        )
        if result.function_space() != self.space.dual():
            raise ValueError("out must belong to the dual sensor space")
        result.dat.data[:] = coefficients @ self._row_values
        return result
