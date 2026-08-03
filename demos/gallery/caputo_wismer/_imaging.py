"""Shared forward and reconstruction routines for sensor-data demos."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from yonderdrake.applications import (
    CaputoWismerInverseProblem,
    CaputoWismerModel,
    reconstruct_initial_pressure,
)


def _solver_parameters(parallel: bool) -> dict[str, Any]:
    if parallel:
        return {
            "snes_type": "ksponly",
            "ksp_type": "cg",
            "ksp_rtol": 1.0e-8,
            "pc_type": "gamg",
        }
    return {
        "snes_type": "ksponly",
        "ksp_type": "preonly",
        "pc_type": "lu",
    }


def reconstruct_sensor_data(
    model: CaputoWismerModel,
    sensor_data: np.ndarray,
    *,
    method: Literal["kaltenbacher", "time_reversal"] = "kaltenbacher",
    regularization: float = 1.0e-6,
    max_iterations: int = 100,
    tolerance: float = 1.0e-5,
    positivity: bool = True,
    compensate_attenuation: bool = True,
    filter_length: float | None = None,
    filter_order: int = 2,
) -> Any:
    """Reconstruct iteratively or by lossless or compensated time reversal."""
    if method == "kaltenbacher":
        problem = CaputoWismerInverseProblem(
            model,
            sensor_data,
            regularization=regularization,
        )
        result = problem.solve(
            max_iterations=max_iterations,
            tolerance=tolerance,
            positivity=positivity,
            warm_start=True,
        )
        status = "converged" if result.converged else "stopped at its cap"
        initial_objective = result.objective_history[0]
        reduction = result.objective / initial_objective if initial_objective else 0.0
        if model.space.mesh().comm.rank == 0:
            print(
                f"Kaltenbacher reconstruction {status} after "
                f"{result.iterations} iterations "
                f"and {result.function_evaluations} evaluations "
                f"(objective {result.objective:.6e}, "
                f"relative {reduction:.3e}, "
                f"forward {result.forward_seconds:.1f}s, "
                f"adjoint {result.adjoint_seconds:.1f}s, "
                f"total {result.elapsed_seconds:.1f}s)"
            )
        return result.pressure
    if method != "time_reversal":
        raise ValueError("method must be 'kaltenbacher' or 'time_reversal'")
    return reconstruct_initial_pressure(
        model,
        sensor_data,
        method=method,
        regularization=regularization,
        max_iterations=max_iterations,
        positivity=positivity,
        compensate_attenuation=compensate_attenuation,
        filter_length=filter_length,
        filter_order=filter_order,
    )


def vessel_values(
    coordinates: Any,
    *,
    dimension: int,
    width: float,
) -> np.ndarray:
    """Return a branching vessel-like initial-pressure phantom."""
    points = np.asarray(coordinates, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != dimension:
        raise ValueError("coordinates have the wrong dimension")
    if dimension == 2:
        polylines = (
            np.column_stack(
                (
                    np.linspace(-0.58, 0.55, 13),
                    -0.12 + 0.14 * np.sin(2.8 * np.linspace(-0.58, 0.55, 13)),
                )
            ),
            np.asarray(((-0.10, -0.10), (0.18, 0.18), (0.42, 0.39))),
            np.asarray(((0.12, -0.08), (0.34, -0.31), (0.48, -0.48))),
            np.asarray(((-0.34, -0.01), (-0.48, 0.25), (-0.54, 0.44))),
        )
    elif dimension == 3:
        axis = np.linspace(-0.55, 0.5, 13)
        polylines = (
            np.column_stack(
                (axis, -0.1 + 0.12 * np.sin(2.8 * axis), 0.08 * np.cos(axis))
            ),
            np.asarray(((-0.08, -0.09, 0.07), (0.18, 0.2, 0.18), (0.4, 0.38, 0.28))),
            np.asarray(((0.10, -0.08, 0.07), (0.31, -0.3, -0.10), (0.45, -0.45, -0.2))),
        )
    else:
        raise ValueError("dimension must be 2 or 3")

    distance_squared = np.full(points.shape[0], np.inf)
    for polyline in polylines:
        for start, end in zip(polyline[:-1], polyline[1:], strict=True):
            direction = end - start
            denominator = float(np.dot(direction, direction))
            parameter = np.clip(
                ((points - start) @ direction) / denominator,
                0.0,
                1.0,
            )
            closest = start + parameter[:, None] * direction
            distance_squared = np.minimum(
                distance_squared,
                np.sum((points - closest) ** 2, axis=1),
            )
    return np.exp(-0.5 * distance_squared / width**2)


def normalized_array(values: Any) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).copy()
    maximum = float(np.max(np.abs(values), initial=0.0))
    return values if maximum == 0.0 else values / maximum


def normalized_values(field: Any) -> np.ndarray:
    return normalized_array(field.dat.data_ro)
