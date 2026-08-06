"""Shared geometric admissibility for Riesz source and target supports."""

from __future__ import annotations

import numpy as np

from yonderdrake.riesz.geometry import SimplexGeometry


def _box_distance(
    left_lower: np.ndarray,
    left_upper: np.ndarray,
    right_lower: np.ndarray,
    right_upper: np.ndarray,
) -> float:
    separation = np.maximum(
        0.0,
        np.maximum(left_lower - right_upper, right_lower - left_upper),
    )
    return float(np.linalg.norm(separation))


def _admissible_bounds(
    left_lower: np.ndarray,
    left_upper: np.ndarray,
    left_support_lower: np.ndarray,
    left_support_upper: np.ndarray,
    right_lower: np.ndarray,
    right_upper: np.ndarray,
    right_support_lower: np.ndarray,
    right_support_upper: np.ndarray,
    eta: float,
) -> bool:
    support_distance = _box_distance(
        left_support_lower,
        left_support_upper,
        right_support_lower,
        right_support_upper,
    )
    distance = _box_distance(
        left_lower,
        left_upper,
        right_lower,
        right_upper,
    )
    left_diameter = float(np.linalg.norm(left_upper - left_lower))
    right_diameter = float(np.linalg.norm(right_upper - right_lower))
    return (
        support_distance > 0.0 and min(left_diameter, right_diameter) <= eta * distance
    )


def _admissible_cells(
    left: SimplexGeometry,
    right: SimplexGeometry,
    eta: float,
) -> bool:
    left_lower = np.min(left.vertices, axis=0)
    left_upper = np.max(left.vertices, axis=0)
    right_lower = np.min(right.vertices, axis=0)
    right_upper = np.max(right.vertices, axis=0)
    return _admissible_bounds(
        left_lower,
        left_upper,
        left_lower,
        left_upper,
        right_lower,
        right_upper,
        right_lower,
        right_upper,
        eta,
    )
