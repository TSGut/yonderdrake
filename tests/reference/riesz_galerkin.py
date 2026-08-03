"""Independent symmetric-energy reference on the unit square."""

from __future__ import annotations

from math import atan2, pi

import numpy as np
from scipy.integrate import quad

from yonderdrake.riesz.dense import RieszMeshData
from yonderdrake.riesz.outer_quadrature import SimplexQuadrature
from yonderdrake.riesz.triangle_action import riesz_normalization_2d


def _square_exterior_kernel(point: np.ndarray, order: float) -> float:
    """Direct polar integral of ``|x-y|^-2-2s`` over square exterior."""
    x, y = point
    cuts = sorted(
        atan2(vertex_y - y, vertex_x - x)
        for vertex_x, vertex_y in ((0, 0), (1, 0), (1, 1), (0, 1))
    )
    cuts.append(cuts[0] + 2.0 * pi)

    def radius(theta: float) -> float:
        direction = np.array([np.cos(theta), np.sin(theta)])
        candidates = []
        if direction[0] > 0.0:
            candidates.append((1.0 - x) / direction[0])
        elif direction[0] < 0.0:
            candidates.append(-x / direction[0])
        if direction[1] > 0.0:
            candidates.append((1.0 - y) / direction[1])
        elif direction[1] < 0.0:
            candidates.append(-y / direction[1])
        return min(value for value in candidates if value > 0.0)

    angular = 0.0
    for left, right in zip(cuts[:-1], cuts[1:], strict=True):
        angular += quad(
            lambda theta: radius(theta) ** (-2.0 * order),
            left,
            right,
            epsabs=2.0e-11,
            epsrel=2.0e-11,
        )[0]
    return angular / (2.0 * order)


def unit_square_symmetric_energy_matrix(
    mesh: RieszMeshData,
    order: float,
    quadrature: SimplexQuadrature,
) -> np.ndarray:
    """Approximate the independent symmetric double-integral bilinear form."""
    dimension = mesh.dimension
    points = []
    weights = []
    basis_values = []
    for cell, geometry, basis in zip(
        mesh.cell_dofs,
        mesh.geometries,
        mesh.local_basis,
        strict=True,
    ):
        for barycentric, reference_weight in zip(
            quadrature.barycentric,
            quadrature.weights,
            strict=True,
        ):
            values = np.zeros(dimension)
            point = barycentric @ geometry.vertices
            values[cell] = [
                polynomial(point) for polynomial in basis
            ]
            points.append(point)
            weights.append(2.0 * geometry.area * reference_weight)
            basis_values.append(values)
    point_array = np.asarray(points)
    weight_array = np.asarray(weights)
    basis_array = np.asarray(basis_values)
    normalization = riesz_normalization_2d(order)
    matrix = np.zeros((dimension, dimension))

    for left in range(point_array.shape[0]):
        for right in range(left + 1, point_array.shape[0]):
            difference = basis_array[left] - basis_array[right]
            distance = np.linalg.norm(point_array[left] - point_array[right])
            coefficient = (
                normalization
                * weight_array[left]
                * weight_array[right]
                / distance ** (2.0 + 2.0 * order)
            )
            matrix += coefficient * np.outer(difference, difference)
        exterior = _square_exterior_kernel(point_array[left], order)
        matrix += (
            normalization
            * weight_array[left]
            * exterior
            * np.outer(basis_array[left], basis_array[left])
        )
    return matrix
