"""Target quadrature on affine triangles and tetrahedra."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, isfinite

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import roots_jacobi


@dataclass(frozen=True)
class SimplexQuadrature:
    """Reference-simplex barycentric nodes and weights."""

    degree: int
    barycentric: np.ndarray
    weights: np.ndarray
    rule: str = "ordinary"
    singular_exponent: float | None = None

    @property
    def num_points(self) -> int:
        return int(self.weights.size)

    @property
    def dimension(self) -> int:
        return int(self.barycentric.shape[1] - 1)


def triangle_quadrature(degree: int) -> SimplexQuadrature:
    """Return an interior tensor-Gauss/Duffy rule of the requested degree."""
    if not isinstance(degree, int) or isinstance(degree, bool) or degree < 1:
        raise ValueError("quadrature_degree must be a positive integer")
    count = max(1, ceil((degree + 2) / 2))
    nodes, weights = leggauss(count)
    unit_nodes = 0.5 * (nodes + 1.0)
    unit_weights = 0.5 * weights
    barycentric = []
    output_weights = []
    for first, first_weight in zip(unit_nodes, unit_weights, strict=True):
        for second, second_weight in zip(unit_nodes, unit_weights, strict=True):
            barycentric.append(
                [1.0 - first - (1.0 - first) * second, first, (1.0 - first) * second]
            )
            output_weights.append(
                first_weight * second_weight * (1.0 - first)
            )
    barycentric_array = np.asarray(barycentric, dtype=np.float64)
    weight_array = np.asarray(output_weights, dtype=np.float64)
    barycentric_array.setflags(write=False)
    weight_array.setflags(write=False)
    return SimplexQuadrature(
        degree=degree,
        barycentric=barycentric_array,
        weights=weight_array,
    )


def edge_triangle_quadrature(
    degree: int,
    order: float,
    *,
    zero_trace: bool,
    field_degree: int,
) -> SimplexQuadrature:
    """Return an edge-fitted Duffy-Jacobi target rule."""
    if not isinstance(degree, int) or isinstance(degree, bool) or degree < 1:
        raise ValueError("quadrature_degree must be a positive integer")
    order = float(order)
    if not isfinite(order) or not 0.0 < order < 1.0:
        raise ValueError("order must satisfy 0 < order < 1")
    if field_degree not in {1, 2}:
        raise ValueError("field_degree must be 1 or 2")

    count = max(1, ceil((degree + 2) / 2))
    exponent = (
        -2.0 * order
        if not zero_trace and order < 0.5
        else min(0.0, 1.0 - 2.0 * order)
    )
    radial_nodes, radial_weights = roots_jacobi(
        count,
        1.0,
        exponent,
    )
    radial_nodes = 0.5 * (radial_nodes + 1.0)
    radial_weights = radial_weights / 2.0 ** (exponent + 2.0)
    tangent_nodes, tangent_weights = leggauss(count)
    tangent_nodes = 0.5 * (tangent_nodes + 1.0)
    tangent_weights = 0.5 * tangent_weights

    barycentric = []
    weights = []
    for opposite in range(3):
        left = (opposite + 1) % 3
        right = (opposite + 2) % 3
        for radial, radial_weight in zip(
            radial_nodes,
            radial_weights,
            strict=True,
        ):
            for tangent, tangent_weight in zip(
                tangent_nodes,
                tangent_weights,
                strict=True,
            ):
                point = np.full(3, radial / 3.0)
                point[left] += (1.0 - radial) * (1.0 - tangent)
                point[right] += (1.0 - radial) * tangent
                barycentric.append(point)
                weights.append(
                    radial_weight
                    * tangent_weight
                    * radial ** (-exponent)
                    / 3.0
                )

    barycentric_array = np.asarray(barycentric, dtype=np.float64)
    weight_array = np.asarray(weights, dtype=np.float64)
    weight_array *= 0.5 / np.sum(weight_array)
    if field_degree == 2:
        # Restore quadratic moments after removing the Jacobi weight.
        quadratic = np.sum(barycentric_array**2, axis=1)
        mean = np.sum(weight_array * quadratic) / 0.5
        denominator = np.sum(
            weight_array * quadratic * (quadratic - mean)
        )
        correction = (
            0.25 - np.sum(weight_array * quadratic)
        ) / denominator
        weight_array *= 1.0 + correction * (quadratic - mean)

    barycentric_array.setflags(write=False)
    weight_array.setflags(write=False)
    return SimplexQuadrature(
        degree=degree,
        barycentric=barycentric_array,
        weights=weight_array,
        rule="boundary",
        singular_exponent=exponent,
    )


def tetrahedron_quadrature(degree: int) -> SimplexQuadrature:
    """Return an interior tensor-Gauss/Duffy tetrahedron rule."""
    if not isinstance(degree, int) or isinstance(degree, bool) or degree < 1:
        raise ValueError("quadrature_degree must be a positive integer")
    count = max(1, ceil((degree + 3) / 2))
    nodes, weights = leggauss(count)
    unit_nodes = 0.5 * (nodes + 1.0)
    unit_weights = 0.5 * weights
    barycentric = []
    output_weights = []
    for first, first_weight in zip(unit_nodes, unit_weights, strict=True):
        for second, second_weight in zip(unit_nodes, unit_weights, strict=True):
            for third, third_weight in zip(
                unit_nodes,
                unit_weights,
                strict=True,
            ):
                barycentric.append(
                    [
                        1.0 - first,
                        first * (1.0 - second),
                        first * second * (1.0 - third),
                        first * second * third,
                    ]
                )
                output_weights.append(
                    first_weight
                    * second_weight
                    * third_weight
                    * first**2
                    * second
                )
    barycentric_array = np.asarray(barycentric, dtype=np.float64)
    weight_array = np.asarray(output_weights, dtype=np.float64)
    barycentric_array.setflags(write=False)
    weight_array.setflags(write=False)
    return SimplexQuadrature(
        degree=degree,
        barycentric=barycentric_array,
        weights=weight_array,
    )


def face_tetrahedron_quadrature(
    degree: int,
    order: float,
    *,
    zero_trace: bool,
    field_degree: int,
) -> SimplexQuadrature:
    """Return a face-fitted Duffy-Jacobi tetrahedron target rule."""
    if not isinstance(degree, int) or isinstance(degree, bool) or degree < 1:
        raise ValueError("quadrature_degree must be a positive integer")
    order = float(order)
    if not isfinite(order) or not 0.0 < order < 1.0:
        raise ValueError("order must satisfy 0 < order < 1")
    if field_degree not in {1, 2}:
        raise ValueError("field_degree must be 1 or 2")

    count = max(1, ceil((degree + 3) / 2))
    exponent = (
        -2.0 * order
        if not zero_trace and order < 0.5
        else min(0.0, 1.0 - 2.0 * order)
    )
    radial_nodes, radial_weights = roots_jacobi(count, 2.0, exponent)
    radial_nodes = 0.5 * (radial_nodes + 1.0)
    radial_weights = radial_weights / 2.0 ** (exponent + 3.0)
    face_rule = triangle_quadrature(degree)
    barycentric = []
    output_weights = []
    for opposite in range(4):
        face_indices = [index for index in range(4) if index != opposite]
        for radial, radial_weight in zip(
            radial_nodes,
            radial_weights,
            strict=True,
        ):
            for face_point, face_weight in zip(
                face_rule.barycentric,
                face_rule.weights,
                strict=True,
            ):
                point = np.full(4, radial / 4.0)
                point[face_indices] += (1.0 - radial) * face_point
                barycentric.append(point)
                output_weights.append(
                    radial_weight
                    * face_weight
                    * radial ** (-exponent)
                )

    barycentric_array = np.asarray(barycentric, dtype=np.float64)
    weight_array = np.asarray(output_weights, dtype=np.float64)
    reference_volume = 1.0 / 6.0
    weight_array *= reference_volume / np.sum(weight_array)
    if field_degree == 2:
        features = [
            barycentric_array[:, index] for index in range(4)
        ]
        desired = [1.0 / 24.0] * 4
        for left in range(4):
            for right in range(left, 4):
                features.append(
                    barycentric_array[:, left] * barycentric_array[:, right]
                )
                desired.append(1.0 / 60.0 if left == right else 1.0 / 120.0)
        moment_matrix = np.asarray(features, dtype=np.float64)
        residual = np.asarray(desired) - moment_matrix @ weight_array
        gram = (moment_matrix * weight_array[None, :]) @ moment_matrix.T
        correction = np.linalg.lstsq(gram, residual, rcond=None)[0]
        weight_array *= 1.0 + moment_matrix.T @ correction

    barycentric_array.setflags(write=False)
    weight_array.setflags(write=False)
    return SimplexQuadrature(
        degree=degree,
        barycentric=barycentric_array,
        weights=weight_array,
        rule="boundary",
        singular_exponent=exponent,
    )
