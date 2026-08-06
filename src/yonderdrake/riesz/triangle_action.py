"""Exact boundary formula for triangle-supported quadratic polynomials."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from math import gamma, isfinite, pi

import numpy as np
from scipy.special import hyp2f1

from yonderdrake.riesz.geometry import SimplexGeometry, TriangleGeometry


class SingularPointError(ValueError):
    """The classical pointwise action diverges at this support interface."""


@dataclass(frozen=True)
class AffinePolynomial:
    """The physical polynomial ``constant + gradient dot x``."""

    constant: float
    gradient: np.ndarray

    def __post_init__(self) -> None:
        gradient = np.asarray(self.gradient, dtype=np.float64)
        if gradient.shape not in {(2,), (3,)} or not np.all(np.isfinite(gradient)):
            raise ValueError("gradient must be a finite vector of length 2 or 3")
        if not isfinite(float(self.constant)):
            raise ValueError("constant must be finite")
        gradient = gradient.copy()
        gradient.setflags(write=False)
        object.__setattr__(self, "constant", float(self.constant))
        object.__setattr__(self, "gradient", gradient)

    def __call__(self, point: object) -> float:
        return float(self.constant + np.dot(self.gradient, point))


@dataclass(frozen=True)
class QuadraticPolynomial:
    """The physical polynomial ``c + g.x + x.H.x/2``."""

    constant: float
    gradient: np.ndarray
    hessian: np.ndarray

    def __post_init__(self) -> None:
        gradient = np.asarray(self.gradient, dtype=np.float64)
        hessian = np.asarray(self.hessian, dtype=np.float64)
        if gradient.shape not in {(2,), (3,)} or not np.all(np.isfinite(gradient)):
            raise ValueError("gradient must be a finite vector of length 2 or 3")
        dimension = gradient.size
        if hessian.shape != (dimension, dimension) or not np.all(np.isfinite(hessian)):
            raise ValueError(
                "hessian must be a finite square matrix matching the gradient"
            )
        if not np.allclose(hessian, hessian.T, rtol=0.0, atol=1.0e-14):
            raise ValueError("hessian must be symmetric")
        if not isfinite(float(self.constant)):
            raise ValueError("constant must be finite")
        gradient = gradient.copy()
        hessian = 0.5 * (hessian + hessian.T)
        gradient.setflags(write=False)
        hessian.setflags(write=False)
        object.__setattr__(self, "constant", float(self.constant))
        object.__setattr__(self, "gradient", gradient)
        object.__setattr__(self, "hessian", hessian)

    def __call__(self, point: object) -> float:
        x = np.asarray(point, dtype=np.float64)
        return float(
            self.constant + np.dot(self.gradient, x) + 0.5 * np.dot(x, self.hessian @ x)
        )

    def gradient_at(self, point: object) -> np.ndarray:
        return self.gradient + self.hessian @ np.asarray(point, dtype=np.float64)


SimplexPolynomial = AffinePolynomial | QuadraticPolynomial


@dataclass(frozen=True)
class SimplexPiece:
    geometry: SimplexGeometry
    polynomial: SimplexPolynomial


def combine_polynomials(
    polynomials: Iterable[SimplexPolynomial],
    coefficients: Iterable[float],
) -> SimplexPolynomial:
    """Form a local polynomial from nodal coefficients."""
    polynomial_values = tuple(polynomials)
    coefficient_values = tuple(float(value) for value in coefficients)
    if len(polynomial_values) != len(coefficient_values):
        raise ValueError("polynomials and coefficients must have equal length")
    constant = sum(
        value * polynomial.constant
        for value, polynomial in zip(
            coefficient_values,
            polynomial_values,
            strict=True,
        )
    )
    dimension = polynomial_values[0].gradient.size if polynomial_values else 2
    if any(polynomial.gradient.size != dimension for polynomial in polynomial_values):
        raise ValueError("all polynomials must have the same spatial dimension")
    gradient = sum(
        (
            value * polynomial.gradient
            for value, polynomial in zip(
                coefficient_values,
                polynomial_values,
                strict=True,
            )
        ),
        np.zeros(dimension),
    )
    if all(
        isinstance(polynomial, AffinePolynomial) for polynomial in polynomial_values
    ):
        return AffinePolynomial(constant, gradient)
    hessian = sum(
        (
            value
            * (
                polynomial.hessian
                if isinstance(polynomial, QuadraticPolynomial)
                else np.zeros((dimension, dimension))
            )
            for value, polynomial in zip(
                coefficient_values,
                polynomial_values,
                strict=True,
            )
        ),
        np.zeros((dimension, dimension)),
    )
    return QuadraticPolynomial(constant, gradient, hessian)


def riesz_normalization_2d(order: float) -> float:
    """Return the Fourier-multiplier normalization ``C_(2,s)``."""
    return riesz_normalization(2, order)


def riesz_normalization(dimension: int, order: float) -> float:
    """Return the Fourier-multiplier normalization ``C_(d,s)``."""
    if dimension not in {2, 3}:
        raise ValueError("dimension must be 2 or 3")
    order = _validate_order(order)
    return float(
        2.0 ** (2.0 * order)
        * order
        * gamma(0.5 * dimension + order)
        / (pi ** (0.5 * dimension) * gamma(1.0 - order))
    )


def _validate_order(order: float) -> float:
    order = float(order)
    if not isfinite(order) or not 0.0 < order < 1.0:
        raise ValueError("order must satisfy 0 < order < 1")
    return order


def line_power_integral(
    left: float,
    right: float,
    distance: float,
    exponent: float,
    *,
    tolerance: float = 1.0e-14,
) -> float:
    r"""Evaluate ``integral_left^right (distance^2+z^2)^(-exponent) dz``."""
    return float(
        _line_power_integral_many(
            np.asarray([left], dtype=np.float64),
            np.asarray([right], dtype=np.float64),
            np.asarray([distance], dtype=np.float64),
            float(exponent),
            tolerance=tolerance,
        )[0]
    )


def _line_power_integral_many(
    left: np.ndarray,
    right: np.ndarray,
    distance: np.ndarray,
    exponent: float,
    *,
    tolerance: float,
) -> np.ndarray:
    """Vectorized equivalent of :func:`line_power_integral`."""
    left_values = np.asarray(left, dtype=np.float64).copy()
    right_values = np.asarray(right, dtype=np.float64).copy()
    distances = np.abs(np.asarray(distance, dtype=np.float64))
    if not (left_values.shape == right_values.shape == distances.shape):
        raise ValueError("line-integral arrays must have matching shapes")
    signs = np.ones(left_values.shape, dtype=np.float64)
    reversed_limits = right_values < left_values
    if bool(np.any(reversed_limits)):
        temporary = left_values[reversed_limits].copy()
        left_values[reversed_limits] = right_values[reversed_limits]
        right_values[reversed_limits] = temporary
        signs[reversed_limits] = -1.0

    result = np.empty(left_values.shape, dtype=np.float64)
    regular = distances > tolerance
    if bool(np.any(regular)):
        regular_indices = np.flatnonzero(regular)
        regular_left = left_values[regular]
        regular_right = right_values[regular]
        regular_distance = distances[regular]
        if abs(exponent - 0.5) <= 8.0 * np.finfo(np.float64).eps:
            result[regular] = np.arcsinh(regular_right / regular_distance) - np.arcsinh(
                regular_left / regular_distance
            )
        else:
            same_side = regular_left * regular_right > 0.0
            away_from_origin = np.minimum(
                np.abs(regular_left),
                np.abs(regular_right),
            )
            reciprocal = same_side & (regular_distance < 0.25 * away_from_origin)
            if bool(np.any(reciprocal)):
                power = 1.0 - 2.0 * exponent
                reciprocal_left = regular_left[reciprocal]
                reciprocal_right = regular_right[reciprocal]
                reciprocal_distance = regular_distance[reciprocal]

                def reciprocal_primitive(values: np.ndarray) -> np.ndarray:
                    radii = np.abs(values)
                    transformed = hyp2f1(
                        exponent,
                        exponent - 0.5,
                        exponent + 0.5,
                        -((reciprocal_distance / radii) ** 2),
                    )
                    return np.sign(values) * radii**power * transformed / power

                result[regular_indices[reciprocal]] = reciprocal_primitive(
                    reciprocal_right
                ) - reciprocal_primitive(reciprocal_left)
            direct = ~reciprocal
            if bool(np.any(direct)):
                direct_left = regular_left[direct]
                direct_right = regular_right[direct]
                direct_distance = regular_distance[direct]
                scale = direct_distance ** (-2.0 * exponent)

                def primitive(values: np.ndarray) -> np.ndarray:
                    ratios = values / direct_distance
                    return (
                        values
                        * scale
                        * hyp2f1(
                            0.5,
                            exponent,
                            1.5,
                            -(ratios * ratios),
                        )
                    )

                result[regular_indices[direct]] = primitive(direct_right) - primitive(
                    direct_left
                )

    zero_distance = ~regular
    if bool(np.any(zero_distance)):
        zero_indices = np.flatnonzero(zero_distance)
        zero_left = left_values[zero_distance]
        zero_right = right_values[zero_distance]
        crosses_origin = (zero_left <= 0.0) & (zero_right >= 0.0)
        power = 1.0 - 2.0 * exponent
        divergent = crosses_origin & (power <= 0.0)
        result[zero_indices[divergent]] = np.inf
        finite = ~divergent
        if bool(np.any(finite)):
            finite_left = zero_left[finite]
            finite_right = zero_right[finite]

            def zero_distance_primitive(values: np.ndarray) -> np.ndarray:
                primitive_values = np.zeros_like(values)
                nonzero = values != 0.0
                if abs(power) <= 8.0 * np.finfo(np.float64).eps:
                    primitive_values[nonzero] = np.sign(values[nonzero]) * np.log(
                        np.abs(values[nonzero])
                    )
                else:
                    primitive_values[nonzero] = (
                        np.sign(values[nonzero])
                        * np.abs(values[nonzero]) ** power
                        / power
                    )
                return primitive_values

            result[zero_indices[finite]] = zero_distance_primitive(
                finite_right
            ) - zero_distance_primitive(finite_left)
    return signs * result


def _line_first_moment_many(
    left: np.ndarray,
    right: np.ndarray,
    distance: np.ndarray,
    exponent: float,
) -> np.ndarray:
    """Evaluate the first-moment line integral over matching arrays."""
    power = 1.0 - exponent
    squared_distance = np.asarray(distance, dtype=np.float64) ** 2
    left_squared = squared_distance + np.asarray(left, dtype=np.float64) ** 2
    right_squared = squared_distance + np.asarray(right, dtype=np.float64) ** 2
    if abs(power) <= 8.0 * np.finfo(np.float64).eps:
        return 0.5 * (np.log(right_squared) - np.log(left_squared))
    return (right_squared**power - left_squared**power) / (2.0 * power)


def _piece_boundary_action_many(
    piece: SimplexPiece,
    points: np.ndarray,
    order: float,
) -> np.ndarray:
    """Evaluate one triangle-supported polynomial at many target points."""
    geometry = piece.geometry
    if not isinstance(geometry, TriangleGeometry):
        raise TypeError("triangle boundary action requires TriangleGeometry")
    polynomial = piece.polynomial
    values_at_points = polynomial.constant + points @ polynomial.gradient
    if isinstance(polynomial, QuadraticPolynomial):
        values_at_points += 0.5 * np.einsum(
            "ij,jk,ik->i",
            points,
            polynomial.hessian,
            points,
        )
        gradients = polynomial.gradient[None, :] + points @ polynomial.hessian
    else:
        gradients = np.broadcast_to(polynomial.gradient, points.shape)
    total = np.zeros(points.shape[0], dtype=np.float64)
    for vertex, length, tangent, normal in zip(
        geometry.vertices,
        geometry.lengths,
        geometry.tangents,
        geometry.outward_normals,
        strict=True,
    ):
        relative_left = vertex[None, :] - points
        left = relative_left @ tangent
        right = left + length
        distance = relative_left @ normal
        away_from_edge = np.abs(distance) > geometry.tolerance
        high = np.zeros(points.shape[0], dtype=np.float64)
        if bool(np.any(away_from_edge)):
            high[away_from_edge] = _line_power_integral_many(
                left[away_from_edge],
                right[away_from_edge],
                distance[away_from_edge],
                1.0 + order,
                tolerance=geometry.tolerance,
            )
            total[away_from_edge] += (
                values_at_points[away_from_edge]
                * distance[away_from_edge]
                * high[away_from_edge]
            )
        low = _line_power_integral_many(
            left,
            right,
            distance,
            order,
            tolerance=geometry.tolerance,
        )
        total += (gradients @ normal) * low
        if isinstance(polynomial, QuadraticPolynomial) and bool(np.any(away_from_edge)):
            first = _line_first_moment_many(
                left[away_from_edge],
                right[away_from_edge],
                distance[away_from_edge],
                1.0 + order,
            )
            second = (
                low[away_from_edge]
                - distance[away_from_edge] ** 2 * high[away_from_edge]
            )
            tangent_hessian = polynomial.hessian @ tangent
            normal_hessian = polynomial.hessian @ normal
            quadratic = (
                float(np.dot(tangent, tangent_hessian)) * second
                + 2.0
                * distance[away_from_edge]
                * float(np.dot(tangent, normal_hessian))
                * first
                + distance[away_from_edge] ** 2
                * float(np.dot(normal, normal_hessian))
                * high[away_from_edge]
            )
            total[away_from_edge] -= (
                order * distance[away_from_edge] * quadratic / (2.0 * (1.0 - order))
            )
    return total


def triangle_affine_action(
    geometry: TriangleGeometry,
    polynomial: AffinePolynomial,
    point: object,
    order: float,
) -> float:
    """Fractional Laplacian of one zero-extended triangle polynomial."""
    x = np.asarray(point, dtype=np.float64)
    if x.shape != (2,):
        raise ValueError("point must be a vector of length 2")
    return float(
        triangle_affine_action_many(
            geometry,
            polynomial,
            x[None, :],
            order,
        )[0]
    )


def triangle_affine_action_many(
    geometry: TriangleGeometry,
    polynomial: AffinePolynomial,
    points: object,
    order: float,
) -> np.ndarray:
    """Evaluate the triangle action over a batch of target points."""
    targets = np.asarray(points, dtype=np.float64)
    if targets.ndim != 2 or targets.shape[1] != 2:
        raise ValueError("points must have shape (num_points, 2)")
    return _triangle_polynomial_action_many(
        geometry,
        polynomial,
        targets,
        order,
    )


def triangle_quadratic_action(
    geometry: TriangleGeometry,
    polynomial: QuadraticPolynomial,
    point: object,
    order: float,
) -> float:
    """Fractional Laplacian of one zero-extended triangle polynomial."""
    x = np.asarray(point, dtype=np.float64)
    if x.shape != (2,):
        raise ValueError("point must be a vector of length 2")
    return float(
        _triangle_polynomial_action_many(
            geometry,
            polynomial,
            x[None, :],
            order,
        )[0]
    )


def _triangle_polynomial_action_many(
    geometry: TriangleGeometry,
    polynomial: SimplexPolynomial,
    targets: np.ndarray,
    order: float,
) -> np.ndarray:
    """Evaluate a triangle polynomial after pointwise singularity checks."""
    order = _validate_order(order)
    if not np.all(np.isfinite(targets)):
        raise ValueError("points must contain only finite values")
    for point in targets:
        if geometry.classify(point) not in {"edge", "vertex"}:
            continue
        trace = abs(polynomial(point))
        trace_scale = max(
            1.0,
            abs(polynomial.constant),
            np.linalg.norm(polynomial.gradient) * geometry.diameter,
        )
        if isinstance(polynomial, QuadraticPolynomial):
            trace_scale = max(
                trace_scale,
                np.linalg.norm(polynomial.hessian) * geometry.diameter**2,
            )
        if trace > geometry.tolerance * trace_scale or order >= 0.5:
            raise SingularPointError(
                "triangle-supported polynomial has a divergent pointwise "
                "action at this support interface"
            )
    raw = _piece_boundary_action_many(
        SimplexPiece(geometry, polynomial),
        targets,
        order,
    )
    return riesz_normalization_2d(order) * raw / (2.0 * order)


def piecewise_affine_action(
    pieces: Iterable[SimplexPiece],
    point: object,
    order: float,
) -> float:
    """Sum local polynomial pieces before interpreting interface limits."""
    order = _validate_order(order)
    piece_tuple = tuple(pieces)
    if not piece_tuple:
        return 0.0
    x = np.asarray(point, dtype=np.float64)
    on_interface = any(
        piece.geometry.classify(x) in {"face", "edge", "vertex"}
        for piece in piece_tuple
    )
    if on_interface and order >= 0.5:
        raise SingularPointError(
            "piecewise-polynomial actions at interfaces require s < 1/2"
        )
    dimension = int(piece_tuple[0].geometry.dimension)
    scale = riesz_normalization(dimension, order) / (2.0 * order)
    return float(
        _scaled_piecewise_affine_action_many(
            piece_tuple,
            x[None, :],
            order,
            scale,
        )[0]
    )


def _scaled_piecewise_affine_action_many(
    pieces: tuple[SimplexPiece, ...],
    points: np.ndarray,
    order: float,
    scale: float,
) -> np.ndarray:
    """Evaluate a piecewise polynomial on a batch of interior targets."""
    if pieces and int(pieces[0].geometry.dimension) == 3:
        from yonderdrake.riesz.tetrahedron_action import (
            _tetrahedron_piece_action_many_unchecked,
        )

        result = np.zeros(points.shape[0], dtype=np.float64)
        for piece in pieces:
            result += _tetrahedron_piece_action_many_unchecked(
                piece,
                points,
                order,
            )
        return result
    result = np.zeros(points.shape[0], dtype=np.float64)
    for piece in pieces:
        result += _piece_boundary_action_many(piece, points, order)
    return scale * result
