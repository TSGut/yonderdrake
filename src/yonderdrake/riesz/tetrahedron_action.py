"""Boundary-reduced action for tetrahedron-supported quadratic polynomials."""

from __future__ import annotations

from functools import cache
from typing import Literal, cast

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.integrate import quad
from scipy.special import hyp2f1

from yonderdrake.riesz._appell_f1 import appell_f1_a_plus_one
from yonderdrake.riesz.geometry import TetrahedronGeometry
from yonderdrake.riesz.triangle_action import (
    AffinePolynomial,
    QuadraticPolynomial,
    SimplexPiece,
    SingularPointError,
    _validate_order,
    riesz_normalization,
)

_EdgeEvaluation = Literal["automatic", "quadrature", "appell"]
# Distance from [0, 1] to the nearest complex edge singularity, measured in
# unit-edge coordinates, below which the 16-node rule is not used.
_AUTOMATIC_EDGE_SEPARATION = 0.4


def _validate_edge_evaluation(edge_evaluation: str) -> _EdgeEvaluation:
    if edge_evaluation not in {"automatic", "quadrature", "appell"}:
        raise ValueError(
            "edge_evaluation must be 'automatic', 'quadrature', or 'appell'"
        )
    return cast(_EdgeEvaluation, edge_evaluation)


@cache
def _unit_legendre(count: int) -> tuple[np.ndarray, np.ndarray]:
    nodes, weights = leggauss(count)
    nodes = 0.5 * (nodes + 1.0)
    weights = 0.5 * weights
    nodes.setflags(write=False)
    weights.setflags(write=False)
    return nodes, weights


def _radial_moment(
    squared_radius: np.ndarray,
    distance: float | np.ndarray,
    exponent: float,
    moment: int,
) -> np.ndarray:
    """Integrate ``rho**moment / (h**2 + a*rho**2)**exponent``."""
    distance_array = np.abs(np.asarray(distance, dtype=np.float64))
    if distance_array.ndim and squared_radius.ndim > distance_array.ndim:
        distance_array = distance_array[..., None]
    if bool(np.any(distance_array == 0.0)):
        if not bool(np.all(distance_array == 0.0)):
            raise ValueError("radial moments cannot mix zero and nonzero distances")
        power = moment + 1.0 - 2.0 * exponent
        if power <= 0.0:
            return np.full_like(squared_radius, np.inf)
        return squared_radius ** (-exponent) / power
    parameter = -(squared_radius / distance_array**2)
    return (
        distance_array ** (-2.0 * exponent)
        * hyp2f1(
            exponent,
            0.5 * (moment + 1.0),
            0.5 * (moment + 3.0),
            parameter,
        )
        / (moment + 1.0)
    )


def _radial_first_moment_value(
    squared_radius: float,
    squared_distance: float,
    exponent: float,
) -> float:
    """Stable scalar form of the first radial moment."""
    if squared_radius == 0.0:
        return float(0.5 * squared_distance ** (-exponent))
    delta = 1.0 - exponent
    logarithm = np.log1p(squared_radius / squared_distance)
    if abs(delta) <= 8.0 * np.finfo(np.float64).eps:
        return float(
            np.log1p(squared_radius / squared_distance) / (2.0 * squared_radius)
        )
    return float(
        squared_distance**delta
        * np.expm1(delta * logarithm)
        / (2.0 * squared_radius * delta)
    )


def _adaptive_edge_radial_integral(
    lower: float,
    upper: float,
    squared_distance_to_line: float,
    squared_face_distance: float,
    edge_length: float,
    exponent: float,
) -> float:
    """Integrate a radial first moment when separated primitives are unstable."""

    def integrand(tau: float) -> float:
        squared_radius = squared_distance_to_line + (edge_length * tau) ** 2
        return _radial_first_moment_value(
            squared_radius,
            squared_face_distance,
            exponent,
        )

    value, _ = quad(
        integrand,
        lower,
        upper,
        epsabs=0.0,
        epsrel=2.0e-13,
        limit=200,
    )
    return float(value)


def _edge_radial_integral_many(
    lower: np.ndarray,
    upper: np.ndarray,
    squared_distance_to_line: np.ndarray,
    squared_face_distance: np.ndarray,
    edge_length: float,
    exponent: float,
) -> np.ndarray:
    r"""Evaluate ``integral M_exponent,1(|u(t)|^2) dt`` along an edge."""
    result = np.empty(lower.shape, dtype=np.float64)
    delta = 1.0 - exponent
    line_scale = edge_length**2
    regular = (squared_distance_to_line > 1.0e-14 * line_scale) & (abs(delta) > 1.0e-5)
    if bool(np.any(regular)):
        indices = np.flatnonzero(regular)
        d2 = squared_distance_to_line[regular]
        h2 = squared_face_distance[regular]
        h2_plus_d2 = h2 + d2
        endpoint_d2 = np.tile(d2, 2)
        endpoint_h2 = np.tile(h2, 2)
        endpoint_h2_plus_d2 = np.tile(h2_plus_d2, 2)

        def primitive_parts(tau: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            tau_squared_edge = (edge_length * tau) ** 2
            first = (
                tau
                * endpoint_h2_plus_d2**delta
                * np.asarray(
                    appell_f1_a_plus_one(
                        0.5,
                        1.0,
                        exponent - 1.0,
                        -tau_squared_edge / endpoint_d2,
                        -tau_squared_edge / endpoint_h2_plus_d2,
                    )
                )
                / endpoint_d2
            )
            second = (
                tau
                * endpoint_h2**delta
                * hyp2f1(0.5, 1.0, 1.5, -tau_squared_edge / endpoint_d2)
                / endpoint_d2
            )
            return first, second

        endpoint_count = indices.size
        first, second = primitive_parts(
            np.concatenate((upper[regular], lower[regular]))
        )
        first_upper = first[:endpoint_count]
        first_lower = first[endpoint_count:]
        second_upper = second[:endpoint_count]
        second_lower = second[endpoint_count:]
        first_difference = first_upper - first_lower
        second_difference = second_upper - second_lower
        numerator = first_difference - second_difference
        scale = np.abs(first_difference) + np.abs(second_difference)
        stable = np.isfinite(numerator) & (
            np.abs(numerator) > 1.0e-3 * np.maximum(scale, np.finfo(np.float64).tiny)
        )
        result[indices[stable]] = numerator[stable] / (2.0 * delta)
        regular[indices[~stable]] = False

    for index in np.flatnonzero(~regular):
        result[index] = _adaptive_edge_radial_integral(
            float(lower[index]),
            float(upper[index]),
            float(squared_distance_to_line[index]),
            float(squared_face_distance[index]),
            edge_length,
            exponent,
        )
    return result


def _adaptive_shifted_edge_power_integral(
    lower: float,
    upper: float,
    squared_base: float,
    squared_scale: float,
    edge_length: float,
    delta: float,
) -> float:
    """Integrate a constant-shifted radial power, continuously through zero."""
    scale_power = squared_scale**delta

    def integrand(tau: float) -> float:
        squared_radius = squared_base + (edge_length * tau) ** 2
        logarithm = np.log(squared_radius / squared_scale)
        if delta == 0.0:
            return float(0.5 * logarithm)
        return float(scale_power * np.expm1(delta * logarithm) / (2.0 * delta))

    value, _ = quad(
        integrand,
        lower,
        upper,
        epsabs=1.0e-14,
        epsrel=2.0e-13,
        limit=200,
    )
    return float(value)


def _shifted_edge_power_integral_many(
    lower: np.ndarray,
    upper: np.ndarray,
    squared_base: np.ndarray,
    squared_scale: np.ndarray,
    edge_length: float,
    exponent: float,
) -> np.ndarray:
    """Integrate the divergence-field factor for a linear numerator."""
    delta = 1.0 - exponent
    if abs(delta) <= 1.0e-5:
        return np.asarray(
            [
                _adaptive_shifted_edge_power_integral(
                    float(left),
                    float(right),
                    float(base),
                    float(scale),
                    edge_length,
                    delta,
                )
                for left, right, base, scale in zip(
                    lower,
                    upper,
                    squared_base,
                    squared_scale,
                    strict=True,
                )
            ],
            dtype=np.float64,
        )

    def primitive(tau: np.ndarray) -> np.ndarray:
        parameter = -((edge_length * tau) ** 2) / squared_base
        powered = tau * squared_base**delta * hyp2f1(0.5, -delta, 1.5, parameter)
        shifted = tau * squared_scale**delta
        return (powered - shifted) / (2.0 * delta)

    return primitive(upper) - primitive(lower)


def _automatic_appell_mask(
    face: np.ndarray,
    normal: np.ndarray,
    points: np.ndarray,
) -> np.ndarray:
    """Select targets whose projected edge integrals need closed primitives."""
    signed_distances = (face[0] - points) @ normal
    projections = points + signed_distances[:, None] * normal[None, :]
    minimum_separation_squared = np.full(points.shape[0], np.inf)
    for left, right in zip(face, np.roll(face, -1, axis=0), strict=True):
        edge = right - left
        squared_length = float(np.dot(edge, edge))
        left_relative = left[None, :] - projections
        edge_parameter = -(left_relative @ edge) / squared_length
        closest = left_relative + edge_parameter[:, None] * edge[None, :]
        squared_distance_to_line = np.maximum(
            np.einsum("ij,ij->i", closest, closest),
            0.0,
        )
        interval_distance = np.maximum.reduce(
            (-edge_parameter, edge_parameter - 1.0, np.zeros_like(edge_parameter))
        )
        separation_squared = (
            signed_distances**2 + squared_distance_to_line
        ) / squared_length + interval_distance**2
        minimum_separation_squared = np.minimum(
            minimum_separation_squared,
            separation_squared,
        )
    return (signed_distances != 0.0) & (
        minimum_separation_squared < _AUTOMATIC_EDGE_SEPARATION**2
    )


def _face_moments_appell_many(
    face: np.ndarray,
    normal: np.ndarray,
    points: np.ndarray,
    low_exponent: float,
    high_exponent: float,
    hessian: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Face moments reduced analytically to Appell and Gauss functions."""
    signed_distances = (face[0] - points) @ normal
    coplanar = signed_distances == 0.0
    if bool(np.any(coplanar)):
        low_total = np.zeros(points.shape[0], dtype=np.float64)
        high_total = np.zeros(points.shape[0], dtype=np.float64)
        hessian_total = np.zeros(points.shape[0], dtype=np.float64)
        regular = ~coplanar
        if bool(np.any(regular)):
            low, high, hessian_values = _face_moments_appell_many(
                face,
                normal,
                points[regular],
                low_exponent,
                high_exponent,
                hessian,
            )
            low_total[regular] = low
            high_total[regular] = high
            hessian_total[regular] = hessian_values
        low, high, hessian_values = _face_moments_many(
            face,
            normal,
            points[coplanar],
            low_exponent,
            high_exponent,
            hessian,
            edge_evaluation="quadrature",
        )
        low_total[coplanar] = low
        high_total[coplanar] = high
        hessian_total[coplanar] = hessian_values
        return low_total, high_total, hessian_total

    projections = points + signed_distances[:, None] * normal[None, :]
    squared_face_distances = signed_distances**2
    low_total = np.zeros(points.shape[0], dtype=np.float64)
    high_total = np.zeros(points.shape[0], dtype=np.float64)
    tangential_hessian_total = np.zeros(points.shape[0], dtype=np.float64)
    face_scale = max(
        float(np.dot(right - left, right - left))
        for left, right in zip(face, np.roll(face, -1, axis=0), strict=True)
    )
    squared_scales = squared_face_distances + face_scale
    hessian_normal = hessian @ normal if hessian is not None else None

    for left, right in zip(face, np.roll(face, -1, axis=0), strict=True):
        edge = right - left
        edge_length = float(np.linalg.norm(edge))
        tangent = edge / edge_length
        conormal = np.cross(tangent, normal)
        left_relative = left[None, :] - projections
        edge_parameter = -(left_relative @ edge) / edge_length**2
        lower = -edge_parameter
        upper = 1.0 - edge_parameter
        closest = left_relative + edge_parameter[:, None] * edge[None, :]
        squared_distance_to_line = np.maximum(
            np.einsum("ij,ij->i", closest, closest),
            0.0,
        )
        signed_jacobians = np.einsum(
            "ij,j->i",
            np.cross(left_relative, edge[None, :]),
            normal,
        )
        low_total += signed_jacobians * _edge_radial_integral_many(
            lower,
            upper,
            squared_distance_to_line,
            squared_face_distances,
            edge_length,
            low_exponent,
        )
        high_total += signed_jacobians * _edge_radial_integral_many(
            lower,
            upper,
            squared_distance_to_line,
            squared_face_distances,
            edge_length,
            high_exponent,
        )
        if hessian_normal is not None:
            tangential_hessian_total += (
                edge_length
                * float(np.dot(hessian_normal, conormal))
                * _shifted_edge_power_integral_many(
                    lower,
                    upper,
                    squared_face_distances + squared_distance_to_line,
                    squared_scales,
                    edge_length,
                    low_exponent,
                )
            )

    if hessian_normal is None:
        hessian_total = np.zeros(points.shape[0], dtype=np.float64)
    else:
        normal_component = float(np.dot(normal, hessian_normal))
        hessian_total = signed_distances * normal_component * low_total + (
            tangential_hessian_total
        )
    return low_total, high_total, hessian_total


def _face_moments_many(
    face: np.ndarray,
    normal: np.ndarray,
    points: np.ndarray,
    low_exponent: float,
    high_exponent: float,
    hessian: np.ndarray | None,
    *,
    tangent_count: int = 16,
    edge_evaluation: _EdgeEvaluation = "appell",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized face moments for interior or exterior target points."""
    edge_evaluation = _validate_edge_evaluation(edge_evaluation)
    if edge_evaluation == "automatic":
        appell = _automatic_appell_mask(face, normal, points)
        if bool(np.all(appell)):
            return _face_moments_appell_many(
                face,
                normal,
                points,
                low_exponent,
                high_exponent,
                hessian,
            )
        if not bool(np.any(appell)):
            return _face_moments_many(
                face,
                normal,
                points,
                low_exponent,
                high_exponent,
                hessian,
                tangent_count=tangent_count,
                edge_evaluation="quadrature",
            )
        low_total = np.empty(points.shape[0], dtype=np.float64)
        high_total = np.empty(points.shape[0], dtype=np.float64)
        hessian_total = np.empty(points.shape[0], dtype=np.float64)
        selections: tuple[tuple[np.ndarray, _EdgeEvaluation], ...] = (
            (appell, "appell"),
            (~appell, "quadrature"),
        )
        for selected, method in selections:
            low, high, hessian_values = _face_moments_many(
                face,
                normal,
                points[selected],
                low_exponent,
                high_exponent,
                hessian,
                tangent_count=tangent_count,
                edge_evaluation=method,
            )
            low_total[selected] = low
            high_total[selected] = high
            hessian_total[selected] = hessian_values
        return low_total, high_total, hessian_total
    if edge_evaluation == "appell":
        return _face_moments_appell_many(
            face,
            normal,
            points,
            low_exponent,
            high_exponent,
            hessian,
        )
    signed_distances = (face[0] - points) @ normal
    coplanar = signed_distances == 0.0
    if bool(np.any(coplanar)):
        low_total = np.zeros(points.shape[0], dtype=np.float64)
        high_total = np.zeros(points.shape[0], dtype=np.float64)
        hessian_total = np.zeros(points.shape[0], dtype=np.float64)
        regular = ~coplanar
        if bool(np.any(regular)):
            low, high, hessian_values = _face_moments_many(
                face,
                normal,
                points[regular],
                low_exponent,
                high_exponent,
                hessian,
                tangent_count=tangent_count,
                edge_evaluation="quadrature",
            )
            low_total[regular] = low
            high_total[regular] = high
            hessian_total[regular] = hessian_values
        from yonderdrake.riesz.outer_quadrature import triangle_quadrature

        rule = triangle_quadrature(24)
        source_points = rule.barycentric @ face
        surface_jacobian = np.linalg.norm(
            np.cross(face[1] - face[0], face[2] - face[0])
        )
        surface_weights = surface_jacobian * rule.weights
        for index in np.flatnonzero(coplanar):
            relative = source_points - points[index]
            squared_distance = np.einsum("ij,ij->i", relative, relative)
            if float(np.min(squared_distance)) == 0.0:
                raise SingularPointError("target lies on a tetrahedron support face")
            low_kernel = squared_distance ** (-low_exponent)
            low_total[index] = float(np.dot(surface_weights, low_kernel))
            if hessian is not None:
                hessian_total[index] = float(
                    np.dot(
                        surface_weights * low_kernel,
                        relative @ (hessian @ normal),
                    )
                )
        return low_total, high_total, hessian_total
    projections = points + signed_distances[:, None] * normal[None, :]
    nodes, weights = _unit_legendre(tangent_count)
    low_total = np.zeros(points.shape[0], dtype=np.float64)
    high_total = np.zeros(points.shape[0], dtype=np.float64)
    hessian_total = np.zeros(points.shape[0], dtype=np.float64)
    for left, right in zip(face, np.roll(face, -1, axis=0), strict=True):
        left_relative = left[None, :] - projections
        right_relative = right[None, :] - projections
        vectors = (
            left_relative[:, None, :]
            + nodes[None, :, None] * (right_relative - left_relative)[:, None, :]
        )
        signed_jacobians = np.einsum(
            "ij,j->i",
            np.cross(left_relative, right_relative),
            normal,
        )
        squared_radius = np.einsum("ijk,ijk->ij", vectors, vectors)
        low_radial = _radial_moment(
            squared_radius,
            signed_distances,
            low_exponent,
            1,
        )
        high_radial = _radial_moment(
            squared_radius,
            signed_distances,
            high_exponent,
            1,
        )
        low_total += signed_jacobians * (low_radial @ weights)
        high_total += signed_jacobians * (high_radial @ weights)
        if hessian is not None:
            quadratic_constants = signed_distances * float(
                np.dot(normal, hessian @ normal)
            )
            quadratic_linear = np.einsum(
                "ijk,k->ij",
                vectors,
                hessian @ normal,
            )
            second_radial = _radial_moment(
                squared_radius,
                signed_distances,
                low_exponent,
                2,
            )
            hessian_total += signed_jacobians * (
                (
                    quadratic_constants[:, None] * low_radial
                    + quadratic_linear * second_radial
                )
                @ weights
            )
    return low_total, high_total, hessian_total


def _tetrahedron_piece_action_many_unchecked(
    piece: SimplexPiece,
    points: np.ndarray,
    order: float,
    *,
    edge_evaluation: _EdgeEvaluation = "appell",
) -> np.ndarray:
    geometry = piece.geometry
    if not isinstance(geometry, TetrahedronGeometry):
        raise TypeError("tetrahedron action requires TetrahedronGeometry")
    polynomial = piece.polynomial
    values_at_points = polynomial.constant + points @ polynomial.gradient
    if isinstance(polynomial, QuadraticPolynomial):
        values_at_points = values_at_points + 0.5 * np.einsum(
            "ij,jk,ik->i",
            points,
            polynomial.hessian,
            points,
        )
        gradients = polynomial.gradient[None, :] + points @ polynomial.hessian
        hessian = polynomial.hessian
    else:
        gradients = np.broadcast_to(polynomial.gradient, points.shape)
        hessian = None
    beta = 1.0 + 2.0 * order
    low_exponent = 0.5 * beta
    high_exponent = 0.5 * (3.0 + 2.0 * order)
    high_boundary = np.zeros(points.shape[0], dtype=np.float64)
    gradient_boundary = np.zeros(points.shape[0], dtype=np.float64)
    hessian_boundary = np.zeros(points.shape[0], dtype=np.float64)
    trace_boundary = np.zeros(points.shape[0], dtype=np.float64)
    for face, normal in zip(
        geometry.faces,
        geometry.face_normals,
        strict=True,
    ):
        signed_distances = (face[0] - points) @ normal
        low, high, hessian_moment = _face_moments_many(
            face,
            normal,
            points,
            low_exponent,
            high_exponent,
            hessian,
            edge_evaluation=edge_evaluation,
        )
        high_boundary += signed_distances * high
        gradient_boundary += (gradients @ normal) * low
        trace_boundary += signed_distances * low
        hessian_boundary += hessian_moment

    result = values_at_points * high_boundary / (2.0 * order) + gradient_boundary / beta
    if hessian is not None:
        result += hessian_boundary / (2.0 * beta)
        result -= (
            float(np.trace(hessian)) * trace_boundary / (4.0 * beta * (1.0 - order))
        )
    return riesz_normalization(3, order) * result


def _tetrahedron_piece_action(
    piece: SimplexPiece,
    point: np.ndarray,
    order: float,
    *,
    edge_evaluation: _EdgeEvaluation = "appell",
) -> float:
    geometry = piece.geometry
    if not isinstance(geometry, TetrahedronGeometry):
        raise TypeError("tetrahedron action requires TetrahedronGeometry")
    polynomial = piece.polynomial
    classification = geometry.classify(point)
    if classification in {"face", "edge", "vertex"}:
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
                "tetrahedron-supported polynomial has a divergent pointwise "
                "action at this support interface"
            )

    return float(
        _tetrahedron_piece_action_many_unchecked(
            piece,
            point[None, :],
            order,
            edge_evaluation=edge_evaluation,
        )[0]
    )


def tetrahedron_action(
    geometry: TetrahedronGeometry,
    polynomial: AffinePolynomial | QuadraticPolynomial,
    point: object,
    order: float,
) -> float:
    """Fractional Laplacian of one zero-extended tetrahedron polynomial."""
    order = _validate_order(order)
    x = np.asarray(point, dtype=np.float64)
    if x.shape != (3,) or not np.all(np.isfinite(x)):
        raise ValueError("point must be a finite vector of length 3")
    if polynomial.gradient.shape != (3,):
        raise ValueError("tetrahedron polynomial must be three-dimensional")
    return _tetrahedron_piece_action(
        SimplexPiece(geometry, polynomial),
        x,
        order,
    )


def tetrahedron_action_many(
    geometry: TetrahedronGeometry,
    polynomial: AffinePolynomial | QuadraticPolynomial,
    points: object,
    order: float,
) -> np.ndarray:
    """Evaluate a tetrahedron-supported polynomial over target points."""
    targets = np.asarray(points, dtype=np.float64)
    if targets.ndim != 2 or targets.shape[1] != 3 or not np.all(np.isfinite(targets)):
        raise ValueError("points must be a finite array with shape (num_points, 3)")
    if polynomial.gradient.shape != (3,):
        raise ValueError("tetrahedron polynomial must be three-dimensional")
    for point in targets:
        classification = geometry.classify(point)
        if classification in {"face", "edge", "vertex"}:
            return np.fromiter(
                (
                    tetrahedron_action(
                        geometry,
                        polynomial,
                        item,
                        order,
                    )
                    for item in targets
                ),
                dtype=np.float64,
                count=targets.shape[0],
            )
    return _tetrahedron_piece_action_many_unchecked(
        SimplexPiece(geometry, polynomial),
        targets,
        _validate_order(order),
    )
