"""Boundary-reduced action for tetrahedron-supported quadratic polynomials."""

from __future__ import annotations

from functools import cache

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import hyp2f1

from yonderdrake.riesz.geometry import TetrahedronGeometry
from yonderdrake.riesz.triangle_action import (
    AffinePolynomial,
    QuadraticPolynomial,
    SimplexPiece,
    SingularPointError,
    _validate_order,
    riesz_normalization,
)


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


def _face_moments_many(
    face: np.ndarray,
    normal: np.ndarray,
    points: np.ndarray,
    low_exponent: float,
    high_exponent: float,
    hessian: np.ndarray | None,
    *,
    tangent_count: int = 16,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized face moments for interior or exterior target points."""
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
                raise SingularPointError(
                    "target lies on a tetrahedron support face"
                )
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
            + nodes[None, :, None]
            * (right_relative - left_relative)[:, None, :]
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
        )
        high_boundary += signed_distances * high
        gradient_boundary += (gradients @ normal) * low
        trace_boundary += signed_distances * low
        hessian_boundary += hessian_moment

    result = (
        values_at_points * high_boundary / (2.0 * order)
        + gradient_boundary / beta
    )
    if hessian is not None:
        result += hessian_boundary / (2.0 * beta)
        result -= (
            float(np.trace(hessian))
            * trace_boundary
            / (4.0 * beta * (1.0 - order))
        )
    return riesz_normalization(3, order) * result


def _tetrahedron_piece_action(
    piece: SimplexPiece,
    point: np.ndarray,
    order: float,
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
    if (
        targets.ndim != 2
        or targets.shape[1] != 3
        or not np.all(np.isfinite(targets))
    ):
        raise ValueError("points must be a finite array with shape (num_points, 3)")
    if polynomial.gradient.shape != (3,):
        raise ValueError("tetrahedron polynomial must be three-dimensional")
    for point in targets:
        classification = geometry.classify(point)
        if classification in {"face", "edge", "vertex"}:
            return np.fromiter(
                (
                    tetrahedron_action(geometry, polynomial, item, order)
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
