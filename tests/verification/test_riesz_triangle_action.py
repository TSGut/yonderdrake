"""Independent verification of the exact triangle boundary formula."""

from __future__ import annotations

from itertools import permutations

import numpy as np
import pytest
from scipy.integrate import quad
from tests.reference.riesz import (
    inside_triangle_polar_oracle,
    outside_triangle_oracle,
)

from yonderdrake.riesz.geometry import TriangleGeometry
from yonderdrake.riesz.triangle_action import (
    AffinePolynomial,
    QuadraticPolynomial,
    SimplexPiece,
    SingularPointError,
    _line_power_integral_many,
    _piece_boundary_action_many,
    combine_polynomials,
    line_power_integral,
    piecewise_affine_action,
    riesz_normalization_2d,
    triangle_affine_action,
    triangle_affine_action_many,
    triangle_quadratic_action,
)


@pytest.mark.verification
@pytest.mark.parametrize(
    ("left", "right", "distance", "exponent"),
    [
        (-0.7, 1.3, 0.4, 0.2),
        (-2.0, -0.2, 0.0, 0.7),
        (-0.4, 0.8, 0.0, 0.3),
        (0.2, 1.5, 1.0e-4, 1.4),
    ],
)
def test_exact_line_integral_against_adaptive_quadrature(
    left: float,
    right: float,
    distance: float,
    exponent: float,
) -> None:
    actual = line_power_integral(left, right, distance, exponent)
    expected = quad(
        lambda z: (distance**2 + z**2) ** (-exponent),
        left,
        right,
        epsabs=1.0e-12,
        epsrel=1.0e-12,
        points=[0.0] if left < 0.0 < right else None,
    )[0]
    assert actual == pytest.approx(expected, rel=2.0e-10, abs=2.0e-11)


@pytest.mark.unit
def test_line_integral_orientations_and_singular_limits() -> None:
    forward = line_power_integral(0.2, 1.4, 0.3, 0.5)
    reverse = line_power_integral(1.4, 0.2, 0.3, 0.5)
    assert reverse == pytest.approx(-forward)
    assert forward == pytest.approx(
        np.arcsinh(1.4 / 0.3) - np.arcsinh(0.2 / 0.3)
    )
    assert np.isinf(line_power_integral(-1.0, 1.0, 0.0, 0.7))
    assert line_power_integral(1.0, 2.0, 0.0, 0.5) == pytest.approx(
        np.log(2.0)
    )


@pytest.mark.verification
def test_batched_line_integrals_match_adaptive_quadrature() -> None:
    left = np.array([-0.7, -2.0, -0.4, 0.2, 1.4])
    right = np.array([1.3, -0.2, 0.8, 1.5, 0.2])
    distance = np.array([0.4, 0.0, 0.0, 1.0e-4, 0.3])
    exponent = 0.7
    actual = _line_power_integral_many(
        left,
        right,
        distance,
        exponent,
        tolerance=1.0e-14,
    )
    expected = []
    for a, b, h in zip(left, right, distance, strict=True):
        if h == 0.0 and a <= 0.0 <= b:
            expected.append(float("inf"))
            continue
        expected.append(
            quad(
                lambda z, distance_value=h: (
                    distance_value**2 + z**2
                )
                ** (-exponent),
                a,
                b,
                epsabs=1.0e-12,
                epsrel=1.0e-12,
            )[0]
        )
    np.testing.assert_allclose(actual, expected, rtol=2.0e-10, atol=2.0e-11)
    with pytest.raises(ValueError, match="matching shapes"):
        _line_power_integral_many(
            np.zeros(2),
            np.zeros(3),
            np.zeros(2),
            exponent,
            tolerance=1.0e-14,
        )


@pytest.mark.unit
def test_polynomial_data_and_combinations_are_validated() -> None:
    with pytest.raises(ValueError, match="gradient"):
        AffinePolynomial(0.0, np.zeros(4))
    with pytest.raises(ValueError, match="constant"):
        AffinePolynomial(float("nan"), np.zeros(2))
    with pytest.raises(ValueError, match="gradient"):
        QuadraticPolynomial(0.0, np.zeros(4), np.eye(4))
    with pytest.raises(ValueError, match="hessian"):
        QuadraticPolynomial(0.0, np.zeros(2), np.zeros((3, 3)))
    with pytest.raises(ValueError, match="symmetric"):
        QuadraticPolynomial(
            0.0,
            np.zeros(2),
            np.array([[0.0, 1.0], [0.0, 0.0]]),
        )
    with pytest.raises(ValueError, match="constant"):
        QuadraticPolynomial(float("nan"), np.zeros(2), np.eye(2))

    affine = AffinePolynomial(1.0, np.array([2.0, -1.0]))
    quadratic = QuadraticPolynomial(0.5, np.zeros(2), np.eye(2))
    with pytest.raises(ValueError, match="equal length"):
        combine_polynomials([affine], [1.0, 2.0])
    combined = combine_polynomials([affine, quadratic], [2.0, -1.0])
    point = np.array([0.2, 0.3])
    assert combined(point) == pytest.approx(
        2.0 * affine(point) - quadratic(point)
    )
    np.testing.assert_allclose(
        quadratic.gradient_at(point),
        point,
    )


@pytest.mark.verification
@pytest.mark.parametrize(
    ("point", "order"),
    [([1.4, 0.2], 0.2), ([-0.1, 0.15], 0.45), ([8.0, -4.0], 0.75)],
)
def test_outside_action_matches_high_precision_source_integral(
    point: list[float],
    order: float,
) -> None:
    geometry = TriangleGeometry.from_vertices([[0, 0], [1, 0], [0.2, 0.9]])
    polynomial = AffinePolynomial(0.7, np.array([0.3, -0.2]))
    actual = triangle_affine_action(geometry, polynomial, point, order)
    expected = float(outside_triangle_oracle(geometry, polynomial, point, order))
    assert actual == pytest.approx(expected, rel=3.0e-10, abs=3.0e-11)


@pytest.mark.verification
@pytest.mark.parametrize("order", [0.2, 0.55, 0.8])
def test_inside_action_matches_independent_polar_pv(order: float) -> None:
    geometry = TriangleGeometry.from_vertices([[0, 0], [1, 0], [0.2, 0.9]])
    polynomial = AffinePolynomial(0.7, np.array([0.3, -0.2]))
    point = np.array([0.3, 0.25])
    actual = triangle_affine_action(geometry, polynomial, point, order)
    expected = float(
        inside_triangle_polar_oracle(geometry, polynomial, point, order)
    )
    assert actual == pytest.approx(expected, rel=5.0e-9, abs=5.0e-10)


@pytest.mark.verification
@pytest.mark.parametrize("order", [0.2, 0.55, 0.8])
@pytest.mark.parametrize("point", [[0.3, 0.25], [1.4, 0.2]])
def test_quadratic_action_matches_independent_integral(
    order: float,
    point: list[float],
) -> None:
    geometry = TriangleGeometry.from_vertices([[0, 0], [1, 0], [0.2, 0.9]])
    polynomial = QuadraticPolynomial(
        0.7,
        np.array([0.3, -0.2]),
        np.array([[0.4, 0.1], [0.1, -0.3]]),
    )
    actual = triangle_quadratic_action(geometry, polynomial, point, order)
    oracle = (
        inside_triangle_polar_oracle
        if geometry.classify(point) == "inside"
        else outside_triangle_oracle
    )
    expected = float(oracle(geometry, polynomial, point, order))
    assert actual == pytest.approx(expected, rel=8.0e-9, abs=8.0e-10)


@pytest.mark.verification
def test_orientation_rigid_motion_and_scaling_covariance() -> None:
    vertices = np.array([[0.0, 0.0], [1.2, 0.1], [0.1, 0.9]])
    polynomial = AffinePolynomial(0.4, np.array([0.3, -0.5]))
    point = np.array([1.7, -0.2])
    order = 0.37
    reference = triangle_affine_action(
        TriangleGeometry.from_vertices(vertices),
        polynomial,
        point,
        order,
    )
    for permutation in permutations(range(3)):
        value = triangle_affine_action(
            TriangleGeometry.from_vertices(vertices[list(permutation)]),
            polynomial,
            point,
            order,
        )
        assert value == pytest.approx(reference, rel=2.0e-12)

    angle = 0.73
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    shift = np.array([2.0, -1.0])
    transformed_vertices = vertices @ rotation.T + shift
    transformed_gradient = rotation @ polynomial.gradient
    transformed_polynomial = AffinePolynomial(
        polynomial.constant - np.dot(transformed_gradient, shift),
        transformed_gradient,
    )
    transformed_point = rotation @ point + shift
    transformed = triangle_affine_action(
        TriangleGeometry.from_vertices(transformed_vertices),
        transformed_polynomial,
        transformed_point,
        order,
    )
    assert transformed == pytest.approx(reference, rel=2.0e-11)

    scale = 3.4
    scaled = triangle_affine_action(
        TriangleGeometry.from_vertices(scale * vertices),
        AffinePolynomial(polynomial.constant, polynomial.gradient / scale),
        scale * point,
        order,
    )
    assert scaled == pytest.approx(scale ** (-2.0 * order) * reference, rel=2.0e-11)


@pytest.mark.verification
def test_far_field_decay() -> None:
    geometry = TriangleGeometry.from_vertices([[0, 0], [1, 0], [0, 1]])
    polynomial = AffinePolynomial(1.0, np.zeros(2))
    order = 0.4
    near = abs(triangle_affine_action(geometry, polynomial, [100.0, 0.0], order))
    far = abs(triangle_affine_action(geometry, polynomial, [200.0, 0.0], order))
    assert near / far == pytest.approx(2.0 ** (2.0 + 2.0 * order), rel=1.5e-2)


@pytest.mark.verification
def test_scalar_and_batched_action_interfaces_agree() -> None:
    geometry = TriangleGeometry.from_vertices([[0, 0], [1, 0], [0, 1]])
    polynomial = AffinePolynomial(0.3, np.array([0.2, -0.1]))
    points = np.array([[0.2, 0.3], [1.1, 0.2], [4.0, -2.0]])
    batched = triangle_affine_action_many(geometry, polynomial, points, 0.4)
    scalar = np.array(
        [
            triangle_affine_action(geometry, polynomial, point, 0.4)
            for point in points
        ]
    )
    tolerance = 8.0 * np.finfo(float).eps
    np.testing.assert_allclose(batched, scalar, rtol=tolerance, atol=tolerance)
    with pytest.raises(ValueError, match="num_points"):
        triangle_affine_action_many(
            geometry,
            polynomial,
            np.zeros(2),
            0.4,
        )


@pytest.mark.verification
@pytest.mark.parametrize(
    "polynomial",
    [
        AffinePolynomial(0.3, np.array([0.2, -0.1])),
        QuadraticPolynomial(
            0.7,
            np.array([0.3, -0.2]),
            np.array([[0.4, 0.1], [0.1, -0.3]]),
        ),
    ],
)
def test_batched_boundary_kernel_matches_independent_integrals(
    polynomial: AffinePolynomial | QuadraticPolynomial,
) -> None:
    geometry = TriangleGeometry.from_vertices(
        [[0.0, 0.0], [1.2, 0.1], [0.1, 0.9]]
    )
    piece = SimplexPiece(geometry, polynomial)
    points = np.array([[0.2, 0.3], [1.1, 0.2], [4.0, -2.0]])
    order = 0.4
    actual = (
        riesz_normalization_2d(order)
        * _piece_boundary_action_many(piece, points, order)
        / (2.0 * order)
    )
    expected = np.asarray(
        [
            float(
                (
                    inside_triangle_polar_oracle
                    if geometry.classify(point) == "inside"
                    else outside_triangle_oracle
                )(geometry, polynomial, point, order)
            )
            for point in points
        ]
    )
    np.testing.assert_allclose(actual, expected, rtol=8.0e-9, atol=8.0e-10)


def hat_patch() -> tuple[SimplexPiece, ...]:
    center = np.array([0.0, 0.0])
    boundary = (
        np.array([1.0, 0.0]),
        np.array([0.0, 1.0]),
        np.array([-1.0, 0.0]),
        np.array([0.0, -1.0]),
    )
    pieces = []
    for left, right in zip(boundary, boundary[1:] + boundary[:1], strict=True):
        geometry = TriangleGeometry.from_vertices([center, left, right])
        matrix = np.vstack(
            [
                np.ones(3),
                geometry.vertices[:, 0],
                geometry.vertices[:, 1],
            ]
        ).T
        coefficients = np.linalg.solve(matrix, np.array([1.0, 0.0, 0.0]))
        pieces.append(
            SimplexPiece(
                geometry,
                AffinePolynomial(coefficients[0], coefficients[1:]),
            )
        )
    return tuple(pieces)


@pytest.mark.verification
def test_global_hat_sums_incident_triangles_at_edge_and_vertex() -> None:
    pieces = hat_patch()
    order = 0.3
    edge_point = np.array([0.25, 0.0])
    edge_value = piecewise_affine_action(pieces, edge_point, order)

    def symmetric_limit(offset: float) -> float:
        left = piecewise_affine_action(
            pieces,
            edge_point + np.array([0.0, offset]),
            order,
        )
        right = piecewise_affine_action(
            pieces,
            edge_point - np.array([0.0, offset]),
            order,
        )
        return 0.5 * (left + right)

    coarse = symmetric_limit(1.0e-4)
    fine = symmetric_limit(1.0e-7)
    assert abs(fine - edge_value) < abs(coarse - edge_value)

    vertex_value = piecewise_affine_action(pieces, [0.0, 0.0], order)
    coarse_vertex = piecewise_affine_action(pieces, [1.0e-4, 2.0e-4], order)
    nearby = piecewise_affine_action(pieces, [1.0e-7, 2.0e-7], order)
    assert abs(nearby - vertex_value) < abs(coarse_vertex - vertex_value)
    with pytest.raises(SingularPointError, match="s < 1/2"):
        piecewise_affine_action(pieces, edge_point, 0.7)


@pytest.mark.unit
def test_boundary_singularities_are_explicit() -> None:
    geometry = TriangleGeometry.from_vertices([[0, 0], [1, 0], [0, 1]])
    with pytest.raises(SingularPointError, match="divergent"):
        triangle_affine_action(
            geometry,
            AffinePolynomial(1.0, np.zeros(2)),
            [0.4, 0.0],
            0.2,
        )
    quadratic = QuadraticPolynomial(1.0, np.zeros(2), np.eye(2))
    with pytest.raises(SingularPointError, match="divergent"):
        triangle_quadratic_action(
            geometry,
            quadratic,
            [0.4, 0.0],
            0.2,
        )
    assert piecewise_affine_action([], [0.2, 0.2], 0.4) == 0.0
    with pytest.raises(ValueError, match="0 < order < 1"):
        triangle_affine_action(
            geometry,
            AffinePolynomial(1.0, np.zeros(2)),
            [0.2, 0.2],
            1.0,
        )


@pytest.mark.unit
def test_zero_trace_interface_actions_remain_finite_below_half_order() -> None:
    geometry = TriangleGeometry.from_vertices([[0, 0], [1, 0], [0, 1]])
    point = [0.4, 0.0]
    affine = AffinePolynomial(0.0, np.array([0.0, 1.0]))
    quadratic = QuadraticPolynomial(
        0.0,
        np.array([0.0, 1.0]),
        np.zeros((2, 2)),
    )
    assert np.isfinite(triangle_affine_action(geometry, affine, point, 0.2))
    assert np.isfinite(
        triangle_quadratic_action(geometry, quadratic, point, 0.2)
    )
