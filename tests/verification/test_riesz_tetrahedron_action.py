"""Independent volume checks for the tetrahedral Riesz action."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import nquad

from yonderdrake.riesz.geometry import TetrahedronGeometry
from yonderdrake.riesz.tetrahedron_action import (
    _automatic_appell_mask,
    _face_moments_many,
    _radial_moment,
    _tetrahedron_piece_action_many_unchecked,
    tetrahedron_action,
    tetrahedron_action_many,
)
from yonderdrake.riesz.triangle_action import (
    AffinePolynomial,
    QuadraticPolynomial,
    SimplexPiece,
    SingularPointError,
    riesz_normalization,
)


@pytest.mark.verification
@pytest.mark.parametrize(
    "polynomial",
    [
        AffinePolynomial(0.7, np.array([0.2, -0.3, 0.4])),
        QuadraticPolynomial(
            0.7,
            np.array([0.2, -0.3, 0.4]),
            np.array(
                [
                    [0.5, 0.1, -0.2],
                    [0.1, -0.4, 0.3],
                    [-0.2, 0.3, 0.2],
                ]
            ),
        ),
    ],
)
def test_tetrahedron_action_matches_independent_outside_volume_integral(
    polynomial: AffinePolynomial | QuadraticPolynomial,
) -> None:
    geometry = TetrahedronGeometry.from_vertices(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    point = np.array([1.2, 0.3, 0.4])
    order = 0.3

    def integrand(third: float, second: float, first: float) -> float:
        source = np.array([first, second, third])
        return polynomial(source) / np.linalg.norm(source - point) ** (
            3.0 + 2.0 * order
        )

    integral, error = nquad(
        integrand,
        [
            lambda second, first: [0.0, 1.0 - first - second],
            lambda first: [0.0, 1.0 - first],
            [0.0, 1.0],
        ],
        opts={"epsabs": 2.0e-11, "epsrel": 2.0e-11},
    )
    expected = -riesz_normalization(3, order) * integral
    actual = tetrahedron_action(geometry, polynomial, point, order)
    assert error < 2.0e-10
    assert actual == pytest.approx(expected, rel=3.0e-10, abs=3.0e-11)
    np.testing.assert_allclose(
        tetrahedron_action_many(geometry, polynomial, [point], order),
        [actual],
    )


@pytest.mark.verification
def test_tetrahedron_action_handles_zero_trace_on_a_support_face() -> None:
    geometry = TetrahedronGeometry.from_vertices(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    point = np.array([0.23, 0.31, 0.0])
    zero_trace = AffinePolynomial(0.0, np.array([0.0, 0.0, 1.0]))
    value = tetrahedron_action(geometry, zero_trace, point, 0.3)
    assert np.isfinite(value)
    np.testing.assert_allclose(
        tetrahedron_action_many(
            geometry,
            zero_trace,
            [[1.2, 0.3, 0.4], point],
            0.3,
        ),
        [
            tetrahedron_action(geometry, zero_trace, [1.2, 0.3, 0.4], 0.3),
            value,
        ],
    )

    nonzero_trace = AffinePolynomial(1.0, np.zeros(3))
    with pytest.raises(SingularPointError, match="divergent"):
        tetrahedron_action(geometry, nonzero_trace, point, 0.3)
    with pytest.raises(SingularPointError, match="divergent"):
        tetrahedron_action(geometry, zero_trace, point, 0.6)


@pytest.mark.verification
def test_vectorized_face_moments_handle_mixed_coplanar_targets() -> None:
    face = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    normal = np.array([0.0, 0.0, 1.0])
    points = np.array([[0.23, 0.31, 0.0], [0.3, 0.2, 0.4]])
    hessian = np.array([[0.4, 0.1, 0.2], [0.1, -0.3, 0.05], [0.2, 0.05, 0.6]])
    together = _face_moments_many(face, normal, points, 0.8, 1.8, hessian)
    separate = [
        _face_moments_many(face, normal, point[None, :], 0.8, 1.8, hessian)
        for point in points
    ]
    for component, expected in zip(together, zip(*separate, strict=True), strict=True):
        np.testing.assert_allclose(component, np.concatenate(expected))


@pytest.mark.verification
@pytest.mark.parametrize(
    ("point", "low_exponent", "high_exponent"),
    [
        (np.array([0.2, 0.3, 0.4]), 0.8, 1.8),
        (np.array([0.5, -0.01, 0.01]), 1.0, 2.0),
    ],
)
def test_appell_face_moments_match_independent_surface_integrals(
    point: np.ndarray,
    low_exponent: float,
    high_exponent: float,
) -> None:
    face = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    normal = np.array([0.0, 0.0, 1.0])
    hessian = np.array([[0.4, 0.1, 0.2], [0.1, -0.3, 0.05], [0.2, 0.05, 0.6]])
    actual = np.array(
        [
            values[0]
            for values in _face_moments_many(
                face,
                normal,
                point[None, :],
                low_exponent,
                high_exponent,
                hessian,
                edge_evaluation="appell",
            )
        ]
    )

    def surface_integral(exponent: float, *, hessian_moment: bool) -> float:
        def integrand(second: float, first: float) -> float:
            relative = np.array([first, second, 0.0]) - point
            value = np.dot(relative, relative) ** (-exponent)
            if hessian_moment:
                value *= np.dot(relative, hessian @ normal)
            return float(value)

        value, error = nquad(
            integrand,
            [lambda first: [0.0, 1.0 - first], [0.0, 1.0]],
            opts={"epsabs": 2.0e-10, "epsrel": 2.0e-10, "limit": 300},
        )
        assert error < 2.0e-7 * max(1.0, abs(value))
        return float(value)

    expected = np.array(
        [
            surface_integral(low_exponent, hessian_moment=False),
            surface_integral(high_exponent, hessian_moment=False),
            surface_integral(low_exponent, hessian_moment=True),
        ]
    )
    np.testing.assert_allclose(actual, expected, rtol=2.0e-9, atol=2.0e-11)


@pytest.mark.verification
@pytest.mark.parametrize("order", [0.2, 0.5, 0.8])
def test_appell_face_moments_match_high_order_edge_quadrature(order: float) -> None:
    face = np.array([[0.1, -0.2, 0.3], [1.2, 0.1, 0.4], [-0.1, 0.9, 0.6]])
    raw_normal = np.cross(face[1] - face[0], face[2] - face[0])
    normal = raw_normal / np.linalg.norm(raw_normal)
    points = np.array([[0.3, 0.2, 1.1], [1.4, -0.4, 0.8], [-0.5, 0.1, -0.2]])
    hessian = np.array([[0.4, 0.1, 0.2], [0.1, -0.3, 0.05], [0.2, 0.05, 0.6]])
    low_exponent = 0.5 + order
    high_exponent = 1.5 + order
    expected = _face_moments_many(
        face,
        normal,
        points,
        low_exponent,
        high_exponent,
        hessian,
        tangent_count=128,
        edge_evaluation="quadrature",
    )
    actual = _face_moments_many(
        face,
        normal,
        points,
        low_exponent,
        high_exponent,
        hessian,
        edge_evaluation="appell",
    )
    for values, reference in zip(actual, expected, strict=True):
        np.testing.assert_allclose(values, reference, rtol=4.0e-10, atol=2.0e-11)


@pytest.mark.verification
def test_automatic_edge_evaluation_routes_by_projected_edge_separation() -> None:
    face = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    normal = np.array([0.0, 0.0, 1.0])
    points = np.array([[0.5, -0.01, 0.01], [3.0, 2.0, 4.0]])
    hessian = np.array([[0.4, 0.1, 0.2], [0.1, -0.3, 0.05], [0.2, 0.05, 0.6]])
    np.testing.assert_array_equal(
        _automatic_appell_mask(face, normal, points),
        [True, False],
    )

    automatic = _face_moments_many(
        face,
        normal,
        points,
        1.0,
        2.0,
        hessian,
        edge_evaluation="automatic",
    )
    near_appell = _face_moments_many(
        face,
        normal,
        points[:1],
        1.0,
        2.0,
        hessian,
        edge_evaluation="appell",
    )
    far_quadrature = _face_moments_many(
        face,
        normal,
        points[1:],
        1.0,
        2.0,
        hessian,
        edge_evaluation="quadrature",
    )
    for values, near, far in zip(
        automatic,
        near_appell,
        far_quadrature,
        strict=True,
    ):
        assert values[0] == near[0]
        assert values[1] == far[0]

    scale = 37.0
    np.testing.assert_array_equal(
        _automatic_appell_mask(scale * face, normal, scale * points),
        [True, False],
    )


@pytest.mark.verification
def test_appell_is_default_for_direct_tetrahedron_actions() -> None:
    geometry = TetrahedronGeometry.from_vertices(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    polynomial = QuadraticPolynomial(
        0.7,
        np.array([0.2, -0.3, 0.4]),
        np.array([[0.5, 0.1, -0.2], [0.1, -0.4, 0.3], [-0.2, 0.3, 0.2]]),
    )
    points = np.array([[0.5, -0.01, 0.01], [1.2, 0.3, 0.4]])
    expected = _tetrahedron_piece_action_many_unchecked(
        SimplexPiece(geometry, polynomial),
        points,
        0.7,
        edge_evaluation="appell",
    )
    np.testing.assert_array_equal(
        tetrahedron_action_many(geometry, polynomial, points, 0.7), expected
    )
    assert tetrahedron_action(geometry, polynomial, points[0], 0.7) == expected[0]


@pytest.mark.verification
def test_radial_moment_zero_distance_limits_and_validation() -> None:
    squared_radius = np.array([0.25, 1.0, 4.0])
    finite = _radial_moment(squared_radius, 0.0, 0.4, 1)
    np.testing.assert_allclose(finite, squared_radius**-0.4 / 1.2)
    assert np.all(np.isinf(_radial_moment(squared_radius, 0.0, 1.2, 1)))
    with pytest.raises(ValueError, match="cannot mix"):
        _radial_moment(
            np.vstack((squared_radius, squared_radius)),
            np.array([0.0, 0.2]),
            0.4,
            1,
        )


@pytest.mark.verification
def test_tetrahedron_action_validates_targets_and_polynomial_dimension() -> None:
    geometry = TetrahedronGeometry.from_vertices(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    polynomial = AffinePolynomial(0.0, np.ones(3))
    with pytest.raises(ValueError, match="length 3"):
        tetrahedron_action(geometry, polynomial, [0.2, 0.3], 0.3)
    with pytest.raises(ValueError, match="shape"):
        tetrahedron_action_many(geometry, polynomial, [0.2, 0.3, 0.4], 0.3)
    with pytest.raises(ValueError, match="finite"):
        tetrahedron_action_many(geometry, polynomial, [[np.nan, 0.3, 0.4]], 0.3)

    two_dimensional = AffinePolynomial(0.0, np.ones(2))
    with pytest.raises(ValueError, match="three-dimensional"):
        tetrahedron_action(geometry, two_dimensional, [1.2, 0.3, 0.4], 0.3)
    with pytest.raises(ValueError, match="three-dimensional"):
        tetrahedron_action_many(geometry, two_dimensional, [[1.2, 0.3, 0.4]], 0.3)
