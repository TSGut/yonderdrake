"""Independent volume checks for the tetrahedral Riesz action."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import nquad

from yonderdrake.riesz.geometry import TetrahedronGeometry
from yonderdrake.riesz.tetrahedron_action import (
    _face_moments_many,
    _radial_moment,
    tetrahedron_action,
    tetrahedron_action_many,
)
from yonderdrake.riesz.triangle_action import (
    AffinePolynomial,
    QuadraticPolynomial,
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
    hessian = np.array(
        [[0.4, 0.1, 0.2], [0.1, -0.3, 0.05], [0.2, 0.05, 0.6]]
    )
    together = _face_moments_many(face, normal, points, 0.8, 1.8, hessian)
    separate = [
        _face_moments_many(face, normal, point[None, :], 0.8, 1.8, hessian)
        for point in points
    ]
    for component, expected in zip(together, zip(*separate, strict=True), strict=True):
        np.testing.assert_allclose(component, np.concatenate(expected))


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
