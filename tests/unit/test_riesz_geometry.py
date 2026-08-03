"""Geometry and classification corpus for affine simplices."""

from __future__ import annotations

import numpy as np
import pytest

from yonderdrake.riesz.geometry import TetrahedronGeometry, TriangleGeometry


@pytest.fixture
def triangle() -> TriangleGeometry:
    return TriangleGeometry.from_vertices([[0.0, 0.0], [2.0, 0.0], [0.0, 1.0]])


@pytest.mark.unit
def test_orientation_is_canonical_and_arrays_are_immutable() -> None:
    forward = TriangleGeometry.from_vertices([[0, 0], [2, 0], [0, 1]])
    reverse = TriangleGeometry.from_vertices([[0, 0], [0, 1], [2, 0]])
    assert forward.area == reverse.area == 1.0
    assert np.all(
        forward.signed_edge_distances([0.2, 0.2]) < 0.0
    )
    with pytest.raises(ValueError):
        forward.vertices[0, 0] = 4.0


@pytest.mark.unit
@pytest.mark.parametrize(
    ("point", "expected"),
    [
        ([0.2, 0.2], "inside"),
        ([0.5, 0.0], "edge"),
        ([0.0, 0.0], "vertex"),
        ([1.2, 0.5], "outside_near"),
        ([10.0, 10.0], "outside_far"),
    ],
)
def test_point_classification(
    triangle: TriangleGeometry,
    point: list[float],
    expected: str,
) -> None:
    assert triangle.classify(point) == expected


@pytest.mark.unit
def test_degenerate_and_malformed_triangles_fail() -> None:
    with pytest.raises(ValueError, match="shape"):
        TriangleGeometry.from_vertices([[0, 0], [1, 0]])
    with pytest.raises(ValueError, match="finite"):
        TriangleGeometry.from_vertices([[0, 0], [1, 0], [0, np.nan]])
    with pytest.raises(ValueError, match="nondegenerate"):
        TriangleGeometry.from_vertices([[0, 0], [0, 0], [0, 0]])
    with pytest.raises(ValueError, match="nondegenerate"):
        TriangleGeometry.from_vertices([[0, 0], [1, 0], [2, 0]])


@pytest.mark.unit
def test_distances_and_barycentric_coordinates(
    triangle: TriangleGeometry,
) -> None:
    point = np.array([0.4, 0.2])
    barycentric = triangle.barycentric_coordinates(point)
    np.testing.assert_allclose(
        barycentric @ triangle.vertices,
        point,
        atol=2.0e-16,
    )
    assert np.sum(barycentric) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="finite vector"):
        triangle.signed_edge_distances([np.nan, 0.0])


@pytest.fixture
def tetrahedron() -> TetrahedronGeometry:
    return TetrahedronGeometry.from_vertices(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 3.0]]
    )


@pytest.mark.unit
def test_tetrahedron_orientation_measure_and_barycentric_coordinates(
    tetrahedron: TetrahedronGeometry,
) -> None:
    reverse = TetrahedronGeometry.from_vertices(
        [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 3.0]]
    )
    assert tetrahedron.volume == reverse.volume == pytest.approx(1.0)
    assert tetrahedron.reference_jacobian == pytest.approx(6.0)
    point = np.array([0.2, 0.2, 0.3])
    barycentric = tetrahedron.barycentric_coordinates(point)
    np.testing.assert_allclose(barycentric @ tetrahedron.vertices, point)
    np.testing.assert_allclose(
        tetrahedron.signed_face_distances(point),
        reverse.signed_face_distances(point),
    )
    assert np.all(tetrahedron.signed_face_distances(point) < 0.0)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("point", "expected"),
    [
        ([0.2, 0.2, 0.2], "inside"),
        ([0.2, 0.2, 0.0], "face"),
        ([0.2, 0.0, 0.0], "edge"),
        ([0.0, 0.0, 0.0], "vertex"),
        ([1.2, 0.5, 0.5], "outside_near"),
        ([10.0, 10.0, 10.0], "outside_far"),
    ],
)
def test_tetrahedron_point_classification(
    tetrahedron: TetrahedronGeometry,
    point: list[float],
    expected: str,
) -> None:
    assert tetrahedron.classify(point) == expected


@pytest.mark.unit
def test_degenerate_and_malformed_tetrahedra_fail() -> None:
    with pytest.raises(ValueError, match="shape"):
        TetrahedronGeometry.from_vertices(np.zeros((3, 3)))
    with pytest.raises(ValueError, match="finite"):
        points = np.eye(4, 3)
        points[0, 0] = np.nan
        TetrahedronGeometry.from_vertices(points)
    with pytest.raises(ValueError, match="nondegenerate"):
        TetrahedronGeometry.from_vertices(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]]
        )
