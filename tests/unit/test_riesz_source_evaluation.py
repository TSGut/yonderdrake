"""Tests for Riesz source-evaluation routing."""

from __future__ import annotations

import numpy as np
import pytest

from yonderdrake.riesz.dense import DenseRieszBackend, RieszMeshData
from yonderdrake.riesz.geometry import TetrahedronGeometry, TriangleGeometry
from yonderdrake.riesz.matfree import MatrixFreeRieszBackend
from yonderdrake.riesz.outer_quadrature import triangle_quadrature
from yonderdrake.riesz.source_evaluation import (
    PreparedSourcePiece,
    SourceActionEvaluator,
    SourceEvaluation,
)
from yonderdrake.riesz.triangle_action import AffinePolynomial, SimplexPiece


def source_piece() -> SimplexPiece:
    geometry = TriangleGeometry.from_vertices([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    return SimplexPiece(
        geometry,
        AffinePolynomial(0.7, np.array([0.2, -0.1])),
    )


def tetrahedron_source_piece() -> SimplexPiece:
    geometry = TetrahedronGeometry.from_vertices(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    return SimplexPiece(
        geometry,
        AffinePolynomial(0.7, np.array([0.2, -0.1, 0.3])),
    )


@pytest.mark.verification
@pytest.mark.parametrize("order", [0.1, 0.5, 0.9])
def test_source_modes_agree_on_well_separated_pairs(order: float) -> None:
    piece = source_piece()
    targets = np.array([[4.0, 3.0], [5.0, -2.0]])
    endpoint = SourceActionEvaluator(2, order, "endpoint", 8)
    hybrid = SourceActionEvaluator(2, order, "hybrid", 8)
    expected = endpoint.action(
        endpoint.prepare(piece),
        targets,
        admissible=True,
        coincident=False,
    )
    sampled = hybrid.action(
        hybrid.prepare(piece),
        targets,
        admissible=True,
        coincident=False,
    )
    np.testing.assert_allclose(sampled, expected, rtol=9.0e-10, atol=2.0e-13)
    assert hybrid.quadrature_evaluations > 0


@pytest.mark.unit
def test_hybrid_uses_endpoint_evaluation_for_near_pairs() -> None:
    piece = source_piece()
    targets = np.array([[0.2, 0.3], [1.1, 0.2]])
    endpoint = SourceActionEvaluator(2, 0.4, "endpoint", 1)
    hybrid = SourceActionEvaluator(2, 0.4, "hybrid", 8)
    expected = endpoint.action(
        endpoint.prepare(piece),
        targets,
        admissible=False,
        coincident=False,
    )
    actual = hybrid.action(
        hybrid.prepare(piece),
        targets,
        admissible=False,
        coincident=False,
    )
    np.testing.assert_array_equal(actual, expected)
    assert hybrid.quadrature_evaluations == 0


@pytest.mark.unit
def test_hybrid_uses_endpoint_evaluation_for_coincident_support() -> None:
    piece = source_piece()
    targets = np.array([[0.2, 0.3], [0.6, 0.2]])
    endpoint = SourceActionEvaluator(2, 0.4, "endpoint", 1)
    hybrid = SourceActionEvaluator(2, 0.4, "hybrid", 8)
    expected = endpoint.action(
        endpoint.prepare(piece),
        targets,
        admissible=True,
        coincident=True,
    )
    actual = hybrid.action(
        hybrid.prepare(piece),
        targets,
        admissible=True,
        coincident=True,
    )
    np.testing.assert_array_equal(actual, expected)
    assert hybrid.quadrature_evaluations == 0


@pytest.mark.unit
@pytest.mark.parametrize(
    ("mode", "expected_edge_evaluation"),
    [("endpoint", "appell"), ("hybrid", "automatic")],
)
def test_three_dimensional_boundary_route_follows_source_evaluation(
    mode: SourceEvaluation,
    expected_edge_evaluation: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fake_action(
        piece: SimplexPiece,
        points: np.ndarray,
        order: float,
        *,
        edge_evaluation: str = "appell",
    ) -> np.ndarray:
        del piece, order
        calls.append(edge_evaluation)
        return np.zeros(points.shape[0])

    monkeypatch.setattr(
        "yonderdrake.riesz.tetrahedron_action."
        "_tetrahedron_piece_action_many_unchecked",
        fake_action,
    )
    evaluator = SourceActionEvaluator(3, 0.4, mode, 8)
    piece = tetrahedron_source_piece()
    evaluator.action(
        evaluator.prepare(piece),
        np.array([[0.3, 0.2, 0.1]]),
        admissible=False,
        coincident=False,
    )
    assert calls == [expected_edge_evaluation]


@pytest.mark.unit
def test_three_dimensional_hybrid_keeps_source_quadrature_for_far_pairs() -> None:
    evaluator = SourceActionEvaluator(3, 0.4, "hybrid", 8)
    piece = tetrahedron_source_piece()
    result = evaluator.action(
        evaluator.prepare(piece),
        np.array([[4.0, 3.0, 5.0]]),
        admissible=True,
        coincident=False,
    )
    assert np.all(np.isfinite(result))
    assert evaluator.quadrature_evaluations == 1
    assert evaluator.endpoint_evaluations == 0


@pytest.mark.unit
def test_source_quadrature_is_stable_under_large_coordinate_offsets() -> None:
    evaluator = SourceActionEvaluator(3, 0.4, "hybrid", 8)
    source = evaluator.prepare(tetrahedron_source_piece())
    offset = np.array([1.0e6, -2.0e6, 0.5e6])
    translated = PreparedSourcePiece(
        source.piece,
        source.points + offset,
        source.weighted_values,
    )
    targets = np.array([[4.0, 3.0, 5.0], [6.0, -2.0, 3.0]]) + offset
    differences = targets[:, None, :] - translated.points[None, :, :]
    squared_distances = np.einsum("ijk,ijk->ij", differences, differences)
    expected = -evaluator.normalization * (
        np.power(squared_distances, -0.5 * (3.0 + 2.0 * evaluator.order))
        @ translated.weighted_values
    )

    actual = evaluator.quadrature_action_many((translated,), targets)

    np.testing.assert_allclose(actual, expected, rtol=2.0e-14, atol=0.0)


@pytest.mark.unit
def test_source_quadrature_degree_is_inert_under_endpoint() -> None:
    coordinates = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    cells = np.array([[0, 1, 3], [1, 2, 3]])
    mesh = RieszMeshData.build(coordinates, cells)
    target = triangle_quadrature(5)
    coefficients = np.array([0.2, 1.0, -0.3, 0.4])
    low = MatrixFreeRieszBackend(
        mesh,
        0.35,
        target,
        source_quadrature_degree=1,
    ).apply(coefficients)
    high = MatrixFreeRieszBackend(
        mesh,
        0.35,
        target,
        source_quadrature_degree=20,
    ).apply(coefficients)
    np.testing.assert_array_equal(low, high)


@pytest.mark.verification
def test_hybrid_preserves_dense_and_matrix_free_agreement() -> None:
    source_evaluation = "hybrid"
    coordinates = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    cells = np.array([[0, 1, 3], [1, 2, 3]])
    mesh = RieszMeshData.build(coordinates, cells)
    target = triangle_quadrature(5)
    coefficients = np.array([0.2, 1.0, -0.3, 0.4])
    dense = DenseRieszBackend(
        mesh,
        0.35,
        target,
        source_evaluation=source_evaluation,
        source_quadrature_degree=8,
    ).apply(coefficients)
    matrix_free = MatrixFreeRieszBackend(
        mesh,
        0.35,
        target,
        source_evaluation=source_evaluation,
        source_quadrature_degree=8,
    ).apply(coefficients)
    np.testing.assert_allclose(matrix_free, dense, rtol=3.0e-12, atol=3.0e-12)
