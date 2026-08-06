"""Tests for Riesz source-evaluation routing."""

from __future__ import annotations

import numpy as np
import pytest

from yonderdrake.riesz.dense import DenseRieszBackend, RieszMeshData
from yonderdrake.riesz.geometry import TriangleGeometry
from yonderdrake.riesz.matfree import MatrixFreeRieszBackend
from yonderdrake.riesz.outer_quadrature import triangle_quadrature
from yonderdrake.riesz.source_evaluation import SourceActionEvaluator
from yonderdrake.riesz.triangle_action import AffinePolynomial, SimplexPiece


def source_piece() -> SimplexPiece:
    geometry = TriangleGeometry.from_vertices([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    return SimplexPiece(
        geometry,
        AffinePolynomial(0.7, np.array([0.2, -0.1])),
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
