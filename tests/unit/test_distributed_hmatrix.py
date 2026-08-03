"""Distributed H-matrix data preparation without Firedrake."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("mpi4py")

from yonderdrake.riesz.distributed_hmatrix import (
    DistributedHierarchicalRieszBackend,
    PairEntryEvaluator,
    distribute_dofs,
    validate_geometry,
)
from yonderdrake.riesz.geometry import TetrahedronGeometry, TriangleGeometry
from yonderdrake.riesz.outer_quadrature import (
    tetrahedron_quadrature,
    triangle_quadrature,
)
from yonderdrake.riesz.triangle_action import AffinePolynomial, SimplexPiece


class SerialComm:
    rank = 0
    size = 1

    @staticmethod
    def bcast(value, root):
        return value

    @staticmethod
    def alltoall(values):
        return values

    @staticmethod
    def allreduce(value):
        return value

    @staticmethod
    def allgather(value):
        return [value]


def triangle_piece() -> SimplexPiece:
    geometry = TriangleGeometry.from_vertices(
        np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    )
    return SimplexPiece(
        geometry,
        AffinePolynomial(1.0, np.zeros(2)),
    )


def tetrahedron_piece() -> SimplexPiece:
    geometry = TetrahedronGeometry.from_vertices(
        np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        )
    )
    return SimplexPiece(
        geometry,
        AffinePolynomial(1.0, np.zeros(3)),
    )


@pytest.mark.unit
def test_distribute_dofs_builds_owned_support() -> None:
    piece = triangle_piece()
    records = distribute_dofs(
        SerialComm(),
        ((0, 1),),
        [(0, np.array([0.0, 0.0]), piece)],
        triangle_quadrature(2),
    )
    assert len(records) == 1
    assert records[0].global_index == 0
    assert records[0].support == (piece,)
    assert np.sum(records[0].row_weights) == pytest.approx(piece.geometry.area)


@pytest.mark.unit
def test_distribute_dofs_builds_three_dimensional_support() -> None:
    piece = tetrahedron_piece()
    records = distribute_dofs(
        SerialComm(),
        ((0, 1),),
        [(0, np.zeros(3), piece)],
        tetrahedron_quadrature(2),
    )
    assert records[0].coordinate.shape == (3,)
    assert records[0].support_lower.shape == (3,)
    assert np.sum(records[0].row_weights) == pytest.approx(
        piece.geometry.volume
    )
    validate_geometry(SerialComm(), [piece.geometry.vertices])
    with pytest.raises(ValueError, match="overlapping cell geometry"):
        validate_geometry(
            SerialComm(),
            [piece.geometry.vertices, piece.geometry.vertices],
        )


@pytest.mark.unit
def test_distribute_dofs_validates_ownership_and_support() -> None:
    piece = triangle_piece()
    with pytest.raises(ValueError, match="has no owner"):
        distribute_dofs(
            SerialComm(),
            ((0, 1),),
            [(1, np.array([0.0, 0.0]), piece)],
            triangle_quadrature(1),
        )
    with pytest.raises(RuntimeError, match="does not match PETSc ownership"):
        distribute_dofs(
            SerialComm(),
            ((0, 1),),
            [],
            triangle_quadrature(1),
        )


@pytest.mark.unit
def test_pair_evaluator_caches_entries_and_columns() -> None:
    piece = triangle_piece()
    records = distribute_dofs(
        SerialComm(),
        ((0, 1),),
        [(0, np.array([0.0, 0.0]), piece)],
        triangle_quadrature(2),
    )
    evaluator = PairEntryEvaluator(records, records, 0.3)
    value = evaluator.entry(0, 0)
    assert evaluator.entry(0, 0) == value
    np.testing.assert_allclose(evaluator.column(np.array([0]), 0), [value])
    assert evaluator.evaluation_count == 1


@pytest.mark.unit
def test_validate_geometry_rejects_overlapping_cells() -> None:
    vertices = triangle_piece().geometry.vertices
    validate_geometry(SerialComm(), [vertices])
    with pytest.raises(ValueError, match="overlapping cell geometry"):
        validate_geometry(SerialComm(), [vertices, vertices])


@pytest.mark.unit
def test_distributed_backend_validates_local_coefficient_shape() -> None:
    piece = triangle_piece()
    records = distribute_dofs(
        SerialComm(),
        ((0, 1),),
        [(0, np.array([0.0, 0.0]), piece)],
        triangle_quadrature(1),
    )
    backend = DistributedHierarchicalRieszBackend(
        SerialComm(),
        records,
        0.3,
        triangle_quadrature(1),
        compression_tolerance=1.0e-6,
        admissibility=1.0,
        leaf_size=1,
    )
    with pytest.raises(ValueError, match="local coefficients must have shape"):
        backend.apply_local(np.empty(0))


@pytest.mark.unit
def test_distributed_backend_accepts_an_empty_rank() -> None:
    backend = DistributedHierarchicalRieszBackend(
        SerialComm(),
        (),
        0.3,
        triangle_quadrature(1),
        compression_tolerance=1.0e-6,
        admissibility=1.0,
        leaf_size=1,
    )
    assert backend.local_dofs == ()
