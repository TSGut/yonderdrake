"""Distributed hierarchical compression of Riesz Galerkin rank blocks."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
from mpi4py import MPI
from scipy.spatial import ConvexHull

from yonderdrake.riesz.geometry import SimplexGeometry
from yonderdrake.riesz.hmatrix import (
    Cluster,
    _aca,
    _admissible,
    _build_cluster,
)
from yonderdrake.riesz.outer_quadrature import SimplexQuadrature
from yonderdrake.riesz.source_evaluation import (
    PreparedSourcePiece,
    SourceActionEvaluator,
    SourceEvaluation,
)
from yonderdrake.riesz.triangle_action import (
    SimplexPiece,
    _scaled_piecewise_affine_action_many,
    riesz_normalization,
)


@dataclass(frozen=True)
class DistributedTargetPart:
    geometry: SimplexGeometry
    points: np.ndarray
    weights: np.ndarray


@dataclass(frozen=True)
class DistributedDof:
    global_index: int
    coordinate: np.ndarray
    support: tuple[SimplexPiece, ...]
    row_points: np.ndarray
    row_weights: np.ndarray
    row_parts: tuple[DistributedTargetPart, ...]
    support_lower: np.ndarray
    support_upper: np.ndarray


@dataclass(frozen=True)
class CompressionRequest:
    near_columns: np.ndarray
    projection_columns: tuple[np.ndarray, ...]
    projection_factors: tuple[np.ndarray, ...]

    @property
    def response_size(self) -> int:
        return int(
            self.near_columns.size
            + sum(factor.shape[1] for factor in self.projection_factors)
        )


@dataclass(frozen=True)
class ResponseLayout:
    near_size: int
    projection_sizes: tuple[int, ...]

    @property
    def response_size(self) -> int:
        return self.near_size + sum(self.projection_sizes)


@dataclass(frozen=True)
class TargetDenseBlock:
    source_rank: int
    rows: np.ndarray
    near_positions: np.ndarray
    values: np.ndarray


@dataclass(frozen=True)
class TargetLowRankBlock:
    source_rank: int
    rows: np.ndarray
    left: np.ndarray
    response_offset: int


class PairEntryEvaluator:
    """Evaluate a local-row/remote-column rank block."""

    def __init__(
        self,
        rows: tuple[DistributedDof, ...],
        columns: tuple[DistributedDof, ...],
        order: float,
        *,
        source_evaluation: SourceEvaluation = "endpoint",
        source_quadrature_degree: int = 4,
        pair_admissible: bool = False,
        source_action: SourceActionEvaluator | None = None,
        prepared_supports: tuple[tuple[PreparedSourcePiece, ...], ...] | None = None,
    ) -> None:
        self.rows = rows
        self.columns = columns
        self.order = order
        records = rows or columns
        dimension = int(records[0].coordinate.size) if records else 2
        self.scale = riesz_normalization(dimension, order) / (2.0 * order)
        self.pair_admissible = pair_admissible
        self.source_action = source_action or SourceActionEvaluator(
            dimension,
            order,
            source_evaluation,
            source_quadrature_degree,
        )
        if prepared_supports is not None:
            self.prepared_supports = prepared_supports
        elif self.source_action.mode == "endpoint":
            self.prepared_supports = tuple(() for _ in columns)
        else:
            self.prepared_supports = tuple(
                tuple(self.source_action.prepare(piece) for piece in column.support)
                for column in columns
            )
        self.evaluation_count = 0
        self._entries: dict[tuple[int, int], float] = {}

    def entry(self, row: int, column: int) -> float:
        key = (int(row), int(column))
        if key in self._entries:
            return self._entries[key]
        self.evaluation_count += 1
        row_record = self.rows[key[0]]
        support = self.columns[key[1]].support
        if self.source_action.mode == "endpoint" or (
            self.source_action.mode == "hybrid" and not self.pair_admissible
        ):
            self.source_action.endpoint_evaluations += len(support)
            actions = _scaled_piecewise_affine_action_many(
                support,
                row_record.row_points,
                self.order,
                self.scale,
            )
            value = float(np.dot(row_record.row_weights, actions))
        else:
            value = 0.0
            for target in row_record.row_parts:
                actions = np.zeros(target.points.shape[0], dtype=np.float64)
                endpoint_pieces = []
                quadrature_sources = []
                for source in self.prepared_supports[key[1]]:
                    coincident = np.array_equal(
                        target.geometry.vertices,
                        source.piece.geometry.vertices,
                    )
                    if self.source_action.uses_quadrature(
                        admissible=self.pair_admissible,
                        coincident=coincident,
                    ):
                        quadrature_sources.append(source)
                    else:
                        endpoint_pieces.append(source.piece)
                if endpoint_pieces:
                    self.source_action.endpoint_evaluations += len(endpoint_pieces)
                    actions += _scaled_piecewise_affine_action_many(
                        tuple(endpoint_pieces),
                        target.points,
                        self.order,
                        self.scale,
                    )
                actions += self.source_action.quadrature_action_many(
                    tuple(quadrature_sources),
                    target.points,
                )
                value += float(np.dot(target.weights, actions))
        self._entries[key] = value
        return value

    def row(self, row: int, columns: np.ndarray) -> np.ndarray:
        return np.fromiter(
            (self.entry(row, int(column)) for column in columns),
            dtype=np.float64,
            count=columns.size,
        )

    def column(self, rows: np.ndarray, column: int) -> np.ndarray:
        return np.fromiter(
            (self.entry(int(row), column) for row in rows),
            dtype=np.float64,
            count=rows.size,
        )

    def block(self, rows: np.ndarray, columns: np.ndarray) -> np.ndarray:
        return np.vstack([self.row(int(row), columns) for row in rows])


def _cluster(
    records: tuple[DistributedDof, ...],
    leaf_size: int,
) -> Cluster:
    coordinates = np.asarray([record.coordinate for record in records])
    support_lower = np.asarray([record.support_lower for record in records])
    support_upper = np.asarray([record.support_upper for record in records])
    return _build_cluster(
        np.arange(len(records), dtype=np.int64),
        coordinates,
        support_lower,
        support_upper,
        leaf_size,
    )


def distribute_dofs(
    comm: Any,
    ownership_ranges: tuple[tuple[int, int], ...],
    contributions: list[tuple[int, np.ndarray, SimplexPiece]],
    quadrature: SimplexQuadrature,
) -> tuple[DistributedDof, ...]:
    """Send every cell-basis piece to its PETSc DOF owner."""

    def owner(index: int) -> int:
        for rank, (start, end) in enumerate(ownership_ranges):
            if start <= index < end:
                return rank
        raise ValueError(f"global DOF {index} has no owner")

    outbound: list[list[tuple[int, np.ndarray, SimplexPiece]]] = [
        [] for _ in ownership_ranges
    ]
    for index, coordinate, piece in contributions:
        outbound[owner(index)].append((index, coordinate, piece))
    inbound = comm.alltoall(outbound)
    grouped: dict[int, tuple[np.ndarray, list[SimplexPiece]]] = {}
    for rank_values in inbound:
        for index, coordinate, piece in rank_values:
            if index not in grouped:
                grouped[index] = (np.asarray(coordinate).copy(), [])
            grouped[index][1].append(piece)
    start, end = ownership_ranges[comm.rank]
    if set(grouped) != set(range(start, end)):
        raise RuntimeError("distributed Riesz support does not match PETSc ownership")
    records = []
    for index in range(start, end):
        coordinate, pieces = grouped[index]
        point_parts = []
        weight_parts = []
        row_parts = []
        dimension = int(coordinate.size)
        support_lower = np.full(dimension, np.inf)
        support_upper = np.full(dimension, -np.inf)
        for piece in pieces:
            points = quadrature.barycentric @ piece.geometry.vertices
            weights = piece.geometry.reference_jacobian * quadrature.weights
            point_parts.append(points)
            row_weights = weights * np.fromiter(
                (piece.polynomial(point) for point in points),
                dtype=np.float64,
                count=points.shape[0],
            )
            weight_parts.append(row_weights)
            row_parts.append(DistributedTargetPart(piece.geometry, points, row_weights))
            support_lower = np.minimum(
                support_lower,
                np.min(piece.geometry.vertices, axis=0),
            )
            support_upper = np.maximum(
                support_upper,
                np.max(piece.geometry.vertices, axis=0),
            )
        records.append(
            DistributedDof(
                global_index=index,
                coordinate=coordinate,
                support=tuple(pieces),
                row_points=np.concatenate(point_parts),
                row_weights=np.concatenate(weight_parts),
                row_parts=tuple(row_parts),
                support_lower=support_lower,
                support_upper=support_upper,
            )
        )
    return tuple(records)


def validate_geometry(
    comm: Any,
    cell_vertices: list[np.ndarray],
) -> None:
    """Reject overlapping or periodic affine cells collectively."""
    local_dimension = int(cell_vertices[0].shape[1]) if cell_vertices else 0
    dimension = max(comm.allgather(local_dimension))
    if dimension not in {2, 3}:
        raise ValueError("Riesz cells must have dimension 2 or 3")
    if dimension == 2:
        local_measure = sum(
            0.5
            * abs(
                (vertices[1, 0] - vertices[0, 0]) * (vertices[2, 1] - vertices[0, 1])
                - (vertices[1, 1] - vertices[0, 1]) * (vertices[2, 0] - vertices[0, 0])
            )
            for vertices in cell_vertices
        )
    else:
        local_measure = sum(
            abs(float(np.linalg.det((vertices[1:] - vertices[0]).T))) / 6.0
            for vertices in cell_vertices
        )
    cell_measure = float(comm.allreduce(local_measure))
    local_points = (
        np.unique(np.concatenate(cell_vertices, axis=0), axis=0)
        if cell_vertices
        else np.empty((0, dimension))
    )
    if local_points.shape[0] > dimension:
        local_points = local_points[ConvexHull(local_points).vertices]
    hull_points = np.unique(
        np.concatenate(comm.allgather(local_points), axis=0),
        axis=0,
    )
    hull_measure = float(ConvexHull(hull_points).volume)
    if cell_measure > hull_measure + 1.0e-10 * max(1.0, hull_measure):
        raise ValueError(
            "overlapping cell geometry is unsupported; "
            "periodic meshes cannot define a zero exterior"
        )


class DistributedHierarchicalRieszBackend:
    """Store local H-matrix rows and exchange compressed source data."""

    def __init__(
        self,
        comm: Any,
        local_dofs: tuple[DistributedDof, ...],
        order: float,
        quadrature: SimplexQuadrature,
        *,
        source_evaluation: SourceEvaluation = "endpoint",
        source_quadrature_degree: int = 4,
        compression_tolerance: float,
        admissibility: float,
        leaf_size: int,
    ) -> None:
        self.comm = comm
        self.local_dofs = local_dofs
        self.order = order
        self.quadrature = quadrature
        self.source_evaluation = source_evaluation
        self.source_quadrature_degree = source_quadrature_degree
        self.compression_tolerance = compression_tolerance
        self.admissibility = admissibility
        self.leaf_size = leaf_size
        self.build_seconds = 0.0
        self.apply_seconds = 0.0
        self.apply_count = 0
        self._dense_blocks: tuple[TargetDenseBlock, ...] = ()
        self._low_rank_blocks: tuple[TargetLowRankBlock, ...] = ()
        self._response_layouts: tuple[ResponseLayout, ...] = ()
        self._incoming_requests: tuple[CompressionRequest, ...] = ()
        self._send_counts = np.empty(0, dtype=np.int32)
        self._receive_counts = np.empty(0, dtype=np.int32)
        self._send_displacements = np.empty(0, dtype=np.int32)
        self._receive_displacements = np.empty(0, dtype=np.int32)
        self._stored_entries = 0
        self._dense_entries = 0
        self._admissible_blocks = 0
        self._near_field_blocks = 0
        self._entry_evaluations = 0
        self._source_endpoint_evaluations = 0
        self._source_gauss_evaluations = 0
        self._far_field_ranks: list[int] = []
        self._build()

    def _build(self) -> None:
        started = perf_counter()
        row_root = (
            _cluster(self.local_dofs, self.leaf_size) if self.local_dofs else None
        )
        dense_work: list[tuple[int, np.ndarray, np.ndarray, np.ndarray]] = []
        low_rank_work: list[
            tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        ] = []
        projection_columns: list[list[np.ndarray]] = [[] for _ in range(self.comm.size)]
        projection_factors: list[list[np.ndarray]] = [[] for _ in range(self.comm.size)]
        near_columns: list[list[np.ndarray]] = [[] for _ in range(self.comm.size)]

        for source_rank in range(self.comm.size):
            source_dofs = self.comm.bcast(
                self.local_dofs if self.comm.rank == source_rank else None,
                root=source_rank,
            )
            if row_root is None or not source_dofs:
                continue
            column_root = _cluster(source_dofs, self.leaf_size)
            source_action = SourceActionEvaluator(
                int(self.local_dofs[0].coordinate.size),
                self.order,
                self.source_evaluation,
                self.source_quadrature_degree,
            )
            near_evaluator = PairEntryEvaluator(
                self.local_dofs,
                source_dofs,
                self.order,
                pair_admissible=False,
                source_action=source_action,
            )
            far_evaluator = PairEntryEvaluator(
                self.local_dofs,
                source_dofs,
                self.order,
                pair_admissible=True,
                source_action=source_action,
                prepared_supports=near_evaluator.prepared_supports,
            )

            def add_blocks(
                row_cluster: Cluster,
                column_cluster: Cluster,
                near_evaluator: PairEntryEvaluator = near_evaluator,
                far_evaluator: PairEntryEvaluator = far_evaluator,
                source_rank: int = source_rank,
            ) -> None:
                if _admissible(
                    row_cluster,
                    column_cluster,
                    self.admissibility,
                ):
                    left, right = _aca(
                        far_evaluator,
                        row_cluster.indices,
                        column_cluster.indices,
                        self.compression_tolerance,
                    )
                    projection_index = len(projection_columns[source_rank])
                    projection_columns[source_rank].append(
                        column_cluster.indices.copy()
                    )
                    projection_factors[source_rank].append(right)
                    low_rank_work.append(
                        (
                            source_rank,
                            row_cluster.indices.copy(),
                            left,
                            column_cluster.indices.copy(),
                            np.asarray([projection_index]),
                        )
                    )
                    self._admissible_blocks += 1
                    self._far_field_ranks.append(int(left.shape[1]))
                    return
                if row_cluster.is_leaf and column_cluster.is_leaf:
                    values = near_evaluator.block(
                        row_cluster.indices,
                        column_cluster.indices,
                    )
                    near_columns[source_rank].append(column_cluster.indices.copy())
                    dense_work.append(
                        (
                            source_rank,
                            row_cluster.indices.copy(),
                            column_cluster.indices.copy(),
                            values,
                        )
                    )
                    self._near_field_blocks += 1
                    return
                if not row_cluster.is_leaf and not column_cluster.is_leaf:
                    assert row_cluster.left is not None
                    assert row_cluster.right is not None
                    assert column_cluster.left is not None
                    assert column_cluster.right is not None
                    for row_child in (row_cluster.left, row_cluster.right):
                        for column_child in (
                            column_cluster.left,
                            column_cluster.right,
                        ):
                            add_blocks(row_child, column_child)
                elif not row_cluster.is_leaf:
                    assert row_cluster.left is not None
                    assert row_cluster.right is not None
                    add_blocks(row_cluster.left, column_cluster)
                    add_blocks(row_cluster.right, column_cluster)
                else:
                    assert column_cluster.left is not None
                    assert column_cluster.right is not None
                    add_blocks(row_cluster, column_cluster.left)
                    add_blocks(row_cluster, column_cluster.right)

            add_blocks(row_root, column_root)
            self._entry_evaluations += (
                near_evaluator.evaluation_count + far_evaluator.evaluation_count
            )
            self._source_endpoint_evaluations += source_action.endpoint_evaluations
            self._source_gauss_evaluations += source_action.quadrature_evaluations

        requests = []
        near_maps = []
        projection_offsets = []
        for source_rank in range(self.comm.size):
            unique_near = (
                np.unique(np.concatenate(near_columns[source_rank]))
                if near_columns[source_rank]
                else np.empty(0, dtype=np.int64)
            )
            near_maps.append(
                {int(column): index for index, column in enumerate(unique_near)}
            )
            offsets = []
            offset = int(unique_near.size)
            for factor in projection_factors[source_rank]:
                offsets.append(offset)
                offset += factor.shape[1]
            projection_offsets.append(offsets)
            requests.append(
                CompressionRequest(
                    near_columns=unique_near,
                    projection_columns=tuple(projection_columns[source_rank]),
                    projection_factors=tuple(projection_factors[source_rank]),
                )
            )
        self._incoming_requests = tuple(self.comm.alltoall(requests))
        self._response_layouts = tuple(
            ResponseLayout(
                near_size=request.near_columns.size,
                projection_sizes=tuple(
                    factor.shape[1] for factor in request.projection_factors
                ),
            )
            for request in requests
        )
        self._dense_blocks = tuple(
            TargetDenseBlock(
                source_rank=source_rank,
                rows=rows,
                near_positions=np.asarray(
                    [near_maps[source_rank][int(column)] for column in columns],
                    dtype=np.int64,
                ),
                values=values,
            )
            for source_rank, rows, columns, values in dense_work
        )
        self._low_rank_blocks = tuple(
            TargetLowRankBlock(
                source_rank=source_rank,
                rows=rows,
                left=left,
                response_offset=projection_offsets[source_rank][
                    int(projection_index[0])
                ],
            )
            for (
                source_rank,
                rows,
                left,
                _columns,
                projection_index,
            ) in low_rank_work
        )
        self._send_counts = np.asarray(
            [request.response_size for request in self._incoming_requests],
            dtype=np.int32,
        )
        self._receive_counts = np.asarray(
            [layout.response_size for layout in self._response_layouts],
            dtype=np.int32,
        )
        self._send_displacements = np.concatenate(
            (
                np.zeros(1, dtype=np.int32),
                np.cumsum(self._send_counts[:-1], dtype=np.int32),
            )
        )
        self._receive_displacements = np.concatenate(
            (
                np.zeros(1, dtype=np.int32),
                np.cumsum(self._receive_counts[:-1], dtype=np.int32),
            )
        )
        local_stored = sum(block.values.size for block in self._dense_blocks) + sum(
            block.left.size for block in self._low_rank_blocks
        )
        local_stored += sum(
            factor.size
            for request in self._incoming_requests
            for factor in request.projection_factors
        )
        self._stored_entries = int(self.comm.allreduce(local_stored))
        local_dimension = len(self.local_dofs)
        global_dimension = int(self.comm.allreduce(local_dimension))
        self._dense_entries = global_dimension**2
        self._admissible_blocks = int(self.comm.allreduce(self._admissible_blocks))
        self._near_field_blocks = int(self.comm.allreduce(self._near_field_blocks))
        self._entry_evaluations = int(self.comm.allreduce(self._entry_evaluations))
        self._source_endpoint_evaluations = int(
            self.comm.allreduce(self._source_endpoint_evaluations)
        )
        self._source_gauss_evaluations = int(
            self.comm.allreduce(self._source_gauss_evaluations)
        )
        all_ranks = self.comm.allgather(self._far_field_ranks)
        self._far_field_ranks = [
            rank for rank_values in all_ranks for rank in rank_values
        ]
        self.build_seconds = max(self.comm.allgather(perf_counter() - started))

    def apply_local(self, coefficients: np.ndarray) -> np.ndarray:
        """Apply to the locally owned PETSc coefficient segment."""
        started = perf_counter()
        values = np.asarray(coefficients, dtype=np.float64)
        if values.shape != (len(self.local_dofs),):
            raise ValueError(
                f"local coefficients must have shape ({len(self.local_dofs)},)"
            )
        send_buffer = np.empty(int(np.sum(self._send_counts)))
        for target_rank, request in enumerate(self._incoming_requests):
            offset = int(self._send_displacements[target_rank])
            near_size = request.near_columns.size
            send_buffer[offset : offset + near_size] = values[request.near_columns]
            offset += near_size
            for columns, factor in zip(
                request.projection_columns,
                request.projection_factors,
                strict=True,
            ):
                projection = factor.T @ values[columns]
                send_buffer[offset : offset + projection.size] = projection
                offset += projection.size
        receive_buffer = np.empty(int(np.sum(self._receive_counts)))
        self.comm.Alltoallv(
            [
                send_buffer,
                self._send_counts,
                self._send_displacements,
                MPI.DOUBLE,
            ],
            [
                receive_buffer,
                self._receive_counts,
                self._receive_displacements,
                MPI.DOUBLE,
            ],
        )
        result = np.zeros_like(values)
        for dense_block in self._dense_blocks:
            source_start = int(self._receive_displacements[dense_block.source_rank])
            source_values = receive_buffer[
                source_start : source_start
                + self._response_layouts[dense_block.source_rank].near_size
            ]
            result[dense_block.rows] += (
                dense_block.values @ source_values[dense_block.near_positions]
            )
        for low_rank_block in self._low_rank_blocks:
            start = (
                int(self._receive_displacements[low_rank_block.source_rank])
                + low_rank_block.response_offset
            )
            rank = low_rank_block.left.shape[1]
            result[low_rank_block.rows] += (
                low_rank_block.left @ receive_buffer[start : start + rank]
            )
        self.apply_count += 1
        self.apply_seconds += perf_counter() - started
        return result

    def diagnostics(self) -> dict[str, float | int | str]:
        return {
            "assembly": "hmatrix",
            "distribution": "rank_block",
            "source_evaluation": self.source_evaluation,
            "source_quadrature_degree": self.source_quadrature_degree,
            "source_endpoint_evaluations": self._source_endpoint_evaluations,
            "source_gauss_evaluations": self._source_gauss_evaluations,
            "target_quadrature_degree": self.quadrature.degree,
            "target_quadrature_rule": self.quadrature.rule,
            "target_quadrature_points_per_cell": self.quadrature.num_points,
            "compression_tolerance": self.compression_tolerance,
            "admissibility": self.admissibility,
            "leaf_size": self.leaf_size,
            "admissible_blocks": self._admissible_blocks,
            "near_field_blocks": self._near_field_blocks,
            "average_far_field_rank": (
                float(np.mean(self._far_field_ranks)) if self._far_field_ranks else 0.0
            ),
            "maximum_far_field_rank": max(
                self._far_field_ranks,
                default=0,
            ),
            "stored_entries": self._stored_entries,
            "dense_entries": self._dense_entries,
            "compression_ratio": (self._stored_entries / self._dense_entries),
            "entry_evaluations": self._entry_evaluations,
            "build_seconds": self.build_seconds,
            "applications": self.apply_count,
            "apply_seconds": self.apply_seconds,
            "replicated_source_dofs": 0,
        }
