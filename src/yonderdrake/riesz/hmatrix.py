"""Serial hierarchical low-rank compression of the exact Riesz weak matrix."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from time import perf_counter
from typing import Protocol

import numpy as np

from yonderdrake.riesz.admissibility import _admissible_bounds
from yonderdrake.riesz.dense import (
    GalerkinEntryEvaluator,
    RieszMeshData,
)
from yonderdrake.riesz.outer_quadrature import SimplexQuadrature
from yonderdrake.riesz.source_evaluation import (
    SourceActionEvaluator,
    SourceEvaluation,
)


class EntryEvaluator(Protocol):
    def row(self, row: int, columns: np.ndarray) -> np.ndarray: ...

    def column(self, rows: np.ndarray, column: int) -> np.ndarray: ...


@dataclass(frozen=True)
class Cluster:
    """One binary cluster of basis-function supports."""

    indices: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    support_lower: np.ndarray
    support_upper: np.ndarray
    left: Cluster | None = None
    right: Cluster | None = None

    @property
    def diameter(self) -> float:
        return float(np.linalg.norm(self.upper - self.lower))

    @property
    def is_leaf(self) -> bool:
        return self.left is None


@dataclass(frozen=True)
class DenseBlock:
    rows: np.ndarray
    columns: np.ndarray
    values: np.ndarray

    @property
    def stored_entries(self) -> int:
        return int(self.values.size)

    def apply(self, coefficients: np.ndarray, result: np.ndarray) -> None:
        result[self.rows] += self.values @ coefficients[self.columns]


@dataclass(frozen=True)
class LowRankBlock:
    rows: np.ndarray
    columns: np.ndarray
    left: np.ndarray
    right: np.ndarray

    @property
    def rank(self) -> int:
        return int(self.left.shape[1])

    @property
    def stored_entries(self) -> int:
        return int(self.left.size + self.right.size)

    def apply(self, coefficients: np.ndarray, result: np.ndarray) -> None:
        result[self.rows] += self.left @ (self.right.T @ coefficients[self.columns])


def _support_bounds(mesh_data: RieszMeshData) -> tuple[np.ndarray, np.ndarray]:
    lower = np.empty_like(mesh_data.coordinates)
    upper = np.empty_like(mesh_data.coordinates)
    for index, support in enumerate(mesh_data.supports):
        vertices = np.concatenate(
            [piece.geometry.vertices for piece in support],
            axis=0,
        )
        lower[index] = np.min(vertices, axis=0)
        upper[index] = np.max(vertices, axis=0)
    return lower, upper


def _build_cluster(
    indices: np.ndarray,
    coordinates: np.ndarray,
    support_lower: np.ndarray,
    support_upper: np.ndarray,
    leaf_size: int,
) -> Cluster:
    lower = np.min(coordinates[indices], axis=0)
    upper = np.max(coordinates[indices], axis=0)
    cluster_support_lower = np.min(support_lower[indices], axis=0)
    cluster_support_upper = np.max(support_upper[indices], axis=0)
    if indices.size <= leaf_size:
        return Cluster(
            indices,
            lower,
            upper,
            cluster_support_lower,
            cluster_support_upper,
        )
    axis = int(np.argmax(upper - lower))
    ordering = np.argsort(coordinates[indices, axis], kind="stable")
    ordered = indices[ordering]
    middle = ordered.size // 2
    return Cluster(
        indices,
        lower,
        upper,
        cluster_support_lower,
        cluster_support_upper,
        _build_cluster(
            ordered[:middle],
            coordinates,
            support_lower,
            support_upper,
            leaf_size,
        ),
        _build_cluster(
            ordered[middle:],
            coordinates,
            support_lower,
            support_upper,
            leaf_size,
        ),
    )


def _admissible(left: Cluster, right: Cluster, eta: float) -> bool:
    return _admissible_bounds(
        left.lower,
        left.upper,
        left.support_lower,
        left.support_upper,
        right.lower,
        right.upper,
        right.support_lower,
        right.support_upper,
        eta,
    )


def _aca(
    evaluator: EntryEvaluator,
    rows: np.ndarray,
    columns: np.ndarray,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray]:
    row_count = rows.size
    column_count = columns.size
    maximum_rank = min(row_count, column_count)
    left_factors: list[np.ndarray] = []
    right_factors: list[np.ndarray] = []
    used_rows: set[int] = set()
    pivot_row = 0
    approximation_norm_squared = 0.0

    for _ in range(maximum_rank):
        left = np.column_stack(left_factors) if left_factors else None
        right = np.column_stack(right_factors) if right_factors else None
        candidates = [pivot_row]
        candidates.extend(
            index
            for index in range(row_count)
            if index != pivot_row and index not in used_rows
        )
        residual_row = np.empty(0, dtype=np.float64)
        pivot_column = 0
        pivot = 0.0
        chosen_row = -1
        for candidate in candidates:
            if candidate in used_rows:
                continue
            residual_row = evaluator.row(int(rows[candidate]), columns)
            if left is not None and right is not None:
                residual_row -= left[candidate] @ right.T
            pivot_column = int(np.argmax(np.abs(residual_row)))
            pivot = float(residual_row[pivot_column])
            scale = max(
                float(np.max(np.abs(residual_row))),
                np.finfo(np.float64).tiny,
            )
            if abs(pivot) > np.finfo(np.float64).eps * scale:
                chosen_row = candidate
                break
            used_rows.add(candidate)
        if chosen_row < 0:
            break

        residual_column = evaluator.column(rows, int(columns[pivot_column]))
        if left is not None and right is not None:
            residual_column -= left @ right[pivot_column]
        column_pivot = float(residual_column[chosen_row])
        if abs(column_pivot) <= np.finfo(np.float64).tiny:
            used_rows.add(chosen_row)
            pivot_row = next(
                (index for index in range(row_count) if index not in used_rows),
                0,
            )
            continue

        left_factor = residual_column / column_pivot
        right_factor = residual_row
        cross_term = sum(
            2.0
            * float(np.dot(previous_left, left_factor))
            * float(np.dot(previous_right, right_factor))
            for previous_left, previous_right in zip(
                left_factors,
                right_factors,
                strict=True,
            )
        )
        term_norm = float(np.linalg.norm(left_factor) * np.linalg.norm(right_factor))
        approximation_norm_squared = max(
            0.0,
            approximation_norm_squared + term_norm**2 + cross_term,
        )
        left_factors.append(left_factor)
        right_factors.append(right_factor)
        used_rows.add(chosen_row)

        approximation_norm = np.sqrt(approximation_norm_squared)
        if term_norm <= tolerance * max(
            approximation_norm,
            np.finfo(np.float64).tiny,
        ):
            break
        remaining = np.abs(left_factor)
        if used_rows:
            remaining[np.fromiter(used_rows, dtype=np.int64)] = -1.0
        pivot_row = int(np.argmax(remaining))
        if remaining[pivot_row] < 0.0:
            break

    if not left_factors:
        return (
            np.zeros((row_count, 0), dtype=np.float64),
            np.zeros((column_count, 0), dtype=np.float64),
        )
    return np.column_stack(left_factors), np.column_stack(right_factors)


class HierarchicalRieszBackend:
    """Compress well-separated Galerkin blocks with ACA."""

    def __init__(
        self,
        mesh_data: RieszMeshData,
        order: float,
        quadrature: SimplexQuadrature,
        *,
        source_evaluation: SourceEvaluation = "endpoint",
        source_quadrature_degree: int = 4,
        compression_tolerance: float = 1.0e-6,
        admissibility: float = 1.0,
        leaf_size: int = 16,
    ) -> None:
        self.mesh_data = mesh_data
        self.order = order
        self.quadrature = quadrature
        self.source_evaluation = source_evaluation
        self.source_quadrature_degree = source_quadrature_degree
        self.compression_tolerance = compression_tolerance
        self.admissibility = admissibility
        self.leaf_size = leaf_size
        self.blocks: tuple[DenseBlock | LowRankBlock, ...] | None = None
        self.build_seconds = 0.0
        self.apply_seconds = 0.0
        self.apply_count = 0
        self._admissible_blocks = 0
        self._near_field_blocks = 0
        self._cluster_count = 0
        self._leaf_cluster_count = 0
        self._entry_evaluations = 0
        self._source_endpoint_evaluations = 0
        self._source_gauss_evaluations = 0
        self._far_field_ranks: list[int] = []

    def _count_clusters(self, cluster: Cluster) -> None:
        self._cluster_count += 1
        if cluster.is_leaf:
            self._leaf_cluster_count += 1
            return
        assert cluster.left is not None
        assert cluster.right is not None
        self._count_clusters(cluster.left)
        self._count_clusters(cluster.right)

    def build(self) -> tuple[DenseBlock | LowRankBlock, ...]:
        if self.blocks is not None:
            return self.blocks
        started = perf_counter()
        dimension = self.mesh_data.coordinates.shape[0]
        indices = np.arange(dimension, dtype=np.int64)
        support_lower, support_upper = _support_bounds(self.mesh_data)
        root = _build_cluster(
            indices,
            self.mesh_data.coordinates,
            support_lower,
            support_upper,
            self.leaf_size,
        )
        self._count_clusters(root)
        source_action = SourceActionEvaluator(
            self.mesh_data.dof_coordinates.shape[1],
            self.order,
            self.source_evaluation,
            self.source_quadrature_degree,
        )
        near_evaluator = GalerkinEntryEvaluator(
            self.mesh_data,
            self.order,
            self.quadrature,
            cache=True,
            admissibility=self.admissibility,
            pair_admissible=False,
            source_action=source_action,
        )
        far_evaluator = GalerkinEntryEvaluator(
            self.mesh_data,
            self.order,
            self.quadrature,
            cache=True,
            admissibility=self.admissibility,
            pair_admissible=True,
            source_action=source_action,
            prepared_supports=near_evaluator.prepared_supports,
        )
        blocks: list[DenseBlock | LowRankBlock] = []

        def add_blocks(row_cluster: Cluster, column_cluster: Cluster) -> None:
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
                rank = int(left.shape[1])
                self._far_field_ranks.append(rank)
                blocks.append(
                    LowRankBlock(
                        row_cluster.indices,
                        column_cluster.indices,
                        left,
                        right,
                    )
                )
                self._admissible_blocks += 1
                return
            if row_cluster.is_leaf and column_cluster.is_leaf:
                blocks.append(
                    DenseBlock(
                        row_cluster.indices,
                        column_cluster.indices,
                        near_evaluator.block(
                            row_cluster.indices,
                            column_cluster.indices,
                        ),
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

        add_blocks(root, root)
        self._entry_evaluations = (
            near_evaluator.evaluation_count + far_evaluator.evaluation_count
        )
        self._source_endpoint_evaluations = source_action.endpoint_evaluations
        self._source_gauss_evaluations = source_action.quadrature_evaluations
        near_evaluator.clear_cache()
        far_evaluator.clear_cache()
        self.blocks = tuple(blocks)
        self.build_seconds = perf_counter() - started
        return self.blocks

    def apply(self, coefficients: np.ndarray) -> np.ndarray:
        values = np.asarray(coefficients, dtype=np.float64)
        dimension = self.mesh_data.coordinates.shape[0]
        if values.shape != (dimension,):
            raise ValueError(f"coefficients must have shape ({dimension},)")
        blocks = self.build()
        started = perf_counter()
        result = np.zeros_like(values)
        for block in blocks:
            block.apply(values, result)
        self.apply_count += 1
        self.apply_seconds += perf_counter() - started
        return result

    def apply_owned(
        self,
        coefficients: np.ndarray,
        target_cells: Iterable[int],
    ) -> np.ndarray:
        """Apply only in serial, where every target cell is owned."""
        owned = tuple(int(index) for index in target_cells)
        if owned != tuple(range(len(self.mesh_data.cells))):
            raise NotImplementedError(
                "assembly='hmatrix' currently supports serial full applies only"
            )
        return self.apply(coefficients)

    def diagnostics(self) -> dict[str, float | int | str]:
        blocks = self.build()
        stored_entries = sum(block.stored_entries for block in blocks)
        dimension = self.mesh_data.coordinates.shape[0]
        dense_entries = dimension**2
        return {
            "assembly": "hmatrix",
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
            "clusters": self._cluster_count,
            "leaf_clusters": self._leaf_cluster_count,
            "admissible_blocks": self._admissible_blocks,
            "near_field_blocks": self._near_field_blocks,
            "average_far_field_rank": (
                float(np.mean(self._far_field_ranks)) if self._far_field_ranks else 0.0
            ),
            "maximum_far_field_rank": max(self._far_field_ranks, default=0),
            "stored_entries": stored_entries,
            "dense_entries": dense_entries,
            "compression_ratio": stored_entries / dense_entries,
            "entry_evaluations": self._entry_evaluations,
            "build_seconds": self.build_seconds,
            "applications": self.apply_count,
            "apply_seconds": self.apply_seconds,
        }
