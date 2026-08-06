"""Serial dense reference Galerkin assembly for the Riesz operator."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy.spatial import ConvexHull

from yonderdrake.riesz.admissibility import _admissible_cells
from yonderdrake.riesz.geometry import (
    SimplexGeometry,
    TetrahedronGeometry,
    TriangleGeometry,
)
from yonderdrake.riesz.outer_quadrature import SimplexQuadrature
from yonderdrake.riesz.source_evaluation import (
    PreparedSourcePiece,
    SourceActionEvaluator,
    SourceEvaluation,
)
from yonderdrake.riesz.triangle_action import (
    AffinePolynomial,
    QuadraticPolynomial,
    SimplexPiece,
    SimplexPolynomial,
    _scaled_piecewise_affine_action_many,
)


def local_polynomial_basis(
    coordinates: np.ndarray,
    degree: int,
) -> tuple[SimplexPolynomial, ...]:
    """Construct the physical nodal basis on one affine simplex."""
    points = np.asarray(coordinates, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] not in {2, 3}:
        raise ValueError("basis coordinates must have shape (N, 2) or (N, 3)")
    if degree not in {1, 2}:
        raise NotImplementedError("Riesz basis supports degree 1 or 2")
    dimension = int(points.shape[1])
    local_dimension = (
        dimension + 1 if degree == 1 else ((dimension + 1) * (dimension + 2) // 2)
    )
    if points.shape != (local_dimension, dimension):
        raise ValueError(
            f"degree-{degree} basis coordinates must have shape "
            f"({local_dimension}, {dimension})"
        )
    if degree == 1:
        coefficients = np.linalg.solve(
            np.column_stack((np.ones(local_dimension), points)),
            np.eye(local_dimension),
        ).T
        return tuple(AffinePolynomial(values[0], values[1:]) for values in coefficients)
    quadratic_columns = [0.5 * points[:, axis] ** 2 for axis in range(dimension)]
    cross_pairs = [
        (left, right)
        for left in range(dimension)
        for right in range(left + 1, dimension)
    ]
    quadratic_columns.extend(
        points[:, left] * points[:, right] for left, right in cross_pairs
    )
    coefficients = np.linalg.solve(
        np.column_stack((np.ones(local_dimension), points, *quadratic_columns)),
        np.eye(local_dimension),
    ).T
    basis = []
    for values in coefficients:
        hessian = np.zeros((dimension, dimension), dtype=np.float64)
        hessian[np.diag_indices(dimension)] = values[1 + dimension : 1 + 2 * dimension]
        for value, (left, right) in zip(
            values[1 + 2 * dimension :],
            cross_pairs,
            strict=True,
        ):
            hessian[left, right] = value
            hessian[right, left] = value
        basis.append(
            QuadraticPolynomial(
                values[0],
                values[1 : 1 + dimension],
                hessian,
            )
        )
    return tuple(basis)


@dataclass
class RieszMeshData:
    vertex_coordinates: np.ndarray
    geometry_cells: np.ndarray
    dof_coordinates: np.ndarray
    cell_dofs: np.ndarray
    degree: int
    geometries: tuple[SimplexGeometry, ...]
    local_basis: tuple[tuple[SimplexPolynomial, ...], ...]
    supports: tuple[tuple[SimplexPiece, ...], ...]
    support_cells: tuple[tuple[int, ...], ...]

    @property
    def coordinates(self) -> np.ndarray:
        """Return field-DOF coordinates."""
        return self.dof_coordinates

    @property
    def cells(self) -> np.ndarray:
        """Return cell-to-field-DOF maps."""
        return self.cell_dofs

    @property
    def dimension(self) -> int:
        return int(self.dof_coordinates.shape[0])

    @classmethod
    def build(
        cls,
        coordinates: object,
        cells: object,
        *,
        dof_coordinates: object | None = None,
        cell_dofs: object | None = None,
        degree: int = 1,
    ) -> RieszMeshData:
        vertex_array = np.asarray(coordinates, dtype=np.float64)
        geometry_cell_array = np.asarray(cells, dtype=np.int64)
        if (
            vertex_array.ndim != 2
            or vertex_array.shape[1] not in {2, 3}
            or geometry_cell_array.ndim != 2
            or geometry_cell_array.shape[1] != vertex_array.shape[1] + 1
        ):
            raise ValueError(
                "Riesz mesh data requires 2D vertices and triangular cells, "
                "or 3D vertices and tetrahedral cells"
            )
        if degree not in {1, 2}:
            raise NotImplementedError("Riesz mesh data supports degree 1 or 2")
        dof_array = np.asarray(
            vertex_array if dof_coordinates is None else dof_coordinates,
            dtype=np.float64,
        )
        field_cell_array = np.asarray(
            geometry_cell_array if cell_dofs is None else cell_dofs,
            dtype=np.int64,
        )
        dimension = int(vertex_array.shape[1])
        if dof_array.ndim != 2 or dof_array.shape[1] != dimension:
            raise ValueError(
                f"Riesz field DOF coordinates must have shape (N, {dimension})"
            )
        local_dimension = (
            dimension + 1 if degree == 1 else ((dimension + 1) * (dimension + 2) // 2)
        )
        if (
            field_cell_array.ndim != 2
            or field_cell_array.shape[0] != geometry_cell_array.shape[0]
            or field_cell_array.shape[1] != local_dimension
        ):
            raise ValueError(
                f"degree-{degree} Riesz cells require {local_dimension} field DOFs"
            )
        geometry_type = TriangleGeometry if dimension == 2 else TetrahedronGeometry
        geometries = tuple(
            geometry_type.from_vertices(vertex_array[cell])
            for cell in geometry_cell_array
        )
        local_basis = tuple(
            local_polynomial_basis(dof_array[cell], degree) for cell in field_cell_array
        )
        cell_measure = sum(geometry.measure for geometry in geometries)
        hull_measure = float(ConvexHull(vertex_array).volume)
        if cell_measure > hull_measure + 1.0e-10 * max(1.0, hull_measure):
            raise ValueError(
                "overlapping cell geometry is unsupported; "
                "periodic meshes cannot define a zero exterior"
            )
        mutable_supports: list[list[SimplexPiece]] = [
            [] for _ in range(dof_array.shape[0])
        ]
        mutable_support_cells: list[list[int]] = [[] for _ in range(dof_array.shape[0])]
        for cell_index, (cell, geometry, basis) in enumerate(
            zip(
                field_cell_array,
                geometries,
                local_basis,
                strict=True,
            )
        ):
            for global_index, polynomial in zip(cell, basis, strict=True):
                mutable_supports[int(global_index)].append(
                    SimplexPiece(geometry, polynomial)
                )
                mutable_support_cells[int(global_index)].append(cell_index)
        return cls(
            vertex_coordinates=vertex_array.copy(),
            geometry_cells=geometry_cell_array.copy(),
            dof_coordinates=dof_array.copy(),
            cell_dofs=field_cell_array.copy(),
            degree=degree,
            geometries=geometries,
            local_basis=local_basis,
            supports=tuple(tuple(support) for support in mutable_supports),
            support_cells=tuple(tuple(cells) for cells in mutable_support_cells),
        )


class GalerkinEntryEvaluator:
    """Sample target-quadrature Galerkin rows and columns."""

    def __init__(
        self,
        mesh_data: RieszMeshData,
        order: float,
        quadrature: SimplexQuadrature,
        *,
        cache: bool = False,
        source_evaluation: SourceEvaluation = "endpoint",
        source_quadrature_degree: int = 4,
        admissibility: float = 1.0,
        pair_admissible: bool | None = None,
        source_action: SourceActionEvaluator | None = None,
        prepared_supports: tuple[tuple[PreparedSourcePiece, ...], ...] | None = None,
    ) -> None:
        self.mesh_data = mesh_data
        self.order = order
        self.quadrature = quadrature
        self.cache = cache
        self.admissibility = admissibility
        self.pair_admissible = pair_admissible
        self.evaluation_count = 0
        self._entries: dict[tuple[int, int], float] = {}
        dimension = int(mesh_data.dof_coordinates.shape[1])
        if quadrature.dimension != dimension:
            raise ValueError("quadrature dimension must match the Riesz mesh")
        self.source_action = source_action or SourceActionEvaluator(
            dimension,
            order,
            source_evaluation,
            source_quadrature_degree,
        )
        row_parts: list[list[tuple[int, np.ndarray, np.ndarray]]] = [
            [] for _ in mesh_data.supports
        ]
        for cell_index, (cell, geometry, basis) in enumerate(
            zip(
                mesh_data.cell_dofs,
                mesh_data.geometries,
                mesh_data.local_basis,
                strict=True,
            )
        ):
            points = quadrature.barycentric @ geometry.vertices
            weights = geometry.reference_jacobian * quadrature.weights
            for global_index, polynomial in zip(cell, basis, strict=True):
                row_parts[int(global_index)].append(
                    (
                        cell_index,
                        points,
                        weights
                        * np.fromiter(
                            (polynomial(point) for point in points),
                            dtype=np.float64,
                            count=points.shape[0],
                        ),
                    )
                )
        self._row_parts = tuple(tuple(parts) for parts in row_parts)
        self._row_points = tuple(
            np.concatenate([part[1] for part in parts], axis=0)
            for parts in self._row_parts
        )
        self._row_weights = tuple(
            np.concatenate([part[2] for part in parts]) for parts in self._row_parts
        )
        if prepared_supports is not None:
            self.prepared_supports = prepared_supports
        elif self.source_action.mode == "endpoint":
            self.prepared_supports = tuple(() for _ in mesh_data.supports)
        else:
            self.prepared_supports = tuple(
                tuple(self.source_action.prepare(piece) for piece in support)
                for support in mesh_data.supports
            )

    def _evaluate(self, row: int, column: int) -> float:
        self.evaluation_count += 1
        if self.source_action.mode == "endpoint" or (
            self.source_action.mode == "hybrid" and self.pair_admissible is False
        ):
            self.source_action.endpoint_evaluations += len(
                self.mesh_data.supports[column]
            )
            actions = _scaled_piecewise_affine_action_many(
                self.mesh_data.supports[column],
                self._row_points[row],
                self.order,
                self.source_action.endpoint_scale,
            )
            return float(np.dot(self._row_weights[row], actions))
        total = 0.0
        for target_cell, points, weights in self._row_parts[row]:
            actions = np.zeros(points.shape[0], dtype=np.float64)
            target_geometry = self.mesh_data.geometries[target_cell]
            endpoint_pieces = []
            quadrature_sources = []
            for source_cell, source in zip(
                self.mesh_data.support_cells[column],
                self.prepared_supports[column],
                strict=True,
            ):
                admissible = (
                    self.pair_admissible
                    if self.pair_admissible is not None
                    else _admissible_cells(
                        target_geometry,
                        source.piece.geometry,
                        self.admissibility,
                    )
                )
                coincident = target_cell == source_cell
                if self.source_action.uses_quadrature(
                    admissible=admissible,
                    coincident=coincident,
                ):
                    quadrature_sources.append(source)
                else:
                    endpoint_pieces.append(source.piece)
            if endpoint_pieces:
                self.source_action.endpoint_evaluations += len(endpoint_pieces)
                actions += _scaled_piecewise_affine_action_many(
                    tuple(endpoint_pieces),
                    points,
                    self.order,
                    self.source_action.endpoint_scale,
                )
            actions += self.source_action.quadrature_action_many(
                tuple(quadrature_sources),
                points,
            )
            total += float(np.dot(weights, actions))
        return total

    def entry(self, row: int, column: int) -> float:
        """Evaluate one target-quadrature matrix entry."""
        key = (int(row), int(column))
        if self.cache and key in self._entries:
            return self._entries[key]
        value = self._evaluate(*key)
        if self.cache:
            self._entries[key] = value
        return value

    def row(self, row: int, columns: np.ndarray) -> np.ndarray:
        """Evaluate one row on the requested column index set."""
        column_indices = np.asarray(columns, dtype=np.int64)
        return np.fromiter(
            (self.entry(int(row), int(column)) for column in column_indices),
            dtype=np.float64,
            count=column_indices.size,
        )

    def column(self, rows: np.ndarray, column: int) -> np.ndarray:
        """Evaluate one column on the requested row index set."""
        row_indices = np.asarray(rows, dtype=np.int64)
        return np.fromiter(
            (self.entry(int(row), int(column)) for row in row_indices),
            dtype=np.float64,
            count=row_indices.size,
        )

    def block(self, rows: np.ndarray, columns: np.ndarray) -> np.ndarray:
        """Materialize one exact sub-block using the shared row primitive."""
        row_indices = np.asarray(rows, dtype=np.int64)
        column_indices = np.asarray(columns, dtype=np.int64)
        return np.vstack([self.row(int(row), column_indices) for row in row_indices])

    def clear_cache(self) -> None:
        """Discard temporary ACA samples after compression."""
        self._entries.clear()


class DenseRieszBackend:
    """Assemble and cache the weak Galerkin matrix."""

    def __init__(
        self,
        mesh_data: RieszMeshData,
        order: float,
        quadrature: SimplexQuadrature,
        *,
        source_evaluation: SourceEvaluation = "endpoint",
        source_quadrature_degree: int = 4,
        admissibility: float = 1.0,
    ) -> None:
        self.mesh_data = mesh_data
        self.order = order
        self.quadrature = quadrature
        self.source_evaluation = source_evaluation
        self.source_quadrature_degree = source_quadrature_degree
        self.admissibility = admissibility
        self.matrix: np.ndarray | None = None
        self.assembly_seconds = 0.0
        self.apply_count = 0
        self._source_endpoint_evaluations = 0
        self._source_gauss_evaluations = 0

    def assemble(self) -> np.ndarray:
        if self.matrix is not None:
            return self.matrix
        started = perf_counter()
        dimension = self.mesh_data.dimension
        indices = np.arange(dimension, dtype=np.int64)
        evaluator = GalerkinEntryEvaluator(
            self.mesh_data,
            self.order,
            self.quadrature,
            source_evaluation=self.source_evaluation,
            source_quadrature_degree=self.source_quadrature_degree,
            admissibility=self.admissibility,
        )
        matrix = evaluator.block(indices, indices)
        self._source_endpoint_evaluations = evaluator.source_action.endpoint_evaluations
        self._source_gauss_evaluations = evaluator.source_action.quadrature_evaluations
        self.matrix = matrix
        self.assembly_seconds = perf_counter() - started
        return matrix

    def apply(self, coefficients: np.ndarray) -> np.ndarray:
        self.apply_count += 1
        return self.assemble() @ np.asarray(coefficients, dtype=np.float64)

    def diagnostics(self) -> dict[str, float | int | str]:
        matrix = self.assemble()
        asymmetry = np.linalg.norm(matrix - matrix.T) / max(
            np.linalg.norm(matrix),
            np.finfo(np.float64).tiny,
        )
        return {
            "assembly": "dense",
            "source_evaluation": self.source_evaluation,
            "source_quadrature_degree": self.source_quadrature_degree,
            "source_endpoint_evaluations": self._source_endpoint_evaluations,
            "source_gauss_evaluations": self._source_gauss_evaluations,
            "target_quadrature_degree": self.quadrature.degree,
            "target_quadrature_rule": self.quadrature.rule,
            "target_quadrature_points_per_cell": self.quadrature.num_points,
            "stored_entries": int(matrix.size),
            "assembly_seconds": self.assembly_seconds,
            "applications": self.apply_count,
            "relative_asymmetry": float(asymmetry),
        }
