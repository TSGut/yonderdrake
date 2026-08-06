"""Serial O(N²) matrix-free reference action using the frozen pointwise kernel."""

from __future__ import annotations

from collections.abc import Iterable
from time import perf_counter

import numpy as np

from yonderdrake.riesz.admissibility import _admissible_cells
from yonderdrake.riesz.dense import RieszMeshData
from yonderdrake.riesz.outer_quadrature import SimplexQuadrature
from yonderdrake.riesz.source_evaluation import (
    SourceActionEvaluator,
    SourceEvaluation,
)
from yonderdrake.riesz.triangle_action import (
    SimplexPiece,
    _scaled_piecewise_affine_action_many,
    combine_polynomials,
    riesz_normalization,
)


class MatrixFreeRieszBackend:
    """Apply without storing an ``N x N`` Galerkin matrix."""

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
        dimension = int(mesh_data.dof_coordinates.shape[1])
        if quadrature.dimension != dimension:
            raise ValueError("quadrature dimension must match the Riesz mesh")
        self._action_scale = riesz_normalization(dimension, order) / (2.0 * order)
        self._source_action = SourceActionEvaluator(
            dimension,
            order,
            source_evaluation,
            source_quadrature_degree,
        )
        self._target_points = tuple(
            quadrature.barycentric @ geometry.vertices
            for geometry in mesh_data.geometries
        )
        self._target_weights = tuple(
            geometry.reference_jacobian * quadrature.weights
            for geometry in mesh_data.geometries
        )
        self.apply_count = 0
        self.apply_seconds = 0.0

    def apply(self, coefficients: np.ndarray) -> np.ndarray:
        return self.apply_owned(
            coefficients,
            range(len(self.mesh_data.cell_dofs)),
        )

    def apply_owned(
        self,
        coefficients: np.ndarray,
        target_cells: Iterable[int],
    ) -> np.ndarray:
        """Apply contributions from a deterministic subset of target cells."""
        started = perf_counter()
        values = np.asarray(coefficients, dtype=np.float64)
        pieces = tuple(
            SimplexPiece(
                geometry,
                combine_polynomials(basis, values[cell]),
            )
            for cell, geometry, basis in zip(
                self.mesh_data.cell_dofs,
                self.mesh_data.geometries,
                self.mesh_data.local_basis,
                strict=True,
            )
        )
        prepared = (
            ()
            if self.source_evaluation == "endpoint"
            else tuple(self._source_action.prepare(piece) for piece in pieces)
        )
        result = np.zeros_like(values)
        for target_index in target_cells:
            index = int(target_index)
            cell = self.mesh_data.cell_dofs[index]
            basis = self.mesh_data.local_basis[index]
            points = self._target_points[index]
            if self.source_evaluation == "endpoint":
                self._source_action.endpoint_evaluations += len(pieces)
                actions = _scaled_piecewise_affine_action_many(
                    pieces,
                    points,
                    self.order,
                    self._action_scale,
                )
            elif self.source_evaluation == "hybrid":
                near_pieces = []
                far_sources = []
                target_geometry = self.mesh_data.geometries[index]
                for piece, source in zip(pieces, prepared, strict=True):
                    if _admissible_cells(
                        target_geometry,
                        piece.geometry,
                        self.admissibility,
                    ):
                        far_sources.append(source)
                    else:
                        near_pieces.append(piece)
                actions = np.zeros(points.shape[0], dtype=np.float64)
                if near_pieces:
                    self._source_action.endpoint_evaluations += len(near_pieces)
                    actions += _scaled_piecewise_affine_action_many(
                        tuple(near_pieces),
                        points,
                        self.order,
                        self._action_scale,
                    )
                actions += self._source_action.quadrature_action_many(
                    tuple(far_sources),
                    points,
                )
            else:
                self._source_action.endpoint_evaluations += 1
                actions = _scaled_piecewise_affine_action_many(
                    (pieces[index],),
                    points,
                    self.order,
                    self._action_scale,
                )
                actions += self._source_action.quadrature_action_many(
                    prepared[:index] + prepared[index + 1 :],
                    points,
                )
            weighted_actions = self._target_weights[index] * actions
            for global_index, polynomial in zip(cell, basis, strict=True):
                basis_values = np.fromiter(
                    (polynomial(point) for point in points),
                    dtype=np.float64,
                    count=points.shape[0],
                )
                result[int(global_index)] += float(
                    np.dot(weighted_actions, basis_values)
                )
        self.apply_count += 1
        self.apply_seconds += perf_counter() - started
        return result

    def diagnostics(self) -> dict[str, float | int | str]:
        return {
            "assembly": "matfree",
            "source_evaluation": self.source_evaluation,
            "source_quadrature_degree": self.source_quadrature_degree,
            "source_endpoint_evaluations": self._source_action.endpoint_evaluations,
            "source_gauss_evaluations": self._source_action.quadrature_evaluations,
            "target_quadrature_degree": self.quadrature.degree,
            "target_quadrature_rule": self.quadrature.rule,
            "target_quadrature_points_per_cell": self.quadrature.num_points,
            "stored_entries": 0,
            "applications": self.apply_count,
            "apply_seconds": self.apply_seconds,
        }
