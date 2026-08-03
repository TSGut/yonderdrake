"""Serial O(N²) matrix-free reference action using the frozen pointwise kernel."""

from __future__ import annotations

from collections.abc import Iterable
from time import perf_counter

import numpy as np

from yonderdrake.riesz.dense import RieszMeshData
from yonderdrake.riesz.outer_quadrature import SimplexQuadrature
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
    ) -> None:
        self.mesh_data = mesh_data
        self.order = order
        self.quadrature = quadrature
        dimension = int(mesh_data.dof_coordinates.shape[1])
        if quadrature.dimension != dimension:
            raise ValueError("quadrature dimension must match the Riesz mesh")
        self._action_scale = riesz_normalization(dimension, order) / (2.0 * order)
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
        result = np.zeros_like(values)
        for target_index in target_cells:
            index = int(target_index)
            cell = self.mesh_data.cell_dofs[index]
            basis = self.mesh_data.local_basis[index]
            points = self._target_points[index]
            weighted_actions = self._target_weights[index] * (
                _scaled_piecewise_affine_action_many(
                    pieces,
                    points,
                    self.order,
                    self._action_scale,
                )
            )
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
            "quadrature_degree": self.quadrature.degree,
            "quadrature_rule": self.quadrature.rule,
            "quadrature_points_per_cell": self.quadrature.num_points,
            "stored_entries": 0,
            "applications": self.apply_count,
            "apply_seconds": self.apply_seconds,
        }
