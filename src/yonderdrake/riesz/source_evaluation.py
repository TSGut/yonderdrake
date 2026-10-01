"""Endpoint and Gaussian source actions for the Riesz fractional Laplacian operator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from yonderdrake.riesz.outer_quadrature import (
    tetrahedron_quadrature,
    triangle_quadrature,
)
from yonderdrake.riesz.triangle_action import (
    SimplexPiece,
    SingularPointError,
    _scaled_piecewise_affine_action_many,
    riesz_normalization,
)

SourceEvaluation = Literal["endpoint", "hybrid"]


@dataclass(frozen=True)
class PreparedSourcePiece:
    piece: SimplexPiece
    points: np.ndarray
    weighted_values: np.ndarray


class SourceActionEvaluator:
    """Route one simplex source according to the source-evaluation policy."""

    def __init__(
        self,
        dimension: int,
        order: float,
        mode: SourceEvaluation,
        quadrature_degree: int,
    ) -> None:
        if dimension not in {2, 3}:
            raise ValueError("source dimension must be 2 or 3")
        if mode not in {"endpoint", "hybrid"}:
            raise ValueError("source_evaluation must be 'endpoint' or 'hybrid'")
        self.dimension = dimension
        self.order = float(order)
        self.mode = mode
        self.quadrature_degree = quadrature_degree
        self.normalization = riesz_normalization(dimension, order)
        self.endpoint_scale = self.normalization / (2.0 * order)
        self.quadrature = (
            None
            if mode == "endpoint"
            else (
                triangle_quadrature(quadrature_degree)
                if dimension == 2
                else tetrahedron_quadrature(quadrature_degree)
            )
        )
        self.endpoint_evaluations = 0
        self.quadrature_evaluations = 0

    def prepare(self, piece: SimplexPiece) -> PreparedSourcePiece:
        """Prepare one source polynomial on its Gaussian nodes."""
        if self.quadrature is None:
            return PreparedSourcePiece(
                piece,
                np.empty((0, self.dimension), dtype=np.float64),
                np.empty(0, dtype=np.float64),
            )
        points = self.quadrature.barycentric @ piece.geometry.vertices
        values = np.fromiter(
            (piece.polynomial(point) for point in points),
            dtype=np.float64,
            count=points.shape[0],
        )
        weighted_values = (
            piece.geometry.reference_jacobian * self.quadrature.weights * values
        )
        return PreparedSourcePiece(piece, points, weighted_values)

    def action(
        self,
        source: PreparedSourcePiece,
        targets: np.ndarray,
        *,
        admissible: bool,
        coincident: bool,
    ) -> np.ndarray:
        """Evaluate one prepared source at a batch of target points."""
        points = np.asarray(targets, dtype=np.float64)
        if not self.uses_quadrature(
            admissible=admissible,
            coincident=coincident,
        ):
            return self.boundary_action_many((source.piece,), points)
        return self.quadrature_action_many((source,), points)

    def boundary_action_many(
        self,
        pieces: tuple[SimplexPiece, ...],
        targets: np.ndarray,
    ) -> np.ndarray:
        """Evaluate source pieces after reduction to their boundaries."""
        self.endpoint_evaluations += len(pieces)
        return _scaled_piecewise_affine_action_many(
            pieces,
            np.asarray(targets, dtype=np.float64),
            self.order,
            self.endpoint_scale,
            tetrahedron_edge_evaluation=(
                "automatic" if self.mode == "hybrid" else "appell"
            ),
        )

    def uses_quadrature(self, *, admissible: bool, coincident: bool) -> bool:
        """Return whether one source pair follows the Gaussian route."""
        return self.mode == "hybrid" and admissible and not coincident

    def quadrature_action_many(
        self,
        sources: tuple[PreparedSourcePiece, ...],
        targets: np.ndarray,
    ) -> np.ndarray:
        """Evaluate smooth source cells in one Gaussian batch."""
        points = np.asarray(targets, dtype=np.float64)
        if not sources:
            return np.zeros(points.shape[0], dtype=np.float64)
        source_points = np.concatenate([source.points for source in sources])
        weighted_values = np.concatenate([source.weighted_values for source in sources])
        origin = points[0] if points.shape[0] else source_points[0]
        centered_points = points - origin
        centered_sources = source_points - origin
        target_norms = np.einsum("ij,ij->i", centered_points, centered_points)
        source_norms = np.einsum("ij,ij->i", centered_sources, centered_sources)
        squared_distances = (
            target_norms[:, None]
            + source_norms[None, :]
            - 2.0 * (centered_points @ centered_sources.T)
        )
        cancellation_scale = target_norms[:, None] + source_norms[None, :]
        cancellation = squared_distances <= (
            32.0 * np.finfo(np.float64).eps * cancellation_scale
        )
        if bool(np.any(cancellation)):
            target_indices, source_indices = np.nonzero(cancellation)
            differences = points[target_indices] - source_points[source_indices]
            squared_distances[target_indices, source_indices] = np.einsum(
                "ij,ij->i",
                differences,
                differences,
            )
        if bool(np.any(squared_distances == 0.0)):
            raise SingularPointError(
                "source quadrature encountered a coincident target point"
            )
        kernel = np.power(
            squared_distances,
            -0.5 * (self.dimension + 2.0 * self.order),
        )
        self.quadrature_evaluations += len(sources)
        return -self.normalization * (kernel @ weighted_values)
