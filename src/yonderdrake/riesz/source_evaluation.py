"""Endpoint and Gaussian source actions for the Riesz operator."""

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
    """Route one simplex source through endpoint or Gaussian evaluation."""

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
            self.endpoint_evaluations += 1
            return _scaled_piecewise_affine_action_many(
                (source.piece,),
                points,
                self.order,
                self.endpoint_scale,
            )
        return self.quadrature_action_many((source,), points)

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
        differences = points[:, None, :] - source_points[None, :, :]
        squared_distances = np.einsum(
            "ijk,ijk->ij",
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
