"""Robust affine-simplex geometry independent of Firedrake."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

PointClass = Literal[
    "inside",
    "face",
    "edge",
    "vertex",
    "outside_near",
    "outside_far",
]


@dataclass(frozen=True)
class TriangleGeometry:
    """Canonical counter-clockwise geometry for one nondegenerate triangle."""

    vertices: np.ndarray
    edges: np.ndarray
    lengths: np.ndarray
    tangents: np.ndarray
    outward_normals: np.ndarray
    area: float
    diameter: float
    tolerance: float

    @property
    def dimension(self) -> int:
        return 2

    @property
    def measure(self) -> float:
        return self.area

    @property
    def reference_jacobian(self) -> float:
        return 2.0 * self.area

    @classmethod
    def from_vertices(
        cls,
        vertices: object,
        *,
        relative_tolerance: float = 1.0e-12,
    ) -> TriangleGeometry:
        points = np.asarray(vertices, dtype=np.float64)
        if points.shape != (3, 2) or not np.all(np.isfinite(points)):
            raise ValueError("vertices must be a finite array with shape (3, 2)")
        first_edge = points[1] - points[0]
        second_edge = points[2] - points[0]
        signed_twice_area = float(
            first_edge[0] * second_edge[1] - first_edge[1] * second_edge[0]
        )
        pair_lengths = np.array(
            [
                np.linalg.norm(points[1] - points[0]),
                np.linalg.norm(points[2] - points[1]),
                np.linalg.norm(points[0] - points[2]),
            ]
        )
        diameter = float(np.max(pair_lengths))
        if diameter == 0.0 or abs(signed_twice_area) <= (
            relative_tolerance * diameter**2
        ):
            raise ValueError("triangle must be nondegenerate at its geometric scale")
        if signed_twice_area < 0.0:
            points = points[[0, 2, 1]]
            signed_twice_area = -signed_twice_area
        points = points.copy()
        edges = np.roll(points, -1, axis=0) - points
        lengths = np.linalg.norm(edges, axis=1)
        tangents = edges / lengths[:, None]
        normals = np.column_stack((tangents[:, 1], -tangents[:, 0]))
        tolerance = max(
            np.finfo(np.float64).eps * 32.0 * diameter,
            relative_tolerance * diameter,
        )
        for array in (points, edges, lengths, tangents, normals):
            array.setflags(write=False)
        return cls(
            vertices=points,
            edges=edges,
            lengths=lengths,
            tangents=tangents,
            outward_normals=normals,
            area=0.5 * signed_twice_area,
            diameter=diameter,
            tolerance=tolerance,
        )

    def signed_edge_distances(self, point: object) -> np.ndarray:
        x = np.asarray(point, dtype=np.float64)
        if x.shape != (2,) or not np.all(np.isfinite(x)):
            raise ValueError("point must be a finite vector of length 2")
        return np.einsum(
            "ij,ij->i",
            x[None, :] - self.vertices,
            self.outward_normals,
        )

    def barycentric_coordinates(self, point: object) -> np.ndarray:
        x = np.asarray(point, dtype=np.float64)
        matrix = np.column_stack(
            (self.vertices[1] - self.vertices[0], self.vertices[2] - self.vertices[0])
        )
        tail = np.linalg.solve(matrix, x - self.vertices[0])
        return np.array([1.0 - tail.sum(), tail[0], tail[1]])

    def classify(self, point: object) -> PointClass:
        x = np.asarray(point, dtype=np.float64)
        vertex_distances = np.linalg.norm(self.vertices - x[None, :], axis=1)
        if float(np.min(vertex_distances)) <= self.tolerance:
            return "vertex"
        signed = self.signed_edge_distances(x)
        if bool(np.all(signed <= self.tolerance)):
            if bool(np.any(np.abs(signed) <= self.tolerance)):
                return "edge"
            return "inside"
        distance = _point_triangle_distance(self, x)
        if distance <= self.diameter:
            return "outside_near"
        return "outside_far"


def _point_segment_distance(
    point: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
) -> float:
    edge = right - left
    parameter = float(np.dot(point - left, edge) / np.dot(edge, edge))
    parameter = min(1.0, max(0.0, parameter))
    return float(np.linalg.norm(point - (left + parameter * edge)))


def _point_triangle_distance(geometry: TriangleGeometry, point: np.ndarray) -> float:
    return min(
        _point_segment_distance(
            point,
            geometry.vertices[index],
            geometry.vertices[(index + 1) % 3],
        )
        for index in range(3)
    )


@dataclass(frozen=True)
class TetrahedronGeometry:
    """Oriented geometry for one nondegenerate affine tetrahedron."""

    vertices: np.ndarray
    faces: np.ndarray
    face_normals: np.ndarray
    face_areas: np.ndarray
    volume: float
    diameter: float
    tolerance: float

    @property
    def dimension(self) -> int:
        return 3

    @property
    def measure(self) -> float:
        return self.volume

    @property
    def reference_jacobian(self) -> float:
        return 6.0 * self.volume

    @classmethod
    def from_vertices(
        cls,
        vertices: object,
        *,
        relative_tolerance: float = 1.0e-12,
    ) -> TetrahedronGeometry:
        points = np.asarray(vertices, dtype=np.float64)
        if points.shape != (4, 3) or not np.all(np.isfinite(points)):
            raise ValueError("vertices must be a finite array with shape (4, 3)")
        edges = points[1:] - points[0]
        signed_six_volume = float(np.linalg.det(edges))
        pair_lengths = np.asarray(
            [
                np.linalg.norm(points[right] - points[left])
                for left in range(4)
                for right in range(left + 1, 4)
            ],
            dtype=np.float64,
        )
        diameter = float(np.max(pair_lengths))
        if diameter == 0.0 or abs(signed_six_volume) <= (
            relative_tolerance * diameter**3
        ):
            raise ValueError("tetrahedron must be nondegenerate at its geometric scale")
        if signed_six_volume < 0.0:
            points = points[[0, 2, 1, 3]]
            signed_six_volume = -signed_six_volume
        points = points.copy()
        face_indices = (
            (1, 2, 3),
            (0, 3, 2),
            (0, 1, 3),
            (0, 2, 1),
        )
        faces = np.asarray(
            [[points[index] for index in face] for face in face_indices],
            dtype=np.float64,
        )
        normals = np.cross(faces[:, 1] - faces[:, 0], faces[:, 2] - faces[:, 0])
        twice_areas = np.linalg.norm(normals, axis=1)
        normals /= twice_areas[:, None]
        face_areas = 0.5 * twice_areas
        tolerance = max(
            np.finfo(np.float64).eps * 32.0 * diameter,
            relative_tolerance * diameter,
        )
        for array in (points, faces, normals, face_areas):
            array.setflags(write=False)
        return cls(
            vertices=points,
            faces=faces,
            face_normals=normals,
            face_areas=face_areas,
            volume=signed_six_volume / 6.0,
            diameter=diameter,
            tolerance=tolerance,
        )

    def barycentric_coordinates(self, point: object) -> np.ndarray:
        x = np.asarray(point, dtype=np.float64)
        if x.shape != (3,) or not np.all(np.isfinite(x)):
            raise ValueError("point must be a finite vector of length 3")
        matrix = (self.vertices[1:] - self.vertices[0]).T
        tail = np.linalg.solve(matrix, x - self.vertices[0])
        return np.concatenate(([1.0 - tail.sum()], tail))

    def signed_face_distances(self, point: object) -> np.ndarray:
        x = np.asarray(point, dtype=np.float64)
        if x.shape != (3,) or not np.all(np.isfinite(x)):
            raise ValueError("point must be a finite vector of length 3")
        return np.einsum(
            "ij,ij->i",
            x[None, :] - self.faces[:, 0],
            self.face_normals,
        )

    def classify(self, point: object) -> PointClass:
        x = np.asarray(point, dtype=np.float64)
        barycentric = self.barycentric_coordinates(x)
        barycentric_tolerance = self.tolerance / self.diameter
        if bool(np.all(barycentric >= -barycentric_tolerance)):
            zeros = int(np.count_nonzero(np.abs(barycentric) <= barycentric_tolerance))
            if zeros >= 3:
                return "vertex"
            if zeros == 2:
                return "edge"
            if zeros == 1:
                return "face"
            return "inside"
        distance = min(
            _point_triangle_distance_3d(x, face) for face in self.faces
        )
        if distance <= self.diameter:
            return "outside_near"
        return "outside_far"


def _point_triangle_distance_3d(point: np.ndarray, triangle: np.ndarray) -> float:
    """Return the Euclidean distance using the closest-point region test."""
    first, second, third = triangle
    first_edge = second - first
    second_edge = third - first
    relative = point - first
    first_projection = float(np.dot(first_edge, relative))
    second_projection = float(np.dot(second_edge, relative))
    if first_projection <= 0.0 and second_projection <= 0.0:
        return float(np.linalg.norm(relative))

    relative = point - second
    third_projection = float(np.dot(first_edge, relative))
    fourth_projection = float(np.dot(second_edge, relative))
    if third_projection >= 0.0 and fourth_projection <= third_projection:
        return float(np.linalg.norm(relative))

    vertex_region = (
        first_projection * fourth_projection
        - third_projection * second_projection
    )
    if vertex_region <= 0.0 and first_projection >= 0.0 and third_projection <= 0.0:
        parameter = first_projection / (first_projection - third_projection)
        return float(np.linalg.norm(point - (first + parameter * first_edge)))

    relative = point - third
    fifth_projection = float(np.dot(first_edge, relative))
    sixth_projection = float(np.dot(second_edge, relative))
    if sixth_projection >= 0.0 and fifth_projection <= sixth_projection:
        return float(np.linalg.norm(relative))

    edge_region = (
        fifth_projection * second_projection
        - first_projection * sixth_projection
    )
    if edge_region <= 0.0 and second_projection >= 0.0 and sixth_projection <= 0.0:
        parameter = second_projection / (second_projection - sixth_projection)
        return float(np.linalg.norm(point - (first + parameter * second_edge)))

    third_edge_region = (
        third_projection * sixth_projection
        - fifth_projection * fourth_projection
    )
    if (
        third_edge_region <= 0.0
        and fourth_projection - third_projection >= 0.0
        and fifth_projection - sixth_projection >= 0.0
    ):
        parameter = (fourth_projection - third_projection) / (
            fourth_projection - third_projection + fifth_projection - sixth_projection
        )
        return float(
            np.linalg.norm(point - (second + parameter * (third - second)))
        )

    denominator = 1.0 / (vertex_region + edge_region + third_edge_region)
    second_coordinate = edge_region * denominator
    third_coordinate = third_edge_region * denominator
    closest = first + first_edge * second_coordinate + second_edge * third_coordinate
    return float(np.linalg.norm(point - closest))


SimplexGeometry = TriangleGeometry | TetrahedronGeometry
