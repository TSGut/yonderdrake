"""Pure numerical verification of hierarchical Riesz compression."""

from __future__ import annotations

import numpy as np
import pytest

from yonderdrake.riesz.dense import DenseRieszBackend, RieszMeshData
from yonderdrake.riesz.hmatrix import (
    HierarchicalRieszBackend,
    LowRankBlock,
    _aca,
)
from yonderdrake.riesz.outer_quadrature import (
    face_tetrahedron_quadrature,
    triangle_quadrature,
)


def structured_unit_square(
    subdivisions: int,
) -> tuple[np.ndarray, np.ndarray]:
    coordinates = np.asarray(
        [
            (column / subdivisions, row / subdivisions)
            for row in range(subdivisions + 1)
            for column in range(subdivisions + 1)
        ],
        dtype=np.float64,
    )
    cells = []
    width = subdivisions + 1
    for row in range(subdivisions):
        for column in range(subdivisions):
            lower_left = row * width + column
            lower_right = lower_left + 1
            upper_left = lower_left + width
            upper_right = upper_left + 1
            cells.extend(
                [
                    (lower_left, lower_right, upper_left),
                    (lower_right, upper_right, upper_left),
                ]
            )
    return coordinates, np.asarray(cells, dtype=np.int64)


def separated_unit_squares(
    subdivisions: int,
) -> tuple[np.ndarray, np.ndarray]:
    coordinates, cells = structured_unit_square(subdivisions)
    dimension = coordinates.shape[0]
    return (
        np.vstack((coordinates, coordinates + np.array([4.0, 0.0]))),
        np.vstack((cells, cells + dimension)),
    )


def relative_error(actual: np.ndarray, expected: np.ndarray) -> float:
    return float(
        np.linalg.norm(actual - expected)
        / max(np.linalg.norm(expected), np.finfo(np.float64).tiny)
    )


def quadratic_mesh(
    coordinates: np.ndarray,
    cells: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    dof_coordinates = [coordinate.copy() for coordinate in coordinates]
    edges: dict[tuple[int, int], int] = {}
    cell_dofs = []
    for cell in cells:
        local = [int(index) for index in cell]
        for left, right in (
            (cell[0], cell[1]),
            (cell[1], cell[2]),
            (cell[2], cell[0]),
        ):
            edge = tuple(sorted((int(left), int(right))))
            if edge not in edges:
                edges[edge] = len(dof_coordinates)
                dof_coordinates.append(
                    0.5 * (coordinates[edge[0]] + coordinates[edge[1]])
                )
            local.append(edges[edge])
        cell_dofs.append(local)
    return np.asarray(dof_coordinates), np.asarray(cell_dofs, dtype=np.int64)


def separated_tetrahedra() -> tuple[np.ndarray, np.ndarray]:
    first = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    return (
        np.vstack((first, first + np.array([4.0, 0.0, 0.0]))),
        np.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.int64),
    )


def quadratic_tetrahedral_mesh(
    coordinates: np.ndarray,
    cells: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    dof_coordinates = [coordinate.copy() for coordinate in coordinates]
    edges: dict[tuple[int, int], int] = {}
    cell_dofs = []
    for cell in cells:
        local = [int(index) for index in cell]
        for left in range(4):
            for right in range(left + 1, 4):
                edge = tuple(sorted((int(cell[left]), int(cell[right]))))
                if edge not in edges:
                    edges[edge] = len(dof_coordinates)
                    dof_coordinates.append(
                        0.5 * (coordinates[edge[0]] + coordinates[edge[1]])
                    )
                local.append(edges[edge])
        cell_dofs.append(local)
    return np.asarray(dof_coordinates), np.asarray(cell_dofs, dtype=np.int64)


@pytest.mark.verification
@pytest.mark.parametrize("order", [0.05, 0.45, 0.95])
def test_hmatrix_matches_dense_across_fractional_orders(order: float) -> None:
    coordinates, cells = separated_unit_squares(3)
    mesh = RieszMeshData.build(coordinates, cells)
    quadrature = triangle_quadrature(3)
    tolerance = 1.0e-3
    dense = DenseRieszBackend(mesh, order, quadrature)
    hierarchical = HierarchicalRieszBackend(
        mesh,
        order,
        quadrature,
        compression_tolerance=tolerance,
        admissibility=1.0,
        leaf_size=4,
    )
    coefficients = np.random.default_rng(1729).standard_normal(coordinates.shape[0])
    error = relative_error(
        hierarchical.apply(coefficients),
        dense.apply(coefficients),
    )
    assert error < 5.0 * tolerance
    diagnostics = hierarchical.diagnostics()
    assert diagnostics["admissible_blocks"] > 0
    assert diagnostics["near_field_blocks"] > 0
    assert diagnostics["compression_ratio"] < 1.0
    blocks = hierarchical.build()
    assert hierarchical.build() is blocks
    assert all(block.rank >= 0 for block in blocks if isinstance(block, LowRankBlock))
    with pytest.raises(ValueError, match="coefficients must have shape"):
        hierarchical.apply(coefficients[:-1])


@pytest.mark.unit
def test_aca_represents_an_exact_zero_block_with_zero_rank() -> None:
    class ZeroEvaluator:
        def row(self, row: int, columns: np.ndarray) -> np.ndarray:
            return np.zeros(columns.size)

        def column(self, rows: np.ndarray, column: int) -> np.ndarray:
            return np.zeros(rows.size)

    rows = np.arange(3, dtype=np.int64)
    columns = np.arange(4, dtype=np.int64)
    left, right = _aca(ZeroEvaluator(), rows, columns, 1.0e-8)
    assert left.shape == (3, 0)
    assert right.shape == (4, 0)


@pytest.mark.verification
def test_cg2_hmatrix_matches_dense() -> None:
    coordinates, cells = separated_unit_squares(2)
    dof_coordinates, cell_dofs = quadratic_mesh(coordinates, cells)
    mesh = RieszMeshData.build(
        coordinates,
        cells,
        dof_coordinates=dof_coordinates,
        cell_dofs=cell_dofs,
        degree=2,
    )
    quadrature = triangle_quadrature(4)
    coefficients = np.random.default_rng(19).standard_normal(mesh.dimension)
    reference = DenseRieszBackend(mesh, 0.4, quadrature).apply(coefficients)
    backend = HierarchicalRieszBackend(
        mesh,
        0.4,
        quadrature,
        compression_tolerance=1.0e-5,
        leaf_size=4,
    )
    assert relative_error(backend.apply(coefficients), reference) < 8.0e-5
    assert backend.diagnostics()["admissible_blocks"] > 0


@pytest.mark.verification
def test_hmatrix_hybrid_routes_near_and_far_blocks() -> None:
    coordinates, cells = separated_unit_squares(2)
    mesh = RieszMeshData.build(coordinates, cells)
    quadrature = triangle_quadrature(5)
    coefficients = np.random.default_rng(23).standard_normal(mesh.dimension)
    reference = DenseRieszBackend(
        mesh,
        0.4,
        quadrature,
        source_evaluation="hybrid",
        source_quadrature_degree=8,
    ).apply(coefficients)
    backend = HierarchicalRieszBackend(
        mesh,
        0.4,
        quadrature,
        source_evaluation="hybrid",
        source_quadrature_degree=8,
        compression_tolerance=1.0e-7,
        leaf_size=2,
    )
    assert relative_error(backend.apply(coefficients), reference) < 8.0e-7
    diagnostics = backend.diagnostics()
    assert diagnostics["source_endpoint_evaluations"] > 0
    assert diagnostics["source_gauss_evaluations"] > 0


@pytest.mark.verification
@pytest.mark.parametrize("degree", [1, 2])
def test_tetrahedral_hmatrix_matches_dense(degree: int) -> None:
    coordinates, cells = separated_tetrahedra()
    if degree == 1:
        dof_coordinates = coordinates
        cell_dofs = cells
    else:
        dof_coordinates, cell_dofs = quadratic_tetrahedral_mesh(
            coordinates,
            cells,
        )
    mesh = RieszMeshData.build(
        coordinates,
        cells,
        dof_coordinates=dof_coordinates,
        cell_dofs=cell_dofs,
        degree=degree,
    )
    quadrature = face_tetrahedron_quadrature(
        1,
        0.3,
        zero_trace=False,
        field_degree=degree,
    )
    coefficients = np.random.default_rng(27).standard_normal(mesh.dimension)
    reference = DenseRieszBackend(mesh, 0.3, quadrature).apply(coefficients)
    backend = HierarchicalRieszBackend(
        mesh,
        0.3,
        quadrature,
        compression_tolerance=1.0e-5,
        admissibility=1.0,
        leaf_size=2,
    )
    assert relative_error(backend.apply(coefficients), reference) < 8.0e-5
    assert backend.diagnostics()["admissible_blocks"] > 0


@pytest.mark.verification
def test_compression_error_falls_below_outer_quadrature_error() -> None:
    coordinates, cells = separated_unit_squares(3)
    mesh = RieszMeshData.build(coordinates, cells)
    coefficients = np.random.default_rng(81).standard_normal(coordinates.shape[0])
    coarse_rule = triangle_quadrature(2)
    fine_rule = triangle_quadrature(6)
    coarse = DenseRieszBackend(mesh, 0.4, coarse_rule).apply(coefficients)
    fine = DenseRieszBackend(mesh, 0.4, fine_rule).apply(coefficients)
    quadrature_error = relative_error(coarse, fine)
    compression_errors = []
    for tolerance in (1.0e-2, 1.0e-3, 1.0e-4):
        compressed = HierarchicalRieszBackend(
            mesh,
            0.4,
            coarse_rule,
            compression_tolerance=tolerance,
            admissibility=1.0,
            leaf_size=4,
        ).apply(coefficients)
        compression_errors.append(relative_error(compressed, coarse))
    assert compression_errors[1] <= compression_errors[0]
    assert compression_errors[2] <= compression_errors[1]
    assert compression_errors[-1] < quadrature_error


@pytest.mark.verification
def test_stricter_admissibility_is_stable_toward_dense() -> None:
    coordinates, cells = structured_unit_square(4)
    mesh = RieszMeshData.build(coordinates, cells)
    quadrature = triangle_quadrature(3)
    coefficients = np.random.default_rng(24).standard_normal(coordinates.shape[0])
    reference = DenseRieszBackend(mesh, 0.65, quadrature).apply(coefficients)
    errors = []
    for eta in (2.0, 1.0, 0.5):
        backend = HierarchicalRieszBackend(
            mesh,
            0.65,
            quadrature,
            compression_tolerance=1.0e-1,
            admissibility=eta,
            leaf_size=4,
        )
        errors.append(relative_error(backend.apply(coefficients), reference))
    assert errors[1] <= 1.05 * errors[0] + 1.0e-13
    assert errors[2] <= 1.05 * errors[1] + 1.0e-13
    np.testing.assert_allclose(
        backend.apply_owned(coefficients, range(len(cells))),
        backend.apply(coefficients),
    )
    with pytest.raises(NotImplementedError, match="serial full applies"):
        backend.apply_owned(coefficients, range(len(cells) - 1))
