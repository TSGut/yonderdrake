"""Pure numerical tests for target quadrature and serial backends."""

from __future__ import annotations

import numpy as np
import pytest
from tests.reference.riesz_galerkin import unit_square_symmetric_energy_matrix

from yonderdrake.riesz.dense import (
    DenseRieszBackend,
    GalerkinEntryEvaluator,
    RieszMeshData,
    local_polynomial_basis,
)
from yonderdrake.riesz.matfree import MatrixFreeRieszBackend
from yonderdrake.riesz.outer_quadrature import (
    edge_triangle_quadrature,
    face_tetrahedron_quadrature,
    tetrahedron_quadrature,
    triangle_quadrature,
)
from yonderdrake.riesz.triangle_action import (
    AffinePolynomial,
    SimplexPiece,
    piecewise_affine_action,
)

COORDINATES = np.array(
    [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
)
CELLS = np.array([[0, 1, 3], [1, 2, 3]])
CG2_COORDINATES = np.array(
    [
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
        [0.5, 0.0],
        [0.5, 0.5],
        [0.0, 0.5],
        [1.0, 0.5],
        [0.5, 1.0],
    ]
)
CG2_CELLS = np.array(
    [
        [0, 1, 3, 4, 5, 6],
        [1, 2, 3, 7, 8, 5],
    ]
)
TETRAHEDRON_COORDINATES = np.array(
    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
)
TETRAHEDRON_CELLS = np.array([[0, 1, 2, 3]])
TETRAHEDRON_CG2_COORDINATES = np.vstack(
    (
        TETRAHEDRON_COORDINATES,
        [
            0.5 * (TETRAHEDRON_COORDINATES[left] + TETRAHEDRON_COORDINATES[right])
            for left in range(4)
            for right in range(left + 1, 4)
        ],
    )
)
TETRAHEDRON_CG2_CELLS = np.arange(10, dtype=np.int64)[None, :]


def structured_mesh(subdivisions: int) -> tuple[np.ndarray, np.ndarray]:
    coordinates = np.asarray(
        [
            (column / subdivisions, row / subdivisions)
            for row in range(subdivisions + 1)
            for column in range(subdivisions + 1)
        ]
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
    return coordinates, np.asarray(cells)


def public_oracle_action(
    mesh: RieszMeshData,
    coefficients: np.ndarray,
    order: float,
    degree: int,
) -> np.ndarray:
    rule = triangle_quadrature(degree)
    pieces = []
    for cell, geometry in zip(mesh.cells, mesh.geometries, strict=True):
        vandermonde = np.column_stack(
            (np.ones(3), geometry.vertices[:, 0], geometry.vertices[:, 1])
        )
        affine = np.linalg.solve(vandermonde, coefficients[cell])
        pieces.append(
            SimplexPiece(
                geometry,
                AffinePolynomial(affine[0], affine[1:]),
            )
        )
    result = np.zeros_like(coefficients)
    for cell, geometry in zip(mesh.cells, mesh.geometries, strict=True):
        points = rule.barycentric @ geometry.vertices
        weights = 2.0 * geometry.area * rule.weights
        for barycentric, point, weight in zip(
            rule.barycentric,
            points,
            weights,
            strict=True,
        ):
            action = piecewise_affine_action(pieces, point, order)
            result[cell] += weight * barycentric * action
    return result


@pytest.mark.unit
def test_triangle_quadrature_integrates_reference_area_and_linear_fields() -> None:
    rule = triangle_quadrature(4)
    assert np.sum(rule.weights) == pytest.approx(0.5)
    np.testing.assert_allclose(
        np.sum(rule.weights[:, None] * rule.barycentric, axis=0),
        np.full(3, 1.0 / 6.0),
    )
    assert np.all(rule.barycentric > 0.0)


@pytest.mark.unit
def test_tetrahedron_quadrature_integrates_quadratic_moments() -> None:
    rule = tetrahedron_quadrature(4)
    assert np.sum(rule.weights) == pytest.approx(1.0 / 6.0)
    np.testing.assert_allclose(
        np.sum(rule.weights[:, None] * rule.barycentric, axis=0),
        np.full(4, 1.0 / 24.0),
    )
    for left in range(4):
        for right in range(4):
            expected = 1.0 / 60.0 if left == right else 1.0 / 120.0
            assert np.sum(
                rule.weights
                * rule.barycentric[:, left]
                * rule.barycentric[:, right]
            ) == pytest.approx(expected)


@pytest.mark.unit
def test_quadrature_arguments_are_validated() -> None:
    for degree in (0, 1.5, True):
        with pytest.raises(ValueError, match="quadrature_degree"):
            triangle_quadrature(degree)
        with pytest.raises(ValueError, match="quadrature_degree"):
            edge_triangle_quadrature(
                degree,
                0.4,
                zero_trace=True,
                field_degree=1,
            )
    with pytest.raises(ValueError, match="0 < order < 1"):
        edge_triangle_quadrature(
            2,
            1.0,
            zero_trace=True,
            field_degree=1,
        )
    with pytest.raises(ValueError, match="field_degree"):
        edge_triangle_quadrature(
            2,
            0.4,
            zero_trace=True,
            field_degree=3,
        )


@pytest.mark.unit
def test_mesh_data_layout_is_validated() -> None:
    with pytest.raises(ValueError, match="basis coordinates"):
        local_polynomial_basis(np.zeros((2, 2)), 1)
    with pytest.raises(ValueError, match="basis coordinates"):
        local_polynomial_basis(np.zeros((3, 2)), 2)
    with pytest.raises(ValueError, match="triangular cells"):
        RieszMeshData.build(np.zeros((3, 3)), np.array([[0, 1, 2]]))
    with pytest.raises(NotImplementedError, match="degree 1 or 2"):
        RieszMeshData.build(COORDINATES, CELLS, degree=3)
    with pytest.raises(ValueError, match="DOF coordinates"):
        RieszMeshData.build(
            COORDINATES,
            CELLS,
            dof_coordinates=np.zeros((4, 3)),
        )
    with pytest.raises(ValueError, match="field DOFs"):
        RieszMeshData.build(
            COORDINATES,
            CELLS,
            cell_dofs=np.array([[0, 1], [1, 2]]),
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("order", "zero_trace", "exponent"),
    [(0.3, True, 0.0), (0.3, False, -0.6), (0.8, True, -0.6)],
)
def test_edge_quadrature_tracks_trace_singularity(
    order: float,
    zero_trace: bool,
    exponent: float,
) -> None:
    rule = edge_triangle_quadrature(
        6,
        order,
        zero_trace=zero_trace,
        field_degree=1,
    )
    assert rule.rule == "boundary"
    assert rule.singular_exponent == pytest.approx(exponent)
    assert rule.num_points == 3 * triangle_quadrature(6).num_points
    assert np.sum(rule.weights) == pytest.approx(0.5)
    np.testing.assert_allclose(
        np.sum(rule.weights[:, None] * rule.barycentric, axis=0),
        np.full(3, 1.0 / 6.0),
    )
    assert np.all(rule.barycentric > 0.0)
    assert np.all(rule.weights > 0.0)


@pytest.mark.unit
@pytest.mark.parametrize("order", [1.0e-6, 0.499999, 0.5, 0.999999])
def test_edge_quadrature_is_finite_at_limiting_orders(order: float) -> None:
    rule = edge_triangle_quadrature(
        6,
        order,
        zero_trace=order >= 0.5,
        field_degree=2,
    )
    assert np.all(np.isfinite(rule.barycentric))
    assert np.all(np.isfinite(rule.weights))
    quadratic = np.sum(rule.barycentric**2, axis=1)
    assert np.sum(rule.weights * quadratic) == pytest.approx(0.25)


@pytest.mark.unit
@pytest.mark.parametrize("order", [1.0e-6, 0.499999, 0.5, 0.999999])
def test_face_quadrature_is_finite_and_preserves_cg2_moments(order: float) -> None:
    rule = face_tetrahedron_quadrature(
        4,
        order,
        zero_trace=order >= 0.5,
        field_degree=2,
    )
    assert rule.rule == "boundary"
    assert np.all(np.isfinite(rule.barycentric))
    assert np.all(np.isfinite(rule.weights))
    assert np.all(rule.weights > 0.0)
    np.testing.assert_allclose(
        rule.weights @ rule.barycentric,
        np.full(4, 1.0 / 24.0),
    )
    for left in range(4):
        for right in range(4):
            expected = 1.0 / 60.0 if left == right else 1.0 / 120.0
            assert np.sum(
                rule.weights
                * rule.barycentric[:, left]
                * rule.barycentric[:, right]
            ) == pytest.approx(expected)


@pytest.mark.verification
def test_edge_quadrature_resolves_high_order_interface_action() -> None:
    coordinates, cells = structured_mesh(3)
    interior = (
        (coordinates[:, 0] > 0.0)
        & (coordinates[:, 0] < 1.0)
        & (coordinates[:, 1] > 0.0)
        & (coordinates[:, 1] < 1.0)
    )
    coordinates[interior] += np.array(
        [[0.06, -0.03], [-0.04, 0.05], [0.03, 0.04], [-0.05, -0.02]]
    )
    mesh = RieszMeshData.build(coordinates, cells)
    indices = np.flatnonzero(interior)
    order = 0.9
    edge_coarse = GalerkinEntryEvaluator(
        mesh,
        order,
        edge_triangle_quadrature(
            6,
            order,
            zero_trace=True,
            field_degree=1,
        ),
    ).block(indices, indices)
    edge_fine = GalerkinEntryEvaluator(
        mesh,
        order,
        edge_triangle_quadrature(
            12,
            order,
            zero_trace=True,
            field_degree=1,
        ),
    ).block(indices, indices)
    ordinary = GalerkinEntryEvaluator(
        mesh,
        order,
        triangle_quadrature(12),
    ).block(indices, indices)
    edge_error = np.linalg.norm(edge_coarse - edge_fine)
    ordinary_error = np.linalg.norm(ordinary - edge_fine)
    assert edge_error < ordinary_error
    edge_asymmetry = np.linalg.norm(edge_coarse - edge_coarse.T)
    ordinary_asymmetry = np.linalg.norm(ordinary - ordinary.T)
    assert edge_asymmetry < ordinary_asymmetry


@pytest.mark.verification
def test_edge_quadrature_resolves_nonzero_trace() -> None:
    mesh = RieszMeshData.build(COORDINATES, CELLS)
    coefficients = np.ones(mesh.dimension)
    order = 0.49
    edge_coarse = MatrixFreeRieszBackend(
        mesh,
        order,
        edge_triangle_quadrature(
            6,
            order,
            zero_trace=False,
            field_degree=1,
        ),
    ).apply(coefficients)
    edge_fine = MatrixFreeRieszBackend(
        mesh,
        order,
        edge_triangle_quadrature(
            12,
            order,
            zero_trace=False,
            field_degree=1,
        ),
    ).apply(coefficients)
    ordinary = MatrixFreeRieszBackend(
        mesh,
        order,
        triangle_quadrature(12),
    ).apply(coefficients)
    assert np.linalg.norm(edge_coarse - edge_fine) < np.linalg.norm(
        ordinary - edge_fine
    )


@pytest.mark.verification
def test_dense_and_matrix_free_weak_actions_agree() -> None:
    mesh = RieszMeshData.build(COORDINATES, CELLS)
    rule = triangle_quadrature(5)
    dense = DenseRieszBackend(mesh, 0.35, rule)
    matrix_free = MatrixFreeRieszBackend(mesh, 0.35, rule)
    coefficients = np.array([0.2, 1.0, -0.3, 0.4])
    np.testing.assert_allclose(
        matrix_free.apply(coefficients),
        dense.apply(coefficients),
        rtol=3.0e-12,
        atol=3.0e-12,
    )
    assert matrix_free.diagnostics()["stored_entries"] == 0


@pytest.mark.verification
def test_cg2_dense_and_matrix_free_weak_actions_agree() -> None:
    mesh = RieszMeshData.build(
        COORDINATES,
        CELLS,
        dof_coordinates=CG2_COORDINATES,
        cell_dofs=CG2_CELLS,
        degree=2,
    )
    rule = triangle_quadrature(6)
    dense = DenseRieszBackend(mesh, 0.35, rule)
    matrix_free = MatrixFreeRieszBackend(mesh, 0.35, rule)
    coefficients = np.random.default_rng(42).standard_normal(mesh.dimension)
    np.testing.assert_allclose(
        matrix_free.apply(coefficients),
        dense.apply(coefficients),
        rtol=4.0e-12,
        atol=4.0e-12,
    )


@pytest.mark.verification
@pytest.mark.parametrize("degree", [1, 2])
def test_tetrahedral_dense_and_matrix_free_actions_agree(degree: int) -> None:
    mesh = RieszMeshData.build(
        TETRAHEDRON_COORDINATES,
        TETRAHEDRON_CELLS,
        dof_coordinates=(
            None if degree == 1 else TETRAHEDRON_CG2_COORDINATES
        ),
        cell_dofs=None if degree == 1 else TETRAHEDRON_CG2_CELLS,
        degree=degree,
    )
    basis = mesh.local_basis[0]
    nodal_values = np.asarray(
        [[polynomial(point) for polynomial in basis] for point in mesh.dof_coordinates]
    )
    np.testing.assert_allclose(nodal_values, np.eye(mesh.dimension), atol=3.0e-14)
    rule = face_tetrahedron_quadrature(
        2,
        0.3,
        zero_trace=False,
        field_degree=degree,
    )
    coefficients = np.random.default_rng(42).standard_normal(mesh.dimension)
    dense = DenseRieszBackend(mesh, 0.3, rule).apply(coefficients)
    matrix_free = MatrixFreeRieszBackend(mesh, 0.3, rule).apply(coefficients)
    np.testing.assert_allclose(matrix_free, dense, rtol=2.0e-12, atol=2.0e-12)
    assert float(coefficients @ dense) > 0.0


@pytest.mark.verification
@pytest.mark.parametrize("order", [0.1, 0.49, 0.8])
def test_matrix_free_fast_path_matches_public_oracle(order: float) -> None:
    mesh = RieszMeshData.build(COORDINATES, CELLS)
    degree = 4
    coefficients = np.array([0.2, 1.0, -0.3, 0.4])
    actual = MatrixFreeRieszBackend(
        mesh,
        order,
        triangle_quadrature(degree),
    ).apply(coefficients)
    expected = public_oracle_action(mesh, coefficients, order, degree)
    np.testing.assert_allclose(actual, expected, rtol=2.0e-14, atol=2.0e-14)


@pytest.mark.verification
def test_dense_form_is_positive_and_asymmetry_refines() -> None:
    mesh = RieszMeshData.build(COORDINATES, CELLS)
    coarse = DenseRieszBackend(mesh, 0.3, triangle_quadrature(2))
    fine = DenseRieszBackend(mesh, 0.3, triangle_quadrature(6))
    coarse.assemble()
    fine_matrix = fine.assemble()
    vector = np.array([0.3, -0.7, 1.1, 0.2])
    assert float(vector @ fine_matrix @ vector) > 0.0
    coarse_asymmetry = float(coarse.diagnostics()["relative_asymmetry"])
    fine_asymmetry = float(fine.diagnostics()["relative_asymmetry"])
    assert fine_asymmetry < coarse_asymmetry
    assert np.linalg.norm(fine_matrix - fine_matrix.T) < 0.03 * np.linalg.norm(
        fine_matrix
    )


@pytest.mark.verification
@pytest.mark.slow
def test_dense_entries_converge_to_independent_symmetric_energy() -> None:
    mesh = RieszMeshData.build(COORDINATES, CELLS)
    dense = DenseRieszBackend(mesh, 0.25, triangle_quadrature(8)).assemble()
    reference = unit_square_symmetric_energy_matrix(
        mesh,
        0.25,
        triangle_quadrature(14),
    )
    relative_error = np.linalg.norm(dense - reference) / np.linalg.norm(reference)
    assert relative_error < 5.0e-2


@pytest.mark.verification
@pytest.mark.slow
def test_cg2_dense_entries_converge_to_independent_symmetric_energy() -> None:
    mesh = RieszMeshData.build(
        COORDINATES,
        CELLS,
        dof_coordinates=CG2_COORDINATES,
        cell_dofs=CG2_CELLS,
        degree=2,
    )
    dense = DenseRieszBackend(mesh, 0.25, triangle_quadrature(8)).assemble()
    reference = unit_square_symmetric_energy_matrix(
        mesh,
        0.25,
        triangle_quadrature(14),
    )
    relative_error = np.linalg.norm(dense - reference) / np.linalg.norm(reference)
    assert relative_error < 7.0e-2


@pytest.mark.unit
def test_quadrature_validation() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        triangle_quadrature(0)
