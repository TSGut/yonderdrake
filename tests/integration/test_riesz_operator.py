"""Firedrake integration tests for all Riesz assembly backends."""

from __future__ import annotations

import numpy as np
import pytest

fd = pytest.importorskip("firedrake")

from yonderdrake import RieszFractionalLaplacian  # noqa: E402


def make_problem(assembly: str):
    mesh = fd.UnitSquareMesh(1, 1)
    space = fd.FunctionSpace(mesh, "CG", 1)
    x, y = fd.SpatialCoordinate(mesh)
    u = fd.Function(space).interpolate(0.2 + x - 0.3 * y)
    operator = RieszFractionalLaplacian(
        u,
        0.3,
        assembly=assembly,
        target_quadrature_degree=5,
        compression_tolerance=1.0e-8,
        admissibility=1.0,
        leaf_size=1,
    )
    return space, u, operator


@pytest.mark.verification
def test_riesz_dense_matfree_and_hmatrix_actions_agree() -> None:
    dense_space, dense_u, dense_operator = make_problem("dense")
    free_space, free_u, free_operator = make_problem("matfree")
    hierarchical_space, hierarchical_u, hierarchical_operator = make_problem("hmatrix")
    dense_action = fd.assemble(dense_operator)
    free_action = fd.assemble(free_operator)
    hierarchical_action = fd.assemble(hierarchical_operator)
    np.testing.assert_allclose(
        dense_action.dat.data_ro,
        free_action.dat.data_ro,
        rtol=3.0e-11,
        atol=3.0e-11,
    )
    np.testing.assert_allclose(
        hierarchical_action.dat.data_ro,
        dense_action.dat.data_ro,
        rtol=2.0e-7,
        atol=2.0e-8,
    )

    dense_test = fd.TestFunction(dense_space)
    weak = fd.assemble(fd.inner(dense_operator, dense_test) * fd.dx)
    weak_reference = fd.assemble(fd.inner(dense_action, dense_test) * fd.dx)
    with weak.dat.vec_ro as actual, weak_reference.dat.vec_ro as expected:
        difference = actual.copy()
        difference.axpy(-1.0, expected)
        assert difference.norm() < 2.0e-11

    assert dense_operator.diagnostics()["stored_entries"] == dense_space.dim() ** 2
    assert free_operator.diagnostics()["stored_entries"] == 0
    assert hierarchical_operator.diagnostics()["assembly"] == "hmatrix"
    assert dense_operator.diagnostics()["target_quadrature_rule"] == "boundary"
    assert dense_u.function_space().dim() == free_u.function_space().dim()
    assert hierarchical_u.function_space().dim() == dense_space.dim()
    assert hierarchical_space.dim() == dense_space.dim()


@pytest.mark.verification
def test_riesz_ordinary_quadrature_remains_selectable() -> None:
    mesh = fd.UnitSquareMesh(1, 1)
    space = fd.FunctionSpace(mesh, "CG", 1)
    operator = RieszFractionalLaplacian(
        fd.Function(space).assign(1.0),
        0.3,
        target_quadrature_degree=4,
        target_quadrature_rule="ordinary",
    )
    fd.assemble(operator)
    diagnostics = operator.diagnostics()
    assert diagnostics["target_quadrature_rule"] == "ordinary"
    assert diagnostics["target_quadrature_points_per_cell"] == 9


@pytest.mark.verification
def test_riesz_edge_quadrature_defaults_are_live() -> None:
    mesh = fd.UnitSquareMesh(1, 1)
    space = fd.FunctionSpace(mesh, "CG", 1)
    boundary = fd.DirichletBC(space, 0.0, "on_boundary")
    operator = RieszFractionalLaplacian(
        fd.Function(space),
        0.7,
        bcs=boundary,
    )
    fd.assemble(operator)
    diagnostics = operator.diagnostics()
    assert diagnostics["target_quadrature_rule"] == "boundary"
    assert diagnostics["target_quadrature_degree"] == 6
    assert diagnostics["target_quadrature_points_per_cell"] == 48


@pytest.mark.verification
def test_riesz_cg2_backends_agree() -> None:
    results = []
    for assembly in ("dense", "matfree", "hmatrix"):
        mesh = fd.UnitSquareMesh(1, 1)
        space = fd.FunctionSpace(mesh, "CG", 2)
        x, y = fd.SpatialCoordinate(mesh)
        u = fd.Function(space).interpolate(0.2 + x - 0.3 * y + 0.4 * x * y)
        operator = RieszFractionalLaplacian(
            u,
            0.3,
            assembly=assembly,
            target_quadrature_degree=6,
            compression_tolerance=1.0e-9,
            leaf_size=2,
        )
        results.append(fd.assemble(operator).dat.data_ro.copy())
    np.testing.assert_allclose(results[1], results[0], rtol=4.0e-11, atol=4.0e-11)
    np.testing.assert_allclose(results[2], results[0], rtol=2.0e-7, atol=2.0e-8)


@pytest.mark.verification
@pytest.mark.parametrize("degree", [1, 2])
def test_riesz_tetrahedral_cg1_and_cg2_fields(degree: int) -> None:
    mesh = fd.UnitCubeMesh(1, 1, 1)
    space = fd.FunctionSpace(mesh, "CG", degree)
    x, y, z = fd.SpatialCoordinate(mesh)
    source = fd.Function(space).interpolate(0.2 + x - 0.3 * y + 0.1 * z + 0.2 * x * y)
    operator = RieszFractionalLaplacian(
        source,
        0.3,
        assembly="matfree",
        target_quadrature_degree=1,
    )
    result = fd.assemble(operator)
    assert result.function_space() == space
    assert np.all(np.isfinite(result.dat.data_ro))
    diagnostics = operator.diagnostics()
    assert diagnostics["target_quadrature_rule"] == "boundary"
    assert diagnostics["assembly"] == "matfree"


@pytest.mark.verification
def test_riesz_tetrahedral_backends_agree() -> None:
    results = []
    for assembly in ("dense", "matfree", "hmatrix"):
        mesh = fd.UnitCubeMesh(1, 1, 1)
        space = fd.FunctionSpace(mesh, "CG", 1)
        x, y, z = fd.SpatialCoordinate(mesh)
        source = fd.Function(space).interpolate(0.2 + x - 0.3 * y + 0.1 * z)
        operator = RieszFractionalLaplacian(
            source,
            0.3,
            assembly=assembly,
            target_quadrature_degree=1,
            compression_tolerance=1.0e-8,
            leaf_size=2,
        )
        results.append(fd.assemble(operator).dat.data_ro.copy())
    np.testing.assert_allclose(results[1], results[0], rtol=3.0e-11, atol=3.0e-11)
    np.testing.assert_allclose(results[2], results[0], rtol=3.0e-11, atol=3.0e-11)


@pytest.mark.verification
def test_riesz_mass_solver_parameters_are_public() -> None:
    mesh = fd.UnitSquareMesh(1, 1)
    space = fd.FunctionSpace(mesh, "CG", 1)
    u = fd.Function(space).assign(1.0)
    parameters = {
        "ksp_type": "preonly",
        "pc_type": "lu",
    }
    operator = RieszFractionalLaplacian(
        u,
        0.3,
        target_quadrature_degree=3,
        mass_solver_parameters=parameters,
    )
    fd.assemble(operator)
    assert operator.diagnostics()["mass_solver_parameters"] == parameters


@pytest.mark.parametrize("assembly", ["dense", "matfree", "hmatrix"])
@pytest.mark.verification
def test_riesz_jacobian_action_matrix_and_adjoint(assembly: str) -> None:
    space, u, operator = make_problem(assembly)
    x, y = fd.SpatialCoordinate(space.mesh())
    direction = fd.Function(space).interpolate(0.1 + x * y)
    direction_operator = RieszFractionalLaplacian(
        direction,
        0.3,
        assembly=assembly,
        target_quadrature_degree=5,
    )
    expected = fd.assemble(direction_operator)
    jacobian = fd.derivative(operator, u, fd.TrialFunction(space))
    action = fd.assemble(fd.action(jacobian, direction))
    np.testing.assert_allclose(action.dat.data_ro, expected.dat.data_ro, rtol=2.0e-11)

    matrix = fd.assemble(jacobian)
    matrix_action = fd.Function(space)
    with direction.dat.vec_ro as source, matrix_action.dat.vec as target:
        matrix.petscmat.mult(source, target)
    np.testing.assert_allclose(
        matrix_action.dat.data_ro,
        expected.dat.data_ro,
        rtol=2.0e-11,
    )

    test = fd.TestFunction(space)
    covector = fd.assemble(fd.inner(direction, test) * fd.dx)
    adjoint_action = fd.assemble(fd.action(fd.adjoint(jacobian), covector))
    adjoint_reference = fd.Cofunction(space.dual())
    with covector.dat.vec_ro as source, adjoint_reference.dat.vec as target:
        matrix.petscmat.multTranspose(source, target)
    with (
        adjoint_action.dat.vec_ro as actual,
        adjoint_reference.dat.vec_ro as reference,
    ):
        difference = actual.copy()
        difference.axpy(-1.0, reference)
        assert difference.norm() < 2.0e-11


@pytest.mark.unit
def test_riesz_validation_is_immediate() -> None:
    with pytest.raises(TypeError, match="real scalar"):
        RieszFractionalLaplacian(object(), object())
    with pytest.raises(ValueError, match="0 < s < 1"):
        RieszFractionalLaplacian(object(), 0.0)
    with pytest.raises(ValueError, match="extension"):
        RieszFractionalLaplacian(object(), 0.4, extension="periodic")
    with pytest.raises(ValueError, match="assembly"):
        RieszFractionalLaplacian(object(), 0.4, assembly="sparse")
    with pytest.raises(ValueError, match="target_quadrature_rule"):
        RieszFractionalLaplacian(
            object(),
            0.4,
            target_quadrature_rule="adaptive",
        )
    for degree in (0, 1.5, True):
        with pytest.raises(ValueError, match="target_quadrature_degree"):
            RieszFractionalLaplacian(
                object(),
                0.4,
                target_quadrature_degree=degree,
            )
    with pytest.raises(TypeError, match="compression_tolerance"):
        RieszFractionalLaplacian(
            object(),
            0.4,
            compression_tolerance=object(),
        )
    with pytest.raises(ValueError, match="compression_tolerance"):
        RieszFractionalLaplacian(
            object(),
            0.4,
            compression_tolerance=0.0,
        )
    with pytest.raises(TypeError, match="admissibility"):
        RieszFractionalLaplacian(
            object(),
            0.4,
            admissibility=object(),
        )
    with pytest.raises(ValueError, match="admissibility"):
        RieszFractionalLaplacian(
            object(),
            0.4,
            admissibility=float("inf"),
        )
    with pytest.raises(ValueError, match="leaf_size"):
        RieszFractionalLaplacian(
            object(),
            0.4,
            leaf_size=True,
        )
    with pytest.raises(TypeError, match="mass_solver_parameters"):
        RieszFractionalLaplacian(
            object(),
            0.4,
            mass_solver_parameters=object(),
        )
    with pytest.raises(TypeError, match="Firedrake Function"):
        RieszFractionalLaplacian(object(), 0.4)

    base_mesh = fd.UnitSquareMesh(1, 1)
    coordinate_space = fd.VectorFunctionSpace(base_mesh, "CG", 2)
    x, y = fd.SpatialCoordinate(base_mesh)
    curved_coordinates = fd.Function(coordinate_space).interpolate(
        fd.as_vector((x, y + 0.1 * x * (1.0 - x)))
    )
    curved_mesh = fd.Mesh(curved_coordinates)
    curved_space = fd.FunctionSpace(curved_mesh, "CG", 1)
    with pytest.raises(NotImplementedError, match="degree-1 affine"):
        RieszFractionalLaplacian(fd.Function(curved_space), 0.4)

    space = fd.FunctionSpace(base_mesh, "CG", 1)
    u = fd.Function(space)
    vector = fd.Function(fd.VectorFunctionSpace(base_mesh, "CG", 1))
    with pytest.raises(NotImplementedError, match="scalar"):
        RieszFractionalLaplacian(vector, 0.4)
    with pytest.raises(NotImplementedError, match="degree 1 or 2"):
        RieszFractionalLaplacian(
            fd.Function(fd.FunctionSpace(base_mesh, "CG", 3)),
            0.4,
        )
    with pytest.raises(ValueError, match="bcs is required"):
        RieszFractionalLaplacian(u, 0.5)
    partial = fd.DirichletBC(space, 0.0, 1)
    with pytest.raises(ValueError, match="complete exterior boundary"):
        RieszFractionalLaplacian(u, 0.5, bcs=partial)
    nonzero = fd.DirichletBC(space, 1.0, "on_boundary")
    with pytest.raises(ValueError, match="homogeneous"):
        RieszFractionalLaplacian(u, 0.5, bcs=nonzero)

    interval_space = fd.FunctionSpace(fd.UnitIntervalMesh(1), "CG", 1)
    with pytest.raises(NotImplementedError, match="2D triangle"):
        RieszFractionalLaplacian(fd.Function(interval_space), 0.4)


@pytest.mark.unit
def test_riesz_operator_data_has_identity_semantics() -> None:
    mesh = fd.UnitSquareMesh(1, 1)
    space = fd.FunctionSpace(mesh, "CG", 1)
    u = fd.Function(space)
    left = RieszFractionalLaplacian(u, 0.3)
    right = RieszFractionalLaplacian(u, 0.3)
    assert left != right
    assert len({left, right}) == 2


@pytest.mark.verification
def test_riesz_cache_invalidation_and_order_immutability() -> None:
    mesh = fd.UnitSquareMesh(1, 1)
    space = fd.FunctionSpace(mesh, "CG", 1)
    order = fd.Constant(0.3)
    operator = RieszFractionalLaplacian(
        fd.Function(space).assign(1.0),
        order,
        target_quadrature_degree=2,
    )
    assert operator.diagnostics()["applications"] == 0
    fd.assemble(operator)
    first_manager = operator.operator_data["manager"]
    operator.invalidate_cache()
    assert operator.diagnostics()["applications"] == 0
    fd.assemble(operator)
    assert operator.operator_data["manager"] is not first_manager

    order.assign(0.4)
    with pytest.raises(RuntimeError, match="changing s"):
        fd.assemble(operator)


@pytest.mark.verification
@pytest.mark.parametrize("degree", [1, 2])
def test_riesz_complete_boundary_conditions_define_high_order_domain(
    degree: int,
) -> None:
    mesh = fd.UnitSquareMesh(2, 2)
    space = fd.FunctionSpace(mesh, "CG", degree)
    x, y = fd.SpatialCoordinate(mesh)
    u = fd.Function(space).interpolate(x * (1.0 - x) * y * (1.0 - y))
    bc = fd.DirichletBC(space, 0.0, "on_boundary")
    operator = RieszFractionalLaplacian(
        u,
        0.75,
        bcs=bc,
        target_quadrature_degree=3,
    )
    result = fd.assemble(operator)
    np.testing.assert_allclose(result.dat.data_ro[bc.nodes], 0.0)


@pytest.mark.verification
@pytest.mark.parametrize("degree", [1, 2])
def test_riesz_tetrahedral_boundary_conditions_define_high_order_domain(
    degree: int,
) -> None:
    mesh = fd.UnitCubeMesh(2, 2, 2)
    space = fd.FunctionSpace(mesh, "CG", degree)
    x, y, z = fd.SpatialCoordinate(mesh)
    source = fd.Function(space).interpolate(
        x * (1.0 - x) * y * (1.0 - y) * z * (1.0 - z)
    )
    bc = fd.DirichletBC(space, 0.0, "on_boundary")
    operator = RieszFractionalLaplacian(
        source,
        0.7,
        bcs=bc,
        target_quadrature_degree=1,
    )
    result = fd.assemble(operator)
    assert np.all(np.isfinite(result.dat.data_ro))
    np.testing.assert_allclose(result.dat.data_ro[bc.nodes], 0.0)


@pytest.mark.unit
def test_riesz_rejects_periodic_geometry() -> None:
    meshes = (
        fd.PeriodicUnitSquareMesh(2, 2, quadrilateral=False),
        fd.PeriodicRectangleMesh(
            2,
            2,
            1.0,
            1.0,
            direction="x",
            quadrilateral=False,
        ),
    )
    for mesh in meshes:
        space = fd.FunctionSpace(mesh, "CG", 1)
        operator = RieszFractionalLaplacian(fd.Function(space), 0.3)
        with pytest.raises(ValueError, match="periodic meshes"):
            fd.assemble(operator)
