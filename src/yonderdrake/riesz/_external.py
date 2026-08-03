"""Firedrake registrations for the zero-exterior Riesz operator."""

from __future__ import annotations

from typing import Any

import numpy as np
from petsc4py import PETSc

from yonderdrake._external import LinearExternalOperator
from yonderdrake.riesz.dense import (
    DenseRieszBackend,
    RieszMeshData,
    local_polynomial_basis,
)
from yonderdrake.riesz.distributed_hmatrix import (
    DistributedHierarchicalRieszBackend,
    distribute_dofs,
    validate_geometry,
)
from yonderdrake.riesz.geometry import TetrahedronGeometry, TriangleGeometry
from yonderdrake.riesz.hmatrix import HierarchicalRieszBackend
from yonderdrake.riesz.matfree import MatrixFreeRieszBackend
from yonderdrake.riesz.outer_quadrature import (
    edge_triangle_quadrature,
    face_tetrahedron_quadrature,
    tetrahedron_quadrature,
    triangle_quadrature,
)
from yonderdrake.riesz.triangle_action import SimplexPiece


class RieszApplyManager:
    """Bridge Riesz Galerkin backends to primal Firedrake fields."""

    def __init__(
        self,
        space: Any,
        order: float,
        quadrature_degree: int,
        quadrature_rule: str,
        assembly: str,
        compression_tolerance: float,
        admissibility: float,
        leaf_size: int,
        bcs: tuple[Any, ...],
        mass_solver_parameters: dict[str, Any],
    ) -> None:
        import firedrake as fd

        self._fd = fd
        self.space = space
        self.bcs = bcs
        mesh = space.mesh()
        dimension = int(mesh.geometric_dimension)
        self._comm = mesh.comm
        self._source = fd.Function(space, name="riesz_source")
        degree = int(space.ufl_element().degree())
        vertex_space = fd.FunctionSpace(mesh, "CG", 1)
        vertex_local_cells = np.asarray(
            vertex_space.cell_node_map().values,
            dtype=np.int64,
        )
        vertex_global_cells = np.asarray(
            vertex_space.dof_dset.lgmap.indices
        )[vertex_local_cells]
        vertex_values = mesh.coordinates.dat.data_ro_with_halos
        dof_coordinate_space = fd.VectorFunctionSpace(mesh, "CG", degree)
        dof_coordinates = fd.Function(dof_coordinate_space).interpolate(
            fd.SpatialCoordinate(mesh)
        )
        field_local_cells = np.asarray(
            space.cell_node_map().values,
            dtype=np.int64,
        )
        field_global_cells = np.asarray(
            space.dof_dset.lgmap.indices
        )[field_local_cells]
        dof_coordinate_values = dof_coordinates.dat.data_ro_with_halos
        owned_cells = int(mesh.cell_set.size)
        vertex_local_cells = vertex_local_cells[:owned_cells]
        vertex_global_cells = vertex_global_cells[:owned_cells]
        field_local_cells = field_local_cells[:owned_cells]
        field_global_cells = field_global_cells[:owned_cells]
        local_records = [
            (
                tuple(int(index) for index in vertex_global_cell),
                np.asarray(vertex_values[vertex_local_cell]).copy(),
                tuple(int(index) for index in field_global_cell),
                np.asarray(dof_coordinate_values[field_local_cell]).copy(),
            )
            for (
                vertex_local_cell,
                vertex_global_cell,
                field_local_cell,
                field_global_cell,
            ) in zip(
                vertex_local_cells,
                vertex_global_cells,
                field_local_cells,
                field_global_cells,
                strict=True,
            )
        ]
        if dimension == 2:
            quadrature = (
                edge_triangle_quadrature(
                    quadrature_degree,
                    order,
                    zero_trace=bool(bcs),
                    field_degree=degree,
                )
                if quadrature_rule == "boundary"
                else triangle_quadrature(quadrature_degree)
            )
            geometry_type = TriangleGeometry
        else:
            quadrature = (
                face_tetrahedron_quadrature(
                    quadrature_degree,
                    order,
                    zero_trace=bool(bcs),
                    field_degree=degree,
                )
                if quadrature_rule == "boundary"
                else tetrahedron_quadrature(quadrature_degree)
            )
            geometry_type = TetrahedronGeometry
        self._distributed_hmatrix = (
            assembly == "hmatrix" and mesh.comm.size > 1
        )
        if self._distributed_hmatrix:
            validate_geometry(
                mesh.comm,
                [record[1] for record in local_records],
            )
            with self._source.dat.vec_ro as vector:
                local_range = vector.getOwnershipRange()
            ownership_ranges = tuple(mesh.comm.allgather(local_range))
            contributions = []
            for (
                _geometry_cell,
                cell_vertices,
                field_cell,
                cell_dof_coordinates,
            ) in local_records:
                geometry = geometry_type.from_vertices(cell_vertices)
                basis = local_polynomial_basis(
                    cell_dof_coordinates,
                    degree,
                )
                for index, coordinate, polynomial in zip(
                    field_cell,
                    cell_dof_coordinates,
                    basis,
                    strict=True,
                ):
                    contributions.append(
                        (
                            int(index),
                            np.asarray(coordinate).copy(),
                            SimplexPiece(geometry, polynomial),
                        )
                    )
            local_dofs = distribute_dofs(
                mesh.comm,
                ownership_ranges,
                contributions,
                quadrature,
            )
            self.backend = DistributedHierarchicalRieszBackend(
                mesh.comm,
                local_dofs,
                order,
                quadrature,
                compression_tolerance=compression_tolerance,
                admissibility=admissibility,
                leaf_size=leaf_size,
            )
        else:
            records_by_rank = mesh.comm.allgather(local_records)
            records = [
                record
                for rank_records in records_by_rank
                for record in rank_records
            ]
            global_vertices = np.zeros(
                (vertex_space.dim(), dimension),
                dtype=np.float64,
            )
            global_dof_coordinates = np.zeros(
                (space.dim(), dimension),
                dtype=np.float64,
            )
            geometry_cells = []
            field_cells = []
            for (
                geometry_cell,
                cell_vertices,
                field_cell,
                cell_dof_coordinates,
            ) in records:
                geometry_cells.append(geometry_cell)
                field_cells.append(field_cell)
                global_vertices[
                    np.asarray(geometry_cell, dtype=np.int64)
                ] = cell_vertices
                global_dof_coordinates[
                    np.asarray(field_cell, dtype=np.int64)
                ] = cell_dof_coordinates
            mesh_data = RieszMeshData.build(
                global_vertices,
                geometry_cells,
                dof_coordinates=global_dof_coordinates,
                cell_dofs=field_cells,
                degree=degree,
            )
            target_start = sum(
                len(records_by_rank[rank])
                for rank in range(mesh.comm.rank)
            )
            self._target_cells = range(
                target_start,
                target_start + len(local_records),
            )
            if assembly == "dense":
                self.backend = DenseRieszBackend(
                    mesh_data,
                    order,
                    quadrature,
                )
            elif assembly == "matfree":
                self.backend = MatrixFreeRieszBackend(
                    mesh_data,
                    order,
                    quadrature,
                )
            else:
                self.backend = HierarchicalRieszBackend(
                    mesh_data,
                    order,
                    quadrature,
                    compression_tolerance=compression_tolerance,
                    admissibility=admissibility,
                    leaf_size=leaf_size,
                )
        trial = fd.TrialFunction(space)
        test = fd.TestFunction(space)
        mass = fd.assemble(fd.inner(trial, test) * fd.dx, bcs=bcs)
        self._mass_solver = fd.LinearSolver(
            mass,
            solver_parameters=mass_solver_parameters,
        )
        self.mass_solver_parameters = dict(mass_solver_parameters)
        self.riesz_solve_count = 0

    def _zero_boundary(self, field: Any) -> None:
        for bc in self.bcs:
            bc.zero(field)

    def weak_apply(self, operand: Any) -> Any:
        fd = self._fd
        try:
            self._source.assign(operand)
        except NotImplementedError:
            self._source.interpolate(operand)
        self._zero_boundary(self._source)
        with self._source.dat.vec_ro as vector:
            start, end = vector.getOwnershipRange()
            local_values = np.asarray(vector.array_r).copy()
        if self._distributed_hmatrix:
            values = self.backend.apply_local(local_values)
            result = fd.Cofunction(self.space.dual())
            with result.dat.vec as vector:
                vector.array[:] = values
                vector.assemble()
            self._zero_boundary(result)
            return result
        coefficient_parts = self._comm.allgather(
            (start, end, local_values)
        )
        coefficients = np.empty(self.space.dim(), dtype=np.float64)
        for part_start, part_end, part_values in coefficient_parts:
            coefficients[part_start:part_end] = part_values
        if self._comm.size == 1 or isinstance(
            self.backend,
            DenseRieszBackend,
        ):
            values = self.backend.apply(coefficients)
        else:
            local_values = self.backend.apply_owned(
                coefficients,
                self._target_cells,
            )
            values = np.empty_like(local_values)
            self._comm.Allreduce(local_values, values)
        result = fd.Cofunction(self.space.dual())
        with result.dat.vec as vector:
            start, end = vector.getOwnershipRange()
            indices = np.arange(start, end, dtype=PETSc.IntType)
            vector.setValues(
                indices,
                values[start:end],
                addv=PETSc.InsertMode.INSERT_VALUES,
            )
            vector.assemble()
        self._zero_boundary(result)
        return result

    def apply(self, operand: Any) -> Any:
        weak = self.weak_apply(operand)
        result = self._fd.Function(self.space, name="riesz_fractional_action")
        self._mass_solver.solve(result, weak)
        self._zero_boundary(result)
        self.riesz_solve_count += 1
        return result

    def diagnostics(self) -> dict[str, Any]:
        result = dict(self.backend.diagnostics())
        result["riesz_mass_solves"] = self.riesz_solve_count
        result["mass_solver_parameters"] = dict(self.mass_solver_parameters)
        return result


class RieszExternalOperator(LinearExternalOperator):
    """Primal L2 representation of a selected weak Riesz backend."""

    basis_name = "riesz"

    def _build_manager(self) -> RieszApplyManager:
        return RieszApplyManager(
            self.function_space(),
            self.operator_data["order"],
            self.operator_data["quadrature_degree"],
            self.operator_data["quadrature_rule"],
            self.operator_data["assembly"],
            self.operator_data["compression_tolerance"],
            self.operator_data["admissibility"],
            self.operator_data["leaf_size"],
            self.operator_data["bcs"],
            self.operator_data["mass_solver_parameters"],
        )

    def diagnostics(self) -> dict[str, Any]:
        manager = self.operator_data.get("manager")
        if manager is None:
            return {
                "assembly": self.operator_data["assembly"],
                "quadrature_degree": self.operator_data["quadrature_degree"],
                "quadrature_rule": self.operator_data["quadrature_rule"],
                "compression_tolerance": self.operator_data["compression_tolerance"],
                "admissibility": self.operator_data["admissibility"],
                "leaf_size": self.operator_data["leaf_size"],
                "mass_solver_parameters": dict(
                    self.operator_data["mass_solver_parameters"]
                ),
                "applications": 0,
                "riesz_mass_solves": 0,
            }
        return manager.diagnostics()
