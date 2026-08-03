"""Shared Firedrake integration utilities."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Set
from typing import Any


class ExternalOperatorData(dict[str, Any]):
    """Mutable external-operator data with identity equality."""

    def __eq__(self, other: object) -> bool:
        return self is other


def require_real_float64_petsc() -> None:
    """Reject unsupported PETSc scalar builds."""
    import numpy as np
    from petsc4py import PETSc

    if np.dtype(PETSc.ScalarType) != np.dtype(np.float64):
        raise NotImplementedError(
            "Yonderdrake requires a real binary64 PETSc scalar build"
        )


def validate_exterior_bcs(
    space: Any,
    bcs: Any,
    *,
    required: bool,
    operator_name: str,
) -> tuple[Any, ...]:
    """Validate complete homogeneous exterior boundary conditions."""
    import firedrake as fd
    import numpy as np

    if bcs is None:
        if required:
            raise ValueError(
                f"bcs is required for {operator_name} at this fractional order"
            )
        return ()
    values = tuple(bcs) if isinstance(bcs, (tuple, list)) else (bcs,)
    if not values:
        raise ValueError("bcs must contain at least one homogeneous DirichletBC")
    for bc in values:
        if not hasattr(bc, "function_space") or not hasattr(
            bc, "function_arg"
        ):
            raise TypeError("bcs must contain Firedrake DirichletBC objects")
        if bc.function_space() != space:
            raise ValueError("every boundary condition must act on u's space")
        value = bc.function_arg
        if hasattr(value, "dat"):
            is_zero = bool(np.all(np.asarray(value.dat.data_ro) == 0.0))
        else:
            try:
                is_zero = float(value) == 0.0
            except (TypeError, ValueError):
                is_zero = False
        if not is_zero:
            raise ValueError("boundary conditions must be homogeneous")
    full_nodes = np.unique(fd.DirichletBC(space, 0.0, "on_boundary").nodes)
    supplied_nodes = np.unique(
        np.concatenate([np.asarray(bc.nodes, dtype=np.int64) for bc in values])
    )
    if full_nodes.size == 0 or not np.array_equal(supplied_nodes, full_nodes):
        raise ValueError(
            "bcs must collectively cover the complete exterior boundary"
        )
    return values


def allocate_external_operator_matrix(
    operator: Any,
    *,
    bcs: tuple[Any, ...] = (),
    assembly_opts: Mapping[str, Any] | None = None,
    integral_types: Set[str] = frozenset({"cell"}),
) -> Any:
    """Allocate an external-operator matrix."""
    builder = getattr(operator, "_matrix_builder", None)
    if builder is None:
        raise RuntimeError(
            "This Firedrake version has no compatible external-operator matrix builder"
        )
    return builder(bcs, dict(assembly_opts or {}), set(integral_types))


def assemble_linear_action_adjoint(
    space: Any,
    action: Callable[[Any], Any],
    covector: Any,
    *,
    basis_name: str,
) -> Any:
    """Apply the adjoint of a linear primal action through basis probing."""
    from firedrake import Cofunction, Function

    result = Cofunction(space.dual())
    basis = Function(space, name=basis_name)
    with covector.dat.vec_ro as source, result.dat.vec as target:
        target_start, target_end = target.getOwnershipRange()
        for column in range(target.getSize()):
            basis.assign(0.0)
            with basis.dat.vec as basis_vector:
                basis_start, basis_end = basis_vector.getOwnershipRange()
                if basis_start <= column < basis_end:
                    basis_vector.setValue(column, 1.0)
                basis_vector.assemble()
            image = action(basis)
            with image.dat.vec_ro as image_vector:
                value = image_vector.dot(source)
            if target_start <= column < target_end:
                target.setValue(column, value)
        target.assemble()
    return result


def assemble_linear_action_matrix(
    operator: Any,
    action: Callable[[Any], Any],
    *,
    assembly_opts: Mapping[str, Any] | None,
    basis_name: str,
) -> Any:
    """Probe a linear action columnwise; Riesz assembly is O(N^3)."""
    import numpy as np
    from firedrake import Function

    matrix = allocate_external_operator_matrix(
        operator,
        assembly_opts=assembly_opts,
    )
    petsc_matrix = matrix.petscmat
    petsc_matrix.setOption(
        petsc_matrix.Option.NEW_NONZERO_ALLOCATION_ERR,
        False,
    )
    petsc_matrix.setOption(
        petsc_matrix.Option.UNUSED_NONZERO_LOCATION_ERR,
        False,
    )
    row_start, row_end = petsc_matrix.getOwnershipRange()
    rows = list(range(row_start, row_end))
    basis = Function(operator.function_space(), name=basis_name)
    for column in range(petsc_matrix.getSize()[1]):
        basis.assign(0.0)
        with basis.dat.vec as basis_vector:
            basis_start, basis_end = basis_vector.getOwnershipRange()
            if basis_start <= column < basis_end:
                basis_vector.setValue(column, 1.0)
            basis_vector.assemble()
        image = action(basis)
        with image.dat.vec_ro as image_vector:
            values = np.asarray(image_vector.array_r).copy().reshape(-1, 1)
        petsc_matrix.setValues(rows, [column], values)
    petsc_matrix.assemble()
    return matrix
