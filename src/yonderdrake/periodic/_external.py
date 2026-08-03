"""Firedrake registrations for the periodic Fourier action."""

from __future__ import annotations

from typing import Any

import numpy as np
from firedrake import Function
from petsc4py import PETSc

from yonderdrake._external import LinearExternalOperator
from yonderdrake.periodic.grid import (
    DistributedPeriodicFourierBackend,
    PeriodicFourierBackend,
    PeriodicGridMap,
)


class PeriodicApplyManager:
    """Bridge serial or slab-distributed FFTs to Firedrake fields."""

    def __init__(self, space: Any, grid: PeriodicGridMap, order: float) -> None:
        import firedrake as fd

        self._fd = fd
        self.space = space
        self._comm = space.mesh().comm
        self._source = fd.Function(space, name="periodic_fourier_source")
        with self._source.dat.vec_ro as vector:
            ownership_range = vector.getOwnershipRange()
        self.backend = (
            PeriodicFourierBackend(grid, order)
            if self._comm.size == 1
            else DistributedPeriodicFourierBackend(
                grid,
                order,
                self._comm,
                ownership_range,
            )
        )

    def apply(self, operand: Any) -> Function:
        try:
            self._source.assign(operand)
        except NotImplementedError:
            self._source.interpolate(operand)
        with self._source.dat.vec_ro as vector:
            start, end = vector.getOwnershipRange()
            local_values = np.asarray(vector.array_r).copy()
        values = self.backend.apply(local_values)
        result = self._fd.Function(
            self.space,
            name="periodic_fractional_action",
        )
        with result.dat.vec as vector:
            start, end = vector.getOwnershipRange()
            indices = np.arange(start, end, dtype=PETSc.IntType)
            vector.setValues(
                indices,
                values,
                addv=PETSc.InsertMode.INSERT_VALUES,
            )
            vector.assemble()
        return result

    def diagnostics(self) -> dict[str, Any]:
        result = self.backend.diagnostics()
        result["ranks"] = self._comm.size
        return result


class PeriodicExternalOperator(LinearExternalOperator):
    """Primal nodal representation of the periodic Fourier multiplier."""

    basis_name = "periodic"

    def _build_manager(self) -> PeriodicApplyManager:
        return PeriodicApplyManager(
            self.function_space(),
            self.operator_data["grid"],
            self.operator_data["order"],
        )

    def diagnostics(self) -> dict[str, Any]:
        manager = self.operator_data.get("manager")
        if manager is not None:
            return manager.diagnostics()
        grid = self.operator_data["grid"]
        return {
            "shape": grid.shape,
            "lengths": grid.lengths,
            "spacing": grid.spacing,
            "fft_backend": (
                "numpy-serial"
                if self.function_space().mesh().comm.size == 1
                else "numpy-mpi-slab"
            ),
            "applications": 0,
            "ranks": self.function_space().mesh().comm.size,
        }
