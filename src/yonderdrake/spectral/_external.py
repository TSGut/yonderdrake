"""Firedrake external-operator registrations for the spectral action."""

from __future__ import annotations

from typing import Any

from firedrake import Cofunction, assemble_method

from yonderdrake._external import LinearExternalOperator
from yonderdrake.spectral.shifted_solves import ShiftedSolveManager


class SpectralExternalOperator(LinearExternalOperator):
    """Primal L2 representation of the discrete spectral fractional power."""

    basis_name = "spectral"

    def _build_manager(self) -> ShiftedSolveManager:
        return ShiftedSolveManager(
            self.function_space(),
            self.operator_data["bcs"],
            self.operator_data["quadrature"],
            self.operator_data["shift_cache"],
            self.operator_data["shift_solver_parameters"],
            self.operator_data["mass_solver_parameters"],
        )

    def diagnostics(self) -> dict[str, Any]:
        manager = self.operator_data.get("manager")
        if manager is None:
            quadrature = self.operator_data["quadrature"]
            return {
                "num_nodes": quadrature.num_nodes,
                "step": quadrature.step,
                "log_shifts": quadrature.log_shifts.copy(),
                "requested_truncation_target": (
                    quadrature.truncation_target
                ),
                "effective_truncation_target": (
                    quadrature.effective_truncation_target
                ),
                "estimated_model_error": quadrature.estimated_model_error,
                "shift_cache": self.operator_data["shift_cache"],
                "cached_shift_solvers": 0,
                "matrix_setups": 0,
                "matrix_assemblies": 0,
                "applications": 0,
                "shift_solves": 0,
                "cache_reuses": 0,
                "mass_solver_parameters": dict(
                    self.operator_data["mass_solver_parameters"]
                ),
            }
        return manager.diagnostics()

    @assemble_method(1, (None, 0))
    def assemble_jacobian_adjoint_action(
        self,
        *,
        assembly_opts: Any = None,
    ) -> Cofunction:
        del assembly_opts
        manager = self._manager()
        return manager.primal_to_dual(
            manager.apply(
                manager.dual_to_primal(self.argument_slots()[0]),
            )
        )
