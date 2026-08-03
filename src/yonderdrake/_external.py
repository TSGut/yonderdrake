"""Common Firedrake registration for linear external operators."""

from __future__ import annotations

from typing import Any

from firedrake import (
    AbstractExternalOperator,
    Cofunction,
    Function,
    assemble_method,
)

from yonderdrake._firedrake import (
    assemble_linear_action_adjoint,
    assemble_linear_action_matrix,
)


class LinearExternalOperator(AbstractExternalOperator):
    """Shared registrations for a cached linear primal action."""

    basis_name = "external_operator"

    def _build_manager(self) -> Any:
        raise NotImplementedError

    def _manager(self) -> Any:
        current = self.operator_data.get("manager")
        if current is None:
            current = self._build_manager()
            self.operator_data["manager"] = current
        if float(self.operator_data["order_operand"]) != self.operator_data["order"]:
            raise RuntimeError(
                "changing s after construction is unsupported; rebuild the operator"
            )
        return current

    def invalidate_cache(self) -> None:
        self.operator_data["manager"] = None

    @assemble_method(0, (0,))
    def assemble_forward(self, *, assembly_opts: Any = None) -> Function:
        del assembly_opts
        return self._manager().apply(self.ufl_operands[0])

    @assemble_method(1, (0, None))
    def assemble_jacobian_action(
        self,
        *,
        assembly_opts: Any = None,
    ) -> Function:
        del assembly_opts
        return self._manager().apply(self.argument_slots()[-1])

    @assemble_method(1, (None, 0))
    def assemble_jacobian_adjoint_action(
        self,
        *,
        assembly_opts: Any = None,
    ) -> Cofunction:
        del assembly_opts
        manager = self._manager()
        return assemble_linear_action_adjoint(
            self.function_space(),
            manager.apply,
            self.argument_slots()[0],
            basis_name=f"{self.basis_name}_adjoint_basis",
        )

    @assemble_method(1, (0, 1))
    @assemble_method(1, (1, 0))
    def assemble_jacobian(self, *, assembly_opts: Any = None) -> Any:
        manager = self._manager()
        return assemble_linear_action_matrix(
            self,
            manager.apply,
            assembly_opts=assembly_opts,
            basis_name=f"{self.basis_name}_basis",
        )
