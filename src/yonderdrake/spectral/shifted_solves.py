"""Cached distributed shifted solves for the spectral operator."""

from __future__ import annotations

from math import exp
from typing import Any


class ShiftedSolveManager:
    """Apply a sinc approximation with cached or streamed shifted solves."""

    def __init__(
        self,
        space: Any,
        bcs: tuple[Any, ...],
        quadrature: Any,
        shift_cache: str,
        solver_parameters: dict[str, Any],
        mass_solver_parameters: dict[str, Any],
    ) -> None:
        import firedrake as fd

        self._fd = fd
        self.space = space
        self.bcs = bcs
        self.quadrature = quadrature
        self.shift_cache = shift_cache
        self.solver_parameters = dict(solver_parameters)
        self.mass_solver_parameters = dict(mass_solver_parameters)
        trial = fd.TrialFunction(space)
        test = fd.TestFunction(space)
        self._mass_form = fd.inner(trial, test) * fd.dx
        self._stiffness_form = fd.inner(fd.grad(trial), fd.grad(test)) * fd.dx
        self._mass_matrix = fd.assemble(self._mass_form, bcs=bcs)
        self._natural_neumann = not bcs
        self._minimum_neumann_shift = 1.0e-12
        self._volume = float(
            fd.assemble(fd.Constant(1.0) * fd.dx(domain=space.mesh()))
        )
        # Reused action buffers.
        self._source = fd.Function(space, name="spectral_source")
        self._mass_action_form = fd.inner(self._source, test) * fd.dx
        self._stiffness_action_form = (
            fd.inner(fd.grad(self._source), fd.grad(test)) * fd.dx
        )
        self._mass_rhs = fd.Cofunction(space.dual())
        self._stiffness_rhs = fd.Cofunction(space.dual())
        self._mass_solver: Any = None
        self._solvers: list[Any] = []
        self._stream_data: tuple[
            tuple[Any, Any, Any, Any],
            tuple[Any, Any, Any, Any],
        ] | None = None
        if shift_cache == "all":
            for log_shift in quadrature.log_shifts:
                y = float(log_shift)
                if y <= 0.0:
                    coefficient = fd.Constant(exp(y))
                    if (
                        self._natural_neumann
                        and float(coefficient) < self._minimum_neumann_shift
                    ):
                        self._solvers.append(None)
                        continue
                    form = self._stiffness_form + coefficient * self._mass_form
                else:
                    coefficient = fd.Constant(exp(-y))
                    form = self._mass_form + coefficient * self._stiffness_form
                matrix = fd.assemble(form, bcs=bcs)
                self._solvers.append(
                    fd.LinearSolver(
                        matrix,
                        solver_parameters=self.solver_parameters,
                    )
                )
        else:
            negative_coefficient = fd.Constant(1.0)
            negative_form = (
                self._stiffness_form
                + negative_coefficient * self._mass_form
            )
            negative_matrix = fd.assemble(negative_form, bcs=bcs)
            negative_solver = fd.LinearSolver(
                negative_matrix,
                solver_parameters=self.solver_parameters,
            )
            positive_coefficient = fd.Constant(1.0)
            positive_form = (
                self._mass_form
                + positive_coefficient * self._stiffness_form
            )
            positive_matrix = fd.assemble(positive_form, bcs=bcs)
            positive_solver = fd.LinearSolver(
                positive_matrix,
                solver_parameters=self.solver_parameters,
            )
            self._stream_data = (
                (
                    negative_coefficient,
                    negative_form,
                    negative_matrix,
                    negative_solver,
                ),
                (
                    positive_coefficient,
                    positive_form,
                    positive_matrix,
                    positive_solver,
                ),
            )
        self._work = fd.Function(space, name="spectral_shift_work")
        self._term = fd.Function(space, name="spectral_integrand")
        self.apply_count = 0
        self.shift_solve_count = 0
        self.matrix_assembly_count = (
            sum(solver is not None for solver in self._solvers) + 1
            if self._stream_data is None
            else 3
        )

    def _zero_boundary(self, field: Any) -> None:
        for bc in self.bcs:
            bc.zero(field)

    def _stream_solver(self, y: float) -> Any:
        if self._stream_data is None:
            raise RuntimeError("streaming solver data is unavailable")
        branch = 0 if y <= 0.0 else 1
        coefficient, form, matrix, solver = self._stream_data[branch]
        coefficient.assign(exp(y) if y <= 0.0 else exp(-y))
        self._fd.assemble(form, tensor=matrix, bcs=self.bcs)
        self.matrix_assembly_count += 1
        return solver

    def apply(self, operand: Any) -> Any:
        fd = self._fd
        try:
            self._source.assign(operand)
        except NotImplementedError:
            self._source.interpolate(operand)
        self._zero_boundary(self._source)
        if self._natural_neumann:
            mean = float(fd.assemble(self._source * fd.dx)) / self._volume
            self._source -= mean
        fd.assemble(self._mass_action_form, tensor=self._mass_rhs)
        fd.assemble(self._stiffness_action_form, tensor=self._stiffness_rhs)
        self._zero_boundary(self._mass_rhs)
        self._zero_boundary(self._stiffness_rhs)
        result = fd.Function(self.space, name="spectral_fractional_action")
        result.assign(0.0)

        nodes = zip(
            self.quadrature.log_shifts,
            self.quadrature.weights,
            strict=True,
        )
        for index, (y_value, weight) in enumerate(nodes):
            y = float(y_value)
            if y <= 0.0:
                shift = exp(y)
                self._term.assign(self._source)
                if not (
                    self._natural_neumann
                    and shift < self._minimum_neumann_shift
                ):
                    solver = (
                        self._solvers[index]
                        if self._stream_data is None
                        else self._stream_solver(y)
                    )
                    solver.solve(self._work, self._mass_rhs)
                    self._term -= shift * self._work
                    self.shift_solve_count += 1
            else:
                solver = (
                    self._solvers[index]
                    if self._stream_data is None
                    else self._stream_solver(y)
                )
                solver.solve(self._work, self._stiffness_rhs)
                # The weight already includes exp(-y).
                self._term.assign(self._work)
                self.shift_solve_count += 1
            result += float(weight) * self._term
        self._zero_boundary(result)
        self.apply_count += 1
        return result

    def dual_to_primal(self, covector: Any) -> Any:
        fd = self._fd
        rhs = covector.copy(deepcopy=True)
        self._zero_boundary(rhs)
        result = fd.Function(self.space, name="spectral_adjoint_riesz")
        if self._mass_solver is None:
            self._mass_solver = fd.LinearSolver(
                self._mass_matrix,
                solver_parameters=self.mass_solver_parameters,
            )
        self._mass_solver.solve(result, rhs)
        return result

    def primal_to_dual(self, function: Any) -> Any:
        fd = self._fd
        test = fd.TestFunction(self.space)
        result = fd.assemble(fd.inner(function, test) * fd.dx)
        self._zero_boundary(result)
        return result

    def diagnostics(self) -> dict[str, Any]:
        return {
            "num_nodes": self.quadrature.num_nodes,
            "step": self.quadrature.step,
            "log_shifts": self.quadrature.log_shifts.copy(),
            "requested_truncation_target": (
                self.quadrature.truncation_target
            ),
            "effective_truncation_target": (
                self.quadrature.effective_truncation_target
            ),
            "estimated_model_error": (
                self.quadrature.estimated_model_error
            ),
            "shift_cache": self.shift_cache,
            "cached_shift_solvers": sum(
                solver is not None for solver in self._solvers
            ),
            "matrix_setups": (
                sum(solver is not None for solver in self._solvers) + 1
                if self._stream_data is None
                else 3
            ),
            "matrix_assemblies": self.matrix_assembly_count,
            "applications": self.apply_count,
            "shift_solves": self.shift_solve_count,
            "minimum_neumann_shift": (
                self._minimum_neumann_shift
                if self._natural_neumann
                else None
            ),
            "cache_reuses": (
                max(0, self.apply_count - 1)
                if self._stream_data is None
                else 0
            ),
            "mass_solver_parameters": dict(self.mass_solver_parameters),
        }
