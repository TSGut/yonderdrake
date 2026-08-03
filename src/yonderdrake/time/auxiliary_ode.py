"""Native monolithic auxiliary-ODE time-memory formulation."""

from __future__ import annotations

from math import gamma, isfinite
from typing import Any

import numpy as np

from yonderdrake._firedrake import require_real_float64_petsc
from yonderdrake.time._stepper_lifecycle import StepperLifecycle
from yonderdrake.time.checkpointing import (
    load_checkpoint_file,
    save_checkpoint_file,
    stepper_metadata,
    validate_stepper_metadata,
)
from yonderdrake.time.coefficients import validate_recurrence_interval
from yonderdrake.time.formulations import AuxiliaryODE
from yonderdrake.time.representations import (
    _SingleExponential,
    validate_checkpoint_representation,
)


class AuxiliaryODEStepper(StepperLifecycle):
    """Advance a mixed physical-plus-diffusive-mode Firedrake state."""

    def __init__(
        self,
        F: Any,
        representation: Any,
        t: Any,
        dt: Any,
        u: Any,
        *,
        formulation: AuxiliaryODE,
        u0: Any = None,
        bcs: Any = None,
        solver_parameters: Any = None,
        appctx: Any = None,
    ) -> None:
        try:
            import firedrake as fd
            import ufl
        except ImportError as error:
            raise RuntimeError(
                "AuxiliaryODE requires an active Firedrake environment"
            ) from error
        require_real_float64_petsc()
        from yonderdrake.time._ufl_marker import (
            ExponentialMemoryMarker,
            RiemannLiouvilleDerivativeMarker,
            evaluate_form_at_end_time,
            find_time_memory_markers,
            replace_time_memory_markers,
        )

        end_time_form = evaluate_form_at_end_time(F, t, dt)
        markers = find_time_memory_markers(end_time_form)
        if len(markers) != 1:
            raise ValueError(
                "the auxiliary-ODE stepper requires exactly one "
                f"time-memory marker, found {len(markers)}"
            )
        marker = markers[0]
        is_exponential = isinstance(marker, ExponentialMemoryMarker)
        if marker.field is not u:
            raise ValueError(
                "the time-memory marker must wrap the stepper solution u"
            )
        if u.ufl_shape != ():
            raise NotImplementedError(
                "AuxiliaryODE supports scalar physical fields only"
            )
        space = u.function_space()
        if space.ufl_element().family() != "Lagrange":
            raise NotImplementedError(
                "AuxiliaryODE supports continuous Lagrange spaces only"
            )
        arguments = F.arguments()
        if len(arguments) != 1:
            raise ValueError(
                "the physical residual must have exactly one test argument"
            )

        self._fd = fd
        self._ufl = ufl
        self.F = F
        self.representation: Any = representation
        self.formulation = formulation
        self.t = t
        self.dt = dt
        self.u = u
        self.bcs = bcs
        self._space = space
        self._parameter_operand = (
            marker.decay_rate if is_exponential else marker.alpha
        )
        self._operator_kind = (
            "exponential_memory"
            if is_exponential
            else (
                "riemann_liouville"
                if isinstance(marker, RiemannLiouvilleDerivativeMarker)
                else "caputo"
            )
        )
        self.parameter = self._read_parameter()
        if is_exponential:
            term_representation: Any = _SingleExponential()
        else:
            assert representation is not None
            term_representation = representation
        self._term_representation = term_representation
        self.spectrum = self._term_representation.spectrum(self.parameter)
        if is_exponential:
            self.decay_rate = self.parameter
        self._solver_parameters = dict(solver_parameters or {})
        self._lower_limit = self._read_time()

        if u0 is not None:
            self.u.assign(u0)
        self._previous = fd.Function(space, name="auxiliary_previous")
        self._previous.assign(self.u)
        self._initial = fd.Function(space, name="time_memory_initial")
        self._initial.assign(self.u)
        self._initial_trace_term = fd.Function(
            space,
            name="fractional_initial_trace_term",
        )
        self._initial_trace_term.assign(0.0)
        self._previous_modes = tuple(
            fd.Function(space, name=f"auxiliary_previous_mode_{index:04d}")
            for index in range(self.spectrum.rates.size)
        )

        self._mixed_space = fd.MixedFunctionSpace(
            [space] * (self.spectrum.rates.size + 1)
        )
        self._state = fd.Function(
            self._mixed_space,
            name="time_memory_auxiliary_state",
        )
        self._state.sub(0).assign(self.u)
        # Reused rollback snapshots.
        self._committed_state = fd.Function(
            self._mixed_space,
            name="time_memory_auxiliary_committed",
        )
        self._committed_u = fd.Function(space, name="auxiliary_committed_u")
        state_fields = fd.split(self._state)
        test_fields = fd.TestFunctions(self._mixed_space)
        physical = state_fields[0]
        physical_test = test_fields[0]
        modes = state_fields[1:]
        mode_tests = test_fields[1:]
        modal_memory_value = sum(
            (
                float(weight) * mode
                for weight, mode in zip(
                    self.spectrum.weights,
                    modes,
                    strict=True,
                )
            ),
            0.0 * physical,
        )
        self._local_correction = fd.Constant(0.0)
        memory_value = modal_memory_value + self._local_correction * (
            physical - self._previous
        )
        physical_residual = replace_time_memory_markers(
            end_time_form,
            lambda _marker: memory_value + self._initial_trace_term,
        )
        physical_residual = ufl.replace(
            physical_residual,
            {
                u: physical,
                arguments[0]: physical_test,
            },
        )

        mode_residuals = []
        measure = fd.dx(domain=space.mesh())
        for rate, mode, old_mode, mode_test in zip(
            self.spectrum.rates,
            modes,
            self._previous_modes,
            mode_tests,
            strict=True,
        ):
            derivative_term = (mode - old_mode) / self.dt
            physical_increment = (physical - self._previous) / self.dt
            if formulation.scheme == "backward_euler":
                mode_equation = (
                    derivative_term
                    + float(rate) * mode
                    - physical_increment
                )
            else:
                mode_equation = (
                    derivative_term
                    + 0.5 * float(rate) * (mode + old_mode)
                    - physical_increment
                )
            mode_residuals.append(fd.inner(mode_equation, mode_test) * measure)
        self._mode_residuals = tuple(mode_residuals)
        self._transformed_residual = physical_residual + sum(
            mode_residuals,
            0,
        )

        original_bcs = (
            ()
            if bcs is None
            else tuple(bcs)
            if isinstance(bcs, (tuple, list))
            else (bcs,)
        )
        self._mixed_bcs = tuple(
            bc.reconstruct(V=self._mixed_space.sub(0)) for bc in original_bcs
        )
        field_names = ("physical",) + tuple(
            f"mode_{index:04d}" for index in range(self.spectrum.rates.size)
        )
        yonderdrake_context = {
            "formulation": "auxiliary_ode",
            "operator_kind": self._operator_kind,
            "scheme": formulation.scheme,
            "field_names": field_names,
            "physical_field": 0,
            "mode_fields": tuple(range(1, self.spectrum.rates.size + 1)),
        }
        self._appctx = dict(appctx or {})
        self._appctx["yonderdrake"] = yonderdrake_context

        self._solver: Any = None
        self._rebuild_solver = True
        self._last_step_size: float | None = None
        self._reset_solver_counters()

    def _read_parameter(self) -> float:
        try:
            value = float(self._parameter_operand)
        except (TypeError, ValueError) as error:
            name = (
                "decay_rate"
                if self._operator_kind == "exponential_memory"
                else "alpha"
            )
            raise TypeError(f"{name} must be a real scalar") from error
        if self._operator_kind == "exponential_memory":
            if not isfinite(value) or value <= 0.0:
                raise ValueError("decay_rate must be finite and positive")
            return value
        if not isfinite(value) or not 0.0 < value < 1.0:
            raise ValueError("alpha must satisfy 0 < alpha < 1")
        return value

    def _update_initial_trace(self, step_size: float) -> None:
        if self._operator_kind != "riemann_liouville":
            return
        elapsed = self._read_time() + step_size - self._lower_limit
        if not isfinite(elapsed) or elapsed <= 0.0:
            raise ValueError(
                "Riemann-Liouville evaluation time must exceed its lower limit"
            )
        scale = elapsed ** (-self.parameter) / gamma(1.0 - self.parameter)
        self._initial_trace_term.assign(scale * self._initial)

    def _build_solver(self) -> None:
        problem = self._fd.NonlinearVariationalProblem(
            self._transformed_residual,
            self._state,
            bcs=self._mixed_bcs,
        )
        self._solver = self._fd.NonlinearVariationalSolver(
            problem,
            solver_parameters=self._solver_parameters,
            appctx=self._appctx,
        )
        self._rebuild_solver = False

    def advance(self) -> None:
        """Solve and atomically commit the physical-plus-mode mixed state."""
        if self._read_parameter() != self.parameter:
            if self._operator_kind != "exponential_memory":
                raise RuntimeError(
                    "changing alpha after construction is unsupported; "
                    "rebuild the stepper"
                )
            raise RuntimeError(
                "changing a time-memory parameter after construction is "
                "unsupported; rebuild the stepper"
            )
        self._last_step_size = self._read_step_size()
        validate_recurrence_interval(
            self.spectrum,
            self._last_step_size,
            final_time=(
                self._read_time()
                + self._last_step_size
                - self._lower_limit
            ),
        )
        if self.spectrum.metadata.get("representation") == "SumOfExponentials":
            arguments = self.spectrum.rates * self._last_step_size
            response = (
                1.0 / (1.0 + arguments)
                if self.formulation.scheme == "backward_euler"
                else 1.0 / (1.0 + 0.5 * arguments)
            )
            exact_local = self._last_step_size ** (-self.parameter) / gamma(
                2.0 - self.parameter
            )
            self._local_correction.assign(
                exact_local - float(np.dot(self.spectrum.weights, response))
            )
        self._update_initial_trace(self._last_step_size)
        if self._rebuild_solver:
            self._build_solver()

        self._committed_state.assign(self._state)
        self._committed_u.assign(self.u)
        try:
            self._solver.solve()
        except Exception:
            self._state.assign(self._committed_state)
            self.u.assign(self._committed_u)
            self._failure_count += 1
            raise

        state_subfunctions = self._state.subfunctions
        self.u.assign(state_subfunctions[0])
        self._previous.assign(self.u)
        for old_mode, new_mode in zip(
            self._previous_modes,
            state_subfunctions[1:],
            strict=True,
        ):
            old_mode.assign(new_mode)
        self._solve_count += 1
        self._nonlinear_iterations += self._solver.snes.getIterationNumber()
        self._linear_iterations += self._solver.snes.getLinearSolveIterations()

    @property
    def history(self) -> tuple[Any, ...]:
        return tuple(
            mode.copy(deepcopy=True) for mode in self._state.subfunctions[1:]
        )

    @property
    def transformed_residual(self) -> Any:
        return self._transformed_residual

    @property
    def mode_residuals(self) -> tuple[Any, ...]:
        """Residual forms for residual-level verification and diagnostics."""
        return self._mode_residuals

    @property
    def appctx(self) -> dict[str, Any]:
        return dict(self._appctx)

    def solver_stats(self) -> dict[str, Any]:
        return {
            "solves": self._solve_count,
            "failures": self._failure_count,
            "nonlinear_iterations": self._nonlinear_iterations,
            "linear_iterations": self._linear_iterations,
            "num_modes": len(self._previous_modes),
            "stored_fields": len(self._previous_modes) + 1,
            "scheme": self.formulation.scheme,
            "last_step_size": self._last_step_size,
        }

    def reset(self, u0: Any, t0: Any = None) -> None:
        self.u.assign(u0)
        self._previous.assign(self.u)
        self._initial.assign(self.u)
        self._initial_trace_term.assign(0.0)
        self._state.sub(0).assign(self.u)
        for previous_mode, mode in zip(
            self._previous_modes,
            self._state.subfunctions[1:],
            strict=True,
        ):
            previous_mode.assign(0.0)
            mode.assign(0.0)
        if t0 is not None:
            self.t.assign(t0)
        self._lower_limit = self._read_time()
        self._last_step_size = None
        self._reset_solver_counters()

    def _checkpoint_metadata(self) -> dict[str, Any]:
        payload = stepper_metadata(
            kind="auxiliary_ode",
            operator_kinds=(self._operator_kind,),
            parameters=(self.parameter,),
            formulation={
                "kind": "auxiliary_ode",
                "scheme": self.formulation.scheme,
            },
            representations=[
                self._term_representation.describe(self.parameter)
            ],
        )
        payload.update({
            "lower_limit": self._lower_limit,
            "time": float(self.t),
            "dt": float(self.dt),
            "stats": self.solver_stats(),
        })
        return payload

    def checkpoint_state(self) -> dict[str, Any]:
        state = self._checkpoint_metadata()
        state.update(
            {
                "physical": self.u.dat.data_ro.tolist(),
                "initial": self._initial.dat.data_ro.tolist(),
                "modes": [
                    mode.dat.data_ro.tolist()
                    for mode in self._state.subfunctions[1:]
                ],
            }
        )
        return state

    def _checkpoint_file_fields(self) -> dict[str, Any]:
        fields = {
            "physical": self.u,
            "initial": self._initial,
        }
        fields.update(
            {
                f"mode_{index:04d}": mode
                for index, mode in enumerate(self._state.subfunctions[1:])
            }
        )
        return fields

    def save_checkpoint(self, checkpoint: Any, *, name: str = "state") -> None:
        """Save collective state to a Firedrake CheckpointFile."""
        save_checkpoint_file(
            checkpoint,
            name=name,
            metadata=self._checkpoint_metadata(),
            fields=self._checkpoint_file_fields(),
        )

    def load_checkpoint(self, checkpoint: Any, *, name: str = "state") -> None:
        """Load collective state from a Firedrake CheckpointFile."""
        fields = self._checkpoint_file_fields()
        state, loaded = load_checkpoint_file(
            checkpoint,
            name=name,
            mesh=self._space.mesh(),
            expected_fields=tuple(fields),
        )
        state["physical"] = loaded["physical"].dat.data_ro.tolist()
        state["initial"] = loaded["initial"].dat.data_ro.tolist()
        state["modes"] = [
            loaded[f"mode_{index:04d}"].dat.data_ro.tolist()
            for index in range(len(self._previous_modes))
        ]
        self.restore_checkpoint(state)

    def restore_checkpoint(self, state: dict[str, Any]) -> None:
        representations = validate_stepper_metadata(
            state,
            kind="auxiliary_ode",
            operator_kinds=(self._operator_kind,),
            parameters=(self.parameter,),
        )
        formulation = dict(state.get("formulation") or {})
        if formulation.get("kind") != "auxiliary_ode":
            raise ValueError("checkpoint formulation does not match")
        if formulation.get("scheme") != self.formulation.scheme:
            raise ValueError("checkpoint auxiliary scheme does not match")
        validate_checkpoint_representation(
            representations[0],
            self._term_representation,
            self.parameter,
        )
        modes = state.get("modes")
        if not isinstance(modes, list) or len(modes) != len(self._previous_modes):
            raise ValueError("checkpoint mode count does not match the stepper")
        physical = np.asarray(state.get("physical"), dtype=np.float64)
        initial = np.asarray(state.get("initial"), dtype=np.float64)
        mode_arrays = [np.asarray(values, dtype=np.float64) for values in modes]
        local_shape = self.u.dat.data_ro.shape
        if (
            physical.shape != local_shape
            or initial.shape != local_shape
            or any(values.shape != local_shape for values in mode_arrays)
        ):
            raise ValueError("checkpoint field has the wrong local shape")
        if (
            not np.all(np.isfinite(physical))
            or not np.all(np.isfinite(initial))
            or any(not np.all(np.isfinite(values)) for values in mode_arrays)
        ):
            raise ValueError("checkpoint fields must contain finite values")
        try:
            checkpoint_time = float(state["time"])
            checkpoint_dt = float(state["dt"])
            checkpoint_lower_limit = float(state["lower_limit"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("checkpoint time metadata is invalid") from error
        if not isfinite(checkpoint_time) or not isfinite(checkpoint_lower_limit):
            raise ValueError("checkpoint times must be finite")
        if not isfinite(checkpoint_dt) or checkpoint_dt <= 0.0:
            raise ValueError("checkpoint dt must be finite and positive")
        try:
            stats = dict(state.get("stats") or {})
            solve_count = int(stats.get("solves", 0))
            failure_count = int(stats.get("failures", 0))
            nonlinear_iterations = int(stats.get("nonlinear_iterations", 0))
            linear_iterations = int(stats.get("linear_iterations", 0))
            last_step_size = stats.get("last_step_size")
            if last_step_size is not None:
                last_step_size = float(last_step_size)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("checkpoint solver statistics are invalid") from error
        if min(
            solve_count,
            failure_count,
            nonlinear_iterations,
            linear_iterations,
        ) < 0:
            raise ValueError("checkpoint solver statistics must be nonnegative")
        if last_step_size is not None and (
            not isfinite(last_step_size) or last_step_size <= 0.0
        ):
            raise ValueError(
                "checkpoint last step size must be finite and positive"
            )

        self.u.dat.data[:] = physical
        self._initial.dat.data[:] = initial
        self._previous.dat.data[:] = physical
        self._state.sub(0).dat.data[:] = physical
        for previous_mode, mode, values in zip(
            self._previous_modes,
            self._state.subfunctions[1:],
            mode_arrays,
            strict=True,
        ):
            previous_mode.dat.data[:] = values
            mode.dat.data[:] = values
        self.t.assign(checkpoint_time)
        self.dt.assign(checkpoint_dt)
        self._lower_limit = checkpoint_lower_limit
        self._initial_trace_term.assign(0.0)
        self._solve_count = solve_count
        self._failure_count = failure_count
        self._nonlinear_iterations = nonlinear_iterations
        self._linear_iterations = linear_iterations
        self._last_step_size = last_step_size
        self._rebuild_solver = True
