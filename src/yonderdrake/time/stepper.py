"""Residual-first eliminated-recurrence time-memory stepper."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass
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
from yonderdrake.time.coefficients import (
    oscillator_coefficients,
    quadratic_recurrence_coefficients,
    recurrence_coefficients,
    validate_recurrence_interval,
)
from yonderdrake.time.formulations import AuxiliaryODE, Oscillator, Recurrence
from yonderdrake.time.representations import (
    AlikhanovL21Sigma,
    BirkSong,
    Diethelm2008,
    Diethelm2022,
    FastObliviousCQ,
    FullHistory,
    LubichCQ,
    SineDiffusive,
    SumOfExponentials,
    YuanAgrawal,
    _SingleExponential,
    validate_checkpoint_representation,
)


class _ModalStepper(StepperLifecycle):
    """Advance one physical field with shared modal state management."""

    def __init__(
        self,
        F: Any,
        representation: Any,
        t: Any,
        dt: Any,
        u: Any,
        *,
        formulation: Any = None,
        u0: Any = None,
        bcs: Any = None,
        solver_parameters: Any = None,
        appctx: Any = None,
    ) -> None:
        try:
            import firedrake as fd
        except ImportError as error:
            raise RuntimeError(
                "time-memory stepping requires an active Firedrake environment"
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
        if not markers:
            raise ValueError(
                "the recurrence stepper requires at least one time-memory marker"
            )
        if any(marker.field is not u for marker in markers):
            raise ValueError(
                "every time-memory marker must wrap the stepper solution u"
            )
        element = u.function_space().ufl_element()
        if element.family() != "Lagrange":
            raise NotImplementedError(
                "Recurrence stepping supports continuous Lagrange spaces only"
            )

        self._fd = fd
        self.F = F
        self.representation: Any = representation
        self.formulation = formulation
        self.t = t
        self.dt = dt
        self.u = u
        self.bcs = bcs
        self._solver_parameters = dict(solver_parameters or {})
        self._appctx = dict(appctx or {})
        self._markers = markers
        self._parameter_operands = tuple(
            (
                marker.decay_rate
                if isinstance(marker, ExponentialMemoryMarker)
                else marker.alpha
            )
            for marker in markers
        )
        self._operator_kinds = tuple(
            (
                "exponential_memory"
                if isinstance(marker, ExponentialMemoryMarker)
                else (
                    "riemann_liouville"
                    if isinstance(marker, RiemannLiouvilleDerivativeMarker)
                    else "caputo"
                )
            )
            for marker in markers
        )
        single_exponential = _SingleExponential()
        self._term_representations: tuple[Any, ...] = tuple(
            (
                single_exponential
                if kind == "exponential_memory"
                else representation
            )
            for kind in self._operator_kinds
        )
        self.parameters = self._read_parameters()
        self.spectra = tuple(
            term_representation.spectrum(parameter)
            for term_representation, parameter in zip(
                self._term_representations,
                self.parameters,
                strict=True,
            )
        )
        self.decay_rates = tuple(
            parameter
            for kind, parameter in zip(
                self._operator_kinds,
                self.parameters,
                strict=True,
            )
            if kind == "exponential_memory"
        )
        self._operator_kind = self._operator_kinds[0]
        self.spectrum = (
            self.spectra[0] if len(self.spectra) == 1 else self.spectra
        )
        self._space = u.function_space()
        self._lower_limit = self._read_time()

        if u0 is not None:
            self.u.assign(u0)
        self._previous = fd.Function(self._space, name="time_memory_previous")
        self._previous.assign(self.u)
        self._penultimate = (
            fd.Function(self._space, name="time_memory_penultimate")
            if isinstance(formulation, Recurrence)
            and formulation.interpolant == "quadratic"
            else None
        )
        if self._penultimate is not None:
            self._penultimate.assign(self.u)
        self._initial = fd.Function(self._space, name="time_memory_initial")
        self._initial.assign(self.u)
        # Reused rollback snapshot.
        self._committed_u = fd.Function(
            self._space,
            name="time_memory_committed",
        )
        self._increment = fd.Function(
            self._space,
            name="time_memory_increment",
        )
        self._mode_groups = tuple(
            tuple(
                fd.Function(
                    self._space,
                    name=f"time_memory_term_{term:02d}_mode_{index:04d}",
                )
                for index in range(
                    spectrum.frequencies.size
                    if isinstance(formulation, Oscillator)
                    else spectrum.rates.size
                )
            )
            for term, spectrum in enumerate(self.spectra)
        )
        self._mode_velocity_groups = (
            tuple(
                tuple(
                    fd.Function(
                        self._space,
                        name=(
                            f"time_memory_term_{term:02d}_velocity_"
                            f"{index:04d}"
                        ),
                    )
                    for index in range(spectrum.frequencies.size)
                )
                for term, spectrum in enumerate(self.spectra)
            )
            if isinstance(formulation, Oscillator)
            else ()
        )
        self._mode_scratch = (
            fd.Function(self._space, name="time_memory_mode_scratch")
            if isinstance(formulation, Oscillator)
            else None
        )
        self._history_terms = tuple(
            fd.Function(
                self._space,
                name=f"time_memory_history_term_{term:02d}",
            )
            for term in range(len(markers))
        )
        for history_term in self._history_terms:
            history_term.assign(0.0)
        self._implicit_weights = tuple(
            fd.Constant(0.0) for _ in markers
        )
        self._initial_trace_terms = tuple(
            fd.Function(
                self._space,
                name=f"fractional_initial_trace_term_{term:02d}",
            )
            for term in range(len(markers))
        )
        for initial_trace_term in self._initial_trace_terms:
            initial_trace_term.assign(0.0)

        replacements = {
            marker: (
                history_term
                + implicit_weight * (self.u - self._previous)
                + initial_trace_term
            )
            for marker, history_term, implicit_weight, initial_trace_term in zip(
                markers,
                self._history_terms,
                self._implicit_weights,
                self._initial_trace_terms,
                strict=True,
            )
        }

        self._transformed_residual = replace_time_memory_markers(
            end_time_form,
            replacements.__getitem__,
        )
        self._solver: Any = None
        self._rebuild_solver = True
        self._last_step_size: float | None = None
        self._coefficient_step_size: float | None = None
        self._coefficient_previous_step_size: float | None = None
        self._recurrence_coefficients: (
            tuple[
                tuple[
                    np.ndarray,
                    np.ndarray,
                    np.ndarray,
                    np.ndarray,
                    float,
                ],
                ...,
            ]
            | None
        ) = None
        self._oscillator_coefficients: (
            tuple[
                tuple[
                    np.ndarray,
                    np.ndarray,
                    np.ndarray,
                    np.ndarray,
                    np.ndarray,
                    np.ndarray,
                ],
                ...,
            ]
            | None
        ) = None
        self._reset_solver_counters()

    @staticmethod
    def _read_parameter_operand(
        parameter_operand: Any,
        operator_kind: str,
    ) -> float:
        try:
            value = float(parameter_operand)
        except (TypeError, ValueError) as error:
            name = (
                "decay_rate"
                if operator_kind == "exponential_memory"
                else "alpha"
            )
            raise TypeError(f"{name} must be a real scalar") from error
        if operator_kind == "exponential_memory":
            if not isfinite(value) or value <= 0.0:
                raise ValueError("decay_rate must be finite and positive")
            return value
        if not isfinite(value) or not 0.0 < value < 1.0:
            raise ValueError("alpha must satisfy 0 < alpha < 1")
        return value

    def _read_parameters(self) -> tuple[float, ...]:
        return tuple(
            self._read_parameter_operand(operand, kind)
            for operand, kind in zip(
                self._parameter_operands,
                self._operator_kinds,
                strict=True,
            )
        )

    def _update_initial_trace(self, step_size: float) -> None:
        if "riemann_liouville" not in self._operator_kinds:
            return
        elapsed = self._read_time() + step_size - self._lower_limit
        if not isfinite(elapsed) or elapsed <= 0.0:
            raise ValueError(
                "Riemann-Liouville evaluation time must exceed its lower limit"
            )
        for kind, parameter, initial_trace_term in zip(
            self._operator_kinds,
            self.parameters,
            self._initial_trace_terms,
            strict=True,
        ):
            if kind == "riemann_liouville":
                scale = elapsed ** (-parameter) / gamma(1.0 - parameter)
                initial_trace_term.assign(scale * self._initial)

    def _build_solver(self) -> None:
        problem = self._fd.NonlinearVariationalProblem(
            self._transformed_residual,
            self.u,
            bcs=self.bcs,
        )
        self._solver = self._fd.NonlinearVariationalSolver(
            problem,
            solver_parameters=self._solver_parameters,
            appctx=self._appctx,
        )
        self._rebuild_solver = False

    def _prepare_recurrence_step(
        self,
        step_size: float,
    ) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], ...]:
        self._update_initial_trace(step_size)
        elapsed = self._read_time() + step_size - self._lower_limit
        for spectrum in self.spectra:
            validate_recurrence_interval(
                spectrum,
                step_size,
                final_time=elapsed,
            )
        if (
            self._coefficient_step_size != step_size
            or self._coefficient_previous_step_size != self._last_step_size
            or self._recurrence_coefficients is None
        ):
            coefficient_groups = []
            for spectrum, implicit_weight_field in zip(
                self.spectra,
                self._implicit_weights,
                strict=True,
            ):
                if self.formulation.interpolant == "quadratic":
                    (
                        decay,
                        interpolation,
                        previous_interpolation,
                        implicit_weight,
                        previous_weight,
                    ) = quadratic_recurrence_coefficients(
                        spectrum,
                        step_size,
                        previous_step_size=self._last_step_size,
                        final_time=elapsed,
                    )
                else:
                    (
                        decay,
                        interpolation,
                        implicit_weight,
                    ) = recurrence_coefficients(
                        spectrum,
                        step_size,
                        final_time=elapsed,
                    )
                    previous_interpolation = np.zeros_like(interpolation)
                    previous_weight = 0.0
                history_weights = spectrum.weights * decay
                implicit_weight_field.assign(implicit_weight)
                coefficient_groups.append(
                    (
                        decay,
                        interpolation,
                        previous_interpolation,
                        history_weights,
                        previous_weight,
                    )
                )
            self._coefficient_step_size = step_size
            self._coefficient_previous_step_size = self._last_step_size
            self._recurrence_coefficients = tuple(coefficient_groups)

        coefficients_by_term = self._recurrence_coefficients
        assert coefficients_by_term is not None
        for modes, history_term, coefficients in zip(
            self._mode_groups,
            self._history_terms,
            coefficients_by_term,
            strict=True,
        ):
            _, _, _, history_weights, previous_weight = coefficients
            with history_term.dat.vec as history_vector:
                history_vector.set(0.0)
                for weight, mode in zip(
                    history_weights,
                    modes,
                    strict=True,
                ):
                    with mode.dat.vec_ro as mode_vector:
                        history_vector.axpy(float(weight), mode_vector)
                if previous_weight:
                    penultimate = self._penultimate
                    assert penultimate is not None
                    with (
                        self._previous.dat.vec_ro as previous_vector,
                        penultimate.dat.vec_ro as penultimate_vector,
                    ):
                        history_vector.axpy(previous_weight, previous_vector)
                        history_vector.axpy(-previous_weight, penultimate_vector)
        return tuple(
            (decay, interpolation, previous_interpolation)
            for decay, interpolation, previous_interpolation, _, _ in (
                coefficients_by_term
            )
        )

    def _prepare_oscillator_step(
        self,
        step_size: float,
    ) -> tuple[
        tuple[
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
        ],
        ...,
    ]:
        self._update_initial_trace(step_size)
        if (
            self._coefficient_step_size != step_size
            or self._oscillator_coefficients is None
        ):
            coefficient_groups = []
            for parameter, spectrum, implicit_weight_field in zip(
                self.parameters,
                self.spectra,
                self._implicit_weights,
                strict=True,
            ):
                (
                    cosine,
                    sine_over_frequency,
                    negative_frequency_sine,
                    position_forcing,
                    velocity_forcing,
                    implicit_weight,
                ) = oscillator_coefficients(
                    spectrum,
                    parameter,
                    step_size,
                )
                implicit_weight_field.assign(implicit_weight)
                coefficient_groups.append(
                    (
                        cosine,
                        sine_over_frequency,
                        negative_frequency_sine,
                        position_forcing,
                        velocity_forcing,
                        spectrum.weights,
                    )
                )
            self._coefficient_step_size = step_size
            self._oscillator_coefficients = tuple(coefficient_groups)

        coefficients_by_term = self._oscillator_coefficients
        assert coefficients_by_term is not None
        for (
            modes,
            velocities,
            history_term,
            coefficients,
        ) in zip(
            self._mode_groups,
            self._mode_velocity_groups,
            self._history_terms,
            coefficients_by_term,
            strict=True,
        ):
            cosine, sine_over_frequency, _, _, _, weights = coefficients
            with history_term.dat.vec as history_vector:
                history_vector.set(0.0)
                for index, (mode, velocity) in enumerate(
                    zip(modes, velocities, strict=True)
                ):
                    with mode.dat.vec_ro as mode_vector:
                        history_vector.axpy(
                            float(weights[index] * cosine[index]),
                            mode_vector,
                        )
                    with velocity.dat.vec_ro as velocity_vector:
                        history_vector.axpy(
                            float(
                                weights[index]
                                * sine_over_frequency[index]
                            ),
                            velocity_vector,
                        )
        self._last_step_size = step_size
        return coefficients_by_term

    def _prepare_step(self, step_size: float) -> Any:
        if isinstance(self.formulation, Oscillator):
            return self._prepare_oscillator_step(step_size)
        return self._prepare_recurrence_step(step_size)

    def advance(self) -> None:
        """Solve for the next state and atomically commit all modal states."""
        current_parameters = self._read_parameters()
        if current_parameters != self.parameters:
            if "exponential_memory" not in self._operator_kinds:
                raise RuntimeError(
                    "changing alpha after construction is unsupported; "
                    "rebuild the stepper"
                )
            raise RuntimeError(
                "changing a time-memory parameter after construction is "
                "unsupported; rebuild the stepper"
            )
        step_size = self._read_step_size()
        coefficients_by_term = self._prepare_step(step_size)
        if self._rebuild_solver:
            self._build_solver()

        self._committed_u.assign(self.u)
        try:
            self._solver.solve()
        except Exception:
            self.u.assign(self._committed_u)
            self._failure_count += 1
            raise

        with (
            self.u.dat.vec_ro as solution_vector,
            self._previous.dat.vec_ro as previous_vector,
            self._increment.dat.vec as increment_vector,
        ):
            increment_vector.waxpy(-1.0, previous_vector, solution_vector)
            if isinstance(self.formulation, Oscillator):
                scratch = self._mode_scratch
                assert scratch is not None
                for modes, velocities, coefficients in zip(
                    self._mode_groups,
                    self._mode_velocity_groups,
                    coefficients_by_term,
                    strict=True,
                ):
                    (
                        cosine,
                        sine_over_frequency,
                        negative_frequency_sine,
                        position_forcing,
                        velocity_forcing,
                        _,
                    ) = coefficients
                    for index, (mode, velocity) in enumerate(
                        zip(modes, velocities, strict=True)
                    ):
                        scratch.assign(mode)
                        with (
                            scratch.dat.vec_ro as old_position_vector,
                            mode.dat.vec as position_vector,
                            velocity.dat.vec as velocity_vector,
                        ):
                            position_vector.scale(float(cosine[index]))
                            position_vector.axpy(
                                float(sine_over_frequency[index]),
                                velocity_vector,
                            )
                            position_vector.axpy(
                                float(position_forcing[index]),
                                increment_vector,
                            )
                            velocity_vector.scale(float(cosine[index]))
                            velocity_vector.axpy(
                                float(negative_frequency_sine[index]),
                                old_position_vector,
                            )
                            velocity_vector.axpy(
                                float(velocity_forcing[index]),
                                increment_vector,
                            )
            else:
                penultimate = self._penultimate
                if penultimate is None:
                    for modes, coefficients in zip(
                        self._mode_groups,
                        coefficients_by_term,
                        strict=True,
                    ):
                        decay, interpolation, _ = coefficients
                        for mode, decay_value, interpolation_value in zip(
                            modes,
                            decay,
                            interpolation,
                            strict=True,
                        ):
                            with mode.dat.vec as mode_vector:
                                mode_vector.scale(float(decay_value))
                                mode_vector.axpy(
                                    float(interpolation_value),
                                    increment_vector,
                                )
                else:
                    with (
                        penultimate.dat.vec_ro as penultimate_vector,
                        self._committed_u.dat.vec as old_increment_vector,
                    ):
                        old_increment_vector.waxpy(
                            -1.0, penultimate_vector, previous_vector
                        )
                        for modes, coefficients in zip(
                            self._mode_groups,
                            coefficients_by_term,
                            strict=True,
                        ):
                            decay, interpolation, previous_interpolation = (
                                coefficients
                            )
                            for mode, decay_value, current_value, old_value in (
                                zip(
                                    modes,
                                    decay,
                                    interpolation,
                                    previous_interpolation,
                                    strict=True,
                                )
                            ):
                                with mode.dat.vec as mode_vector:
                                    mode_vector.scale(float(decay_value))
                                    mode_vector.axpy(
                                        float(current_value), increment_vector
                                    )
                                    mode_vector.axpy(
                                        float(old_value), old_increment_vector
                                    )
        if self._penultimate is not None:
            self._penultimate.assign(self._previous)
        self._previous.assign(self.u)
        self._last_step_size = step_size

        self._solve_count += 1
        self._nonlinear_iterations += self._solver.snes.getIterationNumber()
        self._linear_iterations += self._solver.snes.getLinearSolveIterations()

    @property
    def history(self) -> tuple[Any, ...]:
        """Return defensive copies of the primary memory-mode fields."""
        return tuple(
            mode.copy(deepcopy=True)
            for modes in self._mode_groups
            for mode in modes
        )

    @property
    def term_histories(self) -> tuple[tuple[Any, ...], ...]:
        """Return internal modes grouped by time-memory term."""
        return tuple(
            tuple(mode.copy(deepcopy=True) for mode in modes)
            for modes in self._mode_groups
        )

    @property
    def oscillator_velocities(self) -> tuple[Any, ...]:
        """Return defensive copies of oscillator velocity fields."""
        return tuple(
            velocity.copy(deepcopy=True)
            for velocities in self._mode_velocity_groups
            for velocity in velocities
        )

    @property
    def term_oscillator_velocities(self) -> tuple[tuple[Any, ...], ...]:
        """Return oscillator velocities grouped by time-memory term."""
        return tuple(
            tuple(velocity.copy(deepcopy=True) for velocity in velocities)
            for velocities in self._mode_velocity_groups
        )

    @property
    def transformed_residual(self) -> Any:
        """The ordinary UFL residual supplied to Firedrake."""
        return self._transformed_residual

    def solver_stats(self) -> dict[str, Any]:
        num_fractional_terms = sum(
            kind != "exponential_memory" for kind in self._operator_kinds
        )
        return {
            "solves": self._solve_count,
            "failures": self._failure_count,
            "nonlinear_iterations": self._nonlinear_iterations,
            "linear_iterations": self._linear_iterations,
            "num_modes": sum(len(modes) for modes in self._mode_groups),
            "num_time_memory_terms": len(self._markers),
            "num_fractional_terms": num_fractional_terms,
            "num_exponential_memory_terms": (
                len(self._markers) - num_fractional_terms
            ),
            "modes_per_term": tuple(
                len(modes) for modes in self._mode_groups
            ),
            "fields_per_mode": (
                2 if isinstance(self.formulation, Oscillator) else 1
            ),
            "physical_history_fields": (
                2 if self._penultimate is not None else 1
            ),
            "last_step_size": self._last_step_size,
        }

    def reset(self, u0: Any, t0: Any = None) -> None:
        """Reset the physical state and zero every modal field."""
        self.u.assign(u0)
        self._previous.assign(self.u)
        if self._penultimate is not None:
            self._penultimate.assign(self.u)
        self._initial.assign(self.u)
        for modes in self._mode_groups:
            for mode in modes:
                mode.assign(0.0)
        for velocities in self._mode_velocity_groups:
            for velocity in velocities:
                velocity.assign(0.0)
        for history_term in self._history_terms:
            history_term.assign(0.0)
        for initial_trace_term in self._initial_trace_terms:
            initial_trace_term.assign(0.0)
        if t0 is not None:
            self.t.assign(t0)
        self._lower_limit = self._read_time()
        self._last_step_size = None
        self._coefficient_step_size = None
        self._coefficient_previous_step_size = None
        self._recurrence_coefficients = None
        self._reset_solver_counters()

    def _checkpoint_metadata(self) -> dict[str, Any]:
        oscillator = isinstance(self.formulation, Oscillator)
        payload = stepper_metadata(
            kind="oscillator" if oscillator else "recurrence",
            operator_kinds=self._operator_kinds,
            parameters=self.parameters,
            representations=[
                term_representation.describe(parameter)
                for term_representation, parameter in zip(
                    self._term_representations,
                    self.parameters,
                    strict=True,
                )
            ],
            formulation=(
                {
                    "kind": "oscillator",
                    "interpolant": "linear",
                    "state": "position_velocity",
                }
                if oscillator
                else self.formulation.describe()
            ),
        )
        payload.update({
            "lower_limit": self._lower_limit,
            "time": float(self.t),
            "dt": float(self.dt),
            "stats": self.solver_stats(),
        })
        return payload

    def checkpoint_state(self) -> dict[str, Any]:
        """Return a serializable local-state checkpoint payload."""
        payload = self._checkpoint_metadata()
        payload.update(
            {
                "u": self.u.dat.data_ro.tolist(),
                "initial": self._initial.dat.data_ro.tolist(),
                "previous": self._previous.dat.data_ro.tolist(),
            }
        )
        if self._penultimate is not None:
            payload["penultimate"] = self._penultimate.dat.data_ro.tolist()
        payload["mode_groups"] = [
            [mode.dat.data_ro.tolist() for mode in modes]
            for modes in self._mode_groups
        ]
        if self._mode_velocity_groups:
            payload["mode_velocity_groups"] = [
                [velocity.dat.data_ro.tolist() for velocity in velocities]
                for velocities in self._mode_velocity_groups
            ]
        return payload

    def _checkpoint_file_fields(self) -> dict[str, Any]:
        fields = {
            "u": self.u,
            "initial": self._initial,
            "previous": self._previous,
        }
        if self._penultimate is not None:
            fields["penultimate"] = self._penultimate
        fields.update(
            {
                f"mode_{term:04d}_{index:04d}": mode
                for term, modes in enumerate(self._mode_groups)
                for index, mode in enumerate(modes)
            }
        )
        fields.update(
            {
                f"velocity_{term:04d}_{index:04d}": velocity
                for term, velocities in enumerate(self._mode_velocity_groups)
                for index, velocity in enumerate(velocities)
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
        state["u"] = loaded["u"].dat.data_ro.tolist()
        state["initial"] = loaded["initial"].dat.data_ro.tolist()
        state["previous"] = loaded["previous"].dat.data_ro.tolist()
        if self._penultimate is not None:
            state["penultimate"] = loaded["penultimate"].dat.data_ro.tolist()
        mode_groups = [
            [
                loaded[f"mode_{term:04d}_{index:04d}"].dat.data_ro.tolist()
                for index in range(len(modes))
            ]
            for term, modes in enumerate(self._mode_groups)
        ]
        state["mode_groups"] = mode_groups
        if self._mode_velocity_groups:
            state["mode_velocity_groups"] = [
                [
                    loaded[
                        f"velocity_{term:04d}_{index:04d}"
                    ].dat.data_ro.tolist()
                    for index in range(len(velocities))
                ]
                for term, velocities in enumerate(self._mode_velocity_groups)
            ]
        self.restore_checkpoint(state)

    def restore_checkpoint(self, state: dict[str, Any]) -> None:
        """Restore a payload produced by :meth:`checkpoint_state`."""
        oscillator = isinstance(self.formulation, Oscillator)
        representations = validate_stepper_metadata(
            state,
            kind="oscillator" if oscillator else "recurrence",
            operator_kinds=self._operator_kinds,
            parameters=self.parameters,
        )
        formulation = dict(state.get("formulation") or {})
        expected_formulation = (
            {
                "kind": "oscillator",
                "interpolant": "linear",
                "state": "position_velocity",
            }
            if oscillator
            else self.formulation.describe()
        )
        if formulation != expected_formulation:
            raise ValueError("checkpoint formulation does not match")
        for metadata, term_representation, parameter in zip(
            representations,
            self._term_representations,
            self.parameters,
            strict=True,
        ):
            validate_checkpoint_representation(
                metadata,
                term_representation,
                parameter,
            )
        mode_groups = state.get("mode_groups")
        if (
            not isinstance(mode_groups, list)
            or len(mode_groups) != len(self._mode_groups)
            or any(not isinstance(modes, list) for modes in mode_groups)
            or any(
                len(values) != len(modes)
                for values, modes in zip(
                    mode_groups,
                    self._mode_groups,
                    strict=True,
                )
            )
        ):
            raise ValueError("checkpoint mode count does not match the stepper")
        physical = np.asarray(state.get("u"), dtype=np.float64)
        initial = np.asarray(state.get("initial"), dtype=np.float64)
        previous = np.asarray(state.get("previous"), dtype=np.float64)
        penultimate = (
            np.asarray(state.get("penultimate"), dtype=np.float64)
            if self._penultimate is not None
            else None
        )
        local_shape = self.u.dat.data_ro.shape
        if (
            physical.shape != local_shape
            or initial.shape != local_shape
            or previous.shape != local_shape
            or (penultimate is not None and penultimate.shape != local_shape)
        ):
            raise ValueError("checkpoint physical field has the wrong local shape")
        mode_arrays = [
            [
                np.asarray(values, dtype=np.float64)
                for values in group_values
            ]
            for group_values in mode_groups
        ]
        velocity_mode_groups = state.get("mode_velocity_groups", [])
        if oscillator and (
            not isinstance(velocity_mode_groups, list)
            or len(velocity_mode_groups) != len(self._mode_velocity_groups)
            or any(
                not isinstance(velocities, list)
                for velocities in velocity_mode_groups
            )
            or any(
                len(values) != len(velocities)
                for values, velocities in zip(
                    velocity_mode_groups,
                    self._mode_velocity_groups,
                    strict=True,
                )
            )
        ):
            raise ValueError("checkpoint mode count does not match the stepper")
        velocity_arrays = [
            [
                np.asarray(values, dtype=np.float64)
                for values in group_values
            ]
            for group_values in velocity_mode_groups
        ]
        if any(
            values.shape != local_shape
            for group_values in mode_arrays
            for values in group_values
        ) or any(
            values.shape != local_shape
            for group_values in velocity_arrays
            for values in group_values
        ):
            raise ValueError("checkpoint mode field has the wrong local shape")
        if not all(
            np.all(np.isfinite(values))
            for values in (physical, initial, previous, penultimate)
            if values is not None
        ) or any(
            not np.all(np.isfinite(values))
            for group_values in mode_arrays
            for values in group_values
        ) or any(
            not np.all(np.isfinite(values))
            for group_values in velocity_arrays
            for values in group_values
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
        self._previous.dat.data[:] = previous
        if self._penultimate is not None:
            assert penultimate is not None
            self._penultimate.dat.data[:] = penultimate
        for modes, values_group in zip(
            self._mode_groups,
            mode_arrays,
            strict=True,
        ):
            for mode, values in zip(modes, values_group, strict=True):
                mode.dat.data[:] = values
        for velocities, values_group in zip(
            self._mode_velocity_groups,
            velocity_arrays,
            strict=True,
        ):
            for velocity, values in zip(
                velocities,
                values_group,
                strict=True,
            ):
                velocity.dat.data[:] = values
        self.t.assign(checkpoint_time)
        self.dt.assign(checkpoint_dt)
        self._lower_limit = checkpoint_lower_limit
        for initial_trace_term in self._initial_trace_terms:
            initial_trace_term.assign(0.0)
        self._solve_count = solve_count
        self._failure_count = failure_count
        self._nonlinear_iterations = nonlinear_iterations
        self._linear_iterations = linear_iterations
        self._last_step_size = last_step_size
        self._coefficient_step_size = None
        self._coefficient_previous_step_size = None
        self._recurrence_coefficients = None
        self._rebuild_solver = True


class ExponentialMemoryCompatibilityWarning(UserWarning):
    """Warn that bounded-kernel evolution requires compatible initial data."""


@dataclass(frozen=True)
class _StepperDispatch:
    factory: Callable[..., Any]
    permitted_formulations: tuple[type[Any], ...]
    default_formulation: Callable[[], Any]
    permits_exponential: bool


def _modal_stepper_factory(*args: Any, **kwargs: Any) -> Any:
    return _ModalStepper(*args, **kwargs)


def _auxiliary_ode_stepper_factory(*args: Any, **kwargs: Any) -> Any:
    from yonderdrake.time.auxiliary_ode import AuxiliaryODEStepper

    return AuxiliaryODEStepper(*args, **kwargs)


def _diffusive_stepper_factory(
    *args: Any,
    formulation: Any,
    **kwargs: Any,
) -> Any:
    factory = (
        _modal_stepper_factory
        if isinstance(formulation, Recurrence)
        else _auxiliary_ode_stepper_factory
    )
    return factory(*args, formulation=formulation, **kwargs)


def _full_history_stepper_factory(*args: Any, **kwargs: Any) -> Any:
    from yonderdrake.time.full_history import FullHistoryStepper

    return FullHistoryStepper(*args, **kwargs)


def _lubich_cq_stepper_factory(*args: Any, **kwargs: Any) -> Any:
    from yonderdrake.time.full_history import (
        LubichCQStepper,
    )

    return LubichCQStepper(*args, **kwargs)


def _alikhanov_stepper_factory(*args: Any, **kwargs: Any) -> Any:
    from yonderdrake.time.full_history import AlikhanovL21SigmaStepper

    return AlikhanovL21SigmaStepper(*args, **kwargs)


def _fast_oblivious_cq_stepper_factory(*args: Any, **kwargs: Any) -> Any:
    from yonderdrake.time.fast_cq import FastObliviousCQStepper

    return FastObliviousCQStepper(*args, **kwargs)


def _linear_recurrence() -> Recurrence:
    return Recurrence(interpolant="linear")


_DIFFUSIVE_DISPATCH = _StepperDispatch(
    factory=_diffusive_stepper_factory,
    permitted_formulations=(Recurrence, AuxiliaryODE),
    default_formulation=Recurrence,
    permits_exponential=True,
)
_TIME_STEPPER_DISPATCH: dict[type[Any], _StepperDispatch] = {
    type(None): _DIFFUSIVE_DISPATCH,
    BirkSong: _DIFFUSIVE_DISPATCH,
    Diethelm2008: _DIFFUSIVE_DISPATCH,
    Diethelm2022: _DIFFUSIVE_DISPATCH,
    YuanAgrawal: _DIFFUSIVE_DISPATCH,
    FullHistory: _StepperDispatch(
        factory=_full_history_stepper_factory,
        permitted_formulations=(Recurrence,),
        default_formulation=_linear_recurrence,
        permits_exponential=False,
    ),
    LubichCQ: _StepperDispatch(
        factory=_lubich_cq_stepper_factory,
        permitted_formulations=(Recurrence,),
        default_formulation=_linear_recurrence,
        permits_exponential=False,
    ),
    AlikhanovL21Sigma: _StepperDispatch(
        factory=_alikhanov_stepper_factory,
        permitted_formulations=(Recurrence,),
        default_formulation=_linear_recurrence,
        permits_exponential=False,
    ),
    FastObliviousCQ: _StepperDispatch(
        factory=_fast_oblivious_cq_stepper_factory,
        permitted_formulations=(Recurrence,),
        default_formulation=_linear_recurrence,
        permits_exponential=False,
    ),
    SineDiffusive: _StepperDispatch(
        factory=_modal_stepper_factory,
        permitted_formulations=(Oscillator,),
        default_formulation=Oscillator,
        permits_exponential=False,
    ),
    SumOfExponentials: _DIFFUSIVE_DISPATCH,
}


def _construct_time_stepper(
    F: Any,
    representation: Any | None,
    t: Any,
    dt: Any,
    u: Any,
    *,
    formulation: Any = None,
    u0: Any = None,
    bcs: Any = None,
    solver_parameters: Any = None,
    appctx: Any = None,
    allow_exponential: bool,
    warn_initial_compatibility: bool,
) -> Any:
    from yonderdrake.time._ufl_marker import (
        CaputoDerivativeMarker,
        ExponentialMemoryMarker,
        RiemannLiouvilleDerivativeMarker,
        find_time_memory_markers,
    )

    dispatch = _TIME_STEPPER_DISPATCH.get(type(representation))
    if dispatch is None:
        supported = ", ".join(
            representation_type.__name__
            for representation_type in _TIME_STEPPER_DISPATCH
            if representation_type is not type(None)
        )
        raise TypeError(f"representation must be one of: {supported}")
    if formulation is None:
        formulation = dispatch.default_formulation()
    formulation_types = tuple(
        dict.fromkeys(
            formulation_type
            for candidate in _TIME_STEPPER_DISPATCH.values()
            for formulation_type in candidate.permitted_formulations
        )
    )
    if not isinstance(formulation, formulation_types):
        names = ", ".join(kind.__name__ for kind in formulation_types)
        raise TypeError(f"formulation must be one of: {names}")
    if not isinstance(formulation, dispatch.permitted_formulations):
        if dispatch.permitted_formulations == (Recurrence,):
            raise NotImplementedError(
                f"{type(representation).__name__} supports only the "
                "eliminated Recurrence formulation, not "
                f"{type(formulation).__name__}"
            )
        raise NotImplementedError(
            f"{type(representation).__name__} does not support "
            f"{type(formulation).__name__}"
        )
    if (
        dispatch.factory is not _diffusive_stepper_factory
        and isinstance(formulation, Recurrence)
        and formulation.interpolant != "linear"
    ):
        raise NotImplementedError(
            f"{type(representation).__name__} defines its own history rule and "
            "does not use the Recurrence interpolant"
        )

    markers = find_time_memory_markers(F)
    has_exponential = any(
        isinstance(marker, ExponentialMemoryMarker) for marker in markers
    )
    has_fractional = any(
        isinstance(
            marker,
            (CaputoDerivativeMarker, RiemannLiouvilleDerivativeMarker),
        )
        for marker in markers
    )
    if has_exponential and not allow_exponential:
        raise ValueError("ExponentialMemory requires TimeMemoryStepper")
    if has_exponential and not dispatch.permits_exponential:
        raise NotImplementedError(
            f"{type(representation).__name__} cannot be combined with "
            "ExponentialMemory"
        )
    if has_fractional and representation is None:
        raise TypeError(
            "a representation is required for Caputo and "
            "Riemann-Liouville markers"
        )
    if has_exponential and warn_initial_compatibility:
        warnings.warn(
            "ExponentialMemory is zero at the initial time. If it is used as "
            "the leading evolution operator, the remaining residual and "
            "initial data must satisfy the corresponding compatibility "
            "condition.",
            ExponentialMemoryCompatibilityWarning,
            stacklevel=3,
        )
    return dispatch.factory(
        F,
        representation,
        t,
        dt,
        u,
        formulation=formulation,
        u0=u0,
        bcs=bcs,
        solver_parameters=solver_parameters,
        appctx=appctx,
    )


def TimeMemoryStepper(
    F: Any,
    t: Any,
    dt: Any,
    u: Any,
    *,
    representation: Any = None,
    formulation: Any = None,
    u0: Any = None,
    bcs: Any = None,
    solver_parameters: Any = None,
    appctx: Any = None,
    warn_initial_compatibility: bool = True,
) -> Any:
    """Advance exponential memory and optional fractional time markers."""
    return _construct_time_stepper(
        F,
        representation,
        t,
        dt,
        u,
        formulation=formulation,
        u0=u0,
        bcs=bcs,
        solver_parameters=solver_parameters,
        appctx=appctx,
        allow_exponential=True,
        warn_initial_compatibility=warn_initial_compatibility,
    )


def FractionalTimeStepper(
    F: Any,
    representation: Any,
    t: Any,
    dt: Any,
    u: Any,
    *,
    formulation: Any = None,
    u0: Any = None,
    bcs: Any = None,
    solver_parameters: Any = None,
    appctx: Any = None,
) -> Any:
    """Construct a native formulation for a fractional derivative marker."""
    return _construct_time_stepper(
        F,
        representation,
        t,
        dt,
        u,
        formulation=formulation,
        u0=u0,
        bcs=bcs,
        solver_parameters=solver_parameters,
        appctx=appctx,
        allow_exponential=False,
        warn_initial_compatibility=False,
    )
