"""Direct full-history fractional time stepping."""

from __future__ import annotations

import warnings
from functools import cache
from math import fsum, gamma, isclose, isfinite
from typing import Any

import numpy as np
from scipy.linalg import qr, solve_triangular

from yonderdrake._firedrake import require_real_float64_petsc
from yonderdrake.time._stepper_lifecycle import StepperLifecycle
from yonderdrake.time.checkpointing import (
    inspect_checkpoint_file,
    load_checkpoint_file,
    save_checkpoint_file,
    stepper_metadata,
    validate_stepper_metadata,
)
from yonderdrake.time.representations import (
    AlikhanovL21Sigma,
    FullHistory,
    LubichCQ,
    StartingCorrectionAdvisoryWarning,
    validate_checkpoint_representation,
)
from yonderdrake.time.representations.core import _STARTING_CONDITION_ADVISORY


def _lubich_cq_weights(alpha: float, order: str, count: int) -> np.ndarray:
    """Return coefficients of the selected BDF symbol to ``alpha``."""
    if count < 0:
        raise ValueError("count must be nonnegative")
    weights = np.empty(count, dtype=np.float64)
    if count == 0:
        return weights
    if order == "bdf1":
        weights[0] = 1.0
        for index in range(1, count):
            weights[index] = (
                weights[index - 1] * (index - 1.0 - alpha) / index
            )
        return weights
    if order != "bdf2":
        raise ValueError("order must be 'bdf1' or 'bdf2'")

    symbol = (1.5, -2.0, 0.5)
    weights[0] = symbol[0] ** alpha
    # Miller's power-series recurrence avoids the cancellation of extracting
    # coefficients with an FFT on a circle.
    for index in range(1, count):
        numerator = 0.0
        for coefficient_index in range(1, min(index, 2) + 1):
            numerator += (
                ((alpha + 1.0) * coefficient_index - index)
                * symbol[coefficient_index]
                * weights[index - coefficient_index]
            )
        weights[index] = numerator / (index * symbol[0])
    return weights


@cache
def _lubich_starting_factors(
    alpha: float,
    num_corrections: int,
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """Cache float64 QR factors for every active correction system."""
    factors = []
    condition = 1.0
    for active in range(1, num_corrections + 1):
        indices = np.arange(1, active + 1, dtype=np.float64)
        exponents = alpha * indices
        matrix = np.power(indices[None, :], exponents[:, None])
        condition = float(np.linalg.cond(matrix, p=np.inf))
        orthogonal, triangular = qr(
            matrix,
            mode="economic",
            check_finite=False,
        )
        transpose = np.asarray(orthogonal.T)
        transpose.setflags(write=False)
        triangular.setflags(write=False)
        factors.append((transpose, triangular))
    if condition > _STARTING_CONDITION_ADVISORY:
        warnings.warn(
            f"LubichCQ num_corrections={num_corrections} gives condition "
            f"{condition:.2e} at alpha={alpha}; float64 correction accuracy "
            "may be reduced.",
            StartingCorrectionAdvisoryWarning,
            stacklevel=3,
        )
    return tuple(factors)


def _lubich_starting_weights(
    alpha: float,
    cq_weights: np.ndarray,
    step: int,
    num_corrections: int,
) -> np.ndarray:
    """Solve Lubich's power-exact starting system at one time level."""
    active = min(step, num_corrections)
    if active == 0:
        return np.empty(0, dtype=np.float64)
    if cq_weights.shape != (step + 1,):
        raise ValueError("cq_weights must contain coefficients 0 through step")

    right = np.empty(active, dtype=np.float64)
    for row in range(active):
        exponent = (row + 1) * alpha
        exact = (
            gamma(exponent + 1.0)
            / gamma(exponent + 1.0 - alpha)
            * step ** (exponent - alpha)
        )
        uncorrected = fsum(
            float(cq_weights[step - sample]) * sample**exponent
            for sample in range(1, step + 1)
        )
        right[row] = exact - uncorrected
    transpose, triangular = _lubich_starting_factors(
        alpha,
        num_corrections,
    )[active - 1]
    result = solve_triangular(
        triangular,
        transpose @ right,
        check_finite=False,
    )
    if not np.all(np.isfinite(result)):
        raise ValueError(
            "Lubich starting weights are not finite for "
            f"alpha={alpha}, step={step}, and num_corrections={num_corrections}"
        )
    return result


def _alikhanov_b_coefficient(alpha: float, lag: int) -> float:
    """Evaluate one L2-1-sigma quadratic correction without cancellation."""
    sigma = 1.0 - 0.5 * alpha
    center = lag + sigma - 0.5
    half_width = 0.5 / center
    first_power = 1.0 - alpha
    second_power = 2.0 - alpha
    first_binomial = 1.0
    second_binomial = 1.0
    power = 1.0
    scaled = 0.0
    for degree in range(1, 257):
        power *= half_width
        first_binomial *= (first_power - degree + 1.0) / degree
        second_binomial *= (second_power - degree + 1.0) / degree
        term = 0.0
        if degree > 1 and degree % 2 == 0:
            term -= first_binomial * power
        elif degree > 1:
            term += (
                2.0 * center / second_power * second_binomial * power
            )
        scaled += term
        if degree > 12 and abs(term) <= 2.0e-17 * max(abs(scaled), 1.0):
            break
    return float(center**first_power * scaled)


def _alikhanov_increment_weights(alpha: float, step: int) -> np.ndarray:
    """Return L2-1-sigma weights ordered from oldest to current increment."""
    if step < 1:
        raise ValueError("step must be positive")
    sigma = 1.0 - 0.5 * alpha
    level = step - 1
    if level == 0:
        return np.asarray([sigma ** (1.0 - alpha)], dtype=np.float64)

    lags = np.arange(level + 1, dtype=np.float64)
    a = np.empty(level + 1, dtype=np.float64)
    a[0] = sigma ** (1.0 - alpha)
    upper = lags[1:] + sigma
    lower = lags[1:] - 1.0 + sigma
    power = 1.0 - alpha
    a[1:] = np.power(upper, power) * (
        -np.expm1(power * np.log(lower / upper))
    )
    b = np.zeros(level + 1, dtype=np.float64)
    b[1:] = np.asarray(
        [_alikhanov_b_coefficient(alpha, lag) for lag in range(1, level + 1)],
        dtype=np.float64,
    )

    by_lag = np.empty(level + 1, dtype=np.float64)
    by_lag[0] = a[0] + b[1]
    if level > 1:
        by_lag[1:level] = a[1:level] + b[2:] - b[1:level]
    by_lag[level] = a[level] - b[level]
    return by_lag[::-1]


class FullHistoryStepper(StepperLifecycle):
    """Advance time-derivative markers with direct linear history."""

    def __init__(
        self,
        F: Any,
        representation: FullHistory | LubichCQ,
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
                "FractionalTimeStepper requires an active Firedrake environment"
            ) from error
        require_real_float64_petsc()
        from yonderdrake.time._ufl_marker import (
            RiemannLiouvilleDerivativeMarker,
            find_fractional_derivative_markers,
        )

        markers = find_fractional_derivative_markers(F)
        if not markers:
            raise ValueError(
                "the full-history stepper requires at least one "
                "fractional derivative marker"
            )
        if any(marker.field is not u for marker in markers):
            raise ValueError(
                "every fractional derivative marker must wrap "
                "the stepper solution u"
            )
        element = u.function_space().ufl_element()
        if element.family() != "Lagrange":
            raise NotImplementedError(
                "FullHistory stepping supports continuous Lagrange spaces only"
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
        self._alpha_operands = tuple(marker.alpha for marker in markers)
        self._operator_kinds = tuple(
            (
                "riemann_liouville"
                if isinstance(marker, RiemannLiouvilleDerivativeMarker)
                else "caputo"
            )
            for marker in markers
        )
        self.alphas = self._read_alphas()
        self.alpha = self.alphas[0] if len(self.alphas) == 1 else self.alphas
        self._space = u.function_space()

        if u0 is not None:
            self.u.assign(u0)
        self._previous = fd.Function(self._space, name="caputo_previous")
        self._previous.assign(self.u)
        self._initial = fd.Function(self._space, name="fractional_initial")
        self._initial.assign(self.u)
        self._committed_u = fd.Function(self._space, name="caputo_committed")
        self._history_terms = tuple(
            fd.Function(
                self._space,
                name=f"caputo_history_term_{term:02d}",
            )
            for term in range(len(markers))
        )
        for history_term in self._history_terms:
            history_term.assign(0.0)
        self._implicit_weights = tuple(fd.Constant(0.0) for _ in markers)
        self._initial_trace_terms = tuple(
            fd.Function(
                self._space,
                name=f"fractional_initial_trace_term_{term:02d}",
            )
            for term in range(len(markers))
        )
        for initial_trace_term in self._initial_trace_terms:
            initial_trace_term.assign(0.0)
        self._transformed_residual = self._build_transformed_residual()

        initial_time = self._read_time()
        self._times = [initial_time]
        self._local_shape = self.u.dat.data_ro.shape
        self._local_size = int(np.prod(self._local_shape, dtype=np.int64))
        self._history_values = np.empty(
            (0, self._local_size),
            dtype=np.float64,
        )
        self._history_count = 0
        self._solver: Any = None
        self._rebuild_solver = True
        self._last_step_size: float | None = None
        self._reset_solver_counters()

    def _marker_replacements(self) -> dict[Any, Any]:
        return {
            marker: (
                history_term
                + implicit_weight * (self.u - self._previous)
                + initial_trace_term
            )
            for marker, history_term, implicit_weight, initial_trace_term in zip(
                self._markers,
                self._history_terms,
                self._implicit_weights,
                self._initial_trace_terms,
                strict=True,
            )
        }

    def _build_transformed_residual(self) -> Any:
        from yonderdrake.time._ufl_marker import (
            evaluate_form_at_end_time,
            replace_fractional_derivative_markers,
        )

        end_time_form = evaluate_form_at_end_time(self.F, self.t, self.dt)
        replacements = self._marker_replacements()
        return replace_fractional_derivative_markers(
            end_time_form,
            replacements.__getitem__,
        )

    def _ensure_history_capacity(self, required: int) -> None:
        current = self._history_values.shape[0]
        if current >= required:
            return
        capacity = max(required, 8 if current == 0 else 2 * current)
        expanded = np.empty((capacity, self._local_size), dtype=np.float64)
        if self._history_count:
            expanded[: self._history_count] = self._history_values[
                : self._history_count
            ]
        self._history_values = expanded

    def _assign_history_matvec(
        self,
        history_term: Any,
        weights: np.ndarray,
    ) -> None:
        if weights.shape != (self._history_count,):
            raise ValueError("history weights do not match the committed history")
        target = history_term.dat.data.reshape(-1)
        if self._history_count == 0:
            target.fill(0.0)
            return
        np.matmul(
            weights,
            self._history_values[: self._history_count],
            out=target,
        )

    @staticmethod
    def _read_alpha_operand(alpha_operand: Any) -> float:
        try:
            alpha = float(alpha_operand)
        except (TypeError, ValueError) as error:
            raise TypeError("alpha must be a real scalar") from error
        if not isfinite(alpha) or not 0.0 < alpha < 1.0:
            raise ValueError("alpha must satisfy 0 < alpha < 1")
        return alpha

    def _read_alphas(self) -> tuple[float, ...]:
        return tuple(
            self._read_alpha_operand(operand)
            for operand in self._alpha_operands
        )

    @staticmethod
    def _same_time(left: float, right: float) -> bool:
        return isclose(left, right, rel_tol=1.0e-13, abs_tol=1.0e-14)

    def _require_current_time(self) -> float:
        current = self._read_time()
        if not self._same_time(current, self._times[-1]):
            raise RuntimeError(
                "t must be advanced after each successful FullHistory step"
            )
        return current

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

    def _prepare_step(self, current: float, step_size: float) -> float:
        target = current + step_size
        if not isfinite(target) or target <= current:
            raise ValueError("t + dt must be finite and greater than t")
        if len(self._times) != self._history_count + 1:
            raise RuntimeError("full-history state is inconsistent")

        starts = np.asarray(self._times[:-1], dtype=np.float64)
        ends = np.asarray(self._times[1:], dtype=np.float64)
        widths = ends - starts
        for kind, alpha, history_term, implicit_weight, initial_trace_term in zip(
            self._operator_kinds,
            self.alphas,
            self._history_terms,
            self._implicit_weights,
            self._initial_trace_terms,
            strict=True,
        ):
            power = 1.0 - alpha
            normalization = gamma(2.0 - alpha)
            if self._history_count:
                left_ages = target - starts
                right_ages = target - ends
                differences = np.power(left_ages, power) * (
                    -np.expm1(power * np.log(right_ages / left_ages))
                )
                weights = differences / (widths * normalization)
            else:
                weights = np.empty(0, dtype=np.float64)
            self._assign_history_matvec(history_term, weights)
            implicit_weight.assign(step_size ** (-alpha) / normalization)
            if kind == "riemann_liouville":
                elapsed = target - self._times[0]
                initial_trace_term.assign(
                    elapsed ** (-alpha) / gamma(1.0 - alpha) * self._initial
                )
        self._last_step_size = step_size
        return target

    def advance(self) -> None:
        """Solve for the next state and append its full-history increment."""
        if self._read_alphas() != self.alphas:
            raise RuntimeError(
                "changing alpha after construction is unsupported; rebuild the stepper"
            )
        current = self._require_current_time()
        step_size = self._read_step_size()
        target = self._prepare_step(current, step_size)
        if self._rebuild_solver:
            self._build_solver()

        self._committed_u.assign(self.u)
        try:
            self._solver.solve()
        except Exception:
            self.u.assign(self._committed_u)
            self._failure_count += 1
            raise

        self._ensure_history_capacity(self._history_count + 1)
        self._history_values[self._history_count] = (
            self.u.dat.data_ro.reshape(-1)
            - self._previous.dat.data_ro.reshape(-1)
        )
        self._history_count += 1
        self._times.append(target)
        self._previous.assign(self.u)
        self._solve_count += 1
        self._nonlinear_iterations += self._solver.snes.getIterationNumber()
        self._linear_iterations += self._solver.snes.getLinearSolveIterations()

    @property
    def history(self) -> tuple[Any, ...]:
        """Return defensive copies of committed solution increments."""
        increments = []
        for index in range(self._history_count):
            increment = self._fd.Function(
                self._space,
                name=f"caputo_increment_{index:08d}",
            )
            increment.dat.data[:] = self._history_values[index].reshape(
                self._local_shape
            )
            increments.append(increment)
        return tuple(increments)

    @property
    def transformed_residual(self) -> Any:
        return self._transformed_residual

    def solver_stats(self) -> dict[str, Any]:
        return {
            "solves": self._solve_count,
            "failures": self._failure_count,
            "nonlinear_iterations": self._nonlinear_iterations,
            "linear_iterations": self._linear_iterations,
            "num_fractional_terms": len(self._markers),
            "history_steps": self._history_count,
            "stored_history_fields": self._history_count,
            "last_step_size": self._last_step_size,
        }

    def reset(self, u0: Any, t0: Any = None) -> None:
        self.u.assign(u0)
        self._previous.assign(self.u)
        self._initial.assign(self.u)
        self._history_count = 0
        for history_term in self._history_terms:
            history_term.assign(0.0)
        for initial_trace_term in self._initial_trace_terms:
            initial_trace_term.assign(0.0)
        if t0 is not None:
            self.t.assign(t0)
        self._times = [self._read_time()]
        self._last_step_size = None
        self._reset_solver_counters()

    def _formulation_metadata(self) -> dict[str, Any]:
        return {"kind": "recurrence", "interpolant": "linear"}

    def _checkpoint_metadata(self) -> dict[str, Any]:
        current = self._require_current_time()
        payload = stepper_metadata(
            kind="full_history",
            operator_kinds=self._operator_kinds,
            parameters=self.alphas,
            representations=[
                self.representation.describe(alpha)
                for alpha in self.alphas
            ],
            formulation=self._formulation_metadata(),
        )
        payload.update({
            "times": self._times,
            "history_count": self._history_count,
            "time": current,
            "dt": float(self.dt),
            "stats": self.solver_stats(),
        })
        return payload

    def checkpoint_state(self) -> dict[str, Any]:
        payload = self._checkpoint_metadata()
        payload.update(
            {
                "u": self.u.dat.data_ro.tolist(),
                "initial": self._initial.dat.data_ro.tolist(),
                "previous": self._previous.dat.data_ro.tolist(),
                "increments": [
                    self._history_values[index]
                    .reshape(self._local_shape)
                    .tolist()
                    for index in range(self._history_count)
                ],
            }
        )
        return payload

    def _checkpoint_file_fields(self) -> dict[str, Any]:
        fields = {
            "u": self.u,
            "initial": self._initial,
            "previous": self._previous,
        }
        for index, increment in enumerate(self.history):
            fields[f"increment_{index:08d}"] = increment
        return fields

    def save_checkpoint(self, checkpoint: Any, *, name: str = "state") -> None:
        save_checkpoint_file(
            checkpoint,
            name=name,
            metadata=self._checkpoint_metadata(),
            fields=self._checkpoint_file_fields(),
        )

    def load_checkpoint(self, checkpoint: Any, *, name: str = "state") -> None:
        metadata, _ = inspect_checkpoint_file(checkpoint, name=name)
        try:
            history_count = int(metadata["history_count"])
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            raise ValueError("checkpoint history count is invalid") from error
        if history_count < 0:
            raise ValueError("checkpoint history count is invalid")
        expected_fields = ("u", "initial", "previous") + tuple(
            f"increment_{index:08d}" for index in range(history_count)
        )
        state, loaded = load_checkpoint_file(
            checkpoint,
            name=name,
            mesh=self._space.mesh(),
            expected_fields=expected_fields,
        )
        state["u"] = loaded["u"].dat.data_ro.tolist()
        state["initial"] = loaded["initial"].dat.data_ro.tolist()
        state["previous"] = loaded["previous"].dat.data_ro.tolist()
        state["increments"] = [
            loaded[f"increment_{index:08d}"].dat.data_ro.tolist()
            for index in range(history_count)
        ]
        self.restore_checkpoint(state)

    def restore_checkpoint(self, state: dict[str, Any]) -> None:
        representations = validate_stepper_metadata(
            state,
            kind="full_history",
            operator_kinds=self._operator_kinds,
            parameters=self.alphas,
        )
        formulation = dict(state.get("formulation") or {})
        if formulation != self._formulation_metadata():
            raise ValueError("checkpoint formulation does not match")
        for metadata, alpha in zip(
            representations,
            self.alphas,
            strict=True,
        ):
            validate_checkpoint_representation(
                metadata,
                self.representation,
                alpha,
            )

        increments = state.get("increments")
        if not isinstance(increments, list):
            raise ValueError("checkpoint history is invalid")
        try:
            history_count = int(state["history_count"])
            times = [float(value) for value in state["times"]]
            checkpoint_time = float(state["time"])
            checkpoint_dt = float(state["dt"])
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            raise ValueError("checkpoint time metadata is invalid") from error
        if (
            history_count != len(increments)
            or len(times) != history_count + 1
            or not all(isfinite(value) for value in times)
            or any(
                right <= left
                for left, right in zip(
                    times[:-1],
                    times[1:],
                    strict=True,
                )
            )
            or not self._same_time(checkpoint_time, times[-1])
        ):
            raise ValueError("checkpoint history times are invalid")
        if not isfinite(checkpoint_dt) or checkpoint_dt <= 0.0:
            raise ValueError("checkpoint dt must be finite and positive")

        physical = np.asarray(state.get("u"), dtype=np.float64)
        initial = np.asarray(state.get("initial"), dtype=np.float64)
        previous = np.asarray(state.get("previous"), dtype=np.float64)
        increment_arrays = [
            np.asarray(values, dtype=np.float64) for values in increments
        ]
        local_shape = self._local_shape
        if (
            physical.shape != local_shape
            or initial.shape != local_shape
            or previous.shape != local_shape
            or any(values.shape != local_shape for values in increment_arrays)
        ):
            raise ValueError("checkpoint history field has the wrong local shape")
        if not all(
            np.all(np.isfinite(values))
            for values in (physical, initial, previous, *increment_arrays)
        ):
            raise ValueError("checkpoint fields must contain finite values")

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
        if (
            solve_count != history_count
            or min(
                solve_count,
                failure_count,
                nonlinear_iterations,
                linear_iterations,
            )
            < 0
        ):
            raise ValueError("checkpoint solver statistics are invalid")
        if last_step_size is not None and (
            not isfinite(last_step_size) or last_step_size <= 0.0
        ):
            raise ValueError(
                "checkpoint last step size must be finite and positive"
            )

        self.u.dat.data[:] = physical
        self._initial.dat.data[:] = initial
        self._previous.dat.data[:] = previous
        self._history_values = np.empty(
            (history_count, self._local_size),
            dtype=np.float64,
        )
        for index, values in enumerate(increment_arrays):
            self._history_values[index] = values.reshape(-1)
        self._history_count = history_count
        self._times = times
        self.t.assign(checkpoint_time)
        self.dt.assign(checkpoint_dt)
        for history_term in self._history_terms:
            history_term.assign(0.0)
        for initial_trace_term in self._initial_trace_terms:
            initial_trace_term.assign(0.0)
        self._solve_count = solve_count
        self._failure_count = failure_count
        self._nonlinear_iterations = nonlinear_iterations
        self._linear_iterations = linear_iterations
        self._last_step_size = last_step_size
        self._rebuild_solver = True


class LubichCQStepper(FullHistoryStepper):
    """Advance a uniform-grid Lubich BDF convolution quadrature."""

    representation: LubichCQ

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._uniform_step_size: float | None = None
        self._cq_weight_cache: dict[float, list[float]] = {}
        super().__init__(*args, **kwargs)

    def _weights_through(self, alpha: float, step: int) -> np.ndarray:
        weights = self._cq_weight_cache.setdefault(alpha, [])
        order = self.representation.order
        if not weights:
            weights.append(1.0 if order == "bdf1" else 1.5**alpha)
        while len(weights) <= step:
            index = len(weights)
            if order == "bdf1":
                value = weights[-1] * (index - 1.0 - alpha) / index
            else:
                numerator = (
                    ((alpha + 1.0) - index) * -2.0 * weights[index - 1]
                )
                if index >= 2:
                    numerator += (
                        ((alpha + 1.0) * 2.0 - index)
                        * 0.5
                        * weights[index - 2]
                    )
                value = numerator / (1.5 * index)
            weights.append(value)
        return np.asarray(weights[: step + 1], dtype=np.float64)

    def _prepare_step(self, current: float, step_size: float) -> float:
        target = current + step_size
        if not isfinite(target) or target <= current:
            raise ValueError("t + dt must be finite and greater than t")
        if len(self._times) != self._history_count + 1:
            raise RuntimeError("convolution-quadrature history is inconsistent")
        if self._uniform_step_size is None:
            self._uniform_step_size = step_size
        elif not self._same_time(step_size, self._uniform_step_size):
            raise ValueError(
                "LubichCQ requires a uniform time step"
            )

        step = self._history_count + 1
        corrections = self.representation.num_corrections
        assert corrections is not None
        for kind, alpha, history_term, implicit_weight, initial_trace_term in zip(
            self._operator_kinds,
            self.alphas,
            self._history_terms,
            self._implicit_weights,
            self._initial_trace_terms,
            strict=True,
        ):
            cq_weights = self._weights_through(alpha, step)
            increment_weights = np.cumsum(cq_weights[:-1])[::-1]
            starting = _lubich_starting_weights(
                alpha,
                cq_weights,
                step,
                corrections,
            )
            for increment_index in range(starting.size):
                increment_weights[increment_index] += float(
                    np.sum(starting[increment_index:])
                )
            scale = step_size ** (-alpha)
            self._assign_history_matvec(
                history_term,
                scale * increment_weights[:-1],
            )
            implicit_weight.assign(scale * increment_weights[-1])
            if kind == "riemann_liouville":
                initial_trace_term.assign(
                    scale * float(np.sum(cq_weights)) * self._initial
                )
        self._last_step_size = step_size
        return target

    def solver_stats(self) -> dict[str, Any]:
        stats = super().solver_stats()
        stats.update(
            {
                "history_method": self.representation.order,
                "num_starting_corrections": (
                    self.representation.num_corrections
                ),
                "work_class": "O(N^2) total",
                "storage_class": "O(N) fields",
            }
        )
        return stats

    def reset(self, u0: Any, t0: Any = None) -> None:
        super().reset(u0, t0=t0)
        self._uniform_step_size = None

    def restore_checkpoint(self, state: dict[str, Any]) -> None:
        super().restore_checkpoint(state)
        if self._history_count:
            widths = np.diff(np.asarray(self._times, dtype=np.float64))
            if not np.allclose(
                widths,
                widths[0],
                rtol=1.0e-13,
                atol=1.0e-14,
            ):
                raise ValueError(
                    "LubichCQ checkpoint steps are not uniform"
                )
            self._uniform_step_size = float(widths[0])
        else:
            self._uniform_step_size = None


class AlikhanovL21SigmaStepper(FullHistoryStepper):
    """Advance Alikhanov's L2-1-sigma formula at its offset time."""

    representation: AlikhanovL21Sigma

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._uniform_step_size: float | None = None
        super().__init__(*args, **kwargs)

    def _build_transformed_residual(self) -> Any:
        import ufl

        if any(kind != "caputo" for kind in self._operator_kinds):
            raise NotImplementedError(
                "AlikhanovL21Sigma is defined for Caputo markers only"
            )
        if len(set(self.alphas)) != 1:
            raise NotImplementedError(
                "AlikhanovL21Sigma requires one shared alpha because the "
                "residual has one offset evaluation time"
            )
        sigma = 1.0 - 0.5 * self.alphas[0]
        replacements = self._marker_replacements()
        replacements[self.u] = sigma * self.u + (1.0 - sigma) * self._previous
        if isinstance(self.t, ufl.core.expr.Expr):
            replacements[self.t] = self.t + sigma * self.dt
        return ufl.replace(self.F, replacements)

    def _prepare_step(self, current: float, step_size: float) -> float:
        target = current + step_size
        if not isfinite(target) or target <= current:
            raise ValueError("t + dt must be finite and greater than t")
        if len(self._times) != self._history_count + 1:
            raise RuntimeError("L2-1-sigma history is inconsistent")
        if self._uniform_step_size is None:
            self._uniform_step_size = step_size
        elif not self._same_time(step_size, self._uniform_step_size):
            raise ValueError("AlikhanovL21Sigma requires a uniform time step")

        step = self._history_count + 1
        for alpha, history_term, implicit_weight in zip(
            self.alphas,
            self._history_terms,
            self._implicit_weights,
            strict=True,
        ):
            increment_weights = _alikhanov_increment_weights(alpha, step)
            scale = step_size ** (-alpha) / gamma(2.0 - alpha)
            self._assign_history_matvec(
                history_term,
                scale * increment_weights[:-1],
            )
            implicit_weight.assign(scale * increment_weights[-1])
        self._last_step_size = step_size
        return target

    def _formulation_metadata(self) -> dict[str, Any]:
        return {
            "kind": "direct-history",
            "interpolant": "quadratic",
            "evaluation": "n+sigma",
        }

    def solver_stats(self) -> dict[str, Any]:
        stats = super().solver_stats()
        stats.update(
            {
                "history_method": "L2-1-sigma",
                "evaluation_offset": 1.0 - 0.5 * self.alphas[0],
                "work_class": "O(N^2) total",
                "storage_class": "O(N) fields",
            }
        )
        return stats

    def reset(self, u0: Any, t0: Any = None) -> None:
        super().reset(u0, t0=t0)
        self._uniform_step_size = None

    def restore_checkpoint(self, state: dict[str, Any]) -> None:
        super().restore_checkpoint(state)
        if self._history_count:
            widths = np.diff(np.asarray(self._times, dtype=np.float64))
            if not np.allclose(
                widths,
                widths[0],
                rtol=1.0e-13,
                atol=1.0e-14,
            ):
                raise ValueError(
                    "AlikhanovL21Sigma checkpoint steps are not uniform"
                )
            self._uniform_step_size = float(widths[0])
        else:
            self._uniform_step_size = None
