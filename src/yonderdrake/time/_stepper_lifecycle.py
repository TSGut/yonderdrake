"""Shared lifecycle operations for time-memory steppers."""

from __future__ import annotations

from math import isfinite


class StepperLifecycle:
    """Validation and counters shared outside the numerical update paths."""

    def _read_step_size(self) -> float:
        try:
            step_size = float(self.dt)  # type: ignore[attr-defined]
        except (TypeError, ValueError) as error:
            raise TypeError("dt must be a real scalar") from error
        if not isfinite(step_size) or step_size <= 0.0:
            raise ValueError("dt must be finite and positive")
        return step_size

    def _read_time(self) -> float:
        try:
            time = float(self.t)  # type: ignore[attr-defined]
        except (TypeError, ValueError) as error:
            raise TypeError("t must be a real scalar") from error
        if not isfinite(time):
            raise ValueError("t must be finite")
        return time

    def _reset_solver_counters(self) -> None:
        self._solve_count = 0
        self._failure_count = 0
        self._nonlinear_iterations = 0
        self._linear_iterations = 0

    def invalidate_jacobian(self) -> None:
        self._rebuild_solver = True
