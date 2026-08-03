"""Fast and oblivious convolution quadrature stepping."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import gamma, isclose, isfinite
from typing import Any

import numpy as np

from yonderdrake.time.checkpointing import (
    inspect_checkpoint_file,
    load_checkpoint_file,
    save_checkpoint_file,
    stepper_metadata,
    validate_stepper_metadata,
)
from yonderdrake.time.full_history import FullHistoryStepper
from yonderdrake.time.representations import (
    FastObliviousCQ,
    validate_checkpoint_representation,
)


@dataclass
class _CompletedBlock:
    end: int
    real: np.ndarray
    imaginary: np.ndarray


@dataclass
class _ContourLevel:
    size: int
    resolvent: np.ndarray
    weights: np.ndarray
    real: np.ndarray
    imaginary: np.ndarray
    scratch: np.ndarray
    completed: deque[_CompletedBlock]
    active_start: int | None = None


class _FastCQHistory:
    """One local distributed-field history compressed on dyadic contours."""

    def __init__(
        self,
        representation: FastObliviousCQ,
        alpha: float,
        step_size: float,
        local_size: int,
    ) -> None:
        self.representation = representation
        self.alpha = alpha
        self.step_size = step_size
        self.local_size = local_size
        self.accepted_steps = 0
        self.recent: deque[tuple[int, np.ndarray]] = deque(
            maxlen=representation.direct_steps
        )
        self.levels = []
        nodes = representation.nodes_per_level
        assert nodes is not None
        for level_index in range(representation.num_levels):
            spectrum = representation.contour_spectrum(
                alpha,
                step_size,
                level_index,
            )
            positive = slice(nodes, None)
            rates = spectrum.rates[positive]
            weights = spectrum.weights[positive]
            size = 2**level_index
            retained = representation.direct_steps // size + 5
            self.levels.append(
                _ContourLevel(
                    size=size,
                    resolvent=1.0 / (1.0 - step_size * rates),
                    weights=weights,
                    real=np.zeros((nodes + 1, local_size), dtype=np.float64),
                    imaginary=np.zeros(
                        (nodes + 1, local_size),
                        dtype=np.float64,
                    ),
                    scratch=np.empty((nodes + 1, local_size), dtype=np.float64),
                    completed=deque(maxlen=retained),
                )
            )
        ages = np.arange(representation.direct_steps + 1, dtype=np.float64)
        self._direct_weights = np.empty_like(ages)
        self._direct_weights[0] = 1.0
        for age in range(1, ages.size):
            self._direct_weights[age] = (
                self._direct_weights[age - 1]
                * (age - alpha)
                / age
            )

    def append(self, increment: np.ndarray) -> None:
        values = np.asarray(increment, dtype=np.float64).reshape(-1)
        if values.shape != (self.local_size,):
            raise ValueError("fast-CQ increment has the wrong local shape")
        step = self.accepted_steps + 1
        if step > self.representation.max_steps:
            raise RuntimeError(
                "FastObliviousCQ exceeded max_steps; "
                "increase num_levels"
            )
        self.recent.append((step, values.copy()))
        for level in self.levels:
            start = ((step - 1) // level.size) * level.size + 1
            if level.active_start != start:
                level.real.fill(0.0)
                level.imaginary.fill(0.0)
                level.active_start = start
            real_multiplier = level.resolvent.real[:, None]
            imaginary_multiplier = level.resolvent.imag[:, None]
            np.copyto(level.scratch, level.real)
            level.real[:] = (
                real_multiplier * level.real
                - imaginary_multiplier * level.imaginary
                + real_multiplier * values
            )
            level.imaginary[:] = (
                imaginary_multiplier * level.scratch
                + real_multiplier * level.imaginary
                + imaginary_multiplier * values
            )
            if step % level.size == 0:
                level.completed.append(
                    _CompletedBlock(
                        end=step,
                        real=level.real.copy(),
                        imaginary=level.imaginary.copy(),
                    )
                )
                level.active_start = None
        self.accepted_steps = step

    @staticmethod
    def _block(level: _ContourLevel, end: int) -> _CompletedBlock:
        for block in reversed(level.completed):
            if block.end == end:
                return block
        raise RuntimeError(
            "fast-CQ dyadic block is unavailable; increase direct_steps"
        )

    def past_action(self, step: int, out: np.ndarray) -> None:
        if step != self.accepted_steps + 1:
            raise RuntimeError("fast-CQ step and history are inconsistent")
        target = np.asarray(out, dtype=np.float64).reshape(-1)
        if target.shape != (self.local_size,):
            raise ValueError("fast-CQ output has the wrong local shape")
        target.fill(0.0)
        cutoff = max(0, step - self.representation.direct_steps)

        recent_indices = []
        recent_values = []
        for index, values in self.recent:
            if index > cutoff:
                recent_indices.append(index)
                recent_values.append(values)
        if recent_values:
            coefficients = np.asarray(
                [
                    self.step_size ** (-self.alpha)
                    * self._direct_weights[step - index]
                    for index in recent_indices
                ],
                dtype=np.float64,
            )
            np.matmul(coefficients, np.stack(recent_values), out=target)

        end = cutoff
        while end:
            aligned_size = end & -end
            age = step - end
            age_size = 1 << (max(age // 2, 1).bit_length() - 1)
            size = min(aligned_size, age_size)
            level_index = size.bit_length() - 1
            if level_index >= len(self.levels):
                raise RuntimeError(
                    "FastObliviousCQ exceeded its dyadic levels"
                )
            level = self.levels[level_index]
            block = self._block(level, end)
            multiplier = level.weights * np.power(level.resolvent, age)
            target += multiplier[0].real * block.real[0]
            target -= multiplier[0].imag * block.imaginary[0]
            if multiplier.size > 1:
                target += 2.0 * np.matmul(
                    multiplier[1:].real,
                    block.real[1:],
                )
                target -= 2.0 * np.matmul(
                    multiplier[1:].imag,
                    block.imaginary[1:],
                )
            end -= size

    def field_storage(self) -> int:
        completed = sum(len(level.completed) for level in self.levels)
        return len(self.recent) + 2 * (
            len(self.levels) + completed
        ) * (self.representation.nodes_per_level + 1)  # type: ignore[operator]

    def state(self) -> dict[str, Any]:
        return {
            "accepted_steps": self.accepted_steps,
            "recent": [
                {"index": index, "values": values.tolist()}
                for index, values in self.recent
            ],
            "levels": [
                {
                    "active_start": level.active_start,
                    "real": level.real.tolist(),
                    "imaginary": level.imaginary.tolist(),
                    "completed": [
                        {
                            "end": block.end,
                            "real": block.real.tolist(),
                            "imaginary": block.imaginary.tolist(),
                        }
                        for block in level.completed
                    ],
                }
                for level in self.levels
            ],
        }

    def restore(self, state: dict[str, Any]) -> None:
        try:
            accepted_steps = int(state["accepted_steps"])
            recent = list(state["recent"])
            levels = list(state["levels"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("fast-CQ checkpoint history is invalid") from error
        if accepted_steps < 0 or len(levels) != len(self.levels):
            raise ValueError("fast-CQ checkpoint history is invalid")
        restored_recent: deque[tuple[int, np.ndarray]] = deque(
            maxlen=self.representation.direct_steps
        )
        for entry in recent:
            index = int(entry["index"])
            values = np.asarray(entry["values"], dtype=np.float64)
            if values.shape != (self.local_size,) or not np.all(np.isfinite(values)):
                raise ValueError("fast-CQ checkpoint field is invalid")
            restored_recent.append((index, values.copy()))
        for level, level_state in zip(self.levels, levels, strict=True):
            real = np.asarray(level_state["real"], dtype=np.float64)
            imaginary = np.asarray(level_state["imaginary"], dtype=np.float64)
            if (
                real.shape != level.real.shape
                or imaginary.shape != level.imaginary.shape
            ):
                raise ValueError("fast-CQ checkpoint contour field is invalid")
            if not np.all(np.isfinite(real)) or not np.all(np.isfinite(imaginary)):
                raise ValueError("fast-CQ checkpoint contour field is invalid")
            level.real[:] = real
            level.imaginary[:] = imaginary
            active_start = level_state["active_start"]
            level.active_start = None if active_start is None else int(active_start)
            level.completed.clear()
            for block_state in level_state["completed"]:
                block_real = np.asarray(block_state["real"], dtype=np.float64)
                block_imaginary = np.asarray(
                    block_state["imaginary"],
                    dtype=np.float64,
                )
                if (
                    block_real.shape != level.real.shape
                    or block_imaginary.shape != level.imaginary.shape
                ):
                    raise ValueError("fast-CQ checkpoint contour field is invalid")
                level.completed.append(
                    _CompletedBlock(
                        end=int(block_state["end"]),
                        real=block_real.copy(),
                        imaginary=block_imaginary.copy(),
                    )
                )
        self.accepted_steps = accepted_steps
        self.recent = restored_recent


class FastObliviousCQStepper(FullHistoryStepper):
    """Firedrake residual stepper using logarithmic fast-CQ field state."""

    representation: FastObliviousCQ

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._uniform_step_size: float | None = None
        self._accepted_steps = 0
        self._fast_histories: tuple[_FastCQHistory, ...] = ()
        super().__init__(*args, **kwargs)
        self._current_time = self._read_time()

    def _require_current_time(self) -> float:
        current = self._read_time()
        if not isclose(
            current,
            self._current_time,
            rel_tol=1.0e-13,
            abs_tol=1.0e-14,
        ):
            raise RuntimeError("t must be advanced after each successful fast-CQ step")
        return current

    def _prepare_step(self, current: float, step_size: float) -> float:
        target = current + step_size
        if not isfinite(target) or target <= current:
            raise ValueError("t + dt must be finite and greater than t")
        if self._accepted_steps >= self.representation.max_steps:
            raise RuntimeError(
                "FastObliviousCQ exceeded max_steps; "
                "increase num_levels"
            )
        if self._uniform_step_size is None:
            self._uniform_step_size = step_size
            self._fast_histories = tuple(
                _FastCQHistory(
                    self.representation,
                    alpha,
                    step_size,
                    self._local_size,
                )
                for alpha in self.alphas
            )
        elif not self._same_time(step_size, self._uniform_step_size):
            raise ValueError(
                "FastObliviousCQ requires a uniform time step"
            )
        step = self._accepted_steps + 1
        for kind, alpha, history, history_term, implicit, initial_trace in zip(
            self._operator_kinds,
            self.alphas,
            self._fast_histories,
            self._history_terms,
            self._implicit_weights,
            self._initial_trace_terms,
            strict=True,
        ):
            history.past_action(step, history_term.dat.data.reshape(-1))
            implicit.assign(step_size ** (-alpha))
            if kind == "riemann_liouville":
                coefficient = (
                    step_size ** (-alpha)
                    * gamma(step + 1.0 - alpha)
                    / (gamma(1.0 - alpha) * gamma(step + 1.0))
                )
                initial_trace.assign(coefficient * self._initial)
        self._last_step_size = step_size
        return target

    def advance(self) -> None:
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
        increment = (
            self.u.dat.data_ro.reshape(-1)
            - self._previous.dat.data_ro.reshape(-1)
        )
        for history in self._fast_histories:
            history.append(increment)
        self._previous.assign(self.u)
        self._accepted_steps += 1
        self._current_time = target
        self._solve_count += 1
        self._nonlinear_iterations += self._solver.snes.getIterationNumber()
        self._linear_iterations += self._solver.snes.getLinearSolveIterations()

    @property
    def history(self) -> tuple[Any, ...]:
        fields = []
        for term, history in enumerate(self._fast_histories):
            for index, values in history.recent:
                field = self._fd.Function(
                    self._space,
                    name=f"fast_cq_term_{term:02d}_recent_{index:08d}",
                )
                field.dat.data[:] = values.reshape(self._local_shape)
                fields.append(field)
        return tuple(fields)

    def solver_stats(self) -> dict[str, Any]:
        stored = sum(history.field_storage() for history in self._fast_histories)
        return {
            "solves": self._solve_count,
            "failures": self._failure_count,
            "nonlinear_iterations": self._nonlinear_iterations,
            "linear_iterations": self._linear_iterations,
            "num_fractional_terms": len(self._markers),
            "history_steps": self._accepted_steps,
            "stored_history_fields": stored,
            "last_step_size": self._last_step_size,
            "history_method": "fast-oblivious-bdf1",
            "work_class": "O(N log N) total",
            "storage_class": "O(log N) fields",
        }

    def reset(self, u0: Any, t0: Any = None) -> None:
        self.u.assign(u0)
        self._previous.assign(self.u)
        self._initial.assign(self.u)
        if t0 is not None:
            self.t.assign(t0)
        self._current_time = self._read_time()
        self._accepted_steps = 0
        self._uniform_step_size = None
        self._fast_histories = ()
        self._last_step_size = None
        for history_term in self._history_terms:
            history_term.assign(0.0)
        for initial_trace in self._initial_trace_terms:
            initial_trace.assign(0.0)
        self._reset_solver_counters()

    def _checkpoint_metadata(self) -> dict[str, Any]:
        current = self._require_current_time()
        payload = stepper_metadata(
            kind="fast_oblivious_cq",
            operator_kinds=self._operator_kinds,
            parameters=self.alphas,
            representations=[
                self.representation.describe(alpha) for alpha in self.alphas
            ],
            formulation={
                "kind": "complex-contour-blocks",
                "history_splitting": "dyadic",
                "realization": "real-2x2",
            },
        )
        payload.update(
            {
                "time": current,
                "dt": float(self.dt),
                "accepted_steps": self._accepted_steps,
                "uniform_step_size": self._uniform_step_size,
                "history_layouts": [
                    {
                        "recent_indices": [index for index, _ in history.recent],
                        "levels": [
                            {
                                "active_start": level.active_start,
                                "completed_ends": [
                                    block.end for block in level.completed
                                ],
                            }
                            for level in history.levels
                        ],
                    }
                    for history in self._fast_histories
                ],
                "stats": self.solver_stats(),
            }
        )
        return payload

    def checkpoint_state(self) -> dict[str, Any]:
        payload = self._checkpoint_metadata()
        payload.update(
            {
                "u": self.u.dat.data_ro.tolist(),
                "initial": self._initial.dat.data_ro.tolist(),
                "previous": self._previous.dat.data_ro.tolist(),
                "histories": [history.state() for history in self._fast_histories],
            }
        )
        return payload

    def _field_from_values(self, name: str, values: np.ndarray) -> Any:
        field = self._fd.Function(self._space, name=name)
        field.dat.data[:] = values.reshape(self._local_shape)
        return field

    def _checkpoint_file_fields(self) -> dict[str, Any]:
        fields = {
            "u": self.u,
            "initial": self._initial,
            "previous": self._previous,
        }
        for term, history in enumerate(self._fast_histories):
            for recent_index, (_, values) in enumerate(history.recent):
                name = f"t{term:04d}_recent_{recent_index:04d}"
                fields[name] = self._field_from_values(name, values)
            for level_index, level in enumerate(history.levels):
                for node in range(level.real.shape[0]):
                    real_name = f"t{term:04d}_l{level_index:04d}_r_{node:04d}"
                    imag_name = f"t{term:04d}_l{level_index:04d}_i_{node:04d}"
                    fields[real_name] = self._field_from_values(
                        real_name,
                        level.real[node],
                    )
                    fields[imag_name] = self._field_from_values(
                        imag_name,
                        level.imaginary[node],
                    )
                for block_index, block in enumerate(level.completed):
                    for node in range(block.real.shape[0]):
                        prefix = (
                            f"t{term:04d}_l{level_index:04d}_"
                            f"b{block_index:04d}"
                        )
                        real_name = f"{prefix}_r_{node:04d}"
                        imag_name = f"{prefix}_i_{node:04d}"
                        fields[real_name] = self._field_from_values(
                            real_name,
                            block.real[node],
                        )
                        fields[imag_name] = self._field_from_values(
                            imag_name,
                            block.imaginary[node],
                        )
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
        layouts = metadata.get("history_layouts")
        if not isinstance(layouts, list):
            raise ValueError("fast-CQ checkpoint history layout is invalid")
        nodes = self.representation.nodes_per_level
        assert nodes is not None
        expected = ["u", "initial", "previous"]
        for term, layout in enumerate(layouts):
            recent_indices = layout.get("recent_indices", [])
            levels = layout.get("levels", [])
            for recent_index in range(len(recent_indices)):
                expected.append(f"t{term:04d}_recent_{recent_index:04d}")
            for level_index, level_layout in enumerate(levels):
                for node in range(nodes + 1):
                    expected.extend(
                        (
                            f"t{term:04d}_l{level_index:04d}_r_{node:04d}",
                            f"t{term:04d}_l{level_index:04d}_i_{node:04d}",
                        )
                    )
                for block_index in range(
                    len(level_layout.get("completed_ends", []))
                ):
                    for node in range(nodes + 1):
                        prefix = (
                            f"t{term:04d}_l{level_index:04d}_"
                            f"b{block_index:04d}"
                        )
                        expected.extend(
                            (f"{prefix}_r_{node:04d}", f"{prefix}_i_{node:04d}")
                        )
        state, loaded = load_checkpoint_file(
            checkpoint,
            name=name,
            mesh=self._space.mesh(),
            expected_fields=tuple(expected),
        )
        state["u"] = loaded["u"].dat.data_ro.tolist()
        state["initial"] = loaded["initial"].dat.data_ro.tolist()
        state["previous"] = loaded["previous"].dat.data_ro.tolist()
        histories = []
        for term, layout in enumerate(layouts):
            recent = [
                {
                    "index": index,
                    "values": loaded[
                        f"t{term:04d}_recent_{recent_index:04d}"
                    ].dat.data_ro.reshape(-1).tolist(),
                }
                for recent_index, index in enumerate(layout["recent_indices"])
            ]
            level_states = []
            for level_index, level_layout in enumerate(layout["levels"]):
                real = []
                imaginary = []
                for node in range(nodes + 1):
                    real.append(
                        loaded[
                            f"t{term:04d}_l{level_index:04d}_r_{node:04d}"
                        ].dat.data_ro.reshape(-1).tolist()
                    )
                    imaginary.append(
                        loaded[
                            f"t{term:04d}_l{level_index:04d}_i_{node:04d}"
                        ].dat.data_ro.reshape(-1).tolist()
                    )
                completed = []
                for block_index, end in enumerate(
                    level_layout["completed_ends"]
                ):
                    block_real = []
                    block_imaginary = []
                    prefix = (
                        f"t{term:04d}_l{level_index:04d}_b{block_index:04d}"
                    )
                    for node in range(nodes + 1):
                        block_real.append(
                            loaded[f"{prefix}_r_{node:04d}"]
                            .dat.data_ro.reshape(-1)
                            .tolist()
                        )
                        block_imaginary.append(
                            loaded[f"{prefix}_i_{node:04d}"]
                            .dat.data_ro.reshape(-1)
                            .tolist()
                        )
                    completed.append(
                        {
                            "end": end,
                            "real": block_real,
                            "imaginary": block_imaginary,
                        }
                    )
                level_states.append(
                    {
                        "active_start": level_layout["active_start"],
                        "real": real,
                        "imaginary": imaginary,
                        "completed": completed,
                    }
                )
            histories.append(
                {
                    "accepted_steps": metadata["accepted_steps"],
                    "recent": recent,
                    "levels": level_states,
                }
            )
        state["histories"] = histories
        self.restore_checkpoint(state)

    def restore_checkpoint(self, state: dict[str, Any]) -> None:
        representations = validate_stepper_metadata(
            state,
            kind="fast_oblivious_cq",
            operator_kinds=self._operator_kinds,
            parameters=self.alphas,
        )
        for metadata, alpha in zip(representations, self.alphas, strict=True):
            validate_checkpoint_representation(
                metadata,
                self.representation,
                alpha,
            )
        expected_formulation = {
            "kind": "complex-contour-blocks",
            "history_splitting": "dyadic",
            "realization": "real-2x2",
        }
        if dict(state.get("formulation") or {}) != expected_formulation:
            raise ValueError("checkpoint formulation does not match")
        try:
            current = float(state["time"])
            checkpoint_dt = float(state["dt"])
            accepted_steps = int(state["accepted_steps"])
            step_size = float(state["uniform_step_size"])
            history_states = list(state["histories"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("fast-CQ checkpoint metadata is invalid") from error
        physical = np.asarray(state.get("u"), dtype=np.float64)
        initial = np.asarray(state.get("initial"), dtype=np.float64)
        previous = np.asarray(state.get("previous"), dtype=np.float64)
        if any(
            values.shape != self._local_shape
            for values in (physical, initial, previous)
        ):
            raise ValueError("fast-CQ checkpoint field has the wrong local shape")
        if len(history_states) != len(self.alphas):
            raise ValueError("fast-CQ checkpoint term count does not match")
        histories = tuple(
            _FastCQHistory(
                self.representation,
                alpha,
                step_size,
                self._local_size,
            )
            for alpha in self.alphas
        )
        for history, history_state in zip(histories, history_states, strict=True):
            history.restore(history_state)
            if history.accepted_steps != accepted_steps:
                raise ValueError("fast-CQ checkpoint step count does not match")
        try:
            stats = dict(state.get("stats") or {})
            solve_count = int(stats["solves"])
            failure_count = int(stats["failures"])
            nonlinear_iterations = int(stats["nonlinear_iterations"])
            linear_iterations = int(stats["linear_iterations"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                "fast-CQ checkpoint solver statistics are invalid"
            ) from error
        if solve_count != accepted_steps or min(
            solve_count,
            failure_count,
            nonlinear_iterations,
            linear_iterations,
        ) < 0:
            raise ValueError("fast-CQ checkpoint solver statistics are invalid")
        self.u.dat.data[:] = physical
        self._initial.dat.data[:] = initial
        self._previous.dat.data[:] = previous
        self.t.assign(current)
        self.dt.assign(checkpoint_dt)
        self._current_time = current
        self._accepted_steps = accepted_steps
        self._uniform_step_size = step_size
        self._fast_histories = histories
        self._last_step_size = step_size if accepted_steps else None
        self._solve_count = solve_count
        self._failure_count = failure_count
        self._nonlinear_iterations = nonlinear_iterations
        self._linear_iterations = linear_iterations
        self._rebuild_solver = True
