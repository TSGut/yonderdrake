"""Measure timestep accuracy and cost of time-memory formulations."""

from __future__ import annotations

import argparse
import csv
import platform
import sys
from dataclasses import dataclass
from math import exp, fsum, gamma, lgamma, log, log2
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PowerMeasurement:
    """One manufactured-power solve through the public time stepper."""

    alpha: float
    power: float
    interpolant: str
    num_steps: int
    error: float
    seconds_per_step: float


@dataclass(frozen=True)
class StorageMeasurement:
    """Explicit time-memory storage and warm stepping cost."""

    interpolant: str
    num_modes: int
    num_terms: int
    dofs: int
    state_fields: int
    explicit_fields: int
    explicit_bytes: int
    seconds_per_step: float


@dataclass(frozen=True)
class GradingMeasurement:
    """One variable-step relaxation solve."""

    interpolant: str
    grading: float
    error: float
    seconds_per_step: float


@dataclass(frozen=True)
class AuxiliaryMeasurement:
    """One manufactured-power solve through the auxiliary-ODE stepper."""

    alpha: float
    power: float
    scheme: str
    num_steps: int
    error: float
    seconds_per_step: float


def _power_measurement(
    fd: Any,
    *,
    alpha: float,
    power: float,
    interpolant: str,
    num_modes: int,
    num_steps: int,
    timing_repeats: int,
) -> PowerMeasurement:
    from yonderdrake import (
        BirkSong,
        CaputoDerivative,
        FractionalTimeStepper,
        Recurrence,
    )

    mesh = fd.UnitIntervalMesh(1)
    space = fd.FunctionSpace(mesh, "CG", 1)
    solution = fd.Function(space, name="manufactured_power").assign(0.0)
    test = fd.TestFunction(space)
    time = fd.Constant(0.0)
    step_size = fd.Constant(1.0 / num_steps)
    source = fd.Constant(0.0)
    residual = (
        fd.inner(CaputoDerivative(solution, alpha), test)
        - fd.inner(source, test)
    ) * fd.dx
    stepper = FractionalTimeStepper(
        residual,
        BirkSong(num_modes),
        time,
        step_size,
        solution,
        formulation=Recurrence(interpolant=interpolant),
        solver_parameters={"ksp_type": "preonly", "pc_type": "lu"},
    )
    derivative_scale = gamma(power + 1.0) / gamma(power + 1.0 - alpha)

    def solve() -> tuple[float, float]:
        stepper.reset(0.0, t0=0.0)
        start = perf_counter()
        for index in range(1, num_steps + 1):
            target_time = index / num_steps
            source.assign(derivative_scale * target_time ** (power - alpha))
            stepper.advance()
            time.assign(target_time)
        elapsed = perf_counter() - start
        error = abs(float(solution.dat.data_ro[0]) - 1.0)
        return error, elapsed / num_steps

    error, initial_timing = solve()
    timings = [solve()[1] for _ in range(timing_repeats)]
    return PowerMeasurement(
        alpha,
        power,
        interpolant,
        num_steps,
        error,
        median(timings) if timings else initial_timing,
    )


def run_power_study(
    output: Path,
    *,
    alphas: tuple[float, ...] = (0.3, 0.6, 0.9),
    step_counts: tuple[int, ...] = (100, 200, 400),
    num_modes: int = 120,
    timing_repeats: int = 3,
) -> list[dict[str, object]]:
    """Measure actual-stepper order for smooth and singular powers."""
    import firedrake as fd

    measurements: dict[tuple[float, float, str], list[PowerMeasurement]] = {}
    for alpha in alphas:
        for power in (2.0, alpha):
            for interpolant in ("linear", "quadratic"):
                key = (alpha, power, interpolant)
                measurements[key] = [
                    _power_measurement(
                        fd,
                        alpha=alpha,
                        power=power,
                        interpolant=interpolant,
                        num_modes=num_modes,
                        num_steps=num_steps,
                        timing_repeats=timing_repeats,
                    )
                    for num_steps in step_counts
                ]

    # Doubling the modes at the finest grid checks that the reported slope is
    # temporal. On the reference run the largest relative change was 1.1e-7.
    mode_refinement_changes = {}
    for key, values in measurements.items():
        alpha, power, interpolant = key
        refined = _power_measurement(
            fd,
            alpha=alpha,
            power=power,
            interpolant=interpolant,
            num_modes=2 * num_modes,
            num_steps=step_counts[-1],
            timing_repeats=0,
        )
        mode_refinement_changes[key] = abs(
            refined.error - values[-1].error
        ) / values[-1].error
    maximum_mode_change = max(mode_refinement_changes.values())
    if maximum_mode_change >= 1.0e-4:
        raise RuntimeError(
            "mode refinement changed a finest-grid error by "
            f"{maximum_mode_change:.2e}; increase num_modes"
        )

    runtime = {
        "python": platform.python_version(),
        "firedrake": getattr(fd, "__version__", "unknown"),
        "platform": platform.platform(),
    }
    rows: list[dict[str, object]] = []
    for (alpha, power, interpolant), values in measurements.items():
        for index, value in enumerate(values):
            order = (
                ""
                if index == 0
                else log2(values[index - 1].error / value.error)
            )
            rows.append(
                {
                    "study": "power",
                    "alpha": alpha,
                    "profile": "smooth" if power == 2.0 else "singular",
                    "power": power,
                    "method": f"recurrence-{interpolant}",
                    "num_modes": num_modes,
                    "num_steps": value.num_steps,
                    "grading": 1.0,
                    "final_error": value.error,
                    "observed_order": order,
                    "seconds_per_step": value.seconds_per_step,
                    "mode_refinement_relative_change": (
                        mode_refinement_changes[(alpha, power, interpolant)]
                    ),
                    **runtime,
                }
            )

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def _explicit_time_memory_fields(stepper: Any) -> tuple[Any, ...]:
    fields = []
    for name in (
        "_previous",
        "_penultimate",
        "_initial",
        "_committed_u",
        "_increment",
        "_mode_scratch",
        "_mode_groups",
        "_mode_velocity_groups",
        "_history_terms",
        "_initial_trace_terms",
    ):
        pending = [getattr(stepper, name, None)]
        while pending:
            value = pending.pop()
            if value is None:
                continue
            if isinstance(value, tuple):
                pending.extend(value)
            elif hasattr(value, "dat"):
                fields.append(value)
    unique = {id(field): field for field in fields}
    return tuple(unique.values())


def _storage_measurement(
    fd: Any,
    *,
    interpolant: str,
    mesh_level: int,
    num_modes: int,
    step_size: float,
    steps_per_sample: int,
    repeats: int,
) -> StorageMeasurement:
    from benchmarks.time_fractional_scaling import _skullball_problem

    problem = _skullball_problem(
        fd,
        mesh_level,
        step_size,
        steps_per_sample * step_size,
        "birk-song",
        num_modes,
        interpolant=interpolant,
    )
    problem.advance(step_size)
    samples = []
    next_step = 2
    for _ in range(repeats):
        start = perf_counter()
        for _ in range(steps_per_sample):
            problem.advance(next_step * step_size)
            next_step += 1
        samples.append((perf_counter() - start) / steps_per_sample)

    fields = _explicit_time_memory_fields(problem.stepper)
    local_bytes = sum(field.dat.data_with_halos.nbytes for field in fields)
    global_bytes = problem.mesh.comm.allreduce(local_bytes)
    stats = problem.stepper.solver_stats()
    return StorageMeasurement(
        interpolant,
        int(stats["num_modes"]),
        int(stats["num_time_memory_terms"]),
        problem.space.dim(),
        int(stats["num_modes"] + stats["physical_history_fields"]),
        len(fields),
        global_bytes,
        median(samples),
    )


def run_storage_study(
    output: Path,
    *,
    mesh_level: int = 3,
    num_modes: int = 120,
    step_size: float = 0.01,
    steps_per_sample: int = 100,
    repeats: int = 5,
) -> list[dict[str, object]]:
    """Measure storage and cost on the four-material skullball problem."""
    import firedrake as fd

    # Build and advance both variants before timing so shared Firedrake caches do
    # not favour the variant measured second.
    for interpolant in ("linear", "quadratic"):
        _storage_measurement(
            fd,
            interpolant=interpolant,
            mesh_level=mesh_level,
            num_modes=num_modes,
            step_size=step_size,
            steps_per_sample=1,
            repeats=1,
        )
    measurements = [
        _storage_measurement(
            fd,
            interpolant=interpolant,
            mesh_level=mesh_level,
            num_modes=num_modes,
            step_size=step_size,
            steps_per_sample=steps_per_sample,
            repeats=repeats,
        )
        for interpolant in ("linear", "quadratic")
    ]
    rows = [
        {
            "study": "storage",
            "problem": "four-material-skullball",
            "method": f"recurrence-{value.interpolant}",
            "mesh_level": mesh_level,
            "dofs": value.dofs,
            "num_modes_per_term": num_modes,
            "num_terms": value.num_terms,
            "total_modes": value.num_modes,
            "state_fields": value.state_fields,
            "explicit_time_memory_fields": value.explicit_fields,
            "explicit_time_memory_bytes": value.explicit_bytes,
            "explicit_time_memory_mib": value.explicit_bytes / 2.0**20,
            "seconds_per_step": value.seconds_per_step,
            "steps_per_sample": steps_per_sample,
            "repeats": repeats,
            "python": platform.python_version(),
            "firedrake": getattr(fd, "__version__", "unknown"),
            "platform": platform.platform(),
        }
        for value in measurements
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def _mittag_leffler_negative(alpha: float, magnitude: float) -> float:
    terms = []
    for index in range(10000):
        term = (
            (-1.0 if index % 2 else 1.0)
            * exp(
                index * log(magnitude)
                - lgamma(alpha * index + 1.0)
            )
            if magnitude
            else (1.0 if index == 0 else 0.0)
        )
        terms.append(term)
        if index > 8 and abs(term) < 2.0e-17:
            return fsum(terms)
    raise RuntimeError("Mittag-Leffler reference did not converge")


def _grading_measurement(
    fd: Any,
    *,
    alpha: float,
    interpolant: str,
    grading: float,
    num_modes: int,
    num_steps: int,
    final_time: float,
) -> GradingMeasurement:
    from yonderdrake import (
        BirkSong,
        CaputoDerivative,
        FractionalTimeStepper,
        Recurrence,
    )

    mesh = fd.UnitIntervalMesh(1)
    space = fd.FunctionSpace(mesh, "CG", 1)
    solution = fd.Function(space, name="graded_relaxation").assign(1.0)
    test = fd.TestFunction(space)
    time = fd.Constant(0.0)
    step_size = fd.Constant(final_time * (1.0 / num_steps) ** grading)
    residual = (
        fd.inner(CaputoDerivative(solution, alpha), test)
        + fd.inner(solution, test)
    ) * fd.dx
    stepper = FractionalTimeStepper(
        residual,
        BirkSong(num_modes),
        time,
        step_size,
        solution,
        formulation=Recurrence(interpolant=interpolant),
        solver_parameters={"ksp_type": "preonly", "pc_type": "lu"},
    )
    previous_time = 0.0
    maximum_error = 0.0
    start = perf_counter()
    for index in range(1, num_steps + 1):
        target_time = final_time * (index / num_steps) ** grading
        step_size.assign(target_time - previous_time)
        stepper.advance()
        time.assign(target_time)
        previous_time = target_time
        exact = _mittag_leffler_negative(alpha, target_time**alpha)
        maximum_error = max(
            maximum_error,
            abs(float(solution.dat.data_ro[0]) - exact),
        )
    elapsed = perf_counter() - start
    return GradingMeasurement(
        interpolant,
        grading,
        maximum_error,
        elapsed / num_steps,
    )


def run_grading_study(
    output: Path,
    *,
    alpha: float = 0.6,
    gradings: tuple[float, ...] = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0),
    num_modes: int = 120,
    num_steps: int = 100,
    final_time: float = 2.0,
) -> list[dict[str, object]]:
    """Compare maximum relaxation error on uniform and graded steps."""
    import firedrake as fd

    measurements = [
        _grading_measurement(
            fd,
            alpha=alpha,
            interpolant=interpolant,
            grading=grading,
            num_modes=num_modes,
            num_steps=num_steps,
            final_time=final_time,
        )
        for interpolant in ("linear", "quadratic")
        for grading in gradings
    ]
    refined_measurements = [
        _grading_measurement(
            fd,
            alpha=alpha,
            interpolant=interpolant,
            grading=grading,
            num_modes=2 * num_modes,
            num_steps=num_steps,
            final_time=final_time,
        )
        for interpolant in ("linear", "quadratic")
        for grading in gradings
    ]
    mode_refinement_changes = {
        (value.interpolant, value.grading): abs(value.error - refined.error)
        / value.error
        for value, refined in zip(
            measurements,
            refined_measurements,
            strict=True,
        )
    }
    maximum_mode_change = max(mode_refinement_changes.values())
    if maximum_mode_change >= 1.0e-4:
        raise RuntimeError(
            "mode refinement changed a relaxation error by "
            f"{maximum_mode_change:.2e}; increase num_modes"
        )
    baselines = {
        interpolant: next(
            value.error
            for value in measurements
            if value.interpolant == interpolant and value.grading == 1.0
        )
        for interpolant in ("linear", "quadratic")
    }
    rows = [
        {
            "study": "grading",
            "problem": "caputo-relaxation",
            "alpha": alpha,
            "method": f"recurrence-{value.interpolant}",
            "num_modes": num_modes,
            "num_steps": num_steps,
            "final_time": final_time,
            "grading": value.grading,
            "max_error": value.error,
            "gain_over_uniform": baselines[value.interpolant] / value.error,
            "seconds_per_step": value.seconds_per_step,
            "mode_refinement_relative_change": mode_refinement_changes[
                (value.interpolant, value.grading)
            ],
            "python": platform.python_version(),
            "firedrake": getattr(fd, "__version__", "unknown"),
            "platform": platform.platform(),
        }
        for value in measurements
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def _auxiliary_measurement(
    fd: Any,
    *,
    alpha: float,
    power: float,
    scheme: str,
    num_modes: int,
    num_steps: int,
) -> AuxiliaryMeasurement:
    from yonderdrake import (
        AuxiliaryODE,
        BirkSong,
        CaputoDerivative,
        FractionalTimeStepper,
    )

    mesh = fd.UnitIntervalMesh(1)
    space = fd.FunctionSpace(mesh, "CG", 1)
    solution = fd.Function(space, name="auxiliary_manufactured_power").assign(
        0.0
    )
    test = fd.TestFunction(space)
    time = fd.Constant(0.0)
    step_size = fd.Constant(1.0 / num_steps)
    source = fd.Constant(0.0)
    residual = (
        fd.inner(CaputoDerivative(solution, alpha), test)
        - fd.inner(source, test)
    ) * fd.dx
    stepper = FractionalTimeStepper(
        residual,
        BirkSong(num_modes),
        time,
        step_size,
        solution,
        formulation=AuxiliaryODE(scheme=scheme),
        solver_parameters={"ksp_type": "preonly", "pc_type": "lu"},
    )
    derivative_scale = gamma(power + 1.0) / gamma(power + 1.0 - alpha)
    first_time = 1.0 / num_steps
    source.assign(derivative_scale * first_time ** (power - alpha))
    stepper.advance()
    stepper.reset(0.0, t0=0.0)
    start = perf_counter()
    for index in range(1, num_steps + 1):
        target_time = index / num_steps
        source.assign(derivative_scale * target_time ** (power - alpha))
        stepper.advance()
        time.assign(target_time)
    elapsed = perf_counter() - start
    return AuxiliaryMeasurement(
        alpha,
        power,
        scheme,
        num_steps,
        abs(float(solution.dat.data_ro[0]) - 1.0),
        elapsed / num_steps,
    )


def _auxiliary_scalar_error(
    *,
    alpha: float,
    power: float,
    scheme: str,
    num_modes: int,
    num_steps: int,
) -> float:
    from yonderdrake import BirkSong

    spectrum = BirkSong(num_modes).spectrum(alpha)
    step_size = 1.0 / num_steps
    arguments = spectrum.rates * step_size
    response = (
        1.0 / (1.0 + arguments)
        if scheme == "backward_euler"
        else 1.0 / (1.0 + 0.5 * arguments)
    )
    transition = (
        response
        if scheme == "backward_euler"
        else (1.0 - 0.5 * arguments) * response
    )
    physical_coefficient = float(np.dot(spectrum.weights, response))
    modes = np.zeros(num_modes)
    solution = 0.0
    derivative_scale = gamma(power + 1.0) / gamma(power + 1.0 - alpha)
    for index in range(1, num_steps + 1):
        target_time = index / num_steps
        source = derivative_scale * target_time ** (power - alpha)
        previous_modal_value = float(
            np.dot(spectrum.weights * transition, modes)
        )
        increment = (source - previous_modal_value) / physical_coefficient
        solution += increment
        modes = transition * modes + response * increment
    return abs(solution - 1.0)


def run_auxiliary_study(
    output: Path,
    *,
    alphas: tuple[float, ...] = (0.3, 0.6, 0.9),
    step_counts: tuple[int, ...] = (100, 200, 400),
    num_modes: int = 50,
) -> list[dict[str, object]]:
    """Measure both auxiliary schemes with actual-stepper spot checks."""
    import firedrake as fd

    measurements: dict[
        tuple[float, float, str], list[AuxiliaryMeasurement]
    ] = {}
    actual_checks: dict[tuple[float, float, str], AuxiliaryMeasurement] = {}
    for alpha in alphas:
        for power in (2.0, alpha):
            for scheme in ("backward_euler", "trapezoidal"):
                key = (alpha, power, scheme)
                measurements[key] = [
                    AuxiliaryMeasurement(
                        alpha=alpha,
                        power=power,
                        scheme=scheme,
                        num_steps=num_steps,
                        error=_auxiliary_scalar_error(
                            alpha=alpha,
                            power=power,
                            scheme=scheme,
                            num_modes=num_modes,
                            num_steps=num_steps,
                        ),
                        seconds_per_step=float("nan"),
                    )
                    for num_steps in step_counts
                ]
                actual_checks[key] = _auxiliary_measurement(
                    fd,
                    alpha=alpha,
                    power=power,
                    scheme=scheme,
                    num_modes=num_modes,
                    num_steps=step_counts[-1],
                )

    actual_check_changes = {
        key: abs(actual_checks[key].error - values[-1].error)
        / values[-1].error
        for key, values in measurements.items()
    }
    maximum_actual_change = max(actual_check_changes.values())
    if maximum_actual_change >= 1.0e-7:
        raise RuntimeError(
            "scalar auxiliary equations disagreed with the actual stepper by "
            f"{maximum_actual_change:.2e}"
        )

    mode_refinement_changes = {}
    for key in measurements:
        alpha, power, scheme = key
        reference_error = _auxiliary_scalar_error(
            alpha=alpha,
            power=power,
            scheme=scheme,
            num_modes=num_modes,
            num_steps=step_counts[-1],
        )
        refined_error = _auxiliary_scalar_error(
            alpha=alpha,
            power=power,
            scheme=scheme,
            num_modes=2 * num_modes,
            num_steps=step_counts[-1],
        )
        mode_refinement_changes[key] = (
            abs(refined_error - reference_error) / reference_error
        )
    maximum_mode_change = max(mode_refinement_changes.values())
    if maximum_mode_change >= 1.0e-4:
        raise RuntimeError(
            "mode refinement changed a finest-grid auxiliary error by "
            f"{maximum_mode_change:.2e}; increase num_modes"
        )

    rows: list[dict[str, object]] = []
    for (alpha, power, scheme), values in measurements.items():
        for index, value in enumerate(values):
            rows.append(
                {
                    "study": "auxiliary",
                    "alpha": alpha,
                    "profile": "smooth" if power == 2.0 else "singular",
                    "power": power,
                    "method": f"auxiliary-{scheme}",
                    "num_modes": num_modes,
                    "num_steps": value.num_steps,
                    "final_error": value.error,
                    "observed_order": (
                        ""
                        if index == 0
                        else log2(values[index - 1].error / value.error)
                    ),
                    "seconds_per_step": actual_checks[
                        (alpha, power, scheme)
                    ].seconds_per_step,
                    "actual_stepper_relative_check": actual_check_changes[
                        (alpha, power, scheme)
                    ],
                    "mode_refinement_relative_change": (
                        mode_refinement_changes[(alpha, power, scheme)]
                    ),
                    "python": platform.python_version(),
                    "firedrake": getattr(fd, "__version__", "unknown"),
                    "platform": platform.platform(),
                }
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            Path(__file__).resolve().parent
            / "benchmarks-output"
            / "recurrence-accuracy.csv"
        ),
    )
    parser.add_argument("--num-modes", type=int, default=120)
    parser.add_argument("--auxiliary-num-modes", type=int, default=50)
    parser.add_argument("--timing-repeats", type=int, default=3)
    parser.add_argument("--storage-repeats", type=int, default=5)
    parser.add_argument(
        "--study",
        choices=("power", "storage", "grading", "auxiliary"),
        default="power",
    )
    arguments = parser.parse_args()
    sys.argv[:] = [sys.argv[0]]
    if arguments.study == "power":
        rows = run_power_study(
            arguments.output,
            num_modes=arguments.num_modes,
            timing_repeats=arguments.timing_repeats,
        )
    elif arguments.study == "storage":
        rows = run_storage_study(
            arguments.output,
            num_modes=arguments.num_modes,
            repeats=arguments.storage_repeats,
        )
    elif arguments.study == "grading":
        rows = run_grading_study(
            arguments.output,
            num_modes=arguments.num_modes,
        )
    else:
        rows = run_auxiliary_study(
            arguments.output,
            num_modes=arguments.auxiliary_num_modes,
        )
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
