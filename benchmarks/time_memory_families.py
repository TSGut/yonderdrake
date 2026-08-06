"""Compare time-memory families at matched end-to-end computational cost."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
import tempfile
import textwrap
from collections.abc import Callable
from dataclasses import dataclass
from math import gamma
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from yonderdrake import (
    AlikhanovL21Sigma,
    BirkSong,
    CaputoDerivative,
    Diethelm2008,
    Diethelm2022,
    FastObliviousCQ,
    FractionalTimeStepper,
    FullHistory,
    LubichCQ,
    SineDiffusive,
    SumOfExponentials,
    YuanAgrawal,
)

DEFAULT_BASELINE_MODES = 100
DEFAULT_STEP_SIZE = 0.001
DEFAULT_FINAL_TIME = 0.5


@dataclass(frozen=True)
class _Candidate:
    method: str
    builder: Callable[[], object]
    step_size: float
    parameters: dict[str, object]
    work_class: str
    storage_class: str


@dataclass(frozen=True)
class _Measurement:
    candidate: _Candidate
    wall_seconds: float
    smooth_error: float
    singular_error: float
    peak_fields: int

    @property
    def score(self) -> float:
        return max(self.smooth_error, self.singular_error)


def _stored_fields(stats: dict[str, Any]) -> int:
    if "stored_history_fields" in stats:
        return int(stats["stored_history_fields"])
    if "stored_fields" in stats:
        return int(stats["stored_fields"])
    return int(stats.get("num_modes", 0)) * int(stats.get("fields_per_mode", 1))


def _run_problem(
    candidate: _Candidate,
    *,
    alpha: float,
    power: float,
    final_time: float,
) -> tuple[float, int]:
    try:
        import firedrake as fd
    except ImportError as error:
        raise RuntimeError("this benchmark requires Firedrake") from error

    steps = round(final_time / candidate.step_size)
    if not np.isclose(steps * candidate.step_size, final_time):
        raise ValueError("candidate step_size must divide final_time")
    mesh = fd.UnitIntervalMesh(1)
    space = fd.FunctionSpace(mesh, "CG", 1)
    u = fd.Function(space).assign(0.0)
    test = fd.TestFunction(space)
    time = fd.Constant(0.0)
    dt = fd.Constant(candidate.step_size)
    derivative_scale = gamma(power + 1.0) / gamma(power + 1.0 - alpha)
    source = derivative_scale * time ** (power - alpha)
    residual = (
        fd.inner(CaputoDerivative(u, alpha), test) - fd.inner(source, test)
    ) * fd.dx
    stepper = FractionalTimeStepper(
        residual,
        candidate.builder(),
        time,
        dt,
        u,
        solver_parameters={"ksp_type": "preonly", "pc_type": "lu"},
    )
    peak_fields = 0
    for _ in range(steps):
        stepper.advance()
        time.assign(time + dt)
        peak_fields = max(peak_fields, _stored_fields(stepper.solver_stats()))
    return abs(float(u.dat.data_ro[0]) - final_time**power), peak_fields


def _measure_candidate(
    candidate: _Candidate,
    *,
    alpha: float,
    final_time: float,
    repeats: int,
) -> _Measurement:
    def evaluate() -> tuple[float, float, int]:
        smooth, smooth_fields = _run_problem(
            candidate,
            alpha=alpha,
            power=2.0,
            final_time=final_time,
        )
        singular, singular_fields = _run_problem(
            candidate,
            alpha=alpha,
            power=alpha,
            final_time=final_time,
        )
        return smooth, singular, max(smooth_fields, singular_fields)

    # Compile and initialize the complete path without paying for an extra
    # full-duration benchmark run. Firedrake reuses the resulting kernels in
    # the timed evaluations below.
    _run_problem(
        candidate,
        alpha=alpha,
        power=2.0,
        final_time=candidate.step_size,
    )
    _run_problem(
        candidate,
        alpha=alpha,
        power=alpha,
        final_time=candidate.step_size,
    )
    timings = []
    results = []
    for _ in range(repeats):
        start = perf_counter()
        results.append(evaluate())
        timings.append(perf_counter() - start)
    smooth, singular, fields = results[-1]
    return _Measurement(
        candidate=candidate,
        wall_seconds=float(np.median(timings)),
        smooth_error=smooth,
        singular_error=singular,
        peak_fields=fields,
    )


def _estimate_candidate_seconds(
    candidate: _Candidate,
    *,
    alpha: float,
    final_time: float,
    tuning_steps: int = 200,
) -> float:
    """Estimate full-duration cost from a short run and the work class."""
    full_steps = round(final_time / candidate.step_size)
    sampled_steps = min(full_steps, tuning_steps)
    sampled = _measure_candidate(
        candidate,
        alpha=alpha,
        final_time=sampled_steps * candidate.step_size,
        repeats=1,
    ).wall_seconds
    if sampled_steps == full_steps:
        return sampled
    if candidate.work_class == "O(N^2)":
        factor = (full_steps / sampled_steps) ** 2
    elif candidate.work_class == "O(N log N)":
        factor = (
            full_steps
            * np.log2(max(2, full_steps))
            / (sampled_steps * np.log2(max(2, sampled_steps)))
        )
    else:
        factor = full_steps / sampled_steps
    return sampled * factor


def _modal_candidates(
    name: str,
    constructor: Callable[..., object],
    *,
    baseline_modes: int,
    step_size: float,
    smoke: bool,
) -> list[_Candidate]:
    counts = tuple(
        sorted(
            {
                max(4, baseline_modes // 4),
                max(4, baseline_modes // 2),
                baseline_modes,
            }
        )
    )
    scales = (1.0,) if smoke else (0.3, 1.0, 3.0)
    return [
        _Candidate(
            method=name,
            builder=lambda count=count, scale=scale: constructor(
                count,
                rate_scale=scale,
            ),
            step_size=step_size,
            parameters={"num_modes": count, "rate_scale": scale, "dt": step_size},
            work_class="O(NL)",
            storage_class="O(L) fields",
        )
        for count in counts
        for scale in scales
    ]


def _candidate_families(
    *,
    baseline_modes: int,
    step_size: float,
    final_time: float,
    smoke: bool,
) -> dict[str, list[_Candidate]]:
    fixed_references = {
        name: [
            _baseline_candidate(
                constructor,
                name,
                baseline_modes,
                step_size,
            )
        ]
        for name, constructor in (
            ("Birk-Song", BirkSong),
            ("Diethelm", Diethelm2008),
        )
    }
    families = {
        **fixed_references,
        "Yuan-Agrawal": _modal_candidates(
            "Yuan-Agrawal",
            YuanAgrawal,
            baseline_modes=baseline_modes,
            step_size=step_size,
            smoke=smoke,
        ),
    }
    sine_counts = tuple(
        sorted(
            {
                4,
                6,
                8,
                max(4, baseline_modes // 4),
                max(4, baseline_modes // 2),
                baseline_modes,
            }
        )
    )
    families["Sine diffusive"] = [
        _Candidate(
            method="Sine diffusive",
            builder=lambda count=count: SineDiffusive(count),
            step_size=step_size,
            parameters={"num_modes": count, "dt": step_size},
            work_class="O(NL)",
            storage_class="O(2L) fields",
        )
        for count in sine_counts
    ]

    truncated_count = baseline_modes if baseline_modes % 2 else baseline_modes + 1
    laguerre_count = baseline_modes if baseline_modes % 2 == 0 else baseline_modes - 1
    if smoke:
        diethelm_2022_options = [
            ("trapezoidal", truncated_count, 4.0, 1.0),
            ("gauss-laguerre", max(4, laguerre_count), None, 1.0),
        ]
    else:
        diethelm_2022_options = [
            (quadrature, truncated_count, radius, scale)
            for quadrature in ("trapezoidal", "simpson", "gauss-legendre")
            for radius in (4.0, 8.0)
            for scale in (0.3, 1.0, 3.0)
        ] + [("gauss-laguerre", max(4, laguerre_count), None, 1.0)]
    families["Diethelm 2022"] = [
        _Candidate(
            method="Diethelm 2022",
            builder=(
                lambda quadrature=quadrature, count=count, radius=radius, scale=scale: (
                    Diethelm2022(count, quadrature="gauss-laguerre")
                    if quadrature == "gauss-laguerre"
                    else Diethelm2022(
                        count,
                        quadrature=quadrature,
                        truncation_radius=radius,
                        rate_scale=scale,
                    )
                )
            ),
            step_size=step_size,
            parameters={
                "quadrature": quadrature,
                "num_modes": count,
                "truncation_radius": radius,
                "rate_scale": scale,
                "dt": step_size,
            },
            work_class="O(NL)",
            storage_class="O(L) fields",
        )
        for quadrature, count, radius, scale in diethelm_2022_options
    ]

    soe_targets = (
        (1.0e-1, 1.0e-3)
        if smoke
        else (
            1.0,
            1.0e-1,
            1.0e-2,
            1.0e-3,
            1.0e-4,
            1.0e-6,
        )
    )
    families["Sum of exponentials"] = [
        _Candidate(
            method="Sum of exponentials",
            builder=lambda target=target: SumOfExponentials(
                target_error=target,
                min_step=step_size,
                t_final=final_time,
            ),
            step_size=step_size,
            parameters={
                "target_error": target,
                "min_step": step_size,
                "t_final": final_time,
                "dt": step_size,
            },
            work_class="O(NL)",
            storage_class="O(L) fields",
        )
        for target in soe_targets
    ]

    time_steps = (step_size,)
    families["Full-history L1"] = [
        _Candidate(
            method="Full-history L1",
            builder=FullHistory,
            step_size=dt,
            parameters={"dt": dt},
            work_class="O(N^2)",
            storage_class="O(N) fields",
        )
        for dt in time_steps
    ]
    cq_options = (
        (("bdf2", 2),)
        if smoke
        else (
            ("bdf1", 0),
            ("bdf1", 1),
            ("bdf2", 0),
            ("bdf2", 1),
            ("bdf2", 2),
        )
    )
    families["Lubich CQ"] = [
        _Candidate(
            method="Lubich CQ",
            builder=lambda order=order, corrections=corrections: LubichCQ(
                order=order,
                num_corrections=corrections,
            ),
            step_size=dt,
            parameters={
                "order": order,
                "num_corrections": corrections,
                "dt": dt,
            },
            work_class="O(N^2)",
            storage_class="O(N) fields",
        )
        for order, corrections in cq_options
        for dt in time_steps
    ]
    families["Alikhanov L2-1-sigma"] = [
        _Candidate(
            method="Alikhanov L2-1-sigma",
            builder=AlikhanovL21Sigma,
            step_size=dt,
            parameters={"dt": dt, "sigma": "1-alpha/2"},
            work_class="O(N^2)",
            storage_class="O(N) fields",
        )
        for dt in time_steps
    ]
    fast_nodes = (10,) if smoke else (10, 15)
    fast_direct = (10,) if smoke else (10, 20)
    families["Fast-oblivious CQ"] = [
        _Candidate(
            method="Fast-oblivious CQ",
            builder=lambda nodes=nodes, direct=direct, dt=dt: FastObliviousCQ(
                target_error=1.0e-3 if nodes == 10 else 1.0e-6,
                num_levels=round(final_time / dt).bit_length(),
                nodes_per_level=nodes,
                direct_steps=direct,
            ),
            step_size=dt,
            parameters={
                "contour": "talbot",
                "nodes_per_level": nodes,
                "direct_steps": direct,
                "num_levels": round(final_time / dt).bit_length(),
                "dt": dt,
            },
            work_class="O(N log N)",
            storage_class="O(log N) fields",
        )
        for nodes in fast_nodes
        for direct in fast_direct
        for dt in time_steps
    ]
    return families


def _baseline_candidate(
    constructor: Callable[[int], object],
    name: str,
    modes: int,
    step_size: float,
) -> _Candidate:
    return _Candidate(
        method=name,
        builder=lambda: constructor(modes),
        step_size=step_size,
        parameters={"num_modes": modes, "rate_scale": 1.0, "dt": step_size},
        work_class="O(NL)",
        storage_class="O(L) fields",
    )


def run_cost_matched_comparison(
    output: Path,
    *,
    csv_output: Path,
    baseline_modes: int = DEFAULT_BASELINE_MODES,
    step_size: float = DEFAULT_STEP_SIZE,
    final_time: float = DEFAULT_FINAL_TIME,
    alpha: float = 0.5,
    repeats: int = 3,
    minimum_cost_ratio: float = 0.5,
    maximum_cost_ratio: float = 2.0,
    smoke: bool = False,
    dpi: int = 220,
) -> Path:
    """Tune every family within one measured end-to-end runtime budget."""
    if baseline_modes < 4 or repeats < 1:
        raise ValueError("baseline_modes must be at least 4 and repeats positive")
    steps = round(final_time / step_size)
    if steps < 1 or not np.isclose(steps * step_size, final_time):
        raise ValueError("step_size must divide a positive final_time")
    if not 0.0 < minimum_cost_ratio <= 1.0 <= maximum_cost_ratio:
        raise ValueError("the cost band must contain the reference budget")
    baselines = [
        _baseline_candidate(BirkSong, "Birk-Song", baseline_modes, step_size),
        _baseline_candidate(Diethelm2008, "Diethelm", baseline_modes, step_size),
    ]
    baseline_measurements = [
        _measure_candidate(
            candidate,
            alpha=alpha,
            final_time=final_time,
            repeats=repeats,
        )
        for candidate in baselines
    ]
    budget = float(
        np.mean([measurement.wall_seconds for measurement in baseline_measurements])
    )
    families = _candidate_families(
        baseline_modes=baseline_modes,
        step_size=step_size,
        final_time=final_time,
        smoke=smoke,
    )
    measured_cache = {
        (
            measurement.candidate.method,
            json.dumps(measurement.candidate.parameters, sort_keys=True),
        ): measurement
        for measurement in baseline_measurements
    }
    selected = []
    for method, candidates in families.items():
        print(
            f"tuning {method} ({len(candidates)} candidates)",
            flush=True,
        )
        measurements = []
        estimates = []
        for candidate in candidates:
            key = (method, json.dumps(candidate.parameters, sort_keys=True))
            measurement = measured_cache.get(key)
            if measurement is not None:
                measurements.append(measurement)
                continue
            estimate = _estimate_candidate_seconds(
                candidate,
                alpha=alpha,
                final_time=final_time,
            )
            estimates.append((candidate, estimate))
        plausible = [
            (candidate, estimate)
            for candidate, estimate in estimates
            if 0.4 * budget <= estimate <= 2.5 * budget
        ]
        if not plausible and estimates:
            plausible = [
                min(
                    estimates,
                    key=lambda item: abs(np.log(item[1] / budget)),
                )
            ]
        for candidate, estimate in plausible:
            print(
                f"  measuring predicted ratio {estimate / budget:.3f}: "
                f"{json.dumps(candidate.parameters, sort_keys=True)}",
                flush=True,
            )
            measurements.append(
                _measure_candidate(
                    candidate,
                    alpha=alpha,
                    final_time=final_time,
                    repeats=1,
                )
            )
        feasible = [
            measurement
            for measurement in measurements
            if minimum_cost_ratio * budget
            <= measurement.wall_seconds
            <= maximum_cost_ratio * budget
        ]
        measured_parameters = {
            json.dumps(measurement.candidate.parameters, sort_keys=True)
            for measurement in measurements
        }
        remaining = sorted(
            (
                (candidate, estimate)
                for candidate, estimate in estimates
                if json.dumps(candidate.parameters, sort_keys=True)
                not in measured_parameters
            ),
            key=lambda item: abs(np.log(item[1] / budget)),
        )
        while not feasible and remaining:
            candidate, estimate = remaining.pop(0)
            print(
                f"  expanding search at predicted ratio {estimate / budget:.3f}",
                flush=True,
            )
            measurement = _measure_candidate(
                candidate,
                alpha=alpha,
                final_time=final_time,
                repeats=1,
            )
            measurements.append(measurement)
            if (
                minimum_cost_ratio * budget
                <= measurement.wall_seconds
                <= maximum_cost_ratio * budget
            ):
                feasible.append(measurement)
        if not feasible:
            ratios = [measurement.wall_seconds / budget for measurement in measurements]
            raise RuntimeError(
                f"{method} has no candidate in the cost band "
                f"[{minimum_cost_ratio}, {maximum_cost_ratio}]; "
                f"measured ratios span [{min(ratios):.3g}, {max(ratios):.3g}]"
            )
        ranked = sorted(feasible, key=lambda measurement: measurement.score)
        robust_measurement = None
        for tuning_measurement in ranked:
            key = (
                tuning_measurement.candidate.method,
                json.dumps(
                    tuning_measurement.candidate.parameters,
                    sort_keys=True,
                ),
            )
            candidate_measurement = measured_cache.get(key)
            if candidate_measurement is None:
                candidate_measurement = _measure_candidate(
                    tuning_measurement.candidate,
                    alpha=alpha,
                    final_time=final_time,
                    repeats=repeats,
                )
            if (
                minimum_cost_ratio * budget
                <= candidate_measurement.wall_seconds
                <= maximum_cost_ratio * budget
            ):
                robust_measurement = candidate_measurement
                break
        if robust_measurement is None:
            raise RuntimeError(
                f"{method} had candidates in the tuning cost band, but none "
                "remained there under repeated timing"
            )
        selected.append(robust_measurement)
        print(
            f"selected {method}: "
            f"{robust_measurement.wall_seconds / budget:.3f} x budget",
            flush=True,
        )

    rows = []
    runtime = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
    }
    for measurement in selected:
        rows.append(
            {
                "method": measurement.candidate.method,
                "alpha": alpha,
                "smooth_power": 2.0,
                "singular_power": alpha,
                "smooth_final_error": measurement.smooth_error,
                "singular_final_error": measurement.singular_error,
                "wall_seconds": measurement.wall_seconds,
                "cost_budget_seconds": budget,
                "cost_ratio": measurement.wall_seconds / budget,
                "minimum_cost_ratio": minimum_cost_ratio,
                "maximum_cost_ratio": maximum_cost_ratio,
                "peak_memory_fields": measurement.peak_fields,
                "work_class": measurement.candidate.work_class,
                "storage_class": measurement.candidate.storage_class,
                "chosen_parameters": json.dumps(
                    measurement.candidate.parameters,
                    sort_keys=True,
                ),
                "baseline_modes": baseline_modes,
                "final_time": final_time,
                "timing_repeats": repeats,
                **runtime,
            }
        )
    csv_output.parent.mkdir(parents=True, exist_ok=True)
    with csv_output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    labels = []
    for measurement in selected:
        controls = ", ".join(
            f"{key}={value}"
            for key, value in measurement.candidate.parameters.items()
            if value is not None
        )
        labels.append(
            measurement.candidate.method + "\n" + textwrap.fill(controls, width=46)
        )
    positions = np.arange(len(selected))
    # Imported here, not at module scope: the measurement half of this module is
    # pure numerics and must stay importable without the optional visual extra.
    os.environ.setdefault(
        "MPLCONFIGDIR",
        str(Path(tempfile.gettempdir()) / "yonderdrake-matplotlib"),
    )
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    colors = matplotlib.colormaps["viridis"](np.linspace(0.12, 0.9, len(selected)))
    figure, axes = plt.subplots(2, 2, figsize=(20.0, 14.0))
    axes[0, 0].barh(
        positions,
        [measurement.smooth_error for measurement in selected],
        color=colors,
    )
    axes[0, 1].barh(
        positions,
        [measurement.singular_error for measurement in selected],
        color=colors,
    )
    axes[1, 0].barh(
        positions,
        [measurement.wall_seconds / budget for measurement in selected],
        color=colors,
    )
    axes[1, 1].barh(
        positions,
        [measurement.peak_fields for measurement in selected],
        color=colors,
    )
    for axis in axes.flat:
        axis.set_yticks(positions, labels)
        axis.invert_yaxis()
        axis.grid(axis="x", alpha=0.3)
    axes[0, 0].set_xscale("log")
    axes[0, 1].set_xscale("log")
    axes[1, 1].set_xscale("log")
    axes[0, 0].set_title(r"Smooth solution $u(t)=t^2$")
    axes[0, 1].set_title(rf"Weak initial singularity $u(t)=t^{{{alpha}}}$")
    axes[0, 0].set_xlabel("Final absolute error")
    axes[0, 1].set_xlabel("Final absolute error")
    axes[1, 0].set_title("Measured end-to-end cost")
    axes[1, 0].set_xlabel("Wall time / reference budget")
    axes[1, 0].axvline(1.0, color="#333333", linestyle=":")
    axes[1, 0].axvline(minimum_cost_ratio, color="#a33a2b", linestyle=":")
    axes[1, 0].axvline(maximum_cost_ratio, color="#a33a2b", linestyle=":")
    axes[1, 1].set_title("Peak time-memory field storage")
    axes[1, 1].set_xlabel("Distributed fields")
    figure.suptitle(
        "Time-memory accuracy at matched end-to-end cost",
        fontsize=15,
        fontweight="bold",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path(__file__).resolve().parent / "benchmarks-output",
    )
    parser.add_argument("--figure-directory", type=Path)
    parser.add_argument(
        "--baseline-modes",
        type=int,
        default=DEFAULT_BASELINE_MODES,
    )
    parser.add_argument("--step-size", type=float, default=DEFAULT_STEP_SIZE)
    parser.add_argument("--final-time", type=float, default=DEFAULT_FINAL_TIME)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dpi", type=int, default=220)
    arguments = parser.parse_args()
    sys.argv[:] = [sys.argv[0]]
    output = arguments.output_directory
    figures = arguments.figure_directory or output
    output.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    if arguments.smoke:
        arguments.baseline_modes = 9
        arguments.step_size = 0.1
        arguments.final_time = 1.0
        arguments.repeats = 1
    print(
        run_cost_matched_comparison(
            figures / "time-memory-family-cost-matched.png",
            csv_output=output / "time-memory-family-cost-matched.csv",
            baseline_modes=arguments.baseline_modes,
            step_size=arguments.step_size,
            final_time=arguments.final_time,
            repeats=arguments.repeats,
            smoke=arguments.smoke,
            dpi=arguments.dpi,
        )
    )


if __name__ == "__main__":
    main()
