"""Benchmark diffusive-spectrum accuracy, refinement, and setup cost."""

from __future__ import annotations

import argparse
import csv
import os
import platform
import tempfile
from collections.abc import Callable
from functools import partial
from math import gamma
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from scipy.special import gammainc

from yonderdrake import (
    BirkSong,
    Diethelm2008,
    Diethelm2022,
    SineDiffusive,
    SumOfExponentials,
    YuanAgrawal,
)
from yonderdrake.time.coefficients import (
    oscillator_coefficients,
    quadratic_recurrence_coefficients,
    recurrence_coefficients,
)


def _pyplot() -> tuple[Any, Any]:
    """Import matplotlib on demand.

    The measurement half of this module is pure numerics and must stay
    importable without the optional ``visual`` extra, so the plotting stack and
    its backend selection are deferred to the functions that draw.
    """
    os.environ.setdefault(
        "MPLCONFIGDIR",
        str(Path(tempfile.gettempdir()) / "yonderdrake-matplotlib"),
    )
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    return matplotlib, plt


def _method_specs(
    num_modes: int,
    target_error: float,
) -> tuple[tuple[str, Callable[[], object], str, Any], ...]:
    return tuple(
        (
            label,
            partial(builder, _compatible_mode_count(label, num_modes)),
            color,
            linestyle,
        )
        for label, builder, color, linestyle in _method_builders(target_error)
    )


def _method_builders(
    target_error: float,
) -> tuple[tuple[str, Callable[[int], object], str, Any], ...]:
    return (
        ("Birk–Song 2010", BirkSong, "#168c8c", "-"),
        ("Diethelm 2008", Diethelm2008, "#d59618", "--"),
        (
            "Diethelm 2022 trapezoidal",
            lambda count: Diethelm2022(
                count,
                target_error=target_error,
            ),
            "#d95f4f",
            ":",
        ),
        (
            "Diethelm 2022 Simpson",
            lambda count: Diethelm2022(
                count,
                quadrature="simpson",
                target_error=target_error,
            ),
            "#3567b7",
            (0, (3, 1, 1, 1)),
        ),
        (
            "Diethelm 2022 Gauss–Legendre",
            lambda count: Diethelm2022(
                count,
                quadrature="gauss-legendre",
                target_error=target_error,
            ),
            "#3f7f4c",
            (0, (1, 1)),
        ),
        (
            "Diethelm 2022 Gauss–Laguerre",
            lambda count: Diethelm2022(
                count,
                quadrature="gauss-laguerre",
            ),
            "#ad5c9c",
            (0, (5, 1, 1, 1)),
        ),
        (
            "Yuan–Agrawal 2002",
            YuanAgrawal,
            "#7552a3",
            "-.",
        ),
        (
            "Sine diffusive 2024",
            SineDiffusive,
            "#a44a76",
            (0, (5, 2)),
        ),
    )


def _mode_count_parity(label: str) -> str:
    if label.endswith("Simpson"):
        return "odd"
    if label.endswith("Gauss–Laguerre"):
        return "even"
    return "any"


def _compatible_mode_count(label: str, requested: int) -> int:
    parity = _mode_count_parity(label)
    if parity == "odd" and requested % 2 == 0:
        return requested + 1
    if parity == "even" and requested % 2 != 0:
        return requested - 1
    return requested


def _write_rows(path: Path, rows: list[dict[str, object]]) -> Path:
    if not rows:
        raise ValueError("benchmark produced no rows")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _runtime_metadata() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
    }


def _relative_error_from_spectrum(
    spectrum: object,
    alpha: float,
    times: np.ndarray,
    problem: str,
) -> np.ndarray:
    rates = spectrum.rates
    weights = spectrum.weights
    scaled = np.outer(times, rates)
    if problem == "quadratic":
        response = np.empty_like(scaled)
        small = scaled < 1.0e-3
        z = scaled[small]
        response[small] = 1.0 + z * (
            -1.0 / 3.0
            + z * (1.0 / 12.0 + z * (-1.0 / 60.0 + z * (1.0 / 360.0 - z / 2520.0)))
        )
        z = scaled[~small]
        response[~small] = (2.0 / z) * (1.0 + np.expm1(-z) / z)
        observed = np.square(times) * (response @ weights)
        exact = 2.0 * times ** (2.0 - alpha) / gamma(3.0 - alpha)
    elif problem == "exponential":
        normalized_modes = -np.expm1(-np.outer(times, 1.0 + rates)) / (1.0 + rates)
        observed = normalized_modes @ weights
        exact = gammainc(1.0 - alpha, times)
    else:
        raise ValueError("problem must be 'quadratic' or 'exponential'")
    return np.maximum(
        np.abs(observed / exact - 1.0),
        np.finfo(np.float64).eps,
    )


def _relative_error(
    representation: object,
    alpha: float,
    times: np.ndarray,
    problem: str,
) -> np.ndarray:
    spectrum = representation.spectrum(alpha)
    if isinstance(representation, SumOfExponentials):
        minimum_step = float(spectrum.metadata["min_step"])
        final_time = float(spectrum.metadata["t_final"])
        if times[0] < minimum_step or times[-1] > final_time:
            raise ValueError(
                "comparison times must lie inside the SumOfExponentials interval"
            )
        local_step = np.minimum(times, minimum_step)
        older_time = times - local_step
        local_weight = np.power(local_step, -alpha) / gamma(2.0 - alpha)
        if problem == "quadratic":
            scaled = np.outer(older_time, spectrum.rates)
            response = np.empty_like(scaled)
            small = scaled < 1.0e-3
            z = scaled[small]
            response[small] = 1.0 + z * (
                -1.0 / 3.0
                + z * (1.0 / 12.0 + z * (-1.0 / 60.0 + z * (1.0 / 360.0 - z / 2520.0)))
            )
            z = scaled[~small]
            response[~small] = (2.0 / z) * (1.0 + np.expm1(-z) / z)
            older_modes = (
                np.exp(-np.outer(local_step, spectrum.rates))
                * np.square(older_time[:, None])
                * response
            )
            observed = older_modes @ spectrum.weights
            observed += local_weight * (np.square(times) - np.square(older_time))
            exact = 2.0 * times ** (2.0 - alpha) / gamma(3.0 - alpha)
        elif problem == "exponential":
            shifted_rates = 1.0 + spectrum.rates
            older_modes = (
                np.exp(-np.outer(local_step, shifted_rates))
                * -np.expm1(-np.outer(older_time, shifted_rates))
                / shifted_rates
            )
            observed = older_modes @ spectrum.weights
            observed += local_weight * -np.expm1(-local_step)
            exact = gammainc(1.0 - alpha, times)
        else:
            raise ValueError("problem must be 'quadratic' or 'exponential'")
        return np.maximum(
            np.abs(observed / exact - 1.0),
            np.finfo(np.float64).eps,
        )
    if isinstance(representation, SineDiffusive):
        frequencies = spectrum.frequencies
        scaled = np.outer(times, frequencies)
        forcing_scale = 2.0 * np.cos(np.pi * alpha / 2.0) / np.pi
        if problem == "quadratic":
            ratio = np.empty_like(scaled)
            small = np.abs(scaled) < 1.0e-3
            x = scaled[small]
            ratio[small] = 1.0 / 6.0 - np.square(x) / 120.0 + np.power(x, 4) / 5040.0
            x = scaled[~small]
            ratio[~small] = (x - np.sin(x)) / np.power(x, 3)
            modes = 2.0 * forcing_scale * np.power(times[:, None], 3) * ratio
            observed = modes @ spectrum.weights
            exact = 2.0 * times ** (2.0 - alpha) / gamma(3.0 - alpha)
        elif problem == "exponential":
            exponent = (-1.0 + 1j * frequencies[None, :]) * times[:, None]
            normalized_convolution = np.imag(
                np.expm1(exponent) / (-1.0 + 1j * frequencies[None, :])
            )
            modes = forcing_scale * normalized_convolution / frequencies[None, :]
            observed = modes @ spectrum.weights
            exact = gammainc(1.0 - alpha, times)
        else:
            raise ValueError("problem must be 'quadratic' or 'exponential'")
        return np.maximum(
            np.abs(observed / exact - 1.0),
            np.finfo(np.float64).eps,
        )
    return _relative_error_from_spectrum(spectrum, alpha, times, problem)


def _spectrum_size(spectrum: object) -> int:
    """Return the number of rates or frequencies in a memory spectrum."""
    if hasattr(spectrum, "frequencies"):
        return int(spectrum.frequencies.size)
    return int(spectrum.rates.size)


def plot_comparison(
    output: Path,
    *,
    csv_output: Path | None = None,
    num_modes: int = 41,
    target_error: float = 1.0e-8,
    soe_target_error: float = 1.0e-4,
    dpi: int = 220,
) -> Path:
    """Plot representation errors and return the output path."""
    matplotlib, plt = _pyplot()
    if num_modes < 4:
        raise ValueError("num_modes must be at least 4")
    methods = tuple(
        (label, factory(), color, linestyle)
        for label, factory, color, linestyle in _method_specs(
            num_modes,
            target_error,
        )
    )
    alphas = (0.1, 0.5, 0.9)
    times = np.logspace(-3.0, 3.0, 401)
    sum_of_exponentials = SumOfExponentials(
        target_error=soe_target_error,
        min_step=1.0e-5,
        t_final=float(times[-1]),
    )
    methods += (
        (
            "Jiang sum of exponentials 2017",
            sum_of_exponentials,
            "#333333",
            (0, (7, 2)),
        ),
    )

    plt.rcParams.update(
        {
            "axes.grid": True,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "font.size": 10,
            "grid.alpha": 0.35,
            "legend.frameon": False,
        }
    )
    figure, axes = plt.subplots(
        2,
        3,
        figsize=(16.5, 10.0),
        sharex=True,
        sharey=True,
    )
    rows: list[dict[str, object]] = []
    metadata = _runtime_metadata()
    for row, problem in enumerate(("quadratic", "exponential")):
        for axis, alpha in zip(axes[row], alphas, strict=True):
            for label, representation, color, linestyle in methods:
                errors = _relative_error(
                    representation,
                    alpha,
                    times,
                    problem,
                )
                count = _spectrum_size(representation.spectrum(alpha))
                axis.loglog(
                    times,
                    errors,
                    label=label,
                    color=color,
                    linestyle=linestyle,
                    linewidth=2.0,
                )
                rows.extend(
                    {
                        "method": label,
                        "alpha": alpha,
                        "problem": problem,
                        "time": float(time),
                        "num_modes": count,
                        "target_error": target_error,
                        "soe_target_error": soe_target_error,
                        "relative_error": float(error),
                        **metadata,
                    }
                    for time, error in zip(times, errors, strict=True)
                )
            axis.set_title(rf"Order $\alpha={alpha}$")
            axis.set_xlim(times[0], times[-1])
            axis.set_ylim(1.0e-16, 2.0)
    for axis in axes[-1]:
        axis.set_xlabel(r"Time $t$")
    for axis in axes[:, 0]:
        axis.set_ylabel("Absolute relative error")

    figure.suptitle(
        "Caputo derivative accuracy across time-memory representations",
        fontsize=15,
        fontweight="bold",
        y=0.995,
    )
    figure.text(
        0.5,
        0.945,
        r"$\dot u=2t,\ u(0)=0,\ u=t^2$; "
        r"$D_C^\alpha u=2t^{2-\alpha}/\Gamma(3-\alpha)$",
        ha="center",
        fontsize=12,
    )
    figure.text(
        0.5,
        0.535,
        r"$\dot u=e^t,\ u(0)=1,\ u=e^t$; "
        r"$D_C^\alpha u=e^tP(1-\alpha,t)$",
        ha="center",
        fontsize=12,
    )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.012),
        columnspacing=1.4,
        fontsize=9,
    )
    laguerre_modes = _compatible_mode_count(
        "Diethelm 2022 Gauss–Laguerre",
        num_modes,
    )
    simpson_modes = _compatible_mode_count(
        "Diethelm 2022 Simpson",
        num_modes,
    )
    figure.text(
        0.5,
        0.105,
        rf"Fixed-count methods use {laguerre_modes}–{simpson_modes} modes. "
        "The sum of exponentials "
        rf"uses a {soe_target_error:.0e} target on its declared interval.",
        ha="center",
    )
    figure.subplots_adjust(
        top=0.90,
        bottom=0.18,
        hspace=0.62,
        wspace=0.12,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi)
    plt.close(figure)
    if csv_output is not None:
        _write_rows(csv_output, rows)
    return output


def _median_runtime(
    action: Callable[[], object],
    repeats: int,
) -> float:
    action()
    samples = np.empty(repeats, dtype=np.float64)
    for index in range(repeats):
        start = perf_counter()
        action()
        samples[index] = perf_counter() - start
    return float(np.median(samples))


def _modal_update_workload(
    representation: object,
    *,
    alphas: tuple[float, ...],
    num_steps: int,
    field_size: int,
    time_step: float = 0.01,
    interpolant: str = "linear",
) -> float:
    """Run spectrum setup and the representation-dependent memory updates."""
    if interpolant not in {"linear", "quadratic"}:
        raise ValueError("interpolant must be 'linear' or 'quadratic'")
    coordinates = np.linspace(0.0, 1.0, field_size, dtype=np.float64)
    checksum = 0.0
    for alpha in alphas:
        spectrum = representation.spectrum(alpha)
        if isinstance(representation, SineDiffusive):
            (
                cosine,
                sine_over_frequency,
                negative_frequency_sine,
                position_forcing,
                velocity_forcing,
                _,
            ) = oscillator_coefficients(spectrum, alpha, time_step)
            position = np.zeros((spectrum.frequencies.size, field_size))
            velocity = np.zeros_like(position)
            next_position = np.empty_like(position)
            next_velocity = np.empty_like(velocity)
            previous = np.zeros(field_size)
            for step in range(1, num_steps + 1):
                current = np.sin(0.013 * step + coordinates)
                increment = current - previous
                np.multiply(cosine[:, None], position, out=next_position)
                next_position += sine_over_frequency[:, None] * velocity
                next_position += position_forcing[:, None] * increment
                np.multiply(
                    negative_frequency_sine[:, None],
                    position,
                    out=next_velocity,
                )
                next_velocity += cosine[:, None] * velocity
                next_velocity += velocity_forcing[:, None] * increment
                position, next_position = next_position, position
                velocity, next_velocity = next_velocity, velocity
                previous = current
            checksum += float(np.sum(spectrum.weights @ position))
            continue

        if interpolant == "quadratic":
            _, startup_interpolation, _ = recurrence_coefficients(spectrum, time_step)
            decay, interpolation, old_interpolation, _, _ = (
                quadratic_recurrence_coefficients(
                    spectrum,
                    time_step,
                    previous_step_size=time_step,
                )
            )
        else:
            decay, interpolation, _ = recurrence_coefficients(spectrum, time_step)
            startup_interpolation = interpolation
            old_interpolation = np.zeros_like(interpolation)
        modes = np.zeros((spectrum.rates.size, field_size))
        previous = np.zeros(field_size)
        penultimate = np.zeros(field_size)
        for step in range(1, num_steps + 1):
            current = np.sin(0.013 * step + coordinates)
            modes *= decay[:, None]
            if interpolant == "quadratic" and step > 1:
                modes += interpolation[:, None] * (current - previous)
                modes += old_interpolation[:, None] * (previous - penultimate)
            else:
                modes += startup_interpolation[:, None] * (current - previous)
            penultimate = previous
            previous = current
        checksum += float(np.sum(spectrum.weights @ modes))
    return checksum


def _nearest_mode_count(
    value: float,
    *,
    maximum: int,
    parity: str,
) -> int:
    minimum = 4 if parity == "even" else 3
    candidate = max(minimum, min(maximum, int(round(value))))
    if parity == "odd" and candidate % 2 == 0:
        candidate += 1 if candidate < maximum else -1
    elif parity == "even" and candidate % 2 != 0:
        candidate += 1 if candidate < maximum else -1
    return candidate


def _match_modal_cost(
    builder: Callable[[int], object],
    target_seconds: float,
    *,
    baseline_modes: int,
    repeats: int,
    num_steps: int,
    field_size: int,
    time_step: float,
    parity: str,
    maximum_modes: int = 257,
) -> tuple[int, float]:
    """Find a valid stored-mode count with cost nearest a target."""
    maximum = maximum_modes
    if parity == "odd" and maximum % 2 == 0:
        maximum -= 1
    elif parity == "even" and maximum % 2 != 0:
        maximum -= 1
    timings: dict[int, float] = {}

    def measure(count: int) -> float:
        if count not in timings:
            timings[count] = _median_runtime(
                lambda: _modal_update_workload(
                    builder(count),
                    alphas=(0.1, 0.5, 0.9),
                    num_steps=num_steps,
                    field_size=field_size,
                    time_step=time_step,
                ),
                repeats,
            )
        return timings[count]

    initial = _nearest_mode_count(
        baseline_modes,
        maximum=maximum,
        parity=parity,
    )
    initial_time = measure(initial)
    estimate = _nearest_mode_count(
        initial * target_seconds / initial_time,
        maximum=maximum,
        parity=parity,
    )
    candidates = {
        4 if parity == "even" else 3,
        initial,
        estimate,
        _nearest_mode_count(
            0.8 * estimate,
            maximum=maximum,
            parity=parity,
        ),
        _nearest_mode_count(
            1.2 * estimate,
            maximum=maximum,
            parity=parity,
        ),
    }
    for count in candidates:
        measure(count)
    for _ in range(2):
        best = min(
            timings,
            key=lambda count: abs(np.log(timings[count] / target_seconds)),
        )
        refined = _nearest_mode_count(
            best * target_seconds / timings[best],
            maximum=maximum,
            parity=parity,
        )
        measure(refined)
    best = min(
        timings,
        key=lambda count: abs(np.log(timings[count] / target_seconds)),
    )
    return best, timings[best]


def _match_sum_of_exponentials_cost(
    target_seconds: float,
    *,
    repeats: int,
    num_steps: int,
    field_size: int,
    min_step: float,
    t_final: float,
) -> tuple[SumOfExponentials, float, float]:
    """Select the tolerance whose derived spectrum is nearest a cost target."""
    timings: list[tuple[float, float, SumOfExponentials]] = []
    for requested_error in np.logspace(3.0, -8.0, 12):
        representation = SumOfExponentials(
            target_error=float(requested_error),
            min_step=min_step,
            t_final=t_final,
        )
        elapsed = _median_runtime(
            lambda representation=representation: _modal_update_workload(
                representation,
                alphas=(0.1, 0.5, 0.9),
                num_steps=num_steps,
                field_size=field_size,
                time_step=min_step,
            ),
            repeats,
        )
        timings.append((elapsed, float(requested_error), representation))
    elapsed, requested_error, representation = min(
        timings,
        key=lambda item: abs(np.log(item[0] / target_seconds)),
    )
    return representation, elapsed, requested_error


def plot_cost_matched_comparison(
    output: Path,
    *,
    csv_output: Path | None = None,
    baseline_modes: int = 41,
    target_error: float = 1.0e-8,
    timing_repeats: int = 7,
    timing_steps: int = 512,
    timing_field_size: int = 256,
    dpi: int = 220,
) -> Path:
    """Plot representation error at matched setup-and-update cost."""
    matplotlib, plt = _pyplot()
    if baseline_modes < 3 or baseline_modes % 2 == 0:
        raise ValueError("baseline_modes must be an odd integer of at least 3")
    if timing_repeats < 1 or timing_steps < 1 or timing_field_size < 1:
        raise ValueError("timing controls must be positive")

    comparison_min_step = 1.0e-1
    comparison_final_time = 1.0e1
    builders = _method_builders(target_error)
    baseline_timings = []
    for _, builder, _, _ in builders[:2]:
        baseline_timings.append(
            _median_runtime(
                lambda builder=builder: _modal_update_workload(
                    builder(baseline_modes),
                    alphas=(0.1, 0.5, 0.9),
                    num_steps=timing_steps,
                    field_size=timing_field_size,
                    time_step=comparison_min_step,
                ),
                timing_repeats,
            )
        )
    target_seconds = float(np.mean(baseline_timings))
    matched: list[tuple[str, object, str, Any, str, float]] = []
    for index, (label, builder, color, linestyle) in enumerate(builders):
        if index < 2:
            count = baseline_modes
            elapsed = baseline_timings[index]
        else:
            count, elapsed = _match_modal_cost(
                builder,
                target_seconds,
                baseline_modes=baseline_modes,
                repeats=timing_repeats,
                num_steps=timing_steps,
                field_size=timing_field_size,
                time_step=comparison_min_step,
                parity=_mode_count_parity(label),
                maximum_modes=(
                    42
                    if label.endswith("Gauss–Laguerre")
                    else 127
                    if label.startswith("Sine")
                    else 257
                ),
            )
        matched.append((label, builder(count), color, linestyle, str(count), elapsed))

    sum_of_exponentials, soe_elapsed, matched_soe_target = (
        _match_sum_of_exponentials_cost(
            target_seconds,
            repeats=timing_repeats,
            num_steps=timing_steps,
            field_size=timing_field_size,
            min_step=comparison_min_step,
            t_final=comparison_final_time,
        )
    )
    soe_counts = tuple(
        int(sum_of_exponentials.spectrum(alpha).rates.size) for alpha in (0.1, 0.5, 0.9)
    )
    soe_count_label = (
        str(soe_counts[0])
        if len(set(soe_counts)) == 1
        else f"{min(soe_counts)}–{max(soe_counts)}"
    )
    matched.append(
        (
            "Jiang sum of exponentials 2017",
            sum_of_exponentials,
            "#333333",
            (0, (7, 2)),
            soe_count_label,
            soe_elapsed,
        )
    )

    alphas = (0.1, 0.5, 0.9)
    times = np.logspace(
        np.log10(comparison_min_step),
        np.log10(comparison_final_time),
        401,
    )
    figure, axes = plt.subplots(
        2,
        3,
        figsize=(16.5, 10.0),
        sharex=True,
        sharey=True,
    )
    rows: list[dict[str, object]] = []
    metadata = _runtime_metadata()
    for row, problem in enumerate(("quadratic", "exponential")):
        for axis, alpha in zip(axes[row], alphas, strict=True):
            for (
                label,
                representation,
                color,
                linestyle,
                count_label,
                elapsed,
            ) in matched:
                errors = _relative_error(representation, alpha, times, problem)
                ratio = elapsed / target_seconds
                spectrum = representation.spectrum(alpha)
                count = _spectrum_size(spectrum)
                axis.loglog(
                    times,
                    errors,
                    label=f"{label}, {count_label} modes, {ratio:.2f}× cost",
                    color=color,
                    linestyle=linestyle,
                    linewidth=2.0,
                )
                rows.extend(
                    {
                        "method": label,
                        "alpha": alpha,
                        "problem": problem,
                        "time": float(time),
                        "num_modes": count,
                        "baseline_modes": baseline_modes,
                        "target_error": target_error,
                        "soe_target_error": (
                            matched_soe_target
                            if isinstance(representation, SumOfExponentials)
                            else ""
                        ),
                        "relative_error": float(error),
                        "modal_workload_seconds": elapsed,
                        "target_seconds": target_seconds,
                        "cost_ratio": ratio,
                        "timing_repeats": timing_repeats,
                        "timing_steps": timing_steps,
                        "timing_field_size": timing_field_size,
                        **metadata,
                    }
                    for time, error in zip(times, errors, strict=True)
                )
            axis.set_title(rf"Order $\alpha={alpha}$")
            axis.set_xlim(times[0], times[-1])
            axis.set_ylim(1.0e-16, 2.0)
    for axis in axes[-1]:
        axis.set_xlabel(r"Time $t$")
    for axis in axes[:, 0]:
        axis.set_ylabel("Absolute relative error")

    figure.suptitle(
        "Caputo representation accuracy at matched computational cost",
        fontsize=15,
        fontweight="bold",
        y=0.995,
    )
    figure.text(
        0.5,
        0.945,
        r"$\dot u=2t,\ u(0)=0,\ u=t^2$, "
        r"$D_C^\alpha u=2t^{2-\alpha}/\Gamma(3-\alpha)$",
        ha="center",
        fontsize=12,
    )
    figure.text(
        0.5,
        0.535,
        r"$\dot u=e^t,\ u(0)=1,\ u=e^t$, "
        r"$D_C^\alpha u=e^tP(1-\alpha,t)$",
        ha="center",
        fontsize=12,
    )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, 0.005),
        columnspacing=1.2,
        fontsize=8.5,
    )
    figure.text(
        0.5,
        0.13,
        f"Reference budget: {baseline_modes}-mode Birk–Song and Diethelm2008. "
        f"Each timing includes setup and {timing_steps} static-memory updates "
        f"over {timing_field_size} values. Every legend cost is measured against "
        f"this budget. The matched SOE target is {matched_soe_target:.0e}.",
        ha="center",
        fontsize=9,
    )
    figure.subplots_adjust(
        top=0.90,
        bottom=0.23,
        hspace=0.62,
        wspace=0.12,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    if csv_output is not None:
        _write_rows(csv_output, rows)
    return output


def _prepare_recurrence_coefficients(
    spectrum: object,
    step_size: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Exercise the representation-dependent coefficient path used by a stepper."""
    decay, interpolation, implicit_weight = recurrence_coefficients(
        spectrum,
        step_size,
    )
    history_weights = spectrum.weights * decay
    return decay, interpolation, history_weights, implicit_weight


def _prepare_modal_coefficients(
    spectrum: object,
    alpha: float,
    step_size: float,
) -> object:
    if hasattr(spectrum, "frequencies"):
        return oscillator_coefficients(spectrum, alpha, step_size)
    return _prepare_recurrence_coefficients(spectrum, step_size)


def plot_sum_of_exponentials_refinement(
    output: Path,
    *,
    csv_output: Path | None = None,
    targets: tuple[float, ...] = (1.0e-2, 1.0e-4, 1.0e-6, 1.0e-8),
    repeats: int = 5,
    dpi: int = 220,
) -> Path:
    """Plot tolerance, derived mode count, and construction cost."""
    matplotlib, plt = _pyplot()
    alphas = (0.2, 0.5, 0.8)
    colors = ("#168c8c", "#d59618", "#7552a3")
    times = np.geomspace(1.0e-2, 1.0, 2001)
    rows: list[dict[str, object]] = []
    metadata = _runtime_metadata()
    figure, axes = plt.subplots(1, 3, figsize=(16.5, 5.2))
    for alpha, color in zip(alphas, colors, strict=True):
        errors = []
        counts = []
        timings = []
        for target in targets:
            representation = SumOfExponentials(
                target_error=target,
                t_final=1.0,
                min_step=1.0e-2,
            )
            spectrum = representation.spectrum(alpha)
            power_weights = (
                spectrum.weights * spectrum.rates * gamma(1.0 - alpha) / alpha
            )
            observed = np.exp(-np.outer(times, spectrum.rates)) @ power_weights
            error = float(np.max(np.abs(np.power(times, -1.0 - alpha) - observed)))
            construction = _median_runtime(
                lambda representation=representation, alpha=alpha: (
                    representation.spectrum(alpha)
                ),
                repeats,
            )
            errors.append(error)
            counts.append(spectrum.rates.size)
            timings.append(1.0e3 * construction)
            rows.append(
                {
                    "method": "Jiang sum of exponentials",
                    "alpha": alpha,
                    "target_error": target,
                    "min_step": 1.0e-2,
                    "t_final": 1.0,
                    "num_modes": spectrum.rates.size,
                    "maximum_kernel_error": error,
                    "spectrum_construction_seconds": construction,
                    **metadata,
                }
            )
        label = rf"$\alpha={alpha}$"
        axes[0].loglog(targets, errors, "o-", color=color, label=label)
        axes[1].semilogx(targets, counts, "o-", color=color)
        axes[2].loglog(targets, timings, "o-", color=color)
    axes[0].loglog(targets, targets, color="#333333", linestyle=":")
    axes[0].set_ylabel("Maximum absolute kernel error")
    axes[0].set_title("Verified error on the requested interval")
    axes[1].set_ylabel("Derived mode count")
    axes[1].set_title("Storage selected by the tolerance")
    axes[2].set_ylabel("Median construction time (ms)")
    axes[2].set_title("Spectrum construction")
    for axis in axes:
        axis.set_xlabel("Requested absolute error")
        axis.grid(True, alpha=0.35)
        axis.invert_xaxis()
    axes[0].legend(frameon=False)
    figure.suptitle(
        r"Jiang sum of exponentials on $[10^{-2},1]$",
        fontsize=15,
        fontweight="bold",
    )
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    if csv_output is not None:
        _write_rows(csv_output, rows)
    return output


def plot_timings(
    output: Path,
    *,
    csv_output: Path | None = None,
    num_modes: int = 41,
    target_error: float = 1.0e-8,
    soe_target_error: float = 1.0e-4,
    repeats: int = 25,
    dpi: int = 220,
) -> Path:
    """Plot warm spectrum and recurrence-coefficient preparation timings."""
    matplotlib, plt = _pyplot()
    if num_modes < 4:
        raise ValueError("num_modes must be at least 4")
    if repeats < 1:
        raise ValueError("repeats must be positive")

    specs = _method_specs(num_modes, target_error) + (
        (
            "Jiang sum of exponentials 2017",
            lambda: SumOfExponentials(
                target_error=soe_target_error,
                min_step=1.0e-5,
                t_final=1.0e3,
            ),
            "#333333",
            (0, (7, 2)),
        ),
    )
    alphas = (0.1, 0.5, 0.9)
    construction = np.empty((len(specs), len(alphas)))
    coefficient_preparation = np.empty_like(construction)
    rows: list[dict[str, object]] = []
    metadata = _runtime_metadata()

    for method_index, (_, factory, _, _) in enumerate(specs):
        for alpha_index, alpha in enumerate(alphas):
            construction[method_index, alpha_index] = _median_runtime(
                lambda factory=factory, alpha=alpha: factory().spectrum(alpha),
                repeats,
            )
            spectrum = factory().spectrum(alpha)
            coefficient_preparation[method_index, alpha_index] = _median_runtime(
                lambda spectrum=spectrum, alpha=alpha: _prepare_modal_coefficients(
                    spectrum,
                    alpha,
                    0.01,
                ),
                repeats,
            )
            rows.append(
                {
                    "method": specs[method_index][0],
                    "alpha": alpha,
                    "num_modes": _spectrum_size(spectrum),
                    "target_error": target_error,
                    "soe_target_error": soe_target_error,
                    "step_size": 0.01,
                    "repeats": repeats,
                    "spectrum_construction_seconds": construction[
                        method_index,
                        alpha_index,
                    ],
                    "coefficient_preparation_seconds": coefficient_preparation[
                        method_index,
                        alpha_index,
                    ],
                    **metadata,
                }
            )

    figure, axes = plt.subplots(2, 1, figsize=(16.5, 10.0), sharey=True)
    positions = np.arange(len(specs), dtype=np.float64)
    bar_height = 0.23
    alpha_colors = ("#168c8c", "#d59618", "#7552a3")
    for alpha_index, (alpha, color) in enumerate(
        zip(alphas, alpha_colors, strict=True)
    ):
        offset = (alpha_index - 1) * bar_height
        for axis, values in zip(
            axes,
            (construction, coefficient_preparation),
            strict=True,
        ):
            axis.barh(
                positions + offset,
                1.0e3 * values[:, alpha_index],
                height=bar_height * 0.88,
                color=color,
                label=rf"$\alpha={alpha}$",
            )

    labels = [label for label, _, _, _ in specs]
    for axis in axes:
        axis.set_yticks(positions, labels)
        axis.invert_yaxis()
        axis.grid(axis="x", alpha=0.35)
    from matplotlib.ticker import NullFormatter

    axes[0].set_xscale("log")
    axes[0].xaxis.set_minor_formatter(NullFormatter())
    axes[0].set_xlabel("Median wall time (ms, log scale)")
    axes[1].set_xlabel("Median wall time (ms)")
    axes[0].set_title(r"Warm spectrum construction")
    axes[1].set_title(r"Prepare modal update coefficients for $\Delta t=0.01$")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles[: len(alphas)],
        legend_labels[: len(alphas)],
        loc="lower center",
        ncol=len(alphas),
        frameon=False,
    )
    figure.suptitle(
        "Diffusive representation setup timings",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )
    laguerre_modes = _compatible_mode_count(
        "Diethelm 2022 Gauss–Laguerre",
        num_modes,
    )
    simpson_modes = _compatible_mode_count(
        "Diethelm 2022 Simpson",
        num_modes,
    )
    figure.text(
        0.5,
        0.035,
        f"Median of {repeats} warm runs. Fixed-count methods use "
        f"{laguerre_modes}–{simpson_modes} modes. "
        "The sum of exponentials derives its count from its target.",
        ha="center",
        fontsize=9,
    )
    figure.subplots_adjust(left=0.22, bottom=0.13, top=0.91, hspace=0.36)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi)
    plt.close(figure)
    if csv_output is not None:
        _write_rows(csv_output, rows)
    return output


def plot_diethelm2008_refinement(
    output: Path,
    *,
    csv_output: Path | None = None,
    node_counts: tuple[int, ...] = (8, 16, 32, 64, 128, 256),
    dpi: int = 220,
) -> Path:
    """Plot Diethelm2008-2008 error as its Gauss-Jacobi rule is refined."""
    matplotlib, plt = _pyplot()
    if not node_counts or any(
        isinstance(count, bool) or not 1 <= count <= 256 for count in node_counts
    ):
        raise ValueError("node counts must be integers between 1 and 256")

    alphas = (0.1, 0.5, 0.9)
    times = np.logspace(-3.0, 3.0, 401)
    colors = matplotlib.colormaps["viridis"](np.linspace(0.05, 0.95, len(node_counts)))
    figure, axes = plt.subplots(
        2,
        3,
        figsize=(16.5, 10.0),
        sharex=True,
        sharey=True,
    )
    rows: list[dict[str, object]] = []
    metadata = _runtime_metadata()
    for row, problem in enumerate(("quadratic", "exponential")):
        for axis, alpha in zip(axes[row], alphas, strict=True):
            for count, color in zip(node_counts, colors, strict=True):
                errors = _relative_error(
                    Diethelm2008(count),
                    alpha,
                    times,
                    problem,
                )
                axis.loglog(
                    times,
                    errors,
                    color=color,
                    linewidth=2.0,
                    label=f"{count} nodes",
                )
                rows.extend(
                    {
                        "method": "Diethelm 2008",
                        "alpha": alpha,
                        "problem": problem,
                        "time": float(time),
                        "num_modes": count,
                        "relative_error": float(error),
                        **metadata,
                    }
                    for time, error in zip(times, errors, strict=True)
                )
            axis.set_title(rf"Order $\alpha={alpha}$")
            axis.set_xlim(times[0], times[-1])
            axis.set_ylim(1.0e-16, 2.0)
    for axis in axes[-1]:
        axis.set_xlabel(r"Time $t$")
    for axis in axes[:, 0]:
        axis.set_ylabel("Absolute relative error")

    figure.suptitle(
        "Diethelm 2008 Gauss–Jacobi node refinement",
        fontsize=15,
        fontweight="bold",
        y=0.995,
    )
    figure.text(
        0.5,
        0.945,
        r"$\dot u=2t,\ u(0)=0,\ u=t^2$; "
        r"$D_C^\alpha u=2t^{2-\alpha}/\Gamma(3-\alpha)$",
        ha="center",
        fontsize=12,
    )
    figure.text(
        0.5,
        0.535,
        r"$\dot u=e^t,\ u(0)=1,\ u=e^t$; "
        r"$D_C^\alpha u=e^tP(1-\alpha,t)$",
        ha="center",
        fontsize=12,
    )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.012),
        frameon=False,
    )
    figure.text(
        0.5,
        0.105,
        "Exact auxiliary-mode evolution; no time-step or PDE error.",
        ha="center",
    )
    figure.subplots_adjust(
        top=0.90,
        bottom=0.18,
        hspace=0.62,
        wspace=0.12,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    if csv_output is not None:
        _write_rows(csv_output, rows)
    return output


def _parse_node_counts(value: str) -> tuple[int, ...]:
    try:
        counts = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "node counts must be comma-separated integers"
        ) from error
    if not counts or any(not 1 <= count <= 256 for count in counts):
        raise argparse.ArgumentTypeError("node counts must lie between 1 and 256")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=(Path(__file__).resolve().parent / "benchmarks-output"),
    )
    parser.add_argument(
        "--figure-directory",
        type=Path,
        help="write figures separately from the CSV output directory",
    )
    parser.add_argument(
        "--diethelm-nodes",
        type=_parse_node_counts,
        default=(8, 16, 32, 64, 128, 256),
    )
    parser.add_argument(
        "--timing-repeats",
        type=int,
        default=25,
    )
    parser.add_argument("--cost-baseline-modes", type=int, default=41)
    parser.add_argument("--cost-timing-repeats", type=int, default=7)
    parser.add_argument("--num-modes", type=int, default=41)
    parser.add_argument("--target-error", type=float, default=1.0e-8)
    parser.add_argument("--soe-target-error", type=float, default=1.0e-4)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--smoke", action="store_true")
    arguments = parser.parse_args()
    if arguments.smoke:
        arguments.num_modes = 9
        arguments.diethelm_nodes = (8, 16)
        arguments.timing_repeats = 1
        arguments.cost_timing_repeats = 1
    output = arguments.output_directory
    figures = arguments.figure_directory or output
    output.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    comparison_stem = output / "diffusive-representation-errors"
    timing_stem = output / "diffusive-representation-timings"
    cost_stem = output / "diffusive-representation-cost-matched"
    refinement_stem = output / "diethelm2008-node-refinement-errors"
    exponential_stem = output / "sum-of-exponentials-refinement"
    print(
        plot_comparison(
            figures / "diffusive-representation-errors.png",
            csv_output=comparison_stem.with_suffix(".csv"),
            num_modes=arguments.num_modes,
            target_error=arguments.target_error,
            soe_target_error=arguments.soe_target_error,
            dpi=arguments.dpi,
        )
    )
    print(
        plot_timings(
            figures / "diffusive-representation-timings.png",
            csv_output=timing_stem.with_suffix(".csv"),
            num_modes=arguments.num_modes,
            target_error=arguments.target_error,
            soe_target_error=arguments.soe_target_error,
            repeats=arguments.timing_repeats,
            dpi=arguments.dpi,
        )
    )
    print(
        plot_cost_matched_comparison(
            figures / "diffusive-representation-cost-matched.png",
            csv_output=cost_stem.with_suffix(".csv"),
            baseline_modes=arguments.cost_baseline_modes,
            target_error=arguments.target_error,
            timing_repeats=arguments.cost_timing_repeats,
            timing_steps=16 if arguments.smoke else 512,
            timing_field_size=64 if arguments.smoke else 256,
            dpi=arguments.dpi,
        )
    )
    print(
        plot_sum_of_exponentials_refinement(
            figures / "sum-of-exponentials-refinement.png",
            csv_output=exponential_stem.with_suffix(".csv"),
            targets=(1.0e-2, 1.0e-4)
            if arguments.smoke
            else (
                1.0e-2,
                1.0e-4,
                1.0e-6,
                1.0e-8,
            ),
            repeats=arguments.timing_repeats,
            dpi=arguments.dpi,
        )
    )
    print(
        plot_diethelm2008_refinement(
            figures / "diethelm2008-node-refinement-errors.png",
            csv_output=refinement_stem.with_suffix(".csv"),
            node_counts=arguments.diethelm_nodes,
            dpi=arguments.dpi,
        )
    )


if __name__ == "__main__":
    main()
