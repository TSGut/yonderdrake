"""Tests for the tolerance-driven Jiang sum of exponentials."""

from __future__ import annotations

from math import gamma

import mpmath as mp
import numpy as np
import pytest

from yonderdrake import SumOfExponentials
from yonderdrake.time.coefficients import recurrence_coefficients
from yonderdrake.time.representations import (
    validate_checkpoint_representation,
)


def _jiang_power_weights(spectrum, alpha: float) -> np.ndarray:
    return spectrum.weights * spectrum.rates * gamma(1.0 - alpha) / alpha


@pytest.mark.unit
@pytest.mark.parametrize("alpha", [0.2, 0.5, 0.8])
def test_sum_of_exponentials_meets_its_kernel_target(alpha: float) -> None:
    target_error = 1.0e-6
    representation = SumOfExponentials(
        target_error=target_error,
        t_final=1.0,
        min_step=1.0e-2,
    )
    spectrum = representation.spectrum(alpha)
    times = np.geomspace(1.0e-2, 1.0, 4097)
    observed = np.exp(-np.outer(times, spectrum.rates)) @ (
        _jiang_power_weights(spectrum, alpha)
    )
    error = np.max(np.abs(np.power(times, -1.0 - alpha) - observed))

    assert error <= target_error
    assert spectrum.metadata["achieved_kernel_error"] <= target_error
    assert spectrum.metadata["target_achievable"] is True
    assert spectrum.rates.size == spectrum.metadata["num_modes"]
    assert np.all(spectrum.rates > 0.0)
    assert np.all(spectrum.weights > 0.0)
    assert not hasattr(representation, "num_modes")


@pytest.mark.unit
def test_sum_of_exponentials_derives_more_modes_for_a_tighter_target() -> None:
    loose = SumOfExponentials(
        target_error=1.0e-3,
        t_final=1.0,
        min_step=1.0e-2,
    ).spectrum(0.6)
    tight = SumOfExponentials(
        target_error=1.0e-7,
        t_final=1.0,
        min_step=1.0e-2,
    ).spectrum(0.6)
    assert tight.rates.size > loose.rates.size
    assert (
        tight.metadata["achieved_kernel_error"]
        < loose.metadata["achieved_kernel_error"]
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("alpha", "target_error", "t_final", "min_step", "order", "modes"),
    [
        (0.2, 1.0e-3, 1.0, 1.0e-2, 4, 48),
        (0.2, 1.0e-6, 1.0, 1.0e-2, 6, 78),
        (0.5, 1.0e-6, 1.0, 1.0e-2, 8, 104),
        (0.5, 1.0e-6, 2.0, 2.0e-2, 6, 78),
        (0.8, 1.0e-3, 1.0, 1.0e-2, 6, 72),
        (0.8, 1.0e-6, 2.0, 2.0e-2, 8, 104),
    ],
)
def test_package_jacobi_rule_preserves_accepted_construction(
    alpha: float,
    target_error: float,
    t_final: float,
    min_step: float,
    order: int,
    modes: int,
) -> None:
    spectrum = SumOfExponentials(
        target_error=target_error,
        t_final=t_final,
        min_step=min_step,
    ).spectrum(alpha)
    assert spectrum.metadata["quadrature_order"] == order
    assert spectrum.metadata["num_modes"] == modes
    assert spectrum.metadata["achieved_kernel_error"] <= target_error


def _relaxation(step_size: float) -> float:
    alpha = 0.6
    spectrum = SumOfExponentials(
        target_error=1.0e-8,
        t_final=1.0,
        min_step=0.025,
    ).spectrum(alpha)
    decay, interpolation, implicit = recurrence_coefficients(
        spectrum,
        step_size,
    )
    modes = np.zeros(spectrum.rates.size)
    value = 1.0
    for _ in range(round(1.0 / step_size)):
        history = np.dot(spectrum.weights * decay, modes)
        updated = (implicit * value - history) / (implicit + 1.0)
        modes = decay * modes + interpolation * (updated - value)
        value = updated
    return value


@pytest.mark.unit
def test_sum_of_exponentials_mittag_leffler_error_refines_in_time() -> None:
    with mp.workdps(50):
        exact = float(
            mp.nsum(
                lambda index: (-1) ** index / mp.gamma(0.6 * index + 1.0),
                [0, mp.inf],
            )
        )
    coarse_error = abs(_relaxation(0.1) - exact)
    medium_error = abs(_relaxation(0.05) - exact)
    fine_error = abs(_relaxation(0.025) - exact)
    assert medium_error < coarse_error
    assert fine_error < medium_error
    assert fine_error < 3.0e-3


@pytest.mark.unit
def test_sum_of_exponentials_power_identity_has_l1_accuracy() -> None:
    alpha = 0.6
    step_size = 0.01
    spectrum = SumOfExponentials(
        target_error=1.0e-6,
        t_final=1.0,
        min_step=step_size,
    ).spectrum(alpha)
    decay, interpolation, implicit = recurrence_coefficients(
        spectrum,
        step_size,
    )
    modes = np.zeros(spectrum.rates.size)
    previous = 0.0
    maximum_error = 0.0
    for step in range(1, 101):
        time = step * step_size
        value = time * time
        observed = (
            np.dot(spectrum.weights * decay, modes)
            + implicit * (value - previous)
        )
        exact = 2.0 * time ** (2.0 - alpha) / gamma(3.0 - alpha)
        maximum_error = max(maximum_error, abs(observed - exact))
        modes = decay * modes + interpolation * (value - previous)
        previous = value
    assert maximum_error < 9.0e-4


@pytest.mark.unit
def test_sum_of_exponentials_interval_guards_and_checkpoint_metadata() -> None:
    representation = SumOfExponentials(
        target_error=1.0e-4,
        t_final=0.5,
        min_step=0.05,
    )
    spectrum = representation.spectrum(0.4)
    recurrence_coefficients(spectrum, 0.05, final_time=0.5)
    with pytest.raises(ValueError, match="below.*min_step"):
        recurrence_coefficients(spectrum, 0.025)
    with pytest.raises(ValueError, match="exceeds.*t_final"):
        recurrence_coefficients(spectrum, 0.05, final_time=0.55)
    metadata = representation.describe(0.4)
    validate_checkpoint_representation(metadata, representation, 0.4)
    changed = dict(metadata)
    changed["t_final"] = 0.6
    with pytest.raises(ValueError, match="does not match"):
        validate_checkpoint_representation(changed, representation, 0.4)


@pytest.mark.unit
@pytest.mark.parametrize(
    "arguments,error",
    [
        ({"target_error": 0.0, "t_final": 1.0, "min_step": 0.1}, "positive"),
        ({"target_error": 1.0e-4, "t_final": 0.0, "min_step": 0.1}, "positive"),
        ({"target_error": 1.0e-4, "t_final": 1.0, "min_step": 0.0}, "positive"),
        (
            {"target_error": 1.0e-4, "t_final": 0.1, "min_step": 0.2},
            "must not exceed",
        ),
    ],
)
def test_sum_of_exponentials_validates_its_interval(arguments, error: str) -> None:
    with pytest.raises(ValueError, match=error):
        SumOfExponentials(**arguments)
