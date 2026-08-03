"""Numerical tests for the comparison-only sine diffusive method."""

from __future__ import annotations

from math import gamma

import mpmath as mp
import numpy as np
import pytest

from yonderdrake import ModeCountAdvisoryWarning, SineDiffusive
from yonderdrake.time.coefficients import oscillator_coefficients


@pytest.mark.unit
@pytest.mark.parametrize("alpha", [0.1, 0.5, 0.9])
@pytest.mark.parametrize("num_modes", [8, 32, 128])
def test_generalized_laguerre_rule_matches_high_precision(
    alpha: float,
    num_modes: int,
) -> None:
    with mp.workdps(70):
        reference_nodes, reference_weights = mp.gauss_quadrature(
            num_modes,
            "glaguerre",
            alpha,
        )
        expected_nodes = np.asarray(
            [float(value) for value in reference_nodes],
        )
        expected_weights = np.asarray(
            [
                float(reference_weights[index] * mp.exp(reference_nodes[index]))
                for index in range(num_modes)
            ],
        )
    spectrum = SineDiffusive(num_modes).spectrum(alpha)
    np.testing.assert_allclose(
        spectrum.frequencies,
        expected_nodes,
        rtol=2.0e-12,
        atol=0.0,
    )
    np.testing.assert_allclose(
        spectrum.weights,
        expected_weights,
        rtol=2.0e-12,
        atol=0.0,
    )


@pytest.mark.unit
@pytest.mark.parametrize("alpha", [0.1, 0.5, 0.9])
def test_log_scaled_generalized_laguerre_extends_past_scipy_range(
    alpha: float,
) -> None:
    with mp.workdps(70):
        reference_nodes, reference_weights = mp.gauss_quadrature(
            256,
            "glaguerre",
            alpha,
        )
        expected_nodes = np.asarray(
            [float(value) for value in reference_nodes],
        )
        expected_weights = np.asarray(
            [
                float(reference_weights[index] * mp.exp(reference_nodes[index]))
                for index in range(256)
            ],
        )
    with pytest.warns(ModeCountAdvisoryWarning, match="num_modes=256"):
        spectrum = SineDiffusive(256).spectrum(alpha)
    assert np.all(np.isfinite(spectrum.frequencies))
    assert np.all(np.isfinite(spectrum.weights))
    assert np.all(spectrum.weights > 0.0)
    np.testing.assert_allclose(
        spectrum.frequencies,
        expected_nodes,
        rtol=5.0e-12,
        atol=0.0,
    )
    np.testing.assert_allclose(
        spectrum.weights,
        expected_weights,
        rtol=3.0e-10,
        atol=0.0,
    )


def _sdr_relaxation(
    alpha: float,
    step_size: float,
    num_steps: int,
    num_modes: int,
) -> np.ndarray:
    spectrum = SineDiffusive(num_modes).spectrum(alpha)
    (
        cosine,
        sine_over_frequency,
        negative_frequency_sine,
        position_forcing,
        velocity_forcing,
        implicit_weight,
    ) = oscillator_coefficients(spectrum, alpha, step_size)
    positions = np.zeros(num_modes)
    velocities = np.zeros(num_modes)
    values = np.empty(num_steps + 1)
    values[0] = 1.0
    for step in range(1, num_steps + 1):
        history = np.dot(
            spectrum.weights,
            cosine * positions + sine_over_frequency * velocities,
        )
        value = (
            implicit_weight * values[step - 1] - history
        ) / (implicit_weight + 1.0)
        increment = value - values[step - 1]
        old_positions = positions.copy()
        positions = (
            cosine * old_positions
            + sine_over_frequency * velocities
            + position_forcing * increment
        )
        velocities = (
            negative_frequency_sine * old_positions
            + cosine * velocities
            + velocity_forcing * increment
        )
        values[step] = value
    return values


def _full_history_relaxation(
    alpha: float,
    step_size: float,
    num_steps: int,
) -> np.ndarray:
    values = np.empty(num_steps + 1)
    values[0] = 1.0
    increments: list[float] = []
    implicit_weight = step_size ** (-alpha) / gamma(2.0 - alpha)
    for step in range(1, num_steps + 1):
        target = step * step_size
        history = 0.0
        for interval, increment in enumerate(increments):
            left = interval * step_size
            right = left + step_size
            weight = (
                (target - left) ** (1.0 - alpha)
                - (target - right) ** (1.0 - alpha)
            ) / (step_size * gamma(2.0 - alpha))
            history += weight * increment
        values[step] = (
            implicit_weight * values[step - 1] - history
        ) / (implicit_weight + 1.0)
        increments.append(values[step] - values[step - 1])
    return values


@pytest.mark.unit
@pytest.mark.parametrize("alpha", [0.2, 0.5, 0.8])
def test_linear_power_identity_improves_with_mode_count(alpha: float) -> None:
    exact = 1.0 / gamma(2.0 - alpha)
    errors = []
    for num_modes in (16, 64, 128):
        spectrum = SineDiffusive(num_modes).spectrum(alpha)
        *_, implicit_weight = oscillator_coefficients(spectrum, alpha, 1.0)
        errors.append(abs(implicit_weight - exact))
    assert errors[2] < errors[1] < errors[0]


@pytest.mark.unit
def test_long_time_error_is_bounded_but_does_not_decay() -> None:
    alpha = 0.5
    step_size = 0.05
    num_steps = 1000
    observed = _sdr_relaxation(alpha, step_size, num_steps, 128)
    reference = _full_history_relaxation(alpha, step_size, num_steps)
    error = np.abs(observed - reference)
    assert np.all(np.isfinite(observed))
    assert np.max(error) < 0.2
    assert error[200] > 5.0 * error[20]
    assert np.max(error[500:]) > 0.05


@pytest.mark.unit
def test_exact_rotation_preserves_an_unforced_mode_energy() -> None:
    alpha = 0.4
    step_size = 0.037
    spectrum = SineDiffusive(8).spectrum(alpha)
    cosine, sine_over_frequency, negative_frequency_sine, *_ = (
        oscillator_coefficients(spectrum, alpha, step_size)
    )
    positions = np.linspace(0.1, 0.8, 8)
    velocities = np.linspace(-0.4, 0.3, 8)
    initial_energy = np.square(velocities) + np.square(
        spectrum.frequencies * positions
    )
    old_positions = positions.copy()
    positions = cosine * old_positions + sine_over_frequency * velocities
    velocities = (
        negative_frequency_sine * old_positions + cosine * velocities
    )
    final_energy = np.square(velocities) + np.square(
        spectrum.frequencies * positions
    )
    np.testing.assert_allclose(final_energy, initial_energy, rtol=2.0e-14)
