"""Tests for the Diethelm2008-2022 truncated representation."""

from __future__ import annotations

import json
from math import gamma, pi, sin

import mpmath as mp
import numpy as np
import pytest
from scipy.special import roots_laguerre, roots_legendre

import yonderdrake
from yonderdrake import Diethelm2022, ModeCountAdvisoryWarning
from yonderdrake.time.coefficients import recurrence_coefficients
from yonderdrake.time.representations import (
    validate_checkpoint_representation,
)


def maximum_linear_caputo_error(
    representation: Diethelm2022,
    alpha: float,
) -> float:
    """Compare the modes with the exact Caputo derivative of y(t) = t."""
    spectrum = representation.spectrum(alpha)
    times = np.logspace(-2.0, 2.0, 81)
    mode_values = -np.expm1(
        -np.outer(times, spectrum.rates)
    ) / spectrum.rates
    observed = mode_values @ spectrum.weights
    expected = times ** (1.0 - alpha) / gamma(2.0 - alpha)
    return float(np.max(np.abs(observed / expected - 1.0)))


@pytest.mark.unit
def test_explicit_trapezoidal_coefficients_match_the_real_line_density() -> None:
    alpha = 0.35
    radius = 3.0
    rate_scale = 2.5
    representation = Diethelm2022(
        9,
        truncation_radius=radius,
        rate_scale=rate_scale,
    )
    spectrum = representation.spectrum(alpha)

    nodes = np.linspace(-radius, radius, 9)
    spacing = 2.0 * radius / 8
    endpoint_factors = np.ones(9)
    endpoint_factors[[0, -1]] = 0.5
    np.testing.assert_allclose(
        spectrum.rates,
        rate_scale * np.exp(nodes),
    )
    np.testing.assert_allclose(
        spectrum.weights,
        spacing
        * endpoint_factors
        * (sin(pi * alpha) / pi)
        * rate_scale**alpha
        * np.exp(alpha * nodes),
    )
    assert spectrum.metadata["truncation_radius_source"] == "user"
    assert spectrum.metadata["quadrature"] == "composite-trapezoidal"
    assert spectrum.metadata["ordering"] == "increasing_rate"


@pytest.mark.unit
def test_explicit_simpson_coefficients_match_the_real_line_density() -> None:
    alpha = 0.35
    radius = 3.0
    representation = Diethelm2022(
        9,
        quadrature="simpson",
        truncation_radius=radius,
    )
    spectrum = representation.spectrum(alpha)

    nodes = np.linspace(-radius, radius, 9)
    spacing = 2.0 * radius / 8
    factors = np.array([1.0, 4.0, 2.0, 4.0, 2.0, 4.0, 2.0, 4.0, 1.0])
    np.testing.assert_allclose(spectrum.rates, np.exp(nodes))
    np.testing.assert_allclose(
        spectrum.weights,
        (spacing / 3.0)
        * factors
        * (sin(pi * alpha) / pi)
        * np.exp(alpha * nodes),
    )
    assert spectrum.metadata["quadrature"] == "composite-simpson"


@pytest.mark.unit
def test_explicit_gauss_legendre_coefficients_match_density() -> None:
    alpha = 0.35
    radius = 3.0
    rate_scale = 2.5
    representation = Diethelm2022(
        9,
        quadrature="gauss-legendre",
        truncation_radius=radius,
        rate_scale=rate_scale,
    )
    spectrum = representation.spectrum(alpha)

    reference_nodes, quadrature_weights = roots_legendre(9)
    nodes = radius * reference_nodes
    np.testing.assert_allclose(
        spectrum.rates,
        rate_scale * np.exp(nodes),
    )
    np.testing.assert_allclose(
        spectrum.weights,
        radius
        * quadrature_weights
        * (sin(pi * alpha) / pi)
        * rate_scale**alpha
        * np.exp(alpha * nodes),
    )
    assert spectrum.metadata["quadrature"] == "Gauss-Legendre"
    assert spectrum.metadata["log_rate_spacing"] is None
    assert spectrum.metadata["target_achievable"] is None
    assert (
        spectrum.metadata["discretization_error_model"]
        == "analytic_strip_radius_surrogate"
    )


@pytest.mark.unit
def test_gauss_laguerre_coefficients_match_published_construction() -> None:
    alpha = 0.35
    num_modes = 8
    nodes, laguerre_weights = roots_laguerre(num_modes // 2)
    coefficient = sin(pi * alpha) / pi
    negative_rates = np.exp(-nodes / alpha)
    positive_rates = np.exp(nodes / (1.0 - alpha))
    negative_weights = coefficient * laguerre_weights / alpha
    positive_weights = (
        coefficient
        * laguerre_weights
        * np.exp(nodes / (1.0 - alpha))
        / (1.0 - alpha)
    )

    representation = Diethelm2022(
        num_modes,
        quadrature="gauss-laguerre",
    )
    spectrum = representation.spectrum(alpha)

    np.testing.assert_allclose(
        spectrum.rates,
        np.concatenate((negative_rates[::-1], positive_rates)),
        rtol=2.0e-13,
    )
    np.testing.assert_allclose(
        spectrum.weights,
        np.concatenate((negative_weights[::-1], positive_weights)),
        rtol=2.0e-13,
    )
    assert spectrum.rates.size == num_modes
    assert spectrum.metadata["num_modes"] == num_modes
    assert spectrum.metadata["quadrature"] == "Gauss-Laguerre"
    assert spectrum.metadata["truncation_radius"] is None
    assert spectrum.metadata["target_achievable"] is None
    description = representation.describe()
    assert description["target_error"] is None
    assert description["decay_scale"] is None
    assert description["configurable_parameters"] == ("quadrature",)
    json.dumps(representation.describe(alpha), allow_nan=False)


@pytest.mark.unit
@pytest.mark.parametrize("alpha", [0.1, 0.5, 0.9])
def test_gauss_laguerre_power_identity(alpha: float) -> None:
    spectrum = Diethelm2022(
        32,
        quadrature="gauss-laguerre",
    ).spectrum(alpha)
    observed = np.sum(
        spectrum.weights * -np.expm1(-spectrum.rates) / spectrum.rates
    )
    exact = 1.0 / gamma(2.0 - alpha)
    assert abs(observed / exact - 1.0) < 3.0e-3


def _gauss_laguerre_relaxation(alpha: float) -> float:
    step_size = 0.0025
    spectrum = Diethelm2022(
        32,
        quadrature="gauss-laguerre",
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
@pytest.mark.parametrize("alpha", [0.1, 0.5, 0.9])
def test_gauss_laguerre_mittag_leffler_relaxation(alpha: float) -> None:
    with mp.workdps(50):
        exact = float(
            mp.nsum(
                lambda index: (-1) ** index / mp.gamma(alpha * index + 1.0),
                [0, mp.inf],
            )
        )
    assert abs(_gauss_laguerre_relaxation(alpha) - exact) < 5.0e-4


@pytest.mark.unit
def test_automatic_radius_balances_tail_and_grid_error() -> None:
    alpha = 0.4
    loose = Diethelm2022(65, target_error=1.0e-4)
    strict = Diethelm2022(65, target_error=1.0e-8)
    refined = Diethelm2022(257, target_error=1.0e-8)
    larger_solution = Diethelm2022(
        65,
        target_error=1.0e-4,
        decay_scale=10.0,
    )

    loose_metadata = loose.spectrum(alpha).metadata
    strict_metadata = strict.spectrum(alpha).metadata
    refined_metadata = refined.spectrum(alpha).metadata
    larger_metadata = larger_solution.spectrum(alpha).metadata
    assert (
        strict_metadata["truncation_radius"]
        > loose_metadata["truncation_radius"]
    )
    assert (
        larger_metadata["truncation_radius"]
        > loose_metadata["truncation_radius"]
    )
    assert (
        refined_metadata["estimated_total_bound"]
        < strict_metadata["estimated_total_bound"]
    )
    assert loose_metadata["target_achievable"]
    assert not strict_metadata["target_achievable"]
    assert refined_metadata["target_achievable"]
    assert strict_metadata["estimated_total_bound"] == pytest.approx(
        strict_metadata["estimated_tail_bound"]
        + strict_metadata["estimated_discretization_bound"]
    )
    assert loose_metadata["truncation_radius_source"] == "target_error"
    assert (
        strict_metadata["truncation_radius_source"]
        == "balanced_tail_and_grid"
    )
    assert refined_metadata["truncation_radius_source"] == "target_error"


@pytest.mark.unit
@pytest.mark.parametrize("alpha", [0.25, 0.5, 0.75])
def test_mode_refinement_converges_for_a_linear_caputo_derivative(
    alpha: float,
) -> None:
    coarse_error = maximum_linear_caputo_error(
        Diethelm2022(65, target_error=1.0e-8),
        alpha,
    )
    fine_error = maximum_linear_caputo_error(
        Diethelm2022(257, target_error=1.0e-8),
        alpha,
    )
    assert fine_error < 0.01 * coarse_error
    assert fine_error < 1.0e-7


@pytest.mark.unit
@pytest.mark.parametrize("alpha", [0.25, 0.5, 0.75])
def test_simpson_mode_refinement_converges(
    alpha: float,
) -> None:
    coarse_error = maximum_linear_caputo_error(
        Diethelm2022(
            65,
            quadrature="simpson",
            target_error=1.0e-8,
        ),
        alpha,
    )
    fine_error = maximum_linear_caputo_error(
        Diethelm2022(
            257,
            quadrature="simpson",
            target_error=1.0e-8,
        ),
        alpha,
    )
    assert fine_error < 0.05 * coarse_error


@pytest.mark.unit
@pytest.mark.parametrize("alpha", [0.25, 0.5, 0.75])
def test_gauss_legendre_mode_refinement_converges(
    alpha: float,
) -> None:
    coarse_error = maximum_linear_caputo_error(
        Diethelm2022(
            64,
            quadrature="gauss-legendre",
            target_error=1.0e-8,
        ),
        alpha,
    )
    fine_error = maximum_linear_caputo_error(
        Diethelm2022(
            256,
            quadrature="gauss-legendre",
            target_error=1.0e-8,
        ),
        alpha,
    )
    assert fine_error < 0.05 * coarse_error


@pytest.mark.unit
def test_rate_scale_is_a_log_rate_translation() -> None:
    alpha = 0.37
    baseline = Diethelm2022(
        17,
        truncation_radius=4.0,
    ).spectrum(alpha)
    scaled = Diethelm2022(
        17,
        truncation_radius=4.0,
        rate_scale=25.0,
    ).spectrum(alpha)
    np.testing.assert_allclose(scaled.rates, 25.0 * baseline.rates)
    np.testing.assert_allclose(
        scaled.weights,
        25.0**alpha * baseline.weights,
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("arguments", "keywords", "error"),
    [
        ((True,), {}, "integer"),
        ((2,), {}, "at least 3"),
        ((65_537,), {}, "resource ceiling"),
        ((5,), {"target_error": 0.0}, "target_error"),
        ((5,), {"decay_scale": np.inf}, "decay_scale"),
        ((5,), {"truncation_radius": -1.0}, "truncation_radius"),
        ((5,), {"rate_scale": np.nan}, "rate_scale"),
        ((5,), {"quadrature": "midpoint"}, "quadrature"),
        ((6,), {"quadrature": "simpson"}, "odd num_modes"),
        ((5,), {"quadrature": "gauss-laguerre"}, "even num_modes"),
        (
            (8,),
            {
                "quadrature": "gauss-laguerre",
                "truncation_radius": 3.0,
            },
            "truncation_radius",
        ),
        (
            (8,),
            {"quadrature": "gauss-laguerre", "rate_scale": 2.0},
            "rate_scale",
        ),
        (
            (8,),
            {"quadrature": "gauss-laguerre", "target_error": 1.0e-4},
            "target_error",
        ),
        (
            (8,),
            {"quadrature": "gauss-laguerre", "decay_scale": 2.0},
            "decay_scale",
        ),
    ],
)
def test_configuration_validation(
    arguments,
    keywords,
    error: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=error):
        Diethelm2022(*arguments, **keywords)


@pytest.mark.unit
def test_order_and_float64_range_validation() -> None:
    representation = Diethelm2022(9)
    with pytest.raises(TypeError, match="real scalar"):
        representation.spectrum(object())
    for alpha in (0.0, 1.0, np.nan):
        with pytest.raises(ValueError, match="0 < alpha < 1"):
            representation.spectrum(alpha)
    with pytest.raises(ValueError, match="nonfinite float64 rates"):
        Diethelm2022(
            9,
            truncation_radius=800.0,
        ).spectrum(0.01)


@pytest.mark.unit
def test_high_mode_count_warns_and_is_recorded() -> None:
    assert Diethelm2022(16_385).describe()["mode_count_recommended"] is True
    with pytest.warns(ModeCountAdvisoryWarning, match="num_modes=16386"):
        representation = Diethelm2022(16_386)
    assert representation.describe()["mode_count_recommended"] is False


@pytest.mark.unit
def test_gauss_laguerre_reports_infeasible_float64_spectrum() -> None:
    with pytest.raises(
        ValueError,
        match=r"alpha=0\.9 and num_modes=64",
    ):
        Diethelm2022(
            64,
            quadrature="gauss-laguerre",
        ).spectrum(0.9)


@pytest.mark.unit
def test_gauss_laguerre_checkpoint_metadata_round_trip() -> None:
    representation = Diethelm2022(
        32,
        quadrature="gauss-laguerre",
    )
    metadata = representation.describe(0.5)
    validate_checkpoint_representation(metadata, representation, 0.5)

    changed = dict(metadata)
    changed_nodes = list(changed["quadrature_nodes"])
    changed_nodes[0] += 1.0
    changed["quadrature_nodes"] = tuple(changed_nodes)
    with pytest.raises(ValueError, match="does not match"):
        validate_checkpoint_representation(changed, representation, 0.5)


@pytest.mark.unit
def test_representation_is_described_and_publicly_routed() -> None:
    representation = Diethelm2022(33)
    description = representation.describe()
    assert description["status"] == "supported-not-recommended"
    assert description["configurable_parameters"] == (
        "quadrature",
        "target_error",
        "decay_scale",
        "truncation_radius",
        "rate_scale",
    )
    assert yonderdrake.Diethelm2022 is Diethelm2022
    json.dumps(representation.describe(0.4), allow_nan=False)
