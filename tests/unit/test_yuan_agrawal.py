"""Tests for the original Yuan-Agrawal representation."""

from __future__ import annotations

from math import gamma, pi, sin

import numpy as np
import pytest
from scipy.special import roots_laguerre

from yonderdrake import ModeCountAdvisoryWarning, YuanAgrawal


def maximum_linear_error(representation: YuanAgrawal, alpha: float) -> float:
    spectrum = representation.spectrum(alpha)
    times = np.logspace(-2.0, 2.0, 81)
    modes = -np.expm1(
        -np.outer(times, spectrum.rates)
    ) / spectrum.rates
    observed = modes @ spectrum.weights
    exact = times ** (1.0 - alpha) / gamma(2.0 - alpha)
    return float(np.max(np.abs(observed / exact - 1.0)))


@pytest.mark.unit
def test_coefficients_match_standard_gauss_laguerre() -> None:
    num_modes = 32
    alpha = 0.37
    nodes, laguerre_weights = roots_laguerre(num_modes)
    spectrum = YuanAgrawal(num_modes).spectrum(alpha)

    np.testing.assert_allclose(spectrum.rates, np.square(nodes))
    np.testing.assert_allclose(
        spectrum.weights,
        (2.0 * sin(pi * alpha) / pi)
        * laguerre_weights
        * np.exp(nodes)
        * np.power(nodes, 2.0 * alpha - 1.0),
        rtol=2.0e-11,
    )
    assert spectrum.metadata["quadrature"] == "Gauss-Laguerre"
    assert spectrum.metadata["rate_map"] == "squared"


@pytest.mark.unit
def test_high_order_coefficients_remain_finite() -> None:
    spectrum = YuanAgrawal(512).spectrum(0.1)
    assert np.all(np.isfinite(spectrum.rates))
    assert np.all(np.isfinite(spectrum.weights))
    assert np.all(spectrum.weights > 0.0)


@pytest.mark.unit
@pytest.mark.parametrize("alpha", [0.1, 0.5, 0.9])
def test_refinement_converges(alpha: float) -> None:
    coarse = maximum_linear_error(YuanAgrawal(64), alpha)
    fine = maximum_linear_error(YuanAgrawal(256), alpha)
    assert fine < coarse


@pytest.mark.unit
def test_rate_scale_translates_the_spectrum() -> None:
    alpha = 0.4
    baseline = YuanAgrawal(16).spectrum(alpha)
    scaled = YuanAgrawal(16, rate_scale=9.0).spectrum(alpha)
    np.testing.assert_allclose(scaled.rates, 9.0 * baseline.rates)
    np.testing.assert_allclose(
        scaled.weights,
        9.0**alpha * baseline.weights,
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("arguments", "keywords", "error"),
    [
        ((True,), {}, "integer"),
        ((0,), {}, "positive"),
        ((16_385,), {}, "resource ceiling"),
        ((8,), {"rate_scale": 0.0}, "rate_scale"),
    ],
)
def test_configuration_validation(arguments, keywords, error: str) -> None:
    with pytest.raises((TypeError, ValueError), match=error):
        YuanAgrawal(*arguments, **keywords)


@pytest.mark.unit
def test_description() -> None:
    representation = YuanAgrawal(8)
    assert representation.describe()["status"] == "supported-not-recommended"


@pytest.mark.unit
def test_high_mode_count_warns_and_is_recorded() -> None:
    assert YuanAgrawal(2_048).describe()["mode_count_recommended"] is True
    with pytest.warns(ModeCountAdvisoryWarning, match="num_modes=2049"):
        representation = YuanAgrawal(2_049)
    assert representation.describe()["mode_count_recommended"] is False
