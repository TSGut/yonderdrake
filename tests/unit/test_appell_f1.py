"""Tests for the restricted real Appell ``F1`` evaluator."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import hyp2f1

from yonderdrake.riesz._appell_f1 import appell_f1_a_plus_one


def reference_f1(a: float, b1: float, b2: float, x: float, y: float) -> float:
    largest_argument = max(-x, -y)
    lower_limit = -np.log1p(largest_argument) - max(40.0, 40.0 / a)

    value, error = quad(
        lambda u: np.exp(
            np.log(a)
            + a * u
            - b1 * np.log1p(-x * np.exp(u))
            - b2 * np.log1p(-y * np.exp(u))
        ),
        lower_limit,
        0.0,
        epsabs=0.0,
        epsrel=2.0e-14,
        limit=500,
    )
    assert error < 2.0e-11 * abs(value)
    return value


@pytest.mark.unit
@pytest.mark.parametrize(
    ("a", "b1", "b2", "x", "y"),
    [
        (0.5, 1.0, 0.7, -0.2, -0.6),
        (1.0, 1.0, -0.2, -4.0, -0.7),
        (1.5, 2.0, 0.3, -30.0, -2.0),
        (0.5, 1.0, 0.7, -1.0e4, -3.0),
        (1.5, 2.0, -0.4, -1.0e8, -2.0e3),
    ],
)
def test_restricted_appell_matches_adaptive_euler_integral(
    a: float,
    b1: float,
    b2: float,
    x: float,
    y: float,
) -> None:
    actual = appell_f1_a_plus_one(a, b1, b2, x, y)
    expected = reference_f1(a, b1, b2, x, y)
    assert actual == pytest.approx(expected, rel=8.0e-12, abs=3.0e-25)


@pytest.mark.unit
def test_restricted_appell_vectorizes_and_uses_gauss_reductions() -> None:
    a = 1.5
    b1 = 2.0
    b2 = -0.3
    x = np.array([0.0, -0.2, -1.0, -4.0])
    y = np.array([-0.7, -0.2, 0.0, -1.5])
    actual = appell_f1_a_plus_one(a, b1, b2, x, y)
    expected = np.array(
        [
            hyp2f1(a, b2, a + 1.0, y[0]),
            hyp2f1(a, b1 + b2, a + 1.0, x[1]),
            hyp2f1(a, b1, a + 1.0, x[2]),
            reference_f1(a, b1, b2, x[3], y[3]),
        ]
    )
    np.testing.assert_allclose(actual, expected, rtol=2.0e-13, atol=2.0e-15)


@pytest.mark.unit
def test_restricted_appell_vectorizes_large_negative_arguments() -> None:
    a = 0.5
    b1 = 1.0
    b2 = -0.4
    x = np.array([-1.0e3, -1.0e8, -1.0e12])
    y = np.array([-3.0, -2.0e3, -7.0e7])
    actual = appell_f1_a_plus_one(a, b1, b2, x, y)
    expected = np.array(
        [
            reference_f1(a, b1, b2, x_value, y_value)
            for x_value, y_value in zip(x, y, strict=True)
        ]
    )
    np.testing.assert_allclose(actual, expected, rtol=2.0e-12, atol=1.0e-25)


@pytest.mark.unit
def test_restricted_appell_is_symmetric_in_parameter_argument_pairs() -> None:
    left = appell_f1_a_plus_one(0.5, 1.0, -0.4, -8.0, -0.3)
    right = appell_f1_a_plus_one(0.5, -0.4, 1.0, -0.3, -8.0)
    assert left == pytest.approx(right, rel=2.0e-13, abs=2.0e-15)


@pytest.mark.unit
@pytest.mark.parametrize(
    "arguments",
    [
        (0.0, 1.0, 0.2, -1.0, -2.0),
        (0.5, 1.0, 0.2, 0.1, -2.0),
        (0.5, 1.0, 0.2, -1.0, np.inf),
    ],
)
def test_restricted_appell_rejects_values_outside_its_contract(
    arguments: tuple[float, float, float, float, float],
) -> None:
    with pytest.raises(ValueError):
        appell_f1_a_plus_one(*arguments)


@pytest.mark.unit
def test_restricted_appell_rejects_unattainable_relative_tolerance() -> None:
    with pytest.raises(ValueError, match="rtol is too small"):
        appell_f1_a_plus_one(0.5, 1.0, 0.2, -1.0, -2.0, rtol=1.0e-16)
