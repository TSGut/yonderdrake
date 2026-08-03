"""High-precision tests for Lubich convolution quadrature."""

from __future__ import annotations

import mpmath as mp
import numpy as np
import pytest

from yonderdrake import LubichCQ
from yonderdrake.time.full_history import (
    _lubich_cq_weights,
    _lubich_starting_factors,
    _lubich_starting_weights,
)
from yonderdrake.time.representations import (
    StartingCorrectionAdvisoryWarning,
    validate_checkpoint_representation,
)


def reference_weights(alpha: float, order: str, count: int) -> np.ndarray:
    with mp.workdps(100):
        fractional_order = mp.mpf(str(alpha))
        if order == "bdf1":
            values = [
                (-1) ** index * mp.binomial(fractional_order, index)
                for index in range(count)
            ]
        else:
            # delta(z) = (3/2)(1-z)(1-z/3), so this coefficient
            # convolution is independent of Miller's recurrence.
            values = [
                mp.power(mp.mpf("1.5"), fractional_order)
                * mp.fsum(
                    (-1) ** index
                    * mp.binomial(fractional_order, index)
                    * (-1) ** (degree - index)
                    * mp.binomial(fractional_order, degree - index)
                    / mp.power(3, degree - index)
                    for index in range(degree + 1)
                )
                for degree in range(count)
            ]
        return np.asarray([float(value) for value in values], dtype=np.float64)


def reference_starting_weights(
    alpha: float,
    cq_weights: np.ndarray,
    step: int,
    num_corrections: int,
) -> np.ndarray:
    active = min(step, num_corrections)
    with mp.workdps(80):
        order = mp.mpf(str(alpha))
        matrix = mp.matrix(active, active)
        right = mp.matrix(active, 1)
        for row in range(active):
            exponent = (row + 1) * order
            for column in range(active):
                matrix[row, column] = mp.power(column + 1, exponent)
            exact = (
                mp.gamma(exponent + 1)
                / mp.gamma(exponent + 1 - order)
                * mp.power(step, exponent - order)
            )
            uncorrected = mp.fsum(
                mp.mpf(str(cq_weights[step - sample]))
                * mp.power(sample, exponent)
                for sample in range(1, step + 1)
            )
            right[row] = exact - uncorrected
        correction = mp.lu_solve(matrix, right)
    return np.asarray([float(value) for value in correction], dtype=np.float64)


@pytest.mark.unit
@pytest.mark.parametrize("order", ["bdf1", "bdf2"])
@pytest.mark.parametrize("alpha", [0.1, 0.5, 0.9])
def test_cq_weights_match_high_precision(
    order: str,
    alpha: float,
) -> None:
    observed = _lubich_cq_weights(alpha, order, 160)
    expected = reference_weights(alpha, order, 160)
    np.testing.assert_allclose(observed, expected, rtol=2.0e-13, atol=2.0e-16)


@pytest.mark.unit
@pytest.mark.parametrize("order", ["bdf1", "bdf2"])
@pytest.mark.parametrize("alpha", [0.1, 0.5, 0.9])
def test_starting_weights_make_fractional_powers_exact(
    order: str,
    alpha: float,
) -> None:
    corrections = 2
    for step in (1, 2, 7, 40, 1000):
        weights = _lubich_cq_weights(alpha, order, step + 1)
        starting = _lubich_starting_weights(
            alpha,
            weights,
            step,
            corrections,
        )
        for power_index in range(1, starting.size + 1):
            exponent = power_index * alpha
            observed = sum(
                weights[step - sample] * sample**exponent
                for sample in range(1, step + 1)
            ) + sum(
                starting[sample - 1] * sample**exponent
                for sample in range(1, starting.size + 1)
            )
            expected = (
                mp.gamma(exponent + 1)
                / mp.gamma(exponent + 1 - alpha)
                * step ** (exponent - alpha)
            )
            assert observed == pytest.approx(float(expected), rel=3.0e-13)


@pytest.mark.unit
@pytest.mark.parametrize("step", [10, 100, 1000, 5000])
def test_float64_starting_weights_match_high_precision(step: int) -> None:
    alpha = 0.6
    corrections = 2
    weights = _lubich_cq_weights(alpha, "bdf2", step + 1)
    expected = reference_starting_weights(alpha, weights, step, corrections)
    observed = _lubich_starting_weights(alpha, weights, step, corrections)
    # Cancellation grows with step; 2e-11 leaves a threefold margin over the
    # measured 6.8e-12 difference at step 5000.
    np.testing.assert_allclose(observed, expected, rtol=0.0, atol=2.0e-11)


@pytest.mark.unit
def test_ill_conditioned_starting_system_warns_and_is_recorded() -> None:
    alpha = 0.5
    corrections = 8
    step = 8
    weights = _lubich_cq_weights(alpha, "bdf2", step + 1)
    _lubich_starting_factors.cache_clear()
    with pytest.warns(StartingCorrectionAdvisoryWarning, match="condition"):
        _lubich_starting_weights(alpha, weights, step, corrections)
    metadata = LubichCQ(num_corrections=corrections).describe(alpha)
    assert metadata["starting_system_recommended"] is False


@pytest.mark.unit
def test_cq_configuration_and_checkpoint_metadata() -> None:
    representation = LubichCQ(
        order="BDF2",
        num_corrections=3,
    )
    assert representation.order == "bdf2"
    metadata = representation.describe(0.4)
    assert metadata["starting_exponents"] == pytest.approx((0.4, 0.8, 1.2))
    assert metadata["starting_system_condition"] > 1.0
    validate_checkpoint_representation(metadata, representation, 0.4)

    changed = dict(metadata)
    changed["num_corrections"] = 2
    with pytest.raises(ValueError, match="does not match"):
        validate_checkpoint_representation(changed, representation, 0.4)

    incomplete = dict(metadata)
    del incomplete["order"]
    with pytest.raises(ValueError, match="incomplete"):
        validate_checkpoint_representation(incomplete, representation, 0.4)


@pytest.mark.unit
def test_cq_coefficient_helpers_validate_edge_cases() -> None:
    assert _lubich_cq_weights(0.5, "bdf1", 0).size == 0
    with pytest.raises(ValueError, match="nonnegative"):
        _lubich_cq_weights(0.5, "bdf1", -1)
    with pytest.raises(ValueError, match="bdf1"):
        _lubich_cq_weights(0.5, "trapezoidal", 2)

    weights = _lubich_cq_weights(0.5, "bdf2", 2)
    assert _lubich_starting_weights(0.5, weights, 1, 0).size == 0
    with pytest.raises(ValueError, match="coefficients"):
        _lubich_starting_weights(0.5, weights, 2, 1)

    uncorrected = LubichCQ(num_corrections=0)
    assert uncorrected.describe(0.5)["starting_system_condition"] == 1.0


@pytest.mark.unit
def test_cq_rejects_singular_starting_system(monkeypatch) -> None:
    monkeypatch.setattr(np.linalg, "cond", lambda *_args, **_kwargs: np.inf)
    with pytest.raises(ValueError, match="singular"):
        LubichCQ(num_corrections=2).describe(0.5)


@pytest.mark.unit
@pytest.mark.parametrize("order", ["bdf0", "trapezoidal"])
def test_cq_configuration_rejects_unknown_orders(order: str) -> None:
    with pytest.raises(ValueError, match="order"):
        LubichCQ(order=order)


@pytest.mark.unit
@pytest.mark.parametrize("corrections", [True, 1.5, "2"])
def test_cq_configuration_requires_integer_corrections(corrections: object) -> None:
    with pytest.raises(TypeError, match="integer"):
        LubichCQ(num_corrections=corrections)  # type: ignore[arg-type]


@pytest.mark.unit
@pytest.mark.parametrize("corrections", [-1, 17])
def test_cq_configuration_bounds_corrections(corrections: int) -> None:
    with pytest.raises(ValueError, match="between 0 and 16"):
        LubichCQ(num_corrections=corrections)
