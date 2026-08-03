"""Pure numerical tests for the positive-power sinc rule."""

from __future__ import annotations

import numpy as np
import pytest

from yonderdrake.spectral.sinc import positive_power_sinc


def scalar_action(quadrature, eigenvalue: float) -> float:
    negative = quadrature.log_shifts <= 0.0
    negative_shifts = np.exp(quadrature.log_shifts[negative])
    positive_inverse_shifts = np.exp(-quadrature.log_shifts[~negative])
    return float(
        np.sum(
            quadrature.weights[negative]
            * eigenvalue
            / (negative_shifts + eigenvalue)
        )
        + np.sum(
            quadrature.weights[~negative]
            * eigenvalue
            / (1.0 + positive_inverse_shifts * eigenvalue)
        )
    )


@pytest.mark.unit
@pytest.mark.parametrize("order", [0.1, 0.5, 0.9])
def test_sinc_scalar_positive_power(order: float) -> None:
    eigenvalue = 3.7
    quadrature = positive_power_sinc(order, 1.0e-8)
    approximate = scalar_action(quadrature, eigenvalue)
    assert approximate == pytest.approx(eigenvalue**order, rel=1.0e-8)
    assert (
        quadrature.estimated_model_error
        <= quadrature.truncation_target
    )


@pytest.mark.unit
def test_sinc_target_refines_and_arrays_are_immutable() -> None:
    coarse = positive_power_sinc(0.4, 1.0e-3)
    fine = positive_power_sinc(0.4, 1.0e-8)
    assert fine.num_nodes > coarse.num_nodes
    assert fine.estimated_model_error < coarse.estimated_model_error
    with pytest.raises(ValueError):
        fine.weights[0] = 0.0


@pytest.mark.unit
def test_sinc_target_reduces_fixed_eigenvalue_error() -> None:
    order = 0.63
    eigenvalue = 11.0

    def error(target: float) -> float:
        rule = positive_power_sinc(order, target)
        value = scalar_action(rule, eigenvalue)
        return abs(value - eigenvalue**order)

    assert error(1.0e-7) < error(1.0e-3)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("order", "target"),
    [(0.0, 1.0e-3), (1.0, 1.0e-3), (0.5, 0.0), (0.5, 1.0)],
)
def test_sinc_rejects_invalid_parameters(order: float, target: float) -> None:
    with pytest.raises(ValueError):
        positive_power_sinc(order, target)


@pytest.mark.unit
def test_sinc_warns_and_remains_finite_below_float64_precision() -> None:
    with pytest.warns(RuntimeWarning, match="float64 precision"):
        quadrature = positive_power_sinc(0.9, 1.0e-300)
    assert quadrature.truncation_target == 1.0e-300
    assert (
        quadrature.effective_truncation_target
        == np.finfo(np.float64).eps
    )
    assert np.all(np.isfinite(quadrature.weights))
    assert (
        quadrature.estimated_model_error
        <= quadrature.effective_truncation_target
    )


@pytest.mark.unit
def test_sinc_rejects_impractical_work() -> None:
    with pytest.raises(ValueError, match="more than 100000"):
        positive_power_sinc(1.0e-12, 1.0e-10)
