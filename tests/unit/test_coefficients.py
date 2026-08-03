"""Tests for stable scalar functions and the Gauss-Jacobi helper."""

from __future__ import annotations

import gc
import math
import tracemalloc

import mpmath as mp
import numpy as np
import pytest
from scipy.special import eval_jacobi
from tests.reference.diffusive import gauss_jacobi as reference_gauss_jacobi

from yonderdrake.time.coefficients import (
    exp_neg,
    gauss_jacobi,
    phi1,
    psi,
    quadratic_recurrence_coefficients,
    recurrence_coefficients,
)
from yonderdrake.time.representations import BirkSong


@pytest.mark.unit
@pytest.mark.parametrize(
    "z",
    [0.0, 1.0e-16, 1.0e-12, 1.0e-8, 1.0e-4, 1.0, 50.0, 1.0e3],
)
def test_stable_recurrence_scalars(z: float) -> None:
    with mp.workdps(80):
        high_z = mp.mpf(str(z))
        expected_exp = mp.exp(-high_z)
        expected_phi1 = (
            mp.mpf(1) if z == 0.0 else -mp.expm1(-high_z) / high_z
        )
    assert math.isclose(float(exp_neg(z)), float(expected_exp), rel_tol=2.0e-15)
    assert math.isclose(float(phi1(z)), float(expected_phi1), rel_tol=2.0e-15)


@pytest.mark.unit
def test_phi1_stability_branch_avoids_subtractive_cancellation() -> None:
    z = 1.0e-8
    with mp.workdps(80):
        expected = float(-mp.expm1(-mp.mpf("1e-8")) / mp.mpf("1e-8"))
    stable_error = abs(float(phi1(z)) - expected)
    naive_error = abs((1.0 - math.exp(-z)) / z - expected)
    assert stable_error < 2.0e-15
    assert naive_error > 1.0e-10


@pytest.mark.unit
def test_psi_matches_high_precision_across_its_crossover() -> None:
    arguments = np.concatenate(
        ([0.0], np.geomspace(1.0e-16, 1.0e3, 160))
    )
    with mp.workdps(80):
        expected = [
            0.5
            if value == 0.0
            else float(
                (1.0 + mp.expm1(-mp.mpf(str(value))) / mp.mpf(str(value)))
                / mp.mpf(str(value))
            )
            for value in arguments
        ]
    np.testing.assert_allclose(psi(arguments), expected, rtol=2.0e-13, atol=1.0e-15)

    z = 1.0e-14
    with mp.workdps(80):
        high_z = mp.mpf("1e-14")
        reference = float((1.0 + mp.expm1(-high_z) / high_z) / high_z)
    stable_error = abs(float(psi(z)) - reference)
    naive_error = abs((1.0 - float(phi1(z))) / z - 0.5)
    assert stable_error < 2.0e-15
    assert naive_error > 1.0e-4


@pytest.mark.unit
def test_vectorized_recurrence_scalars_and_validation() -> None:
    arguments = np.array([0.0, 1.0e-12, 1.0, 1.0e3])
    assert np.all(np.isfinite(exp_neg(arguments)))
    assert np.all(np.isfinite(phi1(arguments)))
    assert np.all(np.isfinite(psi(arguments)))
    assert phi1(arguments)[0] == 1.0
    assert psi(arguments)[0] == 0.5
    with pytest.raises(ValueError, match="nonnegative"):
        phi1(-1.0)
    with pytest.raises(ValueError, match="finite"):
        exp_neg(np.inf)


@pytest.mark.unit
def test_recurrence_coefficients_match_the_scalar_primitives() -> None:
    spectrum = BirkSong(4).spectrum(0.6)
    decay, interpolation, implicit_weight = recurrence_coefficients(
        spectrum,
        0.125,
    )
    arguments = spectrum.rates * 0.125
    np.testing.assert_array_equal(decay, exp_neg(arguments))
    np.testing.assert_array_equal(interpolation, phi1(arguments))
    assert implicit_weight == float(np.dot(spectrum.weights, interpolation))


@pytest.mark.unit
def test_quadratic_recurrence_uses_a_linear_starting_step() -> None:
    spectrum = BirkSong(8).spectrum(0.6)
    linear_decay, linear_interpolation, linear_implicit = (
        recurrence_coefficients(spectrum, 0.125)
    )
    decay, current, previous, implicit, previous_weight = (
        quadratic_recurrence_coefficients(
            spectrum,
            0.125,
            previous_step_size=None,
        )
    )
    np.testing.assert_array_equal(decay, linear_decay)
    np.testing.assert_array_equal(current, linear_interpolation)
    np.testing.assert_array_equal(previous, np.zeros_like(previous))
    assert implicit == linear_implicit
    assert previous_weight == 0.0

    _, current, old, implicit, old_implicit = (
        quadratic_recurrence_coefficients(
            spectrum,
            0.125,
            previous_step_size=0.125,
        )
    )
    expected_current = 0.5 * linear_interpolation + psi(
        spectrum.rates * 0.125
    )
    expected_old = 0.5 * linear_interpolation - psi(
        spectrum.rates * 0.125
    )
    np.testing.assert_allclose(current, expected_current, rtol=0.0, atol=1.0e-16)
    np.testing.assert_allclose(old, expected_old, rtol=0.0, atol=1.0e-16)
    assert implicit == pytest.approx(np.dot(spectrum.weights, current))
    assert old_implicit == pytest.approx(np.dot(spectrum.weights, old))
    with pytest.raises(ValueError, match="previous_step_size"):
        quadratic_recurrence_coefficients(
            spectrum,
            0.125,
            previous_step_size=0.0,
        )


def _power_recurrence_error(alpha: float, steps: int, interpolant: str) -> float:
    spectrum = BirkSong(256).spectrum(alpha)
    step_size = 1.0 / steps
    modes = np.zeros(spectrum.rates.size)
    penultimate = 0.0
    previous = 0.0
    previous_step_size = None
    for step in range(1, steps + 1):
        current = (step * step_size) ** 2
        if interpolant == "quadratic":
            decay, current_weight, old_weight, _, _ = (
                quadratic_recurrence_coefficients(
                    spectrum,
                    step_size,
                    previous_step_size=previous_step_size,
                )
            )
        else:
            decay, current_weight, _ = recurrence_coefficients(
                spectrum, step_size
            )
            old_weight = np.zeros_like(current_weight)
        modes = (
            decay * modes
            + current_weight * (current - previous)
            + old_weight * (previous - penultimate)
        )
        penultimate, previous = previous, current
        previous_step_size = step_size
    exact = 2.0 / math.gamma(3.0 - alpha)
    return abs(float(np.dot(spectrum.weights, modes)) - exact)


@pytest.mark.unit
@pytest.mark.parametrize("alpha", [0.3, 0.6, 0.8])
def test_recurrence_interpolants_have_their_expected_orders(alpha: float) -> None:
    counts = (50, 100, 200, 400)
    for interpolant, expected_order in (
        ("linear", 2.0 - alpha),
        ("quadratic", 3.0),
    ):
        errors = [
            _power_recurrence_error(alpha, count, interpolant)
            for count in counts
        ]
        orders = np.log2(np.asarray(errors[:-1]) / errors[1:])
        assert orders[-1] == pytest.approx(expected_order, abs=0.035)
        if interpolant == "quadratic":
            assert np.all(orders > 2.95)


@pytest.mark.unit
@pytest.mark.parametrize(("alpha", "beta"), [(0.0, 0.0), (-0.8, 0.4), (2.2, -0.7)])
def test_gauss_jacobi_integrates_expected_moments(
    alpha: float,
    beta: float,
) -> None:
    num_nodes = 5
    nodes, weights = gauss_jacobi(num_nodes, alpha, beta)
    with mp.workdps(80):
        for degree in range(2 * num_nodes):
            expected = mp.quad(
                lambda x, degree=degree: x**degree
                * (1 - x) ** mp.mpf(str(alpha))
                * (1 + x) ** mp.mpf(str(beta)),
                [-1, 0, 1],
            )
            observed = np.dot(weights, nodes**degree)
            assert math.isclose(
                observed,
                float(expected),
                rel_tol=2.0e-12,
                abs_tol=2.0e-13,
            )


@pytest.mark.unit
def test_gauss_jacobi_rejects_invalid_rules() -> None:
    with pytest.raises(TypeError, match="integer"):
        gauss_jacobi(True, 0.0, 0.0)
    with pytest.raises(ValueError, match="positive"):
        gauss_jacobi(0, 0.0, 0.0)
    with pytest.raises(ValueError, match="greater than -1"):
        gauss_jacobi(2, -1.0, 0.0)
    with pytest.raises(ValueError, match="finite"):
        gauss_jacobi(2, np.inf, 0.0)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("num_nodes", "alpha", "beta"),
    [(64, 0.6, -0.6), (256, 0.0, -0.4), (1024, 0.2, -0.2)],
)
def test_gauss_jacobi_is_exact_on_its_orthogonal_polynomials(
    num_nodes: int,
    alpha: float,
    beta: float,
) -> None:
    nodes, weights = gauss_jacobi(num_nodes, alpha, beta)
    moment = float(np.sum(weights))
    maximum_residual = max(
        abs(np.dot(weights, eval_jacobi(degree, alpha, beta, nodes))) / moment
        for degree in range(1, 2 * num_nodes)
    )
    # The package rule is around 1e-15 here, while scipy roots_jacobi reaches
    # 1e-11 for the singular n=1024 case. This bound distinguishes them.
    assert maximum_residual < 1.0e-13


@pytest.mark.unit
def test_gauss_jacobi_uses_subquadratic_temporary_memory() -> None:
    gc.collect()
    tracemalloc.start()
    try:
        gauss_jacobi(1024, 0.2, -0.2)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    # A returned 1024-square eigenvector matrix alone needs 8 MiB. The selected
    # 128-column chunks measure about 3.2 MiB, with margin for allocator drift.
    assert peak < 6 * 1024**2


@pytest.mark.unit
def test_gauss_jacobi_newton_polish_resolves_a_reference_node() -> None:
    nodes, _ = gauss_jacobi(256, 0.0, -0.4)
    observed = nodes[206]
    with mp.workdps(80):
        root = mp.findroot(
            lambda value: mp.jacobi(256, 0, mp.mpf("-0.4"), value),
            (mp.mpf(str(observed - 1.0e-10)), mp.mpf(str(observed + 1.0e-10))),
        )
        error = abs(mp.mpf(str(observed)) - root)
    # The polished node is within 5e-17 here; the selected eigensolver value
    # before the Newton step is about 2.4e-16 away.
    assert error < mp.mpf("1.2e-16")


@pytest.mark.unit
@pytest.mark.parametrize("order", [0.3, 0.6, 0.9])
def test_high_order_singular_jacobi_rule_matches_high_precision(
    order: float,
) -> None:
    nodes, weights = gauss_jacobi(64, 0.0, order - 1.0)
    with mp.workdps(70):
        reference_nodes, reference_weights = reference_gauss_jacobi(
            64,
            mp.mpf("0"),
            mp.mpf(str(order)) - 1,
        )
    np.testing.assert_allclose(
        nodes,
        [float(value) for value in reference_nodes],
        rtol=0.0,
        atol=3.0e-16,
    )
    np.testing.assert_allclose(
        weights,
        [float(value) for value in reference_weights],
        rtol=2.0e-13,
        atol=0.0,
    )
