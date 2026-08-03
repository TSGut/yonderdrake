"""Coefficient and storage tests for fast-oblivious convolution quadrature."""

from __future__ import annotations

from math import exp, lgamma

import numpy as np
import pytest

from yonderdrake import FastObliviousCQ
from yonderdrake.time.fast_cq import _FastCQHistory
from yonderdrake.time.representations import (
    ComplexContourSpectrum,
    validate_checkpoint_representation,
)


@pytest.mark.unit
@pytest.mark.parametrize("alpha", [0.1, 0.5, 0.9])
@pytest.mark.parametrize("level", [0, 5, 9])
def test_talbot_contour_matches_high_precision_bdf1_weight(
    alpha: float,
    level: int,
) -> None:
    step_size = 0.01
    representation = FastObliviousCQ(
        num_levels=10,
        nodes_per_level=15,
        direct_steps=20,
    )
    spectrum = representation.contour_spectrum(alpha, step_size, level)
    scale = max((representation.direct_steps + 1) // 2, 2**level)
    for age in (2 * scale, 4 * scale - 1):
        resolvent = 1.0 / (1.0 - step_size * spectrum.rates)
        observed = float(np.sum(spectrum.weights * np.power(resolvent, age + 1)).real)
        expected = step_size ** (-alpha) * exp(
            lgamma(age + 1.0 - alpha) - lgamma(1.0 - alpha) - lgamma(age + 1.0)
        )
        assert observed == pytest.approx(expected, rel=4.0e-8)


@pytest.mark.unit
def test_complex_contour_contract_preserves_conjugate_pairs() -> None:
    representation = FastObliviousCQ(num_levels=2)
    spectrum = representation.contour_spectrum(0.4, 0.1, 1)
    assert isinstance(spectrum, ComplexContourSpectrum)
    np.testing.assert_allclose(spectrum.rates, np.conjugate(spectrum.rates[::-1]))
    np.testing.assert_allclose(
        spectrum.weights,
        np.conjugate(spectrum.weights[::-1]),
    )
    with pytest.raises(ValueError, match="conjugate"):
        ComplexContourSpectrum([1j, 1.0, 2j], [1.0, 1.0, 1.0], {})


@pytest.mark.unit
@pytest.mark.parametrize(
    ("rates", "weights", "message"),
    [
        ([[1.0]], [1.0], "one-dimensional"),
        ([1.0], [1.0, 2.0], "equal-sized"),
        ([], [], "odd number"),
        ([1.0, 2.0], [1.0, 2.0], "odd number"),
        ([complex(np.inf), 1.0, complex(np.inf)], [1.0, 1.0, 1.0], "finite"),
        ([1.0, 1.0, 1.0], [1j, 1.0, 2j], "weights"),
    ],
)
def test_complex_contour_contract_rejects_invalid_spectra(
    rates: object,
    weights: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        ComplexContourSpectrum(rates, weights, {})


@pytest.mark.unit
def test_dyadic_history_matches_direct_bdf1_convolution() -> None:
    alpha = 0.5
    step_size = 0.01
    representation = FastObliviousCQ(
        num_levels=10,
        nodes_per_level=15,
        direct_steps=20,
    )
    history = _FastCQHistory(representation, alpha, step_size, 3)
    random = np.random.default_rng(1847)
    increments: list[np.ndarray] = []
    maximum_error = 0.0
    for step in range(1, 301):
        observed = np.empty(3)
        history.past_action(step, observed)
        expected = np.zeros(3)
        coefficient = 1.0
        for age, values in enumerate(reversed(increments), start=1):
            coefficient *= (age - alpha) / age
            expected += step_size ** (-alpha) * coefficient * values
        maximum_error = max(
            maximum_error,
            np.linalg.norm(observed - expected) / max(np.linalg.norm(expected), 1.0),
        )
        increment = random.normal(size=3)
        increments.append(increment)
        history.append(increment)
    assert maximum_error < 8.0e-7


@pytest.mark.unit
def test_fast_history_storage_grows_logarithmically_and_restores() -> None:
    representation = FastObliviousCQ(
        num_levels=12,
        nodes_per_level=10,
        direct_steps=10,
    )
    history = _FastCQHistory(representation, 0.4, 0.01, 2)
    storage = {}
    for step in range(1, 513):
        history.append(np.asarray([step, -step], dtype=np.float64))
        if step in {128, 512}:
            storage[step] = history.field_storage()
    assert storage[512] < 1.35 * storage[128]

    restored = _FastCQHistory(representation, 0.4, 0.01, 2)
    restored.restore(history.state())
    observed = np.empty(2)
    expected = np.empty(2)
    history.past_action(513, expected)
    restored.past_action(513, observed)
    np.testing.assert_allclose(observed, expected)


@pytest.mark.unit
def test_fast_history_rejects_invalid_actions_and_capacity() -> None:
    representation = FastObliviousCQ(
        num_levels=2,
        nodes_per_level=4,
        direct_steps=6,
    )
    history = _FastCQHistory(representation, 0.4, 0.1, 2)
    with pytest.raises(ValueError, match="increment"):
        history.append(np.ones(3))
    with pytest.raises(RuntimeError, match="inconsistent"):
        history.past_action(2, np.empty(2))
    with pytest.raises(ValueError, match="output"):
        history.past_action(1, np.empty(3))

    for _ in range(representation.max_steps):
        history.append(np.ones(2))
    with pytest.raises(RuntimeError, match="max_steps"):
        history.append(np.ones(2))

    history.levels[0].completed.clear()
    with pytest.raises(RuntimeError, match="unavailable"):
        history._block(history.levels[0], 1)


@pytest.mark.unit
def test_fast_history_rejects_corrupt_checkpoint_state() -> None:
    representation = FastObliviousCQ(
        num_levels=3,
        nodes_per_level=4,
        direct_steps=6,
    )
    source = _FastCQHistory(representation, 0.4, 0.1, 2)
    for step in range(7):
        source.append(np.asarray([step, -step], dtype=np.float64))
    state = source.state()

    invalid_top_levels = dict(state)
    invalid_top_levels["levels"] = []
    target = _FastCQHistory(representation, 0.4, 0.1, 2)
    with pytest.raises(ValueError, match="history is invalid"):
        target.restore(invalid_top_levels)

    invalid_recent = dict(state)
    invalid_recent["recent"] = [{"index": 1, "values": [np.nan, 0.0]}]
    with pytest.raises(ValueError, match="field is invalid"):
        target.restore(invalid_recent)

    invalid_contour = source.state()
    invalid_contour["levels"][0]["real"] = [[0.0]]
    with pytest.raises(ValueError, match="contour field"):
        target.restore(invalid_contour)

    nonfinite_contour = source.state()
    nonfinite_contour["levels"][0]["imaginary"][0][0] = np.nan
    with pytest.raises(ValueError, match="contour field"):
        target.restore(nonfinite_contour)

    invalid_block = source.state()
    invalid_block["levels"][0]["completed"][0]["real"] = [[0.0]]
    with pytest.raises(ValueError, match="contour field"):
        target.restore(invalid_block)


@pytest.mark.unit
def test_fast_cq_configuration_and_checkpoint_metadata() -> None:
    representation = FastObliviousCQ(
        target_error=1.0e-6,
        num_levels=8,
    )
    assert representation.nodes_per_level == 15
    assert representation.max_steps == 255
    metadata = representation.describe(0.6)
    validate_checkpoint_representation(metadata, representation, 0.6)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("parameters", "error", "message"),
    [
        ({"target_error": 1.0}, ValueError, "target_error"),
        ({"target_error": np.nan}, ValueError, "target_error"),
        ({"num_levels": 0}, ValueError, "num_levels"),
        ({"num_levels": True}, TypeError, "num_levels"),
        ({"num_levels": 61}, ValueError, "num_levels"),
        ({"nodes_per_level": 2}, ValueError, "nodes_per_level"),
        ({"nodes_per_level": True}, TypeError, "nodes_per_level"),
        ({"nodes_per_level": 65}, ValueError, "nodes_per_level"),
        ({"direct_steps": 1}, ValueError, "direct_steps"),
        ({"direct_steps": True}, TypeError, "direct_steps"),
        ({"direct_steps": 4097}, ValueError, "direct_steps"),
        ({"contour": "circle"}, ValueError, "contour"),
    ],
)
def test_fast_cq_configuration_rejects_invalid_values(
    parameters: dict[str, object],
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        FastObliviousCQ(**parameters)  # type: ignore[arg-type]


@pytest.mark.unit
@pytest.mark.parametrize(
    ("step_size", "level", "error", "message"),
    [
        (0.0, 0, ValueError, "step_size"),
        (0.1, True, TypeError, "level"),
        (0.1, -1, ValueError, "dyadic range"),
        (0.1, 2, ValueError, "dyadic range"),
    ],
)
def test_fast_cq_contour_rejects_invalid_location(
    step_size: float,
    level: object,
    error: type[Exception],
    message: str,
) -> None:
    representation = FastObliviousCQ(num_levels=2)
    with pytest.raises(error, match=message):
        representation.contour_spectrum(0.5, step_size, level)  # type: ignore[arg-type]
