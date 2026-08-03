"""High-precision tests for Alikhanov's L2-1-sigma formula."""

from __future__ import annotations

import mpmath as mp
import numpy as np
import pytest

from yonderdrake import AlikhanovL21Sigma
from yonderdrake.time.full_history import _alikhanov_increment_weights
from yonderdrake.time.representations import (
    validate_checkpoint_representation,
)


def reference_increment_weights(alpha: float, step: int) -> np.ndarray:
    with mp.workdps(100):
        order = mp.mpf(str(alpha))
        sigma = 1 - order / 2
        level = step - 1
        a = [sigma ** (1 - order)]
        a.extend(
            (lag + sigma) ** (1 - order)
            - (lag - 1 + sigma) ** (1 - order)
            for lag in range(1, level + 1)
        )
        b = [mp.mpf("0")]
        b.extend(
            (
                (lag + sigma) ** (2 - order)
                - (lag - 1 + sigma) ** (2 - order)
            )
            / (2 - order)
            - (
                (lag + sigma) ** (1 - order)
                + (lag - 1 + sigma) ** (1 - order)
            )
            / 2
            for lag in range(1, level + 1)
        )
        if level == 0:
            by_lag = [a[0]]
        else:
            by_lag = [a[0] + b[1]]
            by_lag.extend(
                a[lag] + b[lag + 1] - b[lag]
                for lag in range(1, level)
            )
            by_lag.append(a[level] - b[level])
        return np.asarray([float(value) for value in reversed(by_lag)])


@pytest.mark.unit
@pytest.mark.parametrize("alpha", [0.1, 0.5, 0.9])
@pytest.mark.parametrize("step", [1, 2, 7, 100, 2000])
def test_alikhanov_coefficients_match_high_precision(
    alpha: float,
    step: int,
) -> None:
    observed = _alikhanov_increment_weights(alpha, step)
    expected = reference_increment_weights(alpha, step)
    np.testing.assert_allclose(observed, expected, rtol=3.0e-11, atol=2.0e-15)
    assert np.all(observed > 0.0)


@pytest.mark.unit
def test_alikhanov_metadata_round_trip() -> None:
    representation = AlikhanovL21Sigma()
    metadata = representation.describe(0.4)
    assert metadata["sigma"] == pytest.approx(0.8)
    assert metadata["evaluation"] == "t_n_plus_sigma"
    validate_checkpoint_representation(metadata, representation, 0.4)

    changed = dict(metadata)
    changed["sigma"] = 0.7
    with pytest.raises(ValueError, match="does not match"):
        validate_checkpoint_representation(changed, representation, 0.4)


@pytest.mark.unit
def test_alikhanov_step_must_be_positive() -> None:
    with pytest.raises(ValueError, match="positive"):
        _alikhanov_increment_weights(0.5, 0)
