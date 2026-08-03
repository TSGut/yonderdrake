"""Representation-level convergence checks, separate from timestep error."""

from __future__ import annotations

from math import gamma

import numpy as np
import pytest

from yonderdrake import BirkSong, Diethelm2008


def maximum_kernel_error(representation, alpha: float) -> float:
    spectrum = representation.spectrum(alpha)
    times = np.logspace(-3.0, 3.0, 121)
    observed = np.array(
        [
            np.dot(spectrum.weights, np.exp(-spectrum.rates * time))
            for time in times
        ]
    )
    expected = times ** (-alpha) / gamma(1.0 - alpha)
    return float(np.max(np.abs(observed / expected - 1.0)))


@pytest.mark.verification
@pytest.mark.parametrize(
    ("representation_type", "alpha", "limit"),
    [
        (BirkSong, 0.1, 1.0e-7),
        (BirkSong, 0.5, 2.0e-6),
        (BirkSong, 0.9, 5.0e-6),
        (Diethelm2008, 0.1, 5.0e-6),
        (Diethelm2008, 0.5, 8.0e-5),
        (Diethelm2008, 0.9, 3.0e-4),
    ],
)
def test_diffusive_kernel_converges(
    representation_type,
    alpha: float,
    limit: float,
) -> None:
    coarse_error = maximum_kernel_error(representation_type(32), alpha)
    fine_error = maximum_kernel_error(representation_type(64), alpha)
    assert fine_error < coarse_error
    assert fine_error < limit


@pytest.mark.verification
@pytest.mark.parametrize("representation_type", [BirkSong, Diethelm2008])
def test_documented_extreme_construction_is_finite(representation_type) -> None:
    for alpha in [1.0e-6, 1.0 - 1.0e-6]:
        spectrum = representation_type(256).spectrum(alpha)
        assert np.all(np.isfinite(spectrum.rates))
        assert np.all(np.isfinite(spectrum.weights))
