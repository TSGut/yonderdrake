"""Common high-precision checks for time-memory spectrum construction."""

from __future__ import annotations

import mpmath as mp
import numpy as np
import pytest
from tests.reference.diffusive import gauss_jacobi as reference_gauss_jacobi
from tests.reference.diffusive import spectrum as reference_spectrum

from yonderdrake import (
    BirkSong,
    Diethelm2008,
    SineDiffusive,
    SumOfExponentials,
    YuanAgrawal,
)


def _positive_rate_reference(name: str) -> tuple[np.ndarray, np.ndarray]:
    alpha = 0.6
    num_modes = 16
    if name in {"BirkSong", "Diethelm2008"}:
        rates, weights = reference_spectrum(name, alpha, num_modes, dps=70)
        return (
            np.asarray([float(value) for value in rates]),
            np.asarray([float(value) for value in weights]),
        )
    with mp.workdps(70):
        nodes, weighted = mp.gauss_quadrature(
            num_modes,
            "glaguerre",
            0,
        )
        rates = [node**2 for node in nodes]
        weights = [
            2
            * mp.sin(mp.pi * alpha)
            / mp.pi
            * weighted[index]
            * mp.exp(nodes[index])
            * nodes[index] ** (2 * alpha - 1)
            for index in range(num_modes)
        ]
    return (
        np.asarray([float(value) for value in rates]),
        np.asarray([float(value) for value in weights]),
    )


def _sum_of_exponentials_reference(
    spectrum,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    quadrature_order = int(spectrum.metadata["quadrature_order"])
    intervals = int(spectrum.metadata["dyadic_intervals"])
    with mp.workdps(70):
        order = mp.mpf(str(alpha))
        jacobi_nodes, jacobi_weights = reference_gauss_jacobi(
            quadrature_order,
            mp.mpf(0),
            order - 1,
        )
        rates = [(node + 1) / 2 for node in jacobi_nodes]
        power_weights = [
            weight * rate / (mp.power(2, order) * mp.gamma(1 + order))
            for rate, weight in zip(rates, jacobi_weights, strict=True)
        ]
        legendre_nodes, legendre_weights = mp.gauss_quadrature(
            quadrature_order,
            "legendre",
        )
        for interval in range(intervals):
            left = mp.power(2, interval)
            half_width = left / 2
            midpoint = 3 * left / 2
            for node, weight in zip(
                legendre_nodes,
                legendre_weights,
                strict=True,
            ):
                rate = midpoint + half_width * node
                rates.append(rate)
                power_weights.append(
                    half_width * weight * rate**order / mp.gamma(1 + order)
                )
        weights = [
            order * power_weight / (rate * mp.gamma(1 - order))
            for rate, power_weight in zip(rates, power_weights, strict=True)
        ]
    return (
        np.asarray([float(value) for value in rates]),
        np.asarray([float(value) for value in weights]),
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "name",
    ["BirkSong", "Diethelm2008", "SumOfExponentials", "SineDiffusive", "YuanAgrawal"],
)
def test_representation_constructions_have_common_reference_accuracy(
    name: str,
) -> None:
    alpha = 0.6
    if name == "BirkSong":
        observed = BirkSong(16).spectrum(alpha)
        expected_nodes, expected_weights = _positive_rate_reference(name)
        observed_nodes = observed.rates
    elif name == "Diethelm2008":
        observed = Diethelm2008(16).spectrum(alpha)
        expected_nodes, expected_weights = _positive_rate_reference(name)
        observed_nodes = observed.rates
    elif name == "SumOfExponentials":
        observed = SumOfExponentials(
            target_error=1.0e-6,
            t_final=1.0,
            min_step=0.01,
        ).spectrum(alpha)
        expected_nodes, expected_weights = _sum_of_exponentials_reference(
            observed,
            alpha,
        )
        observed_nodes = observed.rates
    elif name == "SineDiffusive":
        observed = SineDiffusive(16).spectrum(alpha)
        with mp.workdps(70):
            nodes, weights = mp.gauss_quadrature(16, "glaguerre", alpha)
            expected_nodes = np.asarray([float(value) for value in nodes])
            expected_weights = np.asarray(
                [
                    float(weights[index] * mp.exp(nodes[index]))
                    for index in range(16)
                ]
            )
        observed_nodes = observed.frequencies
    else:
        observed = YuanAgrawal(16).spectrum(alpha)
        expected_nodes, expected_weights = _positive_rate_reference(name)
        observed_nodes = observed.rates

    node_error = np.max(np.abs(observed_nodes / expected_nodes - 1.0))
    weight_error = np.max(np.abs(observed.weights / expected_weights - 1.0))
    assert max(float(node_error), float(weight_error)) < 3.0e-11
