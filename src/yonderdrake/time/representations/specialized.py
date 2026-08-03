"""Tolerance-driven, log-rate, and Gauss-Laguerre representations."""

from __future__ import annotations

from functools import cache
from math import gamma, isfinite, lgamma, log, pi, sin, sqrt
from types import MappingProxyType
from typing import Any

import numpy as np
from scipy.linalg import eigh_tridiagonal
from scipy.optimize import brentq, minimize_scalar
from scipy.special import (
    eval_genlaguerre,
    gammaincc,
    roots_genlaguerre,
    roots_legendre,
)

from yonderdrake.time.coefficients import gauss_jacobi
from yonderdrake.time.representations.core import (
    DiffusiveSpectrum,
    OscillatorSpectrum,
    _validate_alpha,
    _validate_mode_count,
)

_SINE_RECOMMENDED_MAX_MODES = 128  # Accuracy-saturation advisory.
_SINE_RESOURCE_MAX_MODES = 16_384  # Resource guard for two states per mode.
_YUAN_RECOMMENDED_MAX_MODES = 2_048  # Accuracy-saturation advisory.
_YUAN_RESOURCE_MAX_MODES = 16_384  # Resource guard for O(n²)-work rules.
_DIETHELM_2022_RECOMMENDED_MAX_MODES = 16_385  # Accuracy-saturation advisory.
_DIETHELM_2022_RESOURCE_MAX_MODES = 65_536  # Resource guard for field storage.
_SOE_RESOURCE_MAX_NUM_MODES = 65_536  # Resource guard for derived field storage.
_MIN_LOG_RATE = log(float(np.nextafter(0.0, 1.0)))
_MAX_LOG_RATE = log(float(np.finfo(np.float64).max))


def _positive_finite(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a real scalar") from error
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


class SineDiffusive:
    """Sine diffusive representation of Khosravian-Arab and Dehghan.

    Warning:
        This undamped, slowly convergent representation is provided for
        literature comparison. Use a positive-rate representation for routine
        simulations.
    """

    _name = "SineDiffusive"
    _reference = "doi:10.1016/j.apnum.2024.06.017"

    def __init__(self, num_modes: int) -> None:
        self._mode_count_recommended = _validate_mode_count(
            num_modes,
            minimum=1,
            recommended_maximum=_SINE_RECOMMENDED_MAX_MODES,
            resource_maximum=_SINE_RESOURCE_MAX_MODES,
            representation=self._name,
            reason="the two-state oscillator cost usually outweighs improvement",
        )
        self.num_modes = num_modes

    def spectrum(self, alpha: float) -> OscillatorSpectrum:
        """Construct the generalized Gauss-Laguerre oscillator spectrum."""
        order = _validate_alpha(alpha)
        frequencies, weights = _laguerre_nodes_and_unweighted_weights(
            self.num_modes,
            order,
        )
        metadata = MappingProxyType(
            {
                "representation": self._name,
                "reference": self._reference,
                "alpha": order,
                "num_modes": self.num_modes,
                "mode_count_recommended": self._mode_count_recommended,
                "quadrature": "generalized-Gauss-Laguerre",
                "laguerre_alpha": order,
                "quadrature_nodes": tuple(
                    float(frequency) for frequency in frequencies
                ),
                "ordering": "increasing_frequency",
                "state": "undamped-position-velocity",
                "status": "comparison-only",
            }
        )
        return OscillatorSpectrum(frequencies, weights, metadata)

    def describe(self, alpha: float | None = None) -> dict[str, Any]:
        """Describe the configuration or one generated spectrum."""
        if alpha is not None:
            return dict(self.spectrum(alpha).metadata)
        return {
            "representation": self._name,
            "num_modes": self.num_modes,
            "mode_count_recommended": self._mode_count_recommended,
            "reference": self._reference,
            "configurable_parameters": (),
            "status": "comparison-only",
        }


class SumOfExponentials:
    """Tolerance-driven positive sum of exponentials from Jiang et al."""

    _name = "SumOfExponentials"
    _reference = "doi:10.4208/cicp.OA-2016-0136"

    def __init__(
        self,
        *,
        target_error: float,
        t_final: float,
        min_step: float,
    ) -> None:
        self._target_error = _positive_finite(target_error, "target_error")
        self._t_final = _positive_finite(t_final, "t_final")
        self._min_step = _positive_finite(min_step, "min_step")
        if self._min_step > self._t_final:
            raise ValueError("min_step must not exceed t_final")

    @staticmethod
    def _sample_errors(
        alpha: float,
        dimensionless_times: np.ndarray,
        rates: np.ndarray,
        power_weights: np.ndarray,
        caputo_weights: np.ndarray,
    ) -> tuple[float, float]:
        maximum_power_error = 0.0
        maximum_caputo_error = 0.0
        exact_power = np.power(dimensionless_times, -1.0 - alpha)
        exact_caputo = (
            np.power(dimensionless_times, -alpha) / gamma(1.0 - alpha)
        )
        for start in range(0, dimensionless_times.size, 256):
            times = dimensionless_times[start : start + 256]
            exponentials = np.exp(-np.outer(times, rates))
            power_error = np.abs(
                exact_power[start : start + 256]
                - exponentials @ power_weights
            )
            caputo_error = np.abs(
                exact_caputo[start : start + 256]
                - exponentials @ caputo_weights
            )
            maximum_power_error = max(
                maximum_power_error,
                float(np.max(power_error)),
            )
            maximum_caputo_error = max(
                maximum_caputo_error,
                float(np.max(caputo_error)),
            )
        return maximum_power_error, maximum_caputo_error

    def spectrum(self, alpha: float) -> DiffusiveSpectrum:
        """Construct the positive spectrum on the declared time interval."""
        order = _validate_alpha(alpha)
        beta = 1.0 + order
        time_ratio = self._min_step / self._t_final
        dimensionless_power_target = (
            self._target_error * self._t_final**beta
        )
        dimensionless_caputo_target = (
            self._target_error * self._t_final**order
        )
        if (
            not np.isfinite(dimensionless_power_target)
            or not np.isfinite(dimensionless_caputo_target)
            or min(
                dimensionless_power_target,
                dimensionless_caputo_target,
            )
            <= 32.0 * np.finfo(np.float64).eps
        ):
            raise ValueError(
                "target_error is below meaningful float64 precision for "
                "this time interval"
            )

        num_dyadic_intervals = 0
        truncation_rate = 1.0
        while True:
            power_tail = (
                time_ratio ** (-beta)
                * float(gammaincc(beta, time_ratio * truncation_rate))
            )
            caputo_tail = (
                time_ratio ** (-order)
                * float(gammaincc(order, time_ratio * truncation_rate))
                / gamma(1.0 - order)
            )
            if (
                power_tail <= dimensionless_power_target / 8.0
                and caputo_tail <= dimensionless_caputo_target / 8.0
            ):
                break
            num_dyadic_intervals += 1
            if num_dyadic_intervals > 60:
                raise ValueError(
                    "the requested interval requires nonfinite exponential rates"
                )
            truncation_rate *= 2.0

        sample_times = np.geomspace(time_ratio, 1.0, 2049)
        accepted: tuple[
            np.ndarray,
            np.ndarray,
            np.ndarray,
            float,
            float,
            int,
        ] | None = None
        for quadrature_order in range(4, 65, 2):
            jacobi_nodes, jacobi_weights = gauss_jacobi(
                quadrature_order,
                0.0,
                order - 1.0,
            )
            low_rates = (jacobi_nodes + 1.0) / 2.0
            rate_groups = [low_rates]
            power_weight_groups = [
                jacobi_weights
                * low_rates
                / (2.0**order * gamma(beta))
            ]
            legendre_nodes, legendre_weights = roots_legendre(
                quadrature_order
            )
            for interval in range(num_dyadic_intervals):
                left = 2.0**interval
                right = 2.0 * left
                half_width = 0.5 * (right - left)
                midpoint = 0.5 * (right + left)
                interval_rates = midpoint + half_width * legendre_nodes
                rate_groups.append(interval_rates)
                power_weight_groups.append(
                    half_width
                    * legendre_weights
                    * np.power(interval_rates, order)
                    / gamma(beta)
                )
            dimensionless_rates = np.concatenate(rate_groups)
            dimensionless_power_weights = np.concatenate(
                power_weight_groups
            )
            dimensionless_caputo_weights = (
                order
                * dimensionless_power_weights
                / (dimensionless_rates * gamma(1.0 - order))
            )
            if dimensionless_rates.size > _SOE_RESOURCE_MAX_NUM_MODES:
                raise ValueError(
                    "the requested interval and error require too many modes"
                )
            power_error, caputo_error = self._sample_errors(
                order,
                sample_times,
                dimensionless_rates,
                dimensionless_power_weights,
                dimensionless_caputo_weights,
            )
            if (
                power_error <= dimensionless_power_target / 2.0
                and caputo_error <= dimensionless_caputo_target / 2.0
            ):
                accepted = (
                    dimensionless_rates,
                    dimensionless_power_weights,
                    dimensionless_caputo_weights,
                    power_error / self._t_final**beta,
                    caputo_error / self._t_final**order,
                    quadrature_order,
                )
                break
        if accepted is None:
            raise ValueError(
                "target_error was not achieved by the maximum quadrature order"
            )
        (
            dimensionless_rates,
            dimensionless_power_weights,
            dimensionless_caputo_weights,
            achieved_power_error,
            achieved_caputo_error,
            quadrature_order,
        ) = accepted
        rates = dimensionless_rates / self._t_final
        weights = dimensionless_caputo_weights / self._t_final**order
        metadata = MappingProxyType(
            {
                "representation": self._name,
                "reference": self._reference,
                "alpha": order,
                "num_modes": int(rates.size),
                "target_error": self._target_error,
                "t_final": self._t_final,
                "min_step": self._min_step,
                "quadrature": "Jiang-dyadic-Gaussian",
                "quadrature_order": quadrature_order,
                "dyadic_intervals": num_dyadic_intervals,
                "quadrature_nodes": tuple(float(rate) for rate in rates),
                "ordering": "increasing_rate",
                "kernel": "t^(-1-alpha)",
                "achieved_kernel_error": achieved_power_error,
                "achieved_caputo_kernel_error": achieved_caputo_error,
                "target_achievable": True,
            }
        )
        return DiffusiveSpectrum(rates, weights, metadata)

    def describe(self, alpha: float | None = None) -> dict[str, Any]:
        """Describe the requested interval or one generated spectrum."""
        if alpha is not None:
            return dict(self.spectrum(alpha).metadata)
        return {
            "representation": self._name,
            "target_error": self._target_error,
            "t_final": self._t_final,
            "min_step": self._min_step,
            "reference": self._reference,
            "configurable_parameters": (
                "target_error",
                "t_final",
                "min_step",
            ),
            "status": "supported-alternative",
        }


# Hand-rolled rather than scipy.special.roots_laguerre: the recurrence is carried
# with an explicit log scale because the unweighted weights overflow float64 at
# the node counts this representation needs.
@cache
def _laguerre_nodes_and_log_unweighted_weights(
    num_modes: int,
    laguerre_alpha: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    if laguerre_alpha != 0.0 and num_modes <= _SINE_RECOMMENDED_MAX_MODES:
        # SciPy is most accurate here; beyond this range its weighted values
        # underflow before multiplication by exp(node) restores their scale.
        nodes, weighted = roots_genlaguerre(num_modes, laguerre_alpha)
        log_effective_weights = np.log(weighted) + nodes
        nodes.setflags(write=False)
        log_effective_weights.setflags(write=False)
        return nodes, log_effective_weights
    indices = np.arange(num_modes, dtype=np.float64)
    nodes = eigh_tridiagonal(
        2.0 * indices + laguerre_alpha + 1.0,
        np.sqrt(indices[1:] * (indices[1:] + laguerre_alpha)),
        eigvals_only=True,
    )
    if laguerre_alpha != 0.0:
        derivative = -eval_genlaguerre(
            num_modes - 1,
            laguerre_alpha + 1.0,
            nodes,
        )
        correction = eval_genlaguerre(
            num_modes,
            laguerre_alpha,
            nodes,
        ) / derivative
        if np.all(np.isfinite(correction)):
            nodes -= correction
    previous = np.ones(num_modes, dtype=np.float64)
    current = 1.0 + laguerre_alpha - nodes
    log_scale = np.zeros(num_modes, dtype=np.float64)
    scale = np.maximum(np.abs(previous), np.abs(current))
    previous /= scale
    current /= scale
    log_scale += np.log(scale)
    for degree in range(1, num_modes + 1):
        following = (
            (2.0 * degree + laguerre_alpha + 1.0 - nodes) * current
            - (degree + laguerre_alpha) * previous
        ) / (degree + 1.0)
        scale = np.maximum(np.abs(current), np.abs(following))
        previous = current / scale
        current = following / scale
        log_scale += np.log(scale)
    log_effective_weights = (
        np.log(nodes)
        + lgamma(num_modes + laguerre_alpha + 1.0)
        - lgamma(num_modes + 1.0)
        - 2.0 * log(num_modes + 1.0)
        - 2.0 * (np.log(np.abs(current)) + log_scale)
        + nodes
    )
    nodes.setflags(write=False)
    log_effective_weights.setflags(write=False)
    return nodes, log_effective_weights


@cache
def _laguerre_nodes_and_unweighted_weights(
    num_modes: int,
    laguerre_alpha: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    nodes, log_effective_weights = (
        _laguerre_nodes_and_log_unweighted_weights(
            num_modes,
            laguerre_alpha,
        )
    )
    effective_weights = np.exp(log_effective_weights)
    effective_weights.setflags(write=False)
    return nodes, effective_weights


class YuanAgrawal:
    """Original Yuan-Agrawal Gauss-Laguerre diffusive spectrum.

    Warning:
        This representation is provided for expert use and comparison. It is
        generally less robust per mode than :class:`BirkSong` or
        :class:`Diethelm2008`.
    """

    _name = "YuanAgrawal"
    _reference = "doi:10.1115/1.1448322"

    def __init__(self, num_modes: int, *, rate_scale: float = 1.0) -> None:
        self._mode_count_recommended = _validate_mode_count(
            num_modes,
            minimum=1,
            recommended_maximum=_YUAN_RECOMMENDED_MAX_MODES,
            resource_maximum=_YUAN_RESOURCE_MAX_MODES,
            representation=self._name,
            reason="float64 quadrature accuracy usually saturates first",
        )
        self.num_modes = num_modes
        self._rate_scale = _positive_finite(
            rate_scale,
            "rate_scale",
        )

    def spectrum(self, alpha: float) -> DiffusiveSpectrum:
        """Generate the original squared-node Gauss-Laguerre spectrum."""
        order = _validate_alpha(alpha)
        nodes, effective_weights = _laguerre_nodes_and_unweighted_weights(
            self.num_modes
        )
        rates = self._rate_scale * np.square(nodes)
        weights = (
            (2.0 * sin(pi * order) / pi)
            * effective_weights
            * np.power(nodes, 2.0 * order - 1.0)
            * self._rate_scale**order
        )
        metadata = MappingProxyType(
            {
                "representation": self._name,
                "reference": self._reference,
                "alpha": order,
                "num_modes": self.num_modes,
                "mode_count_recommended": self._mode_count_recommended,
                "rate_scale": self._rate_scale,
                "quadrature": "Gauss-Laguerre",
                "quadrature_nodes": tuple(float(node) for node in nodes),
                "rate_map": "squared",
                "ordering": "increasing_rate",
            }
        )
        return DiffusiveSpectrum(rates, weights, metadata)

    def describe(self, alpha: float | None = None) -> dict[str, Any]:
        """Describe the configuration or one generated spectrum."""
        if alpha is not None:
            return dict(self.spectrum(alpha).metadata)
        return {
            "representation": self._name,
            "num_modes": self.num_modes,
            "mode_count_recommended": self._mode_count_recommended,
            "rate_scale": self._rate_scale,
            "reference": self._reference,
            "configurable_parameters": ("rate_scale",),
            "status": "supported-not-recommended",
        }


class Diethelm2022:
    """Discretize Diethelm2008's 2022 real-line representation.

    ``quadrature`` selects the published Gauss-Laguerre construction or a
    truncated composite trapezoidal, composite Simpson, or Gauss-Legendre
    rule. Gauss-Laguerre requires an even ``num_modes`` because each node
    produces two modes. Simpson quadrature requires an odd ``num_modes``.
    The truncation controls apply only to the three truncated rules.

    Warning:
        This representation is provided for expert use and comparison. It is
        more sensitive to mode count, time range, and scaling than BirkSong or
        Diethelm2008.
    """

    _name = "Diethelm2022"
    _reference = "doi:10.1007/978-981-19-7716-9_1"

    def __init__(
        self,
        num_modes: int,
        *,
        quadrature: str = "trapezoidal",
        target_error: float = 1.0e-8,
        decay_scale: float = 1.0,
        truncation_radius: float | None = None,
        rate_scale: float = 1.0,
    ) -> None:
        self._mode_count_recommended = _validate_mode_count(
            num_modes,
            minimum=3,
            recommended_maximum=_DIETHELM_2022_RECOMMENDED_MAX_MODES,
            resource_maximum=_DIETHELM_2022_RESOURCE_MAX_MODES,
            representation=self._name,
            reason="float64 accuracy is expected to saturate first",
        )
        if quadrature not in {
            "trapezoidal",
            "simpson",
            "gauss-legendre",
            "gauss-laguerre",
        }:
            raise ValueError(
                "quadrature must be 'trapezoidal', 'simpson', "
                "'gauss-legendre', or 'gauss-laguerre'"
            )
        if quadrature == "simpson" and num_modes % 2 == 0:
            raise ValueError(
                "Simpson quadrature requires an odd num_modes"
            )
        if quadrature == "gauss-laguerre" and num_modes % 2 != 0:
            raise ValueError(
                "Gauss-Laguerre quadrature requires an even num_modes"
            )
        self.num_modes = num_modes
        self._quadrature = quadrature
        self._target_error = _positive_finite(
            target_error,
            "target_error",
        )
        self._decay_scale = _positive_finite(
            decay_scale,
            "decay_scale",
        )
        self._rate_scale = _positive_finite(
            rate_scale,
            "rate_scale",
        )
        if truncation_radius is None:
            self._truncation_radius = None
        else:
            self._truncation_radius = _positive_finite(
                truncation_radius,
                "truncation_radius",
            )
        if quadrature == "gauss-laguerre":
            if self._truncation_radius is not None:
                raise ValueError(
                    "truncation_radius is not available for Gauss-Laguerre "
                    "quadrature"
                )
            if self._rate_scale != 1.0:
                raise ValueError(
                    "rate_scale is not available for the canonical "
                    "Gauss-Laguerre quadrature"
                )
            if self._target_error != 1.0e-8:
                raise ValueError(
                    "target_error is not available for Gauss-Laguerre "
                    "quadrature"
                )
            if self._decay_scale != 1.0:
                raise ValueError(
                    "decay_scale is not available for Gauss-Laguerre "
                    "quadrature"
                )

    def _automatic_radius(self, alpha: float) -> float:
        slow_decay = min(alpha, 1.0 - alpha)
        initial_radius = pi * sqrt(
            (self.num_modes - 1) / (2.0 * slow_decay)
        )
        if not isfinite(initial_radius):
            raise ValueError(
                "balanced truncation radius is nonfinite for this alpha"
            )
        log_rate_scale = log(self._rate_scale)
        float64_radius = min(
            log_rate_scale - _MIN_LOG_RATE,
            _MAX_LOG_RATE - log_rate_scale,
        )
        upper_radius = min(
            float64_radius,
            max(8.0, 4.0 * initial_radius),
        )
        if upper_radius <= 1.0:
            raise ValueError(
                "rate_scale leaves no finite interval for automatic truncation"
            )
        result = minimize_scalar(
            lambda radius: self._estimated_total_bound(alpha, radius),
            bounds=(1.0, upper_radius),
            method="bounded",
            options={"xatol": 1.0e-10},
        )
        if not result.success or not isfinite(float(result.x)):
            raise ValueError("could not determine a balanced truncation radius")
        balanced_radius = float(result.x)
        if (
            self._estimated_total_bound(alpha, balanced_radius)
            > self._target_error
        ):
            return balanced_radius
        if self._estimated_total_bound(alpha, 1.0) <= self._target_error:
            return 1.0
        return float(
            brentq(
                lambda radius: (
                    self._estimated_total_bound(alpha, radius)
                    - self._target_error
                ),
                1.0,
                balanced_radius,
                xtol=1.0e-12,
            )
        )

    def _tail_bound(self, alpha: float, radius: float) -> float:
        coefficient = (
            self._decay_scale * abs(sin(pi * alpha)) / pi
        )
        return float(
            coefficient
            * (
                np.exp(-alpha * radius) / alpha
                + np.exp(-(1.0 - alpha) * radius) / (1.0 - alpha)
            )
        )

    def _discretization_bound(self, radius: float) -> float:
        spacing = 2.0 * radius / (self.num_modes - 1)
        trapezoidal_bound = np.exp(-(pi * pi) / spacing)
        if self._quadrature == "trapezoidal":
            return float(trapezoidal_bound)
        if self._quadrature == "gauss-legendre":
            # Radius-selection surrogate; unlike the trapezoidal estimate,
            # this is not treated as an accuracy certificate.
            return float(trapezoidal_bound)
        # Composite Simpson is (4 T_h - T_{2h}) / 3. Bound both terms;
        # the embedded coarser trapezoid controls the asymptotic exponent.
        coarse_trapezoidal_bound = np.exp(
            -(pi * pi) / (2.0 * spacing)
        )
        return float(
            (
                4.0 * trapezoidal_bound
                + coarse_trapezoidal_bound
            )
            / 3.0
        )

    def _estimated_total_bound(self, alpha: float, radius: float) -> float:
        return self._tail_bound(alpha, radius) + self._discretization_bound(
            radius
        )

    def _gauss_laguerre_spectrum(self, alpha: float) -> DiffusiveSpectrum:
        num_nodes = self.num_modes // 2
        nodes, log_unweighted_weights = (
            _laguerre_nodes_and_log_unweighted_weights(num_nodes)
        )
        log_laguerre_weights = log_unweighted_weights - nodes
        coefficient = sin(pi * min(alpha, 1.0 - alpha)) / pi

        negative_log_rates = -nodes / alpha
        positive_log_rates = nodes / (1.0 - alpha)
        negative_log_weights = (
            log(coefficient / alpha) + log_laguerre_weights
        )
        positive_log_weights = (
            log(coefficient / (1.0 - alpha))
            + log_laguerre_weights
            + positive_log_rates
        )
        log_rates = np.concatenate(
            (negative_log_rates[::-1], positive_log_rates)
        )
        log_weights = np.concatenate(
            (negative_log_weights[::-1], positive_log_weights)
        )

        unsafe = (
            np.any(log_rates < _MIN_LOG_RATE)
            or np.any(log_rates > _MAX_LOG_RATE)
            or np.any(log_weights < _MIN_LOG_RATE)
            or np.any(log_weights > _MAX_LOG_RATE)
        )
        # For a rapidly decaying positive-half mode, its limiting contribution
        # is proportional to weight/rate. Require that ratio to remain
        # representable as well as the two factors themselves.
        positive_log_weight_rate_ratios = (
            positive_log_weights - positive_log_rates
        )
        unsafe = unsafe or bool(
            np.any(positive_log_weight_rate_ratios < _MIN_LOG_RATE)
            or np.any(positive_log_weight_rate_ratios > _MAX_LOG_RATE)
        )
        if unsafe:
            raise ValueError(
                "Gauss-Laguerre spectrum cannot be represented safely in "
                f"float64 for alpha={alpha} and num_modes={self.num_modes}"
            )

        rates = np.exp(log_rates)
        weights = np.exp(log_weights)
        if (
            not np.all(np.isfinite(rates))
            or not np.all(rates > 0.0)
            or not np.all(np.isfinite(weights))
            or not np.all(weights > 0.0)
        ):
            raise ValueError(
                "Gauss-Laguerre spectrum cannot be represented safely in "
                f"float64 for alpha={alpha} and num_modes={self.num_modes}"
            )

        quadrature_nodes = np.concatenate((nodes[::-1], nodes))
        metadata = MappingProxyType(
            {
                "representation": self._name,
                "reference": self._reference,
                "alpha": alpha,
                "num_modes": self.num_modes,
                "mode_count_recommended": self._mode_count_recommended,
                "rate_scale": 1.0,
                "quadrature": "Gauss-Laguerre",
                "laguerre_node_count": num_nodes,
                "quadrature_nodes": tuple(
                    float(node) for node in quadrature_nodes
                ),
                "ordering": "increasing_rate",
                "target_error": None,
                "decay_scale": None,
                "truncation_radius": None,
                "truncation_radius_source": "not-applicable",
                "estimated_tail_bound": None,
                "estimated_discretization_bound": None,
                "estimated_total_bound": None,
                "target_achievable": None,
                "discretization_error_model": "Gauss-Laguerre",
                "log_rate_spacing": None,
                "maximum_log_rate_spacing": None,
            }
        )
        return DiffusiveSpectrum(rates, weights, metadata)

    def spectrum(self, alpha: float) -> DiffusiveSpectrum:
        """Generate positive modes for the selected quadrature."""
        order = _validate_alpha(alpha)
        if self._quadrature == "gauss-laguerre":
            return self._gauss_laguerre_spectrum(order)
        radius = (
            self._automatic_radius(order)
            if self._truncation_radius is None
            else self._truncation_radius
        )
        log_rate_scale = log(self._rate_scale)
        if (
            log_rate_scale - radius < _MIN_LOG_RATE
            or log_rate_scale + radius > _MAX_LOG_RATE
        ):
            raise ValueError(
                "truncation interval produces nonfinite float64 rates; "
                "relax target_error or provide a smaller "
                "truncation_radius"
            )

        nominal_spacing = 2.0 * radius / (self.num_modes - 1)
        if self._quadrature == "gauss-legendre":
            reference_nodes, quadrature_factors = roots_legendre(
                self.num_modes
            )
            nodes = radius * reference_nodes
            quadrature_scale = radius
            quadrature_name = "Gauss-Legendre"
            log_rate_spacing = None
            maximum_log_rate_spacing = float(np.max(np.diff(nodes)))
        else:
            nodes = np.linspace(
                -radius,
                radius,
                self.num_modes,
                dtype=np.float64,
            )
            quadrature_factors = np.ones(
                self.num_modes,
                dtype=np.float64,
            )
            log_rate_spacing = nominal_spacing
            maximum_log_rate_spacing = nominal_spacing
            if self._quadrature == "trapezoidal":
                quadrature_factors[[0, -1]] = 0.5
                quadrature_scale = nominal_spacing
                quadrature_name = "composite-trapezoidal"
            else:
                quadrature_factors[1:-1:2] = 4.0
                quadrature_factors[2:-1:2] = 2.0
                quadrature_scale = nominal_spacing / 3.0
                quadrature_name = "composite-simpson"
        estimated_tail_bound = self._tail_bound(order, radius)
        estimated_discretization_bound = self._discretization_bound(radius)
        estimated_total_bound = (
            estimated_tail_bound + estimated_discretization_bound
        )
        target_achievable = bool(
            estimated_total_bound
            <= self._target_error
            * (1.0 + 64.0 * float(np.finfo(np.float64).eps))
        )
        rates = np.exp(log_rate_scale + nodes)
        weights = (
            quadrature_scale
            * quadrature_factors
            * (sin(pi * order) / pi)
            * np.exp(order * (log_rate_scale + nodes))
        )
        metadata = MappingProxyType(
            {
                "representation": self._name,
                "reference": self._reference,
                "alpha": order,
                "num_modes": self.num_modes,
                "mode_count_recommended": self._mode_count_recommended,
                "rate_scale": self._rate_scale,
                "quadrature": quadrature_name,
                "quadrature_nodes": tuple(float(node) for node in nodes),
                "ordering": "increasing_rate",
                "target_error": self._target_error,
                "decay_scale": self._decay_scale,
                "truncation_radius": radius,
                "truncation_radius_source": (
                    (
                        "target_error"
                        if target_achievable
                        else "balanced_tail_and_grid"
                    )
                    if self._truncation_radius is None
                    else "user"
                ),
                "estimated_tail_bound": estimated_tail_bound,
                "estimated_discretization_bound": (
                    estimated_discretization_bound
                ),
                "estimated_total_bound": estimated_total_bound,
                "target_achievable": (
                    None
                    if self._quadrature == "gauss-legendre"
                    else target_achievable
                ),
                "discretization_error_model": (
                    "analytic_strip_radius_surrogate"
                    if self._quadrature == "gauss-legendre"
                    else "analytic_strip_bound"
                ),
                "log_rate_spacing": log_rate_spacing,
                "maximum_log_rate_spacing": maximum_log_rate_spacing,
            }
        )
        return DiffusiveSpectrum(rates, weights, metadata)

    def describe(self, alpha: float | None = None) -> dict[str, Any]:
        """Describe the configuration or one ordered spectrum."""
        if alpha is not None:
            return dict(self.spectrum(alpha).metadata)
        gauss_laguerre = self._quadrature == "gauss-laguerre"
        return {
            "representation": self._name,
            "num_modes": self.num_modes,
            "mode_count_recommended": self._mode_count_recommended,
            "rate_scale": self._rate_scale,
            "quadrature": self._quadrature,
            "target_error": None if gauss_laguerre else self._target_error,
            "decay_scale": None if gauss_laguerre else self._decay_scale,
            "truncation_radius": self._truncation_radius,
            "reference": self._reference,
            "configurable_parameters": (
                ("quadrature",)
                if gauss_laguerre
                else (
                    "quadrature",
                    "target_error",
                    "decay_scale",
                    "truncation_radius",
                    "rate_scale",
                )
            ),
            "status": "supported-not-recommended",
        }
