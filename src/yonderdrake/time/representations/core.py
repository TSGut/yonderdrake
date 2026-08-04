"""Core full-history and Gauss-Jacobi time-memory representations."""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite, log2, log10, pi, sin
from types import MappingProxyType
from typing import Any

import numpy as np

from yonderdrake.time.coefficients import FloatArray, gauss_jacobi

_JACOBI_RECOMMENDED_MAX_MODES = 256  # Accuracy-saturation advisory.
_JACOBI_RESOURCE_MAX_MODES = 16_384  # Resource guard for O(n²)-work rules.
_STARTING_CONDITION_ADVISORY = 1.0e8  # Float64 accuracy-saturation advisory.

# The Jacobi exponents satisfy alpha + beta = power - 2, so a power of exactly
# one lands on the degenerate alpha + beta = -1 recurrence.
_CAYLEY_DEGENERATE_POWER = 1.0
_JACOBI_DEGENERACY_TOLERANCE = 64.0 * np.finfo(np.float64).eps
# Selection from a declared range never returns an exponent narrower than
# Diethelm2008's published two, which is the best choice for narrow spans anyway
# and keeps the automatic path clear of the degenerate power. Beyond twelve
# the rate span exceeds what float64 exponentials resolve.
_CAYLEY_POWER_MIN = 2.0
_CAYLEY_POWER_MAX = 12.0
# Least squares over decades in [2, 16], mode counts in [16, 256], and orders
# in [0.2, 0.8]. The best exponent follows the width of the requested rate
# window and barely moves with the order, which is what makes choosing it from
# the declared time range alone dependable.
_CAYLEY_POWER_FIT = (0.211, 0.4215, 0.2020)


class ModeCountAdvisoryWarning(UserWarning):
    """Warn that a representation is beyond its useful mode-count range."""


class StartingCorrectionAdvisoryWarning(UserWarning):
    """Warn that Lubich starting corrections are poorly conditioned."""


def _validate_mode_count(
    num_modes: int,
    *,
    minimum: int,
    recommended_maximum: int,
    resource_maximum: int,
    representation: str,
    reason: str,
) -> bool:
    if isinstance(num_modes, bool) or not isinstance(num_modes, int):
        raise TypeError("num_modes must be an integer")
    if num_modes < minimum:
        qualifier = "positive" if minimum == 1 else f"at least {minimum}"
        raise ValueError(f"num_modes must be {qualifier}")
    if num_modes > resource_maximum:
        raise ValueError(
            f"num_modes exceeds the resource ceiling of {resource_maximum}"
        )
    recommended = num_modes <= recommended_maximum
    if not recommended:
        warnings.warn(
            f"{representation} num_modes={num_modes} exceeds the recommended "
            f"maximum {recommended_maximum}; {reason}.",
            ModeCountAdvisoryWarning,
            stacklevel=3,
        )
    return recommended


@dataclass(frozen=True, slots=True)
class FullHistory:
    """Direct time history with piecewise-linear interpolation."""

    def describe(self, alpha: float | None = None) -> dict[str, str]:
        return {
            "representation": "FullHistory",
            "interpolant": "linear",
        }


@dataclass(frozen=True, slots=True)
class LubichCQ:
    """Lubich convolution quadrature based on BDF1 or BDF2."""

    order: str = "bdf2"
    num_corrections: int | None = None

    def __post_init__(self) -> None:
        normalized_order = str(self.order).lower()
        if normalized_order not in {"bdf1", "bdf2"}:
            raise ValueError("order must be 'bdf1' or 'bdf2'")
        object.__setattr__(self, "order", normalized_order)
        corrections = self.num_corrections
        if corrections is None:
            corrections = 1 if normalized_order == "bdf1" else 2
        if isinstance(corrections, bool) or not isinstance(corrections, int):
            raise TypeError("num_corrections must be an integer")
        if not 0 <= corrections <= 16:
            raise ValueError("num_corrections must lie between 0 and 16")
        object.__setattr__(self, "num_corrections", corrections)

    def _starting_system_condition(self, alpha: float) -> float:
        corrections = self.num_corrections
        assert corrections is not None
        if corrections == 0:
            return 1.0
        indices = np.arange(1, corrections + 1, dtype=np.float64)
        exponents = alpha * indices
        matrix = np.power(indices[None, :], exponents[:, None])
        condition = float(np.linalg.cond(matrix, p=np.inf))
        if not isfinite(condition):
            raise ValueError(
                "the requested Lubich starting-correction system is singular "
                f"for alpha={alpha} and num_corrections={corrections}"
            )
        return condition

    def describe(self, alpha: float | None = None) -> dict[str, Any]:
        """Describe the CQ rule and its alpha-dependent correction system."""
        description: dict[str, Any] = {
            "representation": "LubichCQ",
            "reference": (
                "doi:10.1137/0517050; doi:10.1007/BF01398686; "
                "doi:10.1007/BF01398687"
            ),
            "order": self.order,
            "num_corrections": self.num_corrections,
            "step_grid": "uniform",
            "storage": "full-history",
        }
        if alpha is not None:
            order = _validate_alpha(alpha)
            corrections = self.num_corrections
            assert corrections is not None
            condition = self._starting_system_condition(order)
            description.update(
                {
                    "alpha": order,
                    "starting_exponents": tuple(
                        float((index + 1) * order)
                        for index in range(corrections)
                    ),
                    "starting_system_condition": condition,
                    "starting_system_recommended": (
                        condition <= _STARTING_CONDITION_ADVISORY
                    ),
                }
            )
        return description


@dataclass(frozen=True, slots=True)
class AlikhanovL21Sigma:
    """Alikhanov's uniform-grid L2-1-sigma Caputo formula."""

    def describe(self, alpha: float | None = None) -> dict[str, Any]:
        """Describe the offset formula and its alpha-dependent sigma."""
        description: dict[str, Any] = {
            "representation": "AlikhanovL21Sigma",
            "reference": "doi:10.1016/j.jcp.2014.09.031",
            "step_grid": "uniform",
            "evaluation": "t_n_plus_sigma",
            "storage": "full-history",
        }
        if alpha is not None:
            order = _validate_alpha(alpha)
            description.update(
                {
                    "alpha": order,
                    "sigma": 1.0 - 0.5 * order,
                }
            )
        return description


def _immutable_complex_array(values: Any) -> np.ndarray:
    result = np.array(values, dtype=np.complex128, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class ComplexContourSpectrum:
    """Conjugate-symmetric complex nodes for one CQ history contour."""

    rates: np.ndarray
    weights: np.ndarray
    metadata: MappingProxyType[str, Any]

    def __post_init__(self) -> None:
        rates = _immutable_complex_array(self.rates)
        weights = _immutable_complex_array(self.weights)
        if rates.ndim != 1 or weights.ndim != 1 or rates.shape != weights.shape:
            raise ValueError(
                "rates and weights must be one-dimensional and equal-sized"
            )
        if rates.size == 0 or rates.size % 2 == 0:
            raise ValueError("a contour spectrum must have an odd number of nodes")
        if not np.all(np.isfinite(rates)) or not np.all(np.isfinite(weights)):
            raise ValueError("contour rates and weights must be finite")
        if not np.allclose(rates, np.conjugate(rates[::-1]), rtol=2e-14, atol=0.0):
            raise ValueError("contour rates must occur in conjugate pairs")
        if not np.allclose(
            weights,
            np.conjugate(weights[::-1]),
            rtol=2e-14,
            atol=0.0,
        ):
            raise ValueError("contour weights must occur in conjugate pairs")
        object.__setattr__(self, "rates", rates)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True, slots=True)
class FastObliviousCQ:
    """Fast BDF1 convolution quadrature with dyadic contour histories."""

    target_error: float = 1.0e-6
    num_levels: int = 16
    nodes_per_level: int | None = None
    direct_steps: int = 20
    contour: str = "talbot"

    def __post_init__(self) -> None:
        target = float(self.target_error)
        if not isfinite(target) or not 0.0 < target < 1.0:
            raise ValueError("target_error must lie strictly between 0 and 1")
        object.__setattr__(self, "target_error", target)
        if isinstance(self.num_levels, bool) or not isinstance(self.num_levels, int):
            raise TypeError("num_levels must be an integer")
        if not 1 <= self.num_levels <= 60:
            raise ValueError("num_levels must lie between 1 and 60")
        nodes = self.nodes_per_level
        if nodes is None:
            digits = -np.log10(target)
            nodes = max(10, int(np.ceil(5.0 + 5.0 * digits / 3.0)))
        if isinstance(nodes, bool) or not isinstance(nodes, int):
            raise TypeError("nodes_per_level must be an integer")
        if not 4 <= nodes <= 64:
            raise ValueError("nodes_per_level must lie between 4 and 64")
        object.__setattr__(self, "nodes_per_level", nodes)
        if isinstance(self.direct_steps, bool) or not isinstance(
            self.direct_steps,
            int,
        ):
            raise TypeError("direct_steps must be an integer")
        if not 6 <= self.direct_steps <= 4096:
            raise ValueError("direct_steps must lie between 6 and 4096")
        contour = str(self.contour).lower()
        if contour != "talbot":
            raise ValueError("contour must be 'talbot'")
        object.__setattr__(self, "contour", contour)

    @property
    def max_steps(self) -> int:
        """Largest supported step index for the configured dyadic levels."""
        return int(2**self.num_levels - 1)

    def contour_spectrum(
        self,
        alpha: float,
        step_size: float,
        level: int,
    ) -> ComplexContourSpectrum:
        """Build the published local contour for one dyadic history level."""
        order = _validate_alpha(alpha)
        step = float(step_size)
        if not isfinite(step) or step <= 0.0:
            raise ValueError("step_size must be finite and positive")
        if isinstance(level, bool) or not isinstance(level, int):
            raise TypeError("level must be an integer")
        if not 0 <= level < self.num_levels:
            raise ValueError("level lies outside the configured dyadic range")
        nodes = self.nodes_per_level
        assert nodes is not None
        indices = np.arange(-nodes, nodes + 1, dtype=np.float64)
        scale_steps = max((self.direct_steps + 1) // 2, 2**level)
        interval_end = 4.0 * scale_steps * step
        theta = indices * np.pi / (nodes + 1.0)
        near_zero = np.abs(theta) < 1.0e-7
        theta_cotangent = np.empty_like(theta)
        derivative = np.empty_like(theta)
        theta_cotangent[near_zero] = 1.0 - theta[near_zero] ** 2 / 3.0
        derivative[near_zero] = -2.0 * theta[near_zero] / 3.0
        regular = ~near_zero
        theta_cotangent[regular] = theta[regular] / np.tan(theta[regular])
        derivative[regular] = (
            1.0 / np.tan(theta[regular])
            - theta[regular] / np.sin(theta[regular]) ** 2
        )
        mu = 8.0 / interval_end
        rates = mu * (theta_cotangent + 0.6j * theta)
        contour_derivative = mu * (derivative + 0.6j)
        quadrature_weights = -0.5j / (nodes + 1.0) * contour_derivative
        parameters = {"mu": mu, "nu": 0.6, "sigma": 0.0}
        transfer = np.power(rates, order - 1.0)
        weights = quadrature_weights * transfer
        metadata = MappingProxyType(
            {
                "representation": "FastObliviousCQ",
                "reference": "doi:10.1137/050623139",
                "alpha": order,
                "step_size": step,
                "level": level,
                "interval_end": interval_end,
                "contour": self.contour,
                "nodes_per_level": nodes,
                **parameters,
            }
        )
        return ComplexContourSpectrum(rates, weights, metadata)

    def describe(self, alpha: float | None = None) -> dict[str, Any]:
        """Describe the bounded dyadic fast-CQ configuration."""
        description: dict[str, Any] = {
            "representation": "FastObliviousCQ",
            "reference": "doi:10.1137/050623139",
            "order": "bdf1",
            "target_error": self.target_error,
            "num_levels": self.num_levels,
            "nodes_per_level": self.nodes_per_level,
            "direct_steps": self.direct_steps,
            "contour": self.contour,
            "history_splitting": "dyadic",
            "max_steps": self.max_steps,
            "work": "O(N log N)",
            "storage": "O(log N)",
        }
        if alpha is not None:
            description["alpha"] = _validate_alpha(alpha)
        return description


def _immutable_float_array(values: Any) -> FloatArray:
    result = np.array(values, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class DiffusiveSpectrum:
    """Normalized positive-rate approximation of a Caputo kernel."""

    rates: FloatArray
    weights: FloatArray
    metadata: MappingProxyType[str, Any]

    def __post_init__(self) -> None:
        rates = _immutable_float_array(self.rates)
        weights = _immutable_float_array(self.weights)
        if rates.ndim != 1 or weights.ndim != 1 or rates.shape != weights.shape:
            raise ValueError(
                "rates and weights must be one-dimensional and equal-sized"
            )
        if rates.size == 0:
            raise ValueError("a spectrum must contain at least one mode")
        if not np.all(np.isfinite(rates)) or not np.all(rates > 0.0):
            raise ValueError("all rates must be finite and positive")
        if not np.all(np.isfinite(weights)) or not np.all(weights > 0.0):
            raise ValueError("all weights must be finite and positive")
        object.__setattr__(self, "rates", rates)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True)
class OscillatorSpectrum:
    """Positive-frequency quadrature for an undamped oscillator memory."""

    frequencies: FloatArray
    weights: FloatArray
    metadata: MappingProxyType[str, Any]

    def __post_init__(self) -> None:
        frequencies = _immutable_float_array(self.frequencies)
        weights = _immutable_float_array(self.weights)
        if (
            frequencies.ndim != 1
            or weights.ndim != 1
            or frequencies.shape != weights.shape
        ):
            raise ValueError(
                "frequencies and weights must be one-dimensional and equal-sized"
            )
        if frequencies.size == 0:
            raise ValueError("a spectrum must contain at least one mode")
        if not np.all(np.isfinite(frequencies)) or not np.all(
            frequencies > 0.0
        ):
            raise ValueError("all frequencies must be finite and positive")
        if not np.all(np.isfinite(weights)) or not np.all(weights > 0.0):
            raise ValueError("all weights must be finite and positive")
        object.__setattr__(self, "frequencies", frequencies)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


class _SingleExponential:
    """Exact one-mode representation of an exponential memory kernel."""

    num_modes = 1

    @staticmethod
    def spectrum(decay_rate: float) -> DiffusiveSpectrum:
        try:
            rate = float(decay_rate)
        except (TypeError, ValueError) as error:
            raise TypeError("decay_rate must be a real scalar") from error
        if not np.isfinite(rate) or rate <= 0.0:
            raise ValueError("decay_rate must be finite and positive")
        metadata = MappingProxyType(
            {
                "representation": "SingleExponential",
                "reference": "doi:10.12785/pfda/010201",
                "decay_rate": rate,
                "num_modes": 1,
                "quadrature": "none-exact",
                "quadrature_nodes": (rate,),
                "ordering": "increasing_rate",
            }
        )
        return DiffusiveSpectrum(
            np.asarray([rate], dtype=np.float64),
            np.asarray([1.0], dtype=np.float64),
            metadata,
        )

    def describe(self, decay_rate: float | None = None) -> dict[str, Any]:
        if decay_rate is None:
            return {
                "representation": "SingleExponential",
                "num_modes": 1,
                "reference": "doi:10.12785/pfda/010201",
                "status": "exact-exponential-memory",
            }
        return dict(self.spectrum(decay_rate).metadata)


def _positive_finite(value: Any, name: str) -> float:
    number = float(value)
    if not isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return number


def _cayley_power_for_range(
    t_final: float,
    min_step: float,
    num_modes: int,
) -> float:
    """Choose the Cayley exponent spanning ``[min_step, t_final]``."""
    decades = log10(t_final / min_step)
    intercept, per_decade, per_octave = _CAYLEY_POWER_FIT
    power = intercept + per_decade * decades + per_octave * log2(num_modes)
    return min(max(power, _CAYLEY_POWER_MIN), _CAYLEY_POWER_MAX)


def _validate_alpha(alpha: float) -> float:
    try:
        value = float(alpha)
    except (TypeError, ValueError) as error:
        raise TypeError("alpha must be a real scalar") from error
    if not np.isfinite(value) or not 0.0 < value < 1.0:
        raise ValueError("alpha must satisfy 0 < alpha < 1")
    return value


def validate_checkpoint_representation(
    metadata: Any,
    representation: Any,
    alpha: float,
) -> None:
    """Validate that checkpoint modes use the stepper's exact spectrum."""
    if not isinstance(metadata, Mapping):
        raise ValueError("checkpoint representation metadata is missing")
    expected = representation.describe(alpha)
    for field, expected_value in expected.items():
        if field == "quadrature_nodes":
            continue
        if field not in metadata:
            raise ValueError(
                "checkpoint representation metadata is incomplete"
            )
        if metadata[field] != expected_value:
            raise ValueError(
                "checkpoint representation does not match the stepper"
            )
    if "quadrature_nodes" in expected:
        try:
            observed_nodes = tuple(
                float(value) for value in metadata["quadrature_nodes"]
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                "checkpoint representation metadata is incomplete"
            ) from error
        if observed_nodes != tuple(expected["quadrature_nodes"]):
            raise ValueError(
                "checkpoint representation does not match the stepper"
            )


class _JacobiRepresentation:
    _name: str
    _reference: str

    def __init__(self, num_modes: int, **method_parameters: Any) -> None:
        self._mode_count_recommended = _validate_mode_count(
            num_modes,
            minimum=1,
            recommended_maximum=_JACOBI_RECOMMENDED_MAX_MODES,
            resource_maximum=_JACOBI_RESOURCE_MAX_MODES,
            representation=self._name,
            reason="float64 quadrature accuracy usually saturates first",
        )
        unknown = set(method_parameters) - {"rate_scale"}
        if unknown:
            names = ", ".join(sorted(unknown))
            raise TypeError(f"unsupported representation parameter(s): {names}")
        rate_scale = float(method_parameters.get("rate_scale", 1.0))
        if not np.isfinite(rate_scale) or rate_scale <= 0.0:
            raise ValueError("rate_scale must be finite and positive")
        self.num_modes = num_modes
        self._rate_scale = rate_scale

    def _jacobi_exponents(self, alpha: float) -> tuple[float, float]:
        raise NotImplementedError

    def _native_coefficients(
        self,
        alpha: float,
        nodes: FloatArray,
        quadrature_weights: FloatArray,
    ) -> tuple[FloatArray, FloatArray]:
        raise NotImplementedError

    def spectrum(self, alpha: float) -> DiffusiveSpectrum:
        """Generate a normalized spectrum for one immutable Caputo order."""
        order = _validate_alpha(alpha)
        jacobi_alpha, jacobi_beta = self._jacobi_exponents(order)
        nodes, quadrature_weights = gauss_jacobi(
            self.num_modes,
            jacobi_alpha,
            jacobi_beta,
        )
        rates, weights = self._native_coefficients(
            order,
            nodes,
            quadrature_weights,
        )
        rates = self._rate_scale * rates
        weights = self._rate_scale**order * weights
        permutation = np.argsort(rates, kind="stable")
        rates = rates[permutation]
        weights = weights[permutation]
        ordered_nodes = nodes[permutation]
        metadata = MappingProxyType(
            {
                "representation": self._name,
                "reference": self._reference,
                "alpha": order,
                "num_modes": self.num_modes,
                "mode_count_recommended": self._mode_count_recommended,
                "rate_scale": self._rate_scale,
                "quadrature": "Gauss-Jacobi",
                "jacobi_alpha": jacobi_alpha,
                "jacobi_beta": jacobi_beta,
                "quadrature_nodes": tuple(float(node) for node in ordered_nodes),
                "ordering": "increasing_rate",
            }
        )
        return DiffusiveSpectrum(rates, weights, metadata)

    def describe(self, alpha: float | None = None) -> dict[str, Any]:
        """Describe configuration, or full generated metadata when ordered."""
        if alpha is None:
            return {
                "representation": self._name,
                "num_modes": self.num_modes,
                "mode_count_recommended": self._mode_count_recommended,
                "rate_scale": self._rate_scale,
                "reference": self._reference,
                "configurable_parameters": ("rate_scale",),
            }
        return dict(self.spectrum(alpha).metadata)


class Cayley(_JacobiRepresentation):
    """Cayley-transform diffusive spectrum with a selectable exponent.

    The rate map is ``((1 - x) / (1 + x)) ** power`` over the Gauss-Jacobi
    reference interval, so ``power`` sets how many decades of relaxation rate
    a given mode count spans. `Diethelm2008` and `BirkSong` are the published
    ``power=2`` and ``power=4`` members of this family.

    Pass ``t_final`` and ``min_step`` to size the exponent from the time range
    the problem needs, or ``power`` to set it directly.
    """

    _name = "Cayley"
    _reference = (
        "doi:10.1007/s11075-008-9193-8, doi:10.1007/s00466-010-0510-4"
    )

    def __init__(
        self,
        num_modes: int,
        *,
        power: float | None = None,
        t_final: float | None = None,
        min_step: float | None = None,
        **method_parameters: Any,
    ) -> None:
        declared_range = t_final is not None or min_step is not None
        if power is not None and declared_range:
            raise ValueError(
                "give either power or the t_final and min_step pair"
            )
        if power is None and not declared_range:
            raise ValueError(
                "Cayley requires power, or t_final together with min_step"
            )
        super().__init__(num_modes, **method_parameters)
        if declared_range:
            if t_final is None or min_step is None:
                raise ValueError("t_final and min_step must be given together")
            final = _positive_finite(t_final, "t_final")
            step = _positive_finite(min_step, "min_step")
            if step >= final:
                raise ValueError("min_step must be smaller than t_final")
            self._power = _cayley_power_for_range(final, step, self.num_modes)
        else:
            chosen = _positive_finite(power, "power")
            if chosen == _CAYLEY_DEGENERATE_POWER:
                raise ValueError(
                    "power=1 places the Jacobi exponents on the degenerate "
                    "alpha + beta = -1 recurrence; every other positive "
                    "power is supported"
                )
            self._power = chosen

    @property
    def power(self) -> float:
        """Return the Cayley exponent this spectrum was built with."""
        return self._power

    def _jacobi_exponents(self, alpha: float) -> tuple[float, float]:
        return (
            self._power * alpha - 1.0,
            self._power * (1.0 - alpha) - 1.0,
        )

    def _native_coefficients(
        self,
        alpha: float,
        nodes: FloatArray,
        quadrature_weights: FloatArray,
    ) -> tuple[FloatArray, FloatArray]:
        denominator = 1.0 + nodes
        rates = np.power((1.0 - nodes) / denominator, self._power)
        weights = (
            (2.0 * self._power * sin(pi * alpha) / pi)
            * quadrature_weights
            / np.power(denominator, self._power)
        )
        return rates, weights

    def describe(self, alpha: float | None = None) -> dict[str, Any]:
        """Describe configuration, or full generated metadata when ordered."""
        described = super().describe(alpha)
        described["power"] = self._power
        if alpha is None:
            described["configurable_parameters"] = ("power", "rate_scale")
        return described


class Jacobi(_JacobiRepresentation):
    """Two-parameter Gauss-Jacobi diffusive representation.

    ``sigma`` controls the rate-map behaviour near ``x=1`` and ``rho``
    controls it near ``x=-1``. Both parameters must be finite and positive.
    """

    _name = "Jacobi"
    _reference = "doi:10.1109/ICFDA58234.2023.10153228"

    def __init__(
        self,
        num_modes: int,
        *,
        sigma: float,
        rho: float,
        **method_parameters: Any,
    ) -> None:
        super().__init__(num_modes, **method_parameters)
        self._sigma = _positive_finite(sigma, "sigma")
        self._rho = _positive_finite(rho, "rho")

    @property
    def sigma(self) -> float:
        """Return the exponent controlling the low-rate end of the map."""
        return self._sigma

    @property
    def rho(self) -> float:
        """Return the exponent controlling the high-rate end of the map."""
        return self._rho

    def _jacobi_exponents(self, alpha: float) -> tuple[float, float]:
        return (
            self._sigma * alpha - 1.0,
            self._rho * (1.0 - alpha) - 1.0,
        )

    def _native_coefficients(
        self,
        alpha: float,
        nodes: FloatArray,
        quadrature_weights: FloatArray,
    ) -> tuple[FloatArray, FloatArray]:
        one_minus = 1.0 - nodes
        one_plus = 1.0 + nodes
        rates = np.power(one_minus, self._sigma) / np.power(
            one_plus,
            self._rho,
        )
        weights = (
            (sin(pi * alpha) / pi)
            * quadrature_weights
            * np.power(one_plus, -self._rho)
            * (self._sigma * one_plus + self._rho * one_minus)
        )
        return rates, weights

    def spectrum(self, alpha: float) -> DiffusiveSpectrum:
        """Generate a normalized spectrum for one immutable Caputo order."""
        order = _validate_alpha(alpha)
        exponent_sum = self._sigma * order + self._rho * (1.0 - order)
        if abs(exponent_sum - 1.0) <= _JACOBI_DEGENERACY_TOLERANCE:
            raise ValueError(
                f"Jacobi(sigma={self._sigma}, rho={self._rho}) is "
                f"degenerate at alpha={order}"
            )
        return super().spectrum(order)

    def describe(self, alpha: float | None = None) -> dict[str, Any]:
        """Describe configuration, or full generated metadata when ordered."""
        described = super().describe(alpha)
        described["rho"] = self._rho
        described["sigma"] = self._sigma
        if alpha is None:
            described["configurable_parameters"] = (
                "sigma",
                "rho",
                "rate_scale",
            )
        return described


class Diethelm2008(_JacobiRepresentation):
    """Diethelm2008's Gauss-Jacobi improvement of the diffusive representation."""

    _name = "Diethelm2008"
    _reference = "doi:10.1007/s11075-008-9193-8"

    def _jacobi_exponents(self, alpha: float) -> tuple[float, float]:
        transformed_order = 2.0 * alpha - 1.0
        return transformed_order, -transformed_order

    def _native_coefficients(
        self,
        alpha: float,
        nodes: FloatArray,
        quadrature_weights: FloatArray,
    ) -> tuple[FloatArray, FloatArray]:
        denominator = 1.0 + nodes
        ratio = (1.0 - nodes) / denominator
        rates = np.square(ratio)
        weights = (
            (4.0 * sin(pi * alpha) / pi)
            * quadrature_weights
            / np.square(denominator)
        )
        return rates, weights


class BirkSong(_JacobiRepresentation):
    """Birk-Song's squared Cayley-transform Gauss-Jacobi spectrum."""

    _name = "BirkSong"
    _reference = "doi:10.1007/s00466-010-0510-4"

    def _jacobi_exponents(self, alpha: float) -> tuple[float, float]:
        transformed_order = 2.0 * alpha - 1.0
        return 2.0 * transformed_order + 1.0, 1.0 - 2.0 * transformed_order

    def _native_coefficients(
        self,
        alpha: float,
        nodes: FloatArray,
        quadrature_weights: FloatArray,
    ) -> tuple[FloatArray, FloatArray]:
        denominator = 1.0 + nodes
        ratio = (1.0 - nodes) / denominator
        rates = np.power(ratio, 4)
        weights = (
            (8.0 * sin(pi * alpha) / pi)
            * quadrature_weights
            / np.power(denominator, 4)
        )
        return rates, weights
