"""Stable scalar coefficients and Gaussian quadrature helpers."""

from __future__ import annotations

from math import cos, exp, gamma, lgamma, pi
from typing import Any, TypeAlias

import numpy as np
from scipy.linalg import eigh_tridiagonal
from scipy.special import eval_jacobi

FloatArray: TypeAlias = Any
ArrayLike: TypeAlias = Any


def _nonnegative_array(z: float | ArrayLike) -> tuple[FloatArray, bool]:
    values = np.asarray(z, dtype=np.float64)
    scalar = values.ndim == 0
    if np.any(values < 0.0):
        raise ValueError("recurrence arguments must be nonnegative")
    if not np.all(np.isfinite(values)):
        raise ValueError("recurrence arguments must be finite")
    return values, scalar


def exp_neg(z: float | ArrayLike) -> float | FloatArray:
    """Evaluate ``exp(-z)`` for finite nonnegative ``z``."""
    values, scalar = _nonnegative_array(z)
    result = np.exp(-values)
    return float(result) if scalar else result


def phi1(z: float | ArrayLike) -> float | FloatArray:
    r"""Evaluate :math:`(1-\exp(-z))/z` with its removable limit at zero."""
    values, scalar = _nonnegative_array(z)
    small = values < 1.0e-4
    result = np.empty_like(values)
    if np.any(small):
        x = values[small]
        result[small] = 1.0 + x * (
            -0.5 + x * (1.0 / 6.0 + x * (-1.0 / 24.0 + x / 120.0))
        )
    if np.any(~small):
        x = values[~small]
        result[~small] = -np.expm1(-x) / x
    return float(result) if scalar else result


def psi(z: float | ArrayLike) -> float | FloatArray:
    r"""Evaluate :math:`(1-\phi_1(z))/z` with its limit at zero."""
    values, scalar = _nonnegative_array(z)
    # A crossover sweep against a 100-digit reference gave the smallest worst
    # float64 error at 1e-3: the series wins below it and the direct form above.
    small = values < 1.0e-3
    result = np.empty_like(values)
    if np.any(small):
        x = values[small]
        result[small] = 0.5 + x * (
            -1.0 / 6.0 + x * (1.0 / 24.0 - x / 120.0)
        )
    if np.any(~small):
        x = values[~small]
        result[~small] = (1.0 - np.asarray(phi1(x))) / x
    return float(result) if scalar else result


def recurrence_coefficients(
    spectrum: Any,
    step_size: float,
    *,
    final_time: float | None = None,
) -> tuple[FloatArray, FloatArray, float]:
    """Return exact linear-interpolant coefficients for one spectrum."""
    validate_recurrence_interval(
        spectrum,
        step_size,
        final_time=final_time,
    )
    arguments = spectrum.rates * step_size
    decay = np.asarray(exp_neg(arguments), dtype=np.float64)
    interpolation = np.asarray(phi1(arguments), dtype=np.float64)
    if spectrum.metadata.get("representation") == "SumOfExponentials":
        alpha = float(spectrum.metadata["alpha"])
        implicit_weight = step_size ** (-alpha) / gamma(2.0 - alpha)
    else:
        implicit_weight = float(np.dot(spectrum.weights, interpolation))
    return decay, interpolation, implicit_weight


def quadratic_recurrence_coefficients(
    spectrum: Any,
    step_size: float,
    *,
    previous_step_size: float | None,
    final_time: float | None = None,
) -> tuple[FloatArray, FloatArray, FloatArray, float, float]:
    """Return exact quadratic-interpolant coefficients for one spectrum.

    The final two scalars multiply the current and previous physical increments
    in the residual.  ``previous_step_size=None`` deliberately selects a linear
    starting step.
    """
    validate_recurrence_interval(
        spectrum,
        step_size,
        final_time=final_time,
    )
    arguments = spectrum.rates * step_size
    decay = np.asarray(exp_neg(arguments), dtype=np.float64)
    linear = np.asarray(phi1(arguments), dtype=np.float64)
    if previous_step_size is None:
        previous = np.zeros_like(linear)
        if spectrum.metadata.get("representation") == "SumOfExponentials":
            alpha = float(spectrum.metadata["alpha"])
            implicit = step_size ** (-alpha) / gamma(2.0 - alpha)
        else:
            implicit = float(np.dot(spectrum.weights, linear))
        return decay, linear, previous, implicit, 0.0

    if not np.isfinite(previous_step_size) or previous_step_size <= 0.0:
        raise ValueError("previous_step_size must be finite and positive")
    ratio = step_size / (step_size + previous_step_size)
    curvature = 2.0 * np.asarray(psi(arguments)) - linear
    current = linear + ratio * curvature
    previous = -(step_size / previous_step_size) * ratio * curvature
    implicit = float(np.dot(spectrum.weights, current))
    previous_weight = float(np.dot(spectrum.weights, previous))
    if spectrum.metadata.get("representation") == "SumOfExponentials":
        alpha = float(spectrum.metadata["alpha"])
        linear_weight = step_size ** (-alpha) / gamma(2.0 - alpha)
        curvature_scale = alpha / (2.0 - alpha)
        implicit = linear_weight * (1.0 + ratio * curvature_scale)
        previous_weight = (
            -linear_weight
            * (step_size / previous_step_size)
            * ratio
            * curvature_scale
        )
    return decay, current, previous, implicit, previous_weight


def oscillator_coefficients(
    spectrum: Any,
    alpha: float,
    step_size: float,
) -> tuple[
    FloatArray,
    FloatArray,
    FloatArray,
    FloatArray,
    FloatArray,
    float,
]:
    """Return the exact rotation and linear-forcing coefficients for SDR."""
    frequencies = spectrum.frequencies
    phase = frequencies * step_size
    cosine = np.cos(phase)
    sinc = np.sinc(phase / pi)
    sine_over_frequency = step_size * sinc
    negative_frequency_sine = -frequencies * np.sin(phase)
    forcing_scale = 2.0 * cos(pi * alpha / 2.0) / pi
    position_forcing = (
        0.5
        * forcing_scale
        * step_size
        * np.square(np.sinc(phase / (2.0 * pi)))
    )
    velocity_forcing = forcing_scale * sinc
    implicit_weight = float(np.dot(spectrum.weights, position_forcing))
    return (
        cosine,
        sine_over_frequency,
        negative_frequency_sine,
        position_forcing,
        velocity_forcing,
        implicit_weight,
    )


def validate_recurrence_interval(
    spectrum: Any,
    step_size: float,
    *,
    final_time: float | None = None,
) -> None:
    """Reject use outside a spectrum's certified time interval."""
    metadata = spectrum.metadata
    if metadata.get("representation") != "SumOfExponentials":
        return
    minimum = float(metadata["min_step"])
    maximum = float(metadata["t_final"])
    tolerance = 64.0 * float(np.finfo(np.float64).eps)
    if step_size < minimum * (1.0 - tolerance):
        raise ValueError(
            "step_size is below the SumOfExponentials min_step"
        )
    if final_time is not None and final_time > maximum * (1.0 + tolerance):
        raise ValueError(
            "integration exceeds the SumOfExponentials t_final"
        )


# Golub-Welsch rather than scipy.special.roots_jacobi: scipy derives its weights
# from evaluations of degree-n Jacobi polynomials, which loses accuracy as the
# degree grows, while eigenvector weights stay near machine precision.
def gauss_jacobi(
    num_nodes: int,
    alpha: float,
    beta: float,
) -> tuple[FloatArray, FloatArray]:
    """Return a weighted Golub-Welsch rule."""
    if isinstance(num_nodes, bool) or not isinstance(num_nodes, int):
        raise TypeError("num_nodes must be an integer")
    if num_nodes < 1:
        raise ValueError("num_nodes must be positive")
    if not alpha > -1.0 or not beta > -1.0:
        raise ValueError("Jacobi exponents must both be greater than -1")
    if not np.isfinite(alpha) or not np.isfinite(beta):
        raise ValueError("Jacobi exponents must be finite")

    ab = alpha + beta
    diagonal = np.empty(num_nodes, dtype=np.float64)
    diagonal[0] = (beta - alpha) / (ab + 2.0)
    off_diagonal = np.empty(num_nodes - 1, dtype=np.float64)
    if num_nodes > 1:
        indices = np.arange(1, num_nodes, dtype=np.float64)
        two_indices = 2.0 * indices + ab
        diagonal[1:] = (beta * beta - alpha * alpha) / (
            two_indices * (two_indices + 2.0)
        )
        off_diagonal[:] = (2.0 / two_indices) * np.sqrt(
            indices
            * (indices + alpha)
            * (indices + beta)
            * (indices + ab)
            / ((two_indices - 1.0) * (two_indices + 1.0))
        )

    # Selected eigenvectors cap the temporary allocation at O(n) memory;
    # requesting the full matrix here would require 8 n² bytes on every rank.
    chunk_size = min(128, num_nodes)
    node_chunks = []
    first_row_chunks = []
    for start in range(0, num_nodes, chunk_size):
        stop = min(start + chunk_size, num_nodes) - 1
        selected_nodes, selected_vectors = eigh_tridiagonal(
            diagonal,
            off_diagonal,
            select="i",
            select_range=(start, stop),
            check_finite=False,
            lapack_driver="stebz",
        )
        node_chunks.append(selected_nodes)
        first_row_chunks.append(selected_vectors[0, :].copy())
    nodes = np.concatenate(node_chunks)
    first_eigenvector_rows = np.concatenate(first_row_chunks)
    # One Newton step on P_n^{(alpha,beta)} refines the nodes; the eigenvector
    # weights below are unaffected. Nodes are built once per problem.
    derivative = (0.5 * (num_nodes + ab + 1.0)) * eval_jacobi(
        num_nodes - 1, alpha + 1.0, beta + 1.0, nodes
    )
    nodes = nodes - eval_jacobi(num_nodes, alpha, beta, nodes) / derivative
    log_moment = (
        (ab + 1.0) * np.log(2.0)
        + lgamma(alpha + 1.0)
        + lgamma(beta + 1.0)
        - lgamma(ab + 2.0)
    )
    weights = exp(log_moment) * np.square(first_eigenvector_rows)
    return nodes.astype(np.float64), weights.astype(np.float64)
