"""Real Appell ``F1`` values needed by tetrahedral edge formulas.

This is not a general Appell-function implementation.  It only intends to cover

``F1(a; b1, b2; a + 1; x, y)``, with ``a > 0`` and real ``x, y <= 0``.

In this region Euler's integral is positive and has no poles. For difficult, very
large negative arguments, composite Gauss integration after ``t=exp(u)``
tries to resolve the thin layer at ``t=0`` without arbitrary precision arithmetic.
"""

from __future__ import annotations

from functools import cache
from math import isfinite

import numpy as np
from scipy.integrate import quad
from scipy.special import hyp2f1, roots_jacobi, roots_legendre

_DEFAULT_ORDERS = (16, 32, 64, 128)
_ADAPTIVE_ARGUMENT_THRESHOLD = 256.0
_CHUNK_SIZE = 4096
_LOG_CHUNK_SIZE = 256
_LOG_PANEL_WIDTH = 2.0
_LOG_RULE_ORDER = 16


@cache
def _euler_rule(a: float, order: int) -> tuple[np.ndarray, np.ndarray]:
    """Return nodes and normalized weights for ``a*t**(a-1)`` on ``[0, 1]``."""
    nodes, weights = roots_jacobi(order, 0.0, a - 1.0)
    nodes = 0.5 * (nodes + 1.0)
    weights = a * 2.0 ** (-a) * weights
    nodes.setflags(write=False)
    weights.setflags(write=False)
    return nodes, weights


def _fixed_euler_many(
    a: float,
    b1: float,
    b2: float,
    x: np.ndarray,
    y: np.ndarray,
    order: int,
) -> np.ndarray:
    nodes, weights = _euler_rule(a, order)
    result = np.empty(x.size, dtype=np.float64)
    for start in range(0, x.size, _CHUNK_SIZE):
        stop = min(start + _CHUNK_SIZE, x.size)
        x_chunk = x[start:stop, None]
        y_chunk = y[start:stop, None]
        logarithm = -b1 * np.log1p(-x_chunk * nodes[None, :]) - b2 * np.log1p(
            -y_chunk * nodes[None, :]
        )
        result[start:stop] = np.exp(logarithm) @ weights
    return result


@cache
def _unit_legendre(order: int) -> tuple[np.ndarray, np.ndarray]:
    nodes, weights = roots_legendre(order)
    nodes = 0.5 * (nodes + 1.0)
    weights = 0.5 * weights
    nodes.setflags(write=False)
    weights.setflags(write=False)
    return nodes, weights


def _log_euler_many(
    a: float,
    b1: float,
    b2: float,
    x: np.ndarray,
    y: np.ndarray,
    *,
    rtol: float,
) -> np.ndarray:
    """Resolve large-argument endpoint layers in a shifted log coordinate."""
    largest_argument = np.maximum(-x, -y)
    upper_limits = np.log1p(largest_argument)
    tail_span = max(32.0, -np.log(rtol) / a + 4.0)
    panel_counts = np.ceil((tail_span + upper_limits) / _LOG_PANEL_WIDTH).astype(
        np.int64
    )
    nodes, weights = _unit_legendre(_LOG_RULE_ORDER)
    result = np.empty(x.size, dtype=np.float64)

    for panel_count in np.unique(panel_counts):
        indices = np.flatnonzero(panel_counts == panel_count)
        locations = (
            np.arange(panel_count, dtype=np.float64)[:, None] + nodes[None, :]
        ).reshape(-1)
        composite_weights = np.tile(weights, panel_count)
        for start in range(0, indices.size, _LOG_CHUNK_SIZE):
            chunk = indices[start : start + _LOG_CHUNK_SIZE]
            widths = (tail_span + upper_limits[chunk]) / panel_count
            log_t = (
                -tail_span
                + widths[:, None] * locations[None, :]
                - upper_limits[chunk, None]
            )
            t = np.exp(log_t)
            logarithm = (
                np.log(a)
                + a * log_t
                - b1 * np.log1p(-x[chunk, None] * t)
                - b2 * np.log1p(-y[chunk, None] * t)
            )
            result[chunk] = widths * (np.exp(logarithm) @ composite_weights)
    return result


def _adaptive_euler(
    a: float,
    b1: float,
    b2: float,
    x: float,
    y: float,
    *,
    rtol: float,
    atol: float,
) -> float:
    largest_argument = max(-x, -y)
    tail_span = max(32.0, -np.log(rtol) / a + 4.0)
    lower_limit = -np.log1p(largest_argument) - tail_span

    def transformed_integrand(u: float) -> float:
        t = np.exp(u)
        logarithm = np.log(a) + a * u - b1 * np.log1p(-x * t) - b2 * np.log1p(-y * t)
        return float(np.exp(logarithm))

    value, _ = quad(
        transformed_integrand,
        lower_limit,
        0.0,
        epsabs=atol,
        epsrel=rtol,
        limit=200,
    )
    return float(value)


def appell_f1_a_plus_one(
    a: float,
    b1: float,
    b2: float,
    x: float | np.ndarray,
    y: float | np.ndarray,
    *,
    rtol: float = 2.0e-13,
    atol: float = 0.0,
) -> float | np.ndarray:
    r"""Evaluate ``F1(a; b1, b2; a+1; x, y)`` for real ``x,y <= 0``.

    ``x`` and ``y`` follow NumPy broadcasting rules.  The returned value is a
    Python ``float`` for scalar inputs and an array otherwise.
    """
    a = float(a)
    b1 = float(b1)
    b2 = float(b2)
    if not all(isfinite(value) for value in (a, b1, b2, rtol, atol)):
        raise ValueError("Appell parameters and tolerances must be finite")
    if a <= 0.0:
        raise ValueError("a must be positive")
    if rtol <= 0.0 or atol < 0.0:
        raise ValueError("rtol must be positive and atol nonnegative")
    if atol == 0.0 and rtol < 50.0 * np.finfo(np.float64).eps:
        raise ValueError("rtol is too small when atol is zero")

    x_array, y_array = np.broadcast_arrays(
        np.asarray(x, dtype=np.float64),
        np.asarray(y, dtype=np.float64),
    )
    scalar = x_array.ndim == 0
    shape = x_array.shape
    x_flat = x_array.reshape(-1)
    y_flat = y_array.reshape(-1)
    if not np.all(np.isfinite(x_flat)) or not np.all(np.isfinite(y_flat)):
        raise ValueError("x and y must be finite")
    if bool(np.any(x_flat > 0.0)) or bool(np.any(y_flat > 0.0)):
        raise ValueError("this evaluator requires x <= 0 and y <= 0")

    result = np.empty(x_flat.size, dtype=np.float64)
    unresolved = np.ones(x_flat.size, dtype=bool)

    equal = x_flat == y_flat
    if bool(np.any(equal)):
        result[equal] = hyp2f1(a, b1 + b2, a + 1.0, x_flat[equal])
        unresolved[equal] = False

    x_zero = unresolved & (x_flat == 0.0)
    if bool(np.any(x_zero)):
        result[x_zero] = hyp2f1(a, b2, a + 1.0, y_flat[x_zero])
        unresolved[x_zero] = False

    y_zero = unresolved & (y_flat == 0.0)
    if bool(np.any(y_zero)):
        result[y_zero] = hyp2f1(a, b1, a + 1.0, x_flat[y_zero])
        unresolved[y_zero] = False

    difficult = unresolved & (
        np.maximum(-x_flat, -y_flat) > _ADAPTIVE_ARGUMENT_THRESHOLD
    )
    regular = unresolved & ~difficult
    if bool(np.any(regular)):
        indices = np.flatnonzero(regular)
        previous = _fixed_euler_many(
            a,
            b1,
            b2,
            x_flat[indices],
            y_flat[indices],
            _DEFAULT_ORDERS[0],
        )
        for order in _DEFAULT_ORDERS[1:]:
            current = _fixed_euler_many(
                a,
                b1,
                b2,
                x_flat[indices],
                y_flat[indices],
                order,
            )
            converged = np.abs(current - previous) <= atol + rtol * np.abs(current)
            if bool(np.any(converged)):
                result[indices[converged]] = current[converged]
            remaining = ~converged
            if not bool(np.any(remaining)):
                indices = indices[:0]
                break
            indices = indices[remaining]
            previous = current[remaining]
        difficult[indices] = True

    difficult_indices = np.flatnonzero(difficult)
    if difficult_indices.size:
        difficult_values = _log_euler_many(
            a,
            b1,
            b2,
            x_flat[difficult_indices],
            y_flat[difficult_indices],
            rtol=rtol,
        )
        result[difficult_indices] = difficult_values
        for local_index in np.flatnonzero(~np.isfinite(difficult_values)):
            index = difficult_indices[local_index]
            result[index] = _adaptive_euler(
                a,
                b1,
                b2,
                float(x_flat[index]),
                float(y_flat[index]),
                rtol=rtol,
                atol=atol,
            )

    reshaped = result.reshape(shape)
    return float(reshaped) if scalar else reshaped
