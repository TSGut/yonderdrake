"""Sinc quadrature for the Balakrishnan positive fractional power."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from math import ceil, exp, isfinite, log, pi

import numpy as np


@dataclass(frozen=True)
class SincQuadrature:
    """Nodes for an exponentially transformed Balakrishnan integral."""

    order: float
    truncation_target: float
    effective_truncation_target: float
    step: float
    indices: np.ndarray
    log_shifts: np.ndarray
    weights: np.ndarray
    estimated_model_error: float

    @property
    def num_nodes(self) -> int:
        return int(self.indices.size)


def positive_power_sinc(
    order: float,
    truncation_target: float,
) -> SincQuadrature:
    """Build the logarithmic sinc rule for a positive fractional power."""
    order = float(order)
    truncation_target = float(truncation_target)
    if not isfinite(order) or not 0.0 < order < 1.0:
        raise ValueError("order must satisfy 0 < order < 1")
    if (
        not isfinite(truncation_target)
        or not 0.0 < truncation_target < 1.0
    ):
        raise ValueError(
            "truncation_target must satisfy 0 < truncation_target < 1"
        )

    requested_target = truncation_target
    meaningful_target = np.finfo(np.float64).eps
    if truncation_target < meaningful_target:
        warnings.warn(
            "truncation_target is below meaningful float64 precision; "
            f"using {meaningful_target:.3e}",
            RuntimeWarning,
            stacklevel=2,
        )
        truncation_target = meaningful_target

    # Reserve margin in the scalar tail model.
    target = log(10.0 / truncation_target)
    step = 2.0 * pi * pi / target
    negative_count = max(1, ceil(target / (order * step)))
    positive_count = max(1, ceil(target / ((1.0 - order) * step)))
    num_nodes = negative_count + positive_count + 1
    if num_nodes > 100_000:
        raise ValueError(
            "requested order and truncation target require more than 100000 "
            "sinc nodes; loosen the target or keep s farther from 0 and 1"
        )
    indices = np.arange(
        -negative_count,
        positive_count + 1,
        dtype=np.int64,
    )
    log_shifts = step * indices.astype(np.float64)
    prefactor = np.sin(pi * order) / pi
    # Fold exp(-y) into positive weights to avoid overflow.
    weights = np.empty_like(log_shifts)
    negative = log_shifts <= 0.0
    weights[negative] = (
        prefactor * step * np.exp(order * log_shifts[negative])
    )
    weights[~negative] = (
        prefactor * step * np.exp((order - 1.0) * log_shifts[~negative])
    )
    estimate = (
        exp(-order * negative_count * step)
        + exp(-(1.0 - order) * positive_count * step)
        + exp(-2.0 * pi * pi / step)
    )
    indices.setflags(write=False)
    log_shifts.setflags(write=False)
    weights.setflags(write=False)
    return SincQuadrature(
        order=order,
        truncation_target=requested_target,
        effective_truncation_target=truncation_target,
        step=step,
        indices=indices,
        log_shifts=log_shifts,
        weights=weights,
        estimated_model_error=estimate,
    )
