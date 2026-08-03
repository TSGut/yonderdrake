"""Independent high-precision diffusive coefficients generated with mpmath."""

from __future__ import annotations

from collections.abc import Sequence

import mpmath as mp


def gauss_jacobi(
    num_nodes: int,
    alpha: mp.mpf,
    beta: mp.mpf,
) -> tuple[list[mp.mpf], list[mp.mpf]]:
    """High-precision Golub-Welsch rule for verification only."""
    ab = alpha + beta
    matrix = mp.matrix(num_nodes)
    matrix[0, 0] = (beta - alpha) / (ab + 2)
    for index in range(1, num_nodes):
        k = mp.mpf(index)
        two_k = 2 * k + ab
        matrix[index, index] = (beta**2 - alpha**2) / (two_k * (two_k + 2))
        off_diagonal = (2 / two_k) * mp.sqrt(
            k
            * (k + alpha)
            * (k + beta)
            * (k + ab)
            / ((two_k - 1) * (two_k + 1))
        )
        matrix[index - 1, index] = off_diagonal
        matrix[index, index - 1] = off_diagonal
    eigenvalues, eigenvectors = mp.eigsy(matrix)
    moment = 2 ** (ab + 1) * mp.gamma(alpha + 1) * mp.gamma(
        beta + 1
    ) / mp.gamma(ab + 2)
    nodes = [eigenvalues[index] for index in range(num_nodes)]
    weights = [
        moment * eigenvectors[0, index] ** 2 for index in range(num_nodes)
    ]
    return nodes, weights


def spectrum(
    representation: str,
    alpha: float,
    num_modes: int,
    rate_scale: float = 1.0,
    dps: int = 80,
) -> tuple[list[mp.mpf], list[mp.mpf]]:
    """Generate Birk-Song or Diethelm2008 coefficients at high precision."""
    with mp.workdps(dps):
        order = mp.mpf(str(alpha))
        scale = mp.mpf(str(rate_scale))
        transformed_order = 2 * order - 1
        if representation == "Diethelm2008":
            jacobi_alpha = transformed_order
            jacobi_beta = -transformed_order
        elif representation == "BirkSong":
            jacobi_alpha = 2 * transformed_order + 1
            jacobi_beta = 1 - 2 * transformed_order
        else:
            raise ValueError(f"unknown representation: {representation}")
        nodes, quadrature_weights = gauss_jacobi(
            num_modes,
            jacobi_alpha,
            jacobi_beta,
        )
        rates = []
        weights = []
        for node, quadrature_weight in zip(
            nodes,
            quadrature_weights,
            strict=True,
        ):
            denominator = 1 + node
            ratio = (1 - node) / denominator
            if representation == "Diethelm2008":
                rate = ratio**2
                weight = (
                    4
                    * mp.sin(mp.pi * order)
                    / mp.pi
                    * quadrature_weight
                    / denominator**2
                )
            else:
                rate = ratio**4
                weight = (
                    8
                    * mp.sin(mp.pi * order)
                    / mp.pi
                    * quadrature_weight
                    / denominator**4
                )
            rates.append(scale * rate)
            weights.append(scale**order * weight)
        permutation = sorted(range(num_modes), key=lambda index: rates[index])
        return (
            [rates[index] for index in permutation],
            [weights[index] for index in permutation],
        )


def as_float(values: Sequence[mp.mpf]) -> list[float]:
    return [float(value) for value in values]
