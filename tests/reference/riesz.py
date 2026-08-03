"""Independent high-precision source-integral oracles for Riesz actions."""

from __future__ import annotations

import mpmath as mp
import numpy as np

from yonderdrake.riesz.geometry import TriangleGeometry
from yonderdrake.riesz.triangle_action import (
    AffinePolynomial,
    QuadraticPolynomial,
    SimplexPolynomial,
    riesz_normalization_2d,
)


def outside_triangle_oracle(
    geometry: TriangleGeometry,
    polynomial: SimplexPolynomial,
    point: object,
    order: float,
    *,
    dps: int = 60,
) -> mp.mpf:
    """Directly integrate the source triangle for a point outside it."""
    mp.mp.dps = dps
    vertices = [[mp.mpf(value) for value in row] for row in geometry.vertices]
    x = [mp.mpf(value) for value in np.asarray(point)]
    gradient = [mp.mpf(value) for value in polynomial.gradient]
    hessian = (
        [[mp.mpf(value) for value in row] for row in polynomial.hessian]
        if isinstance(polynomial, QuadraticPolynomial)
        else [[mp.mpf("0"), mp.mpf("0")], [mp.mpf("0"), mp.mpf("0")]]
    )
    constant = mp.mpf(polynomial.constant)
    first = [vertices[1][index] - vertices[0][index] for index in range(2)]
    second = [vertices[2][index] - vertices[0][index] for index in range(2)]
    jacobian = mp.mpf(2) * mp.mpf(geometry.area)

    def outer(u: mp.mpf) -> mp.mpf:
        def inner(v: mp.mpf) -> mp.mpf:
            y = [
                vertices[0][index] + u * first[index] + v * second[index]
                for index in range(2)
            ]
            value = constant + sum(
                gradient[index] * y[index] for index in range(2)
            )
            value += mp.mpf("0.5") * sum(
                y[row] * hessian[row][column] * y[column]
                for row in range(2)
                for column in range(2)
            )
            radius_squared = sum((y[index] - x[index]) ** 2 for index in range(2))
            return value * radius_squared ** (-1 - order)

        return mp.quad(inner, [0, 1 - u])

    integral = jacobian * mp.quad(outer, [0, 1])
    return -mp.mpf(riesz_normalization_2d(order)) * integral


def inside_triangle_polar_oracle(
    geometry: TriangleGeometry,
    polynomial: SimplexPolynomial,
    point: object,
    order: float,
    *,
    dps: int = 60,
) -> mp.mpf:
    """Evaluate the PV integral by independent polar ray integration."""
    if geometry.classify(point) != "inside":
        raise ValueError("polar oracle requires a strictly interior point")
    mp.mp.dps = dps
    x = np.asarray(point, dtype=np.float64)
    vertex_angles = sorted(
        float(np.arctan2(*(vertex - x)[::-1])) for vertex in geometry.vertices
    )
    base = vertex_angles[0]
    cuts = vertex_angles + [base + 2.0 * np.pi]
    normals = [
        [mp.mpf(value) for value in row] for row in geometry.outward_normals
    ]
    vertices = [[mp.mpf(value) for value in row] for row in geometry.vertices]
    x_mp = [mp.mpf(value) for value in x]
    point_gradient = (
        polynomial.gradient
        if isinstance(polynomial, AffinePolynomial)
        else polynomial.gradient_at(x)
    )
    gradient = [mp.mpf(value) for value in point_gradient]
    hessian = (
        [[mp.mpf(value) for value in row] for row in polynomial.hessian]
        if isinstance(polynomial, QuadraticPolynomial)
        else [[mp.mpf("0"), mp.mpf("0")], [mp.mpf("0"), mp.mpf("0")]]
    )
    value_at_point = mp.mpf(polynomial(x))
    s = mp.mpf(order)

    def integrand(theta: mp.mpf) -> mp.mpf:
        direction = [mp.cos(theta), mp.sin(theta)]
        radii = []
        for vertex, normal in zip(vertices, normals, strict=True):
            denominator = sum(
                direction[index] * normal[index] for index in range(2)
            )
            if denominator > 0:
                numerator = -sum(
                    (x_mp[index] - vertex[index]) * normal[index]
                    for index in range(2)
                )
                radii.append(numerator / denominator)
        radius = min(value for value in radii if value > 0)
        directional_gradient = sum(
            gradient[index] * direction[index] for index in range(2)
        )
        if abs(order - 0.5) < 1.0e-15:
            linear = -directional_gradient * mp.log(radius)
        else:
            linear = (
                -directional_gradient
                * radius ** (1 - 2 * s)
                / (1 - 2 * s)
            )
        exterior = value_at_point * radius ** (-2 * s) / (2 * s)
        directional_hessian = sum(
            direction[row] * hessian[row][column] * direction[column]
            for row in range(2)
            for column in range(2)
        )
        quadratic = (
            -directional_hessian
            * radius ** (2 - 2 * s)
            / (4 * (1 - s))
        )
        return linear + quadratic + exterior

    integral = mp.mpf("0")
    for left, right in zip(cuts[:-1], cuts[1:], strict=True):
        integral += mp.quad(integrand, [left, right])
    return mp.mpf(riesz_normalization_2d(order)) * integral
