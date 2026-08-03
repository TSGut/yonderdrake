"""Fixed spatial operators outside fractional time markers."""

from __future__ import annotations

from math import gamma

import numpy as np
import pytest

fd = pytest.importorskip("firedrake")

from yonderdrake import (  # noqa: E402
    AuxiliaryODE,
    CaputoDerivative,
    Diethelm2008,
    FractionalTimeStepper,
    Recurrence,
    RiemannLiouvilleDerivative,
)


def implicit_mode_weight(
    representation,
    alpha: float,
    step_size: float,
    formulation,
) -> float:
    """Return the first-step coefficient multiplying ``u_new - u_old``."""
    spectrum = representation.spectrum(alpha)
    arguments = spectrum.rates * step_size
    if isinstance(formulation, Recurrence):
        interpolation = -np.expm1(-arguments) / arguments
    elif formulation.scheme == "backward_euler":
        interpolation = 1.0 / (1.0 + arguments)
    else:
        interpolation = 1.0 / (1.0 + 0.5 * arguments)
    return float(np.dot(spectrum.weights, interpolation))


@pytest.mark.verification
@pytest.mark.parametrize(
    "marker,has_initial_trace",
    [
        (CaputoDerivative, False),
        (RiemannLiouvilleDerivative, True),
    ],
)
@pytest.mark.parametrize(
    "formulation",
    [
        Recurrence(),
        AuxiliaryODE(scheme="backward_euler"),
        AuxiliaryODE(scheme="trapezoidal"),
    ],
)
def test_fixed_laplacian_outside_fractional_marker(
    marker,
    has_initial_trace: bool,
    formulation,
) -> None:
    """The Wismer weak form transforms before Firedrake differentiates it."""
    alpha = 0.58
    step_size = 0.04
    damping = 0.35
    reaction = 0.8
    representation = Diethelm2008(10)
    mesh = fd.UnitIntervalMesh(2)
    space = fd.FunctionSpace(mesh, "CG", 1)
    x = fd.SpatialCoordinate(mesh)[0]
    u = fd.Function(space, name="u").interpolate(fd.sin(fd.pi * x))
    boundary = fd.DirichletBC(space, 0.0, "on_boundary")
    boundary.apply(u)
    initial = u.copy(deepcopy=True)
    test = fd.TestFunction(space)
    time = fd.Constant(0.0)
    dt = fd.Constant(step_size)

    fractional_value = marker(u, alpha)
    residual = (
        damping
        * fd.inner(fd.grad(fractional_value), fd.grad(test))
        * fd.dx
        + reaction * fd.inner(u, test) * fd.dx
    )
    stepper = FractionalTimeStepper(
        residual,
        representation,
        time,
        dt,
        u,
        formulation=formulation,
        bcs=boundary,
        solver_parameters={"ksp_type": "preonly", "pc_type": "lu"},
    )
    stepper.advance()

    stiffness = float(
        fd.assemble(fd.inner(fd.grad(initial), fd.grad(initial)) * fd.dx)
    )
    mass = float(fd.assemble(fd.inner(initial, initial) * fd.dx))
    implicit = implicit_mode_weight(
        representation,
        alpha,
        step_size,
        formulation,
    )
    initial_trace = (
        step_size ** (-alpha) / gamma(1.0 - alpha)
        if has_initial_trace
        else 0.0
    )
    expected_amplitude = (
        damping * stiffness * (implicit - initial_trace)
        / (damping * stiffness * implicit + reaction * mass)
    )
    expected = initial.copy(deepcopy=True)
    expected *= expected_amplitude

    assert fd.norm(u - expected) < 2.0e-11
    assert np.allclose(u.dat.data_ro[boundary.nodes], 0.0)
