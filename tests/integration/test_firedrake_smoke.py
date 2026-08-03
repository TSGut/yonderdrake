"""Small integration check for the supported Firedrake environment."""

from __future__ import annotations

import pytest

firedrake = pytest.importorskip("firedrake")


@pytest.mark.verification
def test_trivial_firedrake_solve() -> None:
    mesh = firedrake.UnitSquareMesh(2, 2)
    space = firedrake.FunctionSpace(mesh, "CG", 1)
    u = firedrake.TrialFunction(space)
    v = firedrake.TestFunction(space)
    solution = firedrake.Function(space)
    bc = firedrake.DirichletBC(space, 0.0, "on_boundary")
    a = (firedrake.inner(firedrake.grad(u), firedrake.grad(v)) + u * v) * firedrake.dx
    rhs = v * firedrake.dx
    firedrake.solve(a == rhs, solution, bcs=bc)
    assert solution.dat.data_ro.size > 0
