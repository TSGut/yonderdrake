"""Shared cleanup for Firedrake integration tests."""

from __future__ import annotations

import gc

import pytest


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Destroy cyclic PETSc objects collectively before MPI finalization."""
    del session, exitstatus
    try:
        from petsc4py import PETSc
    except ImportError:
        return
    if PETSc.COMM_WORLD.size > 1:
        gc.collect()
        PETSc.garbage_cleanup(PETSc.COMM_WORLD)
