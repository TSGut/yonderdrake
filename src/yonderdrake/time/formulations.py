"""Public choices for fractional time formulation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Recurrence:
    """Eliminated constant-memory recurrence formulation."""

    interpolant: str = "quadratic"

    def __post_init__(self) -> None:
        if self.interpolant not in {"linear", "quadratic"}:
            raise ValueError("interpolant must be 'linear' or 'quadratic'")

    def describe(self) -> dict[str, str]:
        """Return deterministic checkpoint metadata."""
        return {"kind": "recurrence", "interpolant": self.interpolant}


@dataclass(frozen=True, slots=True)
class Oscillator:
    """Exact two-field rotation for sine diffusive memory modes."""


@dataclass(frozen=True, slots=True)
class AuxiliaryODE:
    """Auxiliary-ODE formulation solving the field and its modes together.

    Where `Recurrence` eliminates the memory modes from the field solve, this
    couples them into one monolithic system on ``V^(m+1)``, which is what makes
    the modes reachable from PETSc.
    """

    scheme: str = "backward_euler"

    def __post_init__(self) -> None:
        if self.scheme not in {"backward_euler", "trapezoidal"}:
            raise ValueError(
                "scheme must be 'backward_euler' or 'trapezoidal'"
            )
