"""Time-dependent sources for Caputo-Wismer propagation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

SourceRegion = Literal["volume", "boundary"]


@dataclass(frozen=True, slots=True, kw_only=True)
class CaputoWismerSource:
    """A separable spatial profile and time signal."""

    profile: Any
    signal: Any
    region: SourceRegion = "volume"
    boundary_id: Any = None

    @classmethod
    def volume(cls, profile: Any, signal: Any) -> CaputoWismerSource:
        """Create a volume forcing ``profile * signal(t)``."""
        return cls(profile=profile, signal=signal, region="volume")

    @classmethod
    def boundary(
        cls,
        profile: Any,
        signal: Any,
        *,
        boundary_id: Any = None,
    ) -> CaputoWismerSource:
        """Create incoming data for a natural or absorbing boundary."""
        return cls(
            profile=profile,
            signal=signal,
            region="boundary",
            boundary_id=boundary_id,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class CaputoWismerArraySource:
    """Independent time signals injected through a sensor-array transpose."""

    array: Any
    signals: Any


@dataclass(frozen=True, slots=True, kw_only=True)
class CaputoWismerImpedanceBoundary:
    """A first-order outgoing-wave condition on a marked boundary."""

    coefficient: Any
    boundary_id: Any = None


__all__ = [
    "CaputoWismerArraySource",
    "CaputoWismerImpedanceBoundary",
    "CaputoWismerSource",
    "SourceRegion",
]
