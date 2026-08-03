"""Caputo-Wismer wave stepping."""

from __future__ import annotations

from collections.abc import Sequence
from math import isfinite
from typing import Any

from yonderdrake.applications.caputo_wismer.model import CaputoWismerMaterial
from yonderdrake.applications.caputo_wismer.sources import (
    CaputoWismerImpedanceBoundary,
)
from yonderdrake.time.marker import CaputoDerivative
from yonderdrake.time.representations import BirkSong, SineDiffusive
from yonderdrake.time.stepper import FractionalTimeStepper


class CaputoWismerStepper:
    """Advance a heterogeneous Caputo-Wismer wave equation."""

    def __init__(
        self,
        u: Any,
        t: Any,
        dt: Any,
        *,
        materials: Sequence[CaputoWismerMaterial],
        representation: Any = None,
        num_modes: int = 32,
        volume_source: Any = None,
        boundary_source: Any = None,
        boundaries: Sequence[CaputoWismerImpedanceBoundary] = (),
        initial_velocity: Any = 0.0,
        bcs: Any = None,
        stiffness_theta: float = 1.0,
        solver_parameters: Any = None,
        appctx: Any = None,
    ) -> None:
        try:
            import firedrake as fd
        except ImportError as error:
            raise RuntimeError(
                "CaputoWismerStepper requires an active Firedrake environment"
            ) from error
        material_values = tuple(materials)
        if not material_values:
            raise ValueError("materials must contain at least one material")
        if any(
            not isinstance(material, CaputoWismerMaterial)
            for material in material_values
        ):
            raise TypeError("materials must contain CaputoWismerMaterial objects")
        if u.ufl_shape != ():
            raise NotImplementedError("CaputoWismerStepper supports scalar fields only")
        space = u.function_space()
        if space.ufl_element().family() != "Lagrange":
            raise NotImplementedError(
                "CaputoWismerStepper supports continuous Lagrange spaces only"
            )
        if (
            not isinstance(num_modes, int)
            or isinstance(num_modes, bool)
            or num_modes < 1
        ):
            raise ValueError("num_modes must be a positive integer")
        try:
            theta = float(stiffness_theta)
        except (TypeError, ValueError) as error:
            raise TypeError("stiffness_theta must be a real scalar") from error
        if not isfinite(theta) or not 0.0 <= theta <= 1.0:
            raise ValueError("stiffness_theta must be between 0 and 1")
        boundary_values = tuple(boundaries)
        if any(
            not isinstance(boundary, CaputoWismerImpedanceBoundary)
            for boundary in boundary_values
        ):
            raise TypeError(
                "boundaries must contain CaputoWismerImpedanceBoundary objects"
            )

        self._fd = fd
        self.u = u
        self.t = t
        self.dt = dt
        self.materials = material_values
        self.representation = (
            BirkSong(num_modes) if representation is None else representation
        )
        if isinstance(self.representation, SineDiffusive):
            raise NotImplementedError(
                "SineDiffusive is available through the time steppers only"
            )
        self.previous = u.copy(deepcopy=True)
        self.older = u.copy(deepcopy=True)
        self.older -= dt * initial_velocity
        self.volume_source = (
            fd.Function(space, name="caputo_wismer_volume_source")
            if volume_source is None
            else volume_source
        )
        self.boundary_source = (
            fd.Function(space, name="caputo_wismer_boundary_source")
            if boundary_source is None
            else boundary_source
        )
        test = fd.TestFunction(space)
        acceleration = (u - 2.0 * self.previous + self.older) / dt**2
        stiffness_field = theta * u + (1.0 - theta) * self.previous
        mass_coefficient = sum(
            (
                material.indicator / (material.density * material.wave_speed**2)
                for material in material_values
            ),
            0,
        )
        stiffness = sum(
            (
                fd.inner(
                    material.indicator / material.density * fd.grad(stiffness_field),
                    fd.grad(test),
                )
                * fd.dx
                for material in material_values
            ),
            0,
        )
        memory = sum(
            (
                fd.inner(
                    material.indicator
                    * material.damping
                    / material.density
                    * fd.grad(CaputoDerivative(u, material.alpha)),
                    fd.grad(test),
                )
                * fd.dx
                for material in material_values
            ),
            0,
        )
        boundary = sum(
            (
                condition.coefficient
                * fd.inner((u - self.previous) / dt, test)
                * (
                    fd.ds
                    if condition.boundary_id is None
                    else fd.ds(condition.boundary_id)
                )
                for condition in boundary_values
            ),
            0,
        )
        residual = (
            mass_coefficient * fd.inner(acceleration, test) * fd.dx
            + stiffness
            + memory
            + boundary
            - fd.inner(self.volume_source, test) * fd.dx
            - fd.inner(self.boundary_source, test) * fd.ds
        )
        self.residual = residual
        self._stepper = FractionalTimeStepper(
            residual,
            self.representation,
            t,
            dt,
            u,
            bcs=bcs,
            solver_parameters=solver_parameters,
            appctx=appctx,
        )

    @property
    def fractional_stepper(self) -> Any:
        return self._stepper

    def advance(self) -> None:
        """Advance one step and commit the two physical wave histories."""
        self._stepper.advance()
        self.older.assign(self.previous)
        self.previous.assign(self.u)

    def reset(
        self,
        u0: Any,
        *,
        initial_velocity: Any = 0.0,
        t0: Any = None,
    ) -> None:
        """Reset the wave field, velocity, and fractional memory."""
        self.u.assign(u0)
        self.previous.assign(self.u)
        self.older.assign(self.u)
        self.older -= self.dt * initial_velocity
        self._stepper.reset(self.u, t0=t0)

    def solver_stats(self) -> dict[str, Any]:
        return dict(self._stepper.solver_stats())
