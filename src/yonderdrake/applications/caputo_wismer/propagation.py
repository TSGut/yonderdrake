"""Forward and adjoint Caputo-Wismer acoustic propagation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite
from typing import Any, Literal

import numpy as np

from yonderdrake.applications.caputo_wismer.model import (
    CaputoWismerMaterial,
    SensorArray,
)
from yonderdrake.applications.caputo_wismer.pml import CaputoWismerPML
from yonderdrake.applications.caputo_wismer.sources import (
    CaputoWismerArraySource,
    CaputoWismerImpedanceBoundary,
    CaputoWismerSource,
)
from yonderdrake.time.coefficients import recurrence_coefficients
from yonderdrake.time.representations import BirkSong, SineDiffusive

AttenuationMode = Literal["dissipative", "none", "reversed"]


def _nonnegative_real(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a real scalar") from error
    if not isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def _positive_integer(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _solver_parameters(parameters: Any, *, parallel: bool) -> dict[str, Any]:
    if parameters is not None:
        return dict(parameters)
    if parallel:
        return {
            "snes_type": "ksponly",
            "ksp_type": "gmres",
            "ksp_rtol": 1.0e-9,
            "ksp_max_it": 500,
            "ksp_error_if_not_converged": True,
            "pc_type": "bjacobi",
            "sub_pc_type": "ilu",
        }
    return {
        "snes_type": "ksponly",
        "ksp_type": "preonly",
        "pc_type": "lu",
    }


def _material_model(
    materials: Sequence[CaputoWismerMaterial],
    attenuation: AttenuationMode,
) -> tuple[CaputoWismerMaterial, ...]:
    values = tuple(materials)
    if not values:
        raise ValueError("materials must contain at least one material")
    if any(not isinstance(value, CaputoWismerMaterial) for value in values):
        raise TypeError("materials must contain CaputoWismerMaterial objects")
    if attenuation not in {"dissipative", "none", "reversed"}:
        raise ValueError("attenuation must be 'dissipative', 'none', or 'reversed'")
    sign = -1.0 if attenuation == "reversed" else 1.0
    if attenuation == "none":
        sign = 0.0
    return tuple(
        CaputoWismerMaterial(
            indicator=value.indicator,
            density=value.density,
            wave_speed=value.wave_speed,
            damping=sign * value.damping,
            alpha=value.alpha,
        )
        for value in values
    )


@dataclass(frozen=True, slots=True)
class CaputoWismerPropagation:
    """Fields and observations returned by one propagation."""

    final_pressure: Any
    sensor_data: np.ndarray | None
    field_history: tuple[Any, ...]


@dataclass(frozen=True, slots=True)
class _PreparedSource:
    load: Any
    signal: np.ndarray


@dataclass(frozen=True, slots=True)
class _PreparedArraySource:
    array: SensorArray
    signals: np.ndarray


class CaputoWismerModel:
    """A conservative acoustic model and its exact discrete adjoint."""

    def __init__(
        self,
        space: Any,
        *,
        materials: Sequence[CaputoWismerMaterial],
        dt: float,
        num_steps: int,
        sensors: SensorArray | None = None,
        sources: Sequence[CaputoWismerSource | CaputoWismerArraySource] = (),
        boundaries: Sequence[CaputoWismerImpedanceBoundary] = (),
        pml: CaputoWismerPML | None = None,
        attenuation: AttenuationMode = "dissipative",
        attenuation_filter_length: float | None = None,
        attenuation_filter_order: int = 2,
        num_modes: int = 32,
        representation: Any = None,
        stiffness_theta: float | None = None,
        solver_parameters: Any = None,
    ) -> None:
        try:
            import firedrake as fd
        except ImportError as error:
            raise RuntimeError(
                "CaputoWismerModel requires an active Firedrake environment"
            ) from error
        if space.ufl_element().family() != "Lagrange":
            raise NotImplementedError(
                "CaputoWismerModel supports continuous Lagrange spaces only"
            )
        if space.value_shape != ():
            raise NotImplementedError(
                "CaputoWismerModel supports scalar pressure fields only"
            )
        dimension = int(space.mesh().geometric_dimension)
        if dimension not in {2, 3}:
            raise NotImplementedError("CaputoWismerModel supports 2D or 3D meshes")
        step_size = _nonnegative_real(dt, "dt")
        if step_size == 0.0:
            raise ValueError("dt must be positive")
        _positive_integer(num_steps, "num_steps")
        _positive_integer(num_modes, "num_modes")
        try:
            theta = (
                1.0
                if stiffness_theta is None and pml is not None
                else 0.0
                if stiffness_theta is None
                else float(stiffness_theta)
            )
        except (TypeError, ValueError) as error:
            raise TypeError("stiffness_theta must be a real scalar or None") from error
        if not isfinite(theta) or not 0.0 <= theta <= 1.0:
            raise ValueError("stiffness_theta must be between 0 and 1")
        if sensors is not None and sensors.space != space:
            raise ValueError("sensors must belong to the model space")
        if pml is not None:
            if not isinstance(pml, CaputoWismerPML):
                raise TypeError("pml must be a CaputoWismerPML")
            if len(pml.damping) != dimension:
                raise ValueError(
                    f"pml.damping must contain {dimension} directional fields"
                )

        filter_length = None
        if attenuation_filter_length is not None:
            filter_length = _nonnegative_real(
                attenuation_filter_length,
                "attenuation_filter_length",
            )
            if filter_length == 0.0:
                raise ValueError("attenuation_filter_length must be positive")
        _positive_integer(attenuation_filter_order, "attenuation_filter_order")
        if attenuation == "reversed" and filter_length is None:
            raise ValueError("reversed attenuation requires attenuation_filter_length")
        if attenuation != "reversed" and filter_length is not None:
            raise ValueError(
                "attenuation_filter_length is only used with reversed attenuation"
            )

        self._fd = fd
        self.space = space
        self.dimension = dimension
        self.dt = step_size
        self.num_steps = num_steps
        self.num_modes = num_modes
        self.sensors = sensors
        self.pml = pml
        self.boundaries = tuple(boundaries)
        if any(
            not isinstance(boundary, CaputoWismerImpedanceBoundary)
            for boundary in self.boundaries
        ):
            raise TypeError(
                "boundaries must contain CaputoWismerImpedanceBoundary objects"
            )
        self.attenuation = attenuation
        self.attenuation_filter_length = filter_length
        self.attenuation_filter_order = attenuation_filter_order
        self.stiffness_theta = theta
        self.sources = tuple(sources)
        self.representation = (
            BirkSong(num_modes) if representation is None else representation
        )
        if isinstance(self.representation, SineDiffusive):
            raise NotImplementedError(
                "SineDiffusive is available through the time steppers only"
            )
        self.materials = tuple(materials)
        self._effective_materials = _material_model(
            self.materials,
            attenuation,
        )
        self._parameters = _solver_parameters(
            solver_parameters,
            parallel=space.mesh().comm.size > 1,
        )
        self.solver_parameters = dict(self._parameters)

        trial = fd.TrialFunction(space)
        test = fd.TestFunction(space)
        domain_zero = fd.Function(
            fd.FunctionSpace(space.mesh(), "DG", 0),
            name="caputo_wismer_domain_zero",
        )
        self._domain_zero = domain_zero
        self._inverse_density = sum(
            (
                material.indicator / material.density
                for material in self._effective_materials
            ),
            0,
        )
        self._mass_coefficient = sum(
            (
                material.indicator / (material.density * material.wave_speed**2)
                for material in self._effective_materials
            ),
            0,
        )
        wave_mass_form = self._mass_coefficient * fd.inner(trial, test) * fd.dx
        l2_mass_form = fd.inner(trial, test) * fd.dx
        stiffness_form = (
            self._inverse_density * fd.inner(fd.grad(trial), fd.grad(test)) * fd.dx
        )
        damping_forms = tuple(
            material.indicator
            * (material.damping + domain_zero)
            / material.density
            * fd.inner(fd.grad(trial), fd.grad(test))
            * fd.dx
            for material in self._effective_materials
        )
        boundary_forms = tuple(
            (boundary.coefficient + domain_zero)
            * fd.inner(trial, test)
            * (fd.ds if boundary.boundary_id is None else fd.ds(boundary.boundary_id))
            for boundary in self.boundaries
        )
        boundary_form = (
            None if not boundary_forms else sum(boundary_forms[1:], boundary_forms[0])
        )

        self._wave_mass = fd.assemble(wave_mass_form, mat_type="aij")
        self._l2_mass = fd.assemble(l2_mass_form, mat_type="aij")
        self._stiffness = fd.assemble(stiffness_form, mat_type="aij")
        self._damping = tuple(
            fd.assemble(form, mat_type="aij") for form in damping_forms
        )
        self._boundary = (
            None
            if boundary_form is None
            else fd.assemble(boundary_form, mat_type="aij")
        )
        self._l2_mass_solver = fd.LinearSolver(
            self._l2_mass,
            solver_parameters=_solver_parameters(
                None,
                parallel=space.mesh().comm.size > 1,
            ),
        )

        spectra = tuple(
            self.representation.spectrum(float(material.alpha))
            for material in self._effective_materials
        )
        recurrence = []
        implicit_weights = []
        for spectrum in spectra:
            decay, interpolation, implicit_weight = recurrence_coefficients(
                spectrum,
                step_size,
                final_time=num_steps * step_size,
            )
            recurrence.append((decay, interpolation, spectrum.weights))
            implicit_weights.append(implicit_weight)
        self._recurrence = tuple(recurrence)
        self._implicit_weights = tuple(implicit_weights)

        self._filter_left = None
        self._filter_solver = None
        self._filter_component_count = 0
        if filter_length is not None:
            self._filter_component_count = attenuation_filter_order
            filter_form = (
                l2_mass_form
                + filter_length**2 * fd.inner(fd.grad(trial), fd.grad(test)) * fd.dx
            )
            self._filter_left = fd.assemble(filter_form, mat_type="aij")
            self._filter_solver = fd.LinearSolver(
                self._filter_left,
                solver_parameters=self._parameters,
            )

        self._mixed_space = None
        self._outer_bc = None
        self._pressure_outer_bc = None
        self._outer_nodes = None
        if pml is None:
            if self._filter_component_count:
                self._build_filtered_scalar_system()
            else:
                left_form = (1.0 / step_size**2) * wave_mass_form
                left_form += theta * stiffness_form
                left_form += sum(
                    (
                        weight * form
                        for weight, form in zip(
                            self._implicit_weights,
                            damping_forms,
                            strict=True,
                        )
                    ),
                    0,
                )
                if boundary_form is not None:
                    left_form += (1.0 / step_size) * boundary_form
                self._left = fd.assemble(left_form, mat_type="aij")
        else:
            self._build_pml_system()
        self._left_solver = fd.LinearSolver(
            self._left,
            solver_parameters=self._parameters,
        )
        self._sources, self._array_sources = self._prepare_sources(self.sources)

    def _add_filter_chain(
        self,
        form: Any,
        trials: tuple[Any, ...],
        tests: tuple[Any, ...],
        *,
        start: int,
        input_field: Any,
    ) -> tuple[Any, Any]:
        fd = self._fd
        assert self.attenuation_filter_length is not None
        previous = input_field
        for offset in range(self._filter_component_count):
            filtered = trials[start + offset]
            filtered_test = tests[start + offset]
            form += fd.inner(filtered, filtered_test) * fd.dx
            form += (
                self.attenuation_filter_length**2
                * fd.inner(fd.grad(filtered), fd.grad(filtered_test))
                * fd.dx
            )
            form -= fd.inner(previous, filtered_test) * fd.dx
            previous = filtered
        return form, previous

    def _build_filtered_scalar_system(self) -> None:
        fd = self._fd
        component_count = 1 + self._filter_component_count
        mixed_space = self.space
        for _ in range(component_count - 1):
            mixed_space *= self.space
        self._mixed_space = mixed_space
        trials = fd.TrialFunctions(mixed_space)
        tests = fd.TestFunctions(mixed_space)
        pressure = trials[0]
        pressure_test = tests[0]
        form: Any = 0
        form, filtered_pressure = self._add_filter_chain(
            form,
            trials,
            tests,
            start=1,
            input_field=pressure,
        )
        form += (
            (1.0 / self.dt**2)
            * self._mass_coefficient
            * fd.inner(pressure, pressure_test)
            * fd.dx
        )
        if self.stiffness_theta:
            form += (
                self.stiffness_theta
                * self._inverse_density
                * fd.inner(fd.grad(pressure), fd.grad(pressure_test))
                * fd.dx
            )
        for material, weight in zip(
            self._effective_materials,
            self._implicit_weights,
            strict=True,
        ):
            form += (
                weight
                * material.indicator
                * (material.damping + self._domain_zero)
                / material.density
                * fd.inner(fd.grad(filtered_pressure), fd.grad(pressure_test))
                * fd.dx
            )
        for boundary in self.boundaries:
            form += (
                (1.0 / self.dt)
                * boundary.coefficient
                * fd.inner(pressure, pressure_test)
                * (
                    fd.ds
                    if boundary.boundary_id is None
                    else fd.ds(boundary.boundary_id)
                )
            )
        self._left = fd.assemble(form, mat_type="aij")

    def _build_pml_system(self) -> None:
        fd = self._fd
        assert self.pml is not None
        include_integral = self.dimension == 3
        physical_component_count = 2 + self.dimension + int(include_integral)
        component_count = physical_component_count + self._filter_component_count
        mixed_space = self.space
        for _ in range(component_count - 1):
            mixed_space *= self.space
        self._mixed_space = mixed_space
        trials = fd.TrialFunctions(mixed_space)
        tests = fd.TestFunctions(mixed_space)
        pressure = trials[0]
        pressure_test = tests[0]
        velocity = trials[1]
        velocity_test = tests[1]
        auxiliaries = trials[2 : 2 + self.dimension]
        auxiliary_tests = tests[2 : 2 + self.dimension]
        integral_index = 2 + self.dimension
        integral = trials[integral_index] if include_integral else None
        integral_test = tests[integral_index] if include_integral else None
        sigma = self.pml.damping
        sigma_sum = sum(sigma)
        sigma_pair_sum = sum(
            sigma[first] * sigma[second]
            for first in range(self.dimension)
            for second in range(first + 1, self.dimension)
        )
        sigma_product = sigma[0] * sigma[1] * sigma[2] if include_integral else 0
        form = (1.0 / self.dt) * fd.inner(pressure, pressure_test) * fd.dx
        form -= fd.inner(velocity, pressure_test) * fd.dx
        form += (
            (1.0 / self.dt)
            * self._mass_coefficient
            * fd.inner(velocity, velocity_test)
            * fd.dx
        )
        form += (
            sigma_sum
            * self._mass_coefficient
            * fd.inner(velocity, velocity_test)
            * fd.dx
        )
        form += (
            sigma_pair_sum
            * self._mass_coefficient
            * fd.inner(pressure, velocity_test)
            * fd.dx
        )
        if self.stiffness_theta:
            form += (
                self.stiffness_theta
                * self._inverse_density
                * fd.inner(fd.grad(pressure), fd.grad(velocity_test))
                * fd.dx
            )
        attenuation_pressure = pressure
        if self._filter_component_count:
            form, attenuation_pressure = self._add_filter_chain(
                form,
                trials,
                tests,
                start=physical_component_count,
                input_field=pressure,
            )
        for material, weight in zip(
            self._effective_materials,
            self._implicit_weights,
            strict=True,
        ):
            form += (
                weight
                * material.indicator
                * (material.damping + self._domain_zero)
                / material.density
                * fd.inner(fd.grad(attenuation_pressure), fd.grad(velocity_test))
                * fd.dx
            )
        for boundary in self.boundaries:
            form += (
                boundary.coefficient
                * fd.inner(velocity, velocity_test)
                * (
                    fd.ds
                    if boundary.boundary_id is None
                    else fd.ds(boundary.boundary_id)
                )
            )
        for axis, auxiliary in enumerate(auxiliaries):
            form += auxiliary * velocity_test.dx(axis) * fd.dx
        if include_integral:
            assert integral is not None
            form += (
                sigma_product
                * self._mass_coefficient
                * fd.inner(integral, velocity_test)
                * fd.dx
            )

        for axis, (auxiliary, auxiliary_test) in enumerate(
            zip(auxiliaries, auxiliary_tests, strict=True)
        ):
            others = [sigma[index] for index in range(self.dimension) if index != axis]
            b_coefficient = sigma[axis] - sum(others)
            c_coefficient = others[0] * others[1] if include_integral else 0
            form += (1.0 / self.dt) * fd.inner(auxiliary, auxiliary_test) * fd.dx
            form += sigma[axis] * fd.inner(auxiliary, auxiliary_test) * fd.dx
            form += (
                self._inverse_density
                * b_coefficient
                * pressure.dx(axis)
                * auxiliary_test
                * fd.dx
            )
            if include_integral:
                assert integral is not None
                form -= (
                    self._inverse_density
                    * c_coefficient
                    * integral.dx(axis)
                    * auxiliary_test
                    * fd.dx
                )
        if include_integral:
            assert integral is not None and integral_test is not None
            form += (1.0 / self.dt) * fd.inner(integral, integral_test) * fd.dx
            form -= fd.inner(pressure, integral_test) * fd.dx

        bcs = None
        if self.pml.outer_boundary:
            self._outer_bc = fd.DirichletBC(mixed_space.sub(0), 0.0, "on_boundary")
            pressure_outer_bc = fd.DirichletBC(self.space, 0.0, "on_boundary")
            self._pressure_outer_bc = pressure_outer_bc
            self._outer_nodes = pressure_outer_bc.nodes
            bcs = self._outer_bc
        self._left = fd.assemble(form, bcs=bcs, mat_type="aij")

    def _prepare_sources(
        self,
        sources: Sequence[CaputoWismerSource | CaputoWismerArraySource],
    ) -> tuple[tuple[_PreparedSource, ...], tuple[_PreparedArraySource, ...]]:
        fd = self._fd
        test = fd.TestFunction(self.space)
        prepared = []
        arrays = []
        for source in sources:
            if isinstance(source, CaputoWismerSource):
                signal = np.asarray(source.signal, dtype=np.float64)
                if signal.shape != (self.num_steps + 1,):
                    raise ValueError(
                        f"source signal must have shape ({self.num_steps + 1},)"
                    )
                if not np.all(np.isfinite(signal)):
                    raise ValueError("source signal must be finite")
                if source.region == "volume":
                    if source.boundary_id is not None:
                        raise ValueError(
                            "boundary_id is only valid for a boundary source"
                        )
                    measure = fd.dx
                elif source.region == "boundary":
                    measure = (
                        fd.ds
                        if source.boundary_id is None
                        else fd.ds(source.boundary_id)
                    )
                else:
                    raise ValueError("source region must be 'volume' or 'boundary'")
                load = fd.assemble(fd.inner(source.profile, test) * measure)
                prepared.append(_PreparedSource(load=load, signal=signal.copy()))
                continue
            if isinstance(source, CaputoWismerArraySource):
                if source.array.space != self.space:
                    raise ValueError("array source must belong to the model space")
                signals = np.asarray(source.signals, dtype=np.float64)
                shape = (self.num_steps + 1, source.array.num_sensors)
                if signals.shape != shape:
                    raise ValueError(f"array source signals must have shape {shape}")
                if not np.all(np.isfinite(signals)):
                    raise ValueError("array source signals must be finite")
                arrays.append(
                    _PreparedArraySource(
                        array=source.array,
                        signals=signals.copy(),
                    )
                )
                continue
            raise TypeError(
                "sources must contain CaputoWismerSource or "
                "CaputoWismerArraySource objects"
            )
        return tuple(prepared), tuple(arrays)

    def _function(self, name: str) -> Any:
        return self._fd.Function(self.space, name=name)

    def _covector(self, name: str) -> Any:
        return self._fd.Cofunction(self.space.dual(), name=name)

    @staticmethod
    def _axpy(target: Any, scale: float, source: Any) -> None:
        with target.dat.vec as target_vector, source.dat.vec_ro as source_vector:
            target_vector.axpy(float(scale), source_vector)

    @staticmethod
    def _matrix_axpy(
        matrix: Any,
        source: Any,
        target: Any,
        scale: float,
        *,
        transpose: bool = False,
        work: Any = None,
    ) -> None:
        temporary = target.copy(deepcopy=True) if work is None else work
        temporary.assign(0.0)
        with source.dat.vec_ro as source_vector, temporary.dat.vec as temporary_vector:
            if transpose:
                matrix.petscmat.multTranspose(source_vector, temporary_vector)
            else:
                matrix.petscmat.mult(source_vector, temporary_vector)
        CaputoWismerModel._axpy(target, scale, temporary)

    def _filter_primal(self, source: Any, *, out: Any = None) -> Any:
        if self._filter_solver is None:
            if out is None:
                return source
            out.assign(source)
            return out
        current = source
        result = self._function("attenuation_filtered_state") if out is None else out
        right_hand_side = self._covector("attenuation_filter_rhs")
        for _ in range(self.attenuation_filter_order):
            right_hand_side.assign(0.0)
            self._matrix_axpy(self._l2_mass, current, right_hand_side, 1.0)
            self._filter_solver.solve(result, right_hand_side)
            current = result
        return result

    def _filter_transpose(self, source: Any, *, out: Any = None) -> Any:
        if self._filter_solver is None:
            if out is None:
                return source
            out.assign(source)
            return out
        current = source
        result = self._covector("attenuation_filter_transpose") if out is None else out
        solution = self._function("attenuation_filter_transpose_solution")
        for _ in range(self.attenuation_filter_order):
            with current.dat.vec_ro as source_vector, solution.dat.vec as target_vector:
                target_vector.set(0.0)
                self._filter_solver.ksp.solveTranspose(source_vector, target_vector)
            result.assign(0.0)
            self._matrix_axpy(self._l2_mass, solution, result, 1.0, transpose=True)
            current = result
        return result

    def _new_pml_state(self) -> tuple[tuple[Any, ...], Any | None]:
        if self.pml is None:
            return (), None
        auxiliaries = tuple(
            self._function(f"pml_auxiliary_{axis}") for axis in range(self.dimension)
        )
        integral = (
            self._function("pml_pressure_integral") if self.dimension == 3 else None
        )
        return auxiliaries, integral

    def _source_rhs(self, step: int, target: Any) -> None:
        for source in self._sources:
            self._axpy(target, source.signal[step], source.load)
        for array_source in self._array_sources:
            load = array_source.array.adjoint_covector(array_source.signals[step])
            self._axpy(target, 1.0, load)

    def _pressure_rhs(
        self,
        current: Any,
        previous: Any,
        modes: tuple[np.ndarray, ...],
        step: int,
    ) -> Any:
        rhs = self._covector("caputo_wismer_pressure_rhs")
        work = self._covector("caputo_wismer_matrix_work")
        material_state = self._function("caputo_wismer_material_state")
        filtered_state = self._function("caputo_wismer_filtered_material_state")
        inverse_dt_squared = 1.0 / self.dt**2
        self._matrix_axpy(
            self._wave_mass,
            current,
            rhs,
            2.0 * inverse_dt_squared,
            work=work,
        )
        self._matrix_axpy(
            self._wave_mass,
            previous,
            rhs,
            -inverse_dt_squared,
            work=work,
        )
        self._matrix_axpy(
            self._stiffness,
            current,
            rhs,
            -(1.0 - self.stiffness_theta),
            work=work,
        )
        if self._boundary is not None:
            self._matrix_axpy(
                self._boundary,
                current,
                rhs,
                1.0 / self.dt,
                work=work,
            )
        for damping, mode_values, coefficients, implicit_weight in zip(
            self._damping,
            modes,
            self._recurrence,
            self._implicit_weights,
            strict=True,
        ):
            decay, _, weights = coefficients
            np.matmul(
                -(decay * weights),
                mode_values,
                out=material_state.dat.data,
            )
            material_state.dat.data[:] += implicit_weight * current.dat.data_ro
            action_state = self._filter_primal(material_state, out=filtered_state)
            self._matrix_axpy(damping, action_state, rhs, 1.0, work=work)
        self._source_rhs(step, rhs)
        return rhs

    def _solve_scalar_step(self, pressure_rhs: Any) -> Any:
        if self._filter_component_count:
            assert self._mixed_space is not None
            rhs = self._fd.Cofunction(
                self._mixed_space.dual(), name="caputo_wismer_filtered_rhs"
            )
            rhs.subfunctions[0].assign(pressure_rhs)
            solution = self._fd.Function(
                self._mixed_space, name="caputo_wismer_filtered_solution"
            )
            self._left_solver.solve(solution, rhs)
            pressure = self._function("caputo_wismer_next_pressure")
            pressure.assign(solution.subfunctions[0])
            return pressure
        pressure = self._function("caputo_wismer_next_pressure")
        self._left_solver.solve(pressure, pressure_rhs)
        return pressure

    def _solve_pml_step(
        self,
        current: Any,
        velocity: Any,
        modes: tuple[np.ndarray, ...],
        auxiliaries: tuple[Any, ...],
        integral: Any | None,
        step: int,
    ) -> tuple[Any, Any, tuple[Any, ...], Any | None]:
        assert self.pml is not None
        assert self._mixed_space is not None
        rhs = self._fd.Cofunction(
            self._mixed_space.dual(), name="caputo_wismer_mixed_rhs"
        )
        work = self._covector("caputo_wismer_pml_rhs_work")
        pressure_block = rhs.subfunctions[0]
        pressure_block.assign(0.0)
        self._matrix_axpy(
            self._l2_mass,
            current,
            pressure_block,
            1.0 / self.dt,
            work=work,
        )
        velocity_block = rhs.subfunctions[1]
        velocity_block.assign(0.0)
        self._matrix_axpy(
            self._wave_mass,
            velocity,
            velocity_block,
            1.0 / self.dt,
            work=work,
        )
        self._matrix_axpy(
            self._stiffness,
            current,
            velocity_block,
            -(1.0 - self.stiffness_theta),
            work=work,
        )
        material_state = self._function("caputo_wismer_pml_material_state")
        filtered_state = self._function("caputo_wismer_pml_filtered_material_state")
        for damping, mode_values, coefficients, implicit_weight in zip(
            self._damping,
            modes,
            self._recurrence,
            self._implicit_weights,
            strict=True,
        ):
            decay, _, weights = coefficients
            np.matmul(
                -(decay * weights),
                mode_values,
                out=material_state.dat.data,
            )
            material_state.dat.data[:] += implicit_weight * current.dat.data_ro
            action_state = self._filter_primal(
                material_state,
                out=filtered_state,
            )
            self._matrix_axpy(
                damping,
                action_state,
                velocity_block,
                1.0,
                work=work,
            )
        self._source_rhs(step, velocity_block)
        for axis, auxiliary in enumerate(auxiliaries):
            block = rhs.subfunctions[2 + axis]
            block.assign(0.0)
            self._matrix_axpy(
                self._l2_mass,
                auxiliary,
                block,
                1.0 / self.dt,
                work=work,
            )
        if integral is not None:
            block = rhs.subfunctions[2 + self.dimension]
            block.assign(0.0)
            self._matrix_axpy(
                self._l2_mass,
                integral,
                block,
                1.0 / self.dt,
                work=work,
            )
        if self._outer_nodes is not None:
            rhs.subfunctions[0].dat.data_with_halos[self._outer_nodes] = 0.0
        solution = self._fd.Function(
            self._mixed_space, name="caputo_wismer_mixed_solution"
        )
        self._left_solver.solve(solution, rhs)
        pressure = self._function("caputo_wismer_next_pressure")
        pressure.assign(solution.subfunctions[0])
        next_velocity = self._function("caputo_wismer_next_velocity")
        next_velocity.assign(solution.subfunctions[1])
        next_auxiliaries = tuple(
            self._function(f"pml_auxiliary_{axis}_next")
            for axis in range(self.dimension)
        )
        for target, source in zip(
            next_auxiliaries,
            solution.subfunctions[2 : 2 + self.dimension],
            strict=True,
        ):
            target.assign(source)
        next_integral = None
        if integral is not None:
            next_integral = self._function("pml_pressure_integral_next")
            next_integral.assign(solution.subfunctions[2 + self.dimension])
        return pressure, next_velocity, next_auxiliaries, next_integral

    def propagate(
        self,
        initial_pressure: Any,
        *,
        record_history: bool = False,
    ) -> CaputoWismerPropagation:
        """Advance an initial pressure and configured time-dependent sources."""
        if initial_pressure.function_space() != self.space:
            raise ValueError("initial_pressure must belong to the model space")
        current = initial_pressure.copy(deepcopy=True)
        if self._pressure_outer_bc is not None:
            self._pressure_outer_bc.apply(current)
        previous = current.copy(deepcopy=True)
        velocity = self._function("caputo_wismer_velocity")
        modes = tuple(
            np.zeros((decay.size, current.dat.data_ro.size), dtype=np.float64)
            for decay, _, _ in self._recurrence
        )
        auxiliaries, integral = self._new_pml_state()
        sensor_data = (
            None
            if self.sensors is None
            else np.empty(
                (self.num_steps + 1, self.sensors.num_sensors),
                dtype=np.float64,
            )
        )
        if sensor_data is not None:
            assert self.sensors is not None
            sensor_data[0] = self.sensors.sample(current)
        history = [current.copy(deepcopy=True)] if record_history else []
        increment = self._function("caputo_wismer_pressure_increment")
        for step in range(self.num_steps):
            if self.pml is None:
                pressure_rhs = self._pressure_rhs(
                    current,
                    previous,
                    modes,
                    step + 1,
                )
                next_pressure = self._solve_scalar_step(pressure_rhs)
                next_velocity = velocity
                next_auxiliaries = ()
                next_integral = None
            else:
                (
                    next_pressure,
                    next_velocity,
                    next_auxiliaries,
                    next_integral,
                ) = self._solve_pml_step(
                    current,
                    velocity,
                    modes,
                    auxiliaries,
                    integral,
                    step + 1,
                )
            increment.assign(next_pressure)
            increment -= current
            for mode_values, coefficients in zip(modes, self._recurrence, strict=True):
                decay, interpolation, _ = coefficients
                mode_values *= decay[:, None]
                mode_values += interpolation[:, None] * increment.dat.data_ro[None, :]
            previous.assign(current)
            current.assign(next_pressure)
            velocity.assign(next_velocity)
            auxiliaries = next_auxiliaries
            integral = next_integral
            if sensor_data is not None:
                assert self.sensors is not None
                sensor_data[step + 1] = self.sensors.sample(current)
            if record_history:
                history.append(current.copy(deepcopy=True))
        current.rename("caputo_wismer_final_pressure")
        return CaputoWismerPropagation(
            final_pressure=current,
            sensor_data=sensor_data,
            field_history=tuple(history),
        )

    def _solve_step_transpose(
        self,
        pressure: Any,
    ) -> Any:
        if self._filter_component_count:
            assert self._mixed_space is not None
            rhs = self._fd.Cofunction(
                self._mixed_space.dual(), name="caputo_wismer_filtered_adjoint_rhs"
            )
            rhs.subfunctions[0].assign(pressure)
            solution = self._fd.Function(
                self._mixed_space,
                name="caputo_wismer_filtered_adjoint_multiplier",
            )
            with rhs.dat.vec_ro as source_vector, solution.dat.vec as target_vector:
                target_vector.set(0.0)
                self._left_solver.ksp.solveTranspose(source_vector, target_vector)
            result = self._function("caputo_wismer_adjoint_multiplier")
            result.assign(solution.subfunctions[0])
            return result
        result = self._function("caputo_wismer_adjoint_multiplier")
        with pressure.dat.vec_ro as source_vector, result.dat.vec as target_vector:
            target_vector.set(0.0)
            self._left_solver.ksp.solveTranspose(source_vector, target_vector)
        return result

    def _solve_pml_step_transpose(
        self,
        pressure: Any,
        velocity: Any,
        auxiliaries: tuple[Any, ...],
        integral: Any | None,
    ) -> tuple[Any, Any, tuple[Any, ...], Any | None]:
        assert self.pml is not None
        assert self._mixed_space is not None
        rhs = self._fd.Cofunction(
            self._mixed_space.dual(), name="caputo_wismer_mixed_adjoint_rhs"
        )
        rhs.subfunctions[0].assign(pressure)
        rhs.subfunctions[1].assign(velocity)
        for target, source in zip(
            rhs.subfunctions[2 : 2 + self.dimension],
            auxiliaries,
            strict=True,
        ):
            target.assign(source)
        if integral is not None:
            rhs.subfunctions[2 + self.dimension].assign(integral)
        if self._outer_nodes is not None:
            rhs.subfunctions[0].dat.data_with_halos[self._outer_nodes] = 0.0
        solution = self._fd.Function(
            self._mixed_space, name="caputo_wismer_mixed_adjoint_multiplier"
        )
        with rhs.dat.vec_ro as source_vector, solution.dat.vec as target_vector:
            target_vector.set(0.0)
            self._left_solver.ksp.solveTranspose(source_vector, target_vector)
        pressure_result = self._function("caputo_wismer_adjoint_multiplier")
        pressure_result.assign(solution.subfunctions[0])
        if self._pressure_outer_bc is not None:
            self._pressure_outer_bc.apply(pressure_result)
        velocity_result = self._function("pml_velocity_adjoint_multiplier")
        velocity_result.assign(solution.subfunctions[1])
        auxiliary_results = tuple(
            self._function(f"pml_adjoint_multiplier_{axis}")
            for axis in range(self.dimension)
        )
        for target, source in zip(
            auxiliary_results,
            solution.subfunctions[2 : 2 + self.dimension],
            strict=True,
        ):
            target.assign(source)
        integral_result = None
        if integral is not None:
            integral_result = self._function("pml_integral_adjoint_multiplier")
            integral_result.assign(solution.subfunctions[2 + self.dimension])
        return pressure_result, velocity_result, auxiliary_results, integral_result

    def _adjoint_covector_pml(self, values: np.ndarray) -> Any:
        assert self.sensors is not None
        adjoint_pressure = self.sensors.adjoint_covector(values[-1])
        adjoint_velocity = self._covector("adjoint_pml_velocity")
        adjoint_auxiliaries = tuple(
            self._covector(f"adjoint_pml_auxiliary_{axis}")
            for axis in range(self.dimension)
        )
        adjoint_integral = (
            self._covector("adjoint_pml_integral") if self.dimension == 3 else None
        )
        adjoint_modes = tuple(
            np.zeros((decay.size, adjoint_pressure.dat.data_ro.size), dtype=np.float64)
            for decay, _, _ in self._recurrence
        )
        mode_contribution = np.empty(
            adjoint_pressure.dat.data_ro.size,
            dtype=np.float64,
        )
        matrix_work = self._covector("adjoint_pml_matrix_work")
        damping_action = self._covector("adjoint_pml_damping_action")
        filtered_action = self._covector("adjoint_pml_filtered_damping_action")
        observation = self._covector("adjoint_pml_sensor_observation")

        for step in range(self.num_steps - 1, -1, -1):
            pressure_contribution = self._covector("adjoint_pml_pressure_input")
            velocity_contribution = self._covector("adjoint_pml_velocity_input")
            for mode_values, coefficients in zip(
                adjoint_modes,
                self._recurrence,
                strict=True,
            ):
                decay, interpolation, _ = coefficients
                np.matmul(
                    interpolation,
                    mode_values,
                    out=mode_contribution,
                )
                adjoint_pressure.dat.data[:] += mode_contribution
                pressure_contribution.dat.data[:] -= mode_contribution
                mode_values *= decay[:, None]

            (
                pressure_multiplier,
                velocity_multiplier,
                auxiliary_multipliers,
                integral_multiplier,
            ) = self._solve_pml_step_transpose(
                adjoint_pressure,
                adjoint_velocity,
                adjoint_auxiliaries,
                adjoint_integral,
            )
            self._matrix_axpy(
                self._l2_mass,
                pressure_multiplier,
                pressure_contribution,
                1.0 / self.dt,
                transpose=True,
                work=matrix_work,
            )
            self._matrix_axpy(
                self._wave_mass,
                velocity_multiplier,
                velocity_contribution,
                1.0 / self.dt,
                transpose=True,
                work=matrix_work,
            )
            self._matrix_axpy(
                self._stiffness,
                velocity_multiplier,
                pressure_contribution,
                -(1.0 - self.stiffness_theta),
                transpose=True,
                work=matrix_work,
            )
            for damping, mode_values, coefficients, implicit_weight in zip(
                self._damping,
                adjoint_modes,
                self._recurrence,
                self._implicit_weights,
                strict=True,
            ):
                decay, _, weights = coefficients
                damping_action.assign(0.0)
                self._matrix_axpy(
                    damping,
                    velocity_multiplier,
                    damping_action,
                    1.0,
                    transpose=True,
                    work=matrix_work,
                )
                material_action = self._filter_transpose(
                    damping_action,
                    out=filtered_action,
                )
                self._axpy(
                    pressure_contribution,
                    implicit_weight,
                    material_action,
                )
                mode_values -= (decay * weights)[:, None] * material_action.dat.data_ro[
                    None, :
                ]

            next_auxiliary_adjoint = []
            for auxiliary_multiplier in auxiliary_multipliers:
                contribution = self._covector("adjoint_pml_auxiliary_input")
                self._matrix_axpy(
                    self._l2_mass,
                    auxiliary_multiplier,
                    contribution,
                    1.0 / self.dt,
                    transpose=True,
                    work=matrix_work,
                )
                next_auxiliary_adjoint.append(contribution)
            next_integral_adjoint = None
            if integral_multiplier is not None:
                next_integral_adjoint = self._covector("adjoint_pml_integral_input")
                self._matrix_axpy(
                    self._l2_mass,
                    integral_multiplier,
                    next_integral_adjoint,
                    1.0 / self.dt,
                    transpose=True,
                    work=matrix_work,
                )
            self.sensors.adjoint_covector(values[step], out=observation)
            self._axpy(pressure_contribution, 1.0, observation)
            adjoint_pressure = pressure_contribution
            adjoint_velocity = velocity_contribution
            adjoint_auxiliaries = tuple(next_auxiliary_adjoint)
            adjoint_integral = next_integral_adjoint

        if self._outer_nodes is not None:
            adjoint_pressure.dat.data_with_halos[self._outer_nodes] = 0.0
        adjoint_pressure.rename("initial_pressure_adjoint")
        return adjoint_pressure

    def adjoint_covector(self, sensor_values: Any) -> Any:
        """Apply the exact transpose of the initial-pressure observation map."""
        if self.sensors is None:
            raise ValueError("the model has no sensor array")
        values = np.asarray(sensor_values, dtype=np.float64)
        expected_shape = (self.num_steps + 1, self.sensors.num_sensors)
        if values.shape != expected_shape:
            raise ValueError(f"sensor_values must have shape {expected_shape}")
        if not np.all(np.isfinite(values)):
            raise ValueError("sensor_values must be finite")
        if self.pml is not None:
            return self._adjoint_covector_pml(values)

        adjoint_current = self.sensors.adjoint_covector(values[-1])
        adjoint_previous = self._covector("adjoint_previous_pressure")
        adjoint_modes = tuple(
            np.zeros((decay.size, adjoint_current.dat.data_ro.size), dtype=np.float64)
            for decay, _, _ in self._recurrence
        )
        inverse_dt_squared = 1.0 / self.dt**2
        mode_contribution = np.empty(adjoint_current.dat.data_ro.size, dtype=np.float64)
        matrix_work = self._covector("adjoint_matrix_work")
        damping_action = self._covector("adjoint_damping_action")
        filtered_action = self._covector("adjoint_filtered_damping_action")
        observation = self._covector("adjoint_sensor_observation")

        for step in range(self.num_steps - 1, -1, -1):
            current_contribution = self._covector("adjoint_current_contribution")
            current_contribution.assign(adjoint_previous)
            previous_contribution = self._covector("adjoint_previous_contribution")
            for mode_values, coefficients in zip(
                adjoint_modes, self._recurrence, strict=True
            ):
                _, interpolation, _ = coefficients
                np.matmul(interpolation, mode_values, out=mode_contribution)
                adjoint_current.dat.data[:] += mode_contribution
                current_contribution.dat.data[:] -= mode_contribution

            multiplier = self._solve_step_transpose(adjoint_current)
            self._matrix_axpy(
                self._wave_mass,
                multiplier,
                current_contribution,
                2.0 * inverse_dt_squared,
                transpose=True,
                work=matrix_work,
            )
            self._matrix_axpy(
                self._wave_mass,
                multiplier,
                previous_contribution,
                -inverse_dt_squared,
                transpose=True,
                work=matrix_work,
            )
            self._matrix_axpy(
                self._stiffness,
                multiplier,
                current_contribution,
                -(1.0 - self.stiffness_theta),
                transpose=True,
                work=matrix_work,
            )
            if self._boundary is not None:
                self._matrix_axpy(
                    self._boundary,
                    multiplier,
                    current_contribution,
                    1.0 / self.dt,
                    transpose=True,
                    work=matrix_work,
                )
            for damping, mode_values, coefficients, implicit_weight in zip(
                self._damping,
                adjoint_modes,
                self._recurrence,
                self._implicit_weights,
                strict=True,
            ):
                decay, _, weights = coefficients
                damping_action.assign(0.0)
                self._matrix_axpy(
                    damping,
                    multiplier,
                    damping_action,
                    1.0,
                    transpose=True,
                    work=matrix_work,
                )
                material_action = self._filter_transpose(
                    damping_action,
                    out=filtered_action,
                )
                self._axpy(
                    current_contribution,
                    implicit_weight,
                    material_action,
                )
                mode_values *= decay[:, None]
                mode_values -= (decay * weights)[:, None] * material_action.dat.data_ro[
                    None, :
                ]

            self.sensors.adjoint_covector(values[step], out=observation)
            self._axpy(current_contribution, 1.0, observation)
            adjoint_current = current_contribution
            adjoint_previous = previous_contribution

        result = self._covector("initial_pressure_adjoint")
        result.assign(adjoint_current)
        self._axpy(result, 1.0, adjoint_previous)
        return result

    def adjoint(self, sensor_values: Any) -> Any:
        """Return the spatial L2 representative of the discrete adjoint."""
        covector = self.adjoint_covector(sensor_values)
        result = self._function("initial_pressure_adjoint")
        self._l2_mass_solver.solve(result, covector)
        return result


__all__ = [
    "AttenuationMode",
    "CaputoWismerModel",
    "CaputoWismerPropagation",
]
