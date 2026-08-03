"""Initial-pressure reconstruction for Caputo-Wismer acoustics."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Literal

import numpy as np
from scipy.optimize import minimize

from yonderdrake.applications.caputo_wismer.propagation import (
    AttenuationMode,
    CaputoWismerModel,
    _nonnegative_real,
    _positive_integer,
)

ReconstructionMethod = Literal["kaltenbacher", "adjoint", "time_reversal"]


@dataclass(frozen=True, slots=True)
class CaputoWismerReconstruction:
    """Result and diagnostics from an iterative reconstruction."""

    pressure: Any
    converged: bool
    iterations: int
    objective: float
    objective_history: tuple[float, ...]
    message: str
    function_evaluations: int
    forward_seconds: float
    adjoint_seconds: float
    elapsed_seconds: float


def _sensor_data(model: CaputoWismerModel, values: Any) -> np.ndarray:
    if model.sensors is None:
        raise ValueError("the model must contain a sensor array")
    data = np.asarray(values, dtype=np.float64)
    expected = (model.num_steps + 1, model.sensors.num_sensors)
    if data.shape != expected:
        raise ValueError(f"sensor_data must have shape {expected}")
    if not np.all(np.isfinite(data)):
        raise ValueError("sensor_data must be finite")
    return data.copy()


class CaputoWismerInverseProblem:
    """Regularized initial-pressure reconstruction for one acoustic model."""

    def __init__(
        self,
        model: CaputoWismerModel,
        sensor_data: Any,
        *,
        regularization: float = 1.0e-6,
    ) -> None:
        if not isinstance(model, CaputoWismerModel):
            raise TypeError("model must be a CaputoWismerModel")
        self.model = model
        self.space = model.space
        self.sensor_data = _sensor_data(model, sensor_data)
        self.regularization = _nonnegative_real(regularization, "regularization")
        self._candidate = model._function("initial_pressure_candidate")
        self._time_weights = np.full(self.sensor_data.shape[0], model.dt)
        self._time_weights[[0, -1]] *= 0.5
        self._function_evaluations = 0
        self._forward_seconds = 0.0
        self._adjoint_seconds = 0.0

    def objective_gradient(self, candidate: Any) -> tuple[float, Any]:
        """Return the Tikhonov objective and coefficient-space gradient."""
        if candidate.function_space() != self.space:
            raise ValueError("candidate must belong to the inversion space")
        started = perf_counter()
        predicted = self.model.propagate(candidate).sensor_data
        assert predicted is not None
        residual = predicted - self.sensor_data
        self._forward_seconds += perf_counter() - started
        weighted_residual = self._time_weights[:, None] * residual
        objective = 0.5 * float(np.sum(residual * weighted_residual))
        started = perf_counter()
        gradient = self.model.adjoint_covector(weighted_residual)
        self._adjoint_seconds += perf_counter() - started
        self._function_evaluations += 1
        if self.regularization:
            objective += (
                0.5
                * self.regularization
                * float(
                    self.model._fd.assemble(candidate * candidate * self.model._fd.dx)
                )
            )
            self.model._matrix_axpy(
                self.model._l2_mass,
                candidate,
                gradient,
                self.regularization,
            )
        return objective, gradient

    def _scaled_adjoint_initial_guess(self, *, positivity: bool) -> Any:
        weighted_data = self._time_weights[:, None] * self.sensor_data
        started = perf_counter()
        direction = self.model.adjoint(weighted_data)
        self._adjoint_seconds += perf_counter() - started
        if positivity:
            direction.dat.data[:] = np.maximum(direction.dat.data_ro, 0.0)
        started = perf_counter()
        traces = self.model.propagate(direction).sensor_data
        assert traces is not None
        self._forward_seconds += perf_counter() - started
        numerator = float(
            np.sum(self.sensor_data * self._time_weights[:, None] * traces)
        )
        denominator = float(np.sum(traces * self._time_weights[:, None] * traces))
        if self.regularization:
            denominator += self.regularization * float(
                self.model._fd.assemble(direction * direction * self.model._fd.dx)
            )
        scale = max(0.0, numerator / denominator) if denominator > 0.0 else 0.0
        direction *= scale
        return direction

    def _solve_distributed(
        self,
        *,
        max_iterations: int,
        tolerance: float,
        positivity: bool,
        started: float,
    ) -> CaputoWismerReconstruction:
        from petsc4py import PETSc

        history: list[float] = []
        solution = self._candidate.copy(deepcopy=True)
        gradient_buffer = self.model._covector("tao_objective_gradient")
        tao = PETSc.TAO().create(comm=self.space.mesh().comm)
        tao.setType(PETSc.TAO.Type.BLMVM if positivity else PETSc.TAO.Type.LMVM)
        tao.setTolerances(gatol=tolerance, grtol=tolerance, gttol=tolerance)
        tao.setMaximumIterations(max_iterations)

        def objective_gradient(
            _tao: Any,
            coefficients: Any,
            output_gradient: Any,
        ) -> float:
            with self._candidate.dat.vec as candidate_vector:
                coefficients.copy(candidate_vector)
            value, gradient = self.objective_gradient(self._candidate)
            with gradient.dat.vec_ro as gradient_vector:
                gradient_vector.copy(output_gradient)
            history.append(value)
            return value

        lower = self.model._function("tao_lower_bound")
        upper = self.model._function("tao_upper_bound")
        lower.assign(0.0)
        upper.assign(PETSc.INFINITY)
        with (
            solution.dat.vec as solution_vector,
            gradient_buffer.dat.vec as gradient_vector,
            lower.dat.vec_ro as lower_vector,
            upper.dat.vec_ro as upper_vector,
        ):
            tao.setObjectiveGradient(objective_gradient, gradient_vector)
            if positivity:
                tao.setVariableBounds((lower_vector, upper_vector))
            tao.solve(solution_vector)

        iterations, objective, _, _, _, reason = tao.getSolutionStatus()
        result = CaputoWismerReconstruction(
            pressure=solution.copy(deepcopy=True),
            converged=int(reason) > 0,
            iterations=int(iterations),
            objective=float(objective),
            objective_history=tuple(history),
            message=str(reason),
            function_evaluations=self._function_evaluations,
            forward_seconds=self._forward_seconds,
            adjoint_seconds=self._adjoint_seconds,
            elapsed_seconds=perf_counter() - started,
        )
        result.pressure.rename("kaltenbacher_initial_pressure")
        tao.destroy()
        return result

    def solve(
        self,
        *,
        initial_guess: Any = None,
        max_iterations: int = 100,
        tolerance: float = 1.0e-5,
        positivity: bool = True,
        warm_start: bool = True,
    ) -> CaputoWismerReconstruction:
        """Minimize the regularized sensor-data misfit."""
        _positive_integer(max_iterations, "max_iterations")
        tolerance_value = _nonnegative_real(tolerance, "tolerance")
        if tolerance_value == 0.0:
            raise ValueError("tolerance must be positive")
        self._function_evaluations = 0
        self._forward_seconds = 0.0
        self._adjoint_seconds = 0.0
        started = perf_counter()
        if initial_guess is not None:
            if initial_guess.function_space() != self.space:
                raise ValueError("initial_guess must belong to the inversion space")
            self._candidate.assign(initial_guess)
        elif warm_start:
            self._candidate.assign(
                self._scaled_adjoint_initial_guess(positivity=positivity)
            )
        else:
            self._candidate.assign(0.0)
        if self.space.mesh().comm.size > 1:
            return self._solve_distributed(
                max_iterations=max_iterations,
                tolerance=tolerance_value,
                positivity=positivity,
                started=started,
            )

        initial_values = np.asarray(
            self._candidate.dat.data_ro, dtype=np.float64
        ).copy()
        history: list[float] = []

        def objective(values: np.ndarray) -> tuple[float, np.ndarray]:
            self._candidate.dat.data[:] = values
            value, gradient = self.objective_gradient(self._candidate)
            history.append(value)
            return value, np.asarray(gradient.dat.data_ro, dtype=np.float64).copy()

        result = minimize(
            objective,
            initial_values,
            method="L-BFGS-B",
            jac=True,
            bounds=[(0.0, None)] * initial_values.size if positivity else None,
            options={
                "ftol": tolerance_value,
                "gtol": tolerance_value,
                "maxiter": max_iterations,
            },
        )
        pressure = self.model._function("kaltenbacher_initial_pressure")
        pressure.dat.data[:] = result.x
        return CaputoWismerReconstruction(
            pressure=pressure,
            converged=bool(result.success),
            iterations=int(result.nit),
            objective=float(result.fun),
            objective_history=tuple(history),
            message=str(result.message),
            function_evaluations=self._function_evaluations,
            forward_seconds=self._forward_seconds,
            adjoint_seconds=self._adjoint_seconds,
            elapsed_seconds=perf_counter() - started,
        )


def _default_filter_length(model: CaputoWismerModel) -> float:
    from mpi4py import MPI

    fd = model._fd
    mesh = model.space.mesh()
    cell_space = fd.FunctionSpace(mesh, "DG", 0)
    diameters = fd.Function(cell_space).interpolate(fd.CellDiameter(mesh))
    local_maximum = float(np.max(diameters.dat.data_ro))
    return 1.5 * float(mesh.comm.allreduce(local_maximum, op=MPI.MAX))


def time_reverse_sensor_data(
    model: CaputoWismerModel,
    sensor_data: Any,
    *,
    compensate_attenuation: bool = True,
    filter_length: float | None = None,
    filter_order: int = 2,
    positivity: bool = False,
) -> Any:
    """Backpropagate sensor traces through a lossless or compensated model."""
    values = _sensor_data(model, sensor_data)
    assert model.sensors is not None
    attenuation: AttenuationMode
    if compensate_attenuation:
        length = (
            _default_filter_length(model) if filter_length is None else filter_length
        )
        attenuation = "reversed"
    else:
        if filter_length is not None:
            raise ValueError(
                "filter_length is only used when attenuation is compensated"
            )
        length = None
        attenuation = "none"
    reverse_model = CaputoWismerModel(
        model.space,
        materials=model.materials,
        dt=model.dt,
        num_steps=model.num_steps,
        sensors=model.sensors,
        boundaries=model.boundaries,
        pml=model.pml,
        attenuation=attenuation,
        attenuation_filter_length=length,
        attenuation_filter_order=filter_order,
        num_modes=model.num_modes,
        representation=model.representation,
        stiffness_theta=model.stiffness_theta,
        # The filtered reverse system need not share the forward solver's symmetry.
        solver_parameters=None,
    )
    pressure = reverse_model.adjoint(model.dt * values)
    pressure.rename("time_reversed_initial_pressure")
    del reverse_model
    communicator = model.space.mesh().comm
    if communicator.size > 1:
        import gc

        from petsc4py import PETSc

        gc.collect()
        PETSc.garbage_cleanup(communicator)
    if positivity:
        pressure.dat.data[:] = np.maximum(pressure.dat.data_ro, 0.0)
    return pressure


def reconstruct_initial_pressure(
    model: CaputoWismerModel,
    sensor_data: Any,
    *,
    method: ReconstructionMethod = "kaltenbacher",
    regularization: float = 1.0e-6,
    initial_guess: Any = None,
    max_iterations: int = 100,
    tolerance: float = 1.0e-5,
    positivity: bool = True,
    warm_start: bool = True,
    compensate_attenuation: bool = True,
    filter_length: float | None = None,
    filter_order: int = 2,
) -> Any:
    """Reconstruct initial pressure with the selected method."""
    values = _sensor_data(model, sensor_data)
    if method == "adjoint":
        weights = np.full(values.shape[0], model.dt)
        weights[[0, -1]] *= 0.5
        return model.adjoint(weights[:, None] * values)
    if method == "time_reversal":
        return time_reverse_sensor_data(
            model,
            values,
            compensate_attenuation=compensate_attenuation,
            filter_length=filter_length,
            filter_order=filter_order,
            positivity=positivity,
        )
    if method != "kaltenbacher":
        raise ValueError("method must be 'kaltenbacher', 'adjoint', or 'time_reversal'")
    problem = CaputoWismerInverseProblem(
        model,
        values,
        regularization=regularization,
    )
    result = problem.solve(
        initial_guess=initial_guess,
        max_iterations=max_iterations,
        tolerance=tolerance,
        positivity=positivity,
        warm_start=warm_start,
    )
    if result.iterations < max_iterations and not result.converged:
        raise RuntimeError(
            f"initial-pressure reconstruction did not converge: {result.message}"
        )
    return result.pressure


__all__ = [
    "CaputoWismerInverseProblem",
    "CaputoWismerReconstruction",
    "ReconstructionMethod",
    "reconstruct_initial_pressure",
    "time_reverse_sensor_data",
]
