# API and supported scope

Signatures and hard limitations. The mathematics behind each entry is in
{ref}`mathematics-and-methods`. The compact overview is the
{ref}`method map <method-map>`.

The {doc}`api-reference` gives the complete callable signatures and public
methods generated from the installed package.

## Fractional time derivatives

```python
CaputoDerivative(u, alpha)
RiemannLiouvilleDerivative(u, alpha)

BirkSong(num_modes, *, rate_scale=1.0)
Diethelm2008(num_modes, *, rate_scale=1.0)
Cayley(num_modes, *, power=None, t_final=None, min_step=None, rate_scale=1.0)
Jacobi(num_modes, *, sigma, rho, rate_scale=1.0)
SumOfExponentials(*, target_error, t_final, min_step)
Diethelm2022(
    num_modes, *, quadrature="trapezoidal", target_error=1e-8,
    decay_scale=1.0, truncation_radius=None, rate_scale=1.0,
)
YuanAgrawal(num_modes, *, rate_scale=1.0)
SineDiffusive(num_modes)
FullHistory()
LubichCQ(order="bdf2", num_corrections=None)
AlikhanovL21Sigma()
FastObliviousCQ(
    target_error=1e-6, num_levels=16, nodes_per_level=None,
    direct_steps=20, contour="talbot",
)

Recurrence(interpolant="quadratic")
AuxiliaryODE(scheme="backward_euler")
Oscillator()

FractionalTimeStepper(
    F, representation, t, dt, u, *,
    formulation=None, u0=None, bcs=None,
    solver_parameters=None, appctx=None,
)
```

| Topic | Support |
| --- | --- |
| Markers | Caputo and left Riemann-Liouville. The latter adds the exact initial trace. Markers must wrap `u` directly. Transformed expressions are currently unsupported. |
| Full history | Direct variable-step linear history with one stored field per accepted step. |
| Lubich CQ | Uniform-step BDF1 or BDF2. Starting corrections default to one for BDF1 and two for BDF2. `num_corrections` may be 0 through 16. Poorly conditioned correction systems emit `StartingCorrectionAdvisoryWarning` and record `starting_system_recommended=False`. Storage grows by one field per accepted step. |
| Alikhanov L2-1$\sigma$ | Uniform-step Caputo formula with one shared $\alpha$. The complete residual is evaluated at $t_{n+\sigma}$, where $\sigma=1-\alpha/2$. Storage grows by one field per accepted step. |
| Fast-oblivious CQ | Uniform-step BDF1 CQ with a Talbot contour, exact recent history, and dyadic older history. `num_levels` supports at most $2^{\mathtt{num\_levels}}-1$ steps. `nodes_per_level` is 4 through 64 and `direct_steps` is 6 through 4096. |
| Diffusive representations | Positive-rate Birk-Song, Cayley, Jacobi, Diethelm2008, Jiang sum-of-exponentials, Diethelm2022, or Yuan-Agrawal spectra. `SineDiffusive` has a separate undamped oscillator spectrum. Equal mode counts do not imply equal accuracy. |
| Recurrence | One or more markers on scalar or fixed-size vector continuous Lagrange fields. `interpolant="quadratic"` is the default and stores one additional physical field for the whole stepper, not one per mode. {doc}`theory/time-stepping` gives the interpolant orders. |
| Auxiliary ODE | One marker on a scalar continuous Lagrange field. The monolithic $V^{m+1}$ solve uses backward Euler or trapezoidal stepping. Use this formulation when PETSc needs field access to the modes. |
| Oscillator | `SineDiffusive` only. Stores position and velocity for every mode, then advances them with an exact rotation and linear Duhamel forcing. Available through the general time steppers. The Caputo-Wismer application uses positive-rate representations. |
| Time | Positive variable `dt`. The residual is evaluated at `t + dt`. `advance()` updates `u` and history. The caller updates `t`. |
| State | History, statistics, reset, collective checkpointing, and Jacobian invalidation. |

Mode counts above the recommended maximum stay available for convergence
experiments. Crossing one emits `ModeCountAdvisoryWarning` and sets
`mode_count_recommended=False` in the representation metadata. The larger hard
ceilings are resource guards.

| Representation | Recommended maximum | Resource ceiling |
| --- | ---: | ---: |
| `BirkSong`, `Cayley`, `Diethelm2008`, `Jacobi` | 256 | 16,384 |
| `SineDiffusive` | 128 | 16,384 |
| `YuanAgrawal` | 2,048 | 16,384 |
| `Diethelm2022` | 16,385 | 65,536 |

`SumOfExponentials` chooses its mode count from the requested interval and
accuracy. It raises if that construction would require more than 65,536
modes.

`Jacobi` requires finite positive `sigma` and `rho`. At each requested order,
$\sigma\alpha+\rho(1-\alpha)=1$ is rejected because the Gauss-Jacobi
recurrence degenerates there. Parameter selection is manual and should cover
the complete time interval and order range of the problem.

:::{warning}
`Diethelm2022`, `YuanAgrawal`, and `SineDiffusive` are literature-comparison
options rather than production choices. Their controls and accuracy are
described in {ref}`diffusive representations <diffusive-representations>`.
:::

`SumOfExponentials` derives an alpha-dependent mode count from
`target_error`, `min_step`, and `t_final`. Its recurrence accepts variable
steps within that interval. The same representation also works with
`AuxiliaryODE`.

## Caputo-Wismer waves and sensors

```python
from yonderdrake.applications import (
    CaputoWismerArraySource,
    CaputoWismerImpedanceBoundary,
    CaputoWismerInverseProblem,
    CaputoWismerMaterial,
    CaputoWismerModel,
    CaputoWismerPML,
    CaputoWismerReconstruction,
    CaputoWismerSource,
    CaputoWismerStepper,
    SensorArray,
    reconstruct_initial_pressure,
    ring_sensor_locations,
    sphere_sensor_locations,
    time_reverse_sensor_data,
)
```

| Topic | Support |
| --- | --- |
| Wave field | Scalar continuous Lagrange spaces on 2D or 3D Firedrake meshes. Wave demos use CG2 by default. |
| Materials | Any fixed UFL indicator, with its own density, wave speed, damping, and Caputo order. |
| Sensors | `SensorArray` accepts user-supplied 2D or 3D locations. `SensorArray.ring(...)` is uniformly spaced in angle. `SensorArray.sphere(...)` uses an approximately uniform golden-angle arrangement. |
| Sources | Separable volume loads, marked-boundary loads, and independent sensor-array signals. Each signal spans `num_steps + 1` time points. |
| Imaging | `reconstruct_initial_pressure(...)` uses the iterative Kaltenbacher method by default, with tolerance `1e-5` and a user-overridable 100-iteration cap. Select `method="adjoint"` or `method="time_reversal"` for one-pass alternatives. |
| Attenuation | `"dissipative"` is the forward default. `"none"` gives lossless propagation. `"reversed"` requires a Helmholtz filter length to regularize high-frequency growth. |
| Sensor adjoint | `SensorArray.adjoint_field(values)` is the spatial L2 adjoint of Gaussian sampling. `adjoint_covector(values)` is its exact coefficient-space transpose. |
| Wave update | Without PML, `CaputoWismerModel` uses the centred update by default and therefore requires a CFL-safe `dt`. PML models default to an implicit first-order pressure and velocity update. `stiffness_theta` remains user-configurable. `CaputoWismerStepper` defaults to implicit stiffness. |
| Boundary | Reflecting natural boundaries by default, marker-aware first-order impedance conditions, and a real auxiliary-field PML for box extensions in 2D or 3D. Sensors are independent of the mesh boundary. |
| Adjoint | The exact discrete transpose includes heterogeneous density, impedance terms, PML auxiliary fields, and the coupled reversed-attenuation filter. Configured sources remain an affine offset to the initial-pressure map. |
| Parallelism | Forward propagation, PML, exact adjoints, time reversal, and iterative reconstruction support distributed meshes. Parallel reconstruction uses PETSc TAO. |

`CaputoWismerStepper` uses `BirkSong(num_modes)` by default. Pass a
tolerance-driven `SumOfExponentials` explicitly as `representation=`. Its
centred wave scheme keeps its own linear modal interpolation, which the
`Recurrence` interpolant option does not change. See
{doc}`examples/caputo-wismer-imaging` for the disk, ball, and BrainWeb vessel
examples.

## Exponential memory

```python
ExponentialMemory(u, decay_rate)
CaputoFabrizioOperator(u, alpha, *, normalization=1.0)

TimeMemoryStepper(
    F, t, dt, u, *,
    representation=None, formulation=None, u0=None, bcs=None,
    solver_parameters=None, appctx=None,
    warn_initial_compatibility=True,
)
```

| Topic | Support |
| --- | --- |
| Operator | Single bounded-kernel fading-memory state. See {ref}`exponential memory <exponential-memory>`. |
| Stepper | `TimeMemoryStepper` only. `FractionalTimeStepper` raises `ValueError`, and `FullHistory` cannot be combined with it. |
| Mixing | The recurrence formulation may combine exponential and fractional markers. `representation` supplies the spectrum for the fractional terms. `AuxiliaryODE` accepts one marker. |
| Initial data | The mode starts at zero, so construction emits `ExponentialMemoryCompatibilityWarning`. Suppress with `warn_initial_compatibility=False`. |
| Caputo-Fabrizio | `normalization` is $B(\alpha)$. The operator is built as a rescaled `ExponentialMemory`, so it carries no quadrature error of its own. |

## Spectral fractional Laplacian

```python
SpectralFractionalLaplacian(
    u, s, *, bcs=None, sinc_truncation_target=1e-10,
    shift_cache="stream", shift_solver_parameters=None,
    mass_solver_parameters=None,
)
```

| Topic | Support |
| --- | --- |
| Realization | Fractional power of the discrete Dirichlet or Neumann Laplacian, $0<s<1$. |
| Boundary | Omitting `bcs` gives natural homogeneous Neumann conditions. Complete homogeneous exterior Dirichlet conditions select the Dirichlet realization. |
| Quadrature | `sinc_truncation_target` controls the sinc model estimate. Mesh resolution controls FE operator error. |
| Cache | `stream` keeps two shifted solvers. `all` caches every shift. |
| Solvers | Shift and adjoint mass solves have separate PETSc dictionaries. Defaults use LU for small reference problems. |
| Diagnostics | Nodes, model estimate, setups, assemblies, solves, and reuse. |

## Riesz/restricted fractional Laplacian

```python
RieszFractionalLaplacian(
    u, s, *, extension="zero", quadrature_degree=6,
    quadrature_rule="boundary", assembly="matfree",
    compression_tolerance=1e-6, admissibility=1.0,
    leaf_size=16, bcs=None, mass_solver_parameters=None,
)
```

| Topic | Support |
| --- | --- |
| Scope | Scalar CG1 or CG2 on affine 2D triangles or 3D tetrahedra, $0<s<1$, zero exterior extension. |
| Boundary | Complete homogeneous `bcs` are required for $s\geq1/2$. |
| Topology | Periodic and overlapping cell geometries are unsupported. |
| Quadrature | Default `boundary` degree 6 is singularity fitted, using edge sectors in 2D and face sectors in 3D. `ordinary` uses Duffy tensor Gauss. |
| `matfree` | Uncompressed MPI backend with $O(N^2)$ work and replicated sources. |
| `dense` | Uncompressed serial reference weak matrix. |
| `hmatrix` | Serial or distributed hierarchy: uncompressed near field plus ACA-compressed admissible blocks. |
| H-matrix controls | ACA tolerance, admissibility, and leaf size. |
| Mass solve | Configurable, with CG/Jacobi as the default. |
| Diagnostics | Storage, timings, solves, blocks, ranks, and compression. |

Treat compression and quadrature as separate errors.

## Periodic Fourier fractional Laplacian

```python
PeriodicFractionalLaplacian(u, s)
```

| Topic | Support |
| --- | --- |
| Realization | Fourier-series multiplier $\lvert 2\pi k/L\rvert^{2s}$ with a zero constant mode. |
| Scope | Scalar degree-one nodal fields on fully periodic uniform 1D intervals, 2D quadrilateral rectangles, or 3D hexahedral boxes. |
| Validation | Uniform spacing, tensor cells, complete periodicity, and a one-to-one global-DOF/grid map are checked collectively during construction. |
| Boundary | No boundary conditions. Every coordinate direction must be periodic. |
| MPI | Supported in 1D, 2D, and 3D. |
| Mesh size | Use at least three Firedrake cells in every periodic direction. Two-cell periodic coordinate localization is ambiguous. |
| Diagnostics | Grid shape, lengths, spacing, backend, ranks, and applications. |

## Checkpoints

```python
stepper.save_checkpoint(checkpoint, name="state")
stepper.load_checkpoint(checkpoint, name="state")
```

These methods use an open Firedrake `CheckpointFile` collectively. Each name
is unique within a file. `checkpoint_state()` and `restore_checkpoint()` are
available for applications that manage an in-memory state dictionary instead
of a Firedrake checkpoint file. Invalid state is rejected without changing the
stepper.
See {doc}`examples/checkpointing`.
