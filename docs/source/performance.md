# Performance and MPI

The {doc}`examples/benchmarks` report setup and steady operation
separately and record all discretization parameters.

## Scaling parameters

Fractional time integration has three controls:

- $h$, which changes the Firedrake mesh and physical degrees of freedom
- $\Delta t$, which changes the number of solves over a fixed duration
- $L$, the number of modes in the diffusive representation.

The benchmark changes one control at a time. Equal $L$ does not imply equal
Birk-Song and Diethelm accuracy. A tolerance-driven sum of exponentials
derives $L$ from its requested interval and error.

Performance drivers warm each application or operator once before recording
cases, then time long enough to avoid reporting sub-millisecond single-call
noise.

The spatial performance benchmark varies $h$, the spectral sinc target, and
the Riesz target quadrature degree. Select the Riesz quadrature rule for each
run. The separate accuracy driver compares sinc and quadrature controls with
fixed references.

## Storage

| Method or operator | Stored state |
| --- | --- |
| Recurrence | $LN$ mode values per fractional term, independent of timestep count |
| Full history | One $N$-value increment per accepted step |
| Auxiliary ODE | $(L+1)N$ mixed values plus committed-state copies |
| Sine diffusive oscillator | $2LN$ mode values per fractional term, independent of timestep count |
| Spectral `stream` | Physical work vectors and two shifted solvers |
| Spectral `all` | One matrix and KSP per sinc shift |
| Riesz dense | $N^2$ weak entries |
| Riesz matrix-free | No $N^2$ matrix. Global source geometry and the current coefficient vector are replicated |
| Riesz H-matrix | Local dense near blocks and low-rank far factors |
| Periodic Fourier | Distributed grid and Fourier work arrays |

## MPI

All time and space operator families support MPI execution. Reference
backends explicitly marked as serial, such as dense Riesz assembly, are the
exceptions. The mass solve used by the Riesz operator is configurable and
defaults to CG/Jacobi.
