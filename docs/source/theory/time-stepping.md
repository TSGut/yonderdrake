(time-stepping)=
# How the memory is advanced

The representation chooses the memory *modes*. The formulation chooses how
they are *advanced* alongside the physical solve. `FullHistory` is an
alternative to both: it keeps the history itself.

```{figure} ../_static/visuals/memory-formulations.svg
:alt: Full history, eliminated recurrence, and auxiliary ODE memory formulations
:class: doc-figure

The same fractional derivative can store its full past, eliminate a fixed set
of memory modes outside the field solve, or expose those modes in one mixed
system.
```

## Eliminated recurrence (default)

`Recurrence()` advances each mode outside the Firedrake solve. Its default
quadratic interpolant uses a quadratic through the current step and the
preceding accepted value. For equal steps, the exact variation-of-constants
update is

$$
\phi_j^{n+1}
=e^{-\lambda_jh}\phi_j^n
+A(\lambda_jh)(u^{n+1}-u^n)
+Q(\lambda_jh)(u^n-u^{n-1}),
$$

where

$$
A(z)=\tfrac12\phi_1(z)+\psi(z),\qquad
Q(z)=\tfrac12\phi_1(z)-\psi(z),\qquad
\psi(z)=\frac{1-\phi_1(z)}{z}.
$$

The first step uses the linear update because no preceding increment exists.
This starting treatment does not change the asymptotic orders measured below.
Unequal accepted steps use the corresponding variable-step quadratic
coefficients.

Set `Recurrence(interpolant="linear")` to use the original piecewise-linear
assumption,

$$
\phi_j^{n+1}
=e^{-\lambda_jh}\phi_j^n
+\phi_1(\lambda_jh)(u^{n+1}-u^n).
$$

Each modal update is exact for its chosen interpolant. Applying the recurrence
to prescribed exact values of a smooth $u$ gives orders $2-\alpha$ and 3 for
linear and quadratic interpolation. Solving for $u$ through the complete
stepper gives orders $2-\alpha$ and $3-\alpha$ instead. The implicit solve
reintroduces the alpha-dependent loss.

The table reports the last observed order from $N=100,200,400$ uniform steps
through the actual stepper, with 120 Birk-Song modes. The smooth manufactured
solution is $u=t^2$. The singular solution $u=t^\alpha$ has the weak initial
singularity typical of time-fractional evolution.

| $\alpha$ | Smooth linear | Smooth quadratic | Singular linear | Singular quadratic |
| ---: | ---: | ---: | ---: | ---: |
| 0.3 | 1.67 | 2.70 | 1.00 | 1.00 |
| 0.6 | 1.39 | 2.40 | 0.99 | 1.00 |
| 0.9 | 1.10 | 2.10 | 0.92 | 1.00 |

Quadratic interpolation gains one order only for smooth solutions. Both choices
are approximately first order for the singular solutions users commonly meet.
Their error constants still differ at $N=400$:

| $\alpha$ | Smooth error reduction | Singular error reduction |
| ---: | ---: | ---: |
| 0.3 | 1129x | 1.42x |
| 0.6 | 544x | 1.95x |
| 0.9 | 315x | 3.37x |

Across these cases, warm median cost at $N=400$ rose from 0.578 to 0.635 ms per
step, an increase of 9.8 percent. Quadratic is the default because it was never
less accurate in this study, often reduced error substantially, and added a
small arithmetic cost. Mode count and timestep still refine independently.

For the more representative relaxation problem
$D_C^{0.6}u+u=0$, the maximum error on $0\leq t\leq2$ occurs near the weak
initial singularity. The table uses 100 steps, 120 Birk-Song modes, and the
graded levels $t_n=2(n/100)^3$. Doubling the mode count changed every error in
the grading sweep by less than $1.5\times10^{-5}$ relatively.

| Time levels | Linear | Quadratic |
| --- | ---: | ---: |
| Uniform | $2.101\times10^{-2}$ (baseline) | $2.101\times10^{-2}$ (1.00x) |
| Graded, $r=3$ | $7.960\times10^{-4}$ (26.4x) | $6.595\times10^{-4}$ (31.9x) |

The uniform-grid maximum is the first-step error, where both interpolants use
the same linear starting update. Grading is therefore the first refinement to
make for this singular solution. The two controls still compose: at $r=3$,
quadratic interpolation improves the graded result by a further 21 percent.
In a sweep through $r=1,\ldots,7$, linear recurrence was best at $r=4$ with a
27.0x reduction, while quadratic recurrence tolerated the coarser late steps
of $r=5$ and reached a 273x reduction. The optimum depends on the problem and
step count. Test the grading exponent for the problem and timestep count.

Mode state is committed only after a successful physical solve, so a failed
step leaves the memory untouched. This path supports variable steps, several
markers in one residual, scalar fields, and fixed-size vector fields. Updates
use PETSc vectors directly. Quadratic interpolation adds one shared physical
history field, independent of the number of modes and time-memory terms.

The storage claim was checked on the four-material skullball problem with 120
modes per material and 4,225 spatial degrees of freedom. Explicit fields count
the modal state, physical history, initial state, rollback storage, update
scratch, and assembled history fields owned by the time-memory stepper.

| Interpolant | Modal plus physical-history fields | All explicit time-memory fields | Explicit storage | Warm wall time |
| --- | ---: | ---: | ---: | ---: |
| Linear | 481 | 492 | 15.859 MiB | 6.04 ms/step |
| Quadratic | 482 | 493 | 15.891 MiB | 6.35 ms/step |

Quadratic therefore adds one field, 0.032 MiB and 0.20 percent in this case.
Its wall time is 5.3 percent higher. The extra field grows with the spatial
problem in 3D, but its ratio to modal storage decreases as the number of modes
or time-memory terms grows. This supports keeping quadratic as the general
recurrence default.

The Caputo-Wismer application retains its dedicated linear modal update. Its
material modes are batched in arrays, and a corresponding quadratic method
would also need one pressure-increment vector shared across all modes and
materials. The linear choice follows from the centred forward and
discrete-adjoint pair. A quadratic update requires a separate coupled
derivation and validation.

## Auxiliary ODE

`AuxiliaryODE` puts every mode into one mixed problem on $V^{m+1}$,

$$
\dot\phi_j+\lambda_j\phi_j=\dot u,\qquad
D^\alpha_Cu\approx\sum_jw_j\phi_j ,
$$

discretized with backward Euler or trapezoidal stepping. This makes the memory
variables visible to PETSc. Field-split metadata is exposed through
`stepper.appctx["yonderdrake"]`. See {doc}`../examples/solvers-and-mpi`. This
formulation enlarges every solve. The eliminated recurrence is usually cheaper.
Both use $O(m)$ storage.

The same solved-power study used above gives the last observed orders from
$N=100,200,400$ steps. The discrete scalar mode equations were evaluated with
50 Birk-Song modes, then checked against the actual Firedrake mixed stepper at
$N=400$. The largest relative disagreement was $3.7\times10^{-9}$. Doubling to
100 modes changed the finest error by at most $4.8\times10^{-5}$ relatively.

| $\alpha$ | Smooth backward Euler | Smooth trapezoidal | Singular backward Euler | Singular trapezoidal |
| ---: | ---: | ---: | ---: | ---: |
| 0.3 | 1.00 | 2.00 | 1.00 | 0.99 |
| 0.6 | 1.00 | 1.98 | 1.00 | 1.00 |
| 0.9 | 1.00 | 1.91 | 1.00 | 1.00 |

Trapezoidal stepping was already approximately second order for smooth data,
so it was ahead of linear recurrence in that regime. It also falls to first
order for $u=t^\alpha$. At $N=400$, its singular-case error was 1.4, 2.5, and
10 times the backward Euler error for the three rows. It therefore offers no
general accuracy advantage on the weakly singular regime.

Warm actual-stepper cost was 12.1 to 12.9 ms per step for these 50-mode
spatially constant two-node mixed systems. The recurrence study above took 0.578
to 0.635 ms per step with 120 modes. `AuxiliaryODE` remains a field-access
option when PETSc requires access to the memory fields. Quadratic
recurrence now has the higher smooth order, while all four choices are about
first order on the singular power.

## Sine diffusive oscillator

`Oscillator()` is the dedicated formulation for `SineDiffusive`. Every mode
has a position and velocity. Under the same piecewise-linear solution
assumption, Yonderdrake applies the exact rotation

$$
\begin{bmatrix}\omega^{n+1}\\\dot\omega^{n+1}\end{bmatrix}
=
\begin{bmatrix}
\cos(zh)&\sin(zh)/z\\
-z\sin(zh)&\cos(zh)
\end{bmatrix}
\begin{bmatrix}\omega^n\\\dot\omega^n\end{bmatrix}
+\text{linear Duhamel forcing}.
$$

This formulation uses two fields per mode and supports variable steps. The
rotation is undamped, so old mode error persists. It is available only with
the comparison-only sine diffusive representation.

## Full history

`FullHistory()` integrates the piecewise-linear solution
history against the Caputo kernel directly. This is the variable-step L1
method:

$$
D_C^\alpha u(t_n)\approx
\frac1{\Gamma(2-\alpha)}
\sum_{k=0}^{n-1}
\frac{u_{k+1}-u_k}{t_{k+1}-t_k}
\left[
(t_n-t_k)^{1-\alpha}-(t_n-t_{k+1})^{1-\alpha}
\right].
$$

There is no quadrature spectrum to tune, which makes it the natural reference
when you want a single error source. The cost is that work and storage grow
with the number of accepted steps: one stored increment per step. See
[Lin and Xu (2007)](https://doi.org/10.1016/j.jcp.2007.02.001) for the
classical L1 construction. Yonderdrake evaluates the same history integral on
variable steps.

Riemann-Liouville markers add their exact initial-trace term to whichever of
these paths is in use.
