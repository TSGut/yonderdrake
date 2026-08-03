# Fractional time derivatives

## Caputo relaxation

[caputo_relaxation.py](https://github.com/TSGut/yonderdrake/blob/main/demos/fractional_time/caputo_relaxation.py)
solves

$$
D_C^{0.6}u+u=0,\qquad u(0)=1,
$$

with 48 Birk-Song modes and the default eliminated recurrence:

```python
F = (inner(CaputoDerivative(u, 0.6), v) + inner(u, v)) * dx
stepper = FractionalTimeStepper(F, BirkSong(48), t, dt, u)
```

Increment `t` after each successful `advance()`. Representation error is
controlled by the mode count, temporal error by `dt`, and algebraic error by
the Firedrake solver parameters. See {doc}`refining`.

## Changing the representation

Only one argument changes:

```python
stepper = FractionalTimeStepper(F, Diethelm2008(48), t, dt, u)
```

Equal mode counts do not imply equal error. To compare fairly, fix the
timestep and problem, refine each mode count independently, and measure both
against the same analytic or high-accuracy reference.
See {ref}`time-derivative-representation-benchmarks` for comparisons across
every available spectrum.

Use direct history instead, with no quadrature spectrum to tune:

```python
stepper = FractionalTimeStepper(F, FullHistory(), t, dt, u)
```

This stores one solution increment per accepted step, so cost grows with the
number of steps. It is the natural reference method
({ref}`time-stepping`).

## Choosing modes from an error target

[caputo_sum_of_exponentials.py](https://github.com/TSGut/yonderdrake/blob/main/demos/fractional_time/caputo_sum_of_exponentials.py)
solves the same relaxation problem with a mode count derived from an absolute
kernel tolerance:

```python
representation = SumOfExponentials(
    target_error=1e-6,
    min_step=0.02,
    t_final=1.0,
)
stepper = FractionalTimeStepper(F, representation, t, dt, u)
```

The declared interval is checked during stepping. Lowering the timestep
therefore also means rebuilding the representation with the new `min_step`.

## Comparing the sine diffusive construction

[caputo_sine_diffusive.py](https://github.com/TSGut/yonderdrake/blob/main/demos/fractional_time/caputo_sine_diffusive.py)
recovers the prescribed solution $u=t$ from its analytic Caputo derivative:

```python
stepper = FractionalTimeStepper(
    F,
    SineDiffusive(128),
    t,
    dt,
    u,
    formulation=Oscillator(),
)
```

The method stores two fields per mode. It is a comparison-only implementation
because its quadrature converges slowly and its undamped errors persist over
long integrations. See {ref}`diffusive representations
<diffusive-representations>` for the representation and its measured
limitations.

## Changing the formulation

`AuxiliaryODE` solves the physical field and all modes together in $m+1$ mixed
fields, which makes the memory available to PETSc:

```python
stepper = FractionalTimeStepper(
    F,
    BirkSong(8),
    t,
    dt,
    u,
    formulation=AuxiliaryODE(scheme="backward_euler"),
)
```

`"trapezoidal"` is the other supported scheme. It is approximately second
order for smooth solutions but falls to first order at a $t^\alpha$ initial
singularity. The default recurrence instead solves one physical field and
stores $m$ histories, and is substantially cheaper. Choose the auxiliary
formulation when PETSc needs field access to the memory variables.
[caputo_auxiliary_ode.py](https://github.com/TSGut/yonderdrake/blob/main/demos/fractional_time/caputo_auxiliary_ode.py)
shows the field split. See
{doc}`solvers-and-mpi` for the PETSc side.

## Riemann-Liouville

Switch only the marker:

```python
F = (
    inner(RiemannLiouvilleDerivative(u, alpha), v)
    + inner(u, v)
) * dx
stepper = FractionalTimeStepper(F, BirkSong(48), t, dt, u)
```

The stepper evaluates

$$
D_{RL}^{\alpha}u(t)
=D_C^\alpha u(t)
+\frac{u(t_0)}{\Gamma(1-\alpha)}(t-t_0)^{-\alpha},
$$

where $t_0$ is the value of `t` at construction and the trace term is exact.
`stepper.reset(u0, t0=new_time)` establishes a new lower limit and trace.
Both formulations and all representations support it, and `FullHistory()`
uses the default eliminated formulation.

The Riemann-Liouville derivative of a nonzero constant includes its initial
trace term. The interface assumes a classical trace $u(t_0)$. Arbitrary prehistory
and fractional-integral initial data are not represented
({ref}`time-derivatives`).

## Exponential memory

`ExponentialMemory` and `CaputoFabrizioOperator` are bounded-kernel
fading-memory operators and use `TimeMemoryStepper`. See
{ref}`exponential-memory`.
