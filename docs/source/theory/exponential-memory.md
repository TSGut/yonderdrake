(exponential-memory)=
# Exponential (fading) memory

`ExponentialMemory(u, decay_rate)` represents

$$
\mathcal E_\lambda u(t)
=\int_{t_0}^{t}e^{-\lambda(t-\tau)}u'(\tau)\,d\tau,
\qquad \lambda>0.
$$

It is nonlocal in $u$, but a single internal state carries its complete
history:

$$
z'(t)+\lambda z(t)=u'(t),\qquad z(t_0)=0,\qquad
\mathcal E_\lambda u=z .
$$

This is one mode of the
{ref}`diffusive representation <diffusive-representations>` used on its own.
Its bounded kernel decays on one timescale and does not follow a power law.
Yonderdrake exposes it as fading memory.

## Exactness

For a step $h$ and Yonderdrake's linear time interpolant, the mode recurrence
is exact:

$$
z_{n+1}=e^{-\lambda h}z_n
+\frac{1-e^{-\lambda h}}{\lambda h}(u_{n+1}-u_n).
$$

The operator therefore costs one stored field per marker and contributes no
quadrature error of its own. Unlike the fractional representations, there is
no mode count to refine.

## Caputo-Fabrizio

`CaputoFabrizioOperator(u, alpha, normalization=1.0)` supplies the
parameterization proposed by
[Caputo and Fabrizio (2015)](https://www.naturalspublishing.com/download.asp?ArtcID=8820), which is
exactly a rescaled exponential memory:

$$
{}^{CF}D^\alpha u(t)
=\frac{B(\alpha)}{1-\alpha}
 \int_{t_0}^t\exp\!\left(-\frac{\alpha}{1-\alpha}(t-\tau)\right)u'(\tau)\,d\tau
=\frac{B(\alpha)}{1-\alpha}\,\mathcal E_{\alpha/(1-\alpha)}u(t),
$$

for $0<\alpha<1$ and positive normalization $B(\alpha)$. Yonderdrake builds it
directly from the marker above.

Its rational transfer function is a one-pole high-pass filter. See
[Ortigueira and Machado (2018)](https://doi.org/10.1016/j.cnsns.2017.12.001).
Power-law kernels use the representations described under
{ref}`Caputo and Riemann-Liouville derivatives <time-derivatives>` and are
selected with `CaputoDerivative`. Refining the exponential model preserves its
one-timescale kernel.

## Stepping

Use `TimeMemoryStepper` for exponential-memory markers:

```python
from yonderdrake import ExponentialMemory, TimeMemoryStepper

F = (inner(ExponentialMemory(u, 1.7), v) - inner(source, v)) * dx
stepper = TimeMemoryStepper(F, t, dt, u)
```

The default recurrence can mix exponential and fractional markers.
`representation` supplies the modes for the fractional terms. `AuxiliaryODE`
supports exactly one exponential marker.
Passing an exponential marker to `FractionalTimeStepper` raises `ValueError`,
and `FullHistory` cannot be combined with it. A worked example is in
{doc}`../examples/exponential-memory`.

:::{warning}
The operator is zero at $t=t_0$, so construction emits an
`ExponentialMemoryCompatibilityWarning`. An equation using it as the leading
time operator requires the remaining residual and the initial data to satisfy
the corresponding compatibility condition. Check that before setting
`warn_initial_compatibility=False`. That restriction on admissible initial
data, and the wider question of whether to model with nonsingular kernels at
all, are discussed in
[Diethelm, Garrappa, Giusti, and Stynes (2020)](https://doi.org/10.1515/fca-2020-0032).
:::
