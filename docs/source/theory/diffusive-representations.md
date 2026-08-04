(diffusive-representations)=
# Diffusive representations of time memory

A Caputo derivative depends on the whole history of $u$. Most diffusive
methods replace that history by a fixed number of first-order memory modes,

$$
D_C^\alpha u(t)\approx\sum_{j=1}^m w_j\phi_j(t),
\qquad
\dot\phi_j+\lambda_j\phi_j=\dot u,
\qquad
\phi_j(0)=0,
$$

with positive rates $\lambda_j$ and weights $w_j$ obtained by quadrature of

$$
D_C^\alpha u(t)
=\frac{\sin(\pi\alpha)}{\pi}
 \int_0^\infty \lambda^{\alpha-1}\phi(t,\lambda)\,d\lambda ,
\qquad
\dot\phi(t,\lambda)+\lambda\phi(t,\lambda)=\dot u(t).
$$

Storage is $O(m)$ per fractional term. These positive-rate methods differ in
how that improper integral is discretized. `SineDiffusive` uses the separate
undamped construction described below.

| Representation | Quadrature | Status | Source |
| --- | --- | --- | --- |
| `Cayley` | Gauss-Jacobi after a Cayley map of selectable exponent | Default | Generalizes [Diethelm (2008)](https://doi.org/10.1007/s11075-008-9193-8) and [Birk and Song (2010)](https://doi.org/10.1007/s00466-010-0510-4) |
| `Jacobi` | Gauss-Jacobi after a two-parameter endpoint map | Expert | [Diethelm (2023)](https://doi.org/10.1109/ICFDA58234.2023.10153228) |
| `SumOfExponentials` | Dyadic Gaussian construction over a requested interval | Supported alternative | [Jiang et al. (2017)](https://doi.org/10.4208/cicp.OA-2016-0136) |
| `Diethelm2022` | Published Gauss-Laguerre rule, or truncated trapezoidal/Simpson/Gauss-Legendre rules | Comparison only | [Diethelm (2022)](https://doi.org/10.3390/math10081245), [(2023)](https://doi.org/10.1007/978-981-19-7716-9_1) |
| `YuanAgrawal` | Original Gauss-Laguerre rule | Comparison only | [Yuan and Agrawal (2002)](https://doi.org/10.1115/1.1448322) |
| `SineDiffusive` | Generalized Gauss-Laguerre quadrature of sine modes | Comparison only | [Khosravian-Arab and Dehghan (2024)](https://doi.org/10.1016/j.apnum.2024.06.017) |

## The Gauss-Jacobi family

`Cayley` maps the semi-infinite rate axis onto the Gauss-Jacobi reference
interval. Writing $r=(1-x)/(1+x)$ and taking rates $\lambda=r^{p}$, the
integral above becomes

$$
\frac{2p\sin(\pi\alpha)}{\pi}
\int_{-1}^{1}(1-x)^{p\alpha-1}(1+x)^{p(1-\alpha)-1}
\,\frac{\phi(t,\lambda(x))}{(1+x)^{p}}\,dx ,
$$

a Jacobi weight against a smooth remainder, which is what makes the rule
converge geometrically rather than at the root-exponential rate the
semi-infinite constructions achieve. Both exponents exceed $-1$ for every
$p>0$ and every $\alpha\in(0,1)$.

That exponent sets how many decades of relaxation rate a given mode count
spans, so a larger $p$ reaches further but resolves each decade less finely.
The best $p$ follows the width of the rate window the problem needs and
barely moves with $\alpha$. Give `Cayley` a `t_final` and a `min_step` and it
sizes the exponent from that declared span. Alternatively, give it `power`
directly to set the exponent yourself (except for exactly $1$ because that
places the Jacobi exponents on the degenerate $\alpha+\beta=-1$ recurrence).

Two of these exponents are published as methods and are available under their
own names: `Diethelm2008(n)` is `Cayley(n, power=2)` and `BirkSong(n)` is
`Cayley(n, power=4)`.

Nothing forces the two ends of the map to share an exponent. Taking
$\lambda(x)=(1-x)^{\sigma}/(1+x)^{\rho}$ gives Jacobi exponents
$\sigma\alpha-1$ and $\rho(1-\alpha)-1$, which exceed $-1$ for every positive
$\sigma$ and $\rho$, so the whole two-parameter family is admissible.
Yonderdrake implements it as `Jacobi(n, sigma=..., rho=...)`, with
$\sigma=\rho=p$ recovering `Cayley(n, power=p)`. Larger $\sigma$ extends the
long-memory end and larger $\rho$ the short-memory end, so the two can be
tuned against each other.

`Jacobi` is an expert option. Its parameters are supplied manually and should
be chosen by measuring kernel error across the time interval and orders the
problem needs, because the diagonal calibration that `Cayley` uses does not obviously
carry over. At each order the pair is rejected when
$\sigma\alpha+\rho(1-\alpha)=1$, where the recurrence degenerates.

Mode counts above the recommended ranges remain available for convergence
experiments. Construction emits `ModeCountAdvisoryWarning` and records
`mode_count_recommended=False` in the metadata. The larger hard ceilings in
{doc}`../api` are resource guards. The Gauss-Jacobi construction computes
selected eigenvectors in bounded-size chunks, so its temporary memory grows
linearly with the mode count even though its work remains quadratic.

For `Diethelm2022`, `quadrature="gauss-laguerre"` is the quadrature published
with the 2022 representation. Each Laguerre node produces one mode from each
half of the real-line integral, so `num_modes` must be even and is twice the
Laguerre node count. This canonical construction has no truncation radius or
rate scaling. It rejects the controls that belong only to truncated rules.
Large orders and mode counts can also exceed the range of positive float64
rates or weights, in which case construction raises an error naming the order
and total mode count.

The `"trapezoidal"`, `"simpson"`, and `"gauss-legendre"` choices instead
truncate the log-rate integral. These variants follow the 2024 analyses by
[Chaudhary and Diethelm](https://doi.org/10.1016/j.ifacol.2024.08.226) and
[Chaudhary and Diethelm](https://doi.org/10.1016/j.ifacol.2024.08.227).
Their truncation and envelope controls do not apply to Gauss-Laguerre.

## Sum of exponentials on a declared interval

`SumOfExponentials` derives its mode count from the requested accuracy and
time range:

```python
representation = SumOfExponentials(
    target_error=1e-6,
    min_step=0.01,
    t_final=2.0,
)
```

The Jiang construction approximates $t^{-1-\alpha}$ by positive
exponentials on `[min_step, t_final]`. Yonderdrake converts the same Gaussian
nodes to the Caputo memory weights. The newest interval retains the exact L1
coefficient and the modes carry the older history. This preserves the kernel
singularity without storing the full solution history.

The achieved `num_modes` is reported by `spectrum(alpha).metadata` and can
differ between fractional orders. A step below `min_step` or an advance past
`t_final` raises an error.

## Sine diffusive oscillators

`SineDiffusive` represents the Caputo derivative as

$$
D_C^\alpha u(t)=\int_0^\infty z^\alpha\omega(z,t)\,dz,
\qquad
\ddot\omega+z^2\omega
=\frac{2\cos(\pi\alpha/2)}{\pi}\dot u.
$$

Each quadrature node stores the pair $(\omega,\dot\omega)$. The `Oscillator`
formulation advances that pair by an exact rotation and integrates a linear
change in $u$ exactly over each step.

Generalized Gauss-Laguerre nodes and effective weights are evaluated in log
scale beyond the recommended range, where the weighted SciPy values begin to
underflow. This keeps finite positive spectra available for high-mode
comparison experiments.

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

The formulation is selected automatically when `formulation` is omitted.
Ordinary SDR converges only as $O(m^{\alpha-1})$. Fixed-problem comparisons
against `FullHistory` also show persistent, oscillatory long-time error because
the modes do not contract. It is included for reproducing and comparing
literature methods. The general time steppers support it. The Caputo-Wismer
application layer supports the positive-rate representations.

## Refining a representation

For the fixed-count positive-rate methods, the mode count and `rate_scale` are
quadrature
parameters of the memory integral. They are **independent of the time grid**.
Halving `dt` does not improve a badly resolved spectrum, and adding modes does
not fix a coarse timestep. For `SumOfExponentials`, lower `target_error` to
derive a finer spectrum, then lower `dt` and rebuild the representation with
the new `min_step`. Refine one control at a time, as in
{doc}`../examples/refining`.

For `SineDiffusive`, increase the mode count at fixed `dt` first. Its slow
quadrature convergence and undamped long-time error should both be measured
against `FullHistory` on the intended time interval.

Equal mode counts across representations do not imply equal accuracy.
Comparisons are only meaningful against a shared analytic or high-accuracy
reference. See {ref}`time-derivative-representation-benchmarks` for the
comparison.
