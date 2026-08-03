(time-derivatives)=
# Caputo and Riemann-Liouville derivatives

Both time derivatives are symbolic markers. They record *which* derivative you
want. The representation and formulation decide how it is computed. Orders are
immutable and satisfy $0<\alpha<1$.

```{figure} ../_static/visuals/derivative-initial-trace.svg
:alt: A constant input has zero Caputo derivative and a decaying Riemann-Liouville initial-trace term
:class: doc-figure

For the same constant input, Caputo annihilates the initial value while the
Riemann-Liouville derivative retains its singular initial-trace term.
```

## Caputo

For a sufficiently regular $u$,

$$
D_C^\alpha u(t)
=\frac{1}{\Gamma(1-\alpha)}
  \int_0^t (t-\tau)^{-\alpha}u'(\tau)\,d\tau .
$$

`CaputoDerivative(u, alpha)` marks this. The Caputo derivative of a constant
is zero, so classical initial data need no special treatment.

## Riemann-Liouville

The supported left derivative with lower limit $t_0$ is

$$
{}_{t_0}D_{RL}^{\alpha}u(t)
=\frac{d}{dt}\left[
  \frac{1}{\Gamma(1-\alpha)}
  \int_{t_0}^t(t-\tau)^{-\alpha}u(\tau)\,d\tau
\right],
$$

and for a $u$ with a classical trace at $t_0$ it splits as

$$
{}_{t_0}D_{RL}^{\alpha}u(t)
={}_{t_0}D_C^\alpha u(t)
+\frac{u(t_0)}{\Gamma(1-\alpha)}(t-t_0)^{-\alpha}.
$$

`RiemannLiouvilleDerivative(u, alpha)` approximates the Caputo part with the
chosen representation and adds the initial-trace term **exactly**. A nonzero
constant retains this trace contribution.

Construction snapshots $t_0$ and $u(t_0)$. `reset(u0, t0=...)` replaces both.
Evaluation at $t_0$, arbitrary prehistory, and fractional-integral initial
data are not currently supported. The equivalence used here follows
[Yuan, Gao, Xiu, and Shi (2020)](https://doi.org/10.11948/20190317).

## Markers must wrap the stepped field

Marker replacement happens before form compilation and differentiation, so the
marker has to sit directly on `u`. Fixed spatial UFL operators may surround
it:

```python
Du = CaputoDerivative(u, alpha)
F_damping = b * inner(grad(Du), grad(v)) * dx
```

This is the weak form of $-b\Delta D_C^\alpha u$. Wrapping a transformed
expression instead of `u` is currently unsupported.

The justification is commutation. Let $L$ be a linear, time-independent
spatial operator on a fixed domain. If $u$ is regular enough for $L$ to pass
through the defining integral, then

$$
L D_C^\alpha u=D_C^\alpha L u,
$$

and for the Riemann-Liouville marker the classical trace must additionally
lie in the domain of $L$. Time-dependent or nonlinear spatial operators,
moving domains, and incompatible boundary realizations break this and are
unsupported.

## Multi-term damping

With fixed indicator functions $\chi_m(x)$ over a material partition, the
recurrence path supports several orders in one residual:

$$
(u_{tt},v)+(c^2(x)\nabla u,\nabla v)
+\sum_m(b_m\chi_m\nabla D_C^{\alpha_m}u,\nabla v)=(f,v).
$$

Each order owns its own diffusive representation. All terms share the field
and the time increment. The head demos in the {doc}`../gallery/index` use this
form.
