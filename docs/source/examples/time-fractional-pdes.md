# Time-fractional PDEs

A fractional time derivative composes with ordinary Firedrake spatial forms.
The marker stays on the stepped field. The spatial operator stays outside it.

## Time-fractional diffusion

```python
F = (
    inner(CaputoDerivative(u, alpha), v)
    + kappa * inner(grad(u), grad(v))
    - inner(source, v)
) * dx
```

[time_fractional_diffusion.py](https://github.com/TSGut/yonderdrake/blob/main/demos/fractional_time/time_fractional_diffusion.py)
uses the ordinary
spatial Laplacian with the Diethelm 2008 representation. Refine modes at a fixed
small `dt`, then `dt` at a fixed large mode count, then the mesh.

## Caputo-Wismer-Kelvin damping

The fractionally damped acoustic wave equation is

$$
u_{tt}-c^2\Delta u-b\Delta(D_C^\alpha u)=0,
\qquad 0<\alpha<1 .
$$

For homogeneous Dirichlet data, integrate both Laplacians by parts and place
the marker directly on the stepped field:

```python
Du = CaputoDerivative(u, alpha)
F = (
    inner(u_tt, v)
    + c**2 * inner(grad(u), grad(v))
    + b * inner(grad(Du), grad(v))
) * dx
stepper = FractionalTimeStepper(F, Diethelm2008(32), t, dt, u, bcs=bc)
```

The last term is $-b\Delta(D_C^\alpha u)$. The derivative is fractional in time
and the spatial Laplacian is classical. On a fixed domain, for a sufficiently
regular field and a time-independent linear spatial operator,

$$
\Delta(D_C^\alpha u)=D_C^\alpha(\Delta u).
$$

Do not wrap a strong CG Laplacian in the marker. Keep the marker on the
evolving field. `RiemannLiouvilleDerivative` uses the same weak form and adds
its exact trace term.

## Several materials at once

With fixed indicator functions $\chi_m$, one residual may carry several
orders, each with its own representation:

```python
F_layered = sum(
    b_m * chi_m * inner(grad(CaputoDerivative(u, alpha_m)), grad(v)) * dx
    for b_m, chi_m, alpha_m in materials
)
```

All terms share the field and the time increment. The BrainWeb and skullball
demos in {doc}`../gallery/index` run this form under MPI. For reusable 2D and
3D wave and sensor-array helpers, see {doc}`caputo-wismer-imaging`.
