# Time-fractional PDEs

A fractional time derivative composes with ordinary Firedrake spatial forms.
The marker stays on the stepped field. The spatial operator stays outside it.

## Time-fractional diffusion

```python
F = (
    fd.inner(CaputoDerivative(u, alpha), v)
    + kappa * fd.inner(fd.grad(u), fd.grad(v))
    - fd.inner(source, v)
) * fd.dx
```

[time_fractional_diffusion.py](https://github.com/TSGut/yonderdrake/blob/main/demos/fractional_time/time_fractional_diffusion.py)
uses the ordinary
spatial Laplacian with the Diethelm 2008 representation. See
{doc}`refining` for the refinement order.

## Caputo-Wismer-Kelvin damping

The fractionally damped acoustic wave equation is

$$
u_{tt}-c^2\Delta u-b\Delta(D_C^\alpha u)=0,
\qquad 0<\alpha<1 .
$$

For homogeneous Dirichlet data, integrate both Laplacians by parts and place
the marker directly on the stepped field:

```python
u_prev, u_older = u.copy(deepcopy=True), u.copy(deepcopy=True)
u_tt = (u - 2 * u_prev + u_older) / dt**2
Du = CaputoDerivative(u, alpha)
F = (
    fd.inner(u_tt, v)
    + c**2 * fd.inner(fd.grad(u), fd.grad(v))
    + b * fd.inner(fd.grad(Du), fd.grad(v))
) * fd.dx
stepper = TimeMemoryStepper(F, Diethelm2008(32), t, dt, u, bcs=bc)
```

`u_tt` is an ordinary backward difference. After each `advance()`, assign
`u_prev` to `u_older` and `u` to `u_prev`, as in
[visual_time_derivative_dragon_wave.py](https://github.com/TSGut/yonderdrake/blob/main/demos/gallery/caputo_wismer/visual_time_derivative_dragon_wave.py).

The last term is $-b\Delta(D_C^\alpha u)$. The classical Laplacian stays
outside the marker, which is justified in
{doc}`../theory/time-derivatives`. `RiemannLiouvilleDerivative` uses the same
weak form and adds its exact trace term.

## Several materials at once

With fixed indicator functions $\chi_m$, one residual may carry several
orders, each with its own representation:

```python
F_layered = sum(
    b_m * chi_m
    * fd.inner(fd.grad(CaputoDerivative(u, alpha_m)), fd.grad(v)) * fd.dx
    for b_m, chi_m, alpha_m in materials
)
```

The BrainWeb and skullball
demos in {doc}`../gallery/index` run this form under MPI. For reusable 2D and
3D wave and sensor-array helpers, see {doc}`caputo-wismer-imaging`.
