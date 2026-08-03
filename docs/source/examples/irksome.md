# Classical time stepping with Irksome

[Irksome](https://www.firedrakeproject.org/Irksome/index.html) generates
Runge-Kutta methods for semi-discrete UFL forms solved with Firedrake. It can
provide the classical time integrator while Yonderdrake provides a spatial
fractional operator:

```python
import firedrake as fd
import irksome
from yonderdrake import SpectralFractionalLaplacian

mesh = fd.UnitSquareMesh(24, 24)
V = fd.FunctionSpace(mesh, "CG", 1)
u = fd.Function(V)
v = fd.TestFunction(V)
t = fd.Constant(0.0)
dt = fd.Constant(0.01)
bc = fd.DirichletBC(V, 0.0, "on_boundary")

F = (
    fd.inner(irksome.Dt(u), v)
    + fd.inner(
        SpectralFractionalLaplacian(u, 0.7, bcs=bc),
        v,
    )
) * fd.dx
stepper = irksome.TimeStepper(
    F,
    irksome.GaussLegendre(1),
    t,
    dt,
    u,
    bcs=bc,
    solver_parameters={
        "mat_type": "matfree",
        "snes_type": "ksponly",
        "ksp_type": "gmres",
    },
)
stepper.advance()
```

Irksome transforms `Dt(u)`. The external operator supplies its action and
derivative. Use a matrix-free Jacobian for this composition, as in the example
above.

Irksome is optional and supplies the classical time integrator. Use
`FractionalTimeStepper` to evolve a residual containing a fractional time
marker. The
{doc}`../gallery/index` shows both compositions.

See the [Irksome repository](https://github.com/firedrakeproject/Irksome) for
installation and its complete time-stepping interface.
