# Exponential memory and Caputo-Fabrizio

Exponential memory is a bounded-kernel fading-memory operator
({ref}`exponential-memory`). It needs no representation, so the stepper
receives `None`.

This example solves

$$
\mathcal E_\lambda u(t)=
\frac{1-e^{-\lambda t}}{\lambda},
\qquad u(0)=0,
$$

whose solution is $u(t)=t$.

```python
from math import exp

import firedrake as fd
from yonderdrake import (
    CaputoFabrizioOperator,
    ExponentialMemory,
    TimeMemoryStepper,
)

mesh = fd.UnitIntervalMesh(1)
V = fd.FunctionSpace(mesh, "CG", 1)
u = fd.Function(V)
v = fd.TestFunction(V)
t, dt = fd.Constant(0.0), fd.Constant(0.05)
source = fd.Constant(0.0)
decay_rate = 1.7

F = (
    fd.inner(ExponentialMemory(u, decay_rate), v)
    - fd.inner(source, v)
) * fd.dx
stepper = TimeMemoryStepper(F, None, t, dt, u)

for _ in range(20):
    target_time = float(t) + float(dt)
    source.assign((1.0 - exp(-decay_rate * target_time)) / decay_rate)
    stepper.advance()
    t.assign(t + dt)
```

The recurrence is exact for this linear history, so the only error left is
algebraic. Construction warns that the memory starts at zero. Here the initial
data are compatible.

For the parameterization used in the
[Caputo-Fabrizio](https://www.naturalspublishing.com/download.asp?ArtcID=8820)
literature, replace the marker with `CaputoFabrizioOperator(u, alpha)`. It
remains the same single exponential memory mode.
