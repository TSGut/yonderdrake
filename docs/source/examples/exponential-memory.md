# Exponential memory and Caputo-Fabrizio

Exponential memory is a bounded-kernel fading-memory operator
({ref}`exponential-memory`). It uses `TimeMemoryStepper`.

This example solves

$$
\mathcal E_\lambda u(t)=
\frac{1-e^{-\lambda t}}{\lambda},
\qquad u(0)=0,
$$

whose solution is $u(t)=t$.

```python
from math import exp

from firedrake import *
from yonderdrake import (
    CaputoFabrizioOperator,
    ExponentialMemory,
    TimeMemoryStepper,
)

mesh = UnitIntervalMesh(1)
V = FunctionSpace(mesh, "CG", 1)
u = Function(V)
v = TestFunction(V)
t, dt = Constant(0.0), Constant(0.05)
source = Constant(0.0)
decay_rate = 1.7

F = (
    inner(ExponentialMemory(u, decay_rate), v)
    - inner(source, v)
) * dx
stepper = TimeMemoryStepper(F, t, dt, u)

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
