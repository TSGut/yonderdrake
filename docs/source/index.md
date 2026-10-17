# Yonderdrake

> **Alpha software:** Yonderdrake has been broadly tested, but bugs and missing
> features may remain. Please [report them as issues](https://github.com/TSGut/yonderdrake/issues).

Yonderdrake adds fractional time derivatives, fading-memory operators, and
nonlocal spatial operators to [UFL](https://docs.fenicsproject.org/ufl/main/)
forms solved with [Firedrake](https://www.firedrakeproject.org/). You write an
ordinary residual with a marker in it. Yonderdrake does the rest.
Equations containing both a nonlocal time operator and a nonlocal space
operator are supported as well.

```python
F = (fd.inner(CaputoDerivative(u, 0.6), v) + fd.inner(u, v)) * fd.dx
stepper = TimeMemoryStepper(F, BirkSong(48), t, dt, u)
stepper.advance()
```

## What it supports

- Caputo and initialized Riemann-Liouville time derivatives with full-history stepping, direct convolution quadrature or static memory
  with $O(m)$ storage.
- Single-exponential memory, including a
  labelled Caputo-Fabrizio operator.
- Spectral (Dirichlet or Neumann), zero-exterior Riesz, and periodic Fourier
  fractional Laplacians, each with its corresponding boundary or exterior realization.
  The Riesz operator supports CG1 and CG2 on triangular 2D and tetrahedral 3D
  meshes.
- Direct 2D and 3D Caputo-Wismer wave support for power-law attenuation, with
  heterogeneous density, volume and boundary sources, impedance boundaries,
  PML, user-defined sensor arrays, exact adjoints, time reversal, and
  regularized inverse reconstruction.
- Variable steps, MPI, PETSc options, and checkpoint/restart.
- Spatial fractional operators combined with
  [Irksome](https://www.firedrakeproject.org/Irksome/index.html) for classical
  time integration.

## Start here

- {doc}`installation`: install into a Firedrake environment.
- {doc}`quickstart`: the core workflow in one page.
- {ref}`guides-and-examples`: one capability at a time.
- {doc}`gallery/index`: larger applications and their animations.
- {ref}`mathematics-and-methods`: definitions, methods, and the papers behind them.
- {doc}`api`: signatures and exact limitations.

(guides-and-examples)=
## Guides and examples

Each guide covers one capability and links to a runnable example.

- {doc}`examples/time-fractional-odes` and {doc}`examples/time-fractional-pdes`:
  fractional time derivatives.
- {doc}`examples/exponential-memory`: single-exponential and Caputo-Fabrizio
  memory.
- {doc}`examples/spatial-operators`: spectral, Riesz, and periodic fractional
  Laplacians.
- {doc}`examples/refining`, {doc}`examples/solvers-and-mpi`,
  {doc}`examples/checkpointing`, and {doc}`examples/irksome`: workflows.
- {doc}`examples/caputo-wismer-imaging`: fractionally damped waves and imaging.

(mathematics-and-methods)=
## Mathematics and methods

- {doc}`methods`: which method to use, with its primary source.
- {doc}`theory/time-memory`: time derivatives, memory representations, and
  time stepping.
- {doc}`theory/spectral-laplacian`, {doc}`theory/riesz-laplacian`, and
  {doc}`theory/periodic-laplacian`: the spatial realizations.
- {doc}`theory/conventions`: notation, normalizations, and terminology.
- {doc}`references`: the complete bibliography.

```{toctree}
:hidden:
:maxdepth: 2
:caption: Getting started

installation
support
quickstart
methods
notebooks
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Nonlocal time operators

examples/time-fractional-odes
examples/time-fractional-pdes
examples/exponential-memory
theory/time-memory
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Nonlocal space operators

examples/spatial-operators
theory/spectral-laplacian
theory/riesz-laplacian
theory/periodic-laplacian
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Workflows

examples/refining
examples/solvers-and-mpi
examples/checkpointing
examples/irksome
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Applications

examples/caputo-wismer-imaging
gallery/index
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Reference

api
api-reference
theory/conventions
examples/benchmarks
references
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Development

contributing
logo
```
