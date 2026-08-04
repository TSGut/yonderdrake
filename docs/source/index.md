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
F = (inner(CaputoDerivative(u, 0.6), v) + inner(u, v)) * dx
stepper = FractionalTimeStepper(F, BirkSong(48), t, dt, u)
stepper.advance()
```

## What it supports

- Caputo and initialized Riemann-Liouville time derivatives.
- Full-history stepping and static memory with $O(m)$ storage.
- Single-exponential memory, including a
  labelled Caputo-Fabrizio operator.
- Dirichlet spectral, zero-exterior Riesz, and periodic Fourier fractional
  Laplacians, each with its corresponding boundary or exterior realization.
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

After the {doc}`quickstart`, choose the {doc}`time-memory workflow
<theory/time-memory>` or the {doc}`space-fractional workflow
<examples/spatial-operators>`. Each guide links to a runnable example. The
{doc}`gallery/index` collects larger applications and animations. Refinement,
parallel execution, checkpointing, and performance guidance are grouped under
Applications and workflow in the sidebar.

(mathematics-and-methods)=
## Mathematics and methods

The method map is the short overview of the available realizations and their
sources. Detailed definitions and numerical guidance live with each operator
family. The complete bibliography is in {doc}`references`.

(method-map)=
### Method map

| Operator | Method | What it is | Use | Primary source |
| --- | --- | --- | --- | --- |
| Caputo and Riemann-Liouville | `Cayley` | Diffusive modes, Gauss-Jacobi after a Cayley map of selectable exponent | Default time memory | Generalizes [Diethelm (2008)](https://doi.org/10.1007/s11075-008-9193-8) and [Birk and Song (2010)](https://doi.org/10.1007/s00466-010-0510-4) |
|  | `Jacobi` | Diffusive modes with independent low-rate and high-rate map exponents | Expert option for measured asymmetric rate ranges | [Diethelm (2023)](https://doi.org/10.1109/ICFDA58234.2023.10153228) |
|  | `SumOfExponentials` | Tolerance-driven positive exponential sum on a declared time interval | Supported alternative | [Jiang et al. (2017)](https://doi.org/10.4208/cicp.OA-2016-0136) |
|  | `Diethelm2022` | Truncated log-rate quadrature | Comparison only | [Diethelm (2022)](https://doi.org/10.3390/math10081245), [(2023)](https://doi.org/10.1007/978-981-19-7716-9_1) |
|  | `YuanAgrawal` | Original Gauss-Laguerre rule | Comparison only | [Yuan and Agrawal (2002)](https://doi.org/10.1115/1.1448322) |
|  | `SineDiffusive` | Generalized Gauss-Laguerre quadrature of undamped sine modes | Comparison only | [Khosravian-Arab and Dehghan (2024)](https://doi.org/10.1016/j.apnum.2024.06.017) |
|  | `LubichCQ` | Uniform-grid BDF1 or BDF2 convolution quadrature with starting corrections | Direct first- or second-order method | [Lubich (1986)](https://doi.org/10.1137/0517050), [(1988, Part I)](https://doi.org/10.1007/BF01398686), [(1988, Part II)](https://doi.org/10.1007/BF01398687) |
|  | `FastObliviousCQ` | BDF1 CQ with dyadic Talbot-contour history | Long uniform-grid histories | [Schädle, López-Fernández, and Lubich (2006)](https://doi.org/10.1137/050623139) |
|  | `AlikhanovL21Sigma` | Uniform-grid quadratic formula at $t_{n+\sigma}$ | Direct second-order method | [Alikhanov (2015)](https://doi.org/10.1016/j.jcp.2014.09.031) |
|  | `FullHistory` | Variable-step L1 history integral | Reference method | [Lin and Xu (2007)](https://doi.org/10.1016/j.jcp.2007.02.001) |
| Exponential memory | `ExponentialMemory` | One-timescale exponential convolution | Direct fading-memory operator | Standard one-state realization |
| Spectral fractional Laplacian | `SpectralFractionalLaplacian` | Power of the homogeneous-Dirichlet or natural-Neumann Laplacian | Spectral realization | [Bonito and Pasciak (2015)](https://doi.org/10.1090/S0025-5718-2015-02937-8) |
| Riesz fractional Laplacian | `RieszFractionalLaplacian` | Whole-space integral on the zero extension | Integral realization | [Acosta and Borthagaray (2017)](https://doi.org/10.1137/15M1033952) |
| Periodic fractional Laplacian | `PeriodicFractionalLaplacian` | Fourier multiplier on a uniform periodic cell | Periodic realization | Standard Fourier-series multiplier |

(convenience-constructors)=
### Convenience constructors

Published members of the families above. Each is the general class at a fixed
setting and behaves identically to it.

| Constructor | Equivalent to | Use | Primary source |
| --- | --- | --- | --- |
| `BirkSong(n)` | `Cayley(n, power=4)` | Default representation | [Birk and Song (2010)](https://doi.org/10.1007/s00466-010-0510-4) |
| `Diethelm2008(n)` | `Cayley(n, power=2)` | Narrow rate spans | [Diethelm (2008)](https://doi.org/10.1007/s11075-008-9193-8) |
| `CaputoFabrizioOperator` | Rescaled `ExponentialMemory` | Labelled Caputo-Fabrizio interface | [Caputo and Fabrizio (2015)](https://www.naturalspublishing.com/download.asp?ArtcID=8820), with the classification in [Ortigueira and Machado (2018)](https://doi.org/10.1016/j.cnsns.2017.12.001) |

```{toctree}
:hidden:
:maxdepth: 2
:caption: Getting started

installation
quickstart
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Nonlocal time operators

examples/time-fractional-odes
examples/time-fractional-pdes
examples/exponential-memory
examples/caputo-wismer-imaging
theory/time-memory
theory/time-stepping
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
:caption: Applications and workflow

examples/refining
examples/solvers-and-mpi
examples/checkpointing
examples/irksome
gallery/index
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Reference

theory/conventions
examples/benchmarks
api
api-reference
performance
references
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Development

contributing
support
logo
```
