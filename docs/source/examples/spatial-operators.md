# Fractional space

All three spatial operators are external UFL operators. Build one and use it
as an ordinary term.

(spectral-vs-riesz)=
## Choosing a spatial realization

`SpectralFractionalLaplacian` raises the eigenvalues of a chosen Dirichlet
operator,

$$
(-\Delta_D)^su=\sum_k\lambda_k^s(u,q_k)q_k,
$$

while `RieszFractionalLaplacian` applies the whole-space singular integral to
the zero extension,

$$
C_{2,s}\operatorname{PV}\int_{\mathbb R^2}
\frac{u(x)-u(y)}{|x-y|^{2+2s}}\,dy .
$$

`PeriodicFractionalLaplacian` acts on a flat periodic cell through the
Fourier-series multiplier

$$
\widehat{(-\Delta_{\mathrm{per}})^s u}_k=|k_L|^{2s}\widehat u_k,
\qquad
k_L=2\pi(k_1/L_1,\ldots,k_d/L_d).
$$

The three realizations have different domains, eigenfunctions, boundary
behaviour, numerical parameters, and solvers. Dirichlet and zero-exterior
realizations remain distinct under a homogeneous trace. Periodic tiling defines
a third problem.

```{figure} ../_static/visuals/spatial-realizations.svg
:alt: Dirichlet spectral, zero-exterior Riesz, and periodic Fourier fractional Laplacians
:class: doc-figure

The same field on the same square produces three different responses. The
Dirichlet spectral and zero-exterior Riesz realizations are shown on triangular
meshes. The periodic Fourier realization is shown on the uniform quadrilateral
periodic mesh required by its discrete Fourier transform.
```

The {ref}`maze gallery experiment <fractional-maze-gallery>` gives another
comparison of the classical, spectral, and zero-exterior Riesz operators on the
same domain.

## Spectral, Dirichlet or Neumann

Pass the homogeneous boundary condition explicitly:

```python
Lu = SpectralFractionalLaplacian(
    u,
    0.4,
    bcs=bc,
    sinc_truncation_target=1.0e-6,
    shift_cache="all",
)
F = (fd.inner(CaputoDerivative(u, alpha), v) + fd.inner(Lu, v)) * fd.dx
```

Omit `bcs` for natural homogeneous Neumann conditions. Constants then form the
zero eigenspace. A steady problem needs compatible forcing and a fixed mean,
as in the {ref}`maze gallery experiment <fractional-maze-gallery>`.

Inspect `Lu.diagnostics()` for node count and shifted-solver reuse. The
refinement order is in {doc}`refining`.

Use `shift_cache="all"` for repeated small problems and `"stream"` for bounded
storage.

## Riesz, zero exterior

For $s<1/2$ no boundary trace is required:

```python
Lu = RieszFractionalLaplacian(u, 0.3)
```

This uses `target_quadrature_degree=6`, the boundary-adapted rule, and the matrix-free
backend. The function is extended by zero outside the polygonal or polyhedral
domain.

For $s\geq1/2$, pass complete homogeneous `bcs`. Supported spaces and meshes
are listed in {doc}`../api`.

A three-dimensional problem uses the same interface:

```python
mesh = fd.UnitCubeMesh(12, 12, 12)
V = fd.FunctionSpace(mesh, "CG", 2)
x, y, z = fd.SpatialCoordinate(mesh)
u = fd.Function(V).interpolate(x * (1 - x) * y * (1 - y) * z * (1 - z))
bc = fd.DirichletBC(V, 0.0, "on_boundary")
Lu = RieszFractionalLaplacian(
    u, 0.7, bcs=bc, assembly="hmatrix", compression_tolerance=1e-8,
)
```

The integral geometry is substantially more expensive in 3D. The H-matrix
backend is intended for larger meshes because it compresses interactions
between well-separated basis-function supports. Refine
`compression_tolerance` separately from `target_quadrature_degree`.

The {ref}`Koch snowflake <fractional-snowflake-gallery>` and
{ref}`maze <fractional-maze-gallery>` gallery experiments compare this
zero-exterior realization with the classical and spectral operators.

Backend trade-offs are tabulated in {doc}`../theory/riesz-laplacian`.

## Periodic Fourier

Use a fully periodic uniform interval, quadrilateral rectangle, or hexahedral
box:

```python
mesh = fd.PeriodicRectangleMesh(
    32, 24, 2*fd.pi, 2*fd.pi, quadrilateral=True, reorder=False,
)
V = fd.FunctionSpace(mesh, "Q", 1)
u = fd.Function(V)
Lu = PeriodicFractionalLaplacian(u, 0.4)
```

The three-dimensional construction uses the same operator:

```python
mesh = fd.PeriodicBoxMesh(
    24, 20, 16, 2*fd.pi, 2*fd.pi, 2*fd.pi,
    hexahedral=True, reorder=False,
)
V = fd.FunctionSpace(mesh, "Q", 1)
Lu = PeriodicFractionalLaplacian(fd.Function(V), 0.62)
```

There are no boundary conditions or quadrature controls. Construction checks
that the field has exactly one scalar degree of freedom at each periodic grid
point and rejects a mesh that is nonuniform, only periodic in one direction,
triangular, deformed, or higher order.

The {ref}`periodic gyroid gallery experiment <periodic-gyroid-gallery>` shows
the three-dimensional operator acting on a multiscale periodic field. See
{doc}`../theory/periodic-laplacian` for the exact scope.
