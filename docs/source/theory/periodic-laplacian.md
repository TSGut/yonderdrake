# Periodic Fourier fractional Laplacian

`PeriodicFractionalLaplacian(u, s)` is the fractional power of the Laplacian
on a flat periodic cell. For a box with side lengths
$L=(L_1,\ldots,L_d)$ and Fourier coefficients $\widehat u_k$,

$$
(-\Delta_{\mathrm{per}})^s u
=\sum_{k\in\mathbb Z^d}
\left|2\pi\left(\frac{k_1}{L_1},\ldots,\frac{k_d}{L_d}\right)\right|^{2s}
\widehat u_k e^{2\pi i\sum_j k_jx_j/L_j}.
$$

The constant mode has multiplier zero. This is neither the fractional power
of a Dirichlet finite-element Laplacian nor the whole-space integral of a
zero extension.

## Discretization

Yonderdrake applies a real multidimensional FFT to the unique periodic nodal
values, multiplies every discrete mode by $|k|^{2s}$, and transforms the
result back into the same Firedrake space. When the returned field appears in
`inner(Lu, v) * dx`, Firedrake supplies the ordinary finite-element mass
action. There is no sinc or singular quadrature parameter.

The operator currently accepts:

- scalar continuous degree-one nodal fields
- uniform `PeriodicIntervalMesh` meshes in one dimension
- fully periodic `PeriodicRectangleMesh(..., quadrilateral=True)` meshes in
  two dimensions
- fully periodic `PeriodicBoxMesh(..., hexahedral=True)` meshes in three
  dimensions.

Use at least three cells in every direction. With only two periodic cells,
Firedrake's localized coordinate field places both wrapped cells on the same
interval, so there is no unique geometric reconstruction of the FFT grid.

Construction validates the complete mesh collectively. It reconstructs each
coordinate level, checks uniform spacing and axis-aligned tensor cells,
requires periodic identification in every direction, and verifies a
one-to-one map between global degrees of freedom and the logical FFT grid.
Nonperiodic, partially periodic, triangular or tetrahedral, deformed,
overlapping, and higher-order spaces are rejected before the operator can be
assembled. The same reconstruction and Fourier kernel are used in 1D, 2D,
and 3D, on either serial or distributed Firedrake fields.

The operator supports serial and MPI execution in 1D, 2D, and 3D.
`diagnostics()` reports the logical shape, physical lengths, spacing, rank
count, backend, and application count.
