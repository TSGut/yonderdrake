# Riesz/restricted fractional Laplacian

`RieszFractionalLaplacian(u, s)` is the whole-space singular integral applied
to the zero extension of the interior function:

$$
(-\Delta)^s u(x)
=C_{d,s}\,\operatorname{PV}\int_{\mathbb R^d}
 \frac{u(x)-u(y)}{|x-y|^{d+2s}}\,dy,
\qquad u=0\quad\text{on }\Omega^c .
$$

"Riesz" and "restricted" both refer to this construction, with the
normalization $C_{d,s}$ fixed in {doc}`conventions` so that the whole-space
multiplier is $|\xi|^{2s}$. The operator follows
[Di Nezza, Palatucci, and Valdinoci (2012)](https://doi.org/10.1016/j.bulsci.2011.12.004)
and its finite-element treatment follows
[Acosta and Borthagaray (2017)](https://doi.org/10.1137/15M1033952).

## Galerkin action

The assembled operator uses the weak Galerkin action

$$
A_{ij}=\int_\Omega\phi_i(x)(-\Delta)^s\phi_j(x)\,dx ,
$$

The default boundary-sector Gauss-Jacobi target rule resolves the remaining
boundary singularity. `target_quadrature_degree` controls it, and
`target_quadrature_rule="ordinary"` selects a Duffy tensor-Gauss alternative.

Scope: scalar CG1 or CG2 on affine 2D triangles or 3D tetrahedra, $0<s<1$,
zero exterior extension. Complete homogeneous `bcs` are **required** for
$s\geq1/2$, because a nonzero trace has infinite zero-extension energy. Below
$1/2$ no trace is needed. Periodic and overlapping cell geometries are
rejected.

## Simplex-supported source action

Yonderdrake removes the source singularity analytically before quadrature. For
a CG1 or CG2 polynomial on one affine simplex, extended by zero, the divergence
theorem reduces the volume integral to its boundary. Triangle sources reduce
to analytic edge integrals. Tetrahedral sources reduce to triangular-face
integrals whose radial direction is integrated analytically and whose
tangential direction uses Gaussian quadrature. For an affine triangle
polynomial $p$,

$$
(-\Delta)^s(p1_T)(x)
=\frac{C_{2,s}}{2s}\sum_{e\subset\partial T}
\left[
p(x)d_e\int_e|y-x|^{-2-2s}\,dl
+(\nabla p\cdot n_e)\int_e|y-x|^{-2s}\,dl
\right],
$$

with $d_e=(y-x)\cdot n_e$. Quadratic terms reduce to the same family of
one-dimensional edge integrals. In 3D the corresponding formula uses all four
tetrahedron faces and the normalization $C_{3,s}$.

`source_evaluation` controls how each simplex source is evaluated:

| Mode | Source action | Use |
| --- | --- | --- |
| `hybrid` (default) | exact formula for near and coincident pairs, source-cell Gauss quadrature on admissible far pairs | general use |
| `endpoint` | exact boundary formula for every source and target pair | reference |

`source_quadrature_degree` controls the Gauss rule for `hybrid` and has no
numerical effect under `endpoint`. The admissibility parameter determines the
near and far split.

## Backends

| Assembly | Storage | Parallel | Use |
| --- | --- | --- | --- |
| `matfree` (default) | no $N^2$ matrix, replicated source geometry | MPI | general use, $O(N^2)$ work |
| `hmatrix` | dense near blocks, low-rank far factors | serial or MPI | larger problems |
| `dense` | $N^2$ weak entries | serial | reference |

`hmatrix` compresses admissible far-field blocks with adaptive cross
approximation, following
[Bebendorf (2000)](https://doi.org/10.1007/PL00005410). `compression_tolerance`,
`admissibility`, and `leaf_size` control it. This is particularly useful in 3D,
where dense storage and uncompressed pairwise work become expensive quickly.

Source quadrature, target quadrature, and compression have independent error
controls. `diagnostics()` reports the selected source mode, source and target
degrees, evaluation counts, storage, timings, solves, blocks, ranks, and
achieved compression.
