# Spectral fractional Laplacian

`SpectralFractionalLaplacian(u, s, bcs=...)` is the fractional power of a
discrete Laplacian. Complete homogeneous exterior `bcs` select the Dirichlet
realization. Omitting `bcs` selects the natural homogeneous Neumann
realization. With eigenpairs $\{(\lambda_k,q_k)\}$ of the selected positive
Laplacian, orthonormal in
$L^2(\Omega)$,

$$
(-\Delta_D)^s u=\sum_k \lambda_k^s (u,q_k)_{L^2(\Omega)}q_k .
$$

The finite-element eigenproblem is $Kq_k=\lambda_kMq_k$ with the selected
boundary realization and $q_i^TMq_j=\delta_{ij}$, so the discrete operator is

$$
A_h^su=Q\,\operatorname{diag}(\lambda_k^s)Q^TMu,
\qquad A_h=M^{-1}K .
$$

## Discretization

Yonderdrake never forms the eigendecomposition. It evaluates the Balakrishnan
integral representation of $A_h^s$ with sinc quadrature, which turns each
application into a sum of shifted elliptic solves that stay distributed.

- `sinc_truncation_target` controls the **quadrature model only**. It is a
  truncation estimate for the sinc rule. Mesh resolution controls the
  finite-element error. Targets below meaningful `float64` precision are clamped with a
  warning. Requests requiring more than 100,000 nodes are rejected.
- `shift_cache` trades setup cost against memory: `"stream"` keeps two shifted
  solvers, `"all"` caches one matrix and KSP per shift.
- `diagnostics()` reports node count, model estimate, setups, assemblies,
  solves, and reuse.

Requirements: $0<s<1$. Dirichlet conditions must be homogeneous and cover the
complete exterior boundary. With natural Neumann conditions, constants form
the zero eigenspace and their fractional-Laplacian action is zero.

Refine the sinc target on a fixed mesh *before* measuring finite-element
convergence. On a coarse mesh the discretization error dominates and tighter
sinc targets change nothing.

## Sources

[Bonito and Pasciak (2015)](https://doi.org/10.1090/S0025-5718-2015-02937-8)
for fractional powers of elliptic operators, and
[Bonito, Lei, and Pasciak (2019)](https://doi.org/10.1515/jnma-2017-0116) for
the sinc quadrature analysis.

The {doc}`Riesz operator <riesz-laplacian>` uses a zero-exterior realization.
The {doc}`periodic Fourier operator <periodic-laplacian>` uses a periodic
realization. See
{ref}`the comparison of spatial realizations <spectral-vs-riesz>`.
