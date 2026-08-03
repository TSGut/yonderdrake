# Conventions and normalizations

These definitions are normative for the whole library.

## Notation

- $\alpha$ is a fractional time order, $0<\alpha<1$.
- $s$ is a fractional power of the Laplacian, $0<s<1$.
- $d$ is the spatial dimension.
- $(-\Delta)^s$ is always written together with its realization, because the
  Dirichlet spectral, zero-exterior Riesz/restricted, and periodic Fourier
  operators are different operators.

Orders are immutable and must lie in the open unit interval.

## Boundary terminology

- **Spectral:** a fractional power of the homogeneous-Dirichlet or
  natural-Neumann elliptic operator. See {doc}`spectral-laplacian`.
- **Zero-exterior Riesz/restricted:** a whole-space integral applied after
  extending the interior function by zero. See {doc}`riesz-laplacian`.
- **Periodic Fourier:** a Fourier-series multiplier on a uniform flat periodic
  cell. See {doc}`periodic-laplacian`.

## Fourier transform

For integrable $u:\mathbb R^d\to\mathbb C$,

$$
\widehat u(\xi)=\int_{\mathbb R^d}e^{-i x\cdot\xi}u(x)\,dx,
\qquad
u(x)=\frac{1}{(2\pi)^d}
      \int_{\mathbb R^d}e^{i x\cdot\xi}\widehat u(\xi)\,d\xi ,
$$

so the whole-space fractional Laplacian has multiplier

$$
\widehat{(-\Delta)^s u}(\xi)=|\xi|^{2s}\widehat u(\xi).
$$

On a periodic box of lengths $L_j$, the continuous frequencies become
$2\pi k_j/L_j$. `PeriodicFractionalLaplacian` applies the corresponding
discrete Fourier multiplier to the unique periodic nodal values and assigns
zero to the constant mode.

## Riesz normalization

Consistency with that multiplier fixes

$$
C_{d,s}
=\frac{4^s\Gamma(d/2+s)}
       {\pi^{d/2}|\Gamma(-s)|}
=\frac{2^{2s}s\Gamma(d/2+s)}
       {\pi^{d/2}\Gamma(1-s)},
\qquad
C_{2,s}=\frac{2^{2s}s\Gamma(1+s)}{\pi\Gamma(1-s)} .
$$

All Riesz backends share this constant.

## Finite-element action

The Riesz implementation assembles the Galerkin action against every
finite-element test function $v$,

$$
\langle v,Lu\rangle=\int_\Omega v(x)Lu(x)\,dx ,
$$

The dense, matrix-free, and H-matrix backends all compute this action. Since
the continuous operator is symmetric, the discrete identity
$\langle v,Lu\rangle=\langle u,Lv\rangle$ is used to check the implementation.
The assembled action is left unchanged, preserving discrete asymmetry as a
diagnostic.

## Numerical precision and requested accuracy

The default methods compute quadrature rates and weights with standard 64-bit
floating-point arithmetic (`float64`). Higher-precision values are used only
for validation.

For the default spectral sinc quadrature, a target below meaningful `float64`
precision is raised to machine epsilon with a warning. A target requiring more
than 100,000 sinc nodes raises an error.
