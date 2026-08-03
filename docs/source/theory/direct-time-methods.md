(direct-time-methods)=
# Direct convolution methods

The diffusive representations replace the power-law kernel with a fixed set of
memory modes. Yonderdrake also provides three uniform-grid methods that act on
the convolution itself. They are useful when a published discretization is
part of the model, when second-order time accuracy is important, or when a
long run makes direct history too expensive.

## Lubich convolution quadrature

`LubichCQ` generates convolution weights from a backward
difference symbol. For the Caputo derivative, it applies the weights to
$u-u(0)$:

$$
D_C^\alpha u(t_n) \approx
h^{-\alpha}\sum_{j=0}^n \omega_{n-j}\left(u_j-u_0\right),
\qquad
\sum_{j=0}^\infty \omega_j\zeta^j=\delta(\zeta)^\alpha.
$$

The available symbols are

$$
\delta_{\mathrm{BDF1}}(\zeta)=1-\zeta,
\qquad
\delta_{\mathrm{BDF2}}(\zeta)=\frac32-2\zeta+\frac12\zeta^2.
$$

BDF2 is the default. `num_corrections` adds Lubich starting weights that are
exact for the first powers $t^{\alpha},t^{2\alpha},\ldots$. Their small dense
systems are solved in higher precision before the coefficients are converted
to float64. The default is one correction for BDF1 and two for BDF2.

This implementation stores the increments in one contiguous local array and
applies all history weights in one batched matrix-vector operation. It still
uses $O(n)$ distributed-field storage and $O(n)$ work at step $n$, giving
$O(N^2)$ total history work. It requires a uniform timestep. BDF2 gives
second-order convergence for sufficiently regular data. Initial singularities
can reduce the observed order, which is why the starting corrections are part
of the default.

## Alikhanov L2-1-sigma

`AlikhanovL21Sigma` uses Alikhanov's quadratic L2-1$\sigma$ approximation with

$$
\sigma=1-\frac{\alpha}{2}.
$$

The complete residual is evaluated at $t_{n+\sigma}$. Yonderdrake therefore
replaces an ordinary occurrence of the solution by
$\sigma u_{n+1}+(1-\sigma)u_n$ and replaces the symbolic time by
$t_n+\sigma h$. The fractional marker and every other term consequently use
the same offset equation.

The method is second order on smooth solutions and usually retains better
accuracy than uncorrected high-order formulas near a weak initial singularity.
It requires a uniform timestep, Caputo markers with one shared $\alpha$, and
stores the full increment history. Its storage and total history work are
$O(N)$ fields and $O(N^2)$ respectively.

## Fast oblivious convolution quadrature

`FastObliviousCQ` accelerates BDF1 convolution quadrature.
Recent increments are applied exactly. Older increments are grouped into
dyadic blocks and integrated on conjugate Talbot contours. Complex contour
states are represented internally by coupled real arrays, so the public
Firedrake problem still uses real scalar fields.

For $N$ accepted steps, the history has $O(\log N)$ work per step and
$O(\log N)$ distributed-field storage, giving $O(N\log N)$ total history work.
The constants include the contour nodes at every active level, so direct
history can remain smaller and faster for short runs. The time discretization
is BDF1 and is therefore first order.

The principal controls are:

| Control | Meaning |
| --- | --- |
| `target_error` | Selects the default number of contour nodes |
| `nodes_per_level` | Explicit contour quadrature size per dyadic level |
| `direct_steps` | Number of recent increments kept exact |
| `num_levels` | Dyadic depth, supporting at most $2^{\mathtt{num\_levels}}-1$ steps |
| `contour` | Currently the published `"talbot"` contour |

The method requires a uniform timestep. Increase `nodes_per_level` to check
contour convergence independently of timestep convergence. Increase
`num_levels` before a run if its configured maximum step count is too small.

## Choosing a time-memory method

`BirkSong` remains the general default because it has fixed storage and accepts
variable timesteps. Use `FullHistory` as a direct variable-step reference.
Use BDF2 `LubichCQ` for classical uniform-grid convolution
quadrature, and `AlikhanovL21Sigma` when the offset second-order PDE formula is
appropriate. Use fast-oblivious CQ for long uniform-grid histories where its
logarithmic scaling outweighs its contour-state constants.
