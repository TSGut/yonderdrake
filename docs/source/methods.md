# Choosing a method

## Time memory

`BirkSong` is the general default because it has fixed storage and accepts
variable timesteps. Use `FullHistory` as a direct variable-step reference.
Use BDF2 `LubichCQ` for classical uniform-grid convolution
quadrature, and `AlikhanovL21Sigma` when the offset second-order PDE formula is
appropriate. Use fast-oblivious CQ for long uniform-grid histories where its
logarithmic scaling outweighs its contour-state constants.

`Diethelm2008` is its direct alternative for narrow rate spans.
`SumOfExponentials` is a supported alternative when the time interval and an
absolute kernel tolerance are known in advance. The remaining diffusive
representations are for literature comparisons. For single-exponential
memory, see {doc}`examples/exponential-memory`.

## Spatial operators

The three spatial operators are different realizations, not interchangeable
discretizations. The model determines which one applies. See
{ref}`choosing a spatial realization <spectral-vs-riesz>`.

(method-map)=
## Method map

| Operator | Method | What it is | Use | Primary source |
| --- | --- | --- | --- | --- |
| Caputo and Riemann-Liouville | `Cayley` | Diffusive modes, Gauss-Jacobi after a Cayley map of selectable exponent | Family of the default `BirkSong`. Requires `power` or a declared time range | Generalizes [Diethelm (2008)](https://doi.org/10.1007/s11075-008-9193-8) and [Birk and Song (2010)](https://doi.org/10.1007/s00466-010-0510-4) |
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
## Convenience constructors

Published members of the families above. Each is the general class at a fixed
setting and behaves identically to it.

| Constructor | Equivalent to | Use | Primary source |
| --- | --- | --- | --- |
| `BirkSong(n)` | `Cayley(n, power=4)` | Default representation | [Birk and Song (2010)](https://doi.org/10.1007/s00466-010-0510-4) |
| `Diethelm2008(n)` | `Cayley(n, power=2)` | Narrow rate spans | [Diethelm (2008)](https://doi.org/10.1007/s11075-008-9193-8) |
| `CaputoFabrizioOperator` | Rescaled `ExponentialMemory` | Labelled Caputo-Fabrizio interface | [Caputo and Fabrizio (2015)](https://www.naturalspublishing.com/download.asp?ArtcID=8820), with the classification in [Ortigueira and Machado (2018)](https://doi.org/10.1016/j.cnsns.2017.12.001) |
