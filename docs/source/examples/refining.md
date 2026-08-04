# Refinement workflow

Vary one control at a time and follow the sequence for the operator in use.

| Operator | Likely refinement sequence | Compare against |
| --- | --- | --- |
| Full history | Reduce $\Delta t$ | Exact solution or the preceding timestep |
| Diffusive memory | Increase mode count $L$ → grade the time levels if the solution has an initial singularity → reduce $\Delta t$ → compare linear and quadratic interpolation if needed | Analytic or high-accuracy reference |
| Sum of exponentials | Lower `target_error` → reduce $\Delta t$ and rebuild with the new `min_step` | Analytic or full-history reference |
| Sine diffusive memory | Increase mode count $L$ → reduce $\Delta t$ → extend the comparison interval | Full history on the same time grid |
| Exponential memory | Reduce $\Delta t$ | The preceding timestep |
| Spectral fractional Laplacian | Tighten `sinc_truncation_target` → refine the mesh | Generalized eigenvalues |
| Riesz, dense or matrix-free | Increase `quadrature_degree` → refine the mesh | Over-resolved direct energy |
| Riesz, H-matrix | Increase `quadrature_degree` → lower `compression_tolerance` → refine the mesh | Over-resolved direct energy |
| Periodic Fourier fractional Laplacian | Increase the uniform grid resolution | Resolved Fourier modes |

For a time-dependent PDE, refine the time and space controls separately. Keep
iterative solver tolerances below the measured discretization error. See
{doc}`benchmarks` for numerical examples.

For smooth solved problems, linear and quadratic recurrence have orders
$2-\alpha$ and $3-\alpha$. Both are approximately first order on uniform steps
for a $t^\alpha$ initial singularity, where grading the steps gains far more
than the interpolant does.
