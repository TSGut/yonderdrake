# Solvers, field splits, and MPI

All solver dictionaries pass straight through to Firedrake. The external
spatial operators supply matrix-free Jacobian actions:

```python
solver_parameters = {
    "snes_type": "ksponly",
    "mat_type": "matfree",
    "ksp_type": "gmres",
    "pc_type": "python",
    "pc_python_type": "firedrake.MassInvPC",
}
```

## Auxiliary-ODE field splits

With `AuxiliaryODE`, `stepper.appctx["yonderdrake"]` gives the physical field
index, the mode indices, and stable names. Group field zero separately from
fields $1..m$, as in
[caputo_auxiliary_ode.py](https://github.com/TSGut/yonderdrake/blob/main/demos/fractional_time/caputo_auxiliary_ode.py).
Production runs should replace that demo's small-problem LU blocks with
scalable preconditioners.

## Spectral shifts

A reasonable starting point for the shifted solves:

```python
shift_solver_parameters = {
    "ksp_type": "cg",
    "ksp_rtol": 1.0e-9,
    "pc_type": "gamg",
}
```

Use `shift_cache="stream"` for bounded storage or `"all"` for repeated small
problems.

## Riesz under MPI

Both `matfree` and `hmatrix` support MPI. The dense reference backend is
serial. The mass solve is configurable and defaults to CG/Jacobi.

## Periodic Fourier under MPI

The periodic Fourier operator supports MPI in 1D, 2D, and 3D.

Storage details are in {doc}`../performance`.
