# Supported platforms and dependencies

Yonderdrake is tested with:

- Python 3.10 through 3.13
- the Firedrake 2026.4 stable release series
- real IEEE 754 binary64 PETSc scalar builds
- Ubuntu Linux and ARM macOS.

Firedrake supplies PETSc, MPI, and scalar precision, so install Yonderdrake
inside a working Firedrake environment.

[Irksome](https://www.firedrakeproject.org/Irksome/index.html) is optional
and locally tested at 2026.0.0.

Intel macOS and native Windows are untested. Windows users can use WSL.

For now, complex PETSc builds are rejected. Use coupled real fields, as in the
{doc}`Schrödinger demo <gallery/index>`. See the {doc}`api` for Riesz limits.
