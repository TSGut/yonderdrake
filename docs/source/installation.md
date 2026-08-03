# Installation

Install and activate a working Firedrake environment by following the
[Firedrake installation instructions](https://www.firedrakeproject.org/install.html).
Then install Yonderdrake into that environment:

```console
git clone https://github.com/TSGut/yonderdrake.git
cd yonderdrake
python -m pip install .
```

Install the plotting dependencies at the same time if you want to generate
gallery media:

```console
python -m pip install '.[visual]'
```

Firedrake supplies MPI and PETSc and must be installed separately. See
{doc}`support` for tested versions and platforms, then {doc}`quickstart` to
run something.

Confirm that the active interpreter sees Yonderdrake:

```console
python -c "import yonderdrake"
python demos/fractional_time/caputo_relaxation.py
```

The [caputo_relaxation.py](https://github.com/TSGut/yonderdrake/blob/main/demos/fractional_time/caputo_relaxation.py)
command finishes with a value near `u(1.00) = 0.415` and reports zero solver
failures. If Python cannot import Firedrake or Yonderdrake, check that
installation and execution use the same activated environment.
