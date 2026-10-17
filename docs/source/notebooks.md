# Notebooks

Two Jupyter notebooks work through complete problems on a graded 2D mesh,
building up one step at a time and plotting as they go.

| Notebook | What it builds |
| --- | --- |
| [01-time-fractional-diffusion.ipynb](https://github.com/TSGut/yonderdrake/blob/main/notebooks/01-time-fractional-diffusion.ipynb) | Graded mesh, a Mittag-Leffler warm-up, the 2D Caputo problem, a manufactured solution, separating mode and timestep error, and graded timesteps |
| [02-spectral-fractional-laplacian.ipynb](https://github.com/TSGut/yonderdrake/blob/main/notebooks/02-spectral-fractional-laplacian.ipynb) | The spectral operator on a known eigenfunction, the sinc quadrature knob and its discretization floor, solving $(-\Delta_D)^s u=1$, why grading pays, and coupling back to a Caputo derivative |

Run them in an activated Firedrake environment. Beyond Yonderdrake itself, they
need only `matplotlib` and `mpmath`. They are stored without outputs, so every
figure is produced by the run in front of you.

A Jupyter kernel is a single process, so the notebooks run serially however the
environment was launched. They show each step of the calculation. For larger
problems, use `mpiexec -n N python your_script.py`, as in `demos/`.
Driving MPI from Jupyter via `ipyparallel` is currently untested.
