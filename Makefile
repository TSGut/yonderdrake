PYTHON ?= python
MPIEXEC ?= mpiexec
MPIEXEC_FLAGS ?=

.PHONY: clean coverage docs lint mpi mpi-2 mpi-4 tests

docs:
	$(MAKE) -C docs html

tests:
	$(PYTHON) -m pytest -m "not parallel and not performance"

clean:
	rm -rf build dist docs/build benchmarks/benchmarks-output src/*.egg-info
	find . -path ./.firedrake -prune -o -type d -name __pycache__ -exec rm -rf {} +
	find . -path ./.firedrake -prune -o -type d \( -name .pytest_cache -o -name .ruff_cache -o -name .mypy_cache \) -exec rm -rf {} +

coverage:
	$(PYTHON) -m coverage erase
	$(PYTHON) -m coverage run -m pytest -m "not parallel"
	$(MPIEXEC) $(MPIEXEC_FLAGS) -n 2 $(PYTHON) -m coverage run -m pytest \
		tests/integration/test_checkpoint_variants.py::test_distributed_spatial_variant_rebuilds_across_checkpoint \
		tests/integration/test_caputo_wismer_mpi.py::test_distributed_caputo_wismer_reconstruction_two_ranks \
		tests/integration/test_caputo_wismer_mpi.py::test_distributed_caputo_wismer_pml_adjoint_two_ranks
	$(MPIEXEC) $(MPIEXEC_FLAGS) -n 2 $(PYTHON) -m coverage run -m pytest \
		tests/integration/test_periodic_mpi.py::test_distributed_periodic_two_ranks \
		tests/integration/test_riesz_mpi.py::test_distributed_hmatrix_two_ranks \
		tests/integration/test_riesz_mpi.py::test_dense_backend_rejects_distributed_mesh
	$(PYTHON) -m coverage combine
	$(PYTHON) -m coverage report

mpi: mpi-2 mpi-4

mpi-2:
	$(MPIEXEC) $(MPIEXEC_FLAGS) -n 2 $(PYTHON) -m pytest \
		tests/integration/test_caputo_wismer_mpi.py::test_distributed_caputo_wismer_reconstruction_two_ranks \
		tests/integration/test_caputo_wismer_mpi.py::test_distributed_caputo_wismer_pml_adjoint_two_ranks
	$(MPIEXEC) $(MPIEXEC_FLAGS) -n 2 $(PYTHON) -m pytest tests/integration/test_periodic_mpi.py::test_distributed_periodic_two_ranks
	$(MPIEXEC) $(MPIEXEC_FLAGS) -n 2 $(PYTHON) -m pytest tests/integration/test_spectral_mpi.py::test_distributed_spectral_two_ranks
	$(MPIEXEC) $(MPIEXEC_FLAGS) -n 2 $(PYTHON) -m pytest tests/integration/test_riesz_mpi.py::test_distributed_matfree_two_ranks
	$(MPIEXEC) $(MPIEXEC_FLAGS) -n 2 $(PYTHON) -m pytest 'tests/integration/test_riesz_mpi.py::test_distributed_hmatrix_two_ranks[1]'
	$(MPIEXEC) $(MPIEXEC_FLAGS) -n 2 $(PYTHON) -m pytest 'tests/integration/test_riesz_mpi.py::test_distributed_hmatrix_two_ranks[2]'
	$(MPIEXEC) $(MPIEXEC_FLAGS) -n 2 $(PYTHON) -m pytest 'tests/integration/test_riesz_mpi.py::test_distributed_tetrahedral_hmatrix_two_ranks[1]'
	$(MPIEXEC) $(MPIEXEC_FLAGS) -n 2 $(PYTHON) -m pytest 'tests/integration/test_riesz_mpi.py::test_distributed_tetrahedral_hmatrix_two_ranks[2]'
	$(MPIEXEC) $(MPIEXEC_FLAGS) -n 2 $(PYTHON) -m pytest tests/integration/test_checkpoint_file.py::test_checkpoint_file_restart_two_ranks
	$(MPIEXEC) $(MPIEXEC_FLAGS) -n 2 $(PYTHON) -m pytest 'tests/integration/test_checkpoint_variants.py::test_distributed_spatial_variant_rebuilds_across_checkpoint[spectral-stream]'
	$(MPIEXEC) $(MPIEXEC_FLAGS) -n 2 $(PYTHON) -m pytest 'tests/integration/test_checkpoint_variants.py::test_distributed_spatial_variant_rebuilds_across_checkpoint[spectral-all]'
	$(MPIEXEC) $(MPIEXEC_FLAGS) -n 2 $(PYTHON) -m pytest 'tests/integration/test_checkpoint_variants.py::test_distributed_spatial_variant_rebuilds_across_checkpoint[riesz-matfree]'
	$(MPIEXEC) $(MPIEXEC_FLAGS) -n 2 $(PYTHON) -m pytest 'tests/integration/test_checkpoint_variants.py::test_distributed_spatial_variant_rebuilds_across_checkpoint[riesz-hmatrix]'
	$(MPIEXEC) $(MPIEXEC_FLAGS) -n 2 $(PYTHON) -m pytest tests/integration/test_full_history.py::test_checkpoint_file_restart_two_ranks
	$(MPIEXEC) $(MPIEXEC_FLAGS) -n 2 $(PYTHON) -m pytest tests/integration/test_full_history.py::test_fast_cq_checkpoint_file_restart_two_ranks

mpi-4:
	$(MPIEXEC) $(MPIEXEC_FLAGS) -n 4 $(PYTHON) -m pytest tests/integration/test_periodic_mpi.py::test_distributed_periodic_four_ranks
	$(MPIEXEC) $(MPIEXEC_FLAGS) -n 4 $(PYTHON) -m pytest tests/integration/test_spectral_mpi.py::test_distributed_spectral_four_ranks
	$(MPIEXEC) $(MPIEXEC_FLAGS) -n 4 $(PYTHON) -m pytest tests/integration/test_riesz_mpi.py::test_distributed_matfree_four_ranks
	$(MPIEXEC) $(MPIEXEC_FLAGS) -n 4 $(PYTHON) -m pytest tests/integration/test_riesz_mpi.py::test_distributed_hmatrix_four_ranks
	$(MPIEXEC) $(MPIEXEC_FLAGS) -n 4 $(PYTHON) -m pytest tests/integration/test_checkpoint_file.py::test_checkpoint_file_restart_four_ranks
	$(MPIEXEC) $(MPIEXEC_FLAGS) -n 4 $(PYTHON) -m pytest tests/integration/test_full_history.py::test_checkpoint_file_restart_four_ranks

lint:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m mypy
