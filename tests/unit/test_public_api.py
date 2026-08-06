"""Tests for the deliberate live public namespace."""

from __future__ import annotations

import inspect
import subprocess
import sys

import pytest

import yonderdrake
from yonderdrake._firedrake import allocate_external_operator_matrix

EXPECTED_PUBLIC = {
    "AlikhanovL21Sigma",
    "AuxiliaryODE",
    "BirkSong",
    "CaputoDerivative",
    "CaputoFabrizioOperator",
    "Cayley",
    "Diethelm2008",
    "Diethelm2022",
    "ExponentialMemory",
    "ExponentialMemoryCompatibilityWarning",
    "FastObliviousCQ",
    "FractionalTimeStepper",
    "FullHistory",
    "Jacobi",
    "LubichCQ",
    "ModeCountAdvisoryWarning",
    "Oscillator",
    "PeriodicFractionalLaplacian",
    "Recurrence",
    "RiemannLiouvilleDerivative",
    "RieszFractionalLaplacian",
    "SineDiffusive",
    "SpectralFractionalLaplacian",
    "SumOfExponentials",
    "StartingCorrectionAdvisoryWarning",
    "TimeMemoryStepper",
    "YuanAgrawal",
    "__version__",
}


@pytest.mark.unit
def test_public_namespace_is_deliberate() -> None:
    assert set(yonderdrake.__all__) == EXPECTED_PUBLIC


@pytest.mark.unit
def test_spatial_operators_validate_before_importing_firedrake() -> None:
    with pytest.raises(ValueError, match="0 < s < 1"):
        yonderdrake.SpectralFractionalLaplacian(None, 0.0, bcs=object())
    with pytest.raises(ValueError, match="0 < s < 1"):
        yonderdrake.PeriodicFractionalLaplacian(None, 1.0)
    with pytest.raises(ValueError, match="extension"):
        yonderdrake.RieszFractionalLaplacian(
            None,
            0.5,
            extension="periodic",
        )
    with pytest.raises(ValueError, match="target_quadrature_rule"):
        yonderdrake.RieszFractionalLaplacian(
            None,
            0.5,
            target_quadrature_rule="adaptive",
        )
    with pytest.raises(ValueError, match="source_evaluation"):
        yonderdrake.RieszFractionalLaplacian(
            None,
            0.5,
            source_evaluation="adaptive",
        )
    for degree in (0, 1.5, True):
        with pytest.raises(ValueError, match="source_quadrature_degree"):
            yonderdrake.RieszFractionalLaplacian(
                None,
                0.5,
                source_quadrature_degree=degree,
            )
    with pytest.raises(ValueError, match="compression_tolerance"):
        yonderdrake.RieszFractionalLaplacian(
            None,
            0.5,
            compression_tolerance=1.0,
        )
    with pytest.raises(ValueError, match="admissibility"):
        yonderdrake.RieszFractionalLaplacian(
            None,
            0.5,
            admissibility=0.0,
        )
    with pytest.raises(ValueError, match="leaf_size"):
        yonderdrake.RieszFractionalLaplacian(
            None,
            0.5,
            leaf_size=0,
        )


@pytest.mark.unit
def test_external_operator_matrix_builder_is_required() -> None:
    with pytest.raises(RuntimeError, match="no compatible"):
        allocate_external_operator_matrix(object())


@pytest.mark.unit
def test_target_signatures_remain_inspectable() -> None:
    signature = inspect.signature(yonderdrake.FractionalTimeStepper)
    assert list(signature.parameters)[:5] == ["F", "representation", "t", "dt", "u"]
    assert signature.parameters["formulation"].default is None
    memory = inspect.signature(yonderdrake.TimeMemoryStepper)
    assert list(memory.parameters)[:4] == ["F", "t", "dt", "u"]
    assert memory.parameters["representation"].default is None
    assert memory.parameters["warn_initial_compatibility"].default is True
    spectral = inspect.signature(yonderdrake.SpectralFractionalLaplacian)
    assert list(spectral.parameters)[:2] == ["u", "s"]
    assert spectral.parameters["bcs"].default is None
    assert spectral.parameters["sinc_truncation_target"].default == 1.0e-10
    assert spectral.parameters["shift_cache"].default == "stream"
    assert spectral.parameters["mass_solver_parameters"].default is None
    periodic = inspect.signature(yonderdrake.PeriodicFractionalLaplacian)
    assert list(periodic.parameters) == ["u", "s"]
    riesz = inspect.signature(yonderdrake.RieszFractionalLaplacian)
    assert list(riesz.parameters)[:2] == ["u", "s"]
    assert riesz.parameters["source_evaluation"].default == "hybrid"
    assert riesz.parameters["source_quadrature_degree"].default == 6
    assert riesz.parameters["target_quadrature_degree"].default == 6
    assert riesz.parameters["target_quadrature_rule"].default == "boundary"
    assert riesz.parameters["assembly"].default == "matfree"
    assert riesz.parameters["compression_tolerance"].default == 1.0e-6
    assert riesz.parameters["admissibility"].default == 1.0
    assert riesz.parameters["leaf_size"].default == 16
    assert riesz.parameters["bcs"].default is None
    assert riesz.parameters["mass_solver_parameters"].default is None
    assert "quadrature_degree" not in riesz.parameters
    assert "quadrature_rule" not in riesz.parameters


@pytest.mark.unit
def test_removed_riesz_quadrature_names_raise_type_errors() -> None:
    with pytest.raises(TypeError, match="quadrature_degree"):
        yonderdrake.RieszFractionalLaplacian(
            object(),
            0.4,
            quadrature_degree=4,
        )
    with pytest.raises(TypeError, match="quadrature_rule"):
        yonderdrake.RieszFractionalLaplacian(
            object(),
            0.4,
            quadrature_rule="ordinary",
        )


@pytest.mark.unit
def test_public_representations_are_live() -> None:
    representations = [
        yonderdrake.BirkSong(4),
        yonderdrake.Diethelm2008(4),
        yonderdrake.Diethelm2022(5),
        yonderdrake.SumOfExponentials(
            target_error=1.0e-3,
            t_final=1.0,
            min_step=0.1,
        ),
        yonderdrake.SineDiffusive(4),
        yonderdrake.YuanAgrawal(4),
    ]
    for representation in representations:
        spectrum = representation.spectrum(0.6)
        nodes = getattr(spectrum, "rates", None)
        if nodes is None:
            nodes = spectrum.frequencies
        assert nodes.size == spectrum.metadata["num_modes"]


@pytest.mark.unit
def test_full_history_configuration_is_live() -> None:
    history = yonderdrake.FullHistory()
    expected = {
        "representation": "FullHistory",
        "interpolant": "linear",
    }
    assert history.describe() == expected
    assert history.describe(0.4) == expected
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        yonderdrake.FullHistory(interpolant="quadratic")


@pytest.mark.unit
def test_recurrence_interpolant_configuration_is_live() -> None:
    quadratic = yonderdrake.Recurrence()
    linear = yonderdrake.Recurrence(interpolant="linear")
    assert quadratic.interpolant == "quadratic"
    assert quadratic != linear
    assert quadratic.describe() == {
        "kind": "recurrence",
        "interpolant": "quadratic",
    }
    with pytest.raises(ValueError, match="linear.*quadratic"):
        yonderdrake.Recurrence(interpolant="constant")


@pytest.mark.unit
def test_lubich_convolution_quadrature_configuration_is_live() -> None:
    cq = yonderdrake.LubichCQ()
    assert cq.order == "bdf2"
    assert cq.num_corrections == 2


@pytest.mark.unit
def test_runtime_import_does_not_load_mpmath() -> None:
    command = "import sys; import yonderdrake; assert 'mpmath' not in sys.modules"
    subprocess.run([sys.executable, "-c", command], check=True)


@pytest.mark.unit
def test_alikhanov_configuration_is_live() -> None:
    method = yonderdrake.AlikhanovL21Sigma()
    assert method.describe(0.6)["sigma"] == pytest.approx(0.7)


@pytest.mark.unit
def test_fast_oblivious_cq_configuration_is_live() -> None:
    method = yonderdrake.FastObliviousCQ(num_levels=8)
    assert method.max_steps == 255
    assert method.describe(0.6)["history_splitting"] == "dyadic"


@pytest.mark.unit
def test_auxiliary_configuration_is_live() -> None:
    formulation = yonderdrake.AuxiliaryODE(scheme="trapezoidal")
    assert formulation.scheme == "trapezoidal"
    assert not hasattr(formulation, "coupling")


@pytest.mark.unit
def test_caputo_fabrizio_parameters_validate_before_importing_ufl() -> None:
    with pytest.raises(ValueError, match="0 < alpha < 1"):
        yonderdrake.CaputoFabrizioOperator(None, 0.0)
    with pytest.raises(TypeError, match="alpha"):
        yonderdrake.CaputoFabrizioOperator(None, object())
    with pytest.raises(ValueError, match="normalization"):
        yonderdrake.CaputoFabrizioOperator(
            None,
            0.5,
            normalization=0.0,
        )
