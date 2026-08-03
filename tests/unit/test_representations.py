"""High-precision and structural tests for diffusive spectra."""

from __future__ import annotations

import subprocess
import sys
import warnings

import numpy as np
import pytest
from tests.reference.diffusive import as_float
from tests.reference.diffusive import spectrum as reference_spectrum

from yonderdrake import (
    BirkSong,
    Cayley,
    Diethelm2008,
    Diethelm2022,
    FullHistory,
    ModeCountAdvisoryWarning,
    SineDiffusive,
    YuanAgrawal,
)
from yonderdrake.time.representations import (
    DiffusiveSpectrum,
    OscillatorSpectrum,
    _SingleExponential,
    validate_checkpoint_representation,
)


@pytest.mark.unit
@pytest.mark.parametrize("representation_type", [BirkSong, Diethelm2008])
@pytest.mark.parametrize("alpha", [0.1, 0.5, 0.9])
@pytest.mark.parametrize("num_modes", [1, 4, 12])
@pytest.mark.parametrize("rate_scale", [1.0e-3, 1.0, 1.0e3])
def test_spectra_match_high_precision(
    representation_type,
    alpha: float,
    num_modes: int,
    rate_scale: float,
) -> None:
    representation = representation_type(num_modes, rate_scale=rate_scale)
    observed = representation.spectrum(alpha)
    expected_rates, expected_weights = reference_spectrum(
        representation_type.__name__,
        alpha,
        num_modes,
        rate_scale,
    )
    np.testing.assert_allclose(
        observed.rates,
        as_float(expected_rates),
        rtol=3.0e-12,
        atol=0.0,
    )
    np.testing.assert_allclose(
        observed.weights,
        as_float(expected_weights),
        rtol=3.0e-12,
        atol=0.0,
    )


@pytest.mark.unit
def test_birk_song_extreme_rates_preserve_cayley_transform_accuracy() -> None:
    alpha = 0.6
    observed = BirkSong(64).spectrum(alpha)
    expected_rates, _ = reference_spectrum("BirkSong", alpha, 64)
    expected = np.asarray(as_float(expected_rates))
    extremes = np.asarray([0, -1])
    relative_error = np.abs(observed.rates[extremes] / expected[extremes] - 1.0)
    assert float(np.max(relative_error)) < 5.0e-13


def _caputo_kernel_error(
    spectrum,
    alpha: float,
    smallest: float,
    largest: float,
) -> float:
    """Worst relative error reproducing t^-alpha / Gamma(1 - alpha)."""
    from math import gamma

    times = np.logspace(np.log10(smallest), np.log10(largest), 120)
    approximated = (
        np.asarray(spectrum.weights)[None, :]
        * np.exp(-np.outer(times, np.asarray(spectrum.rates)))
    ).sum(axis=1)
    exact = times**-alpha / gamma(1.0 - alpha)
    return float(np.max(np.abs(approximated - exact) / exact))


@pytest.mark.unit
@pytest.mark.parametrize(
    ("power", "named_type"), [(2.0, Diethelm2008), (4.0, BirkSong)]
)
@pytest.mark.parametrize("alpha", [0.1, 0.5, 0.9])
def test_named_representations_are_cayley_special_cases(
    power: float,
    named_type,
    alpha: float,
) -> None:
    general = Cayley(24, power=power).spectrum(alpha)
    named = named_type(24).spectrum(alpha)
    np.testing.assert_allclose(general.rates, named.rates, rtol=1.0e-13)
    np.testing.assert_allclose(general.weights, named.weights, rtol=1.0e-13)


@pytest.mark.unit
@pytest.mark.parametrize("power", [1.6, 2.0, 3.0, 4.0, 6.0, 8.0])
@pytest.mark.parametrize("alpha", [0.3, 0.7])
def test_cayley_reproduces_the_kernel_at_every_power(
    power: float,
    alpha: float,
) -> None:
    error = _caputo_kernel_error(
        Cayley(64, power=power).spectrum(alpha), alpha, 1.0e-1, 1.0e1
    )
    assert error < 1.0e-4


@pytest.mark.unit
def test_declared_range_beats_the_fixed_power_when_it_is_wide() -> None:
    alpha, modes = 0.6, 64
    smallest, largest = 1.0e-6, 1.0e6
    ranged = Cayley(modes, t_final=largest, min_step=smallest)
    assert ranged.power > 5.0
    ranged_error = _caputo_kernel_error(
        ranged.spectrum(alpha), alpha, smallest, largest
    )
    fixed_error = _caputo_kernel_error(
        BirkSong(modes).spectrum(alpha), alpha, smallest, largest
    )
    assert ranged_error < 0.5 * fixed_error


@pytest.mark.unit
def test_cayley_rejects_ambiguous_or_incomplete_configuration() -> None:
    with pytest.raises(ValueError, match="either power or"):
        Cayley(16, power=3.0, t_final=1.0, min_step=1.0e-3)
    with pytest.raises(ValueError, match="requires power"):
        Cayley(16)
    with pytest.raises(ValueError, match="given together"):
        Cayley(16, t_final=1.0)
    with pytest.raises(ValueError, match="smaller than t_final"):
        Cayley(16, t_final=1.0e-3, min_step=1.0)
    with pytest.raises(ValueError, match="finite and positive"):
        Cayley(16, power=-1.0)
    with pytest.raises(ValueError, match="degenerate"):
        Cayley(16, power=1.0)


@pytest.mark.unit
@pytest.mark.parametrize("power", [0.5, 0.999, 1.001, 11.0])
def test_cayley_accepts_every_power_but_the_degenerate_one(power: float) -> None:
    spectrum = Cayley(32, power=power).spectrum(0.6)
    assert np.all(np.isfinite(spectrum.rates))
    assert np.all(spectrum.rates > 0.0)
    assert np.all(spectrum.weights > 0.0)


@pytest.mark.unit
def test_declared_range_never_narrows_past_the_published_exponent() -> None:
    narrow = Cayley(16, t_final=1.0, min_step=0.5)
    assert narrow.power == 2.0


@pytest.mark.unit
@pytest.mark.parametrize("representation_type", [BirkSong, Diethelm2008])
def test_spectrum_contract_and_serialization(representation_type) -> None:
    spectrum = representation_type(8).spectrum(0.7)
    assert np.all(np.diff(spectrum.rates) > 0.0)
    assert np.all(spectrum.weights > 0.0)
    assert spectrum.metadata["ordering"] == "increasing_rate"
    assert spectrum.metadata["num_modes"] == 8
    assert len(spectrum.metadata["quadrature_nodes"]) == 8
    with pytest.raises(ValueError):
        spectrum.rates[0] = 0.0
    description = representation_type(8).describe()
    assert description["num_modes"] == 8
    assert description["configurable_parameters"] == ("rate_scale",)


@pytest.mark.unit
@pytest.mark.parametrize("representation_type", [BirkSong, Diethelm2008])
def test_rate_scale_obeys_change_of_variables(representation_type) -> None:
    alpha = 0.37
    baseline = representation_type(6).spectrum(alpha)
    scaled = representation_type(6, rate_scale=25.0).spectrum(alpha)
    np.testing.assert_allclose(scaled.rates, 25.0 * baseline.rates)
    np.testing.assert_allclose(scaled.weights, 25.0**alpha * baseline.weights)


@pytest.mark.unit
@pytest.mark.parametrize("representation_type", [BirkSong, Diethelm2008])
def test_representation_validation(representation_type) -> None:
    with pytest.raises(TypeError, match="integer"):
        representation_type(True)
    with pytest.raises(ValueError, match="positive"):
        representation_type(0)
    with pytest.raises(ValueError, match="resource ceiling"):
        representation_type(16_385)
    with pytest.raises(ValueError, match="positive"):
        representation_type(4, rate_scale=0.0)
    with pytest.raises(TypeError, match="unsupported"):
        representation_type(4, mystery_parameter=1.0)
    representation = representation_type(4)
    for alpha in [0.0, 1.0, np.nan]:
        with pytest.raises(ValueError, match="0 < alpha < 1"):
            representation.spectrum(alpha)
    with pytest.raises(TypeError, match="real scalar"):
        representation.spectrum(object())


@pytest.mark.unit
@pytest.mark.parametrize(
    "rates,weights,error",
    [
        (np.ones((1, 1)), np.ones(1), "one-dimensional"),
        (np.ones(1), np.ones(2), "equal-sized"),
        (np.empty(0), np.empty(0), "at least one"),
        ([0.0], [1.0], "rates"),
        ([1.0], [np.nan], "weights"),
    ],
)
def test_diffusive_spectrum_validates_its_invariants(
    rates,
    weights,
    error: str,
) -> None:
    with pytest.raises(ValueError, match=error):
        DiffusiveSpectrum(rates, weights, {})


@pytest.mark.unit
def test_sine_diffusive_uses_a_separate_oscillator_spectrum() -> None:
    representation = SineDiffusive(32)
    spectrum = representation.spectrum(0.6)
    assert isinstance(spectrum, OscillatorSpectrum)
    assert spectrum.frequencies.size == 32
    assert np.all(np.diff(spectrum.frequencies) > 0.0)
    assert np.all(spectrum.weights > 0.0)
    assert spectrum.metadata["state"] == "undamped-position-velocity"
    assert representation.describe()["status"] == "comparison-only"
    validate_checkpoint_representation(
        representation.describe(0.6),
        representation,
        0.6,
    )
    with pytest.raises(ValueError):
        spectrum.frequencies[0] = 0.0


@pytest.mark.unit
def test_sine_diffusive_validates_mode_count_and_order() -> None:
    with pytest.raises(TypeError, match="integer"):
        SineDiffusive(True)
    with pytest.raises(ValueError, match="positive"):
        SineDiffusive(0)
    with pytest.raises(ValueError, match="resource ceiling"):
        SineDiffusive(16_385)
    for alpha in (0.0, 1.0, np.nan):
        with pytest.raises(ValueError, match="0 < alpha < 1"):
            SineDiffusive(4).spectrum(alpha)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("representation_type", "recommended_maximum"),
    [(BirkSong, 256), (Diethelm2008, 256), (SineDiffusive, 128)],
)
def test_mode_count_advisory_is_recorded(
    representation_type,
    recommended_maximum: int,
) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", ModeCountAdvisoryWarning)
        within_range = representation_type(recommended_maximum)
    assert within_range.describe()["mode_count_recommended"] is True

    with pytest.warns(
        ModeCountAdvisoryWarning,
        match=rf"num_modes={recommended_maximum + 1}",
    ):
        advisory = representation_type(recommended_maximum + 1)
    assert advisory.describe()["mode_count_recommended"] is False


@pytest.mark.unit
def test_advisory_metadata_survives_checkpoint_validation() -> None:
    with pytest.warns(ModeCountAdvisoryWarning):
        representation = BirkSong(257)
    metadata = representation.describe(0.4)
    assert metadata["mode_count_recommended"] is False
    validate_checkpoint_representation(metadata, representation, 0.4)


@pytest.mark.unit
@pytest.mark.parametrize(
    "frequencies,weights,error",
    [
        (np.ones((1, 1)), np.ones(1), "one-dimensional"),
        (np.ones(1), np.ones(2), "equal-sized"),
        (np.empty(0), np.empty(0), "at least one"),
        ([0.0], [1.0], "frequencies"),
        ([1.0], [np.nan], "weights"),
    ],
)
def test_oscillator_spectrum_validates_its_invariants(
    frequencies,
    weights,
    error: str,
) -> None:
    with pytest.raises(ValueError, match=error):
        OscillatorSpectrum(frequencies, weights, {})


@pytest.mark.unit
def test_single_exponential_is_an_exact_one_mode_spectrum() -> None:
    representation = _SingleExponential()
    spectrum = representation.spectrum(2.5)
    np.testing.assert_array_equal(spectrum.rates, [2.5])
    np.testing.assert_array_equal(spectrum.weights, [1.0])
    assert spectrum.metadata["quadrature"] == "none-exact"
    assert representation.describe()["status"] == "exact-exponential-memory"
    assert representation.describe(2.5)["decay_rate"] == 2.5
    with pytest.raises(TypeError, match="real scalar"):
        representation.spectrum(object())
    for invalid in (0.0, -1.0, np.nan):
        with pytest.raises(ValueError, match="finite and positive"):
            representation.spectrum(invalid)


@pytest.mark.unit
def test_checkpoint_representation_requires_complete_exact_metadata() -> None:
    representation = BirkSong(3)
    metadata = representation.describe(0.4)
    validate_checkpoint_representation(metadata, representation, 0.4)
    with pytest.raises(ValueError, match="metadata is missing"):
        validate_checkpoint_representation(None, representation, 0.4)

    incomplete = dict(metadata)
    del incomplete["quadrature_nodes"]
    with pytest.raises(ValueError, match="incomplete"):
        validate_checkpoint_representation(incomplete, representation, 0.4)

    changed = dict(metadata)
    changed["quadrature_nodes"] = tuple(reversed(changed["quadrature_nodes"]))
    with pytest.raises(ValueError, match="does not match"):
        validate_checkpoint_representation(changed, representation, 0.4)

    history = FullHistory()
    validate_checkpoint_representation(
        history.describe(0.4),
        history,
        0.4,
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "representation",
    [
        Diethelm2022(9, truncation_radius=3.0),
        YuanAgrawal(8),
    ],
)
def test_specialized_representation_checkpoint_metadata(representation) -> None:
    metadata = representation.describe(0.4)
    validate_checkpoint_representation(metadata, representation, 0.4)

    changed = dict(metadata)
    changed["rate_scale"] = 2.0
    with pytest.raises(ValueError, match="does not match"):
        validate_checkpoint_representation(changed, representation, 0.4)
@pytest.mark.unit
def test_representation_module_has_no_firedrake_dependency() -> None:
    code = (
        "import builtins\n"
        "original = builtins.__import__\n"
        "def guarded(name, *args, **kwargs):\n"
        "    if name == 'firedrake' or name.startswith('firedrake.'):\n"
        "        raise RuntimeError('unexpected Firedrake import')\n"
        "    return original(name, *args, **kwargs)\n"
        "builtins.__import__ = guarded\n"
        "from yonderdrake.time.representations import BirkSong\n"
        "assert BirkSong(3).spectrum(0.4).rates.size == 3\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
