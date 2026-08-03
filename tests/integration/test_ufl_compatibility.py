"""Compatibility checks for the private UFL hook used by marker registration."""

from __future__ import annotations

import pytest

pytest.importorskip("ufl")

import ufl  # noqa: E402
from ufl.corealg.multifunction import MultiFunction  # noqa: E402

from yonderdrake.time._ufl_marker import (  # noqa: E402
    CaputoDerivativeMarker,
    ExponentialMemoryMarker,
    RiemannLiouvilleDerivativeMarker,
    _clear_multifunction_handlers_cache,
    evaluate_form_at_end_time,
)


@pytest.mark.verification
def test_multifunction_handlers_cache_contract_is_explicit() -> None:
    message = (
        "UFL changed MultiFunction._handlers_cache; update Yonderdrake's "
        "fractional-marker registration compatibility layer"
    )
    assert hasattr(MultiFunction, "_handlers_cache"), message
    cache = MultiFunction._handlers_cache
    assert isinstance(cache, dict), message

    _clear_multifunction_handlers_cache()
    assert cache == {}


@pytest.mark.verification
def test_missing_multifunction_cache_fails_loudly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delattr(MultiFunction, "_handlers_cache")
    with pytest.raises(
        RuntimeError,
        match=r"MultiFunction\._handlers_cache is required",
    ):
        _clear_multifunction_handlers_cache()


@pytest.mark.verification
def test_non_dictionary_multifunction_cache_fails_loudly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(MultiFunction, "_handlers_cache", ())
    with pytest.raises(RuntimeError, match="no longer a dictionary"):
        _clear_multifunction_handlers_cache()


@pytest.mark.unit
@pytest.mark.parametrize(
    "marker_type",
    [
        CaputoDerivativeMarker,
        RiemannLiouvilleDerivativeMarker,
        ExponentialMemoryMarker,
    ],
)
def test_marker_scalar_order_and_string_contract(marker_type) -> None:
    field = ufl.as_ufl(1.0)
    marker = marker_type(field, 0.4)
    assert marker.field == field
    parameter = (
        marker.decay_rate
        if isinstance(marker, ExponentialMemoryMarker)
        else marker.alpha
    )
    assert float(parameter) == 0.4
    assert marker_type.__name__.removesuffix("Marker") in str(marker)
    with pytest.raises(ValueError, match="must be scalar"):
        marker_type(field, ufl.as_vector((0.4, 0.5)))


@pytest.mark.unit
def test_plain_numeric_time_leaves_form_unchanged() -> None:
    form = object()
    assert evaluate_form_at_end_time(form, 0.0, 0.1) is form
