"""Unit tests for Caputo-Wismer application helpers."""

from __future__ import annotations

import numpy as np
import pytest

from yonderdrake.applications import (
    ring_sensor_locations,
    sphere_sensor_locations,
)


@pytest.mark.unit
def test_ring_sensor_locations_are_uniform() -> None:
    points = ring_sensor_locations(8, 2.5, center=(1.0, -1.0))
    radii = np.linalg.norm(points - np.asarray([1.0, -1.0]), axis=1)
    np.testing.assert_allclose(radii, 2.5)
    np.testing.assert_allclose(points.mean(axis=0), [1.0, -1.0], atol=1.0e-15)


@pytest.mark.unit
def test_sphere_sensor_locations_have_requested_radius() -> None:
    points = sphere_sensor_locations(17, 1.8, center=(0.5, -0.25, 1.0))
    radii = np.linalg.norm(
        points - np.asarray([0.5, -0.25, 1.0]),
        axis=1,
    )
    np.testing.assert_allclose(radii, 1.8)


@pytest.mark.unit
@pytest.mark.parametrize(
    "factory,arguments,error",
    [
        (ring_sensor_locations, (0, 1.0), "positive integer"),
        (sphere_sensor_locations, (0, 1.0), "positive integer"),
        (ring_sensor_locations, (True, 1.0), "positive integer"),
        (ring_sensor_locations, (4, object()), "real scalar"),
        (ring_sensor_locations, (4, 0.0), "finite and positive"),
        (ring_sensor_locations, (4, 1.0), "2 finite"),
        (sphere_sensor_locations, (4, 1.0), "3 finite"),
    ],
)
def test_sensor_location_validation(factory, arguments, error: str) -> None:
    keywords = {}
    if "finite" in error and isinstance(arguments[1], float) and arguments[1] > 0.0:
        keywords["center"] = (0.0,)
    with pytest.raises((TypeError, ValueError), match=error):
        factory(*arguments, **keywords)
