"""Pure NumPy checks for the periodic Fourier backend."""

from __future__ import annotations

import numpy as np
import pytest

from yonderdrake.periodic.grid import PeriodicFourierBackend, PeriodicGridMap


def identity_grid(
    shape: tuple[int, ...],
    lengths: tuple[float, ...],
) -> PeriodicGridMap:
    return PeriodicGridMap(
        shape=shape,
        lengths=lengths,
        origins=(0.0,) * len(shape),
        global_to_flat=np.arange(np.prod(shape), dtype=np.int64),
    )


def cell_records(
    shape: tuple[int, ...],
    lengths: tuple[float, ...] | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    if lengths is None:
        lengths = tuple(float(count) for count in shape)
    records = []
    for lower in np.ndindex(shape):
        dofs = []
        coordinates = []
        for offset in np.ndindex(*(2,) * len(shape)):
            raw = tuple(
                lower[axis] + offset[axis] for axis in range(len(shape))
            )
            wrapped = tuple(
                raw[axis] % shape[axis] for axis in range(len(shape))
            )
            dofs.append(np.ravel_multi_index(wrapped, shape))
            coordinates.append(
                tuple(
                    raw[axis] * lengths[axis] / shape[axis]
                    for axis in range(len(shape))
                )
            )
        records.append(
            (
                np.asarray(dofs, dtype=np.int64),
                np.asarray(coordinates, dtype=np.float64),
            )
        )
    return records


@pytest.mark.unit
def test_periodic_fourier_one_dimensional_modes_and_nullspace() -> None:
    count = 16
    length = 2.0 * np.pi
    order = 0.37
    x = np.arange(count) * length / count
    values = 2.5 + np.sin(3.0 * x) - 0.4 * np.cos(5.0 * x)
    expected = 3.0 ** (2.0 * order) * np.sin(3.0 * x)
    expected -= 0.4 * 5.0 ** (2.0 * order) * np.cos(5.0 * x)
    backend = PeriodicFourierBackend(identity_grid((count,), (length,)), order)

    np.testing.assert_allclose(backend.apply(values), expected, atol=2.0e-13)
    np.testing.assert_allclose(backend.apply(np.ones(count)), 0.0, atol=1.0e-14)
    assert backend.diagnostics() == {
        "shape": (count,),
        "lengths": (length,),
        "spacing": (length / count,),
        "fft_backend": "numpy-serial",
        "applications": 2,
    }


@pytest.mark.unit
def test_periodic_fourier_two_dimensional_rectangular_mode() -> None:
    shape = (12, 10)
    lengths = (3.0, 5.0)
    order = 0.63
    x = np.arange(shape[0]) * lengths[0] / shape[0]
    y = np.arange(shape[1]) * lengths[1] / shape[1]
    xx, yy = np.meshgrid(x, y, indexing="ij")
    values = np.sin(2.0 * np.pi * 2.0 * xx / lengths[0])
    values *= np.cos(2.0 * np.pi * 3.0 * yy / lengths[1])
    eigenvalue = (2.0 * np.pi * 2.0 / lengths[0]) ** 2
    eigenvalue += (2.0 * np.pi * 3.0 / lengths[1]) ** 2
    backend = PeriodicFourierBackend(identity_grid(shape, lengths), order)

    result = backend.apply(values.reshape(-1)).reshape(shape)

    np.testing.assert_allclose(result, eigenvalue**order * values, atol=2.0e-13)


@pytest.mark.unit
def test_periodic_grid_and_fourier_kernel_are_dimension_generic() -> None:
    shape = (6, 5, 4)
    lengths = (2.0, 3.0, 5.0)
    grid = PeriodicGridMap.from_cell_records(
        cell_records(shape, lengths),
        dimension=3,
        global_size=np.prod(shape),
    )
    axes = [
        np.arange(count) * length / count
        for count, length in zip(shape, lengths, strict=True)
    ]
    xx, yy, zz = np.meshgrid(*axes, indexing="ij")
    values = np.sin(2.0 * np.pi * xx / lengths[0])
    values *= np.cos(2.0 * np.pi * yy / lengths[1])
    values *= np.cos(2.0 * np.pi * zz / lengths[2])
    eigenvalue = sum((2.0 * np.pi / length) ** 2 for length in lengths)

    result = PeriodicFourierBackend(grid, 0.4).apply(values.reshape(-1))

    assert grid.shape == shape
    assert grid.lengths == pytest.approx(lengths)
    np.testing.assert_allclose(
        result.reshape(shape),
        eigenvalue**0.4 * values,
        atol=3.0e-13,
    )


@pytest.mark.unit
def test_periodic_fourier_respects_global_grid_permutation() -> None:
    shape = (3, 4)
    permutation = np.asarray([8, 2, 9, 1, 4, 11, 0, 7, 5, 10, 3, 6])
    grid = PeriodicGridMap(
        shape=shape,
        lengths=(2.0, 3.0),
        origins=(-1.0, 4.0),
        global_to_flat=permutation,
    )
    canonical = np.arange(grid.size, dtype=np.float64)
    coefficients = canonical[permutation]
    backend = PeriodicFourierBackend(grid, 0.5)
    canonical_result = PeriodicFourierBackend(
        identity_grid(shape, grid.lengths),
        0.5,
    ).apply(canonical)

    np.testing.assert_allclose(
        backend.apply(coefficients),
        canonical_result[permutation],
    )


@pytest.mark.unit
def test_periodic_fourier_rejects_wrong_vector_size() -> None:
    backend = PeriodicFourierBackend(identity_grid((4,), (1.0,)), 0.5)
    with pytest.raises(ValueError, match="wrong periodic-grid size"):
        backend.apply(np.zeros(5))


@pytest.mark.unit
def test_periodic_grid_rejects_malformed_cell_records() -> None:
    with pytest.raises(ValueError, match="at least one cell"):
        PeriodicGridMap.from_cell_records([], dimension=1, global_size=0)

    malformed = cell_records((3,))
    malformed[0] = (malformed[0][0][:-1], malformed[0][1][:-1])
    with pytest.raises(ValueError, match="each corner"):
        PeriodicGridMap.from_cell_records(
            malformed,
            dimension=1,
            global_size=3,
        )

    with pytest.raises(ValueError, match="at least two"):
        PeriodicGridMap.from_cell_records(
            cell_records((1,)),
            dimension=1,
            global_size=1,
        )

    nonfinite = cell_records((3,))
    nonfinite[0][1][0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        PeriodicGridMap.from_cell_records(
            nonfinite,
            dimension=1,
            global_size=3,
        )

    nonuniform = cell_records((3,))
    for _, coordinates in nonuniform:
        axis_values = coordinates[:, 0]
        axis_values[np.isclose(axis_values, 1.0)] = 0.7
    with pytest.raises(ValueError, match="uniform"):
        PeriodicGridMap.from_cell_records(
            nonuniform,
            dimension=1,
            global_size=3,
        )

    incomplete = cell_records((3,))
    incomplete.pop(1)
    with pytest.raises(ValueError, match="complete nonoverlapping"):
        PeriodicGridMap.from_cell_records(
            incomplete,
            dimension=1,
            global_size=3,
        )

    with pytest.raises(ValueError, match="fully periodic"):
        PeriodicGridMap.from_cell_records(
            cell_records((3,)),
            dimension=1,
            global_size=4,
        )


@pytest.mark.unit
def test_periodic_grid_rejects_malformed_topology() -> None:
    nonrectangular = cell_records((3, 2))
    nonrectangular[0][1][-1, 0] = 2.0
    with pytest.raises(ValueError, match="axis-aligned"):
        PeriodicGridMap.from_cell_records(
            nonrectangular,
            dimension=2,
            global_size=6,
        )

    missing_corner = cell_records((3, 2))
    missing_corner[0][1][-1] = missing_corner[0][1][0]
    with pytest.raises(ValueError, match="every tensor-product corner"):
        PeriodicGridMap.from_cell_records(
            missing_corner,
            dimension=2,
            global_size=6,
        )

    overlapping = cell_records((3, 2))
    overlapping[-1] = (
        overlapping[0][0].copy(),
        overlapping[0][1].copy(),
    )
    with pytest.raises(ValueError, match="overlapping Cartesian"):
        PeriodicGridMap.from_cell_records(
            overlapping,
            dimension=2,
            global_size=6,
        )

    inconsistent = cell_records((4,))
    inconsistent[1][0][0] = 0
    with pytest.raises(ValueError, match="inconsistent wrapped"):
        PeriodicGridMap.from_cell_records(
            inconsistent,
            dimension=1,
            global_size=4,
        )

    incomplete_numbering = cell_records((4,))
    for dofs, _ in incomplete_numbering:
        dofs[dofs == 3] = 9
    with pytest.raises(ValueError, match="numbering is incomplete"):
        PeriodicGridMap.from_cell_records(
            incomplete_numbering,
            dimension=1,
            global_size=4,
        )
