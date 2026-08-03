"""Logical periodic grids and their Fourier fractional powers."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Any

import numpy as np


def _cluster_levels(values: np.ndarray) -> np.ndarray:
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    if ordered.size == 0 or not np.all(np.isfinite(ordered)):
        raise ValueError("periodic mesh coordinates must be finite")
    scale = max(1.0, float(np.ptp(ordered)))
    tolerance = 128.0 * np.finfo(np.float64).eps * scale
    groups: list[list[float]] = [[float(ordered[0])]]
    for value in ordered[1:]:
        if abs(float(value) - groups[-1][-1]) <= tolerance:
            groups[-1].append(float(value))
        else:
            groups.append([float(value)])
    return np.asarray([np.mean(group) for group in groups])


def _level_index(value: float, levels: np.ndarray, tolerance: float) -> int:
    index = int(np.argmin(np.abs(levels - value)))
    if abs(float(levels[index]) - value) > tolerance:
        raise ValueError("cell vertex does not lie on the reconstructed Cartesian grid")
    return index


@dataclass(frozen=True)
class PeriodicGridMap:
    """Map an owned range of Firedrake dofs onto a periodic grid."""

    shape: tuple[int, ...]
    lengths: tuple[float, ...]
    origins: tuple[float, ...]
    global_to_flat: np.ndarray
    global_offset: int = 0

    @property
    def size(self) -> int:
        return prod(self.shape)

    @property
    def spacing(self) -> tuple[float, ...]:
        return tuple(
            length / count
            for length, count in zip(self.lengths, self.shape, strict=True)
        )

    @classmethod
    def from_space(cls, space: Any) -> PeriodicGridMap:
        """Validate a Firedrake space and reconstruct its logical grid."""
        mesh = space.mesh()
        dimension = int(mesh.geometric_dimension)
        field_map = np.asarray(space.cell_node_map().values, dtype=np.int64)
        coordinate_space = mesh.coordinates.function_space()
        coordinate_map = np.asarray(
            coordinate_space.cell_node_map().values,
            dtype=np.int64,
        )
        owned_cells = int(mesh.cell_set.size)
        field_map = field_map[:owned_cells]
        coordinate_map = coordinate_map[:owned_cells]
        if field_map.shape != coordinate_map.shape:
            raise ValueError(
                "field and coordinate nodes do not define the same tensor grid"
            )
        local_to_global = np.asarray(space.dof_dset.lgmap.indices, dtype=np.int64)
        coordinates = np.asarray(
            mesh.coordinates.dat.data_ro_with_halos,
            dtype=np.float64,
        )
        if dimension == 1 and coordinates.ndim == 1:
            coordinates = coordinates[:, np.newaxis]
        local_records = [
            (
                local_to_global[field_cell].copy(),
                coordinates[coordinate_cell].copy(),
            )
            for field_cell, coordinate_cell in zip(
                field_map,
                coordinate_map,
                strict=True,
            )
        ]
        records = [
            record
            for rank_records in mesh.comm.allgather(local_records)
            for record in rank_records
        ]
        reconstructed = cls.from_cell_records(
            records,
            dimension=dimension,
            global_size=int(space.dim()),
        )
        owned_size = int(space.dof_dset.size)
        owned_global = local_to_global[:owned_size]
        owned_start = int(mesh.comm.exscan(owned_size) or 0)
        owned_end = owned_start + owned_size
        if owned_global.size:
            if not np.array_equal(
                owned_global,
                np.arange(owned_start, owned_end),
            ):
                raise ValueError(
                    "periodic field ownership must use contiguous global dofs"
                )
        return cls(
            shape=reconstructed.shape,
            lengths=reconstructed.lengths,
            origins=reconstructed.origins,
            global_to_flat=reconstructed.global_to_flat[
                owned_start:owned_end
            ].copy(),
            global_offset=owned_start,
        )

    def owned_flat_indices(self, start: int, end: int) -> np.ndarray:
        """Return grid indices for the stored contiguous global range."""
        local_start = start - self.global_offset
        local_end = end - self.global_offset
        if local_start != 0 or local_end != self.global_to_flat.size:
            raise ValueError(
                "periodic grid map does not match the PETSc ownership range"
            )
        return self.global_to_flat

    @classmethod
    def from_cell_records(
        cls,
        records: list[tuple[np.ndarray, np.ndarray]],
        *,
        dimension: int,
        global_size: int,
    ) -> PeriodicGridMap:
        """Validate gathered cell dofs and coordinates as a periodic grid."""
        if not records:
            raise ValueError("periodic mesh must contain at least one cell")
        vertex_count = 2**dimension
        for field_cell, cell_coordinates in records:
            if field_cell.shape != (vertex_count,) or cell_coordinates.shape != (
                vertex_count,
                dimension,
            ):
                raise ValueError(
                    "periodic grid cells must have one degree of freedom at each corner"
                )

        all_coordinates = np.concatenate([record[1] for record in records])
        levels = tuple(
            _cluster_levels(all_coordinates[:, axis]) for axis in range(dimension)
        )
        shape = tuple(len(axis_levels) - 1 for axis_levels in levels)
        if any(count < 2 for count in shape):
            raise ValueError(
                "periodic grid requires at least two reconstructible uniform cells "
                "in every direction; Firedrake periodic meshes need at least three "
                "input cells per direction"
            )
        origins = tuple(float(axis_levels[0]) for axis_levels in levels)
        lengths = tuple(
            float(axis_levels[-1] - axis_levels[0]) for axis_levels in levels
        )
        tolerances = tuple(
            512.0 * np.finfo(np.float64).eps * max(1.0, length)
            for length in lengths
        )
        for axis_levels, tolerance in zip(levels, tolerances, strict=True):
            widths = np.diff(axis_levels)
            if not np.allclose(
                widths,
                widths[0],
                rtol=1.0e-12,
                atol=tolerance,
            ):
                raise ValueError(
                    "periodic mesh spacing must be uniform in every direction"
                )

        expected_cells = prod(shape)
        if len(records) != expected_cells:
            raise ValueError(
                "mesh must be a complete nonoverlapping Cartesian periodic grid"
            )
        if global_size != expected_cells:
            raise ValueError(
                "mesh must be fully periodic in every coordinate direction"
            )

        global_to_grid: dict[int, tuple[int, ...]] = {}
        logical_cells: set[tuple[int, ...]] = set()
        for field_cell, cell_coordinates in records:
            grid_corners = []
            raw_corners = []
            for coordinate in cell_coordinates:
                raw = tuple(
                    _level_index(
                        float(coordinate[axis]),
                        levels[axis],
                        tolerances[axis],
                    )
                    for axis in range(dimension)
                )
                raw_corners.append(raw)
                grid_corners.append(
                    tuple(raw[axis] % shape[axis] for axis in range(dimension))
                )
            axis_indices = tuple(
                sorted({corner[axis] for corner in raw_corners})
                for axis in range(dimension)
            )
            if any(
                len(indices) != 2 or indices[1] != indices[0] + 1
                for indices in axis_indices
            ):
                raise ValueError(
                    "periodic mesh cells must be axis-aligned tensor-product cells"
                )
            expected_corners = {
                tuple(index)
                for index in np.ndindex(*(2,) * dimension)
            }
            normalized_corners = {
                tuple(
                    axis_indices[axis].index(raw[axis])
                    for axis in range(dimension)
                )
                for raw in raw_corners
            }
            if normalized_corners != expected_corners:
                raise ValueError(
                    "periodic mesh cells must contain every tensor-product corner"
                )
            lower = tuple(indices[0] for indices in axis_indices)
            if lower in logical_cells:
                raise ValueError("periodic mesh contains overlapping Cartesian cells")
            logical_cells.add(lower)
            for global_dof, grid_corner in zip(
                field_cell,
                grid_corners,
                strict=True,
            ):
                global_index = int(global_dof)
                previous = global_to_grid.setdefault(global_index, grid_corner)
                if previous != grid_corner:
                    raise ValueError(
                        "periodic degree of freedom has inconsistent wrapped "
                        "coordinates"
                    )

        expected_dofs = set(range(expected_cells))
        if set(global_to_grid) != expected_dofs:
            raise ValueError(
                "periodic mesh global degree-of-freedom numbering is incomplete"
            )
        if set(global_to_grid.values()) != set(np.ndindex(shape)):
            raise ValueError(
                "periodic mesh degrees of freedom do not map one-to-one onto the grid"
            )
        global_to_flat = np.asarray(
            [
                np.ravel_multi_index(global_to_grid[index], shape)
                for index in range(expected_cells)
            ],
            dtype=np.int64,
        )
        return cls(
            shape=shape,
            lengths=lengths,
            origins=origins,
            global_to_flat=global_to_flat,
        )


class PeriodicFourierBackend:
    """Serial real FFT action on a validated logical periodic grid."""

    def __init__(self, grid: PeriodicGridMap, order: float) -> None:
        self.grid = grid
        self.order = order
        spectral_shape = grid.shape[:-1] + (grid.shape[-1] // 2 + 1,)
        wave_number_squared = np.zeros(spectral_shape, dtype=np.float64)
        for axis, (count, length) in enumerate(
            zip(grid.shape, grid.lengths, strict=True)
        ):
            frequency = (
                np.fft.rfftfreq(count, d=length / count)
                if axis == len(grid.shape) - 1
                else np.fft.fftfreq(count, d=length / count)
            )
            reshape = [1] * len(grid.shape)
            reshape[axis] = frequency.size
            wave_number_squared += (
                2.0 * np.pi * frequency.reshape(reshape)
            ) ** 2
        self._multiplier = wave_number_squared**order
        self._multiplier[(0,) * len(grid.shape)] = 0.0
        self.applications = 0

    def apply(self, coefficients: np.ndarray) -> np.ndarray:
        values = np.asarray(coefficients, dtype=np.float64)
        if values.shape != (self.grid.size,):
            raise ValueError("coefficient vector has the wrong periodic-grid size")
        if self.grid.global_to_flat.size != self.grid.size:
            raise ValueError("serial periodic backend requires a complete grid map")
        grid_values = np.empty(self.grid.size, dtype=np.float64)
        grid_values[self.grid.global_to_flat] = values
        transformed = np.fft.rfftn(grid_values.reshape(self.grid.shape))
        result_grid = np.fft.irfftn(
            self._multiplier * transformed,
            s=self.grid.shape,
            axes=tuple(range(len(self.grid.shape))),
        )
        self.applications += 1
        return np.asarray(result_grid).reshape(-1)[self.grid.global_to_flat]

    def diagnostics(self) -> dict[str, Any]:
        return {
            "shape": self.grid.shape,
            "lengths": self.grid.lengths,
            "spacing": self.grid.spacing,
            "fft_backend": "numpy-serial",
            "applications": self.applications,
        }


def _balanced_ranges(size: int, num_ranks: int) -> tuple[tuple[int, int], ...]:
    """Partition a one-dimensional range into balanced contiguous pieces."""
    quotient, remainder = divmod(size, num_ranks)
    ranges = []
    start = 0
    for rank in range(num_ranks):
        width = quotient + (rank < remainder)
        ranges.append((start, start + width))
        start += width
    return tuple(ranges)


def _displacements(counts: np.ndarray) -> np.ndarray:
    """Return zero-based displacements for collective buffer counts."""
    result = np.zeros_like(counts)
    if counts.size > 1:
        result[1:] = np.cumsum(counts[:-1])
    return result


def _owners(indices: np.ndarray, ranges: tuple[tuple[int, int], ...]) -> np.ndarray:
    """Return the owning balanced range for each nonnegative index."""
    ends = np.asarray([end for _, end in ranges], dtype=np.int64)
    result = np.searchsorted(ends, indices, side="right")
    if np.any(result >= len(ranges)):
        raise ValueError("distributed index lies outside its ownership range")
    return np.asarray(result, dtype=np.int64)


class _RedistributionPlan:
    """Reusable all-to-all plan from local values to a contiguous target."""

    def __init__(
        self,
        comm: Any,
        target_indices: np.ndarray,
        target_owners: np.ndarray,
        *,
        target_start: int,
        target_size: int,
    ) -> None:
        from mpi4py import MPI

        self._comm = comm
        self._float_type = MPI.DOUBLE
        self._int_type = MPI.INT64_T
        num_ranks = int(comm.size)
        self._send_order = np.argsort(target_owners, kind="stable")
        self._send_counts = np.bincount(
            target_owners,
            minlength=num_ranks,
        ).astype(np.int64)
        self._send_displacements = _displacements(self._send_counts)
        self._receive_counts = np.empty(num_ranks, dtype=np.int64)
        comm.Alltoall(self._send_counts, self._receive_counts)
        self._receive_displacements = _displacements(self._receive_counts)

        ordered_targets = np.asarray(
            target_indices[self._send_order],
            dtype=np.int64,
        )
        received_targets = np.empty(
            int(np.sum(self._receive_counts)),
            dtype=np.int64,
        )
        comm.Alltoallv(
            [
                ordered_targets,
                self._send_counts,
                self._send_displacements,
                MPI.INT64_T,
            ],
            [
                received_targets,
                self._receive_counts,
                self._receive_displacements,
                MPI.INT64_T,
            ],
        )
        positions = received_targets - target_start
        if (
            positions.size != target_size
            or np.any(positions < 0)
            or np.any(positions >= target_size)
            or not np.array_equal(np.sort(positions), np.arange(target_size))
        ):
            raise ValueError(
                "distributed periodic-grid redistribution is not one-to-one"
            )
        self._receive_positions = positions
        self._target_size = target_size

    def apply(self, values: np.ndarray) -> np.ndarray:
        """Exchange local source values into target-contiguous order."""
        source = np.asarray(values, dtype=np.float64)
        return self._apply(source, np.float64, self._float_type)

    def apply_indices(self, values: np.ndarray) -> np.ndarray:
        """Exchange integer indices through the same routing plan."""
        source = np.asarray(values, dtype=np.int64)
        return self._apply(source, np.int64, self._int_type)

    def _apply(
        self,
        source: np.ndarray,
        dtype: Any,
        mpi_type: Any,
    ) -> np.ndarray:
        if source.size != self._send_order.size:
            raise ValueError("local coefficient vector has the wrong size")
        ordered = np.ascontiguousarray(source[self._send_order], dtype=dtype)
        received = np.empty(
            int(np.sum(self._receive_counts)),
            dtype=dtype,
        )
        self._comm.Alltoallv(
            [
                ordered,
                self._send_counts,
                self._send_displacements,
                mpi_type,
            ],
            [
                received,
                self._receive_counts,
                self._receive_displacements,
                mpi_type,
            ],
        )
        result = np.empty(self._target_size, dtype=dtype)
        result[self._receive_positions] = received
        return result


class DistributedPeriodicFourierBackend:
    """Slab-distributed complex FFT using local NumPy transforms and MPI."""

    def __init__(
        self,
        grid: PeriodicGridMap,
        order: float,
        comm: Any,
        ownership_range: tuple[int, int],
    ) -> None:
        from mpi4py import MPI

        self.grid = grid
        self.order = order
        self._comm = comm
        self._rank = int(comm.rank)
        self._num_ranks = int(comm.size)
        self._complex_type = MPI.DOUBLE_COMPLEX
        self._tail_size = prod(grid.shape[1:])
        self._row_ranges = _balanced_ranges(
            grid.shape[0],
            self._num_ranks,
        )
        self._column_ranges = _balanced_ranges(
            self._tail_size,
            self._num_ranks,
        )
        self._row_start, self._row_end = self._row_ranges[self._rank]
        self._column_start, self._column_end = self._column_ranges[self._rank]
        self._local_rows = self._row_end - self._row_start
        self._local_columns = self._column_end - self._column_start
        self._ownership_start, self._ownership_end = ownership_range
        ownership_ranges = tuple(comm.allgather(ownership_range))

        local_global = np.arange(
            self._ownership_start,
            self._ownership_end,
            dtype=np.int64,
        )
        local_flat = grid.owned_flat_indices(
            self._ownership_start,
            self._ownership_end,
        )
        input_rows = local_flat // self._tail_size
        self._input_plan = _RedistributionPlan(
            comm,
            local_flat,
            _owners(input_rows, self._row_ranges),
            target_start=self._row_start * self._tail_size,
            target_size=self._local_rows * self._tail_size,
        )

        output_global = self._input_plan.apply_indices(local_global)
        ownership_ends = tuple(end for _, end in ownership_ranges)
        output_owners = np.searchsorted(
            np.asarray(ownership_ends, dtype=np.int64),
            output_global,
            side="right",
        )
        self._output_plan = _RedistributionPlan(
            comm,
            output_global,
            np.asarray(output_owners, dtype=np.int64),
            target_start=self._ownership_start,
            target_size=self._ownership_end - self._ownership_start,
        )

        row_counts = np.asarray(
            [end - start for start, end in self._row_ranges],
            dtype=np.int64,
        )
        column_counts = np.asarray(
            [end - start for start, end in self._column_ranges],
            dtype=np.int64,
        )
        self._slab_send_counts = self._local_rows * column_counts
        self._slab_send_displacements = _displacements(
            self._slab_send_counts
        )
        self._slab_receive_counts = row_counts * self._local_columns
        self._slab_receive_displacements = _displacements(
            self._slab_receive_counts
        )
        self._column_send_counts = row_counts * self._local_columns
        self._column_send_displacements = _displacements(
            self._column_send_counts
        )
        self._column_receive_counts = self._local_rows * column_counts
        self._column_receive_displacements = _displacements(
            self._column_receive_counts
        )
        self._multiplier = self._local_multiplier()
        self.applications = 0

    def _local_multiplier(self) -> np.ndarray:
        frequencies = tuple(
            2.0 * np.pi * np.fft.fftfreq(count, d=length / count)
            for count, length in zip(
                self.grid.shape,
                self.grid.lengths,
                strict=True,
            )
        )
        first_squared = frequencies[0] ** 2
        if len(self.grid.shape) == 1:
            tail_squared = np.zeros(1, dtype=np.float64)
        else:
            tail_grids = np.meshgrid(*frequencies[1:], indexing="ij")
            tail_squared = sum(grid**2 for grid in tail_grids).reshape(-1)
        local_tail = tail_squared[self._column_start : self._column_end]
        multiplier = (
            first_squared[:, np.newaxis] + local_tail[np.newaxis, :]
        ) ** self.order
        if self._column_start == 0 and self._local_columns:
            multiplier[0, 0] = 0.0
        return multiplier

    def _slabs_to_columns(self, slabs: np.ndarray) -> np.ndarray:
        send_parts = [
            np.ascontiguousarray(slabs[:, start:end]).reshape(-1)
            for start, end in self._column_ranges
        ]
        send = (
            np.concatenate(send_parts)
            if send_parts
            else np.empty(0, dtype=np.complex128)
        )
        received = np.empty(
            int(np.sum(self._slab_receive_counts)),
            dtype=np.complex128,
        )
        self._comm.Alltoallv(
            [
                send,
                self._slab_send_counts,
                self._slab_send_displacements,
                self._complex_type,
            ],
            [
                received,
                self._slab_receive_counts,
                self._slab_receive_displacements,
                self._complex_type,
            ],
        )
        columns = np.empty(
            (self.grid.shape[0], self._local_columns),
            dtype=np.complex128,
        )
        for source, (start, end) in enumerate(self._row_ranges):
            count = int(self._slab_receive_counts[source])
            offset = int(self._slab_receive_displacements[source])
            columns[start:end, :] = received[offset : offset + count].reshape(
                end - start,
                self._local_columns,
            )
        return columns

    def _columns_to_slabs(self, columns: np.ndarray) -> np.ndarray:
        send_parts = [
            np.ascontiguousarray(columns[start:end, :]).reshape(-1)
            for start, end in self._row_ranges
        ]
        send = (
            np.concatenate(send_parts)
            if send_parts
            else np.empty(0, dtype=np.complex128)
        )
        received = np.empty(
            int(np.sum(self._column_receive_counts)),
            dtype=np.complex128,
        )
        self._comm.Alltoallv(
            [
                send,
                self._column_send_counts,
                self._column_send_displacements,
                self._complex_type,
            ],
            [
                received,
                self._column_receive_counts,
                self._column_receive_displacements,
                self._complex_type,
            ],
        )
        slabs = np.empty(
            (self._local_rows, self._tail_size),
            dtype=np.complex128,
        )
        for source, (start, end) in enumerate(self._column_ranges):
            count = int(self._column_receive_counts[source])
            offset = int(self._column_receive_displacements[source])
            slabs[:, start:end] = received[offset : offset + count].reshape(
                self._local_rows,
                end - start,
            )
        return slabs

    def apply(self, local_coefficients: np.ndarray) -> np.ndarray:
        """Apply the Fourier multiplier without replicating the global field."""
        local_grid = self._input_plan.apply(local_coefficients).reshape(
            (self._local_rows, *self.grid.shape[1:])
        )
        transformed = np.asarray(local_grid, dtype=np.complex128)
        if len(self.grid.shape) > 1:
            transformed = np.fft.fftn(
                transformed,
                axes=tuple(range(1, len(self.grid.shape))),
            )
        columns = self._slabs_to_columns(
            transformed.reshape(self._local_rows, self._tail_size)
        )
        columns = np.fft.fft(columns, axis=0)
        columns *= self._multiplier
        columns = np.fft.ifft(columns, axis=0)
        result_grid = self._columns_to_slabs(columns).reshape(
            (self._local_rows, *self.grid.shape[1:])
        )
        if len(self.grid.shape) > 1:
            result_grid = np.fft.ifftn(
                result_grid,
                axes=tuple(range(1, len(self.grid.shape))),
            )
        self.applications += 1
        return self._output_plan.apply(
            np.asarray(result_grid.real, dtype=np.float64).reshape(-1)
        )

    def diagnostics(self) -> dict[str, Any]:
        return {
            "shape": self.grid.shape,
            "lengths": self.grid.lengths,
            "spacing": self.grid.spacing,
            "fft_backend": "numpy-mpi-slab",
            "decomposition": "slab-alltoallv",
            "replicated_grid_map": False,
            "local_grid_map_values": self.grid.global_to_flat.size,
            "local_real_values": self._local_rows * self._tail_size,
            "local_fourier_values": self.grid.shape[0]
            * self._local_columns,
            "applications": self.applications,
        }
