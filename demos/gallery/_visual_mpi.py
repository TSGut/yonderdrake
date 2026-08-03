"""Small MPI gathering helpers for Firedrake visual demos."""

from __future__ import annotations

from typing import Any

import numpy as np


def gather_p1_animation_data(
    mesh: Any,
    histories: dict[str, np.ndarray],
    communicator: Any,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]] | None:
    """Gather distributed CG1 histories and triangles onto rank zero."""
    owned_coordinates = np.asarray(
        mesh.coordinates.dat.data_ro,
        dtype=float,
    ).copy()
    halo_coordinates = np.asarray(
        mesh.coordinates.dat.data_ro_with_halos,
        dtype=float,
    )
    local_cells = np.asarray(
        mesh.coordinates.function_space().cell_node_map().values,
        dtype=np.int32,
    )
    triangle_coordinates = halo_coordinates[local_cells].copy()
    if any(
        values.shape[1] != owned_coordinates.shape[0]
        for values in histories.values()
    ):
        raise RuntimeError("CG1 history and coordinate ownership do not align")

    gathered_owned = communicator.gather(owned_coordinates, root=0)
    gathered_triangles = communicator.gather(triangle_coordinates, root=0)
    gathered_histories = {
        name: communicator.gather(values, root=0)
        for name, values in histories.items()
    }
    if communicator.rank != 0:
        return None

    assert gathered_owned is not None
    assert gathered_triangles is not None
    all_triangle_coordinates = np.concatenate(
        gathered_triangles,
        axis=0,
    )
    rounded_vertices = np.round(
        all_triangle_coordinates.reshape(-1, 2),
        decimals=12,
    )
    coordinates, inverse = np.unique(
        rounded_vertices,
        axis=0,
        return_inverse=True,
    )
    cells = inverse.reshape(-1, 3).astype(np.int32)
    canonical_cells = np.sort(cells, axis=1)
    _, first_indices = np.unique(
        canonical_cells,
        axis=0,
        return_index=True,
    )
    cells = cells[np.sort(first_indices)]

    coordinate_to_index = {
        tuple(coordinate): index
        for index, coordinate in enumerate(np.round(coordinates, 12))
    }
    global_histories: dict[str, np.ndarray] = {}
    for name, rank_histories in gathered_histories.items():
        assert rank_histories is not None
        number_of_frames = rank_histories[0].shape[0]
        global_values = np.empty(
            (number_of_frames, coordinates.shape[0]),
            dtype=rank_histories[0].dtype,
        )
        assigned = np.zeros(coordinates.shape[0], dtype=bool)
        for rank_coordinates, rank_values in zip(
            gathered_owned,
            rank_histories,
            strict=True,
        ):
            indices = np.fromiter(
                (
                    coordinate_to_index[tuple(coordinate)]
                    for coordinate in np.round(rank_coordinates, 12)
                ),
                dtype=np.int64,
                count=rank_coordinates.shape[0],
            )
            global_values[:, indices] = rank_values
            assigned[indices] = True
        if not np.all(assigned):
            raise RuntimeError(
                f"MPI gather missed {np.count_nonzero(~assigned)} CG1 nodes"
            )
        global_histories[name] = global_values
    return coordinates, cells, global_histories
