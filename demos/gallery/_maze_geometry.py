"""Deterministic square-cell maze geometry for the spatial gallery."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

Cell = tuple[int, int]
Rectangle = tuple[float, float, float, float]


@dataclass(frozen=True)
class MazeGeometry:
    """Rooms, passages, endpoints, and the unique graph route of a maze."""

    rooms: tuple[Rectangle, ...]
    passages: tuple[Rectangle, ...]
    start: np.ndarray
    goal: np.ndarray
    route: np.ndarray


def _farthest(
    adjacency: dict[Cell, list[Cell]],
    origin: Cell,
) -> tuple[Cell, dict[Cell, Cell | None]]:
    parents: dict[Cell, Cell | None] = {origin: None}
    distances = {origin: 0}
    queue = deque((origin,))
    while queue:
        current = queue.popleft()
        for neighbour in adjacency[current]:
            if neighbour in parents:
                continue
            parents[neighbour] = current
            distances[neighbour] = distances[current] + 1
            queue.append(neighbour)
    endpoint = max(distances, key=distances.__getitem__)
    return endpoint, parents


def square_cell_maze(
    columns: int = 7,
    rows: int = 5,
    *,
    seed: int = 19,
    room_width: float = 0.48,
    passage_width: float = 0.28,
) -> MazeGeometry:
    """Generate a perfect maze as a union of rooms and narrow passages."""
    if columns < 2 or rows < 2:
        raise ValueError("maze needs at least two rows and columns")
    if not 0.0 < passage_width < room_width < 1.0:
        raise ValueError("maze widths must satisfy 0 < passage < room < 1")

    generator = np.random.default_rng(seed)
    origin = (0, 0)
    edges: list[tuple[Cell, Cell]] = []
    adjacency = {
        (column, row): []
        for row in range(rows)
        for column in range(columns)
    }
    parent = {cell: cell for cell in adjacency}

    def root(cell: Cell) -> Cell:
        while parent[cell] != cell:
            parent[cell] = parent[parent[cell]]
            cell = parent[cell]
        return cell

    candidates = [
        ((column, row), (column + 1, row))
        for row in range(rows)
        for column in range(columns - 1)
    ]
    candidates.extend(
        ((column, row), (column, row + 1))
        for row in range(rows - 1)
        for column in range(columns)
    )
    generator.shuffle(candidates)
    for left, right in candidates:
        left_root = root(left)
        right_root = root(right)
        if left_root == right_root:
            continue
        parent[right_root] = left_root
        edges.append((left, right))
        adjacency[left].append(right)
        adjacency[right].append(left)
        if len(edges) == columns * rows - 1:
            break

    first, _ = _farthest(adjacency, origin)
    second, parents = _farthest(adjacency, first)
    route = [second]
    while route[-1] != first:
        parent = parents[route[-1]]
        if parent is None:
            raise RuntimeError("maze route is disconnected")
        route.append(parent)
    route.reverse()

    x_shift = 0.5 * (columns - 1)
    y_shift = 0.5 * (rows - 1)

    def centre(cell: Cell) -> np.ndarray:
        return np.asarray((cell[0] - x_shift, cell[1] - y_shift), dtype=float)

    half_room = 0.5 * room_width
    rooms = tuple(
        (
            float(point[0] - half_room),
            float(point[1] - half_room),
            room_width,
            room_width,
        )
        for point in (
            centre((column, row))
            for row in range(rows)
            for column in range(columns)
        )
    )
    passages = []
    for left, right in edges:
        first_point = centre(left)
        second_point = centre(right)
        if left[0] != right[0]:
            passages.append(
                (
                    float(min(first_point[0], second_point[0]) - half_room),
                    float(first_point[1] - 0.5 * passage_width),
                    1.0 + room_width,
                    passage_width,
                )
            )
        else:
            passages.append(
                (
                    float(first_point[0] - 0.5 * passage_width),
                    float(min(first_point[1], second_point[1]) - half_room),
                    passage_width,
                    1.0 + room_width,
                )
            )
    return MazeGeometry(
        rooms=rooms,
        passages=tuple(passages),
        start=centre(first),
        goal=centre(second),
        route=np.asarray([centre(cell) for cell in route]),
    )
