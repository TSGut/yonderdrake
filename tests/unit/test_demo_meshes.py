"""Quality checks for the committed visual-demo meshes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

MESH_DIRECTORY = Path(__file__).resolve().parents[2] / "demos" / "gallery" / "meshes"
MESH_NAMES = (
    "dragon.msh",
    "koch-snowflake.msh",
    "koch-snowflake-smoke.msh",
    "aperiodic-monotile.msh",
    "fractional-maze.msh",
    "fractional-maze-smoke.msh",
)


def _read_msh2_triangles(path: Path) -> tuple[np.ndarray, np.ndarray]:
    lines = path.read_text(encoding="utf-8").splitlines()
    node_start = lines.index("$Nodes")
    node_count = int(lines[node_start + 1])
    node_lines = lines[node_start + 2 : node_start + 2 + node_count]
    coordinates = np.asarray(
        [[float(value) for value in line.split()[1:3]] for line in node_lines]
    )

    element_start = lines.index("$Elements")
    element_count = int(lines[element_start + 1])
    element_lines = lines[element_start + 2 : element_start + 2 + element_count]
    triangles = []
    for line in element_lines:
        values = [int(value) for value in line.split()]
        if values[1] != 2:
            continue
        tag_count = values[2]
        triangles.append([node - 1 for node in values[3 + tag_count : 6 + tag_count]])
    return coordinates, np.asarray(triangles, dtype=np.int32)


@pytest.mark.unit
@pytest.mark.parametrize("name", MESH_NAMES)
def test_demo_mesh_has_no_degenerate_triangles(name: str) -> None:
    coordinates, cells = _read_msh2_triangles(MESH_DIRECTORY / name)
    first = coordinates[cells[:, 1]] - coordinates[cells[:, 0]]
    second = coordinates[cells[:, 2]] - coordinates[cells[:, 0]]
    twice_area = np.abs(first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0])
    edge_square_sum = (
        np.sum(np.square(coordinates[cells[:, 1]] - coordinates[cells[:, 0]]), axis=1)
        + np.sum(
            np.square(coordinates[cells[:, 2]] - coordinates[cells[:, 1]]),
            axis=1,
        )
        + np.sum(
            np.square(coordinates[cells[:, 0]] - coordinates[cells[:, 2]]),
            axis=1,
        )
    )
    quality = 2.0 * np.sqrt(3.0) * twice_area / edge_square_sum

    assert cells.size
    assert np.min(twice_area) > 1.0e-8
    assert np.min(quality) > 0.5
