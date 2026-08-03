"""Portable CSV export for replotting visual demos without Firedrake."""

from __future__ import annotations

import csv
import gzip
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PlotData:
    """Triangulated fields and metadata loaded from a demo CSV."""

    metadata: dict[str, Any]
    times: np.ndarray
    coordinates: np.ndarray
    cells: np.ndarray
    fields: dict[str, np.ndarray]


def _text(value: Any) -> str:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, separators=(",", ":"))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.12g}"
    return str(value)


def _value(text: str) -> Any:
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return text


def load_plot_csv(path: Path) -> PlotData:
    """Load the mesh, fields, times, and metadata written by ``save_plot_csv``."""
    opener = gzip.open if path.suffix == ".gz" else open
    metadata: dict[str, Any] = {}
    coordinates: dict[int, tuple[float, ...]] = {}
    cells: dict[int, tuple[int, int, int]] = {}
    field_rows: dict[int, dict[int, dict[str, float]]] = {}
    frame_times: dict[int, float] = {}
    with opener(path, "rt", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no CSV header")
        field_names = tuple(
            column.removeprefix("field:")
            for column in reader.fieldnames
            if column.startswith("field:")
        )
        for row in reader:
            record_type = row["record_type"]
            if record_type == "metadata":
                metadata[row["name"]] = _value(row["value"])
            elif record_type == "vertex":
                coordinate = (float(row["x"]), float(row["y"]))
                if row["z"]:
                    coordinate += (float(row["z"]),)
                coordinates[int(row["index"])] = coordinate
            elif record_type == "cell":
                cells[int(row["index"])] = (
                    int(row["v0"]),
                    int(row["v1"]),
                    int(row["v2"]),
                )
            elif record_type == "field":
                frame = int(row["frame"])
                vertex = int(row["index"])
                frame_times[frame] = float(row["time"])
                field_rows.setdefault(frame, {})[vertex] = {
                    name: float(row[f"field:{name}"])
                    for name in field_names
                }

    vertex_indices = sorted(coordinates)
    cell_indices = sorted(cells)
    frame_indices = sorted(field_rows)
    if vertex_indices != list(range(len(vertex_indices))):
        raise ValueError(f"{path} contains non-contiguous vertex indices")
    if cell_indices != list(range(len(cell_indices))):
        raise ValueError(f"{path} contains non-contiguous cell indices")
    if frame_indices != list(range(len(frame_indices))):
        raise ValueError(f"{path} contains non-contiguous frame indices")
    vertex_count = len(vertex_indices)
    fields = {
        name: np.asarray(
            [
                [
                    field_rows[frame][vertex][name]
                    for vertex in range(vertex_count)
                ]
                for frame in frame_indices
            ],
            dtype=float,
        )
        for name in field_names
    }
    return PlotData(
        metadata=metadata,
        times=np.asarray(
            [frame_times[frame] for frame in frame_indices],
            dtype=float,
        ),
        coordinates=np.asarray(
            [coordinates[index] for index in vertex_indices],
            dtype=float,
        ),
        cells=np.asarray(
            [cells[index] for index in cell_indices],
            dtype=np.int64,
        ),
        fields=fields,
    )


def save_plot_csv(
    path: Path,
    *,
    times: Sequence[float],
    coordinates: np.ndarray,
    cells: np.ndarray,
    fields: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
    vertex_labels: np.ndarray | None = None,
    anatomy: tuple[np.ndarray, np.ndarray] | None = None,
    sources: np.ndarray | None = None,
    series: Mapping[str, tuple[np.ndarray, np.ndarray]] | None = None,
) -> Path:
    """Write replot-ready simulation data to compressed CSV."""
    coordinates = np.asarray(coordinates)
    cells = np.asarray(cells)
    times_array = np.asarray(times, dtype=float)
    field_arrays = {
        name: np.asarray(values)
        for name, values in fields.items()
    }
    if coordinates.ndim != 2 or coordinates.shape[1] not in (2, 3):
        raise ValueError("coordinates must have shape (n, 2) or (n, 3)")
    if cells.ndim != 2 or cells.shape[1] != 3:
        raise ValueError("cells must have shape (m, 3)")
    expected_shape = (times_array.size, coordinates.shape[0])
    for name, values in field_arrays.items():
        if values.shape != expected_shape:
            raise ValueError(
                f"field {name!r} has shape {values.shape}; "
                f"expected {expected_shape}"
            )
    if vertex_labels is not None:
        vertex_labels = np.asarray(vertex_labels)
        if vertex_labels.shape != (coordinates.shape[0],):
            raise ValueError("vertex_labels must contain one value per vertex")

    field_columns = tuple(f"field:{name}" for name in field_arrays)
    header = (
        "record_type",
        "name",
        "index",
        "frame",
        "time",
        "x",
        "y",
        "z",
        "v0",
        "v1",
        "v2",
        "label",
        "frequency",
        "phase",
        "value",
        *field_columns,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    blank_fields = ("",) * len(field_columns)
    with gzip.open(
        temporary,
        "wt",
        encoding="utf-8",
        newline="",
        compresslevel=5,
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        for key, value in metadata.items():
            writer.writerow(
                (
                    "metadata",
                    key,
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    _text(value),
                    *blank_fields,
                )
            )
        for index, coordinate in enumerate(coordinates):
            z = coordinate[2] if coordinate.shape[0] == 3 else ""
            label = "" if vertex_labels is None else _text(vertex_labels[index])
            writer.writerow(
                (
                    "vertex",
                    "",
                    index,
                    "",
                    "",
                    f"{coordinate[0]:.12g}",
                    f"{coordinate[1]:.12g}",
                    _text(z),
                    "",
                    "",
                    "",
                    label,
                    "",
                    "",
                    "",
                    *blank_fields,
                )
            )
        for index, cell in enumerate(cells):
            writer.writerow(
                (
                    "cell",
                    "",
                    index,
                    "",
                    "",
                    "",
                    "",
                    "",
                    int(cell[0]),
                    int(cell[1]),
                    int(cell[2]),
                    "",
                    "",
                    "",
                    "",
                    *blank_fields,
                )
            )
        if anatomy is not None:
            anatomy_coordinates, anatomy_labels = anatomy
            anatomy_coordinates = np.asarray(anatomy_coordinates)
            anatomy_labels = np.asarray(anatomy_labels)
            for index, (coordinate, label) in enumerate(
                zip(anatomy_coordinates, anatomy_labels, strict=True)
            ):
                writer.writerow(
                    (
                        "anatomy",
                        "",
                        index,
                        "",
                        "",
                        f"{coordinate[0]:.12g}",
                        f"{coordinate[1]:.12g}",
                        "",
                        "",
                        "",
                        "",
                        _text(label),
                        "",
                        "",
                        "",
                        *blank_fields,
                    )
                )
        if sources is not None:
            for index, source in enumerate(np.asarray(sources)):
                writer.writerow(
                    (
                        "source",
                        "",
                        index,
                        "",
                        "",
                        f"{source[0]:.12g}",
                        f"{source[1]:.12g}",
                        "",
                        "",
                        "",
                        "",
                        "",
                        f"{source[2]:.12g}" if source.size > 2 else "",
                        f"{source[3]:.12g}" if source.size > 3 else "",
                        "",
                        *blank_fields,
                    )
                )
        if series is not None:
            for name, (abscissae, values) in series.items():
                for index, (abscissa, value) in enumerate(
                    zip(abscissae, values, strict=True)
                ):
                    writer.writerow(
                        (
                            "series",
                            name,
                            index,
                            "",
                            "",
                            f"{float(abscissa):.12g}",
                            "",
                            "",
                            "",
                            "",
                            "",
                            "",
                            "",
                            "",
                            f"{float(value):.12g}",
                            *blank_fields,
                        )
                    )
        field_names = tuple(field_arrays)
        for frame, time_value in enumerate(times_array):
            values_at_time = tuple(
                field_arrays[name][frame] for name in field_names
            )
            writer.writerows(
                (
                    "field",
                    "",
                    vertex,
                    frame,
                    f"{time_value:.12g}",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    *(f"{values[vertex]:.12g}" for values in values_at_time),
                )
                for vertex in range(coordinates.shape[0])
            )
    temporary.replace(path)
    return path
