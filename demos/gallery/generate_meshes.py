"""Regenerate the committed Gmsh meshes used by visual demos."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from _maze_geometry import square_cell_maze
from caputo_wismer.visual_time_derivative_dragon_wave import dragon_outline
from visual_fractional_heat_snowflake import koch_outline
from visual_fractional_schrodinger_monotile import hat_outline


@dataclass(frozen=True)
class MeshRecipe:
    name: str
    outline: np.ndarray
    minimum_spacing: float
    maximum_spacing: float
    transition_distance: float
    refinement_points: tuple[tuple[float, float], ...] = ()
    refine_boundary: bool = True


def _geo_source(recipe: MeshRecipe) -> str:
    points = [
        f"Point({index}) = {{{x:.16g}, {y:.16g}, 0, "
        f"{recipe.maximum_spacing:.16g}}};"
        for index, (x, y) in enumerate(recipe.outline, start=1)
    ]
    count = len(points)
    lines = [
        f"Line({index}) = {{{index}, {index % count + 1}}};"
        for index in range(1, count + 1)
    ]
    entity_tags = ", ".join(str(index) for index in range(1, count + 1))
    refinement_tags = tuple(
        count + index
        for index in range(1, len(recipe.refinement_points) + 1)
    )
    refinement_points = [
        f"Point({tag}) = {{{x:.16g}, {y:.16g}, 0, "
        f"{recipe.minimum_spacing:.16g}}};"
        for tag, (x, y) in zip(
            refinement_tags,
            recipe.refinement_points,
            strict=True,
        )
    ]
    settings = [
        'SetFactory("Built-in");',
        "Mesh.MshFileVersion = 2.2;",
        "Mesh.Binary = 0;",
        "Mesh.ElementOrder = 1;",
        "Mesh.Algorithm = 6;",
        "Mesh.Smoothing = 10;",
        "Mesh.Optimize = 1;",
        "Mesh.OptimizeNetgen = 1;",
        f"Mesh.MeshSizeMin = {recipe.minimum_spacing:.16g};",
        f"Mesh.MeshSizeMax = {recipe.maximum_spacing:.16g};",
        "Mesh.MeshSizeFromCurvature = 0;",
        "Mesh.MeshSizeExtendFromBoundary = 0;",
    ]
    topology = [
        f"Curve Loop(1) = {{{entity_tags}}};",
        "Plane Surface(1) = {1};",
        *(
            [
                "Point{" + ", ".join(map(str, refinement_tags)) + "} "
                "In Surface{1};"
            ]
            if refinement_tags
            else []
        ),
        f"Physical Curve(1) = {{{entity_tags}}};",
        "Physical Surface(1) = {1};",
    ]
    field_lines = []
    distance_fields = []
    next_field = 1
    if recipe.refine_boundary:
        field_lines.extend(
            (
                f"Field[{next_field}] = Distance;",
                f"Field[{next_field}].CurvesList = {{{entity_tags}}};",
                f"Field[{next_field}].Sampling = 1000;",
            )
        )
        distance_fields.append(next_field)
        next_field += 1
    if refinement_tags:
        point_tags = ", ".join(map(str, refinement_tags))
        field_lines.extend(
            (
                f"Field[{next_field}] = Distance;",
                f"Field[{next_field}].PointsList = {{{point_tags}}};",
            )
        )
        distance_fields.append(next_field)
        next_field += 1
    if len(distance_fields) > 1:
        field_lines.extend(
            (
                f"Field[{next_field}] = Min;",
                f"Field[{next_field}].FieldsList = "
                "{" + ", ".join(map(str, distance_fields)) + "};",
            )
        )
        distance_field = next_field
        next_field += 1
    else:
        distance_field = distance_fields[0]
    field_lines.extend(
        (
            f"Field[{next_field}] = Threshold;",
            f"Field[{next_field}].InField = {distance_field};",
            f"Field[{next_field}].SizeMin = {recipe.minimum_spacing:.16g};",
            f"Field[{next_field}].SizeMax = {recipe.maximum_spacing:.16g};",
            f"Field[{next_field}].DistMin = {recipe.minimum_spacing:.16g};",
            f"Field[{next_field}].DistMax = {recipe.transition_distance:.16g};",
            f"Background Field = {next_field};",
        )
    )
    return "\n".join(
        (
            *settings,
            "",
            *points,
            *refinement_points,
            "",
            *lines,
            "",
            *topology,
            "",
            *field_lines,
            "",
        )
    )


def _maze_geo_source(
    *,
    columns: int,
    rows: int,
    minimum_spacing: float,
    maximum_spacing: float,
) -> str:
    geometry = square_cell_maze(columns, rows)
    rectangles = (*geometry.rooms, *geometry.passages)
    surfaces = []
    for index, (x, y, width, height) in enumerate(rectangles, start=1):
        surfaces.append(
            f"Rectangle({index}) = "
            f"{{{x:.16g}, {y:.16g}, 0, {width:.16g}, {height:.16g}}};"
        )
    tool_tags = ", ".join(str(index) for index in range(2, len(rectangles) + 1))
    settings = [
        'SetFactory("OpenCASCADE");',
        "Mesh.MshFileVersion = 2.2;",
        "Mesh.Binary = 0;",
        "Mesh.ElementOrder = 1;",
        "Mesh.Algorithm = 6;",
        "Mesh.Smoothing = 10;",
        "Mesh.Optimize = 1;",
        "Mesh.OptimizeNetgen = 1;",
        f"Mesh.MeshSizeMin = {minimum_spacing:.16g};",
        f"Mesh.MeshSizeMax = {maximum_spacing:.16g};",
        "Mesh.MeshSizeFromCurvature = 0;",
        "Mesh.MeshSizeExtendFromBoundary = 1;",
    ]
    topology = [
        "maze[] = BooleanUnion{ Surface{1}; Delete; }"
        f"{{ Surface{{{tool_tags}}}; Delete; }};",
        "boundary[] = Boundary{ Surface{maze[]}; };",
        "MeshSize { PointsOf{ Surface{maze[]}; } } = "
        f"{minimum_spacing:.16g};",
        f"Point(10001) = {{{geometry.goal[0]:.16g}, "
        f"{geometry.goal[1]:.16g}, 0, {minimum_spacing:.16g}}};",
        "Point{10001} In Surface{maze[0]};",
        "Field[1] = Distance;",
        "Field[1].PointsList = {10001};",
        "Field[2] = Threshold;",
        "Field[2].InField = 1;",
        f"Field[2].SizeMin = {minimum_spacing:.16g};",
        f"Field[2].SizeMax = {maximum_spacing:.16g};",
        "Field[2].DistMin = 0.10;",
        "Field[2].DistMax = 1.20;",
        "Background Field = 2;",
        "Physical Surface(1) = {maze[]};",
        "Physical Curve(1) = {boundary[]};",
    ]
    return "\n".join((*settings, "", *surfaces, "", *topology, ""))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gmsh",
        type=Path,
        help="Gmsh executable; defaults to the executable on PATH",
    )
    parser.add_argument(
        "--only",
        choices=(
            "dragon",
            "koch-snowflake",
            "koch-snowflake-smoke",
            "aperiodic-monotile",
            "fractional-maze",
            "fractional-maze-smoke",
        ),
        help="regenerate one mesh instead of every gallery mesh",
    )
    args = parser.parse_args()
    gmsh = args.gmsh or (Path(found) if (found := shutil.which("gmsh")) else None)
    if gmsh is None:
        raise SystemExit(
            "Gmsh is required only to regenerate these committed mesh assets."
        )

    polygon_recipes = (
        MeshRecipe(
            "dragon",
            dragon_outline(),
            minimum_spacing=0.045,
            maximum_spacing=0.12,
            transition_distance=0.42,
            refinement_points=((1.82, 0.72),),
        ),
        MeshRecipe(
            "koch-snowflake",
            koch_outline(3),
            minimum_spacing=0.060,
            maximum_spacing=0.15,
            transition_distance=0.38,
        ),
        MeshRecipe(
            "koch-snowflake-smoke",
            koch_outline(1),
            minimum_spacing=0.22,
            maximum_spacing=0.46,
            transition_distance=0.55,
        ),
        MeshRecipe(
            "aperiodic-monotile",
            hat_outline(),
            minimum_spacing=0.095,
            maximum_spacing=0.24,
            transition_distance=0.48,
            refinement_points=((-0.10, 0.05),),
        ),
    )
    sources = {recipe.name: _geo_source(recipe) for recipe in polygon_recipes}
    sources.update(
        {
            "fractional-maze": _maze_geo_source(
                columns=7,
                rows=5,
                minimum_spacing=0.14,
                maximum_spacing=0.30,
            ),
            "fractional-maze-smoke": _maze_geo_source(
                columns=5,
                rows=4,
                minimum_spacing=0.20,
                maximum_spacing=0.34,
            ),
        }
    )
    selected = {args.only: sources[args.only]} if args.only else sources
    destination = Path(__file__).resolve().parent / "meshes"
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="yonderdrake-meshes-") as temporary:
        temporary_directory = Path(temporary)
        for name, source in selected.items():
            geo_path = temporary_directory / f"{name}.geo"
            mesh_path = destination / f"{name}.msh"
            geo_path.write_text(source, encoding="utf-8")
            subprocess.run(
                (
                    str(gmsh),
                    "-2",
                    str(geo_path),
                    "-format",
                    "msh2",
                    "-o",
                    str(mesh_path),
                    "-v",
                    "2",
                ),
                check=True,
                timeout=360,
            )
            print(mesh_path)


if __name__ == "__main__":
    main()
