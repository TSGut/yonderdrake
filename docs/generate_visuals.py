"""Generate the conceptual diagrams embedded in the documentation."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from collections.abc import Callable
from math import gamma, pi
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.axes import Axes
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon

OUTPUT = Path(__file__).parent / "source" / "_static" / "visuals"
TEAL = "#146c6e"
GREEN = "#628b46"
GOLD = "#e7ad2f"
INK = "#243238"
MUTED = "#64757b"
PALE = "#eef8f6"
SAND = "#fbf6e9"
CORAL = "#d7674f"


def _finish(fig: plt.Figure, name: str, *, dpi: int | str = "figure") -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT / name
    fig.savefig(output_path, bbox_inches="tight", facecolor="white", dpi=dpi)
    plt.close(fig)
    if output_path.suffix == ".svg":
        lines = output_path.read_text(encoding="utf-8").splitlines()
        output_path.write_text(
            "\n".join(line.rstrip() for line in lines) + "\n",
            encoding="utf-8",
        )


def _panel(ax: Axes, title: str, subtitle: str) -> None:
    ax.set_aspect("equal")
    ax.set_xlim(-1.16, 1.16)
    ax.set_ylim(-1.16, 1.16)
    ax.axis("off")
    ax.set_title(
        title,
        color=INK,
        fontsize=13,
        fontweight="bold",
        pad=0,
        y=1.105,
    )
    ax.text(
        0.5,
        1.035,
        subtitle,
        color=MUTED,
        fontsize=9,
        ha="center",
        transform=ax.transAxes,
    )


def sensor_array_layouts() -> None:
    """Render ring, sphere, and custom sensor placement in one row."""
    fig, axes = plt.subplots(1, 3, figsize=(11.7, 3.7))
    fig.subplots_adjust(wspace=0.2)

    # A faint field gives the sensor footprints a physical context.
    grid = np.linspace(-1, 1, 180)
    xx, yy = np.meshgrid(grid, grid)
    field = np.exp(-7 * ((xx + 0.2) ** 2 + (yy - 0.1) ** 2))
    field += 0.65 * np.exp(-12 * ((xx - 0.35) ** 2 + (yy + 0.25) ** 2))

    ax = axes[0]
    _panel(ax, "Ring", "64 equally spaced centres · 2D")
    disk = Circle((0, 0), 0.72, facecolor="none", edgecolor=MUTED, linewidth=1.2)
    image = ax.imshow(
        field,
        extent=(-0.72, 0.72, -0.72, 0.72),
        origin="lower",
        cmap="YlGnBu",
        alpha=0.42,
    )
    image.set_clip_path(disk)
    ax.add_patch(disk)
    theta = np.linspace(0, 2 * pi, 64, endpoint=False)
    locations = np.column_stack((0.92 * np.cos(theta), 0.92 * np.sin(theta)))
    for x_coord, y_coord in locations:
        ax.add_patch(
            Circle(
                (x_coord, y_coord),
                0.04,
                facecolor=GOLD,
                edgecolor="white",
                linewidth=0.3,
                alpha=0.86,
            )
        )
    ax.add_patch(Circle((0, 0), 0.92, fill=False, edgecolor=TEAL, linewidth=1))
    ax.annotate(
        "radius",
        xy=(0.65, 0.65),
        xytext=(0.15, 0.28),
        arrowprops={"arrowstyle": "->", "color": TEAL},
        color=TEAL,
        fontsize=8,
    )

    ax = axes[1]
    _panel(ax, "Sphere", "128 golden-angle centres · 3D")
    count = 128
    indices = np.arange(count)
    z_coord = 1 - 2 * (indices + 0.5) / count
    azimuth = pi * (3 - np.sqrt(5)) * indices
    radius = np.sqrt(1 - z_coord**2)
    x_coord = radius * np.cos(azimuth)
    depth = radius * np.sin(azimuth)
    ax.add_patch(Circle((0, 0), 0.88, facecolor=PALE, edgecolor=TEAL, linewidth=1.2))
    for latitude in (-0.5, 0.0, 0.5):
        ax.add_patch(
            Circle(
                (0, 0),
                0.88 * np.sqrt(1 - latitude**2),
                fill=False,
                edgecolor="#b8ccc9",
                linewidth=0.55,
            )
        )
    order = np.argsort(depth)
    sizes = 14 + 22 * (depth[order] + 1) / 2
    colors = np.where(depth[order] > 0, GOLD, "#9fbab7")
    ax.scatter(
        0.88 * x_coord[order],
        0.88 * z_coord[order],
        s=sizes,
        c=colors,
        edgecolors="white",
        linewidths=0.35,
        alpha=np.where(depth[order] > 0, 0.94, 0.48),
    )
    ax.text(0, -1.03, "front and rear surface", ha="center", color=MUTED, fontsize=8)

    ax = axes[2]
    _panel(ax, "Custom", "user-supplied (x, y) rows · 2D")
    boundary = Polygon(
        [
            (-0.82, -0.52),
            (-0.7, 0.52),
            (-0.28, 0.82),
            (0.38, 0.72),
            (0.76, 0.3),
            (0.7, -0.56),
            (0.05, -0.8),
        ],
        closed=True,
        facecolor=PALE,
        edgecolor=TEAL,
        linewidth=1.2,
    )
    ax.add_patch(boundary)
    custom = np.array(
        [
            (-0.94, 0.55),
            (-0.7, 0.88),
            (-0.14, 1.0),
            (0.46, 0.91),
            (0.93, 0.52),
            (0.98, -0.08),
            (0.72, -0.78),
            (0.12, -1.02),
            (-0.58, -0.86),
            (-0.98, -0.35),
        ]
    )
    for x_pos, y_pos in custom:
        ax.add_patch(Circle((x_pos, y_pos), 0.085, facecolor=GOLD, alpha=0.22))
        ax.plot(x_pos, y_pos, "o", color=CORAL, markersize=4.5)
    ax.annotate(
        "width",
        xy=(0.98, -0.08),
        xytext=(0.53, 0.15),
        arrowprops={"arrowstyle": "->", "color": CORAL},
        color=CORAL,
        fontsize=8,
    )

    _finish(fig, "sensor-array-layouts.svg")


def memory_formulations() -> None:
    """Show what is stored and solved by each time formulation."""
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.45))
    fig.subplots_adjust(wspace=0.22)
    for ax in axes:
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 7)
        ax.axis("off")

    ax = axes[0]
    ax.set_title("Full history", color=INK, fontsize=13, fontweight="bold")
    ax.text(
        5, 6.25, "store every accepted increment", ha="center", color=MUTED, fontsize=9
    )
    times = np.linspace(1, 8.7, 8)
    values = 2.1 + 1.5 * np.sin(0.55 * times) * np.exp(-0.05 * times)
    ax.plot(times, values, color=TEAL, linewidth=2)
    ax.scatter(times, values, color=TEAL, s=28, zorder=3)
    for time, value in zip(times[:-1], values[:-1], strict=True):
        ax.add_patch(
            FancyArrowPatch(
                (time, value + 0.15),
                (8.7, values[-1] + 0.15),
                arrowstyle="->",
                color=GOLD,
                alpha=0.45,
                mutation_scale=8,
            )
        )
    ax.text(
        5, 0.75, "work + storage grow with steps", ha="center", color=CORAL, fontsize=9
    )

    ax = axes[1]
    ax.set_title("Eliminated recurrence", color=INK, fontsize=13, fontweight="bold")
    ax.text(
        5,
        6.25,
        "default: update modes outside the solve",
        ha="center",
        color=MUTED,
        fontsize=9,
    )
    ax.add_patch(
        FancyBboxPatch(
            (0.5, 2.35),
            2.1,
            1.4,
            boxstyle="round,pad=0.18",
            facecolor=PALE,
            edgecolor=TEAL,
        )
    )
    ax.text(1.55, 3.05, "$u^{n+1}$", ha="center", va="center", fontsize=14, color=INK)
    for index, y_pos in enumerate((1.3, 2.45, 3.6, 4.75), start=1):
        ax.add_patch(
            FancyBboxPatch(
                (4.1, y_pos),
                2.0,
                0.72,
                boxstyle="round,pad=0.1",
                facecolor=SAND,
                edgecolor=GOLD,
            )
        )
        ax.text(
            5.1, y_pos + 0.36, rf"$\phi_{index}$", ha="center", va="center", color=INK
        )
        ax.annotate(
            "",
            xy=(4.05, y_pos + 0.36),
            xytext=(2.65, 3.05),
            arrowprops={"arrowstyle": "->", "color": TEAL},
        )
    ax.add_patch(
        FancyBboxPatch(
            (7.25, 2.35),
            2.1,
            1.4,
            boxstyle="round,pad=0.18",
            facecolor=PALE,
            edgecolor=TEAL,
        )
    )
    ax.text(
        8.3,
        3.05,
        r"$D_C^\alpha u$",
        ha="center",
        va="center",
        fontsize=13,
        color=INK,
    )
    for y_pos in (1.66, 2.81, 3.96, 5.11):
        ax.annotate(
            "",
            xy=(7.2, 3.05),
            xytext=(6.15, y_pos),
            arrowprops={"arrowstyle": "->", "color": GOLD},
        )
    ax.text(
        5,
        0.55,
        "$O(m)$ memory · physical solve stays small",
        ha="center",
        color=GREEN,
        fontsize=9,
    )

    ax = axes[2]
    ax.set_title("Auxiliary ODE", color=INK, fontsize=13, fontweight="bold")
    ax.text(
        5, 6.25, "solve field and modes together", ha="center", color=MUTED, fontsize=9
    )
    ax.add_patch(
        FancyBboxPatch(
            (1.45, 1.2),
            7.1,
            4.25,
            boxstyle="round,pad=0.25",
            facecolor=PALE,
            edgecolor=TEAL,
            linewidth=1.4,
        )
    )
    ax.text(
        2.4, 4.85, "one mixed PETSc system", color=TEAL, fontsize=9, fontweight="bold"
    )
    rows = [
        "physical field  $u$",
        r"memory mode  $\phi_1$",
        r"$\vdots$",
        r"memory mode  $\phi_m$",
    ]
    for index, label in enumerate(rows):
        y_pos = 4.2 - index * 0.82
        ax.add_patch(
            FancyBboxPatch(
                (2.1, y_pos - 0.27),
                5.8,
                0.55,
                boxstyle="round,pad=0.06",
                facecolor="white",
                edgecolor="#bed4d0",
            )
        )
        ax.text(5, y_pos, label, ha="center", va="center", color=INK, fontsize=10)
    ax.text(
        5,
        0.55,
        "larger solve · modes visible to field splits",
        ha="center",
        color=CORAL,
        fontsize=9,
    )

    _finish(fig, "memory-formulations.svg")


def spatial_realizations() -> None:
    """Compare three fractional Laplacian realizations on one square."""
    grid_size = 241
    order = 0.65
    coordinates = np.linspace(0.0, 1.0, grid_size)
    spacing = float(coordinates[1] - coordinates[0])
    xx, yy = np.meshgrid(coordinates, coordinates)

    # A smooth zero-trace input built from known Dirichlet eigenfunctions.
    mode_11 = np.sin(pi * xx) * np.sin(pi * yy)
    mode_21 = np.sin(2 * pi * xx) * np.sin(pi * yy)
    mode_32 = np.sin(3 * pi * xx) * np.sin(2 * pi * yy)
    input_field = mode_11 + 0.42 * mode_21 - 0.22 * mode_32
    spectral = (
        (2 * pi**2) ** order * mode_11
        + 0.42 * (5 * pi**2) ** order * mode_21
        - 0.22 * (13 * pi**2) ** order * mode_32
    )

    # Zero-padding approximates the whole-space Fourier power of the zero
    # extension. The periodic realization uses the same sampled square as one
    # repeating cell.
    padded_size = 4 * (grid_size - 1)
    padded = np.zeros((padded_size, padded_size))
    offset = (padded_size - (grid_size - 1)) // 2
    interior_samples = input_field[:-1, :-1]
    padded[
        offset : offset + grid_size - 1,
        offset : offset + grid_size - 1,
    ] = interior_samples
    frequencies = 2 * pi * np.fft.fftfreq(padded_size, d=spacing)
    kx, ky = np.meshgrid(frequencies, frequencies)
    symbol = (kx**2 + ky**2) ** order
    riesz_padded = np.fft.ifft2(symbol * np.fft.fft2(padded)).real
    riesz_core = riesz_padded[
        offset : offset + grid_size - 1,
        offset : offset + grid_size - 1,
    ]
    riesz = np.pad(riesz_core, ((0, 1), (0, 1)), mode="edge")

    periodic_frequencies = 2 * pi * np.fft.fftfreq(grid_size - 1, d=spacing)
    periodic_kx, periodic_ky = np.meshgrid(
        periodic_frequencies,
        periodic_frequencies,
    )
    periodic_symbol = (periodic_kx**2 + periodic_ky**2) ** order
    periodic_core = np.fft.ifft2(
        periodic_symbol * np.fft.fft2(interior_samples)
    ).real
    periodic = np.pad(periodic_core, ((0, 1), (0, 1)), mode="wrap")

    spectral /= np.max(np.abs(spectral))
    riesz /= np.max(np.abs(riesz))
    periodic /= np.max(np.abs(periodic))
    input_field /= np.max(np.abs(input_field))

    fig = plt.figure(figsize=(12.0, 7.3))
    layout = fig.add_gridspec(
        2,
        3,
        height_ratios=(1.0, 1.18),
        hspace=0.42,
        wspace=0.18,
        top=0.84,
        bottom=0.08,
        left=0.045,
        right=0.955,
    )
    input_ax = fig.add_subplot(layout[0, 1])
    output_axes = [fig.add_subplot(layout[1, column]) for column in range(3)]
    fig.add_subplot(layout[0, 0]).axis("off")
    fig.add_subplot(layout[0, 2]).axis("off")
    fig.suptitle(
        "One field, three fractional Laplacian realizations",
        color=INK,
        fontsize=16,
        fontweight="bold",
        y=0.965,
    )

    def draw_panel(
        panel: Axes,
        field: np.ndarray,
        title: str,
        subtitle: str,
        mesh_kind: str | None,
    ) -> None:
        panel.imshow(
            field,
            extent=(0.0, 1.0, 0.0, 1.0),
            cmap="viridis",
            vmin=-1.0,
            vmax=1.0,
            origin="lower",
            interpolation="bicubic",
            rasterized=True,
        )
        panel.contour(
            coordinates,
            coordinates,
            field,
            levels=[0.0],
            colors=["white"],
            linewidths=1.15,
            alpha=0.85,
        )
        mesh_coordinates = np.linspace(0.0, 1.0, 13)
        if mesh_kind is not None:
            for value in mesh_coordinates:
                panel.plot(
                    [value, value],
                    [0.0, 1.0],
                    color="white",
                    linewidth=0.38,
                    alpha=0.38,
                )
                panel.plot(
                    [0.0, 1.0],
                    [value, value],
                    color="white",
                    linewidth=0.38,
                    alpha=0.38,
                )
            if mesh_kind == "triangular":
                for row in range(12):
                    for column in range(12):
                        x0 = mesh_coordinates[column]
                        x1 = mesh_coordinates[column + 1]
                        y0 = mesh_coordinates[row]
                        y1 = mesh_coordinates[row + 1]
                        if (row + column) % 2:
                            panel.plot(
                                [x0, x1], [y1, y0], color="white",
                                linewidth=0.32, alpha=0.32,
                            )
                        else:
                            panel.plot(
                                [x0, x1], [y0, y1], color="white",
                                linewidth=0.32, alpha=0.32,
                            )
        panel.add_patch(
            Polygon(
                [(0, 0), (1, 0), (1, 1), (0, 1)],
                closed=True,
                fill=False,
                edgecolor=INK,
                linewidth=1.25,
            )
        )
        panel.set_aspect("equal")
        panel.set_xlim(0.0, 1.0)
        panel.set_ylim(0.0, 1.0)
        panel.set_xticks([])
        panel.set_yticks([])
        panel.set_title(title, color=INK, fontsize=12, fontweight="bold", pad=19)
        panel.text(
            0.5,
            1.025,
            subtitle,
            color=MUTED,
            fontsize=8.5,
            ha="center",
            transform=panel.transAxes,
        )
        for spine in panel.spines.values():
            spine.set_visible(False)

    draw_panel(
        input_ax,
        input_field,
        "Input field $u$",
        "the same square and sampled field",
        None,
    )
    panels = (
        (
            spectral,
            "Dirichlet spectral",
            r"eigenvalue power $\lambda_k^s$ · triangular mesh",
            "triangular",
        ),
        (
            riesz,
            "Zero-exterior Riesz",
            r"whole-space power of $u\,\mathbf{1}_\Omega$ · triangular mesh",
            "triangular",
        ),
        (
            periodic,
            "Periodic Fourier",
            r"Fourier multiplier $|k|^{2s}$ · uniform quadrilateral mesh",
            "quadrilateral",
        ),
    )
    for panel, (field, title, subtitle, mesh_kind) in zip(
        output_axes,
        panels,
        strict=True,
    ):
        draw_panel(panel, field, title, subtitle, mesh_kind)

    arrow_properties = {
        "arrowstyle": "-|>",
        "color": TEAL,
        "linewidth": 1.2,
        "mutation_scale": 12,
        "connectionstyle": "arc3,rad=0",
    }
    for target_x in (0.18, 0.5, 0.82):
        fig.add_artist(
            FancyArrowPatch(
                (0.5, 0.545),
                (target_x, 0.505),
                transform=fig.transFigure,
                **arrow_properties,
            )
        )

    fig.text(
        0.5,
        0.025,
        "Normalized schematic responses at the same fractional order",
        color=MUTED,
        fontsize=9,
        ha="center",
    )
    _finish(fig, "spatial-realizations.svg", dpi=200)


def derivative_initial_trace() -> None:
    """Visualize the initial-trace distinction for a constant function."""
    fig, axes = plt.subplots(1, 2, figsize=(9.3, 3.25), sharex=True)
    time = np.linspace(0.035, 2, 400)
    order = 0.55
    axes[0].plot(time, np.ones_like(time), color=TEAL, linewidth=2.5)
    axes[0].set_title("Same input: $u(t)=1$", color=INK, fontsize=13, fontweight="bold")
    axes[0].set_ylabel("input")
    axes[0].set_ylim(-0.08, 1.25)
    axes[0].annotate(
        "nonzero initial trace",
        xy=(0.05, 1),
        xytext=(0.45, 0.55),
        arrowprops={"arrowstyle": "->", "color": GOLD},
        color=GOLD,
        fontsize=9,
    )

    caputo = np.zeros_like(time)
    riemann_liouville = time ** (-order) / gamma(1 - order)
    axes[1].plot(time, caputo, color=TEAL, linewidth=2.5, label="Caputo")
    axes[1].plot(
        time, riemann_liouville, color=CORAL, linewidth=2.5, label="Riemann–Liouville"
    )
    axes[1].set_title(
        "Different derivatives", color=INK, fontsize=13, fontweight="bold"
    )
    axes[1].set_ylabel("fractional derivative")
    axes[1].legend(frameon=False, fontsize=9)
    axes[1].annotate(
        "exact initial-trace term",
        xy=(0.13, riemann_liouville[20]),
        xytext=(0.62, 2.0),
        arrowprops={"arrowstyle": "->", "color": CORAL},
        color=CORAL,
        fontsize=9,
    )

    for ax in axes:
        ax.set_xlabel("time")
        ax.grid(color="#dfe8e7", linewidth=0.7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(colors=MUTED)
    fig.tight_layout()
    _finish(fig, "derivative-initial-trace.svg")


def _mittag_leffler(alpha: float, argument: float) -> float:
    """Evaluate E_alpha(argument) by its convergent power series."""
    total = 1.0
    power = 1.0
    for index in range(1, 1000):
        power *= argument
        term = power / gamma(alpha * index + 1.0)
        total += term
        if abs(term) <= 5.0e-16 * max(1.0, abs(total)):
            return total
    raise RuntimeError("Mittag-Leffler series did not converge")


def quickstart_spatial_diffusion() -> None:
    """Render classical and spectral Caputo diffusion on a Y-shaped domain."""
    import matplotlib.tri as mtri
    from scipy.sparse import coo_matrix, diags
    from scipy.sparse.linalg import eigsh, factorized

    outline = np.asarray(
        (
            (-0.18, -0.94),
            (0.18, -0.94),
            (0.18, -0.15),
            (0.79, 0.75),
            (0.53, 0.96),
            (0.00, 0.18),
            (-0.53, 0.96),
            (-0.79, 0.75),
            (-0.18, -0.15),
        ),
        dtype=np.float64,
    )
    gmsh = shutil.which("gmsh")
    if gmsh is None:
        raise RuntimeError("quickstart Y-domain visuals require the gmsh executable")
    point_lines = [
        f"Point({index}) = {{{x:.12g}, {y:.12g}, 0, 0.055}};"
        for index, (x, y) in enumerate(outline, start=1)
    ]
    line_lines = [
        f"Line({index}) = {{{index}, {index % len(outline) + 1}}};"
        for index in range(1, len(outline) + 1)
    ]
    source_tag = len(outline) + 1
    junction_tag = source_tag + 1
    geometry = "\n".join(
        (
            'SetFactory("OpenCASCADE");',
            *point_lines,
            *line_lines,
            "Curve Loop(1) = {"
            + ", ".join(map(str, range(1, len(outline) + 1)))
            + "};",
            "Plane Surface(1) = {1};",
            f"Point({source_tag}) = {{0, -0.12, 0, 0.016}};",
            f"Point({junction_tag}) = {{0, 0.02, 0, 0.016}};",
            f"Point{{{source_tag}, {junction_tag}}} In Surface{{1}};",
            "Field[1] = Distance;",
            f"Field[1].PointsList = {{{source_tag}, {junction_tag}}};",
            "Field[2] = Threshold;",
            "Field[2].InField = 1;",
            "Field[2].SizeMin = 0.016;",
            "Field[2].SizeMax = 0.055;",
            "Field[2].DistMin = 0.10;",
            "Field[2].DistMax = 0.58;",
            "Background Field = 2;",
            "Mesh.Algorithm = 6;",
            "Mesh.Smoothing = 12;",
            "Mesh.MeshSizeFromCurvature = 0;",
            "Mesh.MshFileVersion = 2.2;",
        )
    )
    with tempfile.TemporaryDirectory(prefix="yonderdrake-y-mesh-") as directory:
        temporary = Path(directory)
        geometry_path = temporary / "y_domain.geo"
        mesh_path = temporary / "y_domain.msh"
        geometry_path.write_text(geometry + "\n", encoding="utf-8")
        # A self-intersecting outline leaves gmsh meshing indefinitely rather
        # than reporting an error, so an edited outline fails here instead.
        subprocess.run(
            (gmsh, "-2", str(geometry_path), "-format", "msh2", "-o", str(mesh_path)),
            check=True,
            capture_output=True,
            text=True,
            timeout=360,
        )
        lines = mesh_path.read_text(encoding="utf-8").splitlines()
    node_start = lines.index("$Nodes")
    node_count = int(lines[node_start + 1])
    node_rows = lines[node_start + 2 : node_start + 2 + node_count]
    node_ids = []
    node_values = []
    for row in node_rows:
        identifier, x_value, y_value, _ = row.split()
        node_ids.append(int(identifier))
        node_values.append((float(x_value), float(y_value)))
    node_lookup = {
        identifier: index for index, identifier in enumerate(node_ids)
    }
    points = np.asarray(node_values, dtype=np.float64)
    element_start = lines.index("$Elements")
    element_count = int(lines[element_start + 1])
    cells_list = []
    for row in lines[
        element_start + 2 : element_start + 2 + element_count
    ]:
        values = tuple(int(value) for value in row.split())
        if values[1] != 2:
            continue
        tag_count = values[2]
        cells_list.append(
            tuple(node_lookup[value] for value in values[3 + tag_count :])
        )
    cells = np.asarray(cells_list, dtype=np.int64)

    all_edges = np.sort(
        np.vstack((cells[:, (0, 1)], cells[:, (1, 2)], cells[:, (2, 0)])),
        axis=1,
    )
    unique_edges, edge_counts = np.unique(all_edges, axis=0, return_counts=True)
    boundary_nodes = np.unique(unique_edges[edge_counts == 1])
    interior = np.ones(points.shape[0], dtype=bool)
    interior[boundary_nodes] = False
    interior_indices = np.flatnonzero(interior)
    reduced_index = np.full(points.shape[0], -1, dtype=np.int64)
    reduced_index[interior_indices] = np.arange(interior_indices.size)

    mass = np.zeros(interior_indices.size)
    stiffness_rows: list[int] = []
    stiffness_columns: list[int] = []
    stiffness_entries: list[float] = []
    for cell in cells:
        vertices = points[cell]
        coordinate_matrix = np.column_stack((np.ones(3), vertices))
        determinant = float(np.linalg.det(coordinate_matrix))
        area = 0.5 * abs(determinant)
        coefficients = np.linalg.inv(coordinate_matrix)
        gradients = coefficients[1:, :]
        local_stiffness = area * gradients.T @ gradients
        for local_row, global_row in enumerate(cell):
            row = int(reduced_index[global_row])
            if row < 0:
                continue
            mass[row] += area / 3.0
            for local_column, global_column in enumerate(cell):
                column = int(reduced_index[global_column])
                if column < 0:
                    continue
                stiffness_rows.append(row)
                stiffness_columns.append(column)
                stiffness_entries.append(
                    float(local_stiffness[local_row, local_column])
                )
    dimension = interior_indices.size
    mass_matrix = diags(mass, format="csr")
    stiffness = coo_matrix(
        (stiffness_entries, (stiffness_rows, stiffness_columns)),
        shape=(dimension, dimension),
    ).tocsr()

    initial_full = np.exp(
        -45.0 * (points[:, 0] ** 2 + (points[:, 1] + 0.12) ** 2)
    )
    initial_full[~interior] = 0.0
    initial_full /= np.max(initial_full)
    initial = initial_full[interior]

    alpha = 0.65
    spatial_order = 0.4
    time_step = 0.025
    num_steps = 100
    diffusivity = 0.08
    caputo_scale = time_step ** (-alpha) / gamma(2.0 - alpha)
    l1_weights = np.asarray(
        [
            (index + 1) ** (1.0 - alpha) - index ** (1.0 - alpha)
            for index in range(num_steps)
        ]
    )

    def evolve(
        starting_value: np.ndarray,
        solve_step: Callable[[np.ndarray], np.ndarray],
    ) -> list[np.ndarray]:
        states = [starting_value]
        increments: list[np.ndarray] = []
        previous = starting_value
        for step in range(1, num_steps + 1):
            history = np.zeros_like(previous)
            for lag in range(1, step):
                history += l1_weights[lag] * increments[step - lag - 1]
            current = solve_step(caputo_scale * (previous - history))
            increments.append(current - previous)
            states.append(current)
            previous = current
        return states

    classical_system = (
        caputo_scale * mass_matrix + diffusivity * stiffness
    ).tocsc()
    classical_solve = factorized(classical_system)
    classical_reduced_states = evolve(
        initial,
        lambda value: classical_solve(mass * value),
    )

    eigenvalue_count = min(220, dimension - 2)
    eigenvalues, eigenvectors = eigsh(
        stiffness,
        k=eigenvalue_count,
        M=mass_matrix,
        sigma=0.0,
        which="LM",
    )
    ordering = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[ordering]
    eigenvectors = eigenvectors[:, ordering]
    initial_coefficients = eigenvectors.T @ (mass * initial)
    spectral_denominator = (
        caputo_scale + diffusivity * eigenvalues**spatial_order
    )
    spectral_coefficients = evolve(
        initial_coefficients,
        lambda value: value / spectral_denominator,
    )
    spectral_reduced_states = [initial]
    spectral_reduced_states.extend(
        eigenvectors @ coefficients for coefficients in spectral_coefficients[1:]
    )

    def embed(states: list[np.ndarray]) -> list[np.ndarray]:
        embedded = []
        for state in states:
            values = np.zeros(points.shape[0])
            values[interior] = state
            embedded.append(values)
        return embedded

    classical_states = embed(classical_reduced_states)
    spectral_states = embed(spectral_reduced_states)
    triangulation = mtri.Triangulation(points[:, 0], points[:, 1], cells)

    snapshot_steps = (0, 20, 55, 100)

    def render(
        states: list[np.ndarray],
        title: str,
        filename: str,
    ) -> None:
        fig, axes = plt.subplots(1, 4, figsize=(11.8, 3.45))
        fig.subplots_adjust(
            left=0.025,
            right=0.925,
            top=0.78,
            bottom=0.06,
            wspace=0.08,
        )
        image = None
        for axis, step in zip(axes, snapshot_steps, strict=True):
            snapshot = np.maximum(states[step], 0.0)
            snapshot /= np.max(snapshot)
            image = axis.tricontourf(
                triangulation,
                snapshot,
                levels=np.linspace(0.0, 1.0, 41),
                cmap="viridis",
                vmin=0.0,
                vmax=1.0,
                extend="max",
            )
            axis.triplot(
                triangulation,
                color="white",
                linewidth=0.22,
                alpha=0.14,
            )
            closed_outline = np.vstack((outline, outline[0]))
            axis.plot(
                closed_outline[:, 0],
                closed_outline[:, 1],
                color=INK,
                linewidth=0.65,
                solid_joinstyle="miter",
            )
            axis.set_aspect("equal")
            axis.set_xlim(-1.00, 1.00)
            axis.set_ylim(-1.00, 1.00)
            axis.axis("off")
            axis.set_title(
                rf"$t={step * time_step:.2f}$",
                color=INK,
                fontsize=11,
                fontweight="bold",
                pad=4,
            )
        if image is None:
            raise RuntimeError("Y-domain diffusion produced no snapshots")
        colorbar = fig.colorbar(
            image,
            ax=axes,
            fraction=0.025,
            pad=0.025,
            ticks=(0.0, 0.5, 1.0),
        )
        colorbar.set_label("normalized field", color=INK)
        colorbar.ax.tick_params(colors=MUTED)
        fig.suptitle(
            title,
            color=INK,
            fontsize=14,
            fontweight="bold",
            y=0.96,
        )
        _finish(fig, filename, dpi=180)

    render(
        classical_states,
        r"Caputo diffusion with the classical Laplacian, $\alpha=0.65$",
        "quickstart-y-domain-diffusion.png",
    )
    render(
        spectral_states,
        r"Caputo diffusion with the spectral fractional Laplacian, "
        r"$\alpha=0.65$, $s=0.4$",
        "quickstart-y-domain-spectral-diffusion.png",
    )


def _quickstart_relaxation_solution(
    num_modes: int,
    time_step: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the quickstart relaxation for one mode count."""
    import firedrake as fd

    from yonderdrake import BirkSong, CaputoDerivative, FractionalTimeStepper

    alpha = 0.6
    mesh = fd.UnitIntervalMesh(1)
    space = fd.FunctionSpace(mesh, "CG", 1)
    u = fd.Function(space).assign(1.0)
    v = fd.TestFunction(space)
    t = fd.Constant(0.0)
    dt = fd.Constant(time_step)
    residual = (
        fd.inner(CaputoDerivative(u, alpha), v) + fd.inner(u, v)
    ) * fd.dx
    stepper = FractionalTimeStepper(residual, BirkSong(num_modes), t, dt, u)

    times = [0.0]
    computed = [1.0]
    num_steps = round(1.0 / time_step)
    for _ in range(num_steps):
        stepper.advance()
        t.assign(t + dt)
        times.append(float(t))
        computed.append(float(u.dat.data_ro[0]))

    return np.asarray(times), np.asarray(computed)


def quickstart_relaxation() -> None:
    """Plot the quickstart solution and its mode-refinement comparison."""
    alpha = 0.6
    time_values, computed_values = _quickstart_relaxation_solution(48)
    exact_values = np.asarray(
        [
            _mittag_leffler(alpha, -(time_value**alpha))
            for time_value in time_values
        ]
    )
    absolute_error = np.abs(computed_values - exact_values)

    fig, (solution_axis, error_axis) = plt.subplots(
        2,
        1,
        figsize=(9.6, 6.2),
        sharex=True,
        gridspec_kw={"height_ratios": (2.15, 1.0), "hspace": 0.12},
    )
    solution_axis.plot(
        time_values,
        exact_values,
        color=INK,
        linewidth=2.4,
        label=r"Exact $E_{0.6}(-t^{0.6})$",
    )
    solution_axis.plot(
        time_values,
        computed_values,
        color=TEAL,
        linewidth=1.8,
        marker="o",
        markevery=5,
        markersize=4.2,
        label=r"Computed: 48 modes, $\Delta t=0.01$",
    )
    solution_axis.set_ylabel(r"Solution $u(t)$")
    solution_axis.set_title(
        "Time-fractional relaxation in the quickstart",
        color=INK,
        fontsize=14,
        fontweight="bold",
        pad=10,
    )
    solution_axis.legend(frameon=False, loc="upper right")

    error_axis.plot(
        time_values,
        np.where(absolute_error > 0.0, absolute_error, np.nan),
        color=CORAL,
        linewidth=2.0,
    )
    error_axis.set_yscale("log", base=10)
    error_axis.set_xlabel(r"Time $t$")
    error_axis.set_ylabel("Absolute error")

    for axis in (solution_axis, error_axis):
        axis.set_xlim(0.0, 1.0)
        axis.grid(color="#c8d4d4", linewidth=0.7, alpha=0.65)
        axis.spines["right"].set_visible(False)
        axis.spines["top"].set_visible(False)
        axis.tick_params(colors=MUTED)
        axis.yaxis.label.set_color(INK)
        axis.xaxis.label.set_color(INK)

    _finish(fig, "quickstart-relaxation.png", dpi=180)

    refined_time, refined_values = _quickstart_relaxation_solution(256, 0.001)
    refined_exact = np.asarray(
        [
            _mittag_leffler(alpha, -(time_value**alpha))
            for time_value in refined_time
        ]
    )
    refined_error = np.abs(refined_values - refined_exact)
    fig, (solution_axis, error_axis) = plt.subplots(
        2,
        1,
        figsize=(9.6, 6.2),
        sharex=True,
        gridspec_kw={"height_ratios": (2.15, 1.0), "hspace": 0.12},
    )
    solution_axis.plot(
        time_values,
        exact_values,
        color=INK,
        linewidth=2.5,
        label=r"Exact $E_{0.6}(-t^{0.6})$",
    )
    solution_axis.plot(
        time_values,
        computed_values,
        color=TEAL,
        linewidth=1.8,
        marker="o",
        markevery=6,
        markersize=4.0,
        label=r"48 modes, $\Delta t=0.01$",
    )
    solution_axis.plot(
        refined_time,
        refined_values,
        color=GOLD,
        linewidth=1.8,
        linestyle="--",
        marker="s",
        markevery=(30, 60),
        markersize=3.8,
        label=r"256 modes, $\Delta t=0.001$",
    )
    solution_axis.set_ylabel(r"Solution $u(t)$")
    solution_axis.set_title(
        "Refining the memory representation and timestep",
        color=INK,
        fontsize=14,
        fontweight="bold",
        pad=10,
    )
    solution_axis.legend(frameon=False, loc="upper right")

    error_axis.plot(
        time_values,
        np.where(absolute_error > 0.0, absolute_error, np.nan),
        color=TEAL,
        linewidth=2.0,
        label="48 modes",
    )
    error_axis.plot(
        refined_time,
        np.where(refined_error > 0.0, refined_error, np.nan),
        color=GOLD,
        linewidth=1.8,
        linestyle="--",
        label="256 modes",
    )
    error_axis.set_yscale("log", base=10)
    error_axis.set_xlabel(r"Time $t$")
    error_axis.set_ylabel("Absolute error")
    error_axis.legend(frameon=False, loc="upper right")

    for axis in (solution_axis, error_axis):
        axis.set_xlim(0.0, 1.0)
        axis.grid(color="#c8d4d4", linewidth=0.7, alpha=0.65)
        axis.spines["right"].set_visible(False)
        axis.spines["top"].set_visible(False)
        axis.tick_params(colors=MUTED)
        axis.yaxis.label.set_color(INK)
        axis.xaxis.label.set_color(INK)

    _finish(fig, "quickstart-relaxation-refinement.png", dpi=180)


def main() -> None:
    """Generate all documentation diagrams."""
    sensor_array_layouts()
    memory_formulations()
    spatial_realizations()
    derivative_initial_trace()
    quickstart_spatial_diffusion()
    quickstart_relaxation()


if __name__ == "__main__":
    main()
