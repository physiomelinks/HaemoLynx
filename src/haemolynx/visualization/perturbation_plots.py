"""Disk artifacts for a finished perturbation: Alice-style sweep curves, or
pipeline-like plots and statistics CSVs for a single re-solve.

Sweeps already write their grid CSV via the haemodynamics helpers; this module
turns those rows into resistance/flow PNGs with axes that match what was swept.
Non-sweep perturbations get the same class of matplotlib/plotly and statistics
exports ``export_results`` writes for the baseline (no VTK: a perturbation is
not a second published model).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from haemolynx import statistics
from haemolynx.haemodynamics.perturbations import is_sweep_perturbation
from haemolynx.visualization.plot import (
    plot_node_degree_distribution,
    visualize_3d_plotly,
    visualize_edges_and_nodes,
    visualize_geometry_with_branch_orders,
    visualize_geometry_with_edge_resistance,
)

logger = logging.getLogger(__name__)

#: Y-axis metrics shared with the Alice / brain dilation curves.
SWEEP_Y_METRICS: Mapping[str, tuple[str, str, str]] = {
    "equivalent_resistance": (
        "resistance",
        "Equivalent resistance (Pa.s/m^3)",
        "Resistance",
    ),
    "total_inlet_flow": (
        "flow",
        "Total inlet flow (m^3/s)",
        "Flow",
    ),
}


@dataclass(frozen=True)
class SweepAxisSpec:
    """How to draw one sweep type's Alice-style curves."""

    x_key: str
    x_label: str
    #: Column that splits series (``None`` = one line over the x-axis).
    series_key: str | None
    series_label: str
    #: Stem fragment between metric and axis, e.g. ``vs_pericyte_dilation``.
    stem_suffix: str
    title_subject: str


#: Per declared sweep type: x-axis, optional series, and file-stem wording.
SWEEP_AXIS_BY_TYPE: Mapping[str, SweepAxisSpec] = {
    "pressure_sweep": SweepAxisSpec(
        x_key="inlet_pressure_pa",
        x_label="Inlet pressure (Pa)",
        series_key=None,
        series_label="",
        stem_suffix="vs_inlet_pressure",
        title_subject="inlet pressure",
    ),
    "pericyte_dilation_sweep": SweepAxisSpec(
        x_key="dilation_percent",
        x_label="Pericyte dilation (%)",
        series_key="inlet_pressure_pa",
        series_label="{value} Pa inlet",
        stem_suffix="vs_pericyte_dilation",
        title_subject="pericyte dilation",
    ),
    "pressure_and_pericyte_sweep": SweepAxisSpec(
        x_key="dilation_percent",
        x_label="Pericyte dilation (%)",
        series_key="inlet_pressure_pa",
        series_label="{value} Pa inlet",
        stem_suffix="vs_pericyte_dilation",
        title_subject="pericyte dilation",
    ),
    "arteriole_diameter_sweep": SweepAxisSpec(
        x_key="dilation_percent",
        x_label="Arteriole diameter change (%)",
        series_key="inlet_pressure_pa",
        series_label="{value} Pa inlet",
        stem_suffix="vs_arteriole_dilation",
        title_subject="arteriole dilation",
    ),
    "pressure_and_arteriole_sweep": SweepAxisSpec(
        x_key="dilation_percent",
        x_label="Arteriole diameter change (%)",
        series_key="inlet_pressure_pa",
        series_label="{value} Pa inlet",
        stem_suffix="vs_arteriole_dilation",
        title_subject="arteriole dilation",
    ),
    "capillary_diameter_sweep": SweepAxisSpec(
        x_key="dilation_percent",
        x_label="Capillary diameter change (%)",
        series_key="inlet_pressure_pa",
        series_label="{value} Pa inlet",
        stem_suffix="vs_capillary_dilation",
        title_subject="capillary dilation",
    ),
    "pressure_and_capillary_sweep": SweepAxisSpec(
        x_key="dilation_percent",
        x_label="Capillary diameter change (%)",
        series_key="inlet_pressure_pa",
        series_label="{value} Pa inlet",
        stem_suffix="vs_capillary_dilation",
        title_subject="capillary dilation",
    ),
    "pericyte_spacing_sweep": SweepAxisSpec(
        x_key="constriction_spacing_um",
        x_label="Constriction spacing (µm)",
        series_key=None,
        series_label="",
        stem_suffix="vs_pericyte_spacing",
        title_subject="pericyte spacing",
    ),
    "pericyte_length_sweep": SweepAxisSpec(
        x_key="constriction_length_um",
        x_label="Constriction length (µm)",
        series_key=None,
        series_label="",
        stem_suffix="vs_constriction_length",
        title_subject="constriction length",
    ),
}


def sweep_axis_for(perturbation_type: str) -> SweepAxisSpec:
    """Axis / labelling for *perturbation_type*, or a dilation-style fallback."""
    known = SWEEP_AXIS_BY_TYPE.get(str(perturbation_type))
    if known is not None:
        return known
    # Future ``*_sweep`` types: prefer dilation_percent when present in rows.
    return SweepAxisSpec(
        x_key="dilation_percent",
        x_label="Dilation (%)",
        series_key="inlet_pressure_pa",
        series_label="{value} Pa inlet",
        stem_suffix="vs_dilation",
        title_subject="dilation",
    )


def plot_sweep_curves(
    results: Iterable[Mapping[str, object]],
    output_dir: Path | str,
    *,
    axis: SweepAxisSpec,
) -> dict[str, str]:
    """Plot resistance and flow against the swept axis described by *axis*.

    Returns written paths keyed ``<column>_plot_path``, matching
    :func:`~haemolynx.visualization.dilation_curves.plot_dilation_curves`.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = list(results)
    if not rows:
        return {}

    written: dict[str, str] = {}
    for column, (stem_prefix, y_label, y_title) in SWEEP_Y_METRICS.items():
        figure, axes = plt.subplots(figsize=(8, 5))
        if axis.series_key is None:
            ordered = sorted(rows, key=lambda r: float(r[axis.x_key]))
            axes.plot(
                [float(r[axis.x_key]) for r in ordered],
                [float(r[column]) for r in ordered],
                marker="o",
                linewidth=2.0,
            )
        else:
            by_series: dict[Any, list[Mapping[str, object]]] = {}
            for row in rows:
                by_series.setdefault(row[axis.series_key], []).append(row)
            for series_value in sorted(by_series, key=lambda v: float(v)):
                series_rows = sorted(
                    by_series[series_value], key=lambda r: float(r[axis.x_key])
                )
                axes.plot(
                    [float(r[axis.x_key]) for r in series_rows],
                    [float(r[column]) for r in series_rows],
                    marker="o",
                    linewidth=2.0,
                    label=axis.series_label.format(value=series_value),
                )
            axes.legend()
        axes.set_xlabel(axis.x_label)
        axes.set_ylabel(y_label)
        axes.set_title(f"{y_title} vs {axis.title_subject}")
        axes.grid(True, alpha=0.3)
        figure.tight_layout()
        path = output_dir / f"{stem_prefix}_{axis.stem_suffix}.png"
        figure.savefig(path, dpi=200)
        plt.close(figure)
        written[f"{column}_plot_path"] = str(path)

    logger.info(
        "Sweep curves saved to: %s", ", ".join(sorted(written.values()))
    )
    return written


def export_sweep_perturbation_plots(
    perturbation_type: str,
    results: Sequence[Mapping[str, object]],
    output_dir: Path | str,
) -> list[Path]:
    """Alice-style PNGs for a sweep perturbation into *output_dir*."""
    written = plot_sweep_curves(
        results, output_dir, axis=sweep_axis_for(perturbation_type)
    )
    return [Path(path) for path in written.values()]


def _blank_image_for_graph(G: nx.MultiGraph) -> np.ndarray:
    """A tiny zero volume spanning the graph's micron extent (1 µm voxels)."""
    zs: list[float] = []
    ys: list[float] = []
    xs: list[float] = []
    for _node, data in G.nodes(data=True):
        pos = data.get("pos")
        if pos is None:
            continue
        zs.append(float(pos[0]))
        ys.append(float(pos[1]))
        xs.append(float(pos[2]))
    for _u, _v, _key, data in G.edges(keys=True, data=True):
        for point in data.get("voxels") or ():
            zs.append(float(point[0]))
            ys.append(float(point[1]))
            xs.append(float(point[2]))
    if not xs:
        return np.zeros((1, 8, 8), dtype=np.uint8)
    shape = (
        max(1, int(np.ceil(max(zs))) + 1),
        max(1, int(np.ceil(max(ys))) + 1),
        max(1, int(np.ceil(max(xs))) + 1),
    )
    return np.zeros(shape, dtype=np.uint8)


def export_non_sweep_perturbation_artifacts(
    G: nx.MultiGraph,
    output_dir: Path | str,
    settings: Mapping[str, Any],
    *,
    image: np.ndarray | None = None,
    name_stem: str | None = None,
) -> list[Path]:
    """Pipeline-like plots and statistics CSVs for one non-sweep perturbation.

    Matches the disk side of ``export_results`` (statistics CSVs, degree /
    overlay / branch-order figures). Deliberately skips VTK: existing stage
    tests require that a perturbation directory contain no ``.vtp``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    stem = name_stem or Path(str(settings.get("input_path", "perturbation"))).stem
    plot_image = image if image is not None else _blank_image_for_graph(G)
    voxel_size = (1.0, 1.0, 1.0)

    # Always write the tabular and figure set a perturbation comparison needs;
    # the baseline's ``statistics`` / ``visualize_results`` flags control the
    # published run, not these side-by-side arms.
    node_positions = nx.get_node_attributes(G, "pos")
    stats = statistics.compute_comprehensive_vessel_statistics(
        G,
        node_positions=node_positions,
        image_dimensions=plot_image.shape,
        voxel_size=voxel_size,
        statistics_mode=settings.get("statistics_mode", "fast"),
    )
    stats_csv = output_dir / f"{stem}_statistics.csv"
    statistics.export_statistics_to_csv(stats, stats_csv)
    written.append(stats_csv)

    branch_stats = statistics.compute_branch_order_statistics(
        G, node_positions=node_positions
    )
    branch_csv = output_dir / f"{stem}_branch_statistics.csv"
    statistics.export_branch_order_statistics_to_csv(branch_stats, branch_csv)
    written.append(branch_csv)

    degree_path = output_dir / "node_degree_distribution.png"
    plot_node_degree_distribution(
        G, save_path=degree_path, show=False, show_after_save=False
    )
    written.append(degree_path)

    render_mode = settings.get("final_render_mode", "2d")
    if render_mode == "3d":
        overlay_3d = output_dir / "edges_and_nodes_overlay_3d.html"
        visualize_3d_plotly(
            G,
            title="Edges and Nodes Overlay (Interactive 3D)",
            save_html_path=str(overlay_3d),
            show=False,
        )
        written.append(overlay_3d)
    else:
        overlay_2d = output_dir / "edges_and_nodes_overlay.png"
        visualize_edges_and_nodes(
            plot_image,
            G,
            save_path=overlay_2d,
            show=False,
            show_after_save=False,
            voxel_size=voxel_size,
        )
        written.append(overlay_2d)

    if settings.get("inlet_nodes"):
        branch_path = output_dir / "geometry_with_branch_orders.png"
        visualize_geometry_with_branch_orders(
            plot_image,
            G,
            group_above=8,
            save_path=branch_path,
            show=False,
            show_after_save=False,
            voxel_size=voxel_size,
        )
        written.append(branch_path)

    resistance_path = output_dir / "geometry_with_edge_resistance.png"
    visualize_geometry_with_edge_resistance(
        plot_image,
        G,
        save_path=resistance_path,
        show=False,
        show_after_save=False,
        voxel_size=voxel_size,
    )
    if resistance_path.exists():
        written.append(resistance_path)

    logger.info(
        "Non-sweep perturbation artifacts (%d file(s)) in %s",
        len(written),
        output_dir,
    )
    return written


def wants_napari_flow_layer(perturbation_type: str) -> bool:
    """Non-sweep perturbations get a named flow layer; sweeps do not."""
    return not is_sweep_perturbation(perturbation_type)
