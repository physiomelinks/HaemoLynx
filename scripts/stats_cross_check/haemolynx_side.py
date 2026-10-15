"""HaemoLynx's side: the run's own stages on the mask, then its statistics.

The graph is what ``run_pipeline_stages`` builds (segment, skeletonise,
build_network, with the run's settings), each vessel's diameter is read from
the mask by the run's own method (``haemodynamics.edt_diameter``, as the
Diameters stage does without a raw image), and the totals come from the rows
``statistics.compute_vessel_statistics_by_measure`` writes into the statistics
CSV. The per-segment arrays are read off the same graph the same way those
rows are -- bridges contracted, length over the straight distance between the
two nodes -- and checked against the rows' totals and means.

:func:`ledger` says where HaemoLynx's numbers part from a plain skeleton's:
skan on the mask's raw Lee skeleton, skan on the skeleton HaemoLynx cleaned,
the graph after each topology step, smoothing, the graph that is saved, and
the bridges contracted -- each row one definition.
"""
from __future__ import annotations

import copy
import csv
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import networkx as nx
import numpy as np

from .measures import Measurements, tortuosity_of, weighted_median

#: The statistics measures the comparison reads; the rest need inlets,
#: outlets or a solved flow, which a mask alone does not give.
MORPHOMETRY_MEASURES = frozenset({"basic", "tortuosity", "branching", "dead_ends"})


def run_overrides(mask_path: Path, voxel_size_xyz, output_dir: Path) -> dict[str, Any]:
    """What the run is told beyond its config: where the mask is and how big
    its voxels are, and to open no windows."""
    return {
        "input_path": Path(mask_path),
        "use_ilastik_segmentation": False,
        # Mask only: every tool reads the same segmentation and nothing else.
        "raw_image_path": None,
        "voxel_size_override_xyz": [float(v) for v in voxel_size_xyz],
        "voxel_size_policy": "override",
        "do_skeletonize": True,
        "crop_on_import": False,
        "vtk_output_prefix": Path(output_dir) / "run",
        "base_plot_dir": Path(output_dir) / "plots",
        "plot_dir": Path(output_dir) / "plots",
        "visualize_results": False,
        "save_step_artifacts": False,
    }


DEFINITIONS = {
    "skeleton": (
        "Lee thinning of the hole-filled mask (skimage, method='lee') after the run's "
        "skeleton cleaning (closing, gap bridging held to the mask, short-branch removal)"
    ),
    "segment": (
        "an edge of the cleaned graph: thirteen topology steps (cluster collapse, mask-supported "
        "reconnects, stub prune by radius, mask recovery, in-lumen and short-loop removal, "
        "doubled-strand removal, degree-2 merging), centrelines smoothed, zero-resistance "
        "bridges into thick vessels contracted as the statistics do"
    ),
    "length": "arc length of the smoothed centreline (`length`)",
    "tortuosity": "length over the straight distance between the edge's two nodes",
    "diameter": (
        "median over samples along the edge, none within 10 um of a junction, of the mask's "
        "own cross-section as a circle's diameter (`edt_diameter_method` cross_section)"
    ),
    "branch point": "node of degree >= 3 in the graph with parallel edges merged (`Number of Branching Points`)",
    "end point": "node of degree 1 (`Dead Ends`)",
}


@dataclass
class Build:
    """A run built to its graph, and what each step of the way left."""

    graph: nx.MultiGraph
    mask: np.ndarray
    skeleton: np.ndarray
    voxel_size_zyx: tuple[float, float, float]
    settings: dict
    #: ``(topology step, graph_counts after it)``, in order.
    steps: list = field(default_factory=list)
    #: The graph after the last topology step, before smoothing.
    before_smoothing: nx.MultiGraph | None = None


def _settings(config: Path | None, overrides: Mapping[str, Any]) -> tuple[dict, Any]:
    from haemolynx.pipeline import default_schema, resolve_settings

    schema = default_schema()
    if config is None:
        settings = resolve_settings(
            schema.defaults(), schema=schema, config_path=None, overrides=dict(overrides)
        )
    else:
        settings = resolve_settings(schema=schema, config_path=config, overrides=dict(overrides))
    return settings, schema


def graph_counts(G: nx.MultiGraph) -> dict[str, float]:
    """Total length, segments, branch points and end points of *G*, defined as
    the statistics define them (``Total Edge Length``, ``Total Edges``,
    ``Number of Branching Points`` on the graph with parallel edges merged,
    ``Dead Ends``)."""
    simple = nx.Graph(G)
    return {
        "length": float(sum(float(d.get("length") or 0.0) for *_e, d in G.edges(data=True))),
        "segments": G.number_of_edges(),
        "branch_points": sum(1 for n in simple if simple.degree(n) > 2),
        "end_points": sum(1 for n in G if G.degree(n) == 1),
    }


def build_graph(mask_path: Path, voxel_size_xyz, output_dir: Path, config: Path | None = None) -> Build:
    """Run the stages to ``build_network``, keeping what each topology step left."""
    from haemolynx.graph import STEP_LABELS
    from haemolynx.pipeline import run_pipeline_stages
    from haemolynx.pipeline.stages import TOPOLOGY_STEP

    settings, schema = _settings(config, run_overrides(mask_path, voxel_size_xyz, output_dir))
    built: dict[str, Any] = {}
    steps: list = []

    def keep(name: str, output: Any) -> None:
        if name == "build_network":
            built["network"] = output
        elif name.startswith(TOPOLOGY_STEP):
            label = name[len(TOPOLOGY_STEP):]
            steps.append((label, graph_counts(output)))
            if label == STEP_LABELS[-1]:
                built["before_smoothing"] = copy.deepcopy(output)

    G = run_pipeline_stages(settings, schema, on_stage_output=keep, stop_after="build_network")
    network = built["network"]
    return Build(
        graph=G,
        mask=network.volume.image,
        skeleton=network.volume.skeleton,
        voxel_size_zyx=tuple(float(v) for v in network.volume.voxel_size_zyx),
        settings=settings,
        steps=steps,
        before_smoothing=built.get("before_smoothing"),
    )


def measure_diameters(G: nx.MultiGraph, mask: np.ndarray, voxel_size_zyx, settings: Mapping[str, Any], method: str | None = None) -> dict:
    """Each vessel's ``edt_diameter_um``, read as the Diameters stage reads it
    without a raw image (``haemodynamics.apply._measure_edt_diameters``)."""
    from haemolynx.graph.thick_vessel_junctions import IS_ZERO_RESISTANCE
    from haemolynx.haemodynamics import edt_diameter

    vessels = [(u, v, k) for u, v, k, d in G.edges(keys=True, data=True) if not d.get(IS_ZERO_RESISTANCE)]
    return edt_diameter.measure_edge_diameters_from_binary_mask(
        G,
        binary_mask=mask,
        voxel_size_zyx=tuple(float(v) for v in voxel_size_zyx),
        sample_spacing_along_edge_um=float(settings["fwhm_sample_spacing_along_edge_um"]),
        branch_endpoint_exclusion_um=float(settings["edt_junction_proximity_exclusion_um"]),
        aggregation=settings["fwhm_edge_diameter_aggregation"],
        method=method or settings["edt_diameter_method"],
        min_resolvable_diameter_um=float(settings["edt_min_resolvable_diameter_um"]),
        edges=vessels,
    )


def statistics_rows(G: nx.MultiGraph, image_shape, voxel_size_zyx) -> dict[str, Any]:
    """The statistics CSV's rows for *G*, as ``export_results`` computes them."""
    from haemolynx import statistics

    sections = statistics.compute_vessel_statistics_by_measure(
        G,
        node_positions=nx.get_node_attributes(G, "pos"),
        image_dimensions=tuple(image_shape),
        voxel_size=tuple(voxel_size_zyx),
        statistics_mode="fast",
        enabled_measures=MORPHOMETRY_MEASURES,
    )
    return statistics.flatten_statistics_sections(sections, "fast")


def write_statistics_csv(rows: Mapping[str, Any], path: Path) -> Path:
    from haemolynx import statistics

    return statistics.export_statistics_to_csv(dict(rows), path)


#: The statistics CSV's rows the comparison reads. The CSV writes each
#: metric's name without its unit (that goes in the Unit column) and each
#: number to six significant figures.
TOTAL_LENGTH = "Total Edge Length"
SEGMENTS = "Total Edges"
BRANCH_POINTS = "Number of Branching Points"
END_POINTS = "Dead Ends"
MEAN_TORTUOSITY = "Average Tortuosity Index"
ENDS_LEAVING_IMAGE = "Dead Ends Leaving the Image"
BRIDGES = "Zero-resistance Bridges"
CSV_PRECISION = 1e-5


def read_statistics_csv(path: Path) -> dict[str, str]:
    """``{metric: value}`` from a statistics CSV."""
    with Path(path).open(newline="", encoding="utf-8") as f:
        return {row["Metric"]: row["Value"] for row in csv.DictReader(f)}


def vessel_edges(G: nx.MultiGraph):
    """The edges the statistics count, as they count them: zero-resistance
    bridges contracted (``graph.thick_vessel_junctions.contract_bridges``)."""
    from haemolynx.graph.thick_vessel_junctions import IS_ZERO_RESISTANCE, contract_bridges

    if any(d.get(IS_ZERO_RESISTANCE) for *_e, d in G.edges(data=True)):
        G, _became = contract_bridges(G)
    return G


def measurements(G: nx.MultiGraph, rows: Mapping[str, Any], *, tool: str = "HaemoLynx") -> Measurements:
    """*G*'s segments as a :class:`Measurements`, its totals the CSV's *rows*."""
    H = vessel_edges(G)
    pos = nx.get_node_attributes(H, "pos")
    lengths, chords, diameters = [], [], []
    for u, v, d in H.edges(data=True):
        if d.get("length") is None:
            continue
        lengths.append(float(d["length"]))
        chords.append(float(np.linalg.norm(np.asarray(pos[u], float) - np.asarray(pos[v], float))))
        diameters.append(float(d.get("edt_diameter_um", math.nan)))
    tortuosity, loops = tortuosity_of(lengths, chords)
    total = float(rows[TOTAL_LENGTH])
    if not math.isclose(total, sum(lengths), rel_tol=CSV_PRECISION):
        raise AssertionError(f"segments sum to {sum(lengths)} um, the CSV says {total} um")
    mean_tortuosity = float(rows[MEAN_TORTUOSITY])
    finite = tortuosity[np.isfinite(tortuosity)]
    if finite.size and not math.isclose(mean_tortuosity, float(np.mean(finite)), rel_tol=CSV_PRECISION):
        raise AssertionError("per-segment tortuosity does not average to the CSV's row")
    return Measurements(
        tool=tool,
        total_length_um=total,
        segments=int(float(rows[SEGMENTS])),
        branch_points=int(float(rows[BRANCH_POINTS])),
        end_points=int(float(rows[END_POINTS])),
        segment_lengths_um=np.asarray(lengths),
        tortuosity=tortuosity,
        diameters_um=np.asarray(diameters),
        loops=loops,
        definitions=dict(DEFINITIONS),
        extra={
            "Zero-resistance bridges (contracted)": int(float(rows.get(BRIDGES, 0) or 0)),
            "Dead ends leaving the image": int(float(rows.get(ENDS_LEAVING_IMAGE, 0) or 0)),
        },
    )


def smoothed_counts(build: Build) -> dict[str, float] | None:
    """:func:`graph_counts` of the last topology step's graph with only the
    run's centreline smoothing applied -- what smoothing alone changes."""
    from haemolynx import graph
    from haemolynx.io.load import _to_binary_volume_for_skeletonization

    settings = build.settings
    if build.before_smoothing is None or not settings["smooth_centrelines"]:
        return None
    G = copy.deepcopy(build.before_smoothing)
    binary = _to_binary_volume_for_skeletonization(build.mask)
    graph.smooth_graph_centrelines(
        G,
        build.skeleton,
        voxel_size_zyx=build.voxel_size_zyx,
        method=settings["centreline_smoothing_method"],
        iterations=settings["centreline_smoothing_iterations"],
        max_deviation=settings["centreline_max_deviation"],
        radius_at=graph.assemble.mask_radius_sampler(binary, build.voxel_size_zyx, 1.0),
    )
    return graph_counts(G)


#: How many of the nearest centreline samples a skeleton voxel is checked
#: against: the nearest can be a capillary passing a wide vessel's strand.
_LUMEN_NEIGHBOURS = 16


def centreline_samples(G: nx.MultiGraph, step_um: float = 0.5) -> np.ndarray:
    """Points every *step_um* along each vessel's centreline (bridges left
    out: they are not vessel), ``(z, y, x)`` microns."""
    from haemolynx.graph.thick_vessel_junctions import IS_ZERO_RESISTANCE

    pieces = []
    for *_e, d in G.edges(data=True):
        path = np.asarray(d.get("voxels") or (), dtype=float)
        if d.get(IS_ZERO_RESISTANCE) or len(path) < 2:
            continue
        for a, b in zip(path[:-1], path[1:]):
            count = max(1, int(np.ceil(np.linalg.norm(b - a) / step_um)))
            pieces.append(a + np.linspace(0.0, 1.0, count, endpoint=False)[:, None] * (b - a))
        pieces.append(path[-1:])
    return np.vstack(pieces) if pieces else np.empty((0, 3))


def in_drawn_vessels(build: Build, skeleton: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Each voxel of *skeleton* (``(z, y, x)`` microns), and whether it lies
    in a vessel HaemoLynx drew.

    A skeleton voxel is in one when it is within that vessel's lumen radius
    (the mask's distance transform at the centreline) plus a voxel of a point
    on its centreline: the same vessel skeletonised again -- a second strand
    through a wide lumen, a spur, a junction's knot -- which HaemoLynx's
    clean-up takes out by definition. Outside every one, it is a vessel the
    graph does not draw. A flattened lumen's far side counts as outside, so
    "in" is a lower bound."""
    from scipy import ndimage as ndi
    from scipy.spatial import cKDTree

    spacing = np.asarray(build.voxel_size_zyx, dtype=float)
    mask = np.asarray(build.mask) > 0
    distance = ndi.distance_transform_edt(mask, sampling=spacing)
    samples = centreline_samples(build.graph)
    index = np.clip(np.rint(samples / spacing).astype(int), 0, np.asarray(mask.shape) - 1)
    radius = distance[tuple(index.T)]
    margin = float(spacing.max())
    voxels = np.argwhere(np.asarray(skeleton) > 0) * spacing
    gaps, nearest = cKDTree(samples).query(
        voxels, k=_LUMEN_NEIGHBOURS, distance_upper_bound=float(radius.max()) + margin
    )
    found = np.isfinite(gaps)
    reach = np.zeros(gaps.shape)
    reach[found] = radius[nearest[found]] + margin
    return voxels, np.any(found & (gaps <= reach), axis=1)


def skeleton_in_haemolynx_lumen(build: Build, skeleton: np.ndarray) -> dict[str, float]:
    """How much of a plain *skeleton* lies in a vessel HaemoLynx drew
    (:func:`in_drawn_vessels`)."""
    voxels, inside = in_drawn_vessels(build, skeleton)
    return {"skeleton voxels": int(len(voxels)), "share in a drawn vessel": float(inside.mean()) if len(voxels) else math.nan}


def ledger(build: Build, raw_skan: Measurements) -> list[tuple[str, dict[str, float]]]:
    """``(what changed, counts after it)`` from the raw skeleton to the CSV."""
    from . import skan_side

    def counts(m: Measurements) -> dict[str, float]:
        return {"length": m.total_length_um, "segments": m.segments,
                "branch_points": m.branch_points, "end_points": m.end_points}

    cleaned = skan_side.run(build.mask, build.voxel_size_zyx, skeleton=build.skeleton)
    rows = [("skan on the mask's Lee skeleton", counts(raw_skan)),
            ("skan on HaemoLynx's cleaned skeleton (skeleton cleaning)", counts(cleaned))]
    rows += [(f"topology step: {label}", c) for label, c in build.steps]
    smoothed = smoothed_counts(build)
    if smoothed is not None:
        rows.append(("centreline smoothing (alone, on the last step's graph)", smoothed))
    rows.append(("saved graph (clean-up after smoothing, thick-vessel split)", graph_counts(build.graph)))
    rows.append(("statistics: zero-resistance bridges contracted", graph_counts(vessel_edges(build.graph))))
    return rows


def ledger_table(rows: list[tuple[str, dict[str, float]]]) -> str:
    """The ledger as markdown, each row with its change from the one before."""
    lines = ["| After | Length (um) | Segments | Branch points | End points |", "|---|---|---|---|---|"]
    previous = None
    for label, c in rows:
        cells = []
        for key, fmt in (("length", "{:,.0f}"), ("segments", "{:,}"), ("branch_points", "{:,}"), ("end_points", "{:,}")):
            text = fmt.format(c[key])
            if previous is not None and c[key] != previous[key]:
                delta = c[key] - previous[key]
                text += f" ({delta:+,.0f})" if key == "length" else f" ({delta:+,})"
            cells.append(text)
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
        previous = c
    return "\n".join(lines)


#: What :func:`run` adds to the HaemoLynx side's ``extra``: the same edges'
#: diameters by the definition skan and maskel use.
INSCRIBED = "Diameter by HaemoLynx's inscribed_radius method (2 x EDT: skan's and maskel's definition), median / length-weighted median (um)"


def inscribed_diameters(build: Build) -> tuple[float, float]:
    """The vessels' diameters by the ``inscribed_radius`` method, on a copy of
    the graph: median and length-weighted median."""
    G = copy.deepcopy(build.graph)
    measure_diameters(G, build.mask, build.voxel_size_zyx, build.settings, method="inscribed_radius")
    H = vessel_edges(G)
    diameters = np.asarray([float(d.get("edt_diameter_um", np.nan)) for *_e, d in H.edges(data=True)])
    lengths = np.asarray([float(d.get("length") or np.nan) for *_e, d in H.edges(data=True)])
    return float(np.nanmedian(diameters)), weighted_median(diameters, lengths)


def run(mask_path: Path, voxel_size_xyz, output_dir: Path, config: Path | None = None) -> tuple[Measurements, Build, Path]:
    """Build, measure and write the statistics CSV; returns the measurements,
    the build and the CSV's path."""
    output_dir = Path(output_dir)
    build = build_graph(mask_path, voxel_size_xyz, output_dir, config)
    measure_diameters(build.graph, build.mask, build.voxel_size_zyx, build.settings)
    rows = statistics_rows(build.graph, build.mask.shape, build.voxel_size_zyx)
    csv_path = write_statistics_csv(rows, output_dir / f"{Path(mask_path).stem}_statistics.csv")
    result = measurements(build.graph, read_statistics_csv(csv_path))
    result.extra[INSCRIBED] = [round(v, 3) for v in inscribed_diameters(build)]
    return result, build, csv_path
