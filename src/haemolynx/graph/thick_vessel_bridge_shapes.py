"""What shape each thick-vessel bridge has: read-only, for the run's log and
``scripts/graph_artefacts.py`` (PLAN_thick_vessel_bridges.md, Phase 0).

A bridge (:data:`~haemolynx.graph.thick_vessel_junctions.IS_ZERO_RESISTANCE`)
is meant to be the thick vessel's radius between its centreline and a vessel
opening into it. Measured against the centreline tree the split bridges to,
each bridge is one of :data:`BRIDGE_SHAPES`, judged in this order:

- ``fork``: made at a fork of the thick centreline, over a narrower branch's
  stretch inside the trunk's lumen (``bridge_forks``; marked
  :data:`~haemolynx.graph.thick_vessel_junctions.THICK_FORK_BRIDGE`);
- ``along_centreline``: at least half its length within the split's tolerance
  of the tree -- the thick vessel's own centreline given no resistance;
- ``radius_shaped``: its inner end (the one nearer the tree) on the tree, its
  outer end at the thick-vessel region's wall, and no longer than
  :data:`RADIUS_SHAPED_MAX_RATIO` times the thick vessel's lumen radius at the
  tree point nearest its inner end;
- ``longer_than_radius``: its inner end on the tree, but not radius-shaped;
- ``on_another_bridge``: its inner end off the tree, on another bridge;
- ``wall_to_wall``: never reaching the tree, both ends at the region's wall --
  a vessel passing through the region's margin given no resistance;
- ``other``: never reaching the centreline otherwise.

The six after ``fork`` are the categories of the 2026-10-09 audit of E14.5's
bridges (the plan's finding 2), when 290 of 393 gave no resistance to
something that is vessel. Each bridge's length over that lumen radius is
reported too, and whether it keeps its drawn path rather than a straight line
(:data:`~haemolynx.graph.thick_vessel_junctions.DRAWN_BRIDGE`: a bridge whose
straight line would leave the segmentation).
"""
from __future__ import annotations

from typing import Any

import networkx as nx
import numpy as np
from scipy.spatial import cKDTree

from ._helpers import densify_polyline, edge_sample_points, points_inside_mask
from .thick_vessel_junctions import (
    CENTRELINE_TOLERANCE_PER_RADIUS,
    CENTRELINE_TOLERANCE_UM,
    DRAWN_BRIDGE,
    IS_ZERO_RESISTANCE,
    THICK_FORK_BRIDGE,
    _centreline_voxels,
    _inscribed_radius_um,
    _tree_radius_um,
    thick_vessel_centreline_voxels,
)

BRIDGE_SHAPES = (
    "radius_shaped",
    "longer_than_radius",
    "along_centreline",
    "on_another_bridge",
    "wall_to_wall",
    "other",
    "fork",
)

#: How each shape reads in a sentence.
SHAPE_WORDS = {
    "radius_shaped": "radius-shaped",
    "longer_than_radius": "reaching the centreline but longer than the radius",
    "along_centreline": "lying along the thick centreline",
    "on_another_bridge": "ending on another bridge",
    "wall_to_wall": "wall to wall through the margin",
    "other": "never reaching the centreline",
    "fork": "over a branch at a fork",
}

#: A radius-shaped bridge is no longer than this many lumen radii (finding 2's
#: rule, kept so the counts compare with that audit's).
RADIUS_SHAPED_MAX_RATIO = 1.5

#: A bridge with at least this share of its length on the tree lies along it.
ALONG_CENTRELINE_MIN_SHARE = 0.5

#: An end outside the thick-vessel region, or this close (µm) to its outside,
#: is at its wall: the split's flicker length.
WALL_REACH_UM = 2.0

_SAMPLE_STEP_UM = 0.5


def diagnose_thick_vessel_bridges(
    G: nx.MultiGraph,
    thick_mask: np.ndarray,
    centreline: np.ndarray | None = None,
    segmentation: np.ndarray | None = None,
    *,
    voxel_size_zyx: tuple[float, float, float] = (1.0, 1.0, 1.0),
    tree_radius_um: np.ndarray | None = None,
    centreline_tolerance_um: float = CENTRELINE_TOLERANCE_UM,
) -> dict[str, Any]:
    """Every bridge of *G* by shape (:data:`BRIDGE_SHAPES`), against the thick
    vessels' centreline tree. Read-only.

    *centreline* is the tree (a volume of it or its voxels), found again
    from *thick_mask* when not given; *segmentation* the lumen its radius is
    read in (the region stands in without one); *tree_radius_um* that
    radius at each tree point, when a caller has it already. The tolerance is
    the split's: *centreline_tolerance_um* or
    :data:`~haemolynx.graph.thick_vessel_junctions.CENTRELINE_TOLERANCE_PER_RADIUS`
    of the radius, whichever is more.

    Returns ``bridge_count``, ``counts`` and ``median_length_um`` per shape,
    ``length_over_radius`` (``median``, ``p90``, ``max`` over every bridge
    with a radius), ``drawn_path_count`` and ``bridges``: one row per bridge.
    """
    vs = np.asarray(voxel_size_zyx, dtype=float)
    bridges = [
        (u, v, key, data) for u, v, key, data in G.edges(keys=True, data=True)
        if data.get(IS_ZERO_RESISTANCE) and u != v
    ]
    report: dict[str, Any] = {
        "bridge_count": len(bridges),
        "counts": {shape: 0 for shape in BRIDGE_SHAPES},
        "median_length_um": {shape: None for shape in BRIDGE_SHAPES},
        "length_over_radius": {"median": None, "p90": None, "max": None},
        "drawn_path_count": 0,
        "bridges": [],
    }
    thick = np.asarray(thick_mask, dtype=bool)
    if not bridges or not thick.any():
        return report
    tree_voxels = (
        thick_vessel_centreline_voxels(thick, tuple(vs)) if centreline is None
        else _centreline_voxels(centreline, thick.ndim)
    )
    tree_points = np.asarray(tree_voxels, dtype=float) * vs
    if len(tree_points) == 0:
        return report
    if segmentation is not None and np.shape(segmentation) != thick.shape:
        segmentation = None
    lumen = thick if segmentation is None else np.asarray(segmentation, dtype=bool)
    if tree_radius_um is not None and np.shape(tree_radius_um) == (len(tree_points),):
        radius = np.asarray(tree_radius_um, dtype=float)
    else:
        radius = _tree_radius_um(tree_points, lumen, thick, vs)
    finite = np.where(np.isfinite(radius), radius, 0.0)
    tolerance = np.maximum(float(centreline_tolerance_um), CENTRELINE_TOLERANCE_PER_RADIUS * finite)
    tree = cKDTree(tree_points)
    node_pos = {n: np.asarray(d["pos"], dtype=float) for n, d in G.nodes(data=True) if "pos" in d}

    bridges_at: dict[Any, int] = {}
    for u, v, _key, _data in bridges:
        for n in (u, v):
            bridges_at[n] = bridges_at.get(n, 0) + 1
    ends = sorted({n for u, v, _k, _d in bridges for n in (u, v) if n in node_pos}, key=str)
    end_points = np.asarray([node_pos[n] for n in ends], dtype=float).reshape(-1, 3)
    in_region = points_inside_mask(end_points, thick, voxel_size_zyx=tuple(vs))
    depth = np.full(len(ends), 0.0)
    if in_region.any():
        depth[in_region] = _inscribed_radius_um(end_points[in_region], thick, vs)
    at_wall = {n: bool(not inside or d <= WALL_REACH_UM) for n, inside, d in zip(ends, in_region, depth)}

    rows = []
    for u, v, key, data in bridges:
        if u not in node_pos or v not in node_pos:
            continue
        path = densify_polyline(edge_sample_points(u, v, data, node_pos), max_step_um=_SAMPLE_STEP_UM)
        if len(path) < 2:
            continue
        steps = np.linalg.norm(np.diff(path, axis=0), axis=1)
        length = float(steps.sum())
        distance, nearest = tree.query(path)
        on = distance <= tolerance[nearest]
        share = float(steps[on[:-1] & on[1:]].sum() / length) if length > 0 else 0.0
        inner_at = 0 if distance[0] <= distance[-1] else len(path) - 1
        inner, outer = (u, v) if inner_at == 0 else (v, u)
        outer_at = len(path) - 1 - inner_at
        local_radius = float(finite[nearest[inner_at]])
        ratio = length / local_radius if local_radius > 0 else float("inf")
        inner_on_tree = bool(on[inner_at])
        if data.get(THICK_FORK_BRIDGE):
            shape = "fork"
        elif share >= ALONG_CENTRELINE_MIN_SHARE:
            shape = "along_centreline"
        elif inner_on_tree:
            radius_shaped = at_wall.get(outer, False) and ratio <= RADIUS_SHAPED_MAX_RATIO
            shape = "radius_shaped" if radius_shaped else "longer_than_radius"
        elif bridges_at.get(inner, 0) > 1:
            shape = "on_another_bridge"
        elif not on.any() and at_wall.get(inner, False) and at_wall.get(outer, False):
            shape = "wall_to_wall"
        else:
            shape = "other"
        rows.append({
            "u": u, "v": v, "key": key, "shape": shape,
            "length_um": length,
            "local_radius_um": local_radius,
            "length_over_radius": ratio,
            "inner_node": inner, "inner_to_tree_um": float(distance[inner_at]),
            "outer_node": outer, "outer_to_tree_um": float(distance[outer_at]),
            "outer_at_wall": at_wall.get(outer, False),
            "share_on_centreline": share,
            "drawn_path": bool(data.get(DRAWN_BRIDGE)),
        })

    report["bridges"] = rows
    for shape in BRIDGE_SHAPES:
        lengths = [row["length_um"] for row in rows if row["shape"] == shape]
        report["counts"][shape] = len(lengths)
        report["median_length_um"][shape] = float(np.median(lengths)) if lengths else None
    ratios = np.asarray([row["length_over_radius"] for row in rows if np.isfinite(row["length_over_radius"])])
    if len(ratios):
        report["length_over_radius"] = {
            "median": float(np.median(ratios)),
            "p90": float(np.percentile(ratios, 90)),
            "max": float(ratios.max()),
        }
    report["drawn_path_count"] = sum(bool(row["drawn_path"]) for row in rows)
    return report


def format_thick_vessel_bridges_report(report: dict[str, Any]) -> str:
    """A one-line summary of :func:`diagnose_thick_vessel_bridges`."""
    counts = report.get("counts", {})
    shapes = ", ".join(
        f"{counts[shape]} {SHAPE_WORDS[shape]}" for shape in BRIDGE_SHAPES if counts.get(shape)
    )
    ratio = report.get("length_over_radius", {})
    line = f"Thick-vessel bridges: {report.get('bridge_count', 0)} bridge(s)"
    if shapes:
        line += f": {shapes}"
    if ratio.get("median") is not None:
        line += (
            f"; length over the thick vessel's lumen radius median {ratio['median']:.2f} "
            f"(90th percentile {ratio['p90']:.2f}, largest {ratio['max']:.2f})"
        )
    drawn = report.get("drawn_path_count", 0)
    if drawn:
        line += f"; {drawn} drawn along their path, a straight one leaving the segmentation"
    return line + "."
