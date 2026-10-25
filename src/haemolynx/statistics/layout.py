"""How the vessels lie in the tissue: the shape of the junctions, how far the
tissue is from a vessel, and the vessels that run along the stack axis.

Three measures of the statistics CSV (``PLAN_synthetic_physiology.md``; Harvey,
2026-10-25), each its own ``statistics_*`` toggle:

- **Junction shape** (:func:`compute_junction_shape`): at every junction of
  three vessels, the three angles between them, each vessel's direction read
  :data:`JUNCTION_DIRECTION_REACH_UM` along its centreline from the node. A
  wide, flat Y has a smallest angle near 120 degrees and angles summing to 360;
  two vessels leaving a node together make an acute pair
  (:data:`ACUTE_JUNCTION_DEG`).
- **Tissue coverage** (:func:`compute_tissue_coverage`): the distance from
  points through the image box to the nearest vessel centreline -- how much
  of the box lies near a vessel, and how far its holes reach.
- **Vessel columns** (:func:`compute_vessel_columns`): vessels whose chord runs
  within :data:`COLUMN_MAX_TILT_DEG` of the stack (z) axis, their share of the
  network's length and how evenly they are spread across the plane (the
  Clark-Evans ratio of their positions: 1 at random, above 1 evenly spaced,
  below 1 bunched).

On E14.5's MCA run the junctions are wide and flat (smallest angle median 97
degrees, 1% acute, sum 356), 4.8% of the box lies beyond 50 um of a capillary
and its columns are evenly spread (Clark-Evans 0.90); the synthetic networks
of grid D1d were not (``docs/notes/synthetic.md``).
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional, Union

import networkx as nx
import numpy as np
from scipy.spatial import cKDTree

from haemolynx.geometry import resample_at_step
from haemolynx.visualization.geometry import edge_polyline

from ._sampling import REPRODUCIBILITY_SEED

#: How far along each vessel from the node its direction is read, in microns:
#: past a centreline's last voxel of wobble at the node, short of its bend.
#: E14.5's junction shape is the same read at 4, 8 or 15 um.
JUNCTION_DIRECTION_REACH_UM = 8.0
#: Two vessels of a junction closer than this are an acute pair.
ACUTE_JUNCTION_DEG = 45.0
#: The coverage's two distances, as ``synthetic.organisation``'s.
COVERAGE_NEAR_UM = 20.0
COVERAGE_FAR_UM = 50.0
#: Spacing of the probe grid through the box; a box needing more probes than
#: :data:`COVERAGE_MAX_PROBES` gets that many at random (seeded) instead.
COVERAGE_STEP_UM = 10.0
COVERAGE_MAX_PROBES = 200_000
#: Spacing of the points each vessel is sampled at for the coverage.
COVERAGE_SAMPLE_STEP_UM = 2.0
#: A column: a vessel whose chord is at least this long and within this angle
#: of the stack axis.
COLUMN_MAX_TILT_DEG = 30.0
COLUMN_MIN_CHORD_UM = 10.0

_NO_JUNCTIONS = "N/A (no junction of three vessels)"
_NO_BOX = "N/A (no vessels to bound a box)"
_FEW_COLUMNS = "N/A (fewer than five columns)"


def _edges(G: Union[nx.Graph, nx.MultiGraph]) -> list:
    if isinstance(G, (nx.MultiGraph, nx.MultiDiGraph)):
        return list(G.edges(keys=True, data=True))
    return [(u, v, 0, d) for u, v, d in G.edges(data=True)]


def _polyline(G, u, v, data) -> Optional[np.ndarray]:
    try:
        points = np.asarray(edge_polyline(G, u, v, data), dtype=float)
    except (TypeError, ValueError, KeyError):
        return None
    if points.ndim != 2 or points.shape[0] < 2:
        return None
    return points


def _direction(points: np.ndarray, reach_um: float) -> Optional[np.ndarray]:
    """The unit direction from *points*' first point to the point *reach_um*
    along it (its last, when it is shorter)."""
    dense = resample_at_step(points, 1.0)
    k = min(len(dense) - 1, max(1, int(round(reach_um))))
    step = dense[k] - dense[0]
    norm = float(np.linalg.norm(step))
    return step / norm if norm > 1e-9 else None


def junction_directions(
    G: Union[nx.Graph, nx.MultiGraph], *, reach_um: float = JUNCTION_DIRECTION_REACH_UM
) -> Dict[Any, list]:
    """Each node's vessels' directions leaving it (one per vessel end at the
    node: a loop on the node gives two), read *reach_um* along each
    centreline. A vessel too short to give one is left out."""
    out: Dict[Any, list] = {n: [] for n in G}
    for u, v, _k, data in _edges(G):
        points = _polyline(G, u, v, data)
        if points is None:
            continue
        for node, path in ((u, points), (v, points[::-1])):
            direction = _direction(path, reach_um)
            if direction is not None:
                out[node].append(direction)
    return out


def junction_angle_triples(
    G: Union[nx.Graph, nx.MultiGraph], *, reach_um: float = JUNCTION_DIRECTION_REACH_UM
) -> list[tuple[float, float, float]]:
    """The three angles between the vessels of every junction of exactly
    three, smallest first, in degrees."""
    triples = []
    for node, directions in junction_directions(G, reach_um=reach_um).items():
        if G.degree(node) != 3 or len(directions) != 3:
            continue
        a = [
            math.degrees(math.acos(float(np.clip(directions[i] @ directions[j], -1.0, 1.0))))
            for i, j in ((0, 1), (0, 2), (1, 2))
        ]
        triples.append(tuple(sorted(a)))
    return triples


def compute_junction_shape(G: Union[nx.Graph, nx.MultiGraph]) -> Dict[str, Any]:
    """The junction-shape rows of the statistics CSV (see the module)."""
    triples = np.asarray(junction_angle_triples(G), dtype=float).reshape(-1, 3)
    wider = sum(1 for n in G if G.degree(n) >= 4)
    rows: Dict[str, Any] = {
        "Three-Vessel Junctions Measured": int(len(triples)),
        "Junctions of Four or More Vessels": int(wider),
    }
    if not len(triples):
        for name in (
            "Junction Smallest Angle Median (degrees)",
            "Junction Smallest Angle 25th Percentile (degrees)",
            "Junction Largest Angle Median (degrees)",
            "Acute Junction Share (fraction)",
            "Junction Angle Sum Median (degrees)",
        ):
            rows[name] = _NO_JUNCTIONS
        return rows
    rows["Junction Smallest Angle Median (degrees)"] = float(np.median(triples[:, 0]))
    rows["Junction Smallest Angle 25th Percentile (degrees)"] = float(np.percentile(triples[:, 0], 25))
    rows["Junction Largest Angle Median (degrees)"] = float(np.median(triples[:, 2]))
    rows["Acute Junction Share (fraction)"] = float(np.mean(triples[:, 0] < ACUTE_JUNCTION_DEG))
    rows["Junction Angle Sum Median (degrees)"] = float(np.median(triples.sum(axis=1)))
    return rows


def network_box(G: Union[nx.Graph, nx.MultiGraph]) -> Optional[tuple[np.ndarray, np.ndarray, str]]:
    """``(low, high, what)``: the image box (``G.graph["image_extent_um"]``,
    from 0) the run read, else the box the vessels span; None with no vessel."""
    extent = G.graph.get("image_extent_um")
    if extent is not None:
        high = np.asarray(extent, dtype=float).reshape(3)
        if np.all(high > 0):
            return np.zeros(3), high, "image"
    points = [p for u, v, _k, d in _edges(G) for p in [_polyline(G, u, v, d)] if p is not None]
    if not points:
        return None
    every = np.vstack(points)
    return every.min(axis=0), every.max(axis=0), "network bounding box"


def vessel_sample_points(G: Union[nx.Graph, nx.MultiGraph], step_um: float = COVERAGE_SAMPLE_STEP_UM) -> np.ndarray:
    """Every vessel's centreline, a point every *step_um*."""
    chunks = []
    for u, v, _k, data in _edges(G):
        points = _polyline(G, u, v, data)
        if points is not None:
            chunks.append(resample_at_step(points, step_um))
    return np.vstack(chunks) if chunks else np.zeros((0, 3))


def coverage_probes(low: np.ndarray, high: np.ndarray) -> np.ndarray:
    """Points through the box: a grid :data:`COVERAGE_STEP_UM` apart, or
    :data:`COVERAGE_MAX_PROBES` at random (seeded) when that grid has more."""
    size = np.maximum(high - low, 1e-9)
    counts = np.maximum(1, np.floor(size / COVERAGE_STEP_UM).astype(int))
    if int(np.prod(counts)) <= COVERAGE_MAX_PROBES:
        axes = [low[a] + (np.arange(counts[a]) + 0.5) * size[a] / counts[a] for a in range(3)]
        return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    rng = np.random.default_rng(REPRODUCIBILITY_SEED)
    return low + rng.uniform(0.0, 1.0, size=(COVERAGE_MAX_PROBES, 3)) * size


def tissue_distances(G: Union[nx.Graph, nx.MultiGraph]) -> Optional[tuple[np.ndarray, str]]:
    """Each probe's distance to the nearest vessel centreline, and what the
    box was; None without vessels."""
    box = network_box(G)
    vessels = vessel_sample_points(G)
    if box is None or not len(vessels):
        return None
    low, high, what = box
    distances, _ = cKDTree(vessels).query(coverage_probes(low, high))
    return np.asarray(distances, dtype=float), what


def compute_tissue_coverage(G: Union[nx.Graph, nx.MultiGraph]) -> Dict[str, Any]:
    """The coverage rows of the statistics CSV (see the module)."""
    names = (
        "Box Within 20 um of a Vessel (fraction)",
        "Box Beyond 50 um of a Vessel (fraction)",
        "Median Distance to Nearest Vessel (microns)",
        "90th Percentile Distance to Nearest Vessel (microns)",
    )
    measured = tissue_distances(G)
    if measured is None:
        return {**{name: _NO_BOX for name in names}, "Coverage Box": _NO_BOX, "Coverage Probe Points": 0}
    distances, what = measured
    return {
        names[0]: float(np.mean(distances <= COVERAGE_NEAR_UM)),
        names[1]: float(np.mean(distances > COVERAGE_FAR_UM)),
        names[2]: float(np.median(distances)),
        names[3]: float(np.percentile(distances, 90)),
        "Coverage Box": what,
        "Coverage Probe Points": int(len(distances)),
    }


def column_vessels(G: Union[nx.Graph, nx.MultiGraph]) -> tuple[list[tuple[np.ndarray, float, float]], float]:
    """Each column's ``(chord midpoint, arc length, chord length)``, and the
    network's whole length (see the module)."""
    cos_tilt = math.cos(math.radians(COLUMN_MAX_TILT_DEG))
    columns, total = [], 0.0
    for u, v, _k, data in _edges(G):
        points = _polyline(G, u, v, data)
        if points is None:
            continue
        arc = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
        total += arc
        chord_vector = points[-1] - points[0]
        chord = float(np.linalg.norm(chord_vector))
        if chord >= COLUMN_MIN_CHORD_UM and abs(float(chord_vector[0])) / chord >= cos_tilt:
            columns.append(((points[0] + points[-1]) / 2.0, arc, chord))
    return columns, total


def clark_evans_2d(points: np.ndarray, area_um2: float) -> Optional[float]:
    """The mean nearest-neighbour distance of *points* in the plane over a
    random pattern's of the same density (0.5 sqrt(area / n)); None for fewer
    than five. No edge correction: a box's edges lengthen it a little."""
    if len(points) < 5 or area_um2 <= 0:
        return None
    nearest, _ = cKDTree(points).query(points, k=2)
    return float(nearest[:, 1].mean()) / (0.5 * math.sqrt(area_um2 / len(points)))


def compute_vessel_columns(G: Union[nx.Graph, nx.MultiGraph]) -> Dict[str, Any]:
    """The column rows of the statistics CSV (see the module)."""
    columns, total = column_vessels(G)
    box = network_box(G)
    area = float((box[1][1] - box[0][1]) * (box[1][2] - box[0][2])) if box is not None else 0.0
    rows: Dict[str, Any] = {
        "Column Vessels (within 30 degrees of z)": len(columns),
        "Column Share of Vessel Length (fraction)": (
            float(sum(c[1] for c in columns) / total) if total > 0 else _NO_BOX
        ),
        "Columns per mm² of Image Plane": float(len(columns) / area * 1e6) if area > 0 else _NO_BOX,
    }
    xy = np.array([c[0][1:] for c in columns], dtype=float).reshape(-1, 2)
    ratio = clark_evans_2d(xy, area)
    if ratio is None:
        rows["Column Spacing Median (microns)"] = _FEW_COLUMNS
        rows["Column Spacing Clark-Evans Ratio"] = _FEW_COLUMNS
    else:
        nearest, _ = cKDTree(xy).query(xy, k=2)
        rows["Column Spacing Median (microns)"] = float(np.median(nearest[:, 1]))
        rows["Column Spacing Clark-Evans Ratio"] = ratio
    rows["Median Column Tortuosity Index"] = (
        float(np.median([c[1] / c[2] for c in columns])) if columns else "N/A (no column)"
    )
    return rows


__all__ = [
    "ACUTE_JUNCTION_DEG",
    "COLUMN_MAX_TILT_DEG",
    "COLUMN_MIN_CHORD_UM",
    "COVERAGE_FAR_UM",
    "COVERAGE_MAX_PROBES",
    "COVERAGE_NEAR_UM",
    "COVERAGE_STEP_UM",
    "JUNCTION_DIRECTION_REACH_UM",
    "clark_evans_2d",
    "column_vessels",
    "compute_junction_shape",
    "compute_tissue_coverage",
    "compute_vessel_columns",
    "coverage_probes",
    "junction_angle_triples",
    "junction_directions",
    "network_box",
    "tissue_distances",
    "vessel_sample_points",
]
