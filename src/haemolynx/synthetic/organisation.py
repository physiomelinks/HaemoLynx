"""How a network is laid out in its box (``PLAN_synthetic_physiology.md``).

The counts, lengths and widths a template records say nothing about where the
vessels are. A tissue has its own layout -- E14.5's large vessels lie flat in a
plexus on one face, its capillaries are densest there and run up through the
depth from it -- and a synthetic network drawn to the counts alone fills its box
evenly in every direction, which is not that tissue (Harvey, 2026-10-25: D4,
the organisation read from the template).

:func:`measure_organisation` puts a coarse grid over the box (cells of about
:data:`TARGET_CELL_UM`, the same share of the extent on every axis, so a scaled
box keeps its pattern) and records, cell by cell: the capillary vertices and
the capillary length, the arterial and the venous trees' length, and the
direction the capillaries run (their length-weighted orientation tensor); and
for the whole network: how much of the box lies near a capillary, the share of
capillary length along each axis, how far a tree vessel turns from its parent
where it carries the trunk on, and how much vessels curve. The generator draws
its bed and grows its trees to these fields; the comparison scores a synthetic
network's own against the template's.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

import networkx as nx
import numpy as np

from haemolynx.graph.thick_vessel_junctions import IS_ZERO_RESISTANCE

#: A cell's side, roughly: a few capillary spacings, so a cell's counts are
#: not noise, and fine enough to see a plexus 25 um deep in a 300 um box.
TARGET_CELL_UM = 50.0
MAX_CELLS_PER_AXIS = 12
#: The grid coverage is read on, and its two distances (um).
COVERAGE_STEP_UM = 10.0
NEAR_UM = 20.0
FAR_UM = 50.0
#: "Along an axis": within this many degrees of it (13.4% of an isotropic
#: network's length is within 30 degrees of any one axis).
AXIS_CONE_DEG = 30.0
#: The stretch of a vessel its heading is read over at a junction, and the
#: step its curvature is read at (um).
HEADING_UM = 10.0
CURVATURE_STEP_UM = 10.0

#: The tensor's six independent components, in this order.
TENSOR_COMPONENTS = ("zz", "yy", "xx", "zy", "zx", "yx")
ISOTROPIC = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 0.0, 0.0, 0.0)

_ARTERIAL = ("large_arteriole", "arteriole")
_VENOUS = ("large_venule", "venule")


@dataclass(frozen=True)
class Organisation:
    """A network's layout: per-cell fields (each flattened in (z, y, x)
    C order over :attr:`shape`) and whole-network measures. See the module."""

    shape: tuple[int, int, int]
    capillary_nodes: tuple[float, ...]
    capillary_length_um: tuple[float, ...]
    arterial_length_um: tuple[float, ...]
    venous_length_um: tuple[float, ...]
    #: Per cell, :data:`TENSOR_COMPONENTS` of the capillaries' orientation
    #: tensor (trace 1); a cell without capillaries holds the global one.
    orientation: tuple[tuple[float, ...], ...]
    global_orientation: tuple[float, ...]
    #: Share of capillary length within :data:`AXIS_CONE_DEG` of z, y, x.
    axis_share: tuple[float, float, float]
    #: Share of the box within :data:`NEAR_UM` / beyond :data:`FAR_UM` of a capillary.
    within_near: float
    beyond_far: float
    #: Median turn (degrees) from a tree vessel's heading into a junction to
    #: the heading of the child that turns least: how straight a trunk runs.
    trunk_turn_deg: Optional[float]
    #: Mean turn (degrees) per :data:`CURVATURE_STEP_UM` of path.
    capillary_curvature_deg: Optional[float]
    tree_curvature_deg: Optional[float]
    #: Each capillary vertex's position as a share of the box's extent (in
    #: node order), and their median distance to the nearest other one (um):
    #: the finer layout a bed drawn at the template's own size is laid by
    #: (``bed.template_vertices``). None for a template written before them.
    capillary_node_positions_rel: Optional[tuple[tuple[float, float, float], ...]] = None
    capillary_node_spacing_um: Optional[float] = None

    # -- reading -----------------------------------------------------------

    def field(self, name: str) -> np.ndarray:
        """One per-cell field as a (z, y, x) array."""
        return np.asarray(getattr(self, name), dtype=float).reshape(self.shape)

    def tensor(self, cell: int) -> np.ndarray:
        """Cell *cell*'s orientation tensor as a 3 x 3 matrix."""
        return tensor_matrix(self.orientation[cell])

    def cell_of(self, points: np.ndarray, extent_um: Sequence[float]) -> np.ndarray:
        """The flat cell index of each of *points* ((n, 3) um) in a box of *extent_um*."""
        return cell_index(points, extent_um, self.shape)

    def to_dict(self) -> dict:
        return {
            "shape": list(self.shape),
            "capillary_nodes": list(self.capillary_nodes),
            "capillary_length_um": list(self.capillary_length_um),
            "arterial_length_um": list(self.arterial_length_um),
            "venous_length_um": list(self.venous_length_um),
            "orientation": [list(t) for t in self.orientation],
            "global_orientation": list(self.global_orientation),
            "axis_share": list(self.axis_share),
            "within_near": self.within_near,
            "beyond_far": self.beyond_far,
            "trunk_turn_deg": self.trunk_turn_deg,
            "capillary_curvature_deg": self.capillary_curvature_deg,
            "tree_curvature_deg": self.tree_curvature_deg,
            "capillary_node_positions_rel": (
                None if self.capillary_node_positions_rel is None else [list(p) for p in self.capillary_node_positions_rel]
            ),
            "capillary_node_spacing_um": self.capillary_node_spacing_um,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Organisation":
        def opt(name):
            value = data.get(name)
            return None if value is None else float(value)

        return cls(
            shape=tuple(int(v) for v in data["shape"]),
            capillary_nodes=tuple(float(v) for v in data["capillary_nodes"]),
            capillary_length_um=tuple(float(v) for v in data["capillary_length_um"]),
            arterial_length_um=tuple(float(v) for v in data["arterial_length_um"]),
            venous_length_um=tuple(float(v) for v in data["venous_length_um"]),
            orientation=tuple(tuple(float(v) for v in t) for t in data["orientation"]),
            global_orientation=tuple(float(v) for v in data["global_orientation"]),
            axis_share=tuple(float(v) for v in data["axis_share"]),
            within_near=float(data["within_near"]),
            beyond_far=float(data["beyond_far"]),
            trunk_turn_deg=opt("trunk_turn_deg"),
            capillary_curvature_deg=opt("capillary_curvature_deg"),
            tree_curvature_deg=opt("tree_curvature_deg"),
            capillary_node_positions_rel=(
                None
                if data.get("capillary_node_positions_rel") is None
                else tuple(tuple(float(v) for v in p) for p in data["capillary_node_positions_rel"])
            ),
            capillary_node_spacing_um=opt("capillary_node_spacing_um"),
        )


def grid_shape(extent_um: Sequence[float]) -> tuple[int, int, int]:
    """Cells per axis: about :data:`TARGET_CELL_UM` each, 1 to :data:`MAX_CELLS_PER_AXIS`."""
    return tuple(int(np.clip(round(float(e) / TARGET_CELL_UM), 1, MAX_CELLS_PER_AXIS)) for e in extent_um)


def cell_index(points: np.ndarray, extent_um: Sequence[float], shape: Sequence[int]) -> np.ndarray:
    """The flat (z, y, x) C-order cell of each point; points on or past a face
    fall in the cell at it."""
    extent = np.maximum(np.asarray(extent_um, dtype=float), 1e-9)
    dims = np.asarray(shape, dtype=int)
    rel = np.asarray(points, dtype=float).reshape(-1, 3) / extent
    idx = np.clip(np.floor(rel * dims).astype(int), 0, dims - 1)
    return np.ravel_multi_index((idx[:, 0], idx[:, 1], idx[:, 2]), tuple(dims))


def cell_centres(extent_um: Sequence[float], shape: Sequence[int]) -> np.ndarray:
    """The centre of every cell, flat C order, (n, 3) um."""
    extent = np.asarray(extent_um, dtype=float)
    axes = [(np.arange(n) + 0.5) * e / n for n, e in zip(shape, extent)]
    grid = np.meshgrid(*axes, indexing="ij")
    return np.stack([g.ravel() for g in grid], axis=1)


def tensor_matrix(components: Sequence[float]) -> np.ndarray:
    zz, yy, xx, zy, zx, yx = (float(v) for v in components)
    return np.array([[zz, zy, zx], [zy, yy, yx], [zx, yx, xx]])


def _components(matrix: np.ndarray) -> tuple[float, ...]:
    return (
        float(matrix[0, 0]), float(matrix[1, 1]), float(matrix[2, 2]),
        float(matrix[0, 1]), float(matrix[0, 2]), float(matrix[1, 2]),
    )


def _category(label: Any) -> Optional[str]:
    from haemolynx.statistics.bifurcation import _normalize_branch_order_tag, branch_order_category

    tag = _normalize_branch_order_tag(label)
    return None if tag is None else branch_order_category(tag)


def _polyline(G: nx.Graph, u: Any, v: Any, data: Mapping[str, Any]) -> np.ndarray:
    """The edge's centreline running from *u* to *v* (a stored polyline may
    run either way)."""
    voxels = data.get("voxels")
    pu = np.asarray(G.nodes[u]["pos"], dtype=float)
    if voxels is None or len(voxels) < 2:
        return np.asarray([pu, G.nodes[v]["pos"]], dtype=float)
    path = np.asarray(voxels, dtype=float)
    if np.linalg.norm(path[-1] - pu) < np.linalg.norm(path[0] - pu):
        path = path[::-1]
    return path


def _heading(path: np.ndarray, from_start: bool) -> Optional[np.ndarray]:
    """The unit direction leaving *path*'s first (or last) point into the
    vessel, over :data:`HEADING_UM`."""
    pts = path if from_start else path[::-1]
    steps = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    if not len(steps) or steps.sum() <= 0:
        return None
    reach = np.searchsorted(np.cumsum(steps), HEADING_UM)
    end = pts[min(int(reach) + 1, len(pts) - 1)]
    vector = end - pts[0]
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 0 else None


def _turn_deg(a: np.ndarray, b: np.ndarray) -> float:
    return math.degrees(math.acos(float(np.clip(np.dot(a, b), -1.0, 1.0))))


def _curvature(paths: Sequence[np.ndarray]) -> Optional[float]:
    """Mean turn (degrees) between successive :data:`CURVATURE_STEP_UM` chords."""
    turns: list[float] = []
    for path in paths:
        steps = np.linalg.norm(np.diff(path, axis=0), axis=1)
        total = float(steps.sum())
        if total < 3 * CURVATURE_STEP_UM:
            continue
        cumulative = np.concatenate([[0.0], np.cumsum(steps)])
        stations = np.arange(0.0, total, CURVATURE_STEP_UM)
        points = np.stack([np.interp(stations, cumulative, path[:, i]) for i in range(3)], axis=1)
        chords = np.diff(points, axis=0)
        norms = np.linalg.norm(chords, axis=1)
        ok = norms > 0
        units = chords[ok] / norms[ok, None]
        for a, b in zip(units[:-1], units[1:]):
            turns.append(_turn_deg(a, b))
    return float(np.mean(turns)) if turns else None


def measure_organisation(G: nx.Graph, extent_um: Sequence[float]) -> Optional[Organisation]:
    """The layout of *G* (branch orders labelled) in a box of *extent_um*; None
    for a network without capillaries."""
    from scipy.spatial import cKDTree

    extent = np.asarray(extent_um, dtype=float)
    shape = grid_shape(extent)
    cells = int(np.prod(shape))
    capillary_length = np.zeros(cells)
    arterial_length = np.zeros(cells)
    venous_length = np.zeros(cells)
    tensors = np.zeros((cells, 3, 3))
    capillary_points: list[np.ndarray] = []
    capillary_paths: list[np.ndarray] = []
    tree_paths: list[np.ndarray] = []
    capillary_nodes: set = set()
    tree_ends: dict[Any, list[tuple[int, np.ndarray, np.ndarray, str]]] = {}
    axis_lengths = np.zeros(3)
    total_capillary = 0.0
    cos_cone = math.cos(math.radians(AXIS_CONE_DEG))
    for u, v, data in G.edges(data=True):
        if data.get(IS_ZERO_RESISTANCE):
            continue
        category = _category(data.get("branch_order"))
        if category is None:
            continue
        path = _polyline(G, u, v, data)
        steps = np.diff(path, axis=0)
        lengths = np.linalg.norm(steps, axis=1)
        keep = lengths > 0
        if not keep.any():
            continue
        steps, lengths = steps[keep], lengths[keep]
        mids = ((path[1:] + path[:-1]) / 2)[keep]
        where = cell_index(mids, extent, shape)
        if category == "capillary":
            capillary_nodes.update((u, v))
            np.add.at(capillary_length, where, lengths)
            units = steps / lengths[:, None]
            outer = units[:, :, None] * units[:, None, :] * lengths[:, None, None]
            np.add.at(tensors, where, outer)
            axis_lengths += (np.abs(units) >= cos_cone).T @ lengths
            total_capillary += float(lengths.sum())
            capillary_points.append(mids)
            capillary_paths.append(path)
        elif category in _ARTERIAL or category in _VENOUS:
            np.add.at(arterial_length if category in _ARTERIAL else venous_length, where, lengths)
            tree_paths.append(path)
            order = _order_number(data.get("branch_order"))
            # At each end, the direction leaving that node into the vessel.
            start, end = _heading(path, True), _heading(path, False)
            if start is not None:
                tree_ends.setdefault(u, []).append((order, start, category))
            if end is not None:
                tree_ends.setdefault(v, []).append((order, end, category))
    if total_capillary <= 0:
        return None
    node_count = np.zeros(cells)
    positions = [G.nodes[n]["pos"] for n in sorted(capillary_nodes, key=str) if G.nodes[n].get("pos") is not None]
    spacing = None
    if positions:
        np.add.at(node_count, cell_index(np.asarray(positions, float), extent, shape), 1.0)
    if len(positions) >= 2:
        nearest, _ = cKDTree(np.asarray(positions, float)).query(np.asarray(positions, float), k=2)
        spacing = float(np.median(nearest[:, 1]))
    total_tensor = tensors.sum(axis=0) / total_capillary
    global_components = _components(total_tensor)
    orientation = []
    for c in range(cells):
        if capillary_length[c] > 0:
            orientation.append(_components(tensors[c] / capillary_length[c]))
        else:
            orientation.append(global_components)
    grid_axes = [np.arange(COVERAGE_STEP_UM / 2, e, COVERAGE_STEP_UM) for e in extent]
    grid = np.stack(np.meshgrid(*grid_axes, indexing="ij"), axis=-1).reshape(-1, 3)
    distance, _ = cKDTree(np.concatenate(capillary_points)).query(grid)
    return Organisation(
        shape=shape,
        capillary_nodes=tuple(float(v) for v in node_count),
        capillary_length_um=tuple(float(v) for v in capillary_length),
        arterial_length_um=tuple(float(v) for v in arterial_length),
        venous_length_um=tuple(float(v) for v in venous_length),
        orientation=tuple(orientation),
        global_orientation=global_components,
        axis_share=tuple(float(v) for v in axis_lengths / total_capillary),
        within_near=float((distance <= NEAR_UM).mean()),
        beyond_far=float((distance > FAR_UM).mean()),
        trunk_turn_deg=_trunk_turn(tree_ends),
        capillary_curvature_deg=_curvature(capillary_paths),
        tree_curvature_deg=_curvature(tree_paths),
        capillary_node_positions_rel=tuple(
            tuple(float(v) for v in np.asarray(p, float) / np.maximum(extent, 1e-9)) for p in positions
        ),
        capillary_node_spacing_um=spacing,
    )


def _order_number(label: Any) -> int:
    digits = "".join(ch for ch in str(label or "") if ch.isdigit())
    return int(digits) if digits else 0


def _trunk_turn(ends: Mapping[Any, list]) -> Optional[float]:
    """At each tree junction, the parent (lowest order) vessel's heading into
    it against the child that turns least from it."""
    turns: list[float] = []
    for node, items in ends.items():
        if len(items) < 2:
            continue
        items = sorted(items, key=lambda item: item[0])
        parent_order, parent_leaving, parent_category = items[0]
        heading_in = -parent_leaving
        children = [i for i in items[1:] if i[0] > parent_order and i[2] == parent_category]
        if not children:
            continue
        turns.append(min(_turn_deg(heading_in, child[1]) for child in children))
    return float(np.median(turns)) if turns else None


def correlation(a: Sequence[float], b: Sequence[float]) -> Optional[float]:
    """Pearson correlation of two per-cell fields; None when either is flat."""
    x, y = np.asarray(a, float), np.asarray(b, float)
    if len(x) != len(y) or len(x) < 2 or x.std() == 0 or y.std() == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def profile(field: np.ndarray, axis: int) -> np.ndarray:
    """A (z, y, x) field summed over every axis but *axis*."""
    others = tuple(i for i in range(3) if i != axis)
    return field.sum(axis=others)


def share_inside(field: np.ndarray, mask: np.ndarray) -> Optional[float]:
    """Share of *field*'s total in the cells *mask* marks."""
    total = float(field.sum())
    return None if total <= 0 else float(field[mask].sum() / total)


__all__ = [
    "AXIS_CONE_DEG",
    "FAR_UM",
    "NEAR_UM",
    "Organisation",
    "TARGET_CELL_UM",
    "TENSOR_COMPONENTS",
    "cell_centres",
    "cell_index",
    "correlation",
    "grid_shape",
    "measure_organisation",
    "profile",
    "share_inside",
    "tensor_matrix",
]
