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
#: A run along an axis: capillaries joined end to end whose chords all lie
#: within AXIS_CONE_DEG of it; one is counted that spans this much of the box
#: along the axis (um) -- E14.5's columns from the plexus up, a muscle's
#: fibre-parallel capillaries, read off whatever template has them.
RUN_SPAN_UM = 100.0
#: The four classes a capillary's chord falls in: within AXIS_CONE_DEG of z,
#: y or x, or none of them (the cones do not overlap).
DIRECTION_CLASSES = ("z", "y", "x", "oblique")
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
    #: Per cell, the share of its capillary length whose vessels' chords run
    #: within AXIS_CONE_DEG of z, y and x (the rest oblique); a cell without
    #: capillaries holds the whole network's. What the bed's fill matches.
    chord_axis_share_by_cell: Optional[tuple[tuple[float, float, float], ...]] = None
    #: The same over the whole network.
    chord_axis_share: Optional[tuple[float, float, float]] = None
    #: Along z, y and x, the runs spanning RUN_SPAN_UM or more.
    long_runs: Optional[tuple[int, int, int]] = None
    #: The mean chord (um) of the capillaries in each of DIRECTION_CLASSES
    #: (None for a class with none): on E14.5 its vertical vessels reach past
    #: nearer junctions, 1.4 times the mean.
    chord_by_class_um: Optional[tuple[Optional[float], ...]] = None
    #: The trees' shape (Harvey, 2026-10-25, grid D1b): per branch-order
    #: label, how many of its vessels end the tree (a leaf: a short side
    #: branch off a trunk, handing over to capillaries or ending), as
    #: ``(label, count)`` pairs; and the median, over the leaves, of the
    #: straight distance from the root to the leaf over the path along the
    #: tree -- how well the trunks hold their course.
    tree_leaves_by_order: Optional[tuple[tuple[str, int], ...]] = None
    trunk_straightness: Optional[float] = None

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
            "chord_axis_share_by_cell": (
                None if self.chord_axis_share_by_cell is None else [list(s) for s in self.chord_axis_share_by_cell]
            ),
            "chord_axis_share": None if self.chord_axis_share is None else list(self.chord_axis_share),
            "long_runs": None if self.long_runs is None else list(self.long_runs),
            "chord_by_class_um": None if self.chord_by_class_um is None else list(self.chord_by_class_um),
            "tree_leaves_by_order": (
                None if self.tree_leaves_by_order is None else [list(item) for item in self.tree_leaves_by_order]
            ),
            "trunk_straightness": self.trunk_straightness,
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
            chord_axis_share_by_cell=(
                None
                if data.get("chord_axis_share_by_cell") is None
                else tuple(tuple(float(v) for v in s) for s in data["chord_axis_share_by_cell"])
            ),
            chord_axis_share=(
                None if data.get("chord_axis_share") is None else tuple(float(v) for v in data["chord_axis_share"])
            ),
            long_runs=None if data.get("long_runs") is None else tuple(int(v) for v in data["long_runs"]),
            chord_by_class_um=(
                None
                if data.get("chord_by_class_um") is None
                else tuple(None if v is None else float(v) for v in data["chord_by_class_um"])
            ),
            tree_leaves_by_order=(
                None
                if data.get("tree_leaves_by_order") is None
                else tuple((str(label), int(count)) for label, count in data["tree_leaves_by_order"])
            ),
            trunk_straightness=opt("trunk_straightness"),
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


def chord_class(chord: np.ndarray) -> int:
    """The index in :data:`DIRECTION_CLASSES` of a chord's direction."""
    norm = float(np.linalg.norm(chord))
    if norm <= 0:
        return 3
    along = np.abs(np.asarray(chord, float)) / norm
    axis = int(np.argmax(along))
    return axis if along[axis] >= math.cos(math.radians(AXIS_CONE_DEG)) else 3


def long_runs(chords: Sequence[tuple[Any, Any, np.ndarray, np.ndarray]]) -> tuple[int, int, int]:
    """Along z, y and x, how many runs span :data:`RUN_SPAN_UM` or more:
    *chords* ``(u, v, pos_u, pos_v)``, the capillaries joined end to end at
    shared nodes, those of one axis's class taken together."""
    counts = []
    for axis in range(3):
        H = nx.Graph()
        where: dict = {}
        for u, v, pu, pv in chords:
            if chord_class(np.asarray(pv) - np.asarray(pu)) == axis:
                H.add_edge(u, v)
                where[u], where[v] = float(pu[axis]), float(pv[axis])
        spans = [max(where[n] for n in part) - min(where[n] for n in part) for part in nx.connected_components(H)]
        counts.append(int(sum(span >= RUN_SPAN_UM for span in spans)))
    return tuple(counts)


def tree_shape(
    tree_edges: Sequence[tuple[Any, Any, str, float]],
    positions: Mapping[Any, np.ndarray],
    roots: Optional[Sequence[Any]] = None,
) -> tuple[tuple[tuple[str, int], ...], Optional[float]]:
    """The trees' leaves by branch-order label and their trunks' straightness
    (see :class:`Organisation`) from *tree_edges* ``(u, v, label, length)``.
    The roots are *roots* (the inlets and outlets) where given, else every
    tree end on a vessel of order 1."""
    tree = nx.Graph()
    for u, v, label, length in tree_edges:
        tree.add_edge(u, v, label=str(label), length=float(length))
    ends = [n for n in tree if tree.degree(n) == 1]
    if roots is not None:
        root_set = {r for r in roots if r in tree}
    else:
        root_set = {n for n in ends if _order_number(next(iter(tree[n].values()))["label"]) == 1}
    leaves: dict[str, int] = {}
    straightness: list[float] = []
    for component in nx.connected_components(tree):
        sub = tree.subgraph(component)
        component_roots = [r for r in component if r in root_set]
        distance: dict = {}
        paths: dict = {}
        if component_roots:
            distance, paths = nx.multi_source_dijkstra(sub, component_roots, weight="length")
        for n in component:
            if sub.degree(n) != 1 or n in root_set:
                continue
            label = next(iter(sub[n].values()))["label"]
            leaves[label] = leaves.get(label, 0) + 1
            if n in distance and distance[n] > 0:
                nearest = paths[n][0]
                chord = float(np.linalg.norm(np.asarray(positions[n], float) - np.asarray(positions[nearest], float)))
                straightness.append(chord / float(distance[n]))
    return tuple(sorted(leaves.items())), (float(np.median(straightness)) if straightness else None)


def measure_organisation(
    G: nx.Graph, extent_um: Sequence[float], roots: Optional[Sequence[Any]] = None
) -> Optional[Organisation]:
    """The layout of *G* (branch orders labelled) in a box of *extent_um*; None
    for a network without capillaries. *roots*: its inlets and outlets, where
    the trees start (else read off the orders)."""
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
    tree_edges: list[tuple[Any, Any, str, float]] = []
    axis_lengths = np.zeros(3)
    chord_classes = np.zeros((cells, 4))
    chord_sums, chord_counts = np.zeros(4), np.zeros(4)
    chords: list[tuple[Any, Any, np.ndarray, np.ndarray]] = []
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
            ends = (np.asarray(G.nodes[u]["pos"], float), np.asarray(G.nodes[v]["pos"], float))
            chords.append((u, v, *ends))
            middle = cell_index((ends[0] + ends[1]) / 2.0, extent, shape)[0]
            kind = chord_class(ends[1] - ends[0])
            chord_classes[middle, kind] += float(lengths.sum())
            chord_sums[kind] += float(np.linalg.norm(ends[1] - ends[0]))
            chord_counts[kind] += 1
        elif category in _ARTERIAL or category in _VENOUS:
            np.add.at(arterial_length if category in _ARTERIAL else venous_length, where, lengths)
            tree_paths.append(path)
            tree_edges.append((u, v, str(data.get("branch_order")), float(lengths.sum())))
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
    trees = tree_shape(tree_edges, {n: G.nodes[n]["pos"] for n in G}, roots) if tree_edges else ((), None)
    whole = chord_classes.sum(axis=0) / max(float(chord_classes.sum()), 1e-12)
    by_cell = []
    for c in range(cells):
        total = float(chord_classes[c].sum())
        shares = chord_classes[c] / total if total > 0 else whole
        by_cell.append(tuple(float(v) for v in shares[:3]))
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
        chord_axis_share_by_cell=tuple(by_cell),
        chord_axis_share=tuple(float(v) for v in whole[:3]),
        long_runs=long_runs(chords),
        chord_by_class_um=tuple(
            float(chord_sums[k] / chord_counts[k]) if chord_counts[k] else None for k in range(4)
        ),
        tree_leaves_by_order=trees[0] if tree_edges else None,
        trunk_straightness=trees[1] if tree_edges else None,
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
    "DIRECTION_CLASSES",
    "FAR_UM",
    "NEAR_UM",
    "Organisation",
    "RUN_SPAN_UM",
    "TARGET_CELL_UM",
    "TENSOR_COMPONENTS",
    "cell_centres",
    "cell_index",
    "chord_class",
    "correlation",
    "grid_shape",
    "long_runs",
    "measure_organisation",
    "tree_shape",
    "profile",
    "share_inside",
    "tensor_matrix",
]
