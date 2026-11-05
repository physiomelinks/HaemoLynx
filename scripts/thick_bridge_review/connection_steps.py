#!/usr/bin/env python3
"""Which step of graph building made a connection Harvey marked "should not
be connected at all" (N) on the Phase 5 grid of PLAN_thick_vessel_bridges.md.

A case's *course* is the stretch of centreline that makes the connection: a
near-miss's own piece, or, for a bridge, the vessel's way from where it
leaves the thick region to the thick centreline in the graph the split was
handed (the split draws its bridge straight, so the bridge itself is on no
earlier graph). The course is looked for in the graph after every step: it
is there when most of it lies within reach of the graph's centrelines and a
path along them joins its two ends not much longer than it. The step that
made it is the first of the last unbroken run of steps that have it. Where
the first graph (the skeleton's own, ``build_graph_segment_skan_stitched_loops``)
already has it, the skeletons say which step drew it, by which of them join
its two ends: the thick tree, Lee on the thin catchment, the join of thin
arms to the fat ridge (``preprocessing.thick_vessels._join_thin_arms_to_fat_ridge``),
or the skeleton clean-up after it (its gap filling, closing and bundle
refinement, thinned again). A bridge on no course the split was handed was
drawn by the split itself; the marks it leaves on the edge say how.

    python scripts/thick_bridge_review/connection_steps.py --run <run> \\
        --grid outputs/thick_bridge_review --out <folder> [--mark N | --all-bridges]

``--run`` is a folder ``capture_steps.py`` captured from one run:
``steps/NN_<label>.pkl``, the graph after each topology step and each step
of ``stages.finish_network_graph`` in order;
``split_call.pkl`` (``{"G": the graph handed to the split}``);
``graph_built.pkl``; ``skeleton.npz``, ``skel_before_join.npz``,
``skel_after_join.npz`` and ``thick.npz`` (volumes as packed bits with their
shape); ``tree.npy`` (the tree's voxels) and ``voxel_size_zyx.json``.
``--grid`` is ``build_grid.py``'s folder (``cases.json``, ``marks.csv``).
Writes ``trace.csv`` and ``TRACE.md`` into ``--out``.
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import networkx as nx
import numpy as np
from scipy.spatial import cKDTree

#: Course and graph centrelines are compared as points this far apart (um).
DENSIFY_UM = 0.5
#: A course point is covered when a graph centreline passes within this (um).
REACH_UM = 1.5
#: ... and the course is there when this share of its points are covered,
MIN_COVER = 0.8
#: and a path along the graph joins its ends no longer than this many times
#: the course, plus PATH_EXTRA_UM.
PATH_SLACK = 1.5
PATH_EXTRA_UM = 4.0
#: The neighbourhood of a case that is read (um, half the side of a cube).
HALF_WINDOW_UM = 40.0
#: A bridge's course before the split is at most this many thick radii
#: plus MAX_COURSE_EXTRA_UM long: a longer way to the centreline is another
#: vessel's, and the bridge was drawn by the split.
MAX_COURSE_RADII = 3.0
MAX_COURSE_EXTRA_UM = 10.0
#: The graph the split was handed passes this near (um) a bridge's ends
#: where the bridge replaced a vessel's way in.
BRIDGE_END_REACH_UM = 2.0

#: Edge attributes that say which step drew an edge.
PROVENANCE_KEYS = (
    "bridge_kind",
    "bridge_kinds",
    "recovered",
    "reconnected",
    "joined_at_thick_wall_um",
    "thick_centreline_from_tree",
    "thick_centreline_extended_um",
    "thick_bridge_drawn",
    "thick_fork_bridge",
    "thick_taper",
    "retraced",
    "retraced_join",
)

#: The skeletons a course the first graph has is looked for in, last made
#: first: the one graph building was handed (after the clean-up's gap filling,
#: closing and bundle refinement, thinned again), the thickness-gated one
#: after the thin-arm join, before it (Lee on the thin catchment and the
#: tree), and Lee's alone.
SKELETONS = ("final", "after_join", "before_join", "lee")
#: The clean-up thins the skeleton again, moving it two or three voxels where
#: its gap filling and closing merged strands: a course's ends are looked for
#: in the skeletons this far from them (um), and the way between them may be
#: this much longer than the course (um) besides PATH_SLACK.
SKELETON_REACH_UM = 3.0
SKELETON_PATH_EXTRA_UM = 6.0
#: A course shorter than this (um) is judged with its own vessel carried on
#: either side to this length (:func:`extend_course`).
MIN_COURSE_UM = 8.0


def densify(points: Sequence[Sequence[float]], step_um: float = DENSIFY_UM) -> np.ndarray:
    """*points* (a polyline) with points added so no two in a row are more
    than *step_um* apart."""
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    if len(points) < 2:
        return points
    out = [points[:1]]
    for a, b in zip(points[:-1], points[1:]):
        count = max(1, int(np.ceil(np.linalg.norm(b - a) / step_um)))
        t = np.linspace(0.0, 1.0, count + 1)[1:, None]
        out.append(a + t * (b - a))
    return np.vstack(out)


def polyline_length(points: np.ndarray) -> float:
    points = np.asarray(points, dtype=float)
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum()) if len(points) > 1 else 0.0


def _edge_points(G: nx.MultiGraph, u, v, data: Mapping[str, Any]) -> np.ndarray | None:
    from haemolynx.visualization.geometry import edge_polyline

    try:
        return densify(edge_polyline(G, u, v, data))
    except ValueError:
        return None


class StepIndex:
    """One graph's centrelines as points, every edge densified, for the
    neighbourhood of any case to be read from."""

    def __init__(self, G: nx.MultiGraph):
        self.G = G
        self.edges: list[tuple] = []
        self.polylines: list[np.ndarray] = []
        owners = []
        for u, v, k, data in G.edges(keys=True, data=True):
            points = _edge_points(G, u, v, data)
            if points is None or not len(points):
                continue
            owners.append(np.full(len(points), len(self.edges)))
            self.edges.append((u, v, k))
            self.polylines.append(points)
        self.points = np.vstack(self.polylines) if self.polylines else np.zeros((0, 3))
        self.owner = np.concatenate(owners) if owners else np.zeros(0, dtype=int)
        self.tree = cKDTree(self.points) if len(self.points) else None

    def local(self, centre: Sequence[float], half_window_um: float = HALF_WINDOW_UM) -> "LocalGraph":
        """The edges with a point within *half_window_um* of *centre* (a cube)."""
        if self.tree is None:
            return LocalGraph(self, [])
        near = self.tree.query_ball_point(np.asarray(centre, float), half_window_um * np.sqrt(3.0))
        centre = np.asarray(centre, float)
        inside = [i for i in near if np.all(np.abs(self.points[i] - centre) <= half_window_um)]
        return LocalGraph(self, sorted({int(self.owner[i]) for i in inside}))


class PointGraph:
    """Points joined into a graph (*graph*, over *labels*): what covers a
    course, and how far along the graph its two ends are apart."""

    def __init__(self, points: np.ndarray, labels: list, graph: nx.Graph):
        self.points = np.asarray(points, dtype=float).reshape(-1, 3)
        self.labels = labels
        self.graph = graph
        self.tree = cKDTree(self.points) if len(self.points) else None

    def covered(self, course: np.ndarray, reach_um: float = REACH_UM) -> np.ndarray:
        """Which of *course*'s points have a centreline within *reach_um*."""
        if self.tree is None:
            return np.zeros(len(course), dtype=bool)
        distance, _ = self.tree.query(np.asarray(course, float), distance_upper_bound=reach_um)
        return distance <= reach_um

    def path_um(self, a: Sequence[float], b: Sequence[float], *, reach_um: float = REACH_UM,
                cutoff_um: float = np.inf) -> float:
        """The shortest way along the centrelines from a point within
        *reach_um* of *a* to one within *reach_um* of *b*, or inf."""
        if self.tree is None:
            return np.inf
        sources = [self.labels[i] for i in self.tree.query_ball_point(np.asarray(a, float), reach_um)]
        targets = {self.labels[i] for i in self.tree.query_ball_point(np.asarray(b, float), reach_um)}
        if not sources or not targets:
            return np.inf
        if targets.intersection(sources):
            return 0.0
        lengths = nx.multi_source_dijkstra_path_length(self.graph, set(sources), cutoff=cutoff_um)
        return float(min((lengths[t] for t in targets if t in lengths), default=np.inf))

    def path_points(self, a: Sequence[float], b: Sequence[float], *, reach_um: float = REACH_UM,
                    cutoff_um: float = np.inf) -> np.ndarray | None:
        """The points of the shortest way along the graph from within
        *reach_um* of *a* to within *reach_um* of *b*, no longer than
        *cutoff_um*; None when there is none."""
        if self.tree is None:
            return None
        sources = {self.labels[i] for i in self.tree.query_ball_point(np.asarray(a, float), reach_um)}
        targets = {self.labels[i] for i in self.tree.query_ball_point(np.asarray(b, float), reach_um)}
        if not sources or not targets:
            return None
        graph = self.graph.copy()
        end = object()
        graph.add_edges_from((t, end) for t in targets)
        for t in targets:
            graph.edges[t, end]["weight"] = 0.0
        try:
            length, path = nx.multi_source_dijkstra(graph, sources, target=end, cutoff=cutoff_um)
        except nx.NetworkXNoPath:
            return None
        where = {label: i for i, label in enumerate(self.labels)}
        points = [self.points[where[label]] for label in path if label in where]
        if len(points) < 2:
            # *a* and *b* are within reach of one point: the way is between them.
            return densify([a, b])
        return np.asarray(points, float)


class LocalGraph(PointGraph):
    """The edges round a case as a graph of their points."""

    def __init__(self, index: StepIndex, edge_ids: Iterable[int]):
        self.index = index
        self.edge_ids = list(edge_ids)
        graph = nx.Graph()
        points, labels = [], []
        G = index.G
        for e in self.edge_ids:
            u, v, _k = index.edges[e]
            line = index.polylines[e]
            for i in range(len(line)):
                labels.append(("p", e, i))
                points.append(line[i])
                if i:
                    graph.add_edge(("p", e, i - 1), ("p", e, i), weight=float(np.linalg.norm(line[i] - line[i - 1])))
            for end, i in ((u, 0), (v, len(line) - 1)):
                pos = G.nodes[end].get("pos")
                weight = 0.0 if pos is None else float(np.linalg.norm(np.asarray(pos, float) - line[i]))
                graph.add_edge(("n", end), ("p", e, i), weight=weight)
        super().__init__(np.asarray(points, dtype=float), labels, graph)

    def covered_by(self, course: np.ndarray, keep, reach_um: float = REACH_UM) -> np.ndarray:
        """Which of *course*'s points have, within *reach_um*, a point of an
        edge whose data *keep* accepts."""
        G = self.index.G
        lines = [self.index.polylines[e] for e in self.edge_ids if keep(G.edges[self.index.edges[e]])]
        if not lines:
            return np.zeros(len(course), dtype=bool)
        distance, _ = cKDTree(np.vstack(lines)).query(np.asarray(course, float), distance_upper_bound=reach_um)
        return distance <= reach_um

    def edges_covering(self, course: np.ndarray, reach_um: float = REACH_UM) -> list[tuple]:
        """The graph edges with a point within *reach_um* of *course*."""
        if self.tree is None:
            return []
        hits = set()
        for near in self.tree.query_ball_point(np.asarray(course, float), reach_um):
            for i in near:
                hits.add(self.labels[i][1])
        return [self.index.edges[e] for e in sorted(hits)]


def skeleton_graph(voxels: np.ndarray, voxel_size_zyx: Sequence[float]) -> PointGraph:
    """Skeleton voxels (``(n, 3)`` indices) as a :class:`PointGraph` in um,
    each joined to its 26 neighbours."""
    voxels = np.asarray(voxels, dtype=np.int64).reshape(-1, 3)
    spacing = np.asarray(voxel_size_zyx, float)
    at = {tuple(v): i for i, v in enumerate(voxels.tolist())}
    graph = nx.Graph()
    graph.add_nodes_from(range(len(voxels)))
    offsets = [np.array(o) for o in np.ndindex(3, 3, 3) if o != (1, 1, 1)]
    for i, v in enumerate(voxels):
        for o in offsets:
            j = at.get(tuple(v + o - 1))
            if j is not None and j > i:
                graph.add_edge(i, j, weight=float(np.linalg.norm((o - 1) * spacing)))
    return PointGraph(voxels * spacing, list(range(len(voxels))), graph)


def extend_course(local: LocalGraph, course: np.ndarray, extra_um: float) -> np.ndarray:
    """*course* carried on by *extra_um* at each end along the edge that
    covers most of it: a short course is judged with its vessel round it,
    or any centreline passing by would have it."""
    course = densify(course)
    best, best_cover = None, 0.0
    for e in local.edge_ids:
        line = local.index.polylines[e]
        cover = float((cKDTree(line).query(course, distance_upper_bound=REACH_UM)[0] <= REACH_UM).mean())
        if cover > best_cover:
            best, best_cover = line, cover
    if best is None:
        return course
    ends = sorted(int(np.argmin(np.linalg.norm(best - p, axis=1))) for p in (course[0], course[-1]))
    step = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(best, axis=0), axis=1))]
    first = int(np.searchsorted(step, step[ends[0]] - extra_um))
    last = int(np.searchsorted(step, step[ends[1]] + extra_um, side="right")) - 1
    return best[first:last + 1] if last > first else course


def course_present(local: PointGraph, course: np.ndarray, *, reach_um: float = REACH_UM,
                   min_cover: float = MIN_COVER, slack: float = PATH_SLACK,
                   extra_um: float = PATH_EXTRA_UM) -> bool:
    """Whether the graph has *course*: most of it covered, and its ends
    joined along the centrelines by a path not much longer than it."""
    course = densify(course)
    if not len(course) or local.covered(course, reach_um).mean() < min_cover:
        return False
    limit = slack * polyline_length(course) + extra_um
    return local.path_um(course[0], course[-1], reach_um=reach_um, cutoff_um=limit) <= limit


def made_by(labels: Sequence[str], present: Sequence[bool]) -> str | None:
    """The first step of the last unbroken run of steps that have the
    course, ending at the last step; None when the last step lacks it."""
    if not present or not present[-1]:
        return None
    first = len(present) - 1
    while first > 0 and present[first - 1]:
        first -= 1
    return labels[first]


def bridge_course(local: LocalGraph, outer: Sequence[float], inner: Sequence[float], *,
                  reach_um: float = BRIDGE_END_REACH_UM, max_length_um: float = np.inf) -> np.ndarray | None:
    """A bridge's course before the split: the shortest way along the graph
    the split was handed from the bridge's *outer* end (where it leaves the
    thick region) to its *inner* end (on the thick centreline, the tree's
    or the graph's own where it strays), no longer than *max_length_um*.
    None when there is none: the split drew the bridge where no vessel ran."""
    return local.path_points(outer, inner, reach_um=reach_um, cutoff_um=max_length_um)


def skeletons_near(skeletons: Mapping[str, np.ndarray], voxel_size_zyx: Sequence[float],
                   centre_um: Sequence[float], half_window_um: float = HALF_WINDOW_UM
                   ) -> dict[str, PointGraph]:
    """Each skeleton's voxels (``(n, 3)`` indices) within the window round
    *centre_um*, as a :func:`skeleton_graph`."""
    spacing = np.asarray(voxel_size_zyx, float)
    centre = np.asarray(centre_um, float)
    out = {}
    for name, voxels in skeletons.items():
        voxels = np.asarray(voxels).reshape(-1, 3)
        inside = np.all(np.abs(voxels * spacing - centre) <= half_window_um, axis=1)
        out[name] = skeleton_graph(voxels[inside], spacing)
    return out


def ends_joined(skeleton: PointGraph, course: np.ndarray, *, reach_um: float = SKELETON_REACH_UM,
                slack: float = PATH_SLACK, extra_um: float = SKELETON_PATH_EXTRA_UM) -> bool:
    """Whether *skeleton* joins *course*'s two ends (each within
    *reach_um*) by a path no longer than *slack* times the course plus
    *extra_um*, wherever it runs between them: the clean-up thins the
    skeleton again, so a way it only moved is the same way."""
    course = densify(course)
    limit = slack * polyline_length(course) + extra_um
    return skeleton.path_um(course[0], course[-1], reach_um=reach_um, cutoff_um=limit) <= limit


def skeleton_maker(course: np.ndarray, skeletons: Mapping[str, PointGraph],
                   reach_um: float = SKELETON_REACH_UM) -> tuple[str, str]:
    """Which skeleton step drew a course the first graph has, judged by
    which skeletons (:data:`SKELETONS`, one after another) join its ends
    (:func:`ends_joined`): ``"cleanup"`` when only the skeleton after the
    clean-up does, ``"join"`` when the join's does and Lee's and the tree's
    do not, ``"lee"`` when Lee's thin skeleton alone does, else ``"tree"``;
    and which of them do, as ``name=0/1`` pairs."""
    have = {name: ends_joined(skeletons[name], course, reach_um=reach_um) for name in SKELETONS}
    flags = " ".join(f"{name}={int(have[name])}" for name in SKELETONS)
    if not have["after_join"]:
        maker = "cleanup" if have["final"] else "none"
    elif not have["before_join"]:
        maker = "join"
    elif have["lee"]:
        maker = "lee"
    else:
        maker = "tree"
    return maker, flags


def edge_marks(G: nx.MultiGraph, edges: Iterable[tuple]) -> dict[str, Any]:
    """The provenance marks (:data:`PROVENANCE_KEYS`) on *edges*, merged."""
    marks: dict[str, Any] = {}
    for u, v, k in edges:
        data = G.edges[u, v, k]
        for key in PROVENANCE_KEYS:
            value = data.get(key)
            if value not in (None, False, "", 0):
                marks.setdefault(key, value)
    return marks


# --- reading a run -----------------------------------------------------------------------------

def _unpacked(path: Path) -> np.ndarray:
    with np.load(path) as data:
        shape = tuple(int(s) for s in data["shape"])
        return np.unpackbits(data["bits"], count=int(np.prod(shape))).reshape(shape).astype(bool)


def read_layers(run: Path) -> dict[str, np.ndarray]:
    """The skeletons a course is looked for in (:data:`SKELETONS`), as voxel
    indices, from the run folder's files: the one graph building was
    handed, the thickness-gated one after and before the thin-arm join (Lee
    on the thin catchment and the tree), and Lee's alone."""
    before = _unpacked(run / "skel_before_join.npz")
    tree = np.zeros_like(before)
    tree[tuple(np.load(run / "tree.npy").T)] = True
    return {
        "final": np.argwhere(_unpacked(run / "skeleton.npz")),
        "after_join": np.argwhere(_unpacked(run / "skel_after_join.npz")),
        "before_join": np.argwhere(before),
        "lee": np.argwhere(before & ~tree),
    }


def read_steps(run: Path) -> list[tuple[str, nx.MultiGraph]]:
    steps = []
    for path in sorted((run / "steps").glob("*.pkl")):
        with open(path, "rb") as f:
            steps.append((path.stem, pickle.load(f)))
    return steps


def _piece_ends(piece: np.ndarray, tree: cKDTree) -> tuple[np.ndarray, np.ndarray]:
    """``(outer, inner)``: a bridge's end further from the tree first."""
    a, b = piece[0], piece[-1]
    return (a, b) if tree.query(a)[0] >= tree.query(b)[0] else (b, a)


def trace_case(case: Mapping[str, Any], steps: Sequence[tuple[str, StepIndex]], split_input: StepIndex,
               final: StepIndex, tree_points: np.ndarray, layers: Mapping[str, np.ndarray],
               voxel_size_zyx: Sequence[float]) -> dict[str, Any]:
    """One case traced: whether the run still has it, its course, the
    steps that have it, the step that made it, and the skeleton layer."""
    from haemolynx.graph.thick_vessel_junctions import IS_ZERO_RESISTANCE

    piece = densify(case["piece_um"])
    centre = piece[len(piece) // 2]
    tree = cKDTree(tree_points)
    row: dict[str, Any] = {"id": case["id"], "kind": case["kind"], "vox": tuple(case["vox"]),
                           "length_um": round(polyline_length(piece), 2)}
    final_local = final.local(centre)
    covering = final_local.edges_covering(piece, 2.0)
    if case["kind"] == "bridge" and final_local.covered_by(
        piece, lambda data: bool(data.get(IS_ZERO_RESISTANCE)), 2.0
    ).mean() >= 0.7:
        row["now"] = "bridge"
    elif course_present(final_local, piece, reach_um=2.0):
        row["now"] = "vessel" if case["kind"] == "bridge" else "present"
    else:
        row["now"] = "gone"
    row["final_marks"] = json.dumps(edge_marks(final.G, covering), default=str)
    if row["now"] == "gone":
        return row
    course, where = piece, final_local
    labels = [label for label, _ in steps]
    indexes = [index for _, index in steps]
    if case["kind"] == "bridge" and row["now"] == "bridge":
        # The run's own bridge there: the one covering most of the case's.
        bridges = [e for e in final_local.edge_ids if final.G.edges[final.edges[e]].get(IS_ZERO_RESISTANCE)]
        line = max((final.polylines[e] for e in bridges), default=piece, key=lambda line: (
            cKDTree(line).query(piece, distance_upper_bound=2.0)[0] <= 2.0).mean())
        outer, inner = _piece_ends(line, tree)
        radius = float(case.get("local_radius_um") or 0.0)
        where = split_input.local(centre)
        course = bridge_course(where, outer, inner,
                               max_length_um=MAX_COURSE_RADII * max(radius, 1.0) + MAX_COURSE_EXTRA_UM)
        if course is None:
            row["made_by"] = "finish_thick_split"
            row["skeleton"] = ""
            return row
        # Only the steps up to the split's input: the split replaces the course by its bridge.
        last = labels.index("finish_thick_split") if "finish_thick_split" in labels else len(labels)
        labels, indexes = labels[:last], indexes[:last]
    length = polyline_length(course)
    if length < MIN_COURSE_UM:
        course = extend_course(where, course, (MIN_COURSE_UM - length) / 2.0)
    row["course_um"] = round(polyline_length(course), 2)
    present = [course_present(index.local(centre), course) for index in indexes]
    row["present"] = "".join("1" if p else "0" for p in present)
    row["made_by"] = made_by(labels, present) or "(not at the end)"
    row["skeleton"] = ""
    if labels and row["made_by"] == labels[0]:
        row["skeleton"], row["skeleton_has"] = skeleton_maker(
            course, skeletons_near(layers, voxel_size_zyx, centre)
        )
    return row


def bridge_cases(G: nx.MultiGraph, thick: np.ndarray, voxel_size_zyx: Sequence[float]) -> list[dict[str, Any]]:
    """Every bridge of *G* as a case :func:`trace_case` reads: its polyline,
    its middle voxel, and the thick region's inscribed radius at its inner
    end (the end deeper in *thick*) as the lumen radius."""
    from scipy.ndimage import distance_transform_edt

    from haemolynx.graph.thick_vessel_junctions import IS_ZERO_RESISTANCE

    spacing = np.asarray(voxel_size_zyx, float)
    shape = np.asarray(thick.shape)
    cases = []
    for u, v, k, data in G.edges(keys=True, data=True):
        if not data.get(IS_ZERO_RESISTANCE):
            continue
        piece = _edge_points(G, u, v, data)
        if piece is None or len(piece) < 2:
            continue
        radii = []
        for end in (piece[0], piece[-1]):
            voxel = np.clip(np.round(end / spacing).astype(int), 0, shape - 1)
            lo = np.maximum(voxel - 25, 0)
            hi = np.minimum(voxel + 26, shape)
            crop = thick[tuple(slice(a, b) for a, b in zip(lo, hi))]
            radii.append(float(distance_transform_edt(crop, sampling=spacing)[tuple(voxel - lo)]))
        middle = piece[len(piece) // 2]
        cases.append({
            "id": f"{u}-{v}-{k}", "kind": "bridge", "edge": (u, v, k),
            "vox": [int(c) for c in np.round(middle / spacing)],
            "piece_um": piece.tolist(), "local_radius_um": max(radii),
        })
    return cases


def summary_markdown(rows: Sequence[Mapping[str, Any]]) -> str:
    """Counts by what made each connection, with the IDs and positions."""
    def maker(row):
        if row.get("now") == "gone":
            return "gone at this commit"
        if row.get("skeleton"):
            return f"skeleton: {row['skeleton']}"
        return row.get("made_by", "")

    groups: dict[str, list] = {}
    for row in rows:
        groups.setdefault(maker(row), []).append(row)
    lines = ["| made by | cases | IDs (voxel z, y, x) |", "|---|---|---|"]
    for name, members in sorted(groups.items(), key=lambda item: (-len(item[1]), item[0])):
        ids = ", ".join(f"{r['id']} {r['kind'][0]} {tuple(r['vox'])}" for r in members)
        lines.append(f"| {name} | {len(members)} | {ids} |")
    lines.append("")
    lines.append("Now: " + ", ".join(f"{k} {v}" for k, v in sorted(Counter(r.get('now') for r in rows).items())))
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run", required=True, help="the run folder the steps were captured into")
    parser.add_argument("--grid", required=True, help="the grid's folder: cases.json, marks.csv")
    parser.add_argument("--mark", default="N", help="the mark whose cases are traced")
    parser.add_argument("--all-bridges", action="store_true",
                        help="trace every bridge of the run's graph instead of the grid's cases")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    run, grid, out = Path(args.run), Path(args.grid), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    voxel_size = json.loads((run / "voxel_size_zyx.json").read_text())
    tree_points = np.load(run / "tree.npy").astype(float) * np.asarray(voxel_size, float)
    layers = read_layers(run)
    steps = [(label.split("_", 1)[1], StepIndex(G)) for label, G in read_steps(run)]
    with open(run / "split_call.pkl", "rb") as f:
        split_input = StepIndex(pickle.load(f)["G"])
    with open(run / "graph_built.pkl", "rb") as f:
        final = StepIndex(pickle.load(f))
    if args.all_bridges:
        chosen = bridge_cases(final.G, _unpacked(run / "thick.npz"), voxel_size)
        args.mark = "bridge"
    else:
        cases = json.loads((grid / "cases.json").read_text(encoding="utf-8"))
        with open(grid / "marks.csv", encoding="utf-8") as f:
            marks = {int(r["id"]): r["mark"].strip().upper() for r in csv.DictReader(f)}
        chosen = [c for c in cases if marks.get(int(c["id"])) == args.mark.upper()]
    rows = [trace_case(c, steps, split_input, final, tree_points, layers, voxel_size) for c in chosen]
    fields = ["id", "kind", "vox", "length_um", "now", "course_um", "made_by", "skeleton", "present",
              "skeleton_has", "final_marks"]
    with open(out / "trace.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    labels = " ".join(f"{i + 1}={label}" for i, (label, _) in enumerate(steps))
    (out / "TRACE.md").write_text(
        f"# Where the {args.mark} connections come from\n\n{summary_markdown(rows)}\nSteps: {labels}\n",
        encoding="utf-8",
    )
    print((out / "TRACE.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
