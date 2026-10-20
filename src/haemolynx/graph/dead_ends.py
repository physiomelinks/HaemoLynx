"""What each dead end of a vessel network is: one definition for every module that asks.

A dead end is a degree-1 node. It is judged as its whole *chain* -- from the
tip through degree-2 nodes to the first node that is not one (the junction it
hangs from), as the stub prune judges a stub -- and given one kind:

- :data:`LEAVES_IMAGE` -- the vessel runs out of the field of view: its tip
  touches an image face, or the segmented vessel runs on from the tip to a face
  within :data:`FACE_REACH_RADII` of its radius. Open in the tissue: what the
  flow's open-end boundaries are put on.
- :data:`ISLAND_END` -- an end of a piece with no junction.
- :data:`ARTEFACT` -- not a vessel end at all: a chain mostly off the mask, a
  tip inside the lumen of the vessel it leaves or of one close by in the
  network, a strand beside another centreline in one lumen, or a blind end
  shorter than the stub prune's radius rule.
- :data:`STOPS_SHORT` -- the segmented vessel runs on past the tip: a vessel
  the graph stopped short of.
- :data:`BROKEN` -- another vessel lies straight ahead across one stretch of
  background, and it is not one of the dead end's own neighbours in the
  network (or the tip ends inside such a vessel's lumen): a vessel the
  segmentation broke.
- :data:`BLIND` -- a real blind end. Only a person says so (the Post
  processing review's "Mark as blind end"), until marks give a rule.
- :data:`UNKNOWN` -- an interior end that is none of the above: a blind end
  or a broken vessel, for review.

These are first proposals (PLAN_dead_ends.md, section 4), to be calibrated on
marked grids; the measures each verdict rests on travel with it
(:class:`DeadEnd`), so a rule can be fitted to marks rather than to an
assumption (CLAUDE.md, "Marked example grids"). Without a segmented mask only
:data:`ISLAND_END`, the distance part of :data:`LEAVES_IMAGE`, the geometric
part of :data:`BROKEN` and :data:`UNKNOWN` can be told.

A kind a person set (``dead_end_kind_source == "review"``) is kept by
:func:`mark_dead_end_kinds` for as long as the node stays a dead end.
"""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping, Optional, Sequence

import networkx as nx
import numpy as np

from ._helpers import densify_polyline, edge_id, edge_sample_points

LEAVES_IMAGE = "leaves_image"
ISLAND_END = "island_end"
ARTEFACT = "artefact"
STOPS_SHORT = "stops_short"
BROKEN = "broken"
BLIND = "blind"
UNKNOWN = "unknown"
#: Every kind, in the order the census and the review list show them.
DEAD_END_KINDS = (LEAVES_IMAGE, STOPS_SHORT, BROKEN, ARTEFACT, BLIND, UNKNOWN, ISLAND_END)

#: The node attribute :func:`mark_dead_end_kinds` writes, and who set it.
DEAD_END_KIND = "dead_end_kind"
DEAD_END_KIND_SOURCE = "dead_end_kind_source"
#: A kind a person chose in the review, which reclassifying keeps.
SOURCE_REVIEW = "review"
SOURCE_CLASSIFIER = "classifier"
#: Known by construction: the generator of a synthetic network (``haemolynx.synthetic``).
SOURCE_SYNTHETIC = "synthetic"

#: A tip within this many of the finest voxels of a face touches it -- or
#: within its own lumen radius, whichever is larger.
FACE_TOUCH_VOXELS = 2.0
#: A tip whose segmented vessel runs on, along its heading, to a face within
#: this many of its own lumen radii (plus :data:`FACE_TOUCH_VOXELS`) leaves the
#: image: a centreline often stops short of the face its tube runs into.
FACE_REACH_RADII = 4.0
#: How far back from the tip its heading is read, in microns.
HEADING_LENGTH_UM = 3.0
#: A vessel straight ahead is looked for this far, and at most this far off
#: the tip's heading -- the facing join's turn limit.
AHEAD_REACH_UM = 10.0
AHEAD_MAX_TURN_DEG = 60.0
#: A vessel ahead this many junctions or more from the dead end's own junction
#: is not one of its neighbours (the One centreline session's marks on E14.5's
#: seven tips inside another lumen, 2026-10-11: removed at 0-1 junctions away,
#: joined at 2, 3 and 11). Counted no further than :data:`AHEAD_MAX_HOPS` edges.
NEIGHBOUR_MAX_JUNCTIONS = 1
AHEAD_MAX_HOPS = 40
#: How far the mask is followed past a tip to say how far it runs on, microns.
MASK_RUN_MAX_UM = 30.0
#: The tip is "inside" another lumen within that lumen's radius plus this.
LUMEN_MARGIN_UM = 1.0


@dataclass(frozen=True)
class DeadEnd:
    """One dead end, its kind, and every measure the kind rests on."""

    tip: Any
    #: The node its chain ends at: the junction it hangs from, or the far end
    #: of an island chain.
    junction: Any
    kind: str
    #: The chain's edges ``(a, b, key)``, tip first, *a* nearer the tip.
    chain: tuple
    chain_length_um: float
    #: The tip's position, physical ``(z, y, x)`` microns.
    tip_um: tuple
    #: Unit vector out of the tip, or None (a chain too short to read).
    heading: Optional[tuple]
    face_distance_um: Optional[float] = None
    #: The axis (0 = z, 1 = y, 2 = x) of the nearest face.
    face_axis: Optional[int] = None
    #: Whether the mask runs from the tip to a face along its heading.
    reaches_face: bool = False
    tip_radius_um: Optional[float] = None
    junction_radius_um: Optional[float] = None
    #: How far the mask runs on past the tip along its heading (capped at
    #: :data:`MASK_RUN_MAX_UM`), and whether it runs on at all
    #: (``assemble.mask_continues_past``).
    mask_run_um: Optional[float] = None
    mask_continues: bool = False
    #: Share of the chain in the mask.
    in_mask_fraction: Optional[float] = None
    #: Share of the chain's judged samples beside another centreline in one lumen.
    beside_share: Optional[float] = None
    #: Whether the tip lies inside another centreline's lumen, and how many
    #: junctions away in the network that centreline is (None: not reached).
    tip_in_other_lumen: bool = False
    lumen_junctions_away: Optional[int] = None
    #: The nearest other centreline straight ahead, within
    #: :data:`AHEAD_REACH_UM` and :data:`AHEAD_MAX_TURN_DEG`: its distance,
    #: the turn onto it, the edge, how many junctions away it is, and the
    #: background the straight line to it crosses (stretches, microns).
    ahead_um: Optional[float] = None
    ahead_turn_deg: Optional[float] = None
    ahead_edge: Optional[tuple] = None
    ahead_junctions_away: Optional[int] = None
    ahead_background_stretches: Optional[int] = None
    ahead_background_um: Optional[float] = None
    #: The old artefact labels, for the lumen-artefact report:
    #: ``inside_other_lumen``, ``beside_vessel``, ``off_mask``,
    #: ``mask_continues``, ``short``.
    flags: tuple = ()
    #: Who chose the kind: :data:`SOURCE_CLASSIFIER` or :data:`SOURCE_REVIEW`.
    source: str = SOURCE_CLASSIFIER
    extra: Mapping[str, Any] = field(default_factory=dict, compare=False)

    @property
    def interior(self) -> bool:
        """Away from the image faces: anything but :data:`LEAVES_IMAGE`."""
        return self.kind != LEAVES_IMAGE

    def measures(self) -> dict[str, Any]:
        """Its measures as plain values, for a CSV row or a grid key."""
        row = asdict(self)
        row.pop("extra", None)
        row["chain"] = ";".join(f"{a}-{b}" for a, b, _k in self.chain)
        row["flags"] = ";".join(self.flags)
        for name in ("tip_um", "heading", "ahead_edge"):
            value = row.get(name)
            row[name] = "" if value is None else ";".join(f"{v:.4g}" if isinstance(v, float) else str(v) for v in value)
        if self.tip_radius_um and self.junction_radius_um:
            row["parent_to_tip_radius"] = self.junction_radius_um / self.tip_radius_um
        row["depth_um"] = float(self.tip_um[0])
        row.update(self.extra)
        return row


# --- geometry shared by every caller ------------------------------------------------


def chain_to_junction(G: nx.MultiGraph, tip: Any) -> tuple[list, list]:
    """``(nodes, edges)`` from a dead end's *tip* through degree-2 nodes to
    the first node that is not one: the junction it hangs from (or, for an
    isolated chain, its other end). Edges are ``(a, b, key)``, *a* nearer the
    tip. The stub prune's walk (``prune._chain``), shared."""
    from .prune import _chain

    return _chain(G, tip)


def chain_path(G: nx.MultiGraph, edges: Sequence[tuple], *, step_um: float = 0.5) -> np.ndarray:
    """A chain's centreline from its tip to its junction, every *step_um*."""
    pieces = []
    for a, b, key in edges:
        data = G.edges[a, b, key] if G.is_multigraph() else G.edges[a, b]
        pos = {n: np.asarray(G.nodes[n]["pos"], dtype=float) for n in (a, b)}
        piece = edge_sample_points(a, b, data, pos)
        pieces.append(piece if not pieces else piece[1:])
    if not pieces:
        return np.empty((0, 3))
    return densify_polyline(np.vstack(pieces), max_step_um=step_um)


def chain_length_um(G: nx.MultiGraph, edges: Sequence[tuple]) -> float:
    """A chain's length: its edges' ``length``, or their polylines' where unset."""
    from ._helpers import calculate_path_length

    total = 0.0
    for a, b, key in edges:
        data = G.edges[a, b, key] if G.is_multigraph() else G.edges[a, b]
        length = data.get("length")
        if length is None:
            pos = {n: np.asarray(G.nodes[n]["pos"], dtype=float) for n in (a, b)}
            length = calculate_path_length(edge_sample_points(a, b, data, pos).tolist())
        total += float(length)
    return total


def tip_heading(path_from_tip: np.ndarray, length_um: float = HEADING_LENGTH_UM) -> Optional[np.ndarray]:
    """The unit vector out of a tip, read over the last *length_um* of its
    chain (*path_from_tip* runs tip first), or None for a chain with no
    length."""
    path = np.asarray(path_from_tip, dtype=float).reshape(-1, 3)
    if len(path) < 2:
        return None
    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))])
    back = path[min(int(np.searchsorted(arc, length_um)), len(path) - 1)]
    outward = path[0] - back
    norm = float(np.linalg.norm(outward))
    return None if norm <= 0 else outward / norm


def image_extent_um(image_shape: Sequence[int], voxel_size_zyx: Sequence[float]) -> np.ndarray:
    """The image's ``(z, y, x)`` extent in microns: node positions run from 0
    to it."""
    return (np.asarray(image_shape, dtype=float)[:3] - 1.0) * np.asarray(voxel_size_zyx, dtype=float)


def graph_image_extent_um(G: nx.Graph) -> Optional[np.ndarray]:
    """The extent :func:`record_image_extent` stored on *G*, or None."""
    extent = G.graph.get("image_extent_um")
    return None if extent is None else np.asarray(extent, dtype=float)


def record_image_extent(G: nx.Graph, image_shape: Sequence[int], voxel_size_zyx: Sequence[float]) -> None:
    """Store the image's extent on *G*, so stages without the image can still
    tell a vessel leaving it."""
    G.graph["image_extent_um"] = tuple(float(v) for v in image_extent_um(image_shape, voxel_size_zyx))


def face_distance(point_um: Sequence[float], extent_um: Sequence[float]) -> tuple[float, int]:
    """``(distance, axis)`` from *point_um* to the nearest image face."""
    point = np.asarray(point_um, dtype=float)[:3]
    extent = np.asarray(extent_um, dtype=float)[:3]
    both = np.concatenate([point, extent - point])
    nearest = int(np.argmin(both))
    return float(both[nearest]), nearest % 3


def _distance_to_box_along(point: np.ndarray, direction: np.ndarray, extent: np.ndarray) -> Optional[float]:
    limits = []
    for axis in range(3):
        if direction[axis] > 1e-8:
            limits.append((extent[axis] - point[axis]) / direction[axis])
        elif direction[axis] < -1e-8:
            limits.append(point[axis] / -direction[axis])
    return float(min(limits)) if limits else None


def leaves_image(
    tip_um: Sequence[float],
    extent_um: Sequence[float],
    *,
    heading: Optional[Sequence[float]] = None,
    support=None,
    tip_radius_um: Optional[float] = None,
    voxel_size_zyx: Optional[Sequence[float]] = None,
) -> tuple[bool, bool]:
    """``(leaves, reaches_face)``: whether a dead end at *tip_um* is a vessel
    the image cut through, and whether it said so because the mask runs on to
    a face.

    It does when the tip lies within its own lumen radius, or
    :data:`FACE_TOUCH_VOXELS` of the finest voxel, of a face; or when, with
    *support* (``MaskSupport``) and a *heading*, the mask runs without a break
    from the tip along the heading to a face no further than
    :data:`FACE_REACH_RADII` lumen radii (plus the touch distance) away.
    """
    spacing = np.asarray(
        voxel_size_zyx if voxel_size_zyx is not None
        else (support.voxel_size_zyx if support is not None else (1.0, 1.0, 1.0)),
        dtype=float,
    )
    finest = float(spacing.min())
    point = np.asarray(tip_um, dtype=float)[:3]
    extent = np.asarray(extent_um, dtype=float)[:3]
    radius = float(tip_radius_um) if tip_radius_um is not None and np.isfinite(tip_radius_um) else 0.0
    touch = max(FACE_TOUCH_VOXELS * finest, radius)
    distance, _axis = face_distance(point, extent)
    if distance <= touch + 1e-9:
        return True, False
    if support is None or heading is None:
        return False, False
    direction = np.asarray(heading, dtype=float)
    reach = _distance_to_box_along(point, direction, extent)
    if reach is None or reach > FACE_REACH_RADII * max(radius, finest) + touch:
        return False, False
    step = finest / 2.0
    samples = point + (np.arange(1, int(np.ceil(reach / step)) + 1) * step)[:, None] * direction
    samples = samples[np.all((samples >= -1e-6) & (samples <= extent + 1e-6), axis=1)]
    runs_on = len(samples) > 0 and bool(np.all(support.inside(samples)))
    return runs_on, runs_on


def mask_run_um(tip_um: np.ndarray, heading: np.ndarray, support, *, limit_um: float = MASK_RUN_MAX_UM) -> float:
    """How far the mask runs on from the tip along *heading* before the first
    background, starting one lumen radius out (the tip's own lumen does not
    count), capped at *limit_um*."""
    tip = np.asarray(tip_um, dtype=float).reshape(1, 3)
    start = float(support.radius(tip)[0])
    step = float(min(support.voxel_size_zyx)) / 2.0
    offsets = start + np.arange(1, int(np.ceil(limit_um / step)) + 1) * step
    samples = tip + offsets[:, None] * np.asarray(heading, dtype=float)
    inside = np.asarray(support.inside(samples), dtype=bool)
    if inside.all():
        return float(limit_um)
    first_out = int(np.argmin(inside))
    return float(first_out * step)


# --- the classifier -------------------------------------------------------------------


class _Centrelines:
    """Every edge's centreline samples over one graph, with the lumen radius
    at each when a mask is given: for asking what lies ahead of a tip, and
    whether it ends in another vessel's lumen."""

    def __init__(self, G: nx.MultiGraph, support=None, step_um: float = 1.0):
        from scipy.spatial import cKDTree

        points, owners = [], []
        self.edges: list[tuple] = []
        for u, v, key, data in G.edges(keys=True, data=True):
            if u == v or "pos" not in G.nodes[u] or "pos" not in G.nodes[v]:
                continue
            pos = {n: np.asarray(G.nodes[n]["pos"], dtype=float) for n in (u, v)}
            path = densify_polyline(edge_sample_points(u, v, data, pos), max_step_um=step_um)
            points.append(path)
            owners.extend([len(self.edges)] * len(path))
            self.edges.append((u, v, key))
        self.ids = {edge_id(*e): i for i, e in enumerate(self.edges)}
        self.points = np.vstack(points) if points else np.empty((0, 3))
        self.owners = np.asarray(owners, dtype=np.intp)
        self.tree = cKDTree(self.points) if len(self.points) else None
        self.radii = (
            np.asarray(support.radius(self.points), dtype=float).reshape(-1)
            if support is not None and len(self.points) else None
        )

    def near(self, point: np.ndarray, reach: float, exclude: set) -> np.ndarray:
        if self.tree is None:
            return np.empty(0, dtype=np.intp)
        found = np.asarray(self.tree.query_ball_point(np.asarray(point, dtype=float), r=reach), dtype=np.intp)
        if not len(found):
            return found
        return found[~np.isin(self.owners[found], list(exclude))] if exclude else found


def _junctions_away(G: nx.MultiGraph, start: Any, targets: set, *, max_hops: int = AHEAD_MAX_HOPS) -> Optional[int]:
    """How many junctions (nodes of degree 3 or more) a shortest hop path
    from *start* passes before reaching any node of *targets* -- *start*
    itself not counted, a target that is a junction counted -- or None when
    none is reached within *max_hops*. 0 when *start* is in *targets*."""
    if start in targets:
        return 0
    best: dict[Any, int] = {start: 0}
    queue = deque([(start, 0, 0)])
    while queue:
        node, hops, junctions = queue.popleft()
        if hops >= max_hops:
            continue
        for other in G.neighbors(node):
            if other in best:
                continue
            count = junctions + (1 if G.degree(other) >= 3 else 0)
            if other in targets:
                return count
            best[other] = count
            queue.append((other, hops + 1, count))
    return None


def _beside_share(path: np.ndarray, own: set, centrelines: _Centrelines, support, step_um: float) -> Optional[float]:
    from haemolynx.preprocessing.bridge_mask_support import shadowed_samples
    from scipy.spatial import cKDTree

    if centrelines.tree is None or len(path) < 2:
        return None
    reach = 2.0 * float(np.max(support.radius(path))) + 1.0
    found = centrelines.tree.query_ball_point(path, r=reach)
    near = np.unique(np.concatenate([np.asarray(f, dtype=np.intp) for f in np.atleast_1d(found)])) if len(found) else np.empty(0, dtype=np.intp)
    near = near[~np.isin(centrelines.owners[near], list(own))] if len(near) else near
    if not len(near):
        return 0.0
    points = centrelines.points[near]
    _samples, judged, beside = shadowed_samples(
        path, cKDTree(points), points, support.inside, support.radius, step_um=step_um
    )
    return float(beside[judged].mean()) if judged.any() else None


def classify_dead_ends(
    G: nx.MultiGraph,
    *,
    extent_um: Optional[Sequence[float]] = None,
    support=None,
    stub_radius_multiple: float = 3.0,
    ahead_reach_um: float = AHEAD_REACH_UM,
    nodes: Optional[Iterable[Any]] = None,
    face_margin_um: Optional[float] = None,
) -> dict[Any, DeadEnd]:
    """Every dead end of *G* (or of *nodes*), and what it is: ``{tip: DeadEnd}``
    in node order. See the module docstring for the kinds.

    *extent_um* is the image's ``(z, y, x)`` extent in microns
    (:func:`image_extent_um`; the graph's own, :func:`graph_image_extent_um`,
    when None); without one no dead end leaves the image. *support* is a
    ``MaskSupport`` over the segmented mask; without one only the geometric
    kinds are told. *stub_radius_multiple* is the stub prune's radius rule
    (``min_stub_length_radius_multiple``). *face_margin_um*, when given,
    replaces the face test with a fixed distance: a tip that close to a face
    leaves the image, any other does not (0 keeps every tip off the faces --
    what a synthetic test volume wants).

    A kind a person set (:data:`SOURCE_REVIEW`) is reported as it is.
    """
    from haemolynx.preprocessing.bridge_mask_support import _sample_step

    from .assemble import mask_continues_past
    from .facing_ends import _background

    extent = np.asarray(extent_um, dtype=float) if extent_um is not None else graph_image_extent_um(G)
    voxel_size = (
        support.voxel_size_zyx if support is not None
        else G.graph.get("image_voxel_size_zyx", (1.0, 1.0, 1.0))
    )
    step = _sample_step(voxel_size)
    continues = mask_continues_past(support) if support is not None else None
    wanted = [
        n for n in (G.nodes if nodes is None else nodes)
        if n in G and G.degree(n) == 1 and G.nodes[n].get("pos") is not None
    ]
    centrelines = _Centrelines(G, support) if wanted else None
    found: dict[Any, DeadEnd] = {}
    for tip in wanted:
        chain_nodes, chain_edges = chain_to_junction(G, tip)
        junction = chain_nodes[-1]
        own = {centrelines.ids[edge_id(*e)] for e in chain_edges if edge_id(*e) in centrelines.ids}
        tip_um = np.asarray(G.nodes[tip]["pos"], dtype=float)[:3]
        path = chain_path(G, chain_edges, step_um=0.5) if chain_edges else tip_um.reshape(1, 3)
        heading = tip_heading(path)
        length = chain_length_um(G, chain_edges)
        values: dict[str, Any] = dict(
            tip=tip, junction=junction, chain=tuple(chain_edges), chain_length_um=length,
            tip_um=tuple(float(v) for v in tip_um), heading=None if heading is None else tuple(float(v) for v in heading),
        )
        if G.nodes[tip].get(DEAD_END_KIND_SOURCE) == SOURCE_REVIEW and G.nodes[tip].get(DEAD_END_KIND):
            values["source"] = SOURCE_REVIEW
        tip_radius = float(support.radius(tip_um.reshape(1, 3))[0]) if support is not None else None
        values["tip_radius_um"] = tip_radius
        junction_degree = G.degree(junction)
        if support is not None and junction_degree > 1 and G.nodes[junction].get("pos") is not None:
            values["junction_radius_um"] = float(
                support.radius(np.asarray(G.nodes[junction]["pos"], dtype=float).reshape(1, 3))[0]
            )
        leaves = False
        if extent is not None:
            values["face_distance_um"], values["face_axis"] = face_distance(tip_um, extent)
            if face_margin_um is not None:
                leaves = values["face_distance_um"] <= float(face_margin_um)
            else:
                leaves, values["reaches_face"] = leaves_image(
                    tip_um, extent, heading=heading, support=support, tip_radius_um=tip_radius,
                    voxel_size_zyx=voxel_size,
                )
        flags: list[str] = []
        if support is not None and len(path) > 1:
            inside = np.asarray(support.inside(path), dtype=bool)
            values["in_mask_fraction"] = float(inside.mean())
            if 2 * int(np.count_nonzero(inside)) < len(path):
                flags.append("off_mask")
            near = centrelines.near(tip_um, 2.0 * (tip_radius or 0.0) + 30.0, own)
            if len(near):
                distance = np.linalg.norm(centrelines.points[near] - tip_um, axis=1)
                nearest = int(near[int(np.argmin(distance))])
                if float(distance.min()) <= float(centrelines.radii[nearest]) + LUMEN_MARGIN_UM:
                    values["tip_in_other_lumen"] = True
                    flags.append("inside_other_lumen")
                    edge = centrelines.edges[int(centrelines.owners[nearest])]
                    values["lumen_junctions_away"] = _junctions_away(G, junction, {edge[0], edge[1]})
            values["beside_share"] = _beside_share(path, own, centrelines, support, step)
            if (values["beside_share"] or 0.0) >= 0.5:
                flags.append("beside_vessel")
            if heading is not None:
                values["mask_continues"] = bool(continues(tip_um, heading))
                values["mask_run_um"] = mask_run_um(tip_um, heading, support)
                if values["mask_continues"]:
                    flags.append("mask_continues")
            junction_radius = values.get("junction_radius_um")
            if junction_radius and stub_radius_multiple > 0 and length < stub_radius_multiple * junction_radius:
                flags.append("short")
        if heading is not None and centrelines.tree is not None:
            near = centrelines.near(tip_um, ahead_reach_um, own)
            if len(near):
                chords = centrelines.points[near] - tip_um
                gaps = np.linalg.norm(chords, axis=1)
                ok = gaps > 1e-6
                cosines = np.full(len(near), -1.0)
                cosines[ok] = chords[ok] @ heading / gaps[ok]
                ahead = cosines >= float(np.cos(np.radians(AHEAD_MAX_TURN_DEG)))
                if ahead.any():
                    best = int(np.flatnonzero(ahead)[int(np.argmin(gaps[ahead]))])
                    target = centrelines.points[near[best]]
                    edge = centrelines.edges[int(centrelines.owners[near[best]])]
                    values["ahead_um"] = float(gaps[best])
                    values["ahead_turn_deg"] = float(np.degrees(np.arccos(np.clip(cosines[best], -1.0, 1.0))))
                    values["ahead_edge"] = tuple(edge)
                    values["ahead_junctions_away"] = _junctions_away(G, junction, {edge[0], edge[1]})
                    if support is not None:
                        stretches, background = _background(
                            densify_polyline(np.vstack([tip_um, target]), max_step_um=1.0), support
                        )
                        values["ahead_background_stretches"] = stretches
                        values["ahead_background_um"] = background
        values["flags"] = tuple(flags)
        values["kind"] = (
            G.nodes[tip][DEAD_END_KIND] if values.get("source") == SOURCE_REVIEW
            else _kind(values, junction_degree=junction_degree, leaves=leaves, has_mask=support is not None)
        )
        found[tip] = DeadEnd(**values)
    return found


def _not_a_neighbour(junctions_away: Optional[int]) -> bool:
    return junctions_away is None or junctions_away > NEIGHBOUR_MAX_JUNCTIONS


def _kind(values: Mapping[str, Any], *, junction_degree: int, leaves: bool, has_mask: bool) -> str:
    """The first proposal of PLAN_dead_ends.md, section 4, in its order."""
    if junction_degree <= 1:
        return ISLAND_END
    if leaves:
        return LEAVES_IMAGE
    flags = set(values.get("flags", ()))
    if values.get("tip_in_other_lumen") and _not_a_neighbour(values.get("lumen_junctions_away")):
        # It ends inside a vessel the network only reaches round about: a
        # connection the graph missed, not a hook back into its own vessel.
        return BROKEN
    if flags & {"off_mask", "inside_other_lumen", "beside_vessel"}:
        return ARTEFACT
    if values.get("mask_continues"):
        return STOPS_SHORT
    if "short" in flags:
        return ARTEFACT
    if (
        values.get("ahead_um") is not None
        and _not_a_neighbour(values.get("ahead_junctions_away"))
        and (not has_mask or (values.get("ahead_background_stretches") or 0) <= 1)
    ):
        return BROKEN
    return UNKNOWN


def mark_dead_end_kinds(G: nx.MultiGraph, dead_ends: Mapping[Any, DeadEnd]) -> Counter:
    """Write each dead end's kind on its node (:data:`DEAD_END_KIND`, with
    :data:`DEAD_END_KIND_SOURCE`), and take the attributes off every node that
    is no longer a dead end. A person's kind stays while its node is a dead end.
    Returns the count of each kind."""
    for node, data in G.nodes(data=True):
        if node not in dead_ends and (DEAD_END_KIND in data or DEAD_END_KIND_SOURCE in data):
            if G.degree(node) != 1 or data.get(DEAD_END_KIND_SOURCE) != SOURCE_REVIEW:
                data.pop(DEAD_END_KIND, None)
                data.pop(DEAD_END_KIND_SOURCE, None)
    counts: Counter = Counter()
    for node, end in dead_ends.items():
        if node not in G:
            continue
        G.nodes[node][DEAD_END_KIND] = end.kind
        G.nodes[node][DEAD_END_KIND_SOURCE] = end.source
        counts[end.kind] += 1
    return counts


def classify_and_mark(
    G: nx.MultiGraph,
    *,
    extent_um: Optional[Sequence[float]] = None,
    support=None,
    stub_radius_multiple: float = 3.0,
) -> dict[Any, DeadEnd]:
    """:func:`classify_dead_ends` and :func:`mark_dead_end_kinds` in one."""
    ends = classify_dead_ends(G, extent_um=extent_um, support=support, stub_radius_multiple=stub_radius_multiple)
    mark_dead_end_kinds(G, ends)
    return ends


def reclassify_dead_ends(
    G: nx.MultiGraph,
    nodes: Iterable[Any],
    *,
    extent_um: Optional[Sequence[float]] = None,
    support=None,
    stub_radius_multiple: float = 3.0,
) -> dict[Any, DeadEnd]:
    """:func:`classify_dead_ends` over *nodes* alone -- the nodes a hand edit
    touched -- and their marks brought in line, every other node's left as
    it is: a dead end among them gets its kind (:data:`DEAD_END_KIND`,
    :data:`DEAD_END_KIND_SOURCE`); one that is a dead end no longer loses the
    classifier's. A person's kind (:data:`SOURCE_REVIEW`) stays whatever the
    node's degree -- an edit that joins a vessel to the node can be undone,
    and nothing reads the mark while the node is no dead end (see
    :func:`classify_dead_ends`, :func:`node_leaves_image`). Returns the dead
    ends judged, as :func:`classify_dead_ends` does."""
    wanted = [node for node in dict.fromkeys(nodes) if node in G]
    ends = classify_dead_ends(
        G, extent_um=extent_um, support=support, stub_radius_multiple=stub_radius_multiple, nodes=wanted
    )
    for node in wanted:
        data = G.nodes[node]
        if node in ends:
            data[DEAD_END_KIND] = ends[node].kind
            data[DEAD_END_KIND_SOURCE] = ends[node].source
        elif G.degree(node) != 1 and data.get(DEAD_END_KIND_SOURCE) != SOURCE_REVIEW:
            data.pop(DEAD_END_KIND, None)
            data.pop(DEAD_END_KIND_SOURCE, None)
    return ends


def set_dead_end_kind(G: nx.MultiGraph, node: Any, kind: str) -> None:
    """A person's verdict on a dead end (the Post processing review), kept
    through later reclassifying while the node stays a dead end."""
    if kind not in DEAD_END_KINDS:
        raise ValueError(f"kind must be one of {DEAD_END_KINDS}, got {kind!r}")
    if node not in G or G.degree(node) != 1:
        raise ValueError(f"node {node!r} is not a dead end")
    G.nodes[node][DEAD_END_KIND] = kind
    G.nodes[node][DEAD_END_KIND_SOURCE] = SOURCE_REVIEW


def node_leaves_image(G: nx.Graph, node: Any, extent_um: Optional[Sequence[float]] = None) -> bool:
    """Whether dead end *node* leaves the image: its kind when it carries one
    (:func:`mark_dead_end_kinds`), else -- a dead end a later step made, or a
    graph never classified -- the distance part of :func:`leaves_image`, its
    radius read from its vessel's ``diameter_um`` where there is one."""
    if node not in G or G.degree(node) != 1:
        return False
    data = G.nodes[node]
    kind = data.get(DEAD_END_KIND)
    if kind is not None:
        return kind == LEAVES_IMAGE
    extent = np.asarray(extent_um, dtype=float) if extent_um is not None else graph_image_extent_um(G)
    if extent is None or data.get("pos") is None:
        return False
    voxel = np.asarray(G.graph.get("image_voxel_size_zyx", (1.0, 1.0, 1.0)), dtype=float)
    radius = None
    for *_e, edge_data in G.edges(node, data=True):
        diameter = edge_data.get("diameter_um")
        if diameter:
            radius = float(diameter) / 2.0
    return leaves_image(data["pos"], extent, tip_radius_um=radius, voxel_size_zyx=voxel)[0]


def open_end_nodes(G: nx.Graph, *, extent_um: Optional[Sequence[float]] = None) -> list[Any]:
    """The dead ends of *G* that leave the image, in node order
    (:func:`node_leaves_image`)."""
    extent = np.asarray(extent_um, dtype=float) if extent_um is not None else graph_image_extent_um(G)
    return [node for node in G.nodes if G.degree(node) == 1 and node_leaves_image(G, node, extent)]


def remove_pieces_without_junction(G: nx.MultiGraph) -> dict[str, float]:
    """Remove every piece of *G* that has no junction, in place: the chains
    whose dead ends are :data:`ISLAND_END` (a tip whose chain ends at another
    dead end, not at a junction), whatever their length or where they lie.
    Rings and lone nodes have no dead end and stay.

    Returns ``{"pieces", "vessels", "nodes", "length_um"}`` removed.
    """
    removed: set = set()
    pieces = vessels = 0
    length = 0.0
    for tip in [n for n in G.nodes if G.degree(n) == 1]:
        if tip in removed:
            continue
        nodes, edges = chain_to_junction(G, tip)
        if G.degree(nodes[-1]) > 1:
            continue
        pieces += 1
        vessels += len(edges)
        length += chain_length_um(G, edges)
        removed.update(nodes)
    G.remove_nodes_from(removed)
    return {"pieces": pieces, "vessels": vessels, "nodes": len(removed), "length_um": length}


def dead_end_census(dead_ends: Mapping[Any, DeadEnd]) -> dict[str, int]:
    """How many dead ends of each kind, every kind listed."""
    counts = Counter(end.kind for end in dead_ends.values())
    return {kind: int(counts.get(kind, 0)) for kind in DEAD_END_KINDS}


__all__ = [
    "AHEAD_MAX_TURN_DEG",
    "AHEAD_REACH_UM",
    "ARTEFACT",
    "BLIND",
    "BROKEN",
    "DEAD_END_KIND",
    "DEAD_END_KINDS",
    "DEAD_END_KIND_SOURCE",
    "DeadEnd",
    "FACE_REACH_RADII",
    "FACE_TOUCH_VOXELS",
    "ISLAND_END",
    "LEAVES_IMAGE",
    "SOURCE_CLASSIFIER",
    "SOURCE_REVIEW",
    "STOPS_SHORT",
    "UNKNOWN",
    "chain_length_um",
    "chain_path",
    "chain_to_junction",
    "classify_and_mark",
    "classify_dead_ends",
    "dead_end_census",
    "face_distance",
    "graph_image_extent_um",
    "image_extent_um",
    "leaves_image",
    "mark_dead_end_kinds",
    "mask_run_um",
    "node_leaves_image",
    "open_end_nodes",
    "reclassify_dead_ends",
    "record_image_extent",
    "remove_pieces_without_junction",
    "set_dead_end_kind",
    "tip_heading",
]
