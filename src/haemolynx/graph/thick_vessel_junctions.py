"""Bridges where vessels open into a thick vessel: one radius each.

A thick vessel is represented by its centreline alone -- the tree
:func:`preprocessing.thick_vessels.skeletonize_edt_ridge` draws through the
thick-vessel region -- and every vessel opening into it is joined to that
centreline by one bridge: a straight edge from where the vessel enters the
thick vessel's lumen to the nearest point of the centreline, one local radius
long, tagged :data:`IS_ZERO_RESISTANCE` so haemodynamics gives it (near) zero
resistance (:func:`haemodynamics.apply._assign_poiseuille_resistances`). The
radius belongs to the thick vessel: counted as the joining vessel's length,
it would add resistance that is not there.

Graph building leaves a joined vessel running on through the thick vessel's
lumen to wherever skeletonisation joined it: to the centreline, along it, or
onto another vessel's join. Each edge reaching the thick-vessel region is
sampled, and each sample labelled: off the region, on the thick centreline
(within *centreline_tolerance_um* of the tree), or in the region off the
centreline. The off-centreline stretches, joined through their nodes, are
either

- a *graze*, never touching the centreline (a vessel passing through the
  region's margin): vessel, left as it is; or
- a *connector*, touching it. Its samples within one lumen radius of the
  centreline (the segmentation's inscribed radius at the nearest point of the
  tree) are inside the thick vessel and are removed; the rest is the joining
  vessel and stays, and each place it meets a removed stretch gets one
  straight bridge to the nearest point of the graph's thick centreline.

The thick centreline itself is vessel, at its own diameter. A junction on it
that only a removed stretch made is merged away.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

import networkx as nx
import numpy as np
from scipy.ndimage import binary_dilation, generate_binary_structure
from scipy.spatial import cKDTree

from haemolynx.preprocessing.memmap_support import iter_blocks

from ._helpers import (
    calculate_path_length,
    densify_polyline,
    edge_sample_points,
    next_node_id,
    points_inside_mask,
)

logger = logging.getLogger(__name__)

#: Edge attribute marking a bridge: the radius of a thick vessel between its
#: centreline and a vessel opening into it, not vessel material --
#: haemodynamics gives it a negligible resistance instead of computing one
#: from its diameter/length.
IS_ZERO_RESISTANCE = "is_zero_resistance"

#: A sample this close (µm) to the thick centreline tree is on the thick
#: vessel's centreline: graph building moves centrelines by about a micron
#: (smoothing's own limit) after skeletonisation drew the tree.
CENTRELINE_TOLERANCE_UM = 1.5

#: ... or this fraction of the thick vessel's radius there, when that is more:
#: a wide vessel's centreline wanders round the tree by more than a narrow
#: one's (on E14.5 a 12 um vessel's by up to 4 um), and is still its centreline.
CENTRELINE_TOLERANCE_PER_RADIUS = 0.3

#: A bridge's centreline end this close (µm, along the centreline) to a node
#: already there, or to another bridge's, takes that node.
_SNAP_UM = 1.0

#: Edges are sampled this finely (µm) to find where they enter the region and
#: the lumen.
_SAMPLE_STEP_UM = 0.5

#: A stretch shorter than this (2 um) between others is the region's edge or the
#: tree's tolerance flickering along a centreline, not a vessel leaving or
#: entering: it takes its neighbour's label.
_MIN_RUN_UM = 2.0
_MIN_RUN_SAMPLES = int(round(_MIN_RUN_UM / _SAMPLE_STEP_UM))

#: A stretch of an edge near the centreline tree is the thick vessel's own
#: centreline only if it is at least this long (µm): from an end of the edge
#: (shorter, it is the edge's last step into the centreline), or within it (a
#: vessel crossing the thick one passes within the tolerance for about twice
#: its length). A whole edge near the tree always is.
_MIN_CENTRELINE_AT_END_UM = 2.0 * CENTRELINE_TOLERANCE_UM
_MIN_CENTRELINE_WITHIN_UM = 4.0 * CENTRELINE_TOLERANCE_UM

#: How far past a thick vessel's lumen (the segmentation's inscribed radius at
#: the nearest point of its centreline tree) a vessel's centreline may stop
#: and still be drawn on to the thick centreline through the mask (µm):
#: skeletonisation often ends a thin vessel's arm short of the thick region,
#: at its wall, and then nothing reaches the thick centreline.
WALL_JOIN_REACH_UM = 10.0

#: Edge attribute on a vessel drawn on from a thick vessel's wall to its
#: centreline: how much was added, in µm (the part inside the lumen becomes
#: the bridge).
JOINED_AT_WALL_UM = "joined_at_thick_wall_um"

#: A bridge continuing a thick centreline's end, its only other edge, within
#: this cosine of the centreline's own direction there is the thick vessel
#: narrowing into a thin one -- one vessel, a taper -- not a vessel opening
#: into the thick one: no bridge (45 degrees).
TAPER_MIN_COSINE = 0.7

#: How far along a thick centreline's end its direction is read (µm).
TAPER_DIRECTION_UM = 5.0

#: A stretch of the thick centreline tree no graph centreline follows, shorter
#: than this (µm), is the tolerance flickering, not a gap (on E14.5 half the
#: gaps were 2 um or less).
TREE_GAP_MIN_UM = 3.0

#: A gap is filled only where the graph's own way between its two sides is
#: more than this many times as long as the tree's path, or there is none: a
#: graph that already joins them about as directly draws the vessel there,
#: however far off the tree. Fitted on Harvey's grid V (E14.5, 2026-10-15):
#: right on 18 of 20, against 13 for filling none; the drawn path was right
#: at 2.2-32 times (and on one stray at 1.0), wrong at 0.5-2.0 (and once at
#: 4.2).
TREE_FILL_MIN_DETOUR = 2.1

#: Edge attribute on a thick vessel's centreline drawn from the tree where
#: the graph's own strayed off it or missed it (``fill_tree_gaps``).
FROM_TREE = "thick_centreline_from_tree"

#: A dead end is the thick centreline's own when its edge follows the tree
#: (the split's labels, :data:`_MIN_CENTRELINE_AT_END_UM` or more) to within
#: this far (µm) of its tip: the thick vessel as the graph draws it stops
#: there (``extend_thick_ends``).
EXTEND_END_OFF_TREE_UM = 15.0

#: How far (µm) a thick centreline is drawn on along its tree where the tree
#: runs on with no graph beside it, and then on through the segmentation from
#: where the tree ends, looking for the vessel or dead end it reaches. Fitted
#: on Harvey's grid Y (E14.5, both segmentations, 2026-10-15): every way he
#: wanted drawn was 32 um or shorter, the one too long 54 um.
EXTEND_TREE_MAX_UM = 150.0
EXTEND_REACH_UM = 40.0

#: The way on through the segmentation leaves the thick centreline's end within
#: this of its heading (degrees), over its lumen radius and this much more
#: (µm): without it the cheapest way turned back to a vessel beside the thick
#: one (grid W 20, 91 degrees) rather than running on to the one ahead.
EXTEND_START_CONE_DEG = 60.0
EXTEND_START_CONE_UM = 5.0

#: Each micron of the way on through the segmentation costs 1 + this / (its
#: distance to the wall + 0.5 um): the way keeps to the middle of the lumen.
EXTEND_CENTRE_WEIGHT_UM = 4.0

#: Edge attribute on a thick centreline drawn on from where it stopped
#: (``extend_thick_ends``): how much was added, in µm.
EXTENDED_UM = "thick_centreline_extended_um"

#: Edge attribute on the vessel a thick vessel narrows into, where it runs on
#: inside the thick lumen from the thick centreline's end (a taper, left as
#: vessel): the lumen clean-up that follows the split leaves it alone as it
#: does a bridge -- never one of a pair in one lumen, a doubled stretch or
#: the side of a short loop it breaks. Without the mark it took the taper
#: for a second strand and rewired the vessels round it (E14.5, 2026-10-13).
#: A taper is never merged with a vessel at a degree-2 node
#: (:func:`taper_meets_vessel`), so the mark stays on the taper.
THICK_TAPER = "thick_taper"

#: At a fork of the thick centreline, an arm this much narrower than the
#: trunk (its two widest arms) is a branch: the stretch of it inside the
#: trunk's lumen is a bridge, as for any vessel opening into the thick one.
#: Equal arms are a bifurcation of equals, and get none.
FORK_BRANCH_MAX_RADIUS_FRACTION = 0.8

#: Each arm's radius is read over this far (µm) past the fork's own radius,
#: and an arm is a thick-centreline arm when this much of its start is on
#: the tree.
FORK_ARM_UM = 10.0

_OUTSIDE, _CENTRELINE, _INSIDE = 0, 1, 2


def _runs(
    inside_flags: list, *, min_run_length: int = 2, min_run_um: float | None = None, along=None
) -> list[tuple[int, int, Any]]:
    """Contiguous ``(start, end inclusive, value)`` runs over the whole sequence.

    A run shorter than *min_run_length* samples -- or, with *along* (each
    sample's arc length from the first) and *min_run_um*, shorter than that
    in microns -- is a flicker right at a boundary (voxel-grid
    discretisation, not a real crossing) and gets absorbed into a
    neighbouring run rather than becoming its own degenerate edge. Judged
    in microns, because a voxel path densified to half a voxel is sampled
    more often than every 0.5 um: counted in samples, a vessel dipping out
    of the region for 1.5 um came back as a 1.5 um vessel between two
    bridges, a loop inside the thick lumen.
    """
    if not inside_flags:
        return []
    raw: list[list[Any]] = []
    start = 0
    current = inside_flags[0]
    for i in range(1, len(inside_flags)):
        if inside_flags[i] != current:
            raw.append([start, i - 1, current])
            start = i
            current = inside_flags[i]
    raw.append([start, len(inside_flags) - 1, current])

    def long_enough(s: int, e: int) -> bool:
        if min_run_um is not None and along is not None:
            # The crossings lie between samples: the run reaches half a
            # step past its first and last sample, where there is one.
            reach = float(along[e] - along[s])
            if s > 0:
                reach += float(along[s] - along[s - 1]) / 2.0
            if e + 1 < len(inside_flags):
                reach += float(along[e + 1] - along[e]) / 2.0
            return reach >= min_run_um
        return e - s + 1 >= min_run_length

    changed = True
    while changed and len(raw) > 1:
        changed = False
        for i, (s, e, _inside) in enumerate(raw):
            if long_enough(s, e):
                continue
            if i == 0:
                raw[1][0] = s
            else:
                raw[i - 1][1] = e
            del raw[i]
            changed = True
            break
    # Absorbing a short run into its neighbour (above) can leave that
    # neighbour newly adjacent to another run of the same value (the short
    # run used to separate them) -- coalesce those back into one run so a
    # flicker never produces an extra, artificial crossing.
    coalesced: list[list[Any]] = []
    for s, e, inside in raw:
        if coalesced and coalesced[-1][2] == inside:
            coalesced[-1][1] = e
        else:
            coalesced.append([s, e, inside])
    return [(s, e, inside) for s, e, inside in coalesced]


def _labels(points: np.ndarray, on_tree: np.ndarray, in_region: np.ndarray) -> np.ndarray:
    """Each sample of an edge's path labelled outside, on the thick
    centreline, or inside the region off it: flickers under
    :data:`_MIN_RUN_UM` take their neighbours' label, and a stretch near the
    tree is the thick centreline only if long enough (see
    :data:`_MIN_CENTRELINE_AT_END_UM`) -- a whole edge near it always is."""
    labels = np.where(on_tree, _CENTRELINE, np.where(in_region, _INSIDE, _OUTSIDE))
    absorbed = np.empty_like(labels)
    along = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))])
    for s, e, value in _runs(labels.tolist(), min_run_um=_MIN_RUN_UM, along=along):
        absorbed[s : e + 1] = value
    last = len(points) - 1
    for s, e, value in _runs(absorbed.tolist(), min_run_length=1):
        if value != _CENTRELINE or (s == 0 and e == last):
            continue
        needed = _MIN_CENTRELINE_AT_END_UM if (s == 0 or e == last) else _MIN_CENTRELINE_WITHIN_UM
        if along[e] - along[s] < needed:
            absorbed[s : e + 1] = _INSIDE
    return absorbed


def taper_meets_vessel(d1: Mapping[str, Any], d2: Mapping[str, Any]) -> bool:
    """Whether, of two edges meeting at a node, one is a taper
    (:data:`THICK_TAPER`) and the other is not: kept two at a degree-2 node,
    as a bridge is kept from its vessel. Merged, the taper lent the whole
    vessel the lumen clean-up's protection, and each merge after spread it
    further; dropped at the merge, the clean-up took the taper for a second
    strand."""
    return bool(d1.get(THICK_TAPER)) != bool(d2.get(THICK_TAPER))


def thick_vessel_centreline(
    thick_mask: np.ndarray,
    voxel_size_zyx: tuple[float, float, float],
    *,
    use_memmap: bool = False,
    large_stack_tree: bool = False,
) -> np.ndarray:
    """The thick vessel's centreline tree, as skeletonisation drew it.

    A function of the thick-vessel region alone, so a run that loaded its
    region rather than making it finds the same tree again. *use_memmap*
    (the low-RAM option) finds it without a distance map or labels of the
    region's whole box (see :func:`~haemolynx.preprocessing.thick_vessels.skeletonize_edt_ridge`):
    the same tree. *large_stack_tree* draws it as skeletonisation did under
    ``skeleton_thick_vessel_large_stack_tree`` (``pipeline.stages.large_stack_thick_tree``).
    """
    from haemolynx.preprocessing.thick_vessels import skeletonize_edt_ridge

    return np.asarray(
        skeletonize_edt_ridge(
            np.asarray(thick_mask, dtype=bool),
            tuple(voxel_size_zyx),
            use_memmap=use_memmap,
            large_stack_tree=large_stack_tree,
        ),
        dtype=bool,
    )


def thick_vessel_centreline_voxels(
    thick_mask: np.ndarray,
    voxel_size_zyx: tuple[float, float, float],
    *,
    use_memmap: bool = False,
    large_stack_tree: bool = False,
) -> np.ndarray:
    """:func:`thick_vessel_centreline` as its voxels, ``(n, 3)`` in C order
    (``np.argwhere`` of it): all the split and :class:`ThickCentreline`
    read, without a volume to draw the tree in. Skeletonisation hands over
    the tree it drew in this form (``SkeletonisedVolume.thick_vessel_centreline``),
    so a fresh run never draws it twice. *large_stack_tree* as for
    :func:`thick_vessel_centreline`."""
    from haemolynx.preprocessing.thick_vessels import edt_ridge_voxels

    return edt_ridge_voxels(
        np.asarray(thick_mask, dtype=bool),
        tuple(voxel_size_zyx),
        use_memmap=use_memmap,
        large_stack_tree=large_stack_tree,
    )


def _centreline_voxels(centreline: np.ndarray, ndim: int) -> np.ndarray:
    """The tree's voxels in C order, from either form a caller may hold: a
    boolean volume of it, or its voxels already (an ``(n, ndim)`` integer
    array, :func:`thick_vessel_centreline_voxels`)."""
    centreline = np.asarray(centreline)
    if centreline.ndim == 2 and centreline.shape[1] == ndim and np.issubdtype(centreline.dtype, np.integer):
        return centreline
    return np.argwhere(np.asarray(centreline, dtype=bool))


#: How far round its points :func:`_inscribed_radius_um` looks for the wall (µm).
_INSCRIBED_REACH_UM = 60.0


def _inscribed_radius_um(
    points_um: np.ndarray, lumen: np.ndarray, voxel_size_zyx: np.ndarray
) -> np.ndarray:
    """Distance (µm) from each point to the nearest voxel outside *lumen*:
    the radius of the largest ball round it inside the lumen. Inf where the
    lumen has no outside near enough to find (it fills the box searched).

    The wall searched is the outside 6-adjacent to the lumen within the
    points' box, padded by :data:`_INSCRIBED_REACH_UM`. It is found a block of
    the box at a time (``preprocessing.memmap_support.iter_blocks``), each
    block read only for the points within that reach of it, so neither the
    box nor its wall is ever held whole: for a thick-vessel tree spanning a
    stack the box is the stack, and reading it at once took several bytes a
    voxel of it (2026-10-18). A point whose wall is further than the reach is
    then measured against every block. The distances are the same as the box
    read at once: the same wall voxels, each measured the same way.
    """
    if len(points_um) == 0:
        return np.zeros(0)
    vs = np.asarray(voxel_size_zyx, dtype=float)
    points_um = np.asarray(points_um, dtype=float)
    lo_vox = np.floor(points_um.min(axis=0) / vs).astype(int)
    hi_vox = np.ceil(points_um.max(axis=0) / vs).astype(int)
    pad = np.ceil(_INSCRIBED_REACH_UM / vs).astype(int)
    lo = np.maximum(lo_vox - pad, 0)
    hi = np.minimum(hi_vox + pad + 1, np.asarray(lumen.shape))
    distance = np.full(len(points_um), np.inf)
    _lower_to_nearest_wall(points_um, lumen, vs, lo, hi, distance, reach_um=_INSCRIBED_REACH_UM)
    # Exact wherever the wall found is within the reach: a nearer one would
    # lie in a block within the reach, and every such block was read.
    far = np.flatnonzero(distance > _INSCRIBED_REACH_UM)
    if len(far):
        rest = distance[far]
        _lower_to_nearest_wall(points_um[far], lumen, vs, lo, hi, rest, reach_um=np.inf)
        distance[far] = rest
    return distance


def _lower_to_nearest_wall(points_um, lumen, vs, lo, hi, best, *, reach_um) -> None:
    """Lower *best* (each point's distance so far, in place) to its nearest
    wall voxel in the box ``[lo, hi)`` of *lumen*, block by block: a block is
    read only for the points it could bring nearer, within *reach_um* of it.
    Outside the box is outside the lumen, and nothing of it is wall, as for
    the box read whole."""
    six = generate_binary_structure(3, 1)
    for padded, inner, core in iter_blocks(tuple(int(s) for s in hi - lo), halo=1):
        start = np.array([s.start for s in core]) + lo
        last = np.array([s.stop for s in core]) + lo - 1
        # Each point's distance to the block's nearest voxel centre: no wall
        # voxel in it is nearer. The slack keeps rounding from dropping one.
        outside = np.maximum(np.maximum(start * vs - points_um, points_um - last * vs), 0.0)
        to_block = np.sqrt((outside**2).sum(axis=1))
        asking = np.flatnonzero((to_block <= reach_um + 1e-6) & (to_block <= best + 1e-6))
        if not len(asking):
            continue
        window = tuple(slice(int(s.start + o), int(s.stop + o)) for s, o in zip(padded, lo))
        block = np.asarray(lumen[window], dtype=bool)
        wall = (binary_dilation(block, structure=six) & ~block)[inner]
        if not wall.any():
            continue
        tree = cKDTree((np.argwhere(wall) + start) * vs)
        found, _ = tree.query(points_um[asking])
        best[asking] = np.minimum(best[asking], found)


def _tree_radius_um(points_um, lumen, thick, vs) -> np.ndarray:
    """The inscribed radius at each point of the thick centreline tree: in
    *lumen*, else (where the lumen fills the box searched) in the region
    *thick* itself; inf where neither has an outside near enough."""
    radius = _inscribed_radius_um(points_um, lumen, vs)
    unknown = ~np.isfinite(radius)
    if unknown.any() and lumen is not thick:
        radius[unknown] = _inscribed_radius_um(points_um[unknown], thick, vs)
    return radius


class ThickCentreline:
    """The thick vessels' centreline tree, for asking how much of a
    centreline follows it: within the split's own tolerance of the tree
    (:data:`CENTRELINE_TOLERANCE_UM`, or :data:`CENTRELINE_TOLERANCE_PER_RADIUS`
    of the lumen's radius there).

    The lumen clean-up asks it which of two strands in one lumen is the
    thick vessel's own centreline. It runs before the split, when nothing
    marks one yet, and judged by the wall alone it took the strand on the
    tree as often as not (Forclaude.haemorun, E14.5, 2026-10-15: the graph
    followed 95% of the tree after skeletonisation and 83% when built).
    """

    def __init__(self, tree_points: np.ndarray, tolerance_um: np.ndarray, radius_um: np.ndarray | None = None):
        self._points = np.asarray(tree_points, dtype=float).reshape(-1, 3)
        self._tolerance = np.asarray(tolerance_um, dtype=float).reshape(-1)
        self._tree = cKDTree(self._points)
        #: The lumen's inscribed radius at each tree point (inf where none was
        #: found), as :func:`insert_thick_vessel_junction_nodes` measures it:
        #: hand it over (``tree_radius_um``) rather than measure it twice.
        self.radius_um = None if radius_um is None else np.asarray(radius_um, dtype=float).reshape(-1)

    @classmethod
    def from_region(
        cls,
        thick_mask: np.ndarray,
        voxel_size_zyx: tuple[float, float, float],
        *,
        segmentation: np.ndarray | None = None,
        centreline: np.ndarray | None = None,
        centreline_tolerance_um: float = CENTRELINE_TOLERANCE_UM,
    ) -> "ThickCentreline | None":
        """The tree of *thick_mask* (or *centreline*, the tree already
        found: a volume of it or its voxels), with the tolerance
        :func:`insert_thick_vessel_junction_nodes` judges it by. None when
        there is no tree."""
        vs = np.asarray(voxel_size_zyx, dtype=float)
        thick = np.asarray(thick_mask, dtype=bool)
        if not thick.any():
            return None
        tree_voxels = (
            thick_vessel_centreline_voxels(thick, tuple(vs)) if centreline is None
            else _centreline_voxels(centreline, thick.ndim)
        )
        points = tree_voxels * vs
        if len(points) == 0:
            return None
        if segmentation is not None and np.shape(segmentation) != thick.shape:
            segmentation = None
        lumen = thick if segmentation is None else np.asarray(segmentation, dtype=bool)
        measured = _tree_radius_um(points, lumen, thick, vs)
        radius = np.where(np.isfinite(measured), measured, 0.0)
        return cls(
            points, np.maximum(float(centreline_tolerance_um), CENTRELINE_TOLERANCE_PER_RADIUS * radius),
            radius_um=measured,
        )

    def along_um(self, polyline: np.ndarray) -> float:
        """How long a stretch of *polyline* (µm) lies on the tree: its steps,
        sampled every :data:`_SAMPLE_STEP_UM`, with both ends within the
        tolerance of the tree."""
        points = densify_polyline(np.asarray(polyline, dtype=float).reshape(-1, 3), max_step_um=_SAMPLE_STEP_UM)
        if len(points) < 2:
            return 0.0
        distance, nearest = self._tree.query(points)
        on = distance <= self._tolerance[nearest]
        steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
        return float(steps[on[:-1] & on[1:]].sum())


def _decide_what_stays(sampled, inside_runs, find, bridged, through=frozenset()):
    """Set each edge's ``keep``: everything but the stretches of the *bridged*
    connectors inside the thick-vessel region, apart from the runs *through*
    (the thick vessel's own path). Returns, per sample dropped, its
    connector."""
    for e in sampled:
        e.keep = np.ones(len(e.points), dtype=bool)
    dropped_from: dict[tuple[int, int], Any] = {}
    for r, (i, s, t) in enumerate(inside_runs):
        root = find(("run", r))
        if root not in bridged or r in through:
            continue
        sampled[i].keep[s : t + 1] = False
        for k in range(s, t + 1):
            dropped_from[(i, k)] = root
    # A sample left alone between removed ones is not a vessel.
    for i, e in enumerate(sampled):
        for s, t, value in _runs(e.keep.tolist(), min_run_length=1):
            if value and s == t and 0 < s < len(e.points) - 1 and (i, s - 1) in dropped_from:
                e.keep[s] = False
                dropped_from[(i, s)] = dropped_from[(i, s - 1)]
    return dropped_from


def _thick_paths_through_connectors(sampled, inside_runs, find, connectors, contacts):
    """The inside runs on the shortest way, within a connector, between two
    places it meets the thick centreline: the thick vessel's own path where
    its centreline strays off the tree, not a vessel opening into it."""
    runs_of: dict[Any, list[int]] = {}
    for r in range(len(inside_runs)):
        runs_of.setdefault(find(("run", r)), []).append(r)
    through: set[int] = set()
    for root in connectors:
        places = {
            tuple(np.round(sampled[i].points[k], 0)) for i, k in contacts.get(root, ())
        }
        if len(places) < 2:
            continue
        H = nx.Graph()
        met = set()
        for r in runs_of.get(root, ()):
            i, s, t = inside_runs[r]
            e = sampled[i]
            last = len(e.points) - 1
            ends = []
            for k, beyond in ((s, s - 1), (t, t + 1)):
                if k in (0, last):
                    end = ("node", e.u if k == 0 else e.v)
                else:
                    end = ("sample", i, beyond)
                    if e.labels[beyond] == _CENTRELINE:
                        met.add(end)
                ends.append(end)
            length = float(np.sum(np.linalg.norm(np.diff(e.points[s : t + 1], axis=0), axis=1)))
            if H.has_edge(*ends) and H.edges[ends]["weight"] <= length:
                continue
            H.add_edge(*ends, weight=length, run=r)
        for i, k in contacts.get(root, ()):
            node = sampled[i].u if k == 0 else sampled[i].v if k == len(sampled[i].points) - 1 else None
            if node is not None and ("node", node) in H:
                met.add(("node", node))
        met = sorted(met, key=str)
        for a_index, a in enumerate(met):
            for b in met[a_index + 1 :]:
                try:
                    path = nx.shortest_path(H, a, b, weight="weight")
                except nx.NetworkXNoPath:
                    continue
                through.update(H.edges[x, y]["run"] for x, y in zip(path[:-1], path[1:]))
    return through


def _bridge_origins(sampled, edges_at, node_on_centreline, result):
    """Where a kept vessel outside the thick-vessel region meets a removed
    stretch: ``(origin, the dropped samples beside it)`` per place, each place
    once. Only from outside: what stays inside the region is the thick
    vessel's own path, which goes on as it is."""
    found: dict[tuple[str, Any], set] = {}
    for i, e in enumerate(sampled):
        if not e.in_scope:
            continue
        n = len(e.points)
        for k in range(n):
            if not e.keep[k] or e.labels[k] != _OUTSIDE:
                continue
            beside = [(i, j) for j in (k - 1, k + 1) if 0 <= j < n and not e.keep[j]]
            if not beside:
                continue
            origin = ("node", e.u) if k == 0 else ("node", e.v) if k == n - 1 else ("sample", (i, k))
            found.setdefault(origin, set()).update(beside)
    for node, ends in edges_at.items():
        dropped = [(i, end) for i, end in ends if sampled[i].in_scope and not sampled[i].keep[end]]
        kept_outside = result.degree(node) > len(ends) or any(
            sampled[i].keep[end] and sampled[i].labels[end] == _OUTSIDE for i, end in ends
        )
        kept_inside = node in node_on_centreline or any(
            sampled[i].keep[end] and sampled[i].labels[end] != _OUTSIDE for i, end in ends
        )
        if dropped and kept_outside and not kept_inside:
            found.setdefault(("node", node), set()).update(dropped)
    return list(found.items())


@dataclass
class _Edge:
    u: Any
    v: Any
    key: Any
    data: dict
    points: np.ndarray
    labels: np.ndarray
    tree_distance: np.ndarray
    on_tree: np.ndarray
    in_scope: bool
    keep: np.ndarray | None = None
    anchors: set = field(default_factory=set)


def insert_thick_vessel_junction_nodes(
    G: nx.MultiGraph,
    thick_mask: np.ndarray,
    *,
    voxel_size_zyx: tuple[float, float, float],
    edges: Any = None,
    segmentation: np.ndarray | None = None,
    centreline: np.ndarray | None = None,
    centreline_tolerance_um: float = CENTRELINE_TOLERANCE_UM,
    join_dead_ends_at_walls: bool = True,
    bridge_forks: bool = True,
    fill_tree_gaps: bool = True,
    fill_attributes: Mapping[str, Any] | None = None,
    extend_thick_ends: bool = False,
    max_background_gap_um: float | None = None,
    min_mask_fraction: float | None = None,
    max_short_loop_um: float | None = None,
    short_loop_keep_hole_um: float | None = None,
    min_stub_length: float = 10.0,
    min_stub_length_radius_multiple: float = 3.0,
    mask_support=None,
    tree_radius_um: np.ndarray | None = None,
) -> nx.MultiGraph:
    """*G* with a one-radius bridge where each vessel opens into a thick vessel.

    *thick_mask* is the thick-vessel region skeletonisation found;
    *centreline* the tree it drew through it (a volume of it, or its voxels:
    :func:`thick_vessel_centreline_voxels`), found again from the region
    when not given. *segmentation* gives
    the thick vessel's lumen, whose inscribed radius at the centreline is
    where a joining vessel enters it; without one the region stands in,
    which puts that wall about the region's margin further out.

    Three more cases, by Harvey's examples (2026-10-13, PLAN_bridging.md):

    - *join_dead_ends_at_walls* (with a *segmentation*): a dead end stopping
      within :data:`WALL_JOIN_REACH_UM` of a thick vessel's lumen, heading
      for it (the gap bridges' turn and length rule), is first drawn on
      through the mask to the nearest point of the thick centreline -- a
      route crossing no more than *max_background_gap_um* of background, as
      every mask-supported bridge (``preprocessing.MaskSupport``), not
      beside a vessel in the same lumen, closing no loop inside one and no
      loop shorter than *max_short_loop_um* (the lumen clean-up's
      ``remove_short_loops_under_um``; ``lumen_loops.DEFAULT_SHORT_LOOP_UM``
      when None, 0 for none) unless it runs round a hole wider than
      *short_loop_keep_hole_um* (the clean-up's
      ``short_loop_keep_hole_over_um``; ``lumen_loops.DEFAULT_SHORT_LOOP_KEEP_HOLE_UM``
      when None) -- and then bridged like any vessel reaching it. The
      clean-up breaks every short loop but one through a bridge or round a
      wide hole, so a join would make the loops it exists to break.
    - A connector whose bridge would continue a thick centreline's end along
      its own direction (:data:`TAPER_MIN_COSINE`), that end having no other
      edge, is the thick vessel narrowing into a thin one: left as vessel.
    - *bridge_forks*: where the thick centreline forks and one arm is
      narrower than the other two (:data:`FORK_BRANCH_MAX_RADIUS_FRACTION`),
      the stretch of that branch inside the trunk's lumen becomes a bridge.

    And with *fill_tree_gaps* (on by default since 2026-10-15, PLAN_bridging.md 6a), first:
    where no graph centreline follows the tree for :data:`TREE_GAP_MIN_UM` or
    more between two places one does, and the graph's own way between them
    is more than :data:`TREE_FILL_MIN_DETOUR` times as long or missing, the
    tree's own path is added as the thick vessel's centreline
    (:data:`FROM_TREE`, :func:`_fill_tree_gaps`). Every vessel reaching the
    thick vessel there can then be bridged to it. Each stretch drawn also
    takes *fill_attributes* (post-processing marks it edited, so it is
    measured as a vessel drawn by hand is).

    With *extend_thick_ends* (and a *segmentation*; off by default, being
    calibrated on Harvey's grid Y), after the fill: where the graph's thick
    centreline stops at a dead end, it is drawn on -- along the tree where the
    tree runs on with no graph beside it, then through the segmentation past
    where the tree ends -- to the first vessel or dead end it reaches, unless
    the image's face comes first (:func:`_extend_thick_centreline_ends`). At
    grid W's 16, 20 and 23
    the thick region, and its tree, ended where the graph did, while the
    vessel ran on.

    Last, a vessel beyond a bridge that ends blind inside the image goes
    with its bridge -- a bridge joins a thick vessel to a vessel that goes on
    -- when the stub prune would take it as a stub too: shorter than
    *min_stub_length* and than *min_stub_length_radius_multiple* times the
    thick vessel's lumen radius where the bridge meets its centreline
    (:func:`_remove_blind_vessels_beyond_bridges`).

    With *edges* (``(u, v, key)``, either way round) only those are rebuilt,
    and one already a bridge is left as it is; every other edge stays as it
    was, except that a bridge may join it where it is the thick centreline,
    and the fill draws only the gaps one of *edges* reaches, as far as a
    bridge from it looks for the thick centreline. That is how a hand-edited
    vessel gets its bridge without the network's existing ones, or its thick
    centreline elsewhere, being made again.

    Two things a caller may have already, so they are not made again (the
    result is the same): *mask_support*, a ``preprocessing.MaskSupport`` over
    *segmentation*, which the join at the walls uses (with its own limits:
    ``MaskSupport.limited_to``); and *tree_radius_um*, the inscribed radius
    at each point of *centreline* in *segmentation* (``ThickCentreline.radius_um``).

    Returns a new graph; *G* is left as it is.
    """
    vs = np.asarray(voxel_size_zyx, dtype=float)
    thick = np.asarray(thick_mask, dtype=bool)
    result = G.copy()
    if not thick.any():
        return result
    tree_voxels = (
        thick_vessel_centreline_voxels(thick, voxel_size_zyx) if centreline is None
        else _centreline_voxels(centreline, thick.ndim)
    )
    tree_points = tree_voxels * vs
    if len(tree_points) == 0:
        logger.warning(
            "Thick-vessel bridges: the thick-vessel region has no centreline; no bridges made."
        )
        return result
    tree = cKDTree(tree_points)
    if segmentation is not None and np.shape(segmentation) != thick.shape:
        logger.warning(
            "Thick-vessel bridges: the segmentation (%s) is not the shape of the thick-vessel "
            "region (%s); the region stands in for the lumen.",
            np.shape(segmentation), thick.shape,
        )
        segmentation = None
    lumen = thick if segmentation is None else np.asarray(segmentation, dtype=bool)
    if segmentation is not None and tree_radius_um is not None and np.shape(tree_radius_um) == (len(tree_points),):
        tree_radius = np.array(tree_radius_um, dtype=float)
    else:
        tree_radius = _tree_radius_um(tree_points, lumen, thick, vs)
    finite = np.where(np.isfinite(tree_radius), tree_radius, 0.0)
    tree_tolerance = np.maximum(
        float(centreline_tolerance_um), CENTRELINE_TOLERANCE_PER_RADIUS * finite
    )

    def near_tree(points):
        distance, nearest = tree.query(points)
        return distance, distance <= tree_tolerance[nearest]
    if segmentation is None:
        logger.info(
            "Thick-vessel bridges: no segmentation given; the thick-vessel region stands "
            "in for the lumen, so each bridge reaches about the region's margin past the wall."
        )

    chosen = (
        None if edges is None else {(frozenset((u, v)), key) for u, v, key in edges}
    )
    if fill_tree_gaps:
        # A post-processing split fills only beside the edges it rebuilds:
        # filling everywhere redrew the network's thick centreline far from
        # any edit, a stretch a hand edit had taken out included.
        named = None
        if chosen is not None:
            named = [
                densify_polyline(_polyline_from(result, u, v, d), max_step_um=_SAMPLE_STEP_UM)
                for u, v, k, d in result.edges(keys=True, data=True)
                if (frozenset((u, v)), k) in chosen and not d.get(IS_ZERO_RESISTANCE)
                and "pos" in result.nodes[u] and "pos" in result.nodes[v]
            ]
            named = np.vstack(named) if named else np.empty((0, 3))
        filled = _fill_tree_gaps(
            result, tree_voxels, vs, tree_points, tree_tolerance, finite,
            near=named, attributes=fill_attributes,
        )
        if filled:
            logger.info(
                "Thick-vessel bridges: %d stretch(es) of the thick centreline tree the graph "
                "did not join drawn from the tree.",
                filled,
            )
    if extend_thick_ends and segmentation is not None:
        extended = _extend_thick_centreline_ends(
            result, tree_voxels, vs, tree_points, tree_tolerance, finite, lumen, thick, chosen,
            attributes=fill_attributes,
        )
        if extended:
            logger.info(
                "Thick-vessel bridges: %d thick centreline(s) stopping at a dead end drawn on to "
                "the vessel or dead end ahead.",
                extended,
            )
    if join_dead_ends_at_walls and segmentation is not None:
        _join_dead_ends_at_walls(
            result, near_tree, tree, tree_radius, lumen, thick, vs, chosen,
            max_background_gap_um=max_background_gap_um, min_mask_fraction=min_mask_fraction,
            max_short_loop_um=max_short_loop_um, short_loop_keep_hole_um=short_loop_keep_hole_um,
            support=mask_support,
        )
    node_pos = {
        n: np.asarray(d["pos"], dtype=float) for n, d in result.nodes(data=True) if "pos" in d
    }

    # 1. Every edge sampled and labelled.
    sampled: list[_Edge] = []
    for u, v, key, data in result.edges(keys=True, data=True):
        if u not in node_pos or v not in node_pos:
            continue
        points = densify_polyline(
            edge_sample_points(u, v, data, node_pos), max_step_um=_SAMPLE_STEP_UM
        )
        if len(points) < 2:
            continue
        tree_distance, on_tree = near_tree(points)
        in_region = points_inside_mask(points, thick, voxel_size_zyx=tuple(vs))
        if not (on_tree.any() or in_region.any()):
            continue
        absorbed = _labels(points, on_tree, in_region)
        is_bridge = bool(data.get(IS_ZERO_RESISTANCE))
        in_scope = not is_bridge and (
            chosen is None or (frozenset((u, v)), key) in chosen
        )
        sampled.append(_Edge(u, v, key, data, points, absorbed, tree_distance, on_tree, in_scope))

    # 2. The off-centreline stretches of the edges being rebuilt, joined
    # through their nodes into grazes and connectors.
    parent: dict[Any, Any] = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        parent[find(a)] = find(b)

    edges_at: dict[Any, list[tuple[int, int]]] = {}  # node -> (edge index, its end sample)
    for i, e in enumerate(sampled):
        edges_at.setdefault(e.u, []).append((i, 0))
        edges_at.setdefault(e.v, []).append((i, len(e.points) - 1))
    node_on_centreline = {
        n for n in edges_at if near_tree(node_pos[n][None, :])[1][0]
    }

    def label_at(n, i, end):
        return sampled[i].labels[end]

    def node_meets_outside(n) -> bool:
        """Some edge leaves *n* off the region, or one not sampled at all."""
        if result.degree(n) > len(edges_at.get(n, ())):
            return True
        return any(label_at(n, i, end) == _OUTSIDE for i, end in edges_at.get(n, ()))

    inside_runs: list[tuple[int, int, int]] = []  # (edge index, start, end)
    for i, e in enumerate(sampled):
        if not e.in_scope:
            continue
        for s, t, value in _runs(e.labels.tolist(), min_run_length=1):
            if value != _INSIDE:
                continue
            run = ("run", len(inside_runs))
            inside_runs.append((i, s, t))
            find(run)
            if s == 0:
                union(run, ("node", e.u))
            if t == len(e.points) - 1:
                union(run, ("node", e.v))

    touches_centreline: set = set()
    leads_out: set = set()
    for r, (i, s, t) in enumerate(inside_runs):
        e = sampled[i]
        last = len(e.points) - 1
        if (s > 0 and e.labels[s - 1] == _OUTSIDE) or (t < last and e.labels[t + 1] == _OUTSIDE) or any(
            (s == end or t == end) and node_meets_outside(node)
            for end, node in ((0, e.u), (last, e.v))
        ):
            leads_out.add(find(("run", r)))
    for r, (i, s, t) in enumerate(inside_runs):
        e = sampled[i]
        root = find(("run", r))
        if (
            (s > 0 and e.labels[s - 1] == _CENTRELINE)
            or (t < len(e.points) - 1 and e.labels[t + 1] == _CENTRELINE)
            or e.on_tree[s : t + 1].any()
        ):
            touches_centreline.add(root)
        for end, node in ((0, e.u), (len(e.points) - 1, e.v)):
            if (s == end or t == end) and (
                node in node_on_centreline
                or any(label_at(node, j, k) == _CENTRELINE for j, k in edges_at[node])
            ):
                touches_centreline.add(root)

    # A stretch that comes within one lumen radius of the centreline is
    # inside the thick vessel, opening into it, whether or not it meets the
    # centreline; one that never does only skims the region's edge.
    enters_lumen: set = set()
    for r, (i, s, t) in enumerate(inside_runs):
        distance, nearest = tree.query(sampled[i].points[s : t + 1])
        if (distance <= tree_radius[nearest]).any():
            enters_lumen.add(find(("run", r)))
    # A stretch that never leads out of the region is the thick vessel's own
    # centreline straying from the tree (smoothing and collapse move it), or
    # a second strand along it: vessel, as it is.
    touches_centreline = (touches_centreline | enters_lumen) & leads_out

    # 3. What stays. Outside and on the centreline: everything. Off the
    # centreline: a graze all of it; a connector what lies beyond one lumen
    # radius of the centreline and still reaches the rest of the network.
    # 4. Where a kept vessel meets a removed stretch: a bridge, to the
    # nearest point of the graph's own thick centreline.
    # 5. Rebuild each edge as the runs it keeps, cut where bridges leave it
    # and where they join the thick centreline.
    # A connector whose removal would cut anything off -- no centreline near
    # enough to bridge to, or a piece of the network reached only through
    # it -- is left as it was, and the rest decided again without it.
    connectors_of_edge: dict[int, set] = {}
    contacts: dict[Any, list[tuple[int, int]]] = {}
    for r, (i, s, t) in enumerate(inside_runs):
        root = find(("run", r))
        connectors_of_edge.setdefault(i, set()).add(root)
        e = sampled[i]
        last = len(e.points) - 1
        if s > 0 and e.labels[s - 1] == _CENTRELINE:
            contacts.setdefault(root, []).append((i, s - 1))
        if t < last and e.labels[t + 1] == _CENTRELINE:
            contacts.setdefault(root, []).append((i, t + 1))
        for end, node in ((0, e.u), (last, e.v)):
            if s == end or t == end:
                contacts.setdefault(root, []).extend(
                    (j, k) for j, k in edges_at[node]
                    if j != i and sampled[j].labels[k] == _CENTRELINE
                    and not sampled[j].data.get(IS_ZERO_RESISTANCE)
                )
    through = _thick_paths_through_connectors(
        sampled, inside_runs, find, touches_centreline, contacts
    )
    for r in through:
        i, s, t = inside_runs[r]
        contacts.setdefault(find(("run", r)), []).extend((i, k) for k in range(s, t + 1))
    candidates = [
        (i, k)
        for i, e in enumerate(sampled)
        if not e.data.get(IS_ZERO_RESISTANCE)
        for k in np.flatnonzero(e.labels == _CENTRELINE)
    ]
    cand_index = np.asarray(candidates, dtype=int).reshape(-1, 2)
    cand_tree = (
        cKDTree(np.asarray([sampled[i].points[k] for i, k in candidates])) if candidates else None
    )
    restored: set = set()
    # Inside stretches the thick vessel narrows into: kept as vessel, though
    # the connector they belong to may still bridge its other stretches.
    tapers: set[int] = set()
    no_centreline_near = 0
    for _attempt in range(len(touches_centreline) + len(inside_runs) + 2):
        bridges, failed, runs = _choose_bridges(
            sampled, inside_runs, find, touches_centreline - restored, tree, tree_radius,
            edges_at, node_on_centreline, node_pos, result,
            cand_tree, cand_index, contacts, through | tapers,
        )
        if failed - restored:
            no_centreline_near += len(failed - restored)
            restored |= failed
            continue
        built, new_node_at, stats, bridge_ends = _rebuild(
            G, result, sampled, bridges, node_pos,
            tapered=[inside_runs[r] for r in tapers],
        )
        cut_off = _cut_off_edges(G, built, edges_at, new_node_at)
        blamed = set().union(*(connectors_of_edge.get(i, set()) for i in cut_off)) & (
            touches_centreline - restored
        )
        # A bridge carrying a thick centreline's end on along its own line is
        # the thick vessel narrowing: the stretch it replaced stays vessel.
        continuing = set().union(*(runs[b] for b in _taper_bridges(built, bridge_ends))) - tapers
        if not blamed and not continuing:
            break
        tapers |= continuing
        restored |= blamed
    def stub_length_um(inner_pos) -> float:
        # The prune's threshold, read with the thick vessel's lumen radius
        # at the tree point nearest the bridge's inner end.
        radius = float(finite[tree.query(np.asarray(inner_pos, dtype=float))[1]])
        return max(float(min_stub_length), float(min_stub_length_radius_multiple) * radius)

    blind, blind_um, blind_kept = _remove_blind_vessels_beyond_bridges(
        built, G, tree, extent_um=(np.asarray(thick.shape) - 1) * vs, stub_length_um=stub_length_um,
    )
    if blind or blind_kept:
        logger.info(
            "Thick-vessel bridges: %d vessel(s) beyond a bridge ending blind inside the image "
            "removed with their bridge (%.0f um), as short as a stub; %d longer kept.",
            blind, blind_um, blind_kept,
        )
    if tapers:
        logger.info(
            "Thick-vessel bridges: %d vessel(s) carrying on from a thick centreline's end along "
            "its line left as vessel: the thick vessel narrowing, not a vessel opening into it.",
            len(tapers),
        )
    cut_off_kept = len(restored) - no_centreline_near
    if no_centreline_near or cut_off_kept:
        logger.warning(
            "Thick-vessel bridges: %d connector(s) left as they were: %d with no thick "
            "centreline in the graph near enough to bridge to, %d the only way to a piece "
            "of the network.",
            no_centreline_near + cut_off_kept, no_centreline_near, cut_off_kept,
        )
    if bridge_forks:
        forks = _bridge_thick_forks(built, near_tree, lumen, vs, chosen)
        if forks:
            stats["bridges"] += forks
            logger.info(
                "Thick-vessel bridges: %d branch(es) of a thick centreline bridged from where "
                "they leave the trunk's lumen to the fork.",
                forks,
            )
    removed_um = sum(
        float(np.sum(np.linalg.norm(np.diff(e.points, axis=0), axis=1)[~(e.keep[:-1] & e.keep[1:])]))
        for e in sampled
        if e.keep is not None
    )
    grazes = len({find(("run", r)) for r in range(len(inside_runs))} - touches_centreline)
    logger.info(
        "Thick-vessel bridges: %d bridge(s), one radius each; %.0f um of centreline inside "
        "thick vessels' lumens replaced; %d stretch(es) through the region's margin left as "
        "vessel; %d junction(s) only a removed stretch made merged away; %d edges rebuilt "
        "as %d.",
        stats["bridges"], removed_um, grazes, stats["merged"], stats["removed"], stats["rebuilt"],
    )
    return built


def _bridge_reach_um(radius):
    """How far a bridge looks for the thick centreline from where its vessel
    meets the thick-vessel region, given the lumen radius there: the radius,
    the margin, and an oblique way in."""
    return 2.0 * radius + 5.0


def _choose_bridges(
    sampled, inside_runs, find, bridged, tree, tree_radius, edges_at, node_on_centreline,
    node_pos, graph, cand_tree, cand_index, contacts, through,
):
    """Decide what stays (``keep``) and each bridge ``(origin, (edge, sample),
    attributes)``; the connectors that cannot have one; and, per bridge, the
    inside runs (indices into *inside_runs*) it replaces beside its origin.

    A bridge goes to where its connector met the thick centreline, so the
    vessel joins the thick one where skeletonisation joined it; failing that
    (out of reach), to the nearest point of the centreline."""
    dropped_from = _decide_what_stays(sampled, inside_runs, find, bridged, through)
    bridges, failed, runs = [], set(), []
    run_of = {(i, k): r for r, (i, s, t) in enumerate(inside_runs) for k in range(s, t + 1)}
    for origin, beside in _bridge_origins(sampled, edges_at, node_on_centreline, graph):
        if origin[0] == "node":
            where = node_pos[origin[1]]
            source = next(
                (sampled[i] for i, _end in edges_at.get(origin[1], ()) if sampled[i].in_scope),
                None,
            )
        else:
            i, k = origin[1]
            where = sampled[i].points[k]
            source = sampled[i]
        reach = _bridge_reach_um(float(tree_radius[tree.query(where)[1]]))
        met = [
            (float(np.linalg.norm(sampled[i].points[k] - where)), (i, k))
            for root in {dropped_from[c] for c in beside if c in dropped_from}
            for i, k in contacts.get(root, ())
        ]
        met = [m for m in met if m[0] <= reach]
        if met:
            target = min(met)[1]
        elif cand_tree is not None and cand_tree.query(where)[0] <= reach:
            target = tuple(int(x) for x in cand_index[cand_tree.query(where)[1]])
        else:
            failed |= {dropped_from[c] for c in beside if c in dropped_from}
            continue
        bridges.append((origin, target, dict(source.data) if source is not None else {}))
        runs.append({run_of[c] for c in beside if c in run_of})
    return bridges, failed, runs


def _rebuild(G, base, sampled, bridges, node_pos, tapered=()):
    """A copy of *base* with each sampled edge rebuilt as the runs it keeps,
    and the bridges added; with, per bridge, its ``(vessel end, centreline
    end)`` nodes (None for one not drawn). Every edge holding a stretch of
    *tapered* (``(edge, start, end)`` inside runs kept as vessel) is marked
    :data:`THICK_TAPER`."""
    result = base.copy()
    tapered_of: dict[int, list[tuple[int, int]]] = {}
    for i, s, t in tapered:
        tapered_of.setdefault(i, []).append((s, t))

    def holds_taper(i: int, a: int, b: int) -> bool:
        return any(s <= b and a <= t for s, t in tapered_of.get(i, ()))
    node_pos = dict(node_pos)
    for e in sampled:
        e.anchors = set()
    for _origin, (i, k), _data in bridges:
        sampled[i].anchors.add(k)
    reserved_ids: set[Any] = set(result.nodes)
    new_node_at: dict[tuple[int, int], Any] = {}

    def node_for(i: int, k: int) -> Any:
        e = sampled[i]
        if k == 0:
            return e.u
        if k == len(e.points) - 1:
            return e.v
        if (i, k) not in new_node_at:
            node_id = next_node_id(result, reserved_ids)
            reserved_ids.add(node_id)
            result.add_node(node_id, pos=tuple(float(c) for c in e.points[k]))
            node_pos[node_id] = np.asarray(e.points[k], dtype=float)
            new_node_at[(i, k)] = node_id
        return new_node_at[(i, k)]

    snapped: dict[tuple[int, int], int] = {}
    for i, e in enumerate(sampled):
        if not e.anchors:
            continue
        along = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(e.points, axis=0), axis=1))])
        kept_cuts = [0, len(e.points) - 1]
        for k in sorted(e.anchors):
            near = min(kept_cuts, key=lambda c: abs(along[c] - along[k]))
            if abs(along[near] - along[k]) <= _SNAP_UM:
                snapped[(i, k)] = near
            else:
                kept_cuts.append(k)
                snapped[(i, k)] = k

    touched_nodes: set[Any] = set()
    stats = {"removed": 0, "rebuilt": 0, "bridges": 0, "merged": 0}
    for i, e in enumerate(sampled):
        cuts = {snapped[(i, k)] for k in e.anchors} - {0, len(e.points) - 1}
        if e.keep.all() and not cuts:
            if holds_taper(i, 0, len(e.points) - 1):
                result.edges[e.u, e.v, e.key][THICK_TAPER] = True
            continue
        result.remove_edge(e.u, e.v, e.key)
        stats["removed"] += 1
        touched_nodes.update((e.u, e.v))
        for s, t, value in _runs(e.keep.tolist(), min_run_length=1):
            if not value or t <= s:
                continue
            bounds = [s, *sorted(c for c in cuts if s < c < t), t]
            for a, b in zip(bounds[:-1], bounds[1:]):
                attrs = {
                    key: value
                    for key, value in e.data.items()
                    if key not in ("voxels", "length", IS_ZERO_RESISTANCE)
                }
                points = [tuple(float(c) for c in row) for row in e.points[a : b + 1]]
                attrs["voxels"] = points
                attrs["length"] = float(calculate_path_length(points))
                if holds_taper(i, a, b):
                    attrs[THICK_TAPER] = True
                result.add_edge(node_for(i, a), node_for(i, b), **attrs)
                stats["rebuilt"] += 1

    bridge_ends: list[tuple[Any, Any] | None] = []
    for origin, (i, k), source_data in bridges:
        start = origin[1] if origin[0] == "node" else node_for(*origin[1])
        end = node_for(i, snapped[(i, k)])
        if start == end:
            bridge_ends.append(None)
            continue
        bridge_ends.append((start, end))
        a, b = node_pos[start], node_pos[end]
        attrs = {
            key: value
            for key, value in source_data.items()
            if key not in ("voxels", "length", IS_ZERO_RESISTANCE)
        }
        attrs["voxels"] = [tuple(float(c) for c in a), tuple(float(c) for c in b)]
        attrs["length"] = float(np.linalg.norm(b - a))
        attrs[IS_ZERO_RESISTANCE] = True
        result.add_edge(start, end, **attrs)
        stats["bridges"] += 1

    # A node no edge reaches any more goes; a junction only a removed stretch
    # made, two vessels left meeting there, is one vessel again.
    from .degree2 import create_trivial_merged_edge

    for node in touched_nodes:
        if not result.has_node(node):
            continue
        degree = result.degree(node)
        if degree == 0:
            result.remove_node(node)
        elif degree == 2 and base.degree(node) != 2:
            incident = list(result.edges(node, keys=True, data=True))
            (_n, n1, _k1, d1), (_m, n2, _k2, d2) = incident
            if n1 == n2 or n1 == node or n2 == node:
                continue
            if d1.get(IS_ZERO_RESISTANCE) or d2.get(IS_ZERO_RESISTANCE) or taper_meets_vessel(d1, d2):
                continue
            merged_attrs = create_trivial_merged_edge(d1, d2, result.nodes[node].get("pos"))
            result.remove_node(node)
            result.add_edge(n1, n2, **merged_attrs)
            stats["merged"] += 1
    return result, new_node_at, stats, bridge_ends


def _polyline_from(G, node, other, data) -> np.ndarray:
    """An edge's path, starting at *node*."""
    pos = {n: np.asarray(G.nodes[n]["pos"], dtype=float) for n in (node, other)}
    return np.asarray(edge_sample_points(node, other, data, pos), dtype=float)


def _taper_bridges(G, bridge_ends) -> set[int]:
    """The bridges (indices into *bridge_ends*, each ``(vessel end,
    centreline end)`` or None) that carry a thick centreline's end on along
    its own line: at a centreline end with one vessel edge, the bridge
    leaving most nearly along that edge's direction there, if within
    :data:`TAPER_MIN_COSINE` of it. At most one per end -- the thick vessel
    narrows into one vessel; any other leaving the same end opens into it."""
    from .cartwheel_guard import _point_along_polyline

    at_end: dict[Any, list[tuple[int, Any]]] = {}
    for b, ends in enumerate(bridge_ends):
        if ends is not None and G.has_node(ends[0]) and G.has_node(ends[1]):
            at_end.setdefault(ends[1], []).append((b, ends[0]))
    tapers: set[int] = set()
    for end, leaving in at_end.items():
        vessels = [
            (a, b, d) for a, b, d in G.edges(end, data=True) if not d.get(IS_ZERO_RESISTANCE)
        ]
        if len(vessels) != 1:
            continue
        a, b, data = vessels[0]
        other = b if a == end else a
        if other == end:
            continue
        end_pos = np.asarray(G.nodes[end]["pos"], dtype=float)
        outward = end_pos - _point_along_polyline(_polyline_from(G, end, other, data), TAPER_DIRECTION_UM)
        best, best_cos = None, TAPER_MIN_COSINE
        for index, start in leaving:
            onward = np.asarray(G.nodes[start]["pos"], dtype=float) - end_pos
            norms = float(np.linalg.norm(outward) * np.linalg.norm(onward))
            if norms > 0.0 and float(outward @ onward) / norms >= best_cos:
                best, best_cos = index, float(outward @ onward) / norms
        if best is not None:
            tapers.add(best)
    return tapers


def _fill_tree_gaps(G, tree_voxels, vs, tree_points, tree_tolerance, tree_radius, *, near=None, attributes=None):
    """Draw the thick centreline tree into *G*, in place, wherever no graph
    centreline follows it. Returns how many stretches were added. With
    *near* (points, um), only a gap one of them reaches as a bridge would
    (:func:`_bridge_reach_um` of the lumen radius at a gap voxel); each
    stretch drawn also takes *attributes*.

    A gap is a run of tree voxels no non-bridge graph centreline comes within
    the tree's tolerance of, :data:`TREE_GAP_MIN_UM` or longer, with tree
    followed by the graph on at least two sides: the graph's thick
    centreline strays off the tree or stops there and starts again. Only
    where the graph's own way between two sides is more than
    :data:`TREE_FILL_MIN_DETOUR` times the tree's path, or there is none:
    where the graph already joins them, its strand is the vessel. Each
    side becomes a node on the graph edge that follows the tree there
    (``edit.insert_node_on_edge``); two sides are joined by the tree's path
    between them, more by the tree's paths to the gap's widest voxel. A gap
    at a tree end, followed on one side only, is left: the graph's thick
    vessel ends there, and the taper and wall rules decide what carries on.

    Nothing of the graph's own is taken out: where a strand of it crosses
    the gap about as directly as the tree, the gap is not filled (on grid V,
    one such strand was to be replaced and one kept, and no measure told
    them apart), and a long way round stays the vessels it is.
    """
    from scipy.spatial import cKDTree

    from .edit import insert_node_on_edge

    vs = np.asarray(vs, dtype=float)
    tree_vox = np.asarray(tree_voxels)  # the tree's voxels, C order
    if len(tree_vox) < 2:
        return 0
    reaching = None
    if near is not None:
        near = np.asarray(near, dtype=float).reshape(-1, 3)
        if len(near) == 0:
            return 0
        reaching = cKDTree(near)

    # Each edge's samples, kept while the edge, its path and its nodes'
    # positions are the very objects they were: every gap filled had the
    # whole graph sampled again, 2.8 s a time on E14.5, though a fill
    # changes a few edges.
    sampled: dict = {}

    def graph_samples():
        points, owners, arcs = [], [], []
        for u, v, key, data in G.edges(keys=True, data=True):
            if data.get(IS_ZERO_RESISTANCE) or "pos" not in G.nodes[u] or "pos" not in G.nodes[v]:
                continue
            made = (data, data.get("voxels"), G.nodes[u]["pos"], G.nodes[v]["pos"])
            hit = sampled.get((u, v, key))
            if hit is None or any(a is not b for a, b in zip(hit[0], made)):
                path = densify_polyline(_polyline_from(G, u, v, data), max_step_um=_SAMPLE_STEP_UM)
                along = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))])
                hit = sampled[(u, v, key)] = (made, path, [(float(a), float(along[-1])) for a in along])
            _made, path, edge_arcs = hit
            points.append(path)
            owners.extend([(u, v, key)] * len(path))
            arcs.extend(edge_arcs)
        if not points:
            return None, [], None, []
        stacked = np.vstack(points)
        return stacked, owners, cKDTree(stacked), arcs

    points, owners, samples, arcs = graph_samples()
    if samples is None:
        return 0
    distance, _nearest = samples.query(tree_points)
    covered = distance <= tree_tolerance

    T = nx.Graph()
    T.add_nodes_from(range(len(tree_vox)))
    for i, j in cKDTree(tree_vox).query_pairs(1.75):
        T.add_edge(int(i), int(j), weight=float(np.linalg.norm((tree_vox[i] - tree_vox[j]) * vs)))
    uncovered = [i for i in range(len(tree_vox)) if not covered[i]]

    added = 0
    for gap in nx.connected_components(T.subgraph(uncovered)):
        sides = sorted({j for i in gap for j in T.neighbors(i) if covered[j]})
        if len(sides) < 2:
            continue
        local = T.subgraph(set(gap) | set(sides))
        gap_length = sum(d["weight"] for _a, _b, d in T.subgraph(gap).edges(data=True)) + 2 * float(vs.min())
        if gap_length < TREE_GAP_MIN_UM:
            continue
        if reaching is not None:
            voxels_in_gap = sorted(gap)
            if not np.any(
                reaching.query(tree_points[voxels_in_gap])[0] <= _bridge_reach_um(tree_radius[voxels_in_gap])
            ):
                continue
        joined = False
        for x, side in enumerate(sides):
            for other in sides[x + 1:]:
                # The tree's path as it would be drawn: from the graph's point
                # at one side to the graph's point at the other.
                at = [int(samples.query(tree_points[n])[1]) for n in (side, other)]
                route = nx.shortest_path(local, side, other, weight="weight")
                tree_way = float(calculate_path_length(
                    [points[at[0]], *(tree_points[n] for n in route[1:-1]), points[at[1]]]
                ))
                own = _graph_way_um(
                    G, *[(owners[k], *arcs[k]) for k in at], cutoff=TREE_FILL_MIN_DETOUR * tree_way,
                )
                joined = joined or own <= TREE_FILL_MIN_DETOUR * tree_way
        if joined:
            continue
        if len(sides) == 2:
            routes = [nx.shortest_path(local, sides[0], sides[1], weight="weight")]
        else:
            hub = max(gap, key=lambda i: float(tree_radius[i]))
            routes = [nx.shortest_path(local, side, hub, weight="weight") for side in sides]
            routes = [r for r in routes if len(r) >= 2]
        nodes_at: dict[int, Any] = {}

        def node_for_side(side):
            nonlocal points, owners, samples, arcs
            if side in nodes_at:
                return nodes_at[side]
            where = tree_points[side]
            index = int(samples.query(where)[1])
            if not G.has_edge(*owners[index]):
                points, owners, samples, arcs = graph_samples()
                index = int(samples.query(where)[1])
            node = insert_node_on_edge(G, *owners[index], points[index])
            nodes_at[side] = node
            return node

        side_nodes = [node_for_side(side) for side in sides]
        if len(set(side_nodes)) < len(side_nodes):
            continue
        hub_node = None
        if len(sides) > 2:
            from ._helpers import next_node_id

            hub_node = next_node_id(G, set())
            G.add_node(hub_node, pos=tuple(float(c) for c in tree_points[routes[0][-1]]))
        for route, node in zip(routes, side_nodes):
            far = hub_node if hub_node is not None else side_nodes[1]
            path = [np.asarray(G.nodes[node]["pos"], dtype=float)]
            path += [tree_points[i] for i in route[1:-1]]
            path.append(np.asarray(G.nodes[far]["pos"], dtype=float))
            voxels = [tuple(float(c) for c in p) for p in path]
            G.add_edge(node, far, **{**(attributes or {}), "voxels": voxels,
                                     "length": float(calculate_path_length(voxels)),
                                     "centreline_smoothing": "kept_raw", FROM_TREE: True})
            added += 1
        points, owners, samples, arcs = graph_samples()
    return added


def _graph_way_um(G, a, b, *, cutoff: float) -> float:
    """The shortest way (um) along *G*'s vessels between two points on its
    edges, each ``((u, v, key), arc from u, the edge's arc length)``; ``inf``
    beyond *cutoff* or with none. Bridges are not vessel."""

    def weight(_u, _v, d):
        lengths = [float(x.get("length") or 0.0) for x in d.values() if not x.get(IS_ZERO_RESISTANCE)]
        return min(lengths) if lengths else None

    (edge_a, s_a, total_a), (edge_b, s_b, total_b) = a, b
    best = abs(s_a - s_b) if edge_a == edge_b else np.inf
    for end_a, off_a in ((edge_a[0], s_a), (edge_a[1], total_a - s_a)):
        reach = nx.single_source_dijkstra_path_length(G, end_a, cutoff=cutoff, weight=weight)
        for end_b, off_b in ((edge_b[0], s_b), (edge_b[1], total_b - s_b)):
            if end_b in reach:
                best = min(best, off_a + reach[end_b] + off_b)
    return float(best) if best <= cutoff else float("inf")


def _extend_thick_centreline_ends(
    G, tree_voxels, vs, tree_points, tree_tolerance, tree_radius, lumen, thick, chosen, *,
    attributes=None, reach_um: float = EXTEND_REACH_UM,
) -> int:
    """Draw each thick centreline stopping at a dead end on, in *G*, in place.
    Returns how many.

    A dead end is the thick centreline's when its edge follows the tree (the
    split's labels) for :data:`_MIN_CENTRELINE_AT_END_UM` or more, to within
    :data:`EXTEND_END_OFF_TREE_UM` of the tip. From it the thick vessel is
    drawn on:

    1. along the tree, where the tree runs on with no other graph centreline
       following it and leaves ahead of the dead end (within 90 degrees of the
       way its last :data:`TAPER_DIRECTION_UM` head). A tree leading on to
       another graph centreline is a gap with graph on both sides, the fill's,
       and the dead end is left;
    2. then, from where that ends, the cheapest way through the segmentation
       (:data:`EXTEND_CENTRE_WEIGHT_UM`) leaving within
       :data:`EXTEND_START_CONE_DEG` of the way it was heading and never back
       into the thick lumen the graph already draws, to the first other
       vessel's centreline (its lumen counted in) or dead end within
       *reach_um*. When an image face -- not one the start lies within 3 um
       of -- comes first, the vessel leaves the image there and nothing is
       drawn.

    Grid Y (Harvey, 2026-10-15; 18 places, every one on E14.5's two
    segmentations): drawn to a vessel or dead end, right at all 13 where
    the way was 32 um or shorter, wrong at the one of 54 um; drawn to a face,
    right at 1 of 3 (a 4 um step out to the face; wrong 10 um across a vessel
    to a face, and 10 um through a 1 um neck). This rule: 17 of 18, against
    14 for drawing every one.

    The dead end's edge is carried on along that way, as a wall join is
    (:data:`EXTENDED_UM` records how far), and takes *attributes*. With
    *chosen* (a post-processing split's edges), only those edges' dead ends,
    and the edge drawn on joins them.
    """
    from scipy.ndimage import distance_transform_edt
    from skimage.graph import MCP_Geometric

    from .edit import insert_node_on_edge

    vs = np.asarray(vs, dtype=float)
    tree_vox = np.asarray(tree_voxels)  # the tree's voxels, C order
    if len(tree_vox) < 2:
        return 0
    tree = cKDTree(tree_points)
    shape = np.asarray(lumen.shape)
    extent = (shape - 1) * vs
    T = nx.Graph()
    T.add_nodes_from(range(len(tree_vox)))
    for i, j in cKDTree(tree_vox).query_pairs(1.75):
        T.add_edge(int(i), int(j), weight=float(np.linalg.norm((tree_vox[i] - tree_vox[j]) * vs)))
    cone_cos = float(np.cos(np.radians(EXTEND_START_CONE_DEG)))

    def arc(path):
        return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))])

    def point_along(path, um):
        return path[min(int(np.searchsorted(arc(path), um)), len(path) - 1)]

    def thick_end(n):
        """``(tip, other end, key, edge data, heading)`` when *n* is a dead end
        of the thick centreline, else None."""
        if G.degree(n) != 1 or "pos" not in G.nodes[n]:
            return None
        (a, b, key, data), = G.edges(n, keys=True, data=True)
        w = b if a == n else a
        if w == n or data.get(IS_ZERO_RESISTANCE) or "pos" not in G.nodes[w]:
            return None
        if chosen is not None and (frozenset((n, w)), key) not in chosen:
            return None
        path = densify_polyline(_polyline_from(G, n, w, data), max_step_um=_SAMPLE_STEP_UM)
        if len(path) < 2:
            return None
        distance, nearest = tree.query(path)
        on = distance <= tree_tolerance[nearest]
        if not on.any():
            return None
        labels = _labels(path, on, points_inside_mask(path, thick, voxel_size_zyx=tuple(vs)))
        along = arc(path)
        centreline = np.flatnonzero(labels == _CENTRELINE)
        if not len(centreline) or along[centreline[0]] > EXTEND_END_OFF_TREE_UM:
            return None
        first = int(centreline[0])
        off = np.flatnonzero(labels[first:] != _CENTRELINE)
        run = (along[first + off[0] - 1] if len(off) else along[-1]) - along[first]
        if run < _MIN_CENTRELINE_AT_END_UM:
            return None
        heading = path[0] - point_along(path, TAPER_DIRECTION_UM)
        norm = float(np.linalg.norm(heading))
        if norm == 0.0:
            return None
        return path[0], w, key, data, heading / norm

    def vessels():
        points, owners = [], []
        for u, v, key, data in G.edges(keys=True, data=True):
            if data.get(IS_ZERO_RESISTANCE) or "pos" not in G.nodes[u] or "pos" not in G.nodes[v]:
                continue
            path = densify_polyline(_polyline_from(G, u, v, data), max_step_um=_SAMPLE_STEP_UM)
            points.append(path)
            owners.extend([(u, v, key)] * len(path))
        if not points:
            return np.empty((0, 3)), [], None, [frozenset()] * len(tree_points)
        stacked = np.vstack(points)
        samples = cKDTree(stacked)
        ids = [(frozenset(o[:2]), o[2]) for o in owners]
        cover = [frozenset(ids[j] for j in near) for near in samples.query_ball_point(tree_points, tree_tolerance)]
        return stacked, owners, samples, cover

    points, owners, samples, cover = vessels()
    if samples is None:
        return 0
    extended = 0
    for n in sorted([n for n in G.nodes if G.degree(n) == 1], key=str):
        if not G.has_node(n):
            continue
        end = thick_end(n)
        if end is None:
            continue
        tip, w, key, data, heading = end
        own = (frozenset((n, w)), key)

        # 1. The tree on past the tip, where no other graph centreline follows it.
        s = int(tree.query(tip)[1])
        near_tip = set(int(i) for i in tree.query_ball_point(tip, float(tree_tolerance[s]) + 1.5))
        passable = [
            i for i in range(len(tree_vox))
            if not cover[i] or (cover[i] <= {own} and i in near_tip) or i == s
        ]
        sub = T.subgraph(passable)
        reached, routes = nx.single_source_dijkstra(sub, s, cutoff=EXTEND_TREE_MAX_UM, weight="weight")
        beyond = [i for i in reached if not cover[i]]
        if any(cover[j] - {own} for i in beyond for j in T.neighbors(i)):
            continue
        tree_path = np.empty((0, 3))
        if beyond:
            ends = [i for i in beyond if sub.degree(i) == 1]
            route = routes[max(ends or beyond, key=reached.get)]
            first_out = next(j for j, i in enumerate(route) if not cover[i])
            candidate = tree_points[route[first_out:]]
            if float((point_along(np.vstack([tip, candidate]), TAPER_DIRECTION_UM) - tip) @ heading) > 0.0:
                tree_path = candidate
        start, start_heading = tip, heading
        if len(tree_path):
            start = tree_path[-1]
            back = point_along(np.vstack([tip, tree_path])[::-1], TAPER_DIRECTION_UM)
            if np.linalg.norm(start - back) > 0.0:
                start_heading = (start - back) / np.linalg.norm(start - back)

        # 2. On through the segmentation from where that stops.
        pad = np.ceil((reach_um + 3.0) / vs).astype(int)
        at = np.clip(np.rint(start / vs).astype(int), 0, shape - 1)
        lo = np.maximum(at - pad, 0)
        hi = np.minimum(at + pad + 1, shape)
        box = tuple(slice(int(l), int(h)) for l, h in zip(lo, hi))
        crop = np.asarray(lumen[box], dtype=bool).copy()
        crop[tuple(at - lo)] = True
        wall = distance_transform_edt(crop, sampling=tuple(vs))
        allowed = crop.copy()
        s_start = int(tree.query(start)[1])
        region = np.argwhere(np.asarray(thick[box], dtype=bool) & crop)
        if len(region):
            # Back into the thick lumen the graph already draws: segmentation in
            # the region nearest a stretch of tree some graph centreline follows,
            # further along the tree from where the way starts than its radius.
            _d, nearest = tree.query((region + lo) * vs)
            along_tree = nx.single_source_dijkstra_path_length(T, s_start, cutoff=EXTEND_TREE_MAX_UM, weight="weight")
            far = np.array([along_tree.get(int(i), np.inf) for i in nearest])
            drawn = np.array([bool(cover[int(i)]) for i in nearest])
            allowed[tuple(region[drawn & (far > tree_radius[s_start] + 2.0)].T)] = False
        offset = (np.indices(crop.shape).reshape(3, -1).T + lo) * vs - start
        distance = np.linalg.norm(offset, axis=1)
        behind = (distance <= tree_radius[s_start] + EXTEND_START_CONE_UM) & (
            offset @ start_heading < cone_cos * distance
        )
        allowed &= ~behind.reshape(crop.shape)
        allowed[tuple(at - lo)] = True
        router = MCP_Geometric(
            np.where(allowed, 1.0 + EXTEND_CENTRE_WEIGHT_UM / (wall + 0.5), np.inf),
            fully_connected=True, sampling=tuple(vs),
        )
        cost, _ = router.find_costs([tuple(at - lo)])
        length, _ = MCP_Geometric(
            np.where(allowed, 1.0, np.inf), fully_connected=True, sampling=tuple(vs)
        ).find_costs([tuple(at - lo)])
        targets = []  # (cost, voxel in the box, sample index or None for a face)
        inside_box = np.flatnonzero(
            np.all((points >= lo * vs) & (points <= (hi - 1) * vs), axis=1)
        )
        inside_box = [j for j in inside_box if (frozenset(owners[j][:2]), owners[j][2]) != own]
        if inside_box:
            voxel = np.clip(np.rint(points[inside_box] / vs).astype(int) - lo, 0, np.asarray(crop.shape) - 1)
            c = cost[tuple(voxel.T)]
            ok = np.isfinite(c) & (length[tuple(voxel.T)] <= reach_um + wall[tuple(voxel.T)])
            targets += [(float(cj), tuple(vj), int(j)) for cj, vj, j, keep in zip(c, voxel, inside_box, ok) if keep]
        for axis in range(3):
            for plane, gap in ((0, start[axis]), (shape[axis] - 1, extent[axis] - start[axis])):
                if gap <= 3.0 or not lo[axis] <= plane < hi[axis]:
                    continue
                face = [slice(None)] * 3
                face[axis] = slice(plane - lo[axis], plane - lo[axis] + 1)
                c = np.where(np.isfinite(cost[tuple(face)]) & (length[tuple(face)] <= reach_um),
                             cost[tuple(face)], np.inf)
                if np.isfinite(c).any():
                    best = np.unravel_index(int(np.argmin(c)), c.shape)
                    voxel = list(best)
                    voxel[axis] = plane - lo[axis]
                    targets.append((float(c[best]), tuple(int(x) for x in voxel), None))
        if not targets:
            continue
        _c, voxel, j = min(targets, key=lambda t: t[0])
        if j is None:
            continue  # the vessel leaves the image: an open end, not drawn on
        way = (np.asarray(router.traceback(voxel), dtype=float) + lo) * vs
        u, v, k = owners[j]
        target = points[j]
        node = None
        for ending in (u, v):
            if G.degree(ending) == 1 and np.linalg.norm(np.asarray(G.nodes[ending]["pos"]) - target) <= 2.0:
                node = ending
        if node is None:
            node = insert_node_on_edge(G, u, v, k, target)
        way = np.vstack([start, way[1:-1], np.asarray(G.nodes[node]["pos"], dtype=float)])
        if node in (n, w):
            continue
        path = np.vstack([tip, tree_path, way[1:]]) if len(tree_path) else way
        combined = np.vstack([_polyline_from(G, w, n, data), path[1:]])
        attrs = {k: v for k, v in data.items() if k not in ("voxels", "length")}
        attrs["voxels"] = [tuple(float(c) for c in row) for row in combined]
        attrs["length"] = float(calculate_path_length(attrs["voxels"]))
        attrs[EXTENDED_UM] = float(calculate_path_length([tuple(r) for r in path]))
        attrs.update(attributes or {})
        G.remove_node(n)
        new_key = G.add_edge(w, node, **attrs)
        if chosen is not None:
            chosen.add((frozenset((w, node)), new_key))
        extended += 1
        points, owners, samples, cover = vessels()
    return extended


def _centreline_samples(G, near_tree, thick, vs):
    """Every point of *G*'s vessels (not bridges) the split labels thick
    centreline (:func:`_labels`, so not a flicker near the tree), sampled
    every :data:`_SAMPLE_STEP_UM`, and the edge each is on: ``(points,
    owners, KD-tree)``, the tree None when there are none."""
    points, owners = [], []
    for u, v, key, data in G.edges(keys=True, data=True):
        if data.get(IS_ZERO_RESISTANCE) or "pos" not in G.nodes[u] or "pos" not in G.nodes[v]:
            continue
        path = densify_polyline(_polyline_from(G, u, v, data), max_step_um=_SAMPLE_STEP_UM)
        if len(path) < 2:
            continue
        on_tree = near_tree(path)[1]
        if not on_tree.any():
            continue
        in_region = points_inside_mask(path, thick, voxel_size_zyx=tuple(vs))
        on = _labels(path, on_tree, in_region) == _CENTRELINE
        if on.any():
            points.append(path[on])
            owners.extend([(u, v, key)] * int(on.sum()))
    if not points:
        return np.empty((0, 3)), [], None
    stacked = np.vstack(points)
    return stacked, owners, cKDTree(stacked)


def _join_dead_ends_at_walls(
    G, near_tree, tree, tree_radius, lumen, thick, vs, chosen, *,
    max_background_gap_um=None, min_mask_fraction=None, max_short_loop_um=None, short_loop_keep_hole_um=None,
    support=None,
) -> int:
    """Draw each dead end that stops at a thick vessel's wall on to the
    nearest point of the thick centreline in *G*, in place, so the split
    bridges it like any vessel reaching the centreline. Returns how many.

    A dead end qualifies when its tip lies within :data:`WALL_JOIN_REACH_UM`
    of the lumen (the inscribed radius at the nearest tree point) but not on
    the centreline, and the way on to the centreline continues its vessel
    (``build.gap_bridge_continues_terminal``: a vessel at least 3 um long,
    turning at most 80 degrees). The way is the route through the mask
    (``reconnect.route_through_mask``), else the straight line, whichever the
    mask-support test accepts and leaves the tip within that turn itself
    (``reconnect.path_continues_ends``); its part outside the thick-vessel
    region must not run beside a vessel in the same lumen (judged at its
    middle when short), and the join must close no loop inside one lumen.
    With *chosen* (the edges a post-processing split rebuilds), only those
    edges' dead ends, and the joined edge joins them. *support*, a
    ``MaskSupport`` over *lumen*, is used rather than one made here.
    """
    from haemolynx.preprocessing.bridge_mask_support import (
        DEFAULT_MAX_BACKGROUND_GAP_UM, DEFAULT_MIN_MASK_FRACTION, MaskSupport,
    )

    from ._helpers import EdgeSampleIndex, duplicates_existing_vessel
    from .build import MAX_GAP_BRIDGE_TURN_DEG, gap_bridge_continues_terminal
    from .dead_end_extension import _closes_loop_in_lumen, _loop_through
    from .lumen_loops import DEFAULT_SHORT_LOOP_KEEP_HOLE_UM, DEFAULT_SHORT_LOOP_UM, loop_hole_um
    from .edit import insert_node_on_edge
    from .reconnect import path_continues_ends, route_through_mask

    gap_um = DEFAULT_MAX_BACKGROUND_GAP_UM if max_background_gap_um is None else float(max_background_gap_um)
    fraction = DEFAULT_MIN_MASK_FRACTION if min_mask_fraction is None else float(min_mask_fraction)
    if support is not None and np.shape(support.mask) == np.shape(lumen):
        support = support.limited_to(gap_um, fraction)
    else:
        support = MaskSupport(lumen, tuple(vs), max_background_gap_um=gap_um, min_mask_fraction=fraction)
    tips = []
    for n in G.nodes:
        if G.degree(n) != 1 or "pos" not in G.nodes[n]:
            continue
        (a, b, key, data), = G.edges(n, keys=True, data=True)
        w = b if a == n else a
        if w == n or data.get(IS_ZERO_RESISTANCE):
            continue
        if chosen is not None and (frozenset((n, w)), key) not in chosen:
            continue
        p = np.asarray(G.nodes[n]["pos"], dtype=float)
        distance, nearest = tree.query(p)
        radius = float(tree_radius[nearest]) if np.isfinite(tree_radius[nearest]) else 0.0
        if distance > radius + WALL_JOIN_REACH_UM or near_tree(p[None])[1][0]:
            continue
        tips.append((float(distance), n))
    if not tips:
        return 0
    points, owners, samples = _centreline_samples(G, near_tree, thick, vs)
    if samples is None:
        return 0
    index = EdgeSampleIndex(G)
    short_loop_um = DEFAULT_SHORT_LOOP_UM if max_short_loop_um is None else float(max_short_loop_um)
    keep_hole_um = (
        DEFAULT_SHORT_LOOP_KEEP_HOLE_UM if short_loop_keep_hole_um is None else float(short_loop_keep_hole_um)
    )
    joined = refused_turn = refused_background = refused_beside = refused_loop = 0
    for _distance, n in sorted(tips, key=lambda t: (t[0], str(t[1]))):
        if not G.has_node(n) or G.degree(n) != 1:
            continue
        (a, b, key, data), = G.edges(n, keys=True, data=True)
        w = b if a == n else a
        p = np.asarray(G.nodes[n]["pos"], dtype=float)
        radius = tree_radius[tree.query(p)[1]]
        reach = (float(radius) if np.isfinite(radius) else 0.0) + WALL_JOIN_REACH_UM + CENTRELINE_TOLERANCE_UM
        nearby = samples.query_ball_point(p, reach)
        if any(not G.has_edge(*owners[j]) for j in nearby):
            # An earlier join split the edge a sample was on: sample again.
            points, owners, samples = _centreline_samples(G, near_tree, thick, vs)
            nearby = samples.query_ball_point(p, reach)
        found = sorted(
            (float(np.linalg.norm(points[j] - p)), j) for j in nearby if n not in owners[j][:2]
        )
        if not found:
            continue
        q, edge = points[found[0][1]], owners[found[0][1]]
        if not gap_bridge_continues_terminal(G, n, q):
            refused_turn += 1
            continue
        straight = np.vstack([p, q])
        routed = route_through_mask(lumen, p, q, tuple(vs))
        path, turned = None, False
        for candidate in (routed, straight):
            if candidate is None or not support.accepts(candidate):
                continue
            # The turn on the path drawn, as for every mask-supported bridge:
            # a route round background can leave the tip sideways or back
            # where the straight line to the centreline does not.
            if not path_continues_ends(G, candidate, (n, None), MAX_GAP_BRIDGE_TURN_DEG):
                turned = True
                continue
            path = candidate
            break
        if path is None:
            if turned:
                refused_turn += 1
            else:
                refused_background += 1
            continue
        path = densify_polyline(np.asarray(path, dtype=float), max_step_um=_SAMPLE_STEP_UM)
        # Inside the thick lumen the path is the bridge-to-be; outside it, a
        # piece of vessel, which must not be a second strand beside another.
        # Judged at its middle when too short for anything to be judged away
        # from its ends, as a reconnect's bridge is: most joins are.
        distance, nearest = tree.query(path)
        in_lumen = np.flatnonzero(distance <= np.nan_to_num(tree_radius[nearest], posinf=0.0))
        outside = path[: int(in_lumen[0]) if len(in_lumen) else len(path)]
        if (
            len(outside) >= 2
            and calculate_path_length([tuple(r) for r in outside]) >= 2.0
            and duplicates_existing_vessel(
                outside, G, support.inside, support.radius, index=index, judge_short_at_middle=True,
            )
        ):
            refused_beside += 1
            continue
        if _closes_loop_in_lumen(G, n, edge, q, path, support):
            refused_beside += 1
            continue
        loop = _loop_through(G, n, edge, q, path, short_loop_um) if short_loop_um > 0 else None
        # A loop round a wide hole is one the clean-up leaves (grids Q and Q2).
        if loop is not None and not (keep_hole_um > 0 and (loop_hole_um(loop, support) or 0.0) > keep_hole_um):
            refused_loop += 1
            continue
        node = insert_node_on_edge(G, *edge, q)
        if node in (n, w):
            continue
        path[-1] = np.asarray(G.nodes[node]["pos"], dtype=float)
        combined = np.vstack([_polyline_from(G, w, n, data), path[1:]])
        attrs = {k: v for k, v in data.items() if k not in ("voxels", "length")}
        attrs["voxels"] = [tuple(float(c) for c in row) for row in combined]
        attrs["length"] = float(calculate_path_length(attrs["voxels"]))
        attrs[JOINED_AT_WALL_UM] = float(calculate_path_length([tuple(r) for r in path]))
        G.remove_node(n)
        new_key = G.add_edge(w, node, **attrs)
        if chosen is not None:
            chosen.add((frozenset((w, node)), new_key))
        index.add(path)
        joined += 1
    if joined or refused_turn or refused_background or refused_beside or refused_loop:
        logger.info(
            "Thick-vessel bridges: %d dead end(s) at a thick vessel's wall drawn on to its "
            "centreline through the mask; refused %d turning away from it, %d across background, "
            "%d beside a vessel in the same lumen or closing a loop inside one, and %d closing "
            "a loop shorter than %.0f um round a hole no wider than %.0f um.",
            joined, refused_turn, refused_background, refused_beside, refused_loop, short_loop_um, keep_hole_um,
        )
    return joined


def _bridge_thick_forks(G, near_tree, lumen, vs, chosen) -> int:
    """At each fork of the thick centreline, make the stretch of a narrower
    branch inside the trunk's lumen a bridge, in place. Returns how many.

    A fork is a node on the tree with at least three vessel edges whose first
    :data:`FORK_ARM_UM` lie on it. Each arm's radius is the median inscribed
    radius of *lumen* over the :data:`FORK_ARM_UM` past the fork's own; the
    two widest are the trunk, and an arm at most
    :data:`FORK_BRANCH_MAX_RADIUS_FRACTION` of the narrower of them is a
    branch. The branch is cut where it first lies further from the trunk's
    centreline than the trunk's radius there -- where it leaves the trunk's
    lumen -- and the piece from the fork to that point becomes a straight
    bridge. A branch whose first edge ends before it leaves is left alone.
    """
    from .edit import insert_node_on_edge

    vs = np.asarray(vs, dtype=float)
    made = 0
    for n in list(G.nodes):
        if not G.has_node(n) or G.degree(n) < 3 or "pos" not in G.nodes[n]:
            continue
        here = np.asarray(G.nodes[n]["pos"], dtype=float)
        if not near_tree(here[None])[1][0]:
            continue
        arms = []
        for a, b, key, data in G.edges(n, keys=True, data=True):
            other = b if a == n else a
            if data.get(IS_ZERO_RESISTANCE) or other == n:
                continue
            path = densify_polyline(_polyline_from(G, n, other, data), max_step_um=_SAMPLE_STEP_UM)
            arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))])
            if near_tree(path[arc <= FORK_ARM_UM])[1].mean() < 0.8:
                continue
            arms.append(dict(edge=(n, other, key), path=path, arc=arc))
        if len(arms) < 3:
            continue
        r0 = float(_inscribed_radius_um(here[None], lumen, vs)[0])
        if not np.isfinite(r0) or r0 <= 0.0:
            continue
        for arm in arms:
            window = (arm["arc"] >= r0) & (arm["arc"] <= r0 + FORK_ARM_UM)
            radii = _inscribed_radius_um(arm["path"][window], lumen, vs) if window.any() else np.zeros(0)
            radii = radii[np.isfinite(radii)]
            arm["radius"] = float(np.median(radii)) if len(radii) else float("nan")
        if any(not np.isfinite(arm["radius"]) for arm in arms):
            continue
        arms.sort(key=lambda arm: -arm["radius"])
        trunk, branches = arms[:2], arms[2:]
        trunk_points = np.vstack([arm["path"] for arm in trunk])
        trunk_radii = _inscribed_radius_um(trunk_points, lumen, vs)
        trunk_tree = cKDTree(trunk_points)
        narrowest_trunk = min(arm["radius"] for arm in trunk)
        for arm in branches:
            if arm["radius"] > FORK_BRANCH_MAX_RADIUS_FRACTION * narrowest_trunk:
                continue
            u, v, key = arm["edge"]
            if chosen is not None and (frozenset((u, v)), key) not in chosen:
                continue
            distance, nearest = trunk_tree.query(arm["path"])
            left = np.flatnonzero((distance > trunk_radii[nearest]) & (arm["arc"] > 0.0))
            if not len(left) or arm["arc"][left[0]] < 1.0 or left[0] >= len(arm["path"]) - 1:
                continue
            exit_node = insert_node_on_edge(G, u, v, key, arm["path"][left[0]])
            if exit_node in (u, v):
                continue
            for data in G[n][exit_node].values():
                end = np.asarray(G.nodes[exit_node]["pos"], dtype=float)
                data["voxels"] = [tuple(float(c) for c in here), tuple(float(c) for c in end)]
                data["length"] = float(np.linalg.norm(end - here))
                data[IS_ZERO_RESISTANCE] = True
            made += 1
    return made


def _remove_blind_vessels_beyond_bridges(
    built, G, tree, *, extent_um, stub_length_um: Callable[[np.ndarray], float] | None = None,
) -> tuple[int, float, int]:
    """Remove each bridge whose vessel ends blind inside the image, with
    that vessel: followed from the bridge through nodes only it passes, it
    reaches no junction and does not leave the image. A vessel running out
    of the image is kept, and so is one at least *stub_length_um(inner end
    of the bridge)* long, which the stub prune would keep as well.

    Harvey's grid Y (2026-10-15, E14.5): of six vessels this took, the four
    marked keep were 34-154 um long -- stretches of the thick vessel itself,
    and long capillaries -- and the two marked remove 6 and 16 um, all on
    the right side of the prune's 3 lumen radii; so was grid W's 142 um
    capillary chain (13, 14, 24, marked plain-right). Returns how many went,
    their length (um), and how many were kept as longer than a stub."""
    from .degree2 import create_trivial_merged_edge

    extent_um = np.asarray(extent_um, dtype=float)

    def at_image_edge(node):
        p = np.asarray(built.nodes[node]["pos"], dtype=float)
        return bool(np.any(p <= 2.0) or np.any(p >= extent_um - 2.0))

    removed, removed_um, kept = 0, 0.0, 0
    for u, v, key in [
        (u, v, key) for u, v, key, d in built.edges(keys=True, data=True)
        if d.get(IS_ZERO_RESISTANCE)
    ]:
        if not built.has_edge(u, v, key):
            continue
        distance = tree.query(np.stack([built.nodes[u]["pos"], built.nodes[v]["pos"]]))[0]
        inner, outer = (u, v) if distance[0] <= distance[1] else (v, u)
        chain = [(u, v, key)]
        node, seen = outer, {inner, outer}
        while built.degree(node) == 2:
            (a, b, k), = [
                (a, b, k) for a, b, k in built.edges(node, keys=True)
                if (a, b, k) not in chain and (b, a, k) not in chain
            ] or [(None, None, None)]
            if a is None:
                break
            chain.append((a, b, k))
            node = b if a == node else a
            if node in seen:
                break
            seen.add(node)
        if built.degree(node) != 1 or at_image_edge(node):
            continue
        length = sum(float(built.edges[e].get("length", 0.0)) for e in chain[1:])
        if stub_length_um is not None and length >= stub_length_um(built.nodes[inner]["pos"]):
            kept += 1
            continue
        removed += 1
        removed_um += length
        for e in chain:
            built.remove_edge(*e)
        for n in seen - {inner}:
            if built.has_node(n) and built.degree(n) == 0:
                built.remove_node(n)
        if built.degree(inner) == 2 and (inner not in G or G.degree(inner) != 2):
            (_n, n1, _k1, d1), (_m, n2, _k2, d2) = list(built.edges(inner, keys=True, data=True))
            if (
                n1 != n2 and not d1.get(IS_ZERO_RESISTANCE) and not d2.get(IS_ZERO_RESISTANCE)
                and not taper_meets_vessel(d1, d2)
            ):
                merged = create_trivial_merged_edge(d1, d2, built.nodes[inner].get("pos"))
                built.remove_node(inner)
                built.add_edge(n1, n2, **merged)
    return removed, removed_um, kept


def _cut_off_edges(G, built, edges_at, new_node_at) -> set[int]:
    """The sampled edges that reach a piece of *built* cut off from the rest
    of what it was joined to in *G*, or made of nothing *G* had."""
    pieces = list(nx.connected_components(built))
    piece_of = {n: p for p, nodes in enumerate(pieces) for n in nodes}
    cut = {p for p, nodes in enumerate(pieces) if not any(n in G for n in nodes)}
    for component in nx.connected_components(G):
        parts = {piece_of[n] for n in component if n in piece_of}
        if len(parts) > 1:
            keep = max(parts, key=lambda q: len(pieces[q]))
            cut |= parts - {keep}
    if not cut:
        return set()
    edge_of_new = {node: i for (i, _k), node in new_node_at.items()}
    edges: set[int] = set()
    for p in cut:
        for n in pieces[p]:
            if n in edge_of_new:
                edges.add(edge_of_new[n])
            edges.update(i for i, _end in edges_at.get(n, ()))
    return edges


def remove_dangling_bridges(G: nx.MultiGraph) -> int:
    """Take out, in place, every bridge with a dead end -- a bridge is never
    one: it joins a vessel to the thick centreline, so one whose vessel a
    clean-up took out joins nothing. And every vessel whose two ends are
    each nothing but a bridge, both to the same node, with its bridges: a
    piece of strand left in the thick vessel's lumen (where the thick
    centreline drawn from the tree replaced the graph's own, on E14.5), not
    a vessel opening into it; contracted for the statistics, it would be a
    loop on one node. Repeated until none is left, with the nodes left bare.
    Returns how many bridges went."""
    removed = 0
    while True:
        dangling = [
            (u, v, key) for u, v, key, data in G.edges(keys=True, data=True)
            if data.get(IS_ZERO_RESISTANCE) and (G.degree[u] == 1 or G.degree[v] == 1)
        ]
        for u, v, key, data in list(G.edges(keys=True, data=True)):
            if data.get(IS_ZERO_RESISTANCE) or u == v or G.degree[u] != 2 or G.degree[v] != 2:
                continue
            ends = []
            for node in (u, v):
                other = [
                    (a, b, k, d) for a, b, k, d in G.edges(node, keys=True, data=True)
                    if not (k == key and {a, b} == {u, v})
                ]
                if len(other) == 1 and other[0][3].get(IS_ZERO_RESISTANCE):
                    ends.append(other[0])
            if len(ends) == 2 and ends[0][1] == ends[1][1]:
                G.remove_edge(u, v, key)
                dangling.extend((a, b, k) for a, b, k, _d in ends)
        if not dangling:
            return removed
        for u, v, key in dangling:
            if G.has_edge(u, v, key):
                G.remove_edge(u, v, key)
                removed += 1
            for node in (u, v):
                if G.has_node(node) and G.degree[node] == 0:
                    G.remove_node(node)


def contract_bridges(G: nx.MultiGraph) -> tuple[nx.MultiGraph, dict[Any, Any]]:
    """*G* with each zero-resistance bridge's two ends made one node.

    What a vessel opening into a thick vessel is, measured as a network:
    the vessel meets the thick vessel's centreline, with no bridge between.
    Each bridge (or chain of them) becomes the node at its centreline end --
    the one with the most vessels -- so the thick vessel keeps its geometry.

    The nodes and edges share *G*'s own attribute dicts, so whatever is
    written on them -- the per-vessel results statistics colour by -- lands
    on *G*. Returns the graph and, per node of *G*, the node it became.
    """
    parent = {n: n for n in G.nodes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def vessels_at(n):
        return sum(1 for *_e, d in G.edges(n, data=True) if not d.get(IS_ZERO_RESISTANCE))

    for u, v, data in G.edges(data=True):
        if data.get(IS_ZERO_RESISTANCE):
            a, b = find(u), find(v)
            if a != b:
                keep, drop = sorted((a, b), key=lambda n: (-vessels_at(n), str(n)))
                parent[drop] = keep
    became = {n: find(n) for n in G.nodes}
    H = G.__class__()
    H.graph = G.graph
    for n, data in G.nodes(data=True):
        if became[n] == n:
            H.add_node(n)
            H._node[n] = data  # shared, not copied: see the docstring
    edges = G.edges(keys=True, data=True) if G.is_multigraph() else (
        (u, v, None, d) for u, v, d in G.edges(data=True)
    )
    for u, v, _key, data in edges:
        if data.get(IS_ZERO_RESISTANCE):
            continue
        a, b = became[u], became[v]
        if H.is_multigraph():
            key = H.add_edge(a, b)
            H._adj[a][b][key] = data
        else:
            H.add_edge(a, b)
            H._adj[a][b] = data
            H._adj[b][a] = data
    return H, became
