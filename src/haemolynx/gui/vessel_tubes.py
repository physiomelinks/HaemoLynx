"""Vessel tubes from Vectors origin+direction data.

Napari Vectors ``vector_style="line"`` draws two world-fixed ribbons. An
axis-aligned centreline step collapses a ribbon, and ``edge_width=0.6`` µm
then vanishes edge-on. A tube stays visible from every camera angle.

Each vessel is drawn at its own diameter (or, by the Tube diameter choice,
every vessel at one width), and vessels joined end to end at a node no
other vessel meets are drawn as one smooth tube through it. The
centreline it follows is the skeleton's voxel path: steps under a micron
that kink at every voxel, round a capillary six microns across. A tube built
ring by ring on that path pinches, bulges and folds where the vessel does
none of those things. So each tube follows its centreline smoothed over
about half its own radius, and again where it still bends tighter than it
is wide, with rings spaced in proportion to the local radius and its ends
kept on their nodes and rounded.

Where vessels meet, each one's radius eases to the node's: the mean of the
two where two join, and at a junction no wider than the widest of the
others, so the widest vessel's rounded end is inside its widest neighbour
instead of standing out as a ball. This is drawing only, and so is the rest:
a diameter far from its neighbours' is drawn within a factor of two of
theirs, while the graph keeps the values it has.

A zero-resistance bridge, where a vessel opens into a thick one's lumen, is
not vessel, and is not drawn as one: it is left out of the tubes, so the
vessel ends, cut square, where it meets the lumen and the thick vessel runs
on through the junction the bridge made, and its dashes (the vessels layer
draws it dashed) are thin flat-ended pieces of their own, half as wide as
the narrowest vessel it meets.

Nothing here imports napari. The widget hides the Vectors visual and shows a
Surface built from these arrays; hover and colour-by still read the Vectors.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from haemolynx.graph.thick_vessel_junctions import IS_ZERO_RESISTANCE
from haemolynx.gui.results import VESSELS, VESSEL_TUBES

#: Tube radius in microns. Vectors ``edge_width=0.6`` µm is half a voxel and
#: still subpixel at whole-network zoom; 2 µm stays visible. Also the
#: fallback radius for a segment whose own diameter is not yet known (NaN,
#: e.g. before the Diameters stage has run) when tubes are drawn per-vessel
#: diameter -- see :func:`tube_radii_um`.
TUBE_RADIUS_UM = 2.0

#: napari Surface shading the tubes layer starts with. Smooth lights each
#: vertex from the surface round it, so a tube reads as round; ``"flat"``
#: shows its facets and ``"none"`` draws every face one flat colour. Only the
#: starting value: a stage that redraws the tubes keeps whatever the user has
#: since chosen in the layer controls.
TUBE_SHADING = "smooth"

#: Sides round each tube, by the tubes layer's "Render quality" slider: the
#: same smooth tube at every level, rounder and slower to build each step.
TUBE_QUALITY_SIDES = (6, 8, 12, 18, 32)
DEFAULT_TUBE_QUALITY = 2

#: How wide the tubes are drawn, by the "Tube diameter" choice on the view
#: panel and on a tubes layer's controls: each vessel at its own diameter,
#: measured or assigned, or every vessel at the one width
#: :func:`tube_radius_um` gives a vessel with no diameter yet.
TUBE_DIAMETER_PER_VESSEL = "per_vessel"
TUBE_DIAMETER_UNIFORM = "uniform"
TUBE_DIAMETERS = (TUBE_DIAMETER_PER_VESSEL, TUBE_DIAMETER_UNIFORM)
TUBE_DIAMETER_LABELS = {
    TUBE_DIAMETER_PER_VESSEL: "Per vessel",
    TUBE_DIAMETER_UNIFORM: "Uniform",
}
DEFAULT_TUBE_DIAMETER = TUBE_DIAMETER_PER_VESSEL
#: The width (µm) every tube is drawn at while the choice is Uniform, set in
#: the µm box beside it: to start, the width of a vessel with no diameter
#: yet, and never outside this range.
DEFAULT_UNIFORM_TUBE_DIAMETER_UM = 2.0 * TUBE_RADIUS_UM
UNIFORM_TUBE_DIAMETER_RANGE_UM = (0.5, 1000.0)
#: A zero-resistance bridge's dashes are drawn this fraction of the radius
#: of the narrowest vessel it meets: thinner than any vessel beside them.
BRIDGE_RADIUS_FRACTION = 0.5
#: Rings along a tube are this fraction of its local radius apart.
_RING_SPACING_PER_RADIUS = 1.0 / 3.0
#: Fewest ring-to-ring segments along one vessel, so that a vessel shorter
#: than it is wide still eases to the radius of each of its nodes.
_MIN_SEGMENTS_PER_VESSEL = 3
#: Samples along a vessel its ring spacing is worked out from.
_SPACING_SAMPLES = 16
#: The ring centres are smoothed this many times with a 1-4-6-4-1 kernel:
#: three passes at a third of a radius apart is about half a radius.
_SMOOTHING_PASSES = 3
_KERNEL = ((-2, 1.0), (-1, 4.0), (0, 6.0), (1, 4.0), (2, 1.0))
#: A ring whose centreline bends tighter than this many of its radii is
#: smoothed again, up to this many more passes; where that is not enough its
#: radius is cut to this fraction of what clears its neighbouring rings, but
#: never below this fraction of its vessel's own.
_TIGHT_BEND_PER_RADIUS = 1.1
_MAX_BEND_PASSES = 24
_FOLD_MARGIN = 0.95
_MIN_RADIUS_FRACTION = 0.05
#: Two steps join when one ends this close to where the next starts (µm).
_JOIN_TOLERANCE_UM = 1e-6
#: Vessel ends this close together (µm) are at the same node. Wider than the
#: join tolerance: two vessels' ends there are separate copies of its position.
_NODE_TOLERANCE_UM = 1e-3
#: A vessel is drawn no more than this factor wider or narrower than the
#: length-weighted median of its tube and the vessels it runs straight into.
_DIAMETER_CLIP_FACTOR = 2.0
#: Another vessel at a junction carries this one straight on when their
#: directions out of it are at least this opposed (cosine).
_THROUGH_COSINE = -0.5

VESSEL_DRAW_TUBES = "tubes"
VESSEL_DRAW_LINES = "lines"
DEFAULT_VESSEL_DRAW = VESSEL_DRAW_TUBES

_EMPTY_VERTICES = np.empty((0, 3), dtype=float)
_EMPTY_FACES = np.empty((0, 3), dtype=np.intp)
_EMPTY_INDEX = np.empty((0,), dtype=np.intp)


def tube_radius_um(edge_width: float | None = None) -> float:
    """Radius used for the tube mesh: at least :data:`TUBE_RADIUS_UM`."""
    try:
        width = float(edge_width) if edge_width is not None else 0.0
    except (TypeError, ValueError):
        width = 0.0
    if not np.isfinite(width) or width < 0.0:
        width = 0.0
    return max(width, TUBE_RADIUS_UM)


def tube_radii_um(diameter_um: np.ndarray | None) -> np.ndarray | None:
    """Per-segment tube radius from each segment's own diameter, halved.

    ``None`` when *diameter_um* is missing or empty, so a caller falls back
    to the uniform :func:`tube_radius_um`. A non-finite or non-positive
    entry (diameter not yet assigned, e.g. before the Diameters stage has
    run) is left as-is here -- :func:`tubes_from_vectors` substitutes
    :data:`TUBE_RADIUS_UM` for any such entry itself, so every caller gets
    the same fallback without repeating it.
    """
    if diameter_um is None:
        return None
    values = np.asarray(diameter_um, dtype=float)
    if values.size == 0:
        return None
    return values / 2.0


def vessel_tubes_layer_name(vessels_name: str) -> str:
    """HaemoLynx-owned Surface name paired with a vessels Vectors layer."""
    name = str(vessels_name)
    if name == VESSELS:
        return VESSEL_TUBES
    if name.startswith(VESSELS):
        return VESSEL_TUBES + name[len(VESSELS) :]
    return f"{name} tubes"


def _normal_plane_frames(tangents: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Orthonormal (normal, binormal) per row, perpendicular to each tangent.

    Vectorized form of the single-segment case: axis-aligned X/Y/Z would make
    a single cross product vanish, so the reference axis is swapped (per row)
    when it is nearly parallel to that row's tangent. The two checks are
    sequential -- the second re-tests against the reference the first check
    may have just swapped to -- so it is two vectorized passes, not one.
    """
    n = tangents.shape[0]
    ref = np.tile(np.array([0.0, 0.0, 1.0]), (n, 1))
    swap_to_x = np.abs(np.einsum("ij,ij->i", tangents, ref)) > 0.9
    ref[swap_to_x] = np.array([1.0, 0.0, 0.0])
    swap_to_y = swap_to_x & (np.abs(np.einsum("ij,ij->i", tangents, ref)) > 0.9)
    ref[swap_to_y] = np.array([0.0, 1.0, 0.0])
    normal = np.cross(tangents, ref)
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)
    binormal = np.cross(tangents, normal)
    binormal /= np.linalg.norm(binormal, axis=1, keepdims=True)
    return normal, binormal


def _step_radii(radius: float | np.ndarray, count: int) -> np.ndarray:
    """One radius per Vectors row: a single *radius* repeated, or *radius* itself.

    A non-finite or non-positive entry in a per-row array falls back to
    :data:`TUBE_RADIUS_UM` rather than breaking that row's tube.
    """
    radius_arr = np.asarray(radius, dtype=float)
    if radius_arr.ndim == 0:
        if not np.isfinite(radius_arr) or radius_arr <= 0.0:
            raise ValueError(f"radius must be a positive finite number; got {radius}")
        return np.full(count, float(radius_arr))
    if radius_arr.shape != (count,):
        raise ValueError(
            "radius array must have one entry per vectors row "
            f"({count},); got shape {radius_arr.shape!r}"
        )
    radii = radius_arr.copy()
    radii[~np.isfinite(radii) | (radii <= 0.0)] = TUBE_RADIUS_UM
    return radii


def _step_flags(flags: np.ndarray | None, count: int) -> np.ndarray:
    """One flag per Vectors row; anything but ``True`` (NaN, None) is unflagged."""
    if flags is None:
        return np.zeros(count, dtype=bool)
    values = np.asarray(flags)
    if values.shape != (count,):
        raise ValueError(
            "bridges must have one entry per vectors row "
            f"({count},); got shape {values.shape!r}"
        )
    return np.asarray(values == True, dtype=bool)  # noqa: E712


def _node_ids(tips: np.ndarray) -> np.ndarray:
    """One label per vessel end, shared by the ends at the same place."""
    keys = np.round(tips / _NODE_TOLERANCE_UM).astype(np.int64)
    _unique, inverse = np.unique(keys, axis=0, return_inverse=True)
    return inverse.reshape(-1)


def _walk_chains(partner: np.ndarray, count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The vessels in tube order: runs of vessels joined end to end.

    *partner* gives, for each vessel end -- all *count* starts, then all
    ends -- the end it is joined to, or -1. Returns the vessels in turn,
    whether each is walked from its end back to its start, and where each
    run begins. A closed loop of vessels is opened at one of its nodes.
    """
    link = partner.tolist()
    seen = bytearray(count)
    order: list[int] = []
    flipped: list[bool] = []
    runs: list[int] = []

    def walk(vessel: int, entered_at_end: bool) -> None:
        runs.append(len(order))
        while True:
            seen[vessel] = 1
            order.append(vessel)
            flipped.append(entered_at_end)
            joined = link[vessel if entered_at_end else count + vessel]
            if joined < 0:
                return
            vessel, entered_at_end = joined % count, joined >= count
            if seen[vessel]:
                return

    open_ends = np.flatnonzero((partner[:count] < 0) | (partner[count:] < 0))
    for vessel in open_ends.tolist():
        if not seen[vessel]:
            walk(vessel, link[vessel] >= 0)
    for vessel in range(count):
        if not seen[vessel]:
            walk(vessel, False)
    return (
        np.asarray(order, dtype=np.intp),
        np.asarray(flipped, dtype=bool),
        np.asarray(runs, dtype=np.intp),
    )


def _weighted_median(
    group: np.ndarray, value: np.ndarray, weight: np.ndarray, count: int
) -> np.ndarray:
    """The weighted median of *value* in each of *count* groups, none empty:
    midway between the two middle values where the weight splits exactly."""
    order = np.lexsort((value, group))
    group, value, weight = group[order], value[order], weight[order]
    total = np.bincount(group, weights=weight, minlength=count)
    offset = np.repeat(np.cumsum(total) - total, np.bincount(group, minlength=count))
    within = np.cumsum(weight) - offset
    half = 0.5 * total[group]

    def first_where(reached):
        reached = np.flatnonzero(reached)
        return reached[np.r_[True, group[reached][1:] != group[reached][:-1]]]

    lower = first_where(within >= half * (1.0 - 1e-9))
    upper = first_where((within > half * (1.0 + 1e-9)) | (within >= total[group]))
    median = np.empty(count)
    median[group[lower]] = 0.5 * (value[lower] + value[upper])
    return median


def _ramped_radius(own, start, end, length, ramp, s):
    """A vessel's radius *s* along it: its own, eased to each node's over *ramp*."""

    def eased(x):
        x = np.clip(x, 0.0, 1.0)
        return x * x * (3.0 - 2.0 * x)

    return (
        own
        + (start - own) * (1.0 - eased(s / ramp))
        + (end - own) * (1.0 - eased((length - s) / ramp))
    )


def _bend_radius(
    centre: np.ndarray, first: np.ndarray, last: np.ndarray, index: np.ndarray
) -> np.ndarray:
    """Radius of the circle through each *index* ring's centre and its two
    neighbours'; infinite at a tube's ends and wherever it runs straight."""
    bend = np.full(len(index), np.inf)
    keep = (index > first[index]) & (index < last[index])
    inner = index[keep]
    a = centre[inner] - centre[inner - 1]
    b = centre[inner + 1] - centre[inner]
    sides = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) * np.linalg.norm(a + b, axis=1)
    twice_area = np.linalg.norm(np.cross(a, b), axis=1)
    curved = twice_area > 1e-12 * sides
    bend[np.flatnonzero(keep)[curved]] = sides[curved] / (2.0 * twice_area[curved])
    return bend


def _around(index: np.ndarray, reach: int, count: int) -> np.ndarray:
    """*index* with every ring up to *reach* either side of one, sorted."""
    near = np.zeros(count, dtype=bool)
    for offset in range(-reach, reach + 1):
        near[np.clip(index + offset, 0, count - 1)] = True
    return np.flatnonzero(near)


def _smoothing_matrix(first: np.ndarray, last: np.ndarray):
    """One smoothing pass over each tube's ring centres, its two end rings left in place.

    *first* and *last* give, for every ring, its tube's first and last ring.
    Past an end the tube is continued by reflection through that end, which
    carries a straight tube on straight and leaves the end where it was, but
    not the direction it leaves in.
    """
    from scipy.sparse import csr_matrix

    count = len(first)
    index = np.arange(count)
    pinned = (index == first) | (index == last)
    rows, cols, weights = [index[pinned]], [index[pinned]], [np.ones(int(pinned.sum()))]
    free = index[~pinned]
    lo, hi = first[~pinned], last[~pinned]
    for offset, weight in _KERNEL:
        at = free + offset
        before, after = at < lo, at > hi
        mirror = np.clip(np.where(before, 2 * lo - at, np.where(after, 2 * hi - at, at)), lo, hi)
        reflected = before | after
        anchor = np.where(before, lo, hi)
        w = weight / 16.0
        # A point past an end is the end doubled less its mirror image.
        rows += [free, free[reflected]]
        cols += [mirror, anchor[reflected]]
        weights += [np.where(reflected, -w, w), np.full(int(reflected.sum()), 2.0 * w)]
    return csr_matrix(
        (np.concatenate(weights), (np.concatenate(rows), np.concatenate(cols))),
        shape=(count, count),
    )


def _through_neighbours(
    ends: np.ndarray,
    inward: np.ndarray,
    node: np.ndarray,
    at_node: np.ndarray,
    order: np.ndarray,
    first_at_node: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """For each of *ends* on a junction, the other end there that carries it
    straight on, if any: ``(which of ends, that other end)``."""
    on_junction = np.flatnonzero(at_node[ends] >= 3)
    count = at_node[ends[on_junction]]
    asked = np.repeat(on_junction, count)
    rank = np.arange(len(asked)) - np.repeat(np.cumsum(count) - count, count)
    other = order[first_at_node[node[ends[asked]]] + rank]
    distinct = other != ends[asked]
    asked, other = asked[distinct], other[distinct]
    cosine = np.einsum("ij,ij->i", inward[ends[asked]], inward[other])
    best = np.lexsort((cosine, asked))
    best = best[np.r_[True, asked[best][1:] != asked[best][:-1]]] if best.size else best
    best = best[cosine[best] < _THROUGH_COSINE]
    return asked[best], other[best]


def tubes_from_vectors(
    vectors: np.ndarray,
    *,
    radius: float | np.ndarray = TUBE_RADIUS_UM,
    sides: int = TUBE_QUALITY_SIDES[DEFAULT_TUBE_QUALITY],
    groups: np.ndarray | None = None,
    bridges: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One smooth, closed tube with rounded ends per chain of joined vessels.

    *vectors* is ``(M, 2, 3)`` origin+direction data. Consecutive rows are
    one vessel when one ends where the next starts with the same radius and,
    given *groups* (one label per row, e.g. each step's ``edge_index``), the
    same label. Vessel ends in the same place are one node, and vessels that
    are the only two at a node are drawn as one tube through it. Zero-length
    and non-finite rows are skipped.

    *radius* is one value for every row, or one per row -- e.g. each
    vessel's own measured/set diameter halved (see :func:`tube_radii_um`). A
    non-finite or non-positive entry falls back to :data:`TUBE_RADIUS_UM`.
    Each vessel is drawn at that radius along its middle, easing at each end
    to its node's: its own at a terminal, the two vessels' length-weighted
    mean where two meet, and at a junction no wider than the widest of the
    others there.

    *bridges* (one flag per row) marks zero-resistance bridges, which are not
    vessel: they are left out of the vessels' tubes and drawn apart, each run
    of joined rows (a dash, as the vessels layer cuts them) as its own
    flat-ended piece, :data:`BRIDGE_RADIUS_FRACTION` of the radius of the
    narrowest vessel the bridge meets. A vessel's open end where a bridge
    starts is cut square too, else its rounding hides the first dash.

    Returns ``(vertices, faces, segment_index)``: ``segment_index[i]`` is
    the Vectors row ``vertices[i]`` was drawn for, so per-row colours can be
    repeated onto the mesh.
    """
    data = np.asarray(vectors, dtype=float)
    if data.size == 0:
        return _empty_mesh()
    if data.ndim != 3 or data.shape[1:] != (2, 3):
        raise ValueError(f"expected Vectors data of shape (M, 2, 3); got {data.shape!r}")
    sides = int(sides)
    if sides < 3:
        raise ValueError(f"sides must be >= 3; got {sides}")
    radii = _step_radii(radius, data.shape[0])
    flagged = _step_flags(bridges, data.shape[0])
    labels = None if groups is None else np.asarray(groups)
    if not flagged.any():
        return _tubes(data, radii, sides, labels)

    vessel_rows, bridge_rows = np.flatnonzero(~flagged), np.flatnonzero(flagged)
    bridge_labels = (
        labels[bridge_rows] if labels is not None
        else np.cumsum(np.r_[True, np.diff(bridge_rows) > 1])
    )
    bridge_radii = BRIDGE_RADIUS_FRACTION * _bridged_radius(
        data, radii, vessel_rows, bridge_rows, bridge_labels
    )
    pieces = [
        (vessel_rows, _tubes(
            data[vessel_rows], radii[vessel_rows], sides,
            None if labels is None else labels[vessel_rows],
            flat_at=np.concatenate([data[bridge_rows, 0], data[bridge_rows].sum(axis=1)]),
        )),
        (bridge_rows, _tubes(
            data[bridge_rows], bridge_radii, sides, bridge_labels, flat_ends=True
        )),
    ]
    vertices, faces, index, offset = [], [], [], 0
    for rows, (piece_vertices, piece_faces, piece_index) in pieces:
        vertices.append(piece_vertices)
        faces.append(piece_faces + offset)
        index.append(rows[piece_index])
        offset += len(piece_vertices)
    return (
        np.concatenate(vertices),
        np.concatenate(faces).astype(np.intp),
        np.concatenate(index).astype(np.intp),
    )


def _empty_mesh() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return _EMPTY_VERTICES.copy(), _EMPTY_FACES.copy(), _EMPTY_INDEX.copy()


def _bridged_radius(
    data: np.ndarray,
    radii: np.ndarray,
    vessel_rows: np.ndarray,
    bridge_rows: np.ndarray,
    bridge_labels: np.ndarray,
) -> np.ndarray:
    """For each of *bridge_rows*, the radius of the narrowest vessel its
    bridge (its rows sharing a label) meets at a row's end, else its own."""

    def row_ends(rows):
        return np.concatenate([data[rows, 0, :], data[rows, 0, :] + data[rows, 1, :]])

    node = _node_ids(np.concatenate([row_ends(vessel_rows), row_ends(bridge_rows)]))
    at_vessel, at_bridge = node[: 2 * len(vessel_rows)], node[2 * len(vessel_rows) :]
    narrowest = np.full(int(node.max()) + 1, np.inf)
    np.minimum.at(narrowest, at_vessel, np.tile(radii[vessel_rows], 2))
    met = narrowest[at_bridge].reshape(2, -1).min(axis=0)
    bridge = np.unique(bridge_labels, return_inverse=True)[1].reshape(-1)
    met_by_bridge = np.full(int(bridge.max()) + 1, np.inf)
    np.minimum.at(met_by_bridge, bridge, met)
    met = met_by_bridge[bridge]
    return np.where(np.isfinite(met), met, radii[bridge_rows])


def _tubes(
    data: np.ndarray,
    radii: np.ndarray,
    sides: int,
    groups: np.ndarray | None,
    *,
    flat_ends: bool = False,
    flat_at: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """:func:`tubes_from_vectors` for checked data and one radius per row.

    Every end is rounded, or with *flat_ends* cut square; so is an open end
    on one of the points *flat_at*, where a vessel stops and a bridge
    starts, which the rounding would hide the bridge's first dash in.
    """
    empty = _empty_mesh()
    if data.size == 0:
        return empty
    lengths = np.linalg.norm(data[:, 1, :], axis=1)
    keep = np.flatnonzero(np.isfinite(lengths) & (lengths > 0.0))
    n = int(keep.size)
    if n == 0:
        return empty
    origin = data[keep, 0, :]
    end = origin + data[keep, 1, :]
    step_length = lengths[keep]
    radii = radii[keep]

    # Does step i carry on the vessel of step i - 1?
    joins = np.zeros(n, dtype=bool)
    if n > 1:
        joins[1:] = np.all(np.abs(origin[1:] - end[:-1]) <= _JOIN_TOLERANCE_UM, axis=1)
        joins[1:] &= radii[1:] == radii[:-1]
        if groups is not None:
            labels = np.asarray(groups)[keep]
            joins[1:] &= labels[1:] == labels[:-1]
    starts = ~joins
    ends = np.append(starts[1:], True)
    vessel_of_step = np.cumsum(starts) - 1
    n_vessels = int(starts.sum())

    # Each vessel's centreline: its steps' starts, then its last step's end.
    point_of_step = np.arange(n) + np.cumsum(ends) - ends
    first_point = point_of_step[starts]
    last_point = point_of_step[ends] + 1
    points = np.empty((n + n_vessels, 3))
    points[point_of_step] = origin
    points[last_point] = end[ends]
    step_of_point = np.empty(n + n_vessels, dtype=np.intp)
    step_of_point[point_of_step] = np.arange(n)
    gap = np.linalg.norm(np.diff(points, axis=0), axis=1)
    gap[last_point[:-1]] = 1.0  # between vessels; no ring falls in it
    along = np.concatenate([[0.0], np.cumsum(gap)])
    vessel_start = along[first_point]
    vessel_length = along[last_point] - vessel_start
    own_radius = np.bincount(vessel_of_step, weights=radii * step_length) / np.bincount(
        vessel_of_step, weights=step_length
    )

    def on_centreline(at):
        return np.stack([np.interp(at, along, points[:, axis]) for axis in range(3)], axis=1)

    # Vessel ends: every vessel's start, then every vessel's end.
    tips = np.concatenate([points[first_point], points[last_point]])
    node = _node_ids(tips)
    degree = np.bincount(node)
    at_node = degree[node]
    order = np.argsort(node, kind="stable")
    first_at_node = np.concatenate([[0], np.cumsum(degree)[:-1]])
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order)) - first_at_node[node[order]]
    partner = np.full(2 * n_vessels, -1, dtype=np.intp)
    paired = np.flatnonzero(at_node == 2)
    partner[paired] = order[first_at_node[node[paired]] + 1 - rank[paired]]

    chain_vessels, flipped, chain_first = _walk_chains(partner, n_vessels)
    n_tubes = len(chain_first)
    entry_chain = np.repeat(np.arange(n_tubes), np.diff(np.append(chain_first, n_vessels)))
    chain_of_vessel = np.empty(n_vessels, dtype=np.intp)
    chain_of_vessel[chain_vessels] = entry_chain

    # The radius each vessel is drawn at: within a factor of the
    # length-weighted median of its tube and of the vessels carrying the
    # tube straight on.
    chain_last = np.append(chain_first[1:], n_vessels) - 1
    head = chain_vessels[chain_first] + np.where(flipped[chain_first], n_vessels, 0)
    tail = chain_vessels[chain_last] + np.where(flipped[chain_last], 0, n_vessels)
    tube_ends = np.concatenate([head, tail])
    junction_ends = np.flatnonzero(at_node >= 3)
    junction_vessel = junction_ends % n_vessels
    reach = np.minimum(vessel_length[junction_vessel] / 2.0, own_radius[junction_vessel])
    inward = np.zeros((2 * n_vessels, 3))
    inward[junction_ends] = on_centreline(
        vessel_start[junction_vessel]
        + np.where(junction_ends < n_vessels, reach, vessel_length[junction_vessel] - reach)
    ) - tips[junction_ends]
    inward /= np.maximum(np.linalg.norm(inward, axis=1, keepdims=True), 1e-12)
    asked, through = _through_neighbours(tube_ends, inward, node, at_node, order, first_at_node)
    through_vessel = through % n_vessels
    through_chain = np.tile(np.arange(n_tubes), 2)[asked]
    chain_length = np.bincount(chain_of_vessel, weights=vessel_length, minlength=n_tubes)
    median = _weighted_median(
        np.concatenate([chain_of_vessel, through_chain]),
        np.concatenate([own_radius, own_radius[through_vessel]]),
        np.concatenate([
            vessel_length,
            np.minimum(vessel_length[through_vessel], chain_length[through_chain]),
        ]),
        n_tubes,
    )[chain_of_vessel]
    drawn = np.clip(own_radius, median / _DIAMETER_CLIP_FACTOR, median * _DIAMETER_CLIP_FACTOR)

    # The radius each vessel eases to at each of its ends.
    end_radius = np.tile(drawn, 2)
    end_length = np.tile(vessel_length, 2)
    target = end_radius.copy()
    other = partner[paired]
    target[paired] = (
        end_radius[paired] * end_length[paired] + end_radius[other] * end_length[other]
    ) / (end_length[paired] + end_length[other])
    if junction_ends.size:
        widest_first = junction_ends[
            np.lexsort((-end_radius[junction_ends], node[junction_ends]))
        ]
        lead = np.flatnonzero(np.r_[True, node[widest_first][1:] != node[widest_first][:-1]])
        widest = np.zeros(len(degree))
        runner_up = np.zeros(len(degree))
        widest[node[widest_first[lead]]] = end_radius[widest_first[lead]]
        runner_up[node[widest_first[lead + 1]]] = end_radius[widest_first[lead + 1]]
        is_widest = np.zeros(2 * n_vessels, dtype=bool)
        is_widest[widest_first[lead]] = True
        others = np.where(
            is_widest[junction_ends],
            runner_up[node[junction_ends]],
            widest[node[junction_ends]],
        )
        target[junction_ends] = np.minimum(end_radius[junction_ends], others)
    start_target, end_target = target[:n_vessels], target[n_vessels:]
    ramp = np.minimum(vessel_length / 2.0, 2.0 * drawn)

    # Rings along each vessel a fixed fraction of the local radius apart,
    # worked out on samples along it.
    sample_s = vessel_length[:, None] * np.linspace(0.0, 1.0, _SPACING_SAMPLES + 1)[None, :]
    density = 1.0 / (
        _RING_SPACING_PER_RADIUS
        * _ramped_radius(
            drawn[:, None], start_target[:, None], end_target[:, None],
            vessel_length[:, None], ramp[:, None], sample_s,
        )
    )
    rings_along = np.concatenate(
        [
            np.zeros((n_vessels, 1)),
            np.cumsum(
                0.5 * (density[:, 1:] + density[:, :-1]) * (sample_s[:, 1:] - sample_s[:, :-1]),
                axis=1,
            ),
        ],
        axis=1,
    )
    total = rings_along[:, -1]
    segments = np.maximum(_MIN_SEGMENTS_PER_VESSEL, np.ceil(total - 1e-9)).astype(np.intp)
    # One increasing table over every vessel: vessel v's fractions sit in [2v, 2v + 1].
    warp = (2.0 * np.arange(n_vessels))[:, None] + rings_along / total[:, None]

    # The rings in tube order; a vessel after the first in its tube shares
    # its first ring with the end of the one before.
    opens = np.zeros(n_vessels, dtype=bool)
    opens[chain_first] = True
    entry_segments = segments[chain_vessels]
    count = entry_segments + opens
    ring_entry = np.repeat(np.arange(n_vessels), count)
    k = (
        np.arange(len(ring_entry))
        - np.repeat(np.cumsum(count) - count, count)
        + (~opens)[ring_entry]
    )
    seg = entry_segments[ring_entry]
    k = np.where(flipped[ring_entry], seg - k, k)
    ring_vessel = chain_vessels[ring_entry]
    s = np.interp(2.0 * ring_vessel + k / seg, warp.ravel(), sample_s.ravel())
    at = vessel_start[ring_vessel] + s
    centre = on_centreline(at)
    # The step each ring lies on gives it its colour.
    on_point = np.clip(
        np.searchsorted(along, at, side="right") - 1,
        first_point[ring_vessel],
        last_point[ring_vessel] - 1,
    )
    ring_step = step_of_point[on_point]
    ring_radius = _ramped_radius(
        drawn[ring_vessel], start_target[ring_vessel], end_target[ring_vessel],
        vessel_length[ring_vessel], ramp[ring_vessel], s,
    )
    ring_chain = entry_chain[ring_entry]
    rings_in_tube = np.bincount(ring_chain, minlength=n_tubes)
    first_ring = np.concatenate([[0], np.cumsum(rings_in_tube)[:-1]])
    last_ring = first_ring + rings_in_tube - 1
    first, last = first_ring[ring_chain], last_ring[ring_chain]

    smoothing = _smoothing_matrix(first, last)
    for _ in range(_SMOOTHING_PASSES):
        centre = smoothing @ centre
    # Only rings next to one that moved can bend any differently.
    candidates = np.arange(len(centre))
    for _ in range(_MAX_BEND_PASSES):
        bend = _bend_radius(centre, first, last, candidates)
        tight = candidates[bend < _TIGHT_BEND_PER_RADIUS * ring_radius[candidates]]
        if not tight.size:
            break
        rows = _around(tight, 2, len(centre))
        centre[rows] = smoothing[rows] @ centre
        candidates = _around(rows, 1, len(centre))

    # An end ring faces the ring two along, not the first step off its node.
    index = np.arange(len(centre))
    ahead = np.minimum(np.where(index == first, index + 2, index + 1), last)
    behind = np.maximum(np.where(index == last, index - 2, index - 1), first)
    tangent = centre[ahead] - centre[behind]
    norm = np.linalg.norm(tangent, axis=1)
    still = norm <= 1e-12  # a tube folded onto itself: take its step's own way
    tangent[still] = data[keep[ring_step[still]], 1, :]
    norm[still] = step_length[ring_step[still]]
    tangent /= norm[:, None]

    # Each ring narrowed where it would reach through the plane of the next.
    has_next = index < last
    before = index[has_next]
    step = centre[before + 1] - centre[before]
    sine = np.linalg.norm(np.cross(tangent[before], tangent[before + 1]), axis=1)
    bent = sine > 1e-12
    clear_ahead = np.full(len(before), np.inf)
    clear_behind = np.full(len(before), np.inf)
    clear_ahead[bent] = np.einsum("ij,ij->i", step[bent], tangent[before][bent]) / sine[bent]
    clear_behind[bent] = np.einsum("ij,ij->i", step[bent], tangent[before + 1][bent]) / sine[bent]
    ring_radius[before + 1] = np.minimum(ring_radius[before + 1], _FOLD_MARGIN * clear_ahead)
    ring_radius[before] = np.minimum(ring_radius[before], _FOLD_MARGIN * clear_behind)
    ring_radius = np.maximum(ring_radius, _MIN_RADIUS_FRACTION * drawn[ring_vessel])

    # Carry each ring's frame on from the last one's, so the tube does not
    # twist: theta is how far round a ring's own frame its first vertex sits.
    normal, binormal = _normal_plane_frames(tangent)
    turn = np.zeros(len(centre))
    turn[before + 1] = np.arctan2(
        np.einsum("ij,ij->i", normal[before], binormal[before + 1]),
        np.einsum("ij,ij->i", normal[before], normal[before + 1]),
    )
    theta = np.cumsum(turn)
    theta -= theta[first]
    angles = np.linspace(0.0, 2.0 * np.pi, sides, endpoint=False)[None, :] + theta[:, None]
    around = (
        np.cos(angles)[:, :, None] * normal[:, None, :]
        + np.sin(angles)[:, :, None] * binormal[:, None, :]
    )
    ring_vertices = (centre[:, None, :] + around * ring_radius[:, None, None]).reshape(-1, 3)

    k = np.arange(sides, dtype=np.intp)[None, :]

    def band(inner, outer, outward=True):
        """Faces joining two rings of vertex indices, ``(count, sides)`` each."""
        a, b = inner, np.roll(inner, -1, axis=1)
        c, d = outer, np.roll(outer, -1, axis=1)
        if outward:
            triangles = [np.stack([a, b, d], axis=-1), np.stack([a, d, c], axis=-1)]
        else:
            triangles = [np.stack([a, d, b], axis=-1), np.stack([a, c, d], axis=-1)]
        return np.stack(triangles, axis=2).reshape(-1, 3)

    ring_index = index[:, None] * sides + k
    faces = [band(ring_index[before], ring_index[before + 1])]

    # Rounded ends: a hemisphere of latitude rings on each end ring, closed
    # by a pole on the tube's own course; flat, those rings and the pole
    # lie in the end ring's own plane.
    levels = max(2, sides // 4)
    rims = np.concatenate([first_ring, last_ring])
    rise = np.full(len(rims), 0.0 if flat_ends else 1.0)
    if flat_at is not None and len(flat_at):
        at = _node_ids(np.concatenate([tips[tube_ends], flat_at]))
        rise[np.isin(at[: len(tube_ends)], at[len(tube_ends) :]) & (at_node[tube_ends] == 1)] = 0.0
    outward = np.concatenate([-tangent[first_ring], tangent[last_ring]])
    cap_radius = ring_radius[rims]
    latitude = 0.5 * np.pi * np.arange(1, levels) / levels
    cap_vertices = (
        centre[rims][:, None, None, :]
        + outward[:, None, None, :]
        * ((rise * cap_radius)[:, None] * np.sin(latitude))[:, :, None, None]
        + around[rims][:, None, :, :] * (cap_radius[:, None] * np.cos(latitude))[:, :, None, None]
    ).reshape(-1, 3)
    poles = centre[rims] + outward * (rise * cap_radius)[:, None]
    n_caps = rims.size
    cap_base = len(ring_vertices)
    pole_base = cap_base + len(cap_vertices)
    cap_index = (
        cap_base + (np.arange(n_caps)[:, None] * (levels - 1) + np.arange(levels - 1)[None, :])
        * sides
    )[:, :, None] + k[None]  # (caps, levels - 1, sides)
    is_end = np.arange(n_caps) >= n_tubes
    rows = [ring_index[rims]] + [cap_index[:, level] for level in range(levels - 1)]
    for inner, outer in zip(rows[:-1], rows[1:]):
        faces.append(band(inner[is_end], outer[is_end]))
        faces.append(band(inner[~is_end], outer[~is_end], outward=False))
    top = rows[-1]
    pole = np.broadcast_to((pole_base + np.arange(n_caps))[:, None], top.shape)
    turned = np.roll(top, -1, axis=1)
    fans = np.where(
        is_end[:, None, None],
        np.stack([top, turned, pole], axis=-1),
        np.stack([top, pole, turned], axis=-1),
    )
    faces.append(fans.reshape(-1, 3))

    cap_step = ring_step[rims]
    vertices = np.concatenate([ring_vertices, cap_vertices, poles])
    segment_index = np.concatenate([
        np.repeat(keep[ring_step], sides),
        np.repeat(keep[cap_step], (levels - 1) * sides),
        keep[cap_step],
    ]).astype(np.intp)
    return vertices, np.concatenate(faces).astype(np.intp), segment_index


def clamp_tube_quality(quality) -> int:
    """*quality* as a valid level, 0 to ``len(TUBE_QUALITY_SIDES) - 1``."""
    try:
        level = int(quality)
    except (TypeError, ValueError):
        level = DEFAULT_TUBE_QUALITY
    return min(max(level, 0), len(TUBE_QUALITY_SIDES) - 1)


def valid_tube_diameter(diameter) -> str:
    """*diameter* if it is one of :data:`TUBE_DIAMETERS`, else the default."""
    return diameter if diameter in TUBE_DIAMETERS else DEFAULT_TUBE_DIAMETER


def valid_uniform_tube_diameter_um(diameter_um) -> float:
    """*diameter_um* held to :data:`UNIFORM_TUBE_DIAMETER_RANGE_UM`; the
    default for anything that is not a number."""
    try:
        value = float(diameter_um)
    except (TypeError, ValueError):
        return DEFAULT_UNIFORM_TUBE_DIAMETER_UM
    if not np.isfinite(value):
        return DEFAULT_UNIFORM_TUBE_DIAMETER_UM
    low, high = UNIFORM_TUBE_DIAMETER_RANGE_UM
    return min(max(value, low), high)


def tube_mesh(
    vectors: np.ndarray,
    *,
    radius: float | np.ndarray = TUBE_RADIUS_UM,
    quality: int = DEFAULT_TUBE_QUALITY,
    groups: np.ndarray | None = None,
    bridges: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """:func:`tubes_from_vectors` with one quality level's sides
    (see :data:`TUBE_QUALITY_SIDES`)."""
    sides = TUBE_QUALITY_SIDES[clamp_tube_quality(quality)]
    return tubes_from_vectors(
        vectors, radius=radius, sides=sides, groups=groups, bridges=bridges
    )


def _feature_column(features: Any, name: str) -> np.ndarray | None:
    """One column of a layer's features (a DataFrame or a mapping), if it has it."""
    if features is None:
        return None
    try:
        if name not in features:
            return None
        return np.asarray(features[name])
    except (KeyError, TypeError, ValueError):
        return None


def vessel_tube_mesh(
    vectors: np.ndarray,
    features: Any = None,
    *,
    quality: int = DEFAULT_TUBE_QUALITY,
    edge_width: float | None = None,
    diameter: str = DEFAULT_TUBE_DIAMETER,
    uniform_diameter_um: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """:func:`tube_mesh` for a vessels Vectors layer's data and features.

    Each vessel at its own ``diameter_um`` (or, with none, the
    :func:`tube_radius_um` of *edge_width*), its steps told apart from the
    next vessel's by ``edge_index``, and the
    :data:`~haemolynx.graph.IS_ZERO_RESISTANCE` column marking bridges.
    With *diameter* :data:`TUBE_DIAMETER_UNIFORM`, every vessel is drawn
    *uniform_diameter_um* across whatever its ``diameter_um`` -- or, with no
    width given, at that one :func:`tube_radius_um`.
    """
    diameters = _feature_column(features, "diameter_um")
    radii = None
    if valid_tube_diameter(diameter) == TUBE_DIAMETER_UNIFORM:
        diameters = None
        if uniform_diameter_um is not None:
            radii = valid_uniform_tube_diameter_um(uniform_diameter_um) / 2.0
    if diameters is not None:
        try:
            radii = tube_radii_um(diameters.astype(float))
        except (TypeError, ValueError):
            radii = None
    return tube_mesh(
        vectors,
        radius=radii if radii is not None else tube_radius_um(edge_width),
        quality=quality,
        groups=_feature_column(features, "edge_index"),
        bridges=_feature_column(features, IS_ZERO_RESISTANCE),
    )


def colors_for_tube_vertices(
    segment_index: np.ndarray, segment_colors: np.ndarray
) -> np.ndarray:
    """Repeat per-segment RGBA (or RGB) onto the tube vertices."""
    index = np.asarray(segment_index, dtype=int)
    colours = np.asarray(segment_colors, dtype=float)
    if index.size == 0:
        cols = colours.shape[1] if colours.ndim == 2 and colours.shape[-1] else 4
        return np.empty((0, int(cols)), dtype=float)
    return colours[index]
