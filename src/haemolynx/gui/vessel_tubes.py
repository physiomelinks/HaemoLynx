"""Vessel tubes from Vectors origin+direction data.

Napari Vectors ``vector_style="line"`` draws two world-fixed ribbons. An
axis-aligned centreline step collapses a ribbon, and ``edge_width=0.6`` µm
then vanishes edge-on. A tube stays visible from every camera angle.

A vessel has one diameter, so it is drawn as one smooth tube of that
diameter. The centreline it follows is the skeleton's voxel path: steps
under a micron that kink at every voxel, round a capillary six microns
across. A tube built ring by ring on that path pinches, bulges and folds
where the vessel does none of those things. So each tube follows its
centreline smoothed over about half its own radius, with rings spaced in
proportion to that radius, ends kept on the vessel's two nodes and rounded,
so vessels meeting at a node join there.

Nothing here imports napari. The widget hides the Vectors visual and shows a
Surface built from these arrays; hover and colour-by still read the Vectors.
"""
from __future__ import annotations

import numpy as np

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
#: Rings along a tube are this fraction of its radius apart.
_RING_SPACING_PER_RADIUS = 1.0 / 3.0
#: The ring centres are smoothed this many times with a 1-4-6-4-1 kernel:
#: three passes at a third of a radius apart is about half a radius.
_SMOOTHING_PASSES = 3
_KERNEL = ((-2, 1.0), (-1, 4.0), (0, 6.0), (1, 4.0), (2, 1.0))
#: Two steps join when one ends this close to where the next starts (µm).
_JOIN_TOLERANCE_UM = 1e-6

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


def _smooth_tubes(centre: np.ndarray, first: np.ndarray, last: np.ndarray) -> np.ndarray:
    """Each tube's ring centres smoothed along it, its two end rings left in place.

    *first* and *last* give, for every ring, its tube's first and last ring.
    Past an end the tube is continued by reflection through that end, which
    carries a straight tube on straight and leaves the end where it was.
    """
    from scipy.sparse import csr_matrix

    count = len(centre)
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
    smoothing = csr_matrix(
        (np.concatenate(weights), (np.concatenate(rows), np.concatenate(cols))),
        shape=(count, count),
    )
    for _ in range(_SMOOTHING_PASSES):
        centre = smoothing @ centre
    return centre


def tubes_from_vectors(
    vectors: np.ndarray,
    *,
    radius: float | np.ndarray = TUBE_RADIUS_UM,
    sides: int = TUBE_QUALITY_SIDES[DEFAULT_TUBE_QUALITY],
    groups: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One smooth, closed tube with rounded ends per run of joined steps.

    *vectors* is ``(M, 2, 3)`` origin+direction data. Consecutive rows join
    when one ends where the next starts and, given *groups* (one label per
    row, e.g. each step's ``edge_index``), both carry the same label -- so
    two vessels meeting at a node stay two tubes. Zero-length and non-finite
    rows are skipped.

    *radius* is one value for every row, or one per row -- e.g. each
    vessel's own measured/set diameter halved (see :func:`tube_radii_um`). A
    non-finite or non-positive entry falls back to :data:`TUBE_RADIUS_UM`.

    Returns ``(vertices, faces, segment_index)``: ``segment_index[i]`` is
    the Vectors row ``vertices[i]`` was drawn for, so per-row colours can be
    repeated onto the mesh.
    """
    empty = (_EMPTY_VERTICES.copy(), _EMPTY_FACES.copy(), _EMPTY_INDEX.copy())
    data = np.asarray(vectors, dtype=float)
    if data.size == 0:
        return empty
    if data.ndim != 3 or data.shape[1:] != (2, 3):
        raise ValueError(f"expected Vectors data of shape (M, 2, 3); got {data.shape!r}")
    sides = int(sides)
    if sides < 3:
        raise ValueError(f"sides must be >= 3; got {sides}")
    radii = _step_radii(radius, data.shape[0])

    lengths = np.linalg.norm(data[:, 1, :], axis=1)
    keep = np.flatnonzero(np.isfinite(lengths) & (lengths > 0.0))
    n = int(keep.size)
    if n == 0:
        return empty
    origin = data[keep, 0, :]
    end = origin + data[keep, 1, :]
    step_length = lengths[keep]
    radii = radii[keep]

    # Does step i carry on from step i - 1?
    joins = np.zeros(n, dtype=bool)
    if n > 1:
        joins[1:] = np.all(np.abs(origin[1:] - end[:-1]) <= _JOIN_TOLERANCE_UM, axis=1)
        if groups is not None:
            labels = np.asarray(groups)[keep]
            joins[1:] &= labels[1:] == labels[:-1]
    starts = ~joins
    ends = np.append(starts[1:], True)
    tube_of_step = np.cumsum(starts) - 1
    n_tubes = int(starts.sum())

    # Each tube's centreline: its steps' starts, then its last step's end.
    point_of_step = np.arange(n) + np.cumsum(ends) - ends
    last_point = point_of_step[ends] + 1
    points = np.empty((n + n_tubes, 3))
    points[point_of_step] = origin
    points[last_point] = end[ends]
    step_of_point = np.empty(n + n_tubes, dtype=np.intp)
    step_of_point[point_of_step] = np.arange(n)
    gap = np.linalg.norm(np.diff(points, axis=0), axis=1)
    gap[last_point[:-1]] = 1.0  # between tubes; no ring falls in it
    along = np.concatenate([[0.0], np.cumsum(gap)])
    tube_start = along[point_of_step[starts]]
    tube_length = along[last_point] - tube_start

    # Rings evenly along each tube, a fixed fraction of its radius apart.
    tube_radius = np.bincount(tube_of_step, weights=radii * step_length) / np.bincount(
        tube_of_step, weights=step_length
    )
    segments = np.maximum(
        1, np.ceil(tube_length / (tube_radius * _RING_SPACING_PER_RADIUS))
    ).astype(np.intp)
    ring_tube = np.repeat(np.arange(n_tubes), segments + 1)
    first_ring = np.concatenate([[0], np.cumsum(segments + 1)[:-1]])
    last_ring = first_ring + segments
    rank = np.arange(len(ring_tube)) - first_ring[ring_tube]
    at = tube_start[ring_tube] + tube_length[ring_tube] * rank / segments[ring_tube]
    centre = np.stack([np.interp(at, along, points[:, axis]) for axis in range(3)], axis=1)
    # The step each ring lies on gives it its radius and its colour.
    on_point = np.minimum(
        np.searchsorted(along, at, side="right") - 1, last_point[ring_tube] - 1
    )
    ring_step = step_of_point[on_point]
    ring_radius = radii[ring_step]
    first, last = first_ring[ring_tube], last_ring[ring_tube]
    centre = _smooth_tubes(centre, first, last)

    index = np.arange(len(centre))
    tangent = centre[np.minimum(index + 1, last)] - centre[np.maximum(index - 1, first)]
    norm = np.linalg.norm(tangent, axis=1)
    still = norm <= 1e-12  # a tube folded onto itself: take its step's own way
    tangent[still] = data[keep[ring_step[still]], 1, :]
    norm[still] = step_length[ring_step[still]]
    tangent /= norm[:, None]

    # Carry each ring's frame on from the last one's, so the tube does not
    # twist: theta is how far round a ring's own frame its first vertex sits.
    normal, binormal = _normal_plane_frames(tangent)
    has_next = index < last
    before = index[has_next]
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
    # by a pole on the tube's own course.
    levels = max(2, sides // 4)
    rims = np.concatenate([first_ring, last_ring])
    outward = np.concatenate([-tangent[first_ring], tangent[last_ring]])
    cap_radius = ring_radius[rims]
    latitude = 0.5 * np.pi * np.arange(1, levels) / levels
    cap_vertices = (
        centre[rims][:, None, None, :]
        + outward[:, None, None, :] * (cap_radius[:, None] * np.sin(latitude))[:, :, None, None]
        + around[rims][:, None, :, :] * (cap_radius[:, None] * np.cos(latitude))[:, :, None, None]
    ).reshape(-1, 3)
    poles = centre[rims] + outward * cap_radius[:, None]
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


def tube_mesh(
    vectors: np.ndarray,
    *,
    radius: float | np.ndarray = TUBE_RADIUS_UM,
    quality: int = DEFAULT_TUBE_QUALITY,
    groups: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """:func:`tubes_from_vectors` with one quality level's sides
    (see :data:`TUBE_QUALITY_SIDES`)."""
    sides = TUBE_QUALITY_SIDES[clamp_tube_quality(quality)]
    return tubes_from_vectors(vectors, radius=radius, sides=sides, groups=groups)


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
