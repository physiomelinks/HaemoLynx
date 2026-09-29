"""Per-segment vessel tubes from Vectors origin+direction data.

Napari Vectors ``vector_style="line"`` draws two world-fixed ribbons. An
axis-aligned centreline step collapses a ribbon, and ``edge_width=0.6`` µm
then vanishes edge-on. A short N-gon prism per segment, built in that
segment's normal plane, stays visible from every camera angle.

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
DEFAULT_TUBE_SIDES = 6

#: napari Surface shading the tubes layer starts with. Flat lights each face
#: on its own, so a tube's sides and bends read as 3D; ``"none"`` draws every
#: face the same flat colour. Only the starting value: a stage that redraws
#: the tubes keeps whatever the user has since chosen in the layer controls.
TUBE_SHADING = "flat"

#: How round and how joined the tubes are drawn, by the tubes layer's "Render
#: quality" slider. Level 0 is the original drawing -- one separate hexagonal
#: prism per centreline step, flat shaded -- which is cheap but reads as a
#: stack of bands: each prism has its own twist and nothing joins them. Every
#: level above it draws one continuous tube per vessel, its rings mitred at
#: each bend and matched round the tube so it does not twist, smooth shaded,
#: with more sides each step.
TUBE_QUALITY_SIDES = (6, 8, 12, 18, 32)
DEFAULT_TUBE_QUALITY = 0
#: A sharp bend's mitred ring is stretched along the bend to keep the tube's
#: width; past this it would spike outward instead.
_MAX_MITRE_STRETCH = 3.0
#: Two steps join when one ends this close to where the next starts (µm).
_JOIN_TOLERANCE_UM = 1e-6

VESSEL_DRAW_TUBES = "tubes"
VESSEL_DRAW_LINES = "lines"
DEFAULT_VESSEL_DRAW = VESSEL_DRAW_TUBES

_EMPTY_VERTICES = np.empty((0, 3), dtype=float)
_EMPTY_FACES = np.empty((0, 3), dtype=np.intp)
_EMPTY_INDEX = np.empty((0,), dtype=np.intp)


def tube_radius_um(edge_width: float | None = None) -> float:
    """Radius used for the prism mesh: at least :data:`TUBE_RADIUS_UM`."""
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


def tubes_from_vectors(
    vectors: np.ndarray,
    *,
    radius: float | np.ndarray = TUBE_RADIUS_UM,
    sides: int = DEFAULT_TUBE_SIDES,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build independent N-gon prisms from ``(M, 2, 3)`` origin+direction data.

    Returns ``(vertices, faces, segment_index)``. ``segment_index[i]`` is the
    Vectors row that owns ``vertices[i]``, so per-segment colours can be
    repeated onto the prism. Zero-length and non-finite segments are skipped.
    Joins are not mitred: consecutive steps of a polyline produce disjoint
    vertex sets whose end/start rings abut when the steps share a point.

    *radius* is either one value for every segment, or an array of shape
    ``(M,)`` matching *vectors*' own row count -- one radius per segment,
    e.g. each vessel's own measured/set diameter halved (see
    :func:`tube_radii_um`). A non-finite or non-positive entry in a
    per-segment array falls back to :data:`TUBE_RADIUS_UM` rather than
    breaking that segment's prism.
    """
    empty = (_EMPTY_VERTICES.copy(), _EMPTY_FACES.copy(), _EMPTY_INDEX.copy())
    data = np.asarray(vectors, dtype=float)
    if data.size == 0:
        return empty
    if data.ndim != 3 or data.shape[1:] != (2, 3):
        raise ValueError(
            f"expected Vectors data of shape (M, 2, 3); got {data.shape!r}"
        )
    sides = int(sides)
    if sides < 3:
        raise ValueError(f"sides must be >= 3; got {sides}")

    radius_arr = np.asarray(radius, dtype=float)
    per_segment_radius: np.ndarray | None
    scalar_radius = 0.0
    if radius_arr.ndim == 0:
        if not np.isfinite(radius_arr) or radius_arr <= 0.0:
            raise ValueError(f"radius must be a positive finite number; got {radius}")
        per_segment_radius = None
        scalar_radius = float(radius_arr)
    else:
        if radius_arr.shape != (data.shape[0],):
            raise ValueError(
                "radius array must have one entry per vectors row "
                f"({data.shape[0]},); got shape {radius_arr.shape!r}"
            )
        per_segment_radius = radius_arr

    origins = data[:, 0, :]
    directions = data[:, 1, :]
    lengths = np.linalg.norm(directions, axis=1)
    keep = np.flatnonzero(np.isfinite(lengths) & (lengths > 0.0))
    n_keep = int(keep.size)
    if n_keep == 0:
        return empty

    verts_per = sides * 2

    origin = origins[keep]  # (n_keep, 3)
    direction = directions[keep]  # (n_keep, 3)
    tangent = direction / lengths[keep, None]
    normal, binormal = _normal_plane_frames(tangent)

    if per_segment_radius is None:
        kept_radius = np.full(n_keep, scalar_radius)
    else:
        kept_radius = per_segment_radius[keep].copy()
        invalid = ~np.isfinite(kept_radius) | (kept_radius <= 0.0)
        kept_radius[invalid] = TUBE_RADIUS_UM

    angles = np.linspace(0.0, 2.0 * np.pi, sides, endpoint=False)
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)
    # (n_keep, sides, 3): per-segment ring, one cross-section shared by both
    # the origin-end and direction-end rings of that segment's prism.
    ring = (
        cos_a[None, :, None] * normal[:, None, :]
        + sin_a[None, :, None] * binormal[:, None, :]
    ) * kept_radius[:, None, None]

    # verts_per = 2 * sides per segment: ring at the origin, then the ring at
    # origin + direction -- matches the base/base+sides layout faces indexes
    # into below, and what segment_index/vertices consumers expect.
    rings = np.stack([origin[:, None, :] + ring, (origin + direction)[:, None, :] + ring], axis=1)
    vertices = rings.reshape(n_keep * verts_per, 3)
    segment_index = np.repeat(keep.astype(np.intp), verts_per)

    base = np.arange(n_keep, dtype=np.intp)[:, None] * verts_per  # (n_keep, 1)
    k = np.arange(sides, dtype=np.intp)[None, :]  # (1, sides)
    nxt = (k + 1) % sides
    a = base + k
    b = base + nxt
    c = base + sides + k
    d = base + sides + nxt
    # Two triangles per side, in the same (a,b,d) then (a,d,c) order the
    # original per-segment loop emitted for each k, so faces line up with a
    # test fixture built against the scalar version.
    triangles = np.stack([np.stack([a, b, d], axis=-1), np.stack([a, d, c], axis=-1)], axis=2)
    faces = triangles.reshape(n_keep * sides * 2, 3)
    return vertices, faces, segment_index


def joined_tubes_from_vectors(vectors, *, radius=TUBE_RADIUS_UM, sides=12, groups=None):
    """One continuous, capped tube per run of joined steps.

    Consecutive Vectors rows join when one ends where the next starts and,
    given *groups* (one label per row, e.g. each step's ``edge_index``), both
    carry the same label -- so two vessels meeting at a node stay two tubes.
    Returns ``(vertices, faces, segment_index)`` like
    :func:`tubes_from_vectors`: each vertex belongs to the step that starts
    at its ring (the last ring and the end cap to the last step), so colours
    by step still apply, blending along a step towards the next one's.
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

    radius_arr = np.asarray(radius, dtype=float)
    if radius_arr.ndim == 0:
        if not np.isfinite(radius_arr) or radius_arr <= 0.0:
            raise ValueError(f"radius must be a positive finite number; got {radius}")
        radii = np.full(data.shape[0], float(radius_arr))
    else:
        if radius_arr.shape != (data.shape[0],):
            raise ValueError(
                "radius array must have one entry per vectors row "
                f"({data.shape[0]},); got shape {radius_arr.shape!r}"
            )
        radii = radius_arr.copy()
        radii[~np.isfinite(radii) | (radii <= 0.0)] = TUBE_RADIUS_UM

    lengths = np.linalg.norm(data[:, 1, :], axis=1)
    keep = np.flatnonzero(np.isfinite(lengths) & (lengths > 0.0))
    n = int(keep.size)
    if n == 0:
        return empty
    origin = data[keep, 0, :]
    end = origin + data[keep, 1, :]
    tangent = data[keep, 1, :] / lengths[keep, None]
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

    # Rings in drawing order: each step's start ring, then an end ring after
    # the last step of each tube.
    ring_of_start = np.arange(n) + np.cumsum(ends) - ends
    n_rings = n + int(ends.sum())
    ring_centre = np.empty((n_rings, 3))
    ring_tangent = np.empty((n_rings, 3))
    ring_radius = np.empty(n_rings)
    ring_step = np.empty(n_rings, dtype=np.intp)
    ring_centre[ring_of_start] = origin
    ring_radius[ring_of_start] = radii
    ring_step[ring_of_start] = np.arange(n)
    ring_tangent[ring_of_start] = tangent
    end_rings = ring_of_start[ends] + 1
    ring_centre[end_rings] = end[ends]
    ring_tangent[end_rings] = tangent[ends]
    ring_radius[end_rings] = radii[ends]
    ring_step[end_rings] = np.flatnonzero(ends)

    # A joint's ring lies in the plane halfway between its two steps, and is
    # stretched along the bend so the tube keeps its width round it.
    joint = np.flatnonzero(joins)
    stretch_dir = np.zeros((n_rings, 3))
    stretch = np.ones(n_rings)
    if joint.size:
        before, after = tangent[joint - 1], tangent[joint]
        bisector = before + after
        norm = np.linalg.norm(bisector, axis=1)
        turned = norm > 1e-9  # a step straight back on itself keeps its own plane
        rings = ring_of_start[joint]
        ring_tangent[rings[turned]] = bisector[turned] / norm[turned, None]
        ring_radius[rings] = 0.5 * (radii[joint - 1] + radii[joint])
        bend = after - before
        bend_norm = np.linalg.norm(bend, axis=1)
        bent = turned & (bend_norm > 1e-9)
        cos_half = np.einsum("ij,ij->i", ring_tangent[rings], after)
        stretch_dir[rings[bent]] = bend[bent] / bend_norm[bent, None]
        stretch[rings[bent]] = np.minimum(
            1.0 / np.maximum(cos_half[bent], 1e-9), _MAX_MITRE_STRETCH
        )

    normal, binormal = _normal_plane_frames(ring_tangent)
    angles = np.linspace(0.0, 2.0 * np.pi, sides, endpoint=False)
    offsets = (
        np.cos(angles)[None, :, None] * normal[:, None, :]
        + np.sin(angles)[None, :, None] * binormal[:, None, :]
    ) * ring_radius[:, None, None]
    along = np.einsum("rkj,rj->rk", offsets, stretch_dir)
    offsets += (along * (stretch - 1.0)[:, None])[:, :, None] * stretch_dir[:, None, :]
    ring_vertices = (ring_centre[:, None, :] + offsets).reshape(n_rings * sides, 3)

    # Side faces between each step's two rings. Each ring's frame is its own,
    # so turn the next ring's numbering to line up with this one's rather
    # than let the tube twist between them.
    a_ring = ring_of_start
    b_ring = ring_of_start + 1
    phi = np.arctan2(
        np.einsum("ij,ij->i", normal[a_ring], binormal[b_ring]),
        np.einsum("ij,ij->i", normal[a_ring], normal[b_ring]),
    )
    shift = np.rint(phi / (2.0 * np.pi / sides)).astype(np.intp)
    k = np.arange(sides, dtype=np.intp)[None, :]
    a = a_ring[:, None] * sides + k
    b = a_ring[:, None] * sides + (k + 1) % sides
    c = b_ring[:, None] * sides + (k + shift[:, None]) % sides
    d = b_ring[:, None] * sides + (k + 1 + shift[:, None]) % sides
    side_faces = np.stack(
        [np.stack([a, b, d], axis=-1), np.stack([a, d, c], axis=-1)], axis=2
    ).reshape(-1, 3)

    # Caps: a centre vertex fanned to each open end, facing outward.
    start_rings = ring_of_start[starts]
    cap_rings = np.concatenate([start_rings, end_rings])
    n_caps = cap_rings.size
    centre_index = n_rings * sides + np.arange(n_caps, dtype=np.intp)
    rim = cap_rings[:, None] * sides
    j, j1 = rim + k, rim + (k + 1) % sides
    centre = np.broadcast_to(centre_index[:, None], j.shape)
    is_start = np.arange(n_caps) < start_rings.size
    cap_faces = np.where(
        is_start[:, None, None],
        np.stack([centre, j1, j], axis=-1),
        np.stack([centre, j, j1], axis=-1),
    ).reshape(-1, 3)

    vertices = np.concatenate([ring_vertices, ring_centre[cap_rings]])
    faces = np.concatenate([side_faces, cap_faces]).astype(np.intp)
    segment_index = np.concatenate([
        np.repeat(keep[ring_step], sides),
        keep[ring_step[cap_rings]],
    ]).astype(np.intp)
    return vertices, faces, segment_index


def tube_shading_for_quality(quality: int) -> str:
    """The Surface shading a quality level is drawn with.

    Flat at level 0, where each prism is its own set of vertices and smooth
    shading would have nothing to smooth across; smooth above it, where the
    rings are shared and the light runs round the tube.
    """
    return TUBE_SHADING if clamp_tube_quality(quality) == 0 else "smooth"


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
    """The tube mesh for one quality level (see :data:`TUBE_QUALITY_SIDES`).

    Level 0 is :func:`tubes_from_vectors` exactly as it always drew; every
    level above is :func:`joined_tubes_from_vectors` with that level's sides.
    """
    level = clamp_tube_quality(quality)
    sides = TUBE_QUALITY_SIDES[level]
    if level == 0:
        return tubes_from_vectors(vectors, radius=radius, sides=sides)
    return joined_tubes_from_vectors(vectors, radius=radius, sides=sides, groups=groups)


def colors_for_tube_vertices(
    segment_index: np.ndarray, segment_colors: np.ndarray
) -> np.ndarray:
    """Repeat per-segment RGBA (or RGB) onto the prism vertices."""
    index = np.asarray(segment_index, dtype=int)
    colours = np.asarray(segment_colors, dtype=float)
    if index.size == 0:
        cols = colours.shape[1] if colours.ndim == 2 and colours.shape[-1] else 4
        return np.empty((0, int(cols)), dtype=float)
    return colours[index]
