"""Per-segment vessel tubes from Vectors origin+direction data.

Napari Vectors ``vector_style="line"`` draws two world-fixed ribbons. An
axis-aligned centreline step collapses a ribbon, and ``edge_width=0.6`` µm
then vanishes edge-on. A short N-gon prism per segment stays visible from
every camera angle.

A vessel has one diameter, so its tube is drawn as one: every ring is a
circle of the vessel's own radius, facing the way the vessel runs over that
radius rather than the way one centreline step does. A step is under a
micron and a capillary's radius three, so a ring set square to its own step
tilts with every voxel kink and the tube pinches and bulges where the
vessel does not.

Nothing here imports napari. The widget hides the Vectors visual and shows a
Surface built from these arrays; hover and colour-by still read the Vectors.
"""
from __future__ import annotations

from typing import NamedTuple

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
#: quality" slider. Level 0 is one separate hexagonal prism per centreline
#: step, flat shaded -- cheap, but it reads as a stack of bands. Every level
#: above it draws one continuous, capped tube per vessel, smooth shaded, with
#: more sides each step. Both are built on the same rings, so a vessel is
#: the same width at every level.
TUBE_QUALITY_SIDES = (6, 8, 12, 18, 32)
DEFAULT_TUBE_QUALITY = 0
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


class _Rings(NamedTuple):
    """The rings both tube drawings are built on.

    Consecutive steps that join make one tube; its rings, in drawing order,
    are each step's start ring and then one ring closing the tube.
    """

    keep: np.ndarray  # the Vectors row of each drawn step
    radii: np.ndarray  # each drawn step's radius
    start_ring: np.ndarray  # the ring each drawn step starts at
    starts: np.ndarray  # whether a drawn step begins a tube
    end_rings: np.ndarray  # the ring closing each tube
    centre: np.ndarray  # (rings, 3)
    radius: np.ndarray  # (rings,)
    step: np.ndarray  # the drawn step each ring belongs to
    around: np.ndarray  # (rings, sides, 3) unit offsets round each ring


def _tube_rings(vectors, radius, sides, groups) -> _Rings | None:
    """Where every ring sits, how wide it is and which way it faces.

    ``None`` when there is nothing to draw. Raises on malformed input.
    """
    data = np.asarray(vectors, dtype=float)
    if data.size == 0:
        return None
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
        return None
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

    start_ring = np.arange(n) + np.cumsum(ends) - ends
    n_rings = n + int(ends.sum())
    end_rings = start_ring[ends] + 1
    centre = np.empty((n_rings, 3))
    ring_radius = np.empty(n_rings)
    step = np.empty(n_rings, dtype=np.intp)
    local = np.empty((n_rings, 3))
    centre[start_ring] = origin
    ring_radius[start_ring] = radii
    step[start_ring] = np.arange(n)
    local[start_ring] = tangent
    centre[end_rings] = end[ends]
    ring_radius[end_rings] = radii[ends]
    step[end_rings] = np.flatnonzero(ends)
    local[end_rings] = tangent[ends]
    joint = np.flatnonzero(joins)
    if joint.size:
        rings = start_ring[joint]
        ring_radius[rings] = 0.5 * (radii[joint - 1] + radii[joint])
        bisector = tangent[joint - 1] + tangent[joint]
        norm = np.linalg.norm(bisector, axis=1)
        turned = norm > 1e-9  # a step straight back on itself keeps its own plane
        local[rings[turned]] = bisector[turned] / norm[turned, None]

    first = np.zeros(n_rings, dtype=bool)
    first[start_ring[starts]] = True
    tube = np.cumsum(first) - 1
    first_ring = start_ring[starts]

    # Each ring faces along the chord from one radius behind it on the
    # centreline to one radius ahead, clamped to its tube's ends: the way the
    # vessel runs at the scale it is drawn at. On a circular arc that chord is
    # the exact tangent; on a staircase it is the stair's own course.
    gap = np.linalg.norm(np.diff(centre, axis=0), axis=1)
    gap[end_rings[:-1]] = 1.0  # between tubes; never inside a query's range
    along = np.concatenate([[0.0], np.cumsum(gap)])
    lowest = along[first_ring][tube]
    highest = along[end_rings][tube]
    ahead = np.minimum(along + ring_radius, highest)
    behind = np.maximum(along - ring_radius, lowest)
    chord = np.stack(
        [
            np.interp(ahead, along, centre[:, axis]) - np.interp(behind, along, centre[:, axis])
            for axis in range(3)
        ],
        axis=1,
    )
    chord_norm = np.linalg.norm(chord, axis=1)
    usable = chord_norm > 1e-9  # a tube closing on itself keeps its local course
    facing = local.copy()
    facing[usable] = chord[usable] / chord_norm[usable, None]

    # Carry each ring's frame on from the last one's, so the tube does not
    # twist: theta is how far round a ring's own frame its first vertex sits.
    normal, binormal = _normal_plane_frames(facing)
    has_next = np.ones(n_rings, dtype=bool)
    has_next[end_rings] = False
    before = np.flatnonzero(has_next)
    turn = np.zeros(n_rings)
    turn[before + 1] = np.arctan2(
        np.einsum("ij,ij->i", normal[before], binormal[before + 1]),
        np.einsum("ij,ij->i", normal[before], normal[before + 1]),
    )
    theta = np.cumsum(turn)
    theta -= theta[first_ring][tube]
    angles = np.linspace(0.0, 2.0 * np.pi, sides, endpoint=False)[None, :] + theta[:, None]
    around = (
        np.cos(angles)[:, :, None] * normal[:, None, :]
        + np.sin(angles)[:, :, None] * binormal[:, None, :]
    )
    return _Rings(
        keep=keep,
        radii=radii,
        start_ring=start_ring,
        starts=starts,
        end_rings=end_rings,
        centre=centre,
        radius=ring_radius,
        step=step,
        around=around,
    )


def tubes_from_vectors(
    vectors: np.ndarray,
    *,
    radius: float | np.ndarray = TUBE_RADIUS_UM,
    sides: int = DEFAULT_TUBE_SIDES,
    groups: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build independent N-gon prisms from ``(M, 2, 3)`` origin+direction data.

    Returns ``(vertices, faces, segment_index)``. ``segment_index[i]`` is the
    Vectors row that owns ``vertices[i]``, so per-segment colours can be
    repeated onto the prism. Zero-length and non-finite segments are skipped.
    Each prism has its own vertices and open ends; consecutive steps of a
    polyline (of one *groups* label, if given) share the ring between them,
    so their prisms meet edge to edge and the tube keeps its width.

    *radius* is either one value for every segment, or an array of shape
    ``(M,)`` matching *vectors*' own row count -- one radius per segment,
    e.g. each vessel's own measured/set diameter halved (see
    :func:`tube_radii_um`). A non-finite or non-positive entry in a
    per-segment array falls back to :data:`TUBE_RADIUS_UM` rather than
    breaking that segment's prism.
    """
    rings = _tube_rings(vectors, radius, sides, groups)
    if rings is None:
        return _EMPTY_VERTICES.copy(), _EMPTY_FACES.copy(), _EMPTY_INDEX.copy()
    sides = int(sides)
    n_keep = int(rings.keep.size)
    verts_per = sides * 2

    # Each prism is its own step's radius at both ends, so a vessel is drawn
    # at its own width right up to a node it shares with a wider one.
    width = rings.radii[:, None, None]
    first = rings.start_ring
    last = rings.start_ring + 1
    vertices = np.stack(
        [
            rings.centre[first][:, None, :] + rings.around[first] * width,
            rings.centre[last][:, None, :] + rings.around[last] * width,
        ],
        axis=1,
    ).reshape(n_keep * verts_per, 3)
    segment_index = np.repeat(rings.keep.astype(np.intp), verts_per)

    base = np.arange(n_keep, dtype=np.intp)[:, None] * verts_per  # (n_keep, 1)
    k = np.arange(sides, dtype=np.intp)[None, :]  # (1, sides)
    nxt = (k + 1) % sides
    a = base + k
    b = base + nxt
    c = base + sides + k
    d = base + sides + nxt
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
    rings = _tube_rings(vectors, radius, sides, groups)
    if rings is None:
        return _EMPTY_VERTICES.copy(), _EMPTY_FACES.copy(), _EMPTY_INDEX.copy()
    sides = int(sides)
    n_rings = len(rings.centre)
    ring_vertices = (
        rings.centre[:, None, :] + rings.around * rings.radius[:, None, None]
    ).reshape(n_rings * sides, 3)

    # Side faces between each step's two rings, whose frames already line up.
    k = np.arange(sides, dtype=np.intp)[None, :]
    a = rings.start_ring[:, None] * sides + k
    b = rings.start_ring[:, None] * sides + (k + 1) % sides
    c = a + sides
    d = b + sides
    side_faces = np.stack(
        [np.stack([a, b, d], axis=-1), np.stack([a, d, c], axis=-1)], axis=2
    ).reshape(-1, 3)

    # Caps: a centre vertex fanned to each open end, facing outward.
    start_rings = rings.start_ring[rings.starts]
    cap_rings = np.concatenate([start_rings, rings.end_rings])
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

    vertices = np.concatenate([ring_vertices, rings.centre[cap_rings]])
    faces = np.concatenate([side_faces, cap_faces]).astype(np.intp)
    segment_index = np.concatenate([
        np.repeat(rings.keep[rings.step], sides),
        rings.keep[rings.step[cap_rings]],
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

    Level 0 is :func:`tubes_from_vectors`'s separate prisms; every level
    above is :func:`joined_tubes_from_vectors` with that level's sides. Both
    stand on the same rings, so a vessel is the same width at every level.
    """
    level = clamp_tube_quality(quality)
    sides = TUBE_QUALITY_SIDES[level]
    if level == 0:
        return tubes_from_vectors(vectors, radius=radius, sides=sides, groups=groups)
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
