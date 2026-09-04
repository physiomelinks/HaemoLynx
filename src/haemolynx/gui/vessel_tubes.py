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
#: still subpixel at whole-network zoom; 2 µm stays visible. Do not scale by
#: vessel diameter here — that is out of scope.
TUBE_RADIUS_UM = 2.0
DEFAULT_TUBE_SIDES = 6

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
    radius: float = TUBE_RADIUS_UM,
    sides: int = DEFAULT_TUBE_SIDES,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build independent N-gon prisms from ``(M, 2, 3)`` origin+direction data.

    Returns ``(vertices, faces, segment_index)``. ``segment_index[i]`` is the
    Vectors row that owns ``vertices[i]``, so per-segment colours can be
    repeated onto the prism. Zero-length and non-finite segments are skipped.
    Joins are not mitred: consecutive steps of a polyline produce disjoint
    vertex sets whose end/start rings abut when the steps share a point.
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
    radius = float(radius)
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError(f"radius must be a positive finite number; got {radius}")

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

    angles = np.linspace(0.0, 2.0 * np.pi, sides, endpoint=False)
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)
    # (n_keep, sides, 3): per-segment ring, one cross-section shared by both
    # the origin-end and direction-end rings of that segment's prism.
    ring = (
        cos_a[None, :, None] * normal[:, None, :]
        + sin_a[None, :, None] * binormal[:, None, :]
    ) * radius

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
