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


def _normal_plane_frame(direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Orthonormal (normal, binormal) in the plane perpendicular to *direction*.

    Axis-aligned X/Y/Z would make a single cross product vanish, so the
    reference axis is swapped when it is nearly parallel to the tangent.
    """
    tangent = np.asarray(direction, dtype=float)
    length = float(np.linalg.norm(tangent))
    tangent = tangent / length
    ref = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(tangent, ref))) > 0.9:
        ref = np.array([1.0, 0.0, 0.0])
        if abs(float(np.dot(tangent, ref))) > 0.9:
            ref = np.array([0.0, 1.0, 0.0])
    normal = np.cross(tangent, ref)
    normal = normal / np.linalg.norm(normal)
    binormal = np.cross(tangent, normal)
    binormal = binormal / np.linalg.norm(binormal)
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
    faces_per = sides * 2
    vertices = np.empty((n_keep * verts_per, 3), dtype=float)
    faces = np.empty((n_keep * faces_per, 3), dtype=np.intp)
    segment_index = np.empty(n_keep * verts_per, dtype=np.intp)

    angles = np.linspace(0.0, 2.0 * np.pi, sides, endpoint=False)
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)

    face_i = 0
    for out_seg, src in enumerate(keep):
        origin = origins[int(src)]
        direction = directions[int(src)]
        normal, binormal = _normal_plane_frame(direction)
        ring = (cos_a[:, None] * normal + sin_a[:, None] * binormal) * radius
        base = out_seg * verts_per
        vertices[base : base + sides] = origin + ring
        vertices[base + sides : base + verts_per] = origin + direction + ring
        segment_index[base : base + verts_per] = int(src)
        for k in range(sides):
            nxt = (k + 1) % sides
            a = base + k
            b = base + nxt
            c = base + sides + k
            d = base + sides + nxt
            faces[face_i] = (a, b, d)
            faces[face_i + 1] = (a, d, c)
            face_i += 2
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
