"""What the Post processing page's raw panel draws, described without Qt.

A review list asks a person to judge a structure -- a dead end, a short loop,
a piece of segmented vessel no centreline covers -- and the marked grids
showed what that judgement is made on: the structure in its own plane, the
segmentation beside the raw image, the graph drawn over both. The panel
draws the chosen item the same way (``PLAN_raw_confirmation.md``): in its own
plane, through the middle of its points along their two longest directions,
and in the plane across it; in each, the segmentation (its voxels anywhere in
a slab either side of the plane) and the run's raw image (the slab's mean),
the item's vessels and the graph's other centrelines over them, and a 5 um
bar.

Only a window round the item is read (:func:`slab_image`), so a
memory-mapped raw image stays on disk. Numbers in, arrays out: the picture
is an array with row 0 at the top, and every overlay is in fractions of its
width and height, for the panel (``_widget.py``) to draw at any size.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Sequence

import numpy as np

#: Each plane reaches this far past the item's points, and at least
#: :data:`MIN_RADIUS_UM` and at most :data:`MAX_RADIUS_UM` from its middle (um).
MARGIN_UM = 8.0
MIN_RADIUS_UM = 15.0
MAX_RADIUS_UM = 40.0
#: The slab either side of the plane each picture is read through (um).
HALF_THICKNESS_UM = 3.0
#: Sample spacing within the plane and across the slab (um).
STEP_UM = 0.4
DEPTH_STEP_UM = 0.5
#: The raw picture is blurred by this many samples before its contrast is
#: set: both stains are speckled, and the speckle hides the vessel.
RAW_SMOOTHING_SAMPLES = 1.5
#: Contrast: these percentiles of the picture go to black and white.
CONTRAST_PERCENTILES = (2.0, 99.8)
SCALE_BAR_UM = 5.0
OWN_PLANE = "Its own plane"
ACROSS_IT = "Across it"


@dataclass(frozen=True)
class Plane:
    """A square of the volume to draw: its middle and three unit directions,
    physical ``(z, y, x)`` microns, and how far it reaches from the middle."""

    label: str
    centre_um: np.ndarray
    #: Left to right across the picture.
    across: np.ndarray
    #: Bottom to top of the picture.
    up: np.ndarray
    #: Through the picture: the slab's depth.
    normal: np.ndarray
    radius_um: float

    def fractions(self, points_um: Any) -> np.ndarray:
        """*points_um* as ``(x, y)`` fractions of the picture, ``(0, 0)`` its
        top left corner."""
        rel = np.asarray(points_um, dtype=float).reshape(-1, 3) - self.centre_um
        size = 2.0 * self.radius_um
        return np.column_stack([(rel @ self.across + self.radius_um) / size, (self.radius_um - rel @ self.up) / size])

    def holds(self, points_um: Any, half_thickness_um: float = HALF_THICKNESS_UM) -> np.ndarray:
        """Which of *points_um* lie within the slab and inside the square."""
        rel = np.asarray(points_um, dtype=float).reshape(-1, 3) - self.centre_um
        return (
            (np.abs(rel @ self.normal) <= half_thickness_um)
            & (np.abs(rel @ self.across) <= self.radius_um)
            & (np.abs(rel @ self.up) <= self.radius_um)
        )


@dataclass(frozen=True)
class PanelView:
    """One plane's two pictures and what is drawn over them."""

    plane: Plane
    #: ``(rows, columns)`` in ``[0, 1]``, row 0 at the top; None when there
    #: is no such volume.
    segmentation: Optional[np.ndarray]
    raw: Optional[np.ndarray]
    #: The item's vessels, each an ``(n, 2)`` run of fractions, and the
    #: points of an item with no vessels (a piece of uncovered mask).
    item_lines: tuple[np.ndarray, ...] = ()
    item_dots: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))
    #: The graph's other centrelines within the slab.
    other_dots: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))
    #: The 5 um bar: ``(x0, x1, y)`` fractions, bottom left.
    scale_bar: tuple[float, float, float] = (0.0, 0.0, 0.0)


def _signed(vector: np.ndarray) -> np.ndarray:
    """*vector* unit length, its largest component positive: the same plane
    whichever sign the decomposition gave."""
    vector = np.asarray(vector, dtype=float)
    vector = vector / np.linalg.norm(vector)
    return -vector if vector[int(np.argmax(np.abs(vector)))] < 0 else vector


def item_planes(points_um: Any, centre_um: Optional[Sequence[float]] = None) -> tuple[Plane, Plane]:
    """The item's own plane -- through the middle of *points_um*, along their
    two longest directions -- and the plane across it (the longest direction
    and the one through the first). An item of one point, or of points on a
    line, takes the image's own axes for what its points do not settle."""
    points = np.asarray(points_um, dtype=float).reshape(-1, 3)
    if not len(points):
        points = np.asarray(centre_um, dtype=float).reshape(1, 3)
    centre = points.mean(axis=0) if centre_um is None or len(points) > 1 else np.asarray(centre_um, dtype=float)
    spread = points - centre
    directions = []
    if len(points) > 1:
        _u, values, vh = np.linalg.svd(spread, full_matrices=False)
        directions = [vh[k] for k in range(len(values)) if values[k] > 1e-6 * max(values[0], 1e-12)]
    # Fill in from x, then y, then z: a flat item lies in its own image plane.
    for axis in (np.array([0.0, 0.0, 1.0]), np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0])):
        if len(directions) >= 2:
            break
        rest = axis - sum((axis @ d) * d for d in directions)
        if np.linalg.norm(rest) > 1e-6:
            directions.append(rest / np.linalg.norm(rest))
    across = _signed(directions[0])
    up = directions[1] - (directions[1] @ across) * across
    up = up / np.linalg.norm(up)
    # Up is the way that puts the plane's normal mostly along +z, +y or +x:
    # an item lying in one image plane is drawn as napari draws that plane,
    # x to the right and y down.
    normal = np.cross(across, up)
    if normal[int(np.argmax(np.abs(normal)))] < 0:
        up, normal = -up, -normal
    reach = float(np.max(np.abs(np.column_stack([spread @ across, spread @ up])))) if len(points) else 0.0
    radius = float(np.clip(reach + MARGIN_UM, MIN_RADIUS_UM, MAX_RADIUS_UM))
    # Across it, the own plane's normal runs down the picture: deeper is lower.
    return (
        Plane(OWN_PLANE, centre, across, up, normal, radius),
        Plane(ACROSS_IT, centre, across, -normal, np.cross(across, -normal), radius),
    )


def slab_image(
    volume: Any,
    plane: Plane,
    voxel_size_zyx: Sequence[float],
    *,
    reduce: str = "mean",
    half_thickness_um: float = HALF_THICKNESS_UM,
    step_um: float = STEP_UM,
) -> np.ndarray:
    """*volume* (voxel-indexed ``(z, y, x)``) sampled over *plane* through a
    slab *half_thickness_um* either side: the samples' mean (``"mean"``,
    linear interpolation) or whether any is set (``"any"``, nearest voxel, for
    a mask). Rows run top to bottom. Only the window the samples fall in is
    read from *volume*; outside it reads 0."""
    from scipy import ndimage

    spacing = np.asarray(voxel_size_zyx, dtype=float)
    count = max(int(round(2.0 * plane.radius_um / step_um)) + 1, 2)
    along = np.linspace(-plane.radius_um, plane.radius_um, count)
    V, U = np.meshgrid(along[::-1], along, indexing="ij")
    flat = U.reshape(-1, 1) * plane.across + V.reshape(-1, 1) * plane.up + plane.centre_um
    depths = np.arange(-half_thickness_um, half_thickness_um + 1e-9, DEPTH_STEP_UM)
    shape = np.asarray(np.shape(volume)[-3:])
    corners = np.vstack([(flat + w * plane.normal) / spacing for w in (depths[0], depths[-1])])
    lo = np.clip(np.floor(corners.min(axis=0)).astype(int) - 1, 0, shape)
    hi = np.clip(np.ceil(corners.max(axis=0)).astype(int) + 2, 0, shape)
    if np.any(hi <= lo):
        return np.zeros(U.shape)
    window = np.asarray(volume[tuple(slice(a, b) for a, b in zip(lo, hi))])
    window = window.astype(bool) if reduce == "any" else window.astype(np.float32)
    total = np.zeros(U.size, dtype=bool if reduce == "any" else float)
    for w in depths:
        coords = ((flat + w * plane.normal) / spacing - lo).T
        if reduce == "any":
            total |= ndimage.map_coordinates(window, coords, order=0, mode="constant", cval=0).astype(bool)
        else:
            total += ndimage.map_coordinates(window, coords, order=1, mode="constant", cval=0.0)
    image = total.reshape(U.shape)
    return image.astype(float) if reduce == "any" else image / len(depths)


def stretch(image: np.ndarray, percentiles: Sequence[float] = CONTRAST_PERCENTILES) -> np.ndarray:
    """*image* with its *percentiles* at 0 and 1, clipped: a flat picture is 0."""
    image = np.asarray(image, dtype=float)
    low, high = np.percentile(image, list(percentiles)) if image.size else (0.0, 0.0)
    if high <= low:
        return np.zeros_like(image)
    return np.clip((image - low) / (high - low), 0.0, 1.0)


def edge_lines(graph: Any, edges: Iterable[Any]) -> list[np.ndarray]:
    """The polylines of *edges* (``(u, v, key)``) of *graph*, physical
    microns; an edge no longer there, or with nothing to draw, is left out."""
    from haemolynx.visualization.geometry import edge_polyline

    lines = []
    for edge in edges:
        u, v, key = (tuple(edge) + (None,))[:3]
        if graph is None or not graph.has_edge(u, v, key):
            continue
        try:
            lines.append(edge_polyline(graph, u, v, graph.edges[u, v, key]))
        except ValueError:
            continue
    return lines


def centreline_points(graph: Any, step_um: float = 1.0) -> np.ndarray:
    """Every centreline of *graph* as points no more than *step_um* apart,
    physical microns: the other vessels drawn as dots."""
    from haemolynx.graph._helpers import densify_polyline

    lines = edge_lines(graph, graph.edges(keys=True)) if graph is not None else []
    if not lines:
        return np.empty((0, 3))
    return np.vstack([densify_polyline(line, max_step_um=step_um) for line in lines])


def _inside(fractions: np.ndarray) -> np.ndarray:
    return np.all((fractions >= 0.0) & (fractions <= 1.0), axis=1) if len(fractions) else np.zeros(0, dtype=bool)


def panel_views(
    points_um: Any,
    *,
    voxel_size_zyx: Sequence[float],
    centre_um: Optional[Sequence[float]] = None,
    mask: Any = None,
    raw: Any = None,
    item_lines_um: Sequence[np.ndarray] = (),
    other_points_um: Any = None,
    half_thickness_um: float = HALF_THICKNESS_UM,
) -> list[PanelView]:
    """The item at *points_um* (its middle *centre_um*, when it has none) in
    :func:`item_planes`' two planes: *mask* as the segmentation, *raw* as the
    raw image (each None to leave its picture out), *item_lines_um* -- its
    vessels, else its points as dots -- and *other_points_um* over them."""
    points = np.asarray(points_um, dtype=float).reshape(-1, 3)
    others = np.asarray(other_points_um if other_points_um is not None else np.empty((0, 3)), dtype=float)
    views = []
    for plane in item_planes(points, centre_um):
        segmentation = None
        if mask is not None:
            segmentation = slab_image(mask, plane, voxel_size_zyx, reduce="any", half_thickness_um=half_thickness_um)
        picture = None
        if raw is not None:
            from scipy import ndimage

            mean = slab_image(raw, plane, voxel_size_zyx, half_thickness_um=half_thickness_um)
            picture = stretch(ndimage.gaussian_filter(mean, RAW_SMOOTHING_SAMPLES))
        lines = tuple(plane.fractions(line) for line in item_lines_um if len(line))
        dots = np.empty((0, 2)) if lines else plane.fractions(points[plane.holds(points, half_thickness_um)])
        near = others[plane.holds(others, half_thickness_um + 2.0)] if len(others) else others
        bar = SCALE_BAR_UM / (2.0 * plane.radius_um)
        views.append(PanelView(
            plane=plane, segmentation=segmentation, raw=picture, item_lines=lines,
            item_dots=dots[_inside(dots)] if len(dots) else dots,
            other_dots=plane.fractions(near) if len(near) else np.empty((0, 2)),
            scale_bar=(0.04, 0.04 + bar, 0.95),
        ))
    return views


__all__ = [
    "ACROSS_IT",
    "OWN_PLANE",
    "PanelView",
    "Plane",
    "centreline_points",
    "edge_lines",
    "item_planes",
    "panel_views",
    "slab_image",
    "stretch",
]
