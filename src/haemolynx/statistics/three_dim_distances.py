"""3D object-to-vessel distance analysis for binary cell masks.

This module labels connected objects in a 3D binary mask (cells), measures how
far each object lies from the vessels -- the gap between the two as voxelised
volumes, 0 where they touch or overlap -- and names the vessel it is nearest
to (its branch order) when a vessel graph is provided.

The vessels are a vessel mask: the run's own segmentation, or another mask
given for the purpose. Only without either are they drawn from the graph, each
vessel a tube of its own ``diameter_um`` round its centreline.
"""
from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any
import csv

import networkx as nx
import numpy as np
from scipy.ndimage import binary_erosion, find_objects, generate_binary_structure, label
from scipy.spatial import cKDTree

from haemolynx.io import CANONICAL_AXIS_ORDER, voxel_size_zyx_from_xyz
from haemolynx.io.vessel_masks import (
    _VOXEL_SIZE_RELATIVE_TOLERANCE,
    _resolve_mask_voxel_size_xyz,
)
from haemolynx.io.load import (
    _to_binary_volume_for_skeletonization,
    load_binary_mask_voxel_size_and_status,
)
from haemolynx.haemodynamics.fwhm_diameter import build_graph_branch_label_volume
from haemolynx.graph.thick_vessel_junctions import IS_ZERO_RESISTANCE
from haemolynx.io.downsample import graph_downsample, mask_volume_like_graph
from haemolynx.preprocessing.memmap_support import (
    argwhere_by_slab,
    map_blockwise,
    new_memmap_array,
    release_memmap_array,
)

logger = logging.getLogger(__name__)

#: A voxel's six face neighbours: a surface voxel is one with a face neighbour
#: outside. The nearest voxel of a volume to anything outside it is always one
#: of those, so the distances need no others.
_FACE_NEIGHBOURS = generate_binary_structure(rank=3, connectivity=1)

#: How the object stands to the vessel (``ObjectDistanceRecord.vessel_contact``).
CONTACT_SEPARATE = "separate"
CONTACT_TOUCHING = "touching"
CONTACT_OVERLAPPING = "overlapping"
CONTACT_INSIDE = "inside"

#: Centreline samples of one piece of a graph-drawn tube; the piece's bounding
#: box is what is searched for voxels within the vessel's radius.
_TUBE_PIECE_SAMPLES = 32


@dataclass(frozen=True)
class ObjectDistanceRecord:
    """Per-object nearest vessel-distance result.

    Both distances are gaps, each voxel taken as the box it fills: from the
    object to the vessel, and from the object's centroid (a point) to the
    vessel -- 0 where they touch or overlap. ``vessel_contact`` says which of
    ``separate``, ``touching`` (sharing a face, edge or corner),
    ``overlapping`` (some of the object's voxels are vessel) or ``inside`` (all
    of them). The nearest points are vessel voxel centres in microns: where the
    gap is smallest, or, for an object overlapping the vessel, the overlapped
    voxel nearest its centroid. Every position is in the network's frame,
    measured from the crop_on_import box's corner when the image was cut, as
    node ``pos`` is.
    """

    object_id: int
    object_name: str
    voxel_count: int
    centroid_z: float
    centroid_y: float
    centroid_x: float
    edge_to_vessel_distance_microns: float
    edge_to_vessel_nearest_z: float
    edge_to_vessel_nearest_y: float
    edge_to_vessel_nearest_x: float
    edge_to_vessel_nearest_graph_edge_label_id: int | None
    edge_to_vessel_nearest_branch_order: str | None
    centroid_to_vessel_distance_microns: float
    centroid_to_vessel_nearest_z: float
    centroid_to_vessel_nearest_y: float
    centroid_to_vessel_nearest_x: float
    centroid_to_vessel_nearest_graph_edge_label_id: int | None
    centroid_to_vessel_nearest_branch_order: str | None
    centroid_z_microns: float
    centroid_y_microns: float
    centroid_x_microns: float
    vessel_overlap_voxel_count: int
    vessel_contact: str


def label_connected_objects(
    object_mask: np.ndarray,
    *,
    connectivity: int = 3,
    object_name_prefix: str = "Object",
    use_memmap: bool = False,
    memmap_directory: str | Path | None = None,
) -> tuple[np.ndarray, list[tuple[int, str]]]:
    """Label connected components in a binary 3D object mask.

    *use_memmap* (the low-RAM option) labels into a new disk-backed ``int32``
    array, which the caller releases; the labels are the same.
    """
    if object_mask.ndim != 3:
        raise ValueError(f"Expected 3D object mask, got shape {object_mask.shape}.")
    if connectivity not in (1, 2, 3):
        raise ValueError(f"connectivity must be 1, 2, or 3; got {connectivity}.")
    structure = generate_binary_structure(rank=3, connectivity=connectivity)
    if use_memmap:
        labels = new_memmap_array(object_mask.shape, np.int32, directory=memmap_directory)
        n_objects = label(
            np.asanyarray(object_mask, dtype=bool), structure=structure, output=labels
        )
        objects = [
            (i, f"{object_name_prefix}_{i:04d}") for i in range(1, int(n_objects) + 1)
        ]
        return labels, objects
    labels, n_objects = label(object_mask.astype(bool), structure=structure)
    objects = [
        (i, f"{object_name_prefix}_{i:04d}") for i in range(1, int(n_objects) + 1)
    ]
    return labels.astype(np.int32), objects


def _vessel_surface(block: np.ndarray) -> np.ndarray:
    # Beyond the image counts as vessel: a vessel the image cut runs on past
    # its face, which is no wall of it.
    return block & ~binary_erosion(block, structure=_FACE_NEIGHBOURS, border_value=1)


def _compute_vessel_boundary_mask(
    vessel_mask: np.ndarray,
    *,
    use_memmap: bool = False,
    memmap_directory: str | Path | None = None,
) -> np.ndarray:
    """Return the vessel's surface voxels, those with a face neighbour outside it.

    *use_memmap* (the low-RAM option) erodes one padded block at a time into
    a disk-backed array: a face-neighbour erosion reads one voxel away, so a
    halo of one makes every block exact.
    """
    if not np.any(vessel_mask):
        raise ValueError("Vessel mask is empty.")
    if use_memmap:
        mask = np.asanyarray(vessel_mask, dtype=bool)
        boundary = new_memmap_array(mask.shape, bool, directory=memmap_directory)
        map_blockwise(mask, _vessel_surface, boundary, halo=1)
        if not np.any(boundary):
            release_memmap_array(boundary)
            return mask
        return boundary
    mask = np.asarray(vessel_mask, dtype=bool)
    boundary = _vessel_surface(mask)
    if not np.any(boundary):
        # A vessel filling the whole image has no surface inside it.
        boundary = mask
    return boundary


def _compute_object_boundary_indices(object_region: np.ndarray) -> np.ndarray:
    """Return indices of boundary voxels for one connected object."""
    region = object_region.astype(bool)
    boundary = region & ~binary_erosion(region, structure=_FACE_NEIGHBOURS)
    idx = np.argwhere(boundary)
    if idx.size == 0:
        idx = np.argwhere(region)
    return idx


def _edge_label_to_branch_order_map(
    G: nx.MultiGraph,
) -> dict[int, str | None]:
    """Map graph edge label ids to branch-order strings.

    A thick-vessel bridge (``IS_ZERO_RESISTANCE``) is not a vessel, so it is
    left out, and :class:`_VesselIdentity` never names it.
    """
    mapping: dict[int, str | None] = {}
    for u, v, key, data in G.edges(keys=True, data=True):
        if data.get(IS_ZERO_RESISTANCE):
            continue
        edge_label = data.get("graph_edge_label_id")
        if edge_label is None:
            continue
        try:
            edge_label_i = int(edge_label)
        except (TypeError, ValueError):
            continue
        branch_order = data.get("branch_order")
        mapping[edge_label_i] = None if branch_order is None else str(branch_order)
    return mapping


def _build_edge_voxel_lookup(
    G: nx.MultiGraph,
    *,
    volume_shape: tuple[int, int, int],
    voxel_size_zyx: tuple[float, float, float],
    use_memmap: bool = False,
    memmap_directory: str | Path | None = None,
) -> tuple[np.ndarray, dict[int, str | None]]:
    """Rasterize graph edges to a label volume and return branch-order lookup."""
    edge_label_volume, _ = build_graph_branch_label_volume(
        G,
        volume_shape=volume_shape,
        voxel_size_zyx=voxel_size_zyx,
        background_label=0,
        junction_label=-1,
        use_memmap=use_memmap,
        memmap_directory=memmap_directory,
    )
    return edge_label_volume, _edge_label_to_branch_order_map(G)


def _densified(points: np.ndarray, step: float) -> np.ndarray:
    """*points*, a polyline, with samples added so none is more than *step*
    from the next."""
    if len(points) < 2:
        return points
    runs = np.diff(points, axis=0)
    counts = np.maximum(np.ceil(np.linalg.norm(runs, axis=1) / step).astype(int), 1)
    run = np.repeat(np.arange(len(runs)), counts)
    within = np.arange(int(counts.sum())) - np.repeat(np.cumsum(counts) - counts, counts)
    fraction = within / np.repeat(counts, counts)
    return np.vstack([points[run] + runs[run] * fraction[:, None], points[-1:]])


def _graph_vessel_mask(
    G: nx.MultiGraph,
    *,
    volume_shape: tuple[int, int, int],
    voxel_size_zyx: tuple[float, float, float],
    use_memmap: bool = False,
    memmap_directory: str | Path | None = None,
) -> np.ndarray:
    """The vessels as the graph models them, for a measurement with no vessel
    mask: every voxel whose centre lies within an edge's ``diameter_um`` / 2 of
    its centreline, and the voxels its centreline runs through.

    Thick-vessel bridges are not vessel and are left out. An edge with no
    diameter is its centreline alone, and the log says how many there are: a
    distance to one is measured to its centreline, not its wall.
    """
    shape = tuple(int(n) for n in volume_shape)
    spacing = np.asarray(voxel_size_zyx, dtype=float)
    if use_memmap:
        mask = new_memmap_array(shape, bool, directory=memmap_directory)
    else:
        mask = np.zeros(shape, dtype=bool)
    last = np.asarray(shape) - 1
    step = 0.5 * float(spacing.min())
    without_diameter = 0
    for _u, _v, data in G.edges(data=True):
        if data.get(IS_ZERO_RESISTANCE) or not data.get("voxels"):
            continue
        try:
            radius = 0.5 * float(data.get("diameter_um"))
        except (TypeError, ValueError):
            radius = float("nan")
        if not np.isfinite(radius) or radius <= 0.0:
            radius = 0.0
            without_diameter += 1
        points = _densified(np.asarray(data["voxels"], dtype=float).reshape(-1, 3), step)
        centreline = np.clip(np.rint(points / spacing).astype(int), 0, last)
        mask[centreline[:, 0], centreline[:, 1], centreline[:, 2]] = True
        if radius == 0.0:
            continue
        for start in range(0, len(points), _TUBE_PIECE_SAMPLES):
            piece = points[start:start + _TUBE_PIECE_SAMPLES + 1]
            lo = np.maximum(np.floor((piece.min(axis=0) - radius) / spacing), 0).astype(int)
            hi = np.minimum(np.ceil((piece.max(axis=0) + radius) / spacing), last).astype(int)
            if np.any(hi < lo):
                continue
            box = np.stack(
                np.meshgrid(*(np.arange(a, b + 1) for a, b in zip(lo, hi)), indexing="ij"),
                axis=-1,
            ).reshape(-1, 3)
            reach, _ = cKDTree(piece).query(box * spacing, k=1, distance_upper_bound=radius * (1 + 1e-9))
            inside = box[np.isfinite(reach)]
            mask[inside[:, 0], inside[:, 1], inside[:, 2]] = True
    if without_diameter:
        logger.warning(
            "3D distance measurement: no vessel mask, so the vessels are drawn from "
            "the graph; %d vessel(s) have no diameter_um and are drawn as their "
            "centreline alone, so distances to them are to the centreline, not the wall.",
            without_diameter,
        )
    return mask


def _box_gaps(
    a_idx: np.ndarray, b_idx: np.ndarray, spacing: np.ndarray, reach: float
) -> np.ndarray:
    """Distances in microns between *a_idx* and the voxels *b_idx* (indices),
    each voxel the box it fills. *reach* is 1.0 when *a_idx* are voxels too,
    0.5 when they are points."""
    excess = np.maximum(np.abs(b_idx - a_idx) - reach, 0.0) * spacing
    return np.sqrt(np.sum(excess * excess, axis=-1))


def _nearest_gap(
    query_idx: np.ndarray,
    reach: float,
    surface_tree: cKDTree,
    surface_idx: np.ndarray,
    spacing: np.ndarray,
) -> tuple[float, int]:
    """``(gap, surface row)``: the smallest :func:`_box_gaps` from any of
    *query_idx* to the vessel surface, ties to the nearest voxel centre.

    The pair nearest centre to centre need not have the smallest gap, but
    every pair that has one is within the voxel diagonal *reach* spans of it.
    """
    phys = query_idx * spacing
    centre_distances, nearest = surface_tree.query(phys, k=1)
    upper = float(np.min(_box_gaps(query_idx, surface_idx[nearest], spacing, reach)))
    search = upper + float(np.linalg.norm(reach * spacing)) + 1e-9
    rows = np.flatnonzero(centre_distances <= search)
    best = (np.inf, np.inf, -1)
    for row, found in zip(rows, surface_tree.query_ball_point(phys[rows], search)):
        found = np.asarray(found, dtype=np.intp)
        gaps = _box_gaps(query_idx[row], surface_idx[found], spacing, reach)
        centres = np.linalg.norm(surface_idx[found] * spacing - phys[row], axis=1)
        k = np.lexsort((found, centres, gaps))[0]
        candidate = (float(gaps[k]), float(centres[k]), int(found[k]))
        if candidate < best:
            best = candidate
    return best[0], best[2]


class _VesselIdentity:
    """Which vessel a point belongs to: the centreline whose lumen comes
    closest to holding it.

    Each labelled centreline voxel ``c`` has a lumen radius ``r(c)``, its
    distance to the vessel's surface (0 off the mask), and a point ``p``
    belongs to the ``c`` with the least ``|p - c| - r(c)``: how far ``p`` lies
    outside ``c``'s inscribed ball, negative inside it. The nearest centreline
    alone hands a wide vessel's wall to any narrow vessel running within one
    radius of it.
    """

    def __init__(
        self,
        centre_idx: np.ndarray,
        labels: np.ndarray,
        radii: np.ndarray,
        spacing: np.ndarray,
        branch_orders: dict[int, str | None] | None,
    ) -> None:
        self._phys = centre_idx * spacing
        self._tree = cKDTree(self._phys)
        self._labels = labels
        self._radii = radii
        self._widest = float(radii.max())
        self._branch_orders = branch_orders

    def __call__(self, point_phys: np.ndarray) -> tuple[int, str | None]:
        distance, nearest = self._tree.query(point_phys, k=1)
        # No centreline farther than this can beat the nearest one.
        search = float(distance - self._radii[nearest]) + self._widest + 1e-9
        found = np.asarray(self._tree.query_ball_point(point_phys, search), dtype=np.intp)
        centres = np.linalg.norm(self._phys[found] - point_phys, axis=1)
        k = np.lexsort((found, centres, centres - self._radii[found]))[0]
        edge_label = int(self._labels[found[k]])
        if self._branch_orders is None:
            return edge_label, None
        return edge_label, self._branch_orders.get(edge_label)


def _vessel_identity(
    graph_edge_label_volume: np.ndarray | None,
    edge_label_to_branch_order: dict[int, str | None] | None,
    *,
    vessel_mask: np.ndarray,
    surface_tree: cKDTree,
    spacing: np.ndarray,
    use_memmap: bool,
) -> _VesselIdentity | None:
    if graph_edge_label_volume is None:
        return None
    if graph_edge_label_volume.shape != vessel_mask.shape:
        raise ValueError(
            "graph_edge_label_volume shape must match object/vessel masks. "
            f"Got {graph_edge_label_volume.shape} and {vessel_mask.shape}."
        )
    centre_idx = (
        argwhere_by_slab(graph_edge_label_volume, lambda slab: slab > 0)
        if use_memmap
        else np.argwhere(graph_edge_label_volume > 0)
    )
    if centre_idx.size == 0:
        return None
    labels = np.asarray(
        graph_edge_label_volume[centre_idx[:, 0], centre_idx[:, 1], centre_idx[:, 2]]
    ).astype(int)
    if edge_label_to_branch_order is not None:
        named = np.isin(labels, np.fromiter(edge_label_to_branch_order, dtype=int))
        centre_idx, labels = centre_idx[named], labels[named]
        if centre_idx.size == 0:
            return None
    radii, _ = surface_tree.query(centre_idx * spacing, k=1)
    in_mask = np.asarray(
        vessel_mask[centre_idx[:, 0], centre_idx[:, 1], centre_idx[:, 2]], dtype=bool
    )
    radii = np.where(in_mask, radii, 0.0)
    return _VesselIdentity(
        centre_idx.astype(float), labels, radii, spacing, edge_label_to_branch_order
    )


def compute_object_to_vessel_distances(
    *,
    object_mask: np.ndarray,
    vessel_mask: np.ndarray,
    voxel_size_zyx: tuple[float, float, float],
    graph_edge_label_volume: np.ndarray | None = None,
    edge_label_to_branch_order: dict[int, str | None] | None = None,
    connectivity: int = 3,
    object_name_prefix: str = "Object",
    use_memmap: bool = False,
    memmap_directory: str | Path | None = None,
) -> list[ObjectDistanceRecord]:
    """Each object's gap to the vessels, and the vessel it is nearest to.

    Measurements per object (see :class:`ObjectDistanceRecord`):
    1) The gap between the object and the vessel, the two as voxelised
       volumes: 0 where they touch or overlap.
    2) The gap between the object's centroid and the vessel: 0 when the
       centroid lies in a vessel voxel.

    The vessel named for each is the one whose lumen its nearest vessel point
    lies in (:class:`_VesselIdentity`), from *graph_edge_label_volume*'s
    centreline labels and *edge_label_to_branch_order*; a label the map does
    not name (a thick-vessel bridge) is not a vessel.

    Each object is read from its own bounding box (plus the one voxel its
    erosion reads), never from a whole-volume ``labels == id``. *use_memmap*
    (the low-RAM option) keeps every volume-sized array on disk as well:
    labels and the vessel surface are disk-backed and voxel lists are gathered
    a slab at a time. Every voxel list comes out in the same order, so the
    records are identical.
    """
    if object_mask.shape != vessel_mask.shape:
        raise ValueError(
            "object_mask and vessel_mask must share shape. "
            f"Got {object_mask.shape} and {vessel_mask.shape}."
        )
    spacing = np.asarray(voxel_size_zyx, dtype=float)
    if spacing.shape != (3,) or np.any(spacing <= 0):
        raise ValueError(
            f"voxel_size_zyx must be 3 positive values, got {voxel_size_zyx}."
        )

    labels, objects = label_connected_objects(
        object_mask,
        connectivity=connectivity,
        object_name_prefix=object_name_prefix,
        use_memmap=use_memmap,
        memmap_directory=memmap_directory,
    )
    try:
        return _object_to_vessel_distances(
            labels,
            objects,
            vessel_mask=vessel_mask,
            spacing=spacing,
            graph_edge_label_volume=graph_edge_label_volume,
            edge_label_to_branch_order=edge_label_to_branch_order,
            use_memmap=use_memmap,
            memmap_directory=memmap_directory,
        )
    finally:
        if use_memmap:
            release_memmap_array(labels)


def _object_crop_indices(labels: np.ndarray, object_id: int, bbox) -> tuple[np.ndarray, np.ndarray]:
    """``(np.argwhere(labels == id), _compute_object_boundary_indices(labels == id))``
    read from the object's bounding box grown by the one voxel its erosion
    reads -- clipped at the volume, where the erosion's zero border applies
    either way. Row order matches the whole-volume calls: C order within a
    crop is C order in the volume."""
    lo = np.array([max(0, s.start - 1) for s in bbox])
    hi = np.array([min(n, s.stop + 1) for s, n in zip(bbox, labels.shape)])
    region = np.asarray(labels[tuple(slice(a, b) for a, b in zip(lo, hi))]) == int(object_id)
    return np.argwhere(region) + lo, _compute_object_boundary_indices(region) + lo


def _object_to_vessel_distances(
    labels: np.ndarray,
    objects: list[tuple[int, str]],
    *,
    vessel_mask: np.ndarray,
    spacing: np.ndarray,
    graph_edge_label_volume: np.ndarray | None,
    edge_label_to_branch_order: dict[int, str | None] | None,
    use_memmap: bool,
    memmap_directory: str | Path | None,
) -> list[ObjectDistanceRecord]:
    if not objects:
        return []

    boundary_mask = _compute_vessel_boundary_mask(
        vessel_mask, use_memmap=use_memmap, memmap_directory=memmap_directory
    )
    if use_memmap:
        boundary_idx = argwhere_by_slab(boundary_mask)
        if isinstance(boundary_mask, np.memmap) and boundary_mask is not vessel_mask:
            release_memmap_array(boundary_mask)
    else:
        boundary_idx = np.argwhere(boundary_mask)
    if boundary_idx.size == 0:
        raise ValueError("No vessel boundary voxels found.")
    boundary_tree = cKDTree(boundary_idx.astype(float) * spacing.reshape(1, 3))
    identify = _vessel_identity(
        graph_edge_label_volume,
        edge_label_to_branch_order,
        vessel_mask=vessel_mask,
        surface_tree=boundary_tree,
        spacing=spacing,
        use_memmap=use_memmap,
    )

    def _identity(vessel_idx: np.ndarray) -> tuple[int | None, str | None]:
        if identify is None:
            return None, None
        return identify(vessel_idx.astype(float) * spacing)

    last = np.asarray(labels.shape) - 1
    results: list[ObjectDistanceRecord] = []
    bboxes = find_objects(labels)
    for object_id, object_name in objects:
        bbox = bboxes[int(object_id) - 1] if int(object_id) - 1 < len(bboxes) else None
        if bbox is None:
            continue
        object_idx, object_boundary_idx = _object_crop_indices(labels, object_id, bbox)
        if object_idx.size == 0:
            continue
        centroid_idx = np.mean(object_idx.astype(float), axis=0)
        centroid_phys = centroid_idx * spacing

        in_vessel = np.asarray(
            vessel_mask[object_idx[:, 0], object_idx[:, 1], object_idx[:, 2]], dtype=bool
        )
        overlap = int(np.count_nonzero(in_vessel))
        if overlap:
            overlapped = object_idx[in_vessel]
            nearest = int(np.argmin(np.linalg.norm(overlapped * spacing - centroid_phys, axis=1)))
            edge_to_vessel_dist = 0.0
            nearest_vessel_idx_from_edge = overlapped[nearest]
        else:
            edge_to_vessel_dist, row = _nearest_gap(
                object_boundary_idx.astype(float), 1.0, boundary_tree, boundary_idx, spacing
            )
            nearest_vessel_idx_from_edge = boundary_idx[row]
        if overlap == len(object_idx):
            contact = CONTACT_INSIDE
        elif overlap:
            contact = CONTACT_OVERLAPPING
        elif edge_to_vessel_dist == 0.0:
            contact = CONTACT_TOUCHING
        else:
            contact = CONTACT_SEPARATE

        centroid_voxel = np.clip(np.rint(centroid_idx).astype(int), 0, last)
        if bool(vessel_mask[tuple(centroid_voxel)]):
            centroid_to_vessel_dist = 0.0
            nearest_vessel_idx_from_centroid = centroid_voxel
        else:
            centroid_to_vessel_dist, row = _nearest_gap(
                centroid_idx.reshape(1, 3), 0.5, boundary_tree, boundary_idx, spacing
            )
            nearest_vessel_idx_from_centroid = boundary_idx[row]

        nearest_vessel_phys_from_edge = nearest_vessel_idx_from_edge.astype(float) * spacing
        nearest_vessel_phys_from_centroid = (
            nearest_vessel_idx_from_centroid.astype(float) * spacing
        )
        edge_nearest_edge_label_id, edge_nearest_branch_order = _identity(
            nearest_vessel_idx_from_edge
        )
        centroid_nearest_edge_label_id, centroid_nearest_branch_order = _identity(
            nearest_vessel_idx_from_centroid
        )

        results.append(
            ObjectDistanceRecord(
                object_id=int(object_id),
                object_name=object_name,
                voxel_count=int(object_idx.shape[0]),
                centroid_z=float(centroid_idx[0]),
                centroid_y=float(centroid_idx[1]),
                centroid_x=float(centroid_idx[2]),
                edge_to_vessel_distance_microns=float(edge_to_vessel_dist),
                edge_to_vessel_nearest_z=float(nearest_vessel_phys_from_edge[0]),
                edge_to_vessel_nearest_y=float(nearest_vessel_phys_from_edge[1]),
                edge_to_vessel_nearest_x=float(nearest_vessel_phys_from_edge[2]),
                edge_to_vessel_nearest_graph_edge_label_id=edge_nearest_edge_label_id,
                edge_to_vessel_nearest_branch_order=edge_nearest_branch_order,
                centroid_to_vessel_distance_microns=float(centroid_to_vessel_dist),
                centroid_to_vessel_nearest_z=float(nearest_vessel_phys_from_centroid[0]),
                centroid_to_vessel_nearest_y=float(nearest_vessel_phys_from_centroid[1]),
                centroid_to_vessel_nearest_x=float(nearest_vessel_phys_from_centroid[2]),
                centroid_to_vessel_nearest_graph_edge_label_id=centroid_nearest_edge_label_id,
                centroid_to_vessel_nearest_branch_order=centroid_nearest_branch_order,
                centroid_z_microns=float(centroid_phys[0]),
                centroid_y_microns=float(centroid_phys[1]),
                centroid_x_microns=float(centroid_phys[2]),
                vessel_overlap_voxel_count=overlap,
                vessel_contact=contact,
            )
        )
    return results


def export_object_distance_details_csv(
    records: list[ObjectDistanceRecord],
    output_csv_path: str | Path,
) -> Path:
    """Write per-object dual-distance rows to CSV."""
    out = Path(output_csv_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Object ID",
                "Object Name",
                "Voxel Count",
                "Centroid Z (voxel)",
                "Centroid Y (voxel)",
                "Centroid X (voxel)",
                "Centroid Z (um)",
                "Centroid Y (um)",
                "Centroid X (um)",
                "Vessel Contact",
                "Vessel Overlap Voxel Count",
                "Edge-to-Vessel Distance (um)",
                "Edge-to-Vessel Nearest Z (um)",
                "Edge-to-Vessel Nearest Y (um)",
                "Edge-to-Vessel Nearest X (um)",
                "Edge-to-Vessel Nearest Coordinate (z,y,x um)",
                "Edge-to-Vessel Nearest Graph Edge Label ID",
                "Edge-to-Vessel Nearest Branch Order",
                "Centroid-to-Vessel Distance (um)",
                "Delta Distance (Centroid-Edge) (um)",
                "Centroid-to-Vessel Nearest Z (um)",
                "Centroid-to-Vessel Nearest Y (um)",
                "Centroid-to-Vessel Nearest X (um)",
                "Centroid-to-Vessel Nearest Coordinate (z,y,x um)",
                "Centroid-to-Vessel Nearest Graph Edge Label ID",
                "Centroid-to-Vessel Nearest Branch Order",
            ]
        )
        for rec in records:
            writer.writerow(
                [
                    rec.object_id,
                    rec.object_name,
                    rec.voxel_count,
                    f"{rec.centroid_z:.6g}",
                    f"{rec.centroid_y:.6g}",
                    f"{rec.centroid_x:.6g}",
                    f"{rec.centroid_z_microns:.6g}",
                    f"{rec.centroid_y_microns:.6g}",
                    f"{rec.centroid_x_microns:.6g}",
                    rec.vessel_contact,
                    rec.vessel_overlap_voxel_count,
                    f"{rec.edge_to_vessel_distance_microns:.6g}",
                    f"{rec.edge_to_vessel_nearest_z:.6g}",
                    f"{rec.edge_to_vessel_nearest_y:.6g}",
                    f"{rec.edge_to_vessel_nearest_x:.6g}",
                    f"({rec.edge_to_vessel_nearest_z:.6g}, {rec.edge_to_vessel_nearest_y:.6g}, {rec.edge_to_vessel_nearest_x:.6g})",
                    ""
                    if rec.edge_to_vessel_nearest_graph_edge_label_id is None
                    else rec.edge_to_vessel_nearest_graph_edge_label_id,
                    ""
                    if rec.edge_to_vessel_nearest_branch_order is None
                    else rec.edge_to_vessel_nearest_branch_order,
                    f"{rec.centroid_to_vessel_distance_microns:.6g}",
                    f"{(rec.centroid_to_vessel_distance_microns - rec.edge_to_vessel_distance_microns):.6g}",
                    f"{rec.centroid_to_vessel_nearest_z:.6g}",
                    f"{rec.centroid_to_vessel_nearest_y:.6g}",
                    f"{rec.centroid_to_vessel_nearest_x:.6g}",
                    f"({rec.centroid_to_vessel_nearest_z:.6g}, {rec.centroid_to_vessel_nearest_y:.6g}, {rec.centroid_to_vessel_nearest_x:.6g})",
                    ""
                    if rec.centroid_to_vessel_nearest_graph_edge_label_id is None
                    else rec.centroid_to_vessel_nearest_graph_edge_label_id,
                    ""
                    if rec.centroid_to_vessel_nearest_branch_order is None
                    else rec.centroid_to_vessel_nearest_branch_order,
                ]
            )
    return out


def export_object_distance_summary_csv(
    records: list[ObjectDistanceRecord],
    output_csv_path: str | Path,
) -> Path:
    """Write overall and per-branch dual-distance summary to CSV."""
    out = Path(output_csv_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    edge_distances = np.asarray(
        [float(r.edge_to_vessel_distance_microns) for r in records], dtype=float
    )
    centroid_distances = np.asarray(
        [float(r.centroid_to_vessel_distance_microns) for r in records], dtype=float
    )
    grouped_edge: dict[str, list[float]] = {}
    grouped_centroid: dict[str, list[float]] = {}
    grouped_delta_by_edge_group: dict[str, list[float]] = {}
    grouped_delta_by_centroid_group: dict[str, list[float]] = {}
    for rec in records:
        group_edge = rec.edge_to_vessel_nearest_branch_order or "Unassigned"
        group_centroid = rec.centroid_to_vessel_nearest_branch_order or "Unassigned"
        delta = (
            float(rec.centroid_to_vessel_distance_microns)
            - float(rec.edge_to_vessel_distance_microns)
        )
        grouped_edge.setdefault(group_edge, []).append(
            float(rec.edge_to_vessel_distance_microns)
        )
        grouped_centroid.setdefault(group_centroid, []).append(
            float(rec.centroid_to_vessel_distance_microns)
        )
        grouped_delta_by_edge_group.setdefault(group_edge, []).append(delta)
        grouped_delta_by_centroid_group.setdefault(group_centroid, []).append(delta)

    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Group",
                "Measurement Type",
                "Object Count",
                "Mean Distance (um)",
                "Median Distance (um)",
                "Std Distance (um)",
                "Min Distance (um)",
                "Max Distance (um)",
                "Mean Delta (Centroid-Edge) (um)",
                "Median Delta (Centroid-Edge) (um)",
                "Std Delta (Centroid-Edge) (um)",
            ]
        )
        if edge_distances.size == 0:
            writer.writerow(
                ["Overall", "edge_to_vessel", 0, "", "", "", "", "", "", "", ""]
            )
            writer.writerow(
                ["Overall", "centroid_to_vessel", 0, "", "", "", "", "", "", "", ""]
            )
            return out
        deltas = centroid_distances - edge_distances
        for metric_name, distances in (
            ("edge_to_vessel", edge_distances),
            ("centroid_to_vessel", centroid_distances),
        ):
            writer.writerow(
                [
                    "Overall",
                    metric_name,
                    int(distances.size),
                    f"{float(np.mean(distances)):.6g}",
                    f"{float(np.median(distances)):.6g}",
                    f"{float(np.std(distances)):.6g}",
                    f"{float(np.min(distances)):.6g}",
                    f"{float(np.max(distances)):.6g}",
                    f"{float(np.mean(deltas)):.6g}",
                    f"{float(np.median(deltas)):.6g}",
                    f"{float(np.std(deltas)):.6g}",
                ]
            )
        for metric_name, grouped in (
            ("edge_to_vessel", grouped_edge),
            ("centroid_to_vessel", grouped_centroid),
        ):
            for group in sorted(grouped):
                arr = np.asarray(grouped[group], dtype=float)
                if metric_name == "edge_to_vessel":
                    delta_group = np.asarray(
                        grouped_delta_by_edge_group.get(group, []), dtype=float
                    )
                else:
                    delta_group = np.asarray(
                        grouped_delta_by_centroid_group.get(group, []), dtype=float
                    )
                writer.writerow(
                    [
                        group,
                        metric_name,
                        int(arr.size),
                        f"{float(np.mean(arr)):.6g}",
                        f"{float(np.median(arr)):.6g}",
                        f"{float(np.std(arr)):.6g}",
                        f"{float(np.min(arr)):.6g}",
                        f"{float(np.max(arr)):.6g}",
                        ""
                        if delta_group.size == 0
                        else f"{float(np.mean(delta_group)):.6g}",
                        ""
                        if delta_group.size == 0
                        else f"{float(np.median(delta_group)):.6g}",
                        ""
                        if delta_group.size == 0
                        else f"{float(np.std(delta_group)):.6g}",
                    ]
                )
    return out


def run_3d_measurement_to_cell_mask(
    *,
    graph: nx.MultiGraph,
    cell_mask_path: str | Path,
    output_dir: str | Path,
    image_stem: str,
    voxel_size_xyz: tuple[float, float, float],
    vessel_mask_path: str | Path | None = None,
    vessel_mask: np.ndarray | None = None,
    cell_mask_h5_dataset_name: str | None = None,
    vessel_mask_h5_dataset_name: str | None = None,
    connectivity: int = 3,
    axis_order: str = CANONICAL_AXIS_ORDER,
    voxel_size_override_xyz=None,
    voxel_size_policy: str = "auto",
    use_memmap: bool = False,
    memmap_directory: str | Path | None = None,
) -> dict[str, Any]:
    """End-to-end 3D cell-mask distance analysis with CSV export.

    The vessels are *vessel_mask_path* when given -- a mask of the whole input
    image, cut to the network's crop box as the cell mask is -- else
    *vessel_mask*, a mask already in the network's frame (the pipeline passes
    the run's own segmentation), and only without either the vessels as the
    graph models them, tubes of each edge's ``diameter_um``.

    *voxel_size_xyz* is the run's physical ``(x, y, z)`` voxel size, converted
    to canonical array order ``(z, y, x)`` before any index scaling. Each mask
    read from disk is held to it as the run's vessel masks are: one with no
    spacing of its own takes the run's under the ``auto``
    *voxel_size_policy*, *voxel_size_override_xyz* applies to it as to the
    main image, and one whose resolved spacing differs is refused.

    *use_memmap* (the low-RAM option) loads the masks and builds every
    volume-sized intermediate on disk in *memmap_directory*, releasing each
    when done; see :func:`compute_object_to_vessel_distances`. The CSVs are
    the same.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    disk_backed: list[np.ndarray] = []
    try:
        records = _run_3d_measurement_records(
            graph=graph,
            cell_mask_path=cell_mask_path,
            voxel_size_xyz=voxel_size_xyz,
            vessel_mask_path=vessel_mask_path,
            vessel_mask=vessel_mask,
            cell_mask_h5_dataset_name=cell_mask_h5_dataset_name,
            vessel_mask_h5_dataset_name=vessel_mask_h5_dataset_name,
            connectivity=connectivity,
            axis_order=axis_order,
            voxel_size_override_xyz=voxel_size_override_xyz,
            voxel_size_policy=voxel_size_policy,
            use_memmap=use_memmap,
            memmap_directory=memmap_directory,
            disk_backed=disk_backed,
        )
    finally:
        for array in disk_backed:
            if isinstance(array, np.memmap):
                release_memmap_array(array)

    details_csv = output_dir / f"{image_stem}_cell_to_vessel_3d_dual_distances.csv"
    summary_csv = output_dir / f"{image_stem}_cell_to_vessel_3d_dual_distances_summary.csv"
    export_object_distance_details_csv(records, details_csv)
    export_object_distance_summary_csv(records, summary_csv)

    return {
        "object_count": len(records),
        "details_csv_path": details_csv,
        "summary_csv_path": summary_csv,
    }


def _load_mask_in_network_frame(
    path: str | Path,
    *,
    description: str,
    h5_dataset_name: str | None,
    graph: nx.MultiGraph,
    voxel_size_xyz: tuple[float, float, float],
    voxel_size_override_xyz,
    voxel_size_policy: str,
    axis_order: str,
    use_memmap: bool,
    memmap_directory: str | Path | None,
    disk_backed: list[np.ndarray],
) -> tuple[np.ndarray, tuple[int, ...]]:
    """``(mask cut to the network's crop box, the whole mask's shape)``, its
    voxel size held to the run's."""
    mask, metadata_xyz, status = load_binary_mask_voxel_size_and_status(
        path,
        h5_dataset_name=h5_dataset_name,
        axis_order=axis_order,
        description=description,
        use_memmap=use_memmap,
        memmap_directory=memmap_directory,
    )
    disk_backed.append(mask)
    # The mask is on the image's grid as read; the run's voxel size is the
    # downsampled grid's, so a mask with no spacing of its own takes the
    # source's, and one with its own is scaled as the image was.
    downsample = graph_downsample(graph)
    source_voxel_size_xyz = (
        voxel_size_xyz if downsample is None else downsample.source_voxel_size_xyz
    )
    resolved_xyz = _resolve_mask_voxel_size_xyz(
        metadata_xyz,
        status,
        voxel_size_override_xyz=voxel_size_override_xyz,
        voxel_size_policy=voxel_size_policy,
        fallback_voxel_size_xyz=source_voxel_size_xyz,
    )
    if downsample is not None:
        resolved_xyz = downsample.scale_voxel_size_xyz(resolved_xyz)
    if not np.allclose(
        resolved_xyz, voxel_size_xyz, rtol=_VOXEL_SIZE_RELATIVE_TOLERANCE, atol=0.0
    ):
        raise ValueError(
            f"The {description}'s voxel size (x, y, z) is {tuple(resolved_xyz)} um, "
            f"but the run's is {tuple(voxel_size_xyz)} um: the two cannot be overlaid. "
            "Fix: save the mask with the input image's voxel size, or set "
            "voxel_size_override_xyz with voxel_size_policy 'override'."
        )
    whole_shape = tuple(int(n) for n in mask.shape)
    # The graph is rasterised into these masks' shape from its positions,
    # which are measured from the crop_on_import box's corner when the image
    # was cut, on the grid it was downsampled to: the masks are cut to the
    # same box and reduced by the same blocks.
    cropped = mask_volume_like_graph(
        mask, graph, description=description,
        use_memmap=use_memmap, memmap_directory=memmap_directory,
    )
    if cropped is not mask:
        disk_backed.append(cropped)
    return cropped, whole_shape


def _run_3d_measurement_records(
    *,
    graph: nx.MultiGraph,
    cell_mask_path: str | Path,
    voxel_size_xyz: tuple[float, float, float],
    vessel_mask_path: str | Path | None,
    vessel_mask: np.ndarray | None,
    cell_mask_h5_dataset_name: str | None,
    vessel_mask_h5_dataset_name: str | None,
    connectivity: int,
    axis_order: str,
    voxel_size_override_xyz,
    voxel_size_policy: str,
    use_memmap: bool,
    memmap_directory: str | Path | None,
    disk_backed: list[np.ndarray],
) -> list[ObjectDistanceRecord]:
    """Load, build and measure; every disk-backed array made here is appended
    to *disk_backed* for the caller to release."""
    target_voxel_size_xyz = tuple(float(v) for v in voxel_size_xyz)
    target_voxel_size = voxel_size_zyx_from_xyz(target_voxel_size_xyz)
    loading = dict(
        graph=graph,
        voxel_size_xyz=target_voxel_size_xyz,
        voxel_size_override_xyz=voxel_size_override_xyz,
        voxel_size_policy=voxel_size_policy,
        axis_order=axis_order,
        use_memmap=use_memmap,
        memmap_directory=memmap_directory,
        disk_backed=disk_backed,
    )
    cell_mask, whole_cell_shape = _load_mask_in_network_frame(
        cell_mask_path,
        description="cell mask",
        h5_dataset_name=cell_mask_h5_dataset_name,
        **loading,
    )

    if vessel_mask_path is not None:
        vessels, whole_vessel_shape = _load_mask_in_network_frame(
            vessel_mask_path,
            description="3D-distance vessel mask",
            h5_dataset_name=vessel_mask_h5_dataset_name,
            **loading,
        )
        if whole_vessel_shape != whole_cell_shape:
            raise ValueError(
                "cell and vessel masks must share shape for distance analysis. "
                f"Got cell={whole_cell_shape}, vessel={whole_vessel_shape}"
            )
    elif vessel_mask is not None:
        vessels = _to_binary_volume_for_skeletonization(
            vessel_mask, use_memmap=use_memmap, memmap_directory=memmap_directory
        )
        if vessels is not vessel_mask:
            disk_backed.append(vessels)
        if vessels.shape != cell_mask.shape:
            raise ValueError(
                f"The vessel mask is {tuple(vessels.shape)} voxels, but the cell mask "
                f"(cut to the network's crop box) is {tuple(cell_mask.shape)}: the cell "
                "mask must have the input image's shape."
            )
    else:
        vessels = _graph_vessel_mask(
            graph,
            volume_shape=cell_mask.shape,
            voxel_size_zyx=target_voxel_size,
            use_memmap=use_memmap,
            memmap_directory=memmap_directory,
        )
        disk_backed.append(vessels)

    edge_label_volume, edge_to_bo = _build_edge_voxel_lookup(
        graph,
        volume_shape=cell_mask.shape,
        voxel_size_zyx=target_voxel_size,
        use_memmap=use_memmap,
        memmap_directory=memmap_directory,
    )
    disk_backed.append(edge_label_volume)

    return compute_object_to_vessel_distances(
        object_mask=cell_mask,
        vessel_mask=vessels,
        voxel_size_zyx=target_voxel_size,
        graph_edge_label_volume=edge_label_volume,
        edge_label_to_branch_order=edge_to_bo,
        connectivity=connectivity,
        use_memmap=use_memmap,
        memmap_directory=memmap_directory,
    )
