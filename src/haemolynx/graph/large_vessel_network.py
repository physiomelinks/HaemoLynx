"""Inlet/outlet seeding for large vessels kept in the network as Large_Art/Large_Ven.

A large arteriole or venule mask that is meant to stay in the network (see
``cut_at_large_vessel_volumes.py`` for the alternative, where it is cut away)
still needs an inlet/outlet: the point where the real vessel was truncated by
the imaging field of view, not an internal narrowing. Overlap-based terminal
selection (``automated_vessel_assignment.py``) only ever looks at the graph's
*existing* degree-1 nodes and has no notion of "this is where the field of
view cut the vessel off" versus "this is a genuine interior arm tip" -- both
read identically as a degree-1 node overlapping the mask. This module answers
that question from the mask's own geometry instead: a component that touches
the image volume's own face was truncated by the field of view; one that
doesn't was not.
"""
from __future__ import annotations

import logging
from typing import Any, Iterable

import numpy as np
import networkx as nx

from ._helpers import points_inside_mask, sort_nodes
from .boundaries import _terminal_nodes_and_position_map, warn_about_coordinate_snaps

logger = logging.getLogger(__name__)

#: Full 26-neighbour connectivity, matching graph/large_vessels.py's convention.
_STRUCTURE = np.ones((3, 3, 3), dtype=np.uint8)

#: A face contact is a stump only while its footprint is at most this many
#: times the area of the disc its own inscribed radius spans (pi r^2, the
#: vessel's cross-section): a vessel cut by the face leaves about one
#: cross-section (an oblique cut up to 75 degrees stays under 4), while one
#: lying *along* the face -- a pial vessel against the top of a stack --
#: leaves a strip many cross-sections long. On E14.5 MCA (0.98 x 0.98 x 1 um)
#: the real stumps at the x faces read 1.6-3.0 and the vessels' flanks on the
#: z=0 face 5.0-27 (`outputs/` run of 2026-10-11; one dataset, one reader --
#: not a marked grid). Those flanks seeded 8 inlets and 11 outlets, each
#: snapped 11-36 um to some capillary tip.
STUMP_MAX_FOOTPRINT_CROSS_SECTIONS = 4.0


def _footprint_cross_sections(
    footprint: np.ndarray, in_plane_spacing: tuple[float, float]
) -> float:
    """A face footprint's area over the area of the disc its inscribed
    radius spans; ``inf`` for an empty footprint."""
    from scipy.ndimage import distance_transform_edt

    if not footprint.any():
        return float("inf")
    # Padded, so the array's own edge counts as outside the footprint.
    inscribed = distance_transform_edt(np.pad(footprint, 1), sampling=in_plane_spacing)
    radius = float(inscribed[1:-1, 1:-1][footprint].max())
    area = float(np.count_nonzero(footprint)) * float(in_plane_spacing[0]) * float(
        in_plane_spacing[1]
    )
    return area / (np.pi * radius * radius) if radius > 0 else float("inf")


def find_large_vessel_mask_stump_points(
    mask: np.ndarray,
    *,
    voxel_size_zyx: tuple[float, float, float],
    min_component_fraction: float = 0.0,
    max_footprint_cross_sections: float = STUMP_MAX_FOOTPRINT_CROSS_SECTIONS,
) -> list[np.ndarray]:
    """Physical-coordinate points where *mask* is cut off by the volume's edge.

    One point per (face, connected component) touch, at the centroid of
    *that face's own touching voxels only* -- not the whole component, whose
    bulk centroid can sit far from the true stump cross-section for a long
    trunk and would then snap to the wrong graph terminal. Kept per-face
    rather than merged across every face one component touches: a vessel
    that runs corner-to-corner through the whole volume touches two
    different faces at two genuinely different ends, and averaging those
    into one point midway between them would not describe either stump. A
    component that never reaches a face (a vessel that genuinely ends
    inside the tissue, not truncated by the field of view) contributes no
    point.

    Two contacts are not stumps. A component smaller than
    *min_component_fraction* of the mask's largest one is a speck the
    classifier left at the face, not a large vessel (E14.5's arteriole mask
    had 468 components; the vessel was one, the next 0.5% of it). A contact
    whose footprint spans more than *max_footprint_cross_sections* of its own
    cross-section is a vessel lying along the face, not cut by it (see
    :data:`STUMP_MAX_FOOTPRINT_CROSS_SECTIONS`).
    """
    from haemolynx.preprocessing.slab_labels import ForegroundLabels

    mask_bool = np.asarray(mask, dtype=bool)
    if not mask_bool.any():
        return []

    # Components labelled a slab of planes at a time and held at their voxels
    # (``preprocessing.slab_labels.ForegroundLabels``), not in a label volume
    # (8 bytes a voxel once a stack has 2**31 voxels).
    labeled = ForegroundLabels(mask_bool, _STRUCTURE)
    scale = np.asarray(voxel_size_zyx, dtype=float)
    sizes = labeled.sizes
    min_size = float(min_component_fraction) * float(sizes[1:].max())

    points: list[np.ndarray] = []
    skipped_small = 0
    skipped_flank = 0
    for axis in range(mask_bool.ndim):
        in_plane = tuple(float(scale[a]) for a in range(mask_bool.ndim) if a != axis)
        for face in (0, -1):
            face_index = [slice(None)] * mask_bool.ndim
            face_index[axis] = face
            face_axis_value = face_index[axis] % labeled.shape[axis]
            face_box = [slice(0, n) for n in labeled.shape]
            face_box[axis] = slice(face_axis_value, face_axis_value + 1)
            face_labels = np.squeeze(labeled.read(tuple(face_box)), axis=axis)
            for component_id in np.unique(face_labels):
                if component_id == 0:
                    continue
                if sizes[component_id] < min_size:
                    skipped_small += 1
                    continue
                footprint = face_labels == component_id
                if _footprint_cross_sections(footprint, in_plane) > float(
                    max_footprint_cross_sections
                ):
                    skipped_flank += 1
                    continue
                face_coords = np.argwhere(footprint)
                # Restore the dropped axis so every point is a full (z, y, x)
                # voxel index, not the 2D coordinate within the face slice.
                full_coords = np.insert(face_coords, axis, face_axis_value, axis=1)
                centroid_voxels = full_coords.mean(axis=0)
                points.append(centroid_voxels * scale)
    if skipped_small or skipped_flank:
        logger.info(
            "Large-vessel mask stumps: %d kept; %d face contact(s) of components "
            "below %.0f%% of the largest skipped, %d skipped as a vessel lying "
            "along the face (footprint over %.1f cross-sections).",
            len(points),
            skipped_small,
            100.0 * float(min_component_fraction),
            skipped_flank,
            float(max_footprint_cross_sections),
        )
    return points


def _snap_stumps_to_terminals(
    G: nx.Graph,
    points: list[np.ndarray],
    mask: np.ndarray,
    *,
    voxel_size_zyx: tuple[float, float, float],
    coordinates_setting_name: str,
    exclude_nodes: Iterable[Any] | None,
) -> list[Any]:
    """Each stump point's nearest terminal among those inside *mask* (every
    terminal, with a warning, when none is), as the ``coordinates`` boundary
    method snaps and warns (:func:`~haemolynx.graph.boundaries.
    warn_about_coordinate_snaps`), but over those candidates only."""
    terminals, pos = _terminal_nodes_and_position_map(G)
    excluded = set(exclude_nodes or [])
    candidates = [node for node in terminals if node not in excluded]
    if not candidates:
        return []
    inside = points_inside_mask(
        np.asarray([pos[node] for node in candidates], dtype=float),
        np.asarray(mask, dtype=bool),
        voxel_size_zyx=voxel_size_zyx,
    )
    if inside.any():
        candidates = [node for node, is_inside in zip(candidates, inside) if is_inside]
    else:
        logger.warning(
            "No terminal node lies inside the large-vessel mask; its stump snaps "
            "to the nearest terminal of any kind."
        )
    selected: list[Any] = []
    targets: list[np.ndarray] = []
    distances: list[float] = []
    for point in points:
        target = np.asarray(point, dtype=float)
        nearest = min(candidates, key=lambda node: float(np.linalg.norm(pos[node] - target)))
        selected.append(nearest)
        targets.append(target)
        distances.append(float(np.linalg.norm(pos[nearest] - target)))
    warn_about_coordinate_snaps(
        G, targets, distances, setting_name=coordinates_setting_name, pos=pos, terminals=candidates
    )
    return sort_nodes(set(selected))


def select_large_vessel_mask_stump_terminal_nodes_for_role(
    G: nx.Graph,
    mask: np.ndarray,
    *,
    node_role: str,
    voxel_size_zyx: tuple[float, float, float],
    image_shape: tuple[int, ...],
    coordinates_setting_name: str,
    exclude_nodes: Iterable[Any] | None = None,
    min_component_fraction: float = 0.0,
) -> list[Any]:
    """One side's terminal nodes at *mask*'s own image-edge stump.

    *mask*'s face-touching components (see
    :func:`find_large_vessel_mask_stump_points`, which also takes
    *min_component_fraction*) are snapped to their nearest graph terminal
    *inside the mask* via the same "coordinates" method manual boundary
    selection already uses (:func:`haemolynx.graph.boundaries.
    select_boundary_nodes_by_method`), so a snap that lands suspiciously far
    from the mask surfaces the same :class:`~haemolynx.graph.boundaries.
    BoundaryCoordinateWarning` a bad manual coordinate would. Terminals
    outside the mask are not candidates (the stump of a wide vessel used to
    snap to a capillary tip nearer its face centroid than the vessel's own
    centreline end), unless no terminal is inside it at all. A mask with no
    face-touching component (the vessel ends inside the tissue, not at the
    field of view's edge) yields no node. *exclude_nodes* keeps the other
    side's already-chosen node(s) out of reach, so an arteriole and venule
    mask that both happen to touch the same corner of the volume cannot snap
    to the same terminal for both the inlet and the outlet.
    """
    if node_role not in {"inlet", "outlet"}:
        raise ValueError("node_role must be 'inlet' or 'outlet'.")
    del image_shape  # the same signature as the other boundary selectors
    points = find_large_vessel_mask_stump_points(
        mask, voxel_size_zyx=voxel_size_zyx, min_component_fraction=min_component_fraction
    )
    if not points:
        return []
    return _snap_stumps_to_terminals(
        G,
        points,
        mask,
        voxel_size_zyx=voxel_size_zyx,
        coordinates_setting_name=coordinates_setting_name,
        exclude_nodes=exclude_nodes,
    )


def select_large_vessel_stump_terminal_nodes(
    G: nx.Graph,
    *,
    large_arteriole_mask: np.ndarray,
    large_venule_mask: np.ndarray,
    voxel_size_zyx: tuple[float, float, float],
    image_shape: tuple[int, ...],
) -> tuple[list[Any], list[Any]]:
    """Inlet/outlet nodes at the large-vessel masks' own image-edge stumps.

    Both sides of :func:`select_large_vessel_mask_stump_terminal_nodes_for_role`.
    """
    inlet_nodes = select_large_vessel_mask_stump_terminal_nodes_for_role(
        G,
        large_arteriole_mask,
        node_role="inlet",
        voxel_size_zyx=voxel_size_zyx,
        image_shape=image_shape,
        coordinates_setting_name="large_arteriole_mask stump",
    )
    outlet_nodes = select_large_vessel_mask_stump_terminal_nodes_for_role(
        G,
        large_venule_mask,
        node_role="outlet",
        voxel_size_zyx=voxel_size_zyx,
        image_shape=image_shape,
        coordinates_setting_name="large_venule_mask stump",
    )
    return inlet_nodes, outlet_nodes
