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
from typing import Any

import numpy as np
import networkx as nx
from scipy.ndimage import label

from .boundaries import select_boundary_nodes_by_method

logger = logging.getLogger(__name__)

#: Full 26-neighbour connectivity, matching graph/large_vessels.py's convention.
_STRUCTURE = np.ones((3, 3, 3), dtype=np.uint8)


def find_large_vessel_mask_stump_points(
    mask: np.ndarray,
    *,
    voxel_size_zyx: tuple[float, float, float],
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
    """
    mask_bool = np.asarray(mask, dtype=bool)
    if not mask_bool.any():
        return []

    labeled, _n_labels = label(mask_bool, structure=_STRUCTURE)
    scale = np.asarray(voxel_size_zyx, dtype=float)

    points: list[np.ndarray] = []
    for axis in range(mask_bool.ndim):
        for face in (0, -1):
            face_index = [slice(None)] * mask_bool.ndim
            face_index[axis] = face
            face_labels = labeled[tuple(face_index)]
            face_axis_value = face_index[axis] % labeled.shape[axis]
            for component_id in np.unique(face_labels):
                if component_id == 0:
                    continue
                face_coords = np.argwhere(face_labels == component_id)
                # Restore the dropped axis so every point is a full (z, y, x)
                # voxel index, not the 2D coordinate within the face slice.
                full_coords = np.insert(face_coords, axis, face_axis_value, axis=1)
                centroid_voxels = full_coords.mean(axis=0)
                points.append(centroid_voxels * scale)
    return points


def select_large_vessel_stump_terminal_nodes(
    G: nx.Graph,
    *,
    large_arteriole_mask: np.ndarray,
    large_venule_mask: np.ndarray,
    voxel_size_zyx: tuple[float, float, float],
    image_shape: tuple[int, ...],
) -> tuple[list[Any], list[Any]]:
    """Inlet/outlet nodes at the large-vessel masks' own image-edge stumps.

    Each mask's face-touching components (see
    :func:`find_large_vessel_mask_stump_points`) are snapped to their nearest
    graph terminal via the same "coordinates" method manual boundary
    selection already uses (:func:`haemolynx.graph.boundaries.
    select_boundary_nodes_by_method`), so a snap that lands suspiciously far
    from the mask surfaces the same :class:`~haemolynx.graph.boundaries.
    BoundaryCoordinateWarning` a bad manual coordinate would. A mask with no
    face-touching component (the vessel ends inside the tissue, not at the
    field of view's edge) yields no node for that side.
    """
    arteriole_points = find_large_vessel_mask_stump_points(
        large_arteriole_mask, voxel_size_zyx=voxel_size_zyx
    )
    venule_points = find_large_vessel_mask_stump_points(
        large_venule_mask, voxel_size_zyx=voxel_size_zyx
    )
    inlet_nodes = select_boundary_nodes_by_method(
        G,
        image_shape,
        method="coordinates",
        node_role="inlet",
        coordinates=arteriole_points,
        coordinates_setting_name="large_arteriole_mask stump",
    )
    outlet_nodes = select_boundary_nodes_by_method(
        G,
        image_shape,
        method="coordinates",
        node_role="outlet",
        coordinates=venule_points,
        coordinates_setting_name="large_venule_mask stump",
    )
    return inlet_nodes, outlet_nodes
