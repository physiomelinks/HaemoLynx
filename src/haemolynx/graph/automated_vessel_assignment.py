"""Automatic terminal-node assignment from arteriole/venule masks."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterable

import networkx as nx
import numpy as np
from scipy.ndimage import distance_transform_edt

# Aliased because this module also uses ``edge_id`` as a local variable name.
from ._helpers import edge_id as _edge_id, sort_nodes as _sort_nodes
from .cut_at_large_vessel_volumes import CUT_FROM_LARGE_VESSEL
from .large_vessels import (
    exclude_smaller_overlapping_large_vessel_components,
    exclude_smaller_overlapping_small_vessel_components,
)

logger = logging.getLogger(__name__)


def _terminal_nodes_with_position_pairs(G: nx.Graph) -> list[tuple[Any, np.ndarray]]:
    node_pos = nx.get_node_attributes(G, "pos")
    terminals: list[tuple[Any, np.ndarray]] = []
    for node_id, degree in G.degree():
        if degree != 1 or node_id not in node_pos:
            continue
        terminals.append((node_id, np.asarray(node_pos[node_id], dtype=float)))
    return terminals


def _position_to_mask_index(
    position_zyx: np.ndarray,
    voxel_size_zyx: tuple[float, float, float],
    mask_shape: tuple[int, ...],
) -> tuple[int, int, int] | None:
    voxel_size = np.asarray(voxel_size_zyx, dtype=float)
    if voxel_size.shape != (3,) or np.any(voxel_size <= 0):
        raise ValueError(
            f"voxel_size_zyx must be three positive values, got {voxel_size_zyx}."
        )
    if len(mask_shape) != 3:
        raise ValueError(f"Expected a 3D mask shape, got {mask_shape}.")

    voxel_index = np.rint(position_zyx / voxel_size).astype(int)
    if np.any(voxel_index < 0):
        return None
    if np.any(voxel_index >= np.asarray(mask_shape, dtype=int)):
        return None
    return (int(voxel_index[0]), int(voxel_index[1]), int(voxel_index[2]))


def _terminal_edge_sample_points(
    G: nx.Graph,
    node_id: Any,
    node_pos: np.ndarray,
    *,
    max_sample_points: int = 25,
    voxel_size_zyx: tuple[float, float, float] | None = None,
) -> np.ndarray:
    """Collect sample points near a terminal node along its incident edge.

    With *voxel_size_zyx*: the edge's centreline sampled every finest voxel
    spacing, within *max_sample_points* of those steps of the node -- the same
    length of vessel whichever way it runs. Without, its *max_sample_points*
    nearest vertices, which along a 2 um z spanned four times the length they
    did in-plane.
    """
    if max_sample_points <= 0:
        return np.asarray([node_pos], dtype=float)

    edge_voxels: np.ndarray | None = None
    if isinstance(G, nx.MultiGraph):
        incident_edges = list(G.edges(node_id, keys=True, data=True))
        if incident_edges:
            edge_data = incident_edges[0][3]
            voxels = edge_data.get("voxels")
            if voxels is not None:
                arr = np.asarray(voxels, dtype=float)
                if arr.ndim == 2 and arr.shape[1] == 3 and arr.size > 0:
                    edge_voxels = arr
    else:
        incident_edges = list(G.edges(node_id, data=True))
        if incident_edges:
            edge_data = incident_edges[0][2]
            voxels = edge_data.get("voxels")
            if voxels is not None:
                arr = np.asarray(voxels, dtype=float)
                if arr.ndim == 2 and arr.shape[1] == 3 and arr.size > 0:
                    edge_voxels = arr

    if edge_voxels is None:
        return np.asarray([node_pos], dtype=float)

    if voxel_size_zyx is not None:
        step = float(min(voxel_size_zyx))
        evenly = _evenly_spaced(edge_voxels, step)
        distances = np.linalg.norm(evenly - node_pos.reshape(1, 3), axis=1)
        samples = evenly[distances <= max_sample_points * step + 1e-9]
        return np.unique(np.vstack([samples, node_pos.reshape(1, 3)]), axis=0)
    distances = np.linalg.norm(edge_voxels - node_pos.reshape(1, 3), axis=1)
    nearest_idx = np.argsort(distances)[: max_sample_points]
    samples = edge_voxels[nearest_idx]
    samples = np.vstack([samples, node_pos.reshape(1, 3)])
    return np.unique(samples, axis=0)


def _evenly_spaced(polyline: np.ndarray, step_um: float) -> np.ndarray:
    from haemolynx.geometry import resample_at_step

    return resample_at_step(np.asarray(polyline, dtype=float), step_um)


def _mask_midpoint_physical(
    mask: np.ndarray,
    voxel_size_zyx: tuple[float, float, float],
) -> np.ndarray:
    points_zyx = np.argwhere(mask.astype(bool, copy=False))
    if points_zyx.size == 0:
        return np.asarray([np.inf, np.inf, np.inf], dtype=float)
    voxel_size = np.asarray(voxel_size_zyx, dtype=float)
    return np.mean(points_zyx.astype(float), axis=0) * voxel_size


def _mask_principal_axis(
    mask: np.ndarray, voxel_size_zyx: tuple[float, float, float] | None = None
) -> int:
    """The array axis along which *mask* is longest, in microns: counted in
    voxels, a mask 100 um long in z (50 slices) and 75 um in x (150 voxels)
    read as running along x."""
    points_zyx = np.argwhere(mask.astype(bool, copy=False))
    if points_zyx.size == 0:
        return 0
    spans = np.ptp(points_zyx.astype(float), axis=0)
    if voxel_size_zyx is not None:
        spans = spans * np.asarray(voxel_size_zyx, dtype=float)
    return int(np.argmax(spans))


def _cross_section_midpoint_physical(
    mask: np.ndarray,
    voxel_size_zyx: tuple[float, float, float],
    intersection_point: np.ndarray | None,
) -> np.ndarray:
    points_zyx = np.argwhere(mask.astype(bool, copy=False))
    if points_zyx.size == 0 or intersection_point is None:
        return np.asarray([np.inf, np.inf, np.inf], dtype=float)
    axis = _mask_principal_axis(mask, voxel_size_zyx)
    voxel_size = np.asarray(voxel_size_zyx, dtype=float)
    intersection_index = np.rint(intersection_point / voxel_size).astype(int)
    target_slice = int(intersection_index[axis])
    slice_coords = points_zyx[:, axis]
    in_slice = points_zyx[slice_coords == target_slice]
    if in_slice.size == 0:
        nearest_slice = int(
            np.unique(slice_coords)[
                int(np.argmin(np.abs(np.unique(slice_coords) - target_slice)))
            ]
        )
        in_slice = points_zyx[slice_coords == nearest_slice]
    if in_slice.size == 0:
        return np.asarray([np.inf, np.inf, np.inf], dtype=float)
    return np.mean(in_slice.astype(float), axis=0) * voxel_size


def _overlap_fraction_and_intersection(
    sample_points: np.ndarray,
    mask: np.ndarray,
    voxel_size_zyx: tuple[float, float, float],
    node_pos: np.ndarray,
) -> tuple[float, np.ndarray | None]:
    points = np.asarray(sample_points, dtype=float).reshape(-1, 3)
    if not len(points):
        return 0.0, None
    index = np.rint(points / np.asarray(voxel_size_zyx, dtype=float)).astype(np.intp)
    in_bounds = np.all((index >= 0) & (index < np.asarray(mask.shape, dtype=np.intp)), axis=1)
    if not in_bounds.any():
        return 0.0, None
    valid = points[in_bounds]
    in_mask = np.asarray(mask[tuple(index[in_bounds].T)], dtype=bool)
    overlap_fraction = float(np.count_nonzero(in_mask)) / float(len(valid))
    if not in_mask.any():
        return overlap_fraction, None
    in_mask_arr = valid[in_mask]
    dists = np.linalg.norm(in_mask_arr - node_pos.reshape(1, 3), axis=1)
    return overlap_fraction, in_mask_arr[int(np.argmin(dists))]


def resolve_overlapping_terminal_node_assignment(
    G: nx.Graph,
    node_id: Any,
    *,
    node_pos: np.ndarray,
    large_arteriole_mask: np.ndarray,
    large_venule_mask: np.ndarray,
    voxel_size_zyx: tuple[float, float, float],
    max_sample_points: int = 25,
) -> str:
    """Resolve input/output assignment for a terminal node in both masks.

    Decision rule:
    1) Prefer shorter distance from overlap-entry point to vessel cross-section
       midpoint at the entry slice.
    2) If tied, prefer shorter distance to vessel volume midpoint.
    3) If still tied, use higher local overlap percentage near the node.
    """
    metrics = compute_overlapping_terminal_assignment_metrics(
        G,
        node_id,
        node_pos=node_pos,
        large_arteriole_mask=large_arteriole_mask,
        large_venule_mask=large_venule_mask,
        voxel_size_zyx=voxel_size_zyx,
        max_sample_points=max_sample_points,
    )
    arteriole_overlap = float(metrics["arteriole_overlap_fraction"])
    venule_overlap = float(metrics["venule_overlap_fraction"])
    arteriole_cross_section_dist = float(metrics["arteriole_cross_section_midpoint_distance"])
    venule_cross_section_dist = float(metrics["venule_cross_section_midpoint_distance"])
    if arteriole_cross_section_dist < venule_cross_section_dist:
        return "inlet"
    if venule_cross_section_dist < arteriole_cross_section_dist:
        return "outlet"

    arteriole_dist = float(metrics["arteriole_midpoint_distance"])
    venule_dist = float(metrics["venule_midpoint_distance"])
    if arteriole_dist < venule_dist:
        return "inlet"
    if venule_dist < arteriole_dist:
        return "outlet"

    if arteriole_overlap > venule_overlap:
        return "inlet"
    if venule_overlap > arteriole_overlap:
        return "outlet"
    # Final deterministic tie-break.
    return "inlet"


def compute_overlapping_terminal_assignment_metrics(
    G: nx.Graph,
    node_id: Any,
    *,
    node_pos: np.ndarray,
    large_arteriole_mask: np.ndarray,
    large_venule_mask: np.ndarray,
    voxel_size_zyx: tuple[float, float, float],
    max_sample_points: int = 25,
) -> dict[str, Any]:
    """Compute overlap and midpoint-distance metrics for overlap resolution."""
    samples = _terminal_edge_sample_points(
        G,
        node_id,
        node_pos,
        max_sample_points=max_sample_points,
        voxel_size_zyx=voxel_size_zyx,
    )
    arteriole_overlap, arteriole_intersection = _overlap_fraction_and_intersection(
        samples, large_arteriole_mask, voxel_size_zyx, node_pos
    )
    venule_overlap, venule_intersection = _overlap_fraction_and_intersection(
        samples, large_venule_mask, voxel_size_zyx, node_pos
    )
    arteriole_mid = _mask_midpoint_physical(large_arteriole_mask, voxel_size_zyx)
    venule_mid = _mask_midpoint_physical(large_venule_mask, voxel_size_zyx)
    arteriole_cross_section_mid = _cross_section_midpoint_physical(
        large_arteriole_mask, voxel_size_zyx, arteriole_intersection
    )
    venule_cross_section_mid = _cross_section_midpoint_physical(
        large_venule_mask, voxel_size_zyx, venule_intersection
    )
    arteriole_dist = np.inf
    venule_dist = np.inf
    arteriole_cross_section_dist = np.inf
    venule_cross_section_dist = np.inf
    if arteriole_intersection is not None and np.all(np.isfinite(arteriole_mid)):
        arteriole_dist = float(np.linalg.norm(arteriole_intersection - arteriole_mid))
    if venule_intersection is not None and np.all(np.isfinite(venule_mid)):
        venule_dist = float(np.linalg.norm(venule_intersection - venule_mid))
    if arteriole_intersection is not None and np.all(np.isfinite(arteriole_cross_section_mid)):
        arteriole_cross_section_dist = float(
            np.linalg.norm(arteriole_intersection - arteriole_cross_section_mid)
        )
    if venule_intersection is not None and np.all(np.isfinite(venule_cross_section_mid)):
        venule_cross_section_dist = float(
            np.linalg.norm(venule_intersection - venule_cross_section_mid)
        )
    return {
        "arteriole_overlap_fraction": float(arteriole_overlap),
        "venule_overlap_fraction": float(venule_overlap),
        "arteriole_cross_section_midpoint_distance": float(arteriole_cross_section_dist),
        "venule_cross_section_midpoint_distance": float(venule_cross_section_dist),
        "arteriole_midpoint_distance": float(arteriole_dist),
        "venule_midpoint_distance": float(venule_dist),
        "arteriole_intersection": None if arteriole_intersection is None else arteriole_intersection.copy(),
        "venule_intersection": None if venule_intersection is None else venule_intersection.copy(),
        "arteriole_cross_section_midpoint": arteriole_cross_section_mid.copy(),
        "venule_cross_section_midpoint": venule_cross_section_mid.copy(),
        "arteriole_midpoint": arteriole_mid.copy(),
        "venule_midpoint": venule_mid.copy(),
    }


def select_terminal_nodes_from_large_vessel_masks(
    G: nx.Graph,
    large_arteriole_mask: np.ndarray,
    large_venule_mask: np.ndarray,
    *,
    voxel_size_zyx: tuple[float, float, float],
    terminal_node_ids: set[Any] | None = None,
    allow_overlap: bool = False,
    exclude_smaller_overlapping_volumes: bool = False,
) -> tuple[list[Any], list[Any]]:
    """Assign degree-1 nodes to input/output groups by vessel-mask overlap.

    When ``exclude_smaller_overlapping_volumes`` is True, overlapping large-vessel
    components are pre-cleaned by removing only overlap voxels from the smaller
    component in each overlap pair before node assignment. A terminal the
    large-vessel cut made counts as inside the mask it was cut off at
    (``CUT_FROM_LARGE_VESSEL``): it sits one sample outside it.
    """
    if large_arteriole_mask.shape != large_venule_mask.shape:
        raise ValueError(
            "large_arteriole_mask and large_venule_mask must share a shape. "
            f"Got {large_arteriole_mask.shape} and {large_venule_mask.shape}."
        )

    arteriole_mask = large_arteriole_mask.astype(bool, copy=False)
    venule_mask = large_venule_mask.astype(bool, copy=False)
    if exclude_smaller_overlapping_volumes:
        cleaned_arteriole, cleaned_venule = (
            exclude_smaller_overlapping_large_vessel_components(
                arteriole_mask,
                venule_mask,
            )
        )
        if cleaned_arteriole is None or cleaned_venule is None:
            raise RuntimeError(
                "Internal error: expected cleaned masks when large masks are provided."
            )
        arteriole_mask = cleaned_arteriole
        venule_mask = cleaned_venule
    terminal_nodes = _terminal_nodes_with_position_pairs(G)
    if terminal_node_ids is not None:
        allowed = set(terminal_node_ids)
        terminal_nodes = [
            (node_id, node_pos)
            for node_id, node_pos in terminal_nodes
            if node_id in allowed
        ]
    if not terminal_nodes:
        return [], []

    inlet_nodes: set[Any] = set()
    outlet_nodes: set[Any] = set()
    for node_id, node_pos in terminal_nodes:
        index_zyx = _position_to_mask_index(
            node_pos,
            voxel_size_zyx=voxel_size_zyx,
            mask_shape=arteriole_mask.shape,
        )
        if index_zyx is None:
            continue
        cut = _cut_from(G, node_id)
        in_arteriole = bool(arteriole_mask[index_zyx]) or cut == "arteriole"
        in_venule = bool(venule_mask[index_zyx]) or cut == "venule"
        if in_arteriole and in_venule and not allow_overlap:
            assignment = resolve_overlapping_terminal_node_assignment(
                G,
                node_id,
                node_pos=node_pos,
                large_arteriole_mask=arteriole_mask,
                large_venule_mask=venule_mask,
                voxel_size_zyx=voxel_size_zyx,
            )
            if assignment == "inlet":
                inlet_nodes.add(node_id)
            else:
                outlet_nodes.add(node_id)
            continue
        if in_arteriole:
            inlet_nodes.add(node_id)
        if in_venule:
            outlet_nodes.add(node_id)

    if not allow_overlap:
        outlet_nodes -= inlet_nodes

    return _sort_nodes(inlet_nodes), _sort_nodes(outlet_nodes)


def _edge_sample_points_from_data(
    edge_data: dict[str, Any],
    endpoint_positions: tuple[np.ndarray, np.ndarray],
    *,
    step_um: float | None = None,
) -> np.ndarray:
    """Return physical sample points for an edge: its centreline sampled every
    *step_um*, so a fraction of them is a fraction of the vessel's length, or
    without a step its unique vertices -- one per voxel, a quarter as dense per
    micron along a 2 um z as in-plane, which under-weighted the part of a
    vessel running in z."""
    voxels = edge_data.get("voxels")
    if voxels is not None:
        arr = np.asarray(voxels, dtype=float)
        if arr.ndim == 2 and arr.shape[1] == 3 and arr.size > 0:
            if step_um is not None and len(arr) > 1:
                return _evenly_spaced(arr, float(step_um))
            return np.unique(arr, axis=0)
    p_u, p_v = endpoint_positions
    return np.unique(
        np.vstack([p_u.reshape(1, 3), p_v.reshape(1, 3)]),
        axis=0,
    )


def _in_bounds_voxel_indices(
    points_zyx: np.ndarray,
    voxel_size_zyx: tuple[float, float, float],
    mask_shape: tuple[int, ...],
) -> np.ndarray:
    """The voxel index of each physical point that lies inside *mask_shape*,
    as an ``(n, 3)`` integer array; points outside the volume are dropped."""
    points = np.asarray(points_zyx, dtype=float).reshape(-1, 3)
    if not len(points):
        return np.empty((0, 3), dtype=np.intp)
    voxel_size = np.asarray(voxel_size_zyx, dtype=float)
    if voxel_size.shape != (3,) or np.any(voxel_size <= 0):
        raise ValueError(
            f"voxel_size_zyx must be three positive values, got {voxel_size_zyx}."
        )
    if len(mask_shape) != 3:
        raise ValueError(f"Expected a 3D mask shape, got {mask_shape}.")
    index = np.rint(points / voxel_size).astype(np.intp)
    in_bounds = np.all((index >= 0) & (index < np.asarray(mask_shape, dtype=np.intp)), axis=1)
    return index[in_bounds]


def _fraction_in(mask: np.ndarray, voxel_indices: np.ndarray) -> float:
    """The fraction of *voxel_indices* that are True in *mask*; 0 with none."""
    if not len(voxel_indices):
        return 0.0
    return float(np.count_nonzero(mask[tuple(voxel_indices.T)])) / float(len(voxel_indices))


def _sample_overlap_fraction(
    sample_points: np.ndarray,
    mask: np.ndarray,
    *,
    voxel_size_zyx: tuple[float, float, float],
) -> float:
    """Return fraction of valid edge sample points that fall inside a mask.

    Vectorised: read point by point in Python, half a million samples took
    seven seconds against a twentieth of one."""
    return _fraction_in(
        mask, _in_bounds_voxel_indices(sample_points, voxel_size_zyx, mask.shape)
    )


#: An unlabelled edge joining two edges of one vessel type takes that type
#: when at least this fraction of ``minimum_overlap_fraction`` of it lies in
#: that type's mask: hysteresis, so one under-covered stretch of a vessel does
#: not read as the vessel ending twice.
GAP_FILL_FRACTION_OF_THRESHOLD = 0.5


def _label_edge_ids(G: nx.Graph) -> list[tuple[Any, Any, tuple[Any, Any, int], dict[str, Any]]]:
    """``(u, v, edge id, data)`` for every edge, the id in this module's form."""
    if isinstance(G, nx.MultiGraph):
        return [(u, v, _edge_id(u, v, key), data) for u, v, key, data in G.edges(keys=True, data=True)]
    return [
        (u, v, ((u, v, 0) if u <= v else (v, u, 0)), data) for u, v, data in G.edges(data=True)
    ]


def _fill_label_gaps(
    edges: list[tuple[Any, Any, tuple[Any, Any, int], dict[str, Any]]],
    fractions: dict[tuple[Any, Any, int], tuple[float, float]],
    labelled: dict[str, set[tuple[Any, Any, int]]],
    *,
    low_fraction: float | dict[str, float],
    names: tuple[str, str] = ("arteriole", "venule"),
    attribute: str = "mask_vessel_type",
) -> int:
    """Label every unlabelled edge whose two ends each meet another edge of
    one type and which lies at least *low_fraction* (one, or one per name) in
    that type's mask; repeat until nothing changes. *fractions* holds each
    edge's two fractions in the order of *names*, and the label is written
    as *attribute*. Returns how many edges were filled."""
    lows = low_fraction if isinstance(low_fraction, dict) else dict.fromkeys(names, low_fraction)
    filled = 0
    ends = {edge_id: (u, v) for u, v, edge_id, _data in edges}
    data_of = {edge_id: data for _u, _v, edge_id, data in edges}
    while True:
        touching: dict[str, dict[Any, set]] = {name: {} for name in labelled}
        for name, edge_ids in labelled.items():
            for edge_id in edge_ids:
                for node in ends.get(edge_id, ()):
                    touching[name].setdefault(node, set()).add(edge_id)
        taken = set().union(*labelled.values())
        added: list[tuple[str, tuple[Any, Any, int]]] = []
        for edge_id, edge_fractions in fractions.items():
            if edge_id in taken:
                continue
            u, v = ends[edge_id]
            options = []
            for name, fraction in zip(names, edge_fractions):
                if fraction < lows[name]:
                    continue
                if touching[name].get(u, set()) - {edge_id} and touching[name].get(v, set()) - {edge_id}:
                    options.append((fraction, name))
            if options:
                added.append((max(options)[1], edge_id))
        if not added:
            return filled
        for name, edge_id in added:
            labelled[name].add(edge_id)
            data_of[edge_id][attribute] = name
            filled += 1


def _edge_sample_voxels(
    G: nx.Graph,
    edges: list[tuple[Any, Any, tuple[Any, Any, int], dict[str, Any]]],
    *,
    voxel_size_zyx: tuple[float, float, float],
    mask_shape: tuple[int, ...],
) -> dict[tuple[Any, Any, int], np.ndarray]:
    """Each edge's in-volume sample voxels: its centreline sampled every
    finest voxel spacing, so a fraction of them is a fraction of its length.
    An edge whose ends have no position is left out."""
    node_positions = nx.get_node_attributes(G, "pos")
    if not node_positions:
        raise ValueError("Graph has no node positions ('pos').")
    step = float(min(voxel_size_zyx))
    voxels: dict[tuple[Any, Any, int], np.ndarray] = {}
    for u, v, edge_id, edge_data in edges:
        if u not in node_positions or v not in node_positions:
            continue
        pu = np.asarray(node_positions[u], dtype=float)
        pv = np.asarray(node_positions[v], dtype=float)
        samples = _edge_sample_points_from_data(edge_data, (pu, pv), step_um=step)
        voxels[edge_id] = _in_bounds_voxel_indices(samples, voxel_size_zyx, mask_shape)
    return voxels


def _apply_edge_labels(
    edges: list[tuple[Any, Any, tuple[Any, Any, int], dict[str, Any]]],
    fractions: dict[tuple[Any, Any, int], tuple[float, float]],
    *,
    threshold: float,
    allow_overlap: bool,
    labelled: dict[str, set[tuple[Any, Any, int]]],
) -> int:
    """Label each edge in *fractions* by its mask fractions, writing
    ``mask_vessel_type`` and adding it to *labelled*; returns how many edges
    met the threshold in both masks. An edge both masks reach goes to the
    one covering more of it (the arteriole on a tie), or to both with
    *allow_overlap*, labelled ``"overlap"``."""
    overlap_edges = 0
    for _u, _v, edge_id, edge_data in edges:
        if edge_id not in fractions:
            continue
        arteriole_fraction, venule_fraction = fractions[edge_id]
        in_arteriole = arteriole_fraction >= threshold
        in_venule = venule_fraction >= threshold
        if in_arteriole and in_venule:
            overlap_edges += 1
            if allow_overlap:
                labelled["arteriole"].add(edge_id)
                labelled["venule"].add(edge_id)
                edge_data["mask_vessel_type"] = "overlap"
            elif arteriole_fraction >= venule_fraction:
                labelled["arteriole"].add(edge_id)
                edge_data["mask_vessel_type"] = "arteriole"
            else:
                labelled["venule"].add(edge_id)
                edge_data["mask_vessel_type"] = "venule"
        elif in_arteriole:
            labelled["arteriole"].add(edge_id)
            edge_data["mask_vessel_type"] = "arteriole"
        elif in_venule:
            labelled["venule"].add(edge_id)
            edge_data["mask_vessel_type"] = "venule"
    return overlap_edges


def _boundary_state(
    G: nx.Graph,
    edges: list[tuple[Any, Any, tuple[Any, Any, int], dict[str, Any]]],
    arteriole_edges: set[tuple[Any, Any, int]],
    venule_edges: set[tuple[Any, Any, int]],
    *,
    allow_overlap: bool,
) -> dict[str, Any]:
    """The nodes and boundary nodes the edge labels imply, written as node
    ``mask_vessel_type`` too, and the labelled edges themselves. A boundary
    node is a labelled node with an incident edge not of its type: where the
    mask's vessels meet the rest."""
    arteriole_nodes: set[Any] = set()
    venule_nodes: set[Any] = set()
    for u, v, _key in arteriole_edges:
        arteriole_nodes.update((u, v))
    for u, v, _key in venule_edges:
        venule_nodes.update((u, v))
    if not allow_overlap:
        venule_nodes -= arteriole_nodes

    for _node_id, attrs in G.nodes(data=True):
        attrs.pop("mask_vessel_type", None)
    for node_id in arteriole_nodes:
        if node_id in G.nodes:
            G.nodes[node_id]["mask_vessel_type"] = "arteriole"
    for node_id in venule_nodes:
        if node_id in G.nodes and G.nodes[node_id].get("mask_vessel_type") != "arteriole":
            G.nodes[node_id]["mask_vessel_type"] = "venule"

    incident: dict[Any, set[tuple[Any, Any, int]]] = {}
    for u, v, edge_id, _data in edges:
        incident.setdefault(u, set()).add(edge_id)
        incident.setdefault(v, set()).add(edge_id)

    def _boundary_nodes_for(edge_ids: set[tuple[Any, Any, int]], labeled_nodes: set[Any]) -> list[Any]:
        return _sort_nodes(
            node_id
            for node_id in labeled_nodes
            if any(edge_id not in edge_ids for edge_id in incident.get(node_id, ()))
        )

    return {
        "arteriole_boundary_nodes": _boundary_nodes_for(arteriole_edges, arteriole_nodes),
        "venule_boundary_nodes": _boundary_nodes_for(venule_edges, venule_nodes),
        "arteriole_nodes": _sort_nodes(arteriole_nodes),
        "venule_nodes": _sort_nodes(venule_nodes),
        "arteriole_edges": set(arteriole_edges),
        "venule_edges": set(venule_edges),
        "arteriole_edge_count": len(arteriole_edges),
        "venule_edge_count": len(venule_edges),
    }


def _clear_mask_labels(G: nx.Graph, edges) -> None:
    """Drop every ``mask_vessel_type``, so reruns are deterministic."""
    for _node_id, attrs in G.nodes(data=True):
        attrs.pop("mask_vessel_type", None)
    for _u, _v, _edge, attrs in edges:
        attrs.pop("mask_vessel_type", None)


def infer_boundary_nodes_from_small_vessel_masks(
    G: nx.Graph,
    small_arteriole_mask: np.ndarray,
    small_venule_mask: np.ndarray,
    *,
    voxel_size_zyx: tuple[float, float, float],
    minimum_overlap_fraction: float = 0.5,
    allow_overlap: bool = False,
    exclude_smaller_overlapping_volumes: bool = False,
) -> dict[str, Any]:
    """Label mask-overlapping edges/nodes and infer arteriole/venule boundaries.

    Edges are marked as arteriole/venule when the fraction of sampled edge points
    inside the corresponding small-vessel mask meets `minimum_overlap_fraction`.
    Then, with hysteresis, an unlabelled edge joining two edges of one type
    takes that type when at least ``GAP_FILL_FRACTION_OF_THRESHOLD`` of that
    fraction of it lies in the type's mask: one under-covered edge in the
    middle of an arteriole used to leave a gap, and both its ends were then
    reported as the arteriole meeting the capillary bed. Associated endpoint
    nodes are given the same mask vessel type. Boundary nodes are the
    labeled-mask nodes that connect to at least one unlabeled edge, i.e.
    where the small-vessel mask region transitions into the capillary bed.
    ``arteriole_edges``/``venule_edges`` hold the labelled edges'
    ``(u, v, key)`` ids.
    """
    if small_arteriole_mask.shape != small_venule_mask.shape:
        raise ValueError(
            "small_arteriole_mask and small_venule_mask must share a shape. "
            f"Got {small_arteriole_mask.shape} and {small_venule_mask.shape}."
        )
    if not (0.0 <= float(minimum_overlap_fraction) <= 1.0):
        raise ValueError(
            "minimum_overlap_fraction must be in [0.0, 1.0]. "
            f"Got {minimum_overlap_fraction}."
        )

    arteriole_mask = small_arteriole_mask.astype(bool, copy=False)
    venule_mask = small_venule_mask.astype(bool, copy=False)
    if exclude_smaller_overlapping_volumes:
        cleaned_arteriole, cleaned_venule = (
            exclude_smaller_overlapping_small_vessel_components(
                arteriole_mask,
                venule_mask,
            )
        )
        if cleaned_arteriole is None or cleaned_venule is None:
            raise RuntimeError(
                "Internal error: expected cleaned masks when small masks are provided."
            )
        arteriole_mask = cleaned_arteriole
        venule_mask = cleaned_venule

    edges = _label_edge_ids(G)
    _clear_mask_labels(G, edges)
    voxels = _edge_sample_voxels(
        G, edges, voxel_size_zyx=voxel_size_zyx, mask_shape=arteriole_mask.shape
    )
    fractions = {
        edge_id: (_fraction_in(arteriole_mask, index), _fraction_in(venule_mask, index))
        for edge_id, index in voxels.items()
    }
    threshold = float(minimum_overlap_fraction)
    labelled: dict[str, set[tuple[Any, Any, int]]] = {"arteriole": set(), "venule": set()}
    overlap_edges = _apply_edge_labels(
        edges, fractions, threshold=threshold, allow_overlap=allow_overlap, labelled=labelled
    )
    gap_filled = _fill_label_gaps(
        edges, fractions, labelled, low_fraction=GAP_FILL_FRACTION_OF_THRESHOLD * threshold
    )
    if gap_filled:
        logger.info(
            "Small-vessel masks: %d under-covered edge(s) between two edges of one "
            "vessel type took that type (at least %.2f of the edge in its mask).",
            gap_filled,
            GAP_FILL_FRACTION_OF_THRESHOLD * threshold,
        )

    state = _boundary_state(
        G, edges, labelled["arteriole"], labelled["venule"], allow_overlap=allow_overlap
    )
    return {
        **state,
        "overlap_edge_count": overlap_edges,
        "gap_filled_edge_count": gap_filled,
        "minimum_overlap_fraction": threshold,
    }


def tag_vessels_in_global_masks(
    G: nx.Graph,
    *,
    large_mask: np.ndarray | None = None,
    small_mask: np.ndarray | None = None,
    voxel_size_zyx: tuple[float, float, float],
    large_min_overlap_fraction: float = 0.5,
    small_min_overlap_fraction: float = 0.5,
    edges: list[tuple[Any, Any, Any]] | None = None,
) -> dict[str, Any]:
    """Tag each vessel with the global mask it lies in, for its unknown class.

    A global mask holds every vessel of its size, arteriole or venule, so it
    says only how large a vessel is: ``graph.GLOBAL_MASK_CLASS`` becomes
    ``"large"`` on an edge at least *large_min_overlap_fraction* of whose
    length lies in *large_mask*, ``"small"`` on one *small_min_overlap_fraction*
    in *small_mask*, and is removed from the rest (capillaries). An edge
    reaching both goes to the mask covering more of it. As the small-vessel
    labelling does, an under-covered edge between two edges of one class
    takes that class when at least ``GAP_FILL_FRACTION_OF_THRESHOLD`` of the
    threshold of it lies in the mask. A zero-resistance bridge is not tagged.

    *edges* (``(u, v, key)``) limits the tagging to those edges -- the ones a
    hand edit made -- judged beside the tags the others already carry.
    """
    from .branch_order import GLOBAL_MASK_CLASS
    from .thick_vessel_junctions import IS_ZERO_RESISTANCE

    masks = [mask for mask in (large_mask, small_mask) if mask is not None]
    if not masks:
        raise ValueError("tag_vessels_in_global_masks needs a large or a small mask.")
    shape = masks[0].shape
    if any(mask.shape != shape for mask in masks):
        raise ValueError(
            f"The large and small global masks must share a shape; got "
            f"{large_mask.shape} and {small_mask.shape}."
        )
    thresholds = {
        "large": float(large_min_overlap_fraction),
        "small": float(small_min_overlap_fraction),
    }
    for name, threshold in thresholds.items():
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(
                f"{name}_min_overlap_fraction must be in [0.0, 1.0]. Got {threshold}."
            )

    all_edges = _label_edge_ids(G)
    wanted = None if edges is None else {_edge_id(u, v, key) for u, v, key in edges}
    vessels = [
        entry
        for entry in all_edges
        if not entry[3].get(IS_ZERO_RESISTANCE) and (wanted is None or entry[2] in wanted)
    ]
    for _u, _v, _edge, data in vessels:
        data.pop(GLOBAL_MASK_CLASS, None)
    for _u, _v, _edge, data in all_edges:
        if data.get(IS_ZERO_RESISTANCE):
            data.pop(GLOBAL_MASK_CLASS, None)
    voxels = _edge_sample_voxels(G, vessels, voxel_size_zyx=voxel_size_zyx, mask_shape=shape)
    fractions = {
        edge_id: (
            0.0 if large_mask is None else _fraction_in(large_mask, index),
            0.0 if small_mask is None else _fraction_in(small_mask, index),
        )
        for edge_id, index in voxels.items()
    }
    labelled: dict[str, set[tuple[Any, Any, int]]] = {"large": set(), "small": set()}
    for _u, _v, edge_id, data in all_edges:
        if edge_id not in fractions and data.get(GLOBAL_MASK_CLASS) in labelled:
            labelled[data[GLOBAL_MASK_CLASS]].add(edge_id)
    for _u, _v, edge_id, data in vessels:
        if edge_id not in fractions:
            continue
        large_fraction, small_fraction = fractions[edge_id]
        in_large = large_mask is not None and large_fraction >= thresholds["large"]
        in_small = small_mask is not None and small_fraction >= thresholds["small"]
        if in_large and (not in_small or large_fraction >= small_fraction):
            name = "large"
        elif in_small:
            name = "small"
        else:
            continue
        labelled[name].add(edge_id)
        data[GLOBAL_MASK_CLASS] = name
    gap_filled = _fill_label_gaps(
        all_edges,
        {
            edge_id: pair
            for edge_id, pair in fractions.items()
            if edge_id not in labelled["large"] and edge_id not in labelled["small"]
        },
        labelled,
        low_fraction={
            name: GAP_FILL_FRACTION_OF_THRESHOLD * threshold
            for name, threshold in thresholds.items()
        },
        names=("large", "small"),
        attribute=GLOBAL_MASK_CLASS,
    )
    tagged = {name: sum(1 for *_, data in vessels if data.get(GLOBAL_MASK_CLASS) == name)
              for name in labelled}
    return {
        "large_edge_count": tagged["large"],
        "small_edge_count": tagged["small"],
        "capillary_edge_count": len(vessels) - tagged["large"] - tagged["small"],
        "gap_filled_edge_count": gap_filled,
    }


def tag_edges_in_large_vessel_masks(
    G: nx.Graph,
    large_arteriole_mask: np.ndarray,
    large_venule_mask: np.ndarray,
    edges: list[tuple[Any, Any, Any]],
    *,
    voxel_size_zyx: tuple[float, float, float],
    minimum_overlap_fraction: float = 0.5,
    low_memory: bool = False,
) -> dict[str, int]:
    """Tag *edges* (``(u, v, key)``, the ones a hand edit made) with the large
    mask each lies in, ``graph.LARGE_MASK_VESSEL_TYPE``, by the rule
    :func:`infer_boundary_nodes_from_small_vessel_masks` labels every vessel
    with for the Boundaries stage: at least *minimum_overlap_fraction* of its
    length in the mask, the mask covering more of it (the arteriole on a tie)
    when both do, and an under-covered edge between two edges of one type
    gap-filled -- judged beside the tags the other edges carry, which stay as
    they are. An edge in neither mask loses any tag it had (a split copies its
    parent's, a merge its first vessel's)."""
    from .branch_order import LARGE_MASK_VESSEL_TYPE

    return _tag_chosen_edges(
        G,
        large_arteriole_mask,
        large_venule_mask,
        edges,
        attribute=LARGE_MASK_VESSEL_TYPE,
        voxel_size_zyx=voxel_size_zyx,
        minimum_overlap_fraction=minimum_overlap_fraction,
        schedule=[0.0],
        low_memory=low_memory,
    )


def tag_edges_in_small_vessel_masks(
    G: nx.Graph,
    small_arteriole_mask: np.ndarray,
    small_venule_mask: np.ndarray,
    edges: list[tuple[Any, Any, Any]],
    *,
    voxel_size_zyx: tuple[float, float, float],
    minimum_overlap_fraction: float = 0.5,
    max_dilation_microns: float = 0.0,
    dilation_step_microns: float = 5.0,
    low_memory: bool = False,
) -> dict[str, int]:
    """Tag *edges* (``(u, v, key)``, the ones a hand edit made) with the small
    mask each lies in, ``graph.SMALL_MASK_VESSEL_TYPE``, by the rule
    :func:`infer_boundary_nodes_from_small_vessel_masks_progressive_dilation`
    labels every vessel with for the Boundaries stage: at each dilation step
    in turn, at least *minimum_overlap_fraction* of the edge within that
    distance of a mask (the mask covering more of it, the arteriole on a
    tie), or an under-covered edge between two edges of one type
    gap-filled -- judged beside the tags the other edges carry, which stay
    as they are. An edge in neither mask loses any tag it had (a split
    copies its parent's, a merge its first vessel's)."""
    from .branch_order import SMALL_MASK_VESSEL_TYPE

    return _tag_chosen_edges(
        G,
        small_arteriole_mask,
        small_venule_mask,
        edges,
        attribute=SMALL_MASK_VESSEL_TYPE,
        voxel_size_zyx=voxel_size_zyx,
        minimum_overlap_fraction=minimum_overlap_fraction,
        schedule=_progressive_dilation_schedule(max_dilation_microns, dilation_step_microns),
        low_memory=low_memory,
    )


def _tag_chosen_edges(
    G: nx.Graph,
    arteriole_mask: np.ndarray,
    venule_mask: np.ndarray,
    edges: list[tuple[Any, Any, Any]],
    *,
    attribute: str,
    voxel_size_zyx: tuple[float, float, float],
    minimum_overlap_fraction: float,
    schedule: list[float],
    low_memory: bool = False,
) -> dict[str, int]:
    """Write *attribute* (``"arteriole"``/``"venule"``) on *edges* alone, by
    the masks' labelling rule over the dilation *schedule* (``[0.0]``: the
    masks as they are), beside the other edges' *attribute*, left as it is."""
    if arteriole_mask.shape != venule_mask.shape:
        raise ValueError(
            "The arteriole and venule masks must share a shape. "
            f"Got {arteriole_mask.shape} and {venule_mask.shape}."
        )
    threshold = float(minimum_overlap_fraction)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"minimum_overlap_fraction must be in [0.0, 1.0]. Got {threshold}.")
    arteriole_mask = np.asarray(arteriole_mask).astype(bool, copy=False)
    venule_mask = np.asarray(venule_mask).astype(bool, copy=False)
    all_edges = _label_edge_ids(G)
    wanted = {_edge_id(u, v, key) for u, v, key in edges}
    chosen = [entry for entry in all_edges if entry[2] in wanted]
    for *_, data in chosen:
        data.pop(attribute, None)
    voxels = _edge_sample_voxels(
        G, chosen, voxel_size_zyx=voxel_size_zyx, mask_shape=arteriole_mask.shape
    )
    distances: dict[tuple[Any, Any, int], tuple[np.ndarray, np.ndarray]] = {}
    if len(schedule) > 1 and voxels:
        edge_ids = list(voxels)
        stacked = np.concatenate([voxels[edge_id] for edge_id in edge_ids]).reshape(-1, 3)
        splits = np.cumsum([len(voxels[edge_id]) for edge_id in edge_ids])[:-1]
        to_arteriole = np.split(
            _distances_to_mask_microns(arteriole_mask, stacked, voxel_size_zyx=voxel_size_zyx, low_memory=low_memory),
            splits,
        )
        to_venule = np.split(
            _distances_to_mask_microns(venule_mask, stacked, voxel_size_zyx=voxel_size_zyx, low_memory=low_memory),
            splits,
        )
        distances = dict(zip(edge_ids, zip(to_arteriole, to_venule)))

    def fractions_at(dilation: float, edge_ids: list) -> dict:
        if not distances:
            return {
                edge_id: (
                    _fraction_in(arteriole_mask, voxels[edge_id]),
                    _fraction_in(venule_mask, voxels[edge_id]),
                )
                for edge_id in edge_ids
            }
        reach = float(dilation) + 1e-9
        out = {}
        for edge_id in edge_ids:
            art, ven = distances[edge_id]
            out[edge_id] = (
                float(np.count_nonzero(art <= reach)) / float(len(art)) if len(art) else 0.0,
                float(np.count_nonzero(ven <= reach)) / float(len(ven)) if len(ven) else 0.0,
            )
        return out

    labelled: dict[str, set[tuple[Any, Any, int]]] = {"arteriole": set(), "venule": set()}
    for _u, _v, edge_id, data in all_edges:
        if data.get(attribute) in labelled:
            labelled[data[attribute]].add(edge_id)
    data_of = {edge_id: data for _u, _v, edge_id, data in chosen}
    gap_filled = 0
    for dilation in schedule:
        unlabelled = [
            edge_id for edge_id in voxels
            if edge_id not in labelled["arteriole"] and edge_id not in labelled["venule"]
        ]
        if not unlabelled:
            break
        fractions = fractions_at(dilation, unlabelled)
        for edge_id, (arteriole_fraction, venule_fraction) in fractions.items():
            in_arteriole = arteriole_fraction >= threshold
            if in_arteriole and (venule_fraction < threshold or arteriole_fraction >= venule_fraction):
                name = "arteriole"
            elif venule_fraction >= threshold:
                name = "venule"
            else:
                continue
            labelled[name].add(edge_id)
            data_of[edge_id][attribute] = name
        gap_filled += _fill_label_gaps(
            all_edges,
            fractions,
            labelled,
            low_fraction=GAP_FILL_FRACTION_OF_THRESHOLD * threshold,
            attribute=attribute,
        )
    tagged = {
        name: sum(1 for *_, data in chosen if data.get(attribute) == name)
        for name in labelled
    }
    return {
        "arteriole_edge_count": tagged["arteriole"],
        "venule_edge_count": tagged["venule"],
        "gap_filled_edge_count": gap_filled,
    }


def boundary_nodes_from_mask_tags(
    G: nx.Graph, nodes: Iterable[Any] | None = None, *, attribute: str
) -> dict[str, list[Any]]:
    """Which of *nodes* (default: every node) are arteriole and which venule
    boundary nodes, read from the edges' *attribute* tags -- the mask each
    vessel lies in, ``"arteriole"`` or ``"venule"`` -- by the rule the
    Boundaries stage reads the masks' labelling with
    (:func:`infer_boundary_nodes_from_small_vessel_masks`): a node one of
    whose edges lies in the arteriole mask is an arteriole node (else, with
    one in the venule mask, a venule node), and a boundary node when another
    of its edges is not of its type -- where the mask's vessels meet the
    rest. On ``graph.SMALL_MASK_VESSEL_TYPE`` those are the arteriole/venule
    boundary nodes; on ``graph.LARGE_MASK_VESSEL_TYPE`` the Large_Art/Large_Ven
    hand-off nodes. ``{"arteriole": [...], "venule": [...]}``, in the order
    of *nodes*."""
    found: dict[str, list[Any]] = {"arteriole": [], "venule": []}
    for node in G.nodes if nodes is None else nodes:
        if node not in G:
            continue
        types = [data.get(attribute) for _u, _v, data in G.edges(node, data=True)]
        role = "arteriole" if "arteriole" in types else "venule" if "venule" in types else None
        if role is not None and any(kind != role for kind in types):
            found[role].append(node)
    return found


def boundary_nodes_from_small_mask_tags(
    G: nx.Graph, nodes: Iterable[Any] | None = None
) -> dict[str, list[Any]]:
    """:func:`boundary_nodes_from_mask_tags` on ``graph.SMALL_MASK_VESSEL_TYPE``:
    the arteriole and venule boundary nodes among *nodes*."""
    from .branch_order import SMALL_MASK_VESSEL_TYPE

    return boundary_nodes_from_mask_tags(G, nodes, attribute=SMALL_MASK_VESSEL_TYPE)


def filter_io_nodes_to_terminal_degree1(
    G: nx.Graph,
    input_nodes: list[Any],
    output_nodes: list[Any],
) -> tuple[list[Any], list[Any], list[Any], list[Any]]:
    """Keep only degree-1 nodes in input/output assignments.

    Returns:
        (filtered_input_nodes, filtered_output_nodes, dropped_input_nodes, dropped_output_nodes)
    """
    degree1_nodes = {node_id for node_id, degree in G.degree() if int(degree) == 1}
    dropped_input_nodes = [node_id for node_id in input_nodes if node_id not in degree1_nodes]
    dropped_output_nodes = [node_id for node_id in output_nodes if node_id not in degree1_nodes]
    filtered_input_nodes = [node_id for node_id in input_nodes if node_id in degree1_nodes]
    filtered_output_nodes = [node_id for node_id in output_nodes if node_id in degree1_nodes]
    return (
        filtered_input_nodes,
        filtered_output_nodes,
        dropped_input_nodes,
        dropped_output_nodes,
    )


def _build_dilation_schedule_microns(
    *,
    max_dilation_microns: float,
    dilation_step_microns: float,
) -> list[float]:
    """Create dilation schedule including 0 and the exact max dilation."""
    max_dilation = float(max_dilation_microns)
    step = float(dilation_step_microns)
    if max_dilation < 0:
        raise ValueError(
            "max_dilation_microns must be >= 0. "
            f"Got {max_dilation_microns}."
        )
    if step <= 0:
        raise ValueError(
            "dilation_step_microns must be > 0. "
            f"Got {dilation_step_microns}."
        )

    schedule: list[float] = [0.0]
    if max_dilation <= 0:
        return schedule

    epsilon = 1e-9
    current = step
    while current < (max_dilation - epsilon):
        schedule.append(float(current))
        current += step
    if abs(schedule[-1] - max_dilation) > epsilon:
        schedule.append(float(max_dilation))
    return schedule


def _distances_to_mask_microns(
    mask: np.ndarray,
    indices: np.ndarray,
    *,
    voxel_size_zyx: tuple[float, float, float],
    low_memory: bool = False,
) -> np.ndarray:
    """Each of *indices*' distance to the nearest voxel of *mask*, in microns
    (0 inside it; ``inf`` when the mask is empty).

    Read point by point from the mask's surface (a KD-tree over it), not from
    a whole-volume distance transform: the same values
    ``distance_transform_edt(~mask, sampling=voxel_size_zyx)`` gives there.
    *low_memory* (the low-RAM option) reads it from tiles of the surface
    instead of one tree over all of it (``FeatureDistance(by_block=True)``,
    ~69 B a surface voxel otherwise): the same distances.
    """
    from haemolynx.preprocessing.pointwise_distance import FeatureDistance

    binary = np.asanyarray(mask, dtype=bool)
    if len(indices) == 0:
        return np.zeros(0, dtype=float)
    if not binary.any():
        return np.full(len(indices), np.inf)
    if binary.all():
        return np.zeros(len(indices), dtype=float)
    distance = FeatureDistance(binary, feature_value=True, sampling=voxel_size_zyx, by_block=low_memory)
    return distance.at(np.asarray(indices, dtype=np.intp))


def _cut_from(G: nx.Graph, node_id: Any) -> str | None:
    """The mask a terminal was cut off at (``cut_graph_at_large_vessel_volumes``
    tags it), read as inside that mask: the node sits one sample outside."""
    value = G.nodes[node_id].get(CUT_FROM_LARGE_VESSEL)
    return None if value is None else str(value)


def select_terminal_nodes_from_large_vessel_masks_progressive_dilation(
    G: nx.Graph,
    large_arteriole_mask: np.ndarray,
    large_venule_mask: np.ndarray,
    *,
    voxel_size_zyx: tuple[float, float, float],
    max_dilation_microns: float,
    dilation_step_microns: float = 5.0,
    allow_overlap: bool = False,
    exclude_smaller_overlapping_volumes: bool = False,
    low_memory: bool = False,
) -> tuple[list[Any], list[Any]]:
    """Assign I/O nodes over progressive dilation steps without reassignment.

    Assignment steps always include 0 microns first, followed by fixed dilation
    increments (default: 5 microns) up to `max_dilation_microns`. A terminal
    is locked at the first step whose dilation reaches it -- inside the
    arteriole mask grown by that much, an inlet; the venule mask, an outlet --
    and cannot be reassigned at a later step.

    Each terminal's distance to each mask is read once, point by point, and
    its step is the first one at least that far: the same answer growing both
    masks step by step gives, without building two whole-volume masks per step
    to read a handful of voxels from. A terminal both masks reach at the same
    step goes to the nearer one -- it used to be settled by comparing
    whole-mask midpoints, scanning both masks for every such terminal --
    and an exact tie to the arteriole, as before. A terminal the large-vessel
    cut made (``CUT_FROM_LARGE_VESSEL``) is at distance 0 from the mask it
    was cut off at, though it sits one sample outside it.

    ``exclude_smaller_overlapping_volumes`` cleans the two masks' overlap
    once, before any distance is read.
    """
    if large_arteriole_mask.shape != large_venule_mask.shape:
        raise ValueError(
            "large_arteriole_mask and large_venule_mask must share a shape. "
            f"Got {large_arteriole_mask.shape} and {large_venule_mask.shape}."
        )
    schedule = _build_dilation_schedule_microns(
        max_dilation_microns=max_dilation_microns,
        dilation_step_microns=dilation_step_microns,
    )

    terminal_nodes = _terminal_nodes_with_position_pairs(G)
    if not terminal_nodes:
        return [], []

    arteriole_mask = large_arteriole_mask.astype(bool, copy=False)
    venule_mask = large_venule_mask.astype(bool, copy=False)
    if exclude_smaller_overlapping_volumes:
        cleaned_arteriole, cleaned_venule = exclude_smaller_overlapping_large_vessel_components(
            arteriole_mask, venule_mask
        )
        if cleaned_arteriole is not None and cleaned_venule is not None:
            arteriole_mask, venule_mask = cleaned_arteriole, cleaned_venule

    inside: list[tuple[Any, tuple[int, int, int]]] = []
    for node_id, node_pos in terminal_nodes:
        index = _position_to_mask_index(
            node_pos, voxel_size_zyx=voxel_size_zyx, mask_shape=arteriole_mask.shape
        )
        if index is not None:
            inside.append((node_id, index))
    indices = np.asarray([index for _node, index in inside], dtype=np.intp).reshape(-1, 3)
    to_arteriole = _distances_to_mask_microns(
        arteriole_mask, indices, voxel_size_zyx=voxel_size_zyx, low_memory=low_memory
    )
    to_venule = _distances_to_mask_microns(
        venule_mask, indices, voxel_size_zyx=voxel_size_zyx, low_memory=low_memory
    )
    for position, (node_id, _index) in enumerate(inside):
        cut = _cut_from(G, node_id)
        if cut == "arteriole":
            to_arteriole[position] = 0.0
        elif cut == "venule":
            to_venule[position] = 0.0

    steps = np.asarray(schedule, dtype=float)

    def first_step(distance: float) -> int | None:
        reached = np.flatnonzero(steps >= float(distance) - 1e-9)
        return int(reached[0]) if reached.size else None

    assigned_inputs: set[Any] = set()
    assigned_outputs: set[Any] = set()
    per_step = [[0, 0] for _ in schedule]
    for (node_id, _index), d_art, d_ven in zip(inside, to_arteriole, to_venule):
        step_art, step_ven = first_step(d_art), first_step(d_ven)
        if step_art is None and step_ven is None:
            continue
        if step_ven is None or (step_art is not None and step_art < step_ven):
            to_inlet = True
        elif step_art is None or step_ven < step_art:
            to_inlet = False
        else:
            # Both reach it at the same step: the nearer mask, and on an exact
            # tie (or allow_overlap) the arteriole, as the step-mask version did.
            to_inlet = allow_overlap or float(d_art) <= float(d_ven)
        step = step_art if to_inlet else step_ven
        if to_inlet:
            assigned_inputs.add(node_id)
            per_step[step][0] += 1
        else:
            assigned_outputs.add(node_id)
            per_step[step][1] += 1

    logger.info(
        "Automated large-vessel progressive assignment: %d step(s), max_dilation=%.3f microns, "
        "step=%.3f microns; new (inputs, outputs) per step: %s; %d of %d terminals unassigned.",
        len(schedule),
        float(max_dilation_microns),
        float(dilation_step_microns),
        ", ".join(f"{d:g} um: {i}/{o}" for d, (i, o) in zip(schedule, per_step)),
        len(terminal_nodes) - len(assigned_inputs) - len(assigned_outputs),
        len(terminal_nodes),
    )
    return _sort_nodes(assigned_inputs), _sort_nodes(assigned_outputs - assigned_inputs)


def _distance_from_mask_microns(
    mask: np.ndarray,
    *,
    voxel_size_zyx: tuple[float, float, float],
) -> np.ndarray:
    """Compute physical EDT distance from binary mask (in microns)."""
    binary_mask = mask.astype(bool, copy=False)
    sampling_zyx = (
        float(voxel_size_zyx[0]),
        float(voxel_size_zyx[1]),
        float(voxel_size_zyx[2]),
    )
    return distance_transform_edt(~binary_mask, sampling=sampling_zyx)


def _dilated_mask_from_cached_distance(
    base_mask: np.ndarray,
    distance_from_mask: np.ndarray,
    *,
    dilation_microns: float,
) -> np.ndarray:
    """Apply dilation threshold using a cached EDT distance volume."""
    dilation = float(dilation_microns)
    if dilation <= 0:
        return base_mask.astype(bool, copy=False)
    return base_mask.astype(bool, copy=False) | (distance_from_mask <= dilation)


def _progressive_dilation_schedule(
    max_dilation_microns: float, dilation_step_microns: float
) -> list[float]:
    """The small-vessel labelling's dilation steps: 0 um, then every step up
    to the maximum, with a 1 um step put in when the first one is further."""
    schedule = _build_dilation_schedule_microns(
        max_dilation_microns=max_dilation_microns,
        dilation_step_microns=dilation_step_microns,
    )
    if (
        len(schedule) >= 2
        and float(schedule[0]) == 0.0
        and float(schedule[1]) > 1.0
        and float(max_dilation_microns) >= 1.0
    ):
        schedule = sorted({float(s) for s in schedule} | {1.0})
    return schedule


def infer_boundary_nodes_from_small_vessel_masks_progressive_dilation(
    G: nx.Graph,
    small_arteriole_mask: np.ndarray,
    small_venule_mask: np.ndarray,
    *,
    voxel_size_zyx: tuple[float, float, float],
    max_dilation_microns: float,
    dilation_step_microns: float = 5.0,
    minimum_overlap_fraction: float = 0.5,
    allow_overlap: bool = False,
    exclude_smaller_overlapping_volumes: bool = False,
    low_memory: bool = False,
) -> dict[str, Any]:
    """Infer boundary nodes over progressive dilation steps without reassignment.

    Assignment steps always include 0 microns first. When the next scheduled
    dilation exceeds 1 micron, an extra 1 micron step is inserted so sparse masks
    can label nearest-vessel edges before large dilations smear the capillary gap.

    Each edge keeps the label of the first step that reaches it -- the first
    dilation at which at least ``minimum_overlap_fraction`` of it lies within
    that distance of a mask -- and is not judged again at a later step. The
    unlabelled edges are judged again at every step: every edge's label used
    to be locked at the first step that labelled *any* edge, unlabelled ones
    included, so once the 0 um step found one edge the dilation did nothing.
    The boundary nodes are read from the final labels (not accumulated over
    the steps: a node that was the arteriole's end at 0 um is inside it once
    the next edge is labelled).

    An edge's distance to each mask is read once, sample by sample, from the
    mask's surface; the fraction of it within a dilation is the fraction of
    those samples at most that far. That is what dilating the mask by the
    same distance and reading the samples gives, without a whole-volume
    distance transform per mask (float64, eight bytes a voxel) and two
    volume-sized masks per step. A 0 um schedule reads the masks themselves.
    """
    if small_arteriole_mask.shape != small_venule_mask.shape:
        raise ValueError(
            "small_arteriole_mask and small_venule_mask must share a shape. "
            f"Got {small_arteriole_mask.shape} and {small_venule_mask.shape}."
        )
    if not (0.0 <= float(minimum_overlap_fraction) <= 1.0):
        raise ValueError(
            "minimum_overlap_fraction must be in [0.0, 1.0]. "
            f"Got {minimum_overlap_fraction}."
        )
    schedule = _progressive_dilation_schedule(max_dilation_microns, dilation_step_microns)

    arteriole_mask = small_arteriole_mask.astype(bool, copy=False)
    venule_mask = small_venule_mask.astype(bool, copy=False)
    if exclude_smaller_overlapping_volumes:
        cleaned_arteriole, cleaned_venule = exclude_smaller_overlapping_small_vessel_components(
            arteriole_mask, venule_mask
        )
        if cleaned_arteriole is not None and cleaned_venule is not None:
            arteriole_mask, venule_mask = cleaned_arteriole, cleaned_venule

    edges = _label_edge_ids(G)
    _clear_mask_labels(G, edges)
    voxels = _edge_sample_voxels(
        G, edges, voxel_size_zyx=voxel_size_zyx, mask_shape=arteriole_mask.shape
    )
    edge_ids = list(voxels)
    counts = [len(voxels[edge_id]) for edge_id in edge_ids]
    splits = np.cumsum(counts)[:-1]
    if len(schedule) > 1 and edge_ids:
        stacked = (
            np.concatenate([voxels[edge_id] for edge_id in edge_ids])
            if sum(counts)
            else np.empty((0, 3), dtype=np.intp)
        )
        to_arteriole = np.split(
            _distances_to_mask_microns(arteriole_mask, stacked, voxel_size_zyx=voxel_size_zyx, low_memory=low_memory),
            splits,
        )
        to_venule = np.split(
            _distances_to_mask_microns(venule_mask, stacked, voxel_size_zyx=voxel_size_zyx, low_memory=low_memory),
            splits,
        )
        distances = {
            edge_id: (art, ven) for edge_id, art, ven in zip(edge_ids, to_arteriole, to_venule)
        }
    else:
        distances = {}

    def fractions_at(dilation: float, unlocked: list) -> dict:
        if not distances:
            return {
                edge_id: (
                    _fraction_in(arteriole_mask, voxels[edge_id]),
                    _fraction_in(venule_mask, voxels[edge_id]),
                )
                for edge_id in unlocked
            }
        reach = float(dilation) + 1e-9
        out = {}
        for edge_id in unlocked:
            art, ven = distances[edge_id]
            if not len(art):
                out[edge_id] = (0.0, 0.0)
                continue
            out[edge_id] = (
                float(np.count_nonzero(art <= reach)) / float(len(art)),
                float(np.count_nonzero(ven <= reach)) / float(len(ven)),
            )
        return out

    threshold = float(minimum_overlap_fraction)
    labelled: dict[str, set[tuple[Any, Any, int]]] = {"arteriole": set(), "venule": set()}
    locked: set[tuple[Any, Any, int]] = set()
    overlap_edges = 0
    gap_filled = 0
    logger.info(
        "Small-vessel progressive boundary assignment: "
        f"{len(schedule)} step(s), max_dilation={float(max_dilation_microns):.3f} microns, "
        f"step={float(dilation_step_microns):.3f} microns."
    )
    for step_idx, dilation_microns in enumerate(schedule, start=1):
        unlocked = [edge_id for edge_id in edge_ids if edge_id not in locked]
        if not unlocked:
            logger.info(
                "Small-vessel progressive boundary assignment: every edge labelled "
                "before the final dilation step."
            )
            break
        fractions = fractions_at(float(dilation_microns), unlocked)
        before = {name: set(edge_set) for name, edge_set in labelled.items()}
        overlap_edges += _apply_edge_labels(
            edges, fractions, threshold=threshold, allow_overlap=allow_overlap, labelled=labelled
        )
        gap_filled += _fill_label_gaps(
            edges, fractions, labelled, low_fraction=GAP_FILL_FRACTION_OF_THRESHOLD * threshold
        )
        new_arteriole = labelled["arteriole"] - before["arteriole"]
        new_venule = labelled["venule"] - before["venule"]
        locked |= new_arteriole | new_venule
        logger.info(
            "Small-vessel progressive boundary assignment step "
            f"{step_idx}/{len(schedule)} "
            f"(dilation={float(dilation_microns):.3f} microns): "
            f"new_arteriole_edges={len(new_arteriole)}, "
            f"new_venule_edges={len(new_venule)}, "
            f"unlabelled_edges={len(unlocked) - len(new_arteriole | new_venule)}."
        )
    if gap_filled:
        logger.info(
            "Small-vessel masks: %d under-covered edge(s) between two edges of one "
            "vessel type took that type (at least %.2f of the edge in its mask).",
            gap_filled,
            GAP_FILL_FRACTION_OF_THRESHOLD * threshold,
        )

    state = _boundary_state(
        G, edges, labelled["arteriole"], labelled["venule"], allow_overlap=allow_overlap
    )
    return {
        **state,
        "overlap_edge_count": overlap_edges,
        "gap_filled_edge_count": gap_filled,
        "minimum_overlap_fraction": threshold,
    }


def write_automated_vessel_assignment_3d_html(
    G: nx.Graph,
    *,
    large_arteriole_mask: np.ndarray,
    large_venule_mask: np.ndarray,
    input_nodes: list[Any],
    outlet_nodes: list[Any],
    voxel_size_zyx: tuple[float, float, float],
    output_html_path: str | Path,
) -> bool:
    """Write interactive 3D HTML showing masks, graph, and selected nodes."""
    try:
        import plotly.graph_objects as go
    except ModuleNotFoundError:
        return False

    pos = nx.get_node_attributes(G, "pos")
    if not pos:
        raise ValueError("Graph has no node positions ('pos').")
    if large_arteriole_mask.shape != large_venule_mask.shape:
        raise ValueError(
            "large_arteriole_mask and large_venule_mask must share a shape. "
            f"Got {large_arteriole_mask.shape} and {large_venule_mask.shape}."
        )

    output_html_path = Path(output_html_path)
    output_html_path.parent.mkdir(parents=True, exist_ok=True)

    # Edges from node positions; graph positions are stored as (z, y, x).
    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    edge_z: list[float | None] = []
    if isinstance(G, nx.MultiGraph):
        edge_iter = G.edges(keys=True, data=True)
        for u, v, _k, _data in edge_iter:
            pu = np.asarray(pos[u], dtype=float)
            pv = np.asarray(pos[v], dtype=float)
            edge_x += [float(pu[2]), float(pv[2]), None]
            edge_y += [float(pu[1]), float(pv[1]), None]
            edge_z += [float(pu[0]), float(pv[0]), None]
    else:
        edge_iter = G.edges(data=True)
        for u, v, _data in edge_iter:
            pu = np.asarray(pos[u], dtype=float)
            pv = np.asarray(pos[v], dtype=float)
            edge_x += [float(pu[2]), float(pv[2]), None]
            edge_y += [float(pu[1]), float(pv[1]), None]
            edge_z += [float(pu[0]), float(pv[0]), None]

    input_set = set(input_nodes)
    output_set = set(outlet_nodes)
    other_nodes = [n for n in G.nodes if n not in input_set and n not in output_set]

    def _coords(nodes: list[Any]) -> tuple[list[float], list[float], list[float]]:
        xs = [float(np.asarray(pos[n], dtype=float)[2]) for n in nodes if n in pos]
        ys = [float(np.asarray(pos[n], dtype=float)[1]) for n in nodes if n in pos]
        zs = [float(np.asarray(pos[n], dtype=float)[0]) for n in nodes if n in pos]
        return xs, ys, zs

    def _add_volume_trace(mask: np.ndarray, *, name: str, color: str, fig: Any) -> None:
        # Cropped and pooled to a size a page can hold: drawn whole, two
        # 124 x 1024 x 1024 masks ran a 62 GB machine out of memory.
        from haemolynx.visualization.large_vessel_assignment import (
            add_binary_mask_volume_trace,
        )

        add_binary_mask_volume_trace(
            fig, mask, name=name, color=color, opacity=0.12,
            voxel_size_zyx=tuple(float(v) for v in voxel_size_zyx),
        )

    fig = go.Figure()
    _add_volume_trace(
        large_arteriole_mask.astype(bool, copy=False),
        name="Arteriole Mask Volume",
        color="#00FF7F",
        fig=fig,
    )
    _add_volume_trace(
        large_venule_mask.astype(bool, copy=False),
        name="Venule Mask Volume",
        color="#FF3EA5",
        fig=fig,
    )
    fig.add_trace(
        go.Scatter3d(
            x=edge_x,
            y=edge_y,
            z=edge_z,
            mode="lines",
            line=dict(color="rgba(0, 200, 255, 0.7)", width=5),
            name="Edges",
        )
    )
    if other_nodes:
        ox, oy, oz = _coords(other_nodes)
        fig.add_trace(
            go.Scatter3d(
                x=ox,
                y=oy,
                z=oz,
                mode="markers",
                marker=dict(size=4, color="#9E9E9E"),
                name="Other Nodes",
            )
        )
    if input_nodes:
        ix, iy, iz = _coords(input_nodes)
        fig.add_trace(
            go.Scatter3d(
                x=ix,
                y=iy,
                z=iz,
                mode="markers",
                marker=dict(size=8, color="#00FF7F"),
                name="Input Nodes",
            )
        )
    if outlet_nodes:
        ox, oy, oz = _coords(outlet_nodes)
        fig.add_trace(
            go.Scatter3d(
                x=ox,
                y=oy,
                z=oz,
                mode="markers",
                marker=dict(size=8, color="#FF3EA5"),
                name="Output Nodes",
            )
        )
    fig.update_layout(
        title="Automated Vessel Assignment (3D)",
        showlegend=True,
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data",
        ),
    )
    from haemolynx.visualization.plot import write_plotly_html

    write_plotly_html(fig, output_html_path)
    return True


def write_small_vessel_mask_boundary_labelling_3d_html(
    G: nx.Graph,
    *,
    small_arteriole_mask: np.ndarray,
    small_venule_mask: np.ndarray,
    arteriole_boundary_nodes: list[Any],
    venule_boundary_nodes: list[Any],
    voxel_size_zyx: tuple[float, float, float],
    output_html_path: str | Path,
    title: str = "Small Vessel Mask Boundary Labelling (3D)",
) -> bool:
    """Write interactive 3D HTML: small-vessel masks, mask-labelled edges, boundary nodes."""
    try:
        import plotly.graph_objects as go
    except ModuleNotFoundError:
        return False

    pos = nx.get_node_attributes(G, "pos")
    if not pos:
        raise ValueError("Graph has no node positions ('pos').")
    if small_arteriole_mask.shape != small_venule_mask.shape:
        raise ValueError(
            "small_arteriole_mask and small_venule_mask must share a shape. "
            f"Got {small_arteriole_mask.shape} and {small_venule_mask.shape}."
        )

    output_html_path = Path(output_html_path)
    output_html_path.parent.mkdir(parents=True, exist_ok=True)

    def _empty_line_lists() -> tuple[list[float | None], list[float | None], list[float | None]]:
        return [], [], []

    segs: dict[str, tuple[list[float | None], list[float | None], list[float | None]]] = {
        "capillary": _empty_line_lists(),
        "arteriole": _empty_line_lists(),
        "venule": _empty_line_lists(),
        "overlap": _empty_line_lists(),
    }

    def _push_edge(kind: str, pu: np.ndarray, pv: np.ndarray) -> None:
        lx, ly, lz = segs[kind]
        lx += [float(pu[2]), float(pv[2]), None]
        ly += [float(pu[1]), float(pv[1]), None]
        lz += [float(pu[0]), float(pv[0]), None]

    if isinstance(G, nx.MultiGraph):
        for u, v, _k, edge_data in G.edges(keys=True, data=True):
            if u not in pos or v not in pos:
                continue
            pu = np.asarray(pos[u], dtype=float)
            pv = np.asarray(pos[v], dtype=float)
            vt = edge_data.get("mask_vessel_type")
            kind = (
                vt
                if vt in ("arteriole", "venule", "overlap")
                else "capillary"
            )
            _push_edge(kind, pu, pv)
    else:
        for u, v, edge_data in G.edges(data=True):
            if u not in pos or v not in pos:
                continue
            pu = np.asarray(pos[u], dtype=float)
            pv = np.asarray(pos[v], dtype=float)
            vt = edge_data.get("mask_vessel_type")
            kind = (
                vt
                if vt in ("arteriole", "venule", "overlap")
                else "capillary"
            )
            _push_edge(kind, pu, pv)

    def _coords(nodes: list[Any]) -> tuple[list[float], list[float], list[float]]:
        xs = [float(np.asarray(pos[n], dtype=float)[2]) for n in nodes if n in pos]
        ys = [float(np.asarray(pos[n], dtype=float)[1]) for n in nodes if n in pos]
        zs = [float(np.asarray(pos[n], dtype=float)[0]) for n in nodes if n in pos]
        return xs, ys, zs

    def _add_volume_trace(mask: np.ndarray, *, name: str, color: str, fig: Any) -> None:
        # Cropped and pooled to a size a page can hold: drawn whole, two
        # 124 x 1024 x 1024 masks ran a 62 GB machine out of memory.
        from haemolynx.visualization.large_vessel_assignment import (
            add_binary_mask_volume_trace,
        )

        add_binary_mask_volume_trace(
            fig, mask, name=name, color=color, opacity=0.12,
            voxel_size_zyx=tuple(float(v) for v in voxel_size_zyx),
        )

    art_b = set(arteriole_boundary_nodes)
    ven_b = set(venule_boundary_nodes)
    neutral_nodes: list[Any] = []
    art_interior: list[Any] = []
    ven_interior: list[Any] = []
    for n in G.nodes:
        if n in art_b or n in ven_b:
            continue
        t = G.nodes[n].get("mask_vessel_type")
        if t == "arteriole":
            art_interior.append(n)
        elif t == "venule":
            ven_interior.append(n)
        else:
            neutral_nodes.append(n)

    fig = go.Figure()
    _add_volume_trace(
        small_arteriole_mask.astype(bool, copy=False),
        name="Small arteriole mask",
        color="#00FF7F",
        fig=fig,
    )
    _add_volume_trace(
        small_venule_mask.astype(bool, copy=False),
        name="Small venule mask",
        color="#FF3EA5",
        fig=fig,
    )

    cap_x, cap_y, cap_z = segs["capillary"]
    if cap_x:
        fig.add_trace(
            go.Scatter3d(
                x=cap_x,
                y=cap_y,
                z=cap_z,
                mode="lines",
                line=dict(color="rgba(0, 200, 255, 0.45)", width=3),
                name="Edges (capillary / unlabelled)",
            )
        )
    a_x, a_y, a_z = segs["arteriole"]
    if a_x:
        fig.add_trace(
            go.Scatter3d(
                x=a_x,
                y=a_y,
                z=a_z,
                mode="lines",
                line=dict(color="rgba(0, 220, 120, 0.9)", width=5),
                name="Edges (arteriole mask)",
            )
        )
    v_x, v_y, v_z = segs["venule"]
    if v_x:
        fig.add_trace(
            go.Scatter3d(
                x=v_x,
                y=v_y,
                z=v_z,
                mode="lines",
                line=dict(color="rgba(255, 62, 165, 0.9)", width=5),
                name="Edges (venule mask)",
            )
        )
    o_x, o_y, o_z = segs["overlap"]
    if o_x:
        fig.add_trace(
            go.Scatter3d(
                x=o_x,
                y=o_y,
                z=o_z,
                mode="lines",
                line=dict(color="rgba(255, 200, 0, 0.95)", width=6),
                name="Edges (overlap)",
            )
        )

    if neutral_nodes:
        nx_, ny_, nz_ = _coords(neutral_nodes)
        fig.add_trace(
            go.Scatter3d(
                x=nx_,
                y=ny_,
                z=nz_,
                mode="markers",
                marker=dict(size=4, color="#9E9E9E"),
                name="Nodes (unlabelled)",
            )
        )
    if art_interior:
        ix, iy, iz = _coords(art_interior)
        fig.add_trace(
            go.Scatter3d(
                x=ix,
                y=iy,
                z=iz,
                mode="markers",
                marker=dict(size=6, color="#00CC66"),
                name="Nodes (arteriole interior)",
            )
        )
    if ven_interior:
        vx, vy, vz = _coords(ven_interior)
        fig.add_trace(
            go.Scatter3d(
                x=vx,
                y=vy,
                z=vz,
                mode="markers",
                marker=dict(size=6, color="#E040A0"),
                name="Nodes (venule interior)",
            )
        )
    if arteriole_boundary_nodes:
        bx, by, bz = _coords(list(arteriole_boundary_nodes))
        fig.add_trace(
            go.Scatter3d(
                x=bx,
                y=by,
                z=bz,
                mode="markers",
                marker=dict(size=11, color="#00FF7F", symbol="diamond", line=dict(width=1, color="#004422")),
                name="Arteriole boundary",
            )
        )
    if venule_boundary_nodes:
        bx, by, bz = _coords(list(venule_boundary_nodes))
        fig.add_trace(
            go.Scatter3d(
                x=bx,
                y=by,
                z=bz,
                mode="markers",
                marker=dict(size=11, color="#FF3EA5", symbol="diamond", line=dict(width=1, color="#440022")),
                name="Venule boundary",
            )
        )

    fig.update_layout(
        title=title,
        showlegend=True,
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data",
        ),
    )
    from haemolynx.visualization.plot import write_plotly_html

    write_plotly_html(fig, output_html_path)
    return True
