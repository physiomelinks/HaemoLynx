"""Boundary-based node selection helpers."""
from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import networkx as nx


def select_boundary_terminal_nodes(
    G: nx.Graph,
    image_shape: tuple[int, ...],
    *,
    edge_percent: float,
    end_percent: float,
    axis: int = 1,
    boundary_permeability_mode: str = "caged",
    voxel_size: tuple[float, ...] = None
) -> tuple[list[Any], list[Any]]:
    """Select degree-1 nodes with support for Tri-Mode 3D permeability.

    ``voxel_size`` is the (z, y, x) spacing in physical units. Node ``pos`` attributes are
    stored in physical units while ``image_shape`` is in voxels, so the axis extent must be
    scaled before the two are compared. Defaults to unit spacing, under which the comparison
    is unchanged.
    """
    if not (0.0 <= edge_percent <= 100.0 and 0.0 <= end_percent <= 100.0):
        raise ValueError("edge_percent and end_percent must be in [0, 100].")
    if axis < 0 or axis >= len(image_shape):
        raise ValueError(f"axis={axis} out of bounds for image shape {image_shape}.")

    node_pos = nx.get_node_attributes(G, "pos")
    if not node_pos:
        return [], []

    def axis_coord(node_id: Any) -> float:
        return float(np.asarray(node_pos[node_id], dtype=float)[axis])

    axis_spacing = 1.0 if voxel_size is None else float(voxel_size[axis])
    axis_size = float(image_shape[axis] - 1) * axis_spacing
    top_limit = axis_size * (edge_percent / 100.0)
    bottom_start = axis_size * (1.0 - (end_percent / 100.0))

    # TIER 1: Standard Dead-Ends (Degree 1)
    terminal_nodes = [node for node, degree in G.degree() if degree == 1 and node in node_pos]
    
    starting = [node for node in terminal_nodes if axis_coord(node) <= top_limit]
    outputs = [node for node in terminal_nodes if axis_coord(node) >= bottom_start]

    # TIER 2: Spatial Extremes Fallback
    # If the network forms a closed loop cage (0 dead ends) or stitching removed them.
    if not starting or not outputs:
        import logging
        logger = logging.getLogger(__name__)
        logger.info("Tier 1 Boundary Selection failed. Falling back to Spatial Extremes.")
        
        all_nodes_sorted = sorted(list(node_pos.keys()), key=lambda n: axis_coord(n))
        n_select = max(1, len(all_nodes_sorted) // 10)
        starting = all_nodes_sorted[:n_select]
        outputs = all_nodes_sorted[-n_select:]

    starting_set = set(starting)
    outputs_set = set(outputs)
    
    # --- Tri-Mode Boundary Routing ---
    if boundary_permeability_mode == "universal_sink":
        # All remaining dead-ends (on X, Y, or inner Z) are routed directly to the Outlet (Venous Ground)
        for node in terminal_nodes:
            if node not in starting_set and node not in outputs_set:
                outputs.append(node)
                outputs_set.add(node)
                
    elif boundary_permeability_mode == "robin_resistance":
        # Tag remaining dead-ends as Robin Boundaries for dynamic Matrix Ghost Node generation
        for node in terminal_nodes:
            if node not in starting_set and node not in outputs_set:
                G.nodes[node]["is_robin_boundary"] = True
                
    # ---------------------------------

    outputs = [node for node in outputs if node not in starting_set]

    starting.sort(key=lambda n: (axis_coord(n), n))
    outputs.sort(key=lambda n: (-axis_coord(n), n))
    return starting, outputs


def select_boundary_terminal_nodes_by_face(
    G: nx.Graph,
    image_shape: tuple[int, ...],
    *,
    axis: int = 0,
    face_tolerance_voxels: float = 1.0,
    voxel_size: tuple[float, ...] = None,
    boundary_permeability_mode: str = "caged",
) -> tuple[list[Any], list[Any]]:
    """Select pressure boundaries from terminals that cross a region face.

    ``select_boundary_terminal_nodes`` assigns arterial pressure to whichever degree-1 nodes
    fall inside a positional band. On these graphs about **86% of degree-1 nodes are interior**,
    nowhere near a region face: they are skeletonisation spurs and segmentation breaks, not
    vessels entering the volume. The band rule therefore puts arterial pressure on mask defects,
    and the fraction it catches depends on a band width with no anatomical meaning.

    A vessel supplying this region has to cross one of its faces. A dead end in the middle of
    the volume cannot be a pressure inlet whatever its coordinate. This selects on that basis:
    terminals within ``face_tolerance_voxels`` of the low face of ``axis`` are inlets, those on
    the high face are outlets, and everything else is not a pressure boundary.

    **Measured against the band rule** on the six CB3 graphs, varying each rule's own free
    parameter over its plausible range and taking the spread of the shunt ratio per specimen:

    ========================================  ==============
    Rule and parameter range                  Ratio spread
    ========================================  ==============
    band, axis 1, band width 10/25/40%              75.8%
    face, axis 1, tolerance 1/2/4 voxels            13.3%
    ========================================  ==============

    A 5.7-fold reduction, and it comes from the parameter rather than the axis. The band width
    has no principled value, so its whole plausible range is live. The face tolerance is
    anchored to the voxel size: one voxel means "on the face", and the other values are only
    there to show the answer does not depend on it.

    Comparing at a *fixed* second parameter is misleading and initially pointed the other way.
    Axis spread alone is 28.4% for the band rule against 31.5% here, which flatters the band
    rule by holding the parameter that damages it at its default. Both parameters have to move.

    ``axis`` remains a choice without anatomical justification in a mid-organ region. For this
    cohort axis 1 is the only one solvable in all six specimens; axis 0 has no outlet terminal
    in SHR-A and axis 2 has no inlet terminal in SHR-C. That is a selection criterion rather
    than a preference, but it is a property of these graphs and not a general rule.

    Raises rather than falling back when a face carries no terminals. The band method drops to
    the extreme 10% of *all* nodes in that case, which converts an unsolvable region into a
    solved one with invented boundaries.
    """
    if axis < 0 or axis >= len(image_shape):
        raise ValueError(f"axis={axis} out of bounds for image shape {image_shape}.")
    if face_tolerance_voxels < 0:
        raise ValueError("face_tolerance_voxels must be non-negative.")

    node_pos = nx.get_node_attributes(G, "pos")
    if not node_pos:
        return [], []

    spacing = 1.0 if voxel_size is None else float(voxel_size[axis])
    axis_size = float(image_shape[axis] - 1) * spacing
    tol = float(face_tolerance_voxels) * spacing

    terminals = [node for node, degree in G.degree() if degree == 1 and node in node_pos]

    def axis_coord(node_id: Any) -> float:
        return float(np.asarray(node_pos[node_id], dtype=float)[axis])

    inlets = [n for n in terminals if axis_coord(n) <= tol]
    inlet_set = set(inlets)
    # A terminal on both faces would be a region only one voxel thick; assigning it to both
    # would short the solve, so the low face wins and the ambiguity is not silently doubled.
    outlets = [n for n in terminals
               if axis_coord(n) >= axis_size - tol and n not in inlet_set]

    if not inlets or not outlets:
        raise ValueError(
            f"axis {axis}: no terminal nodes on the "
            f"{'low' if not inlets else 'high'} face within "
            f"{face_tolerance_voxels} voxel(s). This region cannot be solved with a pressure "
            f"boundary on this axis. Choose another axis, widen the tolerance deliberately, or "
            f"treat the region as unsuitable. Falling back to a positional band would invent "
            f"boundaries that no vessel crosses."
        )

    if boundary_permeability_mode == "universal_sink":
        outlet_set = set(outlets)
        for node in terminals:
            if node not in inlet_set and node not in outlet_set:
                outlets.append(node)
                outlet_set.add(node)
    elif boundary_permeability_mode == "robin_resistance":
        outlet_set = set(outlets)
        for node in terminals:
            if node not in inlet_set and node not in outlet_set:
                G.nodes[node]["is_robin_boundary"] = True

    inlets.sort(key=lambda n: (axis_coord(n), str(n)))
    outlets.sort(key=lambda n: (-axis_coord(n), str(n)))
    return inlets, outlets


def _terminal_nodes_with_positions(G: nx.Graph) -> tuple[list[Any], dict[Any, np.ndarray]]:
    node_pos = nx.get_node_attributes(G, "pos")
    terminals = [node for node, degree in G.degree() if degree == 1 and node in node_pos]
    pos = {node: np.asarray(node_pos[node], dtype=float) for node in terminals}
    return terminals, pos


def _normalize_point(point: Iterable[float], *, name: str) -> np.ndarray:
    arr = np.asarray(tuple(point), dtype=float)
    if arr.shape != (3,):
        raise ValueError(f"{name} must be a 3D coordinate, got shape {arr.shape}.")
    return arr


def _sort_nodes(nodes: Iterable[Any]) -> list[Any]:
    return sorted(set(nodes), key=lambda n: (str(type(n)), str(n)))


def select_boundary_nodes_by_method(
    G: nx.Graph,
    image_shape: tuple[int, ...],
    *,
    method: str,
    node_role: str,
    coordinates: Iterable[Iterable[float]] | None = None,
    volume_boxes: Iterable[tuple[Iterable[float], Iterable[float]]] | None = None,
    edge_percent: float = 10.0,
    end_percent: float = 10.0,
    axis: int = 1,
    exclude_nodes: Iterable[Any] | None = None,
    starting_nodes_for_distance: Iterable[Any] | None = None,
    distance_from_starting_node: float = 0.0,
) -> list[Any]:
    """Select boundary nodes for one role using the specified method."""
    if node_role not in {"input", "output"}:
        raise ValueError("node_role must be 'input' or 'output'.")

    terminals, pos = _terminal_nodes_with_positions(G)
    if not terminals:
        return []

    method_norm = str(method).strip().lower()
    excluded = set(exclude_nodes or [])
    selected: list[Any]

    if method_norm == "all_degree_1":
        selected = terminals
    elif method_norm == "coordinates":
        points = list(coordinates or [])
        if not points:
            return []
        selected = []
        for idx, point in enumerate(points):
            target = _normalize_point(point, name=f"coordinates[{idx}]")
            nearest = min(
                terminals,
                key=lambda node_id: float(np.linalg.norm(pos[node_id] - target)),
            )
            selected.append(nearest)
    elif method_norm == "volume":
        boxes = list(volume_boxes or [])
        if not boxes:
            return []
        normalized_boxes: list[tuple[np.ndarray, np.ndarray]] = []
        for idx, box in enumerate(boxes):
            corners = tuple(box)
            if len(corners) != 2:
                raise ValueError(
                    "Each volume box must contain exactly two corner points."
                )
            corner_a = _normalize_point(corners[0], name=f"volume_boxes[{idx}][0]")
            corner_b = _normalize_point(corners[1], name=f"volume_boxes[{idx}][1]")
            lo = np.minimum(corner_a, corner_b)
            hi = np.maximum(corner_a, corner_b)
            normalized_boxes.append((lo, hi))
        selected = []
        for node_id in terminals:
            p = pos[node_id]
            if any(np.all(p >= lo) and np.all(p <= hi) for lo, hi in normalized_boxes):
                selected.append(node_id)
    elif method_norm == "edge_percent":
        start_nodes, out_nodes = select_boundary_terminal_nodes(
            G,
            image_shape,
            edge_percent=edge_percent,
            end_percent=end_percent,
            axis=axis,
        )
        selected = start_nodes if node_role == "input" else out_nodes
    elif method_norm == "degree_1_from_starting":
        if distance_from_starting_node < 0:
            raise ValueError("distance_from_starting_node must be non-negative.")
        node_pos_all = nx.get_node_attributes(G, "pos")
        starting_positions = [
            np.asarray(node_pos_all[node_id], dtype=float)
            for node_id in (starting_nodes_for_distance or [])
            if node_id in node_pos_all
        ]
        if not starting_positions:
            return []
        selected = []
        for node_id in terminals:
            nearest_start_dist = min(
                float(np.linalg.norm(pos[node_id] - start_pos))
                for start_pos in starting_positions
            )
            if nearest_start_dist > distance_from_starting_node:
                selected.append(node_id)
    else:
        raise ValueError(
            "Unknown boundary-node method. Supported methods are: "
            "'coordinates', 'all_degree_1', 'volume', 'edge_percent', "
            "'degree_1_from_starting'."
        )

    return [node for node in _sort_nodes(selected) if node not in excluded]

