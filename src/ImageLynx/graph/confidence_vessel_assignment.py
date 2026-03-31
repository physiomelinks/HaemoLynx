"""Confidence-based large-vessel terminal I/O assignment."""
from __future__ import annotations

from typing import Any

import networkx as nx
import numpy as np
from scipy.ndimage import distance_transform_edt, label

from .automated_vessel_assignment import (
    compute_overlapping_terminal_assignment_metrics,
    filter_io_nodes_to_terminal_degree1,
)
from ..coords import physical_xyz_to_index_zyx


def _sort_nodes(nodes: set[Any]) -> list[Any]:
    return sorted(nodes, key=lambda n: (str(type(n)), str(n)))


def _terminal_nodes_with_positions(G: nx.Graph) -> list[tuple[Any, np.ndarray]]:
    node_pos = nx.get_node_attributes(G, "pos")
    terminals: list[tuple[Any, np.ndarray]] = []
    for node_id, degree in G.degree():
        if int(degree) != 1 or node_id not in node_pos:
            continue
        terminals.append((node_id, np.asarray(node_pos[node_id], dtype=float)))
    return terminals


def _position_to_mask_index(
    position_xyz: np.ndarray,
    voxel_size_xyz: tuple[float, float, float],
    mask_shape: tuple[int, ...],
) -> tuple[int, int, int] | None:
    voxel_index = physical_xyz_to_index_zyx(position_xyz, voxel_size_xyz)
    if np.any(voxel_index < 0):
        return None
    if np.any(voxel_index >= np.asarray(mask_shape, dtype=int)):
        return None
    return (int(voxel_index[0]), int(voxel_index[1]), int(voxel_index[2]))


def _build_dilation_schedule_microns(
    *,
    max_dilation_microns: float,
    dilation_step_microns: float,
) -> list[float]:
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


def _distance_from_mask_microns(
    mask: np.ndarray,
    *,
    voxel_size_xyz: tuple[float, float, float],
) -> np.ndarray:
    binary_mask = mask.astype(bool, copy=False)
    sampling_zyx = (
        float(voxel_size_xyz[2]),
        float(voxel_size_xyz[1]),
        float(voxel_size_xyz[0]),
    )
    return distance_transform_edt(~binary_mask, sampling=sampling_zyx)


def _dilated_mask_from_cached_distance(
    base_mask: np.ndarray,
    distance_from_mask: np.ndarray,
    *,
    dilation_microns: float,
) -> np.ndarray:
    dilation = float(dilation_microns)
    if dilation <= 0:
        return base_mask.astype(bool, copy=False)
    return base_mask.astype(bool, copy=False) | (distance_from_mask <= dilation)


def _mask_component_metrics(mask: np.ndarray) -> dict[str, float]:
    binary = mask.astype(bool, copy=False)
    if not np.any(binary):
        return {
            "component_count": 0.0,
            "largest_component_fraction": 0.0,
        }
    labeled, n_components = label(binary)
    counts = np.bincount(labeled.ravel())
    if counts.size <= 1:
        return {"component_count": 1.0, "largest_component_fraction": 1.0}
    component_sizes = counts[1:]
    total = float(np.sum(component_sizes))
    largest = float(np.max(component_sizes)) if component_sizes.size else 0.0
    largest_fraction = 0.0 if np.isclose(total, 0.0) else largest / total
    return {
        "component_count": float(n_components),
        "largest_component_fraction": float(largest_fraction),
    }


def assess_large_vessel_assignment_quality(
    G: nx.Graph,
    *,
    large_arteriole_mask: np.ndarray,
    large_venule_mask: np.ndarray,
    voxel_size_xyz: tuple[float, float, float],
    quality_max_overlap_fraction: float = 0.20,
    quality_min_terminal_coverage: float = 0.20,
    quality_max_component_count: int = 12,
) -> dict[str, Any]:
    """Compute mask/registration quality diagnostics used by robust assignment."""
    if large_arteriole_mask.shape != large_venule_mask.shape:
        raise ValueError(
            "large_arteriole_mask and large_venule_mask must share a shape. "
            f"Got {large_arteriole_mask.shape} and {large_venule_mask.shape}."
        )
    art_mask = large_arteriole_mask.astype(bool, copy=False)
    ven_mask = large_venule_mask.astype(bool, copy=False)
    overlap = np.logical_and(art_mask, ven_mask)
    union = np.logical_or(art_mask, ven_mask)
    overlap_fraction = 0.0
    if np.any(union):
        overlap_fraction = float(np.count_nonzero(overlap)) / float(np.count_nonzero(union))

    art_components = _mask_component_metrics(art_mask)
    ven_components = _mask_component_metrics(ven_mask)

    terminals = _terminal_nodes_with_positions(G)
    coverage_hits = 0
    for _node_id, node_pos in terminals:
        idx = _position_to_mask_index(
            node_pos,
            voxel_size_xyz=voxel_size_xyz,
            mask_shape=art_mask.shape,
        )
        if idx is None:
            continue
        if bool(art_mask[idx]) or bool(ven_mask[idx]):
            coverage_hits += 1
    terminal_coverage = 0.0
    if terminals:
        terminal_coverage = float(coverage_hits) / float(len(terminals))

    poor_overlap = overlap_fraction > float(quality_max_overlap_fraction)
    poor_components = (
        art_components["component_count"] > float(quality_max_component_count)
        or ven_components["component_count"] > float(quality_max_component_count)
    )
    poor_coverage = terminal_coverage < float(quality_min_terminal_coverage)
    poor_quality = bool(poor_overlap or poor_components or poor_coverage)

    return {
        "overlap_fraction": float(overlap_fraction),
        "terminal_coverage": float(terminal_coverage),
        "arteriole_component_count": int(art_components["component_count"]),
        "venule_component_count": int(ven_components["component_count"]),
        "arteriole_largest_component_fraction": float(
            art_components["largest_component_fraction"]
        ),
        "venule_largest_component_fraction": float(
            ven_components["largest_component_fraction"]
        ),
        "poor_quality": poor_quality,
        "poor_quality_reasons": {
            "high_overlap_fraction": bool(poor_overlap),
            "high_component_fragmentation": bool(poor_components),
            "low_terminal_coverage": bool(poor_coverage),
        },
    }


def _edge_label(edge_data: dict[str, Any]) -> str:
    branch_order = edge_data.get("branch_order")
    if branch_order is not None:
        bo = str(branch_order).strip().lower()
        if bo.startswith("art"):
            return "input"
        if bo.startswith("ven"):
            return "output"
    vessel_type = edge_data.get("vessel_type")
    if vessel_type is None:
        vessel_type = edge_data.get("mask_vessel_type")
    vt = str(vessel_type).strip().lower() if vessel_type is not None else ""
    if vt in {"arteriole", "arterial", "art"}:
        return "input"
    if vt in {"venule", "venous", "ven"}:
        return "output"
    return "unknown"


def _terminal_topology_support(
    G: nx.Graph,
    node_id: Any,
) -> dict[str, float]:
    """Gather local support from terminal edge and neighboring edge labels."""
    input_support = 0.0
    output_support = 0.0

    if int(G.degree(node_id)) == 0:
        return {"input": 0.0, "output": 0.0}

    if isinstance(G, nx.MultiGraph):
        incident = list(G.edges(node_id, keys=True, data=True))
        if not incident:
            return {"input": 0.0, "output": 0.0}
        u, v, k, edge_data = incident[0]
        neighbor = v if u == node_id else u
        labels = [_edge_label(edge_data)]
        for nu, nv, nk, nbr_data in G.edges(neighbor, keys=True, data=True):
            if (nu == node_id and nv == neighbor and nk == k) or (
                nv == node_id and nu == neighbor and nk == k
            ):
                continue
            labels.append(_edge_label(nbr_data))
    else:
        incident = list(G.edges(node_id, data=True))
        if not incident:
            return {"input": 0.0, "output": 0.0}
        u, v, edge_data = incident[0]
        neighbor = v if u == node_id else u
        labels = [_edge_label(edge_data)]
        for nu, nv, nbr_data in G.edges(neighbor, data=True):
            if (nu == node_id and nv == neighbor) or (nv == node_id and nu == neighbor):
                continue
            labels.append(_edge_label(nbr_data))

    for idx, label_name in enumerate(labels):
        # Direct terminal edge is strongest evidence; neighboring edges are softer evidence.
        weight = 1.0 if idx == 0 else 0.5
        if label_name == "input":
            input_support += weight
        elif label_name == "output":
            output_support += weight

    return {"input": float(input_support), "output": float(output_support)}


def _distance_penalty(distance_value: float, scale: float = 5.0) -> float:
    if not np.isfinite(distance_value):
        return 1.0
    d = max(0.0, float(distance_value))
    s = max(1e-9, float(scale))
    return float(d / (d + s))


def select_terminal_nodes_from_large_vessel_masks_progressive_dilation_confidence(
    G: nx.Graph,
    large_arteriole_mask: np.ndarray,
    large_venule_mask: np.ndarray,
    *,
    voxel_size_xyz: tuple[float, float, float],
    max_dilation_microns: float,
    dilation_step_microns: float = 5.0,
    confidence_margin: float = 0.08,
    minimum_confidence: float = 0.12,
    topology_penalty: float = 0.12,
    overlap_weight: float = 1.0,
    cross_section_distance_weight: float = 0.35,
    midpoint_distance_weight: float = 0.20,
    quality_max_overlap_fraction: float = 0.20,
    quality_min_terminal_coverage: float = 0.20,
    quality_max_component_count: int = 12,
    conservative_max_dilation_microns: float = 15.0,
) -> dict[str, Any]:
    """Confidence-based progressive terminal I/O assignment with unresolved nodes."""
    if large_arteriole_mask.shape != large_venule_mask.shape:
        raise ValueError(
            "large_arteriole_mask and large_venule_mask must share a shape. "
            f"Got {large_arteriole_mask.shape} and {large_venule_mask.shape}."
        )

    quality_metrics = assess_large_vessel_assignment_quality(
        G,
        large_arteriole_mask=large_arteriole_mask,
        large_venule_mask=large_venule_mask,
        voxel_size_xyz=voxel_size_xyz,
        quality_max_overlap_fraction=quality_max_overlap_fraction,
        quality_min_terminal_coverage=quality_min_terminal_coverage,
        quality_max_component_count=quality_max_component_count,
    )
    poor_quality = bool(quality_metrics["poor_quality"])
    effective_max_dilation = float(max_dilation_microns)
    effective_confidence_margin = float(confidence_margin)
    effective_minimum_confidence = float(minimum_confidence)
    if poor_quality:
        effective_max_dilation = min(
            float(effective_max_dilation),
            float(conservative_max_dilation_microns),
        )
        effective_confidence_margin = max(
            float(effective_confidence_margin),
            float(confidence_margin) * 1.5,
        )
        effective_minimum_confidence = max(
            float(effective_minimum_confidence),
            float(minimum_confidence) * 1.35,
        )

    schedule = _build_dilation_schedule_microns(
        max_dilation_microns=effective_max_dilation,
        dilation_step_microns=dilation_step_microns,
    )
    terminals = _terminal_nodes_with_positions(G)
    if not terminals:
        return {
            "input_nodes": [],
            "output_nodes": [],
            "unresolved_nodes": [],
            "node_confidence": {},
            "quality_metrics": quality_metrics,
            "conservative_mode": poor_quality,
            "effective_max_dilation_microns": float(effective_max_dilation),
            "effective_confidence_margin": float(effective_confidence_margin),
            "effective_minimum_confidence": float(effective_minimum_confidence),
        }

    base_art = large_arteriole_mask.astype(bool, copy=False)
    base_ven = large_venule_mask.astype(bool, copy=False)
    art_distance = _distance_from_mask_microns(base_art, voxel_size_xyz=voxel_size_xyz)
    ven_distance = _distance_from_mask_microns(base_ven, voxel_size_xyz=voxel_size_xyz)

    assigned_inputs: set[Any] = set()
    assigned_outputs: set[Any] = set()
    remaining_ids = {node_id for node_id, _ in terminals}
    node_pos_map = {node_id: node_pos for node_id, node_pos in terminals}
    first_hit_dilation: dict[Any, float] = {}
    node_confidence: dict[Any, dict[str, Any]] = {}

    for dilation_microns in schedule:
        if not remaining_ids:
            break
        step_art = _dilated_mask_from_cached_distance(
            base_art,
            art_distance,
            dilation_microns=float(dilation_microns),
        )
        step_ven = _dilated_mask_from_cached_distance(
            base_ven,
            ven_distance,
            dilation_microns=float(dilation_microns),
        )
        assigned_this_step: list[Any] = []
        for node_id in list(remaining_ids):
            node_pos = node_pos_map[node_id]
            idx = _position_to_mask_index(
                node_pos,
                voxel_size_xyz=voxel_size_xyz,
                mask_shape=step_art.shape,
            )
            if idx is None:
                continue
            in_art = bool(step_art[idx])
            in_ven = bool(step_ven[idx])
            if not in_art and not in_ven:
                continue
            if node_id not in first_hit_dilation:
                first_hit_dilation[node_id] = float(dilation_microns)

            metrics = compute_overlapping_terminal_assignment_metrics(
                G,
                node_id,
                node_pos=node_pos,
                large_arteriole_mask=step_art,
                large_venule_mask=step_ven,
                voxel_size_xyz=voxel_size_xyz,
            )
            top_support = _terminal_topology_support(G, node_id)
            topology_delta = float(topology_penalty) * (
                float(top_support["input"]) - float(top_support["output"])
            )
            score_input = (
                float(overlap_weight) * float(metrics["arteriole_overlap_fraction"])
                - float(cross_section_distance_weight)
                * _distance_penalty(float(metrics["arteriole_cross_section_midpoint_distance"]))
                - float(midpoint_distance_weight)
                * _distance_penalty(float(metrics["arteriole_midpoint_distance"]))
                + topology_delta
            )
            score_output = (
                float(overlap_weight) * float(metrics["venule_overlap_fraction"])
                - float(cross_section_distance_weight)
                * _distance_penalty(float(metrics["venule_cross_section_midpoint_distance"]))
                - float(midpoint_distance_weight)
                * _distance_penalty(float(metrics["venule_midpoint_distance"]))
                - topology_delta
            )
            score_delta = float(score_input - score_output)
            score_gap = float(abs(score_delta))
            max_dilation_safe = max(1e-9, float(effective_max_dilation))
            first_hit = float(first_hit_dilation[node_id])
            first_hit_factor = float(max(0.30, 1.0 - 0.5 * (first_hit / max_dilation_safe)))
            confidence = float(score_gap * first_hit_factor)

            decision = "unresolved"
            if score_delta > 0:
                decision = "input"
            elif score_delta < 0:
                decision = "output"

            reason = "insufficient_confidence"
            if decision == "unresolved":
                reason = "exact_tie"
            elif score_gap < float(effective_confidence_margin):
                reason = "low_score_gap"
                decision = "unresolved"
            elif confidence < float(effective_minimum_confidence):
                reason = "low_confidence"
                decision = "unresolved"
            else:
                reason = "assigned"

            node_confidence[node_id] = {
                "decision": decision,
                "reason": reason,
                "confidence": float(confidence),
                "score_gap": float(score_gap),
                "score_input": float(score_input),
                "score_output": float(score_output),
                "first_hit_dilation_microns": float(first_hit),
                "evaluated_dilation_microns": float(dilation_microns),
                "metrics": {
                    "arteriole_overlap_fraction": float(metrics["arteriole_overlap_fraction"]),
                    "venule_overlap_fraction": float(metrics["venule_overlap_fraction"]),
                    "arteriole_cross_section_midpoint_distance": float(
                        metrics["arteriole_cross_section_midpoint_distance"]
                    ),
                    "venule_cross_section_midpoint_distance": float(
                        metrics["venule_cross_section_midpoint_distance"]
                    ),
                    "arteriole_midpoint_distance": float(metrics["arteriole_midpoint_distance"]),
                    "venule_midpoint_distance": float(metrics["venule_midpoint_distance"]),
                },
                "topology_support": {
                    "input": float(top_support["input"]),
                    "output": float(top_support["output"]),
                },
            }

            if decision == "input":
                assigned_inputs.add(node_id)
                assigned_this_step.append(node_id)
            elif decision == "output":
                assigned_outputs.add(node_id)
                assigned_this_step.append(node_id)
        for node_id in assigned_this_step:
            remaining_ids.discard(node_id)

    # Keep sets disjoint.
    assigned_outputs -= assigned_inputs

    # Post-assignment topology consistency filter.
    posthoc_unresolved: set[Any] = set()
    for node_id in list(assigned_inputs):
        support = _terminal_topology_support(G, node_id)
        if float(support["output"]) > (float(support["input"]) + 0.5):
            assigned_inputs.discard(node_id)
            posthoc_unresolved.add(node_id)
            node_confidence.setdefault(node_id, {})
            node_confidence[node_id]["decision"] = "unresolved"
            node_confidence[node_id]["reason"] = "topology_conflict_input"
    for node_id in list(assigned_outputs):
        support = _terminal_topology_support(G, node_id)
        if float(support["input"]) > (float(support["output"]) + 0.5):
            assigned_outputs.discard(node_id)
            posthoc_unresolved.add(node_id)
            node_confidence.setdefault(node_id, {})
            node_confidence[node_id]["decision"] = "unresolved"
            node_confidence[node_id]["reason"] = "topology_conflict_output"

    unresolved_nodes = _sort_nodes(set(remaining_ids) | posthoc_unresolved)
    sorted_inputs = _sort_nodes(assigned_inputs)
    sorted_outputs = _sort_nodes(assigned_outputs)
    filtered_inputs, filtered_outputs, dropped_inputs, dropped_outputs = filter_io_nodes_to_terminal_degree1(
        G,
        sorted_inputs,
        sorted_outputs,
    )
    for node_id in dropped_inputs + dropped_outputs:
        node_confidence.setdefault(node_id, {})
        node_confidence[node_id]["decision"] = "unresolved"
        node_confidence[node_id]["reason"] = "non_terminal_filtered"
    unresolved_nodes = _sort_nodes(set(unresolved_nodes) | set(dropped_inputs) | set(dropped_outputs))

    return {
        "input_nodes": filtered_inputs,
        "output_nodes": filtered_outputs,
        "unresolved_nodes": unresolved_nodes,
        "node_confidence": node_confidence,
        "quality_metrics": quality_metrics,
        "conservative_mode": bool(poor_quality),
        "effective_max_dilation_microns": float(effective_max_dilation),
        "effective_confidence_margin": float(effective_confidence_margin),
        "effective_minimum_confidence": float(effective_minimum_confidence),
    }
