"""Graph building and topology optimization for vascular networks."""
from .assemble import STEP_LABELS, build_graph_from_skeleton
from .smoothing import (
    SMOOTHING_METHODS,
    chaikin_smooth_polyline,
    smooth_graph_centrelines,
    smooth_polyline,
    taubin_smooth_polyline,
)
from .build import build_graph_segment_skan_stitched_loops
from .reconnect import reconnect_secondary_loop_edges
from .optimise import optimise_graph_topology_fixed, reconnect_orphan_and_dangling_nodes
from .validate import (
    EDGE_ATTRIBUTE_UNITS,
    assert_no_forbidden_edge_attributes,
    validate_skeleton_connection,
)
from .degree2 import (
    safer_simple_remove_all_degree2_nodes,
    trivial_remove_all_degree2_nodes,
    create_trivial_merged_edge,
    smart_multigraph_degree2_removal,
    merge_edges_with_topology_improvement,
)
from .prune import (
    prune_vascular_stubs,
    remove_components_without_connected_io,
    remove_edges_for_self_connected_nodes,
)
from .diagnostics import diagnose_degree2_nodes, format_degree2_diagnostics_report
from .collapse import collapse_node_clusters
from .branch_order import (
    MissingSmallVesselAssignmentWarning,
    assign_branch_orders,
    assign_hierarchical_branch_orders,
    assign_vessel_branch_orders,
)
from .boundaries import (
    BoundaryCoordinateWarning,
    select_boundary_nodes_by_method,
    select_boundary_nodes_for_role,
    select_boundary_terminal_nodes,
)
from .boundary_node_fallback import (
    seed_edges_have_full_mask_coverage,
    select_nodes_at_hop_distance,
)
from .automated_vessel_assignment import (
    compute_overlapping_terminal_assignment_metrics,
    filter_io_nodes_to_terminal_degree1,
    infer_boundary_nodes_from_small_vessel_masks,
    infer_boundary_nodes_from_small_vessel_masks_progressive_dilation,
    resolve_overlapping_terminal_node_assignment,
    select_terminal_nodes_from_large_vessel_masks,
    select_terminal_nodes_from_large_vessel_masks_progressive_dilation,
    write_automated_vessel_assignment_3d_html,
    write_small_vessel_mask_boundary_labelling_3d_html,
)
from .confidence_vessel_assignment import (
    assess_large_vessel_assignment_quality,
    select_terminal_nodes_from_large_vessel_masks_progressive_dilation_confidence,
)
from .mask_continuity import (
    enforce_small_vessel_mask_continuity,
    redefine_small_masks_from_large_tangential_contact,
)
from .large_vessels import (
    dilate_binary_mask_by_microns,
    dilate_large_vessel_masks_by_microns,
    exclude_smaller_overlapping_large_vessel_components,
    exclude_smaller_overlapping_small_vessel_components,
    remove_small_opposite_attached_large_vessel_components,
)
from .cut_at_large_vessel_volumes import cut_graph_at_large_vessel_volumes
from .large_vessel_network import (
    find_large_vessel_mask_stump_points,
    select_large_vessel_mask_stump_terminal_nodes_for_role,
    select_large_vessel_stump_terminal_nodes,
)
from .thick_vessel_junctions import IS_ZERO_RESISTANCE, insert_thick_vessel_junction_nodes
from .mask_component_volume import (
    remove_small_mask_components_by_volume,
    remove_small_vessel_components_by_volume,
)
from ._helpers import (
    add_edge_safe,
    has_edge_safe,
    remove_edge_safe,
    get_all_edge_data,
    create_merged_edge_attributes,
    validate_voxel_path_continuity,
    merge_edge_voxels_at_node,
    merge_voxel_paths_at_node,
    orient_voxel_path_to_node,
    get_line_points_3d,
    calculate_path_length,
    calculate_edge_length,
    is_path_curved,
    merge_curved_edges,
    orient_path_to_endpoint,
    orient_path_from_startpoint,
    improve_straight_edge_with_skeleton,
    trace_skeleton_path,
    parse_skeleton_data,
    find_nearest_skeleton_voxel,
    astar_skeleton_path,
    are_paths_similar,
    should_add_merged_edge,
)

__all__ = [
    "STEP_LABELS",
    "build_graph_from_skeleton",
    "SMOOTHING_METHODS",
    "smooth_graph_centrelines",
    "smooth_polyline",
    "taubin_smooth_polyline",
    "chaikin_smooth_polyline",
    "build_graph_segment_skan_stitched_loops",
    "reconnect_secondary_loop_edges",
    "optimise_graph_topology_fixed",
    "reconnect_orphan_and_dangling_nodes",
    "validate_skeleton_connection",
    "assert_no_forbidden_edge_attributes",
    "EDGE_ATTRIBUTE_UNITS",
    "safer_simple_remove_all_degree2_nodes",
    "trivial_remove_all_degree2_nodes",
    "create_trivial_merged_edge",
    "smart_multigraph_degree2_removal",
    "merge_edges_with_topology_improvement",
    "prune_vascular_stubs",
    "remove_components_without_connected_io",
    "diagnose_degree2_nodes",
    "format_degree2_diagnostics_report",
    "collapse_node_clusters",
    "assign_branch_orders",
    "assign_hierarchical_branch_orders",
    "assign_vessel_branch_orders",
    "MissingSmallVesselAssignmentWarning",
    "BoundaryCoordinateWarning",
    "select_boundary_terminal_nodes",
    "select_boundary_nodes_by_method",
    "select_boundary_nodes_for_role",
    "compute_overlapping_terminal_assignment_metrics",
    "filter_io_nodes_to_terminal_degree1",
    "infer_boundary_nodes_from_small_vessel_masks",
    "infer_boundary_nodes_from_small_vessel_masks_progressive_dilation",
    "resolve_overlapping_terminal_node_assignment",
    "select_terminal_nodes_from_large_vessel_masks",
    "select_terminal_nodes_from_large_vessel_masks_progressive_dilation",
    "assess_large_vessel_assignment_quality",
    "select_terminal_nodes_from_large_vessel_masks_progressive_dilation_confidence",
    "write_automated_vessel_assignment_3d_html",
    "write_small_vessel_mask_boundary_labelling_3d_html",
    "enforce_small_vessel_mask_continuity",
    "redefine_small_masks_from_large_tangential_contact",
    "dilate_binary_mask_by_microns",
    "dilate_large_vessel_masks_by_microns",
    "exclude_smaller_overlapping_large_vessel_components",
    "exclude_smaller_overlapping_small_vessel_components",
    "remove_small_opposite_attached_large_vessel_components",
    "cut_graph_at_large_vessel_volumes",
    "find_large_vessel_mask_stump_points",
    "select_large_vessel_mask_stump_terminal_nodes_for_role",
    "select_large_vessel_stump_terminal_nodes",
    "IS_ZERO_RESISTANCE",
    "insert_thick_vessel_junction_nodes",
    "remove_small_mask_components_by_volume",
    "remove_small_vessel_components_by_volume",
    "remove_edges_for_self_connected_nodes",
    "seed_edges_have_full_mask_coverage",
    "select_nodes_at_hop_distance",
    "add_edge_safe",
    "has_edge_safe",
    "remove_edge_safe",
    "get_all_edge_data",
    "create_merged_edge_attributes",
    "validate_voxel_path_continuity",
    "merge_edge_voxels_at_node",
    "merge_voxel_paths_at_node",
    "orient_voxel_path_to_node",
    "get_line_points_3d",
    "calculate_path_length",
    "calculate_edge_length",
    "is_path_curved",
    "merge_curved_edges",
    "orient_path_to_endpoint",
    "orient_path_from_startpoint",
    "improve_straight_edge_with_skeleton",
    "trace_skeleton_path",
    "parse_skeleton_data",
    "find_nearest_skeleton_voxel",
    "astar_skeleton_path",
    "are_paths_similar",
    "should_add_merged_edge",
]
