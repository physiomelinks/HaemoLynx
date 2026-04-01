"""Graph building and topology optimization for vascular networks."""
from .build import build_graph_segment_skan_stitched_loops
from .reconnect import reconnect_secondary_loop_edges
from .optimise import optimise_graph_topology_fixed, reconnect_orphan_and_dangling_nodes
from .validate import validate_skeleton_connection
from .degree2 import (
    safer_simple_remove_all_degree2_nodes,
    trivial_remove_all_degree2_nodes,
    create_trivial_merged_edge,
    smart_multigraph_degree2_removal,
    merge_edges_with_topology_improvement,
)
from .prune import (
    prune_vascular_stubs,
    remove_edges_for_self_connected_nodes,
    remove_isolated_nodes,
)
from .terminal_edges import remove_terminal_terminal_edges
from .diagnostics import diagnose_degree2_nodes, format_degree2_diagnostics_report
from .collapse import collapse_node_clusters
from .branch_order import assign_branch_orders, assign_hierarchical_branch_orders
from .boundaries import select_boundary_terminal_nodes, select_boundary_nodes_by_method
from .automated_vessel_assignment import (
    compute_overlapping_terminal_assignment_metrics,
    filter_io_nodes_to_terminal_degree1,
    infer_boundary_nodes_from_small_vessel_masks,
    infer_boundary_nodes_from_small_vessel_masks_progressive_dilation,
    resolve_overlapping_terminal_node_assignment,
    select_terminal_nodes_from_large_vessel_masks_progressive_dilation,
    select_terminal_nodes_from_large_vessel_masks,
    write_automated_vessel_assignment_3d_html,
    write_small_vessel_mask_boundary_labelling_3d_html,
)
from .confidence_vessel_assignment import (
    assess_large_vessel_assignment_quality,
    select_terminal_nodes_from_large_vessel_masks_progressive_dilation_confidence,
)
from .large_vessels import (
    dilate_binary_mask_by_microns,
    dilate_large_vessel_masks_by_microns,
    exclude_smaller_overlapping_large_vessel_components,
    exclude_smaller_overlapping_small_vessel_components,
    remove_small_opposite_attached_large_vessel_components,
)
from .mask_component_volume import (
    remove_small_mask_components_by_volume,
    remove_small_vessel_components_by_volume,
)
from .boundary_node_fallback import (
    select_nodes_at_hop_distance,
    seed_edges_have_full_mask_coverage,
)
from .remove_volume import remove_graph_elements_in_volumes
from .mask_continuity import (
    enforce_small_vessel_mask_continuity,
    redefine_small_masks_from_large_tangential_contact,
)
from ._helpers import (
    add_edge_safe,
    has_edge_safe,
    remove_edge_safe,
    get_all_edge_data,
    create_merged_edge_attributes,
    calculate_voxel_path_length,
    validate_voxel_path_continuity,
    merge_edge_voxels_at_node,
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
    "build_graph_segment_skan_stitched_loops",
    "reconnect_secondary_loop_edges",
    "optimise_graph_topology_fixed",
    "reconnect_orphan_and_dangling_nodes",
    "validate_skeleton_connection",
    "safer_simple_remove_all_degree2_nodes",
    "trivial_remove_all_degree2_nodes",
    "create_trivial_merged_edge",
    "smart_multigraph_degree2_removal",
    "merge_edges_with_topology_improvement",
    "prune_vascular_stubs",
    "diagnose_degree2_nodes",
    "format_degree2_diagnostics_report",
    "collapse_node_clusters",
    "assign_branch_orders",
    "assign_hierarchical_branch_orders",
    "select_boundary_terminal_nodes",
    "select_boundary_nodes_by_method",
    "compute_overlapping_terminal_assignment_metrics",
    "filter_io_nodes_to_terminal_degree1",
    "infer_boundary_nodes_from_small_vessel_masks",
    "infer_boundary_nodes_from_small_vessel_masks_progressive_dilation",
    "resolve_overlapping_terminal_node_assignment",
    "select_terminal_nodes_from_large_vessel_masks_progressive_dilation",
    "select_terminal_nodes_from_large_vessel_masks",
    "write_automated_vessel_assignment_3d_html",
    "write_small_vessel_mask_boundary_labelling_3d_html",
    "assess_large_vessel_assignment_quality",
    "select_terminal_nodes_from_large_vessel_masks_progressive_dilation_confidence",
    "dilate_binary_mask_by_microns",
    "dilate_large_vessel_masks_by_microns",
    "exclude_smaller_overlapping_large_vessel_components",
    "exclude_smaller_overlapping_small_vessel_components",
    "remove_small_opposite_attached_large_vessel_components",
    "remove_small_mask_components_by_volume",
    "remove_small_vessel_components_by_volume",
    "select_nodes_at_hop_distance",
    "seed_edges_have_full_mask_coverage",
    "remove_graph_elements_in_volumes",
    "enforce_small_vessel_mask_continuity",
    "redefine_small_masks_from_large_tangential_contact",
    "remove_edges_for_self_connected_nodes",
    "remove_terminal_terminal_edges",
    "remove_isolated_nodes",
    "add_edge_safe",
    "has_edge_safe",
    "remove_edge_safe",
    "get_all_edge_data",
    "create_merged_edge_attributes",
    "calculate_voxel_path_length",
    "validate_voxel_path_continuity",
    "merge_edge_voxels_at_node",
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
