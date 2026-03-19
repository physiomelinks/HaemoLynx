"""Graph building and topology optimization for vascular networks."""
from .build import build_graph_segment_skan_stitched_loops
from .reconnect import reconnect_secondary_loop_edges
from .optimise import optimise_graph_topology_fixed
from .validate import validate_skeleton_connection
from .degree2 import (
    safer_simple_remove_all_degree2_nodes,
    trivial_remove_all_degree2_nodes,
    create_trivial_merged_edge,
    smart_multigraph_degree2_removal,
    merge_edges_with_topology_improvement,
)
from .prune import prune_vascular_stubs, remove_edges_for_self_connected_nodes
from .diagnostics import diagnose_degree2_nodes, format_degree2_diagnostics_report
from .collapse import collapse_node_clusters
from .branch_order import assign_branch_orders
from .boundaries import select_boundary_terminal_nodes, select_boundary_nodes_by_method
from ._helpers import (
    add_edge_safe,
    has_edge_safe,
    remove_edge_safe,
    get_all_edge_data,
    create_merged_edge_attributes,
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
    "select_boundary_terminal_nodes",
    "select_boundary_nodes_by_method",
    "remove_edges_for_self_connected_nodes",
    "add_edge_safe",
    "has_edge_safe",
    "remove_edge_safe",
    "get_all_edge_data",
    "create_merged_edge_attributes",
    "create_merged_edge_attributes_simple",
    "create_merged_edge_attributes_full",
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
