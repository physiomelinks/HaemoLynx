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
from .branch_order import assign_branch_orders

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
    "assign_branch_orders",
    "remove_edges_for_self_connected_nodes",
]
