"""Assemble a cleaned vascular graph from a 3D skeleton."""
from __future__ import annotations

import logging
from collections.abc import Callable
import networkx as nx
import numpy as np
from skan import csr

from .build import build_graph_segment_skan_stitched_loops
from .collapse import collapse_node_clusters
from .degree2 import smart_multigraph_degree2_removal
from .diagnostics import diagnose_degree2_nodes, format_degree2_diagnostics_report
from .optimise import optimise_graph_topology_fixed, reconnect_orphan_and_dangling_nodes
from .prune import prune_vascular_stubs, remove_edges_for_self_connected_nodes
from .reconnect import reconnect_secondary_loop_edges

logger = logging.getLogger(__name__)

StepCallback = Callable[[nx.MultiGraph, str], None]


def _notify_step(
    G: nx.MultiGraph,
    label: str,
    step_callback: StepCallback | None,
) -> None:
    if step_callback is not None:
        step_callback(G, label)


def _log_degree2_diagnostics(G: nx.MultiGraph, max_degree: int, debug: bool) -> None:
    if not debug:
        return
    degree2_diag = diagnose_degree2_nodes(G, max_degree=max_degree)
    print(format_degree2_diagnostics_report(degree2_diag))


def build_graph_from_skeleton(
    skeleton: np.ndarray,
    voxel_size: tuple[float, float, float] = (1.0, 1.0, 1.0),
    graph_reconnect_threshold: float = 10.0,
    final_orphan_reconnect_threshold: float = 3.0,
    cluster_collapse_distance: float = 5.0,
    min_stub_length: float = 10.0,
    debug: bool = False,
    step_callback: StepCallback | None = None,
) -> nx.MultiGraph:
    """
    Build and clean a vascular NetworkX graph from a binary 3D skeleton.

    Runs the full topology pipeline: skan extraction, loop stitching, secondary
    loop reconnection, topology optimisation, degree-2 removal passes, cluster
    collapse, stub pruning, self-edge removal, and orphan reconnection.

    Parameters
    ----------
    skeleton
        Binary 3D skeleton array.
    voxel_size
        Physical voxel size (x, y, z) in microns.
    graph_reconnect_threshold
        Reconnection threshold for initial graph build and topology optimisation.
    final_orphan_reconnect_threshold
        Reconnection threshold for orphan/dangling node repair.
    cluster_collapse_distance
        Distance threshold for collapsing nearby node clusters.
    min_stub_length
        Minimum stub length (microns) retained before pruning.
    debug
        When True, print degree-2 diagnostic reports after cleanup passes.
    step_callback
        Optional ``callback(G, step_label)`` invoked after each topology step.

    Returns
    -------
    nx.MultiGraph
        Cleaned vascular graph.
    """
    degree2_pass1_max_degree = 4
    degree2_pass2_max_degree = 8

    logger.info("Building skan Skeleton object...")
    print("Building skan Skeleton object...")
    sk = csr.Skeleton(skeleton)
    print(f"skan Skeleton built: {sk.n_paths} paths")

    print("Building graph (loop detection + segment extraction)...")
    G, voxel_loops, loop_edges = build_graph_segment_skan_stitched_loops(
        sk,
        skeleton,
        debug=debug,
        voxel_size=voxel_size,
        reconnect_threshold=graph_reconnect_threshold,
    )
    _notify_step(G, "build_graph_segment_skan_stitched_loops", step_callback)

    G = reconnect_secondary_loop_edges(
        G,
        skeleton,
        voxel_size=voxel_size,
        debug=debug,
    )
    _notify_step(G, "reconnect_secondary_loop_edges", step_callback)

    G, _ = optimise_graph_topology_fixed(
        G,
        voxel_loops,
        loop_edges,
        skeleton_data=skeleton,
        debug=debug,
        reconnect_threshold=graph_reconnect_threshold,
    )
    _notify_step(G, "optimise_graph_topology_fixed", step_callback)

    G = smart_multigraph_degree2_removal(
        G,
        skeleton,
        max_degree=degree2_pass1_max_degree,
        debug=debug,
    )
    _notify_step(G, "smart_multigraph_degree2_removal_pass1", step_callback)
    _log_degree2_diagnostics(G, degree2_pass1_max_degree, debug)

    G = collapse_node_clusters(
        G,
        distance_threshold=cluster_collapse_distance,
        debug=debug,
    )
    _notify_step(G, "collapse_node_clusters", step_callback)

    G = smart_multigraph_degree2_removal(
        G,
        skeleton,
        max_degree=degree2_pass2_max_degree,
        debug=debug,
    )
    _notify_step(G, "smart_multigraph_degree2_removal_post_collapse", step_callback)

    G = prune_vascular_stubs(G, debug=debug, min_stub_length=min_stub_length)
    _notify_step(G, "prune_vascular_stubs", step_callback)
    _log_degree2_diagnostics(G, degree2_pass2_max_degree, debug)

    G = smart_multigraph_degree2_removal(
        G,
        skeleton,
        max_degree=degree2_pass2_max_degree,
        debug=debug,
    )
    _notify_step(G, "smart_multigraph_degree2_removal_post_prune", step_callback)
    _log_degree2_diagnostics(G, degree2_pass2_max_degree, debug)

    G = remove_edges_for_self_connected_nodes(G)
    _notify_step(G, "remove_edges_for_self_connected_nodes", step_callback)

    G = reconnect_orphan_and_dangling_nodes(
        G,
        skeleton_data=skeleton,
        reconnect_threshold=final_orphan_reconnect_threshold,
        include_degree1=True,
        max_new_edges_per_node=1,
        validate_reconnections=True,
        debug=debug,
    )
    _notify_step(G, "reconnect_orphan_and_dangling_nodes", step_callback)

    G = smart_multigraph_degree2_removal(
        G,
        skeleton,
        max_degree=degree2_pass1_max_degree,
        debug=debug,
    )
    _notify_step(G, "smart_multigraph_degree2_removal_post_orphan_reconnect", step_callback)
    _log_degree2_diagnostics(G, degree2_pass2_max_degree, debug)

    return G
