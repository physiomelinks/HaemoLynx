"""Assemble a cleaned vascular graph from a 3D skeleton."""
from __future__ import annotations

import logging
from collections.abc import Callable
import networkx as nx
import numpy as np
from skan import csr

from ._platform import skan_numba_warmup_skeleton
from .build import build_graph_segment_skan_stitched_loops
from .collapse import collapse_node_clusters
from .direction_aware_collapse import (
    DEFAULT_MAX_RADIAL_DISPERSION,
    DEFAULT_MIN_DEGREE_FOR_DISPERSION_CHECK,
    collapse_node_clusters_direction_aware,
)
from .cartwheel_guard import DEFAULT_TANGENT_LENGTH_UM
from .persistence_collapse import (
    DEFAULT_SEARCH_RADIUS_MULTIPLE,
    collapse_node_clusters_persistence,
)
from .degree2 import smart_multigraph_degree2_removal
from .diagnostics import diagnose_degree2_nodes, format_degree2_diagnostics_report
from .optimise import optimise_graph_topology_fixed, reconnect_orphan_and_dangling_nodes
from .prune import prune_vascular_stubs, remove_edges_for_self_connected_nodes
from .reconnect import reconnect_secondary_loop_edges

logger = logging.getLogger(__name__)

StepCallback = Callable[[nx.MultiGraph, str], None]

#: Every step `build_graph_from_skeleton` reports, in the order it runs them.
#: A progress bar has to know how many there will be before the first one
#: fires, and a snapshot writer has to know the labels are unique, so the list
#: is declared here rather than left implicit in the call order below.
#: `tests/test_graph_assemble.py` fails if a build stops matching it.
STEP_LABELS: tuple[str, ...] = (
    "build_graph_segment_skan_stitched_loops",
    "reconnect_secondary_loop_edges",
    "optimise_graph_topology_fixed",
    "smart_multigraph_degree2_removal_pass1",
    "collapse_node_clusters",
    "smart_multigraph_degree2_removal_post_collapse",
    "prune_vascular_stubs",
    "smart_multigraph_degree2_removal_post_prune",
    "remove_edges_for_self_connected_nodes",
    "reconnect_orphan_and_dangling_nodes",
    "smart_multigraph_degree2_removal_post_orphan_reconnect",
)


#: Where each label comes in the run, for the line every step logs. A step
#: names itself in that line, and `collapse_node_clusters` names itself in its
#: own summary too, so the `Step n/11` prefix is what tells the two apart.
_STEP_POSITIONS: dict[str, int] = {
    label: position for position, label in enumerate(STEP_LABELS, start=1)
}


def _notify_step(
    G: nx.MultiGraph,
    label: str,
    step_callback: StepCallback | None,
) -> None:
    # Eleven lines a run, ungated: what a step left behind is the answer to
    # "how many branches does the pipeline think there are", and asking for it
    # should not mean asking for the per-node detail as well.
    logger.info(
        "Step %d/%d %s: %d nodes / %d edges",
        _STEP_POSITIONS.get(label, 0),
        len(STEP_LABELS),
        label,
        G.number_of_nodes(),
        G.number_of_edges(),
    )
    if step_callback is not None:
        step_callback(G, label)


def _log_degree2_diagnostics(G: nx.MultiGraph, max_degree: int, debug: bool) -> None:
    if not debug:
        return
    degree2_diag = diagnose_degree2_nodes(G, max_degree=max_degree)
    logger.debug(format_degree2_diagnostics_report(degree2_diag))


def build_graph_from_skeleton(
    skeleton: np.ndarray,
    voxel_size: tuple[float, float, float] = (1.0, 1.0, 1.0),
    graph_reconnect_threshold: float = 10.0,
    final_orphan_reconnect_threshold: float = 3.0,
    cluster_collapse_distance: float = 5.0,
    min_stub_length: float = 10.0,
    debug: bool = False,
    step_callback: StepCallback | None = None,
    cluster_collapse_method: str = "distance_only",
    cluster_collapse_max_radial_dispersion: float = DEFAULT_MAX_RADIAL_DISPERSION,
    cluster_collapse_persistence_search_multiple: float = DEFAULT_SEARCH_RADIUS_MULTIPLE,
    cluster_collapse_direction_aware_min_degree: int = DEFAULT_MIN_DEGREE_FOR_DISPERSION_CHECK,
    cluster_collapse_direction_aware_tangent_length_um: float = DEFAULT_TANGENT_LENGTH_UM,
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
        Spacing of each array axis in microns, in canonical ``(z, y, x)`` order —
        *not* the ``(x, y, z)`` order reported by image metadata. Convert with
        ``haemolynx.io.voxel_size_zyx_from_xyz`` before calling.
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
    cluster_collapse_method
        ``"distance_only"`` (default) is the original, unmodified behaviour --
        every node within ``cluster_collapse_distance`` of another collapses
        via single-linkage clustering, however far that chains. ``"direction_
        aware"`` additionally refuses a merge that would turn the collapsed
        node's incident edges into a cartwheel shape -- see
        ``direction_aware_collapse`` for why and how. ``"persistence"`` cuts
        each local cluster at its own 0-dimensional-persistence gap instead
        of one global distance -- see ``persistence_collapse`` for the cited
        mathematics and why it can leave more than one representative node
        where ``distance_only`` would merge everything into one.
    cluster_collapse_max_radial_dispersion
        Only read when *cluster_collapse_method* is ``"direction_aware"`` --
        see ``direction_aware_collapse.collapse_node_clusters_direction_aware``.
    cluster_collapse_persistence_search_multiple
        Only read when *cluster_collapse_method* is ``"persistence"`` -- see
        ``persistence_collapse.collapse_node_clusters_persistence``.
    cluster_collapse_direction_aware_min_degree, cluster_collapse_direction_aware_tangent_length_um
        Only read when *cluster_collapse_method* is ``"direction_aware"`` --
        deliberately the same ``cartwheel_hub_min_degree`` /
        ``cartwheel_hub_tangent_length_um`` settings the cartwheel hub guard
        itself uses, since this collapse method gates merges with that
        guard's own geometry: tuning one without the other would silently
        decouple the diagnostic from the corrective gate it is modelled on.

    Returns
    -------
    nx.MultiGraph
        Cleaned vascular graph.
    """
    degree2_pass1_max_degree = 4
    degree2_pass2_max_degree = 8

    logger.info("Building skan Skeleton object...")
    warmup = skan_numba_warmup_skeleton()
    if warmup is not None:
        csr.Skeleton(warmup)
    sk = csr.Skeleton(skeleton)
    logger.info(f"skan Skeleton built: {sk.n_paths} paths")

    logger.info("Building graph (loop detection + segment extraction)...")
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

    if cluster_collapse_method == "direction_aware":
        G = collapse_node_clusters_direction_aware(
            G,
            distance_threshold=cluster_collapse_distance,
            max_radial_dispersion=cluster_collapse_max_radial_dispersion,
            min_degree_for_dispersion_check=cluster_collapse_direction_aware_min_degree,
            tangent_length_um=cluster_collapse_direction_aware_tangent_length_um,
            debug=debug,
        )
    elif cluster_collapse_method == "persistence":
        G = collapse_node_clusters_persistence(
            G,
            distance_threshold=cluster_collapse_distance,
            search_radius_multiple=cluster_collapse_persistence_search_multiple,
            debug=debug,
        )
    elif cluster_collapse_method == "distance_only":
        G = collapse_node_clusters(
            G,
            distance_threshold=cluster_collapse_distance,
            debug=debug,
        )
    else:
        raise ValueError(
            f"cluster_collapse_method must be 'distance_only', 'direction_aware' "
            f"or 'persistence', got {cluster_collapse_method!r}."
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
