"""High-level pipeline orchestration."""
import logging
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import networkx as nx

from . import io
from . import preprocessing
from . import graph
from . import hemodynamics
from . import statistics
from . import visualization
from .config import DIAMETER_BY_BRANCH_ORDER_ENHANCED

logger = logging.getLogger(__name__)


def run_pipeline(
    filepath: str,
    *,
    input_format: str = "tif",
    dataset_name: Optional[str] = None,
    starting_nodes: Optional[List[int]] = None,
    custom_edges: Optional[List] = None,
    diameter_config: Optional[Dict] = None,
    min_branch_length: int = 10,
    debug: bool = False,
) -> Tuple[np.ndarray, np.ndarray, nx.MultiGraph, Dict[str, Any]]:
    """
    Run full pipeline: load -> preprocess -> build graph -> optimise -> stats.

    Returns:
        (image, skeleton, G, stats_dict)
    """
    diameter_config = diameter_config or DIAMETER_BY_BRANCH_ORDER_ENHANCED

    if input_format.lower() == "tif":
        image, skeleton = io.load_and_skeletonize_3d_tif(filepath)
    else:
        if not dataset_name:
            raise ValueError("dataset_name required for HDF5 input")
        image, skeleton = io.load_and_skeletonize_3d_h5(
            filepath, dataset_name
        )

    skeleton = preprocessing.preprocess_skeleton_for_graph(
        skeleton, min_branch_length=min_branch_length
    )

    try:
        from skan import csr
        sk = csr.Skeleton(skeleton)
    except ImportError:
        raise ImportError("skan is required for graph building")

    G, voxel_loops, loop_edges = graph.build_graph_segment_skan_stitched_loops(
        sk, skeleton, debug=debug
    )
    G = graph.reconnect_secondary_loop_edges(G, skeleton, debug=debug)
    G, _ = graph.optimise_graph_topology_fixed(
        G, voxel_loops, loop_edges, skeleton_data=skeleton, debug=debug
    )
    G = graph.safer_simple_remove_all_degree2_nodes(G, max_degree=5, debug=debug)
    G = graph.trivial_remove_all_degree2_nodes(G, max_degree=5, debug=debug)
    G = graph.smart_multigraph_degree2_removal(G, skeleton, debug=debug)
    G = graph.prune_vascular_stubs(G, debug=debug)
    G = graph.smart_multigraph_degree2_removal(G, skeleton, debug=debug)

    if starting_nodes:
        graph.assign_branch_orders(G, starting_nodes)
        hemodynamics.set_poiseuille_weights_with_constrictions(
            G, diameter_config
        )
    if custom_edges:
        hemodynamics.set_poiseuille_edge_weights(
            G, custom_edges, 6.0, use_resistance=False
        )

    pos = nx.get_node_attributes(G, "pos")
    stats = statistics.compute_comprehensive_vessel_statistics(
        G,
        node_positions=pos,
        voxel_size=(1.0, 1.0, 1.0),
        image_dimensions=image.shape,
    )

    return image, skeleton, G, stats
