"""High-level pipeline orchestration."""
import logging
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import networkx as nx
from skan import csr

from . import io
from . import preprocessing
from . import graph
from . import hemodynamics
from . import statistics
from . import visualization

logger = logging.getLogger(__name__)

DEFAULT_DIAMETER_BY_BRANCH_ORDER_ENHANCED = {
    "BO1": {"d1": 6.2, "d2": 6.2},
    "BO2": {"d1": 4.0, "d2": 3.2},
    "BO3": {"d1": 5.0, "d2": 4.0},
    "BO4": {"d1": 5.0, "d2": 4.0},
    "BO5": {"d1": 4.0, "d2": 3.2},
    "BO6": {"d1": 4.0, "d2": 3.2},
    "BO7": {"d1": 4.0, "d2": 3.2},
    "BO8": {"d1": 4.0, "d2": 3.2},
    "BO9": {"d1": 4.0, "d2": 3.2},
    "B10": {"d1": 4.0, "d2": 3.2},
    "B11": {"d1": 4.0, "d2": 3.2},
    "B12": {"d1": 4.0, "d2": 3.2},
    "B13": {"d1": 4.0, "d2": 3.2},
    "B14": {"d1": 4.0, "d2": 3.2},
    "B15": {"d1": 4.0, "d2": 3.2},
    "B16": {"d1": 4.0, "d2": 3.2},
    "B17": {"d1": 4.0, "d2": 3.2},
    "B18": {"d1": 4.0, "d2": 3.2},
    "B19": {"d1": 4.0, "d2": 3.2},
    "B20": {"d1": 4.0, "d2": 3.2},
    "B21": {"d1": 4.0, "d2": 3.2},
    "B22": {"d1": 4.0, "d2": 3.2},
    "B23": {"d1": 4.0, "d2": 3.2},
    "B24": {"d1": 4.0, "d2": 3.2},
    "B25": {"d1": 4.0, "d2": 3.2},
    "B26": {"d1": 4.0, "d2": 3.2},
}


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
    diameter_config = diameter_config or DEFAULT_DIAMETER_BY_BRANCH_ORDER_ENHANCED

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

    sk = csr.Skeleton(skeleton)

    G, voxel_loops, loop_edges = graph.build_graph_segment_skan_stitched_loops(
        sk, skeleton, debug=debug
    )
    G = graph.reconnect_secondary_loop_edges(G, skeleton, debug=debug)
    G, _ = graph.optimise_graph_topology_fixed(
        G, voxel_loops, loop_edges, skeleton_data=skeleton, debug=debug
    )
    # Keep only topology-aware degree-2 removal to avoid introducing
    # straight-line shortcuts from aggressive simple merges.
    G = graph.smart_multigraph_degree2_removal(G, skeleton, debug=debug)
    G = graph.prune_vascular_stubs(G, debug=debug)
    G = graph.smart_multigraph_degree2_removal(G, skeleton, debug=debug)

    poiseuille_model = hemodynamics.PoiseuilleModel(
        constriction_length=40.0,
        constriction_spacing=100.0,
    )
    if starting_nodes:
        graph.assign_branch_orders(G, starting_nodes)
        poiseuille_model.set_poiseuille_weights_with_constrictions(G, diameter_config)
    if custom_edges:
        poiseuille_model.set_poiseuille_edge_weights(
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
