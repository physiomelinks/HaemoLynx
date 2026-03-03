#Ability to compare datasets - Dave's suggestion
#Summarise by BO in statistics
#Resistance should be from start of arteriole to end of venule
#Mean distance of object (classifier) to each capillary type and BO
#Overall list of every vessel and its properties

#!/usr/bin/env python3
"""Refactored full pipeline example using ImageLynx package."""
import logging
import sys
from pathlib import Path
from skan import csr
import tifffile
import numpy as np
import networkx as nx

# Ensure package is importable when running from repo root.
root_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


from ImageLynx import graph, hemodynamics, io, preprocessing, statistics, visualization

# ---------------------------
# Beginner-friendly settings
# ---------------------------
INPUT_PATH = root_dir / "examples" / "images" / "Nerve_capillaries.tif"
PLOT_DIR = root_dir / "examples" / "plots" 
if not PLOT_DIR.exists():
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
INPUT_FORMAT = "tif"  # "tif" or "h5"
H5_DATASET_NAME = None  # For h5 input, e.g. "data"
# STARTING_NODES = [426, 184, 509]
STARTING_NODES = [3, 95, 169]
#HD note - eventually add script to run resistance measurements between every BO1 (arteriole) and every (non-arteriole) capillary node, and between every node.
# RESISTANCE_NODE_PAIR = (426, 509)  # (source_node_id, target_node_id)
RESISTANCE_NODE_PAIR = (3, 166)  # (source_node_id, target_node_id)
VISUALIZE_RESULTS = True
VISUALIZE_VTK = False
VERBOSE_LOGGING = True
DO_SKELETONIZE = True
CONSTRICT_AT_PERICYTES = True
MIN_BRANCH_LENGTH = 10
VTK_OUTPUT_PREFIX = root_dir / "examples" / "outputs" / "resistance_network"


def main() -> None:

    # TODO these diameters etc should be automated 
    #HD note - there should be a manual option, as per below, to add in in vivo diameters, and a option to read in diameters from the original image (via FWHM)
    #HD note - this no longer features the ability to manually define a limited number of user determined vessels (ie endoneurial vessels), which can't be done automatically. Not relevant for alice but relevant generally.
    """Configuration defaults for diameter maps."""

    # Diameter by branch order (simple scalar)
    DIAMETER_BY_BRANCH_ORDER = {
        "BO1": 6.2,
        "BO2": 4.0,
        "BO3": 5.0,
        "BO4": 5.0,
        "BO5": 4.0,
        "BO6": 4.0,
        "BO7": 4.0,
        "BO8": 4.0,
        "BO9": 4.0,
        "B10": 4.0,
        "B11": 4.0,
        "B12": 4.0,
        "B13": 4.0,
        "B14": 4.0,
        "B15": 4.0,
        "B16": 4.0,
        "B17": 4.0,
        "B18": 4.0,
        "B19": 4.0,
        "B20": 4.0,
        "B21": 4.0,
        "B22": 4.0,
        "B23": 4.0,
        "B24": 4.0,
        "B25": 4.0,
        "B26": 4.0,
    }

    # Enhanced: passive (d1) and constricted (d2) diameters
    DIAMETER_BY_BRANCH_ORDER_ENHANCED = {
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
    
    # These are vesses that constrict differently (e.g. endoneurial vessels).
    custom_edges= [
        (103, 262),
        (103, 104),
        (309, 363),
        (363, 746),
        (363, 745),
        (746, 874),
        (745, 766),
        (874, 1140),
        (221, 309),
        (103, 106),
        (34, 222),
        (222, 258),
        (233, 236),
        (123, 176),
        (234, 235),
        (35, 65),
        (32, 35),
        (260,290),
        (290,846),
        (290,846),
        (766, 846),
        (766, 845)
    ]  
    
    logging.basicConfig(
        level=logging.DEBUG if VERBOSE_LOGGING else logging.INFO,
        format="[%(levelname)s] %(message)s",
    )

    # 1) Load image and skeletonize.
    skeleton_path = INPUT_PATH.with_name(f"{INPUT_PATH.stem}_skeleton.npy")
    if DO_SKELETONIZE:
        if INPUT_FORMAT == "tif":
            image, skeleton = io.load_and_skeletonize_3d_tif(INPUT_PATH)
        elif INPUT_FORMAT == "h5":
            if not H5_DATASET_NAME:
                raise ValueError("Set H5_DATASET_NAME when INPUT_FORMAT is 'h5'.")
            image, skeleton = io.load_and_skeletonize_3d_h5(INPUT_PATH, H5_DATASET_NAME)
        else:
            raise ValueError("INPUT_FORMAT must be 'tif' or 'h5'.")
        
        # save the skeleton
        np.save(skeleton_path, skeleton)
    else:
        # load the skeleton
        skeleton = np.load(skeleton_path)
        image = tifffile.imread(INPUT_PATH)

    # 2) Clean skeleton before graph conversion.
    skeleton = preprocessing.preprocess_skeleton_for_graph(
        skeleton,
        min_branch_length=MIN_BRANCH_LENGTH,
    )
    # visualise the skeleton TODO
    # visualization.visualise_skeleton(skeleton)

    # 3) Convert skeleton to graph.
    sk = csr.Skeleton(skeleton)

    G, voxel_loops, loop_edges = graph.build_graph_segment_skan_stitched_loops(
        sk,
        skeleton,
        debug=VERBOSE_LOGGING,
    )
    # visualization.visualize_edges_and_nodes(image, G, label_nodes=True)
    G = graph.reconnect_secondary_loop_edges(G, skeleton, debug=VERBOSE_LOGGING)
    visualization.visualize_edges_and_nodes(image, G, label_nodes=True, save_path=PLOT_DIR / "reconnect_secondary_loop_edges.png")
    
    G, _ = graph.optimise_graph_topology_fixed(
        G,
        voxel_loops,
        loop_edges,
        skeleton_data=skeleton,
        debug=VERBOSE_LOGGING,
    )
    visualization.visualize_edges_and_nodes(image, G, label_nodes=True)
    G = graph.safer_simple_remove_all_degree2_nodes(
        G,
        max_degree=5,
        debug=VERBOSE_LOGGING,
    )
    visualization.visualize_edges_and_nodes(image, G, label_nodes=True)
    G = graph.trivial_remove_all_degree2_nodes(
        G,
        max_degree=5,
        debug=VERBOSE_LOGGING,
    )
    visualization.visualize_edges_and_nodes(image, G, label_nodes=True)
    G = graph.smart_multigraph_degree2_removal(
        G,
        skeleton,
        debug=VERBOSE_LOGGING,
    )
    visualization.visualize_edges_and_nodes(image, G, label_nodes=True, save_path=PLOT_DIR / "smart_multigraph_degree2_removal.png")
    G = graph.prune_vascular_stubs(G, debug=VERBOSE_LOGGING)

    # Here we should visualise the network with node numbers so we can choose the starting nodes.
    print(f"Starting nodes are: {STARTING_NODES}")
    print("Check that they correspond to the correct node numbers in the image")
    visualization.visualize_edges_and_nodes(image, G, label_nodes=True, save_path=PLOT_DIR / "prune_vascular_stubs.png")
    
    G = graph.smart_multigraph_degree2_removal(
        G,
        skeleton,
        debug=VERBOSE_LOGGING,
    )
    visualization.visualize_edges_and_nodes(image, G, label_nodes=True, save_path=PLOT_DIR / "smart_multigraph_degree2_removal_REPEAT.png")

    # 4) Add branch orders and hemodynamic edge weights.
    #HD note - eventually pericyte localisation should be able to be either determined by this manual method, or via loading in a segmented image of pericytes?
    #HD note - eventually add in probability of pericyte contraction?
    if STARTING_NODES:
        graph.assign_branch_orders(G, STARTING_NODES)
        poiseuille_model = hemodynamics.PoiseuilleModel(
            constriction_length=40.0,
            constriction_spacing=100.0,
        )
        if CONSTRICT_AT_PERICYTES:
            poiseuille_model.set_poiseuille_edge_weights(
                G,
                custom_edges,
                edge_diameter=6.0,
                use_resistance=False,
            )
        else:
            poiseuille_model.set_poiseuille_weights_with_constrictions(
                G,
                DIAMETER_BY_BRANCH_ORDER_ENHANCED,
            )

    # 5) Export vessels/pericytes/nodes to VTK and optionally visualize in PyVista.
    # TODO VTK Export currently not creating a connected graph. Don't use this yet.
    vtk_export = visualization.graph_to_vtk(G, VTK_OUTPUT_PREFIX)
    print("\n=== VTK Export ===")
    print(f"  Vessels:   {vtk_export['vessels_path']}")
    print(f"  Pericytes: {vtk_export['pericytes_path']}")
    print(f"  Nodes:     {vtk_export['nodes_path']}")
    print(f"  Counts: vessels={vtk_export['vessel_line_count']}, "
          f"pericytes={vtk_export['pericyte_count']}, nodes={vtk_export['node_count']}")
    if VISUALIZE_VTK:
        visualization.visualize_vtk_network(
            vtk_export["vessels_path"],
            vtk_export["pericytes_path"],
            vtk_export["nodes_path"],
            show_nodes=False,
        )

    # 6) Compute effective resistance between two selected nodes.
    conductance, node_list = hemodynamics.build_conductance_matrix_from_graph(G)
    node_to_idx = {node_id: idx for idx, node_id in enumerate(node_list)}

    source_node, target_node = RESISTANCE_NODE_PAIR
    if source_node in node_to_idx and target_node in node_to_idx:
        laplacian = hemodynamics.calc_laplacian_from_conductance_matrix(conductance)
        two_point_resistance = hemodynamics.calc_two_point_from_laplacian_matrix_nodeID(
            laplacian,
            G,
            source_node,
            target_node,
        )
        print(
            f"\nEffective resistance between nodes {source_node} and "
            f"{target_node}: {two_point_resistance}"
        )
    else:
        print(
            f"\nSkipped two-point resistance: nodes {RESISTANCE_NODE_PAIR} "
            "are not both present in the graph."
        )

    # 7) Compute and print vessel statistics.
    node_positions = nx.get_node_attributes(G, "pos")
    stats = statistics.compute_comprehensive_vessel_statistics(
        G,
        node_positions=node_positions,
        image_dimensions=image.shape,
    )

    print("\n=== Statistics ===")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # 8) Optional matplotlib visualization.
    if VISUALIZE_RESULTS:
        visualization.plot_node_degree_distribution(G)
        visualization.visualize_edges_and_nodes(image, G)
        visualization.interactive_3d_graph(G)
        #HD note - need visualisation of pericyte localisations (ie based upon constriction data)
        
        if STARTING_NODES:
            visualization.visualize_geometry_with_branch_orders(
                image,
                G,
                group_above=8,
            )


if __name__ == "__main__":
    main()
