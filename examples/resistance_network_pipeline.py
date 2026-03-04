#Ability to compare datasets - Dave's suggestion
#Summarise by BO in statistics
#Resistance should be from start of arteriole to end of venule
#Mean distance of object (classifier) to each capillary type and BO
#Overall list of every vessel and its properties

#!/usr/bin/env python3
"""Refactored full pipeline example using ImageLynx package."""
import logging
import sys
import pickle
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
# STARTING NODES and OUTPUT Nodes are now calculated automatically by looking for degree 1 nodes at start or
# end of the image.
EDGE_PERCENT = 10.0
END_PERCENT = 10.0
# For 3D skeletons this is usually the y-axis in (z, y, x).
NODE_EDGE_AXIS = 1
STARTING_NODES: list[int] = []
OUTPUT_NODES: list[int] = []
# TODO HD note - eventually add script to run resistance measurements between every BO1 (arteriole) and every (non-arteriole) capillary node, and between every node.
# TODO automate the selection of resistance node pairs
# RESISTANCE_NODE_PAIR = (426, 509)  # (source_node_id, target_node_id)
RESISTANCE_NODE_PAIR = (918, 47)  # (source_node_id, target_node_id)
INPUT_P_BC = 1000 # Pa 
OUTPUT_P_BC = 500 # Pa
VISUALIZE_RESULTS = True
VISUALIZE_VTK = False
VERBOSE_LOGGING = False
DO_SKELETONIZE = True
DO_GRAPH_BUILDING = True
DO_RESISTANCE_CALCULATION = False
CONSTRICT_AT_PERICYTES = True
MIN_BRANCH_LENGTH = 10
VTK_OUTPUT_PREFIX = root_dir / "examples" / "outputs" / "resistance_network"
SKELETON_CLOSING_RADIUS = 3
SKELETON_BRIDGE_GAP_SIZE = 4
SKELETON_MIN_BRANCH_LENGTH = 3
SKELETON_MAX_BRIDGE_DISTANCE = 0
SKELETON_COMPONENT_CONNECTIVITY = 3
# Keep only connected components at or above this percentage of total
# skeleton voxels (e.g. 5.0 -> keep components >= 5% of total skeleton voxels).
SKELETON_MIN_COMPONENT_PERCENT = 5.0


def select_boundary_terminal_nodes(
    G: nx.Graph,
    image_shape: tuple[int, ...],
    *,
    edge_percent: float,
    end_percent: float,
    axis: int = 1,
) -> tuple[list[int], list[int]]:
    """Select degree-1 nodes in top and bottom image bands along one axis."""
    if not (0.0 <= edge_percent <= 100.0 and 0.0 <= end_percent <= 100.0):
        raise ValueError("edge_percent and end_percent must be in [0, 100].")
    if axis < 0 or axis >= len(image_shape):
        raise ValueError(f"axis={axis} out of bounds for image shape {image_shape}.")

    node_pos = nx.get_node_attributes(G, "pos")
    terminal_nodes = [node for node, degree in G.degree() if degree == 1 and node in node_pos]
    if not terminal_nodes:
        return [], []

    axis_size = float(image_shape[axis] - 1)
    top_limit = axis_size * (edge_percent / 100.0)
    bottom_start = axis_size * (1.0 - (end_percent / 100.0))

    def axis_coord(node_id: int) -> float:
        return float(np.asarray(node_pos[node_id], dtype=float)[axis])

    starting = [node for node in terminal_nodes if axis_coord(node) <= top_limit]
    outputs = [node for node in terminal_nodes if axis_coord(node) >= bottom_start]
    starting_set = set(starting)
    outputs = [node for node in outputs if node not in starting_set]

    starting.sort(key=lambda n: (axis_coord(n), n))
    outputs.sort(key=lambda n: (-axis_coord(n), n))
    return starting, outputs


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
    graph_path = INPUT_PATH.with_name(f"{INPUT_PATH.stem}_graph.pkl")
    projection_path = PLOT_DIR / "skeleton_projection.png"

    if DO_SKELETONIZE:
        if INPUT_FORMAT == "tif":
            image, skeleton = io.load_and_skeletonize_3d_tif(
                INPUT_PATH,
                closing_radius=SKELETON_CLOSING_RADIUS,
                bridge_gap_size=SKELETON_BRIDGE_GAP_SIZE,
            )
        elif INPUT_FORMAT == "h5":
            if not H5_DATASET_NAME:
                raise ValueError("Set H5_DATASET_NAME when INPUT_FORMAT is 'h5'.")
            image, skeleton = io.load_and_skeletonize_3d_h5(
                INPUT_PATH,
                H5_DATASET_NAME,
                closing_radius=SKELETON_CLOSING_RADIUS,
                bridge_gap_size=SKELETON_BRIDGE_GAP_SIZE,
            )
        else:
            raise ValueError("INPUT_FORMAT must be 'tif' or 'h5'.")
        
        preprocessing.print_skeleton_connectivity_stats(
            "raw",
            skeleton,
            component_connectivity=SKELETON_COMPONENT_CONNECTIVITY,
        )

        skeleton = preprocessing.preprocess_skeleton_for_graph(
            skeleton,
            min_branch_length=SKELETON_MIN_BRANCH_LENGTH,
            max_bridge_distance=SKELETON_MAX_BRIDGE_DISTANCE,
            component_connectivity=SKELETON_COMPONENT_CONNECTIVITY,
            min_component_fraction=SKELETON_MIN_COMPONENT_PERCENT / 100.0,
        )
        preprocessing.print_skeleton_connectivity_stats(
            "cleaned",
            skeleton,
            component_connectivity=SKELETON_COMPONENT_CONNECTIVITY,
        )
        
        # save the skeleton
        np.save(skeleton_path, skeleton)
    else:
        # load the skeleton
        skeleton = np.load(skeleton_path)
        image = tifffile.imread(INPUT_PATH)

    # Optional interactive skeleton viewer (disabled by default for debug runs).
    if VISUALIZE_RESULTS:
        visualization.visualize_skeleton(skeleton, save_path=projection_path)

    if DO_GRAPH_BUILDING:
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
        visualization.visualize_edges_and_nodes(image, G, label_nodes=True, save_path=PLOT_DIR / "optimise_graph_topology_fixed.png")
        G = graph.safer_simple_remove_all_degree2_nodes(
            G,
            max_degree=5,
            debug=VERBOSE_LOGGING,
        )
        visualization.visualize_edges_and_nodes(image, G, label_nodes=True, save_path=PLOT_DIR / "safer_simple_remove_all_degree2_nodes.png")
        G = graph.trivial_remove_all_degree2_nodes(
            G,
            max_degree=5,
            debug=VERBOSE_LOGGING,
        )
        visualization.visualize_edges_and_nodes(image, G, label_nodes=True, save_path=PLOT_DIR / "trivial_remove_all_degree2_nodes.png")
        G = graph.smart_multigraph_degree2_removal(
            G,
            skeleton,
            debug=VERBOSE_LOGGING,
        )
        visualization.visualize_edges_and_nodes(image, G, label_nodes=True, save_path=PLOT_DIR / "smart_multigraph_degree2_removal.png")
        G = graph.prune_vascular_stubs(G, debug=VERBOSE_LOGGING)

        # remove any nodes that are connected to themselves with no nodes in between
        G = graph.remove_edges_for_self_connected_nodes(G)

        # Visualize node labels for debugging/verification of auto-selected boundary nodes.
        visualization.visualize_edges_and_nodes(image, G, label_nodes=True, save_path=PLOT_DIR / "prune_vascular_stubs.png")
        
        # G = graph.smart_multigraph_degree2_removal(
        #     G,
        #     skeleton,
        #     debug=VERBOSE_LOGGING,
        # )
        # visualization.visualize_edges_and_nodes(image, G, label_nodes=True, save_path=PLOT_DIR / "smart_multigraph_degree2_removal_REPEAT.png")
    
        with graph_path.open("wb") as f:
            pickle.dump(G, f)
        print(f"Saved graph to: {graph_path}")
    else:
        if not graph_path.exists():
            raise FileNotFoundError(
                f"Graph file not found at {graph_path}. "
                "Set DO_GRAPH_BUILDING=True to generate it first."
            )
        with graph_path.open("rb") as f:
            G = pickle.load(f)
        print(f"Loaded graph from: {graph_path}")

    STARTING_NODES[:] = []
    OUTPUT_NODES[:] = []
    start_nodes, out_nodes = select_boundary_terminal_nodes(
        G,
        image.shape,
        edge_percent=EDGE_PERCENT,
        end_percent=END_PERCENT,
        axis=NODE_EDGE_AXIS,
    )
    STARTING_NODES.extend(start_nodes)
    OUTPUT_NODES.extend(out_nodes)
    print(
        f"Auto-selected {len(STARTING_NODES)} STARTING_NODES "
        f"(top {EDGE_PERCENT}%) and {len(OUTPUT_NODES)} OUTPUT_NODES "
        f"(bottom {END_PERCENT}%) along axis {NODE_EDGE_AXIS}."
    )
    print(f"Starting nodes are: {STARTING_NODES}")
    print(f"Output nodes are: {OUTPUT_NODES}")

    resistance_node_pair = RESISTANCE_NODE_PAIR
    if STARTING_NODES and OUTPUT_NODES:
        resistance_node_pair = (STARTING_NODES[0], OUTPUT_NODES[0])
        print(f"Auto-selected resistance node pair: {resistance_node_pair}")

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
    # FA I have no idea if pericyte location is correct. AI did that part.
    # FA I don't fully understand how pericyte location is currently determined?
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

    if DO_RESISTANCE_CALCULATION:
        source_node, target_node = resistance_node_pair
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
                f"\nSkipped two-point resistance: nodes {resistance_node_pair} "
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

    # 8) Also solve for flow throughout the network using the conductance matrix 
    # and the input and output pressures.
    flow, vtk_export = hemodynamics.solve_flow_from_conductance_matrix(
        conductance,
        node_list,
        INPUT_P_BC,
        OUTPUT_P_BC,
        STARTING_NODES,
        OUTPUT_NODES,
        vtk_export,
    )
    print("Flow through the network solved")
    print(f"Vtk file with flow data saved to: {vtk_export['vessels_path']}")

    # 9) Optional matplotlib visualization.
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
