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
BASE_PLOT_DIR = root_dir / "examples" / "plots" 
if not BASE_PLOT_DIR.exists():
    BASE_PLOT_DIR.mkdir(parents=True, exist_ok=True)
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
INPUT_P_BC = 1000 # Pa 
OUTPUT_P_BC = 500 # Pa
VISUALIZE_RESULTS = True
VISUALIZE_VTK = False
VERBOSE_LOGGING = False
DO_SKELETONIZE = False
DO_GRAPH_BUILDING = False
DO_EQUIV_RESISTANCE_CALCULATION = False
MIN_BRANCH_LENGTH = 10
VTK_OUTPUT_PREFIX = root_dir / "examples" / "outputs" / "resistance_network"
SKELETON_CLOSING_RADIUS = 2
SKELETON_BRIDGE_GAP_SIZE = 3
SKELETON_MIN_BRANCH_LENGTH = 3
SKELETON_MAX_BRIDGE_DISTANCE = 0
SKELETON_COMPONENT_CONNECTIVITY = 3
MIN_STUB_LENGTH = 10.0
# Keep only connected components at or above this percentage of total
# skeleton voxels (e.g. 5.0 -> keep components >= 5% of total skeleton voxels).
SKELETON_MIN_COMPONENT_PERCENT = 5.0
# TODO these diameters etc should be automated 
#HD note - there should be a manual option, as per below, to add in in vivo diameters, and a option to read in diameters from the original image (via FWHM)
#HD note - this no longer features the ability to manually define a limited number of user determined vessels (ie endoneurial vessels), which can't be done automatically. Not relevant for alice but relevant generally.
SET_STUBS_TO_OUTLET_PRESSURE = False
"""Configuration defaults for diameter maps."""

# Diameter by branch order (simple scalar)
print("TODO HARVEY CHANGE THIS ALL_DIAMS_CONST BACK TO FALSE FOR ORIGINAL RUN")
ALL_DIAMS_CONST = True

DIAMETER_BY_BRANCH_ORDER = {}
if ALL_DIAMS_CONST:
    for i in range(1,52):
        DIAMETER_BY_BRANCH_ORDER[f"B{i:02d}"] = 4.0
else:
    DIAMETER_BY_BRANCH_ORDER = {
        "B01": 6.2,
        "B02": 4.0,
        "B03": 5.0,
        "B04": 5.0,
    }
    for i in range(5, 52):
        DIAMETER_BY_BRANCH_ORDER[f"B{i:02d}"] = 4.0

# Enhanced: passive (d1) and constricted (d2) diameters
DIAMETER_BY_BRANCH_ORDER_ENHANCED = {
    "B01": {"d1": 6.2, "d2": 6.2},
    "B02": {"d1": 4.0, "d2": 3.2},
    "B03": {"d1": 5.0, "d2": 4.0},
    "B04": {"d1": 5.0, "d2": 4.0},
}
# repeat for 50 entries
for i in range(5, 52):
    DIAMETER_BY_BRANCH_ORDER_ENHANCED[f"B{i:02d}"] = {"d1": 4.0, "d2": 3.2}

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

def image_to_model_pipeline(image_path=INPUT_PATH, 
                            diameter_by_branch_order=DIAMETER_BY_BRANCH_ORDER_ENHANCED, # TODO this doesn't work with the DIAMETER_BY_BRANCH_ORDER dictionary it needs d2
                            plot_dir=BASE_PLOT_DIR,
                            verbose_logging=VERBOSE_LOGGING, 
                            do_skeletonize=DO_SKELETONIZE, 
                            do_graph_building=DO_GRAPH_BUILDING, 
                            do_equiv_resistance_calculation=DO_EQUIV_RESISTANCE_CALCULATION, 
                            min_branch_length=MIN_BRANCH_LENGTH, 
                            min_stub_length=MIN_STUB_LENGTH,
                            vtk_output_prefix=VTK_OUTPUT_PREFIX, 
                            skeleton_closing_radius=SKELETON_CLOSING_RADIUS, 
                            skeleton_bridge_gap_size=SKELETON_BRIDGE_GAP_SIZE, 
                            skeleton_min_branch_length=SKELETON_MIN_BRANCH_LENGTH, 
                            skeleton_max_bridge_distance=SKELETON_MAX_BRIDGE_DISTANCE, 
                            skeleton_component_connectivity=SKELETON_COMPONENT_CONNECTIVITY, 
                            skeleton_min_component_percent=SKELETON_MIN_COMPONENT_PERCENT, 
                            edge_percent=EDGE_PERCENT, 
                            end_percent=END_PERCENT, 
                            node_edge_axis=NODE_EDGE_AXIS, 
                            starting_nodes=STARTING_NODES, 
                            output_nodes=OUTPUT_NODES, 
                            input_p_bc=INPUT_P_BC, 
                            output_p_bc=OUTPUT_P_BC, 
                            set_stubs_to_outlet_pressure=SET_STUBS_TO_OUTLET_PRESSURE,
                            visualize_results=VISUALIZE_RESULTS, 
                            visualize_vtk=VISUALIZE_VTK) -> None:
                        
    # get image format from image_path
    input_format = image_path.suffix[1:].lower()
    if input_format not in ["tif", "h5"]:
        raise ValueError(f"Invalid image format: {input_format}")

    image_path = Path(image_path)
    vtk_output_prefix = Path(vtk_output_prefix)

    logging.basicConfig(
        level=logging.DEBUG if verbose_logging else logging.INFO,
        format="[%(levelname)s] %(message)s",
    )

    # 1) Load image and skeletonize.
    skeleton_path = image_path.with_name(f"{image_path.stem}_skeleton.npy")
    graph_path = image_path.with_name(f"{image_path.stem}_graph.pkl")
    projection_path = plot_dir / "skeleton_projection.png"
    if not plot_dir.exists():
        plot_dir.mkdir(parents=True, exist_ok=True)

    if do_skeletonize:
        if input_format == "tif":
            image, skeleton = io.load_and_skeletonize_3d_tif(
                image_path,
                closing_radius=skeleton_closing_radius,
                bridge_gap_size=skeleton_bridge_gap_size,
            )
        elif input_format == "h5":
            if not H5_DATASET_NAME:
                raise ValueError("Set H5_DATASET_NAME when INPUT_FORMAT is 'h5'.")
            image, skeleton = io.load_and_skeletonize_3d_h5(
                image_path,
                H5_DATASET_NAME,
                closing_radius=skeleton_closing_radius,
                bridge_gap_size=skeleton_bridge_gap_size,
            )
        else:
            raise ValueError("INPUT_FORMAT must be 'tif' or 'h5'.")
        
        preprocessing.print_skeleton_connectivity_stats(
            "raw",
            skeleton,
            component_connectivity=skeleton_component_connectivity,
        )
        visualization.visualize_skeleton(skeleton, save_path=plot_dir / "raw_skeleton.png")

        skeleton = preprocessing.preprocess_skeleton_for_graph(
            skeleton,
            min_branch_length=skeleton_min_branch_length,
            max_bridge_distance=skeleton_max_bridge_distance,
            component_connectivity=skeleton_component_connectivity,
            min_component_fraction=skeleton_min_component_percent / 100.0,
        )
        preprocessing.print_skeleton_connectivity_stats(
            "cleaned",
            skeleton,
            component_connectivity=skeleton_component_connectivity,
        )
        
        # save the skeleton
        np.save(skeleton_path, skeleton)
    else:
        # load the skeleton
        skeleton = np.load(skeleton_path)
        image = tifffile.imread(image_path)

    visualization.visualize_skeleton(skeleton, save_path=projection_path)

    if do_graph_building:
        # 3) Convert skeleton to graph.
        sk = csr.Skeleton(skeleton)

        G, voxel_loops, loop_edges = graph.build_graph_segment_skan_stitched_loops(
            sk,
            skeleton,
            debug=verbose_logging,
        )
        visualization.visualize_edges_and_nodes(image, G, label_nodes=True, save_path=plot_dir / "build_graph_segment_skan_stitched_loops.png")
        G = graph.reconnect_secondary_loop_edges(G, skeleton, debug=verbose_logging)
        visualization.visualize_edges_and_nodes(image, G, label_nodes=True, save_path=plot_dir / "reconnect_secondary_loop_edges.png")
        
        G, _ = graph.optimise_graph_topology_fixed(
            G,
            voxel_loops,
            loop_edges,
            skeleton_data=skeleton,
            debug=verbose_logging,
        )
        visualization.visualize_edges_and_nodes(image, G, label_nodes=True, save_path=plot_dir / "optimise_graph_topology_fixed.png")
        # Use only the topology-aware degree-2 removal path here. The legacy
        # simple/trivial passes can collapse curved paths into straight shortcuts
        # before smart merging has a chance to preserve topology.
        G = graph.smart_multigraph_degree2_removal(
            G,
            skeleton,
            debug=verbose_logging,
        )
        visualization.visualize_edges_and_nodes(image, G, label_nodes=True, save_path=plot_dir / "smart_multigraph_degree2_removal.png")
        G = graph.prune_vascular_stubs(G, debug=verbose_logging, min_stub_length=min_stub_length)

        # remove any nodes that are connected to themselves with no nodes in between
        G = graph.remove_edges_for_self_connected_nodes(G)

        # Visualize node labels for debugging/verification of auto-selected boundary nodes.
        visualization.visualize_edges_and_nodes(image, G, label_nodes=True, save_path=plot_dir / "prune_vascular_stubs.png")
        
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

    starting_nodes[:] = []
    output_nodes[:] = []
    start_nodes, out_nodes = graph.select_boundary_terminal_nodes(
        G,
        image.shape,
        edge_percent=edge_percent,
        end_percent=end_percent,
        axis=node_edge_axis,
    )
    starting_nodes.extend(start_nodes)
    output_nodes.extend(out_nodes)
    print(
        f"Auto-selected {len(starting_nodes)} STARTING_NODES "
        f"(top {edge_percent}%) and {len(output_nodes)} OUTPUT_NODES "
        f"(bottom {end_percent}%) along axis {node_edge_axis}."
    )
    print(f"Starting nodes are: {starting_nodes}")
    print(f"Output nodes are: {output_nodes}")

    if starting_nodes and output_nodes:
        resistance_node_pair = (starting_nodes[0], output_nodes[0])
        print(f"Auto-selected resistance node pair: {resistance_node_pair}")
    else:
        raise ValueError(f"No starting or output nodes found in input {edge_percent}% or output {end_percent}%")

    # 4) Add branch orders and hemodynamic edge weights.
    #HD note - eventually pericyte localisation should be able to be either determined by this manual method, or via loading in a segmented image of pericytes?
    #HD note - eventually add in probability of pericyte contraction?
    if starting_nodes:
        graph.assign_branch_orders(G, starting_nodes)
        poiseuille_model = hemodynamics.PoiseuilleModel(
            constriction_length=40.0,
            constriction_spacing=100.0,
        )

        G, results = poiseuille_model.set_poiseuille_weights_with_constrictions(
            G,
            diameter_by_branch_order,
        )

        print(f"Results from set_poiseuille_weights_with_constrictions: {results}")

        G, results_2 = poiseuille_model.set_poiseuille_edge_weights(
            G,
            custom_edges,
            edge_diameter=6.0,
            use_resistance=False,
        )

        print(f"Results from set_poiseuille_edge_weights: {results_2}")
        # create list of resistances of all edges
        resistances = []
        # TODO DEBUG
        # for u, v, key in G.edges(keys=True, data=True):
        #     resistances.append(G[u][v][key]['weight'])
        # print(f"Resistances of all edges: {resistances}")

    # visualize pre vtk
    visualization.visualize_edges_and_nodes(image, G, label_nodes=True, save_path=plot_dir / "pre_vtk.png")
    # 5) Export vessels/pericytes/nodes to VTK and optionally visualize in PyVista.
    # FA I have no idea if pericyte location is correct. AI did that part.
    # FA I don't fully understand how pericyte location is currently determined?
    vtk_export = visualization.graph_to_vtk(G, vtk_output_prefix)
    print("\n=== VTK Export ===")
    print(f"  Vessels:   {vtk_export['vessels_path']}")
    print(f"  Pericytes: {vtk_export['pericytes_path']}")
    print(f"  Nodes:     {vtk_export['nodes_path']}")
    print(f"  Counts: vessels={vtk_export['vessel_line_count']}, "
          f"pericytes={vtk_export['pericyte_count']}, nodes={vtk_export['node_count']}")
    if visualize_vtk:
        visualization.visualize_vtk_network(
            vtk_export["vessels_path"],
            vtk_export["pericytes_path"],
            vtk_export["nodes_path"],
            show_nodes=False,
        )

    # 6) Compute effective resistance between two selected nodes.
    conductance, node_list = hemodynamics.build_conductance_matrix_from_graph(G)
    node_to_idx = {node_id: idx for idx, node_id in enumerate(node_list)}

    if do_equiv_resistance_calculation:
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

    flow_output_nodes = list(output_nodes)
    if set_stubs_to_outlet_pressure:
        starting_node_set = set(starting_nodes)
        output_node_set = set(flow_output_nodes)
        stub_nodes = sorted(
            node_id
            for node_id, degree in G.degree()
            if degree == 1
            and node_id not in starting_node_set
            and node_id not in output_node_set
        )
        if stub_nodes:
            flow_output_nodes.extend(stub_nodes)
            print(
                "Applied outlet pressure to terminal stubs: "
                f"added {len(stub_nodes)} node(s) -> {stub_nodes}"
            )
        else:
            print("No additional terminal stubs found for outlet pressure assignment.")

    # 8) Also solve for flow throughout the network using the conductance matrix 
    # and the input and output pressures.
    flow, vtk_export = hemodynamics.solve_flow_from_conductance_matrix(
        conductance,
        node_list,
        input_p_bc,
        output_p_bc,
        starting_nodes,
        flow_output_nodes,
        vtk_export,
    )
    print("Flow through the network solved")
    print(f"Vtk file with flow data saved to: {vtk_export['vessels_path']}")

    # 9) Optional matplotlib visualization.
    if visualize_results:
        visualization.plot_node_degree_distribution(G)
        visualization.visualize_edges_and_nodes(image, G)
        # visualization.interactive_3d_graph(G)
        #HD note - need visualisation of pericyte localisations (ie based upon constriction data)
        
        if starting_nodes:
            visualization.visualize_geometry_with_branch_orders(
                image,
                G,
                group_above=8,
            )


if __name__ == "__main__":
    plot_dir = BASE_PLOT_DIR / "nerve"
    image_to_model_pipeline(plot_dir=plot_dir, set_stubs_to_outlet_pressure=True)
