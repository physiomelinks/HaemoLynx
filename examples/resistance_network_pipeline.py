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
import json
from pathlib import Path
from skan import csr
import tifffile
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# Ensure package is importable when running from repo root.
root_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


from ImageLynx import graph, hemodynamics, io, preprocessing, statistics, visualization 

# ---------------------------
# Beginner-friendly settings
# ---------------------------
INPUT_PATH = root_dir / "examples" / "images" / "Nerve_capillaries.tif"
USE_ILASTIK_SEGMENTATION = False
# INPUT_PATH is the segmented image path used when USE_ILASTIK_SEGMENTATION=False.
# Required when USE_ILASTIK_SEGMENTATION=True: path to the raw/unsegmented image.
ILASTIK_UNSEGMENTED_IMAGE_PATH = root_dir / "examples" / "images" / "Nerve_capillaries.tif"
ILASTIK_CLASSIFIER_PATH = root_dir / "examples" / "classifiers" / "nerve_classifier.ilp"
ILASTIK_EXECUTABLE = "ilastik.exe"
ILASTIK_OUTPUT_DIR = root_dir / "examples" / "outputs" / "segmentations"
# Output extension for ilastik segmentation result. Supported: ".tif", ".tiff", ".h5"
ILASTIK_OUTPUT_SUFFIX = ".tif"
USE_LARGE_VESSEL_MASKS = False
# Toggle large-vessel input mode:
# - False: use pre-segmented arteriole/venule masks from LARGE_*_MASK_PATH.
# - True: use raw arteriole/venule images and segment both with ilastik.
USE_ILASTIK_LARGE_VESSEL_SEGMENTATION = False
LARGE_VESSEL_MASK_DILATION_MICRONS = 0.0
LARGE_ARTERIOLE_MASK_PATH = root_dir / "examples" / "images" / "large_arteriole_mask.tif"
LARGE_VENULE_MASK_PATH = root_dir / "examples" / "images" / "large_venule_mask.tif"
ILASTIK_UNSEGMENTED_ARTERIOLE_IMAGE_PATH = root_dir / "examples" / "images" / "large_arteriole_mask.tif"
ILASTIK_UNSEGMENTED_VENULE_IMAGE_PATH = root_dir / "examples" / "images" / "large_venule_mask.tif"
ILASTIK_ARTERIOLE_CLASSIFIER_PATH = root_dir / "examples" / "classifiers" / "arteriole_classifier.ilp"
ILASTIK_VENULE_CLASSIFIER_PATH = root_dir / "examples" / "classifiers" / "venule_classifier.ilp"
BASE_PLOT_DIR = root_dir / "examples" / "plots" 
if not BASE_PLOT_DIR.exists():
    BASE_PLOT_DIR.mkdir(parents=True, exist_ok=True)
# STARTING NODES and OUTPUT Nodes are now calculated automatically by looking for degree 1 nodes at start or
# end of the image.
SET_INPUT_NODE_METHOD = "coordinates" # "coordinates" or "edge_percent"
SET_OUTPUT_NODE_METHOD = "degree_1_from_starting" # "coordinates" or "edge_percent"
DISTANCE_FROM_STARTING_NODE = 300.0
EDGE_PERCENT = 10.0
END_PERCENT = 10.0
# For 3D skeletons this is usually the y-axis in (z, y, x).
NODE_EDGE_AXIS = 1
STARTING_NODE_COORDINATES = [(152.0, 340.0, 527.0), (160.0, 350.0, 545.0), # top right
                             (202.0, 1303.0, 132.0), (104.0, 1321.0, 133.0), #bottom left
                             (361.0, 332.0, 120.0), (321.0, 334.0, 163.0)] #top right

OUTPUT_NODE_COORDINATES = []
STARTING_NODE_VOLUMES: list[tuple[tuple[float, float, float], tuple[float, float, float]]] = []
OUTPUT_NODE_VOLUMES: list[tuple[tuple[float, float, float], tuple[float, float, float]]] = []
STARTING_NODES: list[int] = []
OUTPUT_NODES: list[int] = []
# TODO HD note - eventually add script to run resistance measurements between every BO1 (arteriole) and every (non-arteriole) capillary node, and between every node.
# TODO automate the selection of resistance node pairs
# RESISTANCE_NODE_PAIR = (426, 509)  # (source_node_id, target_node_id)
INPUT_P_BC = 4500# Pa 
OUTPUT_P_BC = 1000 # Pa
VISUALIZE_RESULTS = True
INTERACTIVE_PLOTS = False
# When True, keep saving PNGs and also display visualization windows
# in a non-blocking way during pipeline execution.
SHOW_PLOTS_IN_IDE = True
# Control how many plots are shown interactively in the IDE while still
# saving all configured output PNGs.
# - "all": show every plot in the visualize_results block
# - "final_only": show only the final edges/nodes overlay
# - "none": do not show IDE windows (save only)
IDE_PLOT_MODE = "final_only"
# Keep matplotlib windows open at the end of the run when plotting to IDE.
HOLD_IDE_PLOTS_OPEN = True
FINAL_RENDER_MODE = "3d"  # "2d" or "3d"
VTK_export = True
STATISTICS = False
# "fast" uses bounded/approximate graph metrics to avoid long runtimes.
# "full" restores exact, potentially much slower statistics calculations.
STATISTICS_MODE = "fast"
VISUALIZE_VTK = False
VERBOSE_LOGGING = False
DO_SKELETONIZE = True
DO_GRAPH_BUILDING = True
DO_EQUIV_RESISTANCE_CALCULATION = False
MIN_BRANCH_LENGTH = 10
VTK_OUTPUT_PREFIX = root_dir / "examples" / "outputs" / "resistance_network"
SKELETON_CLOSING_RADIUS = 2
SKELETON_BRIDGE_GAP_SIZE = 3
SKELETON_MIN_BRANCH_LENGTH = 3
SKELETON_MAX_BRIDGE_DISTANCE = 4
SKELETON_COMPONENT_CONNECTIVITY = 3
GRAPH_RECONNECT_THRESHOLD = 10.0
# Keep final orphan/dangling reconnect local-only to avoid creating
# long cross-links in dense regions.
FINAL_ORPHAN_RECONNECT_THRESHOLD = 3.0
MIN_STUB_LENGTH = 10.0
CLUSTER_COLLAPSE_DISTANCE = 5.0
# Keep only connected components at or above this percentage of total
# skeleton voxels (e.g. 5.0 -> keep components >= 5% of total skeleton voxels).
SKELETON_MIN_COMPONENT_PERCENT = 0.0
# TODO these diameters etc should be automated 
#HD note - there should be a manual option, as per below, to add in in vivo diameters, and a option to read in diameters from the original image (via FWHM)
#HD note - this no longer features the ability to manually define a limited number of user determined vessels (ie endoneurial vessels), which can't be done automatically. Not relevant for alice but relevant generally.
"""Configuration defaults for diameter maps."""

# Diameter by branch order (simple scalar)
print("TODO HARVEY CHANGE THIS ALL_DIAMS_CONST BACK TO FALSE FOR ORIGINAL RUN")
ALL_DIAMS_CONST = True
DO_PERICYTE_CONSTRUCTION = False

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

CONSTRICTION_BY_BRANCH_ORDER = {
    "B01": 1.0,
}
for i in range(2, 52):
    CONSTRICTION_BY_BRANCH_ORDER[f"B{i:02d}"] = 0.8

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


def _save_graph_snapshot(
    G: nx.MultiGraph,
    image: np.ndarray,
    output_dir: Path,
    plot_dir: Path,
    image_stem: str,
    step_name: str,
) -> None:
    """Save graph artifacts after a named processing step."""
    safe_step = step_name.strip().replace(" ", "_")
    graph_snapshot_path = output_dir / f"{image_stem}_graph_after_{safe_step}.pkl"
    with graph_snapshot_path.open("wb") as f:
        pickle.dump(G, f)
    print(f"Saved graph after '{step_name}': {graph_snapshot_path}")

    plot_snapshot_path = plot_dir / f"graph_after_{safe_step}.png"
    visualization.visualize_edges_and_nodes(
        image,
        G,
        label_nodes=True,
        save_path=plot_snapshot_path,
    )
    print(f"Saved graph plot after '{step_name}': {plot_snapshot_path}")


def image_to_model_pipeline(image_path=INPUT_PATH,
                            use_ilastik_segmentation=USE_ILASTIK_SEGMENTATION,
                            ilastik_unsegmented_image_path=ILASTIK_UNSEGMENTED_IMAGE_PATH,
                            ilastik_classifier_path=ILASTIK_CLASSIFIER_PATH,
                            ilastik_executable=ILASTIK_EXECUTABLE,
                            ilastik_output_dir=ILASTIK_OUTPUT_DIR,
                            ilastik_output_suffix=ILASTIK_OUTPUT_SUFFIX,
                            use_large_vessel_masks=USE_LARGE_VESSEL_MASKS,
                            use_ilastik_large_vessel_segmentation=USE_ILASTIK_LARGE_VESSEL_SEGMENTATION,
                            large_vessel_mask_dilation_microns=LARGE_VESSEL_MASK_DILATION_MICRONS,
                            large_arteriole_mask_path=LARGE_ARTERIOLE_MASK_PATH,
                            large_venule_mask_path=LARGE_VENULE_MASK_PATH,
                            ilastik_unsegmented_arteriole_image_path=ILASTIK_UNSEGMENTED_ARTERIOLE_IMAGE_PATH,
                            ilastik_unsegmented_venule_image_path=ILASTIK_UNSEGMENTED_VENULE_IMAGE_PATH,
                            ilastik_arteriole_classifier_path=ILASTIK_ARTERIOLE_CLASSIFIER_PATH,
                            ilastik_venule_classifier_path=ILASTIK_VENULE_CLASSIFIER_PATH,
                            diameter_by_branch_order=DIAMETER_BY_BRANCH_ORDER,
                            constriction_by_branch_order=CONSTRICTION_BY_BRANCH_ORDER,
                            do_pericyte_constriction=DO_PERICYTE_CONSTRUCTION,
                            plot_dir=BASE_PLOT_DIR,
                            verbose_logging=VERBOSE_LOGGING,
                            do_skeletonize=DO_SKELETONIZE,
                            do_graph_building=DO_GRAPH_BUILDING,
                            do_equiv_resistance_calculation=DO_EQUIV_RESISTANCE_CALCULATION,
                            min_branch_length=MIN_BRANCH_LENGTH,
                            min_stub_length=MIN_STUB_LENGTH,
                            cluster_collapse_distance=CLUSTER_COLLAPSE_DISTANCE,
                            vtk_output_prefix=VTK_OUTPUT_PREFIX,
                            skeleton_closing_radius=SKELETON_CLOSING_RADIUS,
                            skeleton_bridge_gap_size=SKELETON_BRIDGE_GAP_SIZE,
                            skeleton_min_branch_length=SKELETON_MIN_BRANCH_LENGTH,
                            skeleton_max_bridge_distance=SKELETON_MAX_BRIDGE_DISTANCE,
                            skeleton_component_connectivity=SKELETON_COMPONENT_CONNECTIVITY,
                            skeleton_min_component_percent=SKELETON_MIN_COMPONENT_PERCENT,
                            graph_reconnect_threshold=GRAPH_RECONNECT_THRESHOLD,
                            final_orphan_reconnect_threshold=FINAL_ORPHAN_RECONNECT_THRESHOLD,
                            set_input_node_method=SET_INPUT_NODE_METHOD,
                            set_output_node_method=SET_OUTPUT_NODE_METHOD,
                            starting_node_coordinates=STARTING_NODE_COORDINATES,
                            output_node_coordinates=OUTPUT_NODE_COORDINATES,
                            starting_node_volumes=STARTING_NODE_VOLUMES,
                            output_node_volumes=OUTPUT_NODE_VOLUMES,
                            distance_from_starting_node=DISTANCE_FROM_STARTING_NODE,
                            edge_percent=EDGE_PERCENT, 
                            end_percent=END_PERCENT, 
                            node_edge_axis=NODE_EDGE_AXIS, 
                            starting_nodes=STARTING_NODES, 
                            output_nodes=OUTPUT_NODES, 
                            input_p_bc=INPUT_P_BC, 
                            output_p_bc=OUTPUT_P_BC, 
                            visualize_results=VISUALIZE_RESULTS, 
                            interactive_plots=INTERACTIVE_PLOTS,
                            show_plots_in_ide=SHOW_PLOTS_IN_IDE,
                            ide_plot_mode=IDE_PLOT_MODE,
                            hold_ide_plots_open=HOLD_IDE_PLOTS_OPEN,
                            final_render_mode=FINAL_RENDER_MODE,
                            visualize_vtk=VISUALIZE_VTK,
                            statistics_mode=STATISTICS_MODE) -> None:
    image_path = Path(image_path)
    if use_ilastik_segmentation:
        unsegmented_image_path = Path(ilastik_unsegmented_image_path)
        unsegmented_image_path = io.resolve_image_path_with_optional_zip(unsegmented_image_path)
        if ilastik_classifier_path is None:
            raise ValueError(
                "ilastik_classifier_path must be set when use_ilastik_segmentation=True."
            )
        ilastik_output_dir = Path(ilastik_output_dir)
        ilastik_segmented_path = ilastik_output_dir / (
            f"{unsegmented_image_path.stem}_segmented{ilastik_output_suffix}"
        )
        print(f"Running ilastik segmentation for unsegmented image: {unsegmented_image_path}")
        image_path = io.run_ilastik_headless_segmentation(
            input_image_path=unsegmented_image_path,
            classifier_path=Path(ilastik_classifier_path),
            output_path=ilastik_segmented_path,
            ilastik_executable=ilastik_executable,
        )
        print(f"Using ilastik-segmented image: {image_path}")
    else:
        print(f"Using segmented input image: {image_path}")

    image_path = io.resolve_image_path_with_optional_zip(image_path)
    # get image format from image_path
    input_format = image_path.suffix[1:].lower()
    if input_format not in ["tif", "tiff", "h5"]:
        raise ValueError(f"Invalid image format: {input_format}")
    vtk_output_prefix = Path(vtk_output_prefix)
    output_dir = vtk_output_prefix.parent
    valid_final_render_modes = {"2d", "3d"}
    if final_render_mode not in valid_final_render_modes:
        raise ValueError(
            f"Invalid final_render_mode='{final_render_mode}'. "
            f"Choose one of {sorted(valid_final_render_modes)}."
        )

    logging.basicConfig(
        level=logging.DEBUG if verbose_logging else logging.INFO,
        format="[%(levelname)s] %(message)s",
    )

    # 1) Load image and skeletonize.
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
    skeleton_path = output_dir / f"{image_path.stem}_skeleton.npy"
    voxel_meta_path = output_dir / f"{image_path.stem}_voxel_size.json"
    graph_path = output_dir / f"{image_path.stem}_graph.pkl"
    projection_path = plot_dir / "skeleton_projection.png"
    if not plot_dir.exists():
        plot_dir.mkdir(parents=True, exist_ok=True)

    if do_skeletonize:
        if input_format in {"tif", "tiff"}:
            image, skeleton, voxel_size_x, voxel_size_y, voxel_size_z = io.load_and_skeletonize_3d_tif(
                image_path,
            )
            voxel_size = (
                float(voxel_size_x),
                float(voxel_size_y),
                float(voxel_size_z),
            )
        elif input_format == "h5":
            image, skeleton, voxel_size_x, voxel_size_y, voxel_size_z = io.load_and_skeletonize_3d_h5(
                image_path,
            )
            voxel_size = (
                float(voxel_size_x),
                float(voxel_size_y),
                float(voxel_size_z),
            )
        else:
            raise ValueError("INPUT_FORMAT must be 'tif', 'tiff', or 'h5'.")
        
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
            closing_radius=skeleton_closing_radius,
            bridge_gap_size=skeleton_bridge_gap_size,
        )
        preprocessing.print_skeleton_connectivity_stats(
            "cleaned",
            skeleton,
            component_connectivity=skeleton_component_connectivity,
        )
        
        # save the skeleton
        np.save(skeleton_path, skeleton)
        voxel_meta_path.write_text(json.dumps({"voxel_size": voxel_size}))
        print(f"Saved skeleton to: {skeleton_path}")
    else:
        # load the skeleton
        skeleton = np.load(skeleton_path)
        image = tifffile.imread(image_path)
        if voxel_meta_path.exists():
            voxel_size = tuple(json.loads(voxel_meta_path.read_text())["voxel_size"])
        else:
            voxel_size = (1.0, 1.0, 1.0)
        print(f"Loaded skeleton from: {skeleton_path}")

    print("Visualizing skeleton projection...")
    visualization.visualize_skeleton(skeleton, save_path=projection_path)
    print("Skeleton projection saved.")

    if use_ilastik_large_vessel_segmentation and not use_large_vessel_masks:
        raise ValueError(
            "use_ilastik_large_vessel_segmentation=True requires "
            "use_large_vessel_masks=True."
        )

    effective_large_arteriole_mask_path = large_arteriole_mask_path
    effective_large_venule_mask_path = large_venule_mask_path
    if use_large_vessel_masks and use_ilastik_large_vessel_segmentation:
        if ilastik_unsegmented_arteriole_image_path is None:
            raise ValueError(
                "ilastik_unsegmented_arteriole_image_path must be set when "
                "use_ilastik_large_vessel_segmentation=True."
            )
        if ilastik_unsegmented_venule_image_path is None:
            raise ValueError(
                "ilastik_unsegmented_venule_image_path must be set when "
                "use_ilastik_large_vessel_segmentation=True."
            )
        if ilastik_arteriole_classifier_path is None:
            raise ValueError(
                "ilastik_arteriole_classifier_path must be set when "
                "use_ilastik_large_vessel_segmentation=True."
            )
        if ilastik_venule_classifier_path is None:
            raise ValueError(
                "ilastik_venule_classifier_path must be set when "
                "use_ilastik_large_vessel_segmentation=True."
            )

        ilastik_output_dir = Path(ilastik_output_dir)
        unsegmented_arteriole_image_path = io.resolve_image_path_with_optional_zip(
            Path(ilastik_unsegmented_arteriole_image_path)
        )
        unsegmented_venule_image_path = io.resolve_image_path_with_optional_zip(
            Path(ilastik_unsegmented_venule_image_path)
        )
        ilastik_segmented_arteriole_path = ilastik_output_dir / (
            f"{unsegmented_arteriole_image_path.stem}_segmented{ilastik_output_suffix}"
        )
        ilastik_segmented_venule_path = ilastik_output_dir / (
            f"{unsegmented_venule_image_path.stem}_segmented{ilastik_output_suffix}"
        )

        print(
            "Running ilastik segmentation for large arteriole image: "
            f"{unsegmented_arteriole_image_path}"
        )
        effective_large_arteriole_mask_path = io.run_ilastik_headless_segmentation(
            input_image_path=unsegmented_arteriole_image_path,
            classifier_path=Path(ilastik_arteriole_classifier_path),
            output_path=ilastik_segmented_arteriole_path,
            ilastik_executable=ilastik_executable,
        )
        print(
            "Running ilastik segmentation for large venule image: "
            f"{unsegmented_venule_image_path}"
        )
        effective_large_venule_mask_path = io.run_ilastik_headless_segmentation(
            input_image_path=unsegmented_venule_image_path,
            classifier_path=Path(ilastik_venule_classifier_path),
            output_path=ilastik_segmented_venule_path,
            ilastik_executable=ilastik_executable,
        )
        print(
            "Using ilastik-segmented large-vessel masks: "
            f"arteriole={effective_large_arteriole_mask_path}, "
            f"venule={effective_large_venule_mask_path}"
        )

    (
        large_arteriole_mask,
        large_venule_mask,
        large_arteriole_mask_voxel_size,
        large_venule_mask_voxel_size,
    ) = io.load_large_vessel_masks(
        enabled=use_large_vessel_masks,
        large_arteriole_mask_path=effective_large_arteriole_mask_path,
        large_venule_mask_path=effective_large_venule_mask_path,
    )
    if large_arteriole_mask is not None and large_venule_mask is not None:
        if large_arteriole_mask.shape != image.shape:
            raise ValueError(
                "large_arteriole_mask shape does not match input image shape: "
                f"{large_arteriole_mask.shape} != {image.shape}"
            )
        if large_venule_mask.shape != image.shape:
            raise ValueError(
                "large_venule_mask shape does not match input image shape: "
                f"{large_venule_mask.shape} != {image.shape}"
            )
        print(
            "Loaded large-vessel masks: "
            f"arteriole={large_arteriole_mask.shape}, "
            f"venule={large_venule_mask.shape}"
        )
        print(
            "Large-vessel mask voxel sizes (x, y, z): "
            f"arteriole={large_arteriole_mask_voxel_size}, "
            f"venule={large_venule_mask_voxel_size}"
        )
        main_voxel_size_xyz = tuple(float(v) for v in voxel_size)
        arteriole_voxel_size_xyz = tuple(float(v) for v in large_arteriole_mask_voxel_size)
        venule_voxel_size_xyz = tuple(float(v) for v in large_venule_mask_voxel_size)
        voxel_match_main_vs_arteriole = np.allclose(
            main_voxel_size_xyz,
            arteriole_voxel_size_xyz,
            rtol=0.0,
            atol=0.0,
        )
        voxel_match_main_vs_venule = np.allclose(
            main_voxel_size_xyz,
            venule_voxel_size_xyz,
            rtol=0.0,
            atol=0.0,
        )
        voxel_match_arteriole_vs_venule = np.allclose(
            arteriole_voxel_size_xyz,
            venule_voxel_size_xyz,
            rtol=0.0,
            atol=0.0,
        )
        if not (
            voxel_match_main_vs_arteriole
            and voxel_match_main_vs_venule
            and voxel_match_arteriole_vs_venule
        ):
            error_message = (
                "Voxel-size mismatch detected across input image and large-vessel masks. "
                f"main={main_voxel_size_xyz}, "
                f"arteriole={arteriole_voxel_size_xyz}, "
                f"venule={venule_voxel_size_xyz}. "
                "All three must match exactly in x, y, and z."
            )
            print(error_message)
            raise ValueError(error_message)
        print(
            "Voxel-size check passed. Arteriole and venule masks are aligned "
            "to the same physical voxel units as the main image."
        )
        if large_vessel_mask_dilation_microns > 0:
            large_arteriole_mask, large_venule_mask = (
                graph.dilate_large_vessel_masks_by_microns(
                    large_arteriole_mask=large_arteriole_mask,
                    large_venule_mask=large_venule_mask,
                    dilation_microns=large_vessel_mask_dilation_microns,
                    voxel_size_xyz=main_voxel_size_xyz,
                )
            )
            print(
                "Dilated large-vessel masks by "
                f"{float(large_vessel_mask_dilation_microns):.3f} microns."
            )
    else:
        print("Large-vessel masks disabled; skipping arteriole/venule mask loading.")

    if do_graph_building:
        # 3) Convert skeleton to graph.
        print("Building skan Skeleton object...")
        sk = csr.Skeleton(skeleton)
        print(f"skan Skeleton built: {sk.n_paths} paths")

        print("Building graph (loop detection + segment extraction)...")
        G, voxel_loops, loop_edges = graph.build_graph_segment_skan_stitched_loops(
            sk,
            skeleton,
            debug=verbose_logging,
            voxel_size=voxel_size,
            reconnect_threshold=graph_reconnect_threshold,
        )
        _save_graph_snapshot(
            G, image, output_dir, plot_dir, image_path.stem,
            "build_graph_segment_skan_stitched_loops",
        )
        visualization.visualize_edges_and_nodes(image, G, label_nodes=True, save_path=plot_dir / "build_graph_segment_skan_stitched_loops.png")
        G = graph.reconnect_secondary_loop_edges(
            G,
            skeleton,
            voxel_size=voxel_size,
            debug=verbose_logging,
        )
        _save_graph_snapshot(
            G, image, output_dir, plot_dir, image_path.stem,
            "reconnect_secondary_loop_edges",
        )
        visualization.visualize_edges_and_nodes(image, G, label_nodes=True, save_path=plot_dir / "reconnect_secondary_loop_edges.png")
        
        G, _ = graph.optimise_graph_topology_fixed(
            G,
            voxel_loops,
            loop_edges,
            skeleton_data=skeleton,
            debug=verbose_logging,
            reconnect_threshold=graph_reconnect_threshold,
        )
        _save_graph_snapshot(
            G, image, output_dir, plot_dir, image_path.stem,
            "optimise_graph_topology_fixed",
        )
        visualization.visualize_edges_and_nodes(image, G, label_nodes=True, save_path=plot_dir / "optimise_graph_topology_fixed.png")
        degree2_pass1_max_degree = 4
        degree2_pass2_max_degree = 8
        G = graph.smart_multigraph_degree2_removal(
            G,
            skeleton,
            max_degree=degree2_pass1_max_degree,
            debug=verbose_logging,
        )
        _save_graph_snapshot(
            G, image, output_dir, plot_dir, image_path.stem,
            "smart_multigraph_degree2_removal_pass1",
        )
        visualization.visualize_edges_and_nodes(image, G, label_nodes=True, save_path=plot_dir / "smart_multigraph_degree2_removal.png")
        degree2_diag = graph.diagnose_degree2_nodes(
            G, max_degree=degree2_pass1_max_degree
        )
        print(graph.format_degree2_diagnostics_report(degree2_diag))

        G = graph.collapse_node_clusters(
            G,
            distance_threshold=cluster_collapse_distance,
            debug=verbose_logging,
        )
        _save_graph_snapshot(
            G, image, output_dir, plot_dir, image_path.stem,
            "collapse_node_clusters",
        )
        visualization.visualize_edges_and_nodes(image, G, label_nodes=True, save_path=plot_dir / "collapse_node_clusters.png")

        # Collapsing clusters can create new degree-2 pass-through nodes;
        # run a second degree-2 cleanup pass with a higher threshold since
        # remaining degree-2 nodes typically neighbour high-degree junctions.
        G = graph.smart_multigraph_degree2_removal(
            G,
            skeleton,
            max_degree=degree2_pass2_max_degree,
            debug=verbose_logging,
        )
        _save_graph_snapshot(
            G, image, output_dir, plot_dir, image_path.stem,
            "smart_multigraph_degree2_removal_post_collapse",
        )
        visualization.visualize_edges_and_nodes(image, G, label_nodes=True, save_path=plot_dir / "smart_multigraph_degree2_removal_post_collapse.png")

        G = graph.prune_vascular_stubs(G, debug=verbose_logging, min_stub_length=min_stub_length)
        _save_graph_snapshot(
            G, image, output_dir, plot_dir, image_path.stem,
            "prune_vascular_stubs",
        )
        visualization.visualize_edges_and_nodes(image, G, label_nodes=True, save_path=plot_dir / "prune_vascular_stubs.png")
        degree2_diag = graph.diagnose_degree2_nodes(
            G, max_degree=degree2_pass2_max_degree
        )
        print(graph.format_degree2_diagnostics_report(degree2_diag))

        G = graph.smart_multigraph_degree2_removal(
            G,
            skeleton,
            max_degree=degree2_pass2_max_degree,
            debug=verbose_logging,
        )
        _save_graph_snapshot(
            G, image, output_dir, plot_dir, image_path.stem,
            "smart_multigraph_degree2_removal_post_prune",
        )
        visualization.visualize_edges_and_nodes(
            image,
            G,
            label_nodes=True,
            save_path=plot_dir / "smart_multigraph_degree2_removal_post_prune.png",
        )
        degree2_diag = graph.diagnose_degree2_nodes(
            G, max_degree=degree2_pass2_max_degree
        )
        print(graph.format_degree2_diagnostics_report(degree2_diag))

        # remove any nodes that are connected to themselves with no nodes in between
        G = graph.remove_edges_for_self_connected_nodes(G)
        _save_graph_snapshot(
            G, image, output_dir, plot_dir, image_path.stem,
            "remove_edges_for_self_connected_nodes",
        )

        # Final topology repair:
        # 1) reconnect remaining orphan/dangling nodes only if a skeleton path
        #    validates the link, then
        # 2) remove any new degree-2 pass-through nodes that remain.
        G = graph.reconnect_orphan_and_dangling_nodes(
            G,
            skeleton_data=skeleton,
            reconnect_threshold=final_orphan_reconnect_threshold,
            include_degree1=True,
            max_new_edges_per_node=1,
            validate_reconnections=True,
            debug=verbose_logging,
        )
        _save_graph_snapshot(
            G, image, output_dir, plot_dir, image_path.stem,
            "reconnect_orphan_and_dangling_nodes",
        )
        visualization.visualize_edges_and_nodes(
            image,
            G,
            label_nodes=True,
            save_path=plot_dir / "reconnect_orphan_and_dangling_nodes.png",
        )

        G = graph.smart_multigraph_degree2_removal(
            G,
            skeleton,
            max_degree=4,
            debug=verbose_logging,
        )
        _save_graph_snapshot(
            G, image, output_dir, plot_dir, image_path.stem,
            "smart_multigraph_degree2_removal_post_orphan_reconnect",
        )
        visualization.visualize_edges_and_nodes(
            image,
            G,
            label_nodes=True,
            save_path=plot_dir / "smart_multigraph_degree2_removal_post_orphan_reconnect.png",
        )
        degree2_diag = graph.diagnose_degree2_nodes(
            G, max_degree=degree2_pass2_max_degree
        )
        print(graph.format_degree2_diagnostics_report(degree2_diag))

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

    # Store physical voxel-unit metadata used for skeleton/graph geometry and mask alignment.
    G.graph["image_voxel_size_xyz"] = tuple(float(v) for v in voxel_size)
    if large_arteriole_mask is not None and large_venule_mask is not None:
        G.graph["large_arteriole_mask_voxel_size_xyz"] = tuple(
            float(v) for v in large_arteriole_mask_voxel_size
        )
        G.graph["large_venule_mask_voxel_size_xyz"] = tuple(
            float(v) for v in large_venule_mask_voxel_size
        )
    
    # Visualize final graph used for boundary-node verification.
    if final_render_mode == "3d":
        final_graph_3d_path = plot_dir / "final_graph_3d.html"
        visualization.visualize_3d_plotly(
            G,
            title="Final Graph (Interactive 3D)",
            save_html_path=str(final_graph_3d_path),
            show=show_plots_in_ide or interactive_plots,
        )
        print(f"Saved interactive 3D final graph to: {final_graph_3d_path}")
    else:
        visualization.visualize_edges_and_nodes(
            image,
            G,
            label_nodes=False,
            save_path=plot_dir / "final_graph.png",
            show_coordinates_degree_1=True,
        )

    starting_nodes[:] = []
    output_nodes[:] = []
    start_nodes = graph.select_boundary_nodes_by_method(
        G,
        image.shape,
        method=set_input_node_method,
        node_role="input",
        coordinates=starting_node_coordinates,
        volume_boxes=starting_node_volumes,
        edge_percent=edge_percent,
        end_percent=end_percent,
        axis=node_edge_axis,
    )
    out_nodes = graph.select_boundary_nodes_by_method(
        G,
        image.shape,
        method=set_output_node_method,
        node_role="output",
        coordinates=output_node_coordinates,
        volume_boxes=output_node_volumes,
        edge_percent=edge_percent,
        end_percent=end_percent,
        axis=node_edge_axis,
        exclude_nodes=start_nodes,
        starting_nodes_for_distance=start_nodes,
        distance_from_starting_node=distance_from_starting_node,
    )
    starting_nodes.extend(start_nodes)
    output_nodes.extend(out_nodes)
    print(
        f"Selected {len(starting_nodes)} STARTING_NODES using "
        f"method='{set_input_node_method}' and {len(output_nodes)} OUTPUT_NODES "
        f"using method='{set_output_node_method}'."
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
        if do_pericyte_constriction:
            diameter_by_branch_order_enhanced = {}
            for branch_order, diameter in diameter_by_branch_order.items():
                diameter_by_branch_order_enhanced[branch_order] = {
                    "d1": diameter,
                    "d2": diameter * constriction_by_branch_order[branch_order],
                }

            G, results = poiseuille_model.set_poiseuille_weights_with_constrictions(
                G,
                diameter_by_branch_order_enhanced,
            )
        else:
            G, results = poiseuille_model.set_poiseuille_weights(
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
        conductances = []
        # TODO DEBUG
        for u, v, key in G.edges(keys=True):
            conductance = G[u][v][key]['weight']
            # print(f"Conductance of edge ({u}, {v}, {key}): {conductance}")
            conductances.append(conductance)

        # print(f"Conductances of all edges: {conductances}")

    # 5) Export vessels/pericytes/nodes to VTK and optionally visualize in PyVista.
    # FA I have no idea if pericyte location is correct. AI did that part.
    # FA I don't fully understand how pericyte location is currently determined?
    if VTK_export:
        vtk_export = visualization.graph_to_vtk(G, vtk_output_prefix)
        print("\n=== VTK Export ===")
        print(f"  Vessels:   {vtk_export['vessels_path']}")
        print(f"  Pericytes: {vtk_export['pericytes_path']}")
        print(f"  Nodes:     {vtk_export['nodes_path']}")
        print(f"  Counts: vessels={vtk_export['vessel_line_count']}, "
          f"pericytes={vtk_export['pericyte_count']}, nodes={vtk_export['node_count']}")
    if visualize_vtk and VTK_export:
        visualization.visualize_vtk_network(
            vtk_export["vessels_path"],
            vtk_export["pericytes_path"],
            vtk_export["nodes_path"],
            show_nodes=False,
        )
    if visualize_vtk and not VTK_export:
        print("VTK visualization requested but VTK export is disabled. Set VTK_export=True to enable.")
    else:
        print("VTK visualization skipped.") 
    # 6) Compute effective resistance between two selected nodes.
    conductance, node_list = hemodynamics.build_conductance_matrix_from_graph(G)
    node_to_idx = {node_id: idx for idx, node_id in enumerate(node_list)}
    print(f"Conductance matrix built with shape {conductance.shape} and node_list length {len(node_list)}.")
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
    print("\nComputing vessel statistics...")
    if STATISTICS:
        valid_statistics_modes = {"fast", "full"}
        if statistics_mode not in valid_statistics_modes:
            raise ValueError(
                f"Invalid statistics_mode='{statistics_mode}'. "
                f"Choose one of {sorted(valid_statistics_modes)}."
            )
        node_positions = nx.get_node_attributes(G, "pos")
        stats = statistics.compute_comprehensive_vessel_statistics(
            G,
            node_positions=node_positions,
            image_dimensions=image.shape,
            statistics_mode=statistics_mode,
        )

        print("\n=== Statistics ===")
        for key, value in stats.items():
            print(f"  {key}: {value}")

        weighted_measurements = statistics.compute_betweenness_and_community_measurements(G)
        print("\n=== Weighted Betweenness and Communities ===")
        for model_name, model_results in weighted_measurements.items():
            print(f"  [{model_name}]")
            for metric_name, metric_values in model_results.items():
                print(f"    {metric_name}: {metric_values}")

        inv_weight_path = output_dir / f"{image_path.stem}_betweenness_communities_inverse_weight.json"
        inv_weight_path.write_text(
            json.dumps(weighted_measurements["inverse_edge_weight"], indent=2)
        )
        length_path = output_dir / f"{image_path.stem}_betweenness_communities_edge_length.json"
        length_path.write_text(
            json.dumps(weighted_measurements["edge_length"], indent=2)
        )
        print(f"Saved inverse-weight stats to: {inv_weight_path}")
        print(f"Saved edge-length stats to: {length_path}")
    else:
        print("Vessel statistics skipped.")

    # 8) Also solve for flow throughout the network using the conductance matrix 
    # and the input and output pressures.
    print("\nSolving flow through the network...")
    flow, vtk_export = hemodynamics.solve_flow_from_conductance_matrix(
        conductance,
        node_list,
        input_p_bc,
        output_p_bc,
        starting_nodes,
        output_nodes,
        vtk_export,
    )
    print("Flow through the network solved")
    print(f"Vtk file with flow data saved to: {vtk_export['vessels_path']}")

    # 9) Optional matplotlib visualization.
    if visualize_results:
        print("\nGenerating visualizations...")
        valid_plot_modes = {"all", "final_only", "none"}
        if ide_plot_mode not in valid_plot_modes:
            raise ValueError(
                f"Invalid ide_plot_mode='{ide_plot_mode}'. "
                f"Choose one of {sorted(valid_plot_modes)}."
            )
        show_any_ide_plot = show_plots_in_ide and ide_plot_mode != "none"
        show_degree_plot = show_plots_in_ide and ide_plot_mode == "all"
        show_overlay_plot = show_any_ide_plot and final_render_mode == "2d"
        show_3d_plot = show_any_ide_plot and final_render_mode == "3d"
        show_branch_order_plot = show_plots_in_ide and ide_plot_mode == "all"
        visualization.plot_node_degree_distribution(
            G,
            save_path=None if interactive_plots else plot_dir / "node_degree_distribution.png",
            show=interactive_plots or show_degree_plot,
            show_after_save=show_degree_plot and not interactive_plots,
        )
        if final_render_mode == "3d":
            overlay_3d_path = None if interactive_plots else plot_dir / "edges_and_nodes_overlay_3d.html"
            visualization.visualize_3d_plotly(
                G,
                title="Edges and Nodes Overlay (Interactive 3D)",
                save_html_path=str(overlay_3d_path) if overlay_3d_path else None,
                show=interactive_plots or show_3d_plot,
            )
            if overlay_3d_path is not None:
                print(f"Saved interactive 3D overlay to: {overlay_3d_path}")
        else:
            visualization.visualize_edges_and_nodes(
                image,
                G,
                save_path=None if interactive_plots else plot_dir / "edges_and_nodes_overlay.png",
                show=interactive_plots or show_overlay_plot,
                show_after_save=show_overlay_plot and not interactive_plots,
            )
        #HD note - need visualisation of pericyte localisations (ie based upon constriction data)
        
        if starting_nodes:
            visualization.visualize_geometry_with_branch_orders(
                image,
                G,
                group_above=8,
                save_path=None if interactive_plots else plot_dir / "geometry_with_branch_orders.png",
                show=interactive_plots or show_branch_order_plot,
                show_after_save=show_branch_order_plot and not interactive_plots,
            )
        if (
            hold_ide_plots_open
            and show_any_ide_plot
            and not interactive_plots
            and plt.get_fignums()
        ):
            print("Holding plot windows open. Close them to finish the script.")
            plt.show(block=True)
    else:
        print("Matplotlib visualizations skipped.")


if __name__ == "__main__":
    plot_dir = BASE_PLOT_DIR / "nerve"
    image_to_model_pipeline(plot_dir=plot_dir)
