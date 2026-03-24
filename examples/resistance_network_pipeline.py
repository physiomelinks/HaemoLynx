#Summarise by BO in statistics
#Mean distance of object (classifier) to each capillary type and BO
#Overall list of every vessel and its properties

#!/usr/bin/env python3
"""Refactored full pipeline example using ImageLynx package."""
import logging
import sys
import pickle
import json
from pathlib import Path
from typing import Optional
from skan import csr
import tifffile
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# Ensure package is importable when running from repo root.
root_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


from ImageLynx import graph, haemodynamics, io, preprocessing, statistics, visualization
from ImageLynx.haemodynamics import pericyte_comparison as pericyte_comparison_haemodynamics
from ImageLynx.haemodynamics import pericyte_mask as pericyte_mask_haemodynamics

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

#Do you want to use large vessel masks to determine input and output nodes?
USE_LARGE_VESSEL_MASKS = False
# Toggle large-vessel input mode:
# - False: use pre-segmented arteriole/venule masks from LARGE_*_MASK_PATH.
# - True: use raw arteriole and artery/venule and vein images and segment both with ilastik.
# If true, these vessels should be the largest vessels, that will not be modelled in the graph.
USE_ILASTIK_LARGE_VESSEL_SEGMENTATION = False
LARGE_VESSEL_MASK_DILATION_MICRONS = 0.0

LARGE_ARTERIOLE_MASK_PATH = root_dir / "examples" / "images" / "large_arteriole_mask.tif"
LARGE_VENULE_MASK_PATH = root_dir / "examples" / "images" / "large_venule_mask.tif"
ILASTIK_UNSEGMENTED_ARTERIOLE_IMAGE_PATH = root_dir / "examples" / "images" / "large_arteriole_mask.tif"
ILASTIK_UNSEGMENTED_VENULE_IMAGE_PATH = root_dir / "examples" / "images" / "large_venule_mask.tif"
ILASTIK_ARTERIOLE_CLASSIFIER_PATH = root_dir / "examples" / "classifiers" / "arteriole_classifier.ilp"
ILASTIK_VENULE_CLASSIFIER_PATH = root_dir / "examples" / "classifiers" / "venule_classifier.ilp"

# Small-vessel masks can be used to auto-detect arteriole/venule boundary nodes
# Do this to automatically determine terminal arteriole/venule-to-capillary transition points for automated hierarchical Art/Ven/Capillary branch ordering.
USE_SMALL_VESSEL_MASKS_FOR_BOUNDARY_ASSIGNMENT = False
USE_ILASTIK_SMALL_VESSEL_SEGMENTATION = False
SMALL_VESSEL_MASK_MIN_OVERLAP_FRACTION = 0.5
# When small-vessel masks assign boundaries, save rotatable Plotly HTML (requires plotly).
WRITE_SMALL_VESSEL_BOUNDARY_LABELLING_3D_HTML = True
SMALL_ARTERIOLE_MASK_PATH = root_dir / "examples" / "images" / "small_arteriole_mask.tif"
SMALL_VENULE_MASK_PATH = root_dir / "examples" / "images" / "small_venule_mask.tif"
ILASTIK_UNSEGMENTED_SMALL_ARTERIOLE_IMAGE_PATH = root_dir / "examples" / "images" / "small_arteriole_mask.tif"
ILASTIK_UNSEGMENTED_SMALL_VENULE_IMAGE_PATH = root_dir / "examples" / "images" / "small_venule_mask.tif"
ILASTIK_SMALL_ARTERIOLE_CLASSIFIER_PATH = ILASTIK_ARTERIOLE_CLASSIFIER_PATH
ILASTIK_SMALL_VENULE_CLASSIFIER_PATH = ILASTIK_VENULE_CLASSIFIER_PATH
BASE_PLOT_DIR = root_dir / "examples" / "plots" 
if not BASE_PLOT_DIR.exists():
    BASE_PLOT_DIR.mkdir(parents=True, exist_ok=True)
# Boundary-node assignment modes:
# - Manual: set AUTOMATED_VESSEL_ASSIGNMENT=False and supply STARTING_NODE_COORDINATES and OUTPUT_NODE_COORDINATES.
# - Automated: set AUTOMATED_VESSEL_ASSIGNMENT=True and use large-vessel masks.
AUTOMATED_VESSEL_ASSIGNMENT = False
# Manual node-selection methods (when AUTOMATED_VESSEL_ASSIGNMENT=False):
# - "coordinates": choose nearest degree-1 node to each provided point
# - "volume": choose all degree-1 nodes inside provided volume boxes
STARTING_NODE_SELECTION_METHOD = "coordinates"
OUTPUT_NODE_SELECTION_METHOD = "coordinates"
ARTERIOLE_BOUNDARY_SELECTION_METHOD = "coordinates"
VENULE_BOUNDARY_SELECTION_METHOD = "coordinates"
STARTING_NODE_COORDINATES = [(152.0, 340.0, 527.0), (160.0, 350.0, 545.0), # top right
                             (202.0, 1303.0, 132.0), (104.0, 1321.0, 133.0), #bottom left
                             (361.0, 332.0, 120.0), (321.0, 334.0, 163.0)] #top right

OUTPUT_NODE_COORDINATES = []
ARTERIOLE_BOUNDARY_NODE_COORDINATES = []
VENULE_BOUNDARY_NODE_COORDINATES = []

#Assign by volume boxes
# - Volume boxes: set USE_VOLUME_BOXES=True and supply STARTING_NODE_VOLUMES and OUTPUT_NODE_VOLUMES.
# - Coordinates: set USE_VOLUME_BOXES=False and supply STARTING_NODE_COORDINATES and OUTPUT_NODE_COORDINATES.
USE_VOLUME_BOXES = False
STARTING_NODE_VOLUMES: list[tuple[tuple[float, float, float], tuple[float, float, float]]] = []
OUTPUT_NODE_VOLUMES: list[tuple[tuple[float, float, float], tuple[float, float, float]]] = []
ARTERIOLE_BOUNDARY_NODE_VOLUMES: list[tuple[tuple[float, float, float], tuple[float, float, float]]] = []
VENULE_BOUNDARY_NODE_VOLUMES: list[tuple[tuple[float, float, float], tuple[float, float, float]]] = []
STARTING_NODES: list[int] = []
OUTPUT_NODES: list[int] = []
ARTERIOLE_BOUNDARY_NODES: list[int] = []
VENULE_BOUNDARY_NODES: list[int] = []
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
# Toggle cell-mask to vessel 3D distance measurement.
MEASUREMENT_3D_TO_CELL_MASK = False
CELL_MASK_PATH: Optional[Path] = None
CELL_MASK_H5_DATASET_NAME: Optional[str] = None
# Optional explicit vessel mask. If omitted, vessel volume is rasterized from graph
# edges (same strategy used by automated FWHM branch-label volume generation).
MEASUREMENT_3D_VESSEL_MASK_PATH: Optional[Path] = None
MEASUREMENT_3D_VESSEL_MASK_H5_DATASET_NAME: Optional[str] = None
# Optional reference image used only to define shape when rasterizing vessel volume
# from graph edges (for example, the same raw stack used by FWHM measurement).
MEASUREMENT_3D_REFERENCE_IMAGE_PATH: Optional[Path] = None
MEASUREMENT_3D_REFERENCE_H5_DATASET_NAME: Optional[str] = None
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
# -----------------------------------------------------------------------------
# Vessel diameter for Poiseuille weights (manual branch-order vs automated FWHM)
# -----------------------------------------------------------------------------
# Manual mode (default): USE_FWHM_EDGE_DIAMETERS=False. Diameters come from
# DIAMETER_BY_BRANCH_ORDER (built from ALL_DIAMS_CONST, DEFAULT_DIAMETER, and
# MANUAL_*_DIAMETER_BY_BRANCH_ORDER). Used by PoiseuilleModel.set_poiseuille_weights.
#
# Automated mode: USE_FWHM_EDGE_DIAMETERS=True. Requires FWHM_RAW_TIFF_PATH to a
# single-channel raw fluorescence TIFF aligned with the graph. Per-edge
# ``fwhm_diameter_um`` comes from haemodynamics.automated (Gaussian transverse fit).
# With DO_PERICYTE_CONSTRUCTION=False, plain Poiseuille uses that per edge (branch-order
# fallback if no fit). With DO_PERICYTE_CONSTRUCTION=True, integrated constriction
# uses FWHM as passive d1 and d2 = d1 * CONSTRICTION_BY_BRANCH_ORDER (same multipliers
# as manual mode), with scalar DIAMETER_BY_BRANCH_ORDER as fallback d1 when FWHM
# is missing on an edge.
#
# Optional: set FWHM_RAW_TIFF_PATH = ILASTIK_UNSEGMENTED_IMAGE_PATH when the raw
# stack is the same file as your unsegmented input (see top of this file).
#HD note - manual overrides for in vivo diameters; endoneurial custom vessels not in graph.
# -----------------------------------------------------------------------------

print("TODO HARVEY CHANGE THIS ALL_DIAMS_CONST BACK TO FALSE FOR ORIGINAL RUN")
ALL_DIAMS_CONST = True
DO_PERICYTE_CONSTRUCTION = False
# Optional mask-driven pericyte mode:
# - False: keep existing artificial periodic constriction placement.
# - True: use pericyte mask connected-component centroids as constriction centers.
USE_PERICYTE_MASK_CONSTRICTION = False
PERICYTE_MASK_PATH: Optional[Path] = None
PERICYTE_MASK_H5_DATASET_NAME: Optional[str] = None
# Optional probabilistic constriction:
# Example: probability=0.8 means ~80% of pericytes are active per run.
USE_PROBABILISTIC_PERICYTE_CONSTRICTION = False
PERICYTE_CONSTRICTION_PROBABILITY = 0.8
RUN_PERICYTE_RESISTANCE_COMPARISON = False
PERICYTE_COMPARISON_BASELINE_SCALE = 1.0
PERICYTE_COMPARISON_CONSTRICTED_SCALE = 0.8

MAX_BRANCH_ORDER = 51
DEFAULT_DIAMETER = 4.0

MANUAL_CAPILLARY_DIAMETER_BY_BRANCH_ORDER = {
    "B01": 6.2,
    "B02": 4.0,
    "B03": 5.0,
    "B04": 5.0,
}
MANUAL_ARTERIOLE_DIAMETER_BY_BRANCH_ORDER = {}
MANUAL_VENULE_DIAMETER_BY_BRANCH_ORDER = {}

DIAMETER_BY_BRANCH_ORDER = haemodynamics.build_diameter_by_branch_order(
    all_diams_const=ALL_DIAMS_CONST,
    max_branch_order=MAX_BRANCH_ORDER,
    default_diameter=DEFAULT_DIAMETER,
    manual_capillary_diameter_by_branch_order=MANUAL_CAPILLARY_DIAMETER_BY_BRANCH_ORDER,
    manual_arteriole_diameter_by_branch_order=MANUAL_ARTERIOLE_DIAMETER_BY_BRANCH_ORDER,
    manual_venule_diameter_by_branch_order=MANUAL_VENULE_DIAMETER_BY_BRANCH_ORDER,
)

CONSTRICTION_BY_BRANCH_ORDER = {
    "B01": 1.0,
}
for i in range(2, MAX_BRANCH_ORDER + 1):
    CONSTRICTION_BY_BRANCH_ORDER[f"B{i:02d}"] = 0.8
CONSTRICTION_BY_BRANCH_ORDER["Art1"] = 1.0
CONSTRICTION_BY_BRANCH_ORDER["Ven1"] = 1.0
for i in range(2, MAX_BRANCH_ORDER + 1):
    CONSTRICTION_BY_BRANCH_ORDER[f"Art{i}"] = 0.8
    CONSTRICTION_BY_BRANCH_ORDER[f"Ven{i}"] = 0.8

# --- Toggle: False = manual branch-order diameters only; True = run FWHM pipeline ---
USE_FWHM_EDGE_DIAMETERS = False
# Raw single-channel 3D TIFF (required if USE_FWHM_EDGE_DIAMETERS is True).
FWHM_RAW_TIFF_PATH: Optional[Path] = None
# Interval along the vessel centerline between transverse profiles (µm).
FWHM_SAMPLE_SPACING_ALONG_EDGE_UM = 2.0
# Sample spacing along each transverse line profile / line resolution (µm).
FWHM_TRANSVERSE_PROFILE_STEP_UM = 0.25
# Initial maximum half-length of the transverse line on each side of the center (µm);
# may grow with measured width (see FWHM_MIN_TOTAL_EXTENT_MULTIPLIER in automated.py).
FWHM_TRANSVERSE_HALF_EXTENT_UM = 6.0
FWHM_DIAMETER_GUESS_UM = None
FWHM_MIN_TOTAL_EXTENT_MULTIPLIER = 3.0
FWHM_BACKGROUND_LABEL = 0
FWHM_JUNCTION_LABEL = -1
# If False (default), transverse rays stop at junction-labelled voxels to avoid crossing
# into neighbouring branches and over-estimating diameter near bifurcations.
FWHM_ALLOW_JUNCTION_CROSSING = False
# Transverse profile baseline for Gaussian fit: "wings" uses outer medians (less shoulder
# bias from neighbours); "percentile" uses global 10th percentile (legacy).
FWHM_PROFILE_BASELINE_MODE = "wings"
FWHM_PROFILE_BASELINE_WING_FRACTION = 0.2
# If True, keep the fitted baseline near the wing/percentile anchor (stricter; try if
# shoulders still bias the optimiser).
FWHM_CONSTRAIN_FITTED_BASELINE = False
FWHM_BASELINE_CONSTRAINT_HALF_WIDTH_PTP = 0.35
# Clip each transverse profile to the central vessel lobe so a second peak from a
# neighbouring branch does not inflate diameter.
FWHM_CLIP_PROFILE_TO_SINGLE_VESSEL = True
FWHM_CLIP_MIN_DROP_FRACTION_OF_CENTER = 0.35
FWHM_CLIP_RE_RISE_FRACTION_OF_CENTER = 0.08
# Exclude samples this close (µm) to bifurcation endpoints (degree > 1).
FWHM_BRANCH_ENDPOINT_EXCLUSION_UM = 10.0
# Auto-detected junction-proximity exclusion (µm) using rasterized junction voxels.
FWHM_JUNCTION_PROXIMITY_EXCLUSION_UM = 10.0
# Prevent profile rays from re-entering distant parts of the same edge (zig-zag guard).
FWHM_ENFORCE_SAME_EDGE_LOCALITY = True
FWHM_SAME_EDGE_ARC_WINDOW_UM = 3.0
FWHM_SAME_EDGE_ARC_WINDOW_MULTIPLIER = 1.0
FWHM_SAME_EDGE_ARC_WINDOW_MIN_UM = 1.0
FWHM_CAP_HALF_EXTENT_BY_NONLOCAL_SAME_EDGE_DISTANCE = True
FWHM_NONLOCAL_SAME_EDGE_ARC_SEPARATION_UM = 6.0
FWHM_NONLOCAL_SAME_EDGE_HALF_EXTENT_FACTOR = 0.45
FWHM_REJECT_SAMPLES_WITH_CENTER_OFFSET = True
FWHM_MAX_FIT_CENTER_OFFSET_UM = 1.5
FWHM_REJECT_SAMPLES_WITH_LOW_FIT_R2 = True
FWHM_MIN_FIT_R2 = 0.85

# These are vesses that constrict differently (e.g. endoneurial vessels).
custom_edges= [
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
                            use_small_vessel_masks_for_boundary_assignment=USE_SMALL_VESSEL_MASKS_FOR_BOUNDARY_ASSIGNMENT,
                            use_ilastik_small_vessel_segmentation=USE_ILASTIK_SMALL_VESSEL_SEGMENTATION,
                            small_vessel_mask_min_overlap_fraction=SMALL_VESSEL_MASK_MIN_OVERLAP_FRACTION,
                            write_small_vessel_boundary_labelling_3d_html=WRITE_SMALL_VESSEL_BOUNDARY_LABELLING_3D_HTML,
                            automated_vessel_assignment=AUTOMATED_VESSEL_ASSIGNMENT,
                            large_arteriole_mask_path=LARGE_ARTERIOLE_MASK_PATH,
                            large_venule_mask_path=LARGE_VENULE_MASK_PATH,
                            small_arteriole_mask_path=SMALL_ARTERIOLE_MASK_PATH,
                            small_venule_mask_path=SMALL_VENULE_MASK_PATH,
                            ilastik_unsegmented_arteriole_image_path=ILASTIK_UNSEGMENTED_ARTERIOLE_IMAGE_PATH,
                            ilastik_unsegmented_venule_image_path=ILASTIK_UNSEGMENTED_VENULE_IMAGE_PATH,
                            ilastik_arteriole_classifier_path=ILASTIK_ARTERIOLE_CLASSIFIER_PATH,
                            ilastik_venule_classifier_path=ILASTIK_VENULE_CLASSIFIER_PATH,
                            ilastik_unsegmented_small_arteriole_image_path=ILASTIK_UNSEGMENTED_SMALL_ARTERIOLE_IMAGE_PATH,
                            ilastik_unsegmented_small_venule_image_path=ILASTIK_UNSEGMENTED_SMALL_VENULE_IMAGE_PATH,
                            ilastik_small_arteriole_classifier_path=ILASTIK_SMALL_ARTERIOLE_CLASSIFIER_PATH,
                            ilastik_small_venule_classifier_path=ILASTIK_SMALL_VENULE_CLASSIFIER_PATH,
                            diameter_by_branch_order=DIAMETER_BY_BRANCH_ORDER,
                            constriction_by_branch_order=CONSTRICTION_BY_BRANCH_ORDER,
                            do_pericyte_constriction=DO_PERICYTE_CONSTRUCTION,
                            use_pericyte_mask_constriction=USE_PERICYTE_MASK_CONSTRICTION,
                            pericyte_mask_path=PERICYTE_MASK_PATH,
                            pericyte_mask_h5_dataset_name=PERICYTE_MASK_H5_DATASET_NAME,
                            use_probabilistic_pericyte_constriction=USE_PROBABILISTIC_PERICYTE_CONSTRICTION,
                            pericyte_constriction_probability=PERICYTE_CONSTRICTION_PROBABILITY,
                            run_pericyte_resistance_comparison=RUN_PERICYTE_RESISTANCE_COMPARISON,
                            pericyte_comparison_baseline_scale=PERICYTE_COMPARISON_BASELINE_SCALE,
                            pericyte_comparison_constricted_scale=PERICYTE_COMPARISON_CONSTRICTED_SCALE,
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
                            starting_node_selection_method=STARTING_NODE_SELECTION_METHOD,
                            output_node_selection_method=OUTPUT_NODE_SELECTION_METHOD,
                            arteriole_boundary_selection_method=ARTERIOLE_BOUNDARY_SELECTION_METHOD,
                            venule_boundary_selection_method=VENULE_BOUNDARY_SELECTION_METHOD,
                            starting_node_coordinates=STARTING_NODE_COORDINATES,
                            output_node_coordinates=OUTPUT_NODE_COORDINATES,
                            arteriole_boundary_node_coordinates=ARTERIOLE_BOUNDARY_NODE_COORDINATES,
                            venule_boundary_node_coordinates=VENULE_BOUNDARY_NODE_COORDINATES,
                            starting_node_volumes=STARTING_NODE_VOLUMES,
                            output_node_volumes=OUTPUT_NODE_VOLUMES,
                            arteriole_boundary_node_volumes=ARTERIOLE_BOUNDARY_NODE_VOLUMES,
                            venule_boundary_node_volumes=VENULE_BOUNDARY_NODE_VOLUMES,
                            starting_nodes=STARTING_NODES, 
                            output_nodes=OUTPUT_NODES, 
                            arteriole_boundary_nodes=ARTERIOLE_BOUNDARY_NODES,
                            venule_boundary_nodes=VENULE_BOUNDARY_NODES,
                            input_p_bc=INPUT_P_BC, 
                            output_p_bc=OUTPUT_P_BC, 
                            visualize_results=VISUALIZE_RESULTS, 
                            interactive_plots=INTERACTIVE_PLOTS,
                            show_plots_in_ide=SHOW_PLOTS_IN_IDE,
                            ide_plot_mode=IDE_PLOT_MODE,
                            hold_ide_plots_open=HOLD_IDE_PLOTS_OPEN,
                            final_render_mode=FINAL_RENDER_MODE,
                            visualize_vtk=VISUALIZE_VTK,
                            measurement_3d_to_cell_mask=MEASUREMENT_3D_TO_CELL_MASK,
                            cell_mask_path=CELL_MASK_PATH,
                            cell_mask_h5_dataset_name=CELL_MASK_H5_DATASET_NAME,
                            measurement_3d_vessel_mask_path=MEASUREMENT_3D_VESSEL_MASK_PATH,
                            measurement_3d_vessel_mask_h5_dataset_name=MEASUREMENT_3D_VESSEL_MASK_H5_DATASET_NAME,
                            measurement_3d_reference_image_path=MEASUREMENT_3D_REFERENCE_IMAGE_PATH,
                            measurement_3d_reference_h5_dataset_name=MEASUREMENT_3D_REFERENCE_H5_DATASET_NAME,
                            statistics_mode=STATISTICS_MODE,
                            use_fwhm_edge_diameters=USE_FWHM_EDGE_DIAMETERS,
                            fwhm_raw_tiff_path=FWHM_RAW_TIFF_PATH,
                            fwhm_sample_spacing_along_edge_um=FWHM_SAMPLE_SPACING_ALONG_EDGE_UM,
                            fwhm_transverse_profile_step_um=FWHM_TRANSVERSE_PROFILE_STEP_UM,
                            fwhm_transverse_half_extent_um=FWHM_TRANSVERSE_HALF_EXTENT_UM,
                            fwhm_diameter_guess_um=FWHM_DIAMETER_GUESS_UM,
                            fwhm_min_total_extent_multiplier=FWHM_MIN_TOTAL_EXTENT_MULTIPLIER,
                            fwhm_background_label=FWHM_BACKGROUND_LABEL,
                            fwhm_junction_label=FWHM_JUNCTION_LABEL,
                            fwhm_allow_junction_crossing=FWHM_ALLOW_JUNCTION_CROSSING,
                            fwhm_profile_baseline_mode=FWHM_PROFILE_BASELINE_MODE,
                            fwhm_profile_baseline_wing_fraction=FWHM_PROFILE_BASELINE_WING_FRACTION,
                            fwhm_constrain_fitted_baseline=FWHM_CONSTRAIN_FITTED_BASELINE,
                            fwhm_baseline_constraint_half_width_ptp=FWHM_BASELINE_CONSTRAINT_HALF_WIDTH_PTP,
                            fwhm_clip_profile_to_single_vessel=FWHM_CLIP_PROFILE_TO_SINGLE_VESSEL,
                            fwhm_clip_min_drop_fraction_of_center=FWHM_CLIP_MIN_DROP_FRACTION_OF_CENTER,
                            fwhm_clip_re_rise_fraction_of_center=FWHM_CLIP_RE_RISE_FRACTION_OF_CENTER,
                            fwhm_branch_endpoint_exclusion_um=FWHM_BRANCH_ENDPOINT_EXCLUSION_UM,
                            fwhm_junction_proximity_exclusion_um=FWHM_JUNCTION_PROXIMITY_EXCLUSION_UM,
                            fwhm_enforce_same_edge_locality=FWHM_ENFORCE_SAME_EDGE_LOCALITY,
                            fwhm_same_edge_arc_window_um=FWHM_SAME_EDGE_ARC_WINDOW_UM,
                            fwhm_same_edge_arc_window_multiplier=FWHM_SAME_EDGE_ARC_WINDOW_MULTIPLIER,
                            fwhm_same_edge_arc_window_min_um=FWHM_SAME_EDGE_ARC_WINDOW_MIN_UM,
                            fwhm_cap_half_extent_by_nonlocal_same_edge_distance=FWHM_CAP_HALF_EXTENT_BY_NONLOCAL_SAME_EDGE_DISTANCE,
                            fwhm_nonlocal_same_edge_arc_separation_um=FWHM_NONLOCAL_SAME_EDGE_ARC_SEPARATION_UM,
                            fwhm_nonlocal_same_edge_half_extent_factor=FWHM_NONLOCAL_SAME_EDGE_HALF_EXTENT_FACTOR,
                            fwhm_reject_samples_with_center_offset=FWHM_REJECT_SAMPLES_WITH_CENTER_OFFSET,
                            fwhm_max_fit_center_offset_um=FWHM_MAX_FIT_CENTER_OFFSET_UM,
                            fwhm_reject_samples_with_low_fit_r2=FWHM_REJECT_SAMPLES_WITH_LOW_FIT_R2,
                            fwhm_min_fit_r2=FWHM_MIN_FIT_R2) -> None:
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
    if not use_large_vessel_masks:
        # Keep the loader contract strict: disabled mode must not receive mask paths.
        effective_large_arteriole_mask_path = None
        effective_large_venule_mask_path = None
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

    if (
        use_ilastik_small_vessel_segmentation
        and not use_small_vessel_masks_for_boundary_assignment
    ):
        raise ValueError(
            "use_ilastik_small_vessel_segmentation=True requires "
            "use_small_vessel_masks_for_boundary_assignment=True."
        )

    effective_small_arteriole_mask_path = small_arteriole_mask_path
    effective_small_venule_mask_path = small_venule_mask_path
    if not use_small_vessel_masks_for_boundary_assignment:
        # Keep the loader contract strict: disabled mode must not receive mask paths.
        effective_small_arteriole_mask_path = None
        effective_small_venule_mask_path = None
    if (
        use_small_vessel_masks_for_boundary_assignment
        and use_ilastik_small_vessel_segmentation
    ):
        if ilastik_unsegmented_small_arteriole_image_path is None:
            raise ValueError(
                "ilastik_unsegmented_small_arteriole_image_path must be set when "
                "use_ilastik_small_vessel_segmentation=True."
            )
        if ilastik_unsegmented_small_venule_image_path is None:
            raise ValueError(
                "ilastik_unsegmented_small_venule_image_path must be set when "
                "use_ilastik_small_vessel_segmentation=True."
            )
        if ilastik_small_arteriole_classifier_path is None:
            raise ValueError(
                "ilastik_small_arteriole_classifier_path must be set when "
                "use_ilastik_small_vessel_segmentation=True."
            )
        if ilastik_small_venule_classifier_path is None:
            raise ValueError(
                "ilastik_small_venule_classifier_path must be set when "
                "use_ilastik_small_vessel_segmentation=True."
            )

        ilastik_output_dir = Path(ilastik_output_dir)
        unsegmented_small_arteriole_image_path = io.resolve_image_path_with_optional_zip(
            Path(ilastik_unsegmented_small_arteriole_image_path)
        )
        unsegmented_small_venule_image_path = io.resolve_image_path_with_optional_zip(
            Path(ilastik_unsegmented_small_venule_image_path)
        )
        ilastik_segmented_small_arteriole_path = ilastik_output_dir / (
            f"{unsegmented_small_arteriole_image_path.stem}_segmented{ilastik_output_suffix}"
        )
        ilastik_segmented_small_venule_path = ilastik_output_dir / (
            f"{unsegmented_small_venule_image_path.stem}_segmented{ilastik_output_suffix}"
        )

        print(
            "Running ilastik segmentation for small arteriole image: "
            f"{unsegmented_small_arteriole_image_path}"
        )
        effective_small_arteriole_mask_path = io.run_ilastik_headless_segmentation(
            input_image_path=unsegmented_small_arteriole_image_path,
            classifier_path=Path(ilastik_small_arteriole_classifier_path),
            output_path=ilastik_segmented_small_arteriole_path,
            ilastik_executable=ilastik_executable,
        )
        print(
            "Running ilastik segmentation for small venule image: "
            f"{unsegmented_small_venule_image_path}"
        )
        effective_small_venule_mask_path = io.run_ilastik_headless_segmentation(
            input_image_path=unsegmented_small_venule_image_path,
            classifier_path=Path(ilastik_small_venule_classifier_path),
            output_path=ilastik_segmented_small_venule_path,
            ilastik_executable=ilastik_executable,
        )
        print(
            "Using ilastik-segmented small-vessel masks: "
            f"arteriole={effective_small_arteriole_mask_path}, "
            f"venule={effective_small_venule_mask_path}"
        )

    (
        small_arteriole_mask,
        small_venule_mask,
        small_arteriole_mask_voxel_size,
        small_venule_mask_voxel_size,
    ) = io.load_large_vessel_masks(
        enabled=use_small_vessel_masks_for_boundary_assignment,
        large_arteriole_mask_path=effective_small_arteriole_mask_path,
        large_venule_mask_path=effective_small_venule_mask_path,
    )
    if small_arteriole_mask is not None and small_venule_mask is not None:
        if small_arteriole_mask.shape != image.shape:
            raise ValueError(
                "small_arteriole_mask shape does not match input image shape: "
                f"{small_arteriole_mask.shape} != {image.shape}"
            )
        if small_venule_mask.shape != image.shape:
            raise ValueError(
                "small_venule_mask shape does not match input image shape: "
                f"{small_venule_mask.shape} != {image.shape}"
            )
        main_voxel_size_xyz = tuple(float(v) for v in voxel_size)
        small_arteriole_voxel_size_xyz = tuple(
            float(v) for v in small_arteriole_mask_voxel_size
        )
        small_venule_voxel_size_xyz = tuple(float(v) for v in small_venule_mask_voxel_size)
        if not (
            np.allclose(main_voxel_size_xyz, small_arteriole_voxel_size_xyz, rtol=0.0, atol=0.0)
            and np.allclose(main_voxel_size_xyz, small_venule_voxel_size_xyz, rtol=0.0, atol=0.0)
            and np.allclose(
                small_arteriole_voxel_size_xyz,
                small_venule_voxel_size_xyz,
                rtol=0.0,
                atol=0.0,
            )
        ):
            raise ValueError(
                "Voxel-size mismatch detected across input image and small-vessel masks. "
                f"main={main_voxel_size_xyz}, "
                f"small_arteriole={small_arteriole_voxel_size_xyz}, "
                f"small_venule={small_venule_voxel_size_xyz}. "
                "All must match exactly in x, y, and z."
            )
        print(
            "Loaded small-vessel masks for boundary assignment: "
            f"arteriole={small_arteriole_mask.shape}, venule={small_venule_mask.shape}, "
            f"min_overlap_fraction={float(small_vessel_mask_min_overlap_fraction):.3f}"
        )
    else:
        print(
            "Small-vessel-mask boundary assignment disabled; "
            "manual arteriole/venule boundary-node selection remains available."
        )

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

    auto_start_nodes: list[int] = []
    auto_output_nodes: list[int] = []
    if automated_vessel_assignment:
        if large_arteriole_mask is None or large_venule_mask is None:
            raise ValueError(
                "automated_vessel_assignment=True requires arteriole and venule masks. "
                "Set use_large_vessel_masks=True and provide mask paths."
            )
        auto_start_nodes, auto_output_nodes = (
            graph.select_terminal_nodes_from_large_vessel_masks(
                G,
                large_arteriole_mask=large_arteriole_mask,
                large_venule_mask=large_venule_mask,
                voxel_size_xyz=tuple(float(v) for v in voxel_size),
                allow_overlap=False,
            )
        )
        if not auto_start_nodes:
            raise ValueError(
                "automated_vessel_assignment=True found no terminal nodes in the "
                "arteriole mask (after any configured dilation)."
            )
        if not auto_output_nodes:
            raise ValueError(
                "automated_vessel_assignment=True found no terminal nodes in the "
                "venule mask (after any configured dilation)."
            )
        starting_node_coordinates = [
            tuple(np.asarray(G.nodes[node_id]["pos"], dtype=float))
            for node_id in auto_start_nodes
        ]
        output_node_coordinates = [
            tuple(np.asarray(G.nodes[node_id]["pos"], dtype=float))
            for node_id in auto_output_nodes
        ]
        automated_assignment_html_path = plot_dir / "automated_vessel_assignment_3d.html"
        wrote_assignment_html = graph.write_automated_vessel_assignment_3d_html(
            G,
            large_arteriole_mask=large_arteriole_mask,
            large_venule_mask=large_venule_mask,
            input_nodes=auto_start_nodes,
            output_nodes=auto_output_nodes,
            voxel_size_xyz=tuple(float(v) for v in voxel_size),
            output_html_path=automated_assignment_html_path,
        )
        if wrote_assignment_html:
            print(
                "Saved automated vessel-assignment 3D visualization to: "
                f"{automated_assignment_html_path}"
            )
        else:
            print(
                "Skipped automated vessel-assignment 3D visualization "
                "(plotly is not installed)."
            )
        print(
            "Automated vessel assignment selected "
            f"{len(starting_node_coordinates)} input coordinates from arteriole-mask overlap "
            f"and {len(output_node_coordinates)} output coordinates from venule-mask overlap."
        )

    starting_nodes[:] = []
    output_nodes[:] = []
    arteriole_boundary_nodes[:] = []
    venule_boundary_nodes[:] = []
    if automated_vessel_assignment:
        # Use direct terminal-node overlap assignment from vessel masks.
        start_nodes = auto_start_nodes
        out_nodes = [node_id for node_id in auto_output_nodes if node_id not in set(start_nodes)]
    else:
        start_nodes = graph.select_boundary_nodes_by_method(
            G,
            image.shape,
            method=starting_node_selection_method,
            node_role="input",
            coordinates=starting_node_coordinates,
            volume_boxes=starting_node_volumes,
        )
        out_nodes = graph.select_boundary_nodes_by_method(
            G,
            image.shape,
            method=output_node_selection_method,
            node_role="output",
            coordinates=output_node_coordinates,
            volume_boxes=output_node_volumes,
            exclude_nodes=start_nodes,
        )
    starting_nodes.extend(start_nodes)
    output_nodes.extend(out_nodes)
    used_nodes = set(starting_nodes) | set(output_nodes)
    if arteriole_boundary_node_coordinates or arteriole_boundary_node_volumes:
        art_boundary = graph.select_boundary_nodes_by_method(
            G,
            image.shape,
            method=arteriole_boundary_selection_method,
            node_role="input",
            coordinates=arteriole_boundary_node_coordinates,
            volume_boxes=arteriole_boundary_node_volumes,
            exclude_nodes=list(used_nodes),
        )
        arteriole_boundary_nodes.extend(art_boundary)
        used_nodes.update(arteriole_boundary_nodes)

    if venule_boundary_node_coordinates or venule_boundary_node_volumes:
        ven_boundary = graph.select_boundary_nodes_by_method(
            G,
            image.shape,
            method=venule_boundary_selection_method,
            node_role="output",
            coordinates=venule_boundary_node_coordinates,
            volume_boxes=venule_boundary_node_volumes,
            exclude_nodes=list(used_nodes),
        )
        venule_boundary_nodes.extend(ven_boundary)
    if use_small_vessel_masks_for_boundary_assignment:
        if small_arteriole_mask is None or small_venule_mask is None:
            raise ValueError(
                "use_small_vessel_masks_for_boundary_assignment=True requires "
                "small_arteriole_mask_path and small_venule_mask_path."
            )
        inferred_boundary_results = graph.infer_boundary_nodes_from_small_vessel_masks(
            G,
            small_arteriole_mask=small_arteriole_mask,
            small_venule_mask=small_venule_mask,
            voxel_size_xyz=tuple(float(v) for v in voxel_size),
            minimum_overlap_fraction=float(small_vessel_mask_min_overlap_fraction),
            allow_overlap=False,
        )
        arteriole_boundary_nodes[:] = list(
            inferred_boundary_results["arteriole_boundary_nodes"]
        )
        venule_boundary_nodes[:] = list(inferred_boundary_results["venule_boundary_nodes"])
        print(
            "Small-vessel mask boundary assignment selected "
            f"{len(arteriole_boundary_nodes)} arteriole boundary nodes and "
            f"{len(venule_boundary_nodes)} venule boundary nodes "
            f"(min_overlap_fraction={float(small_vessel_mask_min_overlap_fraction):.3f})."
        )
        print(
            "Small-vessel mask edge labels: "
            f"arteriole_edges={inferred_boundary_results['arteriole_edge_count']}, "
            f"venule_edges={inferred_boundary_results['venule_edge_count']}, "
            f"overlap_edges={inferred_boundary_results['overlap_edge_count']}."
        )
        if write_small_vessel_boundary_labelling_3d_html:
            boundary_html = Path(plot_dir) / "small_vessel_mask_boundary_labelling_3d.html"
            Path(plot_dir).mkdir(parents=True, exist_ok=True)
            ok = graph.write_small_vessel_mask_boundary_labelling_3d_html(
                G,
                small_arteriole_mask=small_arteriole_mask,
                small_venule_mask=small_venule_mask,
                arteriole_boundary_nodes=arteriole_boundary_nodes,
                venule_boundary_nodes=venule_boundary_nodes,
                voxel_size_xyz=tuple(float(v) for v in voxel_size),
                output_html_path=boundary_html,
            )
            if ok:
                print(f"Saved interactive 3D small-vessel boundary view: {boundary_html}")
            else:
                print(
                    "Small-vessel boundary 3D HTML not written (install plotly to enable)."
                )
    if automated_vessel_assignment:
        print(
            f"Selected {len(starting_nodes)} STARTING_NODES and {len(output_nodes)} "
            "OUTPUT_NODES directly from terminal-node overlap with vessel masks."
        )
    else:
        print(
            f"Selected {len(starting_nodes)} STARTING_NODES and {len(output_nodes)} "
            "OUTPUT_NODES from manual coordinates."
        )
    print(f"Starting nodes are: {starting_nodes}")
    print(f"Output nodes are: {output_nodes}")
    print(f"Arteriole boundary nodes are: {arteriole_boundary_nodes}")
    print(f"Venule boundary nodes are: {venule_boundary_nodes}")

    if starting_nodes and output_nodes:
        resistance_node_pair = (starting_nodes[0], output_nodes[0])
        print(f"Auto-selected resistance node pair: {resistance_node_pair}")
    else:
        if automated_vessel_assignment:
            raise ValueError(
                "No starting or output nodes found from terminal-node overlap with "
                "arteriole/venule masks."
            )
        raise ValueError(
            "No starting or output nodes found from manual input coordinates."
        )

    # 4) Add branch orders and hemodynamic edge weights.
    #HD note - eventually pericyte localisation should be able to be either determined by this manual method, or via loading in a segmented image of pericytes?
    #HD note - eventually add in probability of pericyte contraction?
    if starting_nodes:
        use_hierarchical_assignment = bool(
            arteriole_boundary_nodes and venule_boundary_nodes and output_nodes
        )
        if use_hierarchical_assignment:
            branch_assignment_results = graph.assign_hierarchical_branch_orders(
                G,
                starting_nodes=starting_nodes,
                output_nodes=output_nodes,
                arteriole_boundary_nodes=arteriole_boundary_nodes,
                venule_boundary_nodes=venule_boundary_nodes,
            )
            print(
                "Assigned hierarchical branch orders "
                "(Art*/Ven* first, then capillary B* from arteriole boundary)."
            )
            print(f"Branch assignment summary: {branch_assignment_results}")
        else:
            graph.assign_branch_orders(G, starting_nodes)
            print(
                "Assigned capillary branch orders from STARTING_NODES only "
                "(no arteriole/venule boundary-node sets supplied)."
            )

        vessel_type_3d_path = plot_dir / "vessel_types_assigned_3d.html"
        visualization.visualize_3d_plotly_vessel_types(
            G,
            title="Assigned Vessel Types (Interactive 3D)",
            save_html_path=str(vessel_type_3d_path),
            show=False,
        )
        print(
            "Saved vessel-type 3D visualization after branch assignment to: "
            f"{vessel_type_3d_path}"
        )
        if use_fwhm_edge_diameters:
            if fwhm_raw_tiff_path is None:
                raise ValueError(
                    "use_fwhm_edge_diameters=True requires fwhm_raw_tiff_path."
                )
            raw_p = io.resolve_image_path_with_optional_zip(Path(fwhm_raw_tiff_path))
            voxel_sz = tuple(
                float(v) for v in G.graph.get("image_voxel_size_xyz", voxel_size)
            )
            fwhm_summary = haemodynamics.automated.measure_edge_diameters_fwhm_from_raw_tiff(
                G,
                raw_tiff_path=raw_p,
                voxel_size_xyz=voxel_sz,
                sample_spacing_along_edge_um=float(fwhm_sample_spacing_along_edge_um),
                transverse_profile_step_um=float(fwhm_transverse_profile_step_um),
                transverse_half_extent_um=float(fwhm_transverse_half_extent_um),
                diameter_guess_um=(
                    None
                    if fwhm_diameter_guess_um is None
                    else float(fwhm_diameter_guess_um)
                ),
                background_label=int(fwhm_background_label),
                junction_label=int(fwhm_junction_label),
                min_total_extent_multiplier=float(fwhm_min_total_extent_multiplier),
                profile_baseline_mode=fwhm_profile_baseline_mode,
                profile_baseline_wing_fraction=float(fwhm_profile_baseline_wing_fraction),
                constrain_fitted_baseline=bool(fwhm_constrain_fitted_baseline),
                allow_junction_crossing=bool(fwhm_allow_junction_crossing),
                baseline_constraint_half_width_ptp=float(
                    fwhm_baseline_constraint_half_width_ptp
                ),
                clip_profile_to_single_vessel=bool(fwhm_clip_profile_to_single_vessel),
                clip_min_drop_fraction_of_center=float(
                    fwhm_clip_min_drop_fraction_of_center
                ),
                clip_re_rise_fraction_of_center=float(
                    fwhm_clip_re_rise_fraction_of_center
                ),
                branch_endpoint_exclusion_um=float(
                    fwhm_branch_endpoint_exclusion_um
                ),
                junction_proximity_exclusion_um=float(
                    fwhm_junction_proximity_exclusion_um
                ),
                enforce_same_edge_locality=bool(fwhm_enforce_same_edge_locality),
                same_edge_arc_window_um=(
                    None
                    if fwhm_same_edge_arc_window_um is None
                    else float(fwhm_same_edge_arc_window_um)
                ),
                same_edge_arc_window_multiplier=float(
                    fwhm_same_edge_arc_window_multiplier
                ),
                same_edge_arc_window_min_um=float(
                    fwhm_same_edge_arc_window_min_um
                ),
                cap_half_extent_by_nonlocal_same_edge_distance=bool(
                    fwhm_cap_half_extent_by_nonlocal_same_edge_distance
                ),
                nonlocal_same_edge_arc_separation_um=float(
                    fwhm_nonlocal_same_edge_arc_separation_um
                ),
                nonlocal_same_edge_half_extent_factor=float(
                    fwhm_nonlocal_same_edge_half_extent_factor
                ),
                reject_samples_with_center_offset=bool(
                    fwhm_reject_samples_with_center_offset
                ),
                max_fit_center_offset_um=float(
                    fwhm_max_fit_center_offset_um
                ),
                reject_samples_with_low_fit_r2=bool(
                    fwhm_reject_samples_with_low_fit_r2
                ),
                min_fit_r2=float(fwhm_min_fit_r2),
            )
            print(f"FWHM diameter measurement summary: {fwhm_summary}")
            if do_pericyte_constriction:
                print(
                    "Pericyte mode: passive diameter d1 from per-edge FWHM where available, "
                    "else DIAMETER_BY_BRANCH_ORDER; d2 = d1 * CONSTRICTION_BY_BRANCH_ORDER."
                )
        elif not use_fwhm_edge_diameters:
            print(
                "Vessel diameters: manual mode (DIAMETER_BY_BRANCH_ORDER / "
                "set_poiseuille_weights without per-edge FWHM)."
            )
        if run_pericyte_resistance_comparison:
            comparison_csv_path = output_dir / f"{image_path.stem}_pericyte_resistance_comparison.csv"
            comparison_results = (
                pericyte_comparison_haemodynamics.compare_baseline_vs_pericyte_constriction(
                    G,
                    diameter_by_branch_order=diameter_by_branch_order,
                    constriction_factor_by_branch_order=constriction_by_branch_order,
                    resistance_node_pair=resistance_node_pair,
                    output_csv_path=comparison_csv_path,
                    baseline_factor_scale=float(pericyte_comparison_baseline_scale),
                    constricted_factor_scale=float(pericyte_comparison_constricted_scale),
                    use_pericyte_mask_constriction=bool(use_pericyte_mask_constriction),
                    pericyte_mask_path=pericyte_mask_path,
                    pericyte_mask_h5_dataset_name=pericyte_mask_h5_dataset_name,
                    prefer_edge_fwhm_baseline=bool(use_fwhm_edge_diameters),
                    constriction_length=40.0,
                    constriction_spacing=100.0,
                    use_probabilistic_pericyte_constriction=bool(
                        use_probabilistic_pericyte_constriction
                    ),
                    pericyte_constriction_probability=float(
                        pericyte_constriction_probability
                    ),
                )
            )
            print(
                "Pericyte resistance comparison complete: "
                f"baseline={comparison_results['baseline_resistance']:.6f}, "
                f"constricted={comparison_results['constricted_resistance']:.6f}, "
                f"delta={comparison_results['delta']:.6f}, "
                f"change={comparison_results['percent_change']:.3f}%."
            )
            print(
                "Saved pericyte resistance comparison CSV to: "
                f"{comparison_results['output_csv_path']}"
            )
        poiseuille_model = haemodynamics.PoiseuilleModel(
            constriction_length=40.0,
            constriction_spacing=100.0,
        )
        if do_pericyte_constriction:
            if use_pericyte_mask_constriction:
                if pericyte_mask_path is None:
                    raise ValueError(
                        "pericyte_mask_path must be set when "
                        "use_pericyte_mask_constriction=True."
                    )
                G, results = pericyte_mask_haemodynamics.set_poiseuille_weights_with_pericyte_mask(
                    G,
                    diameter_by_branch_order=diameter_by_branch_order,
                    constriction_factor_by_branch_order=constriction_by_branch_order,
                    pericyte_mask_path=pericyte_mask_path,
                    pericyte_mask_h5_dataset_name=pericyte_mask_h5_dataset_name,
                    prefer_edge_fwhm_baseline=bool(use_fwhm_edge_diameters),
                    constriction_length=40.0,
                    use_probabilistic_constriction=bool(
                        use_probabilistic_pericyte_constriction
                    ),
                    constriction_probability=float(pericyte_constriction_probability),
                )
                print(
                    "Results from set_poiseuille_weights_with_pericyte_mask "
                    f"(centroid-based d2 from mask): {results}"
                )
            else:
                if use_fwhm_edge_diameters:
                    G, results = poiseuille_model.set_poiseuille_weights_with_constrictions(
                        G,
                        diameter_by_branch_order,
                        prefer_edge_fwhm_baseline=True,
                        constriction_factor_by_branch_order=constriction_by_branch_order,
                    )
                    print(
                        "Results from set_poiseuille_weights_with_constrictions "
                        f"(FWHM baseline d1, constriction factors): {results}"
                    )
                else:
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
                    print(
                        f"Results from set_poiseuille_weights_with_constrictions: {results}"
                    )
        else:
            G, results = poiseuille_model.set_poiseuille_weights(
                G,
                diameter_by_branch_order,
                prefer_edge_fwhm_diameter=bool(use_fwhm_edge_diameters),
            )
            _diam_mode = (
                "per-edge FWHM (Gaussian fit) with branch-order fallback"
                if use_fwhm_edge_diameters
                else "branch-order table only"
            )
            print(f"Results from set_poiseuille_weights ({_diam_mode}): {results}")

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
    conductance, node_list = haemodynamics.build_conductance_matrix_from_graph(G)
    node_to_idx = {node_id: idx for idx, node_id in enumerate(node_list)}
    print(f"Conductance matrix built with shape {conductance.shape} and node_list length {len(node_list)}.")
    if do_equiv_resistance_calculation:
        source_node, target_node = resistance_node_pair
        if source_node in node_to_idx and target_node in node_to_idx:
            laplacian = haemodynamics.calc_laplacian_from_conductance_matrix(conductance)
            two_point_resistance = haemodynamics.calc_two_point_from_laplacian_matrix_nodeID(
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

        stats_csv_path = output_dir / f"{image_path.stem}_statistics.csv"
        statistics.export_statistics_to_csv(stats, stats_csv_path)
        print(f"Saved statistics CSV to: {stats_csv_path}")

        branch_stats = statistics.compute_branch_order_statistics(
            G,
            node_positions=node_positions,
        )
        branch_stats_csv_path = output_dir / f"{image_path.stem}_branch_statistics.csv"
        statistics.export_branch_order_statistics_to_csv(
            branch_stats,
            branch_stats_csv_path,
        )
        print(f"Saved branch-order statistics CSV to: {branch_stats_csv_path}")

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

    # 8) Optional: nearest 3D distance from objects in a cell mask to vessel edge.
    if measurement_3d_to_cell_mask:
        if cell_mask_path is None:
            raise ValueError(
                "measurement_3d_to_cell_mask=True requires cell_mask_path."
            )
        distance_summary = statistics.run_3d_measurement_to_cell_mask(
            graph=G,
            cell_mask_path=Path(cell_mask_path),
            output_dir=output_dir,
            image_stem=image_path.stem,
            voxel_size_xyz=tuple(float(v) for v in voxel_size),
            vessel_mask_path=(
                None
                if measurement_3d_vessel_mask_path is None
                else Path(measurement_3d_vessel_mask_path)
            ),
            vessel_reference_image_path=(
                None
                if measurement_3d_reference_image_path is None
                else Path(measurement_3d_reference_image_path)
            ),
            cell_mask_h5_dataset_name=cell_mask_h5_dataset_name,
            vessel_mask_h5_dataset_name=measurement_3d_vessel_mask_h5_dataset_name,
            vessel_reference_h5_dataset_name=measurement_3d_reference_h5_dataset_name,
        )
        print(
            "3D cell-mask vessel-distance summary: "
            f"{distance_summary}"
        )
    else:
        print("3D cell-mask vessel-distance measurement skipped.")

    # 9) Also solve for flow throughout the network using the conductance matrix 
    # and the input and output pressures.
    print("\nSolving flow through the network...")
    flow, vtk_export = haemodynamics.solve_flow_from_conductance_matrix(
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

    # 10) Optional matplotlib visualization.
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
    import argparse

    parser = argparse.ArgumentParser(description="Resistance network pipeline example.")
    parser.add_argument(
        "--run-small-vessel-boundary-labelling-tests",
        action="store_true",
        help="Run pytest on tests/test_small_vessel_mask_boundary_labelling.py and exit.",
    )
    parser.add_argument(
        "--use-fwhm-edge-diameters",
        action="store_true",
        help=(
            "Override USE_FWHM_EDGE_DIAMETERS: measure diameters from raw TIFF "
            "(Gaussian transverse fit). Requires --fwhm-raw-tiff unless "
            "FWHM_RAW_TIFF_PATH is set in this file."
        ),
    )
    parser.add_argument(
        "--fwhm-raw-tiff",
        type=Path,
        default=None,
        help="Path to raw single-channel TIFF for FWHM (overrides FWHM_RAW_TIFF_PATH).",
    )
    cli = parser.parse_args()
    if cli.run_small_vessel_boundary_labelling_tests:
        import pytest

        raise SystemExit(
            pytest.main([str(root_dir / "tests" / "test_small_vessel_mask_boundary_labelling.py"), "-q"])
        )
    plot_dir = BASE_PLOT_DIR / "nerve"
    pipeline_kwargs: dict = {}
    if cli.use_fwhm_edge_diameters:
        pipeline_kwargs["use_fwhm_edge_diameters"] = True
    if cli.fwhm_raw_tiff is not None:
        pipeline_kwargs["fwhm_raw_tiff_path"] = cli.fwhm_raw_tiff
    image_to_model_pipeline(plot_dir=plot_dir, **pipeline_kwargs)
