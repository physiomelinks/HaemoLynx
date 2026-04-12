#!/usr/bin/env python3
"""Refactored full pipeline example using ImageLynx package."""
import logging
import sys
import pickle
import os
import subprocess
from pathlib import Path
from skan import csr
import tifffile
import numpy as np
import networkx as nx
from dataclasses import dataclass, field

# Ensure package is importable when running from repo root.
root_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


from ImageLynx import graph, hemodynamics, io, preprocessing, statistics, visualization 

# ---------------------------
# Beginner-friendly settings
# ---------------------------

# Ilastik configuration settings
RUN_ILASTIK = False
ILASTIK_OUTPUT_PROBABILITIES = True # Set to True for Probabilities, False for Simple Segmentation
# The channel index in the probability map that corresponds to vessels (usually 0 or 1).
ILASTIK_VESSEL_CHANNEL = 0
ILASTIK_BINARY_PATH = "/home/dsas627/Desktop/ilastik-1.4.1rc2-gpu-Linux/run_ilastik.sh"
ILASTIK_PROJECT_PATH = root_dir / "examples" / "images" / "cb_wky_2x2x2_A.ilp"
RAW_IMAGE_DIR = root_dir / "examples" / "images" / "ilastik_batch_processing_input_images"
ILASTIK_OUTPUT_DIR = root_dir / "examples" / "images" / "ilastik_batch_processing_output_images"

# Paths for multi-input Ilastik features (e.g., Raw + Frangi)
RAW_IMAGE_PATH = RAW_IMAGE_DIR / "C1-CB3-WKY-CB-A-2x2x2_vessels.tif"
FRANGI_IMAGE_PATH = RAW_IMAGE_DIR / "C1-CB3-WKY-CB-A-2x2x2_vesselness_map.tif"

INPUT_PATH = None
BASE_PLOT_DIR = root_dir / "examples" / "plots" 
if not BASE_PLOT_DIR.exists():
    BASE_PLOT_DIR.mkdir(parents=True, exist_ok=True)
H5_DATASET_NAME = None  # For h5 input, e.g. "data"
# STARTING NODES and OUTPUT Nodes are now calculated automatically by looking for degree 1 nodes at start or
# end of the image.
EDGE_PERCENT = 25.0
END_PERCENT = 25.0
# For 3D skeletons this is usually the y-axis in (z, y, x).
NODE_EDGE_AXIS = 0
STARTING_NODES: list[int] = []
OUTPUT_NODES: list[int] = []
INPUT_P_BC = 1000 # Pa 
OUTPUT_P_BC = 500 # Pa
VISUALIZE_RESULTS = False
VISUALIZE_MASK_ONLY = False
# ---------------------------
# Vedo Visualization Style (image_to_model style)
# ---------------------------
VISUALIZE_VEDO = True
# Mode: 'lego' (exact voxels) or 'iso' (smooth surface)
VISUALIZE_VEDO_MODE = 'iso' 
# Smoothing iterations (only for 'iso' mode)
VISUALIZE_VEDO_SMOOTH_ITER = 15
# Voxel spacing [z, y, x]
VISUALIZE_VEDO_SPACING = (1.0, 1.0, 1.0)
# If True, attempts to read spacing from TIF metadata automatically.
VISUALIZE_VEDO_AUTO_SPACING = True

VISUALIZE_VEDO_OPACITY = 0.5
VISUALIZE_OVERLAY_PREVIEW = False

VISUALIZE_MASK_OPACITY = 1.0
VISUALIZE_VTK = False
VERBOSE_LOGGING = False
DO_SKELETONIZE = True
DO_GRAPH_BUILDING = True
DO_RESISTANCE_CALCULATION = True
CONSTRICT_AT_PERICYTES = False
MIN_BRANCH_LENGTH = 10
VTK_OUTPUT_PREFIX = root_dir / "examples" / "outputs" / "resistance_network"
SKELETON_CLOSING_RADIUS = 1
SKELETON_BRIDGE_GAP_SIZE = 1
SKELETON_MIN_BRANCH_LENGTH = 3
SKELETON_MAX_BRIDGE_DISTANCE = 0
SKELETON_COMPONENT_CONNECTIVITY = 3
# Keep only connected components at or above this percentage of total
# skeleton voxels (e.g. 5.0 -> keep components >= 5% of total skeleton voxels).
SKELETON_MIN_COMPONENT_PERCENT = 5.0

# ---------------------------
# Skeleton Bundle Cleanup settings (Added 06/04/2026)
# ---------------------------
# Window size used to detect dense skeleton "bundles" (must be odd).
SKELETON_BUNDLE_SCAN_SIZE = 9
# Density threshold (0 to 1) above which a region is collapsed into a single hub.
# Lower values are more aggressive at removing tangled "blobs".
SKELETON_BUNDLE_DENSITY_FRACTION = 0.025
# Maximum number of paths to keep when reconnecting a collapsed hub.
SKELETON_BUNDLE_MAX_CONNECTIONS = 5
# Minimum spacing between hub centers. 0 to disable.
SKELETON_BUNDLE_HUB_MIN_SPACING = 0

# ---------------------------
# Advanced Efficiency settings (Added 19/03/2026)
# ---------------------------
# Downsample factor for 3D skeletonization (e.g. 2.0 reduces each dimension by half).
# Set to 1.0 to disable downsampling.
SKELETON_DOWNSAMPLE_FACTOR = 1.0

# Enable local padded slicing for much faster loop detection on large skeletons.
SKELETON_USE_PADDED_SLICING = True
# Voxel padding for the local slicing crops.
SKELETON_PADDED_SLICING_PADDING = 3

# Prune the binary mask to keep only the largest N connected components BEFORE skeletonization.
# This speeds up skeletonization by removing noise fragments. Set to 0 to disable.
SKELETON_PRUNE_MASK_BEFORE_SKELETONIZATION = 1

# If True, keeps only the single largest connected component of the final mathematical graph.
# This ensures zero "floating islands" exist before flow solving.
GRAPH_KEEP_LARGEST_COMPONENT_ONLY = True

# Sub-volume / ROI settings. 
# SKELETON_SUB_VOLUME_PERCENTAGE: percentage of original volume to keep (0.0 to 1.0). Set to 1.0 for full volume.
SKELETON_SUB_VOLUME_PERCENTAGE = 0.25

# Center offsets for the ROI (as percentage of original dimensions, -0.5 to 0.5).
SKELETON_SUB_VOLUME_CENTER_OFFSET_Z = 0.0
SKELETON_SUB_VOLUME_CENTER_OFFSET_Y = 0.0
SKELETON_SUB_VOLUME_CENTER_OFFSET_X = 0.0

# ---------------------------
# Probability Map Post-processing settings (Added 30/03/2026)
# ---------------------------
# Median filter window size (e.g., 3 for 3x3x3). 0 to disable.
MEDIAN_FILTER_SIZE = 7

# Smoothing of probability maps before thresholding. 0.0 to disable.
PROBABILITY_SMOOTHING_SIGMA = 0.0

# Morphological opening radius (applied to probability map before thresholding). 0 to disable.
MORPHOLOGICAL_OPENING_RADIUS = 1

# If True, uses Hysteresis thresholding instead of global Otsu.
ENABLE_HYSTERESIS_THRESHOLD = True
# Lower threshold for connectivity (keeps voxels if they connect to a 'high' seed).
HYSTERESIS_THRESHOLD_LOW = 0.2
# Upper threshold for seeds (defines definitely-vessel voxels).
HYSTERESIS_THRESHOLD_HIGH = 0.4

# Enable filling internal holes in the binary mask.
ENABLE_HOLE_FILLING = True

# ---------------------------
# Shannon Entropy settings (Added 30/03/2026)
# ---------------------------
# If True, uses Shannon Entropy to identify and reject uncertain voxels.
ENABLE_SHANNON_ENTROPY = True
# Voxels with normalized entropy above this threshold are forced to background.
SHANNON_ENTROPY_THRESHOLD = 0.95

# Visualize the post-processed binary mask and exit (Added 30/03/2026)
VISUALIZE_POST_PROCESSED_MASK = False
"""Configuration defaults for diameter maps."""

# Diameter by branch order (dict with d1 and d2 for pericyte constriction simulation)
DIAMETER_BY_BRANCH_ORDER = {
    "BO1": {"d1": 4.0, "d2": 4.0},
    "BO2": {"d1": 4.0, "d2": 4.0},
    "BO3": {"d1": 4.0, "d2": 4.0},
    "BO4": {"d1": 4.0, "d2": 4.0},
    "BO5": {"d1": 4.0, "d2": 4.0},
    "BO6": {"d1": 4.0, "d2": 4.0},
    "BO7": {"d1": 4.0, "d2": 4.0},
    "BO8": {"d1": 4.0, "d2": 4.0},
    "BO9": {"d1": 4.0, "d2": 4.0},
    "DEFAULT": {"d1": 4.0, "d2": 4.0},
}

DIAMETER_BY_BRANCH_ORDER_ENHANCED = None

# These are vesses that constrict differently (e.g. endoneurial vessels).
custom_edges= []  



@dataclass
class PreprocessingConfig:
    median_filter_size: int = 7
    probability_smoothing_sigma: float = 0.0
    morphological_opening_radius: int = 1
    enable_hysteresis_threshold: bool = True
    hysteresis_threshold_low: float = 0.2
    hysteresis_threshold_high: float = 0.4
    enable_hole_filling: bool = True
    ilastik_vessel_channel: int = 0
    enable_shannon_entropy: bool = True
    shannon_entropy_threshold: float = 0.95

@dataclass
class SkeletonConfig:
    closing_radius: int = 1
    bridge_gap_size: int = 1
    min_branch_length: int = 3
    max_bridge_distance: int = 0
    component_connectivity: int = 3
    min_component_percent: float = 5.0
    downsample_factor: float = 1.0
    use_padded_slicing: bool = True
    padded_slicing_padding: int = 3
    prune_mask_before: int = 1
    sub_volume_percentage: float = 0.25
    sub_volume_offset_z: float = 0.0
    sub_volume_offset_y: float = 0.0
    sub_volume_offset_x: float = 0.0
    bundle_scan_size: int = 9
    bundle_density_fraction: float = 0.025
    bundle_max_connections: int = 5
    bundle_hub_min_spacing: int = 0

@dataclass
class GraphConfig:
    keep_largest_component_only: bool = True
    edge_percent: float = 25.0
    end_percent: float = 25.0
    node_edge_axis: int = 0
    starting_nodes: list = field(default_factory=list)
    output_nodes: list = field(default_factory=list)

@dataclass
class HemodynamicsConfig:
    constrict_at_pericytes: bool = False
    input_p_bc: float = 1000.0
    output_p_bc: float = 500.0
    diameter_by_branch_order: dict = field(default_factory=dict)

@dataclass
class VisualizationConfig:
    visualize_results: bool = False
    visualize_mask_only: bool = False
    visualize_vedo: bool = True
    visualize_overlay_preview: bool = False
    visualize_vedo_mode: str = 'iso'
    visualize_vedo_smooth_iter: int = 15
    visualize_vedo_spacing: tuple = (1.0, 1.0, 1.0)
    visualize_vedo_auto_spacing: bool = True
    visualize_vedo_opacity: float = 0.5
    visualize_mask_opacity: float = 1.0
    visualize_vtk: bool = False
    visualize_post_processed_mask: bool = False

@dataclass
class PipelineConfig:
    do_skeletonize: bool = True
    do_graph_building: bool = True
    do_resistance_calculation: bool = True
    verbose_logging: bool = False
    min_branch_length: int = 10
    vtk_output_prefix: Path = None
    plot_dir: Path = None

class IlastikClassifier():
    """Wrapper for the Ilastik headless engine to perform pixel classification."""
    def __init__(self, ilastik_binary_path, project_file_path):
        self.binary = ilastik_binary_path
        self.project = project_file_path

        if not os.path.exists(self.binary):
            raise FileNotFoundError(f"Ilastik binary not found at: {self.binary}")
        if not os.path.exists(self.project):
            raise FileNotFoundError(f"Project file not found at: {self.project}")

    def segment_images(self, image_paths, output_dir, export_source="Simple Segmentation"):
        """Runs the segmentation on a volume composed of multiple input feature files."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Ensure all paths exist and are absolute strings
        formatted_paths = []
        for p in image_paths:
            p = Path(p)
            if not p.exists():
                raise FileNotFoundError(f"Input image not found: {p}")
            formatted_paths.append(str(p))

        print(f"Starting Ilastik engine for {export_source} with {len(formatted_paths)} input features...")

        # Determine suffix for naming (seg or probs)
        suffix = "probs" if export_source == "Probabilities" else "seg"

        # Base command - Use multipage tiff to support 3D volumes
        cmd = [
            str(self.binary),
            "--headless",
            f"--project={self.project}",
            "--output_format=multipage tiff", 
            f"--export_source={export_source}",
            f"--output_filename_format={output_dir}/{{nickname}}_{suffix}.tif"
        ]
        
        # Add each file as a separate raw_data entry to fill Ilastik input slots
        for path_str in formatted_paths:
            cmd.append(f"--raw_data={path_str}")

        try:
            process = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print("Ilastik Output Log (Success):\n", process.stdout)
            print(f"Results saved in: {output_dir}")
        except subprocess.CalledProcessError as e:
            print("\n!!! Error occurred during Ilastik processing !!!")
            print("--- ILASTIK ERROR OUTPUT ---\n", e.stderr)
            raise e

def run_ilastik_segmentation(ilastik_bin=ILASTIK_BINARY_PATH, 
                             project_path=ILASTIK_PROJECT_PATH, 
                             raw_image_path=RAW_IMAGE_PATH,
                             frangi_image_path=FRANGI_IMAGE_PATH,
                             output_dir=ILASTIK_OUTPUT_DIR, 
                             output_probabilities=ILASTIK_OUTPUT_PROBABILITIES):
    """
    Stand-alone function to trigger the Ilastik headless segmentation.
    
    Args:
        ilastik_bin (str): Path to the run_ilastik.sh executable.
        project_path (str): Path to the .ilp project file.
        raw_image_path (str): Path to the raw CB image.
        frangi_image_path (str): Path to the frangi vesselness map.
        output_dir (str): Directory where the result will be saved.
        output_probabilities (bool): If True, exports "Probabilities". If False, "Simple Segmentation".
        
    Returns:
        Path: The absolute path to the generated segmentation/probability TIFF file.
    """
    
    # Bundle input features (Order matters! Matches Ilastik slots)
    input_features = [raw_image_path, frangi_image_path]
    
    classifier = IlastikClassifier(ilastik_bin, project_path)
    
    # Set export source based on toggle
    export_src = "Probabilities" if output_probabilities else "Simple Segmentation"
    suffix = "probs" if output_probabilities else "seg"

    # Trigger the segmentation
    classifier.segment_images(
        image_paths=input_features,
        output_dir=output_dir,
        export_source=export_src
    )
    
    # Identify the resulting filename
    # Ilastik headless can be unpredictable with nicknames and extensions (.tif vs .tiff)
    possible_stems = [Path(p).stem for p in input_features]
    exts = [".tif", ".tiff"]
    
    result_path = None
    for stem in possible_stems:
        for ext in exts:
            test_path = Path(output_dir) / f"{stem}_{suffix}{ext}"
            if test_path.exists():
                result_path = test_path
                break
        if result_path: break

    if not result_path:
        # Final fallback: look for ANY file in the output dir modified in the last 60 seconds
        print(f"Warning: Specific output not found. Searching {output_dir} for recent results...")
        recent_files = sorted(Path(output_dir).glob(f"*_{suffix}.tif*"), key=os.path.getmtime, reverse=True)
        if recent_files:
            result_path = recent_files[0]
            print(f"Detected output file: {result_path}")
        else:
            raise FileNotFoundError(f"Could not find Ilastik output in {output_dir}")
        
    return result_path



def _preview_raw_volume(image, image_path, input_format, vis_config):
    if vis_config.visualize_vedo:
        print(f"Visualizing 3D volume with VEDO ({vis_config.visualize_vedo_mode}, smooth={vis_config.visualize_vedo_smooth_iter}).")
        current_spacing = vis_config.visualize_vedo_spacing
        if vis_config.visualize_vedo_auto_spacing and input_format == "tif":
            detected = io.get_tif_spacing(image_path)
            print(f"  Auto-detected spacing (z,y,x): {detected}")
            current_spacing = detected

        visualization.visualize_volume_vedo(
            image,
            title=f"Vedo 3D Image ({vis_config.visualize_vedo_mode})",
            mode=vis_config.visualize_vedo_mode,
            spacing=current_spacing,
            alpha=vis_config.visualize_vedo_opacity,
            smooth_iter=vis_config.visualize_vedo_smooth_iter
        )
    else:
        print(f"Visualizing PRE-OTSU intensity volume (cropped, opacity={vis_config.visualize_mask_opacity}). Close window to exit.")
        visualization.visualize_volume(image, title="3D Pre-Otsu Intensity Image", opacity=vis_config.visualize_mask_opacity)
    print("Exiting pipeline as requested.")
    import sys
    sys.exit(0)

def _preview_post_processed_mask(binary, image_path, input_format, vis_config, pipeline_config):
    mask_plot_path = pipeline_config.plot_dir / "post_processed_mask.png"
    print(f"Visualizing post-processed binary mask with Vedo (opacity=0.5). Voxel count: {binary.sum()}. Saving to {mask_plot_path}. Close window to exit.")
    
    current_spacing = vis_config.visualize_vedo_spacing
    if vis_config.visualize_vedo_auto_spacing and input_format == "tif":
        current_spacing = io.get_tif_spacing(image_path)

    visualization.visualize_volume_vedo(
        binary, 
        title="3D Post-Processed Binary Mask (Vedo)", 
        mode=vis_config.visualize_vedo_mode,
        spacing=current_spacing,
        alpha=0.5,
        smooth_iter=vis_config.visualize_vedo_smooth_iter
    )
    print("Exiting pipeline as requested after post-processed mask visualization.")
    import sys
    sys.exit(0)

def _visualize_final_results(G, image, starting_nodes, vis_config):
    visualization.plot_node_degree_distribution(G)
    visualization.visualize_edges_and_nodes(image, G)
    
    if starting_nodes:
        visualization.visualize_geometry_with_branch_orders(
            image,
            G,
            group_above=8,
        )

def _load_and_preprocess_image(image_path, input_format, pre_config, skel_config, vis_config, pipeline_config):
    if input_format == "tif":
        image = io.load_3d_tif(image_path)
    elif input_format == "h5":
        if not H5_DATASET_NAME:
            raise ValueError("Set H5_DATASET_NAME when INPUT_FORMAT is 'h5'.")
        image = io.load_3d_h5(image_path, H5_DATASET_NAME)
    else:
        raise ValueError("INPUT_FORMAT must be 'tif' or 'h5'.")

    print(f"Loaded image shape: {image.shape}")

    # 1.5) Sub-volume / ROI Cropping (Supports 4D)
    if 0 < skel_config.sub_volume_percentage < 1.0 or skel_config.sub_volume_offset_z != 0 or        skel_config.sub_volume_offset_y != 0 or skel_config.sub_volume_offset_x != 0:
        
        print(f"Applying ROI crop (sub-volume={skel_config.sub_volume_percentage})...")
        image = preprocessing.crop_roi(
            image,
            sub_volume_percentage=skel_config.sub_volume_percentage,
            offset_z=skel_config.sub_volume_offset_z,
            offset_y=skel_config.sub_volume_offset_y,
            offset_x=skel_config.sub_volume_offset_x
        )
        print(f"  ROI new shape: {image.shape}")

    entropy_map = None
    if image.ndim == 4:
        # Calculate entropy before extracting the vessel channel
        if pre_config.enable_shannon_entropy:
            entropy_map = preprocessing.calculate_entropy_map(image)

        dims = np.array(image.shape)
        c_axis = np.argmin(dims)
        print(f"Detected 4D image. Assuming channel is at axis {c_axis} (size {dims[c_axis]}).")
        
        if c_axis == 0:
            image = image[pre_config.ilastik_vessel_channel, :, :, :]
        elif c_axis == 1:
            image = image[:, pre_config.ilastik_vessel_channel, :, :]
        elif c_axis == 2:
            image = image[:, :, pre_config.ilastik_vessel_channel, :]
        else:
            image = image[:, :, :, pre_config.ilastik_vessel_channel]
            
        print(f"Extracted vessel channel {pre_config.ilastik_vessel_channel}. New spatial shape: {image.shape}")

        if entropy_map is not None:
            uncertain_mask = entropy_map > pre_config.shannon_entropy_threshold
            print(f"Applying Shannon Entropy Refinement (threshold={pre_config.shannon_entropy_threshold})...")
            print(f"  Rejecting {uncertain_mask.sum()} uncertain voxels.")
            image[uncertain_mask] = 0.0

    print(f"Image probability range: min={image.min():.4f}, max={image.max():.4f}, mean={image.mean():.4f}")

    if vis_config.visualize_mask_only:
        _preview_raw_volume(image, image_path, input_format, vis_config)

    if pre_config.median_filter_size > 0:
        print(f"Applying median filter (size={pre_config.median_filter_size})...")
        image = preprocessing.median_filter_image(image, size=pre_config.median_filter_size)

    if pre_config.morphological_opening_radius > 0:
        print(f"Applying morphological opening (radius={pre_config.morphological_opening_radius})...")
        image = preprocessing.morphological_opening(image, radius=pre_config.morphological_opening_radius)

    if pre_config.probability_smoothing_sigma > 0:
        image = preprocessing.smooth_probability_map(image, sigma=pre_config.probability_smoothing_sigma)

    if pre_config.enable_hysteresis_threshold:
        binary_raw = preprocessing.hysteresis_threshold(
            image, low=pre_config.hysteresis_threshold_low, high=pre_config.hysteresis_threshold_high
        )
    else:
        from skimage.filters import threshold_otsu
        threshold = threshold_otsu(image)
        binary_raw = image > threshold
    
    binary = binary_raw.copy()

    if pre_config.enable_hole_filling:
        binary = preprocessing.skeleton.fill_holes_3d(binary)

    if skel_config.closing_radius > 0:
        binary = preprocessing.skeleton.close_binary_mask(binary, radius=skel_config.closing_radius)
    if skel_config.bridge_gap_size > 0:
        binary = preprocessing.skeleton.bridge_gaps(binary, max_gap=skel_config.bridge_gap_size)

    if skel_config.prune_mask_before > 0:
        print(f"Pruning binary mask to keep largest {skel_config.prune_mask_before} components...")
        binary = preprocessing.skeleton.keep_largest_mask_components(
            binary, n_components=skel_config.prune_mask_before, connectivity=skel_config.component_connectivity
        )

    if vis_config.visualize_post_processed_mask:
        _preview_post_processed_mask(binary, image_path, input_format, vis_config, pipeline_config)
    
    return image, binary

def _run_skeletonization_phase(binary, skel_config):
    if skel_config.downsample_factor > 1.0:
        print(f"Applying downsampled skeletonization (factor={skel_config.downsample_factor})...")
        skeleton = preprocessing.skeleton.rescale_and_skeletonize_3d(binary, downsample_factor=skel_config.downsample_factor)
    else:
        skeleton = preprocessing.skeleton.skeletonize_3d(binary)
    
    preprocessing.print_skeleton_connectivity_stats(
        "raw",
        skeleton,
        component_connectivity=skel_config.component_connectivity,
    )

    skeleton = preprocessing.preprocess_skeleton_for_graph(
        skeleton,
        min_branch_length=skel_config.min_branch_length,
        max_bridge_distance=skel_config.max_bridge_distance,
        component_connectivity=skel_config.component_connectivity,
        min_component_fraction=skel_config.min_component_percent / 100.0,
        bundle_scan_size=skel_config.bundle_scan_size,
        bundle_density_fraction=skel_config.bundle_density_fraction,
        bundle_max_connections_per_hub=skel_config.bundle_max_connections,
        bundle_hub_min_spacing=skel_config.bundle_hub_min_spacing,
    )
    preprocessing.print_skeleton_connectivity_stats(
        "cleaned",
        skeleton,
        component_connectivity=skel_config.component_connectivity,
    )
    return skeleton

def _preview_overlay(binary, skeleton, G, image_path, input_format, vis_config):
    print(f"Visualizing 3D overlay PREVIEW with optimized graph (mask opacity=0.3). Close window to exit.")
    
    current_spacing = vis_config.visualize_vedo_spacing
    if vis_config.visualize_vedo_auto_spacing and input_format == "tif":
        current_spacing = io.get_tif_spacing(image_path)

    if vis_config.visualize_vedo:
        visualization.visualize_overlay_vedo(
            binary, 
            skeleton, 
            title="3D Skeleton & Graph Overlay Preview (Vedo)", 
            alpha=0.3,
            mode=vis_config.visualize_vedo_mode,
            smooth_iter=vis_config.visualize_vedo_smooth_iter,
            spacing=current_spacing,
            separate_windows=True,
            G=G
        )
    else:
        visualization.visualize_overlay(binary, skeleton, title="3D Skeleton Overlay Preview", vessel_opacity=0.3)
    print("Exiting pipeline as requested (Preview Mode).")
    sys.exit(0)

def _build_and_optimize_graph(skeleton, image, image_path, input_format, skel_config, graph_config, pipeline_config):
    sk = csr.Skeleton(skeleton)

    G, voxel_loops, loop_edges = graph.build_graph_segment_skan_stitched_loops(
        sk,
        skeleton,
        debug=pipeline_config.verbose_logging,
        use_padded_slicing=skel_config.use_padded_slicing,
        padding=skel_config.padded_slicing_padding,
    )
    G = graph.reconnect_secondary_loop_edges(G, skeleton, debug=pipeline_config.verbose_logging)
    visualization.visualize_edges_and_nodes(image, G, label_nodes=True, save_path=pipeline_config.plot_dir / "reconnect_secondary_loop_edges.png")
    
    G, _ = graph.optimise_graph_topology_fixed(
        G,
        voxel_loops,
        loop_edges,
        skeleton_data=skeleton,
        debug=pipeline_config.verbose_logging,
    )
    visualization.visualize_edges_and_nodes(image, G, label_nodes=True, save_path=pipeline_config.plot_dir / "optimise_graph_topology_fixed.png")
    
    G = graph.smart_multigraph_degree2_removal(
        G,
        skeleton,
        debug=pipeline_config.verbose_logging,
    )
    visualization.visualize_edges_and_nodes(image, G, label_nodes=True, save_path=pipeline_config.plot_dir / "smart_multigraph_degree2_removal.png")

    current_spacing = (1.0, 1.0, 1.0)
    if input_format == "tif":
        current_spacing = io.get_tif_spacing(image_path)
        print(f"  Using detected spacing for pruning (z,y,x): {current_spacing}")

    G = graph.prune_vascular_stubs(G, debug=pipeline_config.verbose_logging, voxel_size=current_spacing)
    G = graph.remove_edges_for_self_connected_nodes(G)
    visualization.visualize_edges_and_nodes(image, G, label_nodes=True, save_path=pipeline_config.plot_dir / "prune_vascular_stubs.png")
    
    if graph_config.keep_largest_component_only:
        n_before = G.number_of_nodes()
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
        n_after = G.number_of_nodes()
        if n_after < n_before:
            print(f"Final Graph Pruning: Removed {n_before - n_after} nodes in floating islands. Keeping largest component ({n_after} nodes).")
            
    return G

def _setup_boundary_conditions_and_hemodynamics(G, image, hemo_config, graph_config):
    starting_nodes = []
    output_nodes = []
    start_nodes, out_nodes = graph.select_boundary_terminal_nodes(
        G,
        image.shape,
        edge_percent=graph_config.edge_percent,
        end_percent=graph_config.end_percent,
        axis=graph_config.node_edge_axis,
    )
    starting_nodes.extend(start_nodes)
    output_nodes.extend(out_nodes)
    print(
        f"Auto-selected {len(starting_nodes)} STARTING_NODES "
        f"(top {graph_config.edge_percent}%) and {len(output_nodes)} OUTPUT_NODES "
        f"(bottom {graph_config.end_percent}%) along axis {graph_config.node_edge_axis}."
    )
    print(f"Starting nodes are: {starting_nodes}")
    print(f"Output nodes are: {output_nodes}")

    if starting_nodes and output_nodes:
        resistance_node_pair = (starting_nodes[0], output_nodes[0])
        print(f"Auto-selected resistance node_pair: {resistance_node_pair}")
    else:
        raise ValueError(f"No starting or output nodes found in input {graph_config.edge_percent}% or output {graph_config.end_percent}%")

    if starting_nodes:
        graph.assign_branch_orders(G, starting_nodes)
        poiseuille_model = hemodynamics.PoiseuilleModel(
            constriction_length=40.0,
            constriction_spacing=100.0,
        )
        if hemo_config.constrict_at_pericytes:
            poiseuille_model.set_poiseuille_edge_weights(
                G,
                custom_edges,
                edge_diameter=6.0,
                use_resistance=False,
            )
        else:
            poiseuille_model.set_poiseuille_weights_with_constrictions(
                G,
                hemo_config.diameter_by_branch_order,
            )
            
    return starting_nodes, output_nodes, resistance_node_pair

def _export_and_solve_hemodynamics(G, image, starting_nodes, output_nodes, resistance_node_pair, hemo_config, vis_config, pipeline_config):
    visualization.visualize_edges_and_nodes(image, G, label_nodes=True, save_path=pipeline_config.plot_dir / "pre_vtk.png")
    
    vtk_export = visualization.graph_to_vtk(G, pipeline_config.vtk_output_prefix)
    print("\n=== VTK Export ===")
    print(f"  Vessels:   {vtk_export['vessels_path']}")
    print(f"  Pericytes: {vtk_export['pericytes_path']}")
    print(f"  Nodes:     {vtk_export['nodes_path']}")
    print(f"  Counts: vessels={vtk_export['vessel_line_count']}, "
          f"pericytes={vtk_export['pericyte_count']}, nodes={vtk_export['node_count']}")
          
    if vis_config.visualize_vtk:
        visualization.visualize_vtk_network(
            vtk_export["vessels_path"],
            vtk_export["pericytes_path"],
            vtk_export["nodes_path"],
            show_nodes=False,
        )

    conductance, node_list = hemodynamics.build_conductance_matrix_from_graph(G)
    node_to_idx = {node_id: idx for idx, node_id in enumerate(node_list)}

    if pipeline_config.do_resistance_calculation:
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

    node_positions = nx.get_node_attributes(G, "pos")
    stats = statistics.compute_comprehensive_vessel_statistics(
        G,
        node_positions=node_positions,
        image_dimensions=image.shape,
    )

    print("\n=== Statistics ===")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    flow, vtk_export = hemodynamics.solve_flow_from_conductance_matrix(
        conductance,
        node_list,
        hemo_config.input_p_bc,
        hemo_config.output_p_bc,
        starting_nodes,
        output_nodes,
        vtk_export,
    )
    print("Flow through the network solved")
    print(f"Vtk file with flow data saved to: {vtk_export['vessels_path']}")

    if vis_config.visualize_results:
        _visualize_final_results(G, image, starting_nodes, vis_config)

def carotid_image_to_model(image_path: Path | str, 
                           pre_config: PreprocessingConfig = None,
                           skel_config: SkeletonConfig = None,
                           graph_config: GraphConfig = None,
                           hemo_config: HemodynamicsConfig = None,
                           vis_config: VisualizationConfig = None,
                           pipeline_config: PipelineConfig = None) -> None:
    if pre_config is None: pre_config = PreprocessingConfig()
    if skel_config is None: skel_config = SkeletonConfig()
    if graph_config is None: graph_config = GraphConfig()
    if hemo_config is None: hemo_config = HemodynamicsConfig()
    if vis_config is None: vis_config = VisualizationConfig()
    if pipeline_config is None: pipeline_config = PipelineConfig()

    image_path = Path(image_path)
    input_format = image_path.suffix[1:].lower()
    if input_format not in ["tif", "tiff", "h5"]:
        raise ValueError(f"Invalid image format: {input_format}")
    if input_format == "tiff":
        input_format = "tif"

    if pipeline_config.plot_dir is None:
        pipeline_config.plot_dir = Path("plots")
    if pipeline_config.vtk_output_prefix is None:
        pipeline_config.vtk_output_prefix = Path("outputs/resistance_network")

    logging.basicConfig(
        level=logging.DEBUG if pipeline_config.verbose_logging else logging.INFO,
        format="[%(levelname)s] %(message)s",
    )

    skeleton_path = image_path.with_name(f"{image_path.stem}_skeleton.npy")
    graph_path = image_path.with_name(f"{image_path.stem}_graph.pkl")
    projection_path = pipeline_config.plot_dir / "skeleton_projection.png"
    if not pipeline_config.plot_dir.exists():
        pipeline_config.plot_dir.mkdir(parents=True, exist_ok=True)

    if pipeline_config.do_skeletonize:
        image, binary = _load_and_preprocess_image(image_path, input_format, pre_config, skel_config, vis_config, pipeline_config)
        skeleton = _run_skeletonization_phase(binary, skel_config)
        np.save(skeleton_path, skeleton)
    else:
        skeleton = np.load(skeleton_path)
        image = tifffile.imread(image_path)

    if vis_config.visualize_results:
        visualization.visualize_skeleton(skeleton, save_path=projection_path)

    if pipeline_config.do_graph_building:
        G = _build_and_optimize_graph(skeleton, image, image_path, input_format, skel_config, graph_config, pipeline_config)
        
        if vis_config.visualize_overlay_preview and pipeline_config.do_skeletonize:
            _preview_overlay(binary, skeleton, G, image_path, input_format, vis_config)

        with graph_path.open("wb") as f:
            pickle.dump(G, f)
        print(f"Saved graph to: {graph_path}")
    else:
        if not graph_path.exists():
            raise FileNotFoundError(f"Graph file not found at {graph_path}. Set DO_GRAPH_BUILDING=True to generate it first.")
        with graph_path.open("rb") as f:
            G = pickle.load(f)
        print(f"Loaded graph from: {graph_path}")

    starting_nodes, output_nodes, resistance_node_pair = _setup_boundary_conditions_and_hemodynamics(G, image, hemo_config, graph_config)
    _export_and_solve_hemodynamics(G, image, starting_nodes, output_nodes, resistance_node_pair, hemo_config, vis_config, pipeline_config)
    
if __name__ == "__main__":
    plot_dir = BASE_PLOT_DIR / "carotid"
    
    # 1. Run Ilastik Segmentation (if enabled)
    if RUN_ILASTIK:
        # Example using explicit kwargs for clarity
        target_input_mask_path = run_ilastik_segmentation(
            ilastik_bin=ILASTIK_BINARY_PATH,
            project_path=ILASTIK_PROJECT_PATH,
            raw_image_path=RAW_IMAGE_PATH,
            frangi_image_path=FRANGI_IMAGE_PATH,
            output_dir=ILASTIK_OUTPUT_DIR,
            output_probabilities=ILASTIK_OUTPUT_PROBABILITIES
        )
        print(f"Acquired Ilastik output: {target_input_mask_path}")
    else:
        # Use the newly generated probabilities file
        target_input_mask_path = ILASTIK_OUTPUT_DIR / "C1-CB3-WKY-CB-A-2x2x2_vesselness_map_probs.tiff"

    # 2. Run the Network Pipeline
    pre_config = PreprocessingConfig(
        median_filter_size=MEDIAN_FILTER_SIZE,
        probability_smoothing_sigma=PROBABILITY_SMOOTHING_SIGMA,
        morphological_opening_radius=MORPHOLOGICAL_OPENING_RADIUS,
        enable_hysteresis_threshold=ENABLE_HYSTERESIS_THRESHOLD,
        hysteresis_threshold_low=HYSTERESIS_THRESHOLD_LOW,
        hysteresis_threshold_high=HYSTERESIS_THRESHOLD_HIGH,
        enable_hole_filling=ENABLE_HOLE_FILLING,
        ilastik_vessel_channel=ILASTIK_VESSEL_CHANNEL,
        enable_shannon_entropy=ENABLE_SHANNON_ENTROPY,
        shannon_entropy_threshold=SHANNON_ENTROPY_THRESHOLD,
    )
    skel_config = SkeletonConfig(
        closing_radius=SKELETON_CLOSING_RADIUS,
        bridge_gap_size=SKELETON_BRIDGE_GAP_SIZE,
        min_branch_length=SKELETON_MIN_BRANCH_LENGTH,
        max_bridge_distance=SKELETON_MAX_BRIDGE_DISTANCE,
        component_connectivity=SKELETON_COMPONENT_CONNECTIVITY,
        min_component_percent=SKELETON_MIN_COMPONENT_PERCENT,
        downsample_factor=SKELETON_DOWNSAMPLE_FACTOR,
        use_padded_slicing=SKELETON_USE_PADDED_SLICING,
        padded_slicing_padding=SKELETON_PADDED_SLICING_PADDING,
        prune_mask_before=SKELETON_PRUNE_MASK_BEFORE_SKELETONIZATION,
        sub_volume_percentage=SKELETON_SUB_VOLUME_PERCENTAGE,
        sub_volume_offset_z=SKELETON_SUB_VOLUME_CENTER_OFFSET_Z,
        sub_volume_offset_y=SKELETON_SUB_VOLUME_CENTER_OFFSET_Y,
        sub_volume_offset_x=SKELETON_SUB_VOLUME_CENTER_OFFSET_X,
        bundle_scan_size=SKELETON_BUNDLE_SCAN_SIZE,
        bundle_density_fraction=SKELETON_BUNDLE_DENSITY_FRACTION,
        bundle_max_connections=SKELETON_BUNDLE_MAX_CONNECTIONS,
        bundle_hub_min_spacing=SKELETON_BUNDLE_HUB_MIN_SPACING,
    )
    graph_config = GraphConfig(
        keep_largest_component_only=GRAPH_KEEP_LARGEST_COMPONENT_ONLY,
        edge_percent=EDGE_PERCENT,
        end_percent=END_PERCENT,
        node_edge_axis=NODE_EDGE_AXIS,
        starting_nodes=STARTING_NODES,
        output_nodes=OUTPUT_NODES,
    )
    hemo_config = HemodynamicsConfig(
        constrict_at_pericytes=CONSTRICT_AT_PERICYTES,
        input_p_bc=INPUT_P_BC,
        output_p_bc=OUTPUT_P_BC,
        diameter_by_branch_order=DIAMETER_BY_BRANCH_ORDER,
    )
    vis_config = VisualizationConfig(
        visualize_results=VISUALIZE_RESULTS,
        visualize_mask_only=VISUALIZE_MASK_ONLY,
        visualize_vedo=VISUALIZE_VEDO,
        visualize_overlay_preview=VISUALIZE_OVERLAY_PREVIEW,
        visualize_vedo_mode=VISUALIZE_VEDO_MODE,
        visualize_vedo_smooth_iter=VISUALIZE_VEDO_SMOOTH_ITER,
        visualize_vedo_spacing=VISUALIZE_VEDO_SPACING,
        visualize_vedo_auto_spacing=VISUALIZE_VEDO_AUTO_SPACING,
        visualize_vedo_opacity=VISUALIZE_VEDO_OPACITY,
        visualize_mask_opacity=VISUALIZE_MASK_OPACITY,
        visualize_vtk=VISUALIZE_VTK,
        visualize_post_processed_mask=VISUALIZE_POST_PROCESSED_MASK,
    )
    pipeline_config = PipelineConfig(
        do_skeletonize=DO_SKELETONIZE,
        do_graph_building=DO_GRAPH_BUILDING,
        do_resistance_calculation=DO_RESISTANCE_CALCULATION,
        verbose_logging=VERBOSE_LOGGING,
        min_branch_length=MIN_BRANCH_LENGTH,
        vtk_output_prefix=VTK_OUTPUT_PREFIX,
        plot_dir=plot_dir,
    )

    carotid_image_to_model(
        image_path=target_input_mask_path,
        pre_config=pre_config,
        skel_config=skel_config,
        graph_config=graph_config,
        hemo_config=hemo_config,
        vis_config=vis_config,
        pipeline_config=pipeline_config
    )

