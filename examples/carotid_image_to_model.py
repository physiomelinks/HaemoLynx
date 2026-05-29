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

# Setup logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Ensure package is importable when running from repo root.
root_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


from ImageLynx import graph, haemodynamics, io, preprocessing, statistics, visualization 

# ---------------------------
# Beginner-friendly settings
# ---------------------------

# Ilastik configuration settings
RUN_ILASTIK = False
ILASTIK_OUTPUT_PROBABILITIES = True # Set to True for Probabilities, False for Simple Segmentation
ILASTIK_BINARY_PATH = "/home/dsas627/Desktop/ilastik-1.4.1rc2-gpu-Linux/run_ilastik.sh"
ILASTIK_PROJECT_PATH = root_dir / "examples" / "images" / "cb_wky_2x2x2_A.ilp"
RAW_IMAGE_DIR = root_dir / "examples" / "images" / "ilastik_batch_processing_input_images"
ILASTIK_OUTPUT_DIR = root_dir / "examples" / "images" / "ilastik_batch_processing_output_images"

# Paths for multi-input Ilastik features (e.g., Raw + Frangi)
RAW_IMAGE_PATH = RAW_IMAGE_DIR / "C1-CB3-WKY-CB-A-2x2x2_vessels.tif"
FRANGI_IMAGE_PATH = RAW_IMAGE_DIR / "C1-CB3-WKY-CB-A-2x2x2_vesselness_map.tif"

INPUT_PATH = None
H5_DATASET_NAME = None  # For h5 input, e.g. "data"

"""Configuration defaults for diameter maps."""

# Diameter by branch order (dict with d1 and d2 for pericyte constriction simulation)
DIAMETER_BY_BRANCH_ORDER = {
    "DEFAULT": {"d1": 4.0, "d2": 4.0},
}

DIAMETER_BY_BRANCH_ORDER_ENHANCED = None

# These are vesses that constrict differently (e.g. endoneurial vessels).
custom_edges= []  



@dataclass
class PreprocessingConfig:
    """Configuration for probability map noise removal, smoothing, and binary thresholding."""
    median_filter_size: int = 7
    probability_smoothing_sigma: float = 0.0
    morphological_opening_radius: int = 1
    morphological_closing_radius: int = 0
    enable_hysteresis_threshold: bool = True
    hysteresis_threshold_low: float = 0.2
    hysteresis_threshold_high: float = 0.4
    enable_hole_filling: bool = True
    ilastik_vessel_channel: int = 0
    enable_shannon_entropy: bool = True
    shannon_entropy_threshold: float = 0.95

@dataclass
class SkeletonConfig:
    """Configuration for 3D skeletonization, artifact pruning, and topological cleanup."""
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
    sub_volume_percentage: float = 0.15
    sub_volume_offset_z: float = 0.0
    sub_volume_offset_y: float = 0.0
    sub_volume_offset_x: float = 0.0
    bundle_scan_size: int = 9
    bundle_density_fraction: float = 0.025
    bundle_max_connections: int = 5
    bundle_hub_min_spacing: int = 0
    smoothing_alpha: float = 0.75
    prune_by_tortuosity: float = 5.0
    core_dead_end_resolution_mode: str = "none"
    core_safe_zone_percent: float = 5.0
    core_stitch_max_distance_um: float = 15.0
    core_stitch_max_degree: int = 4

@dataclass
class GraphConfig:
    """Configuration for mathematical graph generation and boundary node selection."""
    keep_largest_component_only: bool = True
    edge_percent: float = 25.0
    end_percent: float = 25.0
    node_edge_axis: int = 0
    starting_nodes: list = field(default_factory=list)
    output_nodes: list = field(default_factory=list)

@dataclass
class HaemodynamicsConfig:
    """Configuration for fluid dynamics simulation, pressures, and vessel diameters."""
    constrict_at_pericytes: bool = True
    # To match dimensions of viscosity (mPa*s) and lengths (um) yielding flow (Q) in um^3/s:
    # Pressure must be provided in milliPascals (mPa).
    input_p_bc: float = 13.332e6 ### mPa (MAP of 100 mmHg = 13.332 kPa = 13.332e6 mPa)
    output_p_bc: float = 0.27e6 ### mPa (CVP of 2 mmHg = 0.267 kPa = 0.27e6 mPa)
    diameter_by_branch_order: dict = field(default_factory=dict)
    
    # --- Baseline Radius Assignment ---
    radius_assignment_mode: str = "fwhm_radius" # Options: "fwhm_radius" or "constant_radius"
    constant_radius_um: float = 5.0 # Used only if radius_assignment_mode == "constant_radius"
    
    # --- Sphincter / Constriction Configuration ---
    constriction_mode: str = "sphincter"  # Options: "sphincter" or "periodic"
    sphincter_length_um: float = 5.0      # Physical length of the pinched region (um)
    
    # Severity modifiers (1.0 = no constriction, 0.5 = 50% constriction)
    intimal_cushion_constriction_ratio: float = 0.60
    pre_capillary_constriction_ratio: float = 0.50
    pre_capillary_topological_offset: int = 1
    
    # --- FWHM Ray-Casting Configuration ---
    fwhm_sample_spacing_along_edge_um: float = 2.0
    fwhm_transverse_profile_step_um: float = 0.5
    fwhm_transverse_half_extent_um: float = 15.0
    
    # --- Rheology Solver Parameters ---
    rheology_max_iterations: int = 15
    rheology_tolerance: float = 1e-4
    blood_plasma_viscosity_cP: float = 1.2

    def __post_init__(self):
        """Validates configuration bounds to prevent mathematical crashes in the physics engines."""
        if self.input_p_bc <= self.output_p_bc:
            raise ValueError(f"Input pressure ({self.input_p_bc}) must be strictly greater than Output pressure ({self.output_p_bc}).")

        if self.radius_assignment_mode not in ("fwhm_radius", "constant_radius"):
            raise ValueError(f"radius_assignment_mode must be 'fwhm_radius' or 'constant_radius', got: {self.radius_assignment_mode}")

        if self.constriction_mode not in ("sphincter", "periodic"):
            raise ValueError(f"constriction_mode must be 'sphincter' or 'periodic', got: {self.constriction_mode}")
            
        if self.constant_radius_um <= 0.0:
            raise ValueError("constant_radius_um must be strictly positive.")

        if self.sphincter_length_um < 0.0:
            raise ValueError("sphincter_length_um cannot be negative.")

        # Prevent 0.0 constriction ratios (which means diameter=0 -> infinite resistance -> matrix crash)
        if self.intimal_cushion_constriction_ratio <= 0.01:
            print(f"Warning: intimal_cushion_constriction_ratio {self.intimal_cushion_constriction_ratio} is too low. Clamping to 0.01 to prevent singularities.")
            self.intimal_cushion_constriction_ratio = 0.01

        if self.pre_capillary_constriction_ratio <= 0.01:
            print(f"Warning: pre_capillary_constriction_ratio {self.pre_capillary_constriction_ratio} is too low. Clamping to 0.01 to prevent singularities.")
            self.pre_capillary_constriction_ratio = 0.01

@dataclass
class PerfusionConfig:
    """Configuration for steady-state tissue diffusion modeling."""
    do_perfusion_modeling: bool = True
    grid_resolution_xyz: tuple[float, float, float] = (10.0, 10.0, 10.0) # micrometers
    grid_opacity: float = 0.3 # 0.0 to 1.0

    # Physiological Constants (from CellML Blueprints)
    # -----------------------------------------------
    # sigma_diff: Diffusion coefficient of O2 in tissue (m^2/s)
    sigma_diff: float = 1.5e-9 
    
    # Endothelial Barrier Model
    use_endothelial_barrier_model: bool = True
    permeability_o2_cm_s: float = 1.0e-4 # Permeability coefficient for O2 (cm/s)
    
    # Multi-Species Coupling (CO2 & pH)
    use_multi_species_model: bool = True
    sigma_diff_co2: float = 3.0e-8 # Tissue diffusion coefficient for CO2 (m^2/s) - diffuses ~20x faster than O2
    permeability_co2_cm_s: float = 2.0e-3 # Permeability coefficient for CO2 (cm/s)
    respiratory_quotient: float = 0.82 # Ratio of CO2 produced to O2 consumed
    
    # Blood & Tissue Baselines
    systemic_hematocrit: float = 0.45
    po2_arterial_mmHg: float = 100.0
    pco2_arterial: float = 40.0 # Arterial PCO2 (mmHg)
    hco3_tissue: float = 24.0 # Tissue bicarbonate buffer (mmol/L)
    
    # Picard Solver Parameters
    picard_max_iterations: int = 50
    picard_tolerance: float = 1e-4
    
    # M_max: Maximum metabolic consumption rate (mmol / L / s)
    M_max: float = 0.005

    # k_reduce: Metabolic reduction constant for hypoxic zones (per mmol)
    k_reduce: float = 0.1

    # C_arterial: Oxygen concentration entering the network (mmol / L)
    C_arterial: float = 0.13


@dataclass
class VisualizationConfig:
    """Configuration for 3D/2D visualization tools and interactive previews."""
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
    """Top-level configuration for enabling/disabling major pipeline phases and I/O paths."""
    do_skeletonize: bool = True
    do_graph_building: bool = True
    do_resistance_calculation: bool = True
    run_benchmarking: bool = False
    optimize_preprocessing_trials: int = 0
    optimize_patience: int = 15
    verbose_logging: bool = False
    min_branch_length: int = 10
    vtk_output_prefix: Path = Path(__file__).resolve().parents[1] / "examples" / "outputs" / "resistance_network"
    plot_dir: Path = Path(__file__).resolve().parents[1] / "examples" / "plots" / "carotid" 

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
    """Helper function to visualize the raw 3D image volume before any processing."""
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
    """Helper function to visualize the fully cleaned binary mask before skeletonization."""
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
    """Helper function to generate final 2D summary plots (node degree, branch orders)."""
    visualization.plot_node_degree_distribution(G)
    visualization.visualize_edges_and_nodes(image, G)
    
    if starting_nodes:
        visualization.visualize_geometry_with_branch_orders(
            image,
            G,
            group_above=8,
        )

def _apply_preprocessing_filters(raw_prob_map, entropy_map, pre_config_dict):
    """Applies preprocessing filters returning a materialized binary mask."""
    image = raw_prob_map.copy()
    
    if entropy_map is not None and pre_config_dict.get("enable_shannon_entropy", True):
        threshold = pre_config_dict.get("shannon_entropy_threshold", 0.95)
        uncertain_mask = entropy_map > threshold
        image[uncertain_mask] = 0.0
        
    # --- Virtual Z-Padding (Boundary Caging Fix) ---
    # To prevent morphological filters from smearing/caging open vessels at the cut plane,
    # we duplicate the top/bottom slices outward to create "virtual tunnels".
    z_pad = 10
    image = np.pad(image, pad_width=((z_pad, z_pad), (0, 0), (0, 0)), mode='edge')
    
    median_size = pre_config_dict.get("median_filter_size", 0)
    if median_size > 0:
        image = preprocessing.median_filter_image(image, size=median_size)
        
    opening_radius = pre_config_dict.get("morphological_opening_radius", 0)
    if opening_radius > 0:
        image = preprocessing.morphological_opening(image, radius=opening_radius)
        
    closing_radius = pre_config_dict.get("morphological_closing_radius", 0)
    if closing_radius > 0:
        image = preprocessing.morphological_closing(image, radius=closing_radius)
        
    if pre_config_dict.get("probability_smoothing_sigma", 0) > 0:
        image = preprocessing.smooth_probability_map(image, sigma=pre_config_dict["probability_smoothing_sigma"])
        
    if pre_config_dict.get("enable_hysteresis_threshold", True):
        binary = preprocessing.hysteresis_threshold(
            image, 
            low=pre_config_dict.get("hysteresis_threshold_low", 0.2), 
            high=pre_config_dict.get("hysteresis_threshold_high", 0.4)
        )
    else:
        from skimage.filters import threshold_otsu
        binary = image > threshold_otsu(image)
        
    if pre_config_dict.get("enable_hole_filling", True):
        binary = preprocessing.skeleton.fill_holes_3d(binary)
        
    # --- Remove Virtual Z-Padding ---
    # Slicing the padding off cleanly amputates the vessels, guaranteeing open Degree-1 dead ends.
    image = image[z_pad:-z_pad, :, :]
    binary = binary[z_pad:-z_pad, :, :]
        
    return image, binary

def _load_and_preprocess_image(image_path, input_format, pre_config, skel_config, vis_config, pipeline_config):
    """
    Phase 1: Loads the image, handles 4D channels/entropy, crops the ROI,
    removes noise, and applies hysteresis thresholding to generate a binary mask.
    """
    # Load the 3D or 4D volume using lazy loading to save memory
    if input_format == "tif":
        image = io.load_3d_tif(image_path, lazy=True)
    elif input_format == "h5":
        if not H5_DATASET_NAME:
            raise ValueError("Set H5_DATASET_NAME when INPUT_FORMAT is 'h5'.")
        image = io.load_3d_h5(image_path, H5_DATASET_NAME, lazy=True)
    else:
        raise ValueError("INPUT_FORMAT must be 'tif' or 'h5'.")

    is_lazy = preprocessing.image._is_dask_array(image)
    if is_lazy:
        import dask.array as da
        logger.info("Using Dask for lazy out-of-core preprocessing.")

    print(f"Loaded image shape: {image.shape}")

    # Slice the array into a smaller sub-volume to speed up testing/debugging
    # (Virtual operation if image is a Dask array)
    if 0 < skel_config.sub_volume_percentage < 1.0 or skel_config.sub_volume_offset_z != 0 or \
       skel_config.sub_volume_offset_y != 0 or skel_config.sub_volume_offset_x != 0:
        
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
    # If the image is 4D (e.g., from Ilastik with multiple probability channels)
    if image.ndim == 4:
        # Calculate entropy before extracting the vessel channel
        if pre_config.enable_shannon_entropy:
            entropy_map = preprocessing.calculate_entropy_map(image)

        dims = np.array(image.shape)
        c_axis = np.argmin(dims)
        print(f"Detected 4D image. Assuming channel is at axis {c_axis} (size {dims[c_axis]}).")
        
        # Extract only the specific channel containing our target vessel probabilities
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
            # We must compute entropy_map for the optimizer
            if is_lazy:
                print("Computing entropy map for preprocessing...")
                entropy_map = entropy_map.compute()

    # Materialize the raw probability map for the optimizer
    if is_lazy:
        print("Computing cropped raw probability map...")
        raw_prob_map = image.compute()
    else:
        raw_prob_map = image.copy()
        
    print(f"Image probability range: min={raw_prob_map.min():.4f}, max={raw_prob_map.max():.4f}, mean={raw_prob_map.mean():.4f}")

    if vis_config.visualize_mask_only:
        _preview_raw_volume(raw_prob_map, image_path, input_format, vis_config)

    # --- Optuna Preprocessing Optimization ---
    if pipeline_config.optimize_preprocessing_trials > 0:
        import ImageLynx.statistics.benchmarking as benchmarking
        import ImageLynx.statistics.auto_tuner as auto_tuner
        import copy
        
        print(f"\n--- Launching Optuna Preprocessing Auto-Tuner ({pipeline_config.optimize_preprocessing_trials} trials) ---")
        
        def pre_eval_callback(suggested_kwargs):
            # 1. Apply filters
            test_config_dict = pre_config.__dict__.copy()
            test_config_dict.update(suggested_kwargs)
            
            _, test_binary = _apply_preprocessing_filters(raw_prob_map, entropy_map, test_config_dict)
            
            # 2. Score the mask
            bench_results = benchmarking.run_all_preprocessing_benchmarks(raw_prob_map, test_binary)
            return bench_results

        best_pre_params = auto_tuner.run_optuna_preprocessing_optimization(
            pre_eval_callback,
            n_trials=pipeline_config.optimize_preprocessing_trials,
            output_dir=pipeline_config.vtk_output_prefix.parent,
            patience=pipeline_config.optimize_patience
        )
        
        print("\nApplying optimal preprocessing parameters to pipeline...")
        for k, v in best_pre_params.items():
            setattr(pre_config, k, v)
    # ------------------------------------------

    # Apply the (potentially optimized) filters
    filtered_image, binary = _apply_preprocessing_filters(raw_prob_map, entropy_map, pre_config.__dict__)

    # Smooth the bumpy outer walls
    if skel_config.closing_radius > 0:
        binary = preprocessing.skeleton.close_binary_mask(binary, radius=skel_config.closing_radius)
    
    # Draw localized bridges across tiny gaps
    if skel_config.bridge_gap_size > 0:
        binary = preprocessing.skeleton.bridge_gaps(binary, max_gap=skel_config.bridge_gap_size)

    # Delete all floating background noise blobs
    if skel_config.prune_mask_before > 0:
        print(f"Pruning binary mask to keep largest {skel_config.prune_mask_before} components...")
        binary = preprocessing.skeleton.keep_largest_mask_components(
            binary, n_components=skel_config.prune_mask_before, connectivity=skel_config.component_connectivity
        )

    if vis_config.visualize_post_processed_mask:
        _preview_post_processed_mask(binary, image_path, input_format, vis_config, pipeline_config)
    
    # Return the materialized image and binary mask
    return (image.compute() if preprocessing.image._is_dask_array(image) else image), binary

def _run_skeletonization_phase(binary, skel_config):
    """
    Phase 2: Converts the solid binary mask into a 1D centerline skeleton,
    then runs topological cleanup (bundle collapsing, stub removal).
    """
    # Convert the thick 3D tubes into 1-voxel-wide centerlines
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

    # Clean the skeleton topology: remove tiny stubs, collapse dense spiderweb bundles into clean hubs,
    # and filter out floating fragments based on their volume percentage.
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

def _preview_overlay(binary, skeleton, G, image_path, input_format, vis_config, perf_config=None):
    """Helper function to launch the 4-panel Vedo verification viewer."""
    print(f"Visualizing 3D overlay PREVIEW with optimized graph (mask opacity=0.3). Close window to exit.")
    import sys
    
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
            G=G,
            perf_config=perf_config
        )
        sys.exit(0)
    else:
        visualization.visualize_overlay(binary, skeleton, title="3D Skeleton Overlay Preview", vessel_opacity=0.3)
    print("Exiting pipeline as requested (Preview Mode).")
    sys.exit(0)

def _build_and_optimize_graph(skeleton, image, image_path, input_format, skel_config, graph_config, pipeline_config):
    """
    Phase 3: Extracts a mathematical graph (nodes/edges) from the physical skeleton,
    and rigorously optimizes the topology (merging close nodes, removing degree-2 paths).
    """
    # Trace the voxel centerline to identify mathematical Nodes (intersections) and Edges (vessel paths)
    sk = csr.Skeleton(skeleton)

    # Build the networkx MultiGraph. Crucially, detect and stitch tiny 1-voxel circular artifacts (voxel loops) so the graph doesn't shatter.
    G, voxel_loops, loop_edges = graph.build_graph_segment_skan_stitched_loops(
        sk,
        skeleton,
        debug=pipeline_config.verbose_logging,
        use_padded_slicing=skel_config.use_padded_slicing,
        padding=skel_config.padded_slicing_padding,
    )
    # Ensure any branches that touched the stitched loop are properly reconnected to the new central hub node
    G = graph.reconnect_secondary_loop_edges(G, skeleton, debug=pipeline_config.verbose_logging)
    visualization.visualize_edges_and_nodes(image, G, label_nodes=True, save_path=pipeline_config.plot_dir / "reconnect_secondary_loop_edges.png")
    
    # Merge nodes that are physically right next to each other, and resolve "triangle" intersections into clean "Y" bifurcations
    G, _ = graph.optimise_graph_topology_fixed(
        G,
        voxel_loops,
        loop_edges,
        skeleton_data=skeleton,
        debug=pipeline_config.verbose_logging,
    )
    visualization.visualize_edges_and_nodes(image, G, label_nodes=True, save_path=pipeline_config.plot_dir / "optimise_graph_topology_fixed.png")
    
    # Delete redundant middle nodes sitting on straight lines, merging their edges, without destroying the physical curvature of the vessel
    G = graph.smart_multigraph_degree2_removal(
        G,
        skeleton,
        debug=pipeline_config.verbose_logging,
    )
    visualization.visualize_edges_and_nodes(image, G, label_nodes=True, save_path=pipeline_config.plot_dir / "smart_multigraph_degree2_removal.png")

    # Automatically detect the physical image resolution to accurately calculate vessel lengths in microns
    current_spacing = (1.0, 1.0, 1.0)
    if input_format == "tif":
        current_spacing = io.get_tif_spacing(image_path)
        print(f"  Using detected spacing for pruning (z,y,x): {current_spacing}")

    # Delete dead-end branches (stubs) that are physically shorter than the minimum branch length threshold
    G = graph.prune_vascular_stubs(G, debug=pipeline_config.verbose_logging, voxel_size=current_spacing)
    # Delete impossible edges that start and end on the exact same node with no other connections
    G = graph.remove_edges_for_self_connected_nodes(G)
    visualization.visualize_edges_and_nodes(image, G, label_nodes=True, save_path=pipeline_config.plot_dir / "prune_vascular_stubs.png")
    
    # --- Plan A & B: Core Dead-End Resolution ---
    if skel_config.core_dead_end_resolution_mode in ["eradicate", "stitch"]:
        voxel_size_xyz = tuple(float(v) for v in current_spacing)
        stats = graph.resolve_core_dead_ends(
            G,
            image_shape=image.shape,
            voxel_size_xyz=voxel_size_xyz,
            mode=skel_config.core_dead_end_resolution_mode,
            safe_zone_percent=skel_config.core_safe_zone_percent,
            max_stitch_distance_um=skel_config.core_stitch_max_distance_um,
            max_degree=skel_config.core_stitch_max_degree
        )
        print(f"\n--- Core Dead-End Resolution [{skel_config.core_dead_end_resolution_mode.upper()} MODE] ---")
        print(f"Initial edges: {stats.get('initial_edges', 0)}")
        if stats.get('edges_added', 0) > 0:
            print(f"Stitched edges added: {stats['edges_added']} ({stats['edges_added_pct']}%)")
        if stats.get('edges_removed', 0) > 0:
            print(f"Eradicated edges removed: {stats['edges_removed']} ({stats['edges_removed_pct']}%)")
            if stats.get('fallback_eradicated', 0) > 0:
                print(f" (Includes {stats['fallback_eradicated']} un-stitchable edges safely eradicated as fallback)")
    # --------------------------------------------
    
    # Smooth the physical 3D paths (voxels) of all edges using B-Splines to ensure realistic biological curvature
    print("Smoothing all edge centerlines in parallel using Joblib and B-Splines...")
    smooth_stats = graph.smooth_graph_edge_centerlines_continuous(
        G,
        skeleton,
        smoothing_method="bspline",
        bspline_smoothness=0.75,
        debug=pipeline_config.verbose_logging,
        voxel_size=current_spacing
    )
    print(f"Continuous centerline smoothing summary: {smooth_stats}")

    # Final safety net: delete any floating graph islands that were accidentally severed during the topological optimization steps
    if graph_config.keep_largest_component_only:
        n_before = G.number_of_nodes()
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
        n_after = G.number_of_nodes()
        if n_after < n_before:
            print(f"Final Graph Pruning: Removed {n_before - n_after} nodes in floating islands. Keeping largest component ({n_after} nodes).")
            
    return G

def _setup_boundary_conditions_and_haemodynamics(G, image, hemo_config, graph_config, image_path, input_format):
    """
    Phase 4: Selects inlet/outlet nodes, calculates branch hierarchies,
    and assigns physical resistances based on Poiseuille's law.
    """
    starting_nodes = []
    output_nodes = []
    # Auto-detect Inlet (start) and Outlet (output) nodes by finding dead-ends at the physical boundaries of the image volume
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
        # Crawl the network from the inlets to assign a Branch Order (e.g. B01, B02) to every vessel based on bifurcations passed
        # 1. Capture the BFS results to see exactly how deep this network goes
        bo_results = graph.assign_branch_orders(G, starting_nodes)
        unique_branch_orders = list(bo_results["branch_order_counts"].keys())
        
        print(f"Auto-detected {len(unique_branch_orders)} unique branch orders in the network:")
        print(f"  {sorted(unique_branch_orders)}")

        # 2. Extract the current dictionary and the user's intended default fallback
        current_diam_dict = hemo_config.diameter_by_branch_order
        default_diam_vals = current_diam_dict.get("DEFAULT", {"d1": 4.0, "d2": 4.0})

        # 3. Smart-fill using 3-Point Boundary Fitted Exponential Scaling
        import re
        
        # Define your three biological anchor points
        D_start_d1, D_start_d2 = 15.0, 15.0  # Arterial Inlet (B01)
        D_mid_d1,   D_mid_d2   = 4.0,  2.0   # Capillary Bed (Middle)
        D_end_d1,   D_end_d2   = 20.0, 20.0  # Venous Outlet (Max Branch)
        
        # Find the network boundaries
        all_generations = []
        for bo_label in unique_branch_orders:
            match = re.search(r'\d+', bo_label)
            if match:
                all_generations.append(int(match.group()))
                
        n_start = min(all_generations) if all_generations else 1
        n_end = max(all_generations) if all_generations else 1
        n_mid = (n_start + n_end) / 2.0  
        
        # Calculate the custom exponential scaling factors
        delta_art = max(n_mid - n_start, 1e-6) # Prevent division by zero
        delta_ven = max(n_end - n_mid, 1e-6)
        
        factor_art_d1 = (D_start_d1 / D_mid_d1) ** (1.0 / delta_art)
        factor_art_d2 = (D_start_d2 / D_mid_d2) ** (1.0 / delta_art)
        
        factor_ven_d1 = (D_end_d1 / D_mid_d1) ** (1.0 / delta_ven)
        factor_ven_d2 = (D_end_d2 / D_mid_d2) ** (1.0 / delta_ven)

        for bo_label in unique_branch_orders:
            if bo_label not in current_diam_dict:
                try:
                    match = re.search(r'\d+', bo_label)
                    if match:
                        n = int(match.group())
                        
                        if n <= n_mid:
                            # We are on the arterial side (shrinking)
                            dist = n_mid - n
                            calc_d1 = D_mid_d1 * (factor_art_d1 ** dist)
                        else:
                            # We are on the venous side (expanding)
                            dist = n - n_mid
                            calc_d1 = D_mid_d1 * (factor_ven_d1 ** dist)
                            
                        # Apply Physiological Sphincter Constrictions (d2)
                        # By default, vessels are unconstricted (d2 = d1)
                        calc_d2 = calc_d1
                        
                        # (i) Intimal Cushion at the origin of the carotid (B01)
                        if n == 1:
                            calc_d2 = calc_d1 * hemo_config.intimal_cushion_constriction_ratio
                            
                        # (ii) Pre-Capillary Sphincters at the origin of capillary beds
                        # We define the transition zone (n_mid) as the start of the capillary beds
                        elif n == int(n_mid) or n == int(n_mid) - hemo_config.pre_capillary_topological_offset:
                            calc_d2 = calc_d1 * hemo_config.pre_capillary_constriction_ratio
                        
                        current_diam_dict[bo_label] = {"d1": calc_d1, "d2": calc_d2}
                    else:
                        current_diam_dict[bo_label] = default_diam_vals
                except Exception as e:
                    print(f"Warning: Could not apply boundary-fitted law to {bo_label}: {e}")
                    current_diam_dict[bo_label] = default_diam_vals
                
        # 4. Save the safely expanded dictionary back to the config
        hemo_config.diameter_by_branch_order = current_diam_dict
        
        # 5. Automatically measure exact physical vessel diameters from the raw image
        if hemo_config.radius_assignment_mode == "fwhm_radius":
            print("Measuring exact physical vessel diameters using 3D FWHM ray-casting...")
            
            # Use the detected spacing (or default)
            fwhm_spacing = io.get_tif_spacing(image_path) if input_format == "tif" else (1.0, 1.0, 1.0)
            
            # We pass the pre-loaded, pre-cropped 3D `image` array directly into the FWHM algorithm.
            stats_dict = haemodynamics.measure_edge_diameters_fwhm_from_raw_tiff(
                G,
                raw_tiff_path=image,  # We pass the numpy array directly (the backend has been patched to handle this)
                voxel_size_xyz=fwhm_spacing,
                sample_spacing_along_edge_um=hemo_config.fwhm_sample_spacing_along_edge_um,
                transverse_profile_step_um=hemo_config.fwhm_transverse_profile_step_um,
                transverse_half_extent_um=hemo_config.fwhm_transverse_half_extent_um,
            )
            print(f"FWHM measurement complete. Processed {len(stats_dict)} edges.")
        else:
            print(f"Bypassing FWHM ray-casting. Using '{hemo_config.radius_assignment_mode}' ({hemo_config.constant_radius_um} um)")

        # Initialize the haemodynamics solver to calculate physical flow resistance using Poiseuille's Law
        poiseuille_model = haemodynamics.PoiseuilleModel(
            constriction_length=hemo_config.sphincter_length_um,
            constriction_spacing=100.0, # Not used in sphincter mode
            mode=hemo_config.constriction_mode
        )
        if hemo_config.constrict_at_pericytes:
            poiseuille_model.set_poiseuille_resistances_with_constrictions(
                G,
                hemo_config.diameter_by_branch_order,
                radius_assignment_mode=hemo_config.radius_assignment_mode,
                constant_radius_um=hemo_config.constant_radius_um
            )
        else:
            # For non-constricted mode, extract d1 from the config dicts
            simple_diameters = {
                k: (v["d1"] if isinstance(v, dict) else v)
                for k, v in hemo_config.diameter_by_branch_order.items()
            }
            poiseuille_model.set_poiseuille_resistances(
                G,
                simple_diameters,
                radius_assignment_mode=hemo_config.radius_assignment_mode,
                constant_radius_um=hemo_config.constant_radius_um
            )
            
    return starting_nodes, output_nodes, resistance_node_pair

def _export_and_solve_haemodynamics(G, image, binary, starting_nodes, output_nodes, resistance_node_pair, hemo_config, vis_config, pipeline_config, perf_config=None):
    """
    Phase 5: Builds the Laplacian matrix, solves the flow equations,
    calculates comprehensive statistics, and exports all data to VTK files.
    """
    if pipeline_config.run_benchmarking and binary is not None:
        import ImageLynx.statistics.benchmarking as benchmarking
        # Use the detected spacing (or default)
        voxel_size_xyz = (1.0, 1.0, 1.0)
        
        bench_results = benchmarking.run_all_benchmarks(G, binary, voxel_size_xyz)
        print("\n=== Skeletonization Benchmarks ===")
        import json
        print(json.dumps(bench_results, indent=2))
        
    if pipeline_config.plot_dir is not None:
        visualization.visualize_edges_and_nodes(image, G, label_nodes=True, save_path=pipeline_config.plot_dir / "pre_vtk.png")
    
    # Export the geometric network to standardized VTK PolyData files for viewing in ParaView
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

    # Convert the networkx graph into a massive symmetric Conductance Matrix representing flow ease between all nodes
    conductance, node_list = haemodynamics.build_conductance_matrix_from_graph(G)
    node_to_idx = {node_id: idx for idx, node_id in enumerate(node_list)}

    # Optional: Calculate the exact effective mathematical resistance between a single specific inlet and outlet pair
    if pipeline_config.do_resistance_calculation:
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

    node_positions = nx.get_node_attributes(G, "pos")
    # Calculate physical and topological statistics (e.g. total length, mean tortuosity, degree distribution)
    stats = statistics.compute_comprehensive_vessel_statistics(
        G,
        node_positions=node_positions,
        image_dimensions=image.shape,
    )

    print("\n=== Statistics ===")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Inject boundary pressures and solve the system of linear equations to find pressure at every node and flow in every edge
    print("Running Iterative Flow-Hematocrit solver (Phase Separation and Fåhræus–Lindqvist effect)...")
    import ImageLynx.haemodynamics.rheology as rheo
    G, final_pressure = rheo.solve_coupled_flow_and_hematocrit(
        G,
        starting_nodes,
        output_nodes,
        hemo_config.input_p_bc,
        hemo_config.output_p_bc,
        systemic_hematocrit=perf_config.systemic_hematocrit,
        max_iterations=hemo_config.rheology_max_iterations,
        tolerance=hemo_config.rheology_tolerance
    )
    
    # We still need to export the final flow data to VTK
    # Let's rebuild the final conductance matrix now that the iterative solver updated all the resistances
    conductance, _ = haemodynamics.build_conductance_matrix_from_graph(G)
    flow, vtk_export = haemodynamics.solve_flow_from_conductance_matrix(
        conductance,
        node_list,
        hemo_config.input_p_bc,
        hemo_config.output_p_bc,
        starting_nodes,
        output_nodes,
        vtk_export,
    )
    
    # Now manually inject hematocrit and viscosity into the VTK file
    import pyvista as pv
    vessels = pv.read(vtk_export['vessels_path'])
    edge_u = np.asarray(vessels.cell_data.get("edge_u", []))
    edge_v = np.asarray(vessels.cell_data.get("edge_v", []))
    edge_k = np.asarray(vessels.cell_data.get("edge_k", np.zeros_like(edge_u)))
    
    hematocrit_array = np.full(vessels.n_cells, 0.45, dtype=float)
    viscosity_array = np.full(vessels.n_cells, 1.2, dtype=float)
    wss_array = np.zeros(vessels.n_cells, dtype=float)
    
    for ii in range(vessels.n_cells):
        u, v, k = int(edge_u[ii]), int(edge_v[ii]), int(edge_k[ii])
        if G.has_edge(u, v, k):
            hematocrit_array[ii] = G[u][v][k].get("hematocrit", 0.45)
            viscosity_array[ii] = G[u][v][k].get("viscosity", 1.2)
            wss_array[ii] = G[u][v][k].get("wall_shear_stress_pa", 0.0)
            
    vessels.cell_data["hematocrit"] = hematocrit_array
    vessels.cell_data["viscosity"] = viscosity_array
    vessels.cell_data["wall_shear_stress_pa"] = wss_array
    vessels.save(vtk_export['vessels_path'])
    
    print("Flow through the network solved and VTK updated with Rheology fields.")
    print(f"Vtk file with flow data saved to: {vtk_export['vessels_path']}")

    # Write flow back into the NetworkX Graph (flow is actually already in G from the solver, but we'll do this for redundancy)
    vessels = pv.read(vtk_export['vessels_path'])
    edge_u = np.asarray(vessels.cell_data.get("edge_u", []))
    edge_v = np.asarray(vessels.cell_data.get("edge_v", []))
    edge_key = np.asarray(vessels.cell_data.get("edge_key", []))
    flow_abs = np.asarray(vessels.cell_data.get("flow_abs", []))
    
    for i in range(vessels.n_cells):
        u, v, k = int(edge_u[i]), int(edge_v[i]), int(edge_key[i])
        if G.has_edge(u, v, key=k):
            G[u][v][k]["flow_abs"] = flow_abs[i]

    # Phase 6: Perfusion Modeling
    if perf_config and perf_config.do_perfusion_modeling:
        print("\n=== Perfusion Modeling ===")
        # 1. Generate the mathematical grid
        grid = haemodynamics.PerfusionGrid(G, perf_config.grid_resolution_xyz)
        
        # 2. Map the 1D vessels to the 3D grid
        # This identifies which tissue blocks are perfused by which vessels
        cell_mapping = haemodynamics.map_vessels_to_grid(G, grid)
        
        # 3. Build Advection-Diffusion-Reaction (ADR) Matrix
        A, q_total, s_incoming = haemodynamics.build_adr_matrix(grid, cell_mapping, perf_config)
        
        # 4. Solve the Non-Linear Steady-State Perfusion field
        if getattr(perf_config, 'use_multi_species_model', False):
            print("  Running Fully Coupled Multi-Species (O2, CO2, pH) Perfusion Solver...")
            PO2_steady, PCO2_steady, pH_steady = haemodynamics.solve_multi_species_perfusion(grid, G, starting_nodes, cell_mapping, perf_config)
            mean_c = np.mean(PO2_steady); max_c = np.max(PO2_steady); min_c = np.min(PO2_steady)
            print(f"  Perfusion solve complete. Mean tissue PO2: {mean_c:.4e} mmHg (Min: {min_c:.4e}, Max: {max_c:.4e})")
            print(f"                            Mean tissue PCO2: {np.mean(PCO2_steady):.4e} mmHg")
            print(f"                            Mean tissue pH: {np.mean(pH_steady):.4f}")
            
            vti_path = pipeline_config.vtk_output_prefix.with_name(pipeline_config.vtk_output_prefix.name + "_perfusion.vti")
            visualization.export_perfusion_grid_to_vti(grid, PO2_steady, vti_path, array_name="PO2_mmHg")
            
            import pyvista as pv
            vol = pv.read(vti_path)
            vol.cell_data["PCO2_mmHg"] = PCO2_steady.flatten(order='F') if PO2_steady.ndim > 1 else PCO2_steady
            vol.cell_data["pH"] = pH_steady.flatten(order='F') if PO2_steady.ndim > 1 else pH_steady
            vol.save(vti_path)
            
        elif getattr(perf_config, 'use_endothelial_barrier_model', False):
            print("  Running Fully Coupled 1D-3D Endothelial Permeability Solver...")
            PO2_steady = haemodynamics.solve_coupled_1d3d_perfusion(grid, G, starting_nodes, cell_mapping, perf_config)
            mean_c = np.mean(PO2_steady); max_c = np.max(PO2_steady); min_c = np.min(PO2_steady)
            print(f"  Perfusion solve complete. Mean tissue PO2: {mean_c:.4e} mmHg (Min: {min_c:.4e}, Max: {max_c:.4e})")
            vti_path = pipeline_config.vtk_output_prefix.with_name(pipeline_config.vtk_output_prefix.name + "_perfusion.vti")
            visualization.export_perfusion_grid_to_vti(grid, PO2_steady, vti_path, array_name="PO2_mmHg")
            
        else:
            print("  Running Instant-Equilibrium Perfusion Solver...")
            PO2_steady = haemodynamics.solve_perfusion_steady_state(grid, A, q_total, s_incoming, perf_config)
            mean_c = np.mean(PO2_steady); max_c = np.max(PO2_steady); min_c = np.min(PO2_steady)
            print(f"  Perfusion solve complete. Mean tissue PO2: {mean_c:.4e} mmHg (Min: {min_c:.4e}, Max: {max_c:.4e})")
            vti_path = pipeline_config.vtk_output_prefix.with_name(pipeline_config.vtk_output_prefix.name + "_perfusion.vti")
            visualization.export_perfusion_grid_to_vti(grid, PO2_steady, vti_path, array_name="PO2_mmHg")
            
        print(f"  Saved 3D Perfusion Field to: {vti_path}")

    if vis_config.visualize_results:
        _visualize_final_results(G, image, starting_nodes, vis_config)

def carotid_image_to_model(image_path: Path | str, 
                           pre_config: PreprocessingConfig = None,
                           skel_config: SkeletonConfig = None,
                           graph_config: GraphConfig = None,
                           hemo_config: HaemodynamicsConfig = None,
                           perf_config: PerfusionConfig = None,
                           vis_config: VisualizationConfig = None,
                           pipeline_config: PipelineConfig = None) -> None:
    """
    Main orchestrator for the ImageLynx Carotid Pipeline.
    This pipeline executes image preprocessing, skeletonization, topological graph optimization,
    and hemodynamic simulation in sequential order using the provided configurations.
    """
    if pre_config is None: pre_config = PreprocessingConfig()
    if skel_config is None: skel_config = SkeletonConfig()
    if graph_config is None: graph_config = GraphConfig()
    if hemo_config is None: hemo_config = HaemodynamicsConfig()
    if perf_config is None: perf_config = PerfusionConfig()
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
        
        # --- Optuna Hyperparameter Optimization ---
        if args.optimize_skeleton > 0:
            import copy
            import ImageLynx.statistics.benchmarking as benchmarking
            import ImageLynx.statistics.auto_tuner as auto_tuner
            
            print(f"\n--- Launching Optuna Skeletonization Auto-Tuner ({args.optimize_skeleton} trials) ---")
            
            # Setup static dependencies
            voxel_size_xyz = io.get_tif_spacing(image_path) if input_format == "tif" else (1.0, 1.0, 1.0)
            
            def pipeline_eval_callback(suggested_kwargs):
                # Apply suggested parameters
                test_skel_config = copy.deepcopy(skel_config)
                for k, v in suggested_kwargs.items():
                    setattr(test_skel_config, k, v)
                    
                # 1. Skeletonize
                test_skeleton = _run_skeletonization_phase(binary, test_skel_config)
                
                # 2. Build Graph (Silently)
                test_pipeline_config = copy.deepcopy(pipeline_config)
                test_pipeline_config.verbose_logging = False # Reduce log spam
                
                # We do not want to save intermediate files during optimization
                import tempfile
                import os
                
                # Temporarily disable file I/O for the graph builder by passing dummy paths or patching
                # Since _build_and_optimize_graph writes to disk, we run it normally but ignore output
                # (The pipeline function does I/O, but we only care about the returned G)
                test_G = _build_and_optimize_graph(
                    test_skeleton, image, image_path, input_format, 
                    test_skel_config, graph_config, test_pipeline_config
                )
                
                if test_G.number_of_nodes() == 0 or test_G.number_of_edges() == 0:
                    return None # Prune
                    
                # 3. Evaluate Benchmarks
                bench_results = benchmarking.run_all_benchmarks(test_G, binary, voxel_size_xyz)
                return bench_results

            # Run Optuna
            best_params = auto_tuner.run_optuna_skeleton_optimization(
                pipeline_eval_callback,
                n_trials=args.optimize_skeleton,
                output_dir=pipeline_config.vtk_output_prefix.parent,
                patience=pipeline_config.optimize_patience
            )
            
            # Apply Best Parameters permanently
            print("\nApplying optimal parameters to pipeline...")
            for k, v in best_params.items():
                setattr(skel_config, k, v)
        # ------------------------------------------

        skeleton = _run_skeletonization_phase(binary, skel_config)
        np.save(skeleton_path, skeleton)
        
        # Export the post-processed binary volume to .vti for ParaView
        import pyvista as pv
        try:
            print("Exporting post-processed binary volume to VTK...")
            vtk_vol = pv.ImageData()
            vtk_vol.dimensions = np.array(binary.shape)
            
            # Use detected spacing if available, otherwise default to 1x1x1
            spacing = io.get_tif_spacing(image_path) if input_format == "tif" else (1.0, 1.0, 1.0)
            # Ensure spacing aligns with the Z, Y, X array shape
            vtk_vol.spacing = (spacing[2], spacing[1], spacing[0]) # VTK uses X, Y, Z
            
            # PyVista expects flat Fortran-ordered arrays
            vtk_vol.point_data["vessel_mask"] = binary.flatten(order="F").astype(np.uint8)
            binary_vtk_path = pipeline_config.vtk_output_prefix.with_name(f"{pipeline_config.vtk_output_prefix.name}_vessel_mask.vti")
            # Ensure the outputs folder exists
            binary_vtk_path.parent.mkdir(parents=True, exist_ok=True)
            vtk_vol.save(binary_vtk_path)
            print(f"Saved binary vessel volume to: {binary_vtk_path}")
        except Exception as e:
            print(f"Warning: Failed to export binary volume to VTK: {e}")
            
    else:
        skeleton = np.load(skeleton_path)
        image = tifffile.imread(image_path)
        binary = None

    if vis_config.visualize_results:
        visualization.visualize_skeleton(skeleton, save_path=projection_path)

    if pipeline_config.do_graph_building:
        G = _build_and_optimize_graph(skeleton, image, image_path, input_format, skel_config, graph_config, pipeline_config)
        
        if vis_config.visualize_overlay_preview and pipeline_config.do_skeletonize:
            _preview_overlay(binary, skeleton, G, image_path, input_format, vis_config, perf_config)

        with graph_path.open("wb") as f:
            pickle.dump(G, f)
        print(f"Saved graph to: {graph_path}")
    else:
        if not graph_path.exists():
            raise FileNotFoundError(f"Graph file not found at {graph_path}. Set DO_GRAPH_BUILDING=True to generate it first.")
        with graph_path.open("rb") as f:
            G = pickle.load(f)
        print(f"Loaded graph from: {graph_path}")

    starting_nodes, output_nodes, resistance_node_pair = _setup_boundary_conditions_and_haemodynamics(G, image, hemo_config, graph_config, image_path, input_format)
    _export_and_solve_haemodynamics(G, image, binary, starting_nodes, output_nodes, resistance_node_pair, hemo_config, vis_config, pipeline_config, perf_config)
    
def update_dataclass_from_dict(obj, config_dict):
    """Updates a dataclass instance with values from a dictionary."""
    if not config_dict:
        return
    for key, value in config_dict.items():
        if hasattr(obj, key):
            setattr(obj, key, value)
        else:
            logger.warning(f"Config key '{key}' ignored (not a valid parameter for {type(obj).__name__}).")

if __name__ == "__main__":
    import argparse
    import yaml
    
    parser = argparse.ArgumentParser(description="ImageLynx Carotid Pipeline")
    parser.add_argument("--sub-volume", type=float, default=None, help="Override sub_volume_percentage (0.0 to 1.0)")
    parser.add_argument("--config", type=str, default=None, help="Path to a YAML configuration file to override default parameters.")
    parser.add_argument("--optimize-skeleton", type=int, default=0, help="Run Bayesian optimization (Optuna) for N trials before continuing.")
    parser.add_argument("--optimize-preprocessing", type=int, default=0, help="Run Bayesian optimization for preprocessing filters for N trials.")
    parser.add_argument("--optimize-patience", type=int, default=None, help="Override the EarlyStoppingCallback patience limit.")
    parser.add_argument("--core-resolution", type=str, choices=["eradicate", "stitch", "none"], default=None, help="Mode for resolving internal core dead-ends.")
    args = parser.parse_args()

    # 1. Initialize Default Configurations
    pre_config = PreprocessingConfig()
    skel_config = SkeletonConfig()
    graph_config = GraphConfig()
    hemo_config = HaemodynamicsConfig(diameter_by_branch_order=DIAMETER_BY_BRANCH_ORDER)
    vis_config = VisualizationConfig(visualize_overlay_preview=False)
    pipeline_config = PipelineConfig()
    perf_config = PerfusionConfig()

    # 2. Load YAML Overrides (if provided)
    if args.config:
        config_path = Path(args.config)
        if config_path.exists():
            print(f"Loading configuration overrides from: {config_path}")
            with open(config_path, "r") as f:
                yaml_data = yaml.safe_load(f)
                
            if yaml_data:
                update_dataclass_from_dict(pre_config, yaml_data.get("PreprocessingConfig", {}))
                update_dataclass_from_dict(skel_config, yaml_data.get("SkeletonConfig", {}))
                update_dataclass_from_dict(graph_config, yaml_data.get("GraphConfig", {}))
                update_dataclass_from_dict(hemo_config, yaml_data.get("HaemodynamicsConfig", {}))
                update_dataclass_from_dict(perf_config, yaml_data.get("PerfusionConfig", {}))
                update_dataclass_from_dict(vis_config, yaml_data.get("VisualizationConfig", {}))
                update_dataclass_from_dict(pipeline_config, yaml_data.get("PipelineConfig", {}))
        else:
            print(f"Warning: Configuration file not found at {config_path}")

    # 3. CLI Overrides
    if args.sub_volume is not None:
        skel_config.sub_volume_percentage = args.sub_volume
        
    if args.core_resolution is not None:
        skel_config.core_dead_end_resolution_mode = args.core_resolution
        
    pipeline_config.optimize_preprocessing_trials = args.optimize_preprocessing
    
    if args.optimize_patience is not None:
        pipeline_config.optimize_patience = args.optimize_patience

    # 4. Run Ilastik Segmentation (if enabled)
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

    # 5. Run the Network Pipeline
    carotid_image_to_model(
        image_path=target_input_mask_path,
        pre_config=pre_config,
        skel_config=skel_config,
        graph_config=graph_config,
        hemo_config=hemo_config,
        perf_config=perf_config,
        vis_config=vis_config,
        pipeline_config=pipeline_config
    )

