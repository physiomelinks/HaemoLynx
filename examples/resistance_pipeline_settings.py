#!/usr/bin/env python3
import sys
from functools import partial
from pathlib import Path
from typing import Optional

# Ensure package is importable when running from repo root.
root_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ImageLynx import haemodynamics
from presets import (
    apply_settings_to_namespace as _apply_settings_to_namespace,
    build_settings_for_preset as _build_settings_for_preset,
    collect_base_settings,
    collect_setting_names,
    get_preset_definitions,
    load_config_yaml as _load_config_yaml,
    list_presets as _list_presets,
    parse_cli_override as _parse_cli_override,
    save_effective_config_yaml as _save_effective_config_yaml,
)

# ---------------------------
# Input and ilastik settings
# ---------------------------
# Input segmented image used by the pipeline.
INPUT_PATH = root_dir / "examples" / "images" / "brain_microvessels.tiff"
# Toggle ilastik segmentation for the main input image.
USE_ILASTIK_SEGMENTATION = False
# Raw image path used as ilastik input when segmentation is enabled.
ILASTIK_UNSEGMENTED_IMAGE_PATH = root_dir / "examples" / "images" / "Nerve_capillaries.tif"
# Ilastik project/classifier path for main image segmentation.
ILASTIK_CLASSIFIER_PATH = root_dir / "examples" / "classifiers" / "nerve_classifier.ilp"
# Executable name or path for ilastik headless mode.
ILASTIK_EXECUTABLE = "ilastik.exe"
# Output directory for ilastik-generated segmentations.
ILASTIK_OUTPUT_DIR = root_dir / "examples" / "outputs" / "segmentations"
# File suffix for ilastik segmentation outputs.
ILASTIK_OUTPUT_SUFFIX = ".tif"
# Optional manual override with mixed units:
#   x, y: px/um (pixels per micron)
#   z:    um/px (microns per pixel; already spacing)
VOXEL_SIZE_OVERRIDE_XYZ_PX_PER_UM: tuple[float, float, float] | None = [1.6862, 1.6862, 2.0]
# Policy for voxel-size resolution: "auto", "override", or "metadata_only".
VOXEL_SIZE_POLICY = "auto"

# ---------------------------
# Vessel-mask settings
# ---------------------------
# Toggle use of large-vessel masks for automated start/output assignment.
USE_LARGE_VESSEL_MASKS = True
# Toggle ilastik segmentation for large-vessel masks.
USE_ILASTIK_LARGE_VESSEL_SEGMENTATION = False
# Dilation size (microns) applied to large-vessel masks before node selection.
LARGE_VESSEL_MASK_DILATION_MICRONS = 0.0

# Pre-segmented large arteriole mask path.
LARGE_ARTERIOLE_MASK_PATH = root_dir / "examples" / "images" / "brain_large_arterioles.tiff"
# Pre-segmented large venule mask path.
LARGE_VENULE_MASK_PATH = root_dir / "examples" / "images" / "brain_large_venules.tiff"
# Raw arteriole image path used when ilastik large-vessel mode is enabled.
ILASTIK_UNSEGMENTED_ARTERIOLE_IMAGE_PATH = root_dir / "examples" / "images" / "large_arteriole_mask.tif"
# Raw venule image path used when ilastik large-vessel mode is enabled.
ILASTIK_UNSEGMENTED_VENULE_IMAGE_PATH = root_dir / "examples" / "images" / "large_venule_mask.tif"
# Ilastik classifier path for arteriole segmentation.
ILASTIK_ARTERIOLE_CLASSIFIER_PATH = root_dir / "examples" / "classifiers" / "arteriole_classifier.ilp"
# Ilastik classifier path for venule segmentation.
ILASTIK_VENULE_CLASSIFIER_PATH = root_dir / "examples" / "classifiers" / "venule_classifier.ilp"

# Toggle small-vessel masks for automated arteriole/venule boundary assignment.
USE_SMALL_VESSEL_MASKS_FOR_BOUNDARY_ASSIGNMENT = False
# Toggle ilastik segmentation for small-vessel masks.
USE_ILASTIK_SMALL_VESSEL_SEGMENTATION = False
# Minimum edge-overlap fraction required for mask-based boundary assignment.
SMALL_VESSEL_MASK_MIN_OVERLAP_FRACTION = 0.5
# Toggle writing interactive 3D HTML for small-vessel boundary labelling.
WRITE_SMALL_VESSEL_BOUNDARY_LABELLING_3D_HTML = True
# Pre-segmented small arteriole mask path.
SMALL_ARTERIOLE_MASK_PATH = root_dir / "examples" / "images" / "small_arteriole_mask.tif"
# Pre-segmented small venule mask path.
SMALL_VENULE_MASK_PATH = root_dir / "examples" / "images" / "small_venule_mask.tif"
# Raw small-arteriole image path used when ilastik small-vessel mode is enabled.
ILASTIK_UNSEGMENTED_SMALL_ARTERIOLE_IMAGE_PATH = root_dir / "examples" / "images" / "small_arteriole_mask.tif"
# Raw small-venule image path used when ilastik small-vessel mode is enabled.
ILASTIK_UNSEGMENTED_SMALL_VENULE_IMAGE_PATH = root_dir / "examples" / "images" / "small_venule_mask.tif"
# Ilastik classifier path for small arteriole segmentation.
ILASTIK_SMALL_ARTERIOLE_CLASSIFIER_PATH = ILASTIK_ARTERIOLE_CLASSIFIER_PATH
# Ilastik classifier path for small venule segmentation.
ILASTIK_SMALL_VENULE_CLASSIFIER_PATH = ILASTIK_VENULE_CLASSIFIER_PATH

# ---------------------------
# Boundary-node assignment
# ---------------------------
# Base output directory for plot artifacts.
BASE_PLOT_DIR = root_dir / "examples" / "plots"
if not BASE_PLOT_DIR.exists():
    BASE_PLOT_DIR.mkdir(parents=True, exist_ok=True)

# Toggle automated selection of start/output nodes from masks.
AUTOMATED_VESSEL_ASSIGNMENT = False
# Method used to choose manual starting nodes.
STARTING_NODE_SELECTION_METHOD = "coordinates"
# Method used to choose manual output nodes.
OUTPUT_NODE_SELECTION_METHOD = "coordinates"
# Method used to choose manual arteriole boundary nodes.
ARTERIOLE_BOUNDARY_SELECTION_METHOD = "coordinates"
# Method used to choose manual venule boundary nodes.
VENULE_BOUNDARY_SELECTION_METHOD = "coordinates"
# Manual coordinate list for starting node selection.
STARTING_NODE_COORDINATES = [
    (152.0, 340.0, 527.0),
    (160.0, 350.0, 545.0),  # top right
    (202.0, 1303.0, 132.0),
    (104.0, 1321.0, 133.0),  # bottom left
    (361.0, 332.0, 120.0),
    (321.0, 334.0, 163.0),  # top right
]

# Manual coordinate list for output node selection.
OUTPUT_NODE_COORDINATES = []
# Manual coordinate list for arteriole boundary selection.
ARTERIOLE_BOUNDARY_NODE_COORDINATES = []
# Manual coordinate list for venule boundary selection.
VENULE_BOUNDARY_NODE_COORDINATES = []

# Toggle volume-box based boundary node selection mode.
USE_VOLUME_BOXES = False
# Volume boxes used to select starting nodes.
STARTING_NODE_VOLUMES: list[tuple[tuple[float, float, float], tuple[float, float, float]]] = []
# Volume boxes used to select output nodes.
OUTPUT_NODE_VOLUMES: list[tuple[tuple[float, float, float], tuple[float, float, float]]] = []
# Volume boxes used to select arteriole boundary nodes.
ARTERIOLE_BOUNDARY_NODE_VOLUMES: list[tuple[tuple[float, float, float], tuple[float, float, float]]] = []
# Volume boxes used to select venule boundary nodes.
VENULE_BOUNDARY_NODE_VOLUMES: list[tuple[tuple[float, float, float], tuple[float, float, float]]] = []
# Runtime container for selected starting node IDs.
STARTING_NODES: list[int] = []
# Runtime container for selected output node IDs.
OUTPUT_NODES: list[int] = []
# Runtime container for selected arteriole boundary node IDs.
ARTERIOLE_BOUNDARY_NODES: list[int] = []
# Runtime container for selected venule boundary node IDs.
VENULE_BOUNDARY_NODES: list[int] = []
# Enforce strict hierarchical branch-order prerequisites.
STRICT_BRANCH_ORDER_ASSIGNMENT = False

# ---------------------------
# Solver and output settings
# ---------------------------
# Inlet pressure boundary condition (Pa).
INPUT_P_BC = 4500  # Pa
# Outlet pressure boundary condition (Pa).
OUTPUT_P_BC = 1000  # Pa
# Toggle final visualization output generation.
VISUALIZE_RESULTS = True
# Toggle interactive plotting behavior.
INTERACTIVE_PLOTS = False
# Toggle showing plots in IDE windows while running.
SHOW_PLOTS_IN_IDE = True

# Select which IDE plots are displayed when SHOW_PLOTS_IN_IDE is enabled.
IDE_PLOT_MODE = "final_only"
# Keep IDE matplotlib windows open at script end.
HOLD_IDE_PLOTS_OPEN = True
# Choose 2D or 3D rendering mode for final graph views.
FINAL_RENDER_MODE = "3d"  # "2d" or "3d"
# Toggle VTK export of vessels/pericytes/nodes.
VTK_export = True
# Toggle VTK visualization viewer launch.
VISUALIZE_VTK = False
# Toggle verbose logging output.
VERBOSE_LOGGING = False

# ---------------------------
# Pipeline-stage and topology settings
# ---------------------------
# Toggle skeletonization step execution.
DO_SKELETONIZE = True
# Toggle graph-building step execution.
DO_GRAPH_BUILDING = True
# Toggle haemodynamics pipeline execution.
RUN_HAEMODYNAMICS = True
# Toggle two-point equivalent resistance calculation.
DO_EQUIV_RESISTANCE_CALCULATION = True
# Minimum branch length threshold used by graph operations.
MIN_BRANCH_LENGTH = 10
# Output path prefix used for VTK artifacts.
VTK_OUTPUT_PREFIX = root_dir / "examples" / "outputs" / "resistance_network"
# Closing radius used in skeleton preprocessing.
SKELETON_CLOSING_RADIUS = 2
# Maximum gap size for skeleton bridge operations.
SKELETON_BRIDGE_GAP_SIZE = 3
# Minimum branch length kept during skeleton cleaning.
SKELETON_MIN_BRANCH_LENGTH = 3
# Maximum distance for skeleton bridge reconnection.
SKELETON_MAX_BRIDGE_DISTANCE = 4
# Connectivity mode used for skeleton component analysis.
SKELETON_COMPONENT_CONNECTIVITY = 3
# Reconnection threshold for graph topology repair.
GRAPH_RECONNECT_THRESHOLD = 10.0

# Final reconnect threshold for orphan/dangling nodes.
FINAL_ORPHAN_RECONNECT_THRESHOLD = 3.0
# Minimum stub length retained before pruning.
MIN_STUB_LENGTH = 10.0
# Distance threshold used for collapsing node clusters.
CLUSTER_COLLAPSE_DISTANCE = 5.0

# Minimum component size percentage to keep after skeleton cleanup.
SKELETON_MIN_COMPONENT_PERCENT = 0.0

# ---------------------------
# Statistics and measurements
# ---------------------------
# Toggle global vessel statistics computation.
STATISTICS = False
# Toggle 3D distance-to-cell-mask measurement.
MEASUREMENT_3D_TO_CELL_MASK = False
# Path to the cell mask used for 3D distance measurements.
CELL_MASK_PATH: Optional[Path] = None
# Optional H5 dataset name for the cell mask.
CELL_MASK_H5_DATASET_NAME: Optional[str] = None
# Optional explicit vessel mask path for 3D distance measurements.
MEASUREMENT_3D_VESSEL_MASK_PATH: Optional[Path] = None
# Optional H5 dataset name for the vessel mask.
MEASUREMENT_3D_VESSEL_MASK_H5_DATASET_NAME: Optional[str] = None
# Optional reference image path used for vessel-volume raster shape.
MEASUREMENT_3D_REFERENCE_IMAGE_PATH: Optional[Path] = None
# Optional H5 dataset name for the reference image.
MEASUREMENT_3D_REFERENCE_H5_DATASET_NAME: Optional[str] = None
# Statistics execution mode ("fast" or "full").
STATISTICS_MODE = "fast"

# ---------------------------
# Diameter and pericyte settings
# ---------------------------
# Toggle constant diameter behavior across branch orders.
ALL_DIAMS_CONST = True
# Toggle pericyte constriction modelling in haemodynamics.
DO_PERICYTE_CONSTRUCTION = False
# Toggle pericyte-mask-driven constriction placement.
USE_PERICYTE_MASK_CONSTRICTION = False
# Path to pericyte mask used for constriction placement.
PERICYTE_MASK_PATH: Optional[Path] = None
# Optional H5 dataset name for the pericyte mask.
PERICYTE_MASK_H5_DATASET_NAME: Optional[str] = None
# Maximum distance (um) to assign pericyte centroids to vessel edges.
PERICYTE_MAX_ASSIGNMENT_DISTANCE_UM = 3.0
# Minimum pericyte diameter (um) used in mask workflow.
PERICYTE_MIN_DIAMETER_UM = 5.0
# Maximum pericyte diameter (um) used in mask workflow.
PERICYTE_MAX_DIAMETER_UM = 12.0
# Toggle probabilistic pericyte constriction activation.
USE_PROBABILISTIC_PERICYTE_CONSTRICTION = False
# Activation probability used in probabilistic constriction mode.
PERICYTE_CONSTRICTION_PROBABILITY = 0.8
# Toggle baseline-vs-constricted pericyte resistance comparison run.
RUN_PERICYTE_RESISTANCE_COMPARISON = False
# Baseline comparison multiplier used in pericyte comparison mode.
PERICYTE_COMPARISON_BASELINE_VALUE = 1.0
# Constricted comparison multiplier used in pericyte comparison mode.
PERICYTE_COMPARISON_CONSTRICTED_VALUE = 0.8
# Reuse selected probabilistic pericyte cohort from comparison in main run.
REUSE_COMPARISON_PERICYTE_COHORT_FOR_MAIN_RUN = False

# Maximum branch-order index used to build diameter/constriction tables.
MAX_BRANCH_ORDER = 51
# Default vessel diameter used when no branch-order override is present.
DEFAULT_DIAMETER = 4.0

# Manual capillary diameter overrides keyed by branch-order label.
MANUAL_CAPILLARY_DIAMETER_BY_BRANCH_ORDER = {
    "B01": 6.2,
    "B02": 4.0,
    "B03": 5.0,
    "B04": 5.0,
}
# Manual arteriole diameter overrides keyed by branch-order label.
MANUAL_ARTERIOLE_DIAMETER_BY_BRANCH_ORDER = {}
# Manual venule diameter overrides keyed by branch-order label.
MANUAL_VENULE_DIAMETER_BY_BRANCH_ORDER = {}

# Derived branch-order diameter lookup used by haemodynamics.
DIAMETER_BY_BRANCH_ORDER = haemodynamics.build_diameter_by_branch_order(
    all_diams_const=ALL_DIAMS_CONST,
    max_branch_order=MAX_BRANCH_ORDER,
    default_diameter=DEFAULT_DIAMETER,
    manual_capillary_diameter_by_branch_order=MANUAL_CAPILLARY_DIAMETER_BY_BRANCH_ORDER,
    manual_arteriole_diameter_by_branch_order=MANUAL_ARTERIOLE_DIAMETER_BY_BRANCH_ORDER,
    manual_venule_diameter_by_branch_order=MANUAL_VENULE_DIAMETER_BY_BRANCH_ORDER,
)

# Derived branch-order constriction lookup used by haemodynamics.
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

# ---------------------------
# FWHM settings
# ---------------------------
# Toggle FWHM-based automated edge diameter measurement.
USE_FWHM_EDGE_DIAMETERS = False
# Raw single-channel image path used by FWHM measurement.
FWHM_RAW_TIFF_PATH: Optional[Path] = None
# Spacing (um) between sampled centerline positions for FWHM.
FWHM_SAMPLE_SPACING_ALONG_EDGE_UM = 2.0
# Step size (um) for sampling each transverse profile.
FWHM_TRANSVERSE_PROFILE_STEP_UM = 0.25
# Initial transverse half-extent (um) sampled on either side of centerline.
FWHM_TRANSVERSE_HALF_EXTENT_UM = 6.0
# Optional starting diameter guess (um) for Gaussian fitting.
FWHM_DIAMETER_GUESS_UM = None
# Minimum total profile extent multiplier relative to fitted width.
FWHM_MIN_TOTAL_EXTENT_MULTIPLIER = 3.0
# Background label value used in rasterized branch label volume.
FWHM_BACKGROUND_LABEL = 0
# Junction label value used in rasterized branch label volume.
FWHM_JUNCTION_LABEL = -1
# Allow profiles to pass through junction labels during FWHM sampling.
FWHM_ALLOW_JUNCTION_CROSSING = False
# Baseline estimation mode for transverse Gaussian profile fitting.
FWHM_PROFILE_BASELINE_MODE = "wings"
# Wing fraction used by wing-based baseline estimation mode.
FWHM_PROFILE_BASELINE_WING_FRACTION = 0.2
# Constrain fitted baseline near baseline-anchor estimate.
FWHM_CONSTRAIN_FITTED_BASELINE = False
# Half-width constraint for baseline fitting as fraction of profile range.
FWHM_BASELINE_CONSTRAINT_HALF_WIDTH_PTP = 0.35
# Clip profile to single-vessel lobe to avoid neighboring-branch peaks.
FWHM_CLIP_PROFILE_TO_SINGLE_VESSEL = True
# Minimum center-drop fraction for profile clipping.
FWHM_CLIP_MIN_DROP_FRACTION_OF_CENTER = 0.35
# Re-rise fraction threshold that terminates clipped lobe.
FWHM_CLIP_RE_RISE_FRACTION_OF_CENTER = 0.08
# Exclusion distance (um) from branch endpoints for sampling.
FWHM_BRANCH_ENDPOINT_EXCLUSION_UM = 10.0
# Exclusion distance (um) near detected junction voxels.
FWHM_JUNCTION_PROXIMITY_EXCLUSION_UM = 10.0
# Enforce local same-edge sampling neighborhood guard.
FWHM_ENFORCE_SAME_EDGE_LOCALITY = True
# Absolute arc-window size (um) for same-edge locality checks.
FWHM_SAME_EDGE_ARC_WINDOW_UM = 3.0
# Arc-window multiplier used in same-edge locality checks.
FWHM_SAME_EDGE_ARC_WINDOW_MULTIPLIER = 1.0
# Minimum allowed arc-window size (um) for locality checks.
FWHM_SAME_EDGE_ARC_WINDOW_MIN_UM = 1.0
# Cap profile half-extent using nonlocal same-edge distance constraints.
FWHM_CAP_HALF_EXTENT_BY_NONLOCAL_SAME_EDGE_DISTANCE = True
# Arc separation (um) to classify same-edge points as nonlocal.
FWHM_NONLOCAL_SAME_EDGE_ARC_SEPARATION_UM = 6.0
# Scaling factor to cap half-extent from nonlocal same-edge distance.
FWHM_NONLOCAL_SAME_EDGE_HALF_EXTENT_FACTOR = 0.45
# Reject samples when fitted center is too far from expected center.
FWHM_REJECT_SAMPLES_WITH_CENTER_OFFSET = True
# Maximum allowed fitted center offset (um).
FWHM_MAX_FIT_CENTER_OFFSET_UM = 1.5
# Reject samples whose Gaussian fit quality is below threshold.
FWHM_REJECT_SAMPLES_WITH_LOW_FIT_R2 = True
# Minimum acceptable R^2 for transverse Gaussian fits.
FWHM_MIN_FIT_R2 = 0.85

# List of edge IDs that use custom edge-diameter assignment behavior.
custom_edges = []


# ---------------------------
# Preset system
# ---------------------------
PRESET_DEFINITIONS: dict[str, dict[str, object]] = get_preset_definitions(root_dir)
VALID_SETTING_NAMES = collect_setting_names(globals(), PRESET_DEFINITIONS)
BASE_SETTINGS_TEMPLATE = collect_base_settings(globals(), VALID_SETTING_NAMES)

list_presets = partial(_list_presets, preset_definitions=PRESET_DEFINITIONS)
parse_cli_override = partial(
    _parse_cli_override,
    valid_setting_names=VALID_SETTING_NAMES,
)
build_settings_for_preset = partial(
    _build_settings_for_preset,
    preset_definitions=PRESET_DEFINITIONS,
    valid_setting_names=VALID_SETTING_NAMES,
    base_settings_template=BASE_SETTINGS_TEMPLATE,
    haemodynamics_module=haemodynamics,
)
apply_settings_to_namespace = partial(
    _apply_settings_to_namespace,
    valid_setting_names=VALID_SETTING_NAMES,
)
load_config_yaml = partial(
    _load_config_yaml,
    valid_setting_names=VALID_SETTING_NAMES,
    available_preset_names=set(PRESET_DEFINITIONS.keys()),
)
save_effective_config_yaml = _save_effective_config_yaml
