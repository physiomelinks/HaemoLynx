#!/usr/bin/env python3
"""Default settings for the resistance network pipeline, grouped by concern."""
import ast
import copy
import sys
from pathlib import Path
from typing import Optional

# Ensure package is importable when running from repo root.
root_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ImageLynx import haemodynamics
from presets import get_preset_definitions

# ---------------------------
# Input and ilastik settings
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

# ---------------------------
# Vessel-mask settings
# ---------------------------
# Do you want to use large vessel masks to determine input and output nodes?
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

# ---------------------------
# Boundary-node assignment
# ---------------------------
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
STARTING_NODE_COORDINATES = [
    (152.0, 340.0, 527.0),
    (160.0, 350.0, 545.0),  # top right
    (202.0, 1303.0, 132.0),
    (104.0, 1321.0, 133.0),  # bottom left
    (361.0, 332.0, 120.0),
    (321.0, 334.0, 163.0),  # top right
]

OUTPUT_NODE_COORDINATES = []
ARTERIOLE_BOUNDARY_NODE_COORDINATES = []
VENULE_BOUNDARY_NODE_COORDINATES = []

# Assign by volume boxes
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
STRICT_BRANCH_ORDER_ASSIGNMENT = False

# ---------------------------
# Solver and output settings
# ---------------------------
INPUT_P_BC = 4500  # Pa
OUTPUT_P_BC = 1000  # Pa
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
VISUALIZE_VTK = False
VERBOSE_LOGGING = False

# ---------------------------
# Pipeline-stage and topology settings
# ---------------------------
DO_SKELETONIZE = True
DO_GRAPH_BUILDING = True
RUN_HAEMODYNAMICS = True
DO_EQUIV_RESISTANCE_CALCULATION = True
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

# ---------------------------
# Statistics and measurements
# ---------------------------
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

# ---------------------------
# Diameter and pericyte settings
# ---------------------------
# Vessel diameter for Poiseuille weights (manual branch-order vs automated FWHM)
# -------------------------------------------------------------------------------
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
# HD note - manual overrides for in vivo diameters; endoneurial custom vessels not in graph.
# -------------------------------------------------------------------------------
ALL_DIAMS_CONST = True
DO_PERICYTE_CONSTRUCTION = False
# Optional mask-driven pericyte mode:
# - False: keep existing artificial periodic constriction placement.
# - True: use pericyte mask connected-component centroids as constriction centers.
USE_PERICYTE_MASK_CONSTRICTION = False
PERICYTE_MASK_PATH: Optional[Path] = None
PERICYTE_MASK_H5_DATASET_NAME: Optional[str] = None
PERICYTE_MAX_ASSIGNMENT_DISTANCE_UM = 3.0
PERICYTE_MIN_DIAMETER_UM = 5.0
PERICYTE_MAX_DIAMETER_UM = 12.0
# Optional probabilistic constriction:
# Example: probability=0.8 means ~80% of pericytes are active per run.
USE_PROBABILISTIC_PERICYTE_CONSTRICTION = False
PERICYTE_CONSTRICTION_PROBABILITY = 0.8
RUN_PERICYTE_RESISTANCE_COMPARISON = False
# Absolute comparison values: when comparison is enabled, these override
# CONSTRICTION_BY_BRANCH_ORDER magnitudes for the comparison pass.
PERICYTE_COMPARISON_BASELINE_VALUE = 1.0
PERICYTE_COMPARISON_CONSTRICTED_VALUE = 0.8
# If True and probabilistic mode is enabled, reuse the exact
# pericyte cohort selected during comparison for the final haemodynamics solve.
REUSE_COMPARISON_PERICYTE_COHORT_FOR_MAIN_RUN = False

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

# ---------------------------
# FWHM settings
# ---------------------------
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

# These are vessels that constrict differently (e.g. endoneurial vessels).
custom_edges = []


# ---------------------------
# Preset system
# ---------------------------
PRESET_DEFINITIONS: dict[str, dict[str, object]] = get_preset_definitions(root_dir)


def _collect_setting_names() -> set[str]:
    return {
        name
        for name in globals()
        if (
            (name.isupper() or name == "custom_edges")
            and not name.startswith("_")
            and name not in {"PRESET_DEFINITIONS"}
        )
    }


VALID_SETTING_NAMES = _collect_setting_names()


def _collect_base_settings() -> dict[str, object]:
    base: dict[str, object] = {}
    for name in VALID_SETTING_NAMES:
        base[name] = copy.deepcopy(globals()[name])
    return base


def _recompute_derived_settings(settings: dict[str, object]) -> None:
    settings["DIAMETER_BY_BRANCH_ORDER"] = haemodynamics.build_diameter_by_branch_order(
        all_diams_const=bool(settings["ALL_DIAMS_CONST"]),
        max_branch_order=int(settings["MAX_BRANCH_ORDER"]),
        default_diameter=float(settings["DEFAULT_DIAMETER"]),
        manual_capillary_diameter_by_branch_order=settings[
            "MANUAL_CAPILLARY_DIAMETER_BY_BRANCH_ORDER"
        ],
        manual_arteriole_diameter_by_branch_order=settings[
            "MANUAL_ARTERIOLE_DIAMETER_BY_BRANCH_ORDER"
        ],
        manual_venule_diameter_by_branch_order=settings[
            "MANUAL_VENULE_DIAMETER_BY_BRANCH_ORDER"
        ],
    )
    max_branch_order = int(settings["MAX_BRANCH_ORDER"])
    constriction: dict[str, float] = {"B01": 1.0}
    for i in range(2, max_branch_order + 1):
        constriction[f"B{i:02d}"] = 0.8
    constriction["Art1"] = 1.0
    constriction["Ven1"] = 1.0
    for i in range(2, max_branch_order + 1):
        constriction[f"Art{i}"] = 0.8
        constriction[f"Ven{i}"] = 0.8
    settings["CONSTRICTION_BY_BRANCH_ORDER"] = constriction


def list_presets() -> dict[str, str]:
    return {
        name: str(payload["description"])
        for name, payload in PRESET_DEFINITIONS.items()
    }


def parse_cli_override(override_text: str) -> tuple[str, object]:
    if "=" not in override_text:
        raise ValueError(
            f"Invalid override '{override_text}'. Use KEY=VALUE format."
        )
    key_text, raw_value = override_text.split("=", 1)
    key = key_text.strip()
    if not key:
        raise ValueError(f"Invalid override '{override_text}': empty KEY.")
    key = key if key == "custom_edges" else key.upper()
    if key not in VALID_SETTING_NAMES:
        available = ", ".join(sorted(VALID_SETTING_NAMES))
        raise ValueError(
            f"Unknown setting '{key}' in override '{override_text}'. "
            f"Valid settings: {available}"
        )
    value = _parse_cli_value(raw_value.strip())
    value = _coerce_path_like_value(key, value)
    return key, value


def _parse_cli_value(text: str) -> object:
    lowered = text.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    try:
        return ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return text


def _coerce_path_like_value(key: str, value: object) -> object:
    if value is None:
        return None
    if isinstance(value, Path):
        return value
    if isinstance(value, str):
        if key.endswith("_PATH") or key.endswith("_DIR") or key.endswith("_PREFIX"):
            return Path(value)
    return value


def build_settings_for_preset(
    preset_name: str = "default",
    manual_overrides: dict[str, object] | None = None,
) -> dict[str, object]:
    if preset_name not in PRESET_DEFINITIONS:
        available = ", ".join(sorted(PRESET_DEFINITIONS))
        raise ValueError(
            f"Unknown preset '{preset_name}'. Available presets: {available}"
        )
    settings = _collect_base_settings()
    preset_overrides = dict(PRESET_DEFINITIONS[preset_name]["overrides"])
    settings.update(preset_overrides)
    if manual_overrides:
        unknown = [k for k in manual_overrides if k not in VALID_SETTING_NAMES]
        if unknown:
            available = ", ".join(sorted(VALID_SETTING_NAMES))
            raise ValueError(
                f"Unknown manual override settings: {unknown}. "
                f"Valid settings: {available}"
            )
        for key, value in manual_overrides.items():
            settings[key] = _coerce_path_like_value(key, value)
    _recompute_derived_settings(settings)
    return settings


def apply_settings_to_namespace(settings: dict[str, object], namespace: dict) -> None:
    for key, value in settings.items():
        if key in VALID_SETTING_NAMES:
            namespace[key] = value
