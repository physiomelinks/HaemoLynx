#!/usr/bin/env python3
"""Preset definitions for example pipelines."""
from pathlib import Path


def get_preset_definitions(root_dir: Path) -> dict[str, dict[str, object]]:
    """Return named preset profiles and their setting overrides."""
    return {
        "default": {
            "description": "Current baseline behavior from this settings file.",
            "overrides": {},
        },
        "quick_debug": {
            "description": (
                "Fast iteration mode: disables heavy outputs/analysis and keeps logs concise."
            ),
            "overrides": {
                "VERBOSE_LOGGING": True,
                "VISUALIZE_RESULTS": False,
                "SHOW_PLOTS_IN_IDE": False,
                "HOLD_IDE_PLOTS_OPEN": False,
                "INTERACTIVE_PLOTS": False,
                "VTK_export": False,
                "VISUALIZE_VTK": False,
                "STATISTICS": False,
                "MEASUREMENT_3D_TO_CELL_MASK": False,
                "DO_EQUIV_RESISTANCE_CALCULATION": False,
                "RUN_PERICYTE_RESISTANCE_COMPARISON": False,
                "DO_PERICYTE_CONSTRUCTION": False,
                "USE_FWHM_EDGE_DIAMETERS": False,
                "WRITE_SMALL_VESSEL_BOUNDARY_LABELLING_3D_HTML": False,
            },
        },
        "publication": {
            "description": (
                "High-detail reporting mode: writes full stats/artifacts with non-interactive rendering."
            ),
            "overrides": {
                "VERBOSE_LOGGING": False,
                "VISUALIZE_RESULTS": True,
                "SHOW_PLOTS_IN_IDE": False,
                "HOLD_IDE_PLOTS_OPEN": False,
                "INTERACTIVE_PLOTS": False,
                "VTK_export": True,
                "VISUALIZE_VTK": False,
                "STATISTICS": True,
                "STATISTICS_MODE": "full",
                "FINAL_RENDER_MODE": "3d",
                "WRITE_SMALL_VESSEL_BOUNDARY_LABELLING_3D_HTML": True,
            },
        },
        "full_fwhm": {
            "description": (
                "Enable full FWHM diameter workflow with detailed outputs for raw-image-aligned analysis."
            ),
            "overrides": {
                "USE_FWHM_EDGE_DIAMETERS": True,
                "FWHM_RAW_TIFF_PATH": root_dir
                / "examples"
                / "images"
                / "Nerve_capillaries.tif",
                "VERBOSE_LOGGING": True,
                "VISUALIZE_RESULTS": True,
                "SHOW_PLOTS_IN_IDE": False,
                "HOLD_IDE_PLOTS_OPEN": False,
                "INTERACTIVE_PLOTS": False,
                "STATISTICS": True,
                "STATISTICS_MODE": "full",
            },
        },
        "all_automated": {
            "description": (
                "Fully automated profile: large/small vessel masks, pericyte-mask constriction, and FWHM diameters."
            ),
            "overrides": {
                # Automated boundary assignment from large-vessel masks
                "AUTOMATED_VESSEL_ASSIGNMENT": True,
                "USE_LARGE_VESSEL_MASKS": True,
                "USE_ILASTIK_LARGE_VESSEL_SEGMENTATION": False,
                # Automated arteriole/venule boundary inference from small-vessel masks
                "USE_SMALL_VESSEL_MASKS_FOR_BOUNDARY_ASSIGNMENT": True,
                "USE_ILASTIK_SMALL_VESSEL_SEGMENTATION": False,
                "WRITE_SMALL_VESSEL_BOUNDARY_LABELLING_3D_HTML": True,
                # Automated diameter measurement
                "USE_FWHM_EDGE_DIAMETERS": True,
                "FWHM_RAW_TIFF_PATH": root_dir
                / "examples"
                / "images"
                / "Nerve_capillaries.tif",
                # Pericyte mask-driven constriction
                "DO_PERICYTE_CONSTRUCTION": True,
                "USE_PERICYTE_MASK_CONSTRICTION": True,
                "USE_PROBABILISTIC_PERICYTE_CONSTRICTION": False,
                "RUN_PERICYTE_RESISTANCE_COMPARISON": False,
                # Keep this as None by default; users should provide with --set PERICYTE_MASK_PATH=...
                # if they are not using the default image naming/location workflow.
                "PERICYTE_MASK_PATH": None,
                # Output/reporting defaults
                "VERBOSE_LOGGING": True,
                "VISUALIZE_RESULTS": True,
                "SHOW_PLOTS_IN_IDE": False,
                "HOLD_IDE_PLOTS_OPEN": False,
                "INTERACTIVE_PLOTS": False,
                "VTK_export": True,
                "VISUALIZE_VTK": False,
                "STATISTICS": True,
                "STATISTICS_MODE": "full",
                "FINAL_RENDER_MODE": "3d",
            },
        },
        "automated_assignment": {
            "description": (
                "Automated vessel assignment profile using pre-segmented large and small vessel masks (no ilastik)."
            ),
            "overrides": {
                # Automated input/output node assignment from large-vessel masks
                "AUTOMATED_VESSEL_ASSIGNMENT": True,
                "USE_LARGE_VESSEL_MASKS": True,
                "USE_ILASTIK_LARGE_VESSEL_SEGMENTATION": False,
                # Automated arteriole/venule boundary inference from small-vessel masks
                "USE_SMALL_VESSEL_MASKS_FOR_BOUNDARY_ASSIGNMENT": True,
                "USE_ILASTIK_SMALL_VESSEL_SEGMENTATION": False,
                "WRITE_SMALL_VESSEL_BOUNDARY_LABELLING_3D_HTML": True,
                # Keep automated diameter/pericyte workflows disabled
                "USE_FWHM_EDGE_DIAMETERS": False,
                "DO_PERICYTE_CONSTRUCTION": False,
                "USE_PERICYTE_MASK_CONSTRICTION": False,
                "USE_PROBABILISTIC_PERICYTE_CONSTRICTION": False,
                "RUN_PERICYTE_RESISTANCE_COMPARISON": False,
                "PERICYTE_MASK_PATH": None,
                # Output/reporting defaults
                "VERBOSE_LOGGING": True,
                "VISUALIZE_RESULTS": True,
                "SHOW_PLOTS_IN_IDE": False,
                "HOLD_IDE_PLOTS_OPEN": False,
                "INTERACTIVE_PLOTS": False,
                "VTK_export": True,
                "VISUALIZE_VTK": False,
                "STATISTICS": True,
                "STATISTICS_MODE": "full",
                "FINAL_RENDER_MODE": "3d",
            },
        },
        "automated_assignment_ilastik": {
            "description": (
                "Automated vessel assignment profile using ilastik segmentation for large and small vessel masks."
            ),
            "overrides": {
                # Automated input/output node assignment from large-vessel masks
                "AUTOMATED_VESSEL_ASSIGNMENT": True,
                "USE_LARGE_VESSEL_MASKS": True,
                "USE_ILASTIK_LARGE_VESSEL_SEGMENTATION": True,
                # Automated arteriole/venule boundary inference from small-vessel masks
                "USE_SMALL_VESSEL_MASKS_FOR_BOUNDARY_ASSIGNMENT": True,
                "USE_ILASTIK_SMALL_VESSEL_SEGMENTATION": True,
                "WRITE_SMALL_VESSEL_BOUNDARY_LABELLING_3D_HTML": True,
                # Keep automated diameter/pericyte workflows disabled
                "USE_FWHM_EDGE_DIAMETERS": False,
                "DO_PERICYTE_CONSTRUCTION": False,
                "USE_PERICYTE_MASK_CONSTRICTION": False,
                "USE_PROBABILISTIC_PERICYTE_CONSTRICTION": False,
                "RUN_PERICYTE_RESISTANCE_COMPARISON": False,
                "PERICYTE_MASK_PATH": None,
                # Output/reporting defaults
                "VERBOSE_LOGGING": True,
                "VISUALIZE_RESULTS": True,
                "SHOW_PLOTS_IN_IDE": False,
                "HOLD_IDE_PLOTS_OPEN": False,
                "INTERACTIVE_PLOTS": False,
                "VTK_export": True,
                "VISUALIZE_VTK": False,
                "STATISTICS": True,
                "STATISTICS_MODE": "full",
                "FINAL_RENDER_MODE": "3d",
            },
        },
        "statistics_only": {
            "description": (
                "Skeletonize/build graph + assign vessels + compute statistics with haemodynamics enabled but no equivalent-resistance solve."
            ),
            "overrides": {
                "DO_SKELETONIZE": True,
                "DO_GRAPH_BUILDING": True,
                "RUN_HAEMODYNAMICS": True,
                "DO_EQUIV_RESISTANCE_CALCULATION": False,
                "DO_PERICYTE_CONSTRUCTION": False,
                "USE_PERICYTE_MASK_CONSTRICTION": False,
                "USE_PROBABILISTIC_PERICYTE_CONSTRICTION": False,
                "RUN_PERICYTE_RESISTANCE_COMPARISON": False,
                "USE_FWHM_EDGE_DIAMETERS": False,
                "VTK_export": False,
                "VISUALIZE_VTK": False,
                "STATISTICS": True,
                "STATISTICS_MODE": "full",
                "VERBOSE_LOGGING": True,
                "VISUALIZE_RESULTS": True,
                "SHOW_PLOTS_IN_IDE": False,
                "HOLD_IDE_PLOTS_OPEN": False,
                "INTERACTIVE_PLOTS": False,
                "FINAL_RENDER_MODE": "3d",
            },
        },
        "distance_to_mask_3d": {
            "description": (
                "Run 3D distance-to-mask measurement workflow with vessel assignment (no statistics)."
            ),
            "overrides": {
                "DO_SKELETONIZE": True,
                "DO_GRAPH_BUILDING": True,
                # Ensure branch-order vessel assignment is performed via mask-driven automation.
                "AUTOMATED_VESSEL_ASSIGNMENT": True,
                "USE_LARGE_VESSEL_MASKS": True,
                "USE_ILASTIK_LARGE_VESSEL_SEGMENTATION": False,
                "USE_SMALL_VESSEL_MASKS_FOR_BOUNDARY_ASSIGNMENT": True,
                "USE_ILASTIK_SMALL_VESSEL_SEGMENTATION": False,
                "WRITE_SMALL_VESSEL_BOUNDARY_LABELLING_3D_HTML": True,
                "RUN_HAEMODYNAMICS": False,
                "DO_EQUIV_RESISTANCE_CALCULATION": False,
                "STATISTICS": False,
                "MEASUREMENT_3D_TO_CELL_MASK": True,
                # Keep unset by default; provide per run:
                # --set CELL_MASK_PATH='C:/path/to/cell_mask.tif'
                "CELL_MASK_PATH": None,
                "CELL_MASK_H5_DATASET_NAME": None,
                "MEASUREMENT_3D_VESSEL_MASK_PATH": None,
                "MEASUREMENT_3D_VESSEL_MASK_H5_DATASET_NAME": None,
                "MEASUREMENT_3D_REFERENCE_IMAGE_PATH": None,
                "MEASUREMENT_3D_REFERENCE_H5_DATASET_NAME": None,
                "VERBOSE_LOGGING": True,
                "VISUALIZE_RESULTS": True,
                "SHOW_PLOTS_IN_IDE": False,
                "HOLD_IDE_PLOTS_OPEN": False,
                "INTERACTIVE_PLOTS": False,
                "VTK_export": False,
                "VISUALIZE_VTK": False,
                "FINAL_RENDER_MODE": "3d",
            },
        },
    }
