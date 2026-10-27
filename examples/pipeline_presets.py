"""Named sets of setting overrides for the resistance pipeline.

A preset is just a partial config: the settings it names replace what the
config file says, and everything else is untouched. The panel's presets
(``haemolynx.pipeline.presets``) are offered here too, beside these, which
name files and command-line plotting of their own. Every name is validated
against the pipeline's schema when this module is imported, so a preset
cannot quietly set something that no longer exists.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _path in (_ROOT / "src", _ROOT / "examples"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from haemolynx.pipeline.presets import PRESETS as PANEL_PRESETS, check_presets  # noqa: E402
from haemolynx.pipeline.schema import default_schema  # noqa: E402

SCHEMA = default_schema()

_EXAMPLE_PRESETS: dict[str, dict] = {
    "all_automated": {
        "description": (
            "Fully automated profile: large/small vessel masks and FWHM "
            "diameters. Baseline never places focal pericytes "
            "(do_pericyte_construction is inert); add a pericyte-typed "
            "perturbation for mask/periodic constrictions."
        ),
        "overrides": {
            'automated_vessel_assignment': True,
            'do_pericyte_construction': False,
            'final_render_mode': '3d',
            'raw_image_path': '/home/farg967/Documents/git_projects/haemolynx/examples/images/Nerve_capillaries.tif',
            'hold_ide_plots_open': False,
            'interactive_plots': False,
            'pericyte_mask_path': None,
            'run_pericyte_resistance_comparison': False,
            'show_plots_in_ide': False,
            'statistics': True,
            'statistics_mode': 'full',
            'measure_diameters_from_raw_image': True,
            'raw_image_stain': 'plasma_column',
            'use_ilastik_large_vessel_segmentation': False,
            'use_ilastik_small_vessel_segmentation': False,
            'use_large_vessel_masks': True,
            'use_pericyte_mask_constriction': True,
            'use_probabilistic_pericyte_constriction': False,
            'use_small_vessel_masks_for_boundary_assignment': True,
            'verbose_logging': True,
            'visualize_results': True,
            'visualize_vtk': False,
            'vtk_export': True,
            'write_small_vessel_boundary_labelling_3d_html': True,
        },
    },
    "automated_assignment": {
        "description": 'Automated vessel assignment profile using pre-segmented large and small vessel masks (no ilastik).',
        "overrides": {
            'automated_vessel_assignment': True,
            'do_pericyte_construction': False,
            'final_render_mode': '3d',
            'hold_ide_plots_open': False,
            'interactive_plots': False,
            'pericyte_mask_path': None,
            'run_pericyte_resistance_comparison': False,
            'show_plots_in_ide': False,
            'statistics': True,
            'statistics_mode': 'full',
            'measure_diameters_from_raw_image': False,
            'use_ilastik_large_vessel_segmentation': False,
            'use_ilastik_small_vessel_segmentation': False,
            'use_large_vessel_masks': True,
            'use_pericyte_mask_constriction': False,
            'use_probabilistic_pericyte_constriction': False,
            'use_small_vessel_masks_for_boundary_assignment': True,
            'verbose_logging': True,
            'visualize_results': True,
            'visualize_vtk': False,
            'vtk_export': True,
            'write_small_vessel_boundary_labelling_3d_html': True,
        },
    },
    "automated_assignment_ilastik": {
        "description": 'Automated vessel assignment profile using ilastik segmentation for large and small vessel masks.',
        "overrides": {
            'automated_vessel_assignment': True,
            'do_pericyte_construction': False,
            'final_render_mode': '3d',
            'hold_ide_plots_open': False,
            'interactive_plots': False,
            'pericyte_mask_path': None,
            'run_pericyte_resistance_comparison': False,
            'show_plots_in_ide': False,
            'statistics': True,
            'statistics_mode': 'full',
            'measure_diameters_from_raw_image': False,
            'use_ilastik_large_vessel_segmentation': True,
            'use_ilastik_small_vessel_segmentation': True,
            'use_large_vessel_masks': True,
            'use_pericyte_mask_constriction': False,
            'use_probabilistic_pericyte_constriction': False,
            'use_small_vessel_masks_for_boundary_assignment': True,
            'verbose_logging': True,
            'visualize_results': True,
            'visualize_vtk': False,
            'vtk_export': True,
            'write_small_vessel_boundary_labelling_3d_html': True,
        },
    },
    "default": {
        "description": 'Current baseline behavior from this settings file.',
        "overrides": {},
    },
    "size_classes": {
        "description": (
            "Vessel classes by size, without masks: every vessel classed by its measured "
            "width, mid at least 6 um and wide at least 15 um (Harvey's convention, "
            "2026-10-27). No branch orders or sides. The widths are the mask's own "
            "where the raw image is not measured."
        ),
        "overrides": {
            'automated_vessel_assignment': False,
            'classify_vessels_by_size': True,
            'vessel_size_mid_min_diameter_um': 6.0,
            'vessel_size_wide_min_diameter_um': 15.0,
            'use_edt_diameter_crosscheck': True,
            'edt_diameter_prefer_over_table_on_fwhm_failure': True,
        },
    },
    "distance_to_mask_3d": {
        "description": 'Run 3D distance-to-mask measurement workflow with vessel assignment (no statistics).',
        "overrides": {
            'automated_vessel_assignment': True,
            'cell_mask_h5_dataset_name': None,
            'cell_mask_path': None,
            'do_equiv_resistance_calculation': False,
            'do_graph_building': True,
            'do_skeletonize': True,
            'final_render_mode': '3d',
            'hold_ide_plots_open': False,
            'interactive_plots': False,
            'measurement_3d_to_cell_mask': True,
            'measurement_3d_vessel_mask_h5_dataset_name': None,
            'measurement_3d_vessel_mask_path': None,
            'run_haemodynamics': False,
            'show_plots_in_ide': False,
            'statistics': False,
            'strict_branch_order_assignment': True,
            'use_ilastik_large_vessel_segmentation': False,
            'use_ilastik_small_vessel_segmentation': False,
            'use_large_vessel_masks': True,
            'use_small_vessel_masks_for_boundary_assignment': True,
            'verbose_logging': True,
            'visualize_results': True,
            'visualize_vtk': False,
            'vtk_export': False,
            'write_small_vessel_boundary_labelling_3d_html': True,
        },
    },
    "full_fwhm": {
        "description": 'Enable full FWHM diameter workflow with detailed outputs for raw-image-aligned analysis.',
        "overrides": {
            'raw_image_path': '/home/farg967/Documents/git_projects/haemolynx/examples/images/Nerve_capillaries.tif',
            'hold_ide_plots_open': False,
            'interactive_plots': False,
            'show_plots_in_ide': False,
            'statistics': True,
            'statistics_mode': 'full',
            'measure_diameters_from_raw_image': True,
            'raw_image_stain': 'plasma_column',
            'verbose_logging': True,
            'visualize_results': True,
        },
    },
    "publication": {
        "description": 'High-detail reporting mode: writes full stats/artifacts with non-interactive rendering.',
        "overrides": {
            'final_render_mode': '3d',
            'hold_ide_plots_open': False,
            'interactive_plots': False,
            'show_plots_in_ide': False,
            'statistics': True,
            'statistics_mode': 'full',
            'verbose_logging': False,
            'visualize_results': True,
            'visualize_vtk': False,
            'vtk_export': True,
            'write_small_vessel_boundary_labelling_3d_html': True,
        },
    },
    "quick_debug": {
        "description": 'Fast iteration mode: disables heavy outputs/analysis and keeps logs concise.',
        "overrides": {
            'do_equiv_resistance_calculation': False,
            'do_pericyte_construction': False,
            'hold_ide_plots_open': False,
            'interactive_plots': False,
            'measurement_3d_to_cell_mask': False,
            'run_pericyte_resistance_comparison': False,
            'show_plots_in_ide': False,
            'statistics': False,
            'measure_diameters_from_raw_image': False,
            'verbose_logging': True,
            'visualize_results': False,
            'visualize_vtk': False,
            'vtk_export': False,
            'write_small_vessel_boundary_labelling_3d_html': False,
        },
    },
    "statistics_only": {
        "description": 'Skeletonize/build graph + assign vessels + compute statistics with haemodynamics enabled but no equivalent-resistance solve.',
        "overrides": {
            'do_equiv_resistance_calculation': False,
            'do_graph_building': True,
            'do_pericyte_construction': False,
            'do_skeletonize': True,
            'final_render_mode': '3d',
            'hold_ide_plots_open': False,
            'interactive_plots': False,
            'run_haemodynamics': True,
            'run_pericyte_resistance_comparison': False,
            'show_plots_in_ide': False,
            'statistics': True,
            'statistics_mode': 'full',
            'measure_diameters_from_raw_image': False,
            'use_pericyte_mask_constriction': False,
            'use_probabilistic_pericyte_constriction': False,
            'verbose_logging': True,
            'visualize_results': True,
            'visualize_vtk': False,
            'vtk_export': False,
        },
    },
}


_shared = sorted(PANEL_PRESETS.keys() & _EXAMPLE_PRESETS.keys())
if _shared:
    raise ValueError(f"Presets named both here and in the panel's list: {_shared}")
PRESETS: dict[str, dict] = {**PANEL_PRESETS, **_EXAMPLE_PRESETS}

check_presets(PRESETS, SCHEMA)

