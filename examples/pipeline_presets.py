"""Named sets of setting overrides for the resistance pipeline.

A preset is just a partial config: the settings it names replace what the
config file says, and everything else is untouched. Every name is validated
against the pipeline's schema when this module is imported, so
a preset cannot quietly set something that no longer exists.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _path in (_ROOT / "src", _ROOT / "examples"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from haemolynx.pipeline.schema import default_schema  # noqa: E402

SCHEMA = default_schema()

PRESETS: dict[str, dict] = {
    "all_automated": {
        "description": 'Fully automated profile: large/small vessel masks, pericyte-mask constriction, and FWHM diameters.',
        "overrides": {
            'automated_vessel_assignment': True,
            'do_pericyte_construction': True,
            'final_render_mode': '3d',
            'fwhm_raw_tiff_path': '/home/farg967/Documents/git_projects/haemolynx/examples/images/Nerve_capillaries.tif',
            'hold_ide_plots_open': False,
            'interactive_plots': False,
            'pericyte_mask_path': None,
            'run_pericyte_resistance_comparison': False,
            'show_plots_in_ide': False,
            'statistics': True,
            'statistics_mode': 'full',
            'use_fwhm_edge_diameters': True,
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
            'use_fwhm_edge_diameters': False,
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
            'use_fwhm_edge_diameters': False,
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
            'measurement_3d_reference_h5_dataset_name': None,
            'measurement_3d_reference_image_path': None,
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
            'fwhm_raw_tiff_path': '/home/farg967/Documents/git_projects/haemolynx/examples/images/Nerve_capillaries.tif',
            'hold_ide_plots_open': False,
            'interactive_plots': False,
            'show_plots_in_ide': False,
            'statistics': True,
            'statistics_mode': 'full',
            'use_fwhm_edge_diameters': True,
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
            'use_fwhm_edge_diameters': False,
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
            'use_fwhm_edge_diameters': False,
            'use_pericyte_mask_constriction': False,
            'use_probabilistic_pericyte_constriction': False,
            'verbose_logging': True,
            'visualize_results': True,
            'visualize_vtk': False,
            'vtk_export': False,
        },
    },
}


def _check() -> None:
    """Fail at import if a preset names or mis-types a setting.

    Each value is checked on its own rather than through ``Schema.validate``: a
    preset is a partial config, so its prerequisites are usually satisfied by
    the config file it is applied to, not by the preset itself.
    """
    for name, spec in PRESETS.items():
        unknown = sorted(k for k in spec["overrides"] if k not in SCHEMA)
        if unknown:
            raise ValueError(f"Preset '{name}' sets unknown settings: {unknown}")
        for setting_name, value in spec["overrides"].items():
            SCHEMA[setting_name].coerce(value)


_check()

