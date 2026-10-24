"""Presets: named sets of settings to start a run from.

A preset is a partial config. Loading one sets the settings it names and
leaves every other as it is, so a preset holds nothing once loaded: a config
loaded afterwards, or a row changed by hand, replaces what it set. The
panel's Preset menu (under Load config...) and the examples' ``--preset``
flag both read :data:`PRESETS`.

The :data:`USAGE_PRESETS` are what a run is for. Between them they name
every setting any of them sets, wherever it would be read, so choosing one
gives the same run whichever was chosen before it. The rest add one thing --
vessel masks, thick vessels, a large image, a full report -- and name only
their own settings, so they go on top of a usage.

A preset picks switches and choices only. It names no file, which belongs to
the image being analysed, and no numeric threshold, which is calibrated on
marked data rather than chosen for a usage.
"""
from __future__ import annotations

from typing import Any, Mapping, Optional

from haemolynx.parsers.schema import Schema

#: The menu entry for no preset, the panel's starting point: it loads nothing.
NO_PRESET = "None"

PRESETS: dict[str, dict[str, Any]] = {
    "quick_look": {
        "label": "Quick look",
        "description": (
            "A fast first pass to check the network: the flow is solved, but "
            "no statistics, measurements, equivalent resistance or VTK files"
        ),
        "overrides": {
            "run_haemodynamics": True,
            "measure_diameters_from_raw_image": False,
            "use_edt_diameter_crosscheck": False,
            "do_equiv_resistance_calculation": False,
            "statistics": False,
            "compute_vascular_communities": False,
            "measurement_3d_to_cell_mask": False,
            "measure_tissue_volume": False,
            "vtk_export": False,
        },
    },
    "network_structure": {
        "label": "Network structure, no flow",
        "description": (
            "The network and its statistics (lengths, branching, tortuosity, "
            "connectivity) without solving blood flow"
        ),
        "overrides": {
            "run_haemodynamics": False,
            "statistics": True,
            "statistics_network_analysis": True,
            "measurement_3d_to_cell_mask": False,
        },
    },
    "flow_segmentation_diameters": {
        "label": "Blood flow, segmentation diameters",
        "description": (
            "Blood flow solved with each vessel's diameter measured from the "
            "segmented image, so no raw image is needed, then the statistics "
            "and VTK files"
        ),
        "overrides": {
            "run_haemodynamics": True,
            "measure_diameters_from_raw_image": False,
            "use_edt_diameter_crosscheck": True,
            "edt_diameter_prefer_over_table_on_fwhm_failure": True,
            "do_equiv_resistance_calculation": True,
            "statistics": True,
            "statistics_network_analysis": True,
            "measurement_3d_to_cell_mask": False,
            "vtk_export": True,
        },
    },
    "flow_raw_image_diameters": {
        "label": "Blood flow, raw image diameters",
        "description": (
            "Blood flow solved with each vessel's diameter measured from the "
            "raw image by the method its stain calls for, and from the "
            "segmentation where the raw image cannot be read, then the "
            "statistics and VTK files; needs the raw image on the Input tab"
        ),
        "overrides": {
            "run_haemodynamics": True,
            "measure_diameters_from_raw_image": True,
            "use_edt_diameter_crosscheck": True,
            "edt_diameter_prefer_over_table_on_fwhm_failure": True,
            "do_equiv_resistance_calculation": True,
            "statistics": True,
            "statistics_network_analysis": True,
            "measurement_3d_to_cell_mask": False,
            "vtk_export": True,
        },
    },
    "cell_distances": {
        "label": "Cell-to-vessel distances",
        "description": (
            "How far each object in a cell mask lies from the vessels, without "
            "solving blood flow or computing statistics; choose the cell mask "
            "on the Additional measurements tab"
        ),
        "overrides": {
            "run_haemodynamics": False,
            "statistics": False,
            "measurement_3d_to_cell_mask": True,
        },
    },
    "arteriole_venule_masks": {
        "label": "Arteriole and venule masks",
        "description": (
            "Inlets, outlets and arteriole/venule boundaries taken from large "
            "and small vessel masks, filled in from files named like the input "
            "image or chosen on the Vessel masks tab"
        ),
        "overrides": {
            "automated_vessel_assignment": True,
            "use_large_vessel_masks": True,
            "inlets_outlets_from_vessel_masks": True,
            "cut_network_at_large_vessel_volumes": True,
            "use_small_vessel_masks_for_boundary_assignment": True,
        },
    },
    "thick_vessels": {
        "label": "Thick vessels",
        "description": (
            "For large arterioles or venules among the capillaries: each fat "
            "vessel skeletonised as one centreline tree instead of Lee "
            "thinning's sheet, the capillaries still by Lee"
        ),
        "overrides": {
            "use_thick_vessel_skeletonisation": True,
        },
    },
    "large_image": {
        "label": "Large image, low memory",
        "description": (
            "For a stack too large to hold in memory: the volumes kept on disk "
            "rather than in RAM, and the flow and equivalent resistance solved "
            "with sparse matrices; the same results to rounding, read more slowly"
        ),
        "overrides": {
            "use_memmap_loading": True,
            "haemodynamics_solver": "sparse",
            "equivalent_resistance_solver": "sparse",
        },
    },
    "full_report": {
        "label": "Full report",
        "description": (
            "For a run whose numbers will be reported: the statistics with "
            "path efficiency, communities and betweenness computed exactly "
            "rather than sampled (slow on a large network), vascular "
            "communities, VTK files and the citations file"
        ),
        "overrides": {
            "statistics": True,
            "statistics_network_analysis": True,
            "statistics_mode": "full",
            "compute_vascular_communities": True,
            "vtk_export": True,
            "export_citations": True,
        },
    },
}

#: The presets that say what a run is for; every other adds to one of them.
USAGE_PRESETS = (
    "quick_look",
    "network_structure",
    "flow_segmentation_diameters",
    "flow_raw_image_diameters",
    "cell_distances",
)


def check_presets(presets: Mapping[str, Mapping[str, Any]], schema: Schema) -> None:
    """Raise ValueError if a preset names or mis-types a setting.

    Each value is checked on its own rather than through ``Schema.validate``: a
    preset is a partial config, so its prerequisites are usually satisfied by
    the settings it is loaded over, not by the preset itself.
    """
    for name, spec in presets.items():
        unknown = sorted(k for k in spec["overrides"] if k not in schema)
        if unknown:
            raise ValueError(f"Preset '{name}' sets unknown settings: {unknown}")
        for setting_name, value in spec["overrides"].items():
            schema[setting_name].coerce(value)


def preset_settings(name: str, schema: Schema) -> dict[str, Any]:
    """The settings preset *name* sets, each coerced to its setting's kind."""
    if name not in PRESETS:
        raise KeyError(f"No preset '{name}'; the presets are {sorted(PRESETS)}")
    return {
        setting: schema[setting].coerce(value)
        for setting, value in PRESETS[name]["overrides"].items()
    }


def preset_menu() -> list[Optional[tuple[str, Optional[str]]]]:
    """The panel's Preset menu, in order: ``(text, preset name)`` entries,
    :data:`NO_PRESET` first standing for no preset (``None``), then the
    usages, then the rest, each group after a separator (a ``None`` entry)."""
    usages = [name for name in PRESETS if name in USAGE_PRESETS]
    add_ons = [name for name in PRESETS if name not in USAGE_PRESETS]
    menu: list[Optional[tuple[str, Optional[str]]]] = [(NO_PRESET, None)]
    for group in (usages, add_ons):
        menu.append(None)
        menu.extend((PRESETS[name]["label"], name) for name in group)
    return menu


__all__ = [
    "NO_PRESET",
    "PRESETS",
    "USAGE_PRESETS",
    "check_presets",
    "preset_menu",
    "preset_settings",
]
