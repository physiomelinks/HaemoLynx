"""Hover text for napari panel controls that are not schema settings.

Setting rows take their tooltip from ``Setting.help`` via
:func:`haemolynx.gui.form.field_for`. Buttons and checkboxes that live beside
those rows need their own strings, kept here so they stay testable without Qt.
"""
from __future__ import annotations

#: Boundaries tab — shared picture controls.
SHOW_BOUNDARIES_TOOLTIP = (
    "Draw or refresh the BC coordinate and region layers from the current "
    "boundary settings"
)
SNAP_BOUNDARIES_TOOLTIP = (
    "Move each selected BC coordinate to the nearest degree-1 terminal node"
)

#: Boundaries tab — per-role pick/draw controls (when enabled).
ACTION_TOOLTIPS: dict[str, str] = {
    "pick": (
        "Enter add mode on the BC coordinates layer and click to place "
        "points for this role"
    ),
    "draw": (
        "Draw a rectangle in the 2D view; its extruded box becomes this "
        "role's volume region"
    ),
    "depth": (
        "Z extent of the next region drawn for this role, in microns"
    ),
    "move": (
        "Select the BC layer so you can drag or delete points and regions "
        "already placed for this role"
    ),
    "assign": (
        "Give the currently selected coordinates or regions this boundary "
        "role"
    ),
    "clear": (
        "Remove every volume region belonging to this role from the BC "
        "shapes layer"
    ),
}

#: Panel chrome below the stage tabs.
LOAD_CONFIG_TOOLTIP = (
    "Open a YAML config into these form rows without loading the image "
    "paths it names"
)
SAVE_CONFIG_TOOLTIP = (
    "Write the current form values to a YAML config file"
)
RUN_CHECKS_TOOLTIP = (
    "Run preflight checks on the current settings without starting a pipeline"
)
RUN_PIPELINE_TOOLTIP = (
    "Run the pipeline stages with the current settings"
)
CLEAR_LAYERS_TOOLTIP = (
    "Remove HaemoLynx-drawn result and boundary layers from the viewer, "
    "forget in-memory checkpoints, and discard cached resume/checkpoint "
    "pickles on disk"
)
SHOW_RESULTS_TOOLTIP = (
    "After each stage finishes, add its graph or mask layers to the viewer"
)
SHOW_STEPS_TOOLTIP = (
    "During graph build, also show each topology step as its own layer set"
)
USE_LAYER_TOOLTIP = (
    "Point the run at the image layer chosen above (its path, or an export "
    "of its array)"
)
REVERT_STAGE_TOOLTIP = (
    "Reload the checkpoint from the previous stage so later settings can "
    "be changed without rebuilding earlier work"
)

#: Left-hand view dock — display-only; never written into pipeline settings.
Z_PROJECT_TOOLTIP = (
    "Restrict every visible layer to this physical Z window in microns; "
    "full range leaves the display unchanged and does not crop the pipeline"
)
SCALE_BAR_TOOLTIP = (
    "Show napari's scale bar on the canvas, in microns when voxel size is known"
)
