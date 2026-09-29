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
EDIT_GRAPH_TOOLTIP = (
    "Open the graph editor: add a branch by clicking along the segmented "
    "image, or delete a vessel by clicking it, then Regenerate to catch "
    "diameters, haemodynamics, perturbations, measurements and export up "
    "to the edit"
)
REOPEN_VIEW_TOOLTIP = (
    "Reopen the HaemoLynx view panel (Z-depth, vessel draw mode, scale bar "
    "and snapshot) if it has been closed; does nothing while it is already open"
)
SAVE_CONFIG_TOOLTIP = (
    "Write the current form values to a YAML config file; relative "
    "paths stay relative when saving beside the file they came from"
)
RUN_CHECKS_TOOLTIP = (
    "Run preflight checks on the current settings without starting a pipeline"
)
RUN_PIPELINE_TOOLTIP = (
    "Run the full pipeline from the first stage with the current settings"
)
CLEAR_LAYERS_TOOLTIP = (
    "Remove HaemoLynx-drawn result and boundary layers from the viewer, "
    "stop a run in progress, forget in-memory checkpoints, restore skip "
    "toggles, and discard this session's resume/checkpoint pickles on disk"
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
OPTIMISE_SETTINGS_TOOLTIP = (
    "Empirically choose Skeletonise and Graph tab settings from the "
    "segmented input image, and write a config file beside it"
)
CHECK_SEGMENTED_IMAGE_TOOLTIP = (
    "Score the segmented input image out of 10 -- fragmentation, "
    "connectivity, vessels leaving the image, surface noise and resolution "
    "-- and print the breakdown to the log below"
)
REVERT_STAGE_TOOLTIP = (
    "Clear this tab and later layers and checkpoints, then rerun the "
    "pipeline from this stage using the previous tab's saved work"
)

#: View dock — display-only; never written into pipeline settings.
Z_DEPTH_TOOLTIP = (
    "Clip every displayed layer to this physical Z window; full range is the identity and does not crop the pipeline"
)
VESSEL_DRAW_TOOLTIP = (
    "Tubes: each centreline step as a 3D prism that stays visible from every "
    "camera angle. Lines: napari vector ribbons (faster, can vanish edge-on)"
)
LAYER_SET_TOOLTIP = (
    "Which network the vessels, nodes and flow-direction layers show: the "
    "baseline, or one perturbation. Swapping keeps the same kinds of layer on, "
    "so flipping back and forth compares like with like"
)
SCALE_BAR_TOOLTIP = (
    "Show napari's scale bar in the bottom-right of the canvas, in microns when voxel size is known"
)
#: Canvas buttons, bottom-left: centre the data and look straight at a plane.
VIEW_SNAP_TOOLTIPS = {
    "XY": "Centre the view and look down z at the XY plane",
    "XZ": "Centre the view and look along y at the XZ plane, z increasing downwards",
    "YZ": "Centre the view and look along x at the YZ plane, z increasing downwards",
}
SWEEP_TOOLTIP = (
    "Step through the chosen perturbation's sweep: each grid point's flows, "
    "recoloured in place. Shown for a sweep perturbation chosen under Showing; "
    "each sweep keeps its own position while another is shown"
)
TUBE_QUALITY_TOOLTIP = (
    "How the vessel tubes are drawn. Left: separate six-sided prisms per "
    "centreline step, flat shaded (fastest; reads as bands). Each step right: "
    "one continuous, capped tube per vessel with rounder cross-sections and "
    "smooth shading -- slower to build on a large network"
)
SNAPSHOT_TOOLTIP = (
    "Write a TIFF of the current napari view into the pipeline "
    "output folder (cosmetic export only; does not feed the run)"
)
SAVE_RUN_TOOLTIP = (
    "Write this pipeline run to a .haemorun file: settings, graph, "
    "checkpoints, and viewer layers, so it can be opened again later"
)
LOAD_RUN_TOOLTIP = (
    "Clear the current HaemoLynx layers and state, then restore a saved "
    "pipeline run so the viewer looks as it did when that run finished"
)
#: "10. Post processing" tab.
POST_PROCESSING_TOOLTIPS: dict[str, str] = {
    "scan": (
        "Look at the last run's network and list every junction where four "
        "or more vessels meet"
    ),
    "junctions": (
        "Junctions where four or more vessels meet; click one to zoom to it "
        "and list its vessels below"
    ),
    "table": (
        "The vessels meeting at the chosen junction; click or shift/ctrl-click "
        "rows to select vessels, drawn thick and yellow in the viewer"
    ),
    "delete": (
        "Remove the selected vessels; a junction left joining just two "
        "vessels is merged into one, and inlets, outlets and boundary nodes "
        "are never removed"
    ),
    "split": (
        "Turn the chosen junction into bifurcations: the two vessels leaving "
        "in the most similar directions move to a new node joined back by a "
        "connector vessel of the length beside this button"
    ),
    "connector": (
        "Length of the connector vessel a split inserts between the two new "
        "bifurcations, in microns"
    ),
    "leave": "Keep the chosen junction as it is and move on to the next one",
    "click_delete": (
        "Then click vessels in the viewer to remove them, one per click; "
        "inlets, outlets and boundary nodes are never cut off"
    ),
    "add": (
        "Then click two nodes in the viewer to join them with a new vessel, "
        "routed through the segmented image where it can be (straight where "
        "not), with the mean diameter of the vessels already at those nodes"
    ),
    "stop": "Stop deleting or adding by clicking in the viewer",
    "regenerate": (
        "Rerun Diameters, Haemodynamics, Solve, Perturbations and Export on "
        "the edited network, so the 3D view and outputs catch up with the edits"
    ),
}
