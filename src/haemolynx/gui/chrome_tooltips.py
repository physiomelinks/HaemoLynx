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
        "Add volume regions for this role: in 2D draw rectangles (each "
        "extruded by the region depth), in 3D drag a box across the view"
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
    "pick_nodes": (
        "Click nodes of the run's graph to list their IDs for this role; "
        "clicking a listed node again takes it off. Needs '3. Graph' run first"
    ),
    "clear_nodes": (
        "Empty this role's node ID list"
    ),
}

#: The node-picking button while clicks are going to its role.
STOP_PICKING_NODES_TOOLTIP = (
    "Stop picking nodes: clicks in the viewer go back to turning the view"
)

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
    "Reopen the HaemoLynx view panel (Z-depth, vessel draw mode and colouring, "
    "scale bar and snapshot) if it has been closed; does nothing while it is "
    "already open"
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
COLOUR_BY_TOOLTIP = (
    "What the vessels on screen are coloured by, drawn as tubes or as lines: "
    "the same choice as Colour by on the vessels layer's own controls, for "
    "the network chosen under Showing"
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
#: "6. Post processing" tab.
POST_PROCESSING_TOOLTIPS: dict[str, str] = {
    "scan": (
        "Take the last run's network into this tab to edit it, and count the "
        "junctions where four or more vessels meet"
    ),
    "junction_correction": (
        "Show the junctions where four or more vessels meet, to delete some "
        "of their vessels or split them into bifurcations; while it is off "
        "they are neither marked in the viewer nor zoomed to"
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
        "inlets, outlets and boundary nodes are never cut off. Press it again "
        "to stop"
    ),
    "add": (
        "Then click a node, or a point on a vessel where a new node should "
        "form, and click along the new vessel: each click is traced from the "
        "last through the image chosen beside this button, and a click on "
        "another node or vessel finishes it. It takes the mean diameter of the "
        "vessels at its ends; press it again to cancel"
    ),
    "trace_source": (
        "What Add vessel traces each leg through: the segmented mask, or the "
        "raw data (FWHM's image, or the Raw data file on the Input tab)"
    ),
    "branch_ids": (
        "The branchIDs to delete, as the vessel hover shows them, separated "
        "by commas or spaces; Enter deletes them"
    ),
    "delete_ids": (
        "Delete the vessels with the branchIDs typed beside this button; "
        "inlets, outlets and boundary nodes are never cut off"
    ),
    "prune": (
        "Remove every piece of the network that no longer has both an inlet "
        "and an outlet, such as a branch whose only link to the rest was deleted"
    ),
    "log": (
        "Every change made in this tab, oldest first, with the branchIDs as "
        "they were when it was made; the same lines go to the napari log"
    ),
    "regenerate": (
        "After a finished run: bring the edits in line with the rest of the "
        "network, then rerun Haemodynamics, Solve, Perturbations and Export on "
        "it, so the 3D view and outputs catch up with the edits"
    ),
    "regenerate_graph": (
        "Bring the edits in line with the rest of the network: branch orders "
        "assigned again, each edited vessel's length measured along its path and "
        "its diameter by the run's own methods, and a zero-resistance bridge "
        "where it opens into a thick vessel. While a run is paused here it stays "
        "paused; after a finished run, Haemodynamics to Export are run again"
    ),
    "continue": (
        "Carry on the run paused here: bring any edits in line, then run "
        "Haemodynamics, Solve, Perturbations and Export. A run pauses here when "
        "Mid-run postprocessing (1. Input) is on"
    ),
}

#: Diameters tab — the one choice standing for use_fwhm_edge_diameters and
#: use_endothelial_diameters, which cannot both be on.
DIAMETER_SOURCE_TOOLTIP = (
    "Where each vessel's diameter is measured first: FWHM across the plasma "
    "label in the raw image, the lumen inside an endothelial stain's wall, or "
    "neither; a vessel with no measurement falls back to the mask estimate if "
    "it is on, then the diameter table"
)

#: Every tab — the button that shows or hides the less common settings
#: belonging to the row above it.
ADVANCED_SETTINGS_TOOLTIP = (
    "Show or hide the less often changed settings for the option above; "
    "the count says how many of them differ from their defaults"
)
