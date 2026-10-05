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
    "insert_box": (
        "Put a box of the size below in the middle of the view, then move and "
        "resize it to take in the vessel end you want"
    ),
    "box_choice": (
        "Which of this role's boxes the size, arrows and node list act on"
    ),
    "box_size": (
        "The box's size along z, y and x in microns; changing it resizes the "
        "chosen box about its centre"
    ),
    "box_colour": (
        "Choose the colour this role's boxes are drawn in, in 3D and 2D. "
        "Display only: it changes no setting"
    ),
    "box_step": (
        "How far, in microns, one press of an arrow moves the box"
    ),
    "box_move": (
        "Move the box by the step above, as the view shows it: left/right and "
        "up/down across the screen; back/forward through the slices in 2D, "
        "towards or away from you in 3D however the view is turned"
    ),
    "box_scan": (
        "List and mark the nodes inside the chosen box. Moving or resizing "
        "the box does not scan, so it stays quick to steer; scan once it is "
        "where you want it"
    ),
    "box_nodes": (
        "Every node inside the chosen box, open ends (vessel ends) first; they "
        "are drawn yellow in the viewer, open ends larger with a red rim. "
        "Click one to mark it; double-click to use it"
    ),
    "use_node": (
        "Make the node selected in the list one of this role's nodes: it is "
        "added to the role's node IDs and the role switches to node_ids, so a "
        "run takes exactly that node"
    ),
    "remove_box": (
        "Delete the chosen box"
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
HELP_TOOLTIP = (
    "Open the help menu on the tab that is currently open"
)
USE_LAYER_TOOLTIP = (
    "Point the run at the image layer chosen above (its path, or an export "
    "of its array)"
)
OPTIMISE_SETTINGS_TOOLTIP = (
    "Empirically choose Skeletonise and Graph tab settings from the "
    "segmented input image, write a config file beside it, and show what "
    "would change for you to Apply or Discard"
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
#: The view window's Illumination box: how the canvas lights every surface.
ILLUMINATION_TOOLTIPS = {
    "shading": (
        "How the canvas lights a surface: no lighting, flat facets, or "
        "smooth shading"
    ),
    "ambient": (
        "Light that fills every surface equally, including the parts facing "
        "away from the lamp"
    ),
    "diffuse": (
        "Light from the lamp on the canvas, stronger on surfaces facing it"
    ),
    "specular": (
        "Brightness of the highlight on surfaces facing the lamp"
    ),
    "shininess": (
        "How small that highlight is: a higher value is a tighter glint"
    ),
}
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
    "How round the vessel tubes are drawn: one smooth tube per vessel at its "
    "own diameter, with more sides round it each step right -- slower to "
    "build on a large network"
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
#: "7. Post processing" tab.
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
        "Remove every vessel no inlet-to-outlet path runs along: each piece "
        "of the network that no longer has both an inlet and an outlet, such "
        "as a branch whose only link to the rest was deleted, and each "
        "dead-end branch, tree or loop that reaches no outlet. The same rule "
        "as Network handling's remove_disconnected"
    ),
    "connectivity_choice": (
        "Which vessels the connectivity CSV lists: every vessel of the network "
        "in the viewer, or only those some inlet-to-outlet path runs through "
        "(no dead ends, loops off a single node or pieces without both an "
        "inlet and an outlet)"
    ),
    "connectivity_map": (
        "Draw the connectivity CSV exported last (or one you pick) as a 2D map "
        "in the browser: inlets on the left, outlets on the right, every "
        "vessel a line you can hover for its IDs, length and diameter"
    ),
    "export_connectivity": (
        "Save a CSV of how the network is connected: one row per vessel, from "
        "node to node oriented away from the inlets, with its branch order, "
        "length, diameter, upstream and downstream vessels, and the IDs the "
        "VTK and .pkl exports use"
    ),
    "log": (
        "Every change made in this tab, oldest first, with the branchIDs as "
        "they were when it was made; the same lines go to the napari log"
    ),
    "regenerate": (
        "After a finished run: bring the edits in line with the rest of the "
        "network, rerun Haemodynamics on the edited network, then Perturbations "
        "and Export, so the 3D view and outputs catch up with the edits"
    ),
    "regenerate_graph": (
        "Bring the edits in line with the rest of the network -- branch orders "
        "assigned again, each edited vessel's length measured along its path and "
        "its diameter by the run's own methods, and a zero-resistance bridge "
        "where it opens into a thick vessel -- and rerun Haemodynamics on it. "
        "While a run is paused here it stays paused; after a finished run, "
        "Perturbations and Export are run again too"
    ),
    "continue": (
        "Carry on the run paused here: bring any edits in line and rerun "
        "Haemodynamics on them, then run Perturbations and Export. A run pauses "
        "here, after Haemodynamics, when Mid-run postprocessing (1. Input) is on"
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

#: The optimisers' own controls: the review of a run's proposal, and the
#: options each search takes.
OPTIMISE_TOOLTIPS = {
    "review": (
        "What the last optimisation run would change, from your value to its "
        "choice, and any measure it made worse than your settings"
    ),
    "apply": "Write the proposed settings into the panel",
    "discard": "Leave the panel as it is and drop the proposed settings",
    "passes": (
        "Run the whole search this many times, each pass starting where the "
        "last one ended, to catch settings that only help together; stops "
        "early once a pass changes nothing"
    ),
    "fwhm_options": "Show or hide the FWHM optimiser's own options",
    "fwhm_choose_groups": (
        "Restrict Optimise FWHM settings to only the ticked group(s) below, "
        "instead of every FWHM setting"
    ),
    "fwhm_sample": (
        "How many of the network's vessels each FWHM trial measures; Auto "
        "times one measurement and picks a count to keep the search near "
        "three minutes"
    ),
}
