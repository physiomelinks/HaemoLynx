"""Hover text for napari panel controls that are not schema settings.

Setting rows take their tooltip from ``Setting.help`` via
:func:`haemolynx.gui.form.field_for`. Buttons and checkboxes that live beside
those rows need their own strings, kept here so they stay testable without Qt.
"""
from __future__ import annotations

#: Boundaries tab — shared picture controls.
SHOW_BOUNDARIES_TOOLTIP = (
    "Draw or refresh the boundary points and regions in the viewer from the "
    "current boundary settings"
)
SNAP_BOUNDARIES_TOOLTIP = (
    "Move each selected boundary point onto the nearest vessel end"
)

#: Boundaries tab — per-role pick/draw controls (when enabled).
ACTION_TOOLTIPS: dict[str, str] = {
    "pick": (
        "Click in the viewer to place boundary points for this role"
    ),
    "draw": (
        "Draw regions for this role: rectangles in 2D, each given the region "
        "depth, or drag a box across the view in 3D"
    ),
    "depth": (
        "How deep in z the next region drawn for this role is, in um"
    ),
    "move": (
        "Select the boundary layer so you can drag or delete this role's "
        "points and regions"
    ),
    "assign": (
        "Give the selected points or regions this boundary role"
    ),
    "clear": (
        "Remove every region drawn for this role"
    ),
    "pick_nodes": (
        "Click nodes in the viewer to add them to this role's list; click a "
        "listed node again to remove it. Run \"3. Graph\" first"
    ),
    "clear_nodes": (
        "Empty this role's node ID list"
    ),
    "insert_box": (
        "Add a box of the size below in the middle of the view; move and "
        "resize it round the vessel end you want"
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
        "Move the box by the step above: across the screen, or back and "
        "forward through the slices in 2D and towards or away from you in 3D"
    ),
    "box_scan": (
        "List and mark the nodes inside the chosen box. Moving the box does "
        "not rescan, so press this once the box is in place"
    ),
    "box_nodes": (
        "The nodes inside the chosen box, vessel ends first, drawn yellow in "
        "the viewer (ends larger, with a red rim). Click one to mark it; "
        "double-click to use it"
    ),
    "use_node": (
        "Add the node selected in the list to this role's nodes and switch "
        "the role to node_ids, so a run uses exactly that node"
    ),
    "remove_box": (
        "Delete the chosen box"
    ),
}
#: The box tools' own parts: one size box per axis, and one button per move
#: (``boundary_picking.MOVE_DIRECTIONS``).
BOX_SIZE_TOOLTIPS: dict[str, str] = {
    axis: (
        f"The box's size along {axis} in microns; changing it resizes the "
        "chosen box about its centre"
    )
    for axis in "zyx"
}
BOX_MOVE_TOOLTIPS: dict[str, str] = {
    "left": "Move the box left across the screen by the move step",
    "right": "Move the box right across the screen by the move step",
    "up": "Move the box up the screen by the move step",
    "down": "Move the box down the screen by the move step",
    "back": (
        "Move the box back by the move step: back through the slices in 2D, "
        "towards you in 3D"
    ),
    "forward": (
        "Move the box forward by the move step: on through the slices in 2D, "
        "away from you in 3D"
    ),
}

#: The node-picking button while clicks are going to its role.
STOP_PICKING_NODES_TOOLTIP = (
    "Stop picking nodes: clicks in the viewer go back to turning the view"
)

#: Panel chrome below the stage tabs.
LOAD_CONFIG_TOOLTIP = (
    "Fill the settings from a YAML config file; nothing it names is opened, "
    "and an image layer already open stays the input"
)
PRESET_TOOLTIP = (
    "Start from a standard set of settings: a preset changes only the rows "
    "it names, and None changes nothing"
)
EDIT_GRAPH_TOOLTIP = (
    "Open the graph editor to add or delete vessels by clicking in the "
    "viewer; Regenerate then brings diameters, flows and outputs up to date"
)
REOPEN_VIEW_TOOLTIP = (
    "Reopen any closed tabs of the left-hand panel: the HaemoLynx view, "
    "Statistics, Layer list and Layer controls"
)
#: A Statistics tab button with no entry in STATISTICS_SECTION_TOOLTIPS.
STATISTICS_SECTION_TOOLTIP = (
    "Show or hide this measure's results from the last run's statistics"
)
#: The Statistics tab's buttons, one per section of the report
#: (``gui.statistics_view``): what the statistic behind each one measures
#: and how to read it. Keyed by section: the statistic measures
#: (``statistics.STATISTIC_MEASURES``), then ``statistics_view.OTHER_SECTIONS``.
STATISTICS_SECTION_TOOLTIPS: dict[str, str] = {
    "basic": (
        "Size of the network: how many nodes and vessels it has, their total "
        "and mean length, and how many vessels meet at a node on average"
    ),
    "tortuosity": (
        "How winding the vessels are: the tortuosity index (centreline length "
        "over the straight line between the ends, 1 is straight) and "
        "curvature, averaged over vessels"
    ),
    "branching": (
        "How many branch points the network has: nodes where three or more "
        "vessels meet"
    ),
    "tree_asymmetry": (
        "How unevenly the network divides: at each branch point, the size "
        "difference between its largest and smallest sub-tree, summed and "
        "scaled by node count. 0 is perfectly balanced"
    ),
    "fractal_dimension": (
        "How completely the vessels fill space, by box counting: about 1 for "
        "a single line, rising towards 3 for vessels filling the volume. R^2 "
        "is the fit's quality"
    ),
    "vessel_density": (
        "Vessel length per unit volume: of the measured tissue if there is "
        "one, else of the box the network spans, and of the whole image"
    ),
    "vessel_classes": (
        "The main measures for each vessel class side by side (arterioles, "
        "capillaries and venules, or vessels by size), including flow, "
        "velocity and wall shear stress once solved"
    ),
    "murray_law": (
        "Whether junction diameters follow Murray's law (parent^3 = sum of "
        "daughters^3): the Murray ratio (1 is a perfect fit) and the fitted "
        "exponent (3 is Murray's)"
    ),
    "daughter_angles": (
        "The angle between sibling vessels leaving a junction, averaged over "
        "every junction, in degrees"
    ),
    "intercapillary_distance": (
        "How far apart neighbouring vessels lie, which sets how far oxygen "
        "must diffuse; once the flow is solved, also for the vessels carrying "
        "flow alone"
    ),
    "junction_shape": (
        "The angles between the three vessels at each junction: smallest and "
        "largest, the share of junctions with two vessels within 45 degrees, "
        "and their sum (360 in one plane)"
    ),
    "tissue_coverage": (
        "How far tissue lies from the nearest vessel: the share within 20 um, "
        "the share beyond 50 um (holes), and the median and 90th percentile "
        "distance"
    ),
    "vessel_columns": (
        "Vessels running along the stack axis (within 30 degrees of z): their "
        "share of length, count per square mm, and how evenly they are spread "
        "(Clark-Evans ratio, 1 at random)"
    ),
    "dead_ends": (
        "Vessels that end without joining another, counted by kind (leaving "
        "the image, broken, blind and so on), per mm of vessel, with their "
        "lengths"
    ),
    "network_robustness": (
        "How much the network hangs on single links: vessels and junctions "
        "whose loss alone splits it, and those whose loss cuts an outlet off "
        "from an inlet"
    ),
    "bottlenecks": (
        "Where inlet-to-outlet flow is most constrained: the fewest vessels "
        "(or least conductance) whose blocking separates inlets from outlets, "
        "and each vessel's share of the shortest routes"
    ),
    "shunts": (
        "Vessels on inlet-to-outlet routes much shorter than the median route, "
        "carrying blood past most of the bed: how many, and the share of "
        "outflow they carry"
    ),
    "occlusion_impact": (
        "What blocking each vessel in turn does: the share of inlet-to-outlet "
        "flow lost and the length of other vessels left underperfused, for the "
        "network, by vessel type and branch order"
    ),
    "occlusion_curves": (
        "How flow falls as vessels are blocked one after another (at random, "
        "busiest first, narrowest first): the area under each curve, 1 if flow "
        "never drops, and the share blocked when flow halves"
    ),
    "current_flow": (
        "How inlet-to-outlet flow divides between parallel routes, as current "
        "does in a resistor network: each vessel's share and the equivalent "
        "resistance; in full mode, the Kirchhoff index"
    ),
    "perfusion_territories": (
        "Which inlet supplies and which outlet drains each part of the "
        "network: the number and size of the territories, and the watershed "
        "vessels on their borders"
    ),
    "capillary_bed_joins": (
        "Where the capillary beds of different arteriolar feeds first meet: "
        "the median capillary order and distance at which beds join "
        "(Kaplan-Meier), and how many never join"
    ),
    "capillary_path_orders": (
        "How many capillary orders lie between arterioles and venules: for "
        "each capillary draining into a venule, the fewest capillaries back to "
        "an arteriole and that route's length"
    ),
    "transit_time": (
        "How long blood takes to cross the network: vessel volume over flow, "
        "summed along inlet-to-outlet routes, with its spread and the capillary "
        "transit-time heterogeneity (CTH). Assumes plug flow"
    ),
    "flux_heterogeneity": (
        "How unevenly the network is perfused: the coefficient of variation of "
        "flow and velocity across vessels carrying flow, and the share of the "
        "network carrying flow"
    ),
    "loop_hierarchy": (
        "The scale of the network's redundancy: the shortest loop through each "
        "vessel as a distribution of loop lengths, the vessels on no loop, and "
        "loops per mm^3"
    ),
    "strahler": (
        "Strahler orders of the arterial and venous trees (tips are order 1; "
        "two vessels of one order meet to make the next), with Horton's ratios "
        "between orders. Capillaries are unordered"
    ),
    "algebraic_connectivity": (
        "How easily the network splits in two: the Fiedler value, near 0 when "
        "a few vessels hold two halves together, and the two halves and the "
        "vessels between them"
    ),
    "cyclomatic_number": (
        "The number of independent loops (anastomoses): vessels minus nodes "
        "plus separate pieces. 0 is a pure tree with no alternative routes"
    ),
    "degree_assortativity": (
        "Whether nodes where many vessels meet link to similar nodes (positive, "
        "up to 1) or to nodes with few (negative, down to -1). Vascular trees "
        "usually read negative"
    ),
    "rich_club": (
        "Whether the best-connected nodes link to each other: of the possible "
        "links among nodes with more than the threshold number of vessels, the "
        "share that exist. Near 1, a backbone"
    ),
    "k_core": (
        "How layered the network is: a node's core number is the largest k "
        "for which it sits in a part where every node has at least k vessels. "
        "A tree's is 1"
    ),
    "flow_hierarchy": (
        "The share of vessels whose solved flow does not run round a loop. "
        "Steady flow cannot circulate, so below 1 points to a modelling "
        "problem such as conflicting boundary conditions"
    ),
    "path_efficiency": (
        "The inverse of the mean shortest path length between pairs of nodes, "
        "with that mean length; higher means nodes are closer. Sampled pairs in "
        "fast mode, all in full"
    ),
    "community": (
        "Groups of nodes more densely connected to each other than to the rest "
        "of the network: their number and sizes in fast mode, every group's "
        "nodes in full"
    ),
    "betweenness": (
        "How often each node lies on the shortest paths between other nodes, "
        "counting vessels, from 0 to 1: the mean, highest and top nodes in "
        "fast mode, every node in full"
    ),
    "haemodynamics": (
        "The network's resistance to flow: between two nodes (the first inlet "
        "and outlet, or those chosen in Post processing), and for the whole "
        "network, pressure drop over total inflow"
    ),
    "branch_orders": (
        "The vessels of each branch order: how many, their mean length, "
        "diameter and its spread, tortuosity, the angle they leave their "
        "parent at, and the pressure drop across them"
    ),
    "weighted_betweenness_communities": (
        "Betweenness and communities with vessels weighted by resistance "
        "(easiest paths for flow), length (shortest) and 1/|flow| (busiest): "
        "the most central nodes and the communities' number and sizes"
    ),
    "zero_resistance_bridges": (
        "The joins drawn inside thick vessels to link the vessels opening "
        "into them to their centreline: how many and how long. They carry no "
        "resistance"
    ),
    "tissue_volume": (
        "The tissue's volume and surface area, measured from the raw image, "
        "which vessel density divides by, with the share of the image it fills "
        "and the vessels inside it"
    ),
    # synthetic.compare.COMPARISON_SECTION: a synthetic network's run only.
    "synthetic_comparison": (
        "How close the synthetic network came to its template run, "
        "characteristic by characteristic: both values, their ratio, and "
        "OUTSIDE where it misses the allowed band"
    ),
}
SAVE_CONFIG_TOOLTIP = (
    "Save the current settings to a YAML config file; relative paths stay "
    "relative when saved beside the file they came from"
)
RUN_CHECKS_TOOLTIP = (
    "Check the current settings for problems, such as missing files, "
    "without starting a run"
)
RUN_PIPELINE_TOOLTIP = (
    "Run the full pipeline from the first stage with the current settings"
)
CLEAR_LAYERS_TOOLTIP = (
    "Remove HaemoLynx's layers from the viewer, stop any run in progress and "
    "forget this session's saved stages, deleting their checkpoint files "
    "after asking"
)
SHOW_RESULTS_TOOLTIP = (
    "After each stage finishes, add its graph or mask layers to the viewer"
)
SHOW_STEPS_TOOLTIP = (
    "While the graph is built, also show the network after each build step "
    "as its own layers"
)
HELP_TOOLTIP = (
    "Open the help menu on the tab that is currently open"
)
#: The box under the buttons where the panel says what it last did.
REPORT_TOOLTIP = (
    "The panel's latest message: a run's progress or why it stopped, what the "
    "checks found, or what a button did; the log window has everything"
)


def stage_tab_tooltip(stage) -> str:
    """A stage tab's hover text: the summary its page opens with, without the
    full stop, so a tab says what it is for before it is opened."""
    return stage.summary.rstrip(".")


#: The log window's text.
LOG_VIEW_TOOLTIP = (
    "What the run says as it goes, newest at the bottom: Log level sets how "
    "much is shown, and Follow keeps the newest line in view"
)
#: The About widget's text.
ABOUT_TOOLTIP = (
    "What HaemoLynx is, this build's version and number of settings, where "
    "the colour controls are and how a config file repeats a run"
)
OPEN_LAYER_TOOLTIP = (
    "The image layers open in the viewer: choosing one, or dropping an image "
    "into the viewer, makes it the run's input"
)
USE_LAYER_TOOLTIP = (
    "Make the image layer chosen above the run's input: its own file, or a "
    "saved copy of its data if it has none"
)
OPTIMISE_SETTINGS_TOOLTIP = (
    "Search for good Skeletonise and Graph settings on the segmented image, "
    "save them as a config beside it, and show the changes to Apply or Discard"
)
#: Both settings optimisers while the run draws a synthetic network, greyed:
#: there is no image for their trials to read. Also what each says if pressed.
OPTIMISE_SYNTHETIC_TOOLTIP = (
    "Unavailable for a synthetic network, which has no image to optimise on. "
    "Set \"Input\" on the Input tab to Segmented image to use it"
)
CHECK_SEGMENTED_IMAGE_TOOLTIP = (
    "Score the segmented image out of 10 on fragmentation, connectivity, "
    "vessels leaving the image, surface noise and resolution, with the "
    "breakdown in the log"
)
#: The Batch run button and its window.
BATCH_RUN_TOOLTIPS = {
    "button": (
        "Run every segmented image in a folder with these settings, each "
        "matched to its raw image and masks by file name and saved in its "
        "own folder"
    ),
    #: The button, greyed, under a synthetic network and while a run goes.
    "button_synthetic": (
        "Unavailable for a synthetic network: a batch runs segmented images. "
        "Set \"Input\" on the Input tab to Segmented image to use it"
    ),
    "button_running": (
        "A run is going: Batch run opens once it ends, or press Cancel to stop it"
    ),
    "segmented": (
        "The folder of segmented images: each image file in it is one run. Not read "
        "while ilastik segments the raw images"
    ),
    "raw": (
        "The folder of raw images; left blank, they are looked for in the segmented "
        "folder. When ilastik segments, each raw image here is one run"
    ),
    "pericyte": (
        "The folder of pericyte masks, each matched to its image by name. Setting it "
        "turns on placing pericytes from a mask for the pericyte perturbations"
    ),
    "cell": (
        "The folder of cell masks, each matched to its image by name. Setting it "
        "turns on the 3D cell distances"
    ),
    "vessel_3d": (
        "The folder of vessel masks for the 3D cell distances, matched by name; left "
        "blank, each image's own segmentation. Setting it turns on the 3D cell "
        "distances"
    ),
    "tissue": (
        "The folder of tissue images, matched by name; left blank, each image's raw "
        "image. Setting it turns on the tissue volume"
    ),
    "capillary_block": (
        "The folder of capillary block masks, matched by name. Setting it turns on "
        "perturbations, adding a capillary block from a mask if there is none"
    ),
    "large": (
        "The folder of large vessel masks, matched to images by name; left blank, "
        "they are looked for in the segmented folder"
    ),
    "small": (
        "The folder of small vessel masks, matched to images by name; left blank, "
        "they are looked for in the segmented folder"
    ),
    "output": "Each image's outputs and .haemorun go in a folder of its own in here",
    "browse": "Choose the folder",
    "images_tab": (
        "The folders, conditions and matched images, with how each image's run "
        "went"
    ),
    "comparison_tab": (
        "The two conditions compared statistic by statistic, as written to the "
        "output folder's comparison CSV"
    ),
    "skip_done": (
        "Leave out images whose folder already has a .haemorun, so a batch "
        "stopped part way picks up where it ended"
    ),
    "table": (
        "Each segmented image, its matched files and its result. Untick to leave "
        "one out, pick a file where several tie, double-click to open a finished "
        "run, right-click for more"
    ),
    "match": (
        "Read the folders again and match every image; this happens by itself when "
        "a folder is chosen"
    ),
    "last_batch": (
        "Fill this window from the batch saved in the output folder (folders, "
        "conditions, comparison, sections) and load the settings it ran with into "
        "the panel"
    ),
    "settings_line": "How many of the panel's settings, which every image runs with, differ from their defaults",
    "show_settings": "List the settings every image runs with that differ from their defaults",
    "peak_ram": (
        "The image's peak memory use during its run. It may read a little low, "
        "as some stages' memory is sampled rather than measured exactly"
    ),
    "tick_all": "Tick every image",
    "tick_none": "Untick every image",
    "tick_failed": "Tick only the images that failed, in this batch or the folder's last one",
    "open_run": "Open the selected image's .haemorun in the panel (leaves the batch run)",
    "open_folder": "Open the selected image's output folder",
    "show_error": "Show why the selected image failed, with its traceback when this batch ran it",
    "run_again": "Run the selected image again now, whether or not it was done before",
    "details": "What Check, Show error and Show said",
    "comparison_table": (
        "The two conditions tested statistic by statistic, most significant first: "
        "each condition's N and median, the difference, P and the "
        "Benjamini-Hochberg q"
    ),
    "refresh_comparison": "Read the comparison CSV in the output folder again",
    "open_comparison": "Open the comparison CSV in the system's program for it",
    "open_batch_folder": "Open the batch's output folder",
    "stop": (
        "Stop after the image now running finishes and is saved; later images are "
        "not run. Cancel stops the current image itself"
    ),
    "run": "Use this batch and run it now, keeping this window open to follow it",
    "check": "Run every ticked image's pre-run checks and show what they say",
    "condition_a": (
        "Text in the file names of one condition's images (any case; several, "
        "comma-separated, for either). Each condition, and all images, is "
        "summarised after the batch"
    ),
    "condition_b": (
        "The other condition's text: with both set, every statistic is compared "
        "between the two (Mann-Whitney U, Benjamini-Hochberg q-values)"
    ),
    "design": (
        "Independent images: each image is one sample (Mann-Whitney U, Welch t). "
        "Paired by subject: each image is paired with the other condition's most "
        "similarly named image (Wilcoxon, paired t)"
    ),
    "subject_pattern": (
        "Paired only: a regular expression whose first group (or whole match) is "
        "the subject ID. Blank uses the part of the file name before the "
        "condition text"
    ),
    "aggregate": (
        "Paired only: a subject with several images in one condition is left out, "
        "or its images are combined by their median or mean"
    ),
    "exclude_sections": (
        "Statistics sections to leave out of the batch statistics and comparison, "
        "comma-separated (e.g. Dead Ends), so the q-values cover only the "
        "statistics left"
    ),
    "compare": (
        "Write the batch statistics for the ticked images already run, using the "
        "current conditions, without running anything"
    ),
    "use": "Close this window and let Run pipeline run the batch later",
    "leave": "Go back to running the one image the Input tab names",
}
REVERT_STAGE_TOOLTIP = (
    "Rerun the pipeline from this stage, using the previous stage's saved "
    "results and clearing this and later stages' layers"
)

#: View dock — display-only; never written into pipeline settings.
Z_DEPTH_TOOLTIP = (
    "Show only this range of z in every layer; display only, the pipeline "
    "still uses the whole image"
)
#: The Z window's two ends, one slider each.
Z_DEPTH_TOOLTIPS = {
    "min": "The lowest z shown, in microns: every displayed layer is clipped below it",
    "max": "The highest z shown, in microns: every displayed layer is clipped above it",
}
#: The arrow-size row napari's controls show for a flow-direction layer.
ARROW_LENGTH_TOOLTIP = (
    "How long the flow-direction arrows are drawn; display only, the flows "
    "are unchanged"
)
VESSEL_DRAW_TOOLTIP = (
    "Tubes: 3D tubes visible from every angle. Lines: faster flat ribbons that "
    "can vanish edge-on. The joins inside thick vessels are dashed in both"
)
LAYER_SET_TOOLTIP = (
    "Which network the vessel, node and flow-direction layers show: the "
    "baseline or one perturbation. Switching keeps the same layers on, for a "
    "like-with-like comparison"
)
COLOUR_BY_TOOLTIP = (
    "What the vessels are coloured by, as tubes or lines, for the network "
    "chosen under Showing; the same as the vessel layer's own Colour by"
)
#: What HaemoLynx adds to a vessel or node layer's own controls on the left:
#: Colour by and Colour map (its colour range takes COLOUR_RANGE_TOOLTIPS).
LAYER_COLOUR_BY_TOOLTIP = (
    "Which of this layer's values its colours show; the colour map and "
    "colour range below follow the choice"
)
LAYER_COLOUR_MAP_TOOLTIP = (
    "The colour map the chosen value is drawn in; hidden while the colours "
    "are categories or directions, which bring their own"
)
#: The vessels layer's "branch tooltip metrics" checkboxes, one per metric
#: (``gui.branch_hover.BRANCH_HOVER_METRICS``).
BRANCH_HOVER_TOOLTIPS: dict[str, str] = {
    "flow_solution": (
        "Show on hover whether the flow was solved in the vessel, and if not "
        "why: behind a dead end, behind a closed end, or no inlet and outlet"
    ),
    "flow": "Show on hover the size of the vessel's flow, in m^3/s",
    "order": "Show on hover the vessel's branch order",
    "size_class": (
        "Show on hover the vessel's size class and the widths it holds, for a "
        "network classed by size"
    ),
    "diameter": "Show on hover the vessel's diameter, in microns",
    "diameter_source": "Show on hover which method gave the vessel its diameter",
    "fwhm_status": "Show on hover what FWHM made of the vessel: measured, or why not",
    "resistance": "Show on hover the vessel's resistance to flow, in Pa.s/m^3",
    "tortuosity": (
        "Show on hover the vessel's tortuosity: its centreline length over the "
        "straight line between its ends"
    ),
    "length": "Show on hover the vessel's centreline length, in microns",
    "drawn_support": (
        "Show on hover how much of a vessel drawn by hand lies in the "
        "segmentation, and the most background it crosses in one run"
    ),
}
#: The view panel's Colour range row, one entry per control.
COLOUR_RANGE_TOOLTIPS = {
    "range": (
        "The range of values the vessel colours span for the network chosen "
        "under Showing; each network keeps its own"
    ),
    "low": "The value drawn at the bottom of the colour map; type a number to change it",
    "high": "The value drawn at the top of the colour map; type a number to change it",
    "fit_all": "Span the smallest and largest value of the vessels on screen",
    "fit_trim": "Ignore the extreme 1% at each end, so the bulk of the vessels spreads",
    "canvas": (
        "Show the colour scale on the canvas for the network chosen under Showing"
    ),
}
SCALE_BAR_TOOLTIP = (
    "Show a scale bar in the bottom-right of the canvas, in microns when the "
    "voxel size is known"
)
#: The view panel's "Show" checkboxes, by the role each one toggles
#: (``gui.layer_sets.VIEW_TOGGLES``). Greyed out while the run has no such
#: layer yet.
VIEW_TOGGLE_TOOLTIPS: dict[str, str] = {
    "flow direction": (
        "Show the flow-direction arrows of the network chosen under Showing, "
        "one arrow per vessel pointing the way its flow runs"
    ),
    "nodes": (
        "Show the nodes of the network chosen under Showing: every junction "
        "and dead end, coloured by how many vessels meet there"
    ),
    "terminal nodes": (
        "Show the terminal nodes the Boundaries stage assigned: the inlets, "
        "outlets and arteriole/venule boundary nodes, coloured by role"
    ),
    "skeleton": (
        "Show the skeleton the graph was built from, over the segmented image"
    ),
    "segmented image": (
        "Show the segmented image the skeleton was made from"
    ),
}
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
    "Step through the sweep perturbation chosen under Showing, recolouring "
    "the vessels with each grid point's flows"
)
#: One slider per axis of a sweep's grid, in the Sweep box.
SWEEP_SLIDER_TOOLTIP = (
    "Step along this axis of the sweep's grid: the vessels are recoloured "
    "with the flows solved at that point, whose value is shown below"
)
TUBE_QUALITY_TOOLTIP = (
    "How round the vessel tubes are drawn: further right gives more sides and "
    "smoother tubes, but is slower on a large network"
)
TUBE_DIAMETER_TOOLTIP = (
    "How wide the tubes are drawn, for display only. Per vessel: each at its "
    "own diameter. Uniform: all at the width beside it, 4 µm to start, so "
    "narrow vessels stay visible"
)
TUBE_UNIFORM_WIDTH_TOOLTIP = (
    "The width every tube is drawn at while \"Tube diameter\" is Uniform, "
    "applied on Enter or each arrow step. Display only: the network keeps its "
    "diameters"
)
SNAPSHOT_TOOLTIP = (
    "Save a TIFF image of the current view into the output folder; it does "
    "not affect the run"
)
#: The view window's Rotation movie box.
ROTATION_TOOLTIPS = {
    "tilt": (
        "How far the movie looks down on the stack as it turns, in degrees: 0 "
        "is side on, 90 straight down from the top, negative from below"
    ),
    "speed": (
        "How fast the stack turns in the movie, in degrees a second: 30 takes "
        "12 s for a full turn"
    ),
    "rotate": (
        "Preview the movie: turn the view about z at this tilt and speed, "
        "writing nothing. Press again to stop; the view turns back, keeping any "
        "zoom or pan"
    ),
    "record": (
        "Turn the view one full circle and save it as a movie (MP4, or GIF if "
        "MP4 support is not installed) in the output folder. Press again to "
        "stop without saving"
    ),
}
SAVE_RUN_TOOLTIP = (
    "Save this run to a .haemorun file (settings, network, saved stages and "
    "layers) to open again later"
)
LOAD_RUN_TOOLTIP = (
    "Open a saved .haemorun run, replacing the current HaemoLynx layers and "
    "state, so the viewer looks as it did when that run finished"
)
#: "7. Post processing" tab.
POST_PROCESSING_TOOLTIPS: dict[str, str] = {
    "scan": (
        "Bring the last run's network into this tab for editing and make the "
        "list chosen under Review"
    ),
    "review_choice": (
        "Which list Scan network makes: 4+ vessel junctions, short loops, a "
        "check on the network (geometry, segmentation, flow, inlets and "
        "outlets, diameters), or None to edit by clicking alone"
    ),
    "region": (
        "List only within this many um of the middle of the view; 0 lists the "
        "whole network. Faster on a large stack"
    ),
    "review_items": (
        "What the chosen check found, ranked. Click one, or its ring in the "
        "viewer, to zoom to it and list its vessels below; kept items are not "
        "listed again"
    ),
    "dead_end_show": (
        "List every kind of dead end the scan found, or only one kind; "
        "pieces with no junction are vessels on their own, both ends dead "
        "ends"
    ),
    "remove_pieces": (
        "Remove every piece with no junction in the list, except those holding "
        "a boundary node or with a vessel drawn onto them since the scan. Each "
        "removal is recorded in the review CSV"
    ),
    "review_table": (
        "The chosen item's vessels; click or shift/ctrl-click rows to select "
        "vessels, drawn yellow in the viewer"
    ),
    "review_delete_selected": (
        "Remove the vessels selected in the table. A junction left with two "
        "vessels merges them; inlets, outlets and boundary nodes are never "
        "removed"
    ),
    "review_delete_item": (
        "Remove every vessel of the chosen item -- a dead end back to its "
        "junction, or a whole unsolved piece; inlets, outlets and boundary "
        "nodes are never removed"
    ),
    "review_merge": (
        "Remove the short vessel between the two junctions and merge them into "
        "one node, so the vessels of both meet there"
    ),
    "review_trace_on": (
        "Start Add vessel at the chosen dead end's tip: click along the "
        "vessel it should go on to, and on a node or vessel to finish it"
    ),
    "review_join_ahead": (
        "Join the chosen dead end to the vessel straight ahead (or a thick "
        "vessel's centreline), tracing a new vessel from its tip through the "
        "image Add vessel uses"
    ),
    "review_mark_blind": (
        "Mark the chosen dead end as a real blind end, such as a sprout. It "
        "stays, and later scans and the statistics treat it as blind"
    ),
    "review_mark_open": (
        "Mark the chosen dead end as a vessel leaving the image. Unless \"Open "
        "end model\" is closed, the next solve gives it a boundary condition"
    ),
    "review_make_inlet": (
        "Make the chosen terminal node an inlet (no longer an outlet). "
        "Regenerate graph solves the network with it"
    ),
    "review_make_outlet": (
        "Make the chosen terminal node an outlet (no longer an inlet). "
        "Regenerate graph solves the network with it"
    ),
    "review_remove_terminal": (
        "Take the chosen node off the inlets or outlets; a vessel leaving the "
        "image there becomes an open end. The last inlet or outlet cannot be "
        "removed"
    ),
    "review_resistance_start": (
        "Measure the two-point resistance from the chosen node (the first "
        "inlet by default); choosing the current end swaps the two. Regenerate "
        "graph solves it"
    ),
    "review_resistance_end": (
        "Measure the two-point resistance to the chosen node (the first outlet "
        "by default); choosing the current start swaps the two. Regenerate "
        "graph solves it"
    ),
    "review_join": (
        "Join the chosen pair of dead ends with a vessel traced between "
        "them through the image Add vessel uses, as wide as the vessels "
        "at its ends"
    ),
    "review_draw_along": (
        "Add a vessel along the chosen piece of skeleton, joining the vessels "
        "it meets at each end. For a piece meeting the network once, trace it "
        "with Add vessel"
    ),
    "review_set_diameter": (
        "Give the vessels selected in the table, or the chosen item's one "
        "vessel, the diameter beside this button; Regenerate graph keeps it "
        "and re-solves"
    ),
    "review_diameter": "The diameter Set diameter gives, in microns",
    "review_keep": (
        "Keep the chosen item as it is and move on to the next. Every decision "
        "is saved with the run and in a review CSV beside its outputs"
    ),
    "junctions": (
        "Junctions where four or more vessels meet; click one to zoom to it "
        "and list its vessels below, or click its ring in the viewer"
    ),
    "table": (
        "The vessels meeting at the chosen junction; click or shift/ctrl-click "
        "rows to select vessels, drawn thick and yellow in the viewer"
    ),
    "delete": (
        "Remove the selected vessels. A junction left with two vessels merges "
        "them; inlets, outlets and boundary nodes are never removed"
    ),
    "split": (
        "Split the chosen junction into bifurcations: the two vessels leaving "
        "in the most similar directions move to a new node, joined back by a "
        "connector vessel"
    ),
    "connector": (
        "Length of the connector vessel a split inserts between the two new "
        "bifurcations, in microns"
    ),
    "leave": "Keep the chosen junction as it is and move on to the next one",
    "loops": (
        "Short loops not yet reviewed, shortest first; click one to zoom to "
        "it, in its own plane in a 3D view, and list its sides below"
    ),
    "loop_table": (
        "The chosen loop's sides: the runs of vessels between where other "
        "vessels or a boundary node meet it. Click a row to select that side"
    ),
    "loop_delete": (
        "Remove the side selected in the table, opening the loop. A junction "
        "left with two vessels merges them; inlets, outlets and boundary nodes "
        "are never removed"
    ),
    "loop_keep": "Keep the chosen loop as it is, take it off the list and move on to the next one",
    "raw_panel": (
        "Draw the chosen loop or item in its own plane and the plane across "
        "it, the segmentation beside the raw image. The first time is slower "
        "while the raw image is read"
    ),
    "raw_picture": (
        "A slab 3 um either side of the plane: segmentation voxels anywhere in "
        "it, or the raw image's mean through it. Chosen item cyan, other "
        "centrelines grey, orange bar 5 um"
    ),
    "click_delete": (
        "Then click vessels in the viewer to remove them, one per click; "
        "inlets, outlets and boundary nodes are never cut off. Press it again "
        "to stop"
    ),
    "add": (
        "Draw a new vessel: click a node or vessel to start, click along its "
        "course, and click another node or vessel to finish. Each leg is traced "
        "through the chosen image. Press again to cancel"
    ),
    "trace_source": (
        "What Add vessel traces each leg through: the segmented mask, or the "
        "raw data (the raw image on the Input tab)"
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
        "Remove every vessel no inlet-to-outlet path runs along: pieces cut off "
        "from an inlet or outlet, and dead-end branches, trees or loops that "
        "reach no outlet"
    ),
    "connectivity_choice": (
        "Which vessels the connectivity CSV lists: every vessel in the viewer, "
        "or only those on some inlet-to-outlet path (no dead ends or pieces cut "
        "off)"
    ),
    "connectivity_map": (
        "Draw the last exported connectivity CSV (or one you pick) as a 2D map "
        "in the browser: inlets left, outlets right, each vessel hoverable for "
        "its details"
    ),
    "export_connectivity": (
        "Save a CSV of how the network is connected: one row per vessel, "
        "oriented away from the inlets, with its branch order, length, diameter "
        "and upstream and downstream vessels"
    ),
    "log": (
        "Every change made in this tab, oldest first, with branchIDs as they "
        "were at the time; also written to the napari log"
    ),
    "regenerate_graph": (
        "Update the edited vessels to match the rest (branch orders, lengths, "
        "diameters, thick-vessel joins) and rerun Haemodynamics; after a "
        "finished run, Perturbations and Export rerun too. A paused run stays "
        "paused"
    ),
    "continue": (
        "Resume the run paused here: update any edits, rerun Haemodynamics, "
        "then Perturbations and Export. A run pauses here when \"Mid-run "
        "postprocessing\" is on"
    ),
}

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
        "last ended, to catch settings that only help together. Stops early if "
        "a pass changes nothing"
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
#: Optimise settings' group checkboxes, shown under "Choose optimisation
#: types": what each group tries. Keyed by ``optimisation.GROUP_NAMES``.
OPTIMISE_GROUP_TOOLTIPS: dict[str, str] = {
    "segmentation_cleanup": (
        "Try the mask clean-up steps before skeletonising (filling holes, "
        "removing whiskers and small pieces, closing gaps, smoothing) and their "
        "sizes, scored as Check segmented image scores"
    ),
    "thick_vessel_gating": (
        "Try the alternate skeletonisation of thick vessels on and off, the "
        "radius above which a vessel counts as thick, and how thin vessels "
        "join its centreline"
    ),
    "min_branch_length": (
        "Try the shortest piece of skeleton kept when it is cleaned (\"Skeleton "
        "min branch length\")"
    ),
    "bundle_refinement": (
        "Try the settings that find dense tangles of skeleton and collapse each "
        "into a hub: scan window, density, links kept per hub and spacing "
        "between hubs"
    ),
    "closing_radius": (
        "Try the radius of the closing that fills small gaps in the mask "
        "before it is skeletonised (\"Skeleton closing radius\")"
    ),
    "gap_bridging": (
        "Try how large a gap in the skeleton is bridged (\"Skeleton bridge gap "
        "size\") and how far a broken branch end reaches for another piece "
        "(\"Skeleton max bridge distance\")"
    ),
    "connectivity_and_component_filter": (
        "Try the voxel connectivity skeleton pieces are counted by, and the "
        "smallest piece kept as a percentage of the skeleton (\"Skeleton min "
        "component percent\")"
    ),
    "reconnect_thresholds": (
        "Try how far apart pieces of the graph may be and still be joined "
        "(\"Graph reconnect threshold\" and \"Final orphan reconnect threshold\")"
    ),
    "cluster_collapse_distance": (
        "Try how nodes lying close together are merged into one: \"Cluster "
        "collapse distance\", the method, and the method's own setting"
    ),
    "min_stub_length": (
        "Try the shortest dead-end stub the graph keeps: \"Min stub length\", "
        "or \"Min stub length radius multiple\" when stubs are judged by their "
        "vessel's radius"
    ),
    "centreline_smoothing": (
        "Try smoothing the vessel centrelines on and off, the smoothing "
        "method, how many passes it makes and how far a smoothed centreline "
        "may move from the skeleton"
    ),
}
#: Optimise FWHM settings' group checkboxes. Keyed by
#: ``optimisation.FWHM_GROUP_NAMES``.
FWHM_OPTIMISE_GROUP_TOOLTIPS: dict[str, str] = {
    "measurement_model": (
        "Try how each profile is sampled and fitted (slice plane or 3D, blurred "
        "lumen or Gaussian, averaging, spacing) and how many a vessel needs, "
        "scored on added vessels of known width"
    ),
    "exclusion_zones": (
        "Try how far from a branch end, and from a junction, profiles are "
        "left unsampled"
    ),
    "extent_and_widening": (
        "Try how far each profile reaches either side of the centreline, how "
        "much wider than the fitted width it must span, and how often it may "
        "widen"
    ),
    "central_lobe_clipping": (
        "Try clipping each profile to its central lobe, so a neighbouring "
        "vessel does not widen the fit, and the drop and re-rise that mark "
        "the lobe's edges"
    ),
    "same_edge_geometry": (
        "Try keeping each profile's samples to a local stretch of its own "
        "vessel, and shortening a profile where another part of the same "
        "vessel comes close"
    ),
    "baseline_estimation": (
        "Try how each profile's background level is estimated, from its wings "
        "or a percentile, and whether the fit's baseline is held near that "
        "estimate"
    ),
    "diameter_guess": (
        "Try where each vessel's fit starts: one diameter guess for every "
        "vessel, or each vessel's own, such as its width in the mask"
    ),
    "rejection_gates": (
        "Try the gates that reject poor profiles (fitted centre off the "
        "centreline, low R^2, flat top) and the threshold of each"
    ),
}

#: "5. Diameters" — "Optimise endothelial settings", shown in place of
#: Optimise FWHM settings when the raw image stains the vessel wall, and its
#: options. Keyed by control; a group's checkbox by its name in
#: ``optimisation.endothelial_search.ENDOTHELIAL_GROUP_NAMES``.
ENDOTHELIAL_OPTIMISE_TOOLTIPS = {
    "button": (
        "Find the endothelial ring fit's best settings (blur, wall, ring "
        "strength) by testing them on a sample of vessels, vessels of known "
        "lumen and decoys, and propose them"
    ),
    "options": "Show or hide the endothelial optimiser's own options",
    "choose_groups": (
        "Restrict Optimise endothelial settings to only the ticked group(s) "
        "below, instead of every ring-fit setting"
    ),
    "blur_and_wall": (
        "Where the image cannot estimate the blur or wall, or you fixed them, "
        "try their estimate and the ends of its 95% interval; otherwise they "
        "are estimated from the image"
    ),
    "ring_gate": (
        "Try half to twice the weakest ring a reading is kept for "
        "(\"Endothelial min ring contrast\"), and the ring fit's own default"
    ),
    "sample": (
        "How many of the network's vessels each trial reads; Auto times one "
        "reading and picks a count to keep the trials near ten minutes, plus "
        "calibration"
    ),
}
