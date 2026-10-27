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
        "towards you in 3D however the view is turned"
    ),
    "forward": (
        "Move the box forward by the move step: on through the slices in 2D, "
        "away from you in 3D however the view is turned"
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
PRESET_TOOLTIP = (
    "Load a standard set of settings to start from: a preset sets the rows "
    "it names and leaves the rest, so a config loaded afterwards or a row "
    "changed by hand replaces what it set; None loads nothing"
)
EDIT_GRAPH_TOOLTIP = (
    "Open the graph editor: add a branch by clicking along the segmented "
    "image, or delete a vessel by clicking it, then Regenerate to catch "
    "diameters, haemodynamics, perturbations, measurements and export up "
    "to the edit"
)
REOPEN_VIEW_TOOLTIP = (
    "Reopen the left-hand panel: whichever of its HaemoLynx view (Z-depth, "
    "vessel draw mode and colouring, scale bar and snapshot), Statistics, "
    "Layer list and Layer controls tabs have been closed, with the HaemoLynx "
    "view in front"
)
#: A Statistics tab button with no entry in STATISTICS_SECTION_TOOLTIPS.
STATISTICS_SECTION_TOOLTIP = (
    "Show or hide what this measure found in the last run's statistics; a "
    "rerun of the statistics replaces it, keeping open what was open"
)
#: The Statistics tab's buttons, one per section of the report
#: (``gui.statistics_view``): what the statistic behind each one measures
#: and how to read it. Keyed by section: the statistic measures
#: (``statistics.STATISTIC_MEASURES``), then ``statistics_view.OTHER_SECTIONS``.
STATISTICS_SECTION_TOOLTIPS: dict[str, str] = {
    "basic": (
        "The size of the network: how many nodes (junctions and ends) and "
        "vessels it has, their total and average length in microns, and the "
        "average number of vessels meeting at a node"
    ),
    "tortuosity": (
        "How winding the vessels are. The tortuosity index is a vessel's "
        "centreline length over the straight line between its ends (1 is "
        "straight); curvature is the share of its length beyond that straight "
        "line. Both are averaged over every vessel"
    ),
    "branching": (
        "How many branch points the network has: nodes where three or more "
        "vessels meet"
    ),
    "tree_asymmetry": (
        "How unevenly the network divides: at each branch point, the "
        "difference in size between its largest and smallest sub-tree, summed "
        "and divided by the number of nodes. 0 is a perfectly balanced tree; "
        "a network with loops is read along its shortest spanning tree"
    ),
    "fractal_dimension": (
        "How completely the vessels fill space, by box counting: about 1 for "
        "a single line, rising towards 3 for vessels filling the volume. "
        "Measured on the node positions and on the centrelines; R^2 says how "
        "well the box counts fit a straight line"
    ),
    "vessel_density": (
        "Vessel length per unit volume (microns of vessel per cubic micron): "
        "divided by the tissue measured from the raw image when the tissue "
        "volume was measured, else by the box the network spans, and also by "
        "the whole image; with measured tissue, also the share of it that is "
        "vessel"
    ),
    "vessel_classes": (
        "The same measures for each class of vessel, side by side: large and "
        "small arterioles, capillaries, small and large venules -- or, with "
        "global vessel masks, large vessels, small vessels and capillaries. "
        "For each: how many vessels, their length and share of the network's, "
        "length, volume and surface per unit volume, diameter and tortuosity, "
        "and once the flow is solved, the flow, velocity, wall shear stress "
        "and pressure in its vessels carrying flow"
    ),
    "murray_law": (
        "Whether diameters at junctions follow Murray's law, parent^3 = the "
        "sum of daughters^3. The Murray ratio is the daughters' sum over the "
        "parent's (1 is a perfect fit); the fitted exponent is the power that "
        "would make each junction fit exactly, 3 being Murray's"
    ),
    "daughter_angles": (
        "The angle between sibling daughter vessels leaving a junction, read "
        "along their centrelines and averaged over every junction, in degrees"
    ),
    "intercapillary_distance": (
        "How far apart neighbouring vessels lie: for each vessel, the median "
        "distance along it to the nearest vessel not joined to it, then the "
        "mean and median over vessels, in microns. How far oxygen must "
        "diffuse into the tissue between them. With a solved flow, the same "
        "over the vessels carrying flow alone, and the ratio of the two "
        "medians: how the spacing opens up where vessels carry no flow"
    ),
    "junction_shape": (
        "The angles between the three vessels of each junction, read along "
        "their centrelines: the smallest and largest, the share of junctions "
        "with two vessels within 45 degrees of each other, and the three "
        "angles summed (360 when the junction lies in one plane)"
    ),
    "tissue_coverage": (
        "How far the tissue lies from a vessel, read at points through the "
        "image box: the share within 20 microns of a vessel, the share beyond "
        "50 microns (the holes), and the median and 90th percentile distance"
    ),
    "vessel_columns": (
        "Vessels running along the stack axis (within 30 degrees of z): their "
        "share of the network's length, how many per square millimetre of the "
        "image plane, and how evenly they are spread across it (the "
        "Clark-Evans ratio: 1 at random, above 1 even, below 1 bunched)"
    ),
    "dead_ends": (
        "Vessels that end without joining another, by kind: leaving the "
        "image, at a thick vessel's wall but not joined to it, stopped short of "
        "the segmented vessel, broken by the segmentation, artefacts, blind "
        "(marked by hand), unknown, or ends of "
        "pieces with no junction. Also the interior ones per mm of vessel, "
        "their length, and how much of the network the image faces cut"
    ),
    "network_robustness": (
        "How much of the network hangs on single connections: bridge "
        "vessels, whose loss alone splits the network, and articulation "
        "points, junctions whose loss splits it. With inlets and outlets, the "
        "critical ones are those whose loss cuts an outlet off from an inlet"
    ),
    "bottlenecks": (
        "Where inlet-to-outlet flow is most constrained: the minimum cut (the "
        "fewest vessels, or least conductance, whose blocking separates every "
        "inlet from every outlet) and each vessel's share of the shortest "
        "inlet-to-outlet routes; a share of 1 means every route runs through it"
    ),
    "shunts": (
        "Vessels on inlet-to-outlet routes much shorter than the network's "
        "median route, which carry blood past most of the bed: how many, on "
        "how many pathways, and the share of the outflow they carry"
    ),
    "occlusion_impact": (
        "What blocking one vessel does, vessel by vessel: the fraction of "
        "inlet-to-outlet flow lost, and the length of other vessels left "
        "below the hypoperfusion threshold of their flow, summarised for the "
        "network, by vessel type and by branch order. Conductances stay fixed "
        "after the block"
    ),
    "occlusion_curves": (
        "How flow falls as vessels are blocked one after another, in three "
        "orders: at random, busiest first and narrowest first. Each curve's "
        "robustness index is the area under its remaining-flow curve (1 if "
        "flow never drops), with the fraction blocked when flow halves"
    ),
    "current_flow": (
        "How the inlet-to-outlet flow divides between parallel routes, as "
        "current does in a resistor network: each vessel's share of it, the "
        "equivalent inlet-to-outlet resistance, and in full mode on small "
        "networks the Kirchhoff index, the summed resistance between every "
        "pair of nodes"
    ),
    "perfusion_territories": (
        "Which inlet supplies and which outlet drains each part of the "
        "network, by the nearest route: the number and size of the arteriolar "
        "(inlet) and venular (outlet) territories, and the watershed vessels "
        "on the borders between them"
    ),
    "capillary_bed_joins": (
        "Where the capillary beds of different arteriolar feeds first meet. "
        "Each capillary belongs to the bed of the feed that reaches it first; "
        "a bed's join order is the lowest capillary B order at which it "
        "touches another bed. The Kaplan-Meier median join order and "
        "distance, the beds joining within or between arteriolar trees, and "
        "those that never join, by why"
    ),
    "capillary_path_orders": (
        "How many capillary branch orders lie between the arterioles and the "
        "venules. For every capillary draining into a venule, the fewest "
        "capillaries on a route to it from an arteriole, through capillaries "
        "alone, the length in microns of the shortest such route, the "
        "straight-line distance between its two ends and the length over "
        "that distance: the mean, median, SD and range of each, and the "
        "capillaries draining into a venule that no arteriole reaches"
    ),
    "transit_time": (
        "How long blood takes to cross the network: each vessel's volume "
        "over its flow, added up along inlet-to-outlet routes, as a mean and "
        "spread (SD, CV), and the capillaries' transit-time heterogeneity "
        "(CTH). Plug flow at the solved mean flow, not red-cell speed"
    ),
    "flux_heterogeneity": (
        "How unevenly the network is perfused: the coefficient of variation "
        "(SD over mean) of flow and of mean velocity across the vessels "
        "carrying flow, for the capillaries and for every vessel, and the "
        "share of the network carrying flow by vessel count, length and "
        "volume. A vessel carries flow when the solve reached it and its flow "
        "is above a floor far below the network's largest"
    ),
    "loop_hierarchy": (
        "The scale of the network's redundancy: the shortest loop through "
        "each vessel, as a distribution of loop lengths in microns overall and "
        "by vessel type, the vessels on no loop at all, and the independent "
        "loops per mm^3"
    ),
    "strahler": (
        "Horton-Strahler orders of the arterial and venous trees: tips are "
        "order 1, and two vessels of one order meeting make the next. Horton's "
        "bifurcation, length and diameter ratios say how count, length and "
        "diameter change from one order to the next; capillaries are unordered"
    ),
    "algebraic_connectivity": (
        "How easily the network splits in two: the second-smallest eigenvalue "
        "of its Laplacian (the Fiedler value), near 0 when a few vessels hold "
        "two halves together. Also the two halves the Fiedler vector divides "
        "it into and the vessels between them"
    ),
    "cyclomatic_number": (
        "The number of independent loops (anastomoses): vessels minus nodes "
        "plus separate pieces. 0 is a pure tree, where every vessel is the "
        "only supply to everything beyond it"
    ),
    "degree_assortativity": (
        "Whether nodes where many vessels meet connect to others like them "
        "(positive, up to 1) or to nodes with few vessels (negative, down to "
        "-1). Branching vascular trees usually read negative"
    ),
    "rich_club": (
        "Whether the best-connected nodes link to each other: of the vessels "
        "that could join the nodes with more than the threshold number of "
        "vessels, the fraction that do. Near 1, they form a backbone of their "
        "own"
    ),
    "k_core": (
        "How layered the network is. A node's core number is the largest k "
        "for which it sits in a sub-network where every node has at least k "
        "vessels; a tree's is 1, and a high core holding many nodes is a "
        "redundant, looped core inside a shell"
    ),
    "flow_hierarchy": (
        "The fraction of vessels whose solved flow does not run round a "
        "directed loop. Flow cannot circulate in a resistive network at steady "
        "state, so a value below 1 points to a modelling problem, such as "
        "boundary conditions fighting each other"
    ),
    "path_efficiency": (
        "The inverse of the mean shortest path length through the network "
        "between pairs of nodes (1/microns), with that mean length itself; "
        "higher means nodes are fewer microns apart. Sampled pairs in fast "
        "mode, every pair in full"
    ),
    "community": (
        "Groups of nodes more densely connected among themselves than to the "
        "rest of the network (modularity communities, from its topology): "
        "their number and sizes in fast mode, every group's nodes in full"
    ),
    "betweenness": (
        "How often each node lies on the shortest paths between other nodes, "
        "counting vessels rather than microns, from 0 to 1: the mean, highest "
        "and top nodes in fast mode, every node in full"
    ),
    "haemodynamics": (
        "The network's resistance to flow, in Pa.s/m^3: between the first "
        "inlet and the first outlet, or the two nodes Post processing chose "
        "(two-point), and for the whole network, "
        "the pressure drop over the total inflow through every inlet"
    ),
    "branch_orders": (
        "The vessels of each branch order (arterioles, capillary orders, "
        "venules): how many, their mean length, diameter and its variation, "
        "tortuosity, the angle at which they leave their parent, and the "
        "pressure drop across them"
    ),
    "weighted_betweenness_communities": (
        "Betweenness and communities with the vessels weighted three ways: "
        "by resistance (the easiest paths for flow), length (the shortest in "
        "microns) and 1/|flow| (the busiest). The mean and highest "
        "betweenness, the most central nodes, and the communities' number and "
        "sizes"
    ),
    "zero_resistance_bridges": (
        "The joins drawn inside a thick vessel's lumen, from where a vessel "
        "opens into it to its centreline: how many and how long. They carry "
        "no resistance, and no other statistic counts them as vessels"
    ),
    "tissue_volume": (
        "The tissue's volume and surface area from the raw image, which "
        "vessel density divides by: the share of the image it fills, how much "
        "of its surface the image faces cut, the vessels inside it, and the "
        "intensities its threshold split"
    ),
    # synthetic.compare.COMPARISON_SECTION: a synthetic network's run only.
    "synthetic_comparison": (
        "How close the synthetic network came to the template run it was drawn "
        "to, characteristic by characteristic: the template's value, the "
        "synthetic network's and their ratio, and whether it lies within the "
        "band the generator is held to (OUTSIDE when not). The same rows as "
        "the run's template comparison CSV"
    ),
}
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
#: The box under the buttons where the panel says what it last did.
REPORT_TOOLTIP = (
    "What the panel last said: a run's progress or why it stopped, what the "
    "checks found, and what a button did; the run's full narration is in the "
    "log window"
)
#: The "9. Additional measurements" tab, which has no stage function: every
#: other stage tab's hover text names the function it runs.
MEASURED_ON_EXPORT_TAB_TOOLTIP = (
    "Measured by export_results(settings, ...): this tab has no stage of its own"
)
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
    "into the viewer, points the run at it"
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
#: Both settings optimisers while the run draws a synthetic network, greyed:
#: there is no image for their trials to read. Also what each says if pressed.
OPTIMISE_SYNTHETIC_TOOLTIP = (
    "The run draws a synthetic network, so there is no image to optimise "
    "settings on: set the Input tab's Input to Segmented image to use this"
)
CHECK_SEGMENTED_IMAGE_TOOLTIP = (
    "Score the segmented input image out of 10 -- fragmentation, "
    "connectivity, vessels leaving the image, surface noise and resolution "
    "-- and print the breakdown to the log below"
)
#: The Batch run button and its window.
BATCH_RUN_TOOLTIPS = {
    "button": (
        "Run every segmented image in a folder with these settings, each "
        "matched to its raw image and vessel masks by file name, saving a "
        ".haemorun and the run's outputs in a folder named after the image"
    ),
    #: The button, greyed, under a synthetic network and while a run goes.
    "button_synthetic": (
        "A batch runs segmented images, and this run draws a synthetic network: "
        "set the Input tab's Input to Segmented image to use Batch run"
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
        "folder. While ilastik segments (Input tab), each raw image here is one run"
    ),
    "pericyte": (
        "The folder of pericyte masks: every file in it is the pericyte mask of the "
        "image sharing its name. A folder here turns on pericytes placed from a mask, "
        "which each pericyte perturbation reads"
    ),
    "cell": (
        "The folder of cell masks: every file in it is the cell mask of the image "
        "sharing its name. A folder here turns on the 3D cell distances"
    ),
    "vessel_3d": (
        "The folder of vessel masks for the 3D cell distances, one per image by name; "
        "left blank, each image's own segmentation. A folder here turns on the 3D "
        "cell distances"
    ),
    "tissue": (
        "The folder of tissue images: every file in it is the tissue image of the image "
        "sharing its name; left blank, each image's raw image. A folder here turns on "
        "the tissue volume"
    ),
    "capillary_block": (
        "The folder of masks of blocks: every file in it is the mask of blocks of the "
        "image sharing its name, read by each capillary block from a mask. A folder "
        "here turns on the perturbations, adding a capillary block from a mask if "
        "there is none"
    ),
    "large": (
        "The folder of large vessel masks; given on its own, every file in it is "
        "a large mask of the image sharing its name. Left blank, the segmented "
        "folder, read by name"
    ),
    "small": (
        "The folder of small vessel masks; given on its own, every file in it is "
        "a small mask of the image sharing its name. Left blank, the segmented "
        "folder, read by name"
    ),
    "output": "Each image's outputs and .haemorun go in a folder of its own in here",
    "browse": "Choose the folder",
    "images_tab": (
        "The folders, the conditions and the images matched in them, with what "
        "each image came to"
    ),
    "comparison_tab": (
        "The two conditions compared statistic by statistic, as the output "
        "folder's batch_condition_comparison.csv has them"
    ),
    "skip_done": (
        "Leave out images whose folder already has a .haemorun, so a batch "
        "stopped part way picks up where it ended"
    ),
    "table": (
        "Each segmented image, what was matched to it and what it came to; untick "
        "an image to leave it out, choose a file where several tie, double-click a "
        "finished image to open it, right-click for what else can be done to it, "
        "and click a heading to sort by it"
    ),
    "match": (
        "Read the folders again and match every image (done by itself when a folder "
        "is typed or chosen)"
    ),
    "last_batch": (
        "The output folder holds a batch's batch_settings.yaml: fill this window "
        "from it (folders, conditions, comparison, sections) and put the settings "
        "it ran with in the panel"
    ),
    "settings_line": "How many of the panel's settings, which every image runs with, differ from their defaults",
    "show_settings": "List the settings every image runs with that differ from their defaults",
    "peak_ram": (
        "The image's peak memory: the largest of its stages' peaks in its run "
        "manifest (where a stage did not raise the process's peak, its peak was "
        "read every 0.2 s, so it may be a little low)"
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
        "The two conditions tested statistic by statistic, smallest q (most "
        "significant) first: each condition's N and median, the difference, the "
        "test's P and the Benjamini-Hochberg q"
    ),
    "refresh_comparison": "Read the output folder's batch_condition_comparison.csv again",
    "open_comparison": "Open batch_condition_comparison.csv with the system's program for it",
    "open_batch_folder": "Open the batch's output folder",
    "stop": (
        "Let the image running finish and be saved, then stop: the images after it "
        "are not run (Cancel stops the image itself)"
    ),
    "run": "Use this batch and run it now, keeping this window open to follow it",
    "check": "Run every ticked image's pre-run checks and show what they say",
    "condition_a": (
        "One condition's images: a phrase their file names hold (ignoring case; "
        "several, comma-separated, for either). Each condition, and all images, "
        "is summarised once the batch is done"
    ),
    "condition_b": (
        "The other condition's phrase: with both set, every statistic is tested "
        "between the two (Mann-Whitney U, Benjamini-Hochberg q-values) into "
        "batch_condition_comparison.csv"
    ),
    "design": (
        "Independent images: every image one sample (Mann-Whitney U, Welch t). "
        "Paired by subject: each image against the image of the other condition "
        "whose name is most alike once the condition phrase is taken out "
        "(Wilcoxon signed-rank, paired t); images with no partner, or a tie, are "
        "left out and listed in batch_condition_pairs.csv"
    ),
    "subject_pattern": (
        "Paired only: a regular expression whose first group (or whole match) is "
        "an image's subject ID. Blank: the part of the file name before its "
        "condition phrase, so 28_3_d_contra_FITC is subject 28_3_d"
    ),
    "aggregate": (
        "Paired only: a subject with more than one image in a condition is left "
        "out, or its images there are combined, statistic by statistic, by their "
        "median or mean, and those are paired. batch_condition_pairs.csv lists "
        "the images in each subject's value"
    ),
    "exclude_sections": (
        "Statistics sections, comma-separated, left out of the batch statistics and "
        "the comparison (with their subsections, ignoring case), so the "
        "Benjamini-Hochberg q-values are over the statistics left; e.g. Dead Ends"
    ),
    "compare": (
        "Write the batch statistics from the ticked images already run in the "
        "output folder, by the conditions as they are now, without running anything"
    ),
    "use": "Close this window and let Run pipeline run the batch later",
    "leave": "Go back to running the one image the Input tab names",
}
REVERT_STAGE_TOOLTIP = (
    "Clear this tab and later layers and checkpoints, then rerun the "
    "pipeline from this stage using the previous tab's saved work"
)

#: View dock — display-only; never written into pipeline settings.
Z_DEPTH_TOOLTIP = (
    "Clip every displayed layer to this physical Z window; full range is the identity and does not crop the pipeline"
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
    "Tubes: each centreline step as a 3D prism that stays visible from every "
    "camera angle. Lines: napari vector ribbons (faster, can vanish edge-on). "
    "Zero-resistance bridges, artificial joins inside a thick vessel's lumen, "
    "are dashed in both"
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
        "Show on hover whether the flow solve reached the vessel, and if not, "
        "why: behind a dead end, behind a closed image-face end, or no inlet "
        "and outlet"
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
        "The range the vessel colours span, for the network chosen under "
        "Showing, drawn as tubes or as lines; each network keeps its own"
    ),
    "low": "The value drawn at the bottom of the colour map; type a number to change it",
    "high": "The value drawn at the top of the colour map; type a number to change it",
    "fit_all": "Span the smallest and largest value of the vessels on screen",
    "fit_trim": "Ignore the extreme 1% at each end, so the bulk of the vessels spreads",
    "canvas": (
        "Draw the colour scale in the canvas, on the tubes or the lines "
        "showing, for the network chosen under Showing"
    ),
}
SCALE_BAR_TOOLTIP = (
    "Show napari's scale bar in the bottom-right of the canvas, in microns when voxel size is known"
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
    "Step through the chosen perturbation's sweep: each grid point's flows, "
    "recoloured in place. Shown for a sweep perturbation chosen under Showing; "
    "each sweep keeps its own position while another is shown"
)
#: One slider per axis of a sweep's grid, in the Sweep box.
SWEEP_SLIDER_TOOLTIP = (
    "Step along this axis of the sweep's grid: the vessels are recoloured "
    "with the flows solved at that point, whose value is shown below"
)
TUBE_QUALITY_TOOLTIP = (
    "How round the vessel tubes are drawn: one smooth tube per vessel, with "
    "more sides round it each step right -- slower to build on a large "
    "network"
)
TUBE_DIAMETER_TOOLTIP = (
    "How wide the vessel tubes are drawn. Per vessel: each at its own "
    "diameter, measured or assigned. Uniform: every vessel at the width in "
    "the µm box beside it, 4 µm to start, so wide vessels do not hide the "
    "narrow ones. Display only -- the network keeps its diameters"
)
TUBE_UNIFORM_WIDTH_TOOLTIP = (
    "The width every vessel tube is drawn at while Tube diameter is Uniform, "
    "in microns; applied on Enter, on leaving the box or on each arrow step. "
    "Display only -- the network keeps its diameters"
)
SNAPSHOT_TOOLTIP = (
    "Write a TIFF of the current napari view into the pipeline "
    "output folder (cosmetic export only; does not feed the run)"
)
#: The view window's Rotation movie box.
ROTATION_TOOLTIPS = {
    "tilt": (
        "How far the movie looks down on the stack while it turns about the "
        "z axis, in degrees: 0 side on, z down the screen as in the XZ view; "
        "90 straight down onto the top slice; negative looks up from below"
    ),
    "speed": (
        "How fast the stack turns in the movie, in degrees a second: 30 takes "
        "12 s for the full turn, played at 30 frames a second"
    ),
    "rotate": (
        "Turn what is on the canvas about the z axis at this tilt and speed, "
        "round and round, to see the movie before recording it; the tilt and "
        "speed can be changed while it turns, and nothing is written. Press "
        "again to stop: the view turns back, keeping any zoom or pan made "
        "meanwhile"
    ),
    "record": (
        "Turn what is on the canvas one full circle and write it as a movie "
        "into the pipeline output folder (MP4, or GIF without imageio-ffmpeg); "
        "the view is put back afterwards. Press again to stop without writing"
    ),
}
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
        "Take the last run's network into this tab to edit it, and make the "
        "list chosen under Review -- only that one, so a scan does no more "
        "work than it needs"
    ),
    "review_choice": (
        "Which list Scan network makes: junctions where four or more vessels "
        "meet, short loops, or one of the checks on the network's geometry, "
        "segmentation, flow, boundaries and diameters, the terminal nodes "
        "to make inlets and outlets or take the role from, or the two nodes "
        "the two-point resistance is found between, to find and change. None "
        "lists nothing and leaves the network to edit by clicking"
    ),
    "region": (
        "Make the list only round the middle of the view, within this many "
        "microns of it; 0 lists the whole network. A check against the "
        "segmented image then reads only that part of it, which is faster "
        "on a large stack"
    ),
    "review_items": (
        "What the chosen check found, in the order it ranks them; click one "
        "to zoom to it and list its vessels below, or click its ring in the "
        "viewer. Kept items are not listed again"
    ),
    "dead_end_show": (
        "List every kind of dead end the scan found, or only one kind; "
        "pieces with no junction are vessels on their own, both ends dead "
        "ends"
    ),
    "remove_pieces": (
        "Remove every piece with no junction the list shows, each as Delete "
        "all its vessels would; a piece with a boundary node on it, or one "
        "a vessel was drawn onto since the scan, stays. Each is recorded in "
        "{stem}_dead_ends_review.csv"
    ),
    "review_table": (
        "The chosen item's vessels; click or shift/ctrl-click rows to select "
        "vessels, drawn yellow in the viewer"
    ),
    "review_delete_selected": (
        "Remove the vessels selected in the table; a junction left joining "
        "just two vessels is merged into one, and inlets, outlets and "
        "boundary nodes are never removed"
    ),
    "review_delete_item": (
        "Remove every vessel of the chosen item -- a dead end back to its "
        "junction, or a whole unsolved piece; inlets, outlets and boundary "
        "nodes are never removed"
    ),
    "review_merge": (
        "Take out the short vessel between the two junctions and make them "
        "one node, between the two (or where a boundary node is); the "
        "vessels at both now meet there"
    ),
    "review_trace_on": (
        "Start Add vessel at the chosen dead end's tip: click along the "
        "vessel it should go on to, and on a node or vessel to finish it"
    ),
    "review_join_ahead": (
        "Join the chosen dead end to the vessel straight ahead of it (at a "
        "thick vessel, to its centreline), with a vessel traced from its tip "
        "to where it meets that vessel through the image Add vessel uses; that "
        "vessel is split there"
    ),
    "review_mark_blind": (
        "Say the chosen dead end is a real blind end (a sprout, or a vessel "
        "that ends there). It stays; the mark is kept on its node and in the "
        "decisions CSV, and later scans and the statistics read it"
    ),
    "review_mark_open": (
        "Say the chosen dead end is a vessel leaving the image. It becomes an "
        "open end: with an open_end_model other than closed, the next solve "
        "gives it a boundary condition"
    ),
    "review_make_inlet": (
        "Make the chosen terminal node an inlet (an outlet no longer, if it "
        "was one). Regenerate graph solves the network with it, and the "
        "decision is written to the decisions CSV"
    ),
    "review_make_outlet": (
        "Make the chosen terminal node an outlet (an inlet no longer, if it "
        "was one). Regenerate graph solves the network with it, and the "
        "decision is written to the decisions CSV"
    ),
    "review_remove_terminal": (
        "Take the chosen node off the inlets or outlets; a dead end leaving "
        "the image is then an open end in the solve (open_end_model). The "
        "last inlet or outlet cannot be removed: make another one first"
    ),
    "review_resistance_start": (
        "Find the two-point resistance from the chosen node instead (the "
        "first inlet until changed); chosen while it is the end, the two "
        "swap. Regenerate graph solves it, and every regenerate after keeps "
        "the pair"
    ),
    "review_resistance_end": (
        "Find the two-point resistance to the chosen node instead (the first "
        "outlet until changed); chosen while it is the start, the two swap. "
        "Regenerate graph solves it, and every regenerate after keeps the pair"
    ),
    "review_join": (
        "Join the chosen pair of dead ends with a vessel traced between "
        "them through the image Add vessel uses, as wide as the vessels "
        "at its ends"
    ),
    "review_draw_along": (
        "Add a vessel along the chosen piece of skeleton, from the vessel it "
        "meets at one end to the one it meets at the other, each split there, "
        "as the skeleton re-trace would draw it; for a piece meeting the "
        "network once, trace it with Add vessel"
    ),
    "review_set_diameter": (
        "Give the vessels selected in the table -- or the chosen item's one "
        "vessel -- the diameter beside this button; Regenerate graph keeps "
        "it and solves the network again with it"
    ),
    "review_diameter": "The diameter Set diameter gives, in microns",
    "review_keep": (
        "Keep the chosen item as it is, take it off the list and move on to "
        "the next one; every decision is kept with the run and written to "
        "{stem}_{list}_review.csv beside the run's output"
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
    "loops": (
        "Short loops not yet reviewed, shortest first; click one to zoom to "
        "it, in its own plane in a 3D view, and list its sides below"
    ),
    "loop_table": (
        "The chosen loop's sides: the runs of vessels between the places "
        "other vessels, or a boundary node, meet it. Click a row to select "
        "that side, drawn yellow in the viewer"
    ),
    "loop_delete": (
        "Remove the side selected in the table, opening the loop; a junction "
        "left joining just two vessels is merged into one, and inlets, "
        "outlets and boundary nodes are never removed"
    ),
    "loop_keep": "Keep the chosen loop as it is, take it off the list and move on to the next one",
    "raw_panel": (
        "Draw the chosen loop or item in its own plane and in the plane across "
        "it, the segmentation beside the raw image on the Input tab, as the "
        "marked grids draw them. The raw image is read the first time"
    ),
    "raw_picture": (
        "A slab 3 µm either side of the plane: the segmentation's voxels "
        "anywhere in it, or the raw image's mean through it. The chosen item "
        "is cyan, the other centrelines grey, and the orange bar 5 µm"
    ),
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
#: Optimise settings' group checkboxes, shown under "Choose optimisation
#: types": what each group tries. Keyed by ``optimisation.GROUP_NAMES``.
OPTIMISE_GROUP_TOOLTIPS: dict[str, str] = {
    "segmentation_cleanup": (
        "Try the clean-up of the mask before it is skeletonised -- filling "
        "cavities, removing whiskers, splitting narrow necks, closing and "
        "reconnecting gaps, smoothing, removing small pieces -- and their "
        "sizes, each scored as Check segmented image scores the mask"
    ),
    "thick_vessel_gating": (
        "Try the alternate skeletonisation of thick vessels on and off, the "
        "radius above which a vessel counts as thick, and how the thick region "
        "is refined and thin vessels are joined onto its centreline"
    ),
    "min_branch_length": (
        "Try the shortest piece of skeleton kept when it is cleaned (Skeleton "
        "min branch length)"
    ),
    "bundle_refinement": (
        "Try the four settings that find dense tangles of skeleton and "
        "collapse each into a hub: the scan window, how dense a window must "
        "be, the links kept per hub and the spacing between hubs"
    ),
    "closing_radius": (
        "Try the radius of the closing that fills small gaps in the mask "
        "before it is skeletonised (Skeleton closing radius)"
    ),
    "gap_bridging": (
        "Try how large a gap in the skeleton is bridged (Skeleton bridge gap "
        "size) and how far a broken branch end reaches for another piece "
        "(Skeleton max bridge distance)"
    ),
    "connectivity_and_component_filter": (
        "Try the voxel connectivity skeleton pieces are counted by, and the "
        "smallest piece kept as a percentage of the skeleton (Skeleton min "
        "component percent)"
    ),
    "reconnect_thresholds": (
        "Try how far apart pieces of the graph may be and still be joined: "
        "the Graph reconnect threshold and the Final orphan reconnect threshold"
    ),
    "cluster_collapse_distance": (
        "Try how nodes lying close together are merged into one: the Cluster "
        "collapse distance and method, and the method's own setting"
    ),
    "min_stub_length": (
        "Try the shortest dead-end stub the graph keeps: Min stub length, or "
        "Min stub length radius multiple when stubs are judged by the radius "
        "of the vessel they leave"
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
        "Try how each profile is sampled and fitted -- in the slice plane or "
        "across the vessel's 3D direction, as a blurred lumen or a Gaussian, "
        "averaged along the vessel, how far apart -- and how many accepted "
        "profiles a vessel needs and how they are combined. Judged on vessels "
        "of known width planted beside the network's own, so it needs the "
        "run's segmentation"
    ),
    "exclusion_zones": (
        "Try how far from a branch end, and from a junction, profiles are "
        "left unsampled"
    ),
    "extent_and_widening": (
        "Try how far either side of the centreline each profile reaches, how "
        "much wider than the fitted width it must span, and how many times it "
        "may widen to make room"
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
        "vessel, or each vessel's own from a graph attribute"
    ),
    "rejection_gates": (
        "Try the gates that throw out poor profiles -- a fitted centre too far "
        "off the centreline, a poor fit (low R^2), a flat-topped plateau -- "
        "and the threshold of each"
    ),
}

#: "5. Diameters" — "Optimise endothelial settings", shown in place of
#: Optimise FWHM settings when the raw image stains the vessel wall, and its
#: options. Keyed by control; a group's checkbox by its name in
#: ``optimisation.endothelial_search.ENDOTHELIAL_GROUP_NAMES``.
ENDOTHELIAL_OPTIMISE_TOOLTIPS = {
    "button": (
        "Choose the endothelial ring fit's settings -- its blur and wall, "
        "estimated from the image or fixed, and how strong a ring must be to "
        "be read -- by running the real ring fit on a sample of the current "
        "network's vessels in the raw image, judged on vessels of known lumen "
        "drawn into a copy of it and on decoys in vessel-free tissue. "
        "Proposes the winners for this tab; the current network's diameters "
        "are not changed"
    ),
    "options": "Show or hide the endothelial optimiser's own options",
    "choose_groups": (
        "Restrict Optimise endothelial settings to only the ticked group(s) "
        "below, instead of every ring-fit setting"
    ),
    "blur_and_wall": (
        "Where the image's wide vessels cannot estimate the blur or the wall, "
        "or you have fixed them, try their estimate fixed and the ends of its "
        "95% interval; otherwise they stay estimated from the image"
    ),
    "ring_gate": (
        "Try half to twice the weakest ring a reading is kept for "
        "(endothelial_min_ring_contrast), and the ring fit's own default"
    ),
    "sample": (
        "How many of the network's vessels each trial reads; Auto times one "
        "reading and picks a count to keep the trials near ten minutes, "
        "with the blur and wall calibrations on top"
    ),
}
