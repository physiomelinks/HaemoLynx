"""Declarative schema for the image-to-model pipeline settings.

Every setting :func:`haemolynx.pipeline.run_pipeline_stages` reads is declared
here once, as a :class:`~haemolynx.parsers.Setting`. That single declaration is
what the YAML config writer, the command line, and a GUI form all read from, so
none of them has to repeat the list of settings.

It lives in the package rather than beside an example because it is the only
way to configure a run: without it there is nothing to hand
:func:`haemolynx.pipeline.resolve_settings`, so an installed copy of HaemoLynx
could not be run at all::

    from haemolynx.pipeline import default_schema, write_default_config

    write_default_config("my_config.yaml")       # commented, every setting
    settings = resolve_settings(schema=default_schema(), config_path="my_config.yaml")

It declares no dependency on the stage code, on ilastik, or on any image: the
settings are plain data, so a GUI can render its whole form from
``default_schema().describe()`` alone.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from ..parsers import Schema, Setting, dump_config

# Path defaults are relative, and so are resolved against the directory a run
# starts in. They name a plain working-directory layout rather than anything
# inside this repository, so a generated config is a starting point on any
# machine; the examples pin their own paths in their config files.
_IMAGES = "images"
_CLASSIFIERS = "classifiers"
_OUTPUTS = "outputs"
_PLOTS = "plots"

#: Axis orders accepted by the loader: any permutation of x, y and z.
AXIS_ORDERS = ("zyx", "zxy", "yzx", "yxz", "xzy", "xyz")

#: Boundary-node selection methods understood by ``select_boundary_nodes_by_method``.
NODE_SELECTION_METHODS = (
    "coordinates",
    "all_degree_1",
    "volume",
    "edge_percent",
    "degree_1_from_starting",
)

_INPUT_AND_SEGMENTATION = "Input and segmentation"
_VESSEL_MASKS = "Vessel masks"
_BOUNDARY_ASSIGNMENT = "Boundary assignment"
_SOLVER_AND_OUTPUT = "Solver and output"
_PIPELINE_STAGES = "Pipeline stages"
# Named "Statistics and measurements" rather than "Statistics" so the YAML
# section heading does not collide with the setting also called `statistics`.
_STATISTICS = "Statistics and measurements"
_DIAMETERS_AND_PERICYTES = "Diameters and pericytes"
_FWHM = "FWHM diameter measurement"


SCHEMA = Schema(
    [
        # ------------------------------------------------------------------
        # Input and segmentation
        # ------------------------------------------------------------------
        Setting(
            name="input_path",
            # No default: there is no image every run should read, and a
            # plausible-looking path that is not there costs more to diagnose
            # than an empty one. The pre-run checks say it is unset; the napari
            # panel fills it in from the layer you have open.
            kind="path",
            default=None,
            help="Read this already-segmented image as the pipeline input",
            section=_INPUT_AND_SEGMENTATION,
            must_exist=True,
            requires=("!use_ilastik_segmentation",)
        ),
        Setting(
            name="use_ilastik_segmentation",
            kind="bool",
            default=False,
            help="Segment the main input image with ilastik instead of reading a segmented file",
            section=_INPUT_AND_SEGMENTATION,
        ),
        Setting(
            name="ilastik_unsegmented_image_path",
            kind="path",
            default=f"{_IMAGES}/Nerve_capillaries.tif",
            help="Read this raw image as the ilastik input for main-image segmentation",
            section=_INPUT_AND_SEGMENTATION,
            requires=("use_ilastik_segmentation",),
            must_exist=True,
        ),
        Setting(
            name="ilastik_classifier_path",
            kind="path",
            default=f"{_CLASSIFIERS}/nerve_classifier.ilp",
            help="Use this trained ilastik project to segment the main image",
            section=_INPUT_AND_SEGMENTATION,
            requires=("use_ilastik_segmentation",),
            must_exist=True,
        ),
        Setting(
            name="ilastik_executable",
            kind="str",
            default="ilastik.exe",
            help="Run this ilastik executable name or path in headless mode",
            section=_INPUT_AND_SEGMENTATION,
        ),
        Setting(
            name="ilastik_output_dir",
            kind="path",
            default=f"{_OUTPUTS}/segmentations",
            help="Write ilastik-generated segmentations into this directory",
            section=_INPUT_AND_SEGMENTATION,
        ),
        Setting(
            name="ilastik_output_suffix",
            kind="str",
            default=".tif",
            help="Give ilastik segmentation outputs this file suffix",
            section=_INPUT_AND_SEGMENTATION,
        ),
        Setting(
            name="voxel_size_override_xyz",
            kind="float_list",
            default=None,
            help="Override the voxel size with these (x, y, z) values instead of reading image metadata",
            section=_INPUT_AND_SEGMENTATION,
            unit="um",
        ),
        Setting(
            name="voxel_size_policy",
            kind="choice",
            default="auto",
            help="Choose how the voxel size is resolved from metadata and the manual override",
            section=_INPUT_AND_SEGMENTATION,
            choices=("auto", "override", "metadata_only"),
        ),
        Setting(
            name="image_axis_order",
            kind="choice",
            default="zyx",
            help="Declare which array axis is which in the input files, so volumes can be transposed to canonical (z, y, x) on load",
            section=_INPUT_AND_SEGMENTATION,
            choices=AXIS_ORDERS,
        ),
        # ------------------------------------------------------------------
        # Vessel masks
        # ------------------------------------------------------------------
        Setting(
            name="use_large_vessel_masks",
            kind="bool",
            default=False,
            help="Load large arteriole and venule masks for automated start/output node assignment",
            section=_VESSEL_MASKS,
        ),
        Setting(
            name="use_ilastik_large_vessel_segmentation",
            kind="bool",
            default=False,
            help="Produce the large-vessel masks with ilastik instead of reading pre-segmented files",
            section=_VESSEL_MASKS,
            requires=("use_large_vessel_masks",),
        ),
        Setting(
            name="large_vessel_mask_dilation_microns",
            kind="float",
            default=0.0,
            help="Dilate the large-vessel masks by this distance before selecting terminal nodes",
            section=_VESSEL_MASKS,
            minimum=0.0,
            unit="um",
            requires=("use_large_vessel_masks",),
        ),
        Setting(
            name="large_arteriole_mask_path",
            kind="path",
            default=f"{_IMAGES}/large_arteriole_mask.tif",
            help="Read this pre-segmented large arteriole mask",
            section=_VESSEL_MASKS,
            requires=("use_large_vessel_masks",),
            must_exist=True,
        ),
        Setting(
            name="large_venule_mask_path",
            kind="path",
            default=f"{_IMAGES}/large_venule_mask.tif",
            help="Read this pre-segmented large venule mask",
            section=_VESSEL_MASKS,
            requires=("use_large_vessel_masks",),
            must_exist=True,
        ),
        Setting(
            name="ilastik_unsegmented_arteriole_image_path",
            kind="path",
            default=f"{_IMAGES}/large_arteriole_mask.tif",
            help="Read this raw image as the ilastik input for the large arteriole mask",
            section=_VESSEL_MASKS,
            requires=("use_large_vessel_masks", "use_ilastik_large_vessel_segmentation"),
        ),
        Setting(
            name="ilastik_unsegmented_venule_image_path",
            kind="path",
            default=f"{_IMAGES}/large_venule_mask.tif",
            help="Read this raw image as the ilastik input for the large venule mask",
            section=_VESSEL_MASKS,
            requires=("use_large_vessel_masks", "use_ilastik_large_vessel_segmentation"),
        ),
        Setting(
            name="ilastik_arteriole_classifier_path",
            kind="path",
            default=f"{_CLASSIFIERS}/arteriole_classifier.ilp",
            help="Use this trained ilastik project to segment arterioles",
            section=_VESSEL_MASKS,
            requires=("use_large_vessel_masks", "use_ilastik_large_vessel_segmentation"),
            must_exist=True,
        ),
        Setting(
            name="ilastik_venule_classifier_path",
            kind="path",
            default=f"{_CLASSIFIERS}/venule_classifier.ilp",
            help="Use this trained ilastik project to segment venules",
            section=_VESSEL_MASKS,
            requires=("use_large_vessel_masks", "use_ilastik_large_vessel_segmentation"),
            must_exist=True,
        ),
        Setting(
            name="use_small_vessel_masks_for_boundary_assignment",
            kind="bool",
            default=False,
            help="Assign arteriole and venule boundary nodes from small-vessel masks",
            section=_VESSEL_MASKS,
        ),
        Setting(
            name="use_ilastik_small_vessel_segmentation",
            kind="bool",
            default=False,
            help="Produce the small-vessel masks with ilastik instead of reading pre-segmented files",
            section=_VESSEL_MASKS,
            requires=("use_small_vessel_masks_for_boundary_assignment",),
        ),
        Setting(
            name="small_vessel_mask_min_overlap_fraction",
            kind="float",
            default=0.5,
            help="Require at least this fraction of an edge to lie inside a small-vessel mask before labelling it",
            section=_VESSEL_MASKS,
            minimum=0.0,
            maximum=1.0,
            requires=("use_small_vessel_masks_for_boundary_assignment",),
        ),
        Setting(
            name="write_small_vessel_boundary_labelling_3d_html",
            kind="bool",
            default=True,
            help="Write an interactive 3D HTML diagnostic of the small-vessel boundary labelling",
            section=_VESSEL_MASKS,
            requires=("use_small_vessel_masks_for_boundary_assignment",),
        ),
        Setting(
            name="small_arteriole_mask_path",
            kind="path",
            default=f"{_IMAGES}/small_arteriole_mask.tif",
            help="Read this pre-segmented small arteriole mask",
            section=_VESSEL_MASKS,
            requires=("use_small_vessel_masks_for_boundary_assignment",),
            must_exist=True,
        ),
        Setting(
            name="small_venule_mask_path",
            kind="path",
            default=f"{_IMAGES}/small_venule_mask.tif",
            help="Read this pre-segmented small venule mask",
            section=_VESSEL_MASKS,
            requires=("use_small_vessel_masks_for_boundary_assignment",),
            must_exist=True,
        ),
        Setting(
            name="ilastik_unsegmented_small_arteriole_image_path",
            kind="path",
            default=f"{_IMAGES}/small_arteriole_mask.tif",
            help="Read this raw image as the ilastik input for the small arteriole mask",
            section=_VESSEL_MASKS,
            requires=(
                "use_small_vessel_masks_for_boundary_assignment",
                "use_ilastik_small_vessel_segmentation",
            ),
        ),
        Setting(
            name="ilastik_unsegmented_small_venule_image_path",
            kind="path",
            default=f"{_IMAGES}/small_venule_mask.tif",
            help="Read this raw image as the ilastik input for the small venule mask",
            section=_VESSEL_MASKS,
            requires=(
                "use_small_vessel_masks_for_boundary_assignment",
                "use_ilastik_small_vessel_segmentation",
            ),
        ),
        Setting(
            name="ilastik_small_arteriole_classifier_path",
            kind="path",
            default=f"{_CLASSIFIERS}/arteriole_classifier.ilp",
            help="Use this trained ilastik project to segment small arterioles",
            section=_VESSEL_MASKS,
            requires=(
                "use_small_vessel_masks_for_boundary_assignment",
                "use_ilastik_small_vessel_segmentation",
            ),
            must_exist=True,
        ),
        Setting(
            name="ilastik_small_venule_classifier_path",
            kind="path",
            default=f"{_CLASSIFIERS}/venule_classifier.ilp",
            help="Use this trained ilastik project to segment small venules",
            section=_VESSEL_MASKS,
            requires=(
                "use_small_vessel_masks_for_boundary_assignment",
                "use_ilastik_small_vessel_segmentation",
            ),
            must_exist=True,
        ),
        # ------------------------------------------------------------------
        # Boundary assignment
        # ------------------------------------------------------------------
        Setting(
            name="base_plot_dir",
            kind="path",
            default=_PLOTS,
            help="Write plot artifacts under this base directory",
            section=_BOUNDARY_ASSIGNMENT,
        ),
        Setting(
            name="automated_vessel_assignment",
            kind="bool",
            default=False,
            help="Select start and output nodes automatically from the large-vessel masks instead of manually",
            section=_BOUNDARY_ASSIGNMENT,
            requires=("use_large_vessel_masks",),
        ),
        # "edge_percent" is the default for the two that every run needs
        # because it is the only method that asks nothing of the dataset: it
        # takes the terminals in the first and last band of the network along
        # one axis, so an image nobody has looked at yet still gets inlets and
        # outlets. "coordinates" and "volume" describe one dataset and no
        # other, which is why they are chosen rather than defaulted to.
        Setting(
            name="starting_node_selection_method",
            kind="choice",
            default="edge_percent",
            help="Choose how manual starting nodes are picked from the graph",
            section=_BOUNDARY_ASSIGNMENT,
            choices=NODE_SELECTION_METHODS,
        ),
        Setting(
            name="output_node_selection_method",
            kind="choice",
            default="edge_percent",
            help="Choose how manual output nodes are picked from the graph",
            section=_BOUNDARY_ASSIGNMENT,
            choices=NODE_SELECTION_METHODS,
        ),
        Setting(
            name="arteriole_boundary_selection_method",
            kind="choice",
            default="coordinates",
            help="Choose how manual arteriole boundary nodes are picked from the graph",
            section=_BOUNDARY_ASSIGNMENT,
            choices=NODE_SELECTION_METHODS,
        ),
        Setting(
            name="venule_boundary_selection_method",
            kind="choice",
            default="coordinates",
            help="Choose how manual venule boundary nodes are picked from the graph",
            section=_BOUNDARY_ASSIGNMENT,
            choices=NODE_SELECTION_METHODS,
        ),
        # What the "edge_percent" method reads. Shared by every role: one axis
        # and one pair of bands describe the whole network.
        Setting(
            name="boundary_axis",
            kind="int",
            default=1,
            help="Split inlets from outlets along this array axis when the edge_percent method is used (0=z, 1=y, 2=x)",
            section=_BOUNDARY_ASSIGNMENT,
            minimum=0,
            maximum=2,
        ),
        Setting(
            name="boundary_first_percent",
            kind="float",
            default=10.0,
            help="Take inlets from the terminals in this much of the network, measured from its start along the boundary axis",
            section=_BOUNDARY_ASSIGNMENT,
            unit="percent",
            minimum=0.0,
            maximum=100.0,
        ),
        Setting(
            name="boundary_last_percent",
            kind="float",
            default=10.0,
            help="Take outlets from the terminals in this much of the network, measured back from its end along the boundary axis",
            section=_BOUNDARY_ASSIGNMENT,
            unit="percent",
            minimum=0.0,
            maximum=100.0,
        ),
        Setting(
            name="boundary_distance_from_starting_node",
            kind="float",
            default=0.0,
            help="Keep only the terminals further than this from a starting node when the degree_1_from_starting method is used",
            section=_BOUNDARY_ASSIGNMENT,
            unit="um",
            minimum=0.0,
        ),
        # The coordinates below apply whenever a role's selection method is
        # "coordinates". They are empty by default because a coordinate is a
        # statement about one dataset: the previous default named six points in
        # one brain stack, which selected six arbitrary terminals in anything
        # else and left the outlets, whose list was empty, unfindable.
        Setting(
            name="starting_node_coordinates",
            kind="any",
            default=[],
            help="Pick starting nodes nearest to these coordinates when the coordinates method is used",
            section=_BOUNDARY_ASSIGNMENT,
        ),
        Setting(
            name="output_node_coordinates",
            kind="any",
            default=[],
            help="Pick output nodes nearest to these coordinates when the coordinates method is used",
            section=_BOUNDARY_ASSIGNMENT,
        ),
        Setting(
            name="arteriole_boundary_node_coordinates",
            kind="any",
            default=[],
            help="Pick arteriole boundary nodes nearest to these coordinates when the coordinates method is used",
            section=_BOUNDARY_ASSIGNMENT,
        ),
        Setting(
            name="venule_boundary_node_coordinates",
            kind="any",
            default=[],
            help="Pick venule boundary nodes nearest to these coordinates when the coordinates method is used",
            section=_BOUNDARY_ASSIGNMENT,
        ),
        # The volume boxes below apply whenever a role's selection method is
        # "volume"; that choice is the switch, so there is no separate flag.
        Setting(
            name="starting_node_volumes",
            kind="any",
            default=[],
            help="Select starting nodes falling inside these (min corner, max corner) boxes",
            section=_BOUNDARY_ASSIGNMENT,
        ),
        Setting(
            name="output_node_volumes",
            kind="any",
            default=[],
            help="Select output nodes falling inside these (min corner, max corner) boxes",
            section=_BOUNDARY_ASSIGNMENT,
        ),
        Setting(
            name="arteriole_boundary_node_volumes",
            kind="any",
            default=[],
            help="Select arteriole boundary nodes falling inside these (min corner, max corner) boxes",
            section=_BOUNDARY_ASSIGNMENT,
        ),
        Setting(
            name="venule_boundary_node_volumes",
            kind="any",
            default=[],
            help="Select venule boundary nodes falling inside these (min corner, max corner) boxes",
            section=_BOUNDARY_ASSIGNMENT,
        ),
        Setting(
            # The pipeline fills these in place (`starting_nodes[:] = []`), so
            # they must stay mutable lists rather than fixed-length tuples.
            name="starting_nodes",
            kind="any",
            default=[],
            help="Hold the starting node IDs chosen during the run; leave empty to let the pipeline fill it",
            section=_BOUNDARY_ASSIGNMENT,
            advanced=True,
        ),
        Setting(
            name="output_nodes",
            kind="any",
            default=[],
            help="Hold the output node IDs chosen during the run; leave empty to let the pipeline fill it",
            section=_BOUNDARY_ASSIGNMENT,
            advanced=True,
        ),
        Setting(
            name="arteriole_boundary_nodes",
            kind="any",
            default=[],
            help="Hold the arteriole boundary node IDs chosen during the run; leave empty to let the pipeline fill it",
            section=_BOUNDARY_ASSIGNMENT,
            advanced=True,
        ),
        Setting(
            name="venule_boundary_nodes",
            kind="any",
            default=[],
            help="Hold the venule boundary node IDs chosen during the run; leave empty to let the pipeline fill it",
            section=_BOUNDARY_ASSIGNMENT,
            advanced=True,
        ),
        Setting(
            name="strict_branch_order_assignment",
            kind="bool",
            default=False,
            help="Fail rather than fall back when the hierarchical branch-order prerequisites are not met",
            section=_BOUNDARY_ASSIGNMENT,
        ),
        # ------------------------------------------------------------------
        # Solver and output
        # ------------------------------------------------------------------
        Setting(
            name="input_p_bc",
            kind="float",
            default=4500.0,
            help="Apply this pressure boundary condition at the inlet nodes",
            section=_SOLVER_AND_OUTPUT,
            minimum=0.0,
            unit="Pa",
        ),
        Setting(
            name="output_p_bc",
            kind="float",
            default=1000.0,
            help="Apply this pressure boundary condition at the outlet nodes",
            section=_SOLVER_AND_OUTPUT,
            minimum=0.0,
            unit="Pa",
        ),
        Setting(
            name="visualize_results",
            kind="bool",
            default=True,
            help="Generate the matplotlib visualizations at the end of the run",
            section=_SOLVER_AND_OUTPUT,
        ),
        Setting(
            name="interactive_plots",
            kind="bool",
            default=False,
            help="Show plots interactively instead of saving them to the plot directory",
            section=_SOLVER_AND_OUTPUT,
            requires=("visualize_results",),
        ),
        Setting(
            name="show_plots_in_ide",
            kind="bool",
            default=True,
            help="Open saved plots in IDE windows while the run proceeds",
            section=_SOLVER_AND_OUTPUT,
            requires=("visualize_results",),
        ),
        Setting(
            name="ide_plot_mode",
            kind="choice",
            default="final_only",
            help="Choose which plots are shown in IDE windows",
            section=_SOLVER_AND_OUTPUT,
            choices=("all", "final_only", "none"),
            requires=("visualize_results", "show_plots_in_ide"),
        ),
        Setting(
            name="hold_ide_plots_open",
            kind="bool",
            default=True,
            help="Block at the end of the script so the IDE plot windows stay open",
            section=_SOLVER_AND_OUTPUT,
            requires=("visualize_results", "show_plots_in_ide"),
        ),
        Setting(
            name="final_render_mode",
            kind="choice",
            default="3d",
            help="Render the final graph views as a 2D overlay or an interactive 3D plot",
            section=_SOLVER_AND_OUTPUT,
            choices=("2d", "3d"),
            requires=("visualize_results",),
        ),
        Setting(
            name="vtk_export",
            kind="bool",
            default=True,
            help="Export vessels, pericytes and nodes as VTK files",
            section=_SOLVER_AND_OUTPUT,
            requires=("run_haemodynamics",),
        ),
        Setting(
            name="visualize_vtk",
            kind="bool",
            default=False,
            help="Open the exported VTK network in the interactive viewer",
            section=_SOLVER_AND_OUTPUT,
            requires=("run_haemodynamics", "vtk_export"),
        ),
        Setting(
            name="verbose_logging",
            kind="bool",
            default=False,
            help="Print verbose per-stage logging while the pipeline runs",
            section=_SOLVER_AND_OUTPUT,
        ),
        # ------------------------------------------------------------------
        # Pipeline stages
        # ------------------------------------------------------------------
        Setting(
            name="do_skeletonize",
            kind="bool",
            default=True,
            help="Run the skeletonization stage instead of reusing the cached skeleton",
            section=_PIPELINE_STAGES,
        ),
        Setting(
            name="do_graph_building",
            kind="bool",
            default=True,
            help="Run the graph-building stage instead of reusing the cached graph",
            section=_PIPELINE_STAGES,
        ),
        Setting(
            name="run_haemodynamics",
            kind="bool",
            default=True,
            help="Run the haemodynamics stage after the graph is built",
            section=_PIPELINE_STAGES,
        ),
        Setting(
            name="do_equiv_resistance_calculation",
            kind="bool",
            default=True,
            help="Compute the two-point equivalent resistance between the chosen node pair",
            section=_PIPELINE_STAGES,
            requires=("run_haemodynamics",),
        ),
        Setting(
            name="vtk_output_prefix",
            kind="path",
            default=f"{_OUTPUTS}/resistance_network",
            help="Prefix every VTK artifact and cached stage file with this output path",
            section=_PIPELINE_STAGES,
        ),
        Setting(
            name="skeleton_closing_radius",
            kind="int",
            default=2,
            help="Close gaps in the mask with a structuring element of this radius before skeletonizing",
            section=_PIPELINE_STAGES,
            minimum=0,
            requires=("do_skeletonize",),
        ),
        Setting(
            name="skeleton_bridge_gap_size",
            kind="int",
            default=3,
            help="Bridge skeleton gaps no larger than this many voxels",
            section=_PIPELINE_STAGES,
            minimum=0,
            requires=("do_skeletonize",),
        ),
        Setting(
            name="skeleton_min_branch_length",
            kind="int",
            default=3,
            help="Drop skeleton branches shorter than this during cleaning",
            section=_PIPELINE_STAGES,
            minimum=0,
            requires=("do_skeletonize",),
        ),
        Setting(
            name="skeleton_max_bridge_distance",
            kind="int",
            default=4,
            help="Reconnect skeleton fragments no further apart than this distance",
            section=_PIPELINE_STAGES,
            minimum=0,
            requires=("do_skeletonize",),
        ),
        Setting(
            name="skeleton_component_connectivity",
            kind="int",
            default=3,
            help="Use this voxel connectivity when analysing skeleton components",
            section=_PIPELINE_STAGES,
            minimum=1,
            maximum=3,
            requires=("do_skeletonize",),
        ),
        Setting(
            name="graph_reconnect_threshold",
            kind="float",
            default=10.0,
            help="Reconnect graph fragments whose endpoints lie within this distance",
            section=_PIPELINE_STAGES,
            minimum=0.0,
            requires=("do_graph_building",),
        ),
        Setting(
            name="final_orphan_reconnect_threshold",
            kind="float",
            default=3.0,
            help="Reconnect leftover orphan and dangling nodes within this distance in the final pass",
            section=_PIPELINE_STAGES,
            minimum=0.0,
            requires=("do_graph_building",),
        ),
        Setting(
            name="min_stub_length",
            kind="float",
            default=10.0,
            help="Prune terminal stubs shorter than this length",
            section=_PIPELINE_STAGES,
            minimum=0.0,
            requires=("do_graph_building",),
        ),
        Setting(
            name="cluster_collapse_distance",
            kind="float",
            default=5.0,
            help="Collapse clusters of nodes lying within this distance of each other",
            section=_PIPELINE_STAGES,
            minimum=0.0,
            requires=("do_graph_building",),
        ),
        Setting(
            name="skeleton_min_component_percent",
            kind="float",
            default=0.0,
            help="Discard skeleton components smaller than this percentage of the total after cleanup",
            section=_PIPELINE_STAGES,
            minimum=0.0,
            maximum=100.0,
            unit="percent",
            requires=("do_skeletonize",),
        ),
        Setting(
            name="smooth_centrelines",
            kind="bool",
            default=True,
            help=(
                "Take the voxel staircase out of each vessel's centreline after "
                "the graph is built, and re-measure its length from the result"
            ),
            section=_PIPELINE_STAGES,
            requires=("do_graph_building",),
        ),
        Setting(
            name="centreline_smoothing_method",
            kind="choice",
            default="taubin",
            help="How to smooth a centreline: taubin recovers true length, chaikin changes it least",
            section=_PIPELINE_STAGES,
            choices=("taubin", "chaikin"),
            requires=("smooth_centrelines",),
        ),
        Setting(
            name="centreline_smoothing_iterations",
            kind="int",
            default=10,
            help=(
                "How many smoothing passes; accuracy plateaus around 15, and past "
                "that the filter slowly starts inflating a centreline instead"
            ),
            section=_PIPELINE_STAGES,
            minimum=0,
            maximum=50,
            requires=("smooth_centrelines",),
        ),
        Setting(
            name="centreline_max_deviation",
            kind="float",
            default=1.0,
            help=(
                "How far a smoothed centreline may sit from the skeleton before it "
                "is blended back towards the original"
            ),
            section=_PIPELINE_STAGES,
            minimum=0.0,
            unit="microns",
            requires=("smooth_centrelines",),
        ),
        # ------------------------------------------------------------------
        # Statistics
        # ------------------------------------------------------------------
        Setting(
            name="statistics",
            kind="bool",
            default=False,
            help="Compute and export the global vessel statistics",
            section=_STATISTICS,
        ),
        Setting(
            name="measurement_3d_to_cell_mask",
            kind="bool",
            default=False,
            help="Measure 3D distances from the vessel network to a cell mask",
            section=_STATISTICS,
        ),
        Setting(
            name="cell_mask_path",
            kind="path",
            default=None,
            help="Read this cell mask for the 3D distance measurement",
            section=_STATISTICS,
            requires=("measurement_3d_to_cell_mask",),
            must_exist=True,
        ),
        Setting(
            name="cell_mask_h5_dataset_name",
            kind="str",
            default=None,
            help="Read this dataset from the cell mask when it is an H5 file",
            section=_STATISTICS,
            requires=("measurement_3d_to_cell_mask",),
        ),
        Setting(
            name="measurement_3d_vessel_mask_path",
            kind="path",
            default=None,
            help="Use this explicit vessel mask for the 3D distance measurement instead of the pipeline input",
            section=_STATISTICS,
            requires=("measurement_3d_to_cell_mask",),
            must_exist=True,
        ),
        Setting(
            name="measurement_3d_vessel_mask_h5_dataset_name",
            kind="str",
            default=None,
            help="Read this dataset from the 3D-distance vessel mask when it is an H5 file",
            section=_STATISTICS,
            requires=("measurement_3d_to_cell_mask",),
        ),
        Setting(
            name="measurement_3d_reference_image_path",
            kind="path",
            default=None,
            help="Take the vessel-volume raster shape from this reference image",
            section=_STATISTICS,
            requires=("measurement_3d_to_cell_mask",),
            must_exist=True,
        ),
        Setting(
            name="measurement_3d_reference_h5_dataset_name",
            kind="str",
            default=None,
            help="Read this dataset from the 3D-distance reference image when it is an H5 file",
            section=_STATISTICS,
            requires=("measurement_3d_to_cell_mask",),
        ),
        Setting(
            name="statistics_mode",
            kind="choice",
            default="fast",
            help="Compute either the fast subset of vessel statistics or the full set",
            section=_STATISTICS,
            choices=("fast", "full"),
            requires=("statistics",),
        ),
        # ------------------------------------------------------------------
        # Diameters and pericytes
        # ------------------------------------------------------------------
        Setting(
            name="all_diams_const",
            kind="bool",
            default=True,
            help="Give every branch order the same vessel diameter instead of a per-order table",
            section=_DIAMETERS_AND_PERICYTES,
        ),
        Setting(
            name="do_pericyte_construction",
            kind="bool",
            default=False,
            help="Model pericyte constrictions when assigning Poiseuille resistances",
            section=_DIAMETERS_AND_PERICYTES,
            requires=("run_haemodynamics",),
        ),
        Setting(
            name="use_pericyte_mask_constriction",
            kind="bool",
            default=False,
            help="Place constrictions at pericyte centroids taken from a mask rather than at regular spacing",
            section=_DIAMETERS_AND_PERICYTES,
            requires=("run_haemodynamics",),
        ),
        Setting(
            name="pericyte_mask_path",
            kind="path",
            default=None,
            help="Read pericyte locations from this mask",
            section=_DIAMETERS_AND_PERICYTES,
            requires=("use_pericyte_mask_constriction",),
            must_exist=True,
        ),
        Setting(
            name="pericyte_mask_h5_dataset_name",
            kind="str",
            default=None,
            help="Read this dataset from the pericyte mask when it is an H5 file",
            section=_DIAMETERS_AND_PERICYTES,
            requires=("use_pericyte_mask_constriction",),
        ),
        Setting(
            name="pericyte_max_assignment_distance_um",
            kind="float",
            default=3.0,
            help="Assign a pericyte centroid to a vessel edge only within this distance",
            section=_DIAMETERS_AND_PERICYTES,
            minimum=0.0,
            unit="um",
            requires=("use_pericyte_mask_constriction",),
        ),
        Setting(
            name="pericyte_min_diameter_um",
            kind="float",
            default=5.0,
            help="Treat mask components below this diameter as too small to be a pericyte",
            section=_DIAMETERS_AND_PERICYTES,
            minimum=0.0,
            unit="um",
            requires=("use_pericyte_mask_constriction",),
        ),
        Setting(
            name="pericyte_max_diameter_um",
            kind="float",
            default=12.0,
            help="Treat mask components above this diameter as too large to be a pericyte",
            section=_DIAMETERS_AND_PERICYTES,
            minimum=0.0,
            unit="um",
            requires=("use_pericyte_mask_constriction",),
        ),
        Setting(
            name="use_probabilistic_pericyte_constriction",
            kind="bool",
            default=False,
            help="Activate each candidate pericyte constriction at random rather than activating all of them",
            section=_DIAMETERS_AND_PERICYTES,
            requires=("run_haemodynamics",),
        ),
        Setting(
            name="pericyte_constriction_probability",
            kind="float",
            default=0.8,
            help="Activate each candidate pericyte constriction with this probability",
            section=_DIAMETERS_AND_PERICYTES,
            minimum=0.0,
            maximum=1.0,
            requires=("use_probabilistic_pericyte_constriction",),
        ),
        Setting(
            name="pericyte_constriction_seed",
            kind="int",
            # Kept equal to haemolynx.haemodynamics.apply's
            # DEFAULT_PERICYTE_CONSTRICTION_SEED (this module stays importable
            # without numpy, so the value is repeated rather than imported).
            default=20240917,
            help=(
                "Draw the random pericyte cohort from this seed, so a run repeats; "
                "another seed constricts a different set of pericytes, and null "
                "draws a fresh cohort every run"
            ),
            section=_DIAMETERS_AND_PERICYTES,
            requires=("use_probabilistic_pericyte_constriction",),
        ),
        Setting(
            name="run_pericyte_resistance_comparison",
            kind="bool",
            default=False,
            help="Run the baseline-versus-constricted pericyte resistance comparison and write its CSV",
            section=_DIAMETERS_AND_PERICYTES,
            requires=("run_haemodynamics",),
        ),
        Setting(
            name="pericyte_comparison_baseline_value",
            kind="float",
            default=1.0,
            help="Use this constriction factor as the baseline arm of the pericyte comparison",
            section=_DIAMETERS_AND_PERICYTES,
            minimum=0.0,
            requires=("run_pericyte_resistance_comparison",),
        ),
        Setting(
            name="pericyte_comparison_constricted_value",
            kind="float",
            default=0.8,
            help="Use this constriction factor as the constricted arm of the pericyte comparison",
            section=_DIAMETERS_AND_PERICYTES,
            minimum=0.0,
            requires=("run_pericyte_resistance_comparison",),
        ),
        Setting(
            name="reuse_comparison_pericyte_cohort_for_main_run",
            kind="bool",
            default=False,
            help="Reuse the randomly selected pericyte cohort from the comparison in the main run",
            section=_DIAMETERS_AND_PERICYTES,
            requires=(
                "run_pericyte_resistance_comparison",
                "use_probabilistic_pericyte_constriction",
            ),
        ),
        Setting(
            name="max_branch_order",
            kind="int",
            default=51,
            help="Build the diameter and constriction tables up to this branch-order index",
            section=_DIAMETERS_AND_PERICYTES,
            minimum=1,
        ),
        Setting(
            name="default_diameter",
            kind="float",
            default=4.0,
            help="Use this vessel diameter where no branch-order override applies",
            section=_DIAMETERS_AND_PERICYTES,
            minimum=0.0,
            unit="um",
        ),
        Setting(
            name="manual_capillary_diameter_by_branch_order",
            kind="mapping",
            default={"B01": 6.2, "B02": 4.0, "B03": 5.0, "B04": 5.0},
            help="Override capillary diameters per branch-order label, e.g. B01",
            section=_DIAMETERS_AND_PERICYTES,
            unit="um",
        ),
        Setting(
            name="manual_arteriole_diameter_by_branch_order",
            kind="mapping",
            default={},
            help="Override arteriole diameters per branch-order label, e.g. Art1",
            section=_DIAMETERS_AND_PERICYTES,
            unit="um",
        ),
        Setting(
            name="manual_venule_diameter_by_branch_order",
            kind="mapping",
            default={},
            help="Override venule diameters per branch-order label, e.g. Ven1",
            section=_DIAMETERS_AND_PERICYTES,
            unit="um",
        ),
        Setting(
            name="diameter_by_branch_order",
            kind="mapping",
            default=None,
            help="Supply the full branch-order diameter lookup; leave unset to derive it from the manual diameter settings",
            section=_DIAMETERS_AND_PERICYTES,
            unit="um",
            advanced=True,
        ),
        Setting(
            name="constriction_by_branch_order",
            kind="mapping",
            default=None,
            help="Supply the full branch-order constriction-factor lookup; leave unset to derive it from the manual diameter settings",
            section=_DIAMETERS_AND_PERICYTES,
            advanced=True,
        ),
        # ------------------------------------------------------------------
        # FWHM diameter measurement
        # ------------------------------------------------------------------
        Setting(
            name="use_fwhm_edge_diameters",
            kind="bool",
            default=False,
            help="Measure per-edge diameters from the raw image by full-width-half-maximum fitting",
            section=_FWHM,
            requires=("run_haemodynamics",),
        ),
        Setting(
            name="fwhm_raw_tiff_path",
            kind="path",
            default=None,
            help="Measure FWHM diameters from this raw single-channel image",
            section=_FWHM,
            requires=("use_fwhm_edge_diameters",),
            must_exist=True,
        ),
        Setting(
            name="fwhm_sample_spacing_along_edge_um",
            kind="float",
            default=2.0,
            help="Sample a transverse profile every this far along the centerline",
            section=_FWHM,
            minimum=0.0,
            unit="um",
            requires=("use_fwhm_edge_diameters",),
        ),
        Setting(
            name="fwhm_transverse_profile_step_um",
            kind="float",
            default=0.25,
            help="Step this far between points when sampling each transverse profile",
            section=_FWHM,
            minimum=0.0,
            unit="um",
            requires=("use_fwhm_edge_diameters",),
            advanced=True,
        ),
        Setting(
            name="fwhm_transverse_half_extent_um",
            kind="float",
            default=6.0,
            help="Sample this far either side of the centerline before any adaptive adjustment",
            section=_FWHM,
            minimum=0.0,
            unit="um",
            requires=("use_fwhm_edge_diameters",),
            advanced=True,
        ),
        Setting(
            name="fwhm_diameter_guess_um",
            kind="float",
            default=None,
            help="Start the Gaussian fit from this diameter guess instead of an automatic one",
            section=_FWHM,
            minimum=0.0,
            unit="um",
            requires=("use_fwhm_edge_diameters",),
            advanced=True,
        ),
        Setting(
            name="fwhm_min_total_extent_multiplier",
            kind="float",
            default=3.0,
            help="Require the sampled profile to span at least this multiple of the fitted width",
            section=_FWHM,
            minimum=0.0,
            requires=("use_fwhm_edge_diameters",),
            advanced=True,
        ),
        Setting(
            name="fwhm_background_label",
            kind="int",
            default=0,
            help="Mark background voxels with this label in the rasterized branch label volume",
            section=_FWHM,
            requires=("use_fwhm_edge_diameters",),
            advanced=True,
        ),
        Setting(
            name="fwhm_junction_label",
            kind="int",
            default=-1,
            help="Mark junction voxels with this label in the rasterized branch label volume",
            section=_FWHM,
            requires=("use_fwhm_edge_diameters",),
            advanced=True,
        ),
        Setting(
            name="fwhm_allow_junction_crossing",
            kind="bool",
            default=False,
            help="Let a transverse profile pass through junction voxels while sampling",
            section=_FWHM,
            requires=("use_fwhm_edge_diameters",),
            advanced=True,
        ),
        Setting(
            name="fwhm_profile_baseline_mode",
            kind="choice",
            default="wings",
            help="Estimate the profile baseline from the profile wings or from a percentile",
            section=_FWHM,
            choices=("wings", "percentile"),
            requires=("use_fwhm_edge_diameters",),
            advanced=True,
        ),
        Setting(
            name="fwhm_profile_baseline_wing_fraction",
            kind="float",
            default=0.2,
            help="Take this fraction of each profile end as the wing used for baseline estimation",
            section=_FWHM,
            minimum=0.0,
            maximum=1.0,
            requires=("use_fwhm_edge_diameters",),
            advanced=True,
        ),
        Setting(
            name="fwhm_constrain_fitted_baseline",
            kind="bool",
            default=False,
            help="Hold the fitted baseline near the anchor estimate rather than letting it float",
            section=_FWHM,
            requires=("use_fwhm_edge_diameters",),
            advanced=True,
        ),
        Setting(
            name="fwhm_baseline_constraint_half_width_ptp",
            kind="float",
            default=0.35,
            help="Allow the fitted baseline to move this fraction of the profile peak-to-peak range either way",
            section=_FWHM,
            minimum=0.0,
            maximum=1.0,
            requires=("use_fwhm_edge_diameters", "fwhm_constrain_fitted_baseline"),
            advanced=True,
        ),
        Setting(
            name="fwhm_clip_profile_to_single_vessel",
            kind="bool",
            default=True,
            help="Clip each profile to the central lobe so neighbouring branches do not bias the fit",
            section=_FWHM,
            requires=("use_fwhm_edge_diameters",),
            advanced=True,
        ),
        Setting(
            name="fwhm_clip_min_drop_fraction_of_center",
            kind="float",
            default=0.35,
            help="Require the profile to fall by this fraction of the centre value before clipping starts",
            section=_FWHM,
            minimum=0.0,
            maximum=1.0,
            requires=("use_fwhm_edge_diameters", "fwhm_clip_profile_to_single_vessel"),
            advanced=True,
        ),
        Setting(
            name="fwhm_clip_re_rise_fraction_of_center",
            kind="float",
            default=0.08,
            help="End the clipped lobe once the profile rises again by this fraction of the centre value",
            section=_FWHM,
            minimum=0.0,
            maximum=1.0,
            requires=("use_fwhm_edge_diameters", "fwhm_clip_profile_to_single_vessel"),
            advanced=True,
        ),
        Setting(
            name="fwhm_branch_endpoint_exclusion_um",
            kind="float",
            default=10.0,
            help="Skip sample positions within this distance of a branch endpoint",
            section=_FWHM,
            minimum=0.0,
            unit="um",
            requires=("use_fwhm_edge_diameters",),
            advanced=True,
        ),
        Setting(
            name="fwhm_junction_proximity_exclusion_um",
            kind="float",
            default=10.0,
            help="Skip sample positions within this distance of a detected junction voxel",
            section=_FWHM,
            minimum=0.0,
            unit="um",
            requires=("use_fwhm_edge_diameters",),
            advanced=True,
        ),
        Setting(
            name="fwhm_enforce_same_edge_locality",
            kind="bool",
            default=True,
            help="Require profile samples to stay within a local arc window of the same edge",
            section=_FWHM,
            requires=("use_fwhm_edge_diameters",),
            advanced=True,
        ),
        Setting(
            name="fwhm_same_edge_arc_window_um",
            kind="float",
            default=3.0,
            help="Use this absolute arc window when checking same-edge locality",
            section=_FWHM,
            minimum=0.0,
            unit="um",
            requires=("use_fwhm_edge_diameters", "fwhm_enforce_same_edge_locality"),
            advanced=True,
        ),
        Setting(
            name="fwhm_same_edge_arc_window_multiplier",
            kind="float",
            default=1.0,
            help="Scale the same-edge arc window by this factor",
            section=_FWHM,
            minimum=0.0,
            requires=("use_fwhm_edge_diameters", "fwhm_enforce_same_edge_locality"),
            advanced=True,
        ),
        Setting(
            name="fwhm_same_edge_arc_window_min_um",
            kind="float",
            default=1.0,
            help="Never shrink the same-edge arc window below this size",
            section=_FWHM,
            minimum=0.0,
            unit="um",
            requires=("use_fwhm_edge_diameters", "fwhm_enforce_same_edge_locality"),
            advanced=True,
        ),
        Setting(
            name="fwhm_cap_half_extent_by_nonlocal_same_edge_distance",
            kind="bool",
            default=True,
            help="Shorten the profile half-extent when a distant part of the same edge comes close",
            section=_FWHM,
            requires=("use_fwhm_edge_diameters",),
            advanced=True,
        ),
        Setting(
            name="fwhm_nonlocal_same_edge_arc_separation_um",
            kind="float",
            default=6.0,
            help="Treat same-edge points separated by more than this arc length as nonlocal",
            section=_FWHM,
            minimum=0.0,
            unit="um",
            requires=(
                "use_fwhm_edge_diameters",
                "fwhm_cap_half_extent_by_nonlocal_same_edge_distance",
            ),
            advanced=True,
        ),
        Setting(
            name="fwhm_nonlocal_same_edge_half_extent_factor",
            kind="float",
            default=0.45,
            help="Cap the profile half-extent at this fraction of the nearest nonlocal same-edge distance",
            section=_FWHM,
            minimum=0.0,
            requires=(
                "use_fwhm_edge_diameters",
                "fwhm_cap_half_extent_by_nonlocal_same_edge_distance",
            ),
            advanced=True,
        ),
        Setting(
            name="fwhm_reject_samples_with_center_offset",
            kind="bool",
            default=True,
            help="Discard samples whose fitted centre drifts too far from the centerline position",
            section=_FWHM,
            requires=("use_fwhm_edge_diameters",),
            advanced=True,
        ),
        Setting(
            name="fwhm_max_fit_center_offset_um",
            kind="float",
            default=1.5,
            help="Discard a sample once its fitted centre is further than this from the expected centre",
            section=_FWHM,
            minimum=0.0,
            unit="um",
            requires=(
                "use_fwhm_edge_diameters",
                "fwhm_reject_samples_with_center_offset",
            ),
            advanced=True,
        ),
        Setting(
            name="fwhm_reject_samples_with_low_fit_r2",
            kind="bool",
            default=True,
            help="Discard samples whose transverse Gaussian fit quality is too low",
            section=_FWHM,
            requires=("use_fwhm_edge_diameters",),
            advanced=True,
        ),
        Setting(
            name="fwhm_min_fit_r2",
            kind="float",
            default=0.85,
            help="Keep a sample only when its transverse Gaussian fit reaches this R-squared",
            section=_FWHM,
            minimum=0.0,
            maximum=1.0,
            requires=("use_fwhm_edge_diameters", "fwhm_reject_samples_with_low_fit_r2"),
            advanced=True,
        ),
        Setting(
            name="custom_edges",
            kind="any",
            default=[],
            help="Apply the custom edge-diameter assignment behaviour to these edge IDs",
            section=_DIAMETERS_AND_PERICYTES,
            advanced=True,
        ),
    ],
    title="Resistance network pipeline",
    description=(
        "Settings for the end-to-end microvascular pipeline: optional ilastik "
        "segmentation, skeletonization, graph building, boundary and branch-order "
        "assignment, haemodynamics, and export."
    ),
)


def default_schema() -> Schema:
    """The pipeline's settings, as a :class:`~haemolynx.parsers.Schema`.

    Every front-end starts here: :func:`resolve_settings` validates a config
    against it, the command line generates a flag per setting from it, and a
    GUI renders ``default_schema().describe()``. Extend it by building a new
    schema from its settings plus your own, which is what the whole-brain
    example does for its sweep::

        Schema(list(default_schema()) + MY_SETTINGS, title="...")
    """
    return SCHEMA


def write_default_config(
    config_path: Path | str,
    *,
    schema: Schema | None = None,
    values: Mapping[str, Any] | None = None,
) -> Path:
    """Write a commented YAML config for *schema* to *config_path*.

    The file is generated from the schema, so every setting arrives with its
    help text, unit, allowed values and prerequisites as comments -- it is the
    documented starting point for a run, and the answer to "what can I
    configure?" on an installed copy with no repository to read::

        write_default_config("my_config.yaml")

    *values* seeds the file with settings already chosen (an existing config,
    say), which is how regenerating picks up new settings without discarding
    what a user has set.
    """
    return dump_config(config_path, schema or default_schema(), values=values)
