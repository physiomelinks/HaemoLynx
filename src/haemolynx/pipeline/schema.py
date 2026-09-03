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
    "degree_1_from_inlet",
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
# Not "Perturbations": the YAML key of a section may not collide with a
# setting name, and `perturbations` is one of the settings in it.
_PERTURBATION_RUNS = "Perturbation runs"


SCHEMA = Schema(
    [
        # ------------------------------------------------------------------
        # Input and segmentation
        # ------------------------------------------------------------------
        # The napari Input tab hides gated rows (``input_path`` vs main-ilastik
        # children) until ``use_ilastik_segmentation`` applies. Shared
        # executable / output knobs stay visible: vessel-mask ilastik on
        # Boundaries also reads them, and schema ``requires`` cannot OR.
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
        # Vessel masks (Boundaries tab: under automated_vessel_assignment)
        # ------------------------------------------------------------------
        # Gate first: the Boundaries panel lists this checkbox, then the mask
        # rows below, then manual methods. Automated assignment overrides the
        # manual inlet/outlet methods when it is applied.
        Setting(
            name="automated_vessel_assignment",
            kind="bool",
            default=False,
            help=(
                "Select inlet and outlet nodes automatically from the large-vessel "
                "masks instead of manually. When applied, automated assignment "
                "overrides the other (manual) inlet/outlet selection methods"
            ),
            section=_VESSEL_MASKS,
        ),
        Setting(
            name="use_large_vessel_masks",
            kind="bool",
            default=False,
            help="Load large arteriole and venule masks for automated inlet/outlet node assignment",
            section=_VESSEL_MASKS,
            requires=("automated_vessel_assignment",),
        ),
        Setting(
            name="use_ilastik_large_vessel_segmentation",
            kind="bool",
            default=False,
            help="Produce the large-vessel masks with ilastik instead of reading pre-segmented files",
            section=_VESSEL_MASKS,
            requires=("use_large_vessel_masks", "automated_vessel_assignment"),
        ),
        Setting(
            name="large_vessel_mask_dilation_microns",
            kind="float",
            default=0.0,
            help=(
                "One-shot load-time dilation of large-vessel masks (microns) applied "
                "when the masks are loaded. Distinct from "
                "large_vessel_assignment_max_dilation_microns, which runs a progressive "
                "0, +5, … µm schedule only during terminal-node assignment"
            ),
            section=_VESSEL_MASKS,
            minimum=0.0,
            unit="um",
            requires=("use_large_vessel_masks", "automated_vessel_assignment"),
        ),
        Setting(
            name="large_vessel_assignment_max_dilation_microns",
            kind="float",
            default=0.0,
            help=(
                "Maximum dilation (microns) for progressive large-vessel terminal "
                "assignment: assign at 0 µm, then in 5 µm steps up to this value, "
                "locking each node at the first step it is claimed. Does not replace "
                "large_vessel_mask_dilation_microns (load-time one-shot dilation)"
            ),
            section=_VESSEL_MASKS,
            minimum=0.0,
            unit="um",
            requires=("use_large_vessel_masks", "automated_vessel_assignment"),
        ),
        Setting(
            name="large_vessel_min_component_volume_um3",
            kind="float",
            default=200.0,
            help=(
                "Remove connected components smaller than this physical volume from "
                "large-vessel masks after dilation (set 0 to disable)"
            ),
            section=_VESSEL_MASKS,
            minimum=0.0,
            unit="um3",
            requires=("use_large_vessel_masks", "automated_vessel_assignment"),
        ),
        Setting(
            name="large_vessel_remove_small_opposite_attached_components",
            kind="bool",
            default=True,
            help=(
                "Remove tiny large-vessel components that sit attached near the "
                "opposite-type mask surface (suppresses small mislabelled attachments)"
            ),
            section=_VESSEL_MASKS,
            requires=("use_large_vessel_masks", "automated_vessel_assignment"),
        ),
        Setting(
            name="large_vessel_opposite_attached_max_component_volume_um3",
            kind="float",
            default=250.0,
            help=(
                "Maximum physical volume of a large-vessel component eligible for "
                "opposite-attached cleanup"
            ),
            section=_VESSEL_MASKS,
            minimum=0.0,
            unit="um3",
            requires=(
                "use_large_vessel_masks",
                "automated_vessel_assignment",
                "large_vessel_remove_small_opposite_attached_components",
            ),
        ),
        Setting(
            name="large_vessel_opposite_attached_max_distance_microns",
            kind="float",
            default=3.0,
            help=(
                "Maximum distance from the opposite large-vessel mask for a tiny "
                "component to count as opposite-attached and be removed"
            ),
            section=_VESSEL_MASKS,
            minimum=0.0,
            unit="um",
            requires=(
                "use_large_vessel_masks",
                "automated_vessel_assignment",
                "large_vessel_remove_small_opposite_attached_components",
            ),
        ),
        Setting(
            name="exclude_smaller_overlapping_volumes",
            kind="bool",
            default=False,
            help=(
                "At load time, remove arteriole/venule overlap voxels from the "
                "smaller overlapping connected component. Assignment-time cleanup "
                "is controlled separately by "
                "automated_vessel_assignment_enable_overlap_cleanup / fast_mode "
                "(large) and small_vessel_boundary_assignment_* (small)"
            ),
            section=_VESSEL_MASKS,
            requires=("use_large_vessel_masks", "automated_vessel_assignment"),
        ),
        Setting(
            name="automated_vessel_assignment_enable_overlap_cleanup",
            kind="bool",
            default=True,
            help=(
                "Master switch for large-vessel overlap cleanup at assignment time. "
                "If False, cleanup is skipped even when fast mode is on"
            ),
            section=_VESSEL_MASKS,
            requires=("use_large_vessel_masks", "automated_vessel_assignment"),
        ),
        Setting(
            name="automated_vessel_assignment_fast_mode",
            kind="bool",
            default=True,
            help=(
                "Pre-clean large arteriole/venule overlap voxels from the smaller "
                "component once before progressive terminal assignment (when "
                "overlap cleanup is enabled)"
            ),
            section=_VESSEL_MASKS,
            requires=(
                "use_large_vessel_masks",
                "automated_vessel_assignment",
                "automated_vessel_assignment_enable_overlap_cleanup",
            ),
        ),
        Setting(
            name="automated_vessel_assignment_apply_overlap_cleanup_in_normal_mode",
            kind="bool",
            default=False,
            help=(
                "When fast mode is off, still apply overlap cleanup inside each "
                "progressive assignment step (requires overlap cleanup enabled)"
            ),
            section=_VESSEL_MASKS,
            requires=(
                "use_large_vessel_masks",
                "automated_vessel_assignment",
                "automated_vessel_assignment_enable_overlap_cleanup",
                "!automated_vessel_assignment_fast_mode",
            ),
        ),
        Setting(
            name="automated_vessel_overlap_parallel_workers",
            kind="int",
            default=8,
            help=(
                "Worker count for parallel overlap tie-break resolution during "
                "large-vessel assignment (0 or 1 = sequential)"
            ),
            section=_VESSEL_MASKS,
            minimum=0,
            requires=(
                "use_large_vessel_masks",
                "automated_vessel_assignment",
                "automated_vessel_assignment_enable_overlap_cleanup",
            ),
        ),
        Setting(
            name="automated_vessel_assignment_use_legacy_mode",
            kind="bool",
            default=True,
            help=(
                "Use legacy progressive large-vessel terminal assignment. Set "
                "False for confidence-based robust assignment with unresolved-node QC"
            ),
            section=_VESSEL_MASKS,
            requires=("use_large_vessel_masks", "automated_vessel_assignment"),
        ),
        Setting(
            name="automated_vessel_confidence_margin",
            kind="float",
            default=0.08,
            help="Minimum score-gap margin required to keep an automated I/O assignment",
            section=_VESSEL_MASKS,
            minimum=0.0,
            requires=(
                "use_large_vessel_masks",
                "automated_vessel_assignment",
                "!automated_vessel_assignment_use_legacy_mode",
            ),
        ),
        Setting(
            name="automated_vessel_min_confidence",
            kind="float",
            default=0.12,
            help="Minimum confidence required to keep an automated I/O assignment",
            section=_VESSEL_MASKS,
            minimum=0.0,
            requires=(
                "use_large_vessel_masks",
                "automated_vessel_assignment",
                "!automated_vessel_assignment_use_legacy_mode",
            ),
        ),
        Setting(
            name="automated_vessel_topology_penalty",
            kind="float",
            default=0.12,
            help="Topology-consistency penalty weight for implausible local assignments",
            section=_VESSEL_MASKS,
            minimum=0.0,
            requires=(
                "use_large_vessel_masks",
                "automated_vessel_assignment",
                "!automated_vessel_assignment_use_legacy_mode",
            ),
        ),
        Setting(
            name="automated_vessel_quality_max_overlap_fraction",
            kind="float",
            default=0.20,
            help=(
                "Quality gate: max arteriole/venule mask overlap fraction before "
                "conservative mode is auto-enabled"
            ),
            section=_VESSEL_MASKS,
            minimum=0.0,
            maximum=1.0,
            unit="fraction",
            requires=(
                "use_large_vessel_masks",
                "automated_vessel_assignment",
                "!automated_vessel_assignment_use_legacy_mode",
            ),
        ),
        Setting(
            name="automated_vessel_quality_min_terminal_coverage",
            kind="float",
            default=0.20,
            help="Quality gate: minimum terminal coverage by either large-vessel mask",
            section=_VESSEL_MASKS,
            minimum=0.0,
            maximum=1.0,
            unit="fraction",
            requires=(
                "use_large_vessel_masks",
                "automated_vessel_assignment",
                "!automated_vessel_assignment_use_legacy_mode",
            ),
        ),
        Setting(
            name="automated_vessel_quality_max_component_count",
            kind="int",
            default=12,
            help=(
                "Quality gate: maximum connected-component count allowed per "
                "large-vessel mask before conservative mode"
            ),
            section=_VESSEL_MASKS,
            minimum=1,
            requires=(
                "use_large_vessel_masks",
                "automated_vessel_assignment",
                "!automated_vessel_assignment_use_legacy_mode",
            ),
        ),
        Setting(
            name="automated_vessel_conservative_max_dilation_microns",
            kind="float",
            default=15.0,
            help="Conservative-mode cap for progressive dilation in robust assignment",
            section=_VESSEL_MASKS,
            minimum=0.0,
            unit="um",
            requires=(
                "use_large_vessel_masks",
                "automated_vessel_assignment",
                "!automated_vessel_assignment_use_legacy_mode",
            ),
        ),
        Setting(
            name="write_fast_mode_preassignment_large_vessel_debug_3d_html",
            kind="bool",
            default=False,
            help=(
                "Write before/after Plotly HTML diagnostics of large-vessel masks "
                "around fast-mode overlap cleanup"
            ),
            section=_VESSEL_MASKS,
            requires=(
                "use_large_vessel_masks",
                "automated_vessel_assignment",
                "automated_vessel_assignment_fast_mode",
            ),
        ),
        Setting(
            name="large_vessel_3d_volume_downsample_stride",
            kind="int",
            default=1,
            help=(
                "Block-max downsample stride for large/small vessel volume traces "
                "in the automated-assignment Plotly HTML view"
            ),
            section=_VESSEL_MASKS,
            minimum=1,
            requires=("use_large_vessel_masks", "automated_vessel_assignment"),
        ),
        Setting(
            name="cut_network_at_large_vessel_volumes",
            kind="bool",
            default=False,
            help=(
                "Before automated terminal assignment, cut the graph at large "
                "arteriole/venule mask boundaries: remove edges inside the "
                "volume and split crossing edges so new degree-1 terminals sit "
                "on the exterior side of the cut"
            ),
            section=_VESSEL_MASKS,
            requires=("use_large_vessel_masks", "automated_vessel_assignment"),
        ),
        Setting(
            name="remove_orphaned_branches_outside_large_vessel_volumes",
            kind="bool",
            default=False,
            help=(
                "After cutting at large-vessel volumes, remove connected "
                "components that remain outside the volume when their total "
                "edge count is below orphaned_branch_max_edge_count"
            ),
            section=_VESSEL_MASKS,
            requires=(
                "use_large_vessel_masks",
                "automated_vessel_assignment",
                "cut_network_at_large_vessel_volumes",
            ),
        ),
        Setting(
            name="orphaned_branch_max_edge_count",
            kind="int",
            default=3,
            help=(
                "Edge-count threshold for orphan cleanup after large-vessel "
                "volume cuts: drop a remaining exterior component when "
                "edge_count < this value"
            ),
            section=_VESSEL_MASKS,
            minimum=1,
            requires=(
                "use_large_vessel_masks",
                "automated_vessel_assignment",
                "cut_network_at_large_vessel_volumes",
                "remove_orphaned_branches_outside_large_vessel_volumes",
            ),
        ),
        Setting(
            name="remove_disconnected_io_components_after_final_assignment",
            kind="bool",
            default=False,
            help=(
                "After final I/O assignment, drop graph components that lack both "
                "an inlet and an outlet node"
            ),
            section=_VESSEL_MASKS,
            requires=("automated_vessel_assignment",),
        ),
        Setting(
            name="large_arteriole_mask_path",
            kind="path",
            default=f"{_IMAGES}/large_arteriole_mask.tif",
            help="Read this pre-segmented large arteriole mask",
            section=_VESSEL_MASKS,
            requires=(
                "use_large_vessel_masks",
                "automated_vessel_assignment",
                "!use_ilastik_large_vessel_segmentation",
            ),
            must_exist=True,
        ),
        Setting(
            name="large_venule_mask_path",
            kind="path",
            default=f"{_IMAGES}/large_venule_mask.tif",
            help="Read this pre-segmented large venule mask",
            section=_VESSEL_MASKS,
            requires=(
                "use_large_vessel_masks",
                "automated_vessel_assignment",
                "!use_ilastik_large_vessel_segmentation",
            ),
            must_exist=True,
        ),
        Setting(
            name="ilastik_unsegmented_arteriole_image_path",
            kind="path",
            default=f"{_IMAGES}/large_arteriole_mask.tif",
            help="Read this raw image as the ilastik input for the large arteriole mask",
            section=_VESSEL_MASKS,
            requires=(
                "use_large_vessel_masks",
                "automated_vessel_assignment",
                "use_ilastik_large_vessel_segmentation",
            ),
        ),
        Setting(
            name="ilastik_unsegmented_venule_image_path",
            kind="path",
            default=f"{_IMAGES}/large_venule_mask.tif",
            help="Read this raw image as the ilastik input for the large venule mask",
            section=_VESSEL_MASKS,
            requires=(
                "use_large_vessel_masks",
                "automated_vessel_assignment",
                "use_ilastik_large_vessel_segmentation",
            ),
        ),
        Setting(
            name="ilastik_arteriole_classifier_path",
            kind="path",
            default=f"{_CLASSIFIERS}/arteriole_classifier.ilp",
            help="Use this trained ilastik project to segment arterioles",
            section=_VESSEL_MASKS,
            requires=(
                "use_large_vessel_masks",
                "automated_vessel_assignment",
                "use_ilastik_large_vessel_segmentation",
            ),
            must_exist=True,
        ),
        Setting(
            name="ilastik_venule_classifier_path",
            kind="path",
            default=f"{_CLASSIFIERS}/venule_classifier.ilp",
            help="Use this trained ilastik project to segment venules",
            section=_VESSEL_MASKS,
            requires=(
                "use_large_vessel_masks",
                "automated_vessel_assignment",
                "use_ilastik_large_vessel_segmentation",
            ),
            must_exist=True,
        ),
        Setting(
            name="use_small_vessel_masks_for_boundary_assignment",
            kind="bool",
            default=False,
            help="Assign arteriole and venule boundary nodes from small-vessel masks",
            requires=("automated_vessel_assignment",),
            section=_VESSEL_MASKS,
        ),
        Setting(
            name="use_ilastik_small_vessel_segmentation",
            kind="bool",
            default=False,
            help="Produce the small-vessel masks with ilastik instead of reading pre-segmented files",
            section=_VESSEL_MASKS,
            requires=("use_small_vessel_masks_for_boundary_assignment", "automated_vessel_assignment"),
        ),
        Setting(
            name="small_vessel_mask_min_overlap_fraction",
            kind="float",
            default=0.5,
            help="Require at least this fraction of an edge to lie inside a small-vessel mask before labelling it",
            section=_VESSEL_MASKS,
            unit="fraction",
            minimum=0.0,
            maximum=1.0,
            requires=("use_small_vessel_masks_for_boundary_assignment", "automated_vessel_assignment"),
        ),
        Setting(
            name="small_vessel_mask_dilation_microns",
            kind="float",
            default=0.0,
            help=(
                "Maximum dilation (microns) for progressive small-vessel boundary "
                "assignment: label at 0 µm, then in 5 µm steps up to this value, "
                "locking each edge at the first step it is classified. There is no "
                "separate load-time small-vessel dilation setting"
            ),
            section=_VESSEL_MASKS,
            minimum=0.0,
            unit="um",
            requires=("use_small_vessel_masks_for_boundary_assignment", "automated_vessel_assignment"),
        ),
        Setting(
            name="small_vessel_boundary_assignment_enable_overlap_cleanup",
            kind="bool",
            default=True,
            help=(
                "Master switch for small-vessel overlap cleanup at assignment time. "
                "If False, cleanup is skipped even when fast mode is on"
            ),
            section=_VESSEL_MASKS,
            requires=("use_small_vessel_masks_for_boundary_assignment", "automated_vessel_assignment"),
        ),
        Setting(
            name="small_vessel_boundary_assignment_fast_mode",
            kind="bool",
            default=True,
            help=(
                "Pre-clean small arteriole/venule overlap voxels from the smaller "
                "component once before progressive boundary labelling (when "
                "overlap cleanup is enabled)"
            ),
            section=_VESSEL_MASKS,
            requires=(
                "use_small_vessel_masks_for_boundary_assignment",
                "automated_vessel_assignment",
                "small_vessel_boundary_assignment_enable_overlap_cleanup",
            ),
        ),
        Setting(
            name="small_vessel_boundary_assignment_apply_overlap_cleanup_in_normal_mode",
            kind="bool",
            default=False,
            help=(
                "When small-vessel fast mode is off, still apply overlap cleanup "
                "inside each progressive labelling step (requires cleanup enabled)"
            ),
            section=_VESSEL_MASKS,
            requires=(
                "use_small_vessel_masks_for_boundary_assignment",
                "automated_vessel_assignment",
                "small_vessel_boundary_assignment_enable_overlap_cleanup",
                "!small_vessel_boundary_assignment_fast_mode",
            ),
        ),
        Setting(
            name="small_vessel_overlap_parallel_workers",
            kind="int",
            default=8,
            help=(
                "Worker count for parallel edge classification during small-vessel "
                "boundary assignment (0 or 1 = sequential)"
            ),
            section=_VESSEL_MASKS,
            minimum=0,
            requires=(
                "use_small_vessel_masks_for_boundary_assignment",
                "automated_vessel_assignment",
                "small_vessel_boundary_assignment_enable_overlap_cleanup",
            ),
        ),
        Setting(
            name="small_vessel_mask_continuity_enable",
            kind="bool",
            default=False,
            help=(
                "Bridge gaps in small-vessel masks with type-locked cylinder links "
                "(same-type small↔small / small↔large) before boundary labelling"
            ),
            section=_VESSEL_MASKS,
            requires=("use_small_vessel_masks_for_boundary_assignment", "automated_vessel_assignment"),
        ),
        Setting(
            name="small_vessel_mask_continuity_allow_small_to_large",
            kind="bool",
            default=True,
            help="Allow continuity bridges from small components to large same-type masks",
            section=_VESSEL_MASKS,
            requires=(
                "use_small_vessel_masks_for_boundary_assignment",
                "automated_vessel_assignment",
                "small_vessel_mask_continuity_enable",
            ),
        ),
        Setting(
            name="small_vessel_mask_continuity_allow_small_to_small",
            kind="bool",
            default=True,
            help="Allow continuity bridges between small components of the same type",
            section=_VESSEL_MASKS,
            requires=(
                "use_small_vessel_masks_for_boundary_assignment",
                "automated_vessel_assignment",
                "small_vessel_mask_continuity_enable",
            ),
        ),
        Setting(
            name="small_vessel_mask_continuity_enforce_cylinder_only",
            kind="bool",
            default=True,
            help="Restrict continuity bridges to cylinder-to-cylinder links only",
            section=_VESSEL_MASKS,
            requires=(
                "use_small_vessel_masks_for_boundary_assignment",
                "automated_vessel_assignment",
                "small_vessel_mask_continuity_enable",
            ),
        ),
        Setting(
            name="small_vessel_mask_continuity_min_cylindricality",
            kind="float",
            default=0.45,
            help="Minimum cylindricality score required for a continuity bridge endpoint",
            section=_VESSEL_MASKS,
            minimum=0.0,
            maximum=1.0,
            requires=(
                "use_small_vessel_masks_for_boundary_assignment",
                "automated_vessel_assignment",
                "small_vessel_mask_continuity_enable",
            ),
        ),
        Setting(
            name="small_vessel_mask_continuity_max_axis_angle_degrees",
            kind="float",
            default=45.0,
            help="Maximum axis-angle (degrees) between bridged cylinder axes",
            section=_VESSEL_MASKS,
            minimum=0.0,
            requires=(
                "use_small_vessel_masks_for_boundary_assignment",
                "automated_vessel_assignment",
                "small_vessel_mask_continuity_enable",
            ),
        ),
        Setting(
            name="small_vessel_mask_continuity_min_facing_cosine",
            kind="float",
            default=0.82,
            help="Minimum facing-cosine between cylinder endpoints for a bridge",
            section=_VESSEL_MASKS,
            minimum=0.0,
            maximum=1.0,
            requires=(
                "use_small_vessel_masks_for_boundary_assignment",
                "automated_vessel_assignment",
                "small_vessel_mask_continuity_enable",
            ),
        ),
        Setting(
            name="small_vessel_mask_continuity_max_radius_ratio",
            kind="float",
            default=3.0,
            help="Maximum allowed radius ratio between bridged cylinder endpoints",
            section=_VESSEL_MASKS,
            minimum=1.0,
            requires=(
                "use_small_vessel_masks_for_boundary_assignment",
                "automated_vessel_assignment",
                "small_vessel_mask_continuity_enable",
            ),
        ),
        Setting(
            name="small_vessel_mask_continuity_max_bridge_distance_microns",
            kind="float",
            default=35.0,
            help="Maximum bridge length (microns) for continuity linking",
            section=_VESSEL_MASKS,
            minimum=0.0,
            unit="um",
            requires=(
                "use_small_vessel_masks_for_boundary_assignment",
                "automated_vessel_assignment",
                "small_vessel_mask_continuity_enable",
            ),
        ),
        Setting(
            name="small_vessel_mask_continuity_corridor_max_distance_microns",
            kind="float",
            default=12.0,
            help="Maximum corridor half-width (microns) around an accepted bridge",
            section=_VESSEL_MASKS,
            minimum=0.0,
            unit="um",
            requires=(
                "use_small_vessel_masks_for_boundary_assignment",
                "automated_vessel_assignment",
                "small_vessel_mask_continuity_enable",
            ),
        ),
        Setting(
            name="small_vessel_mask_continuity_opposite_exclusion_distance_microns",
            kind="float",
            default=3.0,
            help=(
                "Reject bridges that come within this distance of the opposite-type mask"
            ),
            section=_VESSEL_MASKS,
            minimum=0.0,
            unit="um",
            requires=(
                "use_small_vessel_masks_for_boundary_assignment",
                "automated_vessel_assignment",
                "small_vessel_mask_continuity_enable",
            ),
        ),
        Setting(
            name="small_vessel_tangential_redefinition_enable",
            kind="bool",
            default=False,
            help=(
                "Reassign small-vessel components that make tangential near-contact "
                "with a large-vessel mask of the opposite/same type"
            ),
            section=_VESSEL_MASKS,
            requires=("use_small_vessel_masks_for_boundary_assignment", "automated_vessel_assignment"),
        ),
        Setting(
            name="small_vessel_tangential_redefinition_max_contact_distance_microns",
            kind="float",
            default=12.0,
            help="Maximum contact distance (microns) for tangential redefinition",
            section=_VESSEL_MASKS,
            minimum=0.0,
            unit="um",
            requires=(
                "use_small_vessel_masks_for_boundary_assignment",
                "automated_vessel_assignment",
                "small_vessel_tangential_redefinition_enable",
            ),
        ),
        Setting(
            name="small_vessel_tangential_redefinition_touch_distance_microns",
            kind="float",
            default=3.0,
            help="Touch-distance threshold (microns) for tangential redefinition scoring",
            section=_VESSEL_MASKS,
            minimum=0.0,
            unit="um",
            requires=(
                "use_small_vessel_masks_for_boundary_assignment",
                "automated_vessel_assignment",
                "small_vessel_tangential_redefinition_enable",
            ),
        ),
        Setting(
            name="small_vessel_tangential_redefinition_tangency_cosine_max",
            kind="float",
            default=0.35,
            help="Maximum tangency cosine for a contact to count as tangential",
            section=_VESSEL_MASKS,
            minimum=0.0,
            maximum=1.0,
            requires=(
                "use_small_vessel_masks_for_boundary_assignment",
                "automated_vessel_assignment",
                "small_vessel_tangential_redefinition_enable",
            ),
        ),
        Setting(
            name="small_vessel_tangential_redefinition_margin",
            kind="float",
            default=0.10,
            help="Minimum score margin required to switch arteriole/venule class",
            section=_VESSEL_MASKS,
            minimum=0.0,
            requires=(
                "use_small_vessel_masks_for_boundary_assignment",
                "automated_vessel_assignment",
                "small_vessel_tangential_redefinition_enable",
            ),
        ),
        Setting(
            name="small_vessel_tangential_redefinition_parallel_workers",
            kind="int",
            default=8,
            help="Worker count for parallel tangential reassignment scoring",
            section=_VESSEL_MASKS,
            minimum=0,
            requires=(
                "use_small_vessel_masks_for_boundary_assignment",
                "automated_vessel_assignment",
                "small_vessel_tangential_redefinition_enable",
            ),
        ),
        Setting(
            name="use_gpu_mask_continuity_acceleration",
            kind="bool",
            default=False,
            help=(
                "Optional CuPy GPU acceleration for EDT-heavy continuity / "
                "tangential-redefinition steps (falls back to CPU if unavailable)"
            ),
            section=_VESSEL_MASKS,
            requires=("use_small_vessel_masks_for_boundary_assignment", "automated_vessel_assignment"),
        ),
        Setting(
            name="small_vessel_boundary_fallback_to_hop_distance",
            kind="bool",
            default=True,
            help=(
                "If small-vessel mask labelling misses seed-edge coverage from "
                "inlets/outlets, fall back to nodes at a fixed hop distance"
            ),
            section=_VESSEL_MASKS,
            requires=("use_small_vessel_masks_for_boundary_assignment", "automated_vessel_assignment"),
        ),
        Setting(
            name="small_vessel_boundary_fallback_hop_distance",
            kind="int",
            default=1,
            help="Hop distance used when small-vessel boundary hop-distance fallback is enabled",
            section=_VESSEL_MASKS,
            minimum=1,
            requires=(
                "use_small_vessel_masks_for_boundary_assignment",
                "automated_vessel_assignment",
                "small_vessel_boundary_fallback_to_hop_distance",
            ),
        ),
        Setting(
            name="small_vessel_min_component_volume_um3",
            kind="float",
            default=50.0,
            help=(
                "Remove connected components smaller than this physical volume from "
                "small-vessel masks before boundary assignment (set 0 to disable)"
            ),
            section=_VESSEL_MASKS,
            minimum=0.0,
            unit="um3",
            requires=("use_small_vessel_masks_for_boundary_assignment", "automated_vessel_assignment"),
        ),
        Setting(
            name="write_small_vessel_boundary_labelling_3d_html",
            kind="bool",
            default=True,
            help="Write an interactive 3D HTML diagnostic of the small-vessel boundary labelling",
            section=_VESSEL_MASKS,
            requires=("use_small_vessel_masks_for_boundary_assignment", "automated_vessel_assignment"),
        ),
        Setting(
            name="small_arteriole_mask_path",
            kind="path",
            default=f"{_IMAGES}/small_arteriole_mask.tif",
            help="Read this pre-segmented small arteriole mask",
            section=_VESSEL_MASKS,
            requires=(
                "use_small_vessel_masks_for_boundary_assignment",
                "automated_vessel_assignment",
                "!use_ilastik_small_vessel_segmentation",
            ),
            must_exist=True,
        ),
        Setting(
            name="small_venule_mask_path",
            kind="path",
            default=f"{_IMAGES}/small_venule_mask.tif",
            help="Read this pre-segmented small venule mask",
            section=_VESSEL_MASKS,
            requires=(
                "use_small_vessel_masks_for_boundary_assignment",
                "automated_vessel_assignment",
                "!use_ilastik_small_vessel_segmentation",
            ),
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
                "automated_vessel_assignment",
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
                "automated_vessel_assignment",
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
                "automated_vessel_assignment",
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
                "automated_vessel_assignment",
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
        # "edge_percent" is the default for the two that every run needs
        # because it is the only method that asks nothing of the dataset: it
        # takes the terminals in the first and last band of the network along
        # one axis, so an image nobody has looked at yet still gets inlets and
        # outlets. "coordinates" and "volume" describe one dataset and no
        # other, which is why they are chosen rather than defaulted to.
        # Greyed while automated_vessel_assignment is on: that path overrides
        # these manual inlet/outlet methods.
        Setting(
            name="inlet_node_selection_method",
            kind="choice",
            default="edge_percent",
            help="Choose how manual inlet nodes are picked from the graph",
            section=_BOUNDARY_ASSIGNMENT,
            choices=NODE_SELECTION_METHODS,
            requires=("!automated_vessel_assignment",),
        ),
        Setting(
            name="outlet_node_selection_method",
            kind="choice",
            default="edge_percent",
            help="Choose how manual outlet nodes are picked from the graph",
            section=_BOUNDARY_ASSIGNMENT,
            choices=NODE_SELECTION_METHODS,
            requires=("!automated_vessel_assignment",),
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
            name="boundary_distance_from_inlet_node",
            kind="float",
            default=0.0,
            help="Keep only the terminals further than this from an inlet node when the degree_1_from_inlet method is used",
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
            name="inlet_node_coordinates",
            kind="any",
            default=[],
            help="Pick inlet nodes nearest to these (z, y, x) coordinates when the coordinates method is used",
            section=_BOUNDARY_ASSIGNMENT,
            unit="um",
        ),
        Setting(
            name="outlet_node_coordinates",
            kind="any",
            default=[],
            help="Pick outlet nodes nearest to these (z, y, x) coordinates when the coordinates method is used",
            section=_BOUNDARY_ASSIGNMENT,
            unit="um",
        ),
        Setting(
            name="arteriole_boundary_node_coordinates",
            kind="any",
            default=[],
            help="Pick arteriole boundary nodes nearest to these (z, y, x) coordinates when the coordinates method is used",
            section=_BOUNDARY_ASSIGNMENT,
            unit="um",
        ),
        Setting(
            name="venule_boundary_node_coordinates",
            kind="any",
            default=[],
            help="Pick venule boundary nodes nearest to these (z, y, x) coordinates when the coordinates method is used",
            section=_BOUNDARY_ASSIGNMENT,
            unit="um",
        ),
        # The volume boxes below apply whenever a role's selection method is
        # "volume"; that choice is the switch, so there is no separate flag.
        Setting(
            name="inlet_node_volumes",
            kind="any",
            default=[],
            help="Select inlet nodes falling inside these (min corner, max corner) boxes, each corner (z, y, x)",
            section=_BOUNDARY_ASSIGNMENT,
            unit="um",
        ),
        Setting(
            name="outlet_node_volumes",
            kind="any",
            default=[],
            help="Select outlet nodes falling inside these (min corner, max corner) boxes, each corner (z, y, x)",
            section=_BOUNDARY_ASSIGNMENT,
            unit="um",
        ),
        Setting(
            name="arteriole_boundary_node_volumes",
            kind="any",
            default=[],
            help="Select arteriole boundary nodes falling inside these (min corner, max corner) boxes, each corner (z, y, x)",
            section=_BOUNDARY_ASSIGNMENT,
            unit="um",
        ),
        Setting(
            name="venule_boundary_node_volumes",
            kind="any",
            default=[],
            help="Select venule boundary nodes falling inside these (min corner, max corner) boxes, each corner (z, y, x)",
            section=_BOUNDARY_ASSIGNMENT,
            unit="um",
        ),
        Setting(
            # The pipeline fills these in place (`inlet_nodes[:] = []`), so
            # they must stay mutable lists rather than fixed-length tuples.
            name="inlet_nodes",
            kind="any",
            default=[],
            help="Hold the inlet node IDs chosen during the run; leave empty to let the pipeline fill it",
            section=_BOUNDARY_ASSIGNMENT,
            advanced=True,
        ),
        Setting(
            name="outlet_nodes",
            kind="any",
            default=[],
            help="Hold the outlet node IDs chosen during the run; leave empty to let the pipeline fill it",
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
            help=(
                "When small-arteriole/venule terminals were expected but are "
                "missing, raise; when they were never assigned, still label "
                "capillary B* orders from inlets and warn"
            ),
            section=_BOUNDARY_ASSIGNMENT,
        ),
        # ------------------------------------------------------------------
        # Solver and output
        # ------------------------------------------------------------------
        Setting(
            name="inlet_p_bc",
            kind="float",
            default=4500.0,
            help="Apply this pressure boundary condition at the inlet nodes",
            section=_SOLVER_AND_OUTPUT,
            minimum=0.0,
            unit="Pa",
        ),
        Setting(
            name="outlet_p_bc",
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
            name="show_flow_direction_layer",
            kind="bool",
            default=True,
            help=(
                "Add a napari Vectors layer of mid-edge arrows coloured by "
                "flow magnitude after the solve (Export tab results)"
            ),
            section=_SOLVER_AND_OUTPUT,
            requires=("run_haemodynamics",),
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
            unit="voxels",
            minimum=0,
            requires=("do_skeletonize",),
        ),
        Setting(
            name="skeleton_bridge_gap_size",
            kind="int",
            default=3,
            help="Bridge skeleton gaps no larger than this many voxels",
            section=_PIPELINE_STAGES,
            unit="voxels",
            minimum=0,
            requires=("do_skeletonize",),
        ),
        Setting(
            name="skeleton_min_branch_length",
            kind="int",
            default=3,
            help="Drop skeleton branches shorter than this many voxels during cleaning",
            section=_PIPELINE_STAGES,
            unit="voxels",
            minimum=0,
            requires=("do_skeletonize",),
        ),
        Setting(
            name="skeleton_max_bridge_distance",
            kind="int",
            default=4,
            help="Reconnect skeleton fragments no further apart than this distance",
            section=_PIPELINE_STAGES,
            unit="voxels",
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
            unit="um",
            minimum=0.0,
            requires=("do_graph_building",),
        ),
        Setting(
            name="final_orphan_reconnect_threshold",
            kind="float",
            default=3.0,
            help="Reconnect leftover orphan and dangling nodes within this distance in the final pass",
            section=_PIPELINE_STAGES,
            unit="um",
            minimum=0.0,
            requires=("do_graph_building",),
        ),
        Setting(
            name="min_stub_length",
            kind="float",
            default=10.0,
            help="Prune terminal stubs shorter than this length",
            section=_PIPELINE_STAGES,
            unit="um",
            minimum=0.0,
            requires=("do_graph_building",),
        ),
        Setting(
            name="cluster_collapse_distance",
            kind="float",
            default=5.0,
            help="Collapse clusters of nodes lying within this distance of each other",
            section=_PIPELINE_STAGES,
            unit="um",
            minimum=0.0,
            requires=("do_graph_building",),
        ),
        Setting(
            name="save_step_artifacts",
            kind="bool",
            default=False,
            help=(
                "Write a graph pickle and an overlay PNG after each graph-building "
                "step. Turn on to debug a bad topology; it dominates the runtime of "
                "a large run, so it is off by default"
            ),
            section=_PIPELINE_STAGES,
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
            unit="um",
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
            name="viscosity_law",
            kind="choice",
            default="pries",
            choices=["pries", "capillary_power_law", "constant"],
            help=(
                "Which apparent-viscosity law sets the resistances: pries "
                "covers 3.3-1978 um in one expression and reads "
                "diameter_basis, capillary_power_law is the older law "
                "calibrated at 5 um with a placeholder constant above 7 um, "
                "constant is plasma everywhere. Resistances are not comparable "
                "across laws"
            ),
            section=_DIAMETERS_AND_PERICYTES,
            requires=("run_haemodynamics",),
        ),
        Setting(
            name="diameter_basis",
            kind="choice",
            default="plasma_column",
            choices=["plasma_column", "anatomical"],
            help=(
                "What a vessel diameter in this dataset measures, which the "
                "pries law needs and no other reads: plasma_column for the "
                "channel the fluid occupies, as a plasma stain images it, or "
                "anatomical for a wall-to-wall diameter that includes the "
                "~1.1 um endothelial surface layer. Choosing anatomical for a "
                "plasma-stained image subtracts that layer twice and roughly "
                "quintuples capillary resistance"
            ),
            section=_DIAMETERS_AND_PERICYTES,
            requires=("run_haemodynamics",),
        ),
        Setting(
            name="haematocrit",
            kind="float",
            default=0.45,
            help=(
                "Discharge haematocrit the pries law is evaluated at; read by "
                "no other law"
            ),
            section=_DIAMETERS_AND_PERICYTES,
            unit="fraction",
            minimum=0.0,
            maximum=0.99,
            requires=("run_haemodynamics",),
        ),
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
            help=(
                "Legacy flag kept for older configs and CLI; ignored by baseline "
                "Haemodynamics and every perturbation merge (not shown as a "
                "typed-entry control). Focal constrictions come only from "
                "pericyte-typed perturbations via their strategy path"
            ),
            section=_DIAMETERS_AND_PERICYTES,
            requires=("run_haemodynamics",),
        ),
        Setting(
            name="constriction_length_um",
            kind="float",
            default=40.0,
            help=(
                "Axial length of each focal pericyte constriction/dilation "
                "site along the vessel centreline"
            ),
            section=_DIAMETERS_AND_PERICYTES,
            minimum=0.0,
            unit="um",
        ),
        Setting(
            name="constriction_spacing_um",
            kind="float",
            default=100.0,
            help=(
                "Centre-to-centre distance between regularly spaced pericyte "
                "sites when not placing from a mask"
            ),
            section=_DIAMETERS_AND_PERICYTES,
            minimum=0.0,
            unit="um",
        ),
        Setting(
            name="use_pericyte_mask_constriction",
            kind="bool",
            default=False,
            help=(
                "Place focal constrictions/dilations at pericyte centroids "
                "from a mask instead of at regular spacing along vessels"
            ),
            section=_DIAMETERS_AND_PERICYTES,
            requires=("run_haemodynamics",),
        ),
        Setting(
            name="pericyte_mask_path",
            kind="path",
            default=None,
            help=(
                "Read pericyte centroids from this binary or labelled mask "
                "when mask-based placement is on; leave unset to keep the "
                "path from the Diameters tab"
            ),
            section=_DIAMETERS_AND_PERICYTES,
            requires=("use_pericyte_mask_constriction",),
            must_exist=True,
        ),
        Setting(
            name="pericyte_mask_h5_dataset_name",
            kind="str",
            default=None,
            help=(
                "H5 dataset name inside the pericyte mask file; ignored for "
                "TIFF masks"
            ),
            section=_DIAMETERS_AND_PERICYTES,
            requires=("use_pericyte_mask_constriction",),
        ),
        Setting(
            name="pericyte_max_assignment_distance_um",
            kind="float",
            default=3.0,
            help=(
                "Assign a pericyte centroid to the nearest vessel edge only "
                "when they lie within this distance"
            ),
            section=_DIAMETERS_AND_PERICYTES,
            minimum=0.0,
            unit="um",
            requires=("use_pericyte_mask_constriction",),
        ),
        Setting(
            name="pericyte_min_diameter_um",
            kind="float",
            default=5.0,
            help=(
                "Ignore mask components smaller than this equivalent diameter "
                "as too small to be a pericyte"
            ),
            section=_DIAMETERS_AND_PERICYTES,
            minimum=0.0,
            unit="um",
            requires=("use_pericyte_mask_constriction",),
        ),
        Setting(
            name="pericyte_max_diameter_um",
            kind="float",
            default=12.0,
            help=(
                "Ignore mask components larger than this equivalent diameter "
                "as too large to be a pericyte"
            ),
            section=_DIAMETERS_AND_PERICYTES,
            minimum=0.0,
            unit="um",
            requires=("use_pericyte_mask_constriction",),
        ),
        Setting(
            name="use_probabilistic_pericyte_constriction",
            kind="bool",
            default=False,
            help=(
                "Activate each candidate pericyte site at random rather than "
                "constricting/dilating every candidate"
            ),
            section=_DIAMETERS_AND_PERICYTES,
            requires=("run_haemodynamics",),
        ),
        Setting(
            name="pericyte_constriction_probability",
            kind="float",
            default=0.8,
            help=(
                "Fraction of candidate pericyte sites that activate when "
                "probabilistic selection is on (0 = none, 1 = all)"
            ),
            section=_DIAMETERS_AND_PERICYTES,
            unit="fraction",
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
            help=(
                "Legacy comparison-CSV flag; ignored by baseline assign_diameters "
                "and perturbation merges (CLI / library apply() only). Use "
                "Perturbations for pericyte tone"
            ),
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
            requires=("!all_diams_const",),
        ),
        Setting(
            name="manual_arteriole_diameter_by_branch_order",
            kind="mapping",
            default={},
            help="Override arteriole diameters per branch-order label, e.g. Art1",
            section=_DIAMETERS_AND_PERICYTES,
            unit="um",
            requires=("!all_diams_const",),
        ),
        Setting(
            name="manual_venule_diameter_by_branch_order",
            kind="mapping",
            default={},
            help="Override venule diameters per branch-order label, e.g. Ven1",
            section=_DIAMETERS_AND_PERICYTES,
            unit="um",
            requires=("!all_diams_const",),
        ),
        Setting(
            name="diameter_by_branch_order",
            kind="mapping",
            default=None,
            help="Supply the full branch-order diameter lookup; leave unset to derive it from the manual diameter settings",
            section=_DIAMETERS_AND_PERICYTES,
            unit="um",
            advanced=True,
            requires=("!all_diams_const",),
        ),
        Setting(
            name="pericyte_constriction_factor",
            kind="float",
            default=1.0,
            help=(
                "Global focal constriction/dilation factor for every branch "
                "order (1.0 = no local change; <1 narrows, >1 widens at "
                "pericyte sites). constriction_by_branch_order replaces this "
                "for listed orders only — empty map means this global factor "
                "alone"
            ),
            section=_DIAMETERS_AND_PERICYTES,
            minimum=0.0,
            requires=("run_haemodynamics",),
        ),
        Setting(
            name="constriction_by_branch_order",
            kind="mapping",
            default=None,
            help=(
                "Sparse per-order overrides of pericyte_constriction_factor "
                "(<1 constricts, >1 dilates locally). Empty or unset makes no "
                "change (global factor only). A listed order replaces the "
                "global factor for that order alone (e.g. {B01: 1.0} with "
                "global 0.8 leaves B01 unchanged)"
            ),
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
            unit="fraction",
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
            unit="fraction",
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
            unit="fraction",
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
            unit="fraction",
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
            unit="fraction",
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
        # ------------------------------------------------------------------
        # Perturbation runs
        #
        # What to re-solve the finished network for. The napari panel shows
        # only `run_perturbations` / `perturbations` / `perturbation_output_dir`
        # as ordinary rows; sweep ranges and pericyte/constriction knobs are
        # options of a typed perturbation entry (see SETTINGS_FOR_TYPE), not
        # always-on tab settings. `run_pericyte_dilation_sweep` and
        # `sweep_output_dir` remain for the whole-brain example script, which
        # still runs the combined dilation×pressure sweep as a post-step.
        # A prerequisite can only be a bool, so whether a sweep runs is
        # decided by a perturbation's type (or by the brain script's flag).
        # ------------------------------------------------------------------
        Setting(
            name="run_perturbations",
            kind="bool",
            default=False,
            help="Re-solve the finished network once per configured perturbation",
            section=_PERTURBATION_RUNS,
        ),
        Setting(
            name="perturbations",
            kind="any",
            default=[],
            help=(
                "Re-solve the network once for each of these, as a list of "
                "{name, type, overrides} entries; each runs from the same "
                "baseline rather than on top of the one before"
            ),
            section=_PERTURBATION_RUNS,
        ),
        Setting(
            name="perturbation_output_dir",
            kind="path",
            default=None,
            help=(
                "Write each perturbation's CSVs here; leave unset for a "
                "'perturbations' directory beside the other output"
            ),
            section=_PERTURBATION_RUNS,
            requires=("run_perturbations",),
        ),
        Setting(
            name="run_pericyte_dilation_sweep",
            kind="bool",
            default=False,
            help=(
                "After the pipeline, sweep pericyte constriction/dilation "
                "against inlet pressure (whole-brain example script only; "
                "prefer a pericyte_dilation_sweep perturbation in the panel)"
            ),
            section=_PERTURBATION_RUNS,
        ),
        Setting(
            name="pericyte_dilation_min_percent",
            kind="int",
            default=1,
            help=(
                "Start the pericyte constriction/dilation sweep at this "
                "percentage (negative = constriction, positive = dilation; "
                "scale = 1 + percent/100)"
            ),
            section=_PERTURBATION_RUNS,
            unit="percent",
            minimum=-99,
            maximum=100,
        ),
        Setting(
            name="pericyte_dilation_max_percent",
            kind="int",
            default=30,
            help=(
                "End the pericyte constriction/dilation sweep at this "
                "percentage (negative = constriction, positive = dilation)"
            ),
            section=_PERTURBATION_RUNS,
            unit="percent",
            minimum=-99,
            maximum=100,
        ),
        Setting(
            name="pericyte_dilation_step_percent",
            kind="int",
            default=1,
            help=(
                "Step size along the pericyte constriction/dilation percent "
                "axis (positive step; percent itself may be negative)"
            ),
            section=_PERTURBATION_RUNS,
            unit="percent",
            minimum=1,
            maximum=100,
        ),
        Setting(
            name="arteriole_diameter_change_percent",
            kind="float",
            default=0.0,
            help=(
                "Arteriole constriction/dilation: change every arteriole "
                "branch's diameter by this percentage (positive = dilation, "
                "negative = constriction; e.g. 10 widens by 10%, -20 narrows "
                "by 20%); whole-branch scaling, not a focal pericyte site"
            ),
            section=_PERTURBATION_RUNS,
            unit="percent",
            # Scale = 1 + percent/100 must stay > 0.
            minimum=-99.999,
        ),
        Setting(
            name="arteriole_dilation_min_percent",
            kind="int",
            default=0,
            help=(
                "Start the arteriole constriction/dilation sweep at this "
                "percentage change (negative = constriction, positive = "
                "dilation; whole-branch scale = 1 + percent/100)"
            ),
            section=_PERTURBATION_RUNS,
            unit="percent",
            minimum=-99,
            maximum=100,
        ),
        Setting(
            name="arteriole_dilation_max_percent",
            kind="int",
            default=30,
            help=(
                "End the arteriole constriction/dilation sweep at this "
                "percentage change (negative = constriction, positive = "
                "dilation)"
            ),
            section=_PERTURBATION_RUNS,
            unit="percent",
            minimum=-99,
            maximum=100,
        ),
        Setting(
            name="arteriole_dilation_step_percent",
            kind="int",
            default=1,
            help=(
                "Step size along the arteriole constriction/dilation percent "
                "axis (positive step; percent itself may be negative)"
            ),
            section=_PERTURBATION_RUNS,
            unit="percent",
            minimum=1,
            maximum=100,
        ),
        Setting(
            name="capillary_dilation_min_percent",
            kind="int",
            default=0,
            help=(
                "Start the passive capillary constriction/dilation sweep at "
                "this percentage change (whole capillary, not focal; "
                "negative = constriction, positive = dilation; scale = 1 + "
                "percent/100)"
            ),
            section=_PERTURBATION_RUNS,
            unit="percent",
            minimum=-99,
            maximum=100,
        ),
        Setting(
            name="capillary_dilation_max_percent",
            kind="int",
            default=30,
            help=(
                "End the passive capillary constriction/dilation sweep at "
                "this percentage change (whole capillary, not focal; "
                "negative = constriction, positive = dilation)"
            ),
            section=_PERTURBATION_RUNS,
            unit="percent",
            minimum=-99,
            maximum=100,
        ),
        Setting(
            name="capillary_dilation_step_percent",
            kind="int",
            default=1,
            help=(
                "Step size along the capillary constriction/dilation percent "
                "axis (positive step; percent itself may be negative)"
            ),
            section=_PERTURBATION_RUNS,
            unit="percent",
            minimum=1,
            maximum=100,
        ),
        Setting(
            name="pericyte_geometry_dilation_percent",
            kind="int",
            default=0,
            help=(
                "Fixed whole-network constriction/dilation for a pericyte "
                "spacing or length sweep (scale = 1 + percent/100; negative "
                "= constriction, positive = dilation); pressure stays at "
                "inlet_p_bc"
            ),
            section=_PERTURBATION_RUNS,
            unit="percent",
            minimum=-99,
            maximum=100,
        ),
        Setting(
            name="constriction_spacing_min_um",
            kind="float",
            default=50.0,
            help=(
                "Start the pericyte spacing sweep at this centre-to-centre "
                "inter-site distance"
            ),
            section=_PERTURBATION_RUNS,
            unit="um",
            minimum=0.0,
        ),
        Setting(
            name="constriction_spacing_max_um",
            kind="float",
            default=150.0,
            help=(
                "End the pericyte spacing sweep at this centre-to-centre "
                "inter-site distance"
            ),
            section=_PERTURBATION_RUNS,
            unit="um",
            minimum=0.0,
        ),
        Setting(
            name="constriction_spacing_step_um",
            kind="float",
            default=50.0,
            help=(
                "Step size along the pericyte spacing sweep (centre-to-centre "
                "distance between sites)"
            ),
            section=_PERTURBATION_RUNS,
            unit="um",
            minimum=0.0,
        ),
        Setting(
            name="constriction_length_min_um",
            kind="float",
            default=20.0,
            help=(
                "Start the pericyte length sweep at this axial constriction/"
                "dilation site length"
            ),
            section=_PERTURBATION_RUNS,
            unit="um",
            minimum=0.0,
        ),
        Setting(
            name="constriction_length_max_um",
            kind="float",
            default=60.0,
            help=(
                "End the pericyte length sweep at this axial constriction/"
                "dilation site length"
            ),
            section=_PERTURBATION_RUNS,
            unit="um",
            minimum=0.0,
        ),
        Setting(
            name="constriction_length_step_um",
            kind="float",
            default=20.0,
            help=(
                "Step size along the pericyte length sweep (axial site length "
                "along the centreline)"
            ),
            section=_PERTURBATION_RUNS,
            unit="um",
            minimum=0.0,
        ),
        Setting(
            name="inlet_pressure_min_pa",
            kind="int",
            default=4500,
            help=(
                "Start the inlet-pressure sweep at this absolute inlet "
                "boundary pressure"
            ),
            section=_PERTURBATION_RUNS,
            unit="Pa",
            minimum=0,
        ),
        Setting(
            name="inlet_pressure_max_pa",
            kind="int",
            default=6000,
            help=(
                "End the inlet-pressure sweep at this absolute inlet "
                "boundary pressure"
            ),
            section=_PERTURBATION_RUNS,
            unit="Pa",
            minimum=0,
        ),
        Setting(
            name="inlet_pressure_step_pa",
            kind="int",
            default=500,
            help=(
                "Step size along the inlet-pressure sweep (absolute inlet "
                "boundary pressure)"
            ),
            section=_PERTURBATION_RUNS,
            unit="Pa",
            minimum=1,
        ),
        Setting(
            name="sweep_output_dir",
            kind="path",
            default=f"{_OUTPUTS}/pericyte_dilation_sweep",
            help=(
                "Write the brain-script sweep CSV and curves here "
                "(a pericyte_dilation_sweep perturbation writes beside itself)"
            ),
            section=_PERTURBATION_RUNS,
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
