"""The image-to-model pipeline, one function per stage.

Each stage takes the settings dict and whatever the earlier stages produced,
and returns a small dataclass holding what the later ones need. Running them in
order is what :func:`run_pipeline_stages` does; calling them individually is how
you intervene in the middle of a run.

    segment              raw image -> segmented mask (ilastik, or pass through)
    skeletonise          mask -> skeleton, with the voxel size resolved
    build_network        skeleton + vessel masks -> graph
    assign_boundaries    graph -> inlet, outlet and vessel-boundary nodes
    assign_diameters     graph -> branch orders and a diameter per edge
    build_haemodynamic_model  diameters -> resistance and conductance per edge
    solve                conductance -> pressures, flows, equivalent resistance
    run_perturbations    the same baseline, re-solved once per perturbation
    export_results       VTK, statistics, distances and plots

A run says where it has got to through :mod:`haemolynx.pipeline.progress`: pass
``progress=`` a callback and it is handed one event per stage boundary, and one
per topology step inside graph building.
"""
from __future__ import annotations

import csv
import inspect
import json
import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import tifffile

from haemolynx import graph, haemodynamics, io, preprocessing, statistics, visualization
from haemolynx.haemodynamics.apply import (
    HaemodynamicsApplyConfig,
    apply_poiseuille_haemodynamics,
)
from haemolynx.haemodynamics.constriction_strategy import (
    set_resistances_for_constriction_strategy,
)
from haemolynx.haemodynamics.perturbations import (
    PerturbationSpec,
    is_sweep_perturbation,
    perturbation_output_dir,
    perturbation_problems,
    perturbations_from_settings,
)
from haemolynx.haemodynamics.pericyte_geometry_sweep import (
    run_pericyte_geometry_sweep,
)
from haemolynx.haemodynamics.capillary import (
    run_capillary_dilation_pressure_sweep,
)
from haemolynx.haemodynamics.pericyte_sweep import (
    run_arteriole_dilation_pressure_sweep,
    run_pericyte_dilation_pressure_sweep,
    solve_pressure_and_boundary_flow,
)
from haemolynx.haemodynamics.perturbations import plain as plain_values
from haemolynx.haemodynamics.poiseuille import PoiseuilleModel
from haemolynx.io.voxel_validation import resolve_voxel_size_xyz
from haemolynx.visualization.perturbation_plots import (
    export_non_sweep_perturbation_artifacts,
    export_sweep_perturbation_plots,
)
from haemolynx.parsers import Schema, parameters_of, prefixed_arguments
from haemolynx.pipeline.progress import ProgressCallback, RunProgress, StageProgress

#: Called with each stage's name and the object that stage returned, so
#: something watching a run can look at its work as it happens rather than
#: waiting for the files at the end. The napari panel turns each one into
#: layers; a script could pickle them, or count them, or ignore them.
StageOutputCallback = Callable[[str, Any], None]

#: Prefix on the name given to a graph-building step, so a consumer can tell
#: one from a stage: `topology_step:prune_vascular_stubs`.
TOPOLOGY_STEP = "topology_step:"

logger = logging.getLogger(__name__)


@dataclass
class SegmentedInputs:
    """What segmentation settled: which image to analyse, and where output goes."""

    image_path: Path
    output_dir: Path
    #: "tif"/"tiff"/"h5" — which loader skeletonise should use.
    input_format: str = "tif"


@dataclass
class SkeletonisedVolume:
    """The loaded volume, its skeleton, and the voxel size they are measured in."""

    image: np.ndarray
    skeleton: np.ndarray
    voxel_size_xyz: tuple[float, float, float]
    voxel_size_zyx: tuple[float, float, float]
    output_dir: Path


@dataclass
class VesselNetwork:
    """The graph, the volume it came from, and the vessel masks alongside it."""

    graph: nx.MultiGraph
    volume: SkeletonisedVolume
    large_arteriole_mask: np.ndarray | None = None
    large_venule_mask: np.ndarray | None = None
    small_arteriole_mask: np.ndarray | None = None
    small_venule_mask: np.ndarray | None = None


@dataclass
class BoundaryNodes:
    """Where flow enters and leaves, and where vessel types change."""

    inlet_nodes: list[int] = field(default_factory=list)
    outlet_nodes: list[int] = field(default_factory=list)
    arteriole_boundary_nodes: list[int] = field(default_factory=list)
    venule_boundary_nodes: list[int] = field(default_factory=list)
    resistance_node_pair: tuple[int, int] | None = None


@dataclass
class HaemodynamicModel:
    """A graph carrying branch orders, diameters, resistances and conductances."""

    graph: nx.MultiGraph
    results: dict[str, Any] = field(default_factory=dict)


@dataclass
class Solution:
    """What the solve produced: pressures, flows, equivalent resistance."""

    pressure: np.ndarray | None = None
    node_list: list[int] = field(default_factory=list)
    equivalent_resistance: float | None = None
    statistics: dict[str, Any] = field(default_factory=dict)


@dataclass
class PerturbationResult:
    """One perturbation: the network it produced, and where it wrote it.

    *graph* is that perturbation's own copy of the baseline, carrying its
    resistances, pressures and flows -- so the panel can draw it beside the
    baseline rather than instead of it. A `none` perturbation, and one that
    raised, have no graph and no output.
    """

    name: str
    type: str
    graph: nx.MultiGraph | None = None
    output_dir: Path | None = None
    #: Files this perturbation wrote, in the order it wrote them.
    outputs: list[Path] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    #: What went wrong, if it did. One bad perturbation does not stop the rest.
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass
class PerturbationRun:
    """What the perturbations stage did: one result per configured entry."""

    #: The directory the per-perturbation directories are in, if any ran.
    output_dir: Path | None = None
    results: list[PerturbationResult] = field(default_factory=list)
    #: The baseline every perturbation was differenced against.
    baseline: dict[str, Any] = field(default_factory=dict)

    @property
    def solved(self) -> list[PerturbationResult]:
        """Those that produced a network: not `none`, and not a failure."""
        return [result for result in self.results if result.graph is not None]

    @property
    def failures(self) -> list[PerturbationResult]:
        return [result for result in self.results if result.error is not None]


def segment(settings: dict):
    """Produce the segmented mask to analyse, running ilastik when asked to."""
    # Not yet a path when ilastik is doing the segmenting: `input_path` is what
    # this stage *produces* in that case, and the documented way to ask for it
    # is to leave the setting empty. Converting first turned that into
    # `Path(None)` -- a TypeError before ilastik was even reached, on the one
    # configuration the setting exists to support (#127).
    if settings["input_path"] is not None:
        settings["input_path"] = Path(settings["input_path"])
    if settings["use_ilastik_segmentation"]:
        unsegmented_image_path = Path(settings["ilastik_unsegmented_image_path"])
        unsegmented_image_path = io.resolve_image_path_with_optional_zip(unsegmented_image_path)
        if settings["ilastik_classifier_path"] is None:
            raise ValueError(
                "ilastik_classifier_path must be set when use_ilastik_segmentation=True."
            )
        settings["ilastik_output_dir"] = Path(settings["ilastik_output_dir"])
        ilastik_segmented_path = settings["ilastik_output_dir"] / (
            f"{unsegmented_image_path.stem}_segmented{settings['ilastik_output_suffix']}"
        )
        logger.info(f"Running ilastik segmentation for unsegmented image: {unsegmented_image_path}")
        settings["input_path"] = io.run_ilastik_headless_segmentation(
            input_image_path=unsegmented_image_path,
            classifier_path=Path(settings["ilastik_classifier_path"]),
            output_path=ilastik_segmented_path,
            ilastik_executable=settings["ilastik_executable"],
        )
        logger.info(f"Using ilastik-segmented image: {settings['input_path']}")
    else:
        if settings["input_path"] is None:
            raise ValueError(
                "input_path must name a segmented image, or "
                "use_ilastik_segmentation must be on so that one is produced. "
                "Neither is set, so there is nothing to analyse."
            )
        logger.info(f"Using segmented input image: {settings['input_path']}")

    settings["input_path"] = io.resolve_image_path_with_optional_zip(settings["input_path"])
    # get image format from image_path
    input_format = settings["input_path"].suffix[1:].lower()
    if input_format not in ["tif", "tiff", "h5"]:
        raise ValueError(f"Invalid image format: {input_format}")
    settings["image_axis_order"] = io.normalize_axis_order(settings["image_axis_order"], label="IMAGE_AXIS_ORDER")
    if settings["image_axis_order"] != io.CANONICAL_AXIS_ORDER:
        logger.info(
            f"Input axis order '{settings['image_axis_order']}' will be transposed to the canonical "
            f"'{io.CANONICAL_AXIS_ORDER}' layout on load."
        )
    settings["vtk_output_prefix"] = Path(settings["vtk_output_prefix"])
    output_dir = settings["vtk_output_prefix"].parent
    valid_final_render_modes = {"2d", "3d"}
    if settings["final_render_mode"] not in valid_final_render_modes:
        raise ValueError(
            f"Invalid final_render_mode='{settings['final_render_mode']}'. "
            f"Choose one of {sorted(valid_final_render_modes)}."
        )

    return SegmentedInputs(
        image_path=settings["input_path"],
        output_dir=output_dir,
        input_format=input_format,
    )


def skeletonise(settings: dict, inputs: SegmentedInputs):
    """Load the mask, resolve its voxel size, and skeletonise it."""
    output_dir = inputs.output_dir
    input_format = inputs.input_format
    # 1) Load image and skeletonize.
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
    skeleton_path = output_dir / f"{settings['input_path'].stem}_skeleton.npy"
    voxel_meta_path = output_dir / f"{settings['input_path'].stem}_voxel_size.json"
    graph_path = output_dir / f"{settings['input_path'].stem}_graph.pkl"
    projection_path = settings["plot_dir"] / "skeleton_projection.png"
    if not settings["plot_dir"].exists():
        settings["plot_dir"].mkdir(parents=True, exist_ok=True)

    if settings["do_skeletonize"]:
        if input_format in {"tif", "tiff"}:
            (
                image,
                skeleton,
                voxel_size_x,
                voxel_size_y,
                voxel_size_z,
                voxel_meta_status,
            ) = io.load_and_skeletonize_3d_tif(
                settings["input_path"],
                axis_order=settings["image_axis_order"],
            )
            metadata_voxel_size = (
                float(voxel_size_x),
                float(voxel_size_y),
                float(voxel_size_z),
            )
        elif input_format == "h5":
            (
                image,
                skeleton,
                voxel_size_x,
                voxel_size_y,
                voxel_size_z,
                voxel_meta_status,
            ) = io.load_and_skeletonize_3d_h5(
                settings["input_path"],
                axis_order=settings["image_axis_order"],
            )
            metadata_voxel_size = (
                float(voxel_size_x),
                float(voxel_size_y),
                float(voxel_size_z),
            )
        else:
            raise ValueError("INPUT_FORMAT must be 'tif', 'tiff', or 'h5'.")
        voxel_size, voxel_size_source = resolve_voxel_size_xyz(
            metadata_voxel_size_xyz=metadata_voxel_size,
            metadata_status=voxel_meta_status,
            voxel_size_override_xyz=settings["voxel_size_override_xyz"],
            voxel_size_policy=settings["voxel_size_policy"],
        )
        logger.info(
            "Voxel-size resolution: "
            f"source={voxel_size_source}, "
            f"metadata_status={voxel_meta_status.get('status')}, "
            f"metadata={metadata_voxel_size}, "
            f"final={voxel_size}"
        )
        
        preprocessing.log_skeleton_connectivity_stats(
            "raw",
            skeleton,
            component_connectivity=settings["skeleton_component_connectivity"],
        )
        visualization.visualize_skeleton(skeleton, save_path=settings["plot_dir"] / "raw_skeleton.png")

        # The `skeleton_*` settings are this function's parameters with a
        # prefix, so they go in as a group; the percentage is the one exception.
        skeleton = preprocessing.preprocess_skeleton_for_graph(
            skeleton,
            **prefixed_arguments(
                settings,
                "skeleton_",
                parameters_of(preprocessing.preprocess_skeleton_for_graph),
            ),
            min_component_fraction=settings["skeleton_min_component_percent"] / 100.0,
        )
        preprocessing.log_skeleton_connectivity_stats(
            "cleaned",
            skeleton,
            component_connectivity=settings["skeleton_component_connectivity"],
        )
        
        # save the skeleton
        np.save(skeleton_path, skeleton)
        voxel_meta_path.write_text(
            json.dumps(
                {
                    "voxel_size": voxel_size,
                    "voxel_size_source": voxel_size_source,
                    "voxel_metadata_status": voxel_meta_status,
                    "voxel_size_policy": settings["voxel_size_policy"],
                    "voxel_size_override_xyz": settings["voxel_size_override_xyz"],
                }
            )
        )
        logger.info(f"Saved skeleton to: {skeleton_path}")
    else:
        # load the skeleton
        skeleton = np.load(skeleton_path)
        image = io.apply_axis_order(tifffile.imread(settings["input_path"]), settings["image_axis_order"])
        if voxel_meta_path.exists():
            cached_voxel_meta = json.loads(voxel_meta_path.read_text())
            metadata_voxel_size = tuple(cached_voxel_meta["voxel_size"])
            voxel_meta_status = cached_voxel_meta.get(
                "voxel_metadata_status",
                {"source": "cache", "status": "unknown"},
            )
        else:
            metadata_voxel_size = (1.0, 1.0, 1.0)
            voxel_meta_status = {"source": "none", "status": "missing"}
        voxel_size, voxel_size_source = resolve_voxel_size_xyz(
            metadata_voxel_size_xyz=metadata_voxel_size,
            metadata_status=voxel_meta_status,
            voxel_size_override_xyz=settings["voxel_size_override_xyz"],
            voxel_size_policy=settings["voxel_size_policy"],
        )
        logger.info(f"Loaded skeleton from: {skeleton_path}")
        logger.info(
            "Voxel-size resolution (from cache/default): "
            f"source={voxel_size_source}, "
            f"metadata_status={voxel_meta_status.get('status')}, "
            f"final={voxel_size}"
        )

    logger.info("Visualizing skeleton projection...")
    visualization.visualize_skeleton(skeleton, save_path=projection_path)
    logger.info("Skeleton projection saved.")

    main_voxel_size_xyz = tuple(float(v) for v in voxel_size)
    # Image metadata reports (x, y, z); array axes are canonical (z, y, x).
    # Everything downstream that scales array indices uses voxel_size_zyx.
    return SkeletonisedVolume(
        image=image,
        skeleton=skeleton,
        voxel_size_xyz=main_voxel_size_xyz,
        voxel_size_zyx=io.voxel_size_zyx_from_xyz(main_voxel_size_xyz),
        output_dir=output_dir,
    )


def build_network(
    settings: dict,
    volume: SkeletonisedVolume,
    schema: Schema,
    progress: StageProgress | None = None,
    on_step_graph: Callable[[str, Any], None] | None = None,
):
    """Load the vessel masks and turn the skeleton into a graph.

    This is the long stage, so it reports the eleven topology steps of
    :func:`graph.build_graph_from_skeleton` to *progress* as they land -- a
    run's only finer-grained progress than "graph building is happening".
    """
    image, skeleton = volume.image, volume.skeleton
    output_dir = volume.output_dir
    main_voxel_size_xyz = volume.voxel_size_xyz
    voxel_size_zyx = volume.voxel_size_zyx
    graph_path = output_dir / f"{settings['input_path'].stem}_graph.pkl"

    # The vessel-mask and segmentation settings go in as a group; each role
    # picks the ones it uses. `io.VESSEL_MASK_SETTINGS` lists which those are.
    mask_settings = {
        **schema.section_values(settings, "Vessel masks"),
        **schema.section_values(settings, "Input and segmentation"),
    }
    (
        large_arteriole_mask,
        large_venule_mask,
        large_arteriole_mask_voxel_size,
        large_venule_mask_voxel_size,
    ) = io.load_and_validate_vessel_masks(
        **io.vessel_mask_arguments(mask_settings, "large"),
        image_shape=image.shape,
        main_voxel_size_xyz=main_voxel_size_xyz,
    )
    (
        small_arteriole_mask,
        small_venule_mask,
        small_arteriole_mask_voxel_size,
        small_venule_mask_voxel_size,
    ) = io.load_and_validate_vessel_masks(
        **io.vessel_mask_arguments(mask_settings, "small"),
        image_shape=image.shape,
        main_voxel_size_xyz=main_voxel_size_xyz,
        loaded_message_suffix=(
            f"min_overlap_fraction={float(settings['small_vessel_mask_min_overlap_fraction']):.3f}"
        ),
    )

    if settings["do_graph_building"]:
        # 3) Convert skeleton to graph.
        # Every snapshot draws the same volume, and projecting it reads the whole
        # stack, so it is projected once here rather than once per step. Graph
        # building reads `image` and never writes to it, which is what makes one
        # projection good for all eleven steps.
        step_projection = (
            visualization.overlay_z_projection(image)
            if settings["save_step_artifacts"]
            else None
        )

        def _graph_build_step_callback(graph_obj, label: str) -> None:
            # Report before drawing: the snapshots below are the slow part of
            # this step, so a watcher should see it tick over on arrival.
            if progress is not None:
                progress.step(label, total=len(graph.STEP_LABELS))
            # The graph as it stands, for anyone drawing the repair as it
            # happens. It is mid-repair and will change again, which is why it
            # goes out here rather than being kept.
            if on_step_graph is not None:
                on_step_graph(label, graph_obj)
            if not settings["save_step_artifacts"]:
                return
            # `graph_after_<label>.png` and `<label>.png` are the same figure
            # under two names, so it is drawn once and written twice.
            plot_png = label
            if label == "smart_multigraph_degree2_removal_pass1":
                plot_png = "smart_multigraph_degree2_removal"
            visualization.save_graph_snapshot(
                graph_obj,
                image,
                output_dir,
                settings["plot_dir"],
                settings["input_path"].stem,
                label,
                projection=step_projection,
                extra_plot_names=(plot_png,),
            )

        G = graph.build_graph_from_skeleton(
            skeleton,
            voxel_size=voxel_size_zyx,
            graph_reconnect_threshold=settings["graph_reconnect_threshold"],
            final_orphan_reconnect_threshold=settings["final_orphan_reconnect_threshold"],
            cluster_collapse_distance=settings["cluster_collapse_distance"],
            min_stub_length=settings["min_stub_length"],
            debug=settings["verbose_logging"],
            step_callback=_graph_build_step_callback,
        )

        # Last thing before the graph is saved: take the voxel staircase out of
        # each centreline. A skeleton path steps voxel to voxel, so a vessel at
        # an angle comes back 7% longer than it is, and resistance follows
        # length -- so this is a measurement fix, not a cosmetic one.
        if settings["smooth_centrelines"]:
            graph.smooth_graph_centrelines(
                G,
                skeleton,
                voxel_size_zyx=voxel_size_zyx,
                method=settings["centreline_smoothing_method"],
                iterations=settings["centreline_smoothing_iterations"],
                max_deviation=settings["centreline_max_deviation"],
            )

        with graph_path.open("wb") as f:
            pickle.dump(G, f)
        logger.info(f"Saved graph to: {graph_path}")
    else:
        if not graph_path.exists():
            raise FileNotFoundError(
                f"Graph file not found at {graph_path}. "
                "Set DO_GRAPH_BUILDING=True to generate it first."
            )
        with graph_path.open("rb") as f:
            G = pickle.load(f)
        logger.info(f"Loaded graph from: {graph_path}")

    # Store physical voxel-unit metadata used for skeleton/graph geometry and mask alignment.
    G.graph["image_voxel_size_xyz"] = main_voxel_size_xyz
    G.graph["image_voxel_size_zyx"] = voxel_size_zyx
    if large_arteriole_mask is not None and large_venule_mask is not None:
        G.graph["large_arteriole_mask_voxel_size_xyz"] = tuple(
            float(v) for v in large_arteriole_mask_voxel_size
        )
        G.graph["large_venule_mask_voxel_size_xyz"] = tuple(
            float(v) for v in large_venule_mask_voxel_size
        )
    
    # Visualize final graph used for boundary-node verification.
    if settings["final_render_mode"] == "3d":
        final_graph_3d_path = settings["plot_dir"] / "final_graph_3d.html"
        visualization.visualize_3d_plotly(
            G,
            title="Final Graph (Interactive 3D)",
            save_html_path=str(final_graph_3d_path),
            show=settings["show_plots_in_ide"] or settings["interactive_plots"],
        )
        logger.info(f"Saved interactive 3D final graph to: {final_graph_3d_path}")
    else:
        visualization.visualize_edges_and_nodes(
            image,
            G,
            label_nodes=False,
            save_path=settings["plot_dir"] / "final_graph.png",
            show_coordinates_degree_1=True,
        )


    return VesselNetwork(
        graph=G,
        volume=volume,
        large_arteriole_mask=large_arteriole_mask,
        large_venule_mask=large_venule_mask,
        small_arteriole_mask=small_arteriole_mask,
        small_venule_mask=small_venule_mask,
    )


def assign_boundaries(settings: dict, network: VesselNetwork):
    """Choose the inlet, outlet and vessel-boundary nodes for this network."""
    G = network.graph
    image = network.volume.image
    voxel_size_zyx = network.volume.voxel_size_zyx
    large_arteriole_mask = network.large_arteriole_mask
    large_venule_mask = network.large_venule_mask
    small_arteriole_mask = network.small_arteriole_mask
    small_venule_mask = network.small_venule_mask
    auto_inlet_nodes: list[int] = []
    auto_outlet_nodes: list[int] = []
    if settings["automated_vessel_assignment"]:
        if large_arteriole_mask is None or large_venule_mask is None:
            raise ValueError(
                "automated_vessel_assignment=True requires arteriole and venule masks. "
                "Set use_large_vessel_masks=True and provide mask paths."
            )
        auto_inlet_nodes, auto_outlet_nodes = (
            graph.select_terminal_nodes_from_large_vessel_masks(
                G,
                large_arteriole_mask=large_arteriole_mask,
                large_venule_mask=large_venule_mask,
                voxel_size_zyx=voxel_size_zyx,
                allow_overlap=False,
            )
        )
        if not auto_inlet_nodes:
            raise ValueError(
                "automated_vessel_assignment=True found no terminal nodes in the "
                "arteriole mask (after any configured dilation)."
            )
        if not auto_outlet_nodes:
            raise ValueError(
                "automated_vessel_assignment=True found no terminal nodes in the "
                "venule mask (after any configured dilation)."
            )
        settings["inlet_node_coordinates"] = [
            tuple(np.asarray(G.nodes[node_id]["pos"], dtype=float))
            for node_id in auto_inlet_nodes
        ]
        settings["outlet_node_coordinates"] = [
            tuple(np.asarray(G.nodes[node_id]["pos"], dtype=float))
            for node_id in auto_outlet_nodes
        ]
        automated_assignment_html_path = settings["plot_dir"] / "automated_vessel_assignment_3d.html"
        wrote_assignment_html = graph.write_automated_vessel_assignment_3d_html(
            G,
            large_arteriole_mask=large_arteriole_mask,
            large_venule_mask=large_venule_mask,
            input_nodes=auto_inlet_nodes,
            outlet_nodes=auto_outlet_nodes,
            voxel_size_zyx=voxel_size_zyx,
            output_html_path=automated_assignment_html_path,
        )
        if wrote_assignment_html:
            logger.info(
                "Saved automated vessel-assignment 3D visualization to: "
                f"{automated_assignment_html_path}"
            )
        else:
            logger.warning(
                "Skipped automated vessel-assignment 3D visualization "
                "(plotly is not installed)."
            )
        logger.info(
            "Automated vessel assignment selected "
            f"{len(settings['inlet_node_coordinates'])} input coordinates from arteriole-mask overlap "
            f"and {len(settings['outlet_node_coordinates'])} outlet coordinates from venule-mask overlap."
        )

    settings["inlet_nodes"][:] = []
    settings["outlet_nodes"][:] = []
    settings["arteriole_boundary_nodes"][:] = []
    settings["venule_boundary_nodes"][:] = []
    if settings["automated_vessel_assignment"]:
        # Use direct terminal-node overlap assignment from vessel masks.
        inlet_nodes = auto_inlet_nodes
        outlet_nodes = [node_id for node_id in auto_outlet_nodes if node_id not in set(inlet_nodes)]
    else:
        # Each role reads its own three settings; naming the role is enough.
        inlet_nodes = graph.select_boundary_nodes_for_role(
            G, image.shape, settings, "inlet"
        )
        outlet_nodes = graph.select_boundary_nodes_for_role(
            G, image.shape, settings, "outlet", exclude_nodes=inlet_nodes
        )
    settings["inlet_nodes"].extend(inlet_nodes)
    settings["outlet_nodes"].extend(outlet_nodes)
    used_nodes = set(settings["inlet_nodes"]) | set(settings["outlet_nodes"])
    if settings["arteriole_boundary_node_coordinates"] or settings["arteriole_boundary_node_volumes"]:
        art_boundary = graph.select_boundary_nodes_for_role(
            G, image.shape, settings, "arteriole_boundary", exclude_nodes=list(used_nodes)
        )
        settings["arteriole_boundary_nodes"].extend(art_boundary)
        used_nodes.update(settings["arteriole_boundary_nodes"])

    if settings["venule_boundary_node_coordinates"] or settings["venule_boundary_node_volumes"]:
        ven_boundary = graph.select_boundary_nodes_for_role(
            G, image.shape, settings, "venule_boundary", exclude_nodes=list(used_nodes)
        )
        settings["venule_boundary_nodes"].extend(ven_boundary)
    if settings["use_small_vessel_masks_for_boundary_assignment"]:
        if small_arteriole_mask is None or small_venule_mask is None:
            raise ValueError(
                "use_small_vessel_masks_for_boundary_assignment=True requires "
                "small_arteriole_mask_path and small_venule_mask_path."
            )
        inferred_boundary_results = graph.infer_boundary_nodes_from_small_vessel_masks(
            G,
            small_arteriole_mask=small_arteriole_mask,
            small_venule_mask=small_venule_mask,
            voxel_size_zyx=voxel_size_zyx,
            minimum_overlap_fraction=float(settings["small_vessel_mask_min_overlap_fraction"]),
            allow_overlap=False,
        )
        settings["arteriole_boundary_nodes"][:] = list(
            inferred_boundary_results["arteriole_boundary_nodes"]
        )
        settings["venule_boundary_nodes"][:] = list(inferred_boundary_results["venule_boundary_nodes"])
        logger.info(
            "Small-vessel mask boundary assignment selected "
            f"{len(settings['arteriole_boundary_nodes'])} arteriole boundary nodes and "
            f"{len(settings['venule_boundary_nodes'])} venule boundary nodes "
            f"(min_overlap_fraction={float(settings['small_vessel_mask_min_overlap_fraction']):.3f})."
        )
        logger.info(
            "Small-vessel mask edge labels: "
            f"arteriole_edges={inferred_boundary_results['arteriole_edge_count']}, "
            f"venule_edges={inferred_boundary_results['venule_edge_count']}, "
            f"overlap_edges={inferred_boundary_results['overlap_edge_count']}."
        )
        if settings["write_small_vessel_boundary_labelling_3d_html"]:
            boundary_html = Path(settings["plot_dir"]) / "small_vessel_mask_boundary_labelling_3d.html"
            Path(settings["plot_dir"]).mkdir(parents=True, exist_ok=True)
            ok = graph.write_small_vessel_mask_boundary_labelling_3d_html(
                G,
                small_arteriole_mask=small_arteriole_mask,
                small_venule_mask=small_venule_mask,
                arteriole_boundary_nodes=settings["arteriole_boundary_nodes"],
                venule_boundary_nodes=settings["venule_boundary_nodes"],
                voxel_size_zyx=voxel_size_zyx,
                output_html_path=boundary_html,
            )
            if ok:
                logger.info(f"Saved interactive 3D small-vessel boundary view: {boundary_html}")
            else:
                logger.warning(
                    "Small-vessel boundary 3D HTML not written (install plotly to enable)."
                )
    if settings["automated_vessel_assignment"]:
        logger.info(
            f"Selected {len(settings['inlet_nodes'])} STARTING_NODES and {len(settings['outlet_nodes'])} "
            "OUTPUT_NODES directly from terminal-node overlap with vessel masks."
        )
    else:
        logger.info(
            f"Selected {len(settings['inlet_nodes'])} STARTING_NODES "
            f"({settings['inlet_node_selection_method']}) and "
            f"{len(settings['outlet_nodes'])} OUTPUT_NODES "
            f"({settings['outlet_node_selection_method']})."
        )
    logger.info(f"Inlet nodes are: {settings['inlet_nodes']}")
    logger.info(f"Outlet nodes are: {settings['outlet_nodes']}")
    logger.info(f"Arteriole boundary nodes are: {settings['arteriole_boundary_nodes']}")
    logger.info(f"Venule boundary nodes are: {settings['venule_boundary_nodes']}")

    if settings["inlet_nodes"] and settings["outlet_nodes"]:
        resistance_node_pair = (settings["inlet_nodes"][0], settings["outlet_nodes"][0])
        logger.info(f"Auto-selected resistance node pair: {resistance_node_pair}")
    else:
        if settings["automated_vessel_assignment"]:
            raise ValueError(
                "No inlet or outlet nodes found from terminal-node overlap with "
                "arteriole/venule masks."
            )
        # Name the settings and what each of them found: the graph, not the
        # config, is the usual culprit by the time the run gets here, and the
        # counts are what say which of the two to look at.
        terminal_count = sum(1 for _, degree in G.degree() if degree == 1)
        raise ValueError(
            "No inlet or outlet nodes found: "
            f"inlet_node_selection_method="
            f"{settings['inlet_node_selection_method']!r} selected "
            f"{len(settings['inlet_nodes'])} inlet(s) and "
            f"outlet_node_selection_method="
            f"{settings['outlet_node_selection_method']!r} selected "
            f"{len(settings['outlet_nodes'])} outlet(s), from the "
            f"{terminal_count} terminal node(s) in the graph. Fix: change those "
            "two settings or the values they read, or check that the graph has "
            "terminals at both ends of boundary_axis."
        )


    return BoundaryNodes(
        inlet_nodes=settings["inlet_nodes"],
        outlet_nodes=settings["outlet_nodes"],
        arteriole_boundary_nodes=settings["arteriole_boundary_nodes"],
        venule_boundary_nodes=settings["venule_boundary_nodes"],
        resistance_node_pair=resistance_node_pair,
    )


def assign_diameters(settings: dict, network: VesselNetwork, boundaries: BoundaryNodes, schema: Schema):
    """Assign branch orders, then the diameter each edge is modelled with.\n\n    Branch orders come first because they are the key into the diameter\n    table; per-edge FWHM measurements override that table when enabled."""
    G = network.graph
    image = network.volume.image
    output_dir = network.volume.output_dir
    voxel_size_zyx = network.volume.voxel_size_zyx
    resistance_node_pair = boundaries.resistance_node_pair
    # 4) Add branch orders and hemodynamic edge weights.
    if settings["inlet_nodes"]:

        def _vessel_types_after_branch_assign(graph_obj) -> None:
            vessel_type_3d_path = settings["plot_dir"] / "vessel_types_assigned_3d.html"
            visualization.visualize_3d_plotly_vessel_types(
                graph_obj,
                title="Assigned Vessel Types (Interactive 3D)",
                save_html_path=str(vessel_type_3d_path),
                show=False,
            )
            logger.info(
                "Saved vessel-type 3D visualization after branch assignment to: "
                f"{vessel_type_3d_path}"
            )

        branch_summary = graph.assign_vessel_branch_orders(
            G,
            settings["inlet_nodes"],
            outlet_nodes=settings["outlet_nodes"],
            arteriole_boundary_nodes=settings["arteriole_boundary_nodes"],
            venule_boundary_nodes=settings["venule_boundary_nodes"],
            strict_hierarchical=settings["strict_branch_order_assignment"],
            expects_hierarchical=bool(
                settings["automated_vessel_assignment"]
                or settings["use_small_vessel_masks_for_boundary_assignment"]
            ),
            post_assign_callback=_vessel_types_after_branch_assign,
        )
        if branch_summary["mode"] == "hierarchical":
            logger.info(
                "Assigned hierarchical branch orders "
                "(Art*/Ven* first, then capillary B* from arteriole boundary)."
            )
            logger.info(f"Branch assignment summary: {branch_summary}")
        elif branch_summary["mode"] == "capillary":
            logger.info(
                "Assigned capillary branch orders from STARTING_NODES only "
                "(no arteriole/venule boundary-node sets supplied)."
            )

        if not settings["run_haemodynamics"]:
            logger.info(
                "Haemodynamics disabled; skipping diameter fitting and "
                "Poiseuille conductance assignment."
            )
        elif settings["run_haemodynamics"]:
            # Two config sections go in whole rather than as forty-odd
            # keyword arguments; everything else here is computed by this run.
            haemo_config = HaemodynamicsApplyConfig(
                diameters=schema.section_values(settings, "Diameters and pericytes"),
                fwhm=schema.section_values(settings, "FWHM diameter measurement"),
                resistance_node_pair=resistance_node_pair,
                voxel_size_zyx=voxel_size_zyx,
                axis_order=settings["image_axis_order"],
                comparison_output_csv_path=(
                    output_dir / f"{settings['input_path'].stem}_pericyte_resistance_comparison.csv"
                    if settings["run_pericyte_resistance_comparison"]
                    else None
                ),
            )
            G, haemo_results = apply_poiseuille_haemodynamics(G, config=haemo_config)
            if "fwhm" in haemo_results:
                logger.info(f"FWHM diameter measurement summary: {haemo_results['fwhm']}")
                if settings["do_pericyte_construction"]:
                    logger.info(
                        "Pericyte mode: passive diameter d1 from per-edge FWHM where available, "
                        "else DIAMETER_BY_BRANCH_ORDER; d2 = d1 * CONSTRICTION_BY_BRANCH_ORDER."
                    )
            elif settings["use_fwhm_edge_diameters"] is False:
                logger.info(
                    "Vessel diameters: manual mode (DIAMETER_BY_BRANCH_ORDER / "
                    "set_poiseuille_resistances without per-edge FWHM)."
                )
            if "pericyte_comparison" in haemo_results:
                comparison_results = haemo_results["pericyte_comparison"]
                logger.info(
                    "Pericyte resistance comparison complete: "
                    f"baseline={comparison_results['baseline_resistance']:.6f}, "
                    f"constricted={comparison_results['constricted_resistance']:.6f}, "
                    f"delta={comparison_results['delta']:.6f}, "
                    f"change={comparison_results['percent_change']:.3f}%."
                )
                logger.info(
                    "Saved pericyte resistance comparison CSV to: "
                    f"{comparison_results['output_csv_path']}"
                )
            # `weights` has not been a key since resistance and conductance
            # replaced the overloaded `weight` attribute; the summary calls it
            # `resistances`, and this loop had been logging nothing.
            for step_name, step_result in haemo_results.get("resistances", {}).items():
                logger.info(f"Haemodynamics resistances [{step_name}]: {step_result}")
            if "viscosity" in haemo_results:
                logger.info(f"Viscosity law: {haemo_results['viscosity']}")


    return HaemodynamicModel(graph=G, results=locals().get("haemo_results", {}) or {})


def build_haemodynamic_model(settings: dict, model: HaemodynamicModel):
    """Return the model ready to solve.\n\n    `assign_diameters` already applied Poiseuille's law to every edge, so\n    this reports what it produced rather than recomputing it."""
    if settings["run_haemodynamics"]:
        edges_with_resistance = sum(
            1 for _, _, data in model.graph.edges(data=True) if "resistance" in data
        )
        logger.info(
            f"Haemodynamic model: {edges_with_resistance} of "
            f"{model.graph.number_of_edges()} edges carry a resistance"
        )
    return model


def solve(settings: dict, model: HaemodynamicModel, boundaries: BoundaryNodes):
    """Solve the network: equivalent resistance, then pressures and edge flows."""
    G = model.graph
    resistance_node_pair = boundaries.resistance_node_pair
    solution = Solution()
    # 6) Compute effective resistance between two selected nodes.
    if settings["run_haemodynamics"]:
        conductance, node_list = haemodynamics.build_conductance_matrix_from_graph(G)
        node_to_idx = {node_id: idx for idx, node_id in enumerate(node_list)}
        logger.info(f"Conductance matrix built with shape {conductance.shape} and node_list length {len(node_list)}.")
    if settings["run_haemodynamics"] and settings["do_equiv_resistance_calculation"]:
        source_node, target_node = resistance_node_pair
        if source_node in node_to_idx and target_node in node_to_idx:
            laplacian = haemodynamics.calc_laplacian_from_conductance_matrix(conductance)
            solution.equivalent_resistance = haemodynamics.calc_two_point_from_laplacian_matrix_nodeID(
                laplacian,
                G,
                source_node,
                target_node,
            )
            logger.info(
                f"Effective resistance between nodes {source_node} and "
                f"{target_node}: {solution.equivalent_resistance}"
            )
        else:
            logger.warning(
                f"Skipped two-point resistance: nodes {resistance_node_pair} "
                "are not both present in the graph."
            )

    # 9) Also solve for flow throughout the network using the conductance matrix 
    # and the input and output pressures.
    if settings["run_haemodynamics"]:
        logger.info("Solving flow through the network...")
        flow = haemodynamics.solve_flow_from_conductance_matrix(
            conductance,
            node_list,
            inlet_p_bc=settings["inlet_p_bc"],
            outlet_p_bc=settings["outlet_p_bc"],
            inlet_nodes=settings["inlet_nodes"],
            outlet_nodes=settings["outlet_nodes"],
        )
        haemodynamics.set_edge_flows(G, node_list, flow["pressure"])
        logger.info("Flow through the network solved")
    else:
        logger.info("Haemodynamics solve skipped (run_haemodynamics=False).")


    if settings["run_haemodynamics"]:
        solution.pressure = flow["pressure"]
        solution.node_list = list(node_list)
    return solution


#: Per-edge columns of a perturbation's `<name>_edges.csv`, in order.
PERTURBATION_EDGE_COLUMNS = (
    "u",
    "v",
    "key",
    "branch_order",
    "length_um",
    "diameter_um",
    "resistance",
    "conductance",
    "pressure_drop",
    "flow_signed",
    "baseline_resistance",
    "resistance_ratio",
)

#: Columns of a perturbation's `<name>_summary.csv`, in order.
PERTURBATION_SUMMARY_COLUMNS = (
    "name",
    "type",
    "overrides",
    "inlet_pressure_pa",
    "outlet_pressure_pa",
    "total_inlet_flow",
    "total_outlet_flow",
    "flow_balance_error",
    "equivalent_resistance",
    "baseline_equivalent_resistance",
    "delta_vs_baseline",
    "percent_change_vs_baseline",
)


def _poiseuille_model_for(settings: dict) -> PoiseuilleModel:
    """The blood model a perturbation solves with: the run's, unchanged.

    Read from *settings* rather than defaulted so that a perturbation and the
    baseline it is compared with agree on the viscosity law -- which is also
    why a perturbation may not override one (see `INCOMPARABLE_OVERRIDES`).
    """
    return PoiseuilleModel(
        constriction_length=float(settings["constriction_length_um"]),
        constriction_spacing=float(settings["constriction_spacing_um"]),
        viscosity_law=settings["viscosity_law"],
        haematocrit=float(settings["haematocrit"]),
        diameter_basis=settings["diameter_basis"],
    )


def _perturbation_copy(G: nx.MultiGraph) -> nx.MultiGraph:
    """The graph a perturbation is allowed to write on, without the deep copy.

    `nx.MultiGraph.copy` gives every node and every edge a **new attribute
    dict** whose values are the originals: the copy's `voxels` list and `pos`
    array are the very same objects the baseline holds, but a `data["x"] = ...`
    on the copy cannot reach them. `G.graph` is shallow-copied the same way.

    That is enough here because every write a perturbation makes is a rebind
    of a whole attribute:

    * `poiseuille.set_edge_resistance` -- `resistance`, `conductance`
    * `resistance.set_edge_flows` -- `pressure` on nodes; `pressure_u`,
      `pressure_v`, `pressure_drop`, `flow_signed`, `flow_abs` on edges
    * `arteriole.scale_arteriole_diameters`,
      `capillary.scale_capillary_diameters` and
      `pericyte_sweep.dilate_graph_diameters` -- `fwhm_diameter_um`
    * `constriction.apply_constriction_sites` -- `pericyte_count_assigned`,
      and `pericyte_centers_um` rebound to a freshly built list

    Nothing reachable from a perturbation mutates a shared value in place, and
    the geometry (`voxels`, `pos`) is only ever read. Deep-copying it instead
    costs ~10x the time and ~2x the peak memory of this, which on a 200k-edge
    whole-brain graph is minutes of a run spent duplicating arrays nobody
    writes to.

    So: **anything added to a perturbation path that mutates an attribute
    value rather than replacing it -- `voxels.append(...)`,
    `pos[0] = ...`, `data["pericyte_centers_um"].append(...)`,
    `G.graph["some_dict"]["k"] = ...` -- would edit the baseline and every
    other perturbation through this copy.** Rebind instead. The guard tests in
    `tests/test_perturbation_stage.py` snapshot the baseline's `voxels` and
    `pos` across a run of every perturbation type and fail if one moves.
    """
    return G.copy()


def _solve_network(
    G: nx.MultiGraph, settings: dict, boundaries: BoundaryNodes
) -> dict[str, Any]:
    """Pressures, boundary flows and equivalent resistance, on *G* as it is.

    The full re-solve every perturbation gets: the pressures land on the nodes
    and the flows on the edges, so the perturbed network can be drawn and
    exported the same way the baseline is.
    """
    conductance, node_list = haemodynamics.build_conductance_matrix_from_graph(G)
    solved = solve_pressure_and_boundary_flow(
        conductance,
        list(node_list),
        inlet_p_bc=float(settings["inlet_p_bc"]),
        outlet_p_bc=float(settings["outlet_p_bc"]),
        inlet_nodes=list(boundaries.inlet_nodes),
        outlet_nodes=list(boundaries.outlet_nodes),
    )
    haemodynamics.set_edge_flows(G, list(node_list), solved["pressure"])
    return solved


def _write_perturbation_csvs(
    result: PerturbationResult,
    *,
    baseline_graph: nx.MultiGraph,
    solved: dict[str, Any],
    baseline: dict[str, Any],
    settings: dict,
    overrides: dict[str, Any],
) -> None:
    """One row of summary and one row per edge, for this perturbation alone.

    Deliberately not `export_results`: a perturbation is a number to compare,
    not a second published model, so it gets the two CSVs a comparison needs
    and no VTK, statistics or plots.
    """
    output_dir = result.output_dir
    assert output_dir is not None  # only called once a directory exists
    equivalent = float(solved["equivalent_resistance"])
    baseline_equivalent = float(baseline.get("equivalent_resistance", float("nan")))
    delta = equivalent - baseline_equivalent
    summary_path = output_dir / f"{result.name}_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(PERTURBATION_SUMMARY_COLUMNS))
        writer.writeheader()
        writer.writerow(
            {
                "name": result.name,
                "type": result.type,
                "overrides": json.dumps(plain_values(overrides), sort_keys=True),
                "inlet_pressure_pa": float(settings["inlet_p_bc"]),
                "outlet_pressure_pa": float(settings["outlet_p_bc"]),
                "total_inlet_flow": solved["total_inlet_flow"],
                "total_outlet_flow": solved["total_outlet_flow"],
                "flow_balance_error": (
                    solved["total_inlet_flow"] + solved["total_outlet_flow"]
                ),
                "equivalent_resistance": equivalent,
                "baseline_equivalent_resistance": baseline_equivalent,
                "delta_vs_baseline": delta,
                "percent_change_vs_baseline": (
                    delta / baseline_equivalent * 100.0
                    if baseline_equivalent
                    else float("inf")
                ),
            }
        )
    result.outputs.append(summary_path)

    edges_path = output_dir / f"{result.name}_edges.csv"
    G = result.graph
    assert G is not None
    with edges_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(PERTURBATION_EDGE_COLUMNS))
        writer.writeheader()
        for u, v, key, data in G.edges(keys=True, data=True):
            baseline_data = baseline_graph.get_edge_data(u, v, key) or {}
            was = baseline_data.get("resistance")
            now = data.get("resistance")
            writer.writerow(
                {
                    "u": u,
                    "v": v,
                    "key": key,
                    "branch_order": data.get("branch_order"),
                    "length_um": data.get("length"),
                    "diameter_um": data.get("fwhm_diameter_um"),
                    "resistance": now,
                    "conductance": data.get("conductance"),
                    "pressure_drop": data.get("pressure_drop"),
                    "flow_signed": data.get("flow_signed"),
                    "baseline_resistance": was,
                    "resistance_ratio": (
                        float(now) / float(was) if was and now is not None else None
                    ),
                }
            )
    result.outputs.append(edges_path)


def _perturb_one(
    spec: PerturbationSpec,
    settings: dict,
    model: HaemodynamicModel,
    boundaries: BoundaryNodes,
    schema: Schema,
    root: Path,
    baseline: dict[str, Any],
    *,
    image: np.ndarray | None = None,
) -> PerturbationResult:
    """Run one perturbation from the baseline network, and write its output."""
    result = PerturbationResult(name=spec.name, type=spec.type)
    refused = spec.incomparable_overrides()
    if refused:
        raise ValueError(
            f"perturbation '{spec.name}' overrides {list(refused)}. A "
            "perturbation may not change the blood model: its resistances "
            "would not be comparable with the baseline it is differenced "
            "against. Set it for the whole run instead."
        )
    if spec.type == "none":
        logger.info(
            f"Perturbation '{spec.name}': type 'none', so nothing is re-solved "
            "and nothing is written."
        )
        return result

    overrides = spec.applied_overrides(schema)
    perturbed = {**settings, **overrides}
    result.output_dir = root / spec.name
    result.output_dir.mkdir(parents=True, exist_ok=True)

    # A copy per perturbation, so that nothing it does reaches the baseline or
    # another perturbation: they answer independent questions of the same
    # network, and the run's own graph is still exported afterwards. See
    # `_perturbation_copy` for why a shallow graph copy is enough.
    # `G`, not `graph`, because `graph` is this module's imported subpackage.
    G = _perturbation_copy(model.graph)
    diameter_table = perturbed["diameter_by_branch_order"]
    prefer_measured = bool(perturbed["use_fwhm_edge_diameters"])
    summary: dict[str, Any] = {"overrides": overrides}
    #: Sweep helper payload when this type ran a grid; used for Alice plots.
    sweep_payload: dict[str, Any] | None = None

    if spec.type == "pressure_sweep":
        sweep_payload = run_pericyte_dilation_pressure_sweep(
            G,
            perturbed,
            inlet_nodes=list(boundaries.inlet_nodes),
            outlet_nodes=list(boundaries.outlet_nodes),
            output_dir=result.output_dir,
            sweep_dilation=False,
            sweep_pressure=True,
        )
    elif spec.type == "pressure_and_pericyte_sweep":
        sweep_payload = run_pericyte_dilation_pressure_sweep(
            G,
            perturbed,
            inlet_nodes=list(boundaries.inlet_nodes),
            outlet_nodes=list(boundaries.outlet_nodes),
            output_dir=result.output_dir,
            sweep_dilation=True,
            sweep_pressure=True,
        )
    elif spec.type == "pericyte_dilation_sweep":
        sweep_payload = run_pericyte_dilation_pressure_sweep(
            G,
            perturbed,
            inlet_nodes=list(boundaries.inlet_nodes),
            outlet_nodes=list(boundaries.outlet_nodes),
            output_dir=result.output_dir,
            sweep_dilation=True,
            sweep_pressure=False,
        )
    elif spec.type == "arteriole_diameter_change":
        G, _table, scaling = haemodynamics.scale_arteriole_diameters(
            G,
            diameter_table,
            haemodynamics.percent_change_to_scale(
                float(perturbed["arteriole_diameter_change_percent"])
            ),
            model=_poiseuille_model_for(perturbed),
            prefer_edge_fwhm_diameter=prefer_measured,
        )
        summary.update(scaling)
        summary["arteriole_diameter_change_percent"] = float(
            perturbed["arteriole_diameter_change_percent"]
        )
    elif spec.type == "arteriole_diameter_sweep":
        sweep_payload = run_arteriole_dilation_pressure_sweep(
            G,
            perturbed,
            inlet_nodes=list(boundaries.inlet_nodes),
            outlet_nodes=list(boundaries.outlet_nodes),
            output_dir=result.output_dir,
            sweep_dilation=True,
            sweep_pressure=False,
        )
    elif spec.type == "pressure_and_arteriole_sweep":
        sweep_payload = run_arteriole_dilation_pressure_sweep(
            G,
            perturbed,
            inlet_nodes=list(boundaries.inlet_nodes),
            outlet_nodes=list(boundaries.outlet_nodes),
            output_dir=result.output_dir,
            sweep_dilation=True,
            sweep_pressure=True,
        )
    elif spec.type == "capillary_diameter_sweep":
        sweep_payload = run_capillary_dilation_pressure_sweep(
            G,
            perturbed,
            inlet_nodes=list(boundaries.inlet_nodes),
            outlet_nodes=list(boundaries.outlet_nodes),
            output_dir=result.output_dir,
            sweep_dilation=True,
            sweep_pressure=False,
        )
    elif spec.type == "pressure_and_capillary_sweep":
        sweep_payload = run_capillary_dilation_pressure_sweep(
            G,
            perturbed,
            inlet_nodes=list(boundaries.inlet_nodes),
            outlet_nodes=list(boundaries.outlet_nodes),
            output_dir=result.output_dir,
            sweep_dilation=True,
            sweep_pressure=True,
        )
    elif spec.type == "pericyte_spacing_sweep":
        sweep_payload = run_pericyte_geometry_sweep(
            G,
            perturbed,
            inlet_nodes=list(boundaries.inlet_nodes),
            outlet_nodes=list(boundaries.outlet_nodes),
            output_dir=result.output_dir,
            sweep_axis="spacing",
        )
    elif spec.type == "pericyte_length_sweep":
        sweep_payload = run_pericyte_geometry_sweep(
            G,
            perturbed,
            inlet_nodes=list(boundaries.inlet_nodes),
            outlet_nodes=list(boundaries.outlet_nodes),
            output_dir=result.output_dir,
            sweep_axis="length",
        )
    elif spec.type == "pericyte_diameter_change":
        G, strategy, strategy_results = set_resistances_for_constriction_strategy(
            G,
            diameter_by_branch_order=diameter_table,
            constriction_factor_by_branch_order=perturbed["constriction_by_branch_order"],
            use_pericyte_mask_constriction=bool(perturbed["use_pericyte_mask_constriction"]),
            use_probabilistic_constriction=bool(
                perturbed["use_probabilistic_pericyte_constriction"]
            ),
            prefer_edge_fwhm_baseline=prefer_measured,
            constriction_length=float(perturbed["constriction_length_um"]),
            constriction_spacing=float(perturbed["constriction_spacing_um"]),
            viscosity_law=perturbed["viscosity_law"],
            haematocrit=float(perturbed["haematocrit"]),
            diameter_basis=perturbed["diameter_basis"],
            constriction_probability=(
                1.0
                if perturbed["pericyte_constriction_probability"] is None
                else float(perturbed["pericyte_constriction_probability"])
            ),
            pericyte_mask_path=perturbed["pericyte_mask_path"],
            pericyte_mask_h5_dataset_name=perturbed["pericyte_mask_h5_dataset_name"],
            axis_order=perturbed["image_axis_order"],
            seed=perturbed["pericyte_constriction_seed"],
        )
        summary["strategy"] = strategy
        summary.update(strategy_results)
    else:
        # `perturbation_problems` reports an unknown type before a run starts;
        # reaching here means a caller skipped the checks.
        raise ValueError(
            f"perturbation '{spec.name}' has unknown type {spec.type!r}."
        )

    if sweep_payload is not None:
        result.outputs.append(Path(sweep_payload["csv_path"]))
        summary["sweep_points"] = len(sweep_payload["results"])

    solved = _solve_network(G, perturbed, boundaries)
    result.graph = G
    summary.update(
        {
            "equivalent_resistance": float(solved["equivalent_resistance"]),
            "total_inlet_flow": float(solved["total_inlet_flow"]),
            "total_outlet_flow": float(solved["total_outlet_flow"]),
        }
    )
    result.summary = summary
    _write_perturbation_csvs(
        result,
        baseline_graph=model.graph,
        solved=solved,
        baseline=baseline,
        settings=perturbed,
        overrides=overrides,
    )

    # One post-step for every type: Alice-style curves for sweeps; pipeline-like
    # plots/CSVs for a single re-solve. Sweeps drop their graph so napari does
    # not build a flow layer for a grid of answers.
    if is_sweep_perturbation(spec.type):
        if sweep_payload is not None:
            result.outputs.extend(
                export_sweep_perturbation_plots(
                    spec.type, sweep_payload["results"], result.output_dir
                )
            )
        result.graph = None
    else:
        result.outputs.extend(
            export_non_sweep_perturbation_artifacts(
                G,
                result.output_dir,
                perturbed,
                image=image,
                name_stem=spec.name,
            )
        )

    logger.info(
        f"Perturbation '{spec.name}' ({spec.type}): equivalent resistance "
        f"{summary['equivalent_resistance']:.6g} vs baseline "
        f"{baseline.get('equivalent_resistance', float('nan')):.6g}; wrote "
        f"{len(result.outputs)} file(s) to {result.output_dir}"
    )
    return result


def run_perturbations(
    settings: dict,
    model: HaemodynamicModel,
    boundaries: BoundaryNodes,
    schema: Schema,
    progress: StageProgress | None = None,
    network: VesselNetwork | None = None,
) -> PerturbationRun:
    """Re-solve the finished network once per configured perturbation.

    Every one starts from the same baseline and none of them touches it, so
    they do not compose and the order they are listed in does not matter. The
    run's own graph is untouched too: `export_results` still exports the
    baseline afterwards.

    A perturbation that raises is reported and the rest still run -- an hour of
    graph building should not be lost to one bad entry in a list of five.
    """
    specs = perturbations_from_settings(settings)
    if not settings["run_perturbations"]:
        if specs:
            logger.info(
                f"{len(specs)} perturbation(s) configured but run_perturbations "
                "is off; none will run."
            )
        return PerturbationRun()
    if not specs:
        logger.info("run_perturbations is on but no perturbations are configured.")
        return PerturbationRun()
    if not settings["run_haemodynamics"]:
        logger.warning(
            "run_perturbations is on but run_haemodynamics is off, so there are "
            "no resistances to re-solve. Skipping every perturbation."
        )
        return PerturbationRun()

    # Reported, not raised: `preflight` is where a bad entry stops a run, and
    # it has already had its chance. Here, the entries that do work should.
    for message in perturbation_problems(settings, schema):
        logger.error(f"Perturbation configuration: {message}")

    root = perturbation_output_dir(settings)
    run = PerturbationRun(output_dir=root)
    # On a copy, like every perturbation: `_solve_network` writes pressures and
    # flows onto what it solves, and the number differenced against has to come
    # from the same solver as the perturbations' -- but the run's own graph,
    # which `export_results` is about to write out, must leave this stage
    # exactly as it arrived.
    run.baseline = _solve_network(
        _perturbation_copy(model.graph), settings, boundaries
    )
    logger.info(
        f"Perturbations: {len(specs)} to run from a baseline equivalent "
        f"resistance of {run.baseline['equivalent_resistance']:.6g}, into {root}"
    )

    image = None if network is None else getattr(network.volume, "image", None)
    for spec in specs:
        if progress is not None:
            progress.step(spec.name, total=len(specs))
        try:
            run.results.append(
                _perturb_one(
                    spec,
                    settings,
                    model,
                    boundaries,
                    schema,
                    root,
                    run.baseline,
                    image=image,
                )
            )
        except Exception as error:
            logger.exception(f"Perturbation '{spec.name}' failed: {error}")
            run.results.append(
                PerturbationResult(
                    name=spec.name,
                    type=spec.type,
                    error=f"{type(error).__name__}: {error}",
                )
            )
    return run


def export_results(settings: dict, network: VesselNetwork, model: HaemodynamicModel, solution: Solution):
    """Write VTK, statistics and distance measurements, and draw the plots."""
    G = model.graph
    image = network.volume.image
    output_dir = network.volume.output_dir
    voxel_size_zyx = network.volume.voxel_size_zyx
    main_voxel_size_xyz = network.volume.voxel_size_xyz
    # 7) Compute and print vessel statistics.
    logger.info("Computing vessel statistics...")
    if settings["statistics"]:
        valid_statistics_modes = {"fast", "full"}
        if settings["statistics_mode"] not in valid_statistics_modes:
            raise ValueError(
                f"Invalid statistics_mode='{settings['statistics_mode']}'. "
                f"Choose one of {sorted(valid_statistics_modes)}."
            )
        node_positions = nx.get_node_attributes(G, "pos")
        stats = statistics.compute_comprehensive_vessel_statistics(
            G,
            node_positions=node_positions,
            image_dimensions=image.shape,
            # Zipped element-wise with image.shape, which is (z, y, x), so this
            # is the per-array-axis spacing. Left at its (1, 1, 1) default the
            # whole-image volume and density were counted in voxels for any
            # stack whose z spacing is not 1 um.
            voxel_size=voxel_size_zyx,
            statistics_mode=settings["statistics_mode"],
        )

        logger.info("=== Statistics ===")
        for key, value in stats.items():
            logger.info(f"  {key}: {value}")

        stats_csv_path = output_dir / f"{settings['input_path'].stem}_statistics.csv"
        statistics.export_statistics_to_csv(stats, stats_csv_path)
        logger.info(f"Saved statistics CSV to: {stats_csv_path}")

        branch_stats = statistics.compute_branch_order_statistics(
            G,
            node_positions=node_positions,
        )
        branch_stats_csv_path = output_dir / f"{settings['input_path'].stem}_branch_statistics.csv"
        statistics.export_branch_order_statistics_to_csv(
            branch_stats,
            branch_stats_csv_path,
        )
        logger.info(f"Saved branch-order statistics CSV to: {branch_stats_csv_path}")

        if settings["run_haemodynamics"]:
            weighted_measurements = statistics.compute_betweenness_and_community_measurements(G)
        else:
            weighted_measurements = {
                "edge_resistance": {
                    "Betweenness": {
                        "Betweenness Mean": "N/A (haemodynamics disabled)",
                        "Betweenness Max": "N/A (haemodynamics disabled)",
                        "Betweenness Top Nodes": "N/A (haemodynamics disabled)",
                        "Betweenness Method": "N/A (haemodynamics disabled)",
                    },
                    "Communities": {
                        "Community Count": "N/A (haemodynamics disabled)",
                        "Largest Community Size": "N/A (haemodynamics disabled)",
                        "Mean Community Size": "N/A (haemodynamics disabled)",
                        "Community Method": "N/A (haemodynamics disabled)",
                    },
                },
                "edge_length": {
                    "Betweenness": statistics.compute_weighted_betweenness_summary(
                        G,
                        source_attr="length",
                        inverse_source_attr=False,
                    ),
                    "Communities": statistics.compute_weighted_communities_summary(
                        G,
                        source_attr="length",
                        inverse_source_attr=False,
                    ),
                },
            }
        logger.info("=== Weighted Betweenness and Communities ===")
        for model_name, model_results in weighted_measurements.items():
            logger.info(f"  [{model_name}]")
            for metric_name, metric_values in model_results.items():
                logger.info(f"    {metric_name}: {metric_values}")

        resistance_path = output_dir / f"{settings['input_path'].stem}_betweenness_communities_resistance.json"
        resistance_path.write_text(
            json.dumps(weighted_measurements["edge_resistance"], indent=2)
        )
        length_path = output_dir / f"{settings['input_path'].stem}_betweenness_communities_edge_length.json"
        length_path.write_text(
            json.dumps(weighted_measurements["edge_length"], indent=2)
        )
        logger.info(f"Saved edge-resistance stats to: {resistance_path}")
        logger.info(f"Saved edge-length stats to: {length_path}")
    else:
        logger.info("Vessel statistics skipped.")

    # 8) Optional: nearest 3D distance from objects in a cell mask to vessel edge.
    if settings["measurement_3d_to_cell_mask"]:
        if settings["cell_mask_path"] is None:
            raise ValueError(
                "measurement_3d_to_cell_mask=True requires cell_mask_path."
            )
        distance_summary = statistics.run_3d_measurement_to_cell_mask(
            graph=G,
            cell_mask_path=Path(settings["cell_mask_path"]),
            output_dir=output_dir,
            image_stem=settings["input_path"].stem,
            voxel_size_xyz=main_voxel_size_xyz,
            axis_order=settings["image_axis_order"],
            vessel_mask_path=(
                None
                if settings["measurement_3d_vessel_mask_path"] is None
                else Path(settings["measurement_3d_vessel_mask_path"])
            ),
            vessel_reference_image_path=(
                None
                if settings["measurement_3d_reference_image_path"] is None
                else Path(settings["measurement_3d_reference_image_path"])
            ),
            cell_mask_h5_dataset_name=settings["cell_mask_h5_dataset_name"],
            vessel_mask_h5_dataset_name=settings["measurement_3d_vessel_mask_h5_dataset_name"],
            vessel_reference_h5_dataset_name=settings["measurement_3d_reference_h5_dataset_name"],
        )
        logger.info(
            "3D cell-mask vessel-distance summary: "
            f"{distance_summary}"
        )
    else:
        logger.info("3D cell-mask vessel-distance measurement skipped.")

    # 10) Export vessels/pericytes/nodes to VTK and optionally visualize in PyVista.
    # FA I have no idea if pericyte location is correct. AI did that part.
    # FA I don't fully understand how pericyte location is currently determined?
    if settings["run_haemodynamics"] and settings["vtk_export"]:
        vtk_export = visualization.graph_to_vtk(G, settings["vtk_output_prefix"])
        logger.info("=== VTK Export ===")
        logger.info(f"  Vessels:   {vtk_export['vessels_path']}")
        logger.info(f"  Pericytes: {vtk_export['pericytes_path']}")
        logger.info(f"  Nodes:     {vtk_export['nodes_path']}")
        logger.info(f"  Counts: vessels={vtk_export['vessel_line_count']}, "
          f"pericytes={vtk_export['pericyte_count']}, nodes={vtk_export['node_count']}")
    if settings["run_haemodynamics"] and settings["visualize_vtk"] and settings["vtk_export"]:
        visualization.visualize_vtk_network(
            vtk_export["vessels_path"],
            vtk_export["pericytes_path"],
            vtk_export["nodes_path"],
            show_nodes=False,
        )
    if settings["run_haemodynamics"] and settings["visualize_vtk"] and not settings["vtk_export"]:
        logger.warning(
            "VTK visualization requested but VTK export is disabled. "
            "Set vtk_export: true in the config to enable."
        )
    if settings["run_haemodynamics"] and not settings["visualize_vtk"]:
        logger.info("VTK visualization skipped.") 
    # 11) Optional matplotlib visualization.
    if settings["visualize_results"]:
        logger.info("Generating visualizations...")
        valid_plot_modes = {"all", "final_only", "none"}
        if settings["ide_plot_mode"] not in valid_plot_modes:
            raise ValueError(
                f"Invalid ide_plot_mode='{settings['ide_plot_mode']}'. "
                f"Choose one of {sorted(valid_plot_modes)}."
            )
        show_any_ide_plot = settings["show_plots_in_ide"] and settings["ide_plot_mode"] != "none"
        show_degree_plot = settings["show_plots_in_ide"] and settings["ide_plot_mode"] == "all"
        show_overlay_plot = show_any_ide_plot and settings["final_render_mode"] == "2d"
        show_3d_plot = show_any_ide_plot and settings["final_render_mode"] == "3d"
        show_branch_order_plot = settings["show_plots_in_ide"] and settings["ide_plot_mode"] == "all"
        visualization.plot_node_degree_distribution(
            G,
            save_path=None if settings["interactive_plots"] else settings["plot_dir"] / "node_degree_distribution.png",
            show=settings["interactive_plots"] or show_degree_plot,
            show_after_save=show_degree_plot and not settings["interactive_plots"],
        )
        if settings["final_render_mode"] == "3d":
            overlay_3d_path = None if settings["interactive_plots"] else settings["plot_dir"] / "edges_and_nodes_overlay_3d.html"
            visualization.visualize_3d_plotly(
                G,
                title="Edges and Nodes Overlay (Interactive 3D)",
                save_html_path=str(overlay_3d_path) if overlay_3d_path else None,
                show=settings["interactive_plots"] or show_3d_plot,
            )
            if overlay_3d_path is not None:
                logger.info(f"Saved interactive 3D overlay to: {overlay_3d_path}")
        else:
            visualization.visualize_edges_and_nodes(
                image,
                G,
                save_path=None if settings["interactive_plots"] else settings["plot_dir"] / "edges_and_nodes_overlay.png",
                show=settings["interactive_plots"] or show_overlay_plot,
                show_after_save=show_overlay_plot and not settings["interactive_plots"],
            )
        #HD note - need visualisation of pericyte localisations (ie based upon constriction data)
        
        if settings["inlet_nodes"]:
            visualization.visualize_geometry_with_branch_orders(
                image,
                G,
                group_above=8,
                save_path=None if settings["interactive_plots"] else settings["plot_dir"] / "geometry_with_branch_orders.png",
                show=settings["interactive_plots"] or show_branch_order_plot,
                show_after_save=show_branch_order_plot and not settings["interactive_plots"],
            )
        if (
            settings["hold_ide_plots_open"]
            and show_any_ide_plot
            and not settings["interactive_plots"]
            and plt.get_fignums()
        ):
            logger.info("Holding plot windows open. Close them to finish the script.")
            plt.show(block=True)
    else:
        logger.info("Matplotlib visualizations skipped.")


    return solution


def _produced(
    callback: StageOutputCallback | None, stage: str, output: Any
) -> None:
    """Hand one finished stage's output to *callback*, if anyone is watching.

    Called after the stage's ``with`` block rather than inside it. Inside, an
    exception from the callback would be caught by ``RunProgress.stage()``,
    reported as that stage failing, and re-raised -- so a fault in whoever is
    watching would both kill the run and lie about where it died.
    """
    if callback is not None:
        callback(stage, output)


def run_pipeline_stages(
    settings: dict,
    schema: Schema,
    progress: ProgressCallback | None = None,
    on_stage_output: StageOutputCallback | None = None,
) -> nx.MultiGraph | None:
    """Run every stage in order, for one resolved settings dict.

    Returns the graph the run produced, so a caller can do more with it -- the
    whole-brain script sweeps pericyte dilation over exactly this graph.

    *progress*, if given, is called with a
    :class:`~haemolynx.pipeline.progress.ProgressEvent` as each stage starts,
    finishes or fails, and once per topology step inside graph building.

    *on_stage_output*, if given, is called with each stage's name and the object
    that stage returned, once that stage is done. It is how the napari panel
    shows a run's work as it happens; a script could pickle each output, or
    count it, or ignore it.

    Both run on whatever thread the run is on, and must not raise: a run is not
    stopped, or changed in any way, by whoever is watching it. Note the outputs
    are the live objects, not copies -- every stage after ``build_network``
    writes attributes onto the same graph -- so a consumer that wants a
    snapshot must take one there and then.
    """
    run = RunProgress(progress)
    with run.stage("segment"):
        inputs = segment(settings)
    _produced(on_stage_output, "segment", inputs)
    with run.stage("skeletonise"):
        volume = skeletonise(settings, inputs)
    _produced(on_stage_output, "skeletonise", volume)
    with run.stage("build_network") as building:
        network = build_network(
            settings,
            volume,
            schema,
            progress=building,
            on_step_graph=(
                (lambda label, graph_obj: _produced(
                    on_stage_output, f"{TOPOLOGY_STEP}{label}", graph_obj
                ))
                if on_stage_output is not None
                else None
            ),
        )
    _produced(on_stage_output, "build_network", network)
    with run.stage("assign_boundaries"):
        boundaries = assign_boundaries(settings, network)
    _produced(on_stage_output, "assign_boundaries", boundaries)
    with run.stage("assign_diameters"):
        diameters = assign_diameters(settings, network, boundaries, schema)
    _produced(on_stage_output, "assign_diameters", diameters)
    with run.stage("build_haemodynamic_model"):
        model = build_haemodynamic_model(settings, diameters)
    _produced(on_stage_output, "build_haemodynamic_model", model)
    with run.stage("solve"):
        solution = solve(settings, model, boundaries)
    _produced(on_stage_output, "solve", solution)
    # After the baseline is solved, so each perturbation has a solved network to
    # difference against, and before the export, which writes that baseline out.
    with run.stage("run_perturbations") as perturbing:
        perturbations = run_perturbations(
            settings,
            model,
            boundaries,
            schema,
            progress=perturbing,
            network=network,
        )
    _produced(on_stage_output, "run_perturbations", perturbations)
    with run.stage("export_results"):
        export_results(settings, network, model, solution)
    _produced(on_stage_output, "export_results", solution)
    return model.graph
