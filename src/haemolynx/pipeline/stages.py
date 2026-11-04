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
    apply_network_handling    the "Network handling" settings, at the start of the
                              haemodynamics stage (e.g. drop components lacking I/O)
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
import hashlib
import json
import logging
import os
import pickle
import threading
from collections import Counter
from dataclasses import dataclass, field, replace
from pathlib import Path, PurePath
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Sequence

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from haemolynx import graph, haemodynamics, io, preprocessing, statistics, visualization
from haemolynx.preprocessing import raw_stain
from haemolynx.haemodynamics.apply import (
    HaemodynamicsApplyConfig,
    apply_poiseuille_resistances,
    assign_edge_diameters,
    measure_mask_diameters,
)
from haemolynx.haemodynamics.constriction_strategy import (
    constriction_strategy_kwargs,
    placed_pericyte_mask,
    set_resistances_for_constriction_strategy,
)
from haemolynx.haemodynamics.haematocrit_distribution import (
    DISCHARGE_HAEMATOCRIT_ATTR,
    JUNCTION_RULE_NO_SEPARATION,
    JUNCTION_RULE_SPLIT_JUNCTIONS,
    iterate_flow_and_haematocrit,
)
from haemolynx.haemodynamics.perturbations import (
    MEASURED_CONSTRICTION_TYPE,
    PerturbationSpec,
    is_capillary_block,
    is_sweep_perturbation,
    places_pericytes,
    perturbation_folder_name,
    perturbation_output_dir,
    perturbation_problems,
    perturbations_from_settings,
)
from haemolynx.haemodynamics.pericyte_geometry_sweep import (
    run_pericyte_geometry_sweep,
)
from haemolynx.haemodynamics import pericyte_measurements as pericyte_measurement
from haemolynx.haemodynamics.capillary import (
    run_capillary_dilation_pressure_sweep,
)
from haemolynx.haemodynamics.pericyte_sweep import (
    apply_baseline_overrides,
    run_arteriole_dilation_pressure_sweep,
    run_pericyte_dilation_pressure_sweep,
    solve_pressure_and_boundary_flow,
)
from haemolynx.haemodynamics.perturbations import plain as plain_values
from haemolynx.haemodynamics.viscosity import run_diameter_basis, run_viscosity_law
from haemolynx.haemodynamics.poiseuille import (
    DIAMETER_SOURCE_IMAGE_RUN,
    DIAMETER_SOURCE_SYNTHETIC,
    EDGE_DIAMETER_BASIS,
    PoiseuilleModel,
    start_measurement_afresh,
)
from haemolynx.io.crop import GRAPH_ATTRIBUTE as CROP_GRAPH_ATTRIBUTE
from haemolynx.io.crop import CropBox, crop_box_from_settings, crop_volume_and_box
from haemolynx.io.downsample import GRAPH_ATTRIBUTE as DOWNSAMPLE_GRAPH_ATTRIBUTE
from haemolynx.io.downsample import (
    RAW_SETTING as RAW_DOWNSAMPLE_SETTING,
    Downsample,
    downsample_factor,
    downsample_from_settings,
    downsample_volume,
    graph_downsample,
    raw_volume_like_graph,
    record_on_graph as record_downsample_on_graph,
)
from haemolynx.io.ilastik import ilastik_segmented_path
from haemolynx.io.load import (
    _skeletonize_loaded_volume,
    _to_binary_volume_for_skeletonization,
    h5_binarization_rule,
    mask_binarization_rule,
)
from haemolynx.io.voxel_validation import resolve_voxel_size_xyz
from haemolynx.visualization.perturbation_plots import (
    export_non_sweep_perturbation_artifacts,
    export_sweep_perturbation_plots,
)
from haemolynx.parsers import Schema, parameters_of, prefixed_arguments
from haemolynx.pipeline.checks import tissue_raw_image
from haemolynx.pipeline.citations import write_citations
from haemolynx.pipeline.naming import classifier_segmented_path as naming_classifier_segmented_path
from haemolynx.pipeline.naming import (
    classifier_segments,
    run_input_path,
    run_stem,
    segments_raw_image,
)
from haemolynx.synthetic.template import GRAPH_TEMPLATE_RECORD, ROLES as SYNTHETIC_ROLES
from haemolynx.pipeline.progress import ProgressCallback, RunProgress, StageProgress
from haemolynx.pipeline.provenance import record_run
from haemolynx.preprocessing.memmap_support import bincount_by_slab

#: Called with each stage's name and the object that stage returned, so
#: something watching a run can look at its work as it happens rather than
#: waiting for the files at the end. The napari panel turns each one into
#: layers; a script could pickle them, or count them, or ignore them.
StageOutputCallback = Callable[[str, Any], None]

#: Prefix on the name given to a graph-building step, so a consumer can tell
#: one from a stage: `topology_step:prune_vascular_stubs`.
TOPOLOGY_STEP = "topology_step:"

#: Where `export_results` leaves what the run's statistics found, on the graph
#: it measured, so it goes wherever the graph goes (checkpoints, saved runs):
#: ``{"mode", "csv", "sections"}``, *sections* one per measure that ran
#: (`statistics.compute_vessel_statistics_by_measure`), then those below, each
#: ``{row: value}``. What the panel's Statistics tab shows. Absent when the
#: export computed none, and dropped from a resumed network, which the run
#: measures again.
STATISTICS_REPORT = "statistics_report"
#: The report's sections that are not a statistic measure.
HAEMODYNAMICS_SECTION = "haemodynamics"
BRANCH_ORDER_SECTION = "branch_orders"
WEIGHTED_MEASURES_SECTION = "weighted_betweenness_communities"
#: The statistics measures the weighted section and its three JSON files
#: report: written while either is on.
WEIGHTED_MEASURES = frozenset({"betweenness", "community"})
TISSUE_VOLUME_SECTION = "tissue_volume"

#: Pipeline stage calls in run order, for ``start_from`` comparisons.
STAGE_CALLS: tuple[str, ...] = (
    "segment",
    "skeletonise",
    "build_network",
    "assign_boundaries",
    "assign_diameters",
    "build_haemodynamic_model",
    "solve",
    "post_process",
    "run_perturbations",
    "export_results",
)

logger = logging.getLogger(__name__)

#: Logged in place of each read-only consistency check use_memmap_loading
#: skips. Those checks rasterise, distance-transform and label the whole
#: volume, and scipy's distance transform alone needs roughly 70 bytes of
#: plain RAM per voxel whatever it is given to write into -- none of that can
#: be moved to disk, and none of it changes any result.
LOW_RAM_SKIPPED_DIAGNOSTICS_MESSAGE = (
    "Skipping the %s check: Low RAM option is on, and this read-only "
    "diagnostic needs several whole-volume arrays in RAM. It never changes "
    "the skeleton or graph."
)


@dataclass
class SegmentedInputs:
    """What segmentation settled: which image to analyse, and where output goes."""

    image_path: Path
    output_dir: Path
    #: "tif"/"tiff"/"h5" — which loader skeletonise should use; "haemorun"
    #: for a synthetic run, whose ``image_path`` is its template.
    input_format: str = "tif"
    #: A synthetic run (``network_source=synthetic``): nothing is loaded or
    #: skeletonised; the Graph stage draws the network to ``template_path``.
    synthetic: bool = False
    template_path: Path | None = None


@dataclass
class SkeletonisedVolume:
    """The loaded volume, its skeleton, and the voxel size they are measured in.

    A synthetic run has neither (``synthetic``): its stages read the box
    from the graph (``image_extent_um``) and the voxel size from here.
    """

    image: np.ndarray | None
    skeleton: np.ndarray | None
    voxel_size_xyz: tuple[float, float, float]
    voxel_size_zyx: tuple[float, float, float]
    output_dir: Path
    #: The fat/thick vessel region from thickness-gated skeletonisation, if
    #: that ran, or the one a previous run kept beside the skeleton this run
    #: loaded (``thick_vessel_region_cache``); None otherwise.
    thick_vessel_mask: np.ndarray | None = None
    #: The pre-cleanup, canonically-binarised mask, if at least one
    #: segmentation_cleanup_* step ran (whether the skeleton was made or
    #: loaded); None otherwise (including when every step is off) --
    #: same "None costs nothing extra" convention as thick_vessel_mask.
    raw_segmented_image: np.ndarray | None = None
    #: Where ``image`` was cut from the whole input image (``crop_on_import``);
    #: None when it was not cropped. Every position after this is measured
    #: from its corner.
    crop: CropBox | None = None
    #: How ``image`` was reduced from the image as read, after the crop
    #: (``segmented_image_downsample``, io.downsample); None when it was not.
    #: ``voxel_size_xyz`` / ``voxel_size_zyx`` are the reduced grid's.
    downsample: Downsample | None = None
    #: A synthetic run's volume: no image, no skeleton (see the class).
    synthetic: bool = False
    #: The Input tab's settings this volume was made with
    #: (:func:`input_volume_settings`): a run resumed on it with others is
    #: refused (:func:`_check_resumed_input`). None when not recorded.
    input_settings: dict | None = None
    #: The thick region's centreline tree as skeletonisation drew it: its
    #: voxels in C order, ``graph.thick_vessel_junctions.thick_vessel_centreline_voxels``
    #: of ``thick_vessel_mask``, which graph building reads instead of
    #: drawing the tree again (a volume of zeros and a Dijkstra over every
    #: fat voxel). None when this run did not draw it: graph building does.
    thick_vessel_centreline: np.ndarray | None = None
    #: The segmentation as post processing last read it, binarised with its
    #: ``MaskSupport`` (:class:`_PostProcessSegmentation`): kept for the next
    #: Regenerate of this run, which is handed this volume again, while
    #: ``image`` and the settings that shape it are the same.
    post_process_segmentation: Any = field(default=None, repr=False, compare=False)


@dataclass
class VesselNetwork:
    """The graph, the volume it came from, and the vessel masks alongside it."""

    graph: nx.MultiGraph
    volume: SkeletonisedVolume
    large_arteriole_mask: np.ndarray | None = None
    large_venule_mask: np.ndarray | None = None
    small_arteriole_mask: np.ndarray | None = None
    small_venule_mask: np.ndarray | None = None
    #: The tissue measured from the raw image, with what it was measured from
    #: -- see `tissue_measurement`, which fills it on first use and measures
    #: again when a re-run changed either.
    tissue: "TissueCache | None" = None
    #: The width the large masks were split at into large and small ones
    #: (`graph.large_vessel_mask_split_diameter_um`), None when they were not:
    #: a run resumed on the Boundaries tab cuts them again when it changed.
    mask_split_diameter_um: float | None = None
    #: What each role's masks were read from and cleaned with
    #: (`_vessel_mask_sources`): a run resumed on the Boundaries tab loads a
    #: role again when it changed. None when not recorded: kept as they are.
    mask_sources: dict[str, tuple] | None = None


@dataclass(frozen=True)
class TissueCache:
    """One tissue measurement, and the settings and segmentation it came from."""

    key: tuple
    segmentation: Any
    tissue: statistics.TissueVolume


@dataclass
class BoundaryNodes:
    """Where flow enters and leaves, and where vessel types change.

    ``graph`` is the live network after this stage (post large-vessel volume
    cut when that option ran). Napari results and later stages must use this
    graph rather than a pre-cut copy remembered from ``build_network``.
    """

    inlet_nodes: list[int] = field(default_factory=list)
    outlet_nodes: list[int] = field(default_factory=list)
    arteriole_boundary_nodes: list[int] = field(default_factory=list)
    venule_boundary_nodes: list[int] = field(default_factory=list)
    #: Where the Large_Art/Large_Ven tier hands off to Art/Ven, when
    #: assign_large_vessel_branch_orders is on (see graph.large_vessel_network).
    large_arteriole_boundary_nodes: list[int] = field(default_factory=list)
    large_venule_boundary_nodes: list[int] = field(default_factory=list)
    resistance_node_pair: tuple[int, int] | None = None
    graph: nx.MultiGraph | None = None
    #: Post-assignment-prepass large masks (overlap cleanup applied), when set.
    large_arteriole_mask: np.ndarray | None = None
    large_venule_mask: np.ndarray | None = None
    #: The small masks boundary labelling read -- after overlap cleanup,
    #: tangential redefinition and continuity bridging -- so the viewer shows
    #: what decided the arteriole/venule boundaries, not the files as loaded.
    small_arteriole_mask: np.ndarray | None = None
    small_venule_mask: np.ndarray | None = None


@dataclass
class HaemodynamicModel:
    """A graph carrying branch orders and diameters; resistances after Haemodynamics."""

    graph: nx.MultiGraph
    results: dict[str, Any] = field(default_factory=dict)
    #: Intensity volume FWHM was fit against, for the napari review overlay.
    fwhm_raw: np.ndarray | None = None


@dataclass
class Solution:
    """What the solve produced: pressures, flows, equivalent resistance."""

    pressure: np.ndarray | None = None
    node_list: list[int] = field(default_factory=list)
    #: Two-point resistance (Pa.s/m^3) between `equivalent_resistance_nodes`
    #: -- the first inlet and the first outlet, or the two nodes the Post
    #: processing tab chose -- with every other boundary node left floating.
    equivalent_resistance: float | None = None
    equivalent_resistance_nodes: tuple[Any, Any] | None = None
    #: The whole network's resistance (Pa.s/m^3): the inlet-to-outlet pressure
    #: drop over the total flow in through every inlet.
    network_resistance: float | None = None
    statistics: dict[str, Any] = field(default_factory=dict)
    #: The network that was solved (same object ``solve`` wrote flows onto).
    graph: nx.MultiGraph | None = None
    #: The tissue `export_results` measured (``measure_tissue_volume``), for
    #: the viewer's tissue surface; None when it measured none.
    tissue: statistics.TissueVolume | None = None
    #: What `post_process` brought in line before this solve, when this is
    #: the solve it ran again on a hand-edited network (its
    #: ``model.results["post_process"]``); None for the Haemodynamics stage's.
    post_processed: dict[str, Any] | None = None
    #: What `export_results` wrote, by kind ("VTK", "statistics", "plots",
    #: "citations"), for the panel to say; empty before it runs.
    exported: tuple[str, ...] = ()


@dataclass
class PerturbationResult:
    """One perturbation: the network it produced, and where it wrote it.

    *graph* is that perturbation's own copy of the baseline, carrying its
    resistances, pressures and flows -- so the panel can draw it beside the
    baseline rather than instead of it. A `none` perturbation, and one that
    raised, have no graph and no output.

    Sweep types keep *graph* for centreline geometry only, and put every grid
    point's ``flow_abs`` (etc.) on *sweep_flows* so napari can slider-swap
    features without retaining N full MultiGraph copies.
    """

    name: str
    type: str
    graph: nx.MultiGraph | None = None
    output_dir: Path | None = None
    #: Files this perturbation wrote, in the order it wrote them.
    outputs: list[Path] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    #: Compact per-grid flow arrays for sweep types; None for a single re-solve.
    sweep_flows: Any | None = None
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
        """Those that produced a network (or sweep geometry) to draw."""
        return [
            result
            for result in self.results
            if result.graph is not None or result.sweep_flows is not None
        ]

    @property
    def failures(self) -> list[PerturbationResult]:
        return [result for result in self.results if result.error is not None]


@dataclass
class PipelineResume:
    """Earlier-stage outputs so :func:`run_pipeline_stages` can start mid-run.

    The GUI fills this from the previous tab's checkpoint. ``start_from`` is
    the first stage call of the current tab; stages before it that have no
    on-disk load path (boundaries, diameters, haemodynamics, solve,
    perturbations) are reconstructed from *graph* and the node-id lists
    rather than re-run.
    """

    start_from: str
    graph: Any | None = None
    inlet_nodes: tuple[Any, ...] = ()
    outlet_nodes: tuple[Any, ...] = ()
    arteriole_boundary_nodes: tuple[Any, ...] = ()
    venule_boundary_nodes: tuple[Any, ...] = ()
    #: Where the Large_Art/Large_Ven tier hands off to Art/Ven. Only
    #: assign_boundaries finds them, so a run resumed past it that assigns
    #: branch orders again (Diameters, Post processing) needs them handed on.
    large_arteriole_boundary_nodes: tuple[Any, ...] = ()
    large_venule_boundary_nodes: tuple[Any, ...] = ()
    resistance_node_pair: tuple[Any, Any] | None = None
    #: The cleaned (overlap-resolved) large-vessel masks assign_boundaries
    #: produced, when resuming past it. None falls back to whatever raw masks
    #: this run's own build_network loaded -- see _boundaries_from_resume.
    large_arteriole_mask: Any | None = None
    large_venule_mask: Any | None = None
    #: The small masks the Boundaries stage's arteriole/venule labelling read
    #: (after its overlap cleanup, redefinition and bridging), when resuming
    #: past it: Post processing judges an edited vessel against them.
    small_arteriole_mask: Any | None = None
    small_venule_mask: Any | None = None
    #: Thickness-gated skeletonisation's thick-vessel region, for a run whose
    #: *volume* does not carry it (loaded rather than skeletonised): where the
    #: post_process stage splits an edited vessel into a bridge.
    thick_vessel_mask: Any | None = None
    #: What segment, skeletonise and build_network returned in the run being
    #: resumed. Each one given, for a stage before *start_from*, is used as it
    #: is instead of that stage being done again -- no ilastik, no reloading
    #: and re-checking the image, skeleton, masks and graph. ``network`` takes
    #: *graph* in place of its own. None falls back to loading from disk.
    inputs: SegmentedInputs | None = None
    volume: SkeletonisedVolume | None = None
    network: VesselNetwork | None = None


def raw_image_is_set(settings: Mapping[str, Any]) -> bool:
    """Whether the run names a raw image (``raw_image_path``). It is optional:
    without one, everything that reads it -- ilastik segmentation, measuring
    diameters from it, the raw evidence round the mask -- is off, whatever
    its own switch says (the panel greys those switches out), and the run
    goes on from the segmented image alone. Never for a synthetic network,
    which reads no image at all."""
    if graph.network_is_synthetic(settings):
        return False
    return settings.get("raw_image_path") not in (None, "")


def ilastik_segments(settings: Mapping[str, Any]) -> bool:
    """``use_ilastik_segmentation`` in effect: on, and a raw image to segment."""
    return bool(settings.get("use_ilastik_segmentation")) and raw_image_is_set(settings)


def classifier_run_arguments(settings: Mapping[str, Any]) -> dict:
    """How the classifier reads the run's raw image: its axis order and
    voxel-size policy, and the low-RAM option (``use_memmap_loading``)."""
    return dict(
        axis_order=io.axis_order_for_run(settings),
        voxel_size_policy=settings.get("voxel_size_policy") or "auto",
        voxel_size_override_xyz=settings.get("voxel_size_override_xyz"),
        use_memmap=bool(settings.get("use_memmap_loading")),
        memmap_directory=settings.get("memmap_directory"),
    )


@dataclass(frozen=True)
class ObjectMask:
    """A mask the HaemoLynx classifier can make from the raw image for a
    setting that otherwise names a mask file."""

    switch: str
    classifier_setting: str
    label: str
    #: The mask path settings it stands in for.
    readers: tuple[str, ...]
    #: The job its classifier is trained for (``classifier.JOB_PURPOSES``).
    job: str


#: The masks of objects (each blob one object), and the 3D distances' vessel
#: mask, the classifier can make, by the word their output is named with:
#: ``{raw stem}_{word}_segmented.tif``.
OBJECT_MASKS: dict[str, ObjectMask] = {
    "pericytes": ObjectMask(
        "use_pixel_classifier_pericyte_segmentation", "pixel_classifier_pericyte_path", "Pericyte",
        ("pericyte_mask_path", "pericyte_measurement_mask_path"), "pericytes",
    ),
    "blocks": ObjectMask(
        "use_pixel_classifier_block_segmentation", "pixel_classifier_block_path", "Block",
        ("capillary_block_mask_path",), "blocks",
    ),
    "cells": ObjectMask(
        "use_pixel_classifier_cell_segmentation", "pixel_classifier_cell_path", "Cell",
        ("cell_mask_path",), "cells",
    ),
    "vessels_3d": ObjectMask(
        "use_pixel_classifier_3d_vessel_segmentation", "pixel_classifier_3d_vessel_path", "Vessel",
        ("measurement_3d_vessel_mask_path",), "vessels",
    ),
}


def _perturbations_run(settings: Mapping[str, Any]) -> list:
    if not (settings.get("run_perturbations") and settings.get("run_haemodynamics")):
        return []
    from haemolynx.haemodynamics.perturbations import perturbations_from_settings

    try:
        return list(perturbations_from_settings(settings))
    except (TypeError, ValueError):
        return []  # preflight reports a malformed entry


def object_masks_made(settings: Mapping[str, Any]) -> tuple[str, ...]:
    """The masks of objects (:data:`OBJECT_MASKS`) the classifier makes this
    run: its switch on, a raw image to segment, and something that reads the
    mask -- the pericyte measurements, mask-placed pericytes in the solve or a
    perturbation, the measured-constriction perturbation; a capillary block
    perturbation from a mask."""
    if graph.network_is_synthetic(settings) or not raw_image_is_set(settings):
        return ()
    from haemolynx.haemodynamics.perturbations import places_pericytes

    perturbations = _perturbations_run(settings)
    made = []
    if settings.get(OBJECT_MASKS["pericytes"].switch):
        read = (
            bool(settings.get("measure_pericytes"))
            or bool(settings.get("run_haemodynamics") and settings.get("use_pericyte_mask_constriction"))
            or any(
                spec.type == "pericyte_measured_constriction"
                or (places_pericytes(spec.type)
                    and {**settings, **dict(spec.overrides)}.get("use_pericyte_mask_constriction"))
                for spec in perturbations
            )
        )
        if read:
            made.append("pericytes")
    if settings.get(OBJECT_MASKS["blocks"].switch) and any(
        spec.type == "capillary_block_mask" for spec in perturbations
    ):
        made.append("blocks")
    if settings.get("measurement_3d_to_cell_mask"):
        made += [word for word in ("cells", "vessels_3d") if settings.get(OBJECT_MASKS[word].switch)]
    return tuple(made)


def object_mask_output_path(settings: Mapping[str, Any], word: str) -> Path:
    """``{output_dir}/{raw image stem}_{word}_segmented.tif``."""
    stem = Path(settings["raw_image_path"]).stem
    return Path(settings.get("output_dir") or ".") / f"{stem}_{word}_segmented.tif"


def classifier_requests(settings: Mapping[str, Any], *, main: bool = True) -> list[tuple[Path, dict[str, Path]]]:
    """Every ``(classifier, {mask: output})`` the HaemoLynx classifier makes
    this run: the main image's (with *main*), each vessel-mask role's, and
    the masks of :func:`object_masks_made`. Named as their own steps name them,
    so each finds its output current once :func:`segment_with_run_classifiers`
    has made it."""
    if graph.network_is_synthetic(settings) or not raw_image_is_set(settings):
        return []
    from haemolynx.io.vessel_masks import classifier_finds_venules, ilastik_vessel_mask_output_path

    requests: list[tuple[Path, dict[str, Path]]] = []
    if main and classifier_segments(settings) and settings.get("pixel_classifier_path") not in (None, ""):
        requests.append((Path(settings["pixel_classifier_path"]), {"Vessel": naming_classifier_segmented_path(settings)}))
    raw = Path(settings["raw_image_path"])
    folder = Path(settings.get("output_dir") or ".")
    for role in ("large", "small"):
        if not main or not graph.vessel_mask_classifier_in_use(settings, role):
            continue
        path = settings.get(f"pixel_classifier_{role}_vessel_path")
        if path in (None, ""):
            continue
        if classifier_finds_venules(path):
            outputs = {
                "Arteriole": ilastik_vessel_mask_output_path(folder, raw, role, "arteriole", ".tif"),
                "Venule": ilastik_vessel_mask_output_path(folder, raw, role, "venule", ".tif"),
            }
        else:
            outputs = {"Arteriole": ilastik_vessel_mask_output_path(folder, raw, role, "global", ".tif")}
        requests.append((Path(path), outputs))
    for word in object_masks_made(settings):
        mask = OBJECT_MASKS[word]
        path = settings.get(mask.classifier_setting)
        if path not in (None, ""):
            requests.append((Path(path), {mask.label: object_mask_output_path(settings, word)}))
    return requests


def segment_with_run_classifiers(
    settings: Mapping[str, Any], *, main: bool = True, progress: StageProgress | None = None
) -> None:
    """Make every mask the HaemoLynx classifier makes this run
    (:func:`classifier_requests`) in one pass over the raw image, the
    features the classifiers share computed once; masks already current are
    left as they are. The steps that read each mask then find it current."""
    from haemolynx.classifier import segment_with_classifiers

    requests = classifier_requests(settings, main=main)
    if not requests:
        return
    raw = io.resolve_image_path_with_optional_zip(Path(settings["raw_image_path"]))
    segment_with_classifiers(
        raw, requests, poll=progress.alive if progress is not None else None, **classifier_run_arguments(settings)
    )


def segment_object_masks(settings: dict, *, progress: StageProgress | None = None) -> dict[str, Path]:
    """Make each mask of objects this run reads that the classifier is to
    make (:func:`object_masks_made`), or find it current, and point every
    setting that reads it there. Run with the Segment stage, never by
    preflight or the optimisers."""
    from haemolynx.classifier import segment_with_classifier

    made: dict[str, Path] = {}
    for word in object_masks_made(settings):
        mask = OBJECT_MASKS[word]
        classifier = settings.get(mask.classifier_setting)
        if classifier in (None, ""):
            raise ValueError(
                f"{mask.classifier_setting} must be set when {mask.switch}=True: train one in the "
                "panel's training window and save it."
            )
        raw = io.resolve_image_path_with_optional_zip(Path(settings["raw_image_path"]))
        written = segment_with_classifier(
            raw,
            Path(classifier),
            {mask.label: object_mask_output_path(settings, word)},
            poll=progress.alive if progress is not None else None,
            **classifier_run_arguments(settings),
        )[mask.label]
        for name in mask.readers:
            settings[name] = written
        made[word] = written
        logger.info(f"Using the classifier's {word} mask: {written}")
    return made


def measures_diameters_from_raw_image(settings: Mapping[str, Any]) -> bool:
    """``measure_diameters_from_raw_image`` in effect: on, and a raw image to
    measure from. Without one the diameter chain starts at the mask."""
    return bool(settings.get("measure_diameters_from_raw_image")) and raw_image_is_set(settings)


def junction_bridges_in_effect(settings: Mapping[str, Any]) -> bool:
    """``junction_bridges`` in effect: on, with the widths it reads measured
    (haemodynamics on), on an image's network rather than a drawn one."""
    return (
        bool(settings.get("junction_bridges"))
        and bool(settings.get("run_haemodynamics"))
        and not graph.network_is_synthetic(settings)
    )


def segmented_input_path(settings: dict) -> Path | None:
    """The segmented image the run analyses: the Input tab's ``input_path``,
    or -- when ilastik does the segmenting -- ilastik's output, which is not
    the raw image (``raw_image_path``) but what ilastik makes from it.

    With ilastik on, the Input tab's ``input_path`` is not the answer until
    `segment` has replaced it with ilastik's output (it may still hold an
    unrelated file from before ilastik was switched on): it counts only once
    it is the output for this raw image -- in ``ilastik_output_dir``, named
    ``{raw image stem}_segmented`` (ilastik may have written ``.tif`` as
    ``.tiff``). Any file in that folder counted, so a resumed run read
    another raw image's segmentation, or a vessel mask's. Otherwise the
    output is derived where `segment` writes it -- that stage's own naming
    rule. None for a synthetic network, which has no segmented image."""
    return run_input_path(settings)


def current_ilastik_output(settings: Mapping[str, Any]) -> tuple[Path | None, str]:
    """``(path, "")``: the segmented image ilastik made before from the run's
    raw image (:func:`segmented_input_path`), when its record says it is the
    one `segment` would make now -- the same project, raw image and vessel
    label (``io.ilastik.ilastik_output_is_current``) -- else ``(None, why
    not)``. Never runs ilastik: for a caller that must not start a
    segmentation, which can take hours (the panel's Optimise settings)."""
    from haemolynx.io.ilastik import (
        ilastik_label_names,
        ilastik_output_is_current,
        ilastik_provenance,
        resolve_vessel_label,
    )

    output = segmented_input_path(dict(settings))
    if output is not None and classifier_segments(settings):
        from haemolynx.classifier import VESSEL_LABEL, classifier_output_current

        classifier = settings.get("pixel_classifier_path")
        if classifier in (None, "") or not Path(classifier).is_file():
            return None, f"the classifier {classifier} does not exist"
        arguments = classifier_run_arguments(settings)
        try:
            raw = io.resolve_image_path_with_optional_zip(Path(settings["raw_image_path"]))
            current, reason = classifier_output_current(
                raw, classifier, {VESSEL_LABEL: output},
                axis_order=arguments["axis_order"],
                voxel_size_policy=arguments["voxel_size_policy"],
                voxel_size_override_xyz=arguments["voxel_size_override_xyz"],
            )
        except (OSError, ValueError) as error:
            return None, str(error)
        return (output, "") if current else (None, reason)
    if output is None or not ilastik_segments(settings):
        return None, "ilastik does not segment this run's image"
    classifier = settings.get("ilastik_classifier_path")
    if classifier in (None, ""):
        return None, "ilastik_classifier_path is not set"
    classifier = Path(classifier)
    if not classifier.is_file():
        return None, f"the ilastik project {classifier} does not exist"
    try:
        raw = io.resolve_image_path_with_optional_zip(Path(settings["raw_image_path"]))
        names = ilastik_label_names(classifier)
        label = resolve_vessel_label(
            settings.get("ilastik_vessel_label"), names, setting="ilastik_vessel_label", project=classifier
        )
        provenance = ilastik_provenance(raw, classifier, label, label_names=names)
    except (OSError, ValueError) as error:
        return None, str(error)
    current, reason = ilastik_output_is_current(output, provenance)
    return (output, "") if current else (None, f"{output}: {reason}")


def segment(
    settings: dict,
    *,
    segmented_path: Path | str | None = None,
    progress: StageProgress | None = None,
):
    """Produce the segmented mask to analyse, running ilastik when asked to.

    *segmented_path* is a segmented image this run already has -- a resumed
    run's, from the stage it resumes -- so ilastik is not run again; the
    settings are still normalised as a fresh run's would be.

    *progress*, when given, hears a heartbeat about every half second while
    ilastik runs, which is where a watcher can stop the run: whatever its
    callback raises kills ilastik and propagates.
    """
    if graph.network_is_synthetic(settings):
        return _segment_synthetic(settings)
    if segmented_path is not None:
        settings["input_path"] = Path(segmented_path)
    # Not yet a path when ilastik is doing the segmenting: `input_path` is what
    # this stage *produces* in that case, and the documented way to ask for it
    # is to leave the setting empty. Converting first turned that into
    # `Path(None)` -- a TypeError before ilastik was even reached, on the one
    # configuration the setting exists to support (#127).
    if settings["input_path"] is not None:
        settings["input_path"] = Path(settings["input_path"])
    if segmented_path is not None:
        logger.info(f"Resuming with the segmented image already made: {settings['input_path']}")
    elif (
        settings.get("use_pixel_classifier_segmentation")
        and not settings.get("use_ilastik_segmentation")
        and not raw_image_is_set(settings)
    ):
        if settings["input_path"] is None:
            raise ValueError(
                "use_pixel_classifier_segmentation=True segments the raw image, but raw_image_path "
                "is not set, and there is no segmented image (input_path) to read instead."
            )
        logger.warning(
            "use_pixel_classifier_segmentation is on but raw_image_path is not set: the classifier "
            f"is skipped and the segmented image is read instead: {settings['input_path']}"
        )
    elif classifier_segments(settings):
        from haemolynx.classifier import VESSEL_LABEL, segment_with_classifier

        classifier = settings.get("pixel_classifier_path")
        if classifier in (None, ""):
            raise ValueError(
                "pixel_classifier_path must be set when use_pixel_classifier_segmentation=True: "
                "train one in the panel's training window and save it."
            )
        raw_image = io.resolve_image_path_with_optional_zip(Path(settings["raw_image_path"]))
        output = naming_classifier_segmented_path(settings)
        logger.info(f"Segmenting the raw image with the HaemoLynx classifier: {raw_image}")
        settings["input_path"] = segment_with_classifier(
            raw_image,
            Path(classifier),
            {VESSEL_LABEL: output},
            poll=progress.alive if progress is not None else None,
            **classifier_run_arguments(settings),
        )[VESSEL_LABEL]
        logger.info(f"Using the classifier's segmentation: {settings['input_path']}")
    elif settings["use_ilastik_segmentation"] and not raw_image_is_set(settings):
        # The raw image is optional, and ilastik has nothing without it: the
        # run reads the segmented image instead, as it would with the switch
        # off (the panel greys it out while no raw image is set).
        if settings["input_path"] is None:
            raise ValueError(
                "use_ilastik_segmentation=True segments the raw image, but raw_image_path is "
                "not set, and there is no segmented image (input_path) to read instead."
            )
        logger.warning(
            "use_ilastik_segmentation is on but raw_image_path is not set: ilastik is skipped "
            f"and the segmented image is read instead: {settings['input_path']}"
        )
    elif settings["use_ilastik_segmentation"]:
        unsegmented_image_path = Path(settings["raw_image_path"])
        unsegmented_image_path = io.resolve_image_path_with_optional_zip(unsegmented_image_path)
        if settings["ilastik_classifier_path"] is None:
            raise ValueError(
                "ilastik_classifier_path must be set when use_ilastik_segmentation=True."
            )
        settings["ilastik_output_dir"] = Path(settings["ilastik_output_dir"])
        segmented_output = ilastik_segmented_path(
            settings["ilastik_output_dir"], unsegmented_image_path, settings["ilastik_output_suffix"]
        )
        logger.info(f"Running ilastik segmentation for unsegmented image: {unsegmented_image_path}")
        settings["input_path"] = io.run_ilastik_headless_segmentation(
            input_image_path=unsegmented_image_path,
            classifier_path=Path(settings["ilastik_classifier_path"]),
            output_path=segmented_output,
            ilastik_executable=settings["ilastik_executable"],
            timeout=settings["ilastik_timeout_seconds"],
            vessel_label=settings.get("ilastik_vessel_label"),
            label_setting="ilastik_vessel_label",
            reuse_existing=bool(settings.get("ilastik_reuse_existing_output", False)),
            poll=progress.alive if progress is not None else None,
        )
        logger.info(f"Using ilastik-segmented image: {settings['input_path']}")
    else:
        if settings["input_path"] is None:
            raise ValueError(
                "input_path must name a segmented image, or use_ilastik_segmentation or "
                "use_pixel_classifier_segmentation must be on so that one is produced. "
                "Neither is set, so there is nothing to analyse."
            )
        logger.info(f"Using segmented input image: {settings['input_path']}")

    settings["input_path"] = io.resolve_image_path_with_optional_zip(settings["input_path"])
    # get image format from image_path
    input_format = settings["input_path"].suffix[1:].lower()
    if input_format not in ["tif", "tiff", "h5"]:
        raise ValueError(f"Invalid image format: {input_format}")
    settings["image_axis_order"] = io.normalize_axis_order(
        settings["image_axis_order"], label="IMAGE_AXIS_ORDER", allow_auto=True
    )
    # Under auto, each file is read in its own labels' order, and one that
    # names none in this image's (io.axis_order_for_run).
    image_axis_order = io.axis_order_for_file(settings["image_axis_order"], settings["input_path"])
    if image_axis_order != io.CANONICAL_AXIS_ORDER:
        logger.info(
            f"Input axis order '{image_axis_order}' will be transposed to the canonical "
            f"'{io.CANONICAL_AXIS_ORDER}' layout on load."
        )
    settings["output_dir"] = Path(settings["output_dir"])
    output_dir = settings["output_dir"]
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


def _segment_synthetic(settings: dict) -> SegmentedInputs:
    """The Input stage of a synthetic run (``network_source=synthetic``):
    the template read and checked, the output names settled. Nothing is
    segmented; the Graph stage draws the network."""
    from haemolynx.synthetic import load_template

    template_path = settings.get("synthetic_template_path")
    if template_path in (None, ""):
        raise ValueError(
            "network_source='synthetic' needs synthetic_template_path: the .haemorun "
            "file of the microscopy-image run the synthetic network is matched to."
        )
    template_path = Path(template_path)
    settings["synthetic_template_path"] = template_path
    template = load_template(template_path, settings.get("synthetic_template_source", "statistics"))
    settings["output_dir"] = Path(settings["output_dir"])
    output_dir = settings["output_dir"]
    valid_final_render_modes = {"2d", "3d"}
    if settings["final_render_mode"] not in valid_final_render_modes:
        raise ValueError(
            f"Invalid final_render_mode='{settings['final_render_mode']}'. "
            f"Choose one of {sorted(valid_final_render_modes)}."
        )
    logger.info(
        "Synthetic network: matched to %s (%s): %d vessels, %d inlet(s), %d outlet(s), "
        "box %s um, voxels %s um.",
        template_path,
        template.provenance.source,
        template.vessel_count,
        template.inlet_count,
        template.outlet_count,
        tuple(round(v, 1) for v in template.extent_um),
        tuple(round(v, 3) for v in template.voxel_size_zyx),
    )
    for note in template.provenance.notes:
        logger.warning("Synthetic template: %s", note)
    if graph.network_surrounds_image(settings):
        from haemolynx.synthetic.surround import load_image_run, read_open_ends

        image, boundaries = load_image_run(template_path)
        ends = read_open_ends(image, inlets=boundaries.get("inlet", ()), outlets=boundaries.get("outlet", ()))
        logger.info(
            "Surround: the template run's own network (%d vessels, %d open end(s) its image faces cut) "
            "is drawn round, %gx along %s.",
            image.number_of_edges(), len(ends), float(settings.get("synthetic_scale") or 1.0),
            settings.get("synthetic_scale_axes") or "xyz",
        )
    return SegmentedInputs(
        image_path=template_path,
        output_dir=output_dir,
        input_format="haemorun",
        synthetic=True,
        template_path=template_path,
    )


def _synthetic_volume(settings: dict, inputs: SegmentedInputs) -> SkeletonisedVolume:
    """The Skeletonise stage of a synthetic run: nothing to load or thin;
    the template's voxel size, and the output folders made."""
    from haemolynx.synthetic import load_template

    template = load_template(inputs.template_path, settings.get("synthetic_template_source", "statistics"))
    inputs.output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = settings.get("plot_dir")
    if plot_dir:
        Path(plot_dir).mkdir(parents=True, exist_ok=True)
    logger.info("Synthetic network: nothing to skeletonise; the network is drawn in the Graph stage.")
    voxel_size_zyx = tuple(float(v) for v in template.voxel_size_zyx)
    return SkeletonisedVolume(
        image=None,
        skeleton=None,
        voxel_size_xyz=tuple(reversed(voxel_size_zyx)),
        voxel_size_zyx=voxel_size_zyx,
        output_dir=inputs.output_dir,
        synthetic=True,
    )


def synthetic_box_shape(G: nx.Graph) -> tuple[int, int, int]:
    """The voxel shape of a synthetic network's box: the inverse of
    ``graph.dead_ends.image_extent_um``, ``round(extent / voxel) + 1``."""
    extent = graph.dead_ends.graph_image_extent_um(G)
    if extent is None:
        raise ValueError("the synthetic network records no image_extent_um")
    voxel = np.asarray(G.graph.get("image_voxel_size_zyx") or G.graph.get("voxel_size") or (1.0, 1.0, 1.0), dtype=float)
    return tuple(int(round(float(e) / float(v))) + 1 for e, v in zip(extent, voxel))


def synthetic_generation_options(settings: Mapping[str, Any]):
    """The ``synthetic.GenerationOptions`` a synthetic run draws with, from
    its settings: the one reading, for the Graph stage and preflight's plan
    of a surround's bed."""
    from haemolynx import synthetic

    return synthetic.GenerationOptions(
        scale=float(settings.get("synthetic_scale") or 1.0),
        scale_axes=str(settings.get("synthetic_scale_axes") or synthetic.SCALE_AXES_XYZ),
        arteriole_density=float(settings.get("synthetic_extension_arteriole_density", 1.0)),
        venule_density=float(settings.get("synthetic_extension_venule_density", 1.0)),
        interior_dead_ends=bool(settings.get("synthetic_interior_dead_ends", False)),
        bed_model=str(settings.get("synthetic_bed_model") or "delaunay"),
        open_ends=str(settings.get("synthetic_open_ends") or synthetic.OPEN_ENDS_CLOSED),
    )


def _build_synthetic_network(
    settings: dict,
    volume: SkeletonisedVolume,
    *,
    progress: StageProgress | None = None,
    resumed_graph: nx.MultiGraph | None = None,
) -> VesselNetwork:
    """The Graph stage of a synthetic run: the network drawn to the template
    (``haemolynx.synthetic.generate_network``), or a resumed run's handed back."""
    from haemolynx import synthetic

    if resumed_graph is not None:
        logger.info("Resuming with the synthetic network already drawn.")
        G = resumed_graph
    else:
        template = synthetic.load_template(
            settings["synthetic_template_path"], settings.get("synthetic_template_source", "statistics")
        )
        seed = settings.get("synthetic_seed")
        options = synthetic_generation_options(settings)
        logger.info(
            "Drawing a synthetic network to the template (seed %s, scale %g along %s, arterial and venous "
            "trees beyond the template's box at %g and %g of its density, dead ends inside the "
            "tissue %s, vessels leaving the box: %s)...",
            "fresh" if seed is None else seed,
            options.scale,
            options.scale_axes,
            options.arteriole_density,
            options.venule_density,
            "kept" if options.interior_dead_ends else "none",
            "none, a closed network" if options.open_ends == synthetic.OPEN_ENDS_CLOSED else "the template's",
        )
        if graph.network_surrounds_image(settings):
            from haemolynx.synthetic.surround import SURROUND_RECORD, load_image_run

            image, boundaries = load_image_run(settings["synthetic_template_path"])
            G = synthetic.surround_network(
                image, template,
                inlets=boundaries.get("inlet", ()), outlets=boundaries.get("outlet", ()),
                roles={role: nodes for role, nodes in boundaries.items() if role not in ("inlet", "outlet")},
                options=options, seed=None if seed is None else int(seed),
            )
            joined = G.graph[SURROUND_RECORD]
            logger.info(
                "Surround: the template run's %d vessels joined at %d open end(s): %d run on as trunks to "
                "new inlets and outlets, %d join the capillary bed, %d by a connector; %d left open "
                "(on faces the box does not grow past).",
                image.number_of_edges(), len(joined["joined"]), joined["trunks"], joined["bed_joins"],
                joined["connectors"], len(joined["left_open"]),
            )
        else:
            G = synthetic.generate_network(template, seed=None if seed is None else int(seed), options=options)
        if progress is not None:
            progress.step("generate_network", total=1)
        if G.number_of_edges() == 0:
            raise ValueError("The synthetic network has no vessels.")
        record = G.graph[GRAPH_TEMPLATE_RECORD]["generated"]
        logger.info(
            "Synthetic network: %d nodes, %d vessels (%d capillaries, %d tree vessels), "
            "%d independent loops, bed spacing %.1f um.",
            record["nodes"], record["edges"], record["capillaries"], record["tree_edges"],
            record["loops"], record["spacing_um"],
        )
        for note in record.get("notes", ()):
            logger.warning("Synthetic network: %s", note)
        stem = run_stem(settings)
        if stem is not None:
            graph_path = volume.output_dir / f"{stem}_graph.pkl"
            write_pickle_whole(G, graph_path)
            logger.info(f"Saved the synthetic network to: {graph_path}")
    _record_graph_voxel_metadata(G, volume.voxel_size_xyz, volume.voxel_size_zyx, None, None, None, None)
    return VesselNetwork(graph=G, volume=volume)


def _assign_synthetic_boundaries(settings: dict, network: VesselNetwork) -> BoundaryNodes:
    """The Boundaries stage of a synthetic run: the roles the generator
    marked on the nodes (``haemolynx.synthetic.boundary_nodes``) fill the
    run's lists, in place as the manual methods do."""
    from haemolynx.synthetic import boundary_nodes

    G = network.graph
    roles = boundary_nodes(G)
    for role in SYNTHETIC_ROLES:
        name = f"{role}_nodes"
        _reset_run_node_list(settings, name)
        settings[name].extend(roles[role])
    # The large-vessel network's own inlet and outlet are an image run's,
    # from its masks' stumps: nodes of another graph here.
    _reset_run_node_list(settings, "large_vessel_inlet_nodes")
    _reset_run_node_list(settings, "large_vessel_outlet_nodes")
    if not settings["inlet_nodes"] or not settings["outlet_nodes"]:
        raise ValueError(
            "The synthetic network has no inlet or no outlet to solve between: "
            "the template run has none to match."
        )
    resistance_node_pair = (settings["inlet_nodes"][0], settings["outlet_nodes"][0])
    logger.info(
        "Synthetic network: the generator chose %d inlet(s), %d outlet(s), %d arteriole and "
        "%d venule boundary node(s), %d large-arteriole and %d large-venule hand-over node(s).",
        len(settings["inlet_nodes"]), len(settings["outlet_nodes"]),
        len(settings["arteriole_boundary_nodes"]), len(settings["venule_boundary_nodes"]),
        len(settings["large_arteriole_boundary_nodes"]), len(settings["large_venule_boundary_nodes"]),
    )
    return BoundaryNodes(
        inlet_nodes=settings["inlet_nodes"],
        outlet_nodes=settings["outlet_nodes"],
        arteriole_boundary_nodes=settings["arteriole_boundary_nodes"],
        venule_boundary_nodes=settings["venule_boundary_nodes"],
        large_arteriole_boundary_nodes=settings["large_arteriole_boundary_nodes"],
        large_venule_boundary_nodes=settings["large_venule_boundary_nodes"],
        resistance_node_pair=resistance_node_pair,
        graph=G,
    )



def load_volume_for_skeletonise(settings: dict, input_format: str):
    """Image plus voxel metadata; skeletonisation is chosen after size is resolved.

    Public because the settings optimiser (``haemolynx.optimisation``) needs
    the same raw mask and voxel metadata a real run would skeletonise, without
    running ``skeletonise`` itself -- that also applies the *current*
    skeleton_* settings, which is exactly what the optimiser is deciding.

    Goes through io.load_2d's format-dispatching loader rather than
    calling the TIFF/H5 readers directly -- identical behaviour for an
    ordinary 3D input, and a genuinely 2D one is promoted to a
    single-slice volume (with a logged warning) instead of silently
    passing through 2D or raising. Reads settings["input_path"] /
    settings["image_axis_order"] exactly as before, so a 2D file reaches
    the pipeline through the same GUI input row every other file does.

    With both use_memmap_loading and memmap_directory set, this is the one
    load a later run against the same input can skip re-decompressing --
    see io.load_raw_volume_reusing_cache for the invalidation rules.

    With ``crop_on_import`` the image comes back cut to its box (io.crop):
    the cache keeps the whole volume, so changing the box reloads nothing.
    With ``segmented_image_downsample`` it comes back reduced (io.downsample),
    and the voxel size with it is then the reduced grid's, already resolved
    against the settings' override -- its status says so
    (:data:`RUN_GRID_SOURCE`). Read the run's voxel size from the two with
    :func:`run_voxel_size_xyz`, which handles both cases.
    """
    image, metadata_voxel_size, voxel_meta_status, _crop, downsample = _load_and_crop_volume(
        settings, input_format
    )
    if downsample is None:
        return image, metadata_voxel_size, voxel_meta_status
    voxel_size, source = resolve_voxel_size_xyz(
        metadata_voxel_size_xyz=metadata_voxel_size,
        metadata_status=voxel_meta_status,
        voxel_size_override_xyz=settings["voxel_size_override_xyz"],
        voxel_size_policy=settings["voxel_size_policy"],
    )
    status = {
        "source": RUN_GRID_SOURCE,
        "status": "complete",
        "resolved_from": source,
        "downsample": downsample.to_record(),
    }
    return image, downsample.scale_voxel_size_xyz(voxel_size), status


#: The ``voxel_meta_status["source"]`` :func:`load_volume_for_skeletonise`
#: gives with a voxel size it has already resolved and scaled to the reduced
#: grid, which :func:`run_voxel_size_xyz` then takes as it is.
RUN_GRID_SOURCE = "run_grid"


def run_voxel_size_xyz(
    settings: Mapping[str, Any], metadata_voxel_size, voxel_meta_status
) -> tuple[tuple[float, float, float], str]:
    """The run's voxel size ``(x, y, z)`` and where it came from, from what
    :func:`load_volume_for_skeletonise` returned: ``resolve_voxel_size_xyz``
    of the metadata and the settings' override and policy -- unless the
    loader resolved it already (a downsampled image, :data:`RUN_GRID_SOURCE`),
    when the voxel size is the reduced grid's and the override, which is the
    image's as read, must not replace it."""
    status = dict(voxel_meta_status or {})
    if status.get("source") == RUN_GRID_SOURCE:
        return tuple(float(v) for v in metadata_voxel_size), str(status.get("resolved_from", RUN_GRID_SOURCE))  # type: ignore[return-value]
    return resolve_voxel_size_xyz(
        metadata_voxel_size_xyz=metadata_voxel_size,
        metadata_status=voxel_meta_status,
        voxel_size_override_xyz=settings.get("voxel_size_override_xyz"),
        voxel_size_policy=settings.get("voxel_size_policy", "auto"),
    )


@dataclass(frozen=True)
class RunGridVolume:
    """The segmented image as :func:`skeletonise` reads it before it
    skeletonises: cut to the ``crop_on_import`` box and reduced by
    ``segmented_image_downsample``, with the voxel size of the grid that
    leaves, and the box and the reduction themselves, which whatever is read
    beside the image (the vessel masks, the raw image) is put through too."""

    image: np.ndarray
    #: ``(x, y, z)`` microns of the run's grid, resolved as ``skeletonise`` resolves it.
    voxel_size_xyz: tuple[float, float, float]
    voxel_size_source: str
    crop: CropBox | None
    downsample: Downsample | None


def load_volume_on_run_grid(settings: dict, input_format: str) -> RunGridVolume:
    """What :func:`skeletonise` reads, without skeletonising it: for the
    settings optimiser, whose trials must start from the run's mask and grid
    and confine the thick-vessel region as the run does
    (:func:`thick_vessel_restriction_mask` needs the box and the reduction)."""
    image, metadata_voxel_size, voxel_meta_status, crop, downsample = _load_and_crop_volume(
        settings, input_format
    )
    voxel_size, source = resolve_voxel_size_xyz(
        metadata_voxel_size_xyz=metadata_voxel_size,
        metadata_status=voxel_meta_status,
        voxel_size_override_xyz=settings["voxel_size_override_xyz"],
        voxel_size_policy=settings["voxel_size_policy"],
    )
    return RunGridVolume(
        image=image,
        voxel_size_xyz=_run_voxel_size_on_grid(voxel_size, downsample),
        voxel_size_source=source,
        crop=crop,
        downsample=downsample,
    )


#: The binarisation rule last decided on a whole segmented image, keyed on
#: the file's identity and the volume as read: deciding it reads every voxel
#: (a minute or two on a 17 G-voxel stack), and a run may load the image
#: more than once (the panel's checks, a preflight, the run itself).
_SEGMENTED_IMAGE_RULE: dict[str, tuple[tuple, Any]] = {}


def _segmented_image_binarization_rule(input_path: Any, image: np.ndarray, *, from_file=None):
    """:func:`io.load.mask_binarization_rule` of the whole segmented *image*
    read from *input_path*, remembered for the last file. *from_file*, when
    only a box of the image was read, gives the whole file's rule instead
    (:func:`io.load.h5_binarization_rule`)."""
    try:
        stat = Path(input_path).stat()
        key = (
            str(Path(input_path).resolve()), stat.st_size, stat.st_mtime_ns,
            tuple(image.shape), str(image.dtype),
        )
    except (OSError, TypeError):
        key = None
    cached = _SEGMENTED_IMAGE_RULE.get("rule")
    if key is not None and cached is not None and cached[0] == key:
        return cached[1]
    rule = from_file() if from_file is not None else mask_binarization_rule(image)
    if key is not None:
        _SEGMENTED_IMAGE_RULE["rule"] = (key, rule)
    return rule


def _h5_input_crop_box(settings: dict, input_format: str) -> CropBox | None:
    """The crop_on_import box a 3D H5 input image is read within, rather than
    read whole and cut (the AIA stacks: 9.4 GB read for a 5% box); None when
    it is read whole -- another format, no box or the whole image, or the
    low-RAM raw cache keeping the whole image for the next run."""
    if input_format != "h5" or (settings["use_memmap_loading"] and settings["memmap_directory"]):
        return None
    shape = io.h5_volume_shape(settings["input_path"], axis_order=io.axis_order_for_run(settings))
    if len(shape) != 3:
        return None
    box = crop_box_from_settings(settings, shape)
    return None if box is None or box.is_whole_volume else box


def _load_and_crop_volume(settings: dict, input_format: str):
    """:func:`load_volume_for_skeletonise`, with the box ``crop_on_import``
    cut (None while it is off) and the reduction ``segmented_image_downsample``
    made (None while it is ``none``), in that order: the box is voxel indices
    of the image as read, and the reduction's blocks are chosen from the run's
    voxel size (the metadata and the override resolved as the run resolves
    them), so a coarse z axis is reduced by less. The voxel size returned is
    the image's metadata as read, unscaled: what is resolved from it is the
    reduced grid's once scaled (:meth:`Downsample.scale_voxel_size_xyz`).
    A cut box comes back boolean, binarised by the whole image's rule."""
    if input_format not in {"tif", "tiff", "h5"}:
        raise ValueError("INPUT_FORMAT must be 'tif', 'tiff', or 'h5'.")
    box = _h5_input_crop_box(settings, input_format)
    if box is not None:
        # Only the box is read: the voxels cutting the whole image would keep.
        (
            image,
            voxel_size_x,
            voxel_size_y,
            voxel_size_z,
            voxel_meta_status,
        ) = io.load_3d_h5_with_voxel_size(
            settings["input_path"],
            axis_order=io.axis_order_for_run(settings),
            use_memmap=settings["use_memmap_loading"],
            memmap_directory=settings["memmap_directory"],
            box=box.slices,
        )
    elif settings["use_memmap_loading"] and settings["memmap_directory"]:
        (
            image,
            voxel_size_x,
            voxel_size_y,
            voxel_size_z,
            voxel_meta_status,
        ) = io.load_raw_volume_reusing_cache(
            settings["input_path"],
            input_format=input_format,
            axis_order=io.axis_order_for_run(settings),
            memmap_directory=settings["memmap_directory"],
        )
    else:
        (
            image,
            voxel_size_x,
            voxel_size_y,
            voxel_size_z,
            voxel_meta_status,
        ) = io.load_image_with_voxel_size_2d_aware(
            settings["input_path"],
            input_format=input_format,
            axis_order=io.axis_order_for_run(settings),
            use_memmap=settings["use_memmap_loading"],
            memmap_directory=settings["memmap_directory"],
        )
    metadata_voxel_size = (
        float(voxel_size_x),
        float(voxel_size_y),
        float(voxel_size_z),
    )
    if box is not None:
        # Only the box was read: binarised by the rule the whole file gives,
        # as a box cut from the whole image is (decided on the box alone, an
        # ilastik 1/2 box holding only background was solid vessel).
        logger.info("Crop on import, %s: %s", "input image", box.describe())
        rule = _segmented_image_binarization_rule(
            settings["input_path"],
            image,
            from_file=lambda: h5_binarization_rule(settings["input_path"]),
        )
        if rule is not None:
            image = np.asarray(rule(np.asarray(image)), dtype=bool)
        crop = box
    else:
        # A box is binarised as the whole image reads: decided on the box alone,
        # an ilastik 1/2 box holding only background was solid vessel.
        box = crop_box_from_settings(settings, image.shape)
        rule = None
        if box is not None and not box.is_whole_volume:
            rule = _segmented_image_binarization_rule(settings["input_path"], image)
        image, crop = crop_volume_and_box(image, settings, description="input image", binarise=rule)
    downsample = None
    if downsample_factor(settings.get("segmented_image_downsample")) > 1:
        resolved, _source = resolve_voxel_size_xyz(
            metadata_voxel_size_xyz=metadata_voxel_size,
            metadata_status=voxel_meta_status,
            voxel_size_override_xyz=settings["voxel_size_override_xyz"],
            voxel_size_policy=settings["voxel_size_policy"],
        )
        downsample = downsample_from_settings(
            settings, image.shape, io.voxel_size_zyx_from_xyz(tuple(float(v) for v in resolved))
        )
        if downsample is not None:
            # Binarised first: reduced as read, a 1/2-labelled mask (ilastik's)
            # kept a coarse voxel as vessel only where the whole block was --
            # every vessel came out eroded and the thin ones gone.
            binary = _to_binary_volume_for_skeletonization(
                image,
                use_memmap=settings["use_memmap_loading"],
                memmap_directory=settings["memmap_directory"],
            )
            reduced = downsample_volume(
                binary, downsample, reduce="majority", description="segmented image",
                use_memmap=settings["use_memmap_loading"],
                memmap_directory=settings["memmap_directory"],
            )
            if binary is not image and isinstance(binary, np.memmap):
                preprocessing.release_memmap_array(binary)
            image = reduced
    return image, metadata_voxel_size, voxel_meta_status, crop, downsample


#: Skeleton clean-up settings counted in voxels of the finest axis -- each a
#: physical length of that many voxels of the image as read.
SKELETON_VOXEL_SETTINGS: tuple[str, ...] = (
    "skeleton_closing_radius",
    "skeleton_bridge_gap_size",
    "skeleton_min_branch_length",
    "skeleton_max_bridge_distance",
    "skeleton_bundle_scan_size",
    "skeleton_bundle_hub_min_spacing",
)


def finest_spacing_ratio(source_voxel_size_zyx, grid_voxel_size_zyx) -> float:
    """How many voxels of the source's finest axis one voxel of the grid's
    finest axis spans: what a count of finest-axis voxels (the skeleton
    clean-up's) is divided by on the grid. Not the source's finest axis's
    own reduction: a coarse z reduced less can be the grid's finest axis
    (z 1.5 um, x and y 0.5 um at 4x: a 1.5 um grid voxel, not 2 um)."""
    return round(float(min(grid_voxel_size_zyx)) / float(min(source_voxel_size_zyx)), 9)


def skeleton_voxel_settings_on_grid(settings: Mapping[str, Any], downsample: Downsample | None) -> dict:
    """The :data:`SKELETON_VOXEL_SETTINGS` a run on *downsample*'s grid uses:
    each the same physical length as on the image as read, so divided by the
    reduction of the finest axis, rounded half up and never below 1 where it
    was set. Empty without a reduction. Counted on the coarse grid as set, a
    downsampled run bridged gaps and dropped pieces twice as long (at 2x): on
    E14.5 it built 2.5 points less of the full-resolution network and half
    again as many false junctions (2026-10-19)."""
    if downsample is None or downsample.is_identity:
        return {}
    factor = finest_spacing_ratio(downsample.source_voxel_size_zyx, downsample.voxel_size_zyx)
    scaled = {}
    for name in SKELETON_VOXEL_SETTINGS:
        value = settings.get(name)
        if value is None or isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            continue
        scaled[name] = max(1, int(np.floor(value / factor + 0.5)))
    if scaled:
        logger.info(
            "Downsample on import: skeleton clean-up voxel counts kept at their physical size "
            "on the %gx grid: %s.",
            factor, ", ".join(f"{name} {settings[name]} -> {v}" for name, v in scaled.items()),
        )
    return scaled


def missing_vessel_min_voxels_on_grid(settings: Mapping[str, Any], downsample: Downsample | None) -> int:
    """``missing_vessel_min_voxels``, voxels of the image as read, on
    *downsample*'s grid: as many as hold the same volume, rounded half up and
    never below 1. Counted on the coarse grid as set, the missing-vessel
    checks' noise floor was a piece of eight times the volume (at 2x)."""
    value = int(settings["missing_vessel_min_voxels"])
    if downsample is None or downsample.is_identity:
        return value
    block = int(np.prod(downsample.axis_factors_zyx, dtype=np.int64))
    return max(1, int(np.floor(value / block + 0.5)))


def skeleton_voxel_settings_from_grid(
    chosen: Mapping[str, Any], settings: Mapping[str, Any], downsample: Downsample | None
) -> dict:
    """:func:`skeleton_voxel_settings_on_grid` the other way: the
    :data:`SKELETON_VOXEL_SETTINGS` in *chosen*, counted on *downsample*'s
    grid, as the image as read counts them. One a run on that grid would
    have used for its setting in *settings* comes back as set; another is
    multiplied back up by the finest axis's reduction. For the settings
    optimiser, which searches on the run's grid. Empty without a reduction."""
    if downsample is None or downsample.is_identity:
        return {}
    factor = finest_spacing_ratio(downsample.source_voxel_size_zyx, downsample.voxel_size_zyx)
    on_grid = skeleton_voxel_settings_on_grid(settings, downsample)
    restored = {}
    for name in SKELETON_VOXEL_SETTINGS:
        value = chosen.get(name)
        if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        if name in on_grid and value == on_grid[name]:
            restored[name] = settings[name]
        else:
            restored[name] = int(round(value * factor))
    return restored


def _run_voxel_size_on_grid(voxel_size_xyz, downsample: Downsample | None):
    """A resolved ``(x, y, z)`` voxel size of the image as read, on the grid
    the run is on: scaled by *downsample* when there is one."""
    voxel_size = tuple(float(v) for v in voxel_size_xyz)
    return voxel_size if downsample is None else downsample.scale_voxel_size_xyz(voxel_size)


def _vessel_mask_arguments(settings: Mapping[str, Any], mask_role: str) -> dict[str, Any]:
    """``io.vessel_mask_arguments``, loading the role's masks only while its
    own files are in use (``graph.vessel_mask_files_in_use``): a mask switch
    left ticked under an unticked ``automated_vessel_assignment`` loads
    nothing, and small masks cut from the large ones
    (:func:`_split_large_vessel_masks`) are read from no file."""
    return {
        **io.vessel_mask_arguments(settings, mask_role),
        "enabled": graph.vessel_mask_files_in_use(settings, mask_role),
        # Read onto disk and cleaned a slab at a time under the low-RAM option.
        "use_memmap": bool(settings.get("use_memmap_loading", False)),
        "memmap_directory": settings.get("memmap_directory"),
    }


def _vessel_mask_sources(settings: Mapping[str, Any]) -> dict[str, tuple]:
    """Each role's masks as the settings describe them: the loader's arguments
    (files, ilastik, voxel size, load-time clean-up), the width they are split
    at, and whether the run reads every mask as one global mask. The settings
    sit on the Boundaries tab but the masks load in ``build_network``, so a
    network keeps this (`VesselNetwork.mask_sources`) for a run resumed there
    to tell masks it can keep from ones it must load again."""
    common = (
        io.vessel_masks_are_global(settings),
        graph.large_vessel_mask_split_diameter_um(settings),
    )
    return {
        role: common
        + tuple(sorted((name, repr(value)) for name, value in _vessel_mask_arguments(settings, role).items()))
        for role in graph.boundaries.VESSEL_MASK_SWITCHES
    }


def _split_large_vessel_masks(
    settings: Mapping[str, Any], large_arteriole_mask, large_venule_mask, voxel_size_xyz, segmentation
) -> tuple:
    """``(large arteriole, large venule, small arteriole, small venule)``: the
    loaded large masks cut by width (``io.split_large_vessel_masks_by_diameter``),
    for a run whose small masks are cut from them
    (``graph.small_vessel_masks_split_from_large``). The widths are read on
    *segmentation*, the run's segmented image on the masks' grid, binary or
    read as the run binarises it (:class:`_SegmentationBoxes`)."""
    if large_arteriole_mask is None:
        return large_arteriole_mask, large_venule_mask, None, None
    return io.split_large_vessel_masks_by_diameter(
        large_arteriole_mask,
        large_venule_mask,
        diameter_um=graph.large_vessel_mask_split_diameter_um(settings),
        voxel_size_xyz=voxel_size_xyz,
        segmentation=segmentation,
        dilation_um=float(settings.get("large_vessel_mask_dilation_microns") or 0.0),
        use_memmap=bool(settings.get("use_memmap_loading", False)),
        memmap_directory=settings.get("memmap_directory"),
    )


def thick_vessel_restriction_mask(
    binary_image: np.ndarray,
    settings: dict,
    voxel_size_xyz,
    crop: CropBox | None = None,
    downsample: Downsample | None = None,
) -> np.ndarray | None:
    """Union of the configured large/small vessel masks, per
    ``skeleton_thick_vessel_restrict_to_mask`` -- or ``None`` for no
    restriction (the default, and the only path taken while that setting
    is ``"off"``, so an ordinary run pays nothing extra for this feature).

    Reuses ``io.load_and_validate_vessel_masks``/``io.vessel_mask_arguments``
    directly on the full settings dict -- the same functions
    ``build_network`` uses for boundary assignment -- rather than threading
    masks through ``SkeletonisedVolume``: every stage that needs a vessel
    mask already reloads it independently rather than trusting an earlier
    stage's state, and this keeps the checkpoint/resume format untouched.

    *binary_image* is the already-binarised segmented image (not just its
    shape) -- needed to log a per-mask alignment warning when a loaded
    mask's own voxels mostly do not fall on it at all; see
    preprocessing.diagnose_mask_restriction_alignment for what that
    measures and why a low fraction matters here specifically.

    *crop* is the ``crop_on_import`` box *binary_image* was cut to, which the
    masks are cut to as well, and *downsample* how it was reduced after that
    (``segmented_image_downsample``), which they are too.

    Public for the settings optimiser, whose thick-vessel trials are confined
    to the same voxels (see :func:`load_volume_on_run_grid`).
    """
    choice = str(settings["skeleton_thick_vessel_restrict_to_mask"]).strip().lower()
    if choice == "off":
        return None
    image_shape = binary_image.shape
    # Under low-RAM the union is made on disk, a slab of planes at a time.
    restriction = (
        preprocessing.new_memmap_array(image_shape, bool, directory=settings["memmap_directory"])
        if settings["use_memmap_loading"]
        else np.zeros(image_shape, dtype=bool)
    )

    def add(mask: np.ndarray) -> None:
        nonlocal restriction
        if not settings["use_memmap_loading"]:
            restriction |= mask
            return
        from haemolynx.preprocessing.memmap_support import slab_step

        step = slab_step(image_shape)
        for z0 in range(0, image_shape[0], step):
            restriction[z0:z0 + step] = np.asarray(restriction[z0:z0 + step]) | np.asarray(
                mask[z0:z0 + step], dtype=bool
            )

    roles = {"large": ("large",), "small": ("small",), "both": ("large", "small")}.get(choice, ())
    load = dict(image_shape=image_shape, main_voxel_size_xyz=voxel_size_xyz, crop=crop, downsample=downsample)
    # Small masks cut from the large ones come with them.
    split = graph.small_vessel_masks_split_from_large(settings)
    pairs: dict[str, tuple] = {}
    if "large" in roles or (split and "small" in roles):
        pairs["large"] = tuple(
            io.load_and_validate_vessel_masks(**_vessel_mask_arguments(settings, "large"), **load)[:2]
        )
        if split:
            cut = _split_large_vessel_masks(settings, *pairs["large"], voxel_size_xyz, binary_image)
            pairs = {"large": cut[:2], "small": cut[2:]}
    if "small" in roles and "small" not in pairs:
        pairs["small"] = tuple(
            io.load_and_validate_vessel_masks(**_vessel_mask_arguments(settings, "small"), **load)[:2]
        )
    role_masks: dict[str, np.ndarray] = {}
    for role in roles:
        for vessel, mask in zip(("arteriole", "venule"), pairs[role]):
            if mask is not None:
                add(mask)
                role_masks[f"{role}_{vessel}_mask"] = mask

    warn_below = float(settings["skeleton_thick_vessel_restrict_to_mask_warn_below"])
    for report in preprocessing.diagnose_mask_restriction_alignment(
        role_masks, binary_image
    ):
        message = preprocessing.format_mask_restriction_alignment_report(report)
        if report["overlap_fraction"] < warn_below:
            logger.warning(message)
        else:
            logger.info(message)

    return restriction


def _skeletonize_tile_halo_voxels(settings: dict, voxel_size_zyx) -> int:
    """``skeletonize_tile_halo_um`` in Z voxels, or 0 when tiling is off.

    Shared by both of this stage's skeletonize calls (the initial one and
    the re-skeletonize inside ``preprocess_skeleton_for_graph``) so the
    physical-to-voxel conversion is not repeated -- and cannot silently
    drift -- between them.
    """
    if not settings["skeletonize_tile_large_components"]:
        return 0
    return round(float(settings["skeletonize_tile_halo_um"]) / voxel_size_zyx[0])


def _thick_region_arguments(
    settings: dict,
    binary: np.ndarray,
    voxel_size_xyz,
    crop: CropBox | None = None,
    downsample: Downsample | None = None,
) -> dict:
    """What decides the thick-vessel region: shared by skeletonisation and by
    :func:`thick_vessel_mask_from_image`, so the two find the same voxels."""
    return {
        **thick_region_settings(settings),
        "voxel_size_zyx": io.voxel_size_zyx_from_xyz(tuple(float(v) for v in voxel_size_xyz)),
        "restrict_thick_to_mask": thick_vessel_restriction_mask(
            binary, settings, voxel_size_xyz, crop, downsample
        ),
        "use_memmap": settings["use_memmap_loading"],
        "memmap_directory": settings["memmap_directory"],
    }


def thick_region_settings(settings: Mapping[str, Any]) -> dict[str, Any]:
    """The settings that decide the thick-vessel region, as
    ``preprocessing.thick_vessel_region`` and ``skeletonize_thickness_gated``
    take them (the voxel size and the restriction, not settings, aside)."""
    return {
        "min_radius_um": float(settings["skeleton_thick_vessel_min_radius_um"]),
        "fill_mask_holes": bool(settings["skeleton_fill_mask_holes_before_thickness"]),
        "wall_absorption_um": settings["skeleton_thick_vessel_wall_absorption_um"],
    }


def thick_vessel_refinement_arguments(settings: Mapping[str, Any]) -> dict[str, Any]:
    """The rest of what ``skeletonize_thickness_gated`` takes from the
    settings: how the thin arms are joined to the thick tree, and which tree
    is drawn. With :func:`thick_region_settings`, what a run's thick
    skeleton and a settings-optimiser trial's are both made with."""
    return {
        "flake_filter_um": settings["skeleton_thick_vessel_flake_filter_um"],
        "max_bridge_radius_multiple": settings["skeleton_thick_vessel_max_bridge_radius_multiple"],
        "max_bridge_distance_um": settings["skeleton_thick_vessel_max_bridge_distance_um"],
        "bridge_radius_smoothing_um": float(settings["skeleton_thick_vessel_bridge_radius_smoothing_um"]),
        "large_stack_tree": large_stack_thick_tree(settings),
    }


def large_stack_thick_tree(settings: Mapping[str, Any]) -> bool:
    """Whether the thick vessels' centreline tree is drawn for a large stack
    (``skeleton_thick_vessel_large_stack_tree``, which applies under the
    low-RAM option only): skeletonisation, graph building and post
    processing all ask this, so they draw the same tree."""
    return bool(settings.get("use_memmap_loading")) and bool(
        settings.get("skeleton_thick_vessel_large_stack_tree")
    )


def thick_vessel_region_cache(
    settings: dict, output_dir: Path | None = None
) -> preprocessing.ThickVesselRegionCache:
    """Where a run keeps the thick-vessel region it found: beside its saved
    skeleton in *output_dir* (the run's ``output_dir`` when not given), one
    entry per run output. Finding the region is about four minutes on E14.5,
    and a run with the same mask and settings as the last reads it back
    instead (:mod:`~haemolynx.preprocessing.thick_vessel_region_cache`).
    """
    directory = Path(settings["output_dir"]) if output_dir is None else Path(output_dir)
    stem = run_stem(settings)
    return preprocessing.ThickVesselRegionCache(directory / f"{stem}_thick_vessel_region")


def _cached_thick_vessel_region(
    cache: preprocessing.ThickVesselRegionCache | None, binary: np.ndarray, arguments: dict
) -> tuple[str, bool, np.ndarray | None]:
    """``(key, found, region)``: the entry *cache* holds for *binary* under
    the region-deciding *arguments* (:func:`_thick_region_arguments`)."""
    key = preprocessing.thick_vessel_region_key(
        binary,
        min_radius_um=arguments["min_radius_um"],
        voxel_size_zyx=arguments["voxel_size_zyx"],
        fill_mask_holes=arguments["fill_mask_holes"],
        wall_absorption_um=arguments["wall_absorption_um"],
        restrict_thick_to_mask=arguments["restrict_thick_to_mask"],
    )
    if cache is None:
        return key, False, None
    found, region = cache.lookup(
        key, use_memmap=bool(arguments.get("use_memmap")), memmap_directory=arguments.get("memmap_directory")
    )
    return key, found, region


def thick_vessel_mask_from_image(
    settings: dict,
    image: np.ndarray,
    voxel_size_xyz,
    crop: CropBox | None = None,
    *,
    cache: preprocessing.ThickVesselRegionCache | None = None,
    downsample: Downsample | None = None,
) -> np.ndarray | None:
    """The thick-vessel region skeletonisation found in *image*, found again.

    For a run that needs the region but loaded its skeleton instead of making
    it: ``skeletonise`` with ``do_skeletonize`` off, and post processing on a
    resume with no stored stage outputs. None when thickness-gated
    skeletonisation is off, or finds no thick vessel. *crop* is the
    ``crop_on_import`` box *image* was cut to, *downsample* how it was reduced
    after that. With *cache*
    (:func:`thick_vessel_region_cache`), the region a previous run found from
    the same mask and settings is read back, and one found here is kept.
    """
    return _thick_region_and_key(
        settings, image, voxel_size_xyz, crop, cache=cache, downsample=downsample
    )[0]


def _thick_region_and_key(
    settings: dict,
    image: np.ndarray,
    voxel_size_xyz,
    crop: CropBox | None = None,
    *,
    cache: preprocessing.ThickVesselRegionCache | None = None,
    downsample: Downsample | None = None,
    carried: np.ndarray | None = None,
) -> tuple[np.ndarray | None, str | None]:
    """:func:`thick_vessel_mask_from_image`, and the key the region is kept
    under (``thick_vessel_region_key``: the mask and the settings deciding
    the region), None with thickness-gated skeletonisation off. *carried*, a
    region of *image*'s shape, is used when none is kept rather than finding
    it again: a resumed run's (``PipelineResume.thick_vessel_mask``), which a
    loaded ``.haemorun`` carries, the region its skeleton was made with."""
    if not settings.get("use_thick_vessel_skeletonisation"):
        return None, None
    binary = _to_binary_volume_for_skeletonization(
        image,
        use_memmap=settings["use_memmap_loading"],
        memmap_directory=settings["memmap_directory"],
    )
    arguments = _thick_region_arguments(settings, binary, voxel_size_xyz, crop, downsample)
    key, found, region = _cached_thick_vessel_region(cache, binary, arguments)
    if found:
        pass
    elif carried is not None and tuple(np.shape(carried)) == tuple(binary.shape):
        logger.info("No thick-vessel region is kept for this mask and these settings: using the resumed run's.")
        region = np.asarray(carried, dtype=bool) if np.any(carried) else None
    else:
        # Without it graph building makes no thick-vessel bridges: the graph
        # of a run resumed on another machine was not the run's it resumed.
        logger.info("No thick-vessel region is kept for this mask and these settings: finding it.")
        region = preprocessing.thick_vessel_region(binary, **arguments)
        if cache is not None:
            cache.store(key, region, binary.shape)
    preprocessing.release_superseded(binary, image, keep=image)
    return region, key


def _skeletonize_loaded_mask(
    image,
    settings: dict,
    voxel_size_xyz,
    crop: CropBox | None = None,
    cache: preprocessing.ThickVesselRegionCache | None = None,
    downsample: Downsample | None = None,
    centreline: list | None = None,
    region_key: list | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Lee on the whole mask, or a thickness-gated tree when the GUI toggle is on.

    Returns ``(skeleton, thick_vessel_mask)`` -- the mask is None on the
    plain-Lee path, where there is no fat/thin split at all. *crop* is the
    ``crop_on_import`` box *image* was cut to, *downsample* how it was reduced
    after that. With *cache*, the thick-vessel
    region is read back from the run that last found it in this mask with
    these settings, or kept once found. *centreline*, a list, is given the
    thick region's centreline tree as drawn (its voxels), when one is;
    *region_key*, a list, the key the thick region is kept under.
    """
    voxel_size_zyx = io.voxel_size_zyx_from_xyz(tuple(float(v) for v in voxel_size_xyz))
    if settings["use_thick_vessel_skeletonisation"]:
        binary = _to_binary_volume_for_skeletonization(
            image,
            use_memmap=settings["use_memmap_loading"],
            memmap_directory=settings["memmap_directory"],
        )
        arguments = _thick_region_arguments(settings, binary, voxel_size_xyz, crop, downsample)
        key, found, region = _cached_thick_vessel_region(cache, binary, arguments)
        if region_key is not None:
            region_key.append(key)
        if found and region is None:
            # Found before: there is none. No volume of False for it.
            region = np.broadcast_to(np.False_, binary.shape)
        skeleton, thick = preprocessing.skeletonize_thickness_gated(
            binary,
            **arguments,
            **thick_vessel_refinement_arguments(settings),
            return_thick_mask=True,
            tile_large_components=settings["skeletonize_tile_large_components"],
            tile_max_voxels=settings["skeletonize_tile_max_voxels"],
            tile_halo_voxels=_skeletonize_tile_halo_voxels(settings, voxel_size_zyx),
            thick_region=region if found else None,
            centreline_voxels=centreline,
        )
        if cache is not None and not found:
            cache.store(key, thick, binary.shape)
        return skeleton, thick
    return (
        _skeletonize_loaded_volume(
            image,
            use_memmap=settings["use_memmap_loading"],
            memmap_directory=settings["memmap_directory"],
            tile_large_components=settings["skeletonize_tile_large_components"],
            tile_max_voxels=settings["skeletonize_tile_max_voxels"],
            tile_halo_voxels=_skeletonize_tile_halo_voxels(settings, voxel_size_zyx),
        ),
        None,
    )


def segmentation_cleanup_arguments(settings: Mapping[str, Any]) -> dict[str, Any]:
    """The ``segmentation_cleanup_*`` settings as
    ``preprocessing.clean_segmented_mask_for_skeletonisation`` takes them --
    for skeletonising and for the settings optimiser's trials alike."""
    return prefixed_arguments(
        settings, "segmentation_cleanup_",
        parameters_of(preprocessing.clean_segmented_mask_for_skeletonisation),
    )


def segmentation_cleanup_in_effect(settings: Mapping[str, Any]) -> bool:
    """Whether a run with *settings* cleans the segmented image before it
    skeletonises it: ``segmentation_cleanup`` on and at least one step."""
    arguments = segmentation_cleanup_arguments(settings)
    return bool(settings.get("segmentation_cleanup")) and any(
        arguments.get(flag) for flag in _SEGMENTATION_CLEANUP_STEPS
    )


def skeleton_cleanup_arguments(
    settings: Mapping[str, Any], downsample: Downsample | None = None
) -> dict[str, Any]:
    """What ``preprocessing.preprocess_skeleton_for_graph`` takes from the
    settings: the ``skeleton_*`` settings, their voxel counts on
    *downsample*'s grid (:func:`skeleton_voxel_settings_on_grid`), the
    component percentage as a fraction, and the bridge gates graph building
    shares. Not the volumes, nor the low-RAM and tiling choices, which give
    the same skeleton either way. For skeletonising and for the settings
    optimiser's trials alike."""
    return {
        **prefixed_arguments(
            {**settings, **skeleton_voxel_settings_on_grid(settings, downsample)},
            "skeleton_",
            parameters_of(preprocessing.preprocess_skeleton_for_graph),
        ),
        "min_component_fraction": settings["skeleton_min_component_percent"] / 100.0,
        "bridge_require_mask_support": settings["bridge_require_mask_support"],
        "bridge_max_background_gap_um": settings["bridge_max_background_gap_um"],
        "bridge_min_mask_fraction": settings["bridge_min_mask_fraction"],
    }


def centreline_smoothing_arguments(settings: Mapping[str, Any]) -> dict[str, Any]:
    """The centreline smoothing settings as ``graph.smooth_graph_centrelines``
    (and ``smooth_straight_steps``) take them."""
    return {
        "method": settings["centreline_smoothing_method"],
        "iterations": settings["centreline_smoothing_iterations"],
        "max_deviation": settings["centreline_max_deviation"],
    }


def _mask_skeleton_is_made_from(
    settings: dict, image: np.ndarray, voxel_size_xyz
) -> tuple[np.ndarray, np.ndarray | None]:
    """``(image, raw_segmented_image)``: the segmented image as skeletonising
    reads it, after the ``segmentation_cleanup_*`` steps that are on, and the
    binarised image from before them (None when none ran, *image* then
    returned as it is).

    A run that makes its skeleton and one that loads it both go through
    here, so graph building reads the same mask either way -- and the
    thick-vessel region, cached under a digest of this mask, is found again.
    """
    cleanup_kwargs = segmentation_cleanup_arguments(settings)
    if not segmentation_cleanup_in_effect(settings):
        return image, None
    logger.info(
        "Segmentation cleanup: fill_cavities=%s remove_whiskers=%s "
        "split_narrow_necks=%s close_gaps=%s reconnect=%s smooth=%s "
        "remove_small=%s",
        cleanup_kwargs.get("fill_cavities"),
        cleanup_kwargs.get("remove_whiskers"),
        cleanup_kwargs.get("split_narrow_necks"),
        cleanup_kwargs.get("close_gaps"),
        cleanup_kwargs.get("reconnect_gaps"),
        cleanup_kwargs.get("smooth_surfaces"),
        cleanup_kwargs.get("remove_small_volumes"),
    )
    binary_image = _to_binary_volume_for_skeletonization(
        image,
        use_memmap=settings["use_memmap_loading"],
        memmap_directory=settings["memmap_directory"],
    )
    return preprocessing.clean_segmented_mask_for_skeletonisation(
        binary_image,
        voxel_size_zyx=io.voxel_size_zyx_from_xyz(tuple(float(v) for v in voxel_size_xyz)),
        use_memmap=settings["use_memmap_loading"],
        memmap_directory=settings["memmap_directory"],
        **cleanup_kwargs,
    )


def _crop_record(crop: CropBox | None) -> dict | None:
    """The ``crop_on_import`` box a skeleton was cut to, as kept beside it:
    its voxel corners, or None for the whole image."""
    if crop is None or crop.is_whole_volume:
        return None
    return {"start_zyx": [int(v) for v in crop.start_zyx], "stop_zyx": [int(v) for v in crop.stop_zyx]}


def saved_skeleton_grid(crop: CropBox | None, downsample: Downsample | None) -> dict:
    """The grid a skeleton was made on, as its record keeps it (the
    ``downsample`` and ``crop`` of ``{stem}_voxel_size.json``): what a run
    loading it checks against its own (:func:`_check_saved_skeleton_grid`).
    The panel writes it with a skeleton it saves for a resumed run too."""
    return {
        "downsample": None if downsample is None else downsample.to_record(),
        "crop": _crop_record(crop),
    }


def _saved_skeleton_record(path: Path, skeleton_path: Path) -> dict:
    """What a run making its skeleton kept beside it (``{stem}_voxel_size.json``);
    empty when there is none, it cannot be read, or it is older than the
    skeleton: written by another run than the one that made it, as when the
    panel writes a skeleton alone for a resumed run
    (``gui.stage_checkpoints.ensure_skeleton_artefact``)."""
    try:
        if Path(path).stat().st_mtime_ns < Path(skeleton_path).stat().st_mtime_ns:
            return {}
        record = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}
    return record if isinstance(record, dict) else {}


def _check_saved_skeleton_grid(
    skeleton_path: Path, skeleton_shape, image_shape, record: Mapping[str, Any],
    crop: CropBox | None, downsample: Downsample | None,
) -> None:
    """Refuse a saved skeleton made on another grid than the image this run
    reads: another shape, or the same shape cut from another box or reduced
    by other blocks (a box moved, or a larger box reduced), which the shape
    alone let through. A record from before the box was kept is judged by
    its shape and reduction."""
    def blocks(factors) -> list[int] | None:
        factors = None if factors is None else [int(b) for b in factors]
        return None if factors is None or max(factors) <= 1 else factors

    saved_blocks = blocks((record.get("downsample") or {}).get("axis_factors_zyx"))
    if tuple(skeleton_shape) != tuple(image_shape):
        problem = f"is {tuple(skeleton_shape)} voxels but the image is {tuple(image_shape)}"
    elif "crop" in record and record["crop"] != _crop_record(crop):
        problem = f"was cut from the box {record['crop']}, this run from {_crop_record(crop)}"
    elif "downsample" in record and saved_blocks != blocks(
        None if downsample is None else downsample.axis_factors_zyx
    ):
        problem = "was reduced by other blocks than this run's"
    else:
        return
    raise ValueError(
        f"The saved skeleton {skeleton_path} {problem}: it was made with a different "
        "crop_on_import box or segmented_image_downsample, or none. Turn do_skeletonize "
        "back on to make it again."
    )


def _save_skeleton(path: Path, skeleton: np.ndarray) -> None:
    """Save *skeleton* at *path* whole or not at all: written to a ``.part``
    file and moved over *path* once complete, so a run stopped while saving
    leaves the last skeleton rather than half of one."""
    part = path.with_name(f"{path.name}.part")
    with open(part, "wb") as handle:
        np.save(handle, skeleton)
    os.replace(part, path)


def _load_saved_skeleton(path: Path, settings: Mapping[str, Any]) -> np.ndarray:
    """The skeleton a run making it saved at *path*. Under the low-RAM option
    a disk-backed copy, read a slab of planes at a time, not *path* mapped: on
    Windows a mapped file cannot be written again, so while a loaded skeleton
    was held (the panel keeps a run's layers) the next run making one failed to
    save it. A file that will not load is reported as one to make again."""
    from haemolynx.preprocessing.memmap_support import map_by_slab

    try:
        if not settings["use_memmap_loading"]:
            return np.load(path)
        saved = np.load(path, mmap_mode="r")
        mapping = getattr(saved, "_mmap", None)
        try:
            copy = preprocessing.new_memmap_array(
                saved.shape, saved.dtype, directory=settings["memmap_directory"]
            )
            return map_by_slab(saved, lambda slab: slab, copy)
        finally:
            del saved
            if mapping is not None:
                mapping.close()
    except (OSError, ValueError, EOFError) as error:
        raise ValueError(
            f"The saved skeleton {path} cannot be read ({error}). Turn do_skeletonize "
            "back on to make it again."
        ) from error


#: The steps :func:`skeletonise` reports making a skeleton: each one a point
#: a cancelled panel run stops at, where it used to finish the whole stage.
SKELETONISE_STEPS = ("Image read", "Skeletonised", "Skeleton cleaned")


def _warn_if_made_otherwise(
    skeleton_path: Path, record: Mapping[str, Any], mask_digest: str, region_key: str | None
) -> None:
    """Warn that a loaded skeleton was made from another mask, or with other
    thick-vessel settings, than this run reads: graph building then reads a
    mask the skeleton was not made from. A record from before these were kept
    says nothing."""
    remake = "Turn do_skeletonize back on to make it again."
    if record.get("mask_digest") not in (None, mask_digest):
        logger.warning(
            "The saved skeleton %s was made from another segmented mask than this run "
            "reads (the image or its segmentation clean-up changed since). %s",
            skeleton_path, remake,
        )
    elif "thick_region_key" in record and record["thick_region_key"] != region_key:
        logger.warning(
            "The saved skeleton %s was made with other thick-vessel settings than this "
            "run's (thickness-gated skeletonisation, its radius, hole fill, wall or "
            "restriction to the vessel masks). %s",
            skeleton_path, remake,
        )


def skeletonise(
    settings: dict,
    inputs: SegmentedInputs,
    *,
    progress: StageProgress | None = None,
    thick_region: np.ndarray | None = None,
):
    """Load the mask, resolve its voxel size, and skeletonise it.

    *progress* hears each of :data:`SKELETONISE_STEPS` (one step for a
    skeleton loaded instead). *thick_region* is the region a resumed run
    carries, used for a loaded skeleton when none is kept beside it."""

    def step(label: str, total: int, next_step: str | None = None) -> None:
        if progress is not None:
            progress.step(label, total=total, next_step=next_step)

    if inputs.synthetic:
        return _synthetic_volume(settings, inputs)
    output_dir = inputs.output_dir
    input_format = inputs.input_format
    # 1) Load image and skeletonize.
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
    skeleton_path = output_dir / f"{run_stem(settings)}_skeleton.npy"
    voxel_meta_path = output_dir / f"{run_stem(settings)}_voxel_size.json"
    projection_path = settings["plot_dir"] / "skeleton_projection.png"
    # The skeleton's figures are plots like any other (visualize_results): the
    # panel starts with them off, and its working folder may be one nobody
    # can write to.
    plotting = bool(settings["visualize_results"])
    if plotting:
        Path(settings["plot_dir"]).mkdir(parents=True, exist_ok=True)

    # A loaded skeleton's image is cleaned as a fresh run's is; the
    # thick-vessel region is kept beside the skeleton, so a loaded skeleton
    # gets it back when the mask and settings are the run's that found it,
    # and finds it again when they are not.
    thick_vessel_mask: np.ndarray | None = None
    raw_segmented_image: np.ndarray | None = None
    region_cache = thick_vessel_region_cache(settings, output_dir)
    thick_centreline: list[np.ndarray] = []

    if settings["do_skeletonize"]:
        image, metadata_voxel_size, voxel_meta_status, crop, downsample = _load_and_crop_volume(
            settings, input_format
        )
        voxel_size, voxel_size_source = resolve_voxel_size_xyz(
            metadata_voxel_size_xyz=metadata_voxel_size,
            metadata_status=voxel_meta_status,
            voxel_size_override_xyz=settings["voxel_size_override_xyz"],
            voxel_size_policy=settings["voxel_size_policy"],
        )
        # The image as read was resolved; the run is on the reduced grid. The
        # voxel size kept beside the skeleton is the source's, resolved again
        # (and scaled again) when the skeleton is loaded.
        source_voxel_size = voxel_size
        voxel_size = _run_voxel_size_on_grid(voxel_size, downsample)
        logger.info(
            "Voxel-size resolution: "
            f"source={voxel_size_source}, "
            f"metadata_status={voxel_meta_status.get('status')}, "
            f"metadata={metadata_voxel_size}, "
            f"final={voxel_size}"
        )
        if settings["use_thick_vessel_skeletonisation"]:
            logger.info(
                "Thickness-gated skeletonisation: "
                "centreline tree on fat vessels, Lee on the rest"
            )

        image, raw_segmented_image = _mask_skeleton_is_made_from(settings, image, voxel_size)

        # Binarised once and shared: skeletonisation and the reconnection
        # step's segmentation mask both need the same mask, and each call
        # used to rebuild it -- a whole extra volume every time. A pass-
        # through (no copy) when cleanup already left `image` boolean.
        binary = _to_binary_volume_for_skeletonization(
            image,
            use_memmap=settings["use_memmap_loading"],
            memmap_directory=settings["memmap_directory"],
        )
        made_from = preprocessing.mask_digest(binary)
        step(SKELETONISE_STEPS[0], len(SKELETONISE_STEPS), "Skeletonising")
        region_key: list[str] = []
        skeleton, thick_vessel_mask = _skeletonize_loaded_mask(
            binary, settings, voxel_size, crop, cache=region_cache, downsample=downsample,
            centreline=thick_centreline, region_key=region_key,
        )
        step(SKELETONISE_STEPS[1], len(SKELETONISE_STEPS), "Cleaning the skeleton")

        # Purely diagnostic: off by default, and never changes `skeleton` even
        # when on -- see preprocessing.thick_vessel_braid_guard for what this
        # looks for and why.
        if (
            thick_vessel_mask is not None
            and settings["detect_thick_vessel_braiding"]
        ):
            braided = preprocessing.detect_braided_thick_vessel_components(
                thick_vessel_mask,
                skeleton,
                voxel_size_zyx=io.voxel_size_zyx_from_xyz(
                    tuple(float(v) for v in voxel_size)
                ),
                braid_factor_limit=float(settings["thick_vessel_braid_factor_limit"]),
                min_occupied_slices=int(
                    settings["thick_vessel_braid_min_occupied_slices"]
                ),
                low_memory=bool(settings["use_memmap_loading"]),
            )
            if braided:
                logger.warning(preprocessing.format_braided_thick_vessel_report(braided))

        preprocessing.log_skeleton_connectivity_stats(
            "raw",
            skeleton,
            component_connectivity=settings["skeleton_component_connectivity"],
            use_memmap=settings["use_memmap_loading"],
            memmap_directory=settings["memmap_directory"],
        )
        if plotting:
            visualization.visualize_skeleton(skeleton, save_path=settings["plot_dir"] / "raw_skeleton.png")

        # The `skeleton_*` settings are this function's parameters with a
        # prefix, so they go in as a group; the percentage, the mask (not a
        # scalar setting) and the bridge gates graph building shares are the
        # exceptions.
        tile_halo_voxels = _skeletonize_tile_halo_voxels(
            settings, io.voxel_size_zyx_from_xyz(tuple(float(v) for v in voxel_size))
        )
        raw_skeleton = skeleton
        skeleton = preprocessing.preprocess_skeleton_for_graph(
            raw_skeleton,
            **skeleton_cleanup_arguments(settings, downsample),
            voxel_size_zyx=io.voxel_size_zyx_from_xyz(tuple(float(v) for v in voxel_size)),
            segmentation_mask=binary,
            use_memmap=settings["use_memmap_loading"],
            memmap_directory=settings["memmap_directory"],
            tile_large_components=settings["skeletonize_tile_large_components"],
            tile_max_voxels=settings["skeletonize_tile_max_voxels"],
            tile_halo_voxels=tile_halo_voxels,
        )
        # Volume-sized temp files under use_memmap that nothing reads from
        # here on; `image` (returned below) is never one of them.
        preprocessing.release_superseded(raw_skeleton, skeleton, keep=image)
        preprocessing.release_superseded(binary, skeleton, keep=image)
        preprocessing.log_skeleton_connectivity_stats(
            "cleaned",
            skeleton,
            component_connectivity=settings["skeleton_component_connectivity"],
            use_memmap=settings["use_memmap_loading"],
            memmap_directory=settings["memmap_directory"],
        )
        
        step(SKELETONISE_STEPS[2], len(SKELETONISE_STEPS))
        _save_skeleton(skeleton_path, skeleton)
        voxel_meta_path.write_text(
            json.dumps(
                {
                    "voxel_size": source_voxel_size,
                    "run_voxel_size": voxel_size,
                    "voxel_size_source": voxel_size_source,
                    "voxel_metadata_status": voxel_meta_status,
                    "voxel_size_policy": settings["voxel_size_policy"],
                    "voxel_size_override_xyz": settings["voxel_size_override_xyz"],
                    # For the record: the saved skeleton is on the reduced grid,
                    # cut to this box: a run loading it checks it is its own.
                    **saved_skeleton_grid(crop, downsample),
                    # And made from this mask, with these thick-vessel settings.
                    "mask_digest": made_from,
                    "thick_region_key": region_key[0] if region_key else None,
                    # Which thick-vessel tree drew it: the region's key does not
                    # say, and a run loading it bridges to the tree it would draw.
                    **({"large_stack_thick_tree": large_stack_thick_tree(settings)}
                       if settings["use_thick_vessel_skeletonisation"] else {}),
                }
            )
        )
        logger.info(f"Saved skeleton to: {skeleton_path}")
    else:
        skeleton = _load_saved_skeleton(skeleton_path, settings)
        step(
            "Skeleton loaded", 1,
            "Finding the thick-vessel region" if settings["use_thick_vessel_skeletonisation"] else None,
        )
        # The image as the run reads it: cut to the crop box and reduced by the
        # downsample, as the skeleton was when it was made.
        image, metadata_voxel_size, voxel_meta_status, crop, downsample = _load_and_crop_volume(
            settings, input_format
        )
        saved = _saved_skeleton_record(voxel_meta_path, skeleton_path)
        _check_saved_skeleton_grid(skeleton_path, skeleton.shape, image.shape, saved, crop, downsample)
        # Resolved from the image just read, as a run making its skeleton does.
        # The record beside the skeleton holds what that run resolved -- an
        # override among it, read back here as metadata -- and is not there at
        # all for a skeleton the panel wrote for a resumed run (1 um voxels).
        voxel_size, voxel_size_source = resolve_voxel_size_xyz(
            metadata_voxel_size_xyz=metadata_voxel_size,
            metadata_status=voxel_meta_status,
            voxel_size_override_xyz=settings["voxel_size_override_xyz"],
            voxel_size_policy=settings["voxel_size_policy"],
        )
        voxel_size = _run_voxel_size_on_grid(voxel_size, downsample)
        logger.info(f"Loaded skeleton from: {skeleton_path}")
        logger.info(
            "Voxel-size resolution (skeleton loaded): "
            f"source={voxel_size_source}, "
            f"metadata_status={voxel_meta_status.get('status')}, "
            f"final={voxel_size}"
        )
        made_at = saved.get("run_voxel_size")
        if made_at is not None and not np.allclose(np.asarray(made_at, dtype=float), voxel_size):
            logger.warning(
                "The saved skeleton %s was made at a voxel size of %s um (x, y, z); this "
                "run's is %s. Its clean-up was measured at the old size: turn "
                "do_skeletonize back on to make it again.",
                skeleton_path, tuple(made_at), voxel_size,
            )
        image, raw_segmented_image = _mask_skeleton_is_made_from(settings, image, voxel_size)
        binary = _to_binary_volume_for_skeletonization(
            image,
            use_memmap=settings["use_memmap_loading"],
            memmap_directory=settings["memmap_directory"],
        )
        made_from = preprocessing.mask_digest(binary)
        thick_vessel_mask, region_key = _thick_region_and_key(
            settings, binary, voxel_size, crop, cache=region_cache, downsample=downsample,
            carried=thick_region,
        )
        preprocessing.release_superseded(binary, image, keep=image)
        _warn_if_made_otherwise(skeleton_path, saved, made_from, region_key)
        if settings["use_thick_vessel_skeletonisation"]:
            drawn_by = saved.get("large_stack_thick_tree")
            if drawn_by is not None and bool(drawn_by) != large_stack_thick_tree(settings):
                trees = {True: "the large-stack tree", False: "the exact tree"}
                logger.warning(
                    "The saved skeleton %s drew its thick vessels with %s; this run "
                    "draws %s and bridges to it (skeleton_thick_vessel_large_stack_tree "
                    "applies under use_memmap_loading only). Turn do_skeletonize back "
                    "on to make it again.",
                    skeleton_path, trees[bool(drawn_by)], trees[large_stack_thick_tree(settings)],
                )

    if plotting:
        logger.info("Visualizing skeleton projection...")
        visualization.visualize_skeleton(skeleton, save_path=projection_path)
        logger.info("Skeleton projection saved.")

    main_voxel_size_xyz = tuple(float(v) for v in voxel_size)
    # Image metadata reports (x, y, z); array axes are canonical (z, y, x).
    # Everything downstream that scales array indices uses voxel_size_zyx.
    voxel_size_zyx = io.voxel_size_zyx_from_xyz(main_voxel_size_xyz)

    if settings["use_memmap_loading"]:
        logger.info(LOW_RAM_SKIPPED_DIAGNOSTICS_MESSAGE, "skeleton/mask consistency")
    else:
        # Read-only check on the skeleton just produced (or loaded): how much
        # of the segmented image it actually runs through -- see
        # preprocessing.skeleton_consistency. Never changes skeleton or image.
        # Both checks from one distance transform of the skeleton.
        mask_consistency, missing_vessels = preprocessing.diagnose_skeleton_against_mask(
            skeleton, image, voxel_size_zyx=voxel_size_zyx,
            min_vessel_voxels=missing_vessel_min_voxels_on_grid(settings, downsample),
        )
        consistency_report = preprocessing.format_skeleton_mask_consistency_report(
            mask_consistency
        )
        if mask_consistency["coverage_fraction"] < float(
            settings["skeleton_mask_consistency_warn_below"]
        ):
            logger.warning(consistency_report)
        else:
            logger.info(consistency_report)

        # Read-only check on the same skeleton/image pair, the inverse
        # question: not how well-traced the mask is overall, but whether any
        # genuine vessel is missing from the skeleton entirely -- see
        # preprocessing.skeleton_consistency.diagnose_vessels_missing_from_skeleton.
        missing_vessels_report = preprocessing.format_vessels_missing_from_skeleton_report(
            missing_vessels
        )
        if missing_vessels["explained_vessel_fraction"] < float(
            settings["skeleton_missing_vessel_warn_below"]
        ):
            logger.warning(missing_vessels_report)
        else:
            logger.info(missing_vessels_report)

    return SkeletonisedVolume(
        image=image,
        skeleton=skeleton,
        voxel_size_xyz=main_voxel_size_xyz,
        voxel_size_zyx=voxel_size_zyx,
        output_dir=output_dir,
        thick_vessel_mask=thick_vessel_mask,
        raw_segmented_image=raw_segmented_image,
        crop=crop,
        downsample=downsample,
        thick_vessel_centreline=(
            thick_centreline[0] if thick_centreline and thick_vessel_mask is not None else None
        ),
    )


def _want_midrun_plotly_html(settings: dict) -> bool:
    """Whether a stage should write Plotly HTML as it runs, not only at export.

    Follows ``visualize_results`` (Produce IDE plots). Napari starts that off
    so a run does not dump ``final_graph_3d.html`` or copy every centreline
    voxel into Scatter3d. The viewer already shows the network.
    """
    return bool(settings["visualize_results"])


def _write_run_final_graph_3d_html(
    settings: dict,
    graph: nx.Graph,
    *,
    large_arteriole_mask: np.ndarray | None,
    large_venule_mask: np.ndarray | None,
    small_arteriole_mask: np.ndarray | None,
    small_venule_mask: np.ndarray | None,
    voxel_size_zyx: tuple[float, float, float],
    title: str = "Final Graph (Interactive 3D)",
    show: bool | None = None,
) -> None:
    """Write ``final_graph_3d.html`` from already-loaded run masks.

    Uses the full mask arrays on the network, not a Z-projected viewer copy.
    """
    # Imported by name: the `graph` parameter hides the module here.
    from haemolynx.graph.boundaries import vessel_masks_in_use

    plot_dir = Path(settings["plot_dir"])
    plot_dir.mkdir(parents=True, exist_ok=True)
    show_plots = (
        bool(settings["show_plots_in_ide"] or settings["interactive_plots"])
        if show is None
        else bool(show)
    )
    visualization.write_final_graph_3d_html(
        graph,
        save_html_path=plot_dir / "final_graph_3d.html",
        title=title,
        use_large_vessel_masks=vessel_masks_in_use(settings, "large"),
        use_small_vessel_masks=vessel_masks_in_use(settings, "small"),
        large_arteriole_mask=large_arteriole_mask,
        large_venule_mask=large_venule_mask,
        small_arteriole_mask=small_arteriole_mask,
        small_venule_mask=small_venule_mask,
        input_nodes=list(settings.get("inlet_nodes") or []),
        output_nodes=list(settings.get("outlet_nodes") or []),
        arteriole_boundary_nodes=list(settings.get("arteriole_boundary_nodes") or []),
        venule_boundary_nodes=list(settings.get("venule_boundary_nodes") or []),
        voxel_size_zyx=voxel_size_zyx,
        volume_downsample_stride=int(
            settings.get("large_vessel_3d_volume_downsample_stride", 1)
        ),
        show=show_plots,
    )


def _log_graph_consistency_diagnostics(
    settings: dict, G: nx.Graph, skeleton: np.ndarray, image: np.ndarray, voxel_size_zyx,
    downsample: Downsample | None = None, mask_support=None,
) -> None:
    """The four read-only graph checks build_network logs. They never change
    G's nodes or edges; the lumen-artefact report's counts are kept on
    ``G.graph`` (``graph.LUMEN_ARTEFACT_COUNTS``), and the run's manifest
    repeats them. The report reads the mask through *mask_support* (the
    build's, when it has one) held to the run's bridge limits, as the
    clean-up judged it: with the defaults it counted loops the clean-up had
    kept by the run's own limits, and packed the mask a second time."""
    # How much of the skeleton the graph still traces -- see
    # graph.diagnostics.diagnose_skeleton_graph_consistency.
    graph_consistency = graph.diagnose_skeleton_graph_consistency(
        G, skeleton, voxel_size_zyx=voxel_size_zyx
    )
    graph_consistency_report = graph.format_skeleton_graph_consistency_report(
        graph_consistency
    )
    if graph_consistency["coverage_fraction"] < float(
        settings["skeleton_graph_consistency_warn_below"]
    ):
        logger.warning(graph_consistency_report)
    else:
        logger.info(graph_consistency_report)

    # How much of the original segmented image it still runs through -- see
    # graph.diagnostics.diagnose_graph_mask_consistency.
    # Both mask checks from one distance transform of the graph's edges.
    graph_mask_consistency, missing_vessels = graph.diagnose_graph_against_mask(
        G, image, voxel_size_zyx=voxel_size_zyx,
        min_vessel_voxels=missing_vessel_min_voxels_on_grid(settings, downsample),
    )
    graph_mask_consistency_report = graph.format_graph_mask_consistency_report(
        graph_mask_consistency
    )
    if graph_mask_consistency["coverage_fraction"] < float(
        settings["graph_mask_consistency_warn_below"]
    ):
        logger.warning(graph_mask_consistency_report)
    else:
        logger.info(graph_mask_consistency_report)

    # The inverse question on the same pair: not how well-traced the mask is
    # overall, but whether any genuine vessel is missing from the graph
    # entirely -- see graph.diagnostics.diagnose_vessels_missing_from_graph.
    missing_vessels_report = graph.format_vessels_missing_from_graph_report(
        missing_vessels
    )
    if missing_vessels["explained_vessel_fraction"] < float(
        settings["graph_missing_vessel_warn_below"]
    ):
        logger.warning(missing_vessels_report)
    else:
        logger.info(missing_vessels_report)

    # What is left in a segmented vessel that is not a vessel: two edges in
    # one lumen, loops inside one, dead ends by kind -- see
    # graph.diagnostics.diagnose_lumen_artefacts. On the graph that is saved,
    # after the smoothing and the thick-vessel split.
    gap_um = float(settings["bridge_max_background_gap_um"])
    fraction = float(settings["bridge_min_mask_fraction"])
    if mask_support is not None:
        mask_support = mask_support.limited_to(gap_um, fraction)
    else:
        mask_support = preprocessing.MaskSupport(
            _to_binary_volume_for_skeletonization(image), voxel_size_zyx,
            max_background_gap_um=gap_um, min_mask_fraction=fraction,
        )
    artefacts = graph.diagnose_lumen_artefacts(
        G, image, voxel_size_zyx=voxel_size_zyx, mask_support=mask_support,
        stub_radius_multiple=dead_end_stub_radius_multiple(settings),
    )
    G.graph[graph.LUMEN_ARTEFACT_COUNTS] = graph.lumen_artefact_counts(artefacts)
    artefacts_report = graph.format_lumen_artefacts_report(artefacts)
    if graph.lumen_artefacts_found(artefacts):
        logger.warning(artefacts_report)
    else:
        logger.info(artefacts_report)


def _log_thick_vessel_bridge_shapes(G: nx.MultiGraph, context: NetworkContext, settings: Mapping[str, Any]) -> None:
    """What shape each thick-vessel bridge has, against the tree the split
    bridged to (graph.diagnose_thick_vessel_bridges): a warning when any is
    one the split never means to make. Read-only, like the graph checks."""
    report = graph.diagnose_thick_vessel_bridges(
        G, context.thick_vessel_mask, context.thick_tree, context.binary_mask,
        voxel_size_zyx=context.voxel_size_zyx, tree_radius_um=context.thick_tree_radius,
        centreline_tolerance_um=thick_bridge_centreline_tolerance_um(settings),
    )
    if not report["bridge_count"]:
        return
    line = graph.format_thick_vessel_bridges_report(report)
    if report["unintended_count"]:
        logger.warning(line)
    else:
        logger.info(line)


def thick_junction_bridge_arguments(settings: dict) -> dict[str, Any]:
    """What ``graph.insert_thick_vessel_junction_nodes`` takes from a run's
    settings: drawing a vessel on from a thick vessel's wall is a
    mask-supported bridge like any other, so it follows the same switch and
    the same limits on the background it may cross; and a blind vessel
    beyond a bridge goes only when the stub prune would take it."""
    return dict(
        centreline_tolerance_um=thick_bridge_centreline_tolerance_um(settings),
        join_dead_ends_at_walls=bool(settings["bridge_require_mask_support"]),
        max_background_gap_um=float(settings["bridge_max_background_gap_um"]),
        min_mask_fraction=float(settings["bridge_min_mask_fraction"]),
        max_short_loop_um=float(settings.get("remove_short_loops_under_um") or 0.0),
        short_loop_keep_hole_um=_short_loop_keep_hole_um(settings),
        fill_tree_gaps=bool(settings.get("thick_vessel_fill_centreline_gaps", True)),
        extend_thick_ends=bool(settings.get("thick_vessel_extend_centreline_ends", True)),
        min_stub_length=float(settings.get("min_stub_length", 10.0)),
        min_stub_length_radius_multiple=float(settings.get("min_stub_length_radius_multiple", 3.0) or 0.0),
    )


def thick_bridge_centreline_tolerance_um(settings: Mapping[str, Any]) -> float:
    """``thick_bridge_centreline_tolerance_um``: how near the thick vessels'
    centreline tree a graph centreline is their own, for the split and for
    every clean-up keeping the strand on it. Its default when a caller's
    settings predate it."""
    value = settings.get(
        "thick_bridge_centreline_tolerance_um", graph.thick_vessel_junctions.CENTRELINE_TOLERANCE_UM
    )
    return float(graph.thick_vessel_junctions.CENTRELINE_TOLERANCE_UM if value is None else value)


def dead_end_stub_radius_multiple(settings: Mapping[str, Any]) -> float:
    """``min_stub_length_radius_multiple`` as the dead-end kinds and the
    lumen-artefact report read it: the stub prune's own rule, so 0 calls no
    dead end short by its vessel's radius, as the prune then prunes none by
    it. 3, its default, when a caller's settings predate it."""
    value = settings.get("min_stub_length_radius_multiple")
    return 3.0 if value is None else float(value)


def lumen_cleanup_switches(settings: Mapping[str, Any]) -> dict[str, bool]:
    """``lumen_cleanup_rewire_pairs`` and ``lumen_cleanup_guard_rejoins`` as
    ``graph.assemble.lumen_cleanup`` takes them; on (the library's defaults)
    when a caller's settings predate them."""
    return dict(
        rewire_pairs=bool(settings.get("lumen_cleanup_rewire_pairs", graph.lumen_loops.REWIRE_PAIRS)),
        guard_rejoins=bool(settings.get("lumen_cleanup_guard_rejoins", graph.doubled_stretches.GUARD_REJOINS)),
    )


def _short_loop_keep_hole_um(settings: dict) -> float:
    """``short_loop_keep_hole_over_um``, its default when a caller's settings
    predate it."""
    value = settings.get("short_loop_keep_hole_over_um", graph.lumen_loops.DEFAULT_SHORT_LOOP_KEEP_HOLE_UM)
    return float(value or 0.0)


def graph_build_arguments(settings: dict) -> dict[str, Any]:
    """The keyword arguments ``graph.build_graph_from_skeleton`` takes from a
    run's settings: everything but the skeleton, the voxel size, the mask and
    the callbacks. One mapping, for :func:`build_network` and for anything that
    rebuilds a run's graph outside the pipeline (``scripts/graph_artefacts.py``)."""
    return dict(
        graph_reconnect_threshold=settings["graph_reconnect_threshold"],
        final_orphan_reconnect_threshold=settings["final_orphan_reconnect_threshold"],
        cluster_collapse_distance=settings["cluster_collapse_distance"],
        min_stub_length=settings["min_stub_length"],
        min_stub_length_radius_multiple=float(settings.get("min_stub_length_radius_multiple", 0.0) or 0.0),
        debug=settings["verbose_logging"],
        cluster_collapse_method=settings["cluster_collapse_method"],
        cluster_collapse_max_radial_dispersion=float(settings["cluster_collapse_max_radial_dispersion"]),
        cluster_collapse_persistence_search_multiple=float(
            settings["cluster_collapse_persistence_search_multiple"]
        ),
        cluster_collapse_direction_aware_min_degree=int(settings["cartwheel_hub_min_degree"]),
        cluster_collapse_direction_aware_tangent_length_um=float(settings["cartwheel_hub_tangent_length_um"]),
        use_memmap=settings["use_memmap_loading"],
        bridge_require_mask_support=settings["bridge_require_mask_support"],
        bridge_max_background_gap_um=float(settings["bridge_max_background_gap_um"]),
        bridge_min_mask_fraction=float(settings["bridge_min_mask_fraction"]),
        recover_uncovered_mask_vessels=settings["recover_uncovered_mask_vessels"],
        recovery_min_region_volume_um3=float(settings["recovery_min_region_volume_um3"]),
        recovery_min_length_um=float(settings["recovery_min_length_um"]),
        facing_dead_end_max_gap_um=float(settings["facing_dead_end_max_gap_um"]),
        remove_short_loops_under_um=float(settings.get("remove_short_loops_under_um") or 0.0),
        short_loop_keep_hole_over_um=_short_loop_keep_hole_um(settings),
        extend_dead_ends_to_faces=bool(settings.get("extend_dead_ends_to_faces", False)),
        trace_on_dead_ends=bool(settings.get("trace_on_dead_ends", False)),
        join_dead_ends_to_vessel_sides=bool(settings.get("join_dead_ends_to_vessel_sides", False)),
        stub_face_protection=settings.get("stub_face_protection") or "threshold",
        lumen_cleanup_max_rounds=_lumen_cleanup_max_rounds(settings),
        lumen_cleanup_rewire_pairs=lumen_cleanup_switches(settings)["rewire_pairs"],
        lumen_cleanup_guard_rejoins=lumen_cleanup_switches(settings)["guard_rejoins"],
    )


def _lumen_cleanup_max_rounds(settings: dict) -> int:
    """``lumen_cleanup_max_rounds``, its default when a caller's settings
    predate it."""
    value = settings.get("lumen_cleanup_max_rounds", graph.assemble.CONSOLIDATION_MAX_ROUNDS)
    return max(1, int(value or graph.assemble.CONSOLIDATION_MAX_ROUNDS))


def _no_vessels_left_error(
    settings: dict,
    skeleton: np.ndarray,
    binary_mask: np.ndarray | None,
    edges_after_step: Sequence[tuple[str, int]],
) -> ValueError:
    """What :func:`build_network` raises when graph building left no vessels:
    the step that took the last one, how much of the skeleton the image's
    foreground covers, and the settings that remove vessels."""
    if binary_mask is not None:
        # The mask's value at each skeleton voxel, a slab at a time.
        off_mask, on_mask = (
            int(n) for n in bincount_by_slab(binary_mask, where=skeleton, minlength=2)[:2]
        )
        skeleton_voxels = off_mask + on_mask
    else:
        skeleton_voxels = int(np.count_nonzero(skeleton))
    last_taken_at = None
    before = 0
    for label, count in edges_after_step:
        if before and not count:
            last_taken_at = label
        before = count
    if last_taken_at is not None:
        built = (
            f"The skeleton's {skeleton_voxels} voxels built "
            f"{max(count for _, count in edges_after_step)} vessel(s), and the last went at "
            f"the {last_taken_at} step."
        )
    else:
        built = f"The skeleton's {skeleton_voxels} voxels built no vessel at all."
    if binary_mask is not None:
        built += f" {on_mask} of those {skeleton_voxels} voxels lie in the image's foreground."
    radius_multiple = settings.get("min_stub_length_radius_multiple", 0.0) or 0.0
    reasons = [
        f"cluster_collapse_distance ({settings['cluster_collapse_distance']} um) merges nodes "
        "closer than that, so a skeleton whose branches are all shorter collapses to one point",
        f"min_stub_length ({settings['min_stub_length']} um) and min_stub_length_radius_multiple "
        f"({radius_multiple}) set how short a blind end the stub prune removes",
    ]
    if binary_mask is not None:
        reasons.append(
            "the image: edges and stubs lying mostly outside its foreground are removed, so it "
            "must be the segmentation the skeleton was made from"
        )
    if settings.get("remove_pieces_without_junction"):
        reasons.append("remove_pieces_without_junction removes every vessel with no junction on it")
    if settings.get("remove_interior_dead_ends"):
        reasons.append(
            "remove_interior_dead_ends removes every dead end inside the image, so a network "
            "with no loop and no vessel leaving the image goes whole"
        )
    if settings.get("remove_interior_dead_ends_and_merge"):
        reasons.append(
            "remove_interior_dead_ends_and_merge removes every dead end inside the image, so a "
            "network with no loop and no vessel leaving the image goes whole"
        )
    return ValueError(
        f"Graph building left no vessels. {built} Settings to check: " + "; ".join(reasons) + "."
    )


@dataclass
class NetworkContext:
    """What graph building and every step after it read the segmentation
    through, made once per mask (:func:`network_context`): shared by
    :func:`build_network` and the settings optimiser's trials."""

    binary_mask: np.ndarray | None
    voxel_size_zyx: tuple[float, float, float]
    #: The mask's radius at a point, for the stub prune and the smoothing.
    radius_at: Any = None
    #: The one reader of the mask every step asks.
    mask_support: Any = None
    thick_vessel_mask: np.ndarray | None = None
    #: The thick vessels' centreline tree (its voxels), its radius, and what
    #: the clean-ups keep the strand on the tree by.
    thick_tree: Any = None
    thick_tree_radius: Any = None
    along_thick_centreline: Any = None


def network_context(
    settings: Mapping[str, Any],
    binary_mask: np.ndarray | None,
    voxel_size_zyx,
    thick_vessel_mask: np.ndarray | None = None,
    *,
    thick_centreline: np.ndarray | None = None,
) -> NetworkContext:
    """Everything graph building and :func:`finish_network_graph` read the
    segmentation through, made once.

    Stubs are judged against the radius of the vessel they leave, a
    smoothed centreline may stray further inside a wider vessel, and two
    edges joining the same two nodes are one vessel unless background
    separates them: all three read the binary mask the skeleton was made
    from, the radius through one sampler. One reader of the mask serves
    every step that asks about it -- graph building, the clean-ups after
    it, the thick-vessel split and the dead-end marks -- so it is packed to
    bits and its surface tree built once a run, not once a step. The thick
    vessels' centreline tree is found once too (*thick_centreline*, as
    skeletonisation drew it, when it did): the split bridges to it, and
    every clean-up keeps the strand on it of two in one lumen. Judged by the
    wall alone, the clean-up took the strand on the tree as often as not and
    left the thick vessel broken.
    """
    voxel_size_zyx = tuple(float(v) for v in voxel_size_zyx)
    stub_radius_multiple = float(settings.get("min_stub_length_radius_multiple", 0.0) or 0.0)
    radius_at = None
    if binary_mask is not None and (stub_radius_multiple > 0 or settings["smooth_centrelines"]):
        radius_at = graph.assemble.mask_radius_sampler(
            binary_mask, voxel_size_zyx, 1.0, low_memory=settings["use_memmap_loading"]
        )
    mask_support = None
    if binary_mask is not None:
        mask_support = preprocessing.MaskSupport(
            binary_mask,
            voxel_size_zyx,
            max_background_gap_um=float(settings["bridge_max_background_gap_um"]),
            min_mask_fraction=float(settings["bridge_min_mask_fraction"]),
            feature_distance=getattr(radius_at, "feature_distance", None),
            low_memory=settings["use_memmap_loading"],
            memmap_directory=settings["memmap_directory"],
        )
    context = NetworkContext(
        binary_mask=binary_mask, voxel_size_zyx=voxel_size_zyx, radius_at=radius_at,
        mask_support=mask_support, thick_vessel_mask=thick_vessel_mask,
    )
    if thick_vessel_mask is not None:
        thick_tree = thick_centreline
        if thick_tree is None:
            thick_tree = graph.thick_vessel_junctions.thick_vessel_centreline_voxels(
                thick_vessel_mask,
                voxel_size_zyx,
                use_memmap=settings["use_memmap_loading"],
                large_stack_tree=large_stack_thick_tree(settings),
            )
        follower = graph.thick_vessel_junctions.ThickCentreline.from_region(
            thick_vessel_mask, voxel_size_zyx, segmentation=binary_mask, centreline=thick_tree,
            centreline_tolerance_um=thick_bridge_centreline_tolerance_um(settings),
        )
        context.thick_tree = thick_tree
        context.along_thick_centreline = None if follower is None else follower.along_um
        context.thick_tree_radius = None if follower is None else follower.radius_um
    return context


#: What graph building took out and drew, as its steps keep it on the graph
#: (``G.graph``): repeated in the run's manifest (``checks.graph_building``),
#: where a run's outputs had no word of what remove_interior_dead_ends or the
#: removal of pieces without a junction took.
GRAPH_BUILDING_COUNTS: tuple[str, ...] = (
    "edges_off_mask_removed",
    "parallel_lumen_edges_removed",
    "parallel_lumen_edges_rewired",
    "doubled_stretches_resolved",
    "recovered_joins",
    "recovered_dead_ends_removed",
    "recovered_one_ended_removed",
    "interior_dead_ends_removed",
    "interior_dead_ends_removed_um",
    "interior_recovery_drawn_um",
    "interior_recovery_kept_um",
    "interior_dead_end_junctions_merged",
    "pieces_without_junction_removed",
    "pieces_without_junction_removed_um",
)


def graph_building_counts(G: nx.Graph) -> dict[str, float | int]:
    """The :data:`GRAPH_BUILDING_COUNTS` *G* carries, as plain numbers."""
    counts = {}
    for name in GRAPH_BUILDING_COUNTS:
        value = G.graph.get(name)
        if value is not None:
            counts[name] = int(value) if float(value).is_integer() and not name.endswith("_um") else float(value)
    return counts


#: What :func:`build_network` reports after its last step, saving the graph
#: and checking it, as steps of its own: a cancelled run stops at each, and
#: the step bar does not stand finished while they run.
BUILD_TAIL_STEPS: tuple[str, ...] = ("dead-end kinds", "graph saved", "consistency checks")


def finish_step_labels(settings: Mapping[str, Any], context: NetworkContext) -> list[str]:
    """The steps :func:`finish_network_graph` will report for *settings* and
    *context*, in the order it reports them: a progress bar needs the count
    before the first step lands, as ``graph.STEP_LABELS`` gives the build's."""
    masked = context.binary_mask is not None
    supported = masked and bool(settings["bridge_require_mask_support"])
    smooth = bool(settings["smooth_centrelines"])
    labels = []
    if smooth:
        labels.append("centreline smoothing")
        if masked:
            labels.append("consolidate_lumen after centreline smoothing")
    if context.thick_vessel_mask is not None:
        labels.append("thick-vessel split")
        if masked:
            labels.append("consolidate_lumen after the thick-vessel split")
    if settings.get("retrace_lost_skeleton") and _short_loop_keep_hole_um(settings) > 0 and supported:
        labels.append("skeleton re-trace")
    if smooth:
        labels.append("smoothing of straight steps")
    if settings.get("remove_interior_dead_ends") and supported:
        labels.append("removal and recovery of interior dead ends")
    if settings.get("remove_interior_dead_ends_and_merge") and masked:
        labels.append("removal of interior dead ends and merge of the junctions left")
    if settings.get("remove_pieces_without_junction"):
        labels.append("removal of pieces with no junction")
    return labels


def finish_network_graph(
    G: nx.MultiGraph,
    skeleton: np.ndarray,
    settings: Mapping[str, Any],
    context: NetworkContext,
    *,
    edges_after_step: list | None = None,
    step_callback: Callable[[nx.MultiGraph, str], None] | None = None,
) -> nx.MultiGraph:
    """Every step :func:`build_network` takes after
    ``graph.build_graph_from_skeleton``: the centreline smoothing, the
    lumen clean-up after it, the thick-vessel split and the clean-up after
    that, the skeleton re-trace, the smoothing of straight steps, and the
    optional removal of interior dead ends and of pieces with no junction.
    The graph a run saves, and the one a settings-optimiser trial scores.
    *edges_after_step* (a list) is given each step's label and edge count;
    *step_callback*, ``(G, label)`` as graph building's own, is called
    instead after each (the labels :func:`finish_step_labels` names)."""
    steps = edges_after_step if edges_after_step is not None else []

    def finished(label: str) -> None:
        # Reported as each lands, as graph building's steps are: where a
        # cancelled run stops, and what a watcher sees while it runs.
        if step_callback is not None:
            step_callback(G, label)
        else:
            steps.append((label, G.number_of_edges()))
    binary_mask = context.binary_mask
    voxel_size_zyx = context.voxel_size_zyx
    radius_at = context.radius_at
    stub_radius_multiple = float(settings.get("min_stub_length_radius_multiple", 0.0) or 0.0)
    # Last thing before the graph is saved: take the voxel staircase out of
    # each centreline. A skeleton path steps voxel to voxel, so a vessel at
    # an angle comes back 7% longer than it is, and resistance follows
    # length -- so this is a measurement fix, not a cosmetic one.
    if settings["smooth_centrelines"]:
        graph.smooth_graph_centrelines(
            G,
            skeleton,
            voxel_size_zyx=voxel_size_zyx,
            **centreline_smoothing_arguments(settings),
            radius_at=radius_at,
        )
        finished("centreline smoothing")
    # Smoothing moves every centreline, so what graph building's last
    # step judged is judged again on the curves that are saved: on E14.5
    # smoothing alone turned 2 loops inside one lumen into 12. The same
    # clean-up runs once more after the thick-vessel split below: nothing
    # is the last step, and on E14.5 the split alone made 7 loops inside
    # one lumen through its bridges.
    lumen_cleanup = None
    if binary_mask is not None:
        lumen_cleanup = graph.assemble.lumen_cleanup(
            binary_mask,
            voxel_size_zyx,
            image_shape=skeleton.shape,
            min_stub_length=settings["min_stub_length"],
            min_stub_length_radius_multiple=stub_radius_multiple,
            stub_radius_at=radius_at,
            bridge_require_mask_support=settings["bridge_require_mask_support"],
            bridge_max_background_gap_um=float(settings["bridge_max_background_gap_um"]),
            bridge_min_mask_fraction=float(settings["bridge_min_mask_fraction"]),
            debug=settings["verbose_logging"],
            stub_face_protection=settings.get("stub_face_protection") or "threshold",
            remove_short_loops_under_um=float(settings.get("remove_short_loops_under_um") or 0.0),
            short_loop_keep_hole_over_um=_short_loop_keep_hole_um(settings),
            along_thick_centreline=context.along_thick_centreline,
            mask_support=context.mask_support,
            max_rounds=_lumen_cleanup_max_rounds(settings),
            **lumen_cleanup_switches(settings),
        )
        if settings["smooth_centrelines"]:
            G = graph.assemble.consolidate_lumen(G, lumen_cleanup)
            finished("consolidate_lumen after centreline smoothing")
    # After every topology step and the centreline smoothing above: a
    # vessel joined to a thick one runs on through its lumen with no
    # marker of where the vessel ends and the thick vessel begins. Each
    # gets one bridge, from where it enters the lumen (the segmentation's
    # radius round the thick centreline) to that centreline -- see
    # graph.insert_thick_vessel_junction_nodes. Done on the fully-built
    # graph, so nothing has to protect it through the degree-2/collapse/
    # prune passes; the clean-up that follows knows a bridge (never merged
    # with a vessel, never dropped as a second strand, never left dangling).
    if context.thick_vessel_mask is not None:
        G = graph.insert_thick_vessel_junction_nodes(
            G, context.thick_vessel_mask, voxel_size_zyx=voxel_size_zyx,
            segmentation=binary_mask, centreline=context.thick_tree,
            **thick_junction_bridge_arguments(settings),
            mask_support=context.mask_support, tree_radius_um=context.thick_tree_radius,
        )
        finished("thick-vessel split")
        if lumen_cleanup is not None:
            G = graph.assemble.consolidate_lumen(G, lumen_cleanup)
            finished("consolidate_lumen after the thick-vessel split")
    # Skeleton the clean-up took round a wide hole, drawn back after the last
    # clean-up so no rule takes it again (Harvey's grid N): the pair and
    # in-lumen loop rules never read the hole a loop runs round. A hole limit
    # of 0 keeps no loop for its hole, so nothing is drawn back for one.
    if (
        settings.get("retrace_lost_skeleton")
        and _short_loop_keep_hole_um(settings) > 0
        and lumen_cleanup is not None
        and lumen_cleanup.support is not None
    ):
        graph.skeleton_retrace.retrace_lost_skeleton(
            G, skeleton, lumen_cleanup.support, keep_hole_over_um=_short_loop_keep_hole_um(settings),
        )
        retraced = graph.skeleton_retrace.retraced_edges(G)
        if retraced and settings["smooth_centrelines"]:
            graph.smooth_graph_centrelines(
                G,
                skeleton,
                voxel_size_zyx=voxel_size_zyx,
                **centreline_smoothing_arguments(settings),
                radius_at=radius_at,
                edges=retraced,
            )
        finished("skeleton re-trace")
    # The clean-ups after smoothing join a cut piece to the vessel it
    # doubled with a straight chord across the lumen: those vessels only
    # are smoothed again (PLAN_bridging.md 6b).
    if settings["smooth_centrelines"]:
        graph.smooth_straight_steps(
            G,
            skeleton,
            voxel_size_zyx=voxel_size_zyx,
            **centreline_smoothing_arguments(settings),
            radius_at=radius_at,
        )
        finished("smoothing of straight steps")
    # Interior dead ends out, the mask they leave uncovered recovered as
    # vessels joined at both ends, and the clean-up run again (Harvey's
    # grids G5, L1 and L2, 2026-10-21; off by default).
    if (
        settings.get("remove_interior_dead_ends")
        and lumen_cleanup is not None
        and lumen_cleanup.support is not None
    ):
        G = _remove_and_recover_interior_dead_ends(
            settings, G, skeleton=skeleton, binary_mask=binary_mask, voxel_size_zyx=voxel_size_zyx,
            radius_at=radius_at, cleanup=lumen_cleanup, thick_mask=context.thick_vessel_mask,
            thick_tree=context.thick_tree, thick_tree_radius=context.thick_tree_radius,
        )
        finished("removal and recovery of interior dead ends")
    # Interior dead ends out with nothing traced back, and each junction that
    # leaves with two vessels merged into one vessel, beside a junction of
    # four vessels too, where the build's own merge leaves it.
    if settings.get("remove_interior_dead_ends_and_merge") and binary_mask is not None:
        mark_dead_end_kinds(settings, G, context)
        removed = graph.dead_ends.remove_interior_dead_ends_and_merge(G, inside_lumen=lumen_cleanup.inside_lumen)
        G.graph["interior_dead_ends_removed"] = G.graph.get("interior_dead_ends_removed", 0) + removed["dead_ends"]
        G.graph["interior_dead_ends_removed_um"] = (
            G.graph.get("interior_dead_ends_removed_um", 0.0) + removed["length_um"]
        )
        G.graph["interior_dead_end_junctions_merged"] = removed["merged"]
        logger.info(
            "Removed %d interior dead end(s) (%.0f um of centreline) and merged %d junction(s) they left "
            "with two vessels; %d stay, a vessel meeting a thick-vessel bridge or a loop.",
            removed["dead_ends"], removed["length_um"], removed["merged"], removed["left_degree_2"],
        )
        finished("removal of interior dead ends and merge of the junctions left")
    # Pieces with no junction (their dead ends are graph.dead_ends'
    # island ends, or leave the image), taken out last so no step after
    # rejoins or re-makes one. On Harvey's AIA patella grid P1 (2026-10-20)
    # 11 of 14 were specks and 3 real vessel the segmentation cut off.
    if settings.get("remove_pieces_without_junction"):
        removed = graph.dead_ends.remove_pieces_without_junction(G)
        G.graph["pieces_without_junction_removed"] = removed["pieces"]
        G.graph["pieces_without_junction_removed_um"] = removed["length_um"]
        logger.info(
            "Removed %d piece(s) with no junction: %d vessel(s), %.0f um of centreline.",
            removed["pieces"], removed["vessels"], removed["length_um"],
        )
        finished("removal of pieces with no junction")
    return G


def build_network(
    settings: dict,
    volume: SkeletonisedVolume,
    schema: Schema,
    progress: StageProgress | None = None,
    on_step_graph: Callable[[str, Any], None] | None = None,
    resumed_graph: nx.MultiGraph | None = None,
):
    """Load the vessel masks and turn the skeleton into a graph.

    This is the long stage, so it reports the topology steps of
    :func:`graph.build_graph_from_skeleton` to *progress* as they land -- a
    run's only finer-grained progress than "graph building is happening".

    *resumed_graph* is a resumed run's network, built (and checked and drawn) by the
    run that made it: it is used as it is, so ``{stem}_graph.pkl`` is neither
    loaded nor written -- that file stays the one graph building produced.

    A synthetic run draws its network here instead (``network_source``).
    """
    if volume.synthetic:
        return _build_synthetic_network(settings, volume, progress=progress, resumed_graph=resumed_graph)
    image, skeleton = volume.image, volume.skeleton
    output_dir = volume.output_dir
    main_voxel_size_xyz = volume.voxel_size_xyz
    voxel_size_zyx = volume.voxel_size_zyx
    graph_path = output_dir / f"{run_stem(settings)}_graph.pkl"

    def load_vessel_masks():
        """The run's vessel masks, loaded once the graph is: graph building
        reads none of them, and they were held through its peak memory."""
        # The vessel-mask and segmentation settings go in as a group; each role
        # picks the ones it uses. `io.VESSEL_MASK_SETTINGS` lists which those are.
        mask_settings = {
            **schema.section_values(settings, "Vessel masks"),
            **schema.section_values(settings, "Input and segmentation"),
            "use_memmap_loading": settings["use_memmap_loading"],
            "memmap_directory": settings["memmap_directory"],
        }
        (
            large_arteriole_mask,
            large_venule_mask,
            large_arteriole_mask_voxel_size,
            large_venule_mask_voxel_size,
        ) = io.load_and_validate_vessel_masks(
            **_vessel_mask_arguments(mask_settings, "large"),
            image_shape=image.shape,
            main_voxel_size_xyz=main_voxel_size_xyz,
            crop=getattr(volume, "crop", None),
            downsample=getattr(volume, "downsample", None),
        )
        mask_split_diameter_um = graph.large_vessel_mask_split_diameter_um(settings)
        if mask_split_diameter_um is not None:
            # The small masks are the large ones' narrow parts, not files.
            (
                large_arteriole_mask,
                large_venule_mask,
                small_arteriole_mask,
                small_venule_mask,
            ) = _split_large_vessel_masks(
                settings,
                large_arteriole_mask,
                large_venule_mask,
                main_voxel_size_xyz,
                _SegmentationBoxes(image, settings.get("input_path")),
            )
        else:
            small_arteriole_mask, small_venule_mask, *_ = io.load_and_validate_vessel_masks(
                **_vessel_mask_arguments(mask_settings, "small"),
                image_shape=image.shape,
                main_voxel_size_xyz=main_voxel_size_xyz,
                crop=getattr(volume, "crop", None),
                downsample=getattr(volume, "downsample", None),
                loaded_message_suffix=(
                    f"min_overlap_fraction={float(settings['small_vessel_mask_min_overlap_fraction']):.3f}"
                ),
            )
        if io.vessel_masks_are_global(settings):
            # One role given a global mask makes every vessel's class unknown, so
            # a role still given an arteriole/venule pair is one mask too.
            for role, pair in (
                ("large", (large_arteriole_mask, large_venule_mask)),
                ("small", (small_arteriole_mask, small_venule_mask)),
            ):
                if pair[1] is not None:
                    logger.warning(
                        f"The {role}-vessel arteriole and venule masks are read as one "
                        f"global {role}-vessel mask: the other role's mask is global, so "
                        "no vessel is classed arteriole or venule."
                    )
            on_disk = dict(use_memmap=settings["use_memmap_loading"], memmap_directory=settings["memmap_directory"])
            large_arteriole_mask = io.as_global_vessel_mask(large_arteriole_mask, large_venule_mask, **on_disk)
            small_arteriole_mask = io.as_global_vessel_mask(small_arteriole_mask, small_venule_mask, **on_disk)
            large_venule_mask = small_venule_mask = None
            large_venule_mask_voxel_size = None
        return (
            large_arteriole_mask, large_venule_mask, small_arteriole_mask, small_venule_mask,
            large_arteriole_mask_voxel_size, large_venule_mask_voxel_size, mask_split_diameter_um,
        )

    if resumed_graph is not None:
        # Its consistency checks and plots were made when it was built.
        logger.info("Resuming with the network already built; not building or loading one.")
        # Built on another crop box or grid, its positions are off this image.
        _check_saved_graph_fits_image(
            resumed_graph, "The resumed network", image.shape if image is not None else skeleton.shape,
            voxel_size_zyx, getattr(volume, "crop", None),
            remedy="Run from Skeletonise or Graph to build it again on this image.",
        )
        (
            large_arteriole_mask, large_venule_mask, small_arteriole_mask, small_venule_mask,
            large_arteriole_mask_voxel_size, large_venule_mask_voxel_size, mask_split_diameter_um,
        ) = load_vessel_masks()
        _record_graph_voxel_metadata(
            resumed_graph, main_voxel_size_xyz, voxel_size_zyx,
            large_arteriole_mask, large_venule_mask,
            large_arteriole_mask_voxel_size, large_venule_mask_voxel_size,
        )
        _record_crop(resumed_graph, getattr(volume, "crop", None), voxel_size_zyx, getattr(volume, "downsample", None))
        _record_downsample(resumed_graph, settings, volume)
        return VesselNetwork(
            graph=resumed_graph,
            volume=volume,
            large_arteriole_mask=large_arteriole_mask,
            large_venule_mask=large_venule_mask,
            small_arteriole_mask=small_arteriole_mask,
            small_venule_mask=small_venule_mask,
            mask_split_diameter_um=mask_split_diameter_um,
            mask_sources=_vessel_mask_sources(settings),
        )

    if settings["do_graph_building"]:
        # A vessel is a path between touching skeleton voxels. With none, the
        # cause is upstream, in the segmentation or skeletonising, so say so
        # here rather than build an empty graph.
        if not graph.build.skeleton_has_path(skeleton):
            raise _skeleton_without_vessels_error(
                settings, volume, output_dir / f"{run_stem(settings)}_skeleton.npy"
            )
        # 3) Convert skeleton to graph.
        # Every snapshot draws the same volume, and projecting it reads the whole
        # stack, so it is projected once here rather than once per step. Graph
        # building reads `image` and never writes to it, which is what makes one
        # projection good for every step.
        step_projection = (
            visualization.overlay_z_projection(image)
            if settings["save_step_artifacts"]
            else None
        )
        # The vessels left after each step, so a build that ends with none
        # can say which step took the last one.
        edges_after_step: list[tuple[str, int]] = []
        # Every step this stage reports, in order: the build's thirteen, the
        # finish's (finish_step_labels), then saving and checking the graph.
        # Each is where a cancelled run stops.
        step_labels: list[str] = list(graph.STEP_LABELS)

        def _graph_build_step_callback(graph_obj, label: str, *, draw: bool = True) -> None:
            edges_after_step.append((label, graph_obj.number_of_edges()))
            # Report before drawing: the snapshots below are the slow part of
            # this step, so a watcher should see it tick over on arrival.
            if progress is not None:
                done = len(edges_after_step)
                progress.step(
                    label,
                    total=len(step_labels),
                    next_step=step_labels[done] if done < len(step_labels) else None,
                )
            if not draw:
                return
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
                run_stem(settings),
                label,
                projection=step_projection,
                extra_plot_names=(plot_png,),
            )

        # Stubs are judged against the radius of the vessel they leave, the
        # clean-ups read the mask through one MaskSupport, and the thick
        # vessels' centreline tree is found once: all of it made here, once,
        # and shared with the settings optimiser's trials (network_context).
        binary_mask = None
        if image is not None:
            binary_mask = _to_binary_volume_for_skeletonization(
                image,
                use_memmap=settings["use_memmap_loading"],
                memmap_directory=settings["memmap_directory"],
            )
        context = network_context(
            settings, binary_mask, voxel_size_zyx, volume.thick_vessel_mask,
            thick_centreline=getattr(volume, "thick_vessel_centreline", None),
        )
        radius_at, mask_support = context.radius_at, context.mask_support
        step_labels[:] = [*graph.STEP_LABELS, *finish_step_labels(settings, context), *BUILD_TAIL_STEPS]
        G = graph.build_graph_from_skeleton(
            skeleton,
            voxel_size=voxel_size_zyx,
            stub_radius_at=radius_at,
            segmentation_mask=binary_mask,
            step_callback=_graph_build_step_callback,
            along_thick_centreline=context.along_thick_centreline,
            mask_support=mask_support,
            **graph_build_arguments(settings),
        )
        G = finish_network_graph(G, skeleton, settings, context, step_callback=_graph_build_step_callback)
        if context.thick_vessel_mask is not None and G.number_of_edges():
            _log_thick_vessel_bridge_shapes(G, context, settings)

        # Nothing after this can run on no vessels, and what tried failed on
        # something else (the 3D drawing on no node positions, boundary
        # assignment on no terminals). Raised before the graph is saved, so
        # an empty one never replaces a saved graph; the reason is read first,
        # while the mask is still here.
        no_vessels = None
        if G.number_of_edges() == 0:
            no_vessels = _no_vessels_left_error(settings, skeleton, binary_mask, edges_after_step)

        # What each dead end is (graph.dead_ends), judged on the graph that is
        # saved while the mask is still at hand: those leaving the image are
        # what the flow's open ends are put on, the rest what the review lists.
        graph.dead_ends.record_image_extent(G, skeleton.shape, voxel_size_zyx)
        raw_refused = None
        if binary_mask is not None and no_vessels is None:
            mark_dead_end_kinds(settings, G, context)
            try:
                _record_raw_evidence(
                    settings, G, binary_mask, voxel_size_zyx,
                    getattr(volume, "crop", None), getattr(volume, "downsample", None),
                )
            except (OSError, ValueError) as error:
                # Refused, as a raw image that does not lie on the segmentation
                # always is, but once the graph is saved: refused here, the
                # whole build was lost with it.
                raw_refused = error

        # The consistency checks read the mask through the build's support
        # (in RAM; the low-RAM option skips them), not one packed again.
        report_support = None if settings["use_memmap_loading"] else context.mask_support
        if binary_mask is not None:
            # The support holds the mask's bits and surface tree: let them go
            # with the mask.
            context = mask_support = None
            preprocessing.release_superseded(binary_mask, G, keep=image)
            binary_mask = radius_at = None
        if no_vessels is not None:
            raise no_vessels

        # Saved with the box it was built in and the skeleton it was built
        # from, so a run loading it can tell.
        _record_crop(G, getattr(volume, "crop", None), voxel_size_zyx, getattr(volume, "downsample", None))
        G.graph[SKELETON_DIGEST] = skeleton_digest(skeleton)
        # And with the Input settings its volume was made with, so a run
        # resumed on it with others is refused (_check_resumed_input).
        G.graph[INPUT_SETTINGS_RECORD] = getattr(volume, "input_settings", None) or input_volume_settings(settings)
        _graph_build_step_callback(G, BUILD_TAIL_STEPS[0], draw=False)
        # Written whole or not at all: a cancelled or failed save never leaves
        # half a graph for do_graph_building off to load.
        write_pickle_whole(G, graph_path)
        logger.info(f"Saved graph to: {graph_path}")
        if raw_refused is not None:
            logger.warning(
                f"The graph was saved to {graph_path} before the raw image was refused: with the "
                "raw image put right (or cleared), do_graph_building off loads it rather than "
                "building it again."
            )
            raise raw_refused
        _graph_build_step_callback(G, BUILD_TAIL_STEPS[1], draw=False)
    else:
        if not graph_path.exists():
            raise FileNotFoundError(
                f"Graph file not found at {graph_path}. "
                "Set DO_GRAPH_BUILDING=True to generate it first."
            )
        with graph_path.open("rb") as f:
            G = pickle.load(f)
        logger.info(f"Loaded graph from: {graph_path}")
        if G.number_of_edges() == 0:
            raise ValueError(
                f"The graph loaded from {graph_path} has no vessels. "
                "Turn do_graph_building on to build it again from the skeleton."
            )
        _check_saved_graph_fits_image(
            G, f"The saved graph {graph_path}", image.shape if image is not None else skeleton.shape,
            voxel_size_zyx, getattr(volume, "crop", None),
        )
        _warn_if_built_from_another_skeleton(G, graph_path, skeleton)
        report_support = None
        if any(graph.cut_at_large_vessel_volumes.CUT_FROM_LARGE_VESSEL in data for _node, data in G.nodes(data=True)):
            # Until 2026-11-03 the Boundaries stage wrote its cut network over
            # this file: the large vessels are gone from it, whatever the
            # masks and the cut now say.
            logger.warning(
                f"The graph loaded from {graph_path} was already cut at the large-vessel "
                "masks by an earlier run's Boundaries stage, so the vessels inside them are "
                "missing. Turn do_graph_building on to build it again."
            )

    (
        large_arteriole_mask, large_venule_mask, small_arteriole_mask, small_venule_mask,
        large_arteriole_mask_voxel_size, large_venule_mask_voxel_size, mask_split_diameter_um,
    ) = load_vessel_masks()
    _record_graph_voxel_metadata(
        G, main_voxel_size_xyz, voxel_size_zyx,
        large_arteriole_mask, large_venule_mask,
        large_arteriole_mask_voxel_size, large_venule_mask_voxel_size,
    )
    _record_crop(G, getattr(volume, "crop", None), voxel_size_zyx, getattr(volume, "downsample", None))
    _record_downsample(G, settings, volume)
    if "image_extent_um" not in G.graph:
        # A graph saved before dead ends were judged: the extent at least.
        shape = image.shape if image is not None else skeleton.shape
        graph.dead_ends.record_image_extent(G, shape, voxel_size_zyx)

    # Purely diagnostic: off by default, and never changes G even when on --
    # see graph.cartwheel_guard for what this looks for and why.
    if settings["detect_cartwheel_hub_artifacts"]:
        cartwheel_hubs = graph.detect_cartwheel_hubs(
            G,
            min_degree=int(settings["cartwheel_hub_min_degree"]),
            max_radial_dispersion=float(settings["cartwheel_hub_max_radial_dispersion"]),
            tangent_length_um=float(settings["cartwheel_hub_tangent_length_um"]),
        )
        if cartwheel_hubs:
            logger.warning(graph.format_cartwheel_hub_report(cartwheel_hubs))

    if settings["use_memmap_loading"]:
        logger.info(LOW_RAM_SKIPPED_DIAGNOSTICS_MESSAGE, "skeleton/graph and graph/mask consistency")
        # Not measured: a loaded graph's counts are of another build.
        G.graph.pop(graph.LUMEN_ARTEFACT_COUNTS, None)
    else:
        _log_graph_consistency_diagnostics(
            settings, G, skeleton, image, voxel_size_zyx, downsample=volume.downsample,
            mask_support=report_support,
        )
    report_support = None
    if settings["do_graph_building"]:
        _graph_build_step_callback(G, BUILD_TAIL_STEPS[2], draw=False)

    # Visualize final graph used for boundary-node verification.
    if settings["visualize_results"]:
        if settings["final_render_mode"] == "3d":
            final_graph_3d_path = settings["plot_dir"] / "final_graph_3d.html"
            _write_run_final_graph_3d_html(
                settings,
                G,
                large_arteriole_mask=large_arteriole_mask,
                large_venule_mask=large_venule_mask,
                small_arteriole_mask=small_arteriole_mask,
                small_venule_mask=small_venule_mask,
                voxel_size_zyx=voxel_size_zyx,
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

    if settings.get("run_haemodynamics"):
        _warn_about_dense_solver_memory(settings, G)

    return VesselNetwork(
        graph=G,
        volume=volume,
        large_arteriole_mask=large_arteriole_mask,
        large_venule_mask=large_venule_mask,
        small_arteriole_mask=small_arteriole_mask,
        small_venule_mask=small_venule_mask,
        mask_split_diameter_um=mask_split_diameter_um,
        mask_sources=_vessel_mask_sources(settings),
    )


def _check_saved_graph_fits_image(
    G: nx.MultiGraph,
    which: str,
    shape,
    voxel_size_zyx,
    crop: CropBox | None = None,
    *,
    remedy: str = "Turn do_graph_building back on to build it again.",
) -> None:
    """Refuse a graph built on another image grid -- a saved one, or a resumed
    run's network (*which* names it) -- as a saved skeleton is refused: its
    node positions are microns from that image's corner, so on another crop
    box, downsample or voxel size they no longer lie on this run's image and
    masks. Judged by the extent recorded when it was built
    (``image_extent_um``) and, where it was recorded, the crop box; a graph
    saved before either was recorded is not judged by it. *remedy* says
    what to do."""
    saved = graph.dead_ends.graph_image_extent_um(G)
    now = graph.dead_ends.image_extent_um(shape, voxel_size_zyx)
    differs = saved is not None and not (
        saved.shape == now.shape and np.allclose(saved, now, rtol=1e-6, atol=1e-6)
    )
    if CROP_GRAPH_ATTRIBUTE in G.graph:
        box = G.graph[CROP_GRAPH_ATTRIBUTE] or {}
        whole = crop is None or crop.is_whole_volume
        differs = differs or whole or (
            tuple(box.get("start_zyx", ())), tuple(box.get("stop_zyx", ()))
        ) != (tuple(crop.start_zyx), tuple(crop.stop_zyx))
    if not differs:
        return
    built_on = (
        "" if saved is None
        else f" was built on an image {' x '.join(f'{v:g}' for v in saved)} um (z, y, x)"
    )
    raise ValueError(
        f"{which}{built_on} does not fit this run's image, "
        f"{' x '.join(f'{v:g}' for v in now)} um: it was made with a different "
        f"crop_on_import box, segmented_image_downsample or voxel size. {remedy}"
    )


#: The graph attribute holding :func:`skeleton_digest` of the skeleton a
#: saved graph was built from.
SKELETON_DIGEST = "skeleton_digest"


def skeleton_digest(skeleton: np.ndarray) -> str:
    """A digest of *skeleton*'s shape and voxels, read a plane at a time (a
    disk-backed one is never copied whole): what a saved graph was built
    from, so a run loading the graph can tell another skeleton."""
    digest = hashlib.sha256(repr(tuple(int(n) for n in skeleton.shape)).encode())
    for plane in skeleton:
        digest.update(np.packbits(np.asarray(plane, dtype=bool)).tobytes())
    return digest.hexdigest()


def _warn_if_built_from_another_skeleton(G: nx.MultiGraph, graph_path: Path, skeleton: np.ndarray) -> None:
    """Warn when the saved graph's skeleton (:data:`SKELETON_DIGEST`) is not
    this run's: a build that was cancelled, or a skeleton made again since,
    leaves a graph of another skeleton on the same grid, which the grid
    check cannot see. A graph saved before the digest was kept is not judged."""
    saved = G.graph.get(SKELETON_DIGEST)
    if saved is None or saved == skeleton_digest(skeleton):
        return
    logger.warning(
        f"The saved graph {graph_path} was built from another skeleton than this run's: the "
        "skeleton was made again, or the build that would have replaced the graph did not "
        "finish. Turn do_graph_building back on to build the graph from this skeleton."
    )


def write_pickle_whole(obj: Any, path: Path) -> None:
    """Pickle *obj* to *path* through a ``.part`` file moved over it once
    written, so the file is either the old one or the whole new one: a run
    stopped mid-write left half a graph that the next run loaded."""
    part = path.with_name(path.name + ".part")
    try:
        with part.open("wb") as handle:
            pickle.dump(obj, handle)
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    os.replace(part, path)


#: The segmentation-cleanup steps, each switched on by ``segmentation_cleanup_<step>``.
_SEGMENTATION_CLEANUP_STEPS = (
    "fill_cavities",
    "remove_whiskers",
    "split_narrow_necks",
    "close_gaps",
    "reconnect_gaps",
    "smooth_surfaces",
    "remove_small_volumes",
)


def _skeleton_without_vessels_error(
    settings: dict, volume: SkeletonisedVolume, skeleton_path: Path
) -> ValueError:
    """What :func:`build_network` raises when no two skeleton voxels touch --
    an empty skeleton, or one thinned to isolated points -- so there is no
    vessel to build a graph from. The cause is upstream of graph building:
    says how many voxels the skeleton and the segmented image have, and which
    segmentation or skeletonise settings to check."""

    def foreground_voxels(image: np.ndarray) -> int:
        binary = _to_binary_volume_for_skeletonization(
            image,
            use_memmap=settings["use_memmap_loading"],
            memmap_directory=settings["memmap_directory"],
        )
        count = int(np.count_nonzero(binary))
        preprocessing.release_superseded(binary, None, keep=image)
        return count

    skeleton_voxels = int(np.count_nonzero(volume.skeleton))
    if skeleton_voxels:
        found = (
            f"The skeleton has {skeleton_voxels} voxel(s) and no two of them touch, "
            "so there is no vessel to build a graph from."
        )
    else:
        found = "The skeleton is empty, so there is no vessel to build a graph from."
    if not settings["do_skeletonize"]:
        found += (
            f" It was loaded from {skeleton_path}: turn do_skeletonize on to make it "
            "again from the segmented image."
        )
    image_name = settings["input_path"].name
    image_voxels = None if volume.image is None else foreground_voxels(volume.image)
    if image_voxels == 0:
        # `image` is the mask skeletonising read, after any segmentation
        # cleanup; the mask from before cleanup is kept only when it ran.
        raw = volume.raw_segmented_image
        raw_voxels = 0 if raw is None else foreground_voxels(raw)
        if raw_voxels:
            cleanup_on = [
                f"segmentation_cleanup_{step}"
                for step in _SEGMENTATION_CLEANUP_STEPS
                if settings.get(f"segmentation_cleanup_{step}")
            ]
            return ValueError(
                f"{found} Segmentation cleanup removed all {raw_voxels} foreground voxels "
                f"of the segmented image ({image_name}) before skeletonising. Cleanup "
                f"steps that are on: {', '.join(cleanup_on)}."
            )
        return ValueError(
            f"{found} The segmented image ({image_name}) has no foreground either: check "
            "the segmentation that made it."
        )

    if image_voxels:
        found += (
            f" The segmented image ({image_name}) has {image_voxels} foreground voxels, "
            "so skeletonising left no vessel in them."
        )
    removes = []
    if settings["skeleton_min_branch_length"]:
        removes.append(
            f"skeleton_min_branch_length ({settings['skeleton_min_branch_length']} voxels) "
            "drops skeleton pieces shorter than that"
        )
    if settings["skeleton_min_component_percent"]:
        removes.append(
            f"skeleton_min_component_percent ({settings['skeleton_min_component_percent']}%) "
            "drops pieces smaller than that share of the skeleton"
        )
    check = (
        "Check the segmentation (isolated specks thin to isolated points, and a solid "
        "blob can thin to nothing)"
    )
    if removes:
        check += ", and the skeletonise settings that remove skeleton: " + "; ".join(removes)
    return ValueError(f"{found} {check}.")


def _mark_dead_end_kinds(
    settings: dict, G: nx.MultiGraph, binary_mask: np.ndarray, voxel_size_zyx, radius_at=None, *, support=None
) -> None:
    """Judge every dead end of *G* against the mask and write its kind on its
    node (``graph.dead_ends``), saying how many of each there are. *support*
    is the run's ``MaskSupport`` over *binary_mask*, made here when not given."""
    gap_um = float(settings["bridge_max_background_gap_um"])
    fraction = float(settings["bridge_min_mask_fraction"])
    support = support.limited_to(gap_um, fraction) if support is not None else preprocessing.MaskSupport(
        binary_mask,
        voxel_size_zyx,
        max_background_gap_um=gap_um,
        min_mask_fraction=fraction,
        feature_distance=getattr(radius_at, "feature_distance", None),
        low_memory=bool(settings.get("use_memmap_loading")),
        memmap_directory=settings.get("memmap_directory"),
    )
    ends = graph.dead_ends.classify_and_mark(
        G, support=support,
        stub_radius_multiple=dead_end_stub_radius_multiple(settings),
    )
    census = graph.dead_ends.dead_end_census(ends)
    logger.info(
        "Dead ends: %d -- %s.",
        len(ends),
        ", ".join(f"{count} {kind.replace('_', ' ')}" for kind, count in census.items() if count) or "none",
    )


def mark_dead_end_kinds(settings: Mapping[str, Any], G: nx.MultiGraph, context: NetworkContext) -> None:
    """What each dead end of *G* is, judged through *context*'s mask and
    written on its node, as :func:`build_network` marks the graph it saves:
    the image's extent and the thick vessels' centreline tree recorded (so
    every later reader of the graph judges a dead end at a thick vessel's
    wall alike), then every dead end's kind (``graph.dead_ends``). The
    settings optimiser's scorecard counts the same marks on its graphs."""
    graph.dead_ends.record_image_extent(G, context.binary_mask.shape, context.voxel_size_zyx)
    graph.dead_ends.record_thick_centreline(
        G,
        None if context.thick_tree is None else graph.thick_vessel_junctions.thick_centreline_points_um(
            context.thick_tree, context.voxel_size_zyx
        ),
        context.thick_tree_radius,
        tolerance_um=thick_bridge_centreline_tolerance_um(settings),
    )
    _mark_dead_end_kinds(
        settings, G, context.binary_mask, context.voxel_size_zyx, context.radius_at,
        support=context.mask_support,
    )


#: The edge mark on what :func:`_remove_and_recover_interior_dead_ends`'
#: recovery drew; a merge carries it as a share
#: (``graph.degree2.INTERIOR_RECOVERY_FRACTION``).
INTERIOR_RECOVERY = graph.degree2.INTERIOR_RECOVERY


def interior_recovery_um(G: nx.MultiGraph) -> float:
    """How many um of *G* the interior dead ends' recovery drew: every edge
    it marked, and the share of each vessel a merge made from one."""
    return float(sum(
        graph.degree2.drawn_um(data, graph.degree2.INTERIOR_RECOVERY_FRACTION) for *_e, data in G.edges(data=True)
    ))


#: Where :func:`_set_aside_earlier_recovery` keeps the build's own
#: ``recovered`` mark while the interior recovery runs.
_RECOVERED_BEFORE = "_recovered_before_interior_recovery"


def _set_aside_earlier_recovery(G: nx.MultiGraph) -> None:
    """Move the ``recovered`` mark of what the build's own recovery drew to
    :data:`_RECOVERED_BEFORE`, so the interior recovery judges only its own
    edges two-ended. An attribute, not a list of edges: that recovery joins
    by splitting vessels, and the pieces of a split edge copy its attributes
    under new keys -- kept as ``(u, v, key)``, a split earlier-recovered
    vessel came back unmarked."""
    for *_edge, data in G.edges(keys=True, data=True):
        if data.get("recovered") is not None:
            data[_RECOVERED_BEFORE] = data.pop("recovered")


def _restore_earlier_recovery(G: nx.MultiGraph) -> None:
    """Put back what :func:`_set_aside_earlier_recovery` set aside, on every
    piece the interior recovery split it into."""
    for *_edge, data in G.edges(keys=True, data=True):
        if _RECOVERED_BEFORE in data:
            data["recovered"] = data.pop(_RECOVERED_BEFORE)


def _remove_and_recover_interior_dead_ends(
    settings: dict,
    G: nx.MultiGraph,
    *,
    skeleton: np.ndarray,
    binary_mask: np.ndarray,
    voxel_size_zyx,
    radius_at,
    cleanup: "graph.assemble.LumenCleanup",
    thick_mask: np.ndarray | None = None,
    thick_tree=None,
    thick_tree_radius=None,
) -> nx.MultiGraph:
    """``remove_interior_dead_ends``: every interior dead end's chain out
    (``graph.dead_ends.remove_interior_dead_ends``), the mask no centreline's
    own radius then covers traced along the skeleton and kept only where it
    joins the network at both ends (``mask_recovery``'s *two_ended*, coverage
    by the radius alone), what it drew smoothed and bridged into thick
    vessels as graph building does, the lumen clean-up run with
    ``recovery_short_loop_keep_hole_over_um``, and the interior dead ends the
    clean-up leaves taken out again.

    Fitted to Harvey's marks on E14.5 (2026-10-21): G5 (17 of 29 dead ends
    that stop short "stop here"), the recovery he called the best graph yet,
    and grids L1 and L2 for the clean-up's hole limit. Not yet tried on
    another dataset."""
    graph.dead_ends.record_image_extent(G, skeleton.shape, voxel_size_zyx)
    support = cleanup.support
    _mark_dead_end_kinds(settings, G, binary_mask, voxel_size_zyx, radius_at, support=support)
    first = graph.dead_ends.remove_interior_dead_ends(G)
    _set_aside_earlier_recovery(G)
    G = graph.mask_recovery.recover_uncovered_mask_vessels(
        G,
        support,
        attach_reach_um=float(settings["final_orphan_reconnect_threshold"]),
        min_region_volume_um3=float(settings["recovery_min_region_volume_um3"]),
        min_length_um=float(settings["recovery_min_length_um"]),
        skeleton=skeleton,
        coverage_reach_radii=0.0,
        two_ended=True,
    )
    drawn = []
    for u, v, key, data in G.edges(keys=True, data=True):
        if data.get("recovered"):
            data[INTERIOR_RECOVERY] = True
            drawn.append((u, v, key))
    drawn_um = sum(float(G.edges[e].get("length") or 0.0) for e in drawn)
    _restore_earlier_recovery(G)
    if drawn and settings["smooth_centrelines"]:
        graph.smooth_graph_centrelines(
            G,
            skeleton,
            voxel_size_zyx=voxel_size_zyx,
            **centreline_smoothing_arguments(settings),
            radius_at=radius_at,
            edges=drawn,
        )
    if drawn and thick_mask is not None:
        G = graph.insert_thick_vessel_junction_nodes(
            G, thick_mask, voxel_size_zyx=voxel_size_zyx, edges=drawn, segmentation=binary_mask,
            centreline=thick_tree, **thick_junction_bridge_arguments(settings),
            mask_support=support, tree_radius_um=thick_tree_radius,
        )
    after_recovery = replace(
        cleanup, short_loop_keep_hole_um=float(settings["recovery_short_loop_keep_hole_over_um"]), memos={}
    )
    G = graph.assemble.consolidate_lumen(G, after_recovery)
    _mark_dead_end_kinds(settings, G, binary_mask, voxel_size_zyx, radius_at, support=support)
    second = graph.dead_ends.remove_interior_dead_ends(G)
    if second["dead_ends"]:
        G = graph.assemble.smart_multigraph_degree2_removal(
            G, None, max_degree=cleanup.max_degree, debug=cleanup.debug, inside_lumen=cleanup.inside_lumen,
        )
        graph.thick_vessel_junctions.remove_dangling_bridges(G)
    kept_um = interior_recovery_um(G)
    G.graph["interior_dead_ends_removed"] = first["dead_ends"] + second["dead_ends"]
    G.graph["interior_dead_ends_removed_um"] = first["length_um"] + second["length_um"]
    G.graph["interior_recovery_drawn_um"] = drawn_um
    G.graph["interior_recovery_kept_um"] = kept_um
    logger.info(
        "Interior dead ends: removed %d (%.0f um); recovery drew %.0f um joined at both ends, %.0f um of it "
        "still marked after the clean-up (hole limit %.1f um), which left %d more interior dead end(s) (%.0f um), "
        "removed too.",
        first["dead_ends"], first["length_um"], drawn_um, kept_um, after_recovery.short_loop_keep_hole_um,
        second["dead_ends"], second["length_um"],
    )
    return G


def _record_raw_evidence(
    settings: dict,
    G: nx.MultiGraph,
    binary_mask: np.ndarray,
    voxel_size_zyx,
    crop: CropBox | None,
    downsample: Downsample | None = None,
) -> None:
    """Calibrate the run's raw image round the mask (``preprocessing.raw_evidence``)
    and keep the calibration on the graph, so a saved graph says what the raw-image
    rules read. Nothing when the run names no raw image, or it has nothing to read.

    With ``raw_image_stain`` ``auto``, the stain the evidence was read under --
    told by ``raw_stain.read_raw_stain``, the Diameters stage's own rule -- is
    kept on the graph too (``raw_image_stain``, ``raw_image_stain_reading``), so
    that stage takes the build's answer rather than reading the image again, and
    one graph never names its raw image two ways."""
    for key in ("raw_image_stain", "raw_image_stain_reading", RAW_STAIN_IMAGE):
        G.graph.pop(key, None)
    G.graph.pop(preprocessing.raw_evidence.GRAPH_ATTRIBUTE, None)
    evidence = preprocessing.raw_evidence_from_settings(
        settings, binary_mask, voxel_size_zyx, box=crop, downsample=downsample
    )
    if evidence is None:
        G.graph.pop(preprocessing.raw_evidence.GRAPH_ATTRIBUTE, None)
        return
    G.graph[preprocessing.raw_evidence.GRAPH_ATTRIBUTE] = evidence.calibration.as_dict()
    if settings.get("raw_image_stain", "auto") not in raw_stain.STAINS:
        G.graph["raw_image_stain"] = evidence.calibration.stain
        G.graph["raw_image_stain_reading"] = evidence.calibration.reading
        G.graph[RAW_STAIN_IMAGE] = raw_image_identity(settings)


def _record_crop(
    G: nx.Graph, crop: CropBox | None, voxel_size_zyx, downsample: Downsample | None = None
) -> None:
    """Keep the ``crop_on_import`` box on the graph (io.crop.GRAPH_ATTRIBUTE):
    what every later volume read alongside the image is cut by, and what to add
    to a node ``pos`` to place it on the whole image. The box is voxels of the
    image as read, so its offset is measured in that grid's voxel size: with a
    *downsample*, the source's, not the run's."""
    if crop is None or crop.is_whole_volume:
        G.graph.pop(CROP_GRAPH_ATTRIBUTE, None)
        return
    source_voxel_size_zyx = (
        voxel_size_zyx if downsample is None else downsample.source_voxel_size_zyx
    )
    G.graph[CROP_GRAPH_ATTRIBUTE] = crop.to_record(source_voxel_size_zyx)


def _record_downsample(
    G: nx.Graph, settings: Mapping[str, Any], volume: "SkeletonisedVolume | None"
) -> None:
    """Keep on the graph what the network was built on
    (io.downsample.GRAPH_ATTRIBUTE): the segmented image's reduction and
    ``raw_image_downsample``'s factor, which every later volume read beside the
    image is placed by. The raw factor is the settings' as the stage runs, so a
    run resumed at Diameters reads the raw image as its settings now say; the
    image's reduction is the volume's, or the record's when a resumed network
    has no image at hand. A synthetic network has no image to reduce."""
    if graph.network_is_synthetic(settings):
        G.graph.pop(DOWNSAMPLE_GRAPH_ATTRIBUTE, None)
        return
    downsample = getattr(volume, "downsample", None) if volume is not None else None
    image = getattr(volume, "image", None) if volume is not None else None
    existing = G.graph.get(DOWNSAMPLE_GRAPH_ATTRIBUTE)
    if downsample is None and image is None and existing:
        downsample = Downsample.from_record(existing)
    voxel_size_zyx = (
        getattr(volume, "voxel_size_zyx", None) if volume is not None else None
    ) or G.graph.get("image_voxel_size_zyx")
    record_downsample_on_graph(
        G,
        downsample,
        raw_factor=downsample_factor(settings.get(RAW_DOWNSAMPLE_SETTING)),
        image_shape_zyx=None if image is None else tuple(int(n) for n in image.shape),
        voxel_size_zyx=voxel_size_zyx,
    )


def _record_graph_voxel_metadata(
    G: nx.Graph,
    main_voxel_size_xyz,
    voxel_size_zyx,
    large_arteriole_mask,
    large_venule_mask,
    large_arteriole_mask_voxel_size,
    large_venule_mask_voxel_size,
) -> None:
    """Store physical voxel-unit metadata used for skeleton/graph geometry and mask alignment."""
    G.graph["image_voxel_size_xyz"] = main_voxel_size_xyz
    G.graph["image_voxel_size_zyx"] = voxel_size_zyx
    if large_arteriole_mask is not None and large_venule_mask is not None:
        G.graph["large_arteriole_mask_voxel_size_xyz"] = tuple(
            float(v) for v in large_arteriole_mask_voxel_size
        )
        G.graph["large_venule_mask_voxel_size_xyz"] = tuple(
            float(v) for v in large_venule_mask_voxel_size
        )


class _SegmentationBoxes:
    """The run's segmented *image* read a box at a time as the binary mask its
    skeleton was made from, by the whole image's rule
    (:func:`_segmented_image_binarization_rule`, decided on the first read):
    no whole-volume copy. Read as ``> 0``, an ilastik export's background
    (2) was vessel, so on E14.5 every end filled its large mask and a
    capillary took an outlet again (2026-10-26)."""

    def __init__(self, image: np.ndarray, input_path: Any):
        self.image = image
        self.shape = image.shape
        self._input_path = input_path
        self._rule: Any = None

    def __getitem__(self, box) -> np.ndarray:
        part = np.asarray(self.image[box])
        if part.dtype == bool:
            return part
        if self._rule is None:
            self._rule = _segmented_image_binarization_rule(self._input_path, self.image)
        return np.asarray(self._rule(part), dtype=bool)


def _large_vessel_role_terminal_nodes(
    G: nx.Graph,
    image_shape: tuple[int, ...],
    settings: dict,
    voxel_size_zyx: tuple[float, float, float],
    *,
    role: str,
    mask: np.ndarray,
    node_role: str,
    exclude_nodes: list | None = None,
    segmentation: np.ndarray | None = None,
) -> list:
    """One large-vessel-network side's terminal node(s): mask stump, or override.

    ``large_vessel_inlet``/``large_vessel_outlet`` default to "mask_stump" --
    read straight off the mask's own image-edge geometry -- but can be
    overridden with any of the same methods (coordinates, volume, ...) the
    other four boundary roles use, via :func:`graph.select_boundary_nodes_for_role`.
    "mask_stump" is intercepted here rather than inside
    ``graph.boundaries.select_boundary_nodes_by_method``, which stays
    mask-agnostic. *exclude_nodes* keeps the other side's node(s) out of
    reach, mirroring how the plain inlet/outlet pair excludes inlet_nodes
    when it picks the outlet a few lines below in ``assign_boundaries``.
    """
    method = str(settings[f"{role}_node_selection_method"]).strip().lower()
    if method == "mask_stump":
        return graph.select_large_vessel_mask_stump_terminal_nodes_for_role(
            G,
            mask,
            node_role=node_role,
            voxel_size_zyx=voxel_size_zyx,
            image_shape=image_shape,
            coordinates_setting_name=f"{role} mask stump",
            exclude_nodes=exclude_nodes,
            min_component_fraction=float(
                settings["large_vessel_stump_min_component_fraction"]
            ),
            segmentation=segmentation,
        )
    return graph.select_boundary_nodes_for_role(
        G, image_shape, settings, role, exclude_nodes=exclude_nodes
    )


def _reset_run_node_list(settings: dict, name: str) -> None:
    """Empty the run-filled node list *name*, in place when it is a list.

    In place, because the panel reuses one settings dict across runs and
    holds these lists. A config can hold something else -- a lone node ID
    typed into the row (``large_vessel_outlet_nodes: 3316``) -- which has no
    ``[:]`` to clear, so that is replaced with a fresh list.
    """
    if isinstance(settings.get(name), list):
        settings[name][:] = []
    else:
        settings[name] = []


def _vessel_boundary_configured(settings: dict, vessel: str) -> bool:
    """Whether the arteriole or venule boundary has the list its method reads.

    A list another method left behind does not count. Node IDs still set
    after switching back to coordinates used to make the selector raise,
    because it was given no coordinates; with ``edge_percent`` the same
    leftover list made it pick terminals.
    """
    return graph.boundary_role_configured(settings, f"{vessel}_boundary")


def assign_boundaries(settings: dict, network: VesselNetwork):
    """Choose the inlet, outlet and vessel-boundary nodes for this network."""
    if network.volume.synthetic:
        return _assign_synthetic_boundaries(settings, network)
    G = network.graph
    image = network.volume.image
    voxel_size_zyx = network.volume.voxel_size_zyx
    # A role whose switch is out of effect is not used, whatever masks the
    # network carries (a resumed one may hold those of an earlier run).
    large_in_use = graph.vessel_masks_in_use(settings, "large")
    small_in_use = graph.vessel_masks_in_use(settings, "small")
    large_vessel_tier = graph.assigns_large_vessel_branch_orders(settings)
    large_arteriole_mask = network.large_arteriole_mask if large_in_use else None
    large_venule_mask = network.large_venule_mask if large_in_use else None
    small_arteriole_mask = network.small_arteriole_mask if small_in_use else None
    small_venule_mask = network.small_venule_mask if small_in_use else None
    # Global vessel masks (no venule mask entered) do not tell an arteriole
    # from a venule, which every mask step gated below reads them for: the
    # cut, the inlets and outlets, the Large tier and the small-vessel
    # boundaries. They only tag each vessel's size class, further down, and
    # the inlets and outlets come from the Inlet and Outlet tabs.
    global_masks = io.vessel_masks_are_global(settings)
    if global_masks:
        large_in_use = small_in_use = large_vessel_tier = False
    # Whether the masks also choose the inlets and outlets; off, they only
    # drive the large-vessel treatment and the Inlet/Outlet methods choose.
    io_from_masks = graph.inlets_outlets_from_vessel_masks(settings)
    auto_inlet_nodes: list[int] = []
    auto_outlet_nodes: list[int] = []
    if large_in_use:
        if large_arteriole_mask is None or large_venule_mask is None:
            raise ValueError(
                "use_large_vessel_masks is on, but no large arteriole and venule "
                "masks were loaded. Set large_arteriole_mask_path and "
                "large_venule_mask_path."
            )
        # Start from load-time filtered masks (dilation, min-component volume,
        # opposite-attached cleanup, load-time overlap exclusion in
        # io.load_and_validate_vessel_masks during build_network). Re-binarise
        # with the cut helper — never raw dtype=bool, which turns 1/2 labels
        # into an all-True volume and would wipe the network at cut time.
        from haemolynx.graph.cut_at_large_vessel_volumes import (
            _as_large_vessel_foreground,
        )

        assignment_large_arteriole_mask = _as_large_vessel_foreground(
            large_arteriole_mask, role="arteriole"
        )
        assignment_large_venule_mask = _as_large_vessel_foreground(
            large_venule_mask, role="venule"
        )
        cleanup_enabled_for_large = bool(
            settings["automated_vessel_assignment_enable_overlap_cleanup"]
        )
        effective_assignment_fast_mode = bool(
            settings["automated_vessel_assignment_fast_mode"]
        )
        apply_overlap_cleanup_prepass = bool(
            cleanup_enabled_for_large
            and (
                effective_assignment_fast_mode
                or settings[
                    "automated_vessel_assignment_apply_overlap_cleanup_in_normal_mode"
                ]
            )
        )
        use_legacy_large_vessel_assignment = bool(
            settings["automated_vessel_assignment_use_legacy_mode"]
        )
        effective_max_large_vessel_dilation_microns = float(
            settings["large_vessel_assignment_max_dilation_microns"]
        )
        if not use_legacy_large_vessel_assignment:
            quality_metrics = graph.assess_large_vessel_assignment_quality(
                G,
                large_arteriole_mask=assignment_large_arteriole_mask,
                large_venule_mask=assignment_large_venule_mask,
                voxel_size_zyx=voxel_size_zyx,
                quality_max_overlap_fraction=float(
                    settings["automated_vessel_quality_max_overlap_fraction"]
                ),
                quality_min_terminal_coverage=float(
                    settings["automated_vessel_quality_min_terminal_coverage"]
                ),
                quality_max_component_count=int(
                    settings["automated_vessel_quality_max_component_count"]
                ),
            )
            logger.info(
                "Robust large-vessel quality gate: "
                f"overlap_fraction={float(quality_metrics['overlap_fraction']):.4f}, "
                f"terminal_coverage={float(quality_metrics['terminal_coverage']):.4f}, "
                f"poor_quality={bool(quality_metrics['poor_quality'])}."
            )
            if bool(quality_metrics["poor_quality"]):
                effective_assignment_fast_mode = False
                apply_overlap_cleanup_prepass = False
                effective_max_large_vessel_dilation_microns = min(
                    float(effective_max_large_vessel_dilation_microns),
                    float(settings["automated_vessel_conservative_max_dilation_microns"]),
                )
                logger.info(
                    "Robust large-vessel assignment switched to conservative mode: "
                    "disabling overlap-cleanup pre-pass and reducing max dilation "
                    f"to {float(effective_max_large_vessel_dilation_microns):.3f} microns."
                )
        if effective_assignment_fast_mode and cleanup_enabled_for_large:
            volume_stride = int(settings["large_vessel_3d_volume_downsample_stride"])
            if bool(settings["write_fast_mode_preassignment_large_vessel_debug_3d_html"]):
                before_html = (
                    Path(settings["plot_dir"])
                    / "pre_assignment_large_vessel_masks_before_overlap_cleanup_3d.html"
                )
                Path(settings["plot_dir"]).mkdir(parents=True, exist_ok=True)
                visualization.visualize_3d_plotly_large_vessel_assignment(
                    G,
                    large_arteriole_mask=assignment_large_arteriole_mask,
                    large_venule_mask=assignment_large_venule_mask,
                    input_nodes=[],
                    output_nodes=[],
                    voxel_size_zyx=voxel_size_zyx,
                    volume_downsample_stride=volume_stride,
                    title=(
                        "Pre-Assignment Debug View (Fast Mode, Before Overlap Cleanup): "
                        "Large-Vessel Masks + Graph"
                    ),
                    save_html_path=str(before_html),
                    show=False,
                )
                logger.info(
                    "Saved fast-mode pre-assignment (before cleanup) large-vessel "
                    f"debug 3D visualization to: {before_html}"
                )
            cleaned_arteriole_mask, cleaned_venule_mask = (
                graph.exclude_smaller_overlapping_large_vessel_components(
                    assignment_large_arteriole_mask,
                    assignment_large_venule_mask,
                    low_memory=bool(settings["use_memmap_loading"]),
                    memmap_directory=settings["memmap_directory"],
                )
            )
            if cleaned_arteriole_mask is not None and cleaned_venule_mask is not None:
                assignment_large_arteriole_mask = cleaned_arteriole_mask
                assignment_large_venule_mask = cleaned_venule_mask
            logger.info(
                "Automated large-vessel assignment fast mode: applied overlap "
                "cleanup before progressive terminal assignment."
            )
            if bool(settings["write_fast_mode_preassignment_large_vessel_debug_3d_html"]):
                after_html = (
                    Path(settings["plot_dir"])
                    / "pre_assignment_large_vessel_masks_after_overlap_cleanup_3d.html"
                )
                visualization.visualize_3d_plotly_large_vessel_assignment(
                    G,
                    large_arteriole_mask=assignment_large_arteriole_mask,
                    large_venule_mask=assignment_large_venule_mask,
                    input_nodes=[],
                    output_nodes=[],
                    voxel_size_zyx=voxel_size_zyx,
                    volume_downsample_stride=volume_stride,
                    title=(
                        "Pre-Assignment Debug View (Fast Mode, After Overlap Cleanup): "
                        "Large-Vessel Masks + Graph"
                    ),
                    save_html_path=str(after_html),
                    show=False,
                )
                logger.info(
                    "Saved fast-mode pre-assignment (after cleanup) large-vessel "
                    f"debug 3D visualization to: {after_html}"
                )
        elif apply_overlap_cleanup_prepass:
            logger.info(
                "Automated large-vessel assignment: overlap cleanup will run "
                "once, inside the legacy inlet/outlet selection (normal mode)."
            )
        # Cut uses post-filter masks (load-time filters + assignment-time overlap
        # cleanup above). Progressive dilation is assignment-only and does not
        # redefine the cut volume. Cut runs before terminal selection so new
        # degree-1 boundary nodes are available, and network.graph / napari use
        # the post-cut graph.
        network.large_arteriole_mask = assignment_large_arteriole_mask
        network.large_venule_mask = assignment_large_venule_mask
        if bool(settings["cut_network_at_large_vessel_volumes"]):
            G = graph.cut_graph_at_large_vessel_volumes(
                G,
                assignment_large_arteriole_mask,
                assignment_large_venule_mask,
                voxel_size_zyx=voxel_size_zyx,
                enabled=True,
                remove_orphaned_branches=bool(
                    settings["remove_orphaned_branches_outside_large_vessel_volumes"]
                ),
                orphaned_branch_max_edge_count=int(
                    settings["orphaned_branch_max_edge_count"]
                ),
            )
            network.graph = G
            # The ends the cut made, judged as graph building judged the rest
            # (geometry only: the stage holds no mask support), so every dead
            # end carries its kind.
            cut_ends = [
                node for node, data in G.nodes(data=True)
                if data.get(graph.cut_at_large_vessel_volumes.CUT_FROM_LARGE_VESSEL) is not None and G.degree(node) == 1
            ]
            graph.dead_ends.reclassify_dead_ends(
                G, cut_ends, stub_radius_multiple=dead_end_stub_radius_multiple(settings)
            )
            # {stem}_graph.pkl stays graph building's: a run loading it
            # (do_graph_building off) cuts it again by the masks it then has.
            logger.info(
                "Applied large-vessel volume network cut on filtered masks before "
                "automated terminal assignment "
                f"({G.number_of_nodes()} nodes, {G.number_of_edges()} edges remain)."
            )
        if large_vessel_tier:
            # A manual override (e.g. degree_1_from_inlet) reads settings
            # like inlet_nodes through the normal graph.select_boundary_nodes_for_role
            # path -- reset them first so an override cannot see inlet_nodes
            # left over from an earlier run of this same settings dict (the
            # napari GUI reuses one across runs). They are recomputed from
            # scratch below regardless of which branch this run takes.
            _reset_run_node_list(settings, "inlet_nodes")
            _reset_run_node_list(settings, "outlet_nodes")
            segmentation = _SegmentationBoxes(image, settings.get("input_path"))
            auto_inlet_nodes = _large_vessel_role_terminal_nodes(
                G,
                image.shape,
                settings,
                voxel_size_zyx,
                role="large_vessel_inlet",
                mask=assignment_large_arteriole_mask,
                node_role="inlet",
                segmentation=segmentation,
            )
            # Written now, not only after this whole block, so a
            # large_vessel_outlet override using degree_1_from_inlet measures
            # distance from the inlet this run just chose, not an empty list.
            settings["inlet_nodes"][:] = list(auto_inlet_nodes)
            auto_outlet_nodes = _large_vessel_role_terminal_nodes(
                G,
                image.shape,
                settings,
                voxel_size_zyx,
                role="large_vessel_outlet",
                mask=assignment_large_venule_mask,
                node_role="outlet",
                exclude_nodes=auto_inlet_nodes,
                segmentation=segmentation,
            )
            logger.info(
                "Large-vessel network mode: arteriole/venule mask interior "
                "kept in the network and seeded from the image-edge stump "
                f"({len(auto_inlet_nodes)} inlet node(s), "
                f"{len(auto_outlet_nodes)} outlet node(s)), instead of "
                "being cut away."
            )
        elif not io_from_masks:
            # Nothing reads the masks' own terminals: the Inlet and Outlet
            # methods choose, and there is no Large tier to seed.
            pass
        elif use_legacy_large_vessel_assignment:
            auto_inlet_nodes, auto_outlet_nodes = (
                graph.select_terminal_nodes_from_large_vessel_masks_progressive_dilation(
                    G,
                    large_arteriole_mask=assignment_large_arteriole_mask,
                    large_venule_mask=assignment_large_venule_mask,
                    voxel_size_zyx=voxel_size_zyx,
                    max_dilation_microns=float(
                        effective_max_large_vessel_dilation_microns
                    ),
                    dilation_step_microns=5.0,
                    allow_overlap=False,
                    low_memory=bool(settings["use_memmap_loading"]),
                    exclude_smaller_overlapping_volumes=(
                        apply_overlap_cleanup_prepass and not effective_assignment_fast_mode
                    ),
                )
            )
        else:
            robust_assignment = (
                graph.select_terminal_nodes_from_large_vessel_masks_progressive_dilation_confidence(
                    G,
                    large_arteriole_mask=assignment_large_arteriole_mask,
                    large_venule_mask=assignment_large_venule_mask,
                    voxel_size_zyx=voxel_size_zyx,
                    max_dilation_microns=float(
                        effective_max_large_vessel_dilation_microns
                    ),
                    dilation_step_microns=5.0,
                    confidence_margin=float(
                        settings["automated_vessel_confidence_margin"]
                    ),
                    minimum_confidence=float(settings["automated_vessel_min_confidence"]),
                    topology_penalty=float(
                        settings["automated_vessel_topology_penalty"]
                    ),
                    quality_max_overlap_fraction=float(
                        settings["automated_vessel_quality_max_overlap_fraction"]
                    ),
                    quality_min_terminal_coverage=float(
                        settings["automated_vessel_quality_min_terminal_coverage"]
                    ),
                    quality_max_component_count=int(
                        settings["automated_vessel_quality_max_component_count"]
                    ),
                    conservative_max_dilation_microns=float(
                        settings["automated_vessel_conservative_max_dilation_microns"]
                    ),
                    low_memory=bool(settings["use_memmap_loading"]),
                    memmap_directory=settings["memmap_directory"],
                )
            )
            auto_inlet_nodes = list(robust_assignment["input_nodes"])
            auto_outlet_nodes = list(robust_assignment["output_nodes"])
            logger.info(
                "Robust large-vessel assignment confidence mode: "
                f"inputs={len(auto_inlet_nodes)}, outputs={len(auto_outlet_nodes)}, "
                f"unresolved={len(robust_assignment['unresolved_nodes'])}, "
                f"conservative={bool(robust_assignment['conservative_mode'])}, "
                f"effective_max_dilation="
                f"{float(robust_assignment['effective_max_dilation_microns']):.3f} microns."
            )
        large_arteriole_mask = assignment_large_arteriole_mask
        large_venule_mask = assignment_large_venule_mask
    if large_in_use and not io_from_masks:
        logger.info(
            "inlets_outlets_from_vessel_masks is off: the inlets and outlets come from "
            f"inlet_node_selection_method={settings['inlet_node_selection_method']!r} "
            f"and outlet_node_selection_method={settings['outlet_node_selection_method']!r}"
            + (
                f"; the large masks' stumps seed the Large tier only ({len(auto_inlet_nodes)} "
                f"arteriole, {len(auto_outlet_nodes)} venule)."
                if large_vessel_tier
                else "."
            )
        )
    if io_from_masks:
        if not auto_inlet_nodes:
            if large_vessel_tier:
                method = settings["large_vessel_inlet_node_selection_method"]
                raise ValueError(
                    f"large_vessel_inlet_node_selection_method={method!r} found no "
                    "inlet node for the arteriole mask"
                    + (" -- the large arteriole mask touches no edge of the image, so it "
                       "has no stump" if method == "mask_stump" else "")
                    + ". Fix: on the 'Large vessel inlet' tab, choose the node with "
                    "the volume method (Insert a box, then Use selected node) or "
                    "node_ids."
                )
            raise ValueError(
                "automated_vessel_assignment=True found no terminal nodes in the "
                "arteriole mask (after any configured dilation). Fix: widen the "
                "mask's dilation, or turn inlets_outlets_from_vessel_masks off and "
                "choose the inlets on the Inlet tab."
            )
        if not auto_outlet_nodes:
            if large_vessel_tier:
                method = settings["large_vessel_outlet_node_selection_method"]
                raise ValueError(
                    f"large_vessel_outlet_node_selection_method={method!r} found no "
                    "outlet node for the venule mask"
                    + (" -- the large venule mask touches no edge of the image, so it "
                       "has no stump" if method == "mask_stump" else "")
                    + ". Fix: on the 'Large vessel outlet' tab, choose the node with "
                    "the volume method (Insert a box, then Use selected node) or "
                    "node_ids."
                )
            raise ValueError(
                "automated_vessel_assignment=True found no terminal nodes in the "
                "venule mask (after any configured dilation). Fix: widen the "
                "mask's dilation, or turn inlets_outlets_from_vessel_masks off and "
                "choose the outlets on the Outlet tab."
            )
        # The Inlet and Outlet tabs' own coordinates are left as they are:
        # written here, a settings dict reused for a run with the masks off
        # took these nodes as hand-picked inlets.
        logger.info(
            f"Automated vessel assignment selected {len(auto_inlet_nodes)} inlet node(s) "
            f"from arteriole-mask overlap and {len(auto_outlet_nodes)} outlet node(s) from "
            "venule-mask overlap."
        )

    _reset_run_node_list(settings, "inlet_nodes")
    _reset_run_node_list(settings, "outlet_nodes")
    _reset_run_node_list(settings, "arteriole_boundary_nodes")
    _reset_run_node_list(settings, "venule_boundary_nodes")
    _reset_run_node_list(settings, "large_vessel_inlet_nodes")
    _reset_run_node_list(settings, "large_vessel_outlet_nodes")
    if io_from_masks:
        # Use direct terminal-node overlap assignment from vessel masks.
        inlet_nodes = auto_inlet_nodes
        settings["inlet_nodes"].extend(inlet_nodes)
        outlet_nodes = [node_id for node_id in auto_outlet_nodes if node_id not in set(inlet_nodes)]
    else:
        # Each role reads its own three settings; naming the role is enough.
        inlet_nodes = graph.select_boundary_nodes_for_role(
            G, image.shape, settings, "inlet"
        )
        # Written before the outlets are chosen: degree_1_from_inlet measures
        # from them, and found none when they were written after.
        settings["inlet_nodes"].extend(inlet_nodes)
        outlet_nodes = graph.select_boundary_nodes_for_role(
            G, image.shape, settings, "outlet", exclude_nodes=inlet_nodes
        )
    settings["outlet_nodes"].extend(outlet_nodes)
    if large_vessel_tier:
        settings["large_vessel_inlet_nodes"][:] = list(auto_inlet_nodes)
        settings["large_vessel_outlet_nodes"][:] = list(auto_outlet_nodes)
    used_nodes = set(settings["inlet_nodes"]) | set(settings["outlet_nodes"])
    # While the small masks are in use, or any mask is global (no branch
    # orders, so nothing reads them), the panel greys these manual lists and
    # a batch does not read them: neither does the run.
    manual_vessel_boundaries = graph.manual_vessel_boundaries_read(settings)
    if manual_vessel_boundaries and _vessel_boundary_configured(settings, "arteriole"):
        art_boundary = graph.select_boundary_nodes_for_role(
            G, image.shape, settings, "arteriole_boundary", exclude_nodes=list(used_nodes)
        )
        settings["arteriole_boundary_nodes"].extend(art_boundary)
        used_nodes.update(settings["arteriole_boundary_nodes"])

    if manual_vessel_boundaries and _vessel_boundary_configured(settings, "venule"):
        ven_boundary = graph.select_boundary_nodes_for_role(
            G, image.shape, settings, "venule_boundary", exclude_nodes=list(used_nodes)
        )
        settings["venule_boundary_nodes"].extend(ven_boundary)

    _reset_run_node_list(settings, "large_arteriole_boundary_nodes")
    _reset_run_node_list(settings, "large_venule_boundary_nodes")
    if large_vessel_tier:
        # Which edges fall inside the large-vessel masks, and where that
        # coverage ends -- the Large_Art/Large_Ven <-> Art/Ven hand-off
        # point (graph/branch_order.py's hierarchical tier extension).
        # Reuses the small-vessel-mask overlap-labelling machinery against
        # the large masks instead; its mask_vessel_type diagnostic
        # attribute is a visualization-only side effect (Phase 2), not used
        # here -- if use_small_vessel_masks_for_boundary_assignment is also
        # on, whichever of the two calls runs last "wins" that diagnostic
        # attribute, which is harmless since neither branch-order
        # assignment nor resistance reads it back. The tier is read from
        # the large-mask tag written below instead: the boundary nodes
        # alone cannot say which way the vessel runs on at a junction
        # where a side branch leaves it.
        large_boundary_results = graph.infer_boundary_nodes_from_small_vessel_masks(
            G,
            assignment_large_arteriole_mask,
            assignment_large_venule_mask,
            voxel_size_zyx=voxel_size_zyx,
            minimum_overlap_fraction=float(
                settings["large_vessel_mask_min_overlap_fraction"]
            ),
        )
        graph.tag_large_mask_vessel_types(
            G,
            large_boundary_results["arteriole_edges"],
            large_boundary_results["venule_edges"],
        )
        settings["large_arteriole_boundary_nodes"][:] = list(
            large_boundary_results["arteriole_boundary_nodes"]
        )
        settings["large_venule_boundary_nodes"][:] = list(
            large_boundary_results["venule_boundary_nodes"]
        )
        logger.info(
            "Large-vessel-network branch-order tagging: "
            f"{len(large_boundary_results['arteriole_edges'])} edge(s) in the large "
            f"arteriole mask, {len(large_boundary_results['venule_edges'])} in the large "
            f"venule mask; {len(settings['large_arteriole_boundary_nodes'])} Large_Art "
            f"boundary node(s), {len(settings['large_venule_boundary_nodes'])} "
            "Large_Ven boundary node(s) (min_overlap_fraction="
            f"{float(settings['large_vessel_mask_min_overlap_fraction']):.3f})."
        )
    if not small_in_use:
        # Boundary nodes chosen by hand, or none: the tiers end at them, and
        # small-mask tags a resumed graph keeps from an earlier run must not
        # be read as this run's.
        graph.tag_small_mask_vessel_types(G)
    if small_in_use:
        if small_arteriole_mask is None or small_venule_mask is None:
            raise ValueError(
                "use_small_vessel_masks_for_boundary_assignment=True requires "
                "small_arteriole_mask_path and small_venule_mask_path."
            )
        assignment_small_arteriole_mask = np.asarray(small_arteriole_mask, dtype=bool)
        assignment_small_venule_mask = np.asarray(small_venule_mask, dtype=bool)
        cleanup_enabled_for_small = bool(
            settings["small_vessel_boundary_assignment_enable_overlap_cleanup"]
        )
        small_fast_mode = bool(settings["small_vessel_boundary_assignment_fast_mode"])
        apply_small_overlap_cleanup_prepass = bool(
            cleanup_enabled_for_small
            and (
                small_fast_mode
                or settings[
                    "small_vessel_boundary_assignment_apply_overlap_cleanup_in_normal_mode"
                ]
            )
        )
        if apply_small_overlap_cleanup_prepass:
            cleaned_small_arteriole_mask, cleaned_small_venule_mask = (
                graph.exclude_smaller_overlapping_small_vessel_components(
                    assignment_small_arteriole_mask,
                    assignment_small_venule_mask,
                    low_memory=bool(settings["use_memmap_loading"]),
                    memmap_directory=settings["memmap_directory"],
                )
            )
            if (
                cleaned_small_arteriole_mask is not None
                and cleaned_small_venule_mask is not None
            ):
                assignment_small_arteriole_mask = cleaned_small_arteriole_mask
                assignment_small_venule_mask = cleaned_small_venule_mask
            logger.info(
                "Small-vessel boundary assignment: applied overlap cleanup "
                "before progressive labelling "
                f"(fast_mode={small_fast_mode})."
            )
        if bool(settings["small_vessel_tangential_redefinition_enable"]):
            redefinition_result = graph.redefine_small_masks_from_large_tangential_contact(
                small_arteriole_mask=assignment_small_arteriole_mask,
                small_venule_mask=assignment_small_venule_mask,
                large_arteriole_mask=large_arteriole_mask,
                large_venule_mask=large_venule_mask,
                voxel_size_zyx=voxel_size_zyx,
                enable_redefinition=True,
                max_contact_distance_microns=float(
                    settings[
                        "small_vessel_tangential_redefinition_max_contact_distance_microns"
                    ]
                ),
                touch_distance_microns=float(
                    settings["small_vessel_tangential_redefinition_touch_distance_microns"]
                ),
                tangency_cosine_max=float(
                    settings["small_vessel_tangential_redefinition_tangency_cosine_max"]
                ),
                reassignment_margin=float(
                    settings["small_vessel_tangential_redefinition_margin"]
                ),
                min_contact_fraction=float(
                    settings["small_vessel_tangential_redefinition_min_contact_fraction"]
                ),
                enable_sandwiched_component_reassignment=bool(
                    settings["small_vessel_sandwiched_reassignment_enable"]
                ),
                sandwiched_max_endpoint_distance_microns=float(
                    settings["small_vessel_sandwiched_max_gap_microns"]
                ),
                sandwiched_min_facing_cosine=float(
                    settings["small_vessel_sandwiched_min_facing_cosine"]
                ),
                sandwiched_max_axis_angle_degrees=float(
                    settings["small_vessel_sandwiched_max_axis_angle_degrees"]
                ),
                reassignment_parallel_workers=int(
                    settings["small_vessel_tangential_redefinition_parallel_workers"]
                ),
                low_memory=bool(settings["use_memmap_loading"]),
                memmap_directory=settings["memmap_directory"],
            )
            assignment_small_arteriole_mask = np.asarray(
                redefinition_result["small_arteriole_mask"], dtype=bool
            )
            assignment_small_venule_mask = np.asarray(
                redefinition_result["small_venule_mask"], dtype=bool
            )
        if bool(settings["small_vessel_mask_continuity_enable"]):
            continuity_result = graph.enforce_small_vessel_mask_continuity(
                small_arteriole_mask=assignment_small_arteriole_mask,
                small_venule_mask=assignment_small_venule_mask,
                large_arteriole_mask=large_arteriole_mask,
                large_venule_mask=large_venule_mask,
                voxel_size_zyx=voxel_size_zyx,
                enable_continuity=True,
                allow_small_to_large=bool(
                    settings["small_vessel_mask_continuity_allow_small_to_large"]
                ),
                allow_small_to_small=bool(
                    settings["small_vessel_mask_continuity_allow_small_to_small"]
                ),
                enforce_cylinder_only=bool(
                    settings["small_vessel_mask_continuity_enforce_cylinder_only"]
                ),
                min_cylindricality=float(
                    settings["small_vessel_mask_continuity_min_cylindricality"]
                ),
                max_axis_angle_degrees=float(
                    settings["small_vessel_mask_continuity_max_axis_angle_degrees"]
                ),
                min_facing_cosine=float(
                    settings["small_vessel_mask_continuity_min_facing_cosine"]
                ),
                max_radius_ratio=float(
                    settings["small_vessel_mask_continuity_max_radius_ratio"]
                ),
                max_bridge_distance_microns=float(
                    settings["small_vessel_mask_continuity_max_bridge_distance_microns"]
                ),
                corridor_max_distance_microns=float(
                    settings["small_vessel_mask_continuity_corridor_max_distance_microns"]
                ),
                opposite_exclusion_distance_microns=float(
                    settings[
                        "small_vessel_mask_continuity_opposite_exclusion_distance_microns"
                    ]
                ),
                use_gpu_acceleration=bool(
                    settings["use_gpu_mask_continuity_acceleration"]
                ),
                low_memory=bool(settings["use_memmap_loading"]),
                memmap_directory=settings["memmap_directory"],
            )
            assignment_small_arteriole_mask = np.asanyarray(
                continuity_result["small_arteriole_mask"], dtype=bool
            )
            assignment_small_venule_mask = np.asanyarray(
                continuity_result["small_venule_mask"], dtype=bool
            )
            continuity_stats = continuity_result["stats"]
            logger.info(
                "Small-vessel continuity bridging applied: "
                f"arteriole accepted="
                f"{int(continuity_stats['arteriole']['accepted_bridges'])}/"
                f"{int(continuity_stats['arteriole']['attempted_bridges'])}, "
                f"venule accepted="
                f"{int(continuity_stats['venule']['accepted_bridges'])}/"
                f"{int(continuity_stats['venule']['attempted_bridges'])}."
            )
        inferred_boundary_results = (
            graph.infer_boundary_nodes_from_small_vessel_masks_progressive_dilation(
                G,
                small_arteriole_mask=assignment_small_arteriole_mask,
                small_venule_mask=assignment_small_venule_mask,
                voxel_size_zyx=voxel_size_zyx,
                max_dilation_microns=float(settings["small_vessel_mask_dilation_microns"]),
                dilation_step_microns=5.0,
                minimum_overlap_fraction=float(
                    settings["small_vessel_mask_min_overlap_fraction"]
                ),
                allow_overlap=False,
                exclude_smaller_overlapping_volumes=False,
                low_memory=bool(settings["use_memmap_loading"]),
            )
        )
        small_arteriole_mask = assignment_small_arteriole_mask
        small_venule_mask = assignment_small_venule_mask
        settings["arteriole_boundary_nodes"][:] = list(
            inferred_boundary_results["arteriole_boundary_nodes"]
        )
        settings["venule_boundary_nodes"][:] = list(inferred_boundary_results["venule_boundary_nodes"])
        # The labelling the boundary nodes were read from, kept on the
        # vessels. Every junction where a capillary leaves a small vessel is
        # one of its boundary nodes, not only where its mask ends: the Art/Ven
        # tier reads this tag to run on along the vessel past them, and Post
        # processing judges a hand-edited vessel beside it.
        graph.tag_small_mask_vessel_types(
            G,
            inferred_boundary_results["arteriole_edges"],
            inferred_boundary_results["venule_edges"],
        )
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
        fallback = _small_vessel_boundary_fallback(
            settings,
            G,
            small_masks=(assignment_small_arteriole_mask, assignment_small_venule_mask),
            # An inlet's own vessel is covered by the small mask of its class
            # or, with the large vessels kept in the network, by the large
            # one: the inlet then sits on the large arteriole, which the small
            # arteriole mask never covers, and every inlet read as uncovered.
            large_masks=(
                (large_arteriole_mask, large_venule_mask)
                if large_vessel_tier
                and large_arteriole_mask is not None
                and large_venule_mask is not None
                else None
            ),
            voxel_size_zyx=voxel_size_zyx,
        )
        fallback_hops = int(settings["small_vessel_boundary_fallback_hop_distance"])
        for role, seed_key, boundary_key in (
            ("arteriole", "inlet_nodes", "arteriole_boundary_nodes"),
            ("venule", "outlet_nodes", "venule_boundary_nodes"),
        ):
            uncovered, hop_nodes = fallback[boundary_key]
            if not uncovered:
                continue
            logger.warning(
                f"{len(uncovered)} of {len(settings[seed_key])} {seed_key[:-6]} node(s) "
                f"have no edge in the small-{role} mask: {uncovered}."
            )
            if not settings["small_vessel_boundary_fallback_to_hop_distance"]:
                continue
            # Only those seeds, and added to what the masks found: the whole
            # list used to be replaced by every inlet's hop neighbours.
            added = [node_id for node_id in hop_nodes if node_id not in settings[boundary_key]]
            settings[boundary_key].extend(added)
            logger.warning(
                f"Hop-distance fallback: added {len(added)} {role} boundary node(s) "
                f"{fallback_hops} edge(s) from those; the "
                f"{len(settings[boundary_key]) - len(added)} the masks found are kept."
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
    if global_masks:
        _tag_vessels_in_global_masks(
            settings,
            G,
            large_mask=io.as_global_vessel_mask(
                large_arteriole_mask, large_venule_mask,
                use_memmap=settings["use_memmap_loading"], memmap_directory=settings["memmap_directory"],
            ),
            small_mask=io.as_global_vessel_mask(
                small_arteriole_mask, small_venule_mask,
                use_memmap=settings["use_memmap_loading"], memmap_directory=settings["memmap_directory"],
            ),
            voxel_size_zyx=voxel_size_zyx,
        )
    if io_from_masks:
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

    if large_in_use and settings["visualize_results"]:
        if large_arteriole_mask is None or large_venule_mask is None:
            raise ValueError(
                "use_large_vessel_masks=True requires large arteriole/venule masks "
                "for visualization."
            )
        has_small_viz_masks = (
            small_arteriole_mask is not None and small_venule_mask is not None
        )
        if not has_small_viz_masks:
            logger.info(
                "Small-vessel volume overlay unavailable for final large-vessel "
                "assignment view; skipping small-vessel volume rendering."
            )
        final_assignment_html_path = (
            Path(settings["plot_dir"]) / "final_graph_large_vessel_assignment_3d.html"
        )
        Path(settings["plot_dir"]).mkdir(parents=True, exist_ok=True)
        assignment_fig = visualization.visualize_3d_plotly_large_vessel_assignment(
            G,
            large_arteriole_mask=large_arteriole_mask,
            large_venule_mask=large_venule_mask,
            small_arteriole_mask=small_arteriole_mask if has_small_viz_masks else None,
            small_venule_mask=small_venule_mask if has_small_viz_masks else None,
            input_nodes=list(settings["inlet_nodes"]),
            output_nodes=list(settings["outlet_nodes"]),
            arteriole_boundary_nodes=list(settings["arteriole_boundary_nodes"]),
            venule_boundary_nodes=list(settings["venule_boundary_nodes"]),
            voxel_size_zyx=voxel_size_zyx,
            volume_downsample_stride=int(
                settings["large_vessel_3d_volume_downsample_stride"]
            ),
            title="Final Graph with Automated Large+Small Vessel Assignment (3D)",
            save_html_path=str(final_assignment_html_path),
            show=False,
        )
        logger.info(
            "Saved final automated large-vessel assignment 3D visualization to: "
            f"{final_assignment_html_path}"
        )
        if (
            assignment_fig is not None
            and settings["final_render_mode"] == "3d"
            and hasattr(assignment_fig, "write_html")
        ):
            final_graph_3d_path = Path(settings["plot_dir"]) / "final_graph_3d.html"
            visualization.write_plotly_html(assignment_fig, final_graph_3d_path)
            logger.info(f"Saved interactive 3D final graph to: {final_graph_3d_path}")

    if settings["inlet_nodes"] and settings["outlet_nodes"]:
        resistance_node_pair = (settings["inlet_nodes"][0], settings["outlet_nodes"][0])
        logger.info(f"Auto-selected resistance node pair: {resistance_node_pair}")
    else:
        if io_from_masks:
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
        large_arteriole_boundary_nodes=settings["large_arteriole_boundary_nodes"],
        large_venule_boundary_nodes=settings["large_venule_boundary_nodes"],
        resistance_node_pair=resistance_node_pair,
        graph=G,
        # None for a role not in use: never a resumed run's masks, drawn.
        large_arteriole_mask=large_arteriole_mask,
        large_venule_mask=large_venule_mask,
        small_arteriole_mask=small_arteriole_mask,
        small_venule_mask=small_venule_mask,
    )


#: ``G.graph`` key: the raw image an ``auto`` stain recorded on the graph was
#: read off (:func:`raw_image_identity`), so a run naming another one reads
#: it again rather than taking the old image's answer.
RAW_STAIN_IMAGE = "raw_image_stain_image"


def raw_image_identity(settings: Mapping[str, Any]) -> list | None:
    """The run's raw image as a stain reading is kept for: ``[path, channel,
    size, modification time]`` (the last two None for a file not there);
    None with no raw image."""
    path = settings.get("raw_image_path")
    if path in (None, ""):
        return None
    try:
        stat = Path(path).stat()
        size, modified = int(stat.st_size), int(stat.st_mtime_ns)
    except OSError:
        size = modified = None
    try:
        where = str(Path(path).resolve())
    except OSError:
        where = str(path)
    channel = settings.get("raw_image_channel")
    try:
        channel = None if channel is None else int(channel)
    except (TypeError, ValueError):
        channel = str(channel)
    return [where, channel, size, modified]


def _recorded_stain(settings: Mapping[str, Any], G: nx.MultiGraph | None) -> str | None:
    """The ``auto`` stain recorded on *G*, when it was read off the raw image
    the settings name now (a graph recording no image -- one made before the
    image was recorded -- is taken at its word); else None."""
    if G is None or G.graph.get("raw_image_stain") not in raw_stain.STAINS:
        return None
    image = G.graph.get(RAW_STAIN_IMAGE)
    if image is not None and list(image) != raw_image_identity(settings):
        return None
    return G.graph["raw_image_stain"]


def recorded_raw_image_stain(settings: Mapping[str, Any], G: nx.MultiGraph | None = None) -> str:
    """The raw image's stain as this run knows it without reading the image:
    the setting, unless ``auto``; then what the Diameters stage resolved and
    recorded on *G* for the raw image named now; else plasma_column."""
    stain = settings.get("raw_image_stain", "auto")
    if stain in raw_stain.STAINS:
        return stain
    return _recorded_stain(settings, G) or raw_stain.PLASMA_COLUMN


def resolve_raw_image_stain(
    settings: Mapping[str, Any], network: "VesselNetwork | None"
) -> tuple[str, raw_stain.StainReading | None, np.ndarray | None]:
    """``(stain, reading, raw)``: the raw image's stain when ``raw_image_stain``
    is ``auto`` -- the answer the network build recorded on the graph
    (``_record_raw_evidence``, read by the same rule), with nothing loaded;
    else read here from the image and the segmentation (``reading`` says
    how it read, and ``raw`` is the channel loaded for it, cut to the
    network's crop box, for the measurement to use instead of loading it
    again) -- or the setting's own (``reading`` and ``raw`` None). An image
    that reads as neither, or none to read, is taken as a plasma label --
    the segmentation's own kind -- and said so."""
    stain = settings.get("raw_image_stain", "auto")
    if stain in raw_stain.STAINS:
        return stain, None, None
    recorded = _recorded_stain(settings, network.graph if network is not None else None)
    if recorded is not None:
        logger.info(f"raw_image_stain=auto: {recorded}, as the network build read the raw image.")
        return recorded, None, None
    if network is not None and network.graph.graph.get("raw_image_stain") in raw_stain.STAINS:
        logger.info(
            "raw_image_stain=auto: the network build read another raw image, or another "
            "channel; reading the one named now."
        )
    path = settings.get("raw_image_path")
    if not path or network is None or network.volume is None or network.volume.image is None:
        return raw_stain.PLASMA_COLUMN, None, None
    from haemolynx.haemodynamics.fwhm_diameter import load_single_channel_tiff_volume
    from haemolynx.io.load import _to_binary_volume_for_skeletonization

    use_memmap, memmap_directory = settings["use_memmap_loading"], settings["memmap_directory"]
    whole = load_single_channel_tiff_volume(
        io.resolve_image_path_with_optional_zip(Path(path)),
        axis_order=io.axis_order_for_run(settings),
        use_memmap=use_memmap,
        memmap_directory=memmap_directory,
        channel=settings.get("raw_image_channel"),
    )
    # The mask is the image as cropped and downsampled on import; the raw
    # file is whole.
    raw = raw_volume_like_graph(
        whole, network.graph, description="raw image",
        use_memmap=use_memmap, memmap_directory=memmap_directory,
    )
    if raw is not whole and isinstance(whole, np.memmap):
        preprocessing.release_memmap_array(whole)
    mask = _to_binary_volume_for_skeletonization(
        network.volume.image, use_memmap=use_memmap, memmap_directory=memmap_directory
    )
    reading = raw_stain.read_raw_stain(raw, mask, network.volume.voxel_size_zyx)
    if reading.stain is None:
        logger.warning(
            f"raw_image_stain=auto: the raw image {reading.describe()}; measuring it as a plasma "
            "label. Set raw_image_stain if it is an endothelial one."
        )
        return raw_stain.PLASMA_COLUMN, reading, raw
    logger.info(f"raw_image_stain=auto: the raw image {reading.describe()}.")
    return reading.stain, reading, raw


def _haemodynamics_apply_config(
    settings: dict,
    schema: Schema,
    *,
    voxel_size_zyx: tuple[float, float, float],
    resistance_node_pair: tuple[int, int] | None = None,
    stain: str | None = None,
) -> HaemodynamicsApplyConfig:
    """Baseline haemodynamics config: no focal pericyte tone, no comparison CSV.

    The run's one raw image (``raw_image_path``, ``raw_image_channel``) is
    what both measurements read, and ``measure_diameters_from_raw_image``
    turns on the one its *stain* calls for -- FWHM across a plasma label, the
    lumen inside an endothelial wall -- as the apply layer's own switches.
    *stain* is the resolved one (:func:`resolve_raw_image_stain`); unset, it
    is the setting's, and an ``auto`` not yet resolved reads as plasma_column,
    which only a measurement would act on."""
    if stain is None:
        stain = recorded_raw_image_stain(settings)
    diameters = dict(schema.section_values(settings, "Diameters and pericytes"))
    # The run's basis is what the segmented vessels are (the Input tab's
    # segmentation_source): every diameter the segmentation gave is on it,
    # and one read from the raw image carries its own.
    diameters["diameter_basis"] = run_diameter_basis(settings)
    diameters["do_pericyte_construction"] = False
    diameters["run_pericyte_resistance_comparison"] = False
    edt = dict(schema.section_values(settings, "EDT mask diameter estimate"))
    # A synthetic network has no segmentation to measure: the cross-check, on
    # by default, is off there as its prerequisites say.
    if graph.network_is_synthetic(settings):
        edt["use_edt_diameter_crosscheck"] = False
    # The mask is the run's own segmented input -- what the skeleton was made
    # from -- read only when the stage has none in memory (a resumed run).
    edt["edt_mask_path"] = segmented_input_path(settings)
    measure = measures_diameters_from_raw_image(settings)
    fwhm = dict(schema.section_values(settings, "FWHM diameter measurement"))
    fwhm.update(
        use_fwhm_edge_diameters=measure and stain != raw_stain.ENDOTHELIAL,
        fwhm_raw_tiff_path=settings.get("raw_image_path"),
        fwhm_raw_channel=settings.get("raw_image_channel"),
    )
    endothelial = dict(schema.section_values(settings, "Endothelial diameter"))
    endothelial.update(
        use_endothelial_diameters=measure and stain == raw_stain.ENDOTHELIAL,
        endothelial_image_path=settings.get("raw_image_path"),
        endothelial_channel=settings.get("raw_image_channel"),
    )
    return HaemodynamicsApplyConfig(
        diameters=diameters,
        fwhm=fwhm,
        edt=edt,
        endothelial=endothelial,
        resistance_node_pair=resistance_node_pair,
        voxel_size_zyx=tuple(float(v) for v in voxel_size_zyx),
        axis_order=io.axis_order_for_run(settings),
        voxel_size_override_xyz=settings.get("voxel_size_override_xyz"),
        voxel_size_policy=settings.get("voxel_size_policy", "auto"),
        comparison_output_csv_path=None,
        equivalent_resistance_solver=_equivalent_resistance_solver(settings),
        use_memmap=settings["use_memmap_loading"],
        memmap_directory=settings["memmap_directory"],
    )


def _tag_vessels_in_global_masks(
    settings: Mapping[str, Any],
    G: nx.MultiGraph,
    *,
    large_mask: np.ndarray | None,
    small_mask: np.ndarray | None,
    voxel_size_zyx,
    edges: list | None = None,
) -> None:
    """Tag each vessel (or only *edges*) with the global mask it lies in, at
    the overlap fractions the large and small masks' own labelling uses, for
    :func:`_assign_branch_orders` to class it. Without either mask (a
    post-processing run with no network to read them from) nothing is
    tagged, and the tags already on the graph stand."""
    if large_mask is None and small_mask is None:
        return
    counts = graph.tag_vessels_in_global_masks(
        G,
        large_mask=large_mask,
        small_mask=small_mask,
        voxel_size_zyx=voxel_size_zyx,
        large_min_overlap_fraction=float(settings["large_vessel_mask_min_overlap_fraction"]),
        small_min_overlap_fraction=float(settings["small_vessel_mask_min_overlap_fraction"]),
        edges=edges,
    )
    logger.info(
        "Global vessel masks (no venule mask entered): "
        f"{counts['large_edge_count']} vessel(s) in the large mask, "
        f"{counts['small_edge_count']} in the small mask, "
        f"{counts['capillary_edge_count']} capillar{'y' if counts['capillary_edge_count'] == 1 else 'ies'} "
        f"({counts['gap_filled_edge_count']} under-covered between two of one class took it); "
        "they are classed Unknown_Large, Unknown_Small and Unknown_Cap rather than "
        "given branch orders, and the masks chose no inlets or outlets."
    )


def _tag_edited_vessels_in_large_masks(
    settings: Mapping[str, Any],
    G: nx.MultiGraph,
    boundaries: BoundaryNodes,
    edited: list,
    voxel_size_zyx,
) -> None:
    """Tag each *edited* vessel with the large mask it lies in, as the
    Boundaries stage tagged every vessel (``graph.LARGE_MASK_VESSEL_TYPE``):
    the Large_Art/Large_Ven tier is the tagged vessels joined to an inlet or
    outlet through tagged vessels, so a vessel drawn untagged in the large
    arteriole would end the tier there. The other vessels keep their tags.
    Without the masks the edited vessels keep the tags they had."""
    if boundaries.large_arteriole_mask is None or boundaries.large_venule_mask is None:
        logger.warning(
            "Post processing: no large vessel masks to read, so %d edited vessel(s) "
            "keep the large-mask tags they had; a vessel drawn by hand in a large "
            "vessel is left out of its Large_Art/Large_Ven tier.",
            len(edited),
        )
        return
    from haemolynx.graph.cut_at_large_vessel_volumes import _as_large_vessel_foreground

    counts = graph.tag_edges_in_large_vessel_masks(
        G,
        _as_large_vessel_foreground(boundaries.large_arteriole_mask, role="arteriole"),
        _as_large_vessel_foreground(boundaries.large_venule_mask, role="venule"),
        edited,
        voxel_size_zyx=voxel_size_zyx,
        minimum_overlap_fraction=float(settings["large_vessel_mask_min_overlap_fraction"]),
        low_memory=bool(settings.get("use_memmap_loading")),
    )
    logger.info(
        "Post processing: %d of %d edited vessel(s) lie in the large arteriole mask and "
        "%d in the large venule mask (%d under-covered between two of one type took it).",
        counts["arteriole_edge_count"],
        len(edited),
        counts["venule_edge_count"],
        counts["gap_filled_edge_count"],
    )


def _small_vessel_boundary_fallback(
    settings: Mapping[str, Any],
    G: nx.MultiGraph,
    *,
    small_masks: tuple[np.ndarray, np.ndarray],
    large_masks: tuple[np.ndarray, np.ndarray] | None,
    voxel_size_zyx,
) -> dict[str, tuple[list, list]]:
    """``small_vessel_boundary_fallback_to_hop_distance``, for the arteriole
    and the venule boundary list: ``(uncovered, hop_nodes)`` -- the inlets
    (outlets) none of whose vessels lies in the small mask of their class or,
    given *large_masks* (the large vessels kept in the network), in the large
    one, and the nodes ``small_vessel_boundary_fallback_hop_distance``
    vessels from those that the fallback makes boundary nodes (none with it
    off), inlets, outlets and the other side's boundary nodes left out. The
    Boundaries stage adds them to the
    masks' boundary nodes; Post processing judges an edited place by it."""
    enabled = bool(settings["small_vessel_boundary_fallback_to_hop_distance"])
    hops = int(settings["small_vessel_boundary_fallback_hop_distance"])
    if enabled and hops < 1:
        raise ValueError(
            "small_vessel_boundary_fallback_hop_distance must be >= 1 when "
            "small_vessel_boundary_fallback_to_hop_distance=True."
        )
    exclude_io_nodes = set(settings["inlet_nodes"]) | set(settings["outlet_nodes"])
    fallback: dict[str, tuple[list, list]] = {}
    for index, (seed_key, boundary_key) in enumerate(
        (("inlet_nodes", "arteriole_boundary_nodes"), ("outlet_nodes", "venule_boundary_nodes"))
    ):
        coverage_mask = small_masks[index]
        if large_masks is not None:
            from haemolynx.preprocessing.memmap_support import MaskUnion

            # Read where the edges ask, not made as a volume.
            coverage_mask = MaskUnion(coverage_mask, large_masks[index])
        uncovered = graph.seed_nodes_without_mask_coverage(
            G, list(settings[seed_key]), coverage_mask, voxel_size_zyx=voxel_size_zyx
        )
        # Nor a node the other side's masks made its boundary: one node on
        # both lists ended both tiers at once.
        other_side = "venule_boundary_nodes" if index == 0 else "arteriole_boundary_nodes"
        hop_nodes = (
            graph.select_nodes_at_hop_distance(
                G, uncovered, hops, exclude_nodes=exclude_io_nodes | set(settings.get(other_side) or ())
            )
            if enabled and uncovered
            else []
        )
        fallback[boundary_key] = (uncovered, hop_nodes)
    return fallback


def _set_boundary_list(
    settings: dict, boundaries: BoundaryNodes, name: str, nodes: list
) -> None:
    """Make the boundary list *name* hold *nodes*, on *boundaries* and in
    *settings* both -- in place, as :func:`_trim_boundary_lists` keeps them,
    since a fresh run hands the settings' own lists over and a resumed one
    copies -- so the branch orders assigned next read it."""
    listed = getattr(boundaries, name)
    listed[:] = nodes
    in_settings = settings.get(name)
    if not isinstance(in_settings, list):
        settings[name] = list(nodes)
    elif in_settings is not listed:
        in_settings[:] = nodes


def _reassign_small_vessel_boundaries(
    settings: dict,
    G: nx.MultiGraph,
    boundaries: BoundaryNodes,
    edited: list,
    touched: list,
    voxel_size_zyx,
) -> None:
    """Add or drop arteriole/venule boundary nodes at the *touched* nodes --
    the ends of the *edited* vessels and the nodes a deletion took a vessel
    from -- as the Boundaries stage chose them from the small masks: each
    edited vessel is tagged with the small mask it lies in
    (``graph.SMALL_MASK_VESSEL_TYPE``: ``small_vessel_mask_min_overlap_fraction``,
    the dilation steps, the gap fill), judged beside the other vessels'
    tags, which stay; a touched node is then a boundary node by the tags'
    rule (:func:`graph.boundary_nodes_from_small_mask_tags`) or, with
    ``small_vessel_boundary_fallback_to_hop_distance``, when it is the
    fallback's hop distance from an inlet (outlet) the masks miss -- the
    fallback is placed on the network, so a vessel drawn from such an inlet
    gets its boundary node as the Boundaries stage would give it one. Every
    other boundary node stays as it is, and a list is never emptied: that
    would end the hierarchical branch orders. Without the masks the labelling
    read, or tags from the Boundaries stage, the lists stay as they are."""
    if boundaries.small_arteriole_mask is None or boundaries.small_venule_mask is None:
        logger.warning(
            "Post processing: no small vessel masks to read, so the arteriole and "
            "venule boundary nodes stay as the Boundaries stage left them; run again "
            "from Boundaries to place them on the edited network."
        )
        return
    edited_set = set(edited)
    if not any(
        data.get(graph.SMALL_MASK_VESSEL_TYPE)
        for u, v, key, data in G.edges(keys=True, data=True)
        if (u, v, key) not in edited_set
    ):
        logger.warning(
            "Post processing: no vessel carries the small-mask tag the Boundaries stage "
            "writes (a run from before it did), so the arteriole and venule boundary "
            "nodes stay as they were; run again from Boundaries to place them on the "
            "edited network."
        )
        return
    small_masks = (
        np.asarray(boundaries.small_arteriole_mask, dtype=bool),
        np.asarray(boundaries.small_venule_mask, dtype=bool),
    )
    counts = graph.tag_edges_in_small_vessel_masks(
        G,
        *small_masks,
        edited,
        voxel_size_zyx=voxel_size_zyx,
        minimum_overlap_fraction=float(settings["small_vessel_mask_min_overlap_fraction"]),
        max_dilation_microns=float(settings["small_vessel_mask_dilation_microns"]),
        low_memory=bool(settings.get("use_memmap_loading")),
    )
    logger.info(
        "Post processing: %d of %d edited vessel(s) lie in the small arteriole mask and "
        "%d in the small venule mask (%d under-covered between two of one type took it).",
        counts["arteriole_edge_count"],
        len(edited),
        counts["venule_edge_count"],
        counts["gap_filled_edge_count"],
    )
    fallback_nodes: dict[str, list] = {"arteriole_boundary_nodes": [], "venule_boundary_nodes": []}
    if settings["small_vessel_boundary_fallback_to_hop_distance"]:
        large_masks = None
        if (
            graph.assigns_large_vessel_branch_orders(settings)
            and boundaries.large_arteriole_mask is not None
            and boundaries.large_venule_mask is not None
        ):
            from haemolynx.graph.cut_at_large_vessel_volumes import _as_large_vessel_foreground

            large_masks = (
                _as_large_vessel_foreground(boundaries.large_arteriole_mask, role="arteriole"),
                _as_large_vessel_foreground(boundaries.large_venule_mask, role="venule"),
            )
        fallback = _small_vessel_boundary_fallback(
            settings, G, small_masks=small_masks, large_masks=large_masks,
            voxel_size_zyx=voxel_size_zyx,
        )
        fallback_nodes = {name: hop_nodes for name, (_seeds, hop_nodes) in fallback.items()}
    found = graph.boundary_nodes_from_small_mask_tags(G, touched)
    touched_set = set(touched)
    for role, name in (("arteriole", "arteriole_boundary_nodes"), ("venule", "venule_boundary_nodes")):
        current = list(getattr(boundaries, name))
        wanted = set(found[role]) | (set(fallback_nodes[name]) & touched_set)
        kept = [node for node in current if node not in touched_set or node in wanted]
        added = [node for node in touched if node in wanted and node not in current]
        if current and not kept and not added:
            logger.warning(
                "Post processing: the edits leave no %s boundary node, so the %d the "
                "network had are kept; without one the branch orders would no longer "
                "be hierarchical.",
                role,
                len(current),
            )
            continue
        if len(kept) == len(current) and not added:
            continue
        _set_boundary_list(settings, boundaries, name, kept + added)
        logger.info(
            "Post processing: %s boundary node(s) added at %s, dropped at %s.",
            role,
            added,
            [node for node in current if node not in kept],
        )


def _reassign_large_vessel_boundaries(
    settings: dict,
    G: nx.MultiGraph,
    boundaries: BoundaryNodes,
    touched: list,
) -> None:
    """Add or drop Large_Art/Large_Ven hand-off nodes
    (``large_arteriole_boundary_nodes``, ``large_venule_boundary_nodes``) at
    the *touched* nodes -- the ends of the edited vessels and the nodes a
    deletion took a vessel from -- as the Boundaries stage chose them from
    the large masks' labelling: a node of a vessel tagged as lying in a large
    mask (``graph.LARGE_MASK_VESSEL_TYPE``, the edited vessels tagged by
    :func:`_tag_edited_vessels_in_large_masks`) with a vessel not of that
    type leaving it (:func:`graph.boundary_nodes_from_mask_tags`). The tier
    itself is read from the tags, so the orders were already right; the lists
    say whether a tier is tried at all, and the checkpoint and the synthetic
    template's boundary counts read them. Every other node stays as it is,
    and a list is never emptied: that would end its Large tier while the
    vessel still lies in the mask. Without the tags (a run from before the
    Boundaries stage wrote them) the lists stay as they are."""
    if not any(data.get(graph.LARGE_MASK_VESSEL_TYPE) for *_, data in G.edges(data=True)):
        if boundaries.large_arteriole_boundary_nodes or boundaries.large_venule_boundary_nodes:
            logger.warning(
                "Post processing: no vessel carries the large-mask tag the Boundaries stage "
                "writes (a run from before it did), so the Large_Art/Large_Ven hand-off "
                "nodes stay as they were; run again from Boundaries to place them on the "
                "edited network."
            )
        return  # no large vessel in the masks: nothing to hand off from
    found = graph.boundary_nodes_from_mask_tags(
        G, touched, attribute=graph.LARGE_MASK_VESSEL_TYPE
    )
    touched_set = set(touched)
    for role, tier, name in (
        ("arteriole", "Large_Art", "large_arteriole_boundary_nodes"),
        ("venule", "Large_Ven", "large_venule_boundary_nodes"),
    ):
        current = list(getattr(boundaries, name))
        wanted = set(found[role])
        kept = [node for node in current if node not in touched_set or node in wanted]
        added = [node for node in touched if node in wanted and node not in current]
        if current and not kept and not added:
            logger.warning(
                "Post processing: the edits leave no %s hand-off node, so the %d the "
                "network had are kept; without one the %s tier would not be tried "
                "though its vessels still lie in the large %s mask.",
                tier,
                len(current),
                tier,
                role,
            )
            continue
        if len(kept) == len(current) and not added:
            continue
        _set_boundary_list(settings, boundaries, name, kept + added)
        logger.info(
            "Post processing: %s hand-off node(s) added at %s, dropped at %s.",
            tier,
            added,
            [node for node in current if node not in kept],
        )


def branch_order_steps_in_effect(settings: Mapping[str, Any]) -> str:
    """Where the run's tier orders step (``graph.BRANCH_ORDER_STEP_RULES``):
    ``branch_order_steps``, but every junction for a synthetic network or a
    surround, whose generator and template comparison read each tier order as
    one segment between side branches."""
    if graph.network_is_synthetic(settings):
        return graph.EVERY_JUNCTION
    return settings["branch_order_steps"]


def _assign_branch_orders(
    settings: dict, G: nx.MultiGraph, *, post_assign_callback=None
) -> dict[str, Any]:
    """Branch orders on *G* from the run's boundary-node lists in *settings*.

    One call for the Diameters stage and the Post processing stage, which
    assigns them again over a hand-edited network: the two must read the same
    boundary lists and the same hierarchical rules, or an edit would change
    orders the rest of the network was never given.

    With global vessel masks (``io.vessel_masks_are_global``) every vessel is
    given its unknown class instead -- Unknown_Large, Unknown_Small or
    Unknown_Cap, from the tags :func:`_tag_vessels_in_global_masks` left --
    since nothing says which side of the capillary bed a vessel is on. So is
    a network classed by the size convention (``graph.size_convention_of``:
    the tags ``graph.tag_vessels_by_size`` left, recorded on the graph, so a
    hand-edited network is labelled by the convention it was run with).
    """
    convention = graph.size_convention_of(G)
    if io.vessel_masks_are_global(settings) or convention is not None:
        branch_summary = graph.assign_unknown_vessel_classes(
            G, post_assign_callback=post_assign_callback
        )
        if convention is None:
            logger.info(
                "Global vessel masks: vessels classed by size alone, not given branch "
                f"orders: {branch_summary['counts']}."
            )
        else:
            _log_size_convention(convention)
        return branch_summary
    branch_summary = graph.assign_vessel_branch_orders(
        G,
        settings["inlet_nodes"],
        outlet_nodes=settings["outlet_nodes"],
        arteriole_boundary_nodes=settings["arteriole_boundary_nodes"],
        venule_boundary_nodes=settings["venule_boundary_nodes"],
        large_arteriole_boundary_nodes=settings["large_arteriole_boundary_nodes"],
        large_venule_boundary_nodes=settings["large_venule_boundary_nodes"],
        strict_hierarchical=settings["strict_branch_order_assignment"],
        # Hierarchical Art*/Ven* labelling needs small-vessel terminals
        # (auto masks or manual A/V coords/volumes/node IDs). The
        # Large_Art/Large_Ven tier does not: with the large masks alone it
        # runs with B orders beside it (graph.assign_large_tier_branch_orders).
        expects_hierarchical=bool(
            graph.vessel_masks_in_use(settings, "small")
            or _vessel_boundary_configured(settings, "arteriole")
            or _vessel_boundary_configured(settings, "venule")
        ),
        post_assign_callback=post_assign_callback,
        steps=branch_order_steps_in_effect(settings),
    )
    if branch_summary["mode"] == "hierarchical":
        logger.info(
            "Assigned hierarchical branch orders "
            "(Art*/Ven* first, then capillary B* from arteriole boundary)."
        )
        logger.info(f"Branch assignment summary: {branch_summary}")
    elif branch_summary["mode"] == "large_tier":
        logger.info(
            "Assigned Large_Art/Large_Ven orders along the large-vessel masks and "
            "capillary B* orders beside them (no arteriole/venule boundary nodes, so "
            f"no Art*/Ven* tier): {branch_summary['large_arteriole_edge_count']} "
            f"Large_Art and {branch_summary['large_venule_edge_count']} Large_Ven vessel(s)."
        )
    elif branch_summary["mode"] == "capillary":
        logger.info(
            "Assigned capillary branch orders from STARTING_NODES only "
            "(no arteriole/venule boundary-node sets supplied)."
        )
    if branch_summary["mode"] in ("hierarchical", "large_tier") and (
        G.graph.get(graph.BRANCH_ORDER_STEPS) == graph.TIER_BRANCHINGS
    ):
        logger.info(
            "Tier orders (Large_Art, Art, Ven, Large_Ven) step only where the tier itself "
            "branches (branch_order_steps=tier_branchings), not where a side branch leaves it."
        )
    from_venous = branch_summary.get("capillary_from_venous_side_count", 0)
    if from_venous:
        logger.info(
            f"{from_venous} capillary vessel(s) are reached from the arterial side only "
            "through a venule or the large vein, so their B orders are counted from where "
            f"they meet the venous side ({graph.BRANCH_ORDER_BASIS}={graph.VENOUS_SIDE!r})."
        )
    return branch_summary


def _record_vessel_classes(run: Any, output: Any) -> None:
    """The size convention's record of a stage *output*'s graph, for the
    run's manifest (``checks.vessel_classes_source``); nothing for a run
    without one."""
    G = getattr(output, "graph", None)
    convention = graph.size_convention_of(G) if isinstance(G, nx.Graph) else None
    if convention is not None:
        run.checks[graph.VESSEL_CLASSES_SOURCE] = convention
    else:
        run.checks.pop(graph.VESSEL_CLASSES_SOURCE, None)


def _log_size_convention(convention: Mapping[str, Any]) -> None:
    """The run log's account of a size convention's classes: each class by its
    range and count, the vessels with no width, and what the capillary
    readers now act on."""
    mid, wide = convention["mid_min_um"], convention["wide_min_um"]
    counts = convention.get("counts", {})
    unsized = convention.get("unsized", {})
    logger.info(
        "Size convention: vessels classed by their measured width, not given branch "
        "orders or sides: "
        + ", ".join(
            f"{counts.get(name, 0)} {graph.size_class_range(name, mid, wide)} ({name})"
            for name in graph.SIZE_CLASSES
        )
        + f"; widths read on {convention.get('basis_counts', {})}. Of the vessels with "
        f"no width, {unsized.get('neighbours', 0)} took the class of the vessels at "
        f"both ends and {unsized.get('no_width', 0)} were left narrow. Every step that "
        f"acts on capillaries acts on the vessels {graph.size_class_range('narrow', mid, wide)} "
        "wide."
    )


def _size_classes_then_orders(
    settings: dict,
    thresholds: tuple[float, float],
    *,
    edges=None,
    width_attribute: str = "diameter_um",
    post_assign_callback=None,
):
    """What the size convention does once the widths are measured: class the
    vessels (*edges* only, after a hand edit) by *thresholds*, then assign
    the branch orders the class medians and the table read -- the
    ``before_fallbacks`` hook ``assign_edge_diameters`` calls."""
    from haemolynx.haemodynamics.viscosity import run_diameter_basis

    def classify(G: nx.MultiGraph, waiting) -> None:
        graph.tag_vessels_by_size(
            G,
            *thresholds,
            edges=edges,
            unsized=waiting,
            width_attribute=width_attribute,
            default_basis=run_diameter_basis(settings),
        )
        _assign_branch_orders(settings, G, post_assign_callback=post_assign_callback)

    return classify


def assign_diameters(settings: dict, network: VesselNetwork, boundaries: BoundaryNodes, schema: Schema):
    """Assign branch orders, then the diameter each edge is modelled with.\n\n    Branch orders come first because they are the key into the diameter\n    table; per-edge FWHM measurements override that table when enabled."""
    if boundaries.graph is not None:
        network.graph = boundaries.graph
    G = network.graph
    voxel_size_zyx = network.volume.voxel_size_zyx
    # The raw image is read here as raw_image_downsample now says.
    _record_downsample(G, settings, network.volume)
    # Junction bridges are drawn from the widths measured below: any an
    # earlier run drew go back into their vessels, so each is measured whole.
    if graph.has_junction_bridges(G):
        graph.remove_junction_bridges(G, keep=_boundary_node_set(boundaries))
    # The size convention classes the vessels by the widths measured below,
    # and their branch orders follow (graph.vessel_classes_by_size); a network
    # an earlier run classed so, run again without it, loses those classes.
    by_size = graph.vessel_classes_by_size(settings)
    thresholds = (
        graph.check_size_thresholds(
            settings["vessel_size_mid_min_diameter_um"],
            settings["vessel_size_wide_min_diameter_um"],
        )
        if by_size
        else None
    )
    if not by_size:
        graph.clear_vessel_size_classes(G, keep_mask_class=io.vessel_masks_are_global(settings))
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

        post_assign = (
            _vessel_types_after_branch_assign if _want_midrun_plotly_html(settings) else None
        )
        if not by_size:
            _assign_branch_orders(settings, G, post_assign_callback=post_assign)

        if not settings["run_haemodynamics"]:
            if by_size:
                # Nothing is measured without haemodynamics: the classes come
                # from the mask's own widths.
                mask_widths = measure_mask_diameters(
                    G,
                    _haemodynamics_apply_config(
                        settings, schema, voxel_size_zyx=voxel_size_zyx,
                        stain=recorded_raw_image_stain(settings),
                    ),
                    mask_volume=network.volume.image,
                )
                logger.info(
                    "Haemodynamics disabled: vessels classed by size from the mask's own "
                    f"widths (EDT) alone: {mask_widths}"
                )
                _size_classes_then_orders(
                    settings, thresholds, width_attribute="edt_diameter_um",
                    post_assign_callback=post_assign,
                )(G, ())
            logger.info(
                "Haemodynamics disabled; skipping diameter fitting and "
                "Poiseuille conductance assignment."
            )
        elif settings["run_haemodynamics"]:
            # Diameters only: resistances wait for build_haemodynamic_model so
            # a GUI pause can review FWHM fits first. Baseline never places
            # focal pericyte constrictions -- that belongs to typed
            # perturbation entries.
            stain = recorded_raw_image_stain(settings)
            raw_volume = None
            if settings.get("measure_diameters_from_raw_image") and not raw_image_is_set(settings):
                logger.warning(
                    "measure_diameters_from_raw_image is on but raw_image_path is not set: "
                    "diameters are taken from the segmentation (the mask, class medians and "
                    "the table) instead."
                )
            if measures_diameters_from_raw_image(settings):
                stain, reading, raw_volume = resolve_raw_image_stain(settings, network)
                G.graph["raw_image_stain"] = stain
                # The build's own reading stands unless the image was read here.
                if reading is not None or "raw_image_stain_reading" not in G.graph:
                    G.graph["raw_image_stain_reading"] = None if reading is None else reading.as_dict()
                G.graph[RAW_STAIN_IMAGE] = raw_image_identity(settings)
            haemo_config = _haemodynamics_apply_config(
                settings,
                schema,
                voxel_size_zyx=voxel_size_zyx,
                stain=stain,
            )
            calibration_edges = edges_network_handling_keeps(settings, G, boundaries)
            # Only a measurement from the raw image calibrates on the image.
            if measures_diameters_from_raw_image(settings):
                G.graph[CALIBRATION_SAMPLE] = calibration_sample_rule(settings)
            else:
                G.graph.pop(CALIBRATION_SAMPLE, None)
            if calibration_edges is not None and len(calibration_edges) < G.number_of_edges():
                logger.info(
                    "Image calibrations (PSF, endothelial wall, FWHM decoy check) sample "
                    f"only the {len(calibration_edges)} of {G.number_of_edges()} vessels "
                    "on an inlet-to-outlet path: "
                    "boundary_handling=remove_disconnected removes the rest at the start "
                    "of the haemodynamics stage."
                )
            G, haemo_results, fwhm_raw = assign_edge_diameters(
                G, haemo_config, mask_volume=network.volume.image,
                calibration_edges=calibration_edges, raw_volume=raw_volume,
                # Without the convention the call is as it always was.
                **(
                    {"before_fallbacks": _size_classes_then_orders(
                        settings, thresholds, post_assign_callback=post_assign
                    )}
                    if by_size
                    else {}
                ),
            )
            if "fwhm" in haemo_results:
                logger.info(f"FWHM diameter measurement summary: {haemo_results['fwhm']}")
            elif not measures_diameters_from_raw_image(settings):
                logger.info(
                    "Vessel diameters: manual mode (DIAMETER_BY_BRANCH_ORDER / "
                    "table diameters without per-edge FWHM)."
                )
            if "fwhm_decoy_check" in haemo_results:
                check = haemo_results["fwhm_decoy_check"]
                if check.get("skipped"):
                    logger.info(f"FWHM decoy check skipped: {check.get('reason')}")
                else:
                    speck = check.get("speck_width_range_um")
                    logger.info(
                        f"FWHM decoy check: gave {check['decoys_measured']} of {check['decoys']} "
                        f"decoys in vessel-free tissue a width "
                        f"({check['false_positive_rate']:.0%})"
                        + (
                            f", reading {speck[0]:.1f}-{speck[1]:.1f} um there; "
                            f"{check['measured_in_speck_width_range']:.0%} of measured vessels "
                            f"read in that range (edge attribute fwhm_in_speck_width_range)"
                            if speck else ""
                        )
                    )
            if "endothelial" in haemo_results:
                wall = haemo_results["endothelial"]
                if wall.get("skipped"):
                    logger.info(f"Endothelial diameters skipped: {wall.get('reason')}")
                else:
                    logger.info(
                        f"Endothelial diameters: {wall['edges_measured']} of "
                        f"{wall['edges_measured'] + len(wall['edges_skipped'])} edges measured; "
                        f"PSF sigma (z, y, x) {wall.get('psf_sigma_zyx')} um, wall "
                        f"{wall.get('wall_um', float('nan')):.2f} um; readings rejected "
                        f"{wall.get('readings_rejected')}"
                    )
            if "raw_section" in haemo_results:
                section = haemo_results["raw_section"]
                if section.get("skipped"):
                    logger.info(f"Raw cross-section fallback skipped: {section.get('reason')}")
                else:
                    logger.info(
                        f"Raw cross-section fallback: {section['edges_measured']} of "
                        f"{section['edges_measured'] + len(section['edges_skipped'])} edges FWHM "
                        f"failed on measured; PSF sigma (z, y, x) {section.get('psf_sigma_zyx')} um; "
                        f"readings rejected {section.get('readings_rejected')}"
                    )
            if "edt" in haemo_results:
                logger.info(f"EDT mask diameter measurement summary: {haemo_results['edt']}")
            if "diameters" in haemo_results:
                # "table"/"measured" alone doesn't say *why* -- an edge FWHM
                # never even attempted and one where it outright failed both
                # land on "table" identically. fwhm_status distinguishes them.
                status_counts = Counter(
                    "not_attempted" if status is None else status.split(":", 1)[0]
                    for _u, _v, _key, edge_data in G.edges(keys=True, data=True)
                    for status in (edge_data.get("fwhm_status"),)
                )
                logger.info(
                    f"Diameter sources: {haemo_results['diameters']} "
                    f"(FWHM: {dict(status_counts)})"
                )
            if junction_bridges_in_effect(settings):
                haemo_results["junction_bridges"] = _log_junction_bridges(
                    graph.insert_junction_bridges(G)
                )

    return HaemodynamicModel(
        graph=G,
        results=locals().get("haemo_results", {}) or {},
        fwhm_raw=locals().get("fwhm_raw"),
    )


def _boundary_node_set(boundaries: BoundaryNodes) -> set[Any]:
    """Every node a boundary list or the resistance pair names."""
    nodes = {node for name in _BOUNDARY_NODE_LISTS for node in getattr(boundaries, name)}
    nodes.update(boundaries.resistance_node_pair or ())
    return nodes


def _log_junction_bridges(counts: Mapping[str, int]) -> dict[str, int]:
    """Say what :func:`graph.insert_junction_bridges` did; returns its counts."""
    left = {name: n for name, n in counts.items() if name not in ("bridges", "junctions") and n}
    logger.info(
        f"Junction bridges: {counts.get('bridges', 0)} narrow vessel end(s) at "
        f"{counts.get('junctions', 0)} junction(s) bridged to the junction, as far as the "
        "widest vessel's radius there, at zero resistance"
        + (f"; left as they were: {left}" if left else "")
        + "."
    )
    return dict(counts)


class _PostProcessSegmentation:
    """The run's segmentation as post-processing reads it: binarised once --
    a slab at a time and on disk under the low-RAM option -- and read
    through one ``MaskSupport`` at the run's bridge limits. Its three
    readers (the drawn vessels, the edited vessels' thick-vessel bridges and
    the dead ends at the edits) each binarised the whole image in RAM and
    made a support of their own. Made when first asked for.

    :meth:`of` keeps it on the network's volume, so the next Regenerate of
    the run does not binarise the image and build the support's surface tree
    again: on E14.5 that was about 5 of the 6 s post processing took."""

    def __init__(self, settings: dict, network: "VesselNetwork | None"):
        self.settings = settings
        self.image = None if network is None else network.volume.image
        self.voxel_size_zyx = (
            None if network is None else tuple(float(v) for v in network.volume.voxel_size_zyx)
        )
        self.key = self._key(settings, network)
        self._binary = None
        self._support = None

    @staticmethod
    def _key(settings: Mapping[str, Any], network: "VesselNetwork | None") -> tuple:
        """What the binarised image and its support are made from."""
        if network is None:
            return (None,)
        return (
            id(network.volume.image),
            tuple(float(v) for v in network.volume.voxel_size_zyx),
            float(settings["bridge_max_background_gap_um"]),
            float(settings["bridge_min_mask_fraction"]),
            bool(settings.get("use_memmap_loading")),
            str(settings.get("memmap_directory")),
        )

    @classmethod
    def of(cls, settings: dict, network: "VesselNetwork | None") -> "_PostProcessSegmentation":
        """The one kept on *network*'s volume while it still reads its image
        with these settings, else a new one, kept there in its place."""
        if network is None:
            return cls(settings, None)
        volume = network.volume
        kept = volume.post_process_segmentation
        if (
            isinstance(kept, cls)
            and kept.image is volume.image
            and kept.key == cls._key(settings, network)
        ):
            kept.settings = settings
            return kept
        made = cls(settings, network)
        volume.post_process_segmentation = made
        return made

    # The Post processing tab's review lists read it on a worker thread while
    # a run's post processing may: one makes it, the other waits for it.
    _making = threading.RLock()

    @property
    def binary(self) -> np.ndarray | None:
        with self._making:
            if self._binary is None and self.image is not None:
                self._binary = _to_binary_volume_for_skeletonization(
                    self.image,
                    use_memmap=bool(self.settings.get("use_memmap_loading")),
                    memmap_directory=self.settings.get("memmap_directory"),
                )
            return self._binary

    def support(self, voxel_size_zyx=None) -> "preprocessing.MaskSupport | None":
        if self.binary is None:
            return None
        with self._making:
            if self._support is None:
                self._support = preprocessing.MaskSupport(
                    self.binary,
                    self.voxel_size_zyx if voxel_size_zyx is None else tuple(float(v) for v in voxel_size_zyx),
                    max_background_gap_um=float(self.settings["bridge_max_background_gap_um"]),
                    min_mask_fraction=float(self.settings["bridge_min_mask_fraction"]),
                    low_memory=bool(self.settings.get("use_memmap_loading")),
                    memmap_directory=self.settings.get("memmap_directory"),
                )
            return self._support


def post_process_segmentation_support(
    settings: Mapping[str, Any], volume: "SkeletonisedVolume | None"
) -> "preprocessing.MaskSupport | None":
    """The ``MaskSupport`` post processing reads *volume*'s segmentation
    through (the one kept on it, made here when there is none yet), so the
    Post processing tab's review lists read the same one rather than
    binarising a second copy. None without a segmentation, or without the
    bridge settings it is held to."""
    if volume is None or getattr(volume, "image", None) is None:
        return None
    if any(name not in settings for name in ("bridge_max_background_gap_um", "bridge_min_mask_fraction")):
        return None
    holder = SimpleNamespace(volume=volume)
    return _PostProcessSegmentation.of(dict(settings), holder).support()


def _judge_drawn_vessels(
    settings: dict,
    G: nx.MultiGraph,
    network: VesselNetwork | None,
    edited: Sequence[tuple],
    voxel_size_zyx,
    segmentation: _PostProcessSegmentation | None = None,
) -> list[dict[str, Any]] | None:
    """How each vessel the edits drew sits in the segmentation, written on
    it and listed for the summary; None with nothing drawn or no
    segmentation to read (a synthetic run, or a resume without one).

    The vessels :func:`haemolynx.graph.add_vessel_between` added
    (``post_processing_added``; a split half, a merge or a bridge is not one
    a person drew) are read with the run's own bridge reader
    (:class:`haemolynx.preprocessing.MaskSupport`, at the run's
    ``bridge_max_background_gap_um`` and ``bridge_min_mask_fraction``) by
    :func:`haemolynx.graph.record_drawn_vessel_support`: ``drawn_mask_fraction``,
    ``drawn_background_run_um`` and ``drawn_beside_vessel`` go on each. A
    diagnostic only -- a vessel traced through tissue the segmentation never
    showed is visible as such, and kept as drawn.
    """
    drawn = [
        edge for edge in edited
        if G.edges[edge].get("post_processing_added") and not G.edges[edge].get(graph.IS_ZERO_RESISTANCE)
    ]
    if not drawn or network is None or network.volume.image is None:
        return None
    if segmentation is None:
        segmentation = _PostProcessSegmentation.of(settings, network)
    support = segmentation.support(voxel_size_zyx)
    records = graph.record_drawn_vessel_support(G, drawn, support)
    off_segmentation = sum(not record.within_bridge_gate for record in records)
    beside = sum(record.beside_vessel for record in records)
    logger.info(
        "Post processing: %d drawn vessel(s) read against the segmentation: %d partly "
        "off it (graph building's bridge gate would have drawn no bridge there), %d "
        "beside a vessel in the same lumen -- all kept as drawn.",
        len(records), off_segmentation, beside,
    )
    for record in records:
        u, v, _key = record.edge
        logger.info("  vessel %s-%s (%.3g um): %s", u, v, record.length_um, record.describe())
    return [record.as_record() for record in records]


def _bridge_edited_vessels(
    settings: dict,
    G: nx.MultiGraph,
    network: VesselNetwork | None,
    thick_vessel_mask: np.ndarray | None,
    segmentation: _PostProcessSegmentation | None = None,
) -> nx.MultiGraph:
    """*G* with every edited vessel that opens into a thick vessel split there.

    What graph building's last step does to every vessel
    (:func:`haemolynx.graph.insert_thick_vessel_junction_nodes`), for the
    edited ones only: the part inside the thick-vessel region becomes a
    zero-resistance bridge. Only with thickness-gated skeletonisation, the one
    thing that makes bridges. The region is the run's own when its volume
    carries it, else *thick_vessel_mask*, else found again from the
    segmentation. A vessel a person made of a bridge
    (``graph.BRIDGE_SET_BY_HAND`` False) is left whole.
    """
    if not settings.get("use_thick_vessel_skeletonisation"):
        return G
    edited = [
        edge for edge in graph.edited_edges(G)
        if G.edges[edge].get(graph.BRIDGE_SET_BY_HAND) is not False
    ]
    if not edited:
        return G
    mask = network.volume.thick_vessel_mask if network is not None else None
    if mask is None:
        mask = thick_vessel_mask
    if mask is None:
        if network is None:
            logger.warning(
                "Post processing: no thick-vessel region and no segmentation to find "
                "it in; %d edited vessel(s) get no zero-resistance bridge where they "
                "open into a thick vessel.",
                len(edited),
            )
            return G
        if network.volume.image is None:
            logger.info(
                "Post processing: a synthetic network has no segmentation, so no "
                "thick-vessel bridges are made for its edited vessels."
            )
            return G
        logger.info(
            "Post processing: finding the thick-vessel region in the segmentation "
            "again (this run loaded its skeleton rather than making it)."
        )
        mask = thick_vessel_mask_from_image(
            settings, network.volume.image, network.volume.voxel_size_xyz,
            getattr(network.volume, "crop", None),
            cache=thick_vessel_region_cache(settings, network.volume.output_dir),
            downsample=getattr(network.volume, "downsample", None),
        )
        if mask is None:
            return G  # no thick vessel anywhere: graph building made no bridges either
    if network is not None:
        voxel_size_zyx = tuple(float(v) for v in network.volume.voxel_size_zyx)
    else:
        voxel_size_zyx = tuple(float(v) for v in G.graph.get("image_voxel_size_zyx", (1, 1, 1)))

    def edited_bridges(graph_obj: nx.MultiGraph) -> int:
        return sum(
            1
            for _u, _v, data in graph_obj.edges(data=True)
            if data.get(graph.IS_ZERO_RESISTANCE) and data.get(graph.post_processing.EDITED)
        )

    if segmentation is None:
        segmentation = _PostProcessSegmentation.of(settings, network)
    before = edited_bridges(G)
    # The tree skeletonisation drew, when this is the region it drew it in.
    tree = None
    if network is not None and mask is network.volume.thick_vessel_mask:
        tree = getattr(network.volume, "thick_vessel_centreline", None)
    if tree is None:
        tree = graph.thick_vessel_junctions.thick_vessel_centreline_voxels(
            mask,
            voxel_size_zyx,
            use_memmap=bool(settings.get("use_memmap_loading")),
            large_stack_tree=large_stack_thick_tree(settings),
        )
    # What the tree fill draws beside an edit is new vessel, measured as
    # one drawn by hand is.
    bridged = graph.insert_thick_vessel_junction_nodes(
        G, mask, voxel_size_zyx=voxel_size_zyx, edges=edited, segmentation=segmentation.binary,
        centreline=tree,
        mask_support=segmentation.support(voxel_size_zyx),
        fill_attributes={graph.post_processing.EDITED: True},
        **thick_junction_bridge_arguments(settings),
    )
    added = edited_bridges(bridged) - before
    if added:
        logger.info(
            "Post processing: %d edited vessel piece(s) inside a thick vessel became "
            "zero-resistance bridges.",
            added,
        )
    return bridged


def _mark_touched_dead_end_kinds(
    settings: dict,
    G: nx.MultiGraph,
    network: VesselNetwork | None,
    touched: list,
    segmentation: _PostProcessSegmentation | None = None,
) -> dict[str, int] | None:
    """Judge the dead ends among the *touched* nodes -- the edited vessels'
    ends and the nodes a deletion took a vessel from -- against the run's
    segmentation, as :func:`_mark_dead_end_kinds` judged every dead end when
    the network was built, and write their kinds (``graph.dead_ends``,
    :func:`graph.dead_ends.reclassify_dead_ends`): one leaving the image is
    what the solve's open-end boundary condition goes on, the rest what the
    review lists show. A touched node that is a dead end no longer loses the
    classifier's kind; a person's stays. Every other node keeps its mark.
    Without a segmentation to read -- no *network*, a synthetic one, or an
    image with nothing in it -- nothing is judged, and a dead end an edit
    made is read by the distance rule alone
    (``graph.dead_ends.node_leaves_image``). Returns the census of the dead
    ends judged, or None."""
    image = network.volume.image if network is not None else None
    if image is None:
        logger.info(
            "Post processing: no segmentation to judge the dead ends at the edits by, "
            "so an end an edit left is read by its distance to the image faces alone."
        )
        return None
    if segmentation is None:
        segmentation = _PostProcessSegmentation.of(settings, network)
    binary_mask = segmentation.binary
    if not np.any(binary_mask):
        logger.info(
            "Post processing: the segmentation has no vessel in it, so the dead ends "
            "at the edits are not judged against it."
        )
        return None
    voxel_size_zyx = tuple(float(v) for v in network.volume.voxel_size_zyx)
    support = segmentation.support(voxel_size_zyx)
    extent = graph.dead_ends.graph_image_extent_um(G)
    if extent is None:
        extent = graph.dead_ends.image_extent_um(binary_mask.shape, voxel_size_zyx)
    ends = graph.dead_ends.reclassify_dead_ends(
        G,
        touched,
        extent_um=extent,
        support=support,
        stub_radius_multiple=dead_end_stub_radius_multiple(settings),
    )
    census = graph.dead_ends.dead_end_census(ends)
    logger.info(
        "Post processing: %d dead end(s) at the edits -- %s.",
        len(ends),
        ", ".join(f"{count} {kind.replace('_', ' ')}" for kind, count in census.items() if count) or "none",
    )
    return census


def _trim_boundary_lists(
    settings: dict, boundaries: BoundaryNodes, G: nx.MultiGraph
) -> None:
    """Keep only the boundary nodes still in *G*, on *boundaries* and in *settings*.

    In place: assign_boundaries hands the settings' own lists over, and both
    must say the same thing afterwards. The resistance node pair is picked
    again when either of its nodes went.
    """
    for name in _BOUNDARY_NODE_LISTS:
        kept = getattr(boundaries, name)
        kept[:] = [node_id for node_id in kept if node_id in G]
        in_settings = settings.get(name)
        if isinstance(in_settings, list) and in_settings is not kept:
            in_settings[:] = [node_id for node_id in in_settings if node_id in G]
    pair = boundaries.resistance_node_pair
    if pair is not None and any(node_id not in G for node_id in pair):
        boundaries.resistance_node_pair = (
            (boundaries.inlet_nodes[0], boundaries.outlet_nodes[0])
            if boundaries.inlet_nodes and boundaries.outlet_nodes
            else None
        )
        logger.info(f"Re-selected resistance node pair: {boundaries.resistance_node_pair}")


def _take_edited_terminal_roles(
    settings: dict, boundaries: BoundaryNodes, G: nx.MultiGraph
) -> None:
    """The inlets and outlets the Post processing tab set by hand
    (``graph.set_terminal_role``, kept on *G*), on *boundaries* and in
    *settings* in place of the run's own; the resistance node pair is picked
    again when either of its nodes lost its role."""
    edited = G.graph.get(graph.post_processing.BOUNDARY_ROLES)
    if not edited:
        return
    for role in graph.post_processing.TERMINAL_ROLES:
        if role not in edited:
            continue
        name = f"{role}_nodes"
        nodes = list(edited[role])
        before = list(getattr(boundaries, name))
        _set_boundary_list(settings, boundaries, name, nodes)
        added = [n for n in nodes if n not in before]
        removed = [n for n in before if n not in nodes]
        if added or removed:
            logger.info(f"Post processing: {role} nodes set by hand: added {added}, removed {removed}.")
    pair = boundaries.resistance_node_pair
    if pair is None or pair[0] not in boundaries.inlet_nodes or pair[1] not in boundaries.outlet_nodes:
        boundaries.resistance_node_pair = (
            (boundaries.inlet_nodes[0], boundaries.outlet_nodes[0])
            if boundaries.inlet_nodes and boundaries.outlet_nodes
            else None
        )


def _take_chosen_resistance_pair(boundaries: BoundaryNodes, G: nx.MultiGraph) -> None:
    """The two nodes the Post processing tab chose for the two-point
    resistance (``graph.set_resistance_node``, kept on *G*), on *boundaries*
    in place of the run's pair. A chosen node an edit has since taken out is
    forgotten, with a warning, and the run's pair kept."""
    key = graph.post_processing.RESISTANCE_NODE_PAIR
    chosen = G.graph.get(key)
    if chosen is None:
        return
    if not all(node in G for node in chosen):
        G.graph.pop(key)
        logger.warning(
            f"Post processing: the two-point resistance nodes chosen by hand, {tuple(chosen)}, are "
            f"no longer all in the network; it is found between {boundaries.resistance_node_pair} instead."
        )
        return
    pair = (chosen[0], chosen[1])
    if pair != boundaries.resistance_node_pair:
        logger.info(f"Post processing: two-point resistance between nodes {pair[0]} and {pair[1]}, chosen by hand.")
    boundaries.resistance_node_pair = pair


def post_process(
    settings: dict,
    model: HaemodynamicModel,
    boundaries: BoundaryNodes,
    schema: Schema,
    *,
    network: VesselNetwork | None = None,
    thick_vessel_mask: np.ndarray | None = None,
) -> HaemodynamicModel:
    """Bring a network edited by hand in line with the rest of the run.

    The first half of the Post processing tab's stage, which comes after
    Haemodynamics -- so the network is edited with its flows on screen -- and
    before Perturbations; :func:`solve_post_processed` is the second half,
    the haemodynamics solved again on the network this leaves. The tab's
    edits (:mod:`haemolynx.graph.post_processing`) mark what they touched;
    this does to those vessels what the stages before Haemodynamics did to
    every vessel:

    1. the boundary-node lists follow the graph -- a prune can drop an inlet,
       and so can the split in 2 -- and the inlets and outlets set by hand on
       the tab's Terminal nodes list replace the run's
       (:func:`_take_edited_terminal_roles`), as the
       two nodes chosen on its Two-point resistance nodes list replace the
       first inlet and the first outlet (:func:`_take_chosen_resistance_pair`);
    2. an edited vessel that opens into a thick vessel is split where it
       enters, the part inside a zero-resistance bridge
       (:func:`_bridge_edited_vessels`);
    3. each edited vessel's ``length`` is measured along its path;
    4. each edited vessel is tagged with the vessel mask it lies in, as
       Boundaries tagged every vessel -- a global mask's size class, the
       large mask the Large_Art/Large_Ven tier reads, or the small mask the
       arteriole/venule boundary nodes come from -- and the boundary nodes
       those tags give are added or dropped where the edits changed the
       network, by the Boundaries stage's own rule: the Large_Art/Large_Ven
       hand-off nodes (:func:`_reassign_large_vessel_boundaries`) and the
       arteriole/venule boundary nodes
       (:func:`_reassign_small_vessel_boundaries`); then branch orders are
       assigned again over the whole network, as Diameters does (an edit
       changes the orders downstream of it, and a piece an edit cuts off
       from the inlets loses its orders, as a piece built cut off has none);
    5. the dead ends the edits left are judged against the segmentation as
       graph building judged every dead end
       (:func:`_mark_touched_dead_end_kinds`): one leaving the image is
       what the solve's open-end boundary condition goes on, and a node an
       edit joined a vessel to is a dead end no longer;
    6. with haemodynamics on, each edited vessel is measured afresh with the
       run's own diameter methods (FWHM, the raw section, EDT, the
       endothelium -- whichever it uses), keeping the diameter the edit gave
       it as a provisional override: a measurement replaces that, the
       branch-order table does not;
    7. each vessel the edits drew is read against the segmentation as graph
       building reads every bridge it draws (:func:`_judge_drawn_vessels`:
       the fraction of it in the mask, the longest run of background it
       crosses, whether it runs beside a vessel in the same lumen), written
       on the vessel and listed -- never refused: the person drew it where
       the segmentation is missing as often as not;
    8. with ``junction_bridges`` on, the junctions the edits touched are
       judged afresh from the widths now measured
       (:func:`_junction_bridges_after_edit`).

    Then the marks are cleared, and ``model.results["post_process"]`` says
    what was done (``edited_vessels``, the ``boundary_nodes`` lists as they
    now are, the ``dead_ends`` census at the edits when judged, the
    ``diameters`` sources when measured, and ``drawn_vessels`` when there
    was a segmentation to read them against). A network nobody
    edited passes through untouched -- no ``"post_process"`` entry -- so a
    run without the panel does what it did before this stage.

    *thick_vessel_mask* is the thick-vessel region, for a run whose
    *network* volume does not carry one (see :class:`PipelineResume`).
    """
    G = model.graph
    if not graph.has_pending_edits(G):
        logger.info("Post processing: no edits to bring in line.")
        return model
    if network is not None:
        # The edited vessels are measured from the raw image read as
        # raw_image_downsample now says.
        _record_downsample(G, settings, network.volume)
    _take_edited_terminal_roles(settings, boundaries, G)
    had_inlets, had_outlets = bool(boundaries.inlet_nodes), bool(boundaries.outlet_nodes)
    # One binarised segmentation and one support for the three readers below.
    segmentation = _PostProcessSegmentation.of(settings, network)
    # The lists are trimmed after the split: it takes a short blind vessel
    # beyond its bridge, and the node it hung from, out with it.
    G = _bridge_edited_vessels(settings, G, network, thick_vessel_mask, segmentation)
    _trim_boundary_lists(settings, boundaries, G)
    if (had_inlets and not boundaries.inlet_nodes) or (had_outlets and not boundaries.outlet_nodes):
        raise ValueError(
            "The post-processing edits removed every inlet or every outlet; the "
            "network has nothing left to solve between."
        )
    _take_chosen_resistance_pair(boundaries, G)
    model.graph = G
    boundaries.graph = G
    if network is not None:
        network.graph = G
    edited = graph.edited_edges(G)
    for u, v, key in edited:
        data = G.edges[u, v, key]
        points = data.get("voxels")
        if points is not None and len(points) >= 2:
            data["length"] = float(graph.calculate_path_length(points))
    summary: dict[str, Any] = {"edited_vessels": len(edited)}
    voxel_size_zyx = (
        network.volume.voxel_size_zyx
        if network is not None
        else tuple(G.graph.get("image_voxel_size_zyx", (1.0, 1.0, 1.0)))
    )
    drawn = _judge_drawn_vessels(settings, G, network, edited, voxel_size_zyx, segmentation)
    if drawn is not None:
        summary["drawn_vessels"] = drawn
    if edited and io.vessel_masks_are_global(settings):
        # An edited vessel's class comes from the masks, as every other's did.
        in_use = {role: graph.vessel_masks_in_use(settings, role) for role in ("large", "small")}
        _tag_vessels_in_global_masks(
            settings,
            G,
            large_mask=io.as_global_vessel_mask(
                boundaries.large_arteriole_mask, boundaries.large_venule_mask,
                use_memmap=bool(settings.get("use_memmap_loading")),
                memmap_directory=settings.get("memmap_directory"),
            ) if in_use["large"] else None,
            small_mask=io.as_global_vessel_mask(
                boundaries.small_arteriole_mask, boundaries.small_venule_mask,
                use_memmap=bool(settings.get("use_memmap_loading")),
                memmap_directory=settings.get("memmap_directory"),
            ) if in_use["small"] else None,
            voxel_size_zyx=voxel_size_zyx,
            edges=edited,
        )
    if edited and graph.assigns_large_vessel_branch_orders(settings):
        _tag_edited_vessels_in_large_masks(settings, G, boundaries, edited, voxel_size_zyx)
    touched = graph.touched_nodes(G)
    if touched and graph.assigns_large_vessel_branch_orders(settings):
        _reassign_large_vessel_boundaries(settings, G, boundaries, touched)
    if (
        touched
        and graph.vessel_masks_in_use(settings, "small")
        and not io.vessel_masks_are_global(settings)
    ):
        _reassign_small_vessel_boundaries(settings, G, boundaries, edited, touched, voxel_size_zyx)
    # What the lists now hold, for the panel's checkpoints to hand a later
    # regenerate: they keep the Boundaries stage's own otherwise. The re-solve
    # (solve_post_processed) records them again after its network handling.
    _record_post_processed_boundaries(summary, boundaries)
    if touched:
        census = _mark_touched_dead_end_kinds(settings, G, network, touched, segmentation)
        if census is not None:
            summary["dead_ends"] = census
    convention = graph.size_convention_of(G)
    if convention is not None:
        _warn_if_size_settings_differ(settings, convention)
    G, measured = _orders_and_diameters_after_edit(
        settings, schema, G, edited, network=network, voxel_size_zyx=voxel_size_zyx
    )
    summary.update(measured)
    bridged = _junction_bridges_after_edit(settings, G, graph.touched_nodes(G), _boundary_node_set(boundaries))
    if bridged is not None:
        summary["junction_bridges"] = bridged
    diameters = measured.get("diameters")
    model.results["post_process"] = summary
    graph.clear_edit_marks(G)
    G.graph[graph.post_processing.APPLIED] = True
    logger.info(
        f"Post processing: brought {len(edited)} edited vessel(s) in line with the "
        f"network ({G.number_of_edges()} vessels)"
        + (f"; diameter sources: {diameters}" if diameters else "")
        + "."
    )
    return model


def _junction_bridges_after_edit(
    settings: Mapping[str, Any], G: nx.MultiGraph, nodes: Sequence[Any], keep: set[Any]
) -> dict[str, int] | None:
    """The junctions at *nodes* judged afresh for junction bridges
    (:func:`graph.rebridge_junctions`) once an edit or the network handling
    changed them and their vessels were measured; with ``junction_bridges``
    no longer in effect, the bridges there only put back. Returns the counts
    when bridges were drawn, else None. A bridge ending at a node of *keep*
    (a boundary node) stays."""
    if not nodes:
        return None
    if junction_bridges_in_effect(settings):
        return _log_junction_bridges(graph.rebridge_junctions(G, nodes, keep=keep))
    if graph.has_junction_bridges(G):
        graph.remove_junction_bridges(G, nodes=nodes, keep=keep)
    return None


def _orders_and_diameters_after_edit(
    settings: dict,
    schema: Schema,
    G: nx.MultiGraph,
    edited: list,
    *,
    network: VesselNetwork | None,
    voxel_size_zyx,
) -> tuple[nx.MultiGraph, dict[str, Any]]:
    """Branch orders over the whole network again, then the *edited* vessels
    measured afresh with the run's own diameter methods: what
    :func:`post_process` does once a hand edit is in, and the network
    handling once it joins the two vessels a removed branch left meeting.

    Returns the graph and what was measured (``diameters`` and
    ``measurement`` with haemodynamics on, ``vessel_classes`` for a network
    classed by the size convention). With no inlets nothing is done.
    """
    summary: dict[str, Any] = {}
    # A network classed by the size convention re-classes its edited vessels
    # by the thresholds it was run with, once they are measured.
    convention = graph.size_convention_of(G)
    if convention is not None:
        classify = _size_classes_then_orders(
            settings, (convention["mid_min_um"], convention["wide_min_um"]), edges=edited
        )
    if not settings.get("inlet_nodes"):
        return G, summary
    if convention is None:
        _assign_branch_orders(settings, G)

    def haemo_config():
        return _haemodynamics_apply_config(
            settings, schema, voxel_size_zyx=voxel_size_zyx,
            stain=recorded_raw_image_stain(settings, G),
        )

    mask_volume = (
        network.volume.image if network is not None and network.volume is not None else None
    )
    if isinstance(mask_volume, ImageNotRead):
        mask_volume = None  # nothing measured here reads it (resumed_image_readers)
    # Even with no vessel to measure (only deletions): a table or class
    # median diameter follows the branch order just assigned.
    if settings.get("run_haemodynamics"):
        # What an edited vessel brought with it from the vessel it was
        # cut from or merged with is measured afresh -- but a width a network
        # was drawn with (synthetic, or the template run's inside a surround)
        # has no image of this run to be measured from, and is kept.
        edited = [edge for edge in edited if G.edges[edge].get("diameter_source") not in _DRAWN_SOURCES]
        for u, v, key in edited:
            start_measurement_afresh(G.edges[u, v, key])
        G, results, _raw = assign_edge_diameters(
            G,
            haemo_config(),
            mask_volume=mask_volume,
            edges=edited,
            **({} if convention is None else {"before_fallbacks": classify}),
        )
        summary["diameters"] = results.get("diameters")
        summary["measurement"] = results
    elif convention is not None:
        for u, v, key in edited:
            G.edges[u, v, key].pop("edt_diameter_um", None)
        measure_mask_diameters(G, haemo_config(), mask_volume=mask_volume, edges=edited)
        _size_classes_then_orders(
            settings, (convention["mid_min_um"], convention["wide_min_um"]),
            edges=edited, width_attribute="edt_diameter_um",
        )(G, ())
    if convention is not None:
        summary["vessel_classes"] = graph.size_convention_of(G)
    return G, summary


def _warn_if_size_settings_differ(settings: dict, convention: Mapping[str, Any]) -> None:
    """Warn when the settings no longer state the size convention a network
    was classed by: post processing re-classes its edits by the recorded one,
    and new thresholds take effect only by running again from Diameters."""
    recorded = (float(convention["mid_min_um"]), float(convention["wide_min_um"]))
    stated = (
        settings.get("vessel_size_mid_min_diameter_um"),
        settings.get("vessel_size_wide_min_diameter_um"),
    )
    if graph.vessel_classes_by_size(settings) and all(
        value is not None and float(value) == want for value, want in zip(stated, recorded)
    ):
        return
    now = (
        f"mid {stated[0]} um, wide {stated[1]} um"
        if graph.vessel_classes_by_size(settings)
        else "classify_vessels_by_size off"
    )
    logger.warning(
        f"Post processing: this network was classed by size at mid {recorded[0]:g} um and "
        f"wide {recorded[1]:g} um, and its edited vessels are classed by those; the "
        f"settings now say {now}. Run again from Diameters to class it by the settings."
    )


def solve_post_processed(
    settings: dict,
    model: HaemodynamicModel,
    boundaries: BoundaryNodes,
    schema: Schema,
    *,
    network: VesselNetwork | None = None,
) -> Solution:
    """The haemodynamics again, on a network :func:`post_process` brought in line.

    Post processing comes after the solve, so an edited network still carries
    the resistances and flows of the network before the edit. This runs what
    the Haemodynamics tab runs -- :func:`apply_network_handling`,
    :func:`build_haemodynamic_model` and :func:`solve` -- on the edited one,
    so perturbations and the export start from the network as edited. Every
    vessel's resistance is written again, and every flow solved again.

    The returned solution's ``post_processed`` is what was brought in line,
    its ``boundary_nodes`` the lists as this solve left them: with
    ``boundary_handling="remove_disconnected"`` the network handling drops
    the boundary nodes of the pieces it removes, and a later resume must not
    be handed them.
    """
    model = apply_network_handling(settings, model, boundaries, network, schema=schema)
    model = build_haemodynamic_model(settings, model, schema)
    solution = solve(settings, model, boundaries, schema)
    summary = model.results.get("post_process")
    if isinstance(summary, dict):
        _record_post_processed_boundaries(summary, boundaries)
    solution.post_processed = summary
    return solution


def _record_post_processed_boundaries(summary: dict, boundaries: BoundaryNodes) -> None:
    """The boundary-node lists as *boundaries* now holds them, in *summary*."""
    summary["boundary_nodes"] = {
        name: list(getattr(boundaries, name)) for name in _BOUNDARY_NODE_LISTS
    }


#: ``boundary_handling``: what happens to a vessel no inlet-to-outlet path
#: runs along (a branch or tree without both an inlet and an outlet, or a
#: dead end that reaches no outlet) -- kept and marked unsolved, or removed.
BOUNDARY_HANDLING_LEAVE_UNSOLVED = "leave_unsolved"
BOUNDARY_HANDLING_REMOVE_DISCONNECTED = "remove_disconnected"

#: The boundary-node lists a network-handling step keeps in step with the graph.
_BOUNDARY_NODE_LISTS = (
    "inlet_nodes",
    "outlet_nodes",
    "arteriole_boundary_nodes",
    "venule_boundary_nodes",
    "large_arteriole_boundary_nodes",
    "large_venule_boundary_nodes",
)


def apply_network_handling(
    settings: dict,
    model: HaemodynamicModel,
    boundaries: BoundaryNodes,
    network: VesselNetwork | None = None,
    *,
    schema: Schema | None = None,
) -> HaemodynamicModel:
    """What the "Network handling" settings do to the assigned network.

    Runs at the start of the haemodynamics stage, before the model is built,
    so a change on the Haemodynamics tab takes effect when the run restarts
    there. ``boundary_handling="remove_disconnected"`` drops every vessel no
    inlet-to-outlet path runs along
    (:func:`haemolynx.graph.remove_vessels_off_inlet_outlet_paths`): each
    graph component without both an inlet and an outlet, and the dead-end
    branches, trees and loops hanging off one that has both. A vessel
    leaving the image that ``open_end_model`` may open counts as both
    (:func:`_open_ends_network_handling_keeps`), as the solve's flow marks
    count it, so the vessels leading to one stay. The
    boundary-node lists (on *boundaries* and in *settings*) and the
    resistance node pair follow the pruned graph, which becomes *model*'s,
    *boundaries*' and *network*'s. Where a removed branch hung, the two
    vessels left are joined into one (:func:`_merge_where_branches_hung`),
    then measured afresh and the branch orders assigned again, as post
    processing does after a hand edit (*schema*, default the pipeline's,
    gives the diameter settings). ``"leave_unsolved"`` (the default) keeps
    them all, and :func:`solve` marks exactly those vessels unsolved
    (:data:`haemolynx.graph.FLOW_SOLVED`).

    Then, with ``haematocrit_junction_rule`` at ``split_junctions`` (and the
    distributed haematocrit model it belongs to), every junction of four or
    more vessels is split into bifurcations
    (:func:`haemolynx.graph.split_high_degree_junctions`), on a copy that
    likewise becomes all three's graph. Its connectors are
    ``haematocrit_split_connector_length_um`` long, or with that at 0 as long
    as each junction's vessels are wide.
    """
    _remove_vessels_off_inlet_outlet_paths(settings, model, boundaries, network, schema)
    _split_junctions_for_haematocrit(settings, model, boundaries, network)
    return model


def _splits_junctions(settings: dict) -> bool:
    return bool(
        settings.get("run_haemodynamics")
        and _distributes_haematocrit(settings)
        and settings.get("haematocrit_junction_rule") == JUNCTION_RULE_SPLIT_JUNCTIONS
    )


def _split_junctions_for_haematocrit(
    settings: dict,
    model: HaemodynamicModel,
    boundaries: BoundaryNodes,
    network: VesselNetwork | None,
) -> None:
    if not _splits_junctions(settings) or not graph.high_degree_junctions(model.graph):
        return
    # A copy: the graph handed in may be a stage checkpoint, which a rerun
    # with another junction rule must find unsplit.
    G_split = model.graph.copy()
    # 0 (the default): as long as the junction's vessels are wide.
    length = float(settings.get("haematocrit_split_connector_length_um") or 0.0)
    split = graph.split_high_degree_junctions(
        G_split, connector_length_um=length if length > 0 else None
    )
    model.graph = G_split
    boundaries.graph = G_split
    if network is not None:
        network.graph = G_split
    logger.info(
        "haematocrit_junction_rule=split_junctions: split %d junction(s) of four "
        "or more vessels into bifurcations, adding %d connector vessel(s).",
        len(split),
        sum(len(new_nodes) for new_nodes in split.values()),
    )


#: Graph record: the rule the Diameters stage's image calibrations chose
#: their vessels by (:func:`calibration_sample_rule`), so a run from the
#: Haemodynamics tab can tell when the settings now choose others.
CALIBRATION_SAMPLE = "calibration_sample"


def calibration_sample_rule(settings: Mapping[str, Any]) -> list[str]:
    """What decides :func:`edges_network_handling_keeps`: ``boundary_handling``,
    and under ``remove_disconnected`` the open-end model it counts the open
    ends by (closed without haemodynamics)."""
    handling = settings.get("boundary_handling") or BOUNDARY_HANDLING_LEAVE_UNSOLVED
    if handling != BOUNDARY_HANDLING_REMOVE_DISCONNECTED:
        return [handling]
    model = settings.get("open_end_model") if settings.get("run_haemodynamics") else None
    return [handling, model or haemodynamics.open_ends.CLOSED]


def edges_network_handling_keeps(
    settings: dict, G: nx.MultiGraph, boundaries: BoundaryNodes
) -> list[tuple[Any, Any, Any]] | None:
    """The vessels ``(u, v, key)`` of *G* that :func:`apply_network_handling`
    will keep; ``None`` -- every vessel -- with ``leave_unsolved``.

    With ``boundary_handling="remove_disconnected"``, those some
    inlet-to-outlet path runs along (or a path to a vessel leaving the image
    that the solve may open), by the rule the removal itself applies. The diameters stage, which runs first, samples its image
    calibrations from these, so a vessel about to be removed cannot shape
    the widths of the ones kept; the removal stays in the haemodynamics
    stage, so a rerun from that tab still applies it.
    """
    if settings.get("boundary_handling") != BOUNDARY_HANDLING_REMOVE_DISCONNECTED:
        return None
    opened = _open_ends_network_handling_keeps(settings, G, boundaries)
    kept = graph.inlet_to_outlet_vessels(
        G, [*boundaries.inlet_nodes, *opened], [*boundaries.outlet_nodes, *opened]
    )
    return [edge for edge in G.edges(keys=True) if edge in kept]


def _open_ends_network_handling_keeps(
    settings: Mapping[str, Any], G: nx.MultiGraph, boundaries: BoundaryNodes
) -> list[Any]:
    """The vessels leaving the image the solve may open
    (:func:`haemolynx.haemodynamics.open_ends.open_ends_in_effect`): the
    removal counts each as an inlet and an outlet, as the solve's flow marks
    do, so the vessels leading to one stay. None without haemodynamics."""
    model = settings.get("open_end_model") if settings.get("run_haemodynamics") else None
    return haemodynamics.open_ends.open_ends_in_effect(
        model, G, boundaries.inlet_nodes, boundaries.outlet_nodes
    )


def _remove_vessels_off_inlet_outlet_paths(
    settings: dict,
    model: HaemodynamicModel,
    boundaries: BoundaryNodes,
    network: VesselNetwork | None,
    schema: Schema | None = None,
) -> None:
    if settings.get("boundary_handling") != BOUNDARY_HANDLING_REMOVE_DISCONNECTED:
        return
    G_before = model.graph
    opened = _open_ends_network_handling_keeps(settings, model.graph, boundaries)
    G_pruned, io_prune_stats = graph.remove_vessels_off_inlet_outlet_paths(
        model.graph,
        [*boundaries.inlet_nodes, *opened],
        [*boundaries.outlet_nodes, *opened],
    )
    if G_pruned.number_of_nodes() == model.graph.number_of_nodes() and (
        G_pruned.number_of_edges() == model.graph.number_of_edges()
    ):
        return
    model.graph = G_pruned
    boundaries.graph = G_pruned
    if network is not None:
        network.graph = G_pruned
    for name in _BOUNDARY_NODE_LISTS:
        # In place: assign_boundaries hands the settings' own lists over, and
        # both must say the same thing afterwards.
        kept = getattr(boundaries, name)
        kept[:] = [node_id for node_id in kept if node_id in G_pruned]
        in_settings = settings.get(name)
        if isinstance(in_settings, list) and in_settings is not kept:
            in_settings[:] = [node_id for node_id in in_settings if node_id in G_pruned]
    logger.info(
        "boundary_handling=remove_disconnected: removed "
        f"{int(io_prune_stats['removed_components'])} disconnected component(s) "
        "without both an inlet and an outlet "
        f"({int(io_prune_stats['removed_component_vessels'])} vessel(s), "
        f"{int(io_prune_stats['removed_component_nodes'])} node(s), "
        f"{io_prune_stats['removed_component_length_um']:.1f} um), "
        f"{int(io_prune_stats['removed_dead_end_vessels'])} dead-end vessel(s) "
        "that reach no outlet "
        f"({int(io_prune_stats['removed_dead_end_nodes'])} node(s), "
        f"{io_prune_stats['removed_dead_end_length_um']:.1f} um) and "
        f"{int(io_prune_stats['removed_loop_vessels'])} vessel(s) on loops "
        "hanging off one node "
        f"({int(io_prune_stats['removed_loop_nodes'])} node(s), "
        f"{io_prune_stats['removed_loop_length_um']:.1f} um); "
        f"{int(io_prune_stats['remaining_vessels'])} vessel(s) and "
        f"{int(io_prune_stats['remaining_nodes'])} node(s) remain."
        + (
            f" The vessels leading to the {len(opened)} vessel(s) leaving the image that "
            f"open_end_model={settings.get('open_end_model')} may open were kept: each "
            "counts as an inlet and an outlet."
            if opened else ""
        )
    )
    if not boundaries.inlet_nodes or not boundaries.outlet_nodes:
        raise ValueError(
            "After removing disconnected components without both inlet and "
            "outlet nodes, no valid boundary nodes remained."
        )
    pair = boundaries.resistance_node_pair
    if pair is None or any(node_id not in G_pruned for node_id in pair):
        boundaries.resistance_node_pair = (boundaries.inlet_nodes[0], boundaries.outlet_nodes[0])
        logger.info(f"Re-selected resistance node pair: {boundaries.resistance_node_pair}")
    _merge_where_branches_hung(settings, model, boundaries, network, schema, G_before)


#: The vessel-mask tags the Boundaries stage wrote, which the network
#: handling does not read again: two vessels tagged apart -- one leaving a
#: mask where a removed branch hung -- stay two. So do a surround's image
#: vessel and its synthetic shell (``synthetic.surround.NETWORK_PART``): one
#: vessel would carry one part's label, and the image-flows CSV miscount it.
_MASK_TAGS_KEPT_APART = (
    graph.LARGE_MASK_VESSEL_TYPE,
    graph.SMALL_MASK_VESSEL_TYPE,
    graph.GLOBAL_MASK_CLASS,
    "network_part",
)


#: Widths a network was drawn with rather than measured from this run's
#: image: kept through a join, never measured again.
_DRAWN_SOURCES = frozenset({DIAMETER_SOURCE_SYNTHETIC, DIAMETER_SOURCE_IMAGE_RUN})


def _merge_keeping_drawn_widths(G: nx.MultiGraph, node: Any):
    """:func:`graph.merge_pass_through` at *node*, two vessels whose mask tags
    differ kept apart. Two vessels with drawn widths (:data:`_DRAWN_SOURCES`)
    are joined only when their source and basis agree, and the joined vessel
    takes the width that keeps their resistance in series:
    ``(sum L / sum(L / d^4)) ** (1/4)``."""
    parts = [data for *_edge, data in G.edges(node, data=True)]
    drawn = [data.get("diameter_source") in _DRAWN_SOURCES for data in parts]
    if any(drawn):
        if not all(drawn) or len({(d.get("diameter_source"), d.get(EDGE_DIAMETER_BASIS)) for d in parts}) > 1:
            return None
        lengths = [float(d.get("length") or 0.0) for d in parts]
        widths = [float(d.get("diameter_um") or 0.0) for d in parts]
    edge = graph.merge_pass_through(G, node, same=_MASK_TAGS_KEPT_APART)
    if edge is not None and any(drawn) and min(widths) > 0 and sum(lengths) > 0:
        G.edges[edge]["diameter_um"] = (
            sum(lengths) / sum(length / width**4 for length, width in zip(lengths, widths))
        ) ** 0.25
    return edge


def _merge_where_branches_hung(
    settings: dict,
    model: HaemodynamicModel,
    boundaries: BoundaryNodes,
    network: VesselNetwork | None,
    schema: Schema | None,
    G_before: nx.MultiGraph,
) -> None:
    """Join the two vessels the removal left at each node a branch hung
    from into one (:func:`graph.merge_pass_through`), as graph building
    leaves no node where only two vessels meet; then assign the branch orders
    again and measure the joined vessels afresh
    (:func:`_orders_and_diameters_after_edit`), as post processing does after
    a deletion merges two. Boundary nodes and the resistance pair stay, as do
    a thick-vessel bridge and the vessel it opens, and two vessels whose mask
    tags differ (:data:`_MASK_TAGS_KEPT_APART`). A junction a removed branch
    hung from may be one no longer: its junction bridges go back into their
    vessels first, and it is judged afresh once the joined vessels are
    measured (:func:`_junction_bridges_after_edit`)."""
    G = model.graph
    keep = _boundary_node_set(boundaries)
    regathered = (
        graph.remove_junction_bridges(
            G, nodes=[n for n in G if G.degree(n) != G_before.degree(n)], keep=keep
        )
        if graph.has_junction_bridges(G)
        else []
    )
    left = [n for n in G if G.degree(n) == 2 and G_before.degree(n) > 2 and n not in keep]
    joined = [_merge_keeping_drawn_widths(G, node) for node in left]
    merged = [edge for edge in joined if edge is not None and G.has_edge(*edge)]
    if not merged:
        _junction_bridges_after_edit(settings, G, regathered, keep)
        return
    if schema is None:
        from haemolynx.pipeline.schema import default_schema

        schema = default_schema()
    voxel_size_zyx = (
        network.volume.voxel_size_zyx
        if network is not None and network.volume is not None
        else tuple(G.graph.get("image_voxel_size_zyx", (1.0, 1.0, 1.0)))
    )
    G, measured = _orders_and_diameters_after_edit(
        settings, schema, G, merged, network=network, voxel_size_zyx=voxel_size_zyx
    )
    _junction_bridges_after_edit(settings, G, [*regathered, *(n for edge in merged for n in edge[:2])], keep)
    model.graph = G
    boundaries.graph = G
    if network is not None:
        network.graph = G
    logger.info(
        "Network handling: joined the two vessels left at "
        f"{sum(edge is not None for edge in joined)} node(s) a removed branch hung "
        f"from into one; branch orders assigned again and {len(merged)} joined "
        "vessel(s) measured afresh"
        + (f" (diameter sources: {measured['diameters']})" if measured.get("diameters") else "")
        + "."
    )


def build_haemodynamic_model(
    settings: dict,
    model: HaemodynamicModel,
    schema: Schema | None = None,
):
    """Write Poiseuille resistance from diameters already on the graph.

    A haematocrit an earlier solve distributed to each vessel (a solved
    network post processed, edited or loaded) is dropped first: a fixed
    model would compute its resistances from it, and a distributed one would
    start from it rather than from the one value a fresh run starts from."""
    if not settings["run_haemodynamics"]:
        return model
    for *_edge, data in model.graph.edges(data=True):
        data.pop(DISCHARGE_HAEMATOCRIT_ATTR, None)
    if schema is None:
        from haemolynx.pipeline.schema import default_schema

        schema = default_schema()
    voxel_size_zyx = tuple(
        float(v)
        for v in model.graph.graph.get("image_voxel_size_zyx", (1.0, 1.0, 1.0))
    )
    haemo_config = _haemodynamics_apply_config(
        settings,
        schema,
        voxel_size_zyx=voxel_size_zyx,
        stain=recorded_raw_image_stain(settings, model.graph),
    )
    _graph, resistance_results = apply_poiseuille_resistances(model.graph, haemo_config)
    model.results.update(resistance_results)
    for step_name, step_result in resistance_results.get("resistances", {}).items():
        logger.info(f"Haemodynamics resistances [{step_name}]: {step_result}")
    if "viscosity" in resistance_results:
        logger.info(f"Viscosity law: {resistance_results['viscosity']}")
    edges_with_resistance = sum(
        1 for _, _, data in model.graph.edges(data=True) if "resistance" in data
    )
    logger.info(
        f"Haemodynamic model: {edges_with_resistance} of "
        f"{model.graph.number_of_edges()} edges carry a resistance"
    )
    return model


def _distributes_haematocrit(settings: dict) -> bool:
    """Whether ``haematocrit_model`` picked the iterative bifurcation model
    rather than the uniform fixed value."""
    return settings.get("haematocrit_model") == "distributed_iterative"


def _flow_solver(settings: Mapping[str, Any]) -> str:
    """``haemodynamics_solver``, sparse (its default) when a hand-built settings
    dict has none."""
    return settings.get("haemodynamics_solver") or haemodynamics.FLOW_SOLVER_SPARSE


def _equivalent_resistance_solver(settings: Mapping[str, Any]) -> str:
    """``equivalent_resistance_solver``, sparse (its default) when unset."""
    return (
        settings.get("equivalent_resistance_solver")
        or haemodynamics.EQUIVALENT_RESISTANCE_SPARSE
    )


def _warn_about_dense_solver_memory(settings: Mapping[str, Any], G: nx.Graph) -> list[str]:
    """Warn when the run's dense solvers would not fit in the free memory on
    *G* (:func:`haemodynamics.warn_about_dense_solver_memory`): from
    ``build_network`` as soon as the node count is known, so a run can be
    stopped before its diameters are measured, and from ``solve`` just before
    the matrices are made."""
    return haemodynamics.warn_about_dense_solver_memory(G.number_of_nodes(), **solvers_run(settings))


def solvers_run(settings: Mapping[str, Any]) -> dict[str, str | None]:
    """The run's flow and two-point resistance solvers, as
    :func:`haemodynamics.warn_about_dense_solver_memory` takes them: ``None``
    for the two-point resistance when the run does not find it. The stages'
    memory warnings and preflight's (a synthetic network's) read these."""
    return {
        "flow_solver": _flow_solver(settings),
        "equivalent_resistance_solver": (
            _equivalent_resistance_solver(settings)
            if settings.get("do_equiv_resistance_calculation")
            else None
        ),
    }


def resistance_statistics(solution: Solution) -> dict[str, Any]:
    """The solve's two resistances, as the statistics CSV's Haemodynamics
    section: the two-point one between the first inlet and the first outlet
    (or the two nodes the Post processing tab chose), and the whole
    network's (pressure drop over total inflow through the designated
    inlets). Beside them, when open ends leaving the image were
    given a condition (``open_end_model``), the flow entering the network
    through them and leaving through them. Empty when the solve produced
    none of these."""
    rows: dict[str, Any] = {}
    equivalent = getattr(solution, "equivalent_resistance", None)
    if equivalent is not None:
        rows["Two-Point Equivalent Resistance (Pa.s/m^3)"] = float(equivalent)
        nodes = getattr(solution, "equivalent_resistance_nodes", None)
        if nodes is not None:
            rows["Two-Point Resistance Nodes"] = f"{nodes[0]} -> {nodes[1]}"
    network = getattr(solution, "network_resistance", None)
    if network is not None:
        rows["Network Resistance, Pressure Drop / Total Inflow (Pa.s/m^3)"] = float(network)
    open_ends = (getattr(solution, "statistics", None) or {}).get("open_ends")
    if open_ends:
        rows["Open-End Model"] = str(open_ends.get("model"))
        rows["Open Ends Given a Condition"] = int(open_ends.get("open_ends", 0))
        rows["Inflow Through Open Ends (m^3/s)"] = float(open_ends.get("inflow_m3_s", 0.0))
        rows["Outflow Through Open Ends (m^3/s)"] = float(open_ends.get("outflow_m3_s", 0.0))
    return {"Haemodynamics": rows} if rows else {}


def _junction_rule(settings: dict) -> str:
    """``haematocrit_junction_rule``, for a caller whose settings predate it."""
    return settings.get("haematocrit_junction_rule") or JUNCTION_RULE_NO_SEPARATION


def solve(
    settings: dict,
    model: HaemodynamicModel,
    boundaries: BoundaryNodes,
    schema: Schema | None = None,
    progress: StageProgress | None = None,
):
    """Solve the network: equivalent resistance, then pressures and edge flows.

    When ``haematocrit_model`` is ``distributed_iterative``, the flow solve
    and the Pries-Secomb bifurcation haematocrit distribution
    (:mod:`haemolynx.haemodynamics.haematocrit_distribution`) iterate
    together to a fixed point instead of solving once: haematocrit sets
    viscosity, viscosity sets resistance, resistance sets the flow split,
    and the flow split is what the distribution itself needs.

    *progress* (the stage's reporter) hears each haematocrit pass, and the
    flow solve before the two-point resistance: a watching panel shows them,
    and a run asked to stop stops there rather than at the stage's end.
    """
    G = model.graph
    resistance_node_pair = boundaries.resistance_node_pair
    solution = Solution()
    if not settings["run_haemodynamics"]:
        _clear_earlier_solve(G)
    distribute_hct = bool(
        settings["run_haemodynamics"] and _distributes_haematocrit(settings)
    )
    hct_result: dict[str, Any] | None = None
    solve_again = False
    flow_solver = _flow_solver(settings)
    if settings["run_haemodynamics"]:
        _warn_about_dense_solver_memory(settings, G)
        # Which haematocrit the flows were solved with travels with them,
        # beside the law, basis and haematocrit the resistances record.
        G.graph["haematocrit_model"] = "distributed_iterative" if distribute_hct else "fixed"
    open_ends = (
        _open_end_conditions(settings, G)
        if settings["run_haemodynamics"] else haemodynamics.open_ends.OpenEndConditions()
    )

    # 6) Solve for flow (and, on the way, the conductance matrix effective
    # resistance below reads).
    if settings["run_haemodynamics"]:
        if distribute_hct:
            logger.info(
                "Solving flow with Pries-Secomb bifurcation haematocrit "
                "distribution (up to %s iterations, tolerance %s)...",
                settings["haematocrit_distribution_max_iterations"],
                settings["haematocrit_distribution_tolerance"],
            )
            # Which vessels hang behind dead ends, before the loop: each pass
            # sets their flows to exactly 0 (iterate_flow_and_haematocrit).
            graph.mark_flow_solved_edges(
                G, settings["inlet_nodes"], settings["outlet_nodes"], open_ends.nodes
            )
            if schema is None:
                from haemolynx.pipeline.schema import default_schema

                schema = default_schema()
            voxel_size_zyx = tuple(
                float(v) for v in G.graph.get("image_voxel_size_zyx", (1.0, 1.0, 1.0))
            )
            haemo_config = _haemodynamics_apply_config(
                settings, schema, voxel_size_zyx=voxel_size_zyx,
                stain=recorded_raw_image_stain(settings, G),
            )

            def _recompute_resistances() -> None:
                apply_poiseuille_resistances(G, haemo_config)

            hct_result = iterate_flow_and_haematocrit(
                G,
                recompute_resistances=_recompute_resistances,
                inlet_haematocrit=float(haemo_config.diameter("haematocrit")),
                inlet_p_bc=settings["inlet_p_bc"],
                outlet_p_bc=settings["outlet_p_bc"],
                inlet_nodes=settings["inlet_nodes"],
                outlet_nodes=settings["outlet_nodes"],
                max_iterations=int(settings["haematocrit_distribution_max_iterations"]),
                tolerance=float(settings["haematocrit_distribution_tolerance"]),
                junction_rule=_junction_rule(settings),
                solver=flow_solver,
                open_ends=open_ends,
                on_pass=(
                    None if progress is None
                    else lambda done, most: progress.step(
                        f"Haematocrit pass {done}", next_step=f"Haematocrit pass {done + 1} of up to {most}"
                    )
                ),
            )
            node_list = hct_result["node_list"]
            node_to_idx = hct_result["node_to_idx"]
            # The loop ends with a flow solve from the resistances its last
            # pass left (see iterate_flow_and_haematocrit), so its matrix and
            # pressures match the graph: no rebuild.
            conductance = hct_result["conductance_matrix"]
            solution.statistics["haematocrit_distribution"] = {
                "converged": hct_result["converged"],
                "iterations": hct_result["iterations"],
                "max_delta": hct_result["max_delta"],
                "dead_edges": hct_result.get("dead_edges", 0),
                "compound_junctions": hct_result.get("compound_junctions", 0),
                "junction_rule": _junction_rule(settings),
                "non_bifurcation_junctions": hct_result.get("non_bifurcation_junctions", 0),
            }
            logger.info(
                "Haematocrit distribution %s after %d iteration(s) (max "
                "change %.4g).",
                "converged" if hct_result["converged"] else "stopped without converging",
                hct_result["iterations"],
                hct_result["max_delta"],
            )
            # The estimate was made from one haematocrit's resistances, before
            # the loop changed them: it is made again from the ones the loop
            # left, and the flows solved once more with it (step 9).
            if open_ends.model == haemodynamics.open_ends.ESTIMATED and open_ends:
                open_ends = _open_end_conditions(settings, G)
                solve_again = True
                logger.info(
                    "open_end_model=estimated: the open ends' conditions estimated again from "
                    "the resistances the haematocrit distribution left."
                )
        else:
            conductance, node_list = haemodynamics.build_conductance_matrix_from_graph(
                G, solver=flow_solver
            )
            node_to_idx = {node_id: idx for idx, node_id in enumerate(node_list)}
        logger.info(
            f"Conductance matrix built with shape {conductance.shape} ({flow_solver}) "
            f"and node_list length {len(node_list)}."
        )

    # 7) Compute effective resistance between two selected nodes, from
    # whichever conductance matrix the flow solve above finished with.
    if settings["run_haemodynamics"] and settings["do_equiv_resistance_calculation"]:
        if progress is not None:
            progress.step("Conductance matrix", next_step="Two-point resistance")
        if resistance_node_pair is None:
            logger.warning(
                "Skipped two-point resistance: do_equiv_resistance_calculation "
                "is on but no resistance_node_pair was determined (check "
                "inlet_nodes/outlet_nodes and boundary assignment)."
            )
        else:
            source_node, target_node = _resistance_pair_in_one_piece(
                resistance_node_pair, conductance, node_to_idx,
                settings["inlet_nodes"], settings["outlet_nodes"],
            )
            if source_node in node_to_idx and target_node in node_to_idx:
                method = _equivalent_resistance_solver(settings)
                if method == haemodynamics.EQUIVALENT_RESISTANCE_SPARSE:
                    solution.equivalent_resistance = (
                        haemodynamics.calc_two_point_resistance_sparse(
                            conductance,
                            node_list,
                            source_node,
                            target_node,
                            node_to_idx=node_to_idx,
                        )
                    )
                else:
                    dense = conductance.toarray() if hasattr(conductance, "toarray") else conductance
                    laplacian = haemodynamics.calc_laplacian_from_conductance_matrix(dense)
                    solution.equivalent_resistance = haemodynamics.calc_two_point_from_laplacian_matrix_nodeID(
                        laplacian,
                        G,
                        source_node,
                        target_node,
                    )
                    # Kept, the N x N Laplacian would sit beside the flow solve's own.
                    del dense, laplacian
                solution.equivalent_resistance_nodes = (source_node, target_node)
                logger.info(
                    f"Two-point resistance between nodes {source_node} and "
                    f"{target_node} ({method}): {solution.equivalent_resistance}"
                )
            else:
                logger.warning(
                    f"Skipped two-point resistance: nodes {resistance_node_pair} "
                    "are not both present in the graph."
                )

    # 9) Solve for flow throughout the network using the conductance matrix
    # and the input and output pressures -- unless the haematocrit
    # distribution loop above already did this (repeatedly).
    if settings["run_haemodynamics"] and (not distribute_hct or solve_again):
        logger.info("Solving flow through the network...")
        matrix, names, index, extra_in, extra_out = haemodynamics.open_ends.augment(
            conductance, node_list, open_ends, node_to_idx=node_to_idx
        )
        flow = haemodynamics.solve_flow_from_conductance_matrix(
            matrix,
            names,
            inlet_p_bc=settings["inlet_p_bc"],
            outlet_p_bc=settings["outlet_p_bc"],
            inlet_nodes=[*settings["inlet_nodes"], *extra_in],
            outlet_nodes=[*settings["outlet_nodes"], *extra_out],
            node_to_idx=index,
        )
        flow["pressure"] = flow["pressure"][: len(node_list)]
        haemodynamics.set_edge_flows(G, node_list, flow["pressure"], node_to_idx=node_to_idx)
        logger.info("Flow through the network solved")
        solution.pressure = flow["pressure"]
        solution.node_list = list(node_list)
    elif settings["run_haemodynamics"] and distribute_hct:
        solution.pressure = hct_result["pressure"]
        solution.node_list = list(node_list)
    else:
        logger.info("Haemodynamics solve skipped (run_haemodynamics=False).")

    if settings["run_haemodynamics"] and solution.pressure is not None:
        solution.network_resistance = haemodynamics.network_resistance(
            conductance,
            node_list,
            solution.pressure,
            inlet_nodes=settings["inlet_nodes"],
            inlet_p_bc=settings["inlet_p_bc"],
            outlet_p_bc=settings["outlet_p_bc"],
            node_to_idx=node_to_idx,
        )
        logger.info(
            "Network resistance (pressure drop / total inflow): "
            f"{solution.network_resistance}"
        )
        if open_ends:
            through = haemodynamics.open_ends.open_end_flows(
                conductance, solution.pressure, node_to_idx, open_ends,
                inlet_p_bc=settings["inlet_p_bc"], outlet_p_bc=settings["outlet_p_bc"],
            )
            solution.statistics["open_ends"] = {
                "model": open_ends.model,
                "open_ends": len(open_ends.nodes),
                "inflow_m3_s": through["in"],
                "outflow_m3_s": through["out"],
                **{k: v for k, v in open_ends.info.items() if not isinstance(v, (list, dict))},
            }
            logger.info(
                "Open ends (open_end_model=%s): %d vessel(s) leaving the image given a boundary "
                "condition; %.4g m^3/s enters the network through them and %.4g m^3/s leaves.",
                open_ends.model, len(open_ends.nodes), through["in"], through["out"],
            )

    if solution.pressure is not None:
        _mark_flow_solved(G, settings["inlet_nodes"], settings["outlet_nodes"], open_ends.nodes)
    _mark_capillaries_fed_from_open_ends(settings, G, solved=solution.pressure is not None)

    solution.graph = G
    _stamp_solution(G, solution)
    return solution


#: What building the model and solving it write on a graph, besides
#: :data:`haemolynx.graph.SOLVE_DERIVED_EDGE_ATTRIBUTES` and the flow states.
_MODEL_EDGE_ATTRIBUTES = ("resistance", "conductance")
_SOLVE_GRAPH_RECORDS = (
    "viscosity_law",
    "haematocrit",
    "diameter_basis",
    "haematocrit_model",
    haemodynamics.open_ends.GRAPH_RECORD,
)


def _clear_earlier_solve(G: nx.MultiGraph) -> None:
    """Take off *G* what an earlier model and solve left on it -- resistances,
    flows, flow states, node pressures, the law and haematocrit they were
    found with, the open ends' conditions -- so a network solved before and
    not now (``run_haemodynamics`` turned off while a run waited at Post
    processing) is exported as one never solved, not with the old flows."""
    edge_names = (
        *graph.SOLVE_DERIVED_EDGE_ATTRIBUTES, graph.FLOW_SOLVED, graph.FLOW_STATE, *_MODEL_EDGE_ATTRIBUTES,
    )
    for *_edge, data in G.edges(data=True):
        for name in edge_names:
            data.pop(name, None)
    for _node, data in G.nodes(data=True):
        data.pop("pressure", None)
    for name in _SOLVE_GRAPH_RECORDS:
        G.graph.pop(name, None)


def _resistance_pair_in_one_piece(
    pair: tuple[Any, Any], conductance, node_to_idx: Mapping[Any, int], inlets, outlets
) -> tuple[Any, Any]:
    """*pair*, or -- when no vessel carrying a conductance joins its two
    nodes -- the first inlet and outlet that one does join, and say so. The
    pair is the first inlet and the first outlet unless Post processing chose
    two others, and those can lie in different pieces of the network: the
    sparse solver then gave an infinite resistance, the eigendecomposition a
    finite number that meant nothing."""
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components

    if any(node not in node_to_idx for node in pair):
        return pair
    _count, labels = connected_components(sp.csr_matrix(conductance) > 0, directed=False)

    def joined(a, b) -> bool:
        return a in node_to_idx and b in node_to_idx and labels[node_to_idx[a]] == labels[node_to_idx[b]]

    if joined(*pair):
        return pair
    for inlet in inlets:
        for outlet in outlets:
            if joined(inlet, outlet):
                logger.warning(
                    f"Two-point resistance: no vessel joins nodes {pair[0]} and {pair[1]}, so it "
                    f"is found between inlet {inlet} and outlet {outlet} instead, the first pair "
                    "that lie in one piece of the network."
                )
                return inlet, outlet
    return pair


def _mark_capillaries_fed_from_open_ends(settings: dict, G: nx.MultiGraph, *, solved: bool) -> None:
    """Mark the capillaries an open end feeding the network through an image
    face reaches in fewer steps than any arteriole in the image does
    (:func:`haemolynx.graph.mark_capillaries_nearer_open_end_feeds`), by this
    solve's flows. A network not solved keeps no such mark. Perturbations
    re-solve without this, so they keep the baseline's marks, as they keep
    its open-end conditions."""
    from haemolynx.graph.branch_order import capillary_start_nodes

    open_ends = haemodynamics.open_ends
    feeding = [
        node for node, way in (open_ends.open_end_directions(G) if solved else {}).items()
        if way == open_ends.FEEDS
    ]
    feeds, drains = capillary_start_nodes(
        settings["inlet_nodes"],
        settings["outlet_nodes"],
        settings.get("arteriole_boundary_nodes"),
        settings.get("venule_boundary_nodes"),
        graph=G,
    )
    marked = graph.mark_capillaries_nearer_open_end_feeds(G, feeding, feeds=feeds, drains=drains)
    if marked:
        logger.info(
            f"{marked} capillary vessel(s) are fewer steps from an open end feeding the network "
            "through an image face than from any arteriole in the image: their B orders count "
            "from the image's arterioles, but their blood comes in from outside it "
            f"({graph.BRANCH_ORDER_BASIS}={graph.OPEN_END_NEARER!r})."
        )


def _mark_flow_solved(G: nx.MultiGraph, inlet_nodes, outlet_nodes, open_ends=()) -> None:
    """Mark each vessel solved or not (:data:`haemolynx.graph.FLOW_SOLVED`)
    and why (:data:`haemolynx.graph.FLOW_STATE`), set the flows of those
    behind dead ends to exactly zero, and say how many were left unsolved and
    why."""
    from collections import Counter

    unsolved = graph.mark_flow_solved_edges(G, inlet_nodes, outlet_nodes, open_ends)
    graph.zero_flows_behind_dead_ends(G)
    if unsolved:
        states = Counter(
            data.get(graph.FLOW_STATE) for *_e, data in G.edges(data=True) if not data.get(graph.FLOW_SOLVED)
        )
        logger.info(
            f"{unsolved} of {G.number_of_edges()} vessel(s) lie on no inlet-to-outlet path, so "
            "nothing drives a flow through them: "
            f"{states.get('blind_end', 0)} hang off the flowing network behind dead ends inside the "
            f"tissue (their zero flow is the answer), {states.get('closed_end', 0)} hang behind a "
            "vessel leaving the image that has no boundary condition (open_end_model=closed), and "
            f"{states.get('no_boundary', 0)} lie in pieces without both an inlet and an outlet. "
            "They are marked unsolved."
        )


def _open_end_conditions(settings: dict, G: nx.MultiGraph):
    """The conditions ``open_end_model`` puts on the vessels leaving the
    image (:mod:`haemolynx.haemodynamics.open_ends`), kept on the graph so its
    perturbations and sweeps solve with them too. The estimate works on a
    sparse matrix whatever the flow solver (a dense one was N x N only to be
    converted), made only when there is an open end to estimate."""
    open_ends = haemodynamics.open_ends
    model = settings.get("open_end_model") or open_ends.CLOSED
    conductance = node_list = None
    if model == open_ends.ESTIMATED and open_ends.open_end_candidates(
        G, settings["inlet_nodes"], settings["outlet_nodes"]
    ):
        conductance, node_list = haemodynamics.build_conductance_matrix_from_graph(
            G, solver=haemodynamics.FLOW_SOLVER_SPARSE
        )
    # Unset is the default; 0 is a value (a shear weight of 0 estimates by
    # pressure alone), so neither is read with `or`.
    multiple = _plain_optional_float(settings.get("open_end_terminal_resistance_multiple"))
    weight = _plain_optional_float(settings.get("open_end_estimate_shear_weight"))
    conditions = open_ends.open_end_conditions(
        model, G,
        inlet_nodes=settings["inlet_nodes"], outlet_nodes=settings["outlet_nodes"],
        inlet_p_bc=float(settings["inlet_p_bc"]), outlet_p_bc=float(settings["outlet_p_bc"]),
        conductance=conductance, node_list=node_list,
        terminal_resistance_multiple=(
            open_ends.DEFAULT_TERMINAL_RESISTANCE_MULTIPLE if multiple is None else multiple
        ),
        pressure_target_pa=_plain_optional_float(settings.get("open_end_estimate_pressure_target_pa")),
        shear_target_pa=_plain_optional_float(settings.get("open_end_estimate_shear_target_pa")),
        shear_weight=open_ends.DEFAULT_SHEAR_WEIGHT if weight is None else weight,
    )
    open_ends.keep_on_graph(G, conditions)
    return conditions


#: Per-edge columns of a perturbation's `<name>_edges.csv`, in order.
PERTURBATION_EDGE_COLUMNS = (
    "u",
    "v",
    "key",
    "branch_order",
    "length_um",
    "diameter_um",
    "resistance_pa_s_per_m3",
    "conductance_m3_per_pa_s",
    "pressure_drop_pa",
    "flow_signed_m3_s",
    "flow_solved",
    "flow_state",
    "baseline_resistance_pa_s_per_m3",
    "resistance_ratio",
)

#: The edge attribute each `<name>_edges.csv` column read straight off the
#: perturbed vessel holds, through :func:`_solve_value`.
PERTURBATION_EDGE_ATTRIBUTES = {
    "branch_order": "branch_order",
    "length_um": "length",
    "diameter_um": "diameter_um",
    "resistance_pa_s_per_m3": "resistance",
    "conductance_m3_per_pa_s": "conductance",
    "pressure_drop_pa": "pressure_drop",
    "flow_signed_m3_s": "flow_signed",
    "flow_solved": graph.FLOW_SOLVED,
    "flow_state": graph.FLOW_STATE,
}


def _solve_value(data: Mapping[str, Any], name: str) -> Any:
    """Edge attribute *name* as a perturbation's CSVs write it: blank (None)
    when it is one of the solve's (``graph.SOLVE_DERIVED_EDGE_ATTRIBUTES``)
    and the vessel's flow is no answer at all (``graph.flow_is_undefined``:
    no inlet and outlet reach it), as in the VTK -- not the 0 the solve left."""
    if name in graph.SOLVE_DERIVED_EDGE_ATTRIBUTES and graph.flow_is_undefined(data):
        return None
    return data.get(name)

#: Columns of a perturbation's `<name>_summary.csv`, in order.
PERTURBATION_SUMMARY_COLUMNS = (
    "name",
    "type",
    "overrides",
    "inlet_pressure_pa",
    "outlet_pressure_pa",
    "total_inlet_flow_m3_s",
    "total_outlet_flow_m3_s",
    "flow_balance_error_m3_s",
    "equivalent_resistance_pa_s_per_m3",
    "baseline_equivalent_resistance_pa_s_per_m3",
    "delta_vs_baseline_pa_s_per_m3",
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
        viscosity_law=run_viscosity_law(settings),
        haematocrit=float(settings["haematocrit"]),
        # What the segmentation's diameters measure (segmentation_source); a
        # diameter read from the raw image carries its own
        # (poiseuille.EDGE_DIAMETER_BASIS).
        diameter_basis=run_diameter_basis(settings),
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
      `pressure_v`, `pressure_drop`, `flow_signed`, `flow_abs` on edges (and
      `graph.zero_flows_behind_dead_ends` rebinds the last three to 0.0, as
      `capillary_block.zero_stopped_vessel_flows` does the flows of a
      capillary block's stopped vessels)
    * `arteriole.scale_arteriole_diameters`,
      `capillary.scale_capillary_diameters` and
      `capillary.dilate_capillaries` -- `fwhm_diameter_um`
    * `constriction.apply_constriction_sites` and
      `PoiseuilleModel.set_poiseuille_resistances_with_constrictions` --
      `pericyte_count_assigned`, and `pericyte_centers_um` rebound to a
      freshly built list

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
    G: nx.MultiGraph,
    settings: dict,
    boundaries: BoundaryNodes,
    *,
    recompute_resistances: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Pressures, boundary flows and equivalent resistance, on *G* as it is.

    The full re-solve every perturbation gets: the pressures land on the nodes
    and the flows on the edges, so the perturbed network can be drawn and
    exported the same way the baseline is.

    When ``haematocrit_model`` is ``distributed_iterative`` and the caller
    supplies *recompute_resistances* -- a closure that redoes exactly the
    resistance computation this graph's own diameters/constrictions need,
    reading each edge's current ``discharge_haematocrit`` -- flow and
    haematocrit are iterated to a fixed point the same way the main solve
    stage does, and the returned dict also carries a
    ``"haematocrit_distribution"`` diagnostic. Without a closure (a
    perturbation type that has none to offer) this falls back to the single
    solve it has always done, using whatever ``discharge_haematocrit`` is
    already on the graph. Sweeps never reach here: each grid point iterates
    its own (see ``pericyte_sweep.solve_sweep_point``).
    """
    distribute_hct = _distributes_haematocrit(settings)
    # The baseline's conditions on the vessels leaving the image, as it kept
    # them on the graph: a perturbation is compared with the same boundaries.
    open_ends = haemodynamics.open_ends.conditions_on_graph(G)
    if distribute_hct and recompute_resistances is not None:
        hct_result = iterate_flow_and_haematocrit(
            G,
            recompute_resistances=recompute_resistances,
            inlet_haematocrit=float(settings["haematocrit"]),
            inlet_p_bc=float(settings["inlet_p_bc"]),
            outlet_p_bc=float(settings["outlet_p_bc"]),
            inlet_nodes=list(boundaries.inlet_nodes),
            outlet_nodes=list(boundaries.outlet_nodes),
            max_iterations=int(settings["haematocrit_distribution_max_iterations"]),
            tolerance=float(settings["haematocrit_distribution_tolerance"]),
            junction_rule=_junction_rule(settings),
            solver=_flow_solver(settings),
            # The solve just below, with boundary flows, is that final solve.
            final_solve=False,
            open_ends=open_ends,
        )
        # A fresh solve on the now-converged resistances is what a
        # perturbation's own equivalent-resistance comparison wants to be
        # self-consistent with. Node order never changes across that loop, so
        # reuse its node_list/node_to_idx (and a dense run's array).
        conductance, node_list = haemodynamics.build_conductance_matrix_from_graph(
            G,
            node_list=hct_result["node_list"],
            node_to_idx=hct_result["node_to_idx"],
            out=(
                hct_result["conductance_matrix"]
                if _flow_solver(settings) == haemodynamics.FLOW_SOLVER_DENSE
                else None
            ),
            solver=_flow_solver(settings),
        )
        node_to_idx = hct_result["node_to_idx"]
        solved = _solve_with_open_ends(conductance, list(node_list), settings, boundaries, open_ends)
        haemodynamics.set_edge_flows(
            G, list(node_list), solved["pressure"], node_to_idx=node_to_idx
        )
        # The flow states are the baseline's, copied with the graph: a
        # perturbation changes resistances, not which vessels can carry flow.
        graph.zero_flows_behind_dead_ends(G)
        solved["haematocrit_distribution"] = {
            "converged": hct_result["converged"],
            "iterations": hct_result["iterations"],
            "max_delta": hct_result["max_delta"],
        }
        return solved
    if distribute_hct:
        logger.warning(
            "haematocrit_model is distributed_iterative, but this "
            "perturbation has no way to redo its own resistance computation, "
            "so it keeps the discharge_haematocrit already on the graph and "
            "solves flow once instead of re-equilibrating it."
        )
    conductance, node_list = haemodynamics.build_conductance_matrix_from_graph(
        G, solver=_flow_solver(settings)
    )
    solved = _solve_with_open_ends(conductance, list(node_list), settings, boundaries, open_ends)
    haemodynamics.set_edge_flows(G, list(node_list), solved["pressure"])
    graph.zero_flows_behind_dead_ends(G)
    return solved


def _solve_with_open_ends(conductance, node_list: list, settings: dict, boundaries: BoundaryNodes, open_ends):
    """:func:`solve_pressure_and_boundary_flow` with the open ends' ghosts
    added, the pressures trimmed back to the graph's own nodes. The boundary
    flows count the ghosts' too, so the balance closes; the equivalent
    resistance is the drop over the designated inlets' inflow, the run's
    network resistance (`haemodynamics.network_resistance`)."""
    matrix, names, _index, extra_in, extra_out = haemodynamics.open_ends.augment(
        conductance, node_list, open_ends
    )
    solved = solve_pressure_and_boundary_flow(
        matrix,
        names,
        inlet_p_bc=float(settings["inlet_p_bc"]),
        outlet_p_bc=float(settings["outlet_p_bc"]),
        inlet_nodes=[*boundaries.inlet_nodes, *extra_in],
        outlet_nodes=[*boundaries.outlet_nodes, *extra_out],
        resistance_inlet_nodes=list(boundaries.inlet_nodes) if extra_in else None,
    )
    solved["pressure"] = solved["pressure"][: len(node_list)]
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
                "total_inlet_flow_m3_s": solved["total_inlet_flow"],
                "total_outlet_flow_m3_s": solved["total_outlet_flow"],
                "flow_balance_error_m3_s": (
                    solved["total_inlet_flow"] + solved["total_outlet_flow"]
                ),
                "equivalent_resistance_pa_s_per_m3": equivalent,
                "baseline_equivalent_resistance_pa_s_per_m3": baseline_equivalent,
                "delta_vs_baseline_pa_s_per_m3": delta,
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
                    **{
                        column: _solve_value(data, name)
                        for column, name in PERTURBATION_EDGE_ATTRIBUTES.items()
                    },
                    "baseline_resistance_pa_s_per_m3": was,
                    "resistance_ratio": (
                        float(now) / float(was) if was and now is not None else None
                    ),
                }
            )
    result.outputs.append(edges_path)


def _baseline_voxel_size_zyx(
    G: nx.MultiGraph, network: VesselNetwork | None
) -> tuple[float, float, float]:
    """The voxel size of the image the baseline network was built from, in
    array ``(z, y, x)`` order.

    The loaded volume's own, when the run has it; otherwise what graph
    building recorded on the graph (``image_voxel_size_zyx``, or the
    ``voxel_size`` the builder stamps). Only a graph that carries neither
    falls back to 1 um -- and says so, since every volume and density a
    perturbation reports would then be in voxels.
    """
    volume = getattr(network, "volume", None) if network is not None else None
    for candidate in (
        getattr(volume, "voxel_size_zyx", None),
        G.graph.get("image_voxel_size_zyx"),
        G.graph.get("voxel_size"),
    ):
        if candidate is not None and len(candidate) == 3:
            return tuple(float(v) for v in candidate)
    logger.warning(
        "Perturbations: no voxel size on the run or the baseline graph; "
        "their statistics and plots assume 1 um voxels."
    )
    return (1.0, 1.0, 1.0)


def _listed(value: Any) -> list:
    """A list setting as a list, however it was typed: a lone value, or a
    comma-separated string from a text box (``"BO2, BO3"``, ``"12, 40"``)."""
    if value is None:
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


#: Metric/value rows of a capillary block's `<name>_block_comparison.csv`.
CAPILLARY_BLOCK_COMPARISON_COLUMNS = ("metric", "value")
#: Columns of a capillary block's `<name>_blocked_vessels.csv`, in order.
CAPILLARY_BLOCK_VESSEL_COLUMNS = (
    "branch_id",
    "u",
    "v",
    "key",
    "branch_order",
    "length_um",
    "diameter_um",
    "baseline_resistance_pa_s_per_m3",
    "blocked_resistance_pa_s_per_m3",
    "baseline_flow_abs_m3_s",
    "blocked_flow_abs_m3_s",
)
#: Columns of a capillary block's `<name>_flow_by_branch_order.csv`, in order.
CAPILLARY_BLOCK_ORDER_COLUMNS = (
    "branch_order",
    "vessels",
    "blocked",
    "baseline_mean_flow_m3_s",
    "blocked_mean_flow_m3_s",
    "mean_flow_percent_change",
    "hypoperfused",
    "hypoperfused_length_um",
    "reversed",
)


def _capillary_block_seeds(
    settings: Mapping[str, Any], perturbation_type: str = "capillary_block"
) -> list[int | None]:
    """The seed of every draw of a capillary block: one, unless the vessels are
    drawn at random and ``capillary_block_replicates`` asks for more. A block
    read from a mask draws nothing: one draw, no seed."""
    if perturbation_type == "capillary_block_mask":
        return [None]
    count = 1
    if settings["capillary_block_selection"] == "branch_order_probability":
        count = int(settings.get("capillary_block_replicates") or 1)
    return haemodynamics.replicate_seeds(settings["capillary_block_seed"], count)


def _block_capillaries(
    G: nx.MultiGraph,
    settings: Mapping[str, Any],
    schema: Schema,
    *,
    seed: int | None,
    chosen: tuple[list, dict[str, Any]] | None = None,
) -> tuple[list, dict[str, Any], Callable[[], None] | None]:
    """Block one draw's vessels on *G*: ``(blocked vessels, how they were
    chosen, and -- under a distributed haematocrit -- the closure that redoes
    the resistances from each edge's updated haematocrit, else None)``.
    *chosen* is ``(blocked vessels, summary)`` found already (a block mask's);
    without it the settings' selection draws them."""
    if chosen is not None:
        blocked_edges, block_summary = list(chosen[0]), dict(chosen[1])
    else:
        blocked_edges, block_summary = haemodynamics.resolve_blocked_vessels(
            G,
            selection=settings["capillary_block_selection"],
            branch_orders=_listed(settings["capillary_block_branch_orders"]),
            probability=float(settings["capillary_block_probability"]),
            seed=seed,
            vessel_ids=_listed(settings["capillary_block_vessel_ids"]),
        )
    block_factor = float(settings["capillary_block_resistance_factor"])
    haemodynamics.block_vessels(G, blocked_edges, block_factor)
    if not _distributes_haematocrit(settings):
        return blocked_edges, block_summary, None
    block_config = _haemodynamics_apply_config(
        settings,
        schema,
        voxel_size_zyx=tuple(
            float(v) for v in G.graph.get("image_voxel_size_zyx", (1.0, 1.0, 1.0))
        ),
    )

    def recompute_resistances() -> None:
        # The baseline's own resistance computation, from the new
        # discharge_haematocrit, then the same blocks on top of it.
        apply_poiseuille_resistances(G, block_config)
        haemodynamics.block_vessels(G, blocked_edges, block_factor)

    return blocked_edges, block_summary, recompute_resistances


def _capillary_block_replicates(
    model_graph: nx.MultiGraph,
    settings: dict,
    schema: Schema,
    boundaries: BoundaryNodes,
    *,
    seeds: Sequence[int | None],
    baseline_graph: nx.MultiGraph,
    hypoperfusion_fraction: float,
    heartbeat: Callable[[], None] | None = None,
) -> list[tuple[int | None, int, dict[str, Any]]]:
    """``(seed, vessels blocked, comparison with the baseline)`` for each further
    draw of a capillary block: each from its own copy of the baseline, solved
    the way the first draw is. *heartbeat* is called before each."""
    replicates = []
    for seed in seeds:
        if heartbeat is not None:
            heartbeat()
        G = _perturbation_copy(model_graph)
        blocked_edges, _summary, recompute = _block_capillaries(G, settings, schema, seed=seed)
        _solve_network(G, settings, boundaries, recompute_resistances=recompute)
        haemodynamics.zero_stopped_vessel_flows(
            G, list(boundaries.inlet_nodes), list(boundaries.outlet_nodes)
        )
        comparison, _by_order = haemodynamics.compare_block_to_baseline(
            baseline_graph,
            G,
            inlet_nodes=list(boundaries.inlet_nodes),
            outlet_nodes=list(boundaries.outlet_nodes),
            hypoperfusion_fraction=hypoperfusion_fraction,
            red_cell_volume_fl=float(settings["red_cell_volume_fl"]),
        )
        replicates.append((seed, len(blocked_edges), comparison))
    return replicates


#: Columns of a capillary block's `<name>_block_replicates.csv`, before the
#: comparison metrics: one row per draw.
CAPILLARY_BLOCK_REPLICATE_COLUMNS = ("draw", "seed", "blocked_vessels")

#: Columns of a mask capillary block's `<name>_mask_blocks.csv`: one row per
#: blob of the mask, blocking or not (`reason` says why not).
CAPILLARY_BLOCK_MASK_COLUMNS = (
    "block",
    "voxels",
    "volume_um3",
    "centroid_z_um",
    "centroid_y_um",
    "centroid_x_um",
    "voxels_in_vessels",
    "overlap_fraction",
    "blocks",
    "blocked_branch_ids",
    "vessel_voxels",
    "reason",
)

#: Metric/value rows of a perturbation mask's `<name>_mask_distribution.csv`:
#: a pericyte or block mask's objects, how densely they sit on the
#: capillaries, the vessels they sit on, and where they lie in the image's
#: depth (haemodynamics.mask_distribution).
MASK_DISTRIBUTION_COLUMNS = ("metric", "value")
#: Columns of a perturbation mask's `<name>_mask_density_by_branch_order.csv`.
MASK_DENSITY_ORDER_COLUMNS = ("branch_order", "vessels", "length_um", "objects", "objects_per_mm")
#: Columns of a perturbation mask's `<name>_mask_objects.csv`: one row per
#: object, and the vessel it sits on.
MASK_OBJECT_COLUMNS = (
    "object",
    "centroid_z_um",
    "centroid_y_um",
    "centroid_x_um",
    "volume_um3",
    "on_vessel",
    "branch_id",
    "u",
    "v",
    "key",
    "branch_order",
    "capillary",
    "vessel_length_um",
    "vessel_diameter_um",
    "vessel_branch_angle_deg",
)


def _write_mask_distribution(
    distribution: haemodynamics.MaskDistribution,
    *,
    output_dir: Path,
    name: str,
    mask_path: Any,
) -> list[Path]:
    """A perturbation mask's distribution as its three CSVs in *output_dir*:
    the metric/value summary, the density per branch order, and each
    object with its vessel."""
    summary_path = output_dir / f"{name}_mask_distribution.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(MASK_DISTRIBUTION_COLUMNS)
        writer.writerow(["mask_path", Path(mask_path).as_posix()])
        for key, value in distribution.summary.items():
            writer.writerow([key, value])
    orders_path = output_dir / f"{name}_mask_density_by_branch_order.csv"
    with orders_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(MASK_DENSITY_ORDER_COLUMNS))
        writer.writeheader()
        writer.writerows(distribution.by_branch_order)
    objects_path = output_dir / f"{name}_mask_objects.csv"
    with objects_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(MASK_OBJECT_COLUMNS))
        writer.writeheader()
        writer.writerows(distribution.objects)
    rows = distribution.summary
    logger.info(
        f"Perturbation '{name}': {rows['objects_on_vessels']} of the mask's {rows['objects']} "
        f"object(s) lie on vessels, {rows['objects_per_mm_capillary']:.4g} per mm of capillary, "
        f"on vessels {rows['vessel_length_mean_um']:.4g} um long and "
        f"{rows['vessel_diameter_mean_um']:.4g} um wide on average; depth "
        f"{rows['depth_min_um_all']:.4g}-{rows['depth_max_um_all']:.4g} um (mean "
        f"{rows['depth_mean_um_all']:.4g}, mode {rows['depth_mode_um_all']:.4g}) of an image "
        f"{rows['image_depth_um']:.4g} um deep."
    )
    return [summary_path, orders_path, objects_path]


def _pericyte_mask_distribution(
    G: nx.MultiGraph,
    settings: Mapping[str, Any],
    *,
    output_dir: Path,
    name: str,
    placed: Any | None = None,
) -> tuple[haemodynamics.MaskDistribution, list[Path]]:
    """Where a pericyte perturbation's mask puts its pericytes on *G*, placed
    as its constriction strategy places them (*placed*, the placement the
    strategy used; else read and placed with the same arguments,
    `placed_pericyte_mask`), and the two CSVs that say so."""
    if placed is None:
        placed = placed_pericyte_mask(G, settings)
    distribution = haemodynamics.pericyte_mask_distribution(
        G, placed.placement, image_shape=placed.image_shape, voxel_size_zyx=placed.voxel_size_zyx
    )
    paths = _write_mask_distribution(
        distribution, output_dir=output_dir, name=name, mask_path=settings["pericyte_mask_path"]
    )
    return distribution, paths


def _capillary_blocks_from_mask(
    G: nx.MultiGraph,
    settings: Mapping[str, Any],
    *,
    image: np.ndarray | None,
    output_dir: Path,
    name: str,
) -> tuple[tuple[list, dict[str, Any]], list[Path]]:
    """The vessels a block mask's blobs block on *G*, with how they were
    found, and the per-blob CSV and the mask's distribution CSVs written into
    *output_dir*. *image* is the run's segmentation on the network's grid;
    without it the graph's lumens stand in, with a warning."""
    from haemolynx.haemodynamics.capillary_block_mask import (
        blocks_from_mask,
        load_capillary_block_mask,
    )

    mask_path = settings.get("capillary_block_mask_path")
    if mask_path in (None, ""):
        raise ValueError(
            f"perturbation '{name}' names no capillary_block_mask_path: set it to the "
            "mask of blocks"
        )
    block_mask, mask_voxel_size_zyx = load_capillary_block_mask(
        mask_path,
        G,
        h5_dataset_name=settings.get("capillary_block_mask_h5_dataset_name"),
        axis_order=io.axis_order_for_run(settings),
        voxel_size_override_xyz=settings.get("voxel_size_override_xyz"),
        voxel_size_policy=settings.get("voxel_size_policy", "auto"),
        use_memmap=bool(settings.get("use_memmap_loading")),
        memmap_directory=settings.get("memmap_directory"),
    )
    if image is None:
        logger.warning(
            f"Perturbation '{name}': no segmented image to read the blocks against, so "
            "the graph's own lumens (each vessel's centreline and diameter) stand in for "
            "the vessel mask."
        )
    blocked, summary, blocks = blocks_from_mask(
        G,
        block_mask,
        mask_voxel_size_zyx,
        vessel_mask=image,
        min_overlap_fraction=float(settings["capillary_block_mask_min_overlap_fraction"]),
        vessels_per_block=str(settings["capillary_block_mask_vessels_per_block"]),
    )
    summary["capillary_block_mask_path"] = Path(mask_path).as_posix()
    path = output_dir / f"{name}_mask_blocks.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CAPILLARY_BLOCK_MASK_COLUMNS))
        writer.writeheader()
        for block in blocks:
            writer.writerow(
                {
                    "block": block.number,
                    "voxels": block.voxels,
                    "volume_um3": block.volume_um3,
                    "centroid_z_um": block.centroid_um[0],
                    "centroid_y_um": block.centroid_um[1],
                    "centroid_x_um": block.centroid_um[2],
                    "voxels_in_vessels": block.voxels_in_vessels,
                    "overlap_fraction": block.overlap_fraction,
                    "blocks": block.blocks,
                    "blocked_branch_ids": json.dumps(block.blocked_branch_ids),
                    "vessel_voxels": json.dumps(block.vessel_voxels),
                    "reason": block.reason,
                }
            )
    distribution = haemodynamics.block_mask_distribution(
        G, blocks, image_shape=block_mask.shape, voxel_size_zyx=mask_voxel_size_zyx
    )
    summary["mask_distribution"] = distribution.summary
    paths = _write_mask_distribution(distribution, output_dir=output_dir, name=name, mask_path=mask_path)
    return (blocked, summary), [path, *paths]


def _write_capillary_block_comparison(
    result: PerturbationResult,
    *,
    baseline_graph: nx.MultiGraph,
    boundaries: BoundaryNodes,
    hypoperfusion_fraction: float,
    selection_summary: dict[str, Any],
    replicates: Sequence[tuple[int | None, int, dict[str, Any]]] | None = None,
    first_seed: int | None = None,
    red_cell_volume_fl: float = haemodynamics.DEFAULT_RED_CELL_VOLUME_FL,
) -> dict[str, Any]:
    """The blocked network's flow set against the baseline's, as three CSVs:
    what was blocked, the network-level comparison, and the comparison per
    branch order. Returns the network-level comparison.

    With *replicates* -- the further draws' comparisons -- also a fourth CSV
    with one row per draw (the first included), and every metric's mean, SD
    and 2.5-97.5% range over the draws in the comparison, as
    ``<metric>_draws_mean`` and so on."""
    G = result.graph
    output_dir = result.output_dir
    assert G is not None and output_dir is not None
    comparison, by_order = haemodynamics.compare_block_to_baseline(
        baseline_graph,
        G,
        inlet_nodes=list(boundaries.inlet_nodes),
        outlet_nodes=list(boundaries.outlet_nodes),
        hypoperfusion_fraction=hypoperfusion_fraction,
        red_cell_volume_fl=red_cell_volume_fl,
    )

    vessels_path = output_dir / f"{result.name}_blocked_vessels.csv"
    with vessels_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CAPILLARY_BLOCK_VESSEL_COLUMNS))
        writer.writeheader()
        for branch_id, (u, v, key, data) in enumerate(G.edges(keys=True, data=True)):
            if data.get("capillary_block") != "blocked":
                continue
            before = baseline_graph.get_edge_data(u, v, key) or {}
            writer.writerow(
                {
                    "branch_id": branch_id,
                    "u": u,
                    "v": v,
                    "key": key,
                    "branch_order": data.get("branch_order"),
                    "length_um": data.get("length"),
                    "diameter_um": data.get("diameter_um"),
                    "baseline_resistance_pa_s_per_m3": before.get("resistance"),
                    "blocked_resistance_pa_s_per_m3": data.get("resistance"),
                    "baseline_flow_abs_m3_s": _solve_value(before, "flow_abs"),
                    "blocked_flow_abs_m3_s": _solve_value(data, "flow_abs"),
                }
            )
    result.outputs.append(vessels_path)

    comparison_path = output_dir / f"{result.name}_block_comparison.csv"
    with comparison_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(CAPILLARY_BLOCK_COMPARISON_COLUMNS)
        for key in ("selection", "branch_orders", "probability", "seed",
                    "candidate_vessels", "vessel_ids", "capillary_block_mask_path",
                    "mask_blocks", "mask_blocks_on_vessels", "mask_blocks_below_overlap",
                    "mask_blocks_off_graph", "min_overlap_fraction", "vessels_per_block",
                    "vessel_mask_source", "blocked_vessels", "resistance_factor"):
            if key in selection_summary:
                value = selection_summary[key]
                writer.writerow(
                    [key, json.dumps(plain_values(value)) if isinstance(value, (list, tuple)) else value]
                )
        for key, value in comparison.items():
            writer.writerow([key, value])
        if replicates:
            draws = [comparison] + [item[2] for item in replicates]
            spread = haemodynamics.summarise_block_replicates(draws)
            writer.writerow(["draws", len(draws)])
            for metric, stats in spread.items():
                for stat in ("mean", "sd", "low", "high"):
                    writer.writerow([f"{metric}_draws_{stat}", stats[stat]])
            comparison["draws"] = len(draws)
            comparison["draw_statistics"] = spread
    result.outputs.append(comparison_path)

    if replicates:
        rows = [(first_seed, int(selection_summary.get("blocked_vessels", 0)), draws[0])]
        rows += [(seed, blocked, compared) for seed, blocked, compared in replicates]
        metric_names = list(spread)
        replicates_path = output_dir / f"{result.name}_block_replicates.csv"
        with replicates_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(list(CAPILLARY_BLOCK_REPLICATE_COLUMNS) + metric_names)
            for draw, (seed, blocked, compared) in enumerate(rows, start=1):
                writer.writerow(
                    [draw, seed, blocked] + [compared.get(name) for name in metric_names]
                )
        result.outputs.append(replicates_path)

    orders_path = output_dir / f"{result.name}_flow_by_branch_order.csv"
    with orders_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CAPILLARY_BLOCK_ORDER_COLUMNS))
        writer.writeheader()
        writer.writerows(by_order)
    result.outputs.append(orders_path)

    logger.info(
        f"Perturbation '{result.name}': total inflow "
        f"{comparison['total_inflow_percent_change']:+.3g}% vs baseline; "
        f"{comparison['hypoperfused_vessels']} other vessel(s) lost at least "
        f"{hypoperfusion_fraction:.0%} of their flow, "
        f"{comparison['reversed_vessels']} reversed."
    )
    return comparison


def _fwhm_profile_reader(
    settings: Mapping[str, Any],
    G: nx.MultiGraph,
    vessel_mask: np.ndarray,
    voxel_size_zyx: Sequence[float],
) -> pericyte_measurement.DiameterReader:
    """The raw image's FWHM at the pericyte profiles' points, fitted as this
    run's Diameters stage fits its samples: its FWHM settings, its raw image
    and channel on the graph's grid, *vessel_mask* for the lines to stop at
    the vessel beside, and the PSF it held the blur at (``fwhm_psf_sigma_zyx``
    on the graph; estimated from the image as it would be when the run
    measured no FWHM). Read on a copy of *G*, whose widths it writes."""
    from haemolynx.haemodynamics.apply import (
        image_psf_from_settings,
        load_fwhm_raw_volume_like_graph,
        measure_fwhm_diameters,
    )
    from haemolynx.pipeline.schema import default_schema

    if not raw_image_is_set(settings):
        raise ValueError(
            "pericyte_profile_reading is raw_image_fwhm, which reads the raw image, and "
            "raw_image_path is not set."
        )
    if recorded_raw_image_stain(settings, G) == raw_stain.ENDOTHELIAL:
        logger.warning(
            "Pericyte profiles: the raw image stains the vessel wall, and a FWHM across a "
            "wall stain is not the lumen's width; read the segmentation instead "
            "(pericyte_profile_reading: segmentation)."
        )
    voxel = tuple(float(v) for v in G.graph.get("image_voxel_size_zyx", voxel_size_zyx))
    config = _haemodynamics_apply_config(dict(settings), default_schema(), voxel_size_zyx=voxel)
    raw = load_fwhm_raw_volume_like_graph(config, G)
    psf = G.graph.get("fwhm_psf_sigma_zyx")
    if psf is None:
        psf, _details = image_psf_from_settings(
            G, config.fwhm, voxel_size_zyx=voxel, raw_volume=raw, vessel_mask=vessel_mask,
            memmap_directory=config.memmap_directory,
        )
    psf = None if psf is None else tuple(float(v) for v in psf)

    def measure(at_positions):
        summary = measure_fwhm_diameters(
            G.copy(), config, raw_volume=raw, vessel_mask=vessel_mask, psf_sigma_zyx=psf,
            edges=list(at_positions), at_positions=at_positions,
        )
        return summary.get("readings_at_positions", {})

    return pericyte_measurement.fwhm_reader(G, measure)


def measure_around_pericytes(
    settings: Mapping[str, Any],
    G: nx.MultiGraph,
    image: np.ndarray | None,
    *,
    mask_path: Any,
    h5_dataset_name: str | None,
    max_assignment_distance_um: float | None,
    min_pericyte_diameter_um: float | None,
    max_pericyte_diameter_um: float | None,
    profiles: bool,
    junctions: bool,
) -> pericyte_measurement.PericyteMeasurements:
    """The pericytes of the mask at *mask_path* placed on *G*, with what the
    run asks of them: diameter profiles read off *image* (the run's
    segmentation, binarised here as the Diameters stage binarises it) and
    their branches' resistances by the run's blood model (*profiles*), and
    their distances to their vessels' nearest junctions (*junctions*). One
    reader for ``measure_pericytes`` and the measured-constriction
    perturbation, so the two place and measure the pericytes alike."""
    from haemolynx.haemodynamics.pericyte_mask import load_pericyte_mask
    from haemolynx.haemodynamics.sections import default_worker_count

    if mask_path in (None, ""):
        raise ValueError("No pericyte mask to measure around.")
    if profiles and image is None:
        raise ValueError(
            "The pericytes' diameter profiles are read off the run's segmented image, and "
            "this run has none in memory (a synthetic network, or a resumed run without it)."
        )
    use_memmap = bool(settings.get("use_memmap_loading"))
    memmap_directory = settings.get("memmap_directory")
    mask, _voxel_size_xyz, voxel_size_zyx = load_pericyte_mask(
        mask_path,
        G,
        h5_dataset_name=h5_dataset_name,
        axis_order=io.axis_order_for_run(settings),
        voxel_size_override_xyz=settings.get("voxel_size_override_xyz"),
        voxel_size_policy=settings.get("voxel_size_policy", "auto"),
        use_memmap=use_memmap,
        memmap_directory=memmap_directory,
    )
    vessel_mask = reader = None
    if profiles:
        vessel_mask = _to_binary_volume_for_skeletonization(
            image, use_memmap=use_memmap, memmap_directory=memmap_directory
        )
        if settings.get("pericyte_profile_reading") == "raw_image_fwhm":
            reader = _fwhm_profile_reader(settings, G, vessel_mask, voxel_size_zyx)
    try:
        return pericyte_measurement.measure_pericytes(
            G,
            mask,
            voxel_size_zyx,
            vessel_mask=vessel_mask,
            reader=reader,
            profiles=profiles,
            junctions=junctions,
            step_um=float(settings.get("pericyte_profile_step_um") or pericyte_measurement.DEFAULT_PROFILE_STEP_UM),
            reach_um=float(
                pericyte_measurement.DEFAULT_PROFILE_REACH_UM
                if settings.get("pericyte_profile_reach_um") is None
                else settings["pericyte_profile_reach_um"]
            ),
            max_assignment_distance_um=max_assignment_distance_um,
            min_pericyte_diameter_um=min_pericyte_diameter_um,
            max_pericyte_diameter_um=max_pericyte_diameter_um,
            model=_poiseuille_model_for(dict(settings)),
            diameter_by_branch_order=settings.get("diameter_by_branch_order") or {},
            prefer_edge_fwhm_diameter=bool(settings.get("measure_diameters_from_raw_image")),
            workers=default_worker_count(),
            memmap_directory=memmap_directory,
        )
    finally:
        if isinstance(vessel_mask, np.memmap) and vessel_mask is not image:
            preprocessing.release_memmap_array(vessel_mask)


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
    baseline_graph: nx.MultiGraph | None = None,
    voxel_size_zyx: tuple[float, float, float] | None = None,
    tissue_statistics: Mapping[str, Any] | None = None,
    image_shape: tuple[int, ...] | None = None,
    heartbeat: Callable[[], None] | None = None,
) -> PerturbationResult:
    """Run one perturbation from the baseline network, and write its output.

    *baseline_graph* is the baseline as re-solved by this same solver (see
    `run_perturbations`), for a perturbation that compares its flows vessel
    by vessel; `model.graph` stands in when a caller has none.
    *voxel_size_zyx* is the baseline image's own voxel size, which every
    statistic and plot of this perturbation scales by -- see
    `_baseline_voxel_size_zyx`. *tissue_statistics* is the baseline's tissue
    volume (`tissue_statistics_arguments`), which its vessel density divides by.
    *image_shape* is the voxel shape of the box the baseline's statistics
    divide by (`_statistics_image_shape`). *heartbeat* is called once per
    sweep grid point and block draw: a cancelled run raises from it.
    """
    if voxel_size_zyx is None:
        voxel_size_zyx = _baseline_voxel_size_zyx(model.graph, None)
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
    mask = haemodynamics.perturbation_mask_setting(spec.type, {**settings, **spec.applied_overrides(schema)})
    if mask is not None and graph.network_is_synthetic(settings):
        raise ValueError(
            f"perturbation '{spec.name}' reads {mask}, which lies on an image's voxel grid, and "
            "this network is synthetic: nothing places the mask on its vessels."
        )

    overrides = spec.applied_overrides(schema)
    perturbed = {**settings, **overrides}
    # Top-level do_pericyte_construction / comparison flags must not bleed
    # into any perturbation. Non-pericyte types stay on uniform Poiseuille;
    # pericyte types place constrictions via their own strategy call, not
    # because the global flag survived the merge.
    perturbed["do_pericyte_construction"] = False
    perturbed["run_pericyte_resistance_comparison"] = False
    drawn_seed = None
    if (
        places_pericytes(spec.type)
        and perturbed.get("use_probabilistic_pericyte_constriction")
        and perturbed.get("pericyte_constriction_seed") is None
    ):
        # No seed: a fresh cohort this run, but one cohort for the whole
        # perturbation. Drawn at every call, each haematocrit pass and each
        # sweep grid point constricted pericytes of its own.
        drawn_seed = int(np.random.default_rng().integers(0, 2**31 - 1))
        perturbed["pericyte_constriction_seed"] = drawn_seed
        logger.info(
            f"Perturbation '{spec.name}': no pericyte_constriction_seed, so its cohort "
            f"is drawn once from seed {drawn_seed}."
        )
    result.output_dir = root / perturbation_folder_name(spec.name, spec.type)
    result.output_dir.mkdir(parents=True, exist_ok=True)

    # A copy per perturbation, so that nothing it does reaches the baseline or
    # another perturbation: they answer independent questions of the same
    # network, and the run's own graph is still exported afterwards. See
    # `_perturbation_copy` for why a shallow graph copy is enough.
    # `G`, not `graph`, because `graph` is this module's imported subpackage.
    G = _perturbation_copy(model.graph)
    diameter_table = perturbed["diameter_by_branch_order"]
    prefer_measured = bool(perturbed["measure_diameters_from_raw_image"])
    summary: dict[str, Any] = {"overrides": overrides}
    if drawn_seed is not None:
        summary["pericyte_constriction_seed_drawn"] = drawn_seed
    #: Sweep helper payload when this type ran a grid; used for Alice plots.
    sweep_payload: dict[str, Any] | None = None
    #: Redoes this perturbation's own resistance computation from its
    #: current diameters/constrictions, given a freshly updated per-edge
    #: discharge_haematocrit -- set by whichever branch below changes
    #: geometry. Sweeps leave it None: they iterate per grid point
    #: themselves (see pericyte_sweep.solve_sweep_point).
    recompute_resistances: Callable[[], None] | None = None
    # The pericyte mask read and placed once, for every strategy call of this
    # perturbation (each sweep grid point, each haematocrit pass) and its
    # mask distribution. The measured constriction reads its own.
    placed = (
        placed_pericyte_mask(G, perturbed)
        if places_pericytes(spec.type) and spec.type != MEASURED_CONSTRICTION_TYPE
        else None
    )
    if placed is not None and image is not None and tuple(placed.image_shape) != tuple(image.shape):
        raise ValueError(
            f"perturbation '{spec.name}': the pericyte mask is {tuple(placed.image_shape)} voxels on "
            f"the network's grid, and the segmented image {tuple(image.shape)}: it is not a mask "
            "of this image. Save the mask from the image the network was built from."
        )

    if spec.type == "pressure_sweep":
        sweep_payload = run_pericyte_dilation_pressure_sweep(
            G,
            perturbed,
            inlet_nodes=list(boundaries.inlet_nodes),
            outlet_nodes=list(boundaries.outlet_nodes),
            output_dir=result.output_dir,
            heartbeat=heartbeat,
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
            heartbeat=heartbeat,
            sweep_dilation=True,
            sweep_pressure=True,
            pericyte_mask_placed=placed,
        )
    elif spec.type == "pericyte_dilation_sweep":
        sweep_payload = run_pericyte_dilation_pressure_sweep(
            G,
            perturbed,
            inlet_nodes=list(boundaries.inlet_nodes),
            outlet_nodes=list(boundaries.outlet_nodes),
            output_dir=result.output_dir,
            heartbeat=heartbeat,
            sweep_dilation=True,
            sweep_pressure=False,
            pericyte_mask_placed=placed,
        )
    elif spec.type == "arteriole_diameter_change":
        arteriole_model = _poiseuille_model_for(perturbed)
        G, scaled_table, scaling = haemodynamics.scale_arteriole_diameters(
            G,
            diameter_table,
            haemodynamics.percent_change_to_scale(
                float(perturbed["arteriole_diameter_change_percent"])
            ),
            model=arteriole_model,
            prefer_edge_fwhm_diameter=prefer_measured,
        )
        apply_baseline_overrides(G, perturbed, arteriole_model)
        summary.update(scaling)
        summary["arteriole_diameter_change_percent"] = float(
            perturbed["arteriole_diameter_change_percent"]
        )

        def recompute_resistances() -> None:
            # scale_arteriole_diameters already moved every stored diameter
            # once; set_poiseuille_resistances reads diameter_um straight off
            # each edge before ever falling back to scaled_table, so calling
            # it again does not re-scale anything -- only resistance moves,
            # from the just-updated discharge_haematocrit.
            arteriole_model.set_poiseuille_resistances(
                G, scaled_table, prefer_edge_fwhm_diameter=prefer_measured
            )
            apply_baseline_overrides(G, perturbed, arteriole_model)
    elif spec.type == "arteriole_diameter_sweep":
        sweep_payload = run_arteriole_dilation_pressure_sweep(
            G,
            perturbed,
            inlet_nodes=list(boundaries.inlet_nodes),
            outlet_nodes=list(boundaries.outlet_nodes),
            output_dir=result.output_dir,
            heartbeat=heartbeat,
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
            heartbeat=heartbeat,
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
            heartbeat=heartbeat,
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
            heartbeat=heartbeat,
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
            heartbeat=heartbeat,
            sweep_axis="spacing",
            pericyte_mask_placed=placed,
        )
    elif spec.type == "pericyte_length_sweep":
        sweep_payload = run_pericyte_geometry_sweep(
            G,
            perturbed,
            inlet_nodes=list(boundaries.inlet_nodes),
            outlet_nodes=list(boundaries.outlet_nodes),
            output_dir=result.output_dir,
            heartbeat=heartbeat,
            sweep_axis="length",
            pericyte_mask_placed=placed,
        )
    elif spec.type == "pericyte_diameter_change":
        constriction_kwargs = constriction_strategy_kwargs(
            perturbed, diameter_by_branch_order=diameter_table, pericyte_mask_placed=placed
        )
        G, strategy, strategy_results = set_resistances_for_constriction_strategy(
            G, **constriction_kwargs
        )
        apply_baseline_overrides(G, perturbed, _poiseuille_model_for(perturbed))
        summary["strategy"] = strategy
        summary.update(strategy_results)

        def recompute_resistances() -> None:
            # Every strategy re-derives site placement and d1/d2 fresh from
            # branch_order/length/fwhm_diameter_um and the (seeded, so
            # reproducible) site generator -- none of that depends on a prior
            # call's mutations, so redoing it just refreshes resistance from
            # the new discharge_haematocrit without moving the sites.
            set_resistances_for_constriction_strategy(G, **constriction_kwargs)
            apply_baseline_overrides(G, perturbed, _poiseuille_model_for(perturbed))
    elif spec.type == "arteriole_and_pericyte_diameter_change":
        # Arteriole whole-branch % scale first, then focal pericyte
        # constrictions on the scaled diameters. Reverse would wipe
        # constrictions: scale_arteriole_diameters re-solves uniform
        # Poiseuille on the whole graph. Constrictions are applied by
        # type via the strategy call below — not via do_pericyte_construction.
        G, scaled_table, scaling = haemodynamics.scale_arteriole_diameters(
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
        constriction_kwargs = constriction_strategy_kwargs(
            perturbed, diameter_by_branch_order=scaled_table, pericyte_mask_placed=placed
        )
        G, strategy, strategy_results = set_resistances_for_constriction_strategy(
            G, **constriction_kwargs
        )
        apply_baseline_overrides(G, perturbed, _poiseuille_model_for(perturbed))
        summary["strategy"] = strategy
        summary.update(strategy_results)

        def recompute_resistances() -> None:
            # The arteriole scale already moved every stored diameter once
            # (see the arteriole_diameter_change branch above); only the
            # constriction step needs redoing per iteration, using the fixed
            # scaled_table it already closed over.
            set_resistances_for_constriction_strategy(G, **constriction_kwargs)
            apply_baseline_overrides(G, perturbed, _poiseuille_model_for(perturbed))
    elif spec.type == MEASURED_CONSTRICTION_TYPE:
        if not perturbed.get("use_pericyte_mask_constriction") or perturbed.get("pericyte_mask_path") in (None, ""):
            raise ValueError(
                f"perturbation '{spec.name}' narrows the branches a pericyte mask's pericytes "
                "sit on, but reads no mask: turn on use_pericyte_mask_constriction and set "
                "pericyte_mask_path"
            )
        measured = measure_around_pericytes(
            perturbed,
            G,
            image,
            mask_path=perturbed["pericyte_mask_path"],
            h5_dataset_name=perturbed.get("pericyte_mask_h5_dataset_name"),
            max_assignment_distance_um=perturbed.get("pericyte_max_assignment_distance_um"),
            min_pericyte_diameter_um=perturbed.get("pericyte_min_diameter_um"),
            max_pericyte_diameter_um=perturbed.get("pericyte_max_diameter_um"),
            profiles=True,
            junctions=False,
        )
        measured_model = _poiseuille_model_for(perturbed)
        # The baseline's resistances, each branch with pericytes scaled by its
        # measured narrowing (constricted over uniform): whatever else the
        # baseline did to a vessel's resistance stays.
        changed = pericyte_measurement.apply_measured_constriction(G, measured.branches)
        apply_baseline_overrides(G, perturbed, measured_model)
        pericyte_measurement.record_pericyte_sites(G, measured.pericytes)
        result.outputs.append(
            pericyte_measurement.write_branch_csv(
                measured.branches, result.output_dir / f"{spec.name}_pericyte_branches.csv"
            )
        )
        summary.update(measured.summary())
        summary["branches_constricted"] = changed
        logger.info(
            f"Perturbation '{spec.name}': {measured.summary()['placed']} of the mask's "
            f"{len(measured.pericytes)} pericyte(s) on capillaries; {changed} branch(es) "
            "narrowed as measured."
        )
        if _distributes_haematocrit(perturbed):
            measured_config = _haemodynamics_apply_config(
                perturbed, schema,
                voxel_size_zyx=tuple(float(v) for v in G.graph.get("image_voxel_size_zyx", voxel_size_zyx)),
            )

            def recompute_resistances() -> None:
                # The baseline's own resistance computation from the new
                # discharge_haematocrit, then each branch's ratio again at
                # its own haematocrit (the viscosity law reads it).
                apply_poiseuille_resistances(G, measured_config)
                pericyte_measurement.apply_measured_constriction(
                    G,
                    pericyte_measurement.branch_resistances(
                        G, measured.profiles, model=measured_model,
                        diameter_by_branch_order=diameter_table,
                        prefer_edge_fwhm_diameter=prefer_measured,
                        step_um=measured.step_um,
                        reference_diameters=measured.reference_diameters,
                    ),
                )
                apply_baseline_overrides(G, perturbed, measured_model)
    elif is_capillary_block(spec.type):
        block_seeds = _capillary_block_seeds(perturbed, spec.type)
        chosen = None
        if spec.type == "capillary_block_mask":
            chosen, mask_paths = _capillary_blocks_from_mask(
                G, perturbed, image=image, output_dir=result.output_dir, name=spec.name
            )
            result.outputs.extend(mask_paths)
            logger.info(
                f"Perturbation '{spec.name}': {chosen[1]['mask_blocks_on_vessels']} of the "
                f"mask's {chosen[1]['mask_blocks']} block(s) lie on vessels "
                f"(at least {chosen[1]['min_overlap_fraction']:.0%} inside the "
                f"{chosen[1]['vessel_mask_source'].replace('_', ' ')})."
            )
        blocked_edges, block_summary, recompute_resistances = _block_capillaries(
            G, perturbed, schema, seed=block_seeds[0], chosen=chosen
        )
        summary.update(block_summary)
        summary["resistance_factor"] = float(perturbed["capillary_block_resistance_factor"])
        logger.info(
            f"Perturbation '{spec.name}': blocking {len(blocked_edges)} vessel(s) "
            f"({block_summary['selection']}) at "
            f"{summary['resistance_factor']:g}x resistance."
        )
    else:
        # `perturbation_problems` reports an unknown type before a run starts;
        # reaching here means a caller skipped the checks.
        raise ValueError(
            f"perturbation '{spec.name}' has unknown type {spec.type!r}."
        )

    result.graph = G
    result.summary = summary
    if sweep_payload is not None:
        # A sweep solved every grid point on its own copies; G is still the
        # unperturbed baseline, so solving it again would only write the
        # baseline's numbers under this perturbation's name.
        result.outputs.append(Path(sweep_payload["csv_path"]))
        summary["sweep_points"] = len(sweep_payload["results"])
        result.sweep_flows = sweep_payload.get("sweep_flows")
        if sweep_payload.get("haematocrit_distribution") is not None:
            # Each grid point iterated its own haematocrit, as a single
            # re-solve does (see pericyte_sweep.solve_sweep_point).
            summary["haematocrit_distribution"] = sweep_payload["haematocrit_distribution"]
    else:
        solved = _solve_network(
            G, perturbed, boundaries, recompute_resistances=recompute_resistances
        )
        if is_capillary_block(spec.type):
            # The blocked vessels, and every vessel fed only through one,
            # carry the block's leak and nothing else: exactly 0 on this
            # network, so its statistics, its vessels layer and its
            # comparison with the baseline all read the same flows. The
            # pressures stay: the drop across a block is real.
            stopped = haemodynamics.zero_stopped_vessel_flows(
                G, list(boundaries.inlet_nodes), list(boundaries.outlet_nodes)
            )
            logger.info(
                f"Perturbation '{spec.name}': {len(stopped)} vessel(s) carry no flow -- the "
                f"{summary.get('blocked_vessels', 0)} blocked, and those fed only through a "
                "block; the leak the solve left on them is set to 0."
            )
        summary.update(
            {
                "equivalent_resistance": float(solved["equivalent_resistance"]),
                "total_inlet_flow": float(solved["total_inlet_flow"]),
                "total_outlet_flow": float(solved["total_outlet_flow"]),
            }
        )
        if solved.get("haematocrit_distribution") is not None:
            summary["haematocrit_distribution"] = solved["haematocrit_distribution"]
        _write_perturbation_csvs(
            result,
            baseline_graph=model.graph,
            solved=solved,
            baseline=baseline,
            settings=perturbed,
            overrides=overrides,
        )
    if is_capillary_block(spec.type):
        compared_against = model.graph if baseline_graph is None else baseline_graph
        hypoperfusion = float(perturbed["capillary_block_hypoperfusion_fraction"])
        replicates = None
        if len(block_seeds) > 1:
            replicates = _capillary_block_replicates(
                model.graph,
                perturbed,
                schema,
                boundaries,
                seeds=block_seeds[1:],
                baseline_graph=compared_against,
                heartbeat=heartbeat,
                hypoperfusion_fraction=hypoperfusion,
            )
        summary["comparison"] = _write_capillary_block_comparison(
            result,
            baseline_graph=compared_against,
            boundaries=boundaries,
            hypoperfusion_fraction=hypoperfusion,
            selection_summary=summary,
            red_cell_volume_fl=float(perturbed["red_cell_volume_fl"]),
            replicates=replicates,
            first_seed=block_seeds[0],
        )
    if places_pericytes(spec.type) and perturbed.get("use_pericyte_mask_constriction"):
        # Where the mask's pericytes sit, on the baseline's vessels: the
        # strategy above placed them there, and nothing a perturbation
        # changes (diameters, tone) moves a pericyte or a vessel's order.
        distribution, mask_paths = _pericyte_mask_distribution(
            model.graph,
            perturbed,
            output_dir=result.output_dir,
            name=spec.name,
            placed=placed,
        )
        result.outputs.extend(mask_paths)
        summary["mask_distribution"] = distribution.summary

    # Alice-style curves for sweeps; pipeline-like plots/CSVs for a single
    # re-solve. Sweeps keep *graph* for centreline geometry and *sweep_flows*
    # for slider-backed napari layers — not N MultiGraph deepcopies.
    if is_sweep_perturbation(spec.type):
        if sweep_payload is not None:
            result.outputs.extend(
                export_sweep_perturbation_plots(
                    spec.type, sweep_payload["results"], result.output_dir
                )
            )
    else:
        # Everything export_results derives from a solved network is derived
        # again from this one, with the run's own selections: the vascular
        # communities (flow- and resistance-weighted ones move with the
        # flows), then every selected statistic and network analysis -- which
        # also writes each per-vessel analysis onto G for its vessels layer.
        # A perturbation's statistics always run (with the run's own measure
        # selection), so its topological community measure can share the
        # vessels-layer partition; its weighted-community JSONs do not take
        # a precomputed one.
        shared_partition = shared_vascular_community_partition(
            perturbed, G, topology_report=True, weighted_report=False
        )
        if vascular_communities_enabled(perturbed):
            community_summary = graph.assign_vascular_communities(
                G,
                weighting=perturbed["vascular_community_weighting"],
                communities=shared_partition,
            )
            summary["vascular_communities"] = community_summary.community_count
        perturbation_statistics = statistics_arguments(perturbed)
        perturbation_statistics.update(tissue_statistics or {})
        if shared_partition is not None:
            perturbation_statistics["topology_communities"] = shared_partition
        result.outputs.extend(
            export_non_sweep_perturbation_artifacts(
                G,
                result.output_dir,
                perturbed,
                image=image,
                name_stem=spec.name,
                statistics_kwargs=perturbation_statistics,
                voxel_size_zyx=voxel_size_zyx,
                # As the baseline's: only while betweenness or communities are on.
                betweenness_and_communities=bool(perturbed["statistics"])
                and bool(perturbation_statistics["enabled_measures"] & WEIGHTED_MEASURES),
                image_shape=image_shape,
            )
        )

    if sweep_payload is not None:
        outcome = f"{summary['sweep_points']} sweep point(s)"
    else:
        outcome = (
            f"equivalent resistance {summary['equivalent_resistance']:.6g} vs "
            f"baseline {baseline.get('equivalent_resistance', float('nan')):.6g}"
        )
    logger.info(
        f"Perturbation '{spec.name}' ({spec.type}): {outcome}; wrote "
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
    differences = _solved_otherwise(settings, model.graph, boundaries)
    if differences:
        raise ValueError(
            "Perturbations: the network was solved with " + "; ".join(differences)
            + ". Each perturbation re-solves it with this run's settings, so it would be set "
            "against a baseline solved some other way. Run again from the 6. Haemodynamics tab."
        )
    run = PerturbationRun(output_dir=root)
    # On a copy, like every perturbation: `_solve_network` writes pressures and
    # flows onto what it solves, and the number differenced against has to come
    # from the same solver as the perturbations' -- but the run's own graph,
    # which `export_results` is about to write out, must leave this stage
    # exactly as it arrived.
    baseline_graph = _perturbation_copy(model.graph)
    baseline_recompute: Callable[[], None] | None = None
    if _distributes_haematocrit(settings):
        voxel_size_zyx = tuple(
            float(v) for v in baseline_graph.graph.get("image_voxel_size_zyx", (1.0, 1.0, 1.0))
        )
        haemo_config = _haemodynamics_apply_config(
            settings, schema, voxel_size_zyx=voxel_size_zyx,
            stain=recorded_raw_image_stain(settings, baseline_graph),
        )

        def baseline_recompute() -> None:
            apply_poiseuille_resistances(baseline_graph, haemo_config)

    run.baseline = _solve_network(
        baseline_graph, settings, boundaries, recompute_resistances=baseline_recompute
    )
    logger.info(
        f"Perturbations: {len(specs)} to run from a baseline equivalent "
        f"resistance of {run.baseline['equivalent_resistance']:.6g}, into {root}"
    )

    convention = graph.size_convention_of(model.graph)
    if convention is not None:
        narrow = graph.size_class_range("narrow", convention["mid_min_um"], convention["wide_min_um"])
        on_capillaries = [spec.name for spec in specs if "capillary" in spec.type or "pericyte" in spec.type]
        on_arterioles = [spec.name for spec in specs if "arteriole" in spec.type]
        if on_capillaries:
            logger.info(
                f"Size convention: {', '.join(on_capillaries)} act on the vessels {narrow} "
                "wide (the narrow class) where they act on capillaries."
            )
        if on_arterioles:
            logger.warning(
                f"Size convention: {', '.join(on_arterioles)} change arterioles, and a network "
                "classed by size has none, so they change no vessel's diameter."
            )
    image = None if network is None else getattr(network.volume, "image", None)
    voxel_size_zyx = _baseline_voxel_size_zyx(model.graph, network)
    logger.info(f"Perturbations use the baseline image's voxel size (z, y, x): {voxel_size_zyx}")
    # The tissue the baseline's vessel density divides by, so each re-solve's
    # statistics divide by the same one. Only re-solves write statistics.
    tissue_statistics = (
        tissue_statistics_arguments(tissue_measurement(settings, network))
        if any(not is_sweep_perturbation(spec.type) for spec in specs)
        else {}
    )
    # The box the baseline's statistics divide by (export_results'): the
    # image's, or a synthetic network's drawn box on its voxel grid.
    image_shape = _statistics_image_shape(model.graph, image)
    # Once per sweep grid point and block draw, so a run cancelled mid-sweep
    # stops there rather than when the sweep has finished.
    heartbeat = None if progress is None else progress.alive
    for index, spec in enumerate(specs):
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
                    baseline_graph=baseline_graph,
                    voxel_size_zyx=voxel_size_zyx,
                    tissue_statistics=tissue_statistics,
                    image_shape=image_shape,
                    heartbeat=heartbeat,
                )
            )
        except Exception as error:
            if progress is not None:
                # A cancel raised by the heartbeat is the run's, not this
                # perturbation's failure: asked again, it is raised again.
                progress.alive()
            logger.exception(f"Perturbation '{spec.name}' failed: {error}")
            run.results.append(
                PerturbationResult(
                    name=spec.name,
                    type=spec.type,
                    error=f"{type(error).__name__}: {error}",
                )
            )
        if progress is not None:
            # After it ran: a step reports one just finished.
            following = specs[index + 1].name if index + 1 < len(specs) else None
            progress.step(spec.name, total=len(specs), next_step=following)
    return run


def _statistics_image_shape(G: nx.MultiGraph, image: np.ndarray | None) -> tuple[int, ...] | None:
    """The voxel shape `export_results` sizes the baseline's box by: the
    image's, else a synthetic network's (`synthetic_box_shape`); None when
    the graph records no box either."""
    if image is not None:
        return tuple(int(v) for v in image.shape)
    try:
        return synthetic_box_shape(G)
    except ValueError:
        return None


def _solved_otherwise(settings: Mapping[str, Any], G: nx.MultiGraph, boundaries: BoundaryNodes) -> list[str]:
    """What the solve that left *G* used that *settings* do not, one line
    each: its blood model as `build_haemodynamic_model` and `solve` stamped it
    on the graph, its open-end model, and the pressures it held the inlets
    and outlets at. Empty for a run that solved with these settings; a run
    resumed from this tab after a setting upstream changed is not."""
    recorded = G.graph
    differences: list[str] = []
    expected = {
        "viscosity_law": run_viscosity_law(settings),
        "diameter_basis": run_diameter_basis(settings),
        "haematocrit_model": "distributed_iterative" if _distributes_haematocrit(settings) else "fixed",
    }
    for key, value in expected.items():
        if recorded.get(key) not in (None, value):
            differences.append(f"{key} {recorded[key]!r} where these settings give {value!r}")
    if recorded.get("haematocrit") is not None and not np.isclose(
        float(recorded["haematocrit"]), float(settings["haematocrit"]), rtol=1e-9, atol=0.0
    ):
        differences.append(f"haematocrit {recorded['haematocrit']} where these settings give {settings['haematocrit']}")
    open_ends = haemodynamics.open_ends.conditions_on_graph(G)
    if open_ends and open_ends.model != settings.get("open_end_model", open_ends.model):
        differences.append(
            f"open_end_model {open_ends.model!r} where these settings give {settings['open_end_model']!r}"
        )
    for name, nodes in (("inlet_p_bc", boundaries.inlet_nodes), ("outlet_p_bc", boundaries.outlet_nodes)):
        held = [G.nodes[node].get("pressure") for node in nodes if node in G]
        held = [float(value) for value in held if value is not None and np.isfinite(float(value))]
        if held and not np.allclose(held, float(settings[name]), rtol=1e-9, atol=1e-9):
            differences.append(f"{name} {held[0]:g} Pa where these settings give {float(settings[name]):g} Pa")
    return differences


def vascular_communities_enabled(settings: dict) -> bool:
    """Whether a run partitions its network into vascular communities for the
    vessels layer -- its own toggle, independent of Statistics."""
    return bool(settings["compute_vascular_communities"])


def _has_bridges(G: nx.MultiGraph) -> bool:
    """Whether *G* has a zero-resistance bridge into a thick vessel."""
    return any(data.get(graph.IS_ZERO_RESISTANCE) for *_e, data in G.edges(data=True))


def _without_bridges(G: nx.MultiGraph) -> nx.MultiGraph:
    """*G* as the statistics measure it: each zero-resistance bridge's two
    ends one node (``contract_bridges``), or *G* itself when it has none."""
    if not _has_bridges(G):
        return G
    from haemolynx.graph.thick_vessel_junctions import contract_bridges

    return contract_bridges(G)[0]


def shared_vascular_community_partition(
    settings: dict, G: nx.MultiGraph, *, topology_report: bool, weighted_report: bool
) -> "tuple[list, str] | None":
    """The vascular-community partition, computed once here when the
    statistics report computes the very same one -- else ``None``, and
    ``graph.assign_vascular_communities`` partitions on its own.

    *topology_report*: the report's "community" measure runs (it also needs
    ``statistics_network_analysis`` and ``statistics_community``); it is the
    ``topology`` partition -- exact greedy modularity in ``full`` mode,
    ``communities_for_weighting`` in ``fast``, matching
    ``statistics.compute_comprehensive_vessel_statistics`` exactly.
    *weighted_report*: the report's length/resistance/flow-weighted community
    counts run (resistance and flow only with haemodynamics).

    ``None`` too for a network with zero-resistance bridges: the statistics
    partition it with each bridge's two ends one node (:func:`_without_bridges`),
    the vessels layer with the bridges in, and the two are not one partition.
    """
    if not vascular_communities_enabled(settings) or _has_bridges(G):
        return None
    weighting = settings["vascular_community_weighting"]
    if weighting == "topology":
        if not (
            topology_report
            and settings["statistics_network_analysis"]
            and settings["statistics_community"]
        ):
            return None
        if settings["statistics_mode"] == "full":
            simple = nx.Graph(G) if G.is_multigraph() else G
            return graph.communities.greedy_modularity_partition(simple), "greedy_modularity"
        return graph.communities_for_weighting(G, "topology")
    if weighted_report and (weighting == "length" or settings["run_haemodynamics"]):
        return graph.communities_for_weighting(G, weighting)
    return None


def statistics_arguments(settings: dict) -> dict[str, Any]:
    """The run's statistics selection, as keyword arguments to
    `statistics.compute_comprehensive_vessel_statistics`.

    One place, so the baseline's report and every perturbation's are computed
    over the same measures, terminals and analysis settings -- a perturbation
    compared with a baseline that measured something else is no comparison.
    """
    # A network-analysis measure also needs statistics_network_analysis
    # itself on -- the GUI nests it there (see pipeline.schema's own
    # "Connectivity/Network Analysis" section), the same "outer gate
    # checked explicitly, not just implied by requires=" pattern
    # use_edt_diameter_crosscheck's own children use.
    network_analysis_on = settings["statistics_network_analysis"]
    return {
        "statistics_mode": settings["statistics_mode"],
        "enabled_measures": frozenset(
            measure
            for measure in statistics.STATISTIC_MEASURES
            if settings[f"statistics_{measure}"]
            and (measure not in statistics.NETWORK_ANALYSIS_MEASURES or network_analysis_on)
        ),
        "inlet_nodes": settings["inlet_nodes"],
        "outlet_nodes": settings["outlet_nodes"],
        "arteriole_boundary_nodes": settings["arteriole_boundary_nodes"],
        "venule_boundary_nodes": settings["venule_boundary_nodes"],
        # The short-loop rule only runs with mask-supported bridging (see
        # graph_build_arguments), so without it the graph kept every loop.
        "short_loop_removal_um": (
            float(settings.get("remove_short_loops_under_um") or 0.0)
            if settings["bridge_require_mask_support"]
            else 0.0
        ),
        "route_weighting": settings["statistics_route_weighting"],
        "red_cell_volume_fl": float(settings["red_cell_volume_fl"]),
        "shunt_max_route_fraction": float(settings["statistics_shunt_max_route_fraction"]),
        "occlusion_hypoperfusion_fraction": float(
            settings["statistics_occlusion_hypoperfusion_fraction"]
        ),
        "occlusion_curve_max_fraction": float(settings["statistics_occlusion_curve_max_fraction"]),
        "flow_face_margin_um": float(settings.get("statistics_flow_face_margin_um") or 0.0),
    }


#: What a tissue measurement reads besides its raw image and the segmentation.
_TISSUE_SETTINGS: tuple[str, ...] = (
    "tissue_threshold_method",
    "tissue_manual_threshold",
    "tissue_smoothing_sigma_um",
    "tissue_working_voxel_size_um",
    "tissue_min_component_fraction",
    "tissue_fill_holes",
)


def tissue_measurement(
    settings: dict, network: VesselNetwork | None
) -> statistics.TissueVolume | None:
    """The tissue in the run's raw image (``measure_tissue_volume``), or None.

    Measured once per network: the perturbations' statistics and the
    baseline's both divide vessel density by it, and must divide by the same
    tissue. Kept on *network*, and measured again only when the settings, the
    raw file or the segmentation it was measured against changed -- a resumed
    run carries its network over.

    A settings dict that does not name ``measure_tissue_volume`` at all --
    one a caller built by hand with only the keys an export reads -- has it
    off, as it was before the setting existed.
    """
    if network is None or not settings.get("measure_tissue_volume"):
        return None
    if network.volume is None or network.volume.image is None:
        logger.warning("measure_tissue_volume: no loaded segmentation to measure against; skipped.")
        return None
    path, channel, path_name, _channel_name = tissue_raw_image(settings)
    if not path:
        raise ValueError(
            "measure_tissue_volume needs the raw image to find the tissue in: set "
            "raw_image_path, or tissue_raw_tiff_path for another image."
        )
    resolved = io.resolve_image_path_with_optional_zip(Path(path))
    stat = resolved.stat()
    # The raw image is read onto the network's grid by the factor its record
    # holds: recorded as raw_image_downsample says now (the perturbations
    # measure before the export records it), and part of the key.
    _record_downsample(network.graph, settings, network.volume)
    downsample_record = network.graph.graph.get(DOWNSAMPLE_GRAPH_ATTRIBUTE) or {}
    key = (
        str(resolved),
        stat.st_mtime_ns,
        stat.st_size,
        channel,
        io.axis_order_for_run(settings),
        tuple(network.volume.voxel_size_zyx),
        tuple(settings[name] for name in _TISSUE_SETTINGS),
        int(downsample_record.get("raw_factor", 1)),
    )
    segmentation = network.volume.image
    cached = network.tissue
    if cached is not None and cached.key == key and cached.segmentation is segmentation:
        return cached.tissue

    from haemolynx.haemodynamics.fwhm_diameter import load_single_channel_tiff_volume
    from haemolynx.io.load import _to_binary_volume_for_skeletonization

    logger.info(f"Measuring the tissue volume in {resolved} (from {path_name})...")
    use_memmap, memmap_directory = settings["use_memmap_loading"], settings["memmap_directory"]
    whole_raw = load_single_channel_tiff_volume(
        resolved,
        axis_order=io.axis_order_for_run(settings),
        use_memmap=use_memmap,
        memmap_directory=memmap_directory,
        channel=channel,
    )
    # Cut to the crop_on_import box the segmentation was, and reduced onto
    # its grid, when it was.
    raw = raw_volume_like_graph(
        whole_raw, network.graph, description="tissue raw image",
        use_memmap=use_memmap, memmap_directory=memmap_directory,
    )
    if raw is not whole_raw and isinstance(whole_raw, np.memmap):
        preprocessing.release_memmap_array(whole_raw)
    # The segmentation as loaded, which may be labels (an ilastik 1/2 image):
    # its vessels are what skeletonisation called foreground.
    vessel_mask = _to_binary_volume_for_skeletonization(
        segmentation, use_memmap=use_memmap, memmap_directory=memmap_directory
    )
    try:
        tissue = statistics.measure_tissue_volume(
            raw,
            network.volume.voxel_size_zyx,
            vessel_mask=vessel_mask,
            working_voxel_size_um=float(settings["tissue_working_voxel_size_um"]),
            smoothing_sigma_um=float(settings["tissue_smoothing_sigma_um"]),
            threshold_method=settings["tissue_threshold_method"],
            manual_threshold=settings["tissue_manual_threshold"],
            min_component_fraction=float(settings["tissue_min_component_fraction"]),
            fill_holes=bool(settings["tissue_fill_holes"]),
        )
    finally:
        if isinstance(raw, np.memmap):
            preprocessing.release_memmap_array(raw)
        if isinstance(vessel_mask, np.memmap) and vessel_mask is not segmentation:
            preprocessing.release_memmap_array(vessel_mask)
    network.tissue = TissueCache(key=key, segmentation=segmentation, tissue=tissue)
    return tissue


def tissue_statistics_arguments(tissue: statistics.TissueVolume | None) -> dict[str, Any]:
    """*tissue* as keyword arguments to
    `statistics.compute_comprehensive_vessel_statistics`: what vessel density
    divides by. Empty without one, so density falls back to the network's box."""
    if tissue is None:
        return {}
    return {
        "tissue_volume_um3": tissue.volume_um3,
        "vessel_volume_um3": tissue.vessel_volume_um3,
    }


def connected_network_statistics(
    settings: dict,
    G: nx.MultiGraph,
    image_shape: tuple,
    voxel_size_zyx: tuple,
    tissue: statistics.TissueVolume | None,
) -> dict[str, Any] | None:
    """The geometry and topology statistics again, on the inlet/outlet-
    connected network only, as the same flat rows the main report has.

    Only the measures in `statistics.CONNECTED_NETWORK_MEASURES` the run has
    on: the route, flow and boundary-based ones already see only what is
    connected. ``None`` when the run removed disconnected vessels already
    (the main report is that version), when it has no inlet or outlet to
    connect to, or when no measure applies.
    """
    if settings.get("boundary_handling") == BOUNDARY_HANDLING_REMOVE_DISCONNECTED:
        return None
    arguments = statistics_arguments(settings)
    measures = arguments["enabled_measures"] & statistics.CONNECTED_NETWORK_MEASURES
    inlets, outlets = settings["inlet_nodes"], settings["outlet_nodes"]
    if not measures or not inlets or not outlets:
        return None
    connected, _counts = graph.remove_vessels_off_inlet_outlet_paths(G, inlets, outlets)
    if connected.number_of_edges() == 0:
        return None
    arguments["enabled_measures"] = measures
    # The copy's edge dicts are its own, so a per-vessel result a measure
    # writes (loop hierarchy) never lands on the run's graph.
    sections = statistics.compute_vessel_statistics_by_measure(
        connected,
        node_positions=nx.get_node_attributes(connected, "pos"),
        image_dimensions=image_shape,
        voxel_size=voxel_size_zyx,
        **arguments,
        **tissue_statistics_arguments(tissue),
    )
    return statistics.flatten_statistics_sections(sections, settings["statistics_mode"])


def export_tissue_measurement(
    settings: dict, tissue: statistics.TissueVolume, output_dir: Path
) -> list[Path]:
    """Write *tissue*'s measurements (CSV) and surface (``.vtp``, points in
    the ``(z, y, x)`` order the vessels' own ``.vtp`` uses) beside the run's
    other outputs."""
    stem = run_stem(settings)
    csv_path = statistics.export_statistics_to_csv(
        tissue.rows(), output_dir / f"{stem}_tissue_volume.csv"
    )
    surface_path = visualization.surface_mesh_to_vtk(
        tissue.vertices, tissue.faces, output_dir / f"{stem}_tissue_surface.vtp"
    )
    logger.info(f"Saved the tissue volume to: {csv_path}")
    logger.info(f"Saved the tissue surface to: {surface_path}")
    return [csv_path, surface_path]


def export_results(settings: dict, network: VesselNetwork, model: HaemodynamicModel, solution: Solution):
    """Write VTK, statistics and distance measurements, and draw the plots."""
    G = model.graph
    if not graph.network_is_synthetic(settings):
        # The tissue's raw image is read here as raw_image_downsample now says.
        _record_downsample(G, settings, network.volume)
    image = network.volume.image
    # A synthetic network has no image: its box's shape comes from the graph.
    image_shape = tuple(int(v) for v in image.shape) if image is not None else synthetic_box_shape(G)
    output_dir = network.volume.output_dir
    voxel_size_zyx = network.volume.voxel_size_zyx
    main_voxel_size_xyz = network.volume.voxel_size_xyz
    exported: list[str] = []

    # The statistics report (its topological "community" measure, or its
    # weighted-community counts) and the vessels-layer vascular-community
    # assignment (further below) can ask for the exact same partition.
    # Compute it once up front in that case and hand it to both, instead of
    # running greedy modularity on the same graph twice.
    vascular_communities_on = vascular_communities_enabled(settings)
    vascular_weighting = (
        settings["vascular_community_weighting"]
        if vascular_communities_on
        else None
    )
    shared_vascular_communities = shared_vascular_community_partition(
        settings,
        G,
        topology_report=bool(settings["statistics"]),
        weighted_report=bool(settings["statistics"]),
    )

    # 6b) Optional: the tissue's own volume and surface, from the raw image --
    # what vessel density divides by when it is measured.
    tissue = tissue_measurement(settings, network)
    if tissue is not None:
        export_tissue_measurement(settings, tissue, output_dir)
    else:
        logger.info("Tissue volume measurement skipped.")
    solution.tissue = tissue

    # What the panel's Statistics tab shows; only what this export measures.
    G.graph.pop(STATISTICS_REPORT, None)
    report_sections: dict[str, Any] = {}
    stats_csv_path = None

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
        measures = statistics.compute_vessel_statistics_by_measure(
            G,
            node_positions=node_positions,
            image_dimensions=image_shape,
            # Zipped element-wise with image.shape, which is (z, y, x), so this
            # is the per-array-axis spacing. Left at its (1, 1, 1) default the
            # whole-image volume and density were counted in voxels for any
            # stack whose z spacing is not 1 um.
            voxel_size=voxel_size_zyx,
            topology_communities=(
                shared_vascular_communities if vascular_weighting == "topology" else None
            ),
            **statistics_arguments(settings),
            **tissue_statistics_arguments(tissue),
        )
        stats = statistics.flatten_statistics_sections(measures, settings["statistics_mode"])
        report_sections.update(measures)

        resistances = resistance_statistics(solution)
        stats.update(resistances)
        if resistances:
            report_sections[HAEMODYNAMICS_SECTION] = resistances["Haemodynamics"]

        logger.info("=== Statistics ===")
        for key, value in stats.items():
            logger.info(f"  {key}: {value}")

        stats_csv_path = output_dir / f"{run_stem(settings)}_statistics.csv"
        statistics.export_statistics_to_csv(stats, stats_csv_path)
        exported.append("statistics")
        logger.info(f"Saved statistics CSV to: {stats_csv_path}")
        connected_stats = connected_network_statistics(
            settings, G, image_shape, voxel_size_zyx, tissue
        )
        if connected_stats is not None:
            connected_csv_path = (
                output_dir / f"{run_stem(settings)}_statistics_connected.csv"
            )
            statistics.export_statistics_to_csv(connected_stats, connected_csv_path)
            logger.info(
                "Saved the statistics of the inlet/outlet-connected network "
                f"(disconnected vessels left out) to: {connected_csv_path}"
            )
        beds_csv_path = statistics.export_capillary_bed_joins_csv(
            G, output_dir / f"{run_stem(settings)}_capillary_bed_joins.csv"
        )
        if beds_csv_path is not None:
            logger.info(f"Saved capillary bed joins CSV to: {beds_csv_path}")

        branch_stats = statistics.compute_branch_order_statistics(
            G,
            node_positions=node_positions,
        )
        branch_stats_csv_path = output_dir / f"{run_stem(settings)}_branch_statistics.csv"
        statistics.export_branch_order_statistics_to_csv(
            branch_stats,
            branch_stats_csv_path,
        )
        logger.info(f"Saved branch-order statistics CSV to: {branch_stats_csv_path}")
        report_sections[BRANCH_ORDER_SECTION] = branch_stats

        if not statistics_arguments(settings)["enabled_measures"] & WEIGHTED_MEASURES:
            weighted_measurements = None
            logger.info("Weighted betweenness and communities skipped: betweenness and communities are off.")
        elif settings["run_haemodynamics"]:
            precomputed_communities = (
                {vascular_weighting: shared_vascular_communities}
                if shared_vascular_communities is not None and vascular_weighting != "topology"
                else None
            )
            weighted_measurements = statistics.compute_betweenness_and_community_measurements(
                _without_bridges(G), precomputed_communities=precomputed_communities,
            )
        else:
            # Resistance and solved |flow| both require haemodynamics; only
            # length is purely geometric and stays computable either way.
            def _na_weighted_model() -> dict:
                na = "N/A (haemodynamics disabled)"
                return {
                    "Betweenness": {
                        "Betweenness Mean": na,
                        "Betweenness Max": na,
                        "Betweenness Top Nodes": na,
                        "Betweenness Method": na,
                    },
                    "Communities": {
                        "Community Count": na,
                        "Largest Community Size": na,
                        "Mean Community Size": na,
                        "Community Method": na,
                    },
                }

            measured = _without_bridges(G)
            weighted_measurements = {
                "edge_resistance": _na_weighted_model(),
                "edge_flow_abs": _na_weighted_model(),
                "edge_length": {
                    "Betweenness": statistics.compute_weighted_betweenness_summary(
                        measured,
                        source_attr="length",
                        inverse_source_attr=False,
                    ),
                    "Communities": statistics.compute_weighted_communities_summary(
                        measured,
                        source_attr="length",
                        inverse_source_attr=False,
                        precomputed=(
                            shared_vascular_communities
                            if vascular_weighting == "length"
                            else None
                        ),
                    ),
                },
            }
        if weighted_measurements is not None:
            logger.info("=== Weighted Betweenness and Communities ===")
            for model_name, model_results in weighted_measurements.items():
                logger.info(f"  [{model_name}]")
                for metric_name, metric_values in model_results.items():
                    logger.info(f"    {metric_name}: {metric_values}")

            resistance_path = output_dir / f"{run_stem(settings)}_betweenness_communities_resistance.json"
            resistance_path.write_text(
                json.dumps(weighted_measurements["edge_resistance"], indent=2)
            )
            length_path = output_dir / f"{run_stem(settings)}_betweenness_communities_edge_length.json"
            length_path.write_text(
                json.dumps(weighted_measurements["edge_length"], indent=2)
            )
            flow_path = output_dir / f"{run_stem(settings)}_betweenness_communities_edge_flow.json"
            flow_path.write_text(
                json.dumps(weighted_measurements["edge_flow_abs"], indent=2)
            )
            logger.info(f"Saved edge-resistance stats to: {resistance_path}")
            logger.info(f"Saved edge-length stats to: {length_path}")
            logger.info(f"Saved edge-flow stats to: {flow_path}")
            report_sections[WEIGHTED_MEASURES_SECTION] = weighted_measurements
        if graph.network_is_synthetic(settings):
            # How close the synthetic network came to its template, on the
            # statistics just measured: a CSV beside them and a section of
            # the report.
            from haemolynx import synthetic

            source = settings.get("synthetic_template_source", "statistics")
            template = synthetic.load_template(settings["synthetic_template_path"], source)
            # A batch template's runs, each at the average's box: the range
            # each characteristic is reported within beside its band (B4b).
            members = (
                synthetic.batch_members_in_common_box(settings["synthetic_template_path"], source)
                if synthetic.is_batch_template(settings["synthetic_template_path"])
                else ()
            )
            rows = synthetic.compare_run(
                template, G, roles=synthetic.boundary_nodes(G), sections=report_sections,
                image_shape=image_shape, voxel_size_zyx=voxel_size_zyx, members=members,
            )
            comparison_path = synthetic.write_template_comparison_csv(
                rows, output_dir / f"{run_stem(settings)}_template_comparison.csv"
            )
            report_sections[synthetic.COMPARISON_SECTION] = synthetic.comparison_section(rows)
            summary = synthetic.comparison_summary(rows)
            logger.info(
                "Synthetic network against its template: %d of %d banded characteristics within "
                "tolerance (%s); saved to %s",
                summary["within"], summary["fitted"],
                ", ".join(summary["outside"]) if summary["outside"] else "none outside",
                comparison_path,
            )
            if summary["ranged"]:
                logger.info(
                    "Synthetic network against its batch template's %d runs: %d of %d characteristics "
                    "within their range.", len(members), summary["within_members"], summary["ranged"],
                )
    else:
        logger.info("Vessel statistics skipped.")
    if tissue is not None:
        report_sections[TISSUE_VOLUME_SECTION] = tissue.rows()
    if report_sections:
        G.graph[STATISTICS_REPORT] = {
            "mode": settings["statistics_mode"] if settings["statistics"] else None,
            "csv": None if stats_csv_path is None else str(stats_csv_path),
            "sections": report_sections,
        }

    # 7b) Optional: partition the network into vascular communities/domains
    # for colouring in the vessels layer -- independent of the statistics
    # export above, since this writes a graph attribute for visualization,
    # not a report the user may have turned off. Reuses the partition
    # already computed above when the statistics report asked for the same
    # one (see shared_vascular_community_partition).
    if vascular_communities_on:
        community_summary = graph.assign_vascular_communities(
            G, weighting=vascular_weighting,
            communities=shared_vascular_communities,
        )
        logger.info(
            f"Assigned {community_summary.community_count} vascular "
            f"communities ({community_summary.boundary_edge_count} boundary "
            f"edge(s)) via {community_summary.method}."
        )

    # 8) Optional: the gap from each object in a cell mask to the vessels --
    # the run's own segmentation unless a vessel mask of its own is given.
    if settings["measurement_3d_to_cell_mask"]:
        if settings["cell_mask_path"] is None:
            raise ValueError(
                "measurement_3d_to_cell_mask=True requires cell_mask_path."
            )
        if image is None:
            raise ValueError(
                "measurement_3d_to_cell_mask needs the run's segmented image to measure "
                "to, and a synthetic network has none: switch it off (preflight says so)."
            )
        distance_summary = statistics.run_3d_measurement_to_cell_mask(
            graph=G,
            cell_mask_path=Path(settings["cell_mask_path"]),
            output_dir=output_dir,
            image_stem=run_stem(settings),
            voxel_size_xyz=main_voxel_size_xyz,
            axis_order=io.axis_order_for_run(settings),
            vessel_mask_path=(
                None
                if settings["measurement_3d_vessel_mask_path"] is None
                else Path(settings["measurement_3d_vessel_mask_path"])
            ),
            vessel_mask=image,
            cell_mask_h5_dataset_name=settings["cell_mask_h5_dataset_name"],
            vessel_mask_h5_dataset_name=settings["measurement_3d_vessel_mask_h5_dataset_name"],
            voxel_size_override_xyz=settings.get("voxel_size_override_xyz"),
            voxel_size_policy=settings.get("voxel_size_policy", "auto"),
            use_memmap=settings["use_memmap_loading"],
            memmap_directory=settings["memmap_directory"],
        )
        logger.info(
            "3D cell-mask vessel-distance summary: "
            f"{distance_summary}"
        )
    else:
        logger.info("3D cell-mask vessel-distance measurement skipped.")

    # 8b) Optional: around each pericyte of a pericyte mask -- its vessel's
    # diameter along it and its branch's resistance with that narrowing, and
    # how far it is from its vessel's nearest junction.
    if settings.get("measure_pericytes"):
        if image is None:
            raise ValueError(
                "measure_pericytes needs the run's segmented image, and a synthetic network "
                "has none: switch it off (preflight says so)."
            )
        if settings.get("pericyte_measurement_mask_path") in (None, ""):
            raise ValueError("measure_pericytes=True requires pericyte_measurement_mask_path.")
        measured = measure_around_pericytes(
            settings,
            G,
            image,
            mask_path=Path(settings["pericyte_measurement_mask_path"]),
            h5_dataset_name=settings.get("pericyte_measurement_mask_h5_dataset_name"),
            max_assignment_distance_um=settings.get("pericyte_measurement_max_distance_um"),
            min_pericyte_diameter_um=settings.get("pericyte_measurement_min_diameter_um"),
            max_pericyte_diameter_um=settings.get("pericyte_measurement_max_diameter_um"),
            profiles=bool(settings.get("measure_pericyte_diameter_profiles", True)),
            junctions=bool(settings.get("measure_pericyte_junction_distances", True)),
        )
        written = pericyte_measurement.write_pericyte_measurements(measured, output_dir, run_stem(settings))
        found = measured.summary()
        logger.info(
            f"Pericyte measurements: {found['placed']} of the mask's {found['pericytes']} "
            f"pericyte(s) on capillaries; {found['profiles']} diameter profile(s), "
            f"{found['branches']} branch resistance(s), {found['junction_distances']} junction "
            f"distance(s). Saved {len(written)} file(s), the summary to: {written[0]}"
        )
    else:
        logger.info("Pericyte measurements skipped.")

    # 10) Export vessels/pericytes/nodes to VTK and optionally visualize in PyVista.
    # The pericytes file is the periodic strategy's sites at the run's own
    # spacing and length (derive_pericyte_points_from_graph), not a mask's.
    # Geometry and diameters need no solve; flows are written once there are some.
    if settings["vtk_export"]:
        vtk_export = visualization.graph_to_vtk(
            G, output_dir / run_stem(settings),
            constriction_spacing=float(settings.get("constriction_spacing_um", 100.0)),
            constriction_length=float(settings.get("constriction_length_um", 40.0)),
            inlet_nodes=settings["inlet_nodes"], outlet_nodes=settings["outlet_nodes"],
        )
        exported.append("VTK")
        logger.info("=== VTK Export ===")
        logger.info(f"  Vessels:   {vtk_export['vessels_path']}")
        logger.info(f"  Pericytes: {vtk_export['pericytes_path']}")
        logger.info(f"  Nodes:     {vtk_export['nodes_path']}")
        logger.info(f"  Counts: vessels={vtk_export['vessel_line_count']}, "
          f"pericytes={vtk_export['pericyte_count']}, nodes={vtk_export['node_count']}")
        if settings["visualize_vtk"]:
            visualization.visualize_vtk_network(
                vtk_export["vessels_path"],
                vtk_export["pericytes_path"],
                vtk_export["nodes_path"],
                show_nodes=False,
            )
        else:
            logger.info("VTK visualization skipped.")
    elif settings["visualize_vtk"]:
        logger.warning(
            "VTK visualization requested but VTK export is disabled. "
            "Set vtk_export: true in the config to enable."
        )
    if settings["run_haemodynamics"] and graph.network_surrounds_image(settings) and run_stem(settings) is not None:
        # Each of the image's vessels: its flow in the template run and inside the surround.
        from haemolynx.synthetic.surround import image_flow_rows, write_image_flows_csv

        flows_path = write_image_flows_csv(G, output_dir / f"{run_stem(settings)}_image_flows.csv")
        rows = image_flow_rows(G)
        flipped = sum(1 for row in rows if row["direction_flipped"])
        compared = sum(1 for row in rows if row["direction_flipped"] is not None)
        logger.info(
            "Surround: %d of the image's %d vessels compared with the template run's flow, %d running the "
            "other way now; saved to %s", compared, len(rows), flipped, flows_path,
        )
    # 11) Optional matplotlib visualization.
    if settings["visualize_results"] and image is None:
        logger.info("Plots skipped: a synthetic network has no image to overlay on.")
    elif settings["visualize_results"]:
        logger.info("Generating visualizations...")
        # A run resumed here skipped the Skeletonise stage that made it.
        Path(settings["plot_dir"]).mkdir(parents=True, exist_ok=True)
        exported.append("plots")
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
            final_graph_3d_path = Path(settings["plot_dir"]) / "final_graph_3d.html"
            _write_run_final_graph_3d_html(
                settings,
                G,
                large_arteriole_mask=network.large_arteriole_mask,
                large_venule_mask=network.large_venule_mask,
                small_arteriole_mask=network.small_arteriole_mask,
                small_venule_mask=network.small_venule_mask,
                voxel_size_zyx=voxel_size_zyx,
                show=False,
            )
            logger.info(f"Saved interactive 3D final graph to: {final_graph_3d_path}")
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
            # Never off the main thread (a panel run): nothing could close them.
            and visualization.plot.backend_can_display()
            and plt.get_fignums()
        ):
            logger.info("Holding plot windows open. Close them to finish the script.")
            plt.show(block=True)
    else:
        logger.info("Matplotlib visualizations skipped.")

    # 12) Optional: a plain-text reading list of what this run actually used.
    if settings["export_citations"]:
        citations_path = output_dir / f"{run_stem(settings)}_citations.txt"
        write_citations(settings, citations_path, network_nodes=G.number_of_nodes())
        exported.append("citations")
        logger.info(f"Saved citations to: {citations_path}")
    else:
        logger.info("Citation export skipped.")

    solution.exported = tuple(exported)
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


def _stage_index(name: str) -> int:
    return STAGE_CALLS.index(name)


def _run_stage_body(name: str, start_from: str | None) -> bool:
    """Whether to call the stage function, or reconstruct from a resume.

    ``segment``, ``skeletonise`` and ``build_network`` always run: the last
    two load from disk when their skip toggles are off. Later stages are
    skipped when ``start_from`` is after them.
    """
    if start_from is None:
        return True
    if name not in STAGE_CALLS or start_from not in STAGE_CALLS:
        return True
    if name in {"segment", "skeletonise", "build_network"}:
        return True
    return _stage_index(name) >= _stage_index(start_from)


#: Where a graph keeps :func:`input_volume_settings` as it was built
#: (``G.graph[INPUT_SETTINGS_RECORD]``), for a run resumed on it.
INPUT_SETTINGS_RECORD = "input_settings"


def input_volume_settings(settings: Mapping[str, Any]) -> dict[str, Any]:
    """The Input tab's settings that decide the volume, and so the network,
    a run builds on, as plain values: the image read (ilastik's output,
    once `segment` has made it), its axis order and voxel size, the crop
    box, the downsample and the clean-up -- or, for a synthetic network,
    the template and every synthetic_ setting. What a run resumed on a
    stored volume or network must still hold (:func:`_check_resumed_input`)."""

    def plain(value: Any) -> Any:
        if isinstance(value, PurePath):
            return str(value)
        if isinstance(value, (list, tuple)):
            return [plain(item) for item in value]
        return value

    synthetic = graph.network_is_synthetic(settings)
    if synthetic:
        names = {"network_source", *(name for name in settings if name.startswith("synthetic_"))}
    else:
        names = {
            "network_source", "image_axis_order", "voxel_size_policy",
            "segmented_image_downsample", "crop_on_import", "segmentation_cleanup",
        }
        if str(settings.get("voxel_size_policy", "auto")) != "metadata_only":
            names.add("voxel_size_override_xyz")
        if settings.get("crop_on_import"):
            names.update(name for name in settings if name.startswith("crop_"))
        if settings.get("segmentation_cleanup"):
            names.update(name for name in settings if name.startswith("segmentation_cleanup_"))
    recorded = {name: plain(settings.get(name)) for name in sorted(names)}
    if not synthetic:
        image = segmented_input_path(dict(settings))
        recorded["input_path"] = None if image is None else str(Path(image).resolve())
    return recorded


def _check_resumed_input(settings: Mapping[str, Any], recorded: Any, what: str) -> None:
    """Refuse a run resumed on a stored *what* (a volume or a network) made
    with other Input settings than these: it was built on the old volume
    without a word, and its manifest and settings file named the new ones.
    Nothing to compare for one made before this was recorded."""
    if not isinstance(recorded, Mapping):
        return
    now = input_volume_settings(settings)
    changed = sorted(name for name in set(recorded) | set(now) if recorded.get(name) != now.get(name))
    if changed:
        raise ValueError(
            f"The Input tab's {', '.join(changed)} changed since this run's {what} was made, "
            "and a run resumed on it would not use the new values. Run the pipeline from the "
            "Input tab to use them, or put them back to resume."
        )


class ImageNotReadError(RuntimeError):
    """A stage reached for the voxels of an :class:`ImageNotRead`."""


class ImageNotRead:
    """Stands in for the segmented image, or the skeleton made from it, of a
    run resumed on a saved network without reading either: no stage the run
    does reads them (:func:`resumed_image_readers`), and the file may not be
    on this machine at all. It has the image's shape, which the network
    records (:func:`recorded_image_shape`); anything that reads its voxels
    raises, naming the file."""

    def __init__(self, what: str, path: Any, shape: Sequence[int]):
        self.what = what
        self.path = path
        self.shape = tuple(int(n) for n in shape)
        self.ndim = len(self.shape)

    def _refuse(self):
        raise ImageNotReadError(
            f"This run resumed on the saved network without reading the {self.what} "
            f"({self.path}): no stage it runs was to read it, yet one did. Run it from an "
            "earlier tab with the image at that path."
        )

    def __array__(self, *_args, **_kwargs):
        self._refuse()

    def __getitem__(self, _key):
        self._refuse()

    def __iter__(self):
        self._refuse()

    def __len__(self):
        self._refuse()

    def __getattr__(self, name: str):
        # What an array has and this lacks (dtype, max, ...); private and
        # special names stay unknown, for copy and pickle.
        if name.startswith("_"):
            raise AttributeError(name)
        self._refuse()

    def __repr__(self) -> str:
        return f"ImageNotRead({self.what!r}, {str(self.path)!r}, shape={self.shape})"


def volume_holds_image(volume: Any) -> bool:
    """Whether *volume* (a stage's output) holds the image it names: False
    for the :class:`ImageNotRead` a run resumed without it carries."""
    return not isinstance(getattr(volume, "image", None), ImageNotRead)


def recorded_image_shape(G: nx.Graph) -> tuple[int, ...] | None:
    """The shape of the image *G* was built on, from the extent and voxel
    size graph building records on it (the inverse of
    ``graph.dead_ends.image_extent_um``); None when it records neither."""
    extent = graph.dead_ends.graph_image_extent_um(G)
    voxel = G.graph.get("image_voxel_size_zyx") or G.graph.get("voxel_size")
    if extent is None or voxel is None or len(voxel) != len(extent):
        return None
    return tuple(int(round(float(e) / float(v))) + 1 for e, v in zip(extent, voxel))


#: The export's settings that read the segmented image, and what each does.
_EXPORT_IMAGE_READERS = (
    ("measure_tissue_volume", "the tissue volume (measure_tissue_volume), measured against it"),
    ("measurement_3d_to_cell_mask", "the 3D distances to the cell mask (measurement_3d_to_cell_mask)"),
    ("measure_pericytes", "the pericyte measurements (measure_pericytes), read off it"),
    ("visualize_results", "the plots (visualize_results), drawn over it"),
)


def resumed_image_readers(
    settings: Mapping[str, Any], start_from: str | None, resumed_graph: Any = None
) -> tuple[str, ...]:
    """What reads the segmented image, or the skeleton made from it, in a run
    resumed at *start_from* on *resumed_graph* (the saved run's network): a
    phrase each, empty when nothing does. A resumed run handed no volume
    (one on a run loaded from a file) then goes on from that network without
    loading either (:class:`ImageNotRead` stands in), and preflight does not
    ask for the file: the image need not be on this machine.

    Every stage up to Diameters reads it, so a run starting at one does, as
    does one with no network to resume on, or a network that does not record
    the image's shape. After Diameters: the network handling, when
    ``remove_disconnected`` measures the vessels it joins from the image
    (mask or raw-image diameters, or a network classed by size); Post
    processing, with hand edits to bring in line; every perturbation but a
    sweep, whose plots are drawn over the image; and the export settings in
    :data:`_EXPORT_IMAGE_READERS`. A synthetic network has no image.
    """
    if graph.network_is_synthetic(settings):
        return ()
    if start_from not in STAGE_CALLS or _stage_index(start_from) <= _stage_index("assign_diameters"):
        return ("the stages up to Diameters, which build on it",)
    if resumed_graph is None:
        return ("graph building, with no saved network to resume on",)
    if recorded_image_shape(resumed_graph) is None:
        return ("the image's shape, which the saved network does not record",)

    def runs(stage: str) -> bool:
        return _stage_index(stage) >= _stage_index(start_from)

    readers: list[str] = []
    measures_joins = graph.size_convention_of(resumed_graph) is not None or (
        bool(settings.get("run_haemodynamics"))
        and bool(settings.get("use_edt_diameter_crosscheck") or measures_diameters_from_raw_image(settings))
    )
    if (
        runs("build_haemodynamic_model")
        and settings.get("boundary_handling") == BOUNDARY_HANDLING_REMOVE_DISCONNECTED
        and measures_joins
    ):
        readers.append("the network handling (remove_disconnected), measuring the vessels it joins")
    if runs("post_process") and graph.has_pending_edits(resumed_graph):
        readers.append("Post processing, bringing the hand-edited vessels in line")
    if runs("run_perturbations") and settings.get("run_perturbations") and settings.get("run_haemodynamics"):
        readers.extend(
            f"perturbation '{spec.name}', whose plots are drawn over it"
            for spec in perturbations_from_settings(settings)
            if spec.type != "none" and not is_sweep_perturbation(spec.type)
        )
    readers.extend(what for name, what in _EXPORT_IMAGE_READERS if settings.get(name))
    return tuple(readers)


def _resumes_without_image(
    settings: Mapping[str, Any], resume: PipelineResume | None, start_from: str | None
) -> bool:
    """Whether a run resumed on *resume* goes on without loading the
    segmented image and its skeleton: it is handed no volume of this
    session's to use, no stage it runs reads them
    (:func:`resumed_image_readers`), and the settings name the image, as
    the network's record of them does."""
    if resume is None or graph.network_is_synthetic(settings):
        return False
    if _earlier_output(resume, start_from, "skeletonise", "volume") is not None:
        return False
    if segmented_input_path(dict(settings)) is None:
        return False
    resumed_graph = _earlier_output(resume, start_from, "build_network", "graph")
    return resumed_graph is not None and not resumed_image_readers(settings, start_from, resumed_graph)


def _segmented_inputs_unread(settings: dict) -> SegmentedInputs:
    """What :func:`segment` settles, for a run resumed without reading the
    segmented image: the image named as the run names it
    (:func:`segmented_input_path`) and the settings the later stages read
    normalised as `segment` leaves them. Nothing is read, unzipped or
    segmented."""
    settings["input_path"] = Path(segmented_input_path(settings))
    settings["image_axis_order"] = io.normalize_axis_order(
        settings["image_axis_order"], label="IMAGE_AXIS_ORDER", allow_auto=True
    )
    settings["output_dir"] = Path(settings["output_dir"])
    logger.info(
        "Resuming on the saved network: no stage this run does reads the segmented image "
        f"or its skeleton, so neither is loaded ({settings['input_path']})."
    )
    return SegmentedInputs(
        image_path=settings["input_path"],
        output_dir=settings["output_dir"],
        input_format=settings["input_path"].suffix[1:].lower(),
    )


def _volume_not_read(
    settings: Mapping[str, Any], inputs: SegmentedInputs, resume: PipelineResume, start_from: str
) -> SkeletonisedVolume:
    """The volume of a run resumed without reading the image: its image and
    skeleton :class:`ImageNotRead`, the rest as the saved network records it
    (voxel size, crop box, downsample) or *resume* carries it (the thick
    region)."""
    G = _earlier_output(resume, start_from, "build_network", "graph")
    shape = recorded_image_shape(G)
    voxel_size_zyx = tuple(float(v) for v in (G.graph.get("image_voxel_size_zyx") or G.graph["voxel_size"]))
    voxel_size_xyz = G.graph.get("image_voxel_size_xyz") or tuple(reversed(voxel_size_zyx))
    crop = G.graph.get(CROP_GRAPH_ATTRIBUTE)
    return SkeletonisedVolume(
        image=ImageNotRead("segmented image", inputs.image_path, shape),
        skeleton=ImageNotRead("skeleton", inputs.output_dir / f"{run_stem(settings)}_skeleton.npy", shape),
        voxel_size_xyz=tuple(float(v) for v in voxel_size_xyz),
        voxel_size_zyx=voxel_size_zyx,
        output_dir=inputs.output_dir,
        thick_vessel_mask=resume.thick_vessel_mask,
        crop=None if not crop else CropBox.from_record(crop),
        downsample=graph_downsample(G),
        input_settings=input_volume_settings(settings),
    )


def _earlier_output(
    resume: PipelineResume | None, start_from: str | None, stage: str, attribute: str
) -> Any | None:
    """*resume*'s output for *stage*, when the run starts after that stage.
    Not a volume that only stands in for its image (:class:`ImageNotRead`),
    nor a network on one: this run may read the image."""
    if resume is None or start_from not in STAGE_CALLS:
        return None
    if _stage_index(stage) >= _stage_index(start_from):
        return None
    output = getattr(resume, attribute, None)
    return output if volume_holds_image(getattr(output, "volume", output)) else None


def _resumed_segmented_path(
    settings: dict, resume: PipelineResume | None, start_from: str | None
) -> Path | None:
    """The segmented image a run resumed past segmentation already has.

    The resumed run's own, while it is still the image these settings
    read -- else, when ilastik segments and its current output is on disk
    (a run loaded from a file carries no stage outputs), that output, so
    resuming never runs ilastik again. None: segment as usual. The resumed
    run's own image was kept though input_path, or the raw image ilastik
    segments, had changed since, and the Skeletonise tab built on the old
    one while every other Input change took effect.
    """
    if resume is None or start_from not in STAGE_CALLS:
        return None
    current = current_ilastik_output(settings)[0] if segments_raw_image(settings) else None
    inputs = _earlier_output(resume, start_from, "segment", "inputs")
    if inputs is not None:
        if segments_raw_image(settings):
            wanted = current
        elif graph.network_is_synthetic(settings) or settings.get("input_path") in (None, ""):
            wanted = inputs.image_path
        else:
            wanted = io.resolve_image_path_with_optional_zip(Path(settings["input_path"]))
        if wanted is not None and Path(wanted).resolve() == Path(inputs.image_path).resolve():
            return inputs.image_path
        logger.info(
            "The segmented image is not the one the resumed run read (%s): it is read again.",
            inputs.image_path,
        )
        return current
    if _stage_index(start_from) <= _stage_index("segment"):
        return None
    # Only an output its record vouches for: one left without a record (a
    # binarisation that raised) was taken as this run's.
    return current


def _volume_writing_to(volume: SkeletonisedVolume, output_dir: Path) -> SkeletonisedVolume:
    """*volume*, its outputs going to *output_dir*: the run's folder now.

    A resumed volume names the folder of the run it came from; a run resumed
    with another ``output_dir`` wrote its VTK and manifest to the new folder
    and its statistics, citations and the rest into the old one.
    """
    if Path(volume.output_dir) == Path(output_dir):
        return volume
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    return replace(volume, output_dir=Path(output_dir))


def _fill_boundary_settings(settings: dict, resume: PipelineResume) -> None:
    """Write resume node lists into the mutable settings the later stages read."""
    mapping = {
        "inlet_nodes": resume.inlet_nodes,
        "outlet_nodes": resume.outlet_nodes,
        "arteriole_boundary_nodes": resume.arteriole_boundary_nodes,
        "venule_boundary_nodes": resume.venule_boundary_nodes,
        "large_arteriole_boundary_nodes": resume.large_arteriole_boundary_nodes,
        "large_venule_boundary_nodes": resume.large_venule_boundary_nodes,
    }
    for name, values in mapping.items():
        current = settings.get(name)
        if isinstance(current, list):
            current[:] = list(values)
        else:
            settings[name] = list(values)


def _boundaries_from_resume(
    resume: PipelineResume, network: VesselNetwork
) -> BoundaryNodes:
    graph = resume.graph if resume.graph is not None else network.graph
    pair = resume.resistance_node_pair
    if pair is None and resume.inlet_nodes and resume.outlet_nodes:
        pair = (resume.inlet_nodes[0], resume.outlet_nodes[0])
    # network.large_arteriole_mask/large_venule_mask are the raw masks this
    # run's own build_network just loaded -- never overlap-cleaned, since
    # assign_boundaries's body (the only stage that cleans them) is skipped
    # on this path. Prefer the cleaned masks the resume payload carried from
    # the original run so the exported overlay matches what actually
    # determined the resumed graph's cut and terminal-node assignment.
    large_arteriole_mask = (
        resume.large_arteriole_mask
        if resume.large_arteriole_mask is not None
        else network.large_arteriole_mask
    )
    large_venule_mask = (
        resume.large_venule_mask
        if resume.large_venule_mask is not None
        else network.large_venule_mask
    )
    return BoundaryNodes(
        inlet_nodes=list(resume.inlet_nodes),
        outlet_nodes=list(resume.outlet_nodes),
        arteriole_boundary_nodes=list(resume.arteriole_boundary_nodes),
        venule_boundary_nodes=list(resume.venule_boundary_nodes),
        large_arteriole_boundary_nodes=list(resume.large_arteriole_boundary_nodes),
        large_venule_boundary_nodes=list(resume.large_venule_boundary_nodes),
        resistance_node_pair=pair,
        graph=graph,
        large_arteriole_mask=large_arteriole_mask,
        large_venule_mask=large_venule_mask,
        # Only the resumed run's: the network's own are the masks as loaded,
        # not what the labelling read.
        small_arteriole_mask=resume.small_arteriole_mask,
        small_venule_mask=resume.small_venule_mask,
    )


def _vessel_masks_now_in_use(settings: dict, network: VesselNetwork) -> VesselNetwork:
    """A resumed *network* with the vessel masks *settings* now ask for.

    The masks load in ``build_network`` but their switches sit on the
    Boundaries tab, so a run resumed there carries the masks of the run it
    resumes: a role switched off since is dropped, and one switched on since
    is loaded, so the boundaries are assigned -- and the viewer draws -- what
    the Boundaries tab now says. Masks split by width (``split_large_vessel_masks_by_diameter``)
    at another width than now asked, or split where they now are not, or not
    where they now are, are loaded and cut again; so are a role's masks whose
    files, load-time clean-up or global reading changed (`_vessel_mask_sources`).
    They are placed on the network's grid as ``build_network`` places them:
    cut to its crop box, reduced by its downsample, one global mask each when
    the run reads them so.
    """
    volume = network.volume
    changes: dict[str, Any] = {}
    split = graph.large_vessel_mask_split_diameter_um(settings)
    cut_otherwise = split != network.mask_split_diameter_um
    if cut_otherwise:
        changes["mask_split_diameter_um"] = split
    sources = _vessel_mask_sources(settings)
    on_disk = dict(
        use_memmap=bool(settings.get("use_memmap_loading", False)),
        memmap_directory=settings.get("memmap_directory"),
    )

    def as_read(arteriole, venule):
        if io.vessel_masks_are_global(settings):
            return io.as_global_vessel_mask(arteriole, venule, **on_disk), None
        return arteriole, venule

    for role in graph.boundaries.VESSEL_MASK_SWITCHES:
        arteriole, venule = f"{role}_arteriole_mask", f"{role}_venule_mask"
        held = getattr(network, arteriole) is not None or getattr(network, venule) is not None
        if not graph.vessel_masks_in_use(settings, role):
            if held:
                logger.info(f"The {role}-vessel masks are off now: not using the resumed run's.")
                changes[arteriole] = changes[venule] = None
                if role == "large":
                    network.graph.graph.pop("large_arteriole_mask_voxel_size_xyz", None)
                    network.graph.graph.pop("large_venule_mask_voxel_size_xyz", None)
            continue
        # Without a record (a network not built by build_network) the masks
        # it holds are taken to be the ones asked for.
        read_otherwise = network.mask_sources is not None and (
            network.mask_sources.get(role) != sources[role]
        )
        if held and not cut_otherwise and not read_otherwise:
            continue
        if role == "small" and split is not None:
            continue  # cut from the large masks as they loaded
        if held:
            logger.info(f"The {role}-vessel masks are read or split otherwise now: loading them again.")
        else:
            logger.info(f"The {role}-vessel masks are on now: loading them.")
        loaded_arteriole, loaded_venule, arteriole_size, venule_size = (
            io.load_and_validate_vessel_masks(
                **_vessel_mask_arguments(settings, role),
                image_shape=volume.image.shape,
                main_voxel_size_xyz=volume.voxel_size_xyz,
                crop=getattr(volume, "crop", None),
                downsample=getattr(volume, "downsample", None),
            )
        )
        if role == "large" and split is not None:
            loaded_arteriole, loaded_venule, *small = _split_large_vessel_masks(
                settings,
                loaded_arteriole,
                loaded_venule,
                volume.voxel_size_xyz,
                _SegmentationBoxes(volume.image, settings.get("input_path")),
            )
            changes["small_arteriole_mask"], changes["small_venule_mask"] = as_read(*small)
        loaded_arteriole, loaded_venule = as_read(loaded_arteriole, loaded_venule)
        changes[arteriole], changes[venule] = loaded_arteriole, loaded_venule
        if role == "large":
            _record_graph_voxel_metadata(
                network.graph, volume.voxel_size_xyz, volume.voxel_size_zyx,
                loaded_arteriole, loaded_venule, arteriole_size, venule_size,
            )
    if changes:
        changes["mask_sources"] = sources
    return replace(network, **changes) if changes else network


#: Where ``solve`` leaves the results that live only on the Solution, so a
#: later stage that skips the solve (Continue, or Regenerate with nothing
#: edited) can rebuild them. Pressures already sit on the nodes.
_SOLUTION_RECORD = "solution_record"


def _plain_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _stamp_solution(G: nx.MultiGraph, solution: Solution) -> None:
    """Record the solve's resistances and diagnostics on *G* itself.

    Continue starts at post-processing and does not solve an unedited
    network again. The two resistances and the haematocrit-distribution
    report exist only on the Solution, so without this the statistics CSV
    loses its Haemodynamics section.
    """
    nodes = solution.equivalent_resistance_nodes
    G.graph[_SOLUTION_RECORD] = {
        "equivalent_resistance": _plain_optional_float(solution.equivalent_resistance),
        "equivalent_resistance_nodes": None if nodes is None else (nodes[0], nodes[1]),
        "network_resistance": _plain_optional_float(solution.network_resistance),
        "statistics": {
            key: dict(value) if isinstance(value, dict) else value
            for key, value in solution.statistics.items()
        },
    }


def solved_resistance_node_pair(G: nx.MultiGraph) -> tuple[Any, Any] | None:
    """The two nodes the last solve on *G* found the two-point resistance
    between, from the record it left there; None when it found none."""
    nodes = (G.graph.get(_SOLUTION_RECORD) or {}).get("equivalent_resistance_nodes")
    return None if nodes is None else (nodes[0], nodes[1])


def _solution_from_graph(solved: nx.MultiGraph) -> Solution:
    node_list = list(solved)
    pressure = np.asarray(
        [solved.nodes[node_id].get("pressure", np.nan) for node_id in node_list],
        dtype=float,
    )
    record = solved.graph.get(_SOLUTION_RECORD) or {}
    nodes = record.get("equivalent_resistance_nodes")
    return Solution(
        pressure=pressure,
        node_list=node_list,
        graph=solved,
        equivalent_resistance=record.get("equivalent_resistance"),
        equivalent_resistance_nodes=None if nodes is None else tuple(nodes),
        network_resistance=record.get("network_resistance"),
        statistics=dict(record.get("statistics") or {}),
    )


def run_pipeline_stages(
    settings: dict,
    schema: Schema,
    progress: ProgressCallback | None = None,
    on_stage_output: StageOutputCallback | None = None,
    start_from: str | None = None,
    resume: PipelineResume | None = None,
    stop_after: str | None = None,
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

    *start_from* names a stage call to begin work at. Stages before that point
    are not done again: segment, skeletonise and build_network hand on
    *resume*'s stored outputs when it has them (else ilastik's existing output
    and the saved skeleton are loaded, and *resume*'s graph is used without
    touching ``{stem}_graph.pkl``); boundaries / diameters / haemodynamics /
    solve / perturbations are reconstructed from *resume*. With no stored
    volume, and no stage from *start_from* on reading the segmented image
    (:func:`resumed_image_readers`), neither it nor the skeleton is loaded:
    :class:`ImageNotRead` stands in for both.

    *stop_after* names the last stage call to run: the run returns the graph
    as that stage left it (None before there is one), and the stages after it
    neither run nor report. How the napari panel pauses a run on its Post
    processing tab -- ``stop_after="solve"`` -- and later picks it up again
    with ``start_from="post_process"``. That stage brings a hand-edited
    network in line (:func:`post_process`) and solves its haemodynamics again
    (:func:`solve_post_processed`); a network nobody edited passes through.

    Both run on whatever thread the run is on, and must not raise: a run is not
    stopped, or changed in any way, by whoever is watching it. Note the outputs
    are the live objects, not copies -- every stage after ``build_network``
    writes attributes onto the same graph -- so a consumer that wants a
    snapshot must take one there and then.

    The run writes its manifest -- settings, version, commit, input checksums,
    each stage's time and peak memory -- into its output folder as it ends,
    failed or not (:mod:`haemolynx.pipeline.provenance`).
    """
    if resume is not None and start_from is None:
        start_from = resume.start_from
    if stop_after is not None:
        if stop_after not in STAGE_CALLS:
            raise ValueError(
                f"stop_after={stop_after!r} is not a stage. Stages: {', '.join(STAGE_CALLS)}."
            )
        if start_from in STAGE_CALLS and _stage_index(stop_after) < _stage_index(start_from):
            raise ValueError(
                f"stop_after={stop_after!r} comes before start_from={start_from!r}."
            )
    with record_run(
        settings, schema, progress=progress, start_from=start_from, stop_after=stop_after
    ) as run:
        return _run_stages(run, settings, schema, on_stage_output, start_from, resume, stop_after)


def _run_stages(
    run: RunProgress,
    settings: dict,
    schema: Schema,
    on_stage_output: StageOutputCallback | None,
    start_from: str | None,
    resume: PipelineResume | None,
    stop_after: str | None,
) -> nx.MultiGraph | None:
    """:func:`run_pipeline_stages`'s stages, each reported through *run*."""
    # A run resumed past every stage that reads the image goes on without it.
    unread = _resumes_without_image(settings, resume, start_from)
    with run.stage("segment") as segmenting:
        # Every mask the classifier makes, in one pass over the raw image.
        segment_with_run_classifiers(settings, main=not unread, progress=segmenting)
        if unread:
            inputs = _segmented_inputs_unread(settings)
        else:
            inputs = segment(
                settings,
                segmented_path=_resumed_segmented_path(settings, resume, start_from),
                progress=segmenting,
            )
        segment_object_masks(settings, progress=segmenting)
    _produced(on_stage_output, "segment", inputs)
    if stop_after == "segment":
        return None
    with run.stage("skeletonise") as skeletonising:
        volume = _earlier_output(resume, start_from, "skeletonise", "volume")
        if unread:
            volume = _volume_not_read(settings, inputs, resume, start_from)
        elif volume is None:
            volume = skeletonise(
                settings, inputs, progress=skeletonising,
                thick_region=resume.thick_vessel_mask if resume is not None else None,
            )
            if isinstance(volume, SkeletonisedVolume):
                volume = replace(volume, input_settings=input_volume_settings(settings))
        else:
            _check_resumed_input(settings, getattr(volume, "input_settings", None), "image and skeleton")
            logger.info("Resuming with the image and skeleton already loaded.")
            volume = _volume_writing_to(volume, inputs.output_dir)
    _produced(on_stage_output, "skeletonise", volume)
    if stop_after == "skeletonise":
        return None
    with run.stage("build_network") as building:
        resumed_graph = _earlier_output(resume, start_from, "build_network", "graph")
        if resumed_graph is not None:
            _check_resumed_input(settings, resumed_graph.graph.get(INPUT_SETTINGS_RECORD), "network")
        earlier = _earlier_output(resume, start_from, "build_network", "network")
        if unread:
            # No vessel masks loaded: the stages it runs read the Boundaries
            # stage's, which the resume carries (_boundaries_from_resume).
            network = VesselNetwork(graph=resumed_graph, volume=volume)
        elif earlier is not None and resumed_graph is not None:
            logger.info("Resuming with the network and vessel masks already built.")
            network = replace(earlier, graph=resumed_graph, volume=volume)
            if _run_stage_body("assign_boundaries", start_from):
                network = _vessel_masks_now_in_use(settings, network)
        else:
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
                resumed_graph=resumed_graph,
            )
    if resume is not None and resume.graph is not None and not _run_stage_body("build_network", start_from):
        # Only where graph building did not run: a script resuming at or
        # before it would have had the network it just built replaced.
        network.graph = resume.graph
    # The statistics a resumed network carries are of the network before this
    # run changed it; the export measures it again.
    network_graph = getattr(network, "graph", None)
    if network_graph is not None:
        network_graph.graph.pop(STATISTICS_REPORT, None)
        # The lumen-artefact counts of the graph handed on, for the run's
        # manifest; None where the report did not run (low RAM, synthetic).
        run.checks[graph.LUMEN_ARTEFACT_COUNTS] = network_graph.graph.get(graph.LUMEN_ARTEFACT_COUNTS)
        run.checks["graph_building"] = graph_building_counts(network_graph)
    _produced(on_stage_output, "build_network", network)
    if stop_after == "build_network":
        return network.graph
    with run.stage("assign_boundaries"):
        if _run_stage_body("assign_boundaries", start_from):
            boundaries = assign_boundaries(settings, network)
        else:
            if resume is None:
                raise ValueError(
                    "start_from skips assign_boundaries but no PipelineResume "
                    "was supplied to reconstruct inlet and outlet nodes."
                )
            _fill_boundary_settings(settings, resume)
            boundaries = _boundaries_from_resume(resume, network)
            if boundaries.graph is not None:
                network.graph = boundaries.graph
    _produced(on_stage_output, "assign_boundaries", boundaries)
    if stop_after == "assign_boundaries":
        return boundaries.graph if boundaries.graph is not None else network.graph
    with run.stage("assign_diameters"):
        if _run_stage_body("assign_diameters", start_from):
            diameters = assign_diameters(settings, network, boundaries, schema)
        else:
            if resume is not None:
                _fill_boundary_settings(settings, resume)
            diameters = HaemodynamicModel(
                graph=boundaries.graph if boundaries.graph is not None else network.graph
            )
    _record_vessel_classes(run, diameters)
    _produced(on_stage_output, "assign_diameters", diameters)
    if stop_after == "assign_diameters":
        return diameters.graph
    with run.stage("build_haemodynamic_model"):
        if _run_stage_body("build_haemodynamic_model", start_from):
            diameters = apply_network_handling(
                settings, diameters, boundaries, network, schema=schema
            )
            model = build_haemodynamic_model(settings, diameters, schema)
        else:
            model = diameters
            # The network handling ran when this network was built, and may
            # have removed boundary nodes the resumed lists still name.
            if model.graph is not None:
                _trim_boundary_lists(settings, boundaries, model.graph)
    _produced(on_stage_output, "build_haemodynamic_model", model)
    if stop_after == "build_haemodynamic_model":
        return model.graph
    with run.stage("solve") as solving:
        # Off, solving only takes an earlier solve off a resumed network, so
        # it is not exported with flows this run did not find.
        if _run_stage_body("solve", start_from) or not settings.get("run_haemodynamics", True):
            solution = solve(settings, model, boundaries, schema, progress=solving)
        else:
            solution = _solution_from_graph(model.graph)
    _produced(on_stage_output, "solve", solution)
    if stop_after == "solve":
        return model.graph
    with run.stage("post_process"):
        if _run_stage_body("post_process", start_from):
            edited = graph.has_pending_edits(model.graph)
            model = post_process(
                settings,
                model,
                boundaries,
                schema,
                network=network,
                thick_vessel_mask=resume.thick_vessel_mask if resume is not None else None,
            )
            if edited:
                solution = solve_post_processed(
                    settings, model, boundaries, schema, network=network
                )
    _record_vessel_classes(run, model)
    _produced(on_stage_output, "post_process", solution)
    if stop_after == "post_process":
        return model.graph
    # After the baseline is solved, so each perturbation has a solved network to
    # difference against, and before the export, which writes that baseline out.
    with run.stage("run_perturbations") as perturbing:
        if _run_stage_body("run_perturbations", start_from):
            perturbations = run_perturbations(
                settings,
                model,
                boundaries,
                schema,
                progress=perturbing,
                network=network,
            )
        else:
            perturbations = PerturbationRun()
    _produced(on_stage_output, "run_perturbations", perturbations)
    if stop_after == "run_perturbations":
        return model.graph
    with run.stage("export_results"):
        if _run_stage_body("export_results", start_from):
            export_results(settings, network, model, solution)
    _produced(on_stage_output, "export_results", solution)
    return model.graph
