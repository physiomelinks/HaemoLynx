"""Per-stage snapshots from a GUI run, so a tab can re-run without a full rebuild.

After each pipeline stage finishes in the napari panel, a checkpoint records
what that stage put in the viewer and a pickle of the graph (when there is
one), and -- for segment, skeletonise and build_network -- keeps what the
stage returned, in memory only.

**What "Run from this stage" means.** Standing on tab *K* requires the
checkpoint taken at the **end** of tab *M* (the predecessor) to be ready.
Layers and checkpoints for *K* and later tabs are dropped, earlier layers
are kept, and the pipeline starts at the first stage of *K* with *M*'s graph
handed to it directly. Tabs follow :func:`~haemolynx.gui.tabs.tab_titles`; a
stage that shares another's tab (``solve`` on Haemodynamics) does not open
one of its own, so the Haemodynamics tab starts at
``build_haemodynamic_model`` and its end-of-tab checkpoint is ``solve``.

**What is prepared.** The viewer layers through the previous tab, the
:class:`~haemolynx.gui.results.ResultLayers` memory they need, and a
:class:`~haemolynx.pipeline.stages.PipelineResume` carrying the stored stage
outputs, so the run neither re-segments, re-loads and re-checks the image,
skeleton and masks, nor loads or writes ``{stem}_graph.pkl`` -- that file
stays the one graph building made. Without stored outputs (a run loaded from
a file), ``{stem}_skeleton.npy`` is made sure of and ``do_skeletonize``
turned off instead. Starting after diameters also turns off
``do_fwhm_measurement``, so Haemodynamics does not wipe FWHM approvals.
"""
from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from haemolynx.gui.results import BOUNDARY_NODES, MASK_LAYERS, SKELETON, copy_graph
from haemolynx.gui.tabs import tab_title, tab_titles
from haemolynx.pipeline.progress import STAGES
from haemolynx.pipeline.stages import TOPOLOGY_STEP, PipelineResume, segmented_input_path

logger = logging.getLogger(__name__)

#: The toggles a resumed run may turn off, and Run pipeline puts back to the
#: user's own values. ``do_graph_building`` is no longer turned off -- the
#: graph is handed over instead -- but is still put back, for a panel an
#: older run-from left it off in.
SKIP_FOR_RESUME = ("do_skeletonize", "do_graph_building", "do_fwhm_measurement")

#: Stage call -> the PipelineResume field its stored output fills.
RESUMABLE_OUTPUTS = {"segment": "inputs", "skeletonise": "volume", "build_network": "network"}


def skeleton_resume_path(output_dir: Path, stem: str) -> Path:
    """The ``.npy`` ``do_skeletonize=False`` already loads (and preflight checks)."""
    return Path(output_dir) / f"{stem}_skeleton.npy"


def _skeleton_array_from_groups(groups: Sequence[Any]) -> Any | None:
    """Skeleton volume stored in a replayed checkpoint group, if any."""
    for group in groups:
        for spec in getattr(group, "layers", ()) or ():
            if getattr(spec, "name", None) == SKELETON and getattr(spec, "data", None) is not None:
                return spec.data
    return None


def ensure_skeleton_artefact(
    groups: Sequence[Any],
    output_dir: Path,
    stem: str,
) -> Path | None:
    """Make sure ``{stem}_skeleton.npy`` exists for a resumed Run.

    Returns the path when the file is (or was made) present, else None.
    Without it, turning ``do_skeletonize`` off fails preflight and the user
    cannot continue from the next tab after a revert.
    """
    path = skeleton_resume_path(output_dir, stem)
    if path.is_file():
        return path
    skeleton = _skeleton_array_from_groups(groups)
    if skeleton is None:
        return None
    try:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        np.save(path, np.asarray(skeleton))
        logger.info("Wrote resumed skeleton for next Run to %s", path)
        return path
    except Exception:  # noqa: BLE001 - leave do_skeletonize on if we cannot write
        logger.exception("could not write resumed skeleton %s", path)
        return None


def skip_settings_for_resume(
    *,
    skeleton_ready: bool,
    target: str | None = None,
    start_from: str | None = None,
    use_fwhm_edge_diameters: bool = False,
) -> tuple[str, ...]:
    """Which stage toggles to turn off so a run can start mid-pipeline.

    ``start_from`` is the first stage call of the current tab. ``target`` is
    the previous tab's end stage (the checkpoint being loaded); it is used
    only when ``start_from`` is omitted, for older callers.

    ``do_skeletonize`` is off when the run starts after skeletonising and
    has to load the skeleton -- *skeleton_ready*: the matching ``.npy`` is
    on disk and no stored skeleton is handed over; preflight blocks a run
    that skips it without one. ``do_fwhm_measurement`` is off when starting
    *after* diameters *and* FWHM is in use, so Haemodynamics does not wipe
    FWHM approvals. Starting *at* diameters leaves measurement on, because
    that is the stage being re-run. ``do_graph_building`` is never turned
    off: a resumed run is handed its graph.
    """
    order = [stage.call for stage in STAGES if stage.call]
    if start_from is None and target is not None and target in order:
        idx = order.index(target) + 1
        start_from = order[idx] if idx < len(order) else None
    if start_from is None or start_from not in order:
        return ()

    idx = order.index(start_from)
    skips = []
    if idx > order.index("skeletonise") and skeleton_ready:
        skips.append("do_skeletonize")
    if (
        idx > order.index("assign_diameters")
        and use_fwhm_edge_diameters
    ):
        skips.append("do_fwhm_measurement")
    return tuple(skips)


@dataclass(frozen=True)
class StageCheckpoint:
    """One finished stage, as the panel saw it."""

    stage: str
    title: str
    group: Any  # StageLayers
    graph: Any | None = None
    voxel_size_zyx: tuple[float, float, float] = (1.0, 1.0, 1.0)
    geometry_shown: bool = False
    emitted: tuple[str, ...] = ()
    pickle_path: Path | None = None
    inlet_nodes: tuple[Any, ...] = ()
    outlet_nodes: tuple[Any, ...] = ()
    arteriole_boundary_nodes: tuple[Any, ...] = ()
    venule_boundary_nodes: tuple[Any, ...] = ()
    large_arteriole_mask: Any | None = None
    large_venule_mask: Any | None = None
    #: The fat catchment `skeletonise` produced, if thickness-gated
    #: skeletonisation was on -- carried forward so a stage resumed from a
    #: later tab does not silently disable the thick/thin debug toggle for a
    #: run that genuinely used it (results._thick_vessel_mask has no
    #: per-stage-output equivalent the way the skeleton array itself does).
    thick_vessel_mask: Any | None = None
    #: The pre-cleanup segmented mask, if a segmentation_cleanup_* step ran
    #: -- same "carried forward across a resume" reasoning as
    #: thick_vessel_mask, for the raw-vs-corrected debug toggle.
    raw_segmented_image: Any | None = None
    #: What the stage returned, for the stages in RESUMABLE_OUTPUTS: handed
    #: to a resumed run so it is not done again. In memory only -- a saved
    #: run drops it (see run_snapshot.capture_run), so a loaded one falls
    #: back to loading from disk.
    output: Any | None = None


@dataclass(frozen=True)
class RestorePlan:
    """What preparing a mid-pipeline run does, described without Qt."""

    #: Previous tab's end-of-tab stage call (the checkpoint being loaded).
    stage: str
    title: str
    #: Groups to put back in the viewer, in order from the start of the run
    #: through the previous tab.
    groups: tuple[Any, ...]
    checkpoint: StageCheckpoint
    #: Setting names to set False for the resumed run.
    skip_settings: tuple[str, ...] = ()
    #: Tab that was asked to run: stay here, do not bounce to the predecessor.
    tab_title: str = ""
    #: First stage call of that tab; ``run_pipeline_stages`` starts here.
    start_from: str = ""
    #: Earlier-stage outputs the pipeline should not recompute.
    resume: PipelineResume | None = None


def tab_end_stage(title: str, stages: Sequence = STAGES) -> str | None:
    """The stage call whose checkpoint is the end-of-tab state for *title*.

    When several stages share a tab, the last one in run order wins -- so the
    Haemodynamics tab ends at ``solve``, after pressures and flows are written.

    A tab with no stage of its own (``call=None`` -- "8. Additional
    measurements" is the one today, splitting export_results's settings
    across two tabs without splitting the function itself) runs nothing when
    visited, so its end-of-tab state is whatever its predecessor's already
    was; falls back to that tab instead of reporting no checkpoint at all.
    """
    last: str | None = None
    for stage in stages:
        if stage.call and tab_title(stage) == title:
            last = stage.call
    if last is not None:
        return last
    previous = previous_tab(title, stages)
    return tab_end_stage(previous, stages) if previous is not None else None


def tab_start_stage(title: str, stages: Sequence = STAGES) -> str | None:
    """The first stage call that belongs on *title*.

    Haemodynamics starts at ``build_haemodynamic_model`` (``solve`` shares
    that tab and is not a starting point of its own).

    A tab with no stage of its own (see :func:`tab_end_stage`) starts
    whatever the next tab starts -- "Run from this stage" on it must re-run
    the same stage a change on the next tab would.
    """
    for stage in stages:
        if stage.call and tab_title(stage) == title:
            return stage.call
    titles = tab_titles(stages)
    try:
        index = titles.index(title)
    except ValueError:
        return None
    for later in titles[index + 1:]:
        found = tab_start_stage(later, stages)
        if found is not None:
            return found
    return None


def previous_tab(title: str, stages: Sequence | None = None) -> str | None:
    """The tab before *title*, or None when *title* is the first."""
    titles = tab_titles(stages)
    try:
        index = titles.index(title)
    except ValueError:
        return None
    return titles[index - 1] if index > 0 else None


def revert_target_stage(current_tab: str, stages: Sequence | None = None) -> str | None:
    """Stage call whose checkpoint must be ready to run from *current_tab*."""
    previous = previous_tab(current_tab, stages)
    if previous is None:
        return None
    return tab_end_stage(previous, stages if stages is not None else STAGES)


def can_revert_from(current_tab: str, checkpoints: "StageCheckpoints") -> bool:
    """Whether *current_tab* has a previous tab whose checkpoint is on hand."""
    target = revert_target_stage(current_tab)
    return target is not None and checkpoints.has(target)


can_run_from = can_revert_from


def checkpoint_pickle_path(output_dir: Path, stem: str, stage: str) -> Path:
    """Where a GUI run writes the graph pickle for *stage*."""
    return Path(output_dir) / f"{stem}_checkpoint_{stage}.pkl"


def output_dir_from_prefix(vtk_prefix: Any) -> Path | None:
    """Parent of ``vtk_output_prefix``, or None when it is unset or a bare name.

    A FileEdit that was cleared, ``"."``, or a filename with no directory
    would otherwise treat the working directory as the run's output folder
    and delete pickles there. Magicgui resolves a blank picker to
    ``Path.cwd()`` itself, so that value is the same "unset" as ``"."``.
    """
    if vtk_prefix is None:
        return None
    text = str(vtk_prefix).strip()
    if not text or text == ".":
        return None
    path = Path(vtk_prefix)
    try:
        if path.resolve() == Path.cwd().resolve():
            return None
    except OSError:
        pass
    parent = path.parent
    if str(parent) in {"", "."}:
        return None
    return parent


def discard_cached_artefacts(output_dir: Path, stem: str) -> tuple[Path, ...]:
    """Remove on-disk checkpoint pickles for a GUI run.

    Deletes ``{stem}_checkpoint_*.pkl``. Missing files are ignored. Does
    **not** delete ``{stem}_graph.pkl`` or ``{stem}_skeleton.npy``: those
    are graph building's and skeletonisation's own output, which a run with
    ``do_graph_building`` / ``do_skeletonize`` off loads. Returns paths that
    were actually removed.
    """
    output_dir = Path(output_dir)
    removed: list[Path] = []
    for path in output_dir.glob(f"{stem}_checkpoint_*.pkl"):
        if path.is_file():
            path.unlink()
            removed.append(path)
            logger.info("Discarded cached artefact: %s", path)
    return tuple(removed)


def stems_for_cached_artefacts(settings: Mapping[str, Any] | None) -> tuple[str, ...]:
    """The stem this run's checkpoint pickles are named with, if it has one.

    :meth:`StageCheckpoints.record` names them after the segmented input --
    ``{input stem}_checkpoint_{stage}.pkl`` in the output folder -- so that
    is the stem Clear looks for. It used to look under the VTK prefix's name,
    which no pickle carries, and so only ever removed the files this session
    had written itself. Only this run's own names match: the input's stem in
    this run's output folder.
    """
    if not settings:
        return ()
    segmented = segmented_input_path(dict(settings))
    return () if segmented is None else (segmented.stem,)


def discard_cached_artefacts_for_settings(
    settings: Mapping[str, Any] | None,
    extra_paths: Sequence[Path] = (),
) -> tuple[Path, ...]:
    """Like :func:`discard_cached_artefacts` plus any *extra_paths* this session wrote."""
    if not settings:
        extra = tuple(Path(path) for path in extra_paths if Path(path).is_file())
        for path in extra:
            path.unlink()
            logger.info("Discarded cached artefact: %s", path)
        return extra
    output_dir = output_dir_from_prefix(settings.get("vtk_output_prefix"))
    if output_dir is None:
        extra = tuple(Path(path) for path in extra_paths if Path(path).is_file())
        for path in extra:
            path.unlink()
            logger.info("Discarded cached artefact: %s", path)
        return extra
    removed: list[Path] = []
    seen: set[Path] = set()
    for stem in stems_for_cached_artefacts(settings):
        for path in discard_cached_artefacts(output_dir, stem):
            if path not in seen:
                seen.add(path)
                removed.append(path)
    for path in extra_paths:
        path = Path(path)
        if path in seen or not path.is_file():
            continue
        path.unlink()
        logger.info("Discarded cached artefact: %s", path)
        seen.add(path)
        removed.append(path)
    return tuple(removed)


def _stem_and_output_dir(settings: Mapping[str, Any] | None) -> tuple[str, Path] | None:
    if not settings:
        return None
    output_dir = output_dir_from_prefix(settings.get("vtk_output_prefix"))
    input_path = settings.get("input_path")
    if output_dir is None or input_path is None:
        return None
    return Path(input_path).stem, output_dir


def _boundary_ids_from_group(group: Any) -> dict[str, tuple[Any, ...]]:
    """Node ids per boundary role, read from a checkpoint's layer specs."""
    roles: dict[str, list[Any]] = {
        "inlet": [],
        "outlet": [],
        "arteriole_boundary": [],
        "venule_boundary": [],
    }
    for spec in getattr(group, "layers", ()) or ():
        if getattr(spec, "name", None) != BOUNDARY_NODES:
            continue
        features = getattr(spec, "features", None) or {}
        role_col = features.get("role")
        id_col = features.get("node_id")
        if role_col is None or id_col is None:
            continue
        for role, node_id in zip(role_col, id_col):
            key = str(role)
            if key in roles:
                roles[key].append(node_id)
    return {key: tuple(values) for key, values in roles.items()}


def _boundary_masks_from_group(group: Any) -> tuple[Any | None, Any | None]:
    """The cleaned (overlap-resolved) large-vessel masks, read from a checkpoint's layer specs.

    ``assign_boundaries`` is the only stage that resolves overlap between the
    arteriole and venule masks before cutting the graph and assigning
    terminal nodes; the masks land on ``BoundaryNodes`` as
    ``large_arteriole_mask`` / ``large_venule_mask`` and get turned into
    image-volume layer specs (see ``vessel_mask_volume_layers``). Reading
    them back from here -- instead of from a freshly rebuilt
    ``VesselNetwork``, which only has the raw, never-cleaned masks -- is what
    lets a resumed run's exported overlay match what the resumed graph
    actually used.
    """
    arteriole = None
    venule = None
    for spec in getattr(group, "layers", ()) or ():
        name = getattr(spec, "name", None)
        if name == MASK_LAYERS["large_arteriole_mask"]:
            arteriole = getattr(spec, "data", None)
        elif name == MASK_LAYERS["large_venule_mask"]:
            venule = getattr(spec, "data", None)
    return arteriole, venule


def stages_before(start_from: str | None, stages: Sequence = STAGES) -> tuple[str, ...]:
    """Stage calls that run before *start_from*, in run order.

    A run that starts mid-pipeline still passes through these -- it reloads
    or reconstructs their outputs rather than recomputing them -- so their
    checkpoints and layers are already the ones a run-from restored and must
    not be recorded again from the resumed state.
    """
    order = [stage.call for stage in stages if stage.call]
    if start_from is None or start_from not in order:
        return ()
    return tuple(order[: order.index(start_from)])


def _stored_output(stage: str, output: Any) -> Any | None:
    """What a resumed run may be handed for *stage*, or None.

    A shallow copy, taken when the stage finishes: later stages reassign
    fields on the live object -- assign_boundaries gives the network its
    cleaned masks and cut graph -- and a run resumed at Boundaries must
    start from the network as graph building left it.
    """
    if output is None or stage not in RESUMABLE_OUTPUTS:
        return None
    try:
        return replace(output)
    except TypeError:  # not a dataclass: a test double or an older stage
        return None


def _resume_payload(checkpoint: StageCheckpoint, *, graph: Any, start_from: str) -> PipelineResume:
    """The boundary-node lists a checkpoint carries, threaded onto *graph*.

    Shared by :func:`resume_from_checkpoint` (graph = the checkpoint's own)
    and :func:`resume_from_edit` (graph = a hand-edited one) -- the boundary
    roles are a property of the checkpoint, not of which graph is resumed.

    The run gets a copy: every stage from *start_from* on writes onto the
    graph it is given, and the one it was given here is a checkpoint's own
    (or the editor's) -- a second run from the same tab must start from what
    the first one started from, not from what the first one left behind.
    """
    pair = None
    if checkpoint.inlet_nodes and checkpoint.outlet_nodes:
        pair = (checkpoint.inlet_nodes[0], checkpoint.outlet_nodes[0])
    copied = copy_graph(graph)
    return PipelineResume(
        start_from=start_from,
        # A graph that will not pickle is resumed as it is rather than lost.
        graph=copied if copied is not None else graph,
        inlet_nodes=checkpoint.inlet_nodes,
        outlet_nodes=checkpoint.outlet_nodes,
        arteriole_boundary_nodes=checkpoint.arteriole_boundary_nodes,
        venule_boundary_nodes=checkpoint.venule_boundary_nodes,
        resistance_node_pair=pair,
        large_arteriole_mask=checkpoint.large_arteriole_mask,
        large_venule_mask=checkpoint.large_venule_mask,
    )


def resume_from_checkpoint(checkpoint: StageCheckpoint, start_from: str) -> PipelineResume:
    """Build the pipeline resume payload from a previous-tab checkpoint."""
    return _resume_payload(checkpoint, graph=checkpoint.graph, start_from=start_from)


def resume_from_edit(
    checkpoint: StageCheckpoint,
    edited_graph: Any,
    start_from: str = "assign_diameters",
) -> PipelineResume:
    """Build the pipeline resume payload from a hand-edited graph.

    *checkpoint* only supplies the boundary-node-id lists (any checkpoint at
    or after ``assign_boundaries`` carries these forward -- see
    :meth:`StageCheckpoints._carried_boundary_roles` -- so the caller can
    pass whichever is the most recently recorded one); *edited_graph* is
    threaded through in place of the checkpoint's own stored graph.

    Defaults to starting at ``assign_diameters`` rather than
    ``build_haemodynamic_model``: a hand-added edge has no ``branch_order``
    or ``length`` yet, and both must be recomputed for the edited topology
    before haemodynamics can give it a real resistance (see
    ``haemodynamics.poiseuille.set_poiseuille_resistances`` for what an edge
    missing either one costs -- a silent zero-conductance edge).

    If the edit deleted a node that was one of these boundary nodes (or the
    resistance-pair node), ``solve`` raises a clear error rather than
    solving silently wrong -- see
    ``haemodynamics.resistance.solve_flow_from_conductance_matrix``.
    """
    return _resume_payload(checkpoint, graph=edited_graph, start_from=start_from)


class StageCheckpoints:
    """In-memory (and on-disk graph) snapshots keyed by stage call."""

    def __init__(self) -> None:
        self._by_stage: dict[str, StageCheckpoint] = {}
        self._written_paths: list[Path] = []
        self._recording = True

    def clear(self) -> None:
        self._by_stage.clear()
        self._written_paths.clear()
        self._recording = True

    def freeze(self) -> None:
        """Ignore ``record`` until the next run starts (Clear mid-run)."""
        self._recording = False

    def unfreeze(self) -> None:
        self._recording = True

    @property
    def session_artefact_paths(self) -> tuple[Path, ...]:
        """Pickles this session wrote that are still on disk."""
        return tuple(path for path in self._written_paths if path.is_file())

    def remember_path(self, path: Path | None) -> None:
        if path is not None:
            self._written_paths.append(Path(path))

    def has(self, stage: str) -> bool:
        return stage in self._by_stage

    def get(self, stage: str) -> StageCheckpoint | None:
        return self._by_stage.get(stage)

    @property
    def stages(self) -> tuple[str, ...]:
        """Recorded stage calls, in the order they were recorded."""
        return tuple(self._by_stage)

    def records(self) -> tuple[StageCheckpoint, ...]:
        """Checkpoints in the order they were recorded."""
        return tuple(self._by_stage.values())

    def replace_all(self, checkpoints: Sequence[StageCheckpoint]) -> None:
        """Install *checkpoints* as the whole in-memory history of a run."""
        self._by_stage = {item.stage: replace(item, pickle_path=None) for item in checkpoints}
        self._recording = True

    def _carried_boundary_roles(self) -> dict[str, tuple[Any, ...]] | None:
        """Boundary node ids from the most recent checkpoint that recorded them.

        Only ``assign_boundaries`` emits a ``BOUNDARY_NODES`` layer, so every
        later stage's own group has none. Without this, ``record`` would
        overwrite the real ids with empty tuples on every subsequent stage.
        """
        for checkpoint in reversed(list(self._by_stage.values())):
            if checkpoint.inlet_nodes or checkpoint.outlet_nodes:
                return {
                    "inlet": checkpoint.inlet_nodes,
                    "outlet": checkpoint.outlet_nodes,
                    "arteriole_boundary": checkpoint.arteriole_boundary_nodes,
                    "venule_boundary": checkpoint.venule_boundary_nodes,
                }
        return None

    def _carried_boundary_masks(self) -> tuple[Any | None, Any | None] | None:
        """The cleaned large-vessel masks from the most recent checkpoint that recorded them.

        Only ``assign_boundaries`` emits the mask image layers, so every
        later stage's own group has none -- same reasoning as
        :meth:`_carried_boundary_roles`.
        """
        for checkpoint in reversed(list(self._by_stage.values())):
            if checkpoint.large_arteriole_mask is not None or checkpoint.large_venule_mask is not None:
                return checkpoint.large_arteriole_mask, checkpoint.large_venule_mask
        return None

    def record(
        self,
        stage: str,
        group: Any,
        results: Any,
        settings: Mapping[str, Any] | None = None,
        output: Any | None = None,
    ) -> StageCheckpoint | None:
        """Remember *group* (and the graph *results* holds) for *stage*.

        Topology-step events are ignored: they are not tabs. A stage that
        finishes twice in one run replaces the earlier checkpoint. A freeze
        (Clear while a worker is dying) drops the write so pickles cannot
        come back after discard.

        *output* is what the stage returned; it is kept for the stages in
        :data:`RESUMABLE_OUTPUTS` (see :func:`_stored_output`).
        """
        if not self._recording:
            return None
        if not stage or stage.startswith(TOPOLOGY_STEP):
            return None

        graph = copy_graph(getattr(results, "_graph", None))
        pickle_path: Path | None = None
        located = _stem_and_output_dir(settings)
        if graph is not None and located is not None:
            stem, output_dir = located
            output_dir.mkdir(parents=True, exist_ok=True)
            pickle_path = checkpoint_pickle_path(output_dir, stem, stage)
            try:
                with pickle_path.open("wb") as handle:
                    pickle.dump(graph, handle)
                logger.info("Saved stage checkpoint graph: %s", pickle_path)
                self.remember_path(pickle_path)
            except Exception:  # noqa: BLE001 - viewer restore still has the in-memory copy
                logger.exception("could not write stage checkpoint %s", pickle_path)
                pickle_path = None

        roles = _boundary_ids_from_group(group)
        if not any(roles.values()):
            carried = self._carried_boundary_roles()
            if carried is not None:
                roles = carried
        large_arteriole_mask, large_venule_mask = _boundary_masks_from_group(group)
        if large_arteriole_mask is None and large_venule_mask is None:
            carried_masks = self._carried_boundary_masks()
            if carried_masks is not None:
                large_arteriole_mask, large_venule_mask = carried_masks
        checkpoint = StageCheckpoint(
            stage=stage,
            title=getattr(group, "title", stage),
            group=group,
            graph=graph,
            voxel_size_zyx=tuple(
                float(v) for v in getattr(results, "_voxel_size_zyx", (1.0, 1.0, 1.0))
            ),
            geometry_shown=bool(getattr(results, "_geometry_shown", False)),
            emitted=tuple(getattr(results, "emitted", ()) or ()),
            pickle_path=pickle_path,
            inlet_nodes=roles.get("inlet", ()),
            outlet_nodes=roles.get("outlet", ()),
            arteriole_boundary_nodes=roles.get("arteriole_boundary", ()),
            venule_boundary_nodes=roles.get("venule_boundary", ()),
            large_arteriole_mask=large_arteriole_mask,
            large_venule_mask=large_venule_mask,
            thick_vessel_mask=getattr(results, "_thick_vessel_mask", None),
            raw_segmented_image=getattr(results, "_raw_segmented_image", None),
            output=_stored_output(stage, output),
        )
        self._by_stage[stage] = checkpoint
        return checkpoint

    def _stored_outputs(self, start_from: str) -> dict[str, Any]:
        """PipelineResume fields for the stored outputs ahead of *start_from*."""
        before = set(stages_before(start_from))
        return {
            field_name: self._by_stage[stage].output
            for stage, field_name in RESUMABLE_OUTPUTS.items()
            if stage in before
            and stage in self._by_stage
            and self._by_stage[stage].output is not None
        }

    def apply_to_results(self, results: Any, checkpoint: StageCheckpoint) -> None:
        """Put *results* back to how it was at *checkpoint*."""
        results._graph = copy_graph(checkpoint.graph)
        results._voxel_size_zyx = checkpoint.voxel_size_zyx
        results._geometry_shown = checkpoint.geometry_shown
        results._emitted = list(checkpoint.emitted)
        results._thick_vessel_mask = checkpoint.thick_vessel_mask
        results._raw_segmented_image = checkpoint.raw_segmented_image

    def plan_restore(
        self,
        current_tab: str,
        settings: Mapping[str, Any] | None = None,
    ) -> RestorePlan | None:
        """Prepare a run that starts at *current_tab*, or None if impossible.

        Requires the previous tab's end-of-tab checkpoint. Drops layers and
        checkpoints for this tab and later ones, and gathers the resume
        payload (graph, boundary roles, stored stage outputs) and any skip
        toggles so :func:`~haemolynx.pipeline.run_pipeline_stages` can start
        at this tab's first stage.
        """
        return self.plan_run_from(current_tab, settings=settings)

    def plan_run_from(
        self,
        current_tab: str,
        settings: Mapping[str, Any] | None = None,
        *,
        drop: bool = True,
    ) -> RestorePlan | None:
        """See :meth:`plan_restore`.

        With ``drop=False`` the checkpoints of this tab and later ones are
        kept until :meth:`drop_from` is called -- so a caller can check the
        planned run will start before it throws away the work it replaces.
        """
        target = revert_target_stage(current_tab)
        start_from = tab_start_stage(current_tab)
        if target is None or start_from is None or not self.has(target):
            return None
        checkpoint = self._by_stage[target]
        previous = previous_tab(current_tab)
        assert previous is not None

        groups = [
            self._by_stage[name].group
            for name in stages_before(start_from)
            if name in self._by_stage
        ]

        stored = self._stored_outputs(start_from)
        skip = self._write_resume_artefacts(groups, start_from, settings, stored=stored)
        if drop:
            self.drop_from(start_from)

        return RestorePlan(
            stage=target,
            title=checkpoint.title,
            groups=tuple(groups),
            checkpoint=checkpoint,
            skip_settings=skip,
            tab_title=current_tab,
            start_from=start_from,
            resume=replace(resume_from_checkpoint(checkpoint, start_from), **stored),
        )

    def plan_regenerate(
        self,
        edited_graph: Any,
        settings: Mapping[str, Any] | None = None,
        start_from: str = "assign_diameters",
        *,
        drop: bool = True,
    ) -> RestorePlan | None:
        """Prepare a run that continues a hand-edited graph from *start_from*.

        What :meth:`plan_run_from` does for a tab, for the graph editor's
        Regenerate: the edited graph is handed to the run with the stored
        earlier stages, so it does not re-skeletonise the image and rebuild a
        graph only to throw it away for the edited one, and the checkpoints
        from *start_from* on -- made from the graph before the edit -- are
        dropped. The boundary roles come from the most recent
        checkpoint (see :func:`resume_from_edit`). None until Boundaries has
        been recorded: before that there is nothing to solve between.
        """
        if not self.has("assign_boundaries"):
            return None  # no boundary nodes to solve the edited graph between
        checkpoint = self.records()[-1]
        groups = [
            self._by_stage[name].group
            for name in stages_before(start_from)
            if name in self._by_stage
        ]
        stored = self._stored_outputs(start_from)
        skip = self._write_resume_artefacts(groups, start_from, settings, stored=stored)
        if drop:
            self.drop_from(start_from)
        return RestorePlan(
            stage=checkpoint.stage,
            title=checkpoint.title,
            groups=tuple(groups),
            checkpoint=checkpoint,
            skip_settings=skip,
            start_from=start_from,
            resume=replace(resume_from_edit(checkpoint, edited_graph, start_from), **stored),
        )

    def _write_resume_artefacts(
        self,
        groups: Sequence[Any],
        start_from: str,
        settings: Mapping[str, Any] | None,
        *,
        stored: Mapping[str, Any],
    ) -> tuple[str, ...]:
        """Name the toggles a run starting at *start_from* needs turned off.

        The graph is never written: the run is handed it. The skeleton is
        made sure of on disk only when no stored one is handed over (a run
        loaded from a file), since then the run loads it.
        """
        skeleton_path = None
        located = _stem_and_output_dir(settings)
        if located is not None and "volume" not in stored:
            stem, output_dir = located
            output_dir.mkdir(parents=True, exist_ok=True)
            # Only a skeleton written here is this session's to discard: the
            # one skeletonise itself saved is the pipeline's output, which
            # Clear keeps (see discard_cached_artefacts) -- and which a run
            # with the user's own do_skeletonize off still has to load.
            skeleton_existed = skeleton_resume_path(output_dir, stem).is_file()
            skeleton_path = ensure_skeleton_artefact(groups, output_dir, stem)
            if skeleton_path is not None and not skeleton_existed:
                self.remember_path(skeleton_path)
        return skip_settings_for_resume(
            skeleton_ready=skeleton_path is not None,
            start_from=start_from,
            use_fwhm_edge_diameters=bool(settings and settings.get("use_fwhm_edge_diameters")),
        )

    def drop_from(self, start_from: str) -> None:
        """Forget the checkpoints of *start_from* and every later stage."""
        keep = set(stages_before(start_from))
        for name in list(self._by_stage):
            if name not in keep:
                del self._by_stage[name]


def restore_message(plan: RestorePlan) -> str:
    """What the report box says after preparing a run from this stage."""
    start = plan.start_from or "this stage"
    note = f"Running from {plan.tab_title or start} (using {plan.title})."
    if plan.skip_settings:
        note += f" Turned off {', '.join(plan.skip_settings)}."
    return note
