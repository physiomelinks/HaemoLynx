"""Per-stage snapshots from a GUI run, so a tab can re-run without a full rebuild.

After each pipeline stage finishes in the napari panel, a checkpoint records
what that stage put in the viewer and a pickle of the graph (when there is
one), using the same ``pickle.dump`` path the pipeline already uses for
``{stem}_graph.pkl`` and for ``save_graph_snapshot``.

**What "Run from this stage" means.** Standing on tab *K* requires the
checkpoint taken at the **end** of tab *M* (the predecessor) to be ready.
Layers and checkpoints for *K* and later tabs are dropped, earlier layers
are kept, skip toggles load *M*'s graph, and the pipeline starts at the
first stage of *K*. Tabs follow :func:`~haemolynx.gui.tabs.tab_titles`; a
stage that shares another's tab (``solve`` on Haemodynamics) does not open
one of its own, so the Haemodynamics tab starts at
``build_haemodynamic_model`` and its end-of-tab checkpoint is ``solve``.

**What is prepared.** The viewer layers through the previous tab (by replaying
checkpoints from the start through *M*), the
:class:`~haemolynx.gui.results.ResultLayers` memory they need, and -- when the
checkpoint carries a graph at or after ``build_network`` -- the on-disk
``{stem}_graph.pkl`` plus the ``do_skeletonize`` / ``do_graph_building``
toggles so the run loads that graph. Starting after diameters also turns off
``do_fwhm_measurement``, so Haemodynamics does not wipe FWHM approvals.
Preflight requires ``{stem}_skeleton.npy`` whenever ``do_skeletonize`` is
off, so resume also ensures that artefact exists (re-writing it from the
skeletonise checkpoint layers when needed) before naming ``do_skeletonize``
among the skip toggles.
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
from haemolynx.pipeline.stages import TOPOLOGY_STEP, PipelineResume

logger = logging.getLogger(__name__)

#: Stages whose checkpoint is enough for the next Run to skip rebuilding
#: topology: write ``{stem}_graph.pkl`` and turn the two stage toggles off.
GRAPH_RESUME_STAGES = frozenset(
    {
        "build_network",
        "assign_boundaries",
        "assign_diameters",
        "build_haemodynamic_model",
        "solve",
        "run_perturbations",
        "export_results",
    }
)

SKIP_FOR_RESUME = ("do_skeletonize", "do_graph_building", "do_fwhm_measurement")
GRAPH_SKIP_FOR_RESUME = ("do_skeletonize", "do_graph_building")

#: Stages whose checkpoint already carries stamped diameters, so the next Run
#: must not remeasure FWHM (that would wipe approvals).
DIAMETER_RESUME_STAGES = frozenset(
    {
        "assign_diameters",
        "build_haemodynamic_model",
        "solve",
        "run_perturbations",
        "export_results",
    }
)


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
    graph_written: bool,
    skeleton_ready: bool,
    target: str | None = None,
    start_from: str | None = None,
    use_fwhm_edge_diameters: bool = False,
) -> tuple[str, ...]:
    """Which stage toggles to turn off so a run can start mid-pipeline.

    ``start_from`` is the first stage call of the current tab. ``target`` is
    the previous tab's end stage (the checkpoint being loaded); it is used
    only when ``start_from`` is omitted, for older callers.

    ``do_graph_building`` is off when the run starts *after* graph building
    and the pickle was written. ``do_skeletonize`` is only safe when the
    matching ``.npy`` is on disk — otherwise preflight blocks the next Run.
    ``do_fwhm_measurement`` is off when starting *after* diameters *and*
    FWHM is in use, so Haemodynamics does not wipe FWHM approvals. Starting
    *at* diameters leaves measurement on, because that is the stage being
    re-run.
    """
    order = [stage.call for stage in STAGES if stage.call]
    if start_from is None and target is not None and target in order:
        idx = order.index(target) + 1
        start_from = order[idx] if idx < len(order) else None
    if start_from is None or start_from not in order:
        if not graph_written:
            return ()
        skips: list[str] = ["do_graph_building"]
        if skeleton_ready:
            skips.insert(0, "do_skeletonize")
        if target in DIAMETER_RESUME_STAGES and use_fwhm_edge_diameters:
            skips.append("do_fwhm_measurement")
        return tuple(skips)

    idx = order.index(start_from)
    skips = []
    if idx > order.index("skeletonise") and skeleton_ready:
        skips.append("do_skeletonize")
    if idx > order.index("build_network") and graph_written:
        skips.append("do_graph_building")
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
    #: Setting names to set False so the run loads the written graph.
    skip_settings: tuple[str, ...] = ()
    #: Where the graph was written for resume, if it was.
    graph_path: Path | None = None
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
    """
    last: str | None = None
    for stage in stages:
        if stage.call and tab_title(stage) == title:
            last = stage.call
    return last


def tab_start_stage(title: str, stages: Sequence = STAGES) -> str | None:
    """The first stage call that belongs on *title*.

    Haemodynamics starts at ``build_haemodynamic_model`` (``solve`` shares
    that tab and is not a starting point of its own).
    """
    for stage in stages:
        if stage.call and tab_title(stage) == title:
            return stage.call
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


def graph_resume_path(output_dir: Path, stem: str) -> Path:
    """The pickle ``do_graph_building=False`` already loads."""
    return Path(output_dir) / f"{stem}_graph.pkl"


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
    """Remove on-disk resume/checkpoint pickles for a GUI run.

    Deletes ``{stem}_graph.pkl`` and ``{stem}_checkpoint_*.pkl``. Missing
    files are ignored. Does **not** delete ``{stem}_skeleton.npy``. Returns
    paths that were actually removed.
    """
    output_dir = Path(output_dir)
    removed: list[Path] = []
    candidates = [graph_resume_path(output_dir, stem)]
    candidates.extend(output_dir.glob(f"{stem}_checkpoint_*.pkl"))
    for path in candidates:
        if path.is_file():
            path.unlink()
            removed.append(path)
            logger.info("Discarded cached artefact: %s", path)
    return tuple(removed)


def stems_for_cached_artefacts(settings: Mapping[str, Any] | None) -> tuple[str, ...]:
    """Stem names resume/checkpoint pickles may use for *settings*.

    ``vtk_output_prefix``'s final component names the run output. The input
    stem is not included: deleting every pickle that happens to share the
    image's name would take another run's artefacts in the same folder.
    """
    if not settings:
        return ()
    vtk_prefix = settings.get("vtk_output_prefix")
    if vtk_prefix is None:
        return ()
    return (Path(vtk_prefix).name,)


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


def resume_from_checkpoint(checkpoint: StageCheckpoint, start_from: str) -> PipelineResume:
    """Build the pipeline resume payload from a previous-tab checkpoint."""
    pair = None
    if checkpoint.inlet_nodes and checkpoint.outlet_nodes:
        pair = (checkpoint.inlet_nodes[0], checkpoint.outlet_nodes[0])
    return PipelineResume(
        start_from=start_from,
        graph=checkpoint.graph,
        inlet_nodes=checkpoint.inlet_nodes,
        outlet_nodes=checkpoint.outlet_nodes,
        arteriole_boundary_nodes=checkpoint.arteriole_boundary_nodes,
        venule_boundary_nodes=checkpoint.venule_boundary_nodes,
        resistance_node_pair=pair,
        large_arteriole_mask=checkpoint.large_arteriole_mask,
        large_venule_mask=checkpoint.large_venule_mask,
    )


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
    ) -> StageCheckpoint | None:
        """Remember *group* (and the graph *results* holds) for *stage*.

        Topology-step events are ignored: they are not tabs. A stage that
        finishes twice in one run replaces the earlier checkpoint. A freeze
        (Clear while a worker is dying) drops the write so pickles cannot
        come back after discard.
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
        )
        self._by_stage[stage] = checkpoint
        return checkpoint

    def apply_to_results(self, results: Any, checkpoint: StageCheckpoint) -> None:
        """Put *results* back to how it was at *checkpoint*."""
        results._graph = copy_graph(checkpoint.graph)
        results._voxel_size_zyx = checkpoint.voxel_size_zyx
        results._geometry_shown = checkpoint.geometry_shown
        results._emitted = list(checkpoint.emitted)

    def plan_restore(
        self,
        current_tab: str,
        settings: Mapping[str, Any] | None = None,
    ) -> RestorePlan | None:
        """Prepare a run that starts at *current_tab*, or None if impossible.

        Requires the previous tab's end-of-tab checkpoint. Drops layers and
        checkpoints for this tab and later ones, writes the resume graph, and
        names the skip toggles so :func:`~haemolynx.pipeline.run_pipeline_stages`
        can start at this tab's first stage.
        """
        return self.plan_run_from(current_tab, settings=settings)

    def plan_run_from(
        self,
        current_tab: str,
        settings: Mapping[str, Any] | None = None,
    ) -> RestorePlan | None:
        """See :meth:`plan_restore`."""
        target = revert_target_stage(current_tab)
        start_from = tab_start_stage(current_tab)
        if target is None or start_from is None or not self.has(target):
            return None
        checkpoint = self._by_stage[target]
        previous = previous_tab(current_tab)
        assert previous is not None

        order = [stage.call for stage in STAGES if stage.call]
        groups = []
        for name in order:
            if name not in self._by_stage:
                continue
            if name == start_from:
                break
            groups.append(self._by_stage[name].group)

        skip: tuple[str, ...] = ()
        graph_path: Path | None = None
        skeleton_path = None
        located = _stem_and_output_dir(settings)
        if located is not None:
            stem, output_dir = located
            output_dir.mkdir(parents=True, exist_ok=True)
            skeleton_path = ensure_skeleton_artefact(groups, output_dir, stem)
            if skeleton_path is not None:
                self.remember_path(skeleton_path)
            write_graph = (
                checkpoint.graph is not None
                and start_from not in {"segment", "skeletonise", "build_network"}
            )
            if write_graph:
                graph_path = graph_resume_path(output_dir, stem)
                with graph_path.open("wb") as handle:
                    pickle.dump(checkpoint.graph, handle)
                logger.info(
                    "Wrote resumed graph for run-from %s to %s (from stage %s)",
                    start_from,
                    graph_path,
                    target,
                )
                self.remember_path(graph_path)

        skip = skip_settings_for_resume(
            graph_written=graph_path is not None,
            skeleton_ready=skeleton_path is not None,
            start_from=start_from,
            use_fwhm_edge_diameters=bool(
                settings and settings.get("use_fwhm_edge_diameters")
            ),
        )

        keep = set(order[: order.index(start_from)])
        for name in list(self._by_stage):
            if name not in keep:
                del self._by_stage[name]

        return RestorePlan(
            stage=target,
            title=checkpoint.title,
            groups=tuple(groups),
            checkpoint=checkpoint,
            skip_settings=skip,
            graph_path=graph_path,
            tab_title=current_tab,
            start_from=start_from,
            resume=resume_from_checkpoint(checkpoint, start_from),
        )


def restore_message(plan: RestorePlan) -> str:
    """What the report box says after preparing a run from this stage."""
    start = plan.start_from or "this stage"
    note = f"Running from {plan.tab_title or start} (using {plan.title})."
    if plan.graph_path is not None:
        note += (
            f" Loading {plan.graph_path.name}"
            + (
                f" (turned off {', '.join(plan.skip_settings)})"
                if plan.skip_settings
                else ""
            )
            + "."
        )
    elif plan.skip_settings:
        note += f" Turned off {', '.join(plan.skip_settings)}."
    return note
