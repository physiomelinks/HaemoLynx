"""Per-stage snapshots from a GUI run, so a tab can revert without a full re-run.

After each pipeline stage finishes in the napari panel, a checkpoint records
what that stage put in the viewer and a pickle of the graph (when there is
one), using the same ``pickle.dump`` path the pipeline already uses for
``{stem}_graph.pkl`` and for ``save_graph_snapshot``.

**What "previous tab" means.** Standing on tab *K* and pressing "Revert to
previous stage" restores the checkpoint taken at the **end** of tab *M*
(the predecessor). The panel then selects tab *M* — the restored stage —
and must not bounce back to tab *K*. Tabs follow
:func:`~haemolynx.gui.tabs.tab_titles`; a stage that shares another's tab
(``solve`` on Haemodynamics) does not open one of its own, so the
Haemodynamics tab's end-of-tab checkpoint is ``solve``, not
``build_haemodynamic_model``.

**What is restored.** The viewer layers for that earlier stage (by replaying
checkpoints from the start through the target), the
:class:`~haemolynx.gui.results.ResultLayers` memory it needs, and -- when the
checkpoint carries a graph at or after ``build_network`` -- the on-disk
``{stem}_graph.pkl`` plus the ``do_skeletonize`` / ``do_graph_building``
toggles so the next Run loads that graph and continues from later stages
rather than rebuilding topology. Preflight requires ``{stem}_skeleton.npy``
whenever ``do_skeletonize`` is off, so resume also ensures that artefact
exists (re-writing it from the skeletonise checkpoint layers when needed)
before naming ``do_skeletonize`` among the skip toggles.
"""
from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from haemolynx.gui.results import SKELETON
from haemolynx.gui.tabs import tab_title, tab_titles
from haemolynx.pipeline.progress import STAGES
from haemolynx.pipeline.stages import TOPOLOGY_STEP

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

SKIP_FOR_RESUME = ("do_skeletonize", "do_graph_building")


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
) -> tuple[str, ...]:
    """Which stage toggles to turn off after a successful graph resume write.

    ``do_graph_building`` is safe whenever the graph pickle was written.
    ``do_skeletonize`` is only safe when the matching ``.npy`` is on disk —
    otherwise preflight blocks the next Run.
    """
    if not graph_written:
        return ()
    if skeleton_ready:
        return SKIP_FOR_RESUME
    return ("do_graph_building",)

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


@dataclass(frozen=True)
class RestorePlan:
    """What reverting to a checkpoint does, described without Qt."""

    #: Stage call being restored (end of the previous tab).
    stage: str
    title: str
    #: Groups to put back in the viewer, in order from the start of the run.
    groups: tuple[Any, ...]
    checkpoint: StageCheckpoint
    #: Setting names to set False so the next Run loads the written graph.
    skip_settings: tuple[str, ...] = ()
    #: Where the graph was written for resume, if it was.
    graph_path: Path | None = None
    #: Tab title to select after the restore.
    tab_title: str = ""


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


def previous_tab(title: str, stages: Sequence | None = None) -> str | None:
    """The tab before *title*, or None when *title* is the first."""
    titles = tab_titles(stages)
    try:
        index = titles.index(title)
    except ValueError:
        return None
    return titles[index - 1] if index > 0 else None


def revert_target_stage(current_tab: str, stages: Sequence | None = None) -> str | None:
    """Stage call to restore when reverting from *current_tab*."""
    previous = previous_tab(current_tab, stages)
    if previous is None:
        return None
    return tab_end_stage(previous, stages if stages is not None else STAGES)


def can_revert_from(current_tab: str, checkpoints: "StageCheckpoints") -> bool:
    """Whether *current_tab* has a previous tab whose checkpoint is on hand."""
    target = revert_target_stage(current_tab)
    return target is not None and checkpoints.has(target)


def checkpoint_pickle_path(output_dir: Path, stem: str, stage: str) -> Path:
    """Where a GUI run writes the graph pickle for *stage*."""
    return Path(output_dir) / f"{stem}_checkpoint_{stage}.pkl"


def graph_resume_path(output_dir: Path, stem: str) -> Path:
    """The pickle ``do_graph_building=False`` already loads."""
    return Path(output_dir) / f"{stem}_graph.pkl"


def _copy_graph(graph: Any) -> Any | None:
    """A pickle round-trip copy, matching the pipeline's own save path."""
    if graph is None:
        return None
    try:
        return pickle.loads(pickle.dumps(graph))
    except Exception:  # noqa: BLE001 - a graph that will not pickle is still drawable
        logger.exception("could not pickle graph for a stage checkpoint")
        return None


def _stem_and_output_dir(settings: Mapping[str, Any] | None) -> tuple[str, Path] | None:
    if not settings:
        return None
    vtk_prefix = settings.get("vtk_output_prefix")
    input_path = settings.get("input_path")
    if vtk_prefix is None or input_path is None:
        return None
    return Path(input_path).stem, Path(vtk_prefix).parent


class StageCheckpoints:
    """In-memory (and on-disk graph) snapshots keyed by stage call."""

    def __init__(self) -> None:
        self._by_stage: dict[str, StageCheckpoint] = {}

    def clear(self) -> None:
        self._by_stage.clear()

    def has(self, stage: str) -> bool:
        return stage in self._by_stage

    def get(self, stage: str) -> StageCheckpoint | None:
        return self._by_stage.get(stage)

    @property
    def stages(self) -> tuple[str, ...]:
        """Recorded stage calls, in the order they were recorded."""
        return tuple(self._by_stage)

    def record(
        self,
        stage: str,
        group: Any,
        results: Any,
        settings: Mapping[str, Any] | None = None,
    ) -> StageCheckpoint | None:
        """Remember *group* (and the graph *results* holds) for *stage*.

        Topology-step events are ignored: they are not tabs. A stage that
        finishes twice in one run replaces the earlier checkpoint.
        """
        if not stage or stage.startswith(TOPOLOGY_STEP):
            return None

        graph = _copy_graph(getattr(results, "_graph", None))
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
            except Exception:  # noqa: BLE001 - viewer restore still has the in-memory copy
                logger.exception("could not write stage checkpoint %s", pickle_path)
                pickle_path = None

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
        )
        self._by_stage[stage] = checkpoint
        return checkpoint

    def apply_to_results(self, results: Any, checkpoint: StageCheckpoint) -> None:
        """Put *results* back to how it was at *checkpoint*."""
        results._graph = _copy_graph(checkpoint.graph)
        results._voxel_size_zyx = checkpoint.voxel_size_zyx
        results._geometry_shown = checkpoint.geometry_shown
        results._emitted = list(checkpoint.emitted)

    def plan_restore(
        self,
        current_tab: str,
        settings: Mapping[str, Any] | None = None,
    ) -> RestorePlan | None:
        """Everything needed to revert from *current_tab*, or None if impossible."""
        target = revert_target_stage(current_tab)
        if target is None or not self.has(target):
            return None
        checkpoint = self._by_stage[target]
        previous = previous_tab(current_tab)
        assert previous is not None

        # Replay every recorded main stage up to and including the target, in
        # pipeline order, so layers only a later stage added are not left behind.
        order = [stage.call for stage in STAGES if stage.call]
        groups = []
        for name in order:
            if name not in self._by_stage:
                continue
            groups.append(self._by_stage[name].group)
            if name == target:
                break

        skip: tuple[str, ...] = ()
        graph_path: Path | None = None
        if checkpoint.graph is not None and target in GRAPH_RESUME_STAGES:
            located = _stem_and_output_dir(settings)
            if located is not None:
                stem, output_dir = located
                output_dir.mkdir(parents=True, exist_ok=True)
                graph_path = graph_resume_path(output_dir, stem)
                with graph_path.open("wb") as handle:
                    pickle.dump(checkpoint.graph, handle)
                logger.info(
                    "Wrote resumed graph for next Run to %s (from stage %s)",
                    graph_path,
                    target,
                )
                # Preflight refuses do_skeletonize=False without the .npy; write
                # it from the skeletonise checkpoint layers when the file is gone
                # (or was never beside this vtk_output_prefix).
                skeleton_path = ensure_skeleton_artefact(groups, output_dir, stem)
                skip = skip_settings_for_resume(
                    graph_written=True,
                    skeleton_ready=skeleton_path is not None,
                )

        # Drop checkpoints after the target: they describe a future that no
        # longer matches the viewer or the graph on disk.
        keep = set(order[: order.index(target) + 1])
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
            # Select the restored stage's tab (M), not the tab whose Revert
            # button was pressed (K).
            tab_title=previous,
        )


def restore_message(plan: RestorePlan) -> str:
    """What the report box says after a successful revert."""
    note = f"Restored to end of {plan.title}."
    if plan.graph_path is not None:
        note += (
            f" Next Run will load {plan.graph_path.name} "
            f"(turned off {', '.join(plan.skip_settings)})."
        )
    return note
