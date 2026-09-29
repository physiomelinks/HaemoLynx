"""Save and load a finished GUI pipeline run, without Qt.

Config YAML only stores settings. A run snapshot stores the settings *and*
the checkpoints, remembered graph, and layer specs a viewer run produced, so
opening the file in this napari session or a new one reconstructs the panel
as it looked when the run finished.

The file is a gzip-compressed pickle named ``*.haemorun``. Arrays and the
NetworkX graph are the same objects the live panel already pickles for
resume; this bundles them with the form values.
"""
from __future__ import annotations

import gzip
import logging
import pickle
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from haemolynx.gui.results import ResultLayers, copy_graph
from haemolynx.gui.stage_checkpoints import (
    StageCheckpoint,
    StageCheckpoints,
    ensure_skeleton_artefact,
    skeleton_resume_path,
    _stem_and_output_dir,
)
from haemolynx.gui.tabs import tab_title
from haemolynx.pipeline.progress import STAGES

logger = logging.getLogger(__name__)

FORMAT = "haemolynx.run"
VERSION = 1
SUFFIX = ".haemorun"
SAVE_FILTER = "HaemoLynx run (*.haemorun);;All files (*)"
DEFAULT_FILENAME = f"haemolynx-run{SUFFIX}"

NOTHING_TO_SAVE = (
    "Nothing to save: run the pipeline with 'Show each stage in the viewer' first."
)


class RunSnapshotError(ValueError):
    """The file is not a HaemoLynx run snapshot, or cannot be read."""


@dataclass
class RunSnapshot:
    """Everything a panel needs to look like a finished run."""

    settings: dict[str, Any]
    skip_toggle_snapshot: dict[str, bool] = field(default_factory=dict)
    show_results: bool = True
    show_steps: bool = False
    report: str = ""
    results_state: dict[str, Any] | None = None
    checkpoints: tuple[StageCheckpoint, ...] = ()
    #: The stage the run was paused after (Mid-run postprocessing), so a loaded
    #: run can Continue; None for a run that was not paused.
    paused_after: str | None = None

    @property
    def stages(self) -> tuple[str, ...]:
        return tuple(item.stage for item in self.checkpoints)

    @property
    def last_tab_title(self) -> str | None:
        if not self.checkpoints:
            return None
        last = self.checkpoints[-1].stage
        for stage in STAGES:
            if stage.call == last:
                return tab_title(stage)
        return self.checkpoints[-1].title


def ensure_run_suffix(path: Path | str) -> Path:
    """``run`` becomes ``run.haemorun``; an existing suffix is left alone."""
    path = Path(path)
    if path.suffix.lower() == SUFFIX:
        return path
    if path.suffix.lower() in {".gz", ".pkl"}:
        return path
    return path.with_suffix(SUFFIX)


def default_run_path(values: Mapping[str, Any] | None) -> str:
    """Suggested filename beside the run's VTK prefix, or a local default."""
    if not values:
        return DEFAULT_FILENAME
    vtk_prefix = values.get("vtk_output_prefix")
    stem = Path(vtk_prefix).name if vtk_prefix else "haemolynx-run"
    from haemolynx.gui.stage_checkpoints import output_dir_from_prefix

    output_dir = output_dir_from_prefix(vtk_prefix)
    if output_dir is not None:
        return str(output_dir / f"{stem}{SUFFIX}")
    return f"{stem}{SUFFIX}"


def can_capture(checkpoints: StageCheckpoints | None) -> bool:
    """A viewer run has to have recorded at least one stage."""
    return bool(checkpoints is not None and checkpoints.stages)


def capture_run(
    *,
    checkpoints: StageCheckpoints,
    results: ResultLayers | None,
    settings: Mapping[str, Any],
    skip_toggle_snapshot: Mapping[str, bool] | None = None,
    show_results: bool = True,
    show_steps: bool = False,
    report: str = "",
    paused_after: str | None = None,
) -> RunSnapshot:
    """Copy the live panel's run into a snapshot. Raises if there is none."""
    if not can_capture(checkpoints):
        raise RunSnapshotError(NOTHING_TO_SAVE)
    # Stage outputs stay behind: they are this session's live volumes, and a
    # loaded run loads its skeleton from disk instead.
    records = tuple(
        replace(item, pickle_path=None, graph=copy_graph(item.graph), output=None)
        for item in checkpoints.records()
    )
    results_state = results.export_state() if results is not None else None
    return RunSnapshot(
        settings=dict(settings),
        skip_toggle_snapshot=dict(skip_toggle_snapshot or {}),
        show_results=bool(show_results),
        show_steps=bool(show_steps),
        report=str(report or ""),
        results_state=results_state,
        checkpoints=records,
        paused_after=paused_after,
    )


def write_run_snapshot(path: Path | str, snapshot: RunSnapshot) -> Path:
    """Write *snapshot* to *path* (gzip pickle). Returns the path used."""
    dest = ensure_run_suffix(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": FORMAT,
        "version": VERSION,
        "settings": snapshot.settings,
        "skip_toggle_snapshot": snapshot.skip_toggle_snapshot,
        "show_results": snapshot.show_results,
        "show_steps": snapshot.show_steps,
        "report": snapshot.report,
        "results_state": snapshot.results_state,
        "checkpoints": snapshot.checkpoints,
        "paused_after": snapshot.paused_after,
    }
    with gzip.open(dest, "wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("Wrote HaemoLynx run snapshot to %s", dest)
    return dest


_GZIP_MAGIC = b"\x1f\x8b"


def read_run_snapshot(path: Path | str) -> RunSnapshot:
    """Load a file written by :func:`write_run_snapshot`."""
    source = Path(path)
    # Decide by the file's own header rather than by trying gzip and falling
    # back: a gzip file that fails to unpickle would otherwise be reported with
    # the fallback's "invalid load key '\x1f'", hiding what actually went wrong.
    try:
        with source.open("rb") as handle:
            compressed = handle.read(2) == _GZIP_MAGIC
    except OSError as error:
        raise RunSnapshotError(f"Could not read {source}: {error}") from error
    try:
        with (gzip.open(source, "rb") if compressed else source.open("rb")) as handle:
            payload = pickle.load(handle)
    except Exception as error:
        raise RunSnapshotError(
            f"Could not read {source}: {type(error).__name__}: {error}"
        ) from error
    if not isinstance(payload, dict) or payload.get("format") != FORMAT:
        raise RunSnapshotError(f"{source} is not a HaemoLynx run snapshot.")
    version = payload.get("version")
    if version != VERSION:
        raise RunSnapshotError(
            f"{source} is run-snapshot version {version!r}; this HaemoLynx "
            f"reads version {VERSION}."
        )
    raw = payload.get("checkpoints") or ()
    checkpoints = tuple(
        replace(item, pickle_path=None) if isinstance(item, StageCheckpoint) else item
        for item in raw
    )
    if any(not isinstance(item, StageCheckpoint) for item in checkpoints):
        raise RunSnapshotError(f"{source} has a checkpoint that is not a stage snapshot.")
    checkpoints = with_post_process_checkpoint(checkpoints)
    return RunSnapshot(
        settings=dict(payload.get("settings") or {}),
        skip_toggle_snapshot=dict(payload.get("skip_toggle_snapshot") or {}),
        show_results=bool(payload.get("show_results", True)),
        show_steps=bool(payload.get("show_steps", False)),
        report=str(payload.get("report") or ""),
        results_state=payload.get("results_state"),
        checkpoints=checkpoints,
        paused_after=_paused_after(payload.get("paused_after"), checkpoints),
    )


def _paused_after(paused_after: str | None, checkpoints: Sequence[StageCheckpoint]) -> str | None:
    """Where a loaded run was paused, if Continue can still pick it up.

    A run pauses for Post processing after its solve. One saved paused when
    Post processing came before Haemodynamics has no solve to continue from;
    "Run from this stage" on Haemodynamics picks that one up instead.
    """
    if paused_after is None or any(item.stage == "solve" for item in checkpoints):
        return paused_after
    logger.info(
        "This run was paused before Haemodynamics, where Post processing used "
        "to be; run it on from the Haemodynamics tab."
    )
    return None


def with_post_process_checkpoint(
    checkpoints: Sequence[StageCheckpoint],
) -> tuple[StageCheckpoint, ...]:
    """*checkpoints* in today's stage order, with one for ``post_process``
    wherever the run went past that stage.

    Post processing comes after Haemodynamics. A run saved while it came
    between Diameters and Haemodynamics recorded it before the solve: that
    checkpoint goes -- its edits are in every checkpoint after it -- and for
    a run that stopped before Haemodynamics, its edited network becomes
    Diameters', so "Run from this stage" on Haemodynamics starts from it.

    A run that went past the stage without recording it (saved before it
    existed, or just upgraded) gets the solve's checkpoint standing in for
    it: an unedited network passes Post processing untouched, so that is
    Post processing's end-of-tab state -- the one "Run from this stage" on
    Perturbations needs.
    """
    records = list(checkpoints)
    stages = [item.stage for item in records]
    if "post_process" in stages and (
        "solve" not in stages or stages.index("post_process") < stages.index("solve")
    ):
        edited = records.pop(stages.index("post_process"))
        stages.remove("post_process")
        if "solve" not in stages and "assign_diameters" in stages:
            at = stages.index("assign_diameters")
            records[at] = replace(records[at], graph=edited.graph)
    if "post_process" in stages or "solve" not in stages:
        return tuple(records)
    order = [stage.call for stage in STAGES if stage.call]
    later = order[order.index("post_process") + 1:]
    if not any(stage in later for stage in stages):
        return tuple(records)
    solved = records[stages.index("solve")]
    title = next(stage.title for stage in STAGES if stage.call == "post_process")
    stand_in = replace(solved, stage="post_process", title=title)
    at = stages.index("solve") + 1
    return (*records[:at], stand_in, *records[at:])


def replay_groups(snapshot: RunSnapshot) -> tuple[Any, ...]:
    """Layer groups in pipeline order, then any extra recorded stages."""
    order = [stage.call for stage in STAGES if stage.call]
    by_stage = {item.stage: item.group for item in snapshot.checkpoints}
    groups = [by_stage[name] for name in order if name in by_stage]
    extras = [
        item.group
        for item in snapshot.checkpoints
        if item.stage not in order
    ]
    return tuple(groups + extras)


def apply_snapshot_to_checkpoints(
    checkpoints: StageCheckpoints, snapshot: RunSnapshot
) -> None:
    """Replace the panel's in-memory checkpoints with the snapshot's."""
    checkpoints.replace_all(snapshot.checkpoints)


def apply_snapshot_to_results(
    results: ResultLayers, snapshot: RunSnapshot
) -> None:
    """Put ResultLayers back to how it was at the end of the saved run."""
    if snapshot.results_state:
        results.load_state(snapshot.results_state)
        return
    if snapshot.checkpoints:
        last = snapshot.checkpoints[-1]
        results.load_state(
            {
                "graph": last.graph,
                "canonical_graph": last.graph,
                "voxel_size_zyx": last.voxel_size_zyx,
                "geometry_shown": last.geometry_shown,
                "emitted": last.emitted,
                "show_steps": snapshot.show_steps,
                "thick_vessel_mask": last.thick_vessel_mask,
                "raw_segmented_image": last.raw_segmented_image,
            }
        )
        return
    results.reset()


def write_resume_artefacts(
    snapshot: RunSnapshot,
    settings: Mapping[str, Any] | None,
    checkpoints: StageCheckpoints,
) -> None:
    """Make sure the skeleton ``.npy`` is on disk so Run-from still works.

    A loaded run has no stage outputs to hand a resumed run, so that run
    loads its skeleton. Its graph is handed over from the checkpoint, so
    ``{stem}_graph.pkl`` -- graph building's own output -- is not touched.
    """
    located = _stem_and_output_dir(settings)
    if located is None:
        return
    stem, output_dir = located
    output_dir.mkdir(parents=True, exist_ok=True)
    existed = skeleton_resume_path(output_dir, stem).is_file()
    skeleton_path = ensure_skeleton_artefact(replay_groups(snapshot), output_dir, stem)
    if skeleton_path is not None and not existed:
        checkpoints.remember_path(skeleton_path)


