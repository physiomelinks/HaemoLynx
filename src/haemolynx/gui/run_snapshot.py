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
import os
import pickle
from dataclasses import dataclass, field, replace
from pathlib import Path, PurePath
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


def read_run_snapshot(path: Path | str) -> RunSnapshot:
    """Load a file written by :func:`write_run_snapshot`."""
    source = Path(path)
    payload = None
    try:
        with gzip.open(source, "rb") as handle:
            payload = pickle.load(handle)
    except Exception:
        try:
            with source.open("rb") as handle:
                payload = pickle.load(handle)
        except Exception as error:
            raise RunSnapshotError(f"Could not read {source}: {error}") from error
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
    return RunSnapshot(
        settings=dict(payload.get("settings") or {}),
        skip_toggle_snapshot=dict(payload.get("skip_toggle_snapshot") or {}),
        show_results=bool(payload.get("show_results", True)),
        show_steps=bool(payload.get("show_steps", False)),
        report=str(payload.get("report") or ""),
        results_state=payload.get("results_state"),
        checkpoints=with_post_process_checkpoint(checkpoints),
        paused_after=payload.get("paused_after"),
    )


def with_post_process_checkpoint(
    checkpoints: Sequence[StageCheckpoint],
) -> tuple[StageCheckpoint, ...]:
    """*checkpoints*, with one for ``post_process`` when a run saved before that
    stage existed went past it.

    Such a run went straight from Diameters to Haemodynamics, so Post
    processing's end-of-tab state is Diameters' own -- the checkpoint "Run from
    this stage" on Haemodynamics now needs.
    """
    stages = [item.stage for item in checkpoints]
    if "post_process" in stages or "assign_diameters" not in stages:
        return tuple(checkpoints)
    order = [stage.call for stage in STAGES if stage.call]
    later = order[order.index("post_process") + 1:]
    if not any(stage in later for stage in stages):
        return tuple(checkpoints)
    diameters = checkpoints[stages.index("assign_diameters")]
    title = next(stage.title for stage in STAGES if stage.call == "post_process")
    stand_in = replace(diameters, stage="post_process", title=title)
    at = stages.index("assign_diameters") + 1
    return (*checkpoints[:at], stand_in, *checkpoints[at:])


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
) -> str | None:
    """Make sure the skeleton ``.npy`` is on disk so Run-from still works.

    A loaded run has no stage outputs to hand a resumed run, so that run
    loads its skeleton. Its graph is handed over from the checkpoint, so
    ``{stem}_graph.pkl`` -- graph building's own output -- is not touched.

    Optional: a run saved on another machine names an output folder that may
    not exist, or not be writable, here (``/home/...`` on a Mac). The load
    still succeeds; the reason is returned for the panel to show, and None
    when the skeleton is in place or there was nothing to write.
    """
    located = _stem_and_output_dir(settings)
    if located is None:
        return None
    stem, output_dir = located
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        existed = skeleton_resume_path(output_dir, stem).is_file()
        skeleton_path = ensure_skeleton_artefact(replay_groups(snapshot), output_dir, stem)
    except OSError as error:
        logger.warning("Could not write the loaded run's skeleton to %s: %s", output_dir, error)
        return (
            f"The run's output folder {output_dir} cannot be written on this machine "
            f"({error.strerror or error}); set the output folder before running from a stage."
        )
    if skeleton_path is not None and not existed:
        checkpoints.remember_path(skeleton_path)
    return None




#: Path settings named like these are where a run writes, not files it reads.
_OUTPUT_SUFFIXES = ("_dir", "_directory", "_prefix")
#: How deep under the run file's folder a moved input file is looked for.
_SEARCH_DEPTH = 3
_SKIPPED_FOLDERS = {".git", ".venv", "venv", "__pycache__", "node_modules"}


@dataclass(frozen=True)
class RelocatedPath:
    """One path setting a loaded run had to point somewhere else."""

    name: str
    old: str
    new: str


def _can_create(path: Path) -> bool:
    """Whether *path* exists as a folder or could be made on this machine."""
    for candidate in (path, *path.parents):
        if candidate.exists():
            return candidate.is_dir() and os.access(candidate, os.W_OK)
    return False


def _find_by_name(name: str, root: Path) -> Path | None:
    """The shallowest file called *name* under *root*, at most a few folders down."""
    root = root.resolve()
    for depth in range(_SEARCH_DEPTH + 1):
        found = []
        for folder, dirs, files in os.walk(root):
            level = len(Path(folder).relative_to(root).parts)
            dirs[:] = sorted(
                d for d in dirs if not d.startswith(".") and d not in _SKIPPED_FOLDERS
            ) if level < depth else []
            if level == depth and name in files:
                found.append(Path(folder) / name)
        if found:
            return sorted(found)[0]
    return None


def _relocate_one(name: str, value: Any, run_dir: Path) -> Path | None:
    if value is None or not str(value).strip():
        return None
    path = Path(str(value)).expanduser()
    if name.endswith(_OUTPUT_SUFFIXES):
        folder = path.parent if name.endswith("_prefix") else path
        if _can_create(folder):
            return None
        outputs = run_dir / "outputs"
        return outputs if path.name == "outputs" else outputs / path.name
    if path.exists():
        return None
    return _find_by_name(path.name, run_dir)


def relocate_run_paths(snapshot: RunSnapshot, run_path: Path | str) -> list[RelocatedPath]:
    """Point a run made on another machine at this one's files and folders.

    A run saved on the server names ``/home/<user>/...`` paths. For each path
    setting that does not work here:

    * an input file (any path setting not named ``*_dir``, ``*_directory`` or
      ``*_prefix``) that is missing is replaced by the file of the same name
      nearest the run file -- in its folder or up to three folders below;
    * an output folder or prefix that cannot be created here moves into
      ``outputs/`` beside the run file, keeping its own name.

    Paths that already work, and inputs with no file of that name nearby,
    are left as they are. Changes *snapshot*'s settings (and the copy in its
    saved viewer state) in place and returns what moved.
    """
    run_dir = Path(run_path).expanduser().resolve().parent
    moved: dict[str, RelocatedPath] = {}
    state = snapshot.results_state
    copies = [snapshot.settings]
    if isinstance(state, dict) and isinstance(state.get("settings"), dict):
        copies.append(state["settings"])
    for settings in copies:
        for name, value in list(settings.items()):
            is_path_setting = "path" in name or name.endswith(_OUTPUT_SUFFIXES)
            if not isinstance(value, (str, PurePath)) or not is_path_setting:
                continue
            new = _relocate_one(name, value, run_dir)
            if new is None:
                continue
            settings[name] = new if isinstance(value, PurePath) else str(new)
            moved.setdefault(name, RelocatedPath(name, str(value), str(new)))
    return list(moved.values())
