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
    GRAPH_RESUME_STAGES,
    StageCheckpoint,
    StageCheckpoints,
    ensure_skeleton_artefact,
    graph_resume_path,
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
) -> RunSnapshot:
    """Copy the live panel's run into a snapshot. Raises if there is none."""
    if not can_capture(checkpoints):
        raise RunSnapshotError(NOTHING_TO_SAVE)
    records = tuple(
        replace(item, pickle_path=None, graph=copy_graph(item.graph))
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
        checkpoints=checkpoints,
    )


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
            }
        )
        return
    results.reset()


def write_resume_artefacts(
    snapshot: RunSnapshot,
    settings: Mapping[str, Any] | None,
    checkpoints: StageCheckpoints,
) -> None:
    """Write ``{stem}_graph.pkl`` / skeleton ``.npy`` so Run-from still works."""
    located = _stem_and_output_dir(settings)
    if located is None:
        return
    stem, output_dir = located
    output_dir.mkdir(parents=True, exist_ok=True)
    groups = replay_groups(snapshot)
    skeleton_path = ensure_skeleton_artefact(groups, output_dir, stem)
    if skeleton_path is not None:
        checkpoints.remember_path(skeleton_path)
    graph = None
    for item in reversed(snapshot.checkpoints):
        if item.graph is not None and item.stage in GRAPH_RESUME_STAGES:
            graph = item.graph
            break
    if graph is None:
        return
    graph_path = graph_resume_path(output_dir, stem)
    try:
        with graph_path.open("wb") as handle:
            pickle.dump(graph, handle)
        checkpoints.remember_path(graph_path)
        logger.info("Wrote resumed graph from loaded run to %s", graph_path)
    except Exception:  # noqa: BLE001 - layers still restored
        logger.exception("could not write resumed graph %s", graph_path)


