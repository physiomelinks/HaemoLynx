"""Saving and loading a finished GUI pipeline run, without Qt."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from haemolynx.gui.results import ResultLayers, StageLayers
from haemolynx.gui.run_snapshot import (
    DEFAULT_FILENAME,
    FORMAT,
    NOTHING_TO_SAVE,
    SUFFIX,
    VERSION,
    RunSnapshotError,
    apply_snapshot_to_checkpoints,
    apply_snapshot_to_results,
    can_capture,
    capture_run,
    default_run_path,
    ensure_run_suffix,
    read_run_snapshot,
    replay_groups,
    write_run_snapshot,
)
from haemolynx.gui.stage_checkpoints import StageCheckpoints
from test_gui_results import a_graph, built, network


def _group(stage: str, title: str | None = None) -> StageLayers:
    return StageLayers(stage=stage, title=title or stage, note=f"done {stage}")


def _recorded(tmp_path: Path) -> tuple[StageCheckpoints, ResultLayers, dict]:
    checkpoints = StageCheckpoints()
    results = built()
    settings = {
        "input_path": tmp_path / "stack.tif",
        "vtk_output_prefix": tmp_path / "out" / "stack",
        "do_graph_building": True,
    }
    (tmp_path / "out").mkdir()
    checkpoints.record(
        "skeletonise",
        _group("skeletonise", "2. Skeletonise"),
        results,
        settings=settings,
    )
    results.stage_finished("build_network", network(a_graph(resistance=4.0)))
    checkpoints.record(
        "build_network",
        _group("build_network", "3. Graph"),
        results,
        settings=settings,
    )
    return checkpoints, results, settings


def test_ensure_run_suffix_adds_haemorun():
    assert ensure_run_suffix("run") == Path("run.haemorun")
    assert ensure_run_suffix("run.haemorun") == Path("run.haemorun")
    assert ensure_run_suffix(Path("out") / "stack") == Path("out") / "stack.haemorun"


def test_default_run_path_uses_vtk_prefix_parent(tmp_path):
    path = default_run_path(
        {"vtk_output_prefix": tmp_path / "out" / "stack"}
    )
    assert Path(path) == tmp_path / "out" / f"stack{SUFFIX}"
    assert default_run_path({}) == DEFAULT_FILENAME


def test_empty_checkpoints_cannot_be_captured():
    assert can_capture(StageCheckpoints()) is False
    with pytest.raises(RunSnapshotError, match="Nothing to save"):
        capture_run(
            checkpoints=StageCheckpoints(),
            results=ResultLayers(),
            settings={},
        )


def test_a_recorded_run_round_trips_through_a_file(tmp_path):
    checkpoints, results, settings = _recorded(tmp_path)
    snapshot = capture_run(
        checkpoints=checkpoints,
        results=results,
        settings=settings,
        skip_toggle_snapshot={"do_graph_building": True},
        show_results=True,
        show_steps=False,
        report="Finished: 4 nodes.",
    )
    assert snapshot.stages == ("skeletonise", "build_network")
    assert snapshot.last_tab_title == "3. Graph"

    path = write_run_snapshot(tmp_path / "saved", snapshot)
    assert path.suffix == SUFFIX
    loaded = read_run_snapshot(path)

    assert loaded.settings["do_graph_building"] is True
    assert loaded.skip_toggle_snapshot == {"do_graph_building": True}
    assert loaded.stages == snapshot.stages
    assert loaded.results_state is not None
    assert loaded.results_state["graph"].number_of_nodes() == 4
    assert loaded.checkpoints[-1].graph.number_of_edges() == 3
    assert loaded.report == "Finished: 4 nodes."


def test_loading_replaces_live_checkpoints_and_results(tmp_path):
    checkpoints, results, settings = _recorded(tmp_path)
    snapshot = capture_run(
        checkpoints=checkpoints, results=results, settings=settings
    )
    path = write_run_snapshot(tmp_path / "run.haemorun", snapshot)

    fresh_checks = StageCheckpoints()
    fresh_results = ResultLayers()
    apply_snapshot_to_checkpoints(fresh_checks, read_run_snapshot(path))
    apply_snapshot_to_results(fresh_results, read_run_snapshot(path))

    assert fresh_checks.stages == ("skeletonise", "build_network")
    assert fresh_results._graph is not None
    assert fresh_results._graph.number_of_nodes() == 4
    assert fresh_results.colour_options()
    groups = replay_groups(read_run_snapshot(path))
    assert [group.stage for group in groups] == ["skeletonise", "build_network"]


def test_a_wrong_file_is_rejected(tmp_path):
    path = tmp_path / "nope.haemorun"
    path.write_bytes(b"not a snapshot")
    with pytest.raises(RunSnapshotError):
        read_run_snapshot(path)


def test_a_foreign_pickle_is_rejected(tmp_path):
    import gzip
    import pickle

    path = tmp_path / "other.haemorun"
    with gzip.open(path, "wb") as handle:
        pickle.dump({"format": "something.else", "version": VERSION}, handle)
    with pytest.raises(RunSnapshotError, match="not a HaemoLynx run snapshot"):
        read_run_snapshot(path)


def test_format_and_version_are_stable():
    assert FORMAT == "haemolynx.run"
    assert VERSION == 1
    assert NOTHING_TO_SAVE.startswith("Nothing to save")


def test_result_layers_state_round_trips():
    results = built(a_graph(flow_abs=1.5))
    state = results.export_state()
    fresh = ResultLayers()
    fresh.load_state(state)
    assert fresh._graph is not None
    assert fresh._graph is not results._graph
    assert np.allclose(
        list(fresh._graph.edges(data=True))[0][2]["flow_abs"], 1.5
    )
    assert fresh.emitted == results.emitted
    assert fresh._voxel_size_zyx == results._voxel_size_zyx


def test_result_layers_state_round_trips_the_thick_vessel_mask():
    """A resumed/reloaded run must not silently disable the thick/thin
    debug toggle for a run that genuinely used thickness-gated
    skeletonisation -- results._thick_vessel_mask has no per-stage-output
    equivalent the way the skeleton array itself does, so it must travel
    through export_state/load_state explicitly."""
    thick = np.zeros((4, 4, 4), dtype=bool)
    thick[0, 0, 0] = True
    results = ResultLayers()
    results.stage_finished(
        "skeletonise",
        SimpleNamespace(
            image=np.zeros((4, 4, 4)),
            skeleton=np.zeros((4, 4, 4), dtype=bool),
            voxel_size_xyz=(1.0, 1.0, 1.0),
            voxel_size_zyx=(1.0, 1.0, 1.0),
            thick_vessel_mask=thick,
        ),
    )

    fresh = ResultLayers()
    fresh.load_state(results.export_state())

    assert fresh._thick_vessel_mask is not None
    np.testing.assert_array_equal(fresh._thick_vessel_mask, thick)


def test_apply_snapshot_to_results_restores_the_thick_vessel_mask_from_checkpoints():
    """The older, results_state-less fallback path (built straight from the
    last StageCheckpoint) must carry the mask forward too."""
    from haemolynx.gui.run_snapshot import RunSnapshot
    from haemolynx.gui.stage_checkpoints import StageCheckpoint

    thick = np.zeros((3, 3, 3), dtype=bool)
    thick[1, 1, 1] = True
    checkpoint = StageCheckpoint(
        stage="skeletonise", title="skeletonise", group=_group("skeletonise"),
        thick_vessel_mask=thick,
    )
    snapshot = RunSnapshot(settings={}, results_state=None, checkpoints=(checkpoint,))
    fresh = ResultLayers()

    apply_snapshot_to_results(fresh, snapshot)

    assert fresh._thick_vessel_mask is not None
    np.testing.assert_array_equal(fresh._thick_vessel_mask, thick)


def test_the_module_imports_no_gui():
    import ast
    import os
    import subprocess
    import sys

    repo = Path(__file__).resolve().parents[1]
    probe = (
        "import sys; import haemolynx.gui.run_snapshot; "
        "print([m for m in sys.modules if m.split('.')[0] in "
        "{'napari', 'magicgui', 'qtpy', 'PyQt6', 'PyQt5', 'PySide6'}])"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo / "src")
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, env=env, check=False
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]", result.stdout
    source = repo / "src" / "haemolynx" / "gui" / "run_snapshot.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert not imported & {"napari", "magicgui", "qtpy"}
