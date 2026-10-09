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


def test_a_damaged_run_file_reports_the_real_error_not_the_gzip_header(tmp_path):
    """A gzip file that fails to unpickle must say why, not "invalid load key '\\x1f'"."""
    import gzip
    import pickle

    path = tmp_path / "cut.haemorun"
    data = gzip.compress(pickle.dumps({"format": "x", "blob": b"0" * 10_000}))
    path.write_bytes(data[: len(data) // 2])
    with pytest.raises(RunSnapshotError) as caught:
        read_run_snapshot(path)
    message = str(caught.value)
    assert "invalid load key" not in message
    assert "EOFError" in message or "Compressed file ended" in message


def test_an_uncompressed_run_file_still_loads(tmp_path):
    import pickle

    path = tmp_path / "plain.haemorun"
    path.write_bytes(pickle.dumps({"format": "not ours"}))
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


def test_a_saved_run_leaves_the_live_stage_outputs_behind(tmp_path):
    """The stored volumes and network are this session's; a loaded run has
    none and loads its skeleton instead (see write_resume_artefacts)."""
    from haemolynx.pipeline.stages import SegmentedInputs

    checkpoints, results, settings = _recorded(tmp_path)
    checkpoints.record(
        "segment", _group("segment"), results, settings=settings,
        output=SegmentedInputs(image_path=tmp_path / "stack.tif", output_dir=tmp_path / "out"),
    )
    assert checkpoints.get("segment").output is not None

    snapshot = capture_run(checkpoints=checkpoints, results=results, settings=settings)

    assert all(item.output is None for item in snapshot.checkpoints)


def test_loading_a_run_leaves_graph_building_and_skeleton_files_alone(tmp_path):
    """Loading used to write the saved run's last graph over
    {stem}_graph.pkl -- graph building's own output -- and claim it and an
    existing skeleton as this session's, so the next Clear deleted both."""
    from haemolynx.gui.run_snapshot import write_resume_artefacts

    checkpoints, results, settings = _recorded(tmp_path)
    snapshot = capture_run(checkpoints=checkpoints, results=results, settings=settings)
    graph_pkl = tmp_path / "out" / "stack_graph.pkl"
    graph_pkl.write_bytes(b"what graph building made")
    skeleton = tmp_path / "out" / "stack_skeleton.npy"
    np.save(skeleton, np.zeros((2, 2, 2), dtype=bool))
    loaded = StageCheckpoints()

    write_resume_artefacts(snapshot, settings, loaded)

    assert graph_pkl.read_bytes() == b"what graph building made"
    assert loaded.session_artefact_paths == ()


def test_a_run_whose_output_folder_cannot_be_made_here_still_loads(tmp_path):
    """A run saved on a Linux server names /home/<user>/outputs, which a Mac
    cannot create; Load run crashed on it instead of skipping the optional
    resume skeleton."""
    from haemolynx.gui.run_snapshot import write_resume_artefacts

    checkpoints, results, settings = _recorded(tmp_path)
    snapshot = capture_run(checkpoints=checkpoints, results=results, settings=settings)
    blocker = tmp_path / "not_a_folder"
    blocker.write_text("a file where the output folder's parent should be")
    elsewhere = dict(settings, vtk_output_prefix=blocker / "outputs" / "stack")
    loaded = StageCheckpoints()

    note = write_resume_artefacts(snapshot, elsewhere, loaded)

    assert note is not None and str(blocker / "outputs") in note
    assert "set the output folder" in note
    assert loaded.session_artefact_paths == ()
    # Where the folder can be made, nothing is reported.
    assert write_resume_artefacts(snapshot, settings, StageCheckpoints()) is None


def test_a_run_from_another_machine_is_pointed_at_this_ones_files(tmp_path):
    """A run saved on the server names /home/<user>/... paths; loading it here
    finds the same files beside the run file and moves the outputs next to it."""
    from pathlib import PurePosixPath

    from haemolynx.gui.run_snapshot import RunSnapshot, relocate_run_paths

    resources = tmp_path / "resources" / "masks"
    resources.mkdir(parents=True)
    (tmp_path / "resources" / "stack.tiff").write_bytes(b"image")
    (resources / "Large_arteriole.tiff").write_bytes(b"mask")
    here = tmp_path / "already_here.tif"
    here.write_bytes(b"raw")
    blocker = tmp_path / "server_home"
    blocker.write_text("a file, so nothing can be created under it")
    server = blocker / "sliu205"
    settings = {
        "input_path": PurePosixPath(server / "data" / "stack.tiff"),
        "large_arteriole_mask_path": str(server / "segs" / "Large_arteriole.tiff"),
        "fwhm_raw_tiff_path": here,
        "ilastik_classifier_path": server / "classifiers" / "nowhere.ilp",
        "vtk_output_prefix": server / "outputs" / "run",
        "base_plot_dir": server / "project" / "outputs",
        "sweep_output_dir": tmp_path / "writable" / "sweep",
        "verbose_logging": True,
    }
    snapshot = RunSnapshot(settings=settings, results_state={"settings": dict(settings)})

    moved = relocate_run_paths(snapshot, tmp_path / "run.haemorun")

    got = snapshot.settings
    assert Path(got["input_path"]) == tmp_path / "resources" / "stack.tiff"
    assert isinstance(got["input_path"], Path)  # a path stays a path, a string a string
    assert got["large_arteriole_mask_path"] == str(resources / "Large_arteriole.tiff")
    assert got["fwhm_raw_tiff_path"] == here                      # already works
    assert got["ilastik_classifier_path"] == server / "classifiers" / "nowhere.ilp"  # nothing to find
    assert got["vtk_output_prefix"] == tmp_path / "outputs" / "run"
    assert got["base_plot_dir"] == tmp_path / "outputs"
    assert got["sweep_output_dir"] == tmp_path / "writable" / "sweep"  # can be made here
    assert snapshot.results_state["settings"]["vtk_output_prefix"] == tmp_path / "outputs" / "run"
    assert {m.name for m in moved} == {
        "input_path", "large_arteriole_mask_path", "vtk_output_prefix", "base_plot_dir",
    }


def test_relocation_prefers_the_file_nearest_the_run(tmp_path):
    from haemolynx.gui.run_snapshot import RunSnapshot, relocate_run_paths

    (tmp_path / "a" / "b").mkdir(parents=True)
    (tmp_path / "a" / "b" / "stack.tiff").write_bytes(b"deep")
    (tmp_path / "a" / "stack.tiff").write_bytes(b"near")
    snapshot = RunSnapshot(settings={"input_path": "/nowhere/stack.tiff"})
    relocate_run_paths(snapshot, tmp_path / "run.haemorun")
    assert snapshot.settings["input_path"] == str(tmp_path / "a" / "stack.tiff")


# --- a run paused for post-processing, and runs saved before that stage existed ----------


def _paused_after_haemodynamics(tmp_path):
    checkpoints, results, settings = _recorded(tmp_path)
    for stage in ("assign_boundaries", "assign_diameters", "build_haemodynamic_model", "solve"):
        checkpoints.record(stage, _group(stage), results, settings=settings)
    return checkpoints, results, settings


def test_a_paused_run_is_saved_and_loaded_paused(tmp_path):
    checkpoints, results, settings = _paused_after_haemodynamics(tmp_path)
    snapshot = capture_run(
        checkpoints=checkpoints, results=results, settings=settings,
        paused_after="solve",
    )

    loaded = read_run_snapshot(write_run_snapshot(tmp_path / "paused", snapshot))

    assert loaded.paused_after == "solve"
    assert loaded.stages[-1] == "solve", "no Post processing stand-in for a paused run"


def test_a_run_paused_before_haemodynamics_is_not_loaded_paused(tmp_path):
    """Saved while Post processing came before Haemodynamics: there is no solve
    for Continue to pick up from, so the run is picked up from Haemodynamics."""
    checkpoints, results, settings = _recorded(tmp_path)
    for stage in ("assign_boundaries", "assign_diameters"):
        checkpoints.record(stage, _group(stage), results, settings=settings)
    snapshot = capture_run(
        checkpoints=checkpoints, results=results, settings=settings,
        paused_after="assign_diameters",
    )

    loaded = read_run_snapshot(write_run_snapshot(tmp_path / "old", snapshot))

    assert loaded.paused_after is None


def test_a_finished_run_is_not_loaded_paused(tmp_path):
    checkpoints, results, settings = _recorded(tmp_path)
    snapshot = capture_run(checkpoints=checkpoints, results=results, settings=settings)
    loaded = read_run_snapshot(write_run_snapshot(tmp_path / "done", snapshot))
    assert loaded.paused_after is None


def _through(stages):
    checkpoints = StageCheckpoints()
    results = built()
    for stage in stages:
        checkpoints.record(stage, _group(stage), results)
    return checkpoints.records()


#: A whole run's stages as they were recorded before Post processing existed.
_BEFORE_POST_PROCESSING = (
    "assign_boundaries", "assign_diameters", "build_haemodynamic_model", "solve",
    "run_perturbations", "export_results",
)


def test_a_run_saved_before_post_processing_existed_gets_its_checkpoint():
    """An unedited network passes Post processing untouched, so its end-of-tab
    state is the solve's own -- which "Run from this stage" on Perturbations
    needs."""
    from haemolynx.gui.run_snapshot import with_post_process_checkpoint

    upgraded = with_post_process_checkpoint(_through(_BEFORE_POST_PROCESSING))

    assert [c.stage for c in upgraded] == [
        "assign_boundaries", "assign_diameters", "build_haemodynamic_model", "solve",
        "post_process", "run_perturbations", "export_results",
    ]
    stand_in = upgraded[4]
    assert stand_in.title == "7. Post processing"
    assert stand_in.graph is upgraded[3].graph


def test_a_run_saved_with_post_processing_before_haemodynamics_is_reordered():
    """Its Post processing checkpoint held the network before the solve; the
    solve's, which already has the edits, stands in for it after the solve."""
    from haemolynx.gui.run_snapshot import with_post_process_checkpoint

    old = _through((
        "assign_boundaries", "assign_diameters", "post_process",
        "build_haemodynamic_model", "solve", "run_perturbations", "export_results",
    ))
    upgraded = with_post_process_checkpoint(old)

    assert [c.stage for c in upgraded] == [
        "assign_boundaries", "assign_diameters", "build_haemodynamic_model", "solve",
        "post_process", "run_perturbations", "export_results",
    ]
    assert upgraded[4].graph is upgraded[3].graph is old[4].graph


def test_an_old_pause_keeps_its_edits_on_diameters():
    """Paused before Haemodynamics with the edits brought in line: "Run from
    this stage" on Haemodynamics starts from Diameters' checkpoint, so that is
    where the edited network goes."""
    from haemolynx.gui.run_snapshot import with_post_process_checkpoint

    old = _through(("assign_boundaries", "assign_diameters", "post_process"))
    upgraded = with_post_process_checkpoint(old)

    assert [c.stage for c in upgraded] == ["assign_boundaries", "assign_diameters"]
    assert upgraded[1].graph is old[2].graph
    assert upgraded[1].group is old[1].group


def test_a_run_that_stopped_at_the_solve_or_has_the_stage_is_left_alone():
    from haemolynx.gui.run_snapshot import with_post_process_checkpoint

    for stopped in (
        ("assign_boundaries", "assign_diameters"),
        ("assign_diameters", "build_haemodynamic_model", "solve"),
        ("assign_diameters", "build_haemodynamic_model", "solve", "post_process"),
        (*_BEFORE_POST_PROCESSING[:4], "post_process", *_BEFORE_POST_PROCESSING[4:]),
    ):
        current = _through(stopped)
        assert with_post_process_checkpoint(current) == tuple(current), stopped


# --- Manual loop review's decisions ------------------------------------------------

from haemolynx.gui.post_processing import KEPT, SIDE_DELETED, LoopReview  # noqa: E402


def _reviews():
    return (
        LoopReview((1.0, 2.0, 3.0), 25.0, KEPT, time="2026-10-09T15:00:00"),
        LoopReview((4.0, 5.0, 6.0), 40.0, SIDE_DELETED, side=2, time="2026-10-09T15:01:00"),
    )


def test_a_runs_loop_reviews_are_saved_and_loaded_with_it(tmp_path):
    checkpoints, results, settings = _recorded(tmp_path)
    snapshot = capture_run(
        checkpoints=checkpoints, results=results, settings=settings, loop_reviews=_reviews()
    )

    loaded = read_run_snapshot(write_run_snapshot(tmp_path / "reviewed", snapshot))

    assert loaded.loop_reviews == _reviews()


def test_a_run_saved_before_loop_review_loads_with_none_and_a_bad_entry_is_left_out(tmp_path):
    import gzip
    import pickle

    checkpoints, results, settings = _recorded(tmp_path)
    path = write_run_snapshot(
        tmp_path / "old", capture_run(checkpoints=checkpoints, results=results, settings=settings)
    )
    with gzip.open(path, "rb") as handle:
        payload = pickle.load(handle)
    del payload["loop_reviews"]
    with gzip.open(path, "wb") as handle:
        pickle.dump(payload, handle)
    assert read_run_snapshot(path).loop_reviews == ()

    payload["loop_reviews"] = [_reviews()[0].as_dict(), {"decision": KEPT}, "garbage"]
    with gzip.open(path, "wb") as handle:
        pickle.dump(payload, handle)
    assert read_run_snapshot(path).loop_reviews == _reviews()[:1]
