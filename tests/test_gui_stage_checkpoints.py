"""Per-stage checkpoints for reverting a napari tab without a full re-run.

Pure: no napari, no Qt. The widget tests that press the button live in
`test_gui_stage_checkpoints_widget.py`, marked `gui`.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest

from haemolynx.gui.results import ResultLayers, StageLayers, LayerSpec, SKELETON
from haemolynx.gui.stage_checkpoints import (
    GRAPH_RESUME_STAGES,
    GRAPH_SKIP_FOR_RESUME,
    SKIP_FOR_RESUME,
    StageCheckpoints,
    can_revert_from,
    checkpoint_pickle_path,
    discard_cached_artefacts,
    graph_resume_path,
    previous_tab,
    restore_message,
    revert_target_stage,
    skeleton_resume_path,
    skip_settings_for_resume,
    tab_end_stage,
)
from haemolynx.gui.tabs import tab_titles
from haemolynx.pipeline.stages import TOPOLOGY_STEP
from test_gui_results import a_graph, built, network


def _group(stage: str, title: str | None = None) -> StageLayers:
    return StageLayers(stage=stage, title=title or stage, note=f"done {stage}")


def _settings(tmp_path: Path, stem: str = "stack") -> dict:
    return {
        "input_path": tmp_path / f"{stem}.tif",
        "vtk_output_prefix": tmp_path / "out" / stem,
    }


# --- tab / stage mapping -----------------------------------------------------


def test_haemodynamics_tab_ends_at_solve_not_build_model():
    """Solve shares the Haemodynamics tab; revert must land after pressures."""
    assert tab_end_stage("6. Haemodynamics") == "solve"


def test_previous_tab_of_the_first_is_none():
    titles = tab_titles()
    assert previous_tab(titles[0]) is None
    assert previous_tab(titles[1]) == titles[0]


def test_revert_from_diameters_targets_boundaries():
    assert revert_target_stage("5. Diameters") == "assign_boundaries"


def test_revert_from_perturbations_targets_solve():
    assert revert_target_stage("7. Perturbations") == "solve"


def test_can_revert_requires_a_checkpoint_for_the_previous_tab():
    checkpoints = StageCheckpoints()
    assert can_revert_from("5. Diameters", checkpoints) is False
    results = built()
    checkpoints.record(
        "assign_boundaries", _group("assign_boundaries", "4. Boundaries"), results
    )
    assert can_revert_from("5. Diameters", checkpoints) is True


# --- recording ---------------------------------------------------------------


def test_topology_steps_are_not_recorded():
    checkpoints = StageCheckpoints()
    results = built()
    assert (
        checkpoints.record(
            f"{TOPOLOGY_STEP}prune", _group("prune"), results
        )
        is None
    )
    assert checkpoints.stages == ()


def test_record_pickles_the_graph_beside_the_run_output(tmp_path):
    checkpoints = StageCheckpoints()
    graph = a_graph(branch_order="A1")
    results = built(graph)
    settings = _settings(tmp_path)
    (tmp_path / "out").mkdir()

    checkpoint = checkpoints.record(
        "build_network",
        _group("build_network", "3. Graph"),
        results,
        settings=settings,
    )

    assert checkpoint is not None
    assert checkpoint.graph is not None
    assert checkpoint.graph is not graph
    assert checkpoint.graph.number_of_edges() == graph.number_of_edges()
    path = checkpoint_pickle_path(tmp_path / "out", "stack", "build_network")
    assert path.is_file()
    with path.open("rb") as handle:
        restored = pickle.load(handle)
    assert restored.number_of_nodes() == graph.number_of_nodes()


def test_record_keeps_going_when_the_graph_will_not_pickle(tmp_path, monkeypatch):
    checkpoints = StageCheckpoints()
    results = built()

    def boom(*_args, **_kwargs):
        raise TypeError("unpickleable")

    monkeypatch.setattr("haemolynx.gui.stage_checkpoints.pickle.dumps", boom)
    checkpoint = checkpoints.record(
        "build_network",
        _group("build_network"),
        results,
        settings=_settings(tmp_path),
    )
    assert checkpoint is not None
    assert checkpoint.graph is None
    assert checkpoint.pickle_path is None


# --- restore plan ------------------------------------------------------------


def test_plan_restore_replays_groups_through_the_previous_tab(tmp_path):
    checkpoints = StageCheckpoints()
    results = built()
    settings = _settings(tmp_path)
    (tmp_path / "out").mkdir()

    for stage, title in (
        ("skeletonise", "2. Skeletonise"),
        ("build_network", "3. Graph"),
        ("assign_boundaries", "4. Boundaries"),
        ("assign_diameters", "5. Diameters"),
    ):
        if stage == "build_network":
            results.stage_finished("build_network", network(a_graph()))
        checkpoints.record(stage, _group(stage, title), results, settings=settings)

    plan = checkpoints.plan_restore("5. Diameters", settings=settings)

    assert plan is not None
    assert plan.stage == "assign_boundaries"
    assert plan.tab_title == "5. Diameters"
    assert plan.start_from == "assign_diameters"
    assert [group.stage for group in plan.groups] == [
        "skeletonise",
        "build_network",
        "assign_boundaries",
    ]
    # Later checkpoints are dropped so a second run-from cannot jump forward.
    assert "assign_diameters" not in checkpoints.stages
    assert checkpoints.has("assign_boundaries")


def test_plan_restore_writes_graph_pkl_and_skips_graph_building(tmp_path):
    """Without a skeleton artefact, only graph building is skipped — not skeletonise.

    Preflight errors if do_skeletonize is off and the .npy is missing; resume
    must not put the panel in that state.
    """
    checkpoints = StageCheckpoints()
    graph = a_graph(resistance=1.0)
    results = built(graph)
    settings = _settings(tmp_path)
    settings["use_fwhm_edge_diameters"] = True
    (tmp_path / "out").mkdir()
    checkpoints.record(
        "assign_diameters",
        _group("assign_diameters", "5. Diameters"),
        results,
        settings=settings,
    )

    plan = checkpoints.plan_restore("6. Haemodynamics", settings=settings)

    assert plan is not None
    assert plan.stage == "assign_diameters"
    assert plan.skip_settings == ("do_graph_building", "do_fwhm_measurement")
    assert plan.graph_path == graph_resume_path(tmp_path / "out", "stack")
    assert plan.graph_path.is_file()
    with plan.graph_path.open("rb") as handle:
        restored = pickle.load(handle)
    assert restored.edges[0, 1, 0]["resistance"] == 1.0


def test_plan_restore_writes_skeleton_npy_so_both_skip_toggles_are_safe(tmp_path):
    """Resume from a graph stage must leave preflight happy for both skip toggles."""
    checkpoints = StageCheckpoints()
    graph = a_graph(resistance=1.0)
    results = built(graph)
    settings = _settings(tmp_path)
    (tmp_path / "out").mkdir()
    skeleton = np.zeros((2, 3, 4), dtype=bool)
    skeleton_group = StageLayers(
        stage="skeletonise",
        title="2. Skeletonise",
        layers=(
            LayerSpec(kind="labels", name=SKELETON, data=skeleton, scale=(1.0, 1.0, 1.0)),
        ),
    )
    checkpoints.record("skeletonise", skeleton_group, results, settings=settings)
    checkpoints.record(
        "build_network",
        _group("build_network", "3. Graph"),
        results,
        settings=settings,
    )
    checkpoints.record(
        "assign_boundaries",
        _group("assign_boundaries", "4. Boundaries"),
        results,
        settings=settings,
    )

    plan = checkpoints.plan_restore("4. Boundaries", settings=settings)

    assert plan is not None
    assert plan.tab_title == "4. Boundaries"
    assert plan.skip_settings == GRAPH_SKIP_FOR_RESUME
    skel_path = skeleton_resume_path(tmp_path / "out", "stack")
    assert skel_path.is_file()
    assert np.array_equal(np.load(skel_path), skeleton)


def test_revert_from_every_tab_selects_the_restored_predecessor_tab():
    """Run-from on tab K names K as the tab to stay on, and needs end of M."""
    titles = tab_titles()
    for index in range(1, len(titles)):
        current = titles[index]
        restored = titles[index - 1]
        assert previous_tab(current) == restored
        assert revert_target_stage(current) == tab_end_stage(restored)
        from haemolynx.gui.stage_checkpoints import tab_start_stage

        plan_tab = current
        assert tab_start_stage(current) is not None
        assert plan_tab == current


def test_skip_settings_for_resume_requires_skeleton_before_disabling_skeletonize():
    assert skip_settings_for_resume(graph_written=False, skeleton_ready=True) == ()
    assert skip_settings_for_resume(graph_written=True, skeleton_ready=False) == (
        "do_graph_building",
    )
    assert skip_settings_for_resume(graph_written=True, skeleton_ready=True) == (
        GRAPH_SKIP_FOR_RESUME
    )


def test_skip_settings_for_resume_keeps_fwhm_when_diameters_are_already_on_the_graph():
    assert skip_settings_for_resume(
        graph_written=True,
        skeleton_ready=True,
        target="assign_boundaries",
        use_fwhm_edge_diameters=True,
    ) == GRAPH_SKIP_FOR_RESUME
    assert skip_settings_for_resume(
        graph_written=True,
        skeleton_ready=False,
        target="assign_diameters",
        use_fwhm_edge_diameters=False,
    ) == ("do_graph_building",)
    assert skip_settings_for_resume(
        graph_written=True,
        skeleton_ready=False,
        target="assign_diameters",
        use_fwhm_edge_diameters=True,
    ) == ("do_graph_building", "do_fwhm_measurement")
    assert skip_settings_for_resume(
        graph_written=True,
        skeleton_ready=True,
        target="assign_diameters",
        use_fwhm_edge_diameters=True,
    ) == SKIP_FOR_RESUME


def test_skip_settings_for_start_from_diameters_does_not_disable_fwhm():
    """Re-running Diameters should remeasure; Haemodynamics must not."""
    assert skip_settings_for_resume(
        graph_written=True,
        skeleton_ready=True,
        start_from="assign_diameters",
        use_fwhm_edge_diameters=True,
    ) == GRAPH_SKIP_FOR_RESUME
    assert skip_settings_for_resume(
        graph_written=True,
        skeleton_ready=True,
        start_from="build_haemodynamic_model",
        use_fwhm_edge_diameters=True,
    ) == SKIP_FOR_RESUME


def test_tab_start_stage_of_haemodynamics_is_build_model_not_solve():
    from haemolynx.gui.stage_checkpoints import tab_start_stage

    assert tab_start_stage("6. Haemodynamics") == "build_haemodynamic_model"
    assert tab_start_stage("5. Diameters") == "assign_diameters"


def test_frozen_checkpoints_do_not_record(tmp_path):
    checkpoints = StageCheckpoints()
    results = built()
    checkpoints.freeze()
    assert (
        checkpoints.record(
            "build_network",
            _group("build_network"),
            results,
            settings=_settings(tmp_path),
        )
        is None
    )
    assert checkpoints.stages == ()
    checkpoints.unfreeze()
    assert (
        checkpoints.record("build_network", _group("build_network"), results)
        is not None
    )


def test_output_dir_from_prefix_rejects_a_bare_filename():
    from haemolynx.gui.stage_checkpoints import output_dir_from_prefix

    assert output_dir_from_prefix(None) is None
    assert output_dir_from_prefix(".") is None
    assert output_dir_from_prefix("stack") is None
    assert output_dir_from_prefix(Path("out") / "stack") == Path("out")
    checkpoints = StageCheckpoints()
    assert checkpoints.plan_restore("1. Input") is None


def test_apply_to_results_restores_the_remembered_graph():
    checkpoints = StageCheckpoints()
    graph = a_graph(branch_order="C0")
    results = built(graph)
    checkpoints.record("build_network", _group("build_network"), results)
    fresh = ResultLayers()

    checkpoints.apply_to_results(fresh, checkpoints.get("build_network"))

    assert fresh._graph is not None
    assert fresh._graph is not graph
    assert fresh._graph.edges[0, 1, 0]["branch_order"] == "C0"
    assert fresh.emitted == results.emitted


def test_restore_message_mentions_the_resume_graph(tmp_path):
    plan = SimpleNamespace(
        title="4. Boundaries",
        tab_title="5. Diameters",
        start_from="assign_diameters",
        graph_path=tmp_path / "stack_graph.pkl",
        skip_settings=SKIP_FOR_RESUME,
    )
    message = restore_message(plan)
    assert "5. Diameters" in message
    assert "stack_graph.pkl" in message
    assert "do_skeletonize" in message
    assert "do_skeletonize" in message


def test_graph_resume_stages_cover_every_post_topology_stage():
    """A revert from any later tab must be able to seed `{stem}_graph.pkl`."""
    assert "build_network" in GRAPH_RESUME_STAGES
    assert "solve" in GRAPH_RESUME_STAGES
    assert "export_results" in GRAPH_RESUME_STAGES


def test_discard_cached_artefacts_removes_graph_and_checkpoints_not_skeleton(
    tmp_path,
):
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    stem = "stack"
    graph_path = graph_resume_path(output_dir, stem)
    graph_path.write_bytes(b"graph")
    cp1 = checkpoint_pickle_path(output_dir, stem, "build_network")
    cp1.write_bytes(b"cp")
    cp2 = checkpoint_pickle_path(output_dir, stem, "assign_boundaries")
    cp2.write_bytes(b"cp2")
    skel_path = skeleton_resume_path(output_dir, stem)
    skel_path.write_bytes(b"skeleton")
    unrelated = output_dir / "other.pkl"
    unrelated.write_bytes(b"other")

    removed = discard_cached_artefacts(output_dir, stem)

    assert set(removed) == {graph_path, cp1, cp2}
    assert not graph_path.is_file()
    assert not cp1.is_file()
    assert not cp2.is_file()
    assert skel_path.is_file()
    assert unrelated.is_file()


def test_discard_cached_artefacts_ignores_missing_files(tmp_path):
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    assert discard_cached_artefacts(output_dir, "stack") == ()


def test_discard_cached_artefacts_for_settings_covers_vtk_and_input_stems(
    tmp_path,
):
    from haemolynx.gui.stage_checkpoints import (
        discard_cached_artefacts_for_settings,
        stems_for_cached_artefacts,
    )

    output_dir = tmp_path / "out"
    output_dir.mkdir()
    settings = {
        "input_path": tmp_path / "HaemoLynx_image.tif",
        "vtk_output_prefix": tmp_path / "out" / "stack",
    }
    graph_resume_path(output_dir, "stack").write_bytes(b"g")
    graph_resume_path(output_dir, "HaemoLynx_image").write_bytes(b"g2")
    checkpoint_pickle_path(output_dir, "stack", "build_network").write_bytes(b"c")

    assert set(stems_for_cached_artefacts(settings)) == {"stack"}
    removed = discard_cached_artefacts_for_settings(settings)
    assert graph_resume_path(output_dir, "stack") in removed
    assert graph_resume_path(output_dir, "HaemoLynx_image") not in removed
    assert checkpoint_pickle_path(output_dir, "stack", "build_network") in removed
    assert not graph_resume_path(output_dir, "stack").is_file()
    assert graph_resume_path(output_dir, "HaemoLynx_image").is_file()
    assert not checkpoint_pickle_path(output_dir, "stack", "build_network").is_file()
