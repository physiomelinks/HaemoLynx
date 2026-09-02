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

from haemolynx.gui.results import ResultLayers, StageLayers
from haemolynx.gui.stage_checkpoints import (
    GRAPH_RESUME_STAGES,
    SKIP_FOR_RESUME,
    StageCheckpoints,
    can_revert_from,
    checkpoint_pickle_path,
    graph_resume_path,
    previous_tab,
    restore_message,
    revert_target_stage,
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
    assert plan.tab_title == "4. Boundaries"
    assert [group.stage for group in plan.groups] == [
        "skeletonise",
        "build_network",
        "assign_boundaries",
    ]
    # Later checkpoints are dropped so a second revert cannot jump forward again.
    assert "assign_diameters" not in checkpoints.stages
    assert checkpoints.has("assign_boundaries")


def test_plan_restore_writes_graph_pkl_and_names_skip_settings(tmp_path):
    checkpoints = StageCheckpoints()
    graph = a_graph(resistance=1.0)
    results = built(graph)
    settings = _settings(tmp_path)
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
    assert plan.skip_settings == SKIP_FOR_RESUME
    assert plan.graph_path == graph_resume_path(tmp_path / "out", "stack")
    assert plan.graph_path.is_file()
    with plan.graph_path.open("rb") as handle:
        restored = pickle.load(handle)
    assert restored.edges[0, 1, 0]["resistance"] == 1.0


def test_plan_restore_from_input_tab_is_impossible():
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
        graph_path=tmp_path / "stack_graph.pkl",
        skip_settings=SKIP_FOR_RESUME,
    )
    message = restore_message(plan)
    assert "4. Boundaries" in message
    assert "stack_graph.pkl" in message
    assert "do_skeletonize" in message


def test_graph_resume_stages_cover_every_post_topology_stage():
    """A revert from any later tab must be able to seed `{stem}_graph.pkl`."""
    assert "build_network" in GRAPH_RESUME_STAGES
    assert "solve" in GRAPH_RESUME_STAGES
    assert "export_results" in GRAPH_RESUME_STAGES
