"""A real run reports the progress a progress bar is drawn from.

`tests/test_pipeline_progress.py` stubs every stage out, so it pins the
reporting and nothing else. This one runs the pipeline for real on the smallest
committed fixture, because the thing a stub cannot check is that graph building
fires its fifteen steps through the reporter it was handed. That a run with no
callback still works is what every other integration test here does already.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from haemolynx.graph import STEP_LABELS
from haemolynx.pipeline import default_schema, resolve_settings, run_pipeline_stages
from haemolynx.pipeline.stages import BUILD_TAIL_STEPS, SKELETONISE_STEPS
from haemolynx.pipeline.progress import (
    STAGE_FINISHED,
    STAGE_STARTED,
    STAGES,
    STEP,
    ProgressEvent,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_TIFF = REPO_ROOT / "tests" / "data" / "seven_vessel_noisy_3d.tif"

STAGE_NAMES = [stage.call for stage in STAGES if stage.call]


def _settings(tmp_path: Path) -> dict:
    """The fixture run, with the thresholds the other integration tests use.

    The defaults are sized for a real stack and clean this 50-voxel fixture away
    to nothing, so these are the same values `test_image_to_model_pipeline` runs
    it with.
    """
    return resolve_settings(
        {
            "input_path": FIXTURE_TIFF,
            "plot_dir": tmp_path / "plots",
            "output_dir": tmp_path,
            "statistics": False,
            "visualize_results": False,
            "visualize_vtk": False,
            "do_equiv_resistance_calculation": False,
            "skeleton_closing_radius": 1,
            "skeleton_bridge_gap_size": 1,
            "skeleton_min_branch_length": 3,
            "skeleton_max_bridge_distance": 2,
            "skeleton_component_connectivity": 3,
            "skeleton_min_component_percent": 1.0,
            "min_stub_length": 3.0,
            "inlet_node_selection_method": "coordinates",
            "outlet_node_selection_method": "coordinates",
            "inlet_node_coordinates": [(5.0, 5.0, 5.0)],
            "outlet_node_coordinates": [(42.0, 42.0, 42.0)],
            "inlet_nodes": [],
            "outlet_nodes": [],
            "verbose_logging": False,
        },
        schema=default_schema(),
        config_path=None,
    )


@pytest.mark.integration
@pytest.mark.slow
def test_a_real_run_reports_every_stage_and_every_graph_step(tmp_path):
    events: list[ProgressEvent] = []

    graph = run_pipeline_stages(
        _settings(tmp_path), default_schema(), progress=events.append
    )

    assert graph is not None
    started = [event.stage for event in events if event.kind == STAGE_STARTED]
    finished = [event.stage for event in events if event.kind == STAGE_FINISHED]
    assert started == finished == STAGE_NAMES
    steps = [event for event in events if event.kind == STEP]
    assert {event.stage for event in steps} == {"skeletonise", "build_network"}
    graph_steps = [event for event in steps if event.stage == "build_network"]
    labels = [event.step for event in graph_steps]
    # The thirteen topology steps, then the finish's and saving and checking
    # the graph: every step the stage takes, each counted up front.
    assert labels[: len(STEP_LABELS)] == list(STEP_LABELS)
    assert labels[-len(BUILD_TAIL_STEPS):] == list(BUILD_TAIL_STEPS)
    assert {event.step_total for event in graph_steps} == {len(labels)}
    assert [event.next_step for event in graph_steps] == [*labels[1:], None]
    # Skeletonise reports its steps too: each a point a cancelled run stops at.
    skeleton_steps = [event for event in steps if event.stage == "skeletonise"]
    assert [event.step for event in skeleton_steps] == list(SKELETONISE_STEPS)
    assert {event.step_total for event in skeleton_steps} == {len(SKELETONISE_STEPS)}
    assert graph.number_of_nodes() > 0
    assert graph.number_of_edges() > 0
