"""`save_step_artifacts` decides whether a run writes per-step debug artefacts.

Writing a graph pickle and a full-stack overlay after each of the eleven
topology steps was 40% of the wall time of a real run, so a run only does it
when asked. The steps themselves must still be reported either way -- the
progress bar and the napari panel read those, and they are not artefacts.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from haemolynx.graph import STEP_LABELS
from haemolynx.pipeline import default_schema, resolve_settings, run_pipeline_stages
from haemolynx.pipeline.progress import STEP, ProgressEvent

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_TIFF = REPO_ROOT / "tests" / "data" / "seven_vessel_noisy_3d.tif"


def _settings(tmp_path: Path, **overrides) -> dict:
    base = {
        "input_path": FIXTURE_TIFF,
        "plot_dir": tmp_path / "plots",
        "vtk_output_prefix": tmp_path / "run",
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
    }
    base.update(overrides)
    return resolve_settings(base, schema=default_schema(), config_path=None)


def test_the_default_is_not_to_write_step_artifacts():
    assert default_schema()["save_step_artifacts"].default is False


@pytest.mark.integration
@pytest.mark.slow
def test_a_default_run_writes_no_per_step_artifacts(tmp_path):
    events: list[ProgressEvent] = []
    graph = run_pipeline_stages(
        _settings(tmp_path), default_schema(), progress=events.append
    )

    assert graph is not None
    assert list((tmp_path / "plots").glob("graph_after_*.png")) == []
    assert list(tmp_path.glob("*_graph_after_*.pkl")) == []

    # The steps are still reported: a quiet run is not an unobservable one.
    steps = [event.step for event in events if event.kind == STEP]
    assert steps == list(STEP_LABELS)


@pytest.mark.integration
@pytest.mark.slow
def test_turning_it_on_writes_one_pickle_and_two_named_plots_per_step(tmp_path):
    settings = _settings(tmp_path, save_step_artifacts=True)
    graph = run_pipeline_stages(settings, default_schema())
    assert graph is not None

    plot_dir = tmp_path / "plots"
    pickles = sorted(path.name for path in tmp_path.glob("*_graph_after_*.pkl"))
    assert len(pickles) == len(STEP_LABELS)

    for label in STEP_LABELS:
        assert (plot_dir / f"graph_after_{label}.png").is_file(), label

    # The one step whose alias differs from its label, kept for whoever looks
    # for the old filename.
    alias = plot_dir / "smart_multigraph_degree2_removal.png"
    canonical = plot_dir / "graph_after_smart_multigraph_degree2_removal_pass1.png"
    assert alias.is_file()
    assert alias.read_bytes() == canonical.read_bytes()
