"""Sweep flow retention and napari slider indexing."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from haemolynx.haemodynamics.sweep_flows import (  # noqa: E402
    SweepFlowGrid,
    build_sweep_flow_grid,
)
from haemolynx.gui.results import ResultLayers, perturbation_layer_names  # noqa: E402
from haemolynx.pipeline import PerturbationResult  # noqa: E402
from haemolynx.visualization.perturbation_plots import wants_napari_flow_layer  # noqa: E402

from test_gui_results import a_perturbation_run, solved_graph  # noqa: E402
from test_perturbation_stage import (  # noqa: E402
    DILATION_SWEEP,
    PRESSURE_AND_PERICYTE_SWEEP,
    PRESSURE_SWEEP,
    _run,
)


def _grid_1d() -> SweepFlowGrid:
    return SweepFlowGrid(
        axis_names=("dilation_percent",),
        axis_values={"dilation_percent": np.asarray([0, 10, 20])},
        flow_abs=np.asarray(
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
            ],
            dtype=float,
        ),
    )


def _grid_2d() -> SweepFlowGrid:
    # Outer dilation (2), inner pressure (3) -> 6 rows in C-order.
    return SweepFlowGrid(
        axis_names=("dilation_percent", "inlet_pressure_pa"),
        axis_values={
            "dilation_percent": np.asarray([0, 10]),
            "inlet_pressure_pa": np.asarray([4000, 5000, 6000]),
        },
        flow_abs=np.arange(12, dtype=float).reshape(6, 2),
    )


def test_1d_slider_index_picks_the_matching_flow_row():
    grid = _grid_1d()
    assert grid.flat_index(0) == 0
    assert grid.flat_index(2) == 2
    assert np.allclose(grid.flow_abs_at(1), [3.0, 4.0])


def test_2d_two_index_picks_the_matching_flow_row():
    grid = _grid_2d()
    # dilation 1, pressure 2 -> flat 1*3+2 = 5
    assert grid.flat_index(1, 2) == 5
    assert np.allclose(grid.flow_abs_at(0, 1), [2.0, 3.0])
    assert np.allclose(grid.flow_abs_at(1, 0), [6.0, 7.0])


def test_build_sweep_flow_grid_stacks_recorded_rows():
    recorded = [
        {
            "flow_abs": np.asarray([1.0, 2.0]),
            "flow_signed": np.asarray([1.0, -2.0]),
            "pressure_drop": np.asarray([0.1, 0.2]),
            "node_pressure": np.asarray([10.0, 5.0]),
        },
        {
            "flow_abs": np.asarray([3.0, 4.0]),
            "flow_signed": np.asarray([3.0, -4.0]),
            "pressure_drop": np.asarray([0.3, 0.4]),
            "node_pressure": np.asarray([11.0, 4.0]),
        },
    ]
    grid = build_sweep_flow_grid(
        axis_names=("inlet_pressure_pa",),
        axis_values={"inlet_pressure_pa": [4500, 5000]},
        recorded=recorded,
        node_list=[0, 1],
    )
    assert grid.n_points == 2
    assert grid.n_edges == 2
    assert np.allclose(grid.flow_abs_at(1), [3.0, 4.0])
    assert np.allclose(grid.node_pressure_at(0), [10.0, 5.0])


def test_a_dilation_sweep_retains_flow_fields_for_each_grid_point(tmp_path):
    run = _run(tmp_path, [DILATION_SWEEP])
    result = run.results[0]
    assert result.error is None, result.error
    assert result.graph is not None
    assert result.sweep_flows is not None
    assert result.sweep_flows.n_points == result.summary["sweep_points"] == 2
    assert result.sweep_flows.axis_names == ("dilation_percent",)
    assert result.sweep_flows.flow_abs.shape[0] == 2
    assert result.sweep_flows.flow_abs.shape[1] == result.graph.number_of_edges()
    # Distinct solves: relative change across the dilation axis is real.
    ratio = result.sweep_flows.flow_abs[1] / result.sweep_flows.flow_abs[0]
    assert np.all(np.isfinite(ratio))
    assert float(np.max(np.abs(ratio - 1.0))) > 0.01


def test_a_2d_sweep_retains_flows_indexed_by_both_axes(tmp_path):
    run = _run(tmp_path, [PRESSURE_AND_PERICYTE_SWEEP])
    result = run.results[0]
    assert result.error is None, result.error
    sweep = result.sweep_flows
    assert sweep is not None
    assert sweep.axis_names == ("dilation_percent", "inlet_pressure_pa")
    assert sweep.n_points == 4  # 2 dilations x 2 pressures
    # Last dilation, last pressure.
    last = sweep.flow_abs_at(1, 1)
    assert last.shape == (result.graph.number_of_edges(),)
    assert np.allclose(last, sweep.flow_abs[3])


def test_a_pressure_sweep_keeps_alice_plots_and_a_geometry_graph(tmp_path):
    run = _run(tmp_path, [PRESSURE_SWEEP])
    result = run.results[0]
    assert result.graph is not None
    written = {path.name for path in result.output_dir.iterdir()}
    assert "inlet_pressure_sweep.csv" in written
    assert "resistance_vs_inlet_pressure.png" in written
    assert result.sweep_flows is not None


def _sweep_for_graph(graph, *, n_points: int = 2):
    n_edges = graph.number_of_edges()
    rows = np.arange(1, n_points * n_edges + 1, dtype=float).reshape(n_points, n_edges)
    return SweepFlowGrid(
        axis_names=("dilation_percent",),
        axis_values={"dilation_percent": np.arange(n_points) * 10},
        flow_abs=rows,
    )


def test_results_builder_emits_one_vectors_layer_for_a_sweep():
    graph = solved_graph()
    sweep = PerturbationResult(
        name="dilate_grid",
        type="pericyte_dilation_sweep",
        graph=graph,
        sweep_flows=_sweep_for_graph(graph),
    )
    single = PerturbationResult(
        name="art_dilate_20",
        type="arteriole_diameter_change",
        graph=solved_graph(),
    )
    group = ResultLayers().stage_finished(
        "run_perturbations", a_perturbation_run(sweep, single)
    )
    names = [spec.name for spec in group.layers]
    sweep_vessels = perturbation_layer_names("dilate_grid")[0]
    sweep_nodes = perturbation_layer_names("dilate_grid")[1]
    assert sweep_vessels in names
    assert sweep_nodes not in names  # sweeps: Vectors only
    vessels, nodes = perturbation_layer_names("art_dilate_20")
    assert vessels in names and nodes in names
    sweep_spec = next(spec for spec in group.layers if spec.name == sweep_vessels)
    assert sweep_spec.sweep is not None
    assert sweep_spec.kind == "vectors"
    assert sweep_spec.colour_by == "flow_abs"


def test_sweep_layer_initial_flow_matches_grid_point_zero():
    graph = solved_graph(flow=9.0)
    sweep_flows = _sweep_for_graph(graph, n_points=3)
    sweep = PerturbationResult(
        name="dilate_grid",
        type="pericyte_dilation_sweep",
        graph=graph,
        sweep_flows=sweep_flows,
    )
    group = ResultLayers().stage_finished(
        "run_perturbations", a_perturbation_run(sweep)
    )
    vessels = group.layers[0]
    expected = sweep_flows.flow_abs_at(0)
    # Each edge is one segment in the fixture graph.
    assert np.allclose(np.asarray(vessels.features["flow_abs"], dtype=float), expected)


def test_sweeps_want_a_napari_flow_layer():
    assert wants_napari_flow_layer("pericyte_dilation_sweep")
    assert wants_napari_flow_layer("pressure_and_arteriole_sweep")
    assert wants_napari_flow_layer("arteriole_diameter_change")
    assert not wants_napari_flow_layer("none")
