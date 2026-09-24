"""Perturbation disk outputs and napari layer gating."""
from __future__ import annotations

import sys
from pathlib import Path

from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from haemolynx.haemodynamics import (  # noqa: E402
    PERTURBATION_TYPES,
    SWEEP_PERTURBATION_TYPES,
    is_sweep_perturbation,
    perturbation_folder_name,
)
from haemolynx.gui.results import ResultLayers, perturbation_layer_names  # noqa: E402
from haemolynx.pipeline import PerturbationResult, PerturbationRun  # noqa: E402
from haemolynx.visualization.perturbation_plots import (  # noqa: E402
    SWEEP_AXIS_BY_TYPE,
    export_sweep_perturbation_plots,
    wants_napari_flow_layer,
)

# Reuse the stage test fixtures' network builder via a thin local copy so this
# file stays focused on outputs, not on haemodynamic numbers.
from test_perturbation_stage import (  # noqa: E402
    ARTERIOLE_DILATION,
    DILATION_SWEEP,
    SPACING_SWEEP,
    _run,
)


def test_every_declared_sweep_type_is_classified_as_a_sweep():
    declared_sweeps = {name for name in PERTURBATION_TYPES if "sweep" in name}
    assert declared_sweeps == set(SWEEP_PERTURBATION_TYPES)
    for name in declared_sweeps:
        assert is_sweep_perturbation(name)
        assert wants_napari_flow_layer(name)
    for name in PERTURBATION_TYPES:
        if name not in declared_sweeps:
            assert not is_sweep_perturbation(name)
            if name != "none":
                assert wants_napari_flow_layer(name)
            else:
                assert not wants_napari_flow_layer(name)


def test_every_sweep_type_has_axis_labelling():
    for name in SWEEP_PERTURBATION_TYPES:
        assert name in SWEEP_AXIS_BY_TYPE, name


def test_a_dilation_sweep_writes_alice_style_plots_and_keeps_geometry(tmp_path):
    run = _run(tmp_path, [DILATION_SWEEP])
    result = run.results[0]
    assert result.error is None, result.error
    assert result.graph is not None
    assert result.sweep_flows is not None
    assert result.output_dir == (
        tmp_path
        / "out"
        / perturbation_folder_name(DILATION_SWEEP["name"], DILATION_SWEEP["type"])
    )
    written = {path.name for path in result.output_dir.iterdir()}
    assert "pericyte_dilation_sweep.csv" in written
    assert "resistance_vs_pericyte_dilation.png" in written
    assert "flow_vs_pericyte_dilation.png" in written
    assert f"{DILATION_SWEEP['name']}_summary.csv" in written


def test_a_spacing_sweep_writes_axis_corrected_plots(tmp_path):
    run = _run(tmp_path, [SPACING_SWEEP])
    result = run.results[0]
    assert result.error is None, result.error
    assert result.graph is not None
    assert result.sweep_flows is not None
    written = {path.name for path in result.output_dir.iterdir()}
    assert "pericyte_spacing_sweep.csv" in written
    assert "resistance_vs_pericyte_spacing.png" in written
    assert "flow_vs_pericyte_spacing.png" in written


def test_a_non_sweep_writes_pipeline_like_artifacts_and_keeps_its_graph(tmp_path):
    run = _run(tmp_path, [ARTERIOLE_DILATION])
    result = run.results[0]
    assert result.error is None, result.error
    assert result.graph is not None
    assert result.sweep_flows is None
    assert result.output_dir == (
        tmp_path
        / "out"
        / perturbation_folder_name(
            ARTERIOLE_DILATION["name"], ARTERIOLE_DILATION["type"]
        )
    )
    written = {path.name for path in result.output_dir.iterdir()}
    assert f"{ARTERIOLE_DILATION['name']}_summary.csv" in written
    assert f"{ARTERIOLE_DILATION['name']}_edges.csv" in written
    assert f"{ARTERIOLE_DILATION['name']}_statistics.csv" in written
    assert f"{ARTERIOLE_DILATION['name']}_branch_statistics.csv" in written
    assert "node_degree_distribution.png" in written
    assert (
        "edges_and_nodes_overlay.png" in written
        or "edges_and_nodes_overlay_3d.html" in written
    )
    assert not list(result.output_dir.glob("*.vtp"))


def test_results_builder_gives_sweep_a_vectors_layer_not_a_static_pair():
    """Sweeps: one Vectors layer; non-sweeps: vessels + nodes."""
    from test_gui_results import a_perturbation_run, solved_graph
    from haemolynx.haemodynamics.sweep_flows import SweepFlowGrid
    import numpy as np

    sweep_flows = SweepFlowGrid(
        axis_names=("dilation_percent",),
        axis_values={"dilation_percent": np.asarray([0, 10])},
        flow_abs=np.ones((2, solved_graph().number_of_edges()), dtype=float),
    )
    sweep = PerturbationResult(
        name="dilate_grid",
        type="pericyte_dilation_sweep",
        graph=solved_graph(),
        sweep_flows=sweep_flows,
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
    vessels, nodes = perturbation_layer_names("art_dilate_20")
    assert vessels in names and nodes in names
    sweep_vessels, sweep_nodes = perturbation_layer_names("dilate_grid")
    assert sweep_vessels in names
    assert sweep_nodes not in names


def test_export_sweep_plots_accept_raw_rows(tmp_path):
    rows = [
        {
            "dilation_percent": 0,
            "inlet_pressure_pa": 5000,
            "equivalent_resistance": 1.0,
            "total_inlet_flow": 2.0,
        },
        {
            "dilation_percent": 10,
            "inlet_pressure_pa": 5000,
            "equivalent_resistance": 0.5,
            "total_inlet_flow": 4.0,
        },
    ]
    paths = export_sweep_perturbation_plots(
        "arteriole_diameter_sweep", rows, tmp_path
    )
    assert len(paths) == 2
    assert (tmp_path / "resistance_vs_arteriole_dilation.png").is_file()
    assert (tmp_path / "flow_vs_arteriole_dilation.png").is_file()


# --- a single re-solve recomputes everything the baseline derives from its flows -----

from test_perturbation_stage import (  # noqa: E402
    BRANCH_ORDERS,
    NON_SWEEP_TYPES_FOR_RECOMPUTE,
)


def _terminals():
    return {"inlet_nodes": [0], "outlet_nodes": [len(BRANCH_ORDERS)]}


@pytest.mark.parametrize("entry", NON_SWEEP_TYPES_FOR_RECOMPUTE, ids=lambda e: e["type"])
def test_a_single_re_solve_reruns_the_selected_network_analyses_on_its_own_flows(tmp_path, entry):
    """With inlets and outlets set, the flow-based analyses a baseline report
    runs are run again on the perturbed network -- so its CSV has them, and
    its own vessels carry the per-vessel values for its layer."""
    run = _run(tmp_path, [entry], **_terminals())

    result = run.results[0]
    assert result.error is None, result.error
    stats_csv = (result.output_dir / f"{entry['name']}_statistics.csv").read_text()
    assert "Equivalent Inlet-Outlet Resistance" in stats_csv
    assert "Occlusion" not in stats_csv  # off by default, so off here too
    edge = next(iter(result.graph.edges(keys=True, data=True)))[3]
    assert "current_flow_share" in edge
    assert "arteriolar_territory" in edge


def test_a_perturbation_honours_the_runs_measure_selection(tmp_path):
    run = _run(
        tmp_path,
        [ARTERIOLE_DILATION],
        statistics_current_flow=False,
        statistics_cyclomatic_number=False,
        **_terminals(),
    )

    result = run.results[0]
    stats_csv = (result.output_dir / f"{ARTERIOLE_DILATION['name']}_statistics.csv").read_text()
    assert "Equivalent Inlet-Outlet Resistance" not in stats_csv
    assert "Cyclomatic Number" not in stats_csv
    assert all("current_flow_share" not in d for *_e, d in result.graph.edges(keys=True, data=True))


def test_current_flow_shares_follow_the_perturbed_flow_not_the_baselines(tmp_path):
    """Resistance weighting: the share is computed from this perturbation's
    own conductances, which the arteriole dilation has changed."""
    run = _run(
        tmp_path, [ARTERIOLE_DILATION], statistics_route_weighting="resistance", **_terminals()
    )
    stats_csv = (
        run.results[0].output_dir / f"{ARTERIOLE_DILATION['name']}_statistics.csv"
    ).read_text()
    assert "Current Flow Route Weighting,resistance" in stats_csv.replace(", ", ",")


@pytest.mark.parametrize("statistics_on", [True, False])
def test_a_perturbation_writes_the_betweenness_and_community_reports_like_the_run(
    tmp_path, statistics_on
):
    """export_results writes them only with statistics on; so does a perturbation."""
    run = _run(tmp_path, [ARTERIOLE_DILATION], statistics=statistics_on)
    written = {path.name for path in run.results[0].outputs}
    for suffix in ("resistance", "edge_length", "edge_flow"):
        name = f"{ARTERIOLE_DILATION['name']}_betweenness_communities_{suffix}.json"
        assert (name in written) == statistics_on


def test_a_perturbation_reassigns_vascular_communities_on_its_own_network(tmp_path):
    run = _run(
        tmp_path,
        [ARTERIOLE_DILATION],
        compute_vascular_communities=True,
        statistics=True,
        vascular_community_weighting="flow",
    )
    result = run.results[0]
    assert all("vascular_community" in d for *_e, d in result.graph.edges(keys=True, data=True))
    assert result.summary["vascular_communities"] >= 1


def test_perturbation_statistics_scale_by_the_runs_voxel_size(tmp_path, monkeypatch):
    from haemolynx import statistics

    seen = []
    real = statistics.compute_comprehensive_vessel_statistics

    def spy(*args, **kwargs):
        seen.append(tuple(kwargs["voxel_size"]))
        return real(*args, **kwargs)

    monkeypatch.setattr(statistics, "compute_comprehensive_vessel_statistics", spy)
    from test_perturbation_stage import _model, _boundaries, _settings, SCHEMA
    from haemolynx.pipeline import run_perturbations

    model = _model()
    model.graph.graph["image_voxel_size_zyx"] = (2.0, 0.5, 0.5)
    run_perturbations(_settings(tmp_path, [ARTERIOLE_DILATION]), model, _boundaries(), SCHEMA)

    assert seen == [(2.0, 0.5, 0.5)]


def test_perturbations_use_the_baseline_images_voxel_size_for_statistics_and_plots(
    tmp_path, monkeypatch
):
    """The loaded image's voxel size wins over anything else -- never 1 um."""
    from types import SimpleNamespace

    from haemolynx import statistics
    from haemolynx.pipeline import run_perturbations
    from haemolynx.visualization import perturbation_plots
    from test_perturbation_stage import SCHEMA, _boundaries, _model, _settings

    seen = {"statistics": [], "plots": []}
    real_stats = statistics.compute_comprehensive_vessel_statistics
    real_plot = perturbation_plots.visualize_geometry_with_edge_resistance

    def spy_stats(*args, **kwargs):
        seen["statistics"].append(tuple(kwargs["voxel_size"]))
        return real_stats(*args, **kwargs)

    def spy_plot(*args, **kwargs):
        seen["plots"].append(tuple(kwargs["voxel_size"]))
        return real_plot(*args, **kwargs)

    monkeypatch.setattr(statistics, "compute_comprehensive_vessel_statistics", spy_stats)
    monkeypatch.setattr(perturbation_plots, "visualize_geometry_with_edge_resistance", spy_plot)
    model = _model()
    model.graph.graph["image_voxel_size_zyx"] = (9.0, 9.0, 9.0)  # the volume's must win
    network = SimpleNamespace(volume=SimpleNamespace(voxel_size_zyx=(2.0, 0.5, 0.25), image=None))

    run_perturbations(
        _settings(tmp_path, [ARTERIOLE_DILATION]), model, _boundaries(), SCHEMA, network=network
    )

    assert seen == {"statistics": [(2.0, 0.5, 0.25)], "plots": [(2.0, 0.5, 0.25)]}


def test_without_a_volume_the_graphs_own_voxel_size_is_used_and_1um_only_as_a_warned_last_resort(
    caplog,
):
    import logging

    import networkx as nx
    from haemolynx.pipeline.stages import _baseline_voxel_size_zyx

    G = nx.MultiGraph()
    G.graph["voxel_size"] = (3.0, 1.0, 1.0)
    assert _baseline_voxel_size_zyx(G, None) == (3.0, 1.0, 1.0)
    G.graph["image_voxel_size_zyx"] = (4.0, 1.0, 1.0)
    assert _baseline_voxel_size_zyx(G, None) == (4.0, 1.0, 1.0)

    with caplog.at_level(logging.WARNING):
        assert _baseline_voxel_size_zyx(nx.MultiGraph(), None) == (1.0, 1.0, 1.0)
    assert "assume 1 um voxels" in caplog.text


# --- sweeps: the slider moves every flow-derived column ---------------------------------


def test_sweep_direction_columns_pick_each_edges_own_direction_per_grid_point():
    import numpy as np

    from haemolynx.gui.results import sweep_direction_columns

    forward = {"flow_dir_z": np.array([1.0, 0.5, 1.0]), "flow_heading_deg": np.array([0.0, 90.0, 0.0]),
               "flow_dir_rgb": np.zeros(3)}
    backward = {"flow_dir_z": np.array([-1.0, -0.5, -1.0]), "flow_heading_deg": np.array([180.0, 270.0, 180.0]),
                "flow_dir_rgb": np.zeros(3)}

    columns = sweep_direction_columns((forward, backward), np.array([2.0, -3.0, 0.0]))

    assert columns["flow_dir_z"][:2].tolist() == [1.0, -0.5]
    assert np.isnan(columns["flow_dir_z"][2])  # no flow, no direction
    assert columns["flow_heading_deg"][:2].tolist() == [0.0, 270.0]
    assert columns["flow_dir_rgb"].tolist() == [0.0, 0.0, 0.0]


def test_a_sweep_layer_starts_with_its_first_grid_points_directions_and_log_flow(tmp_path):
    import numpy as np

    run = _run(tmp_path, [DILATION_SWEEP])
    result = run.results[0]
    layers = ResultLayers().stage_finished(
        "run_perturbations", PerturbationRun(results=[result], output_dir=tmp_path)
    ).layers
    vessels = next(spec for spec in layers if spec.name == perturbation_layer_names(result.name)[0])

    for column in ("flow_abs_log10", "flow_dir_z", "flow_heading_deg", "flow_dir_rgb"):
        assert column in vessels.features, column
    assert vessels.sweep_directions is not None
    signed0 = np.asarray(result.sweep_flows.flow_signed_at(0), dtype=float)[vessels.sweep_edge_index]
    expected_z = np.where(
        signed0 > 0,
        np.asarray(vessels.sweep_directions[0]["flow_dir_z"]),
        np.asarray(vessels.sweep_directions[1]["flow_dir_z"]),
    )
    assert np.allclose(
        np.asarray(vessels.features["flow_dir_z"]), expected_z[vessels.segment_owner], equal_nan=True
    )


def test_moving_a_sweep_slider_updates_log_flow_and_directions(tmp_path, monkeypatch):
    """Regression: the slider swapped flow_abs but left flow_abs_log10 at grid
    point 0, so colouring by it never moved; directions did not exist at all."""
    import numpy as np

    from haemolynx.gui import _widget
    from haemolynx.haemodynamics.resistance import flow_abs_log10_value

    from test_perturbation_stage import PRESSURE_SWEEP

    run = _run(tmp_path, [PRESSURE_SWEEP])
    result = run.results[0]
    first = np.asarray(result.sweep_flows.flow_abs_at(0), dtype=float)
    spec = next(
        s
        for s in ResultLayers().stage_finished(
            "run_perturbations", PerturbationRun(results=[result], output_dir=tmp_path)
        ).layers
        if s.name == perturbation_layer_names(result.name)[0]
    )
    layer = SimpleNamespace(features=dict(spec.features), metadata={})
    monkeypatch.setattr(_widget, "_is_ours", lambda _layer: True, raising=False)
    layer.metadata[_widget.OURS] = {}
    _widget._store_sweep_metadata(layer, spec)
    monkeypatch.setattr(_widget, "_colour_layer", lambda *a, **k: None)

    last = len(result.sweep_flows.axis_values[result.sweep_flows.axis_names[0]]) - 1
    _widget._apply_sweep_index(layer, (last,))

    flow = np.asarray(result.sweep_flows.flow_abs_at(last), dtype=float)[spec.sweep_edge_index]
    assert not np.allclose(flow, first[spec.sweep_edge_index], rtol=1e-6, atol=0), "the grid points must differ"
    expected_log = np.asarray([flow_abs_log10_value(v) for v in flow])[spec.segment_owner]
    assert np.allclose(np.asarray(layer.features["flow_abs_log10"]), expected_log, equal_nan=True)
    signed = np.asarray(result.sweep_flows.flow_signed_at(last), dtype=float)[spec.sweep_edge_index]
    from haemolynx.gui.results import sweep_direction_columns

    expected_dir = sweep_direction_columns(spec.sweep_directions, signed)["flow_dir_z"][spec.segment_owner]
    assert np.allclose(np.asarray(layer.features["flow_dir_z"]), expected_dir, equal_nan=True)
