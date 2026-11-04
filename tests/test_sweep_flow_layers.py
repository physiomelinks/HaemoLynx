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
    PericyteSites,
    SweepFlowGrid,
    build_sweep_flow_grid,
    edge_pericyte_sites,
)
from haemolynx.gui.results import (  # noqa: E402
    ResultLayers,
    perturbation_layer_names,
    perturbation_pericyte_layer_name,
)
from haemolynx.pipeline import PerturbationResult, PerturbationRun  # noqa: E402
from haemolynx.visualization.perturbation_plots import wants_napari_flow_layer  # noqa: E402

from test_gui_results import a_perturbation_run, solved_graph  # noqa: E402
from test_perturbation_stage import (  # noqa: E402
    DILATION_SWEEP,
    EDGE_LENGTH_UM,
    LENGTH_SWEEP,
    PRESSURE_AND_PERICYTE_SWEEP,
    PRESSURE_SWEEP,
    SPACING_SWEEP,
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
    assert sweep_nodes not in names  # a grid with no node pressures has no nodes to draw
    vessels, nodes = perturbation_layer_names("art_dilate_20")
    assert vessels in names and nodes in names
    sweep_spec = next(spec for spec in group.layers if spec.name == sweep_vessels)
    assert sweep_spec.sweep is not None
    assert sweep_spec.kind == "vectors"
    assert sweep_spec.colour_by == "flow_abs_log10", "as the baseline's vessels are"


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


def test_a_sweeps_bridge_is_dashed_and_follows_the_slider():
    """A zero-resistance bridge is drawn dashed on a sweep's vessels too, and
    each of its dashes takes the bridge's own flow at the grid point."""
    from haemolynx.graph import IS_ZERO_RESISTANCE

    graph = solved_graph(flow=9.0)
    list(graph.edges(keys=True, data=True))[1][3][IS_ZERO_RESISTANCE] = True
    sweep_flows = _sweep_for_graph(graph, n_points=3)
    group = ResultLayers().stage_finished(
        "run_perturbations",
        a_perturbation_run(PerturbationResult(
            name="dilate_grid", type="pericyte_dilation_sweep", graph=graph,
            sweep_flows=sweep_flows,
        )),
    )
    vessels = group.layers[0]
    owner = np.asarray(vessels.segment_owner)
    assert len(owner) == len(vessels.data) > graph.number_of_edges()
    edge = np.asarray(vessels.features["edge_index"])
    np.testing.assert_array_equal(edge, np.asarray(vessels.sweep_edge_index)[owner])
    np.testing.assert_array_equal(np.asarray(vessels.features[IS_ZERO_RESISTANCE]), edge == 1)
    np.testing.assert_allclose(
        np.asarray(vessels.features["flow_abs"], dtype=float),
        sweep_flows.flow_abs_at(0)[edge],
    )


def _turning_sweep(graph):
    """Three grid points on *graph*: every vessel's flow doubling, then the
    second vessel's turning round; the node pressures rising with them."""
    n_edges, nodes = graph.number_of_edges(), list(graph.nodes)
    signed = np.tile(np.arange(1, n_edges + 1, dtype=float) * 1e-12, (3, 1))
    signed[1] *= 2.0
    signed[2, 1] *= -1.0
    return SweepFlowGrid(
        axis_names=("dilation_percent",),
        axis_values={"dilation_percent": np.asarray([0, 10, 20])},
        flow_abs=np.abs(signed),
        flow_signed=signed,
        node_pressure=np.outer([1.0, 2.0, 3.0], np.arange(len(nodes), dtype=float) * 100.0),
        node_list=tuple(nodes),
    )


def _sweep_layers(graph, sweep_flows, **settings):
    group = ResultLayers(settings=settings).stage_finished(
        "run_perturbations",
        a_perturbation_run(PerturbationResult(
            name="dilate_grid", type="pericyte_dilation_sweep", graph=graph,
            sweep_flows=sweep_flows,
        )),
    )
    return {spec.name: spec for spec in group.layers}


def test_a_sweeps_nodes_take_each_grid_points_pressures():
    """A sweep had no nodes layer, so its Nodes box stayed grey; its nodes now
    show each grid point's pressures as its slider moves, on one colour range
    for the whole grid."""
    graph = solved_graph()
    sweep_flows = _turning_sweep(graph)
    layers = _sweep_layers(graph, sweep_flows)

    nodes = layers[perturbation_layer_names("dilate_grid")[1]]
    assert nodes.kind == "points" and nodes.visible is False
    assert nodes.layer_set == "dilate_grid" and nodes.colour_by == "pressure"
    rows = nodes.sweep_rows
    assert len(rows) == 3
    by_node = {node: index for index, node in enumerate(sweep_flows.node_list)}
    for row in range(3):
        data, features = rows.at(row)
        np.testing.assert_array_equal(data, nodes.data)
        expected = [sweep_flows.node_pressure[row, by_node[node]] for node in features["node_id"]]
        np.testing.assert_array_equal(features["pressure"], expected)
    np.testing.assert_array_equal(nodes.features["pressure"], rows.at(0)[1]["pressure"])
    assert nodes.contrast_limits == (0.0, float(sweep_flows.node_pressure.max()))


def test_a_sweeps_arrows_are_a_solved_networks_at_every_grid_point():
    """A sweep had no flow arrows. At each grid point its arrows are the ones
    a network solved to that point's flows would draw: the second vessel's
    turns round at the last."""
    from haemolynx.gui.results import perturbation_flow_direction_layer_name
    from haemolynx.visualization.flow_direction import flow_direction_vectors

    graph = solved_graph()
    sweep_flows = _turning_sweep(graph)
    layers = _sweep_layers(graph, sweep_flows)

    arrows = layers[perturbation_flow_direction_layer_name("dilate_grid")]
    assert arrows.kind == "vectors" and arrows.visible is False
    assert arrows.colour_by == "flow_dir_rgb"
    for row in range(3):
        solved = graph.copy()
        for (*_e, data), signed in zip(solved.edges(keys=True, data=True), sweep_flows.flow_signed[row]):
            data["flow_signed"], data["flow_abs"] = float(signed), abs(float(signed))
        expected_vectors, expected = flow_direction_vectors(solved)
        data, features = arrows.sweep_rows.at(row)
        np.testing.assert_allclose(data, expected_vectors)
        assert set(features) == set(expected)
        for name, values in expected.items():
            np.testing.assert_allclose(features[name], values, err_msg=name)
    # The turned vessel's arrow points the other way.
    first, last = arrows.sweep_rows.at(0)[0], arrows.sweep_rows.at(2)[0]
    np.testing.assert_allclose(last[1, 1], -first[1, 1])


def test_a_sweeps_arrows_keep_the_axis_columns_out_when_asked():
    from haemolynx.gui.results import FLOW_DIR_COLUMNS, perturbation_flow_direction_layer_name

    graph = solved_graph()
    layers = _sweep_layers(graph, _turning_sweep(graph), flow_direction_colouring=False)
    arrows = layers[perturbation_flow_direction_layer_name("dilate_grid")]
    assert not set(arrows.features) & FLOW_DIR_COLUMNS
    assert not set(arrows.sweep_rows.at(2)[1]) & FLOW_DIR_COLUMNS


def test_a_real_sweep_has_nodes_and_arrows_for_every_grid_point(tmp_path):
    from haemolynx.gui.results import perturbation_flow_direction_layer_name

    result = _run(tmp_path, [PRESSURE_AND_PERICYTE_SWEEP]).results[0]
    group = ResultLayers().stage_finished(
        "run_perturbations", PerturbationRun(results=[result], output_dir=tmp_path)
    )
    layers = {spec.name: spec for spec in group.layers}
    nodes = layers[perturbation_layer_names(result.name)[1]]
    arrows = layers[perturbation_flow_direction_layer_name(result.name)]
    points = result.sweep_flows.n_points
    assert len(nodes.sweep_rows) == len(arrows.sweep_rows) == points
    # Each inlet pressure moves the nodes; every vessel carries flow at each.
    inlet = list(result.graph.nodes).index(0)
    pressures = [nodes.sweep_rows.at(row)[1]["pressure"][inlet] for row in range(points)]
    assert len(set(pressures)) == len(result.sweep_flows.axis_values["inlet_pressure_pa"])
    assert all(len(arrows.sweep_rows.at(row)[0]) == result.graph.number_of_edges() for row in range(points))


def test_sweeps_want_a_napari_flow_layer():
    assert wants_napari_flow_layer("pericyte_dilation_sweep")
    assert wants_napari_flow_layer("pressure_and_arteriole_sweep")
    assert wants_napari_flow_layer("arteriole_diameter_change")
    assert not wants_napari_flow_layer("none")


# --- where each grid point put its pericytes ---------------------------------


def _sites_by_edge(sites: PericyteSites) -> dict[int, list[float]]:
    by_edge: dict[int, list[float]] = {}
    for index, centre in zip(sites.edge_index.tolist(), sites.arc_length_um.tolist()):
        by_edge.setdefault(index, []).append(centre)
    return by_edge


def test_edge_pericyte_sites_flatten_each_edges_centres_in_edge_order():
    graph = solved_graph()
    edges = list(graph.edges(keys=True, data=True))
    edges[0][3]["pericyte_centers_um"] = [1.0, 2.0]
    edges[2][3]["pericyte_centers_um"] = [3.0]
    edges[1][3]["pericyte_centers_um"] = []

    sites = edge_pericyte_sites(graph)

    assert len(sites) == 3
    assert sites.edge_index.tolist() == [0, 0, 2]
    assert sites.arc_length_um.tolist() == [1.0, 2.0, 3.0]
    assert len(edge_pericyte_sites(solved_graph())) == 0


def test_a_grid_refuses_pericyte_sites_that_do_not_match_its_points():
    one = PericyteSites(edge_index=np.asarray([0]), arc_length_um=np.asarray([1.0]))
    with pytest.raises(ValueError, match="pericyte_sites has 1 entries"):
        SweepFlowGrid(
            axis_names=("dilation_percent",),
            axis_values={"dilation_percent": np.asarray([0, 10])},
            flow_abs=np.ones((2, 2)),
            pericyte_sites=(one,),
        )


def test_a_spacing_sweep_records_where_each_grid_point_put_its_pericytes(tmp_path):
    """The sites move with the spacing; the graph the sweep keeps has none."""
    result = _run(tmp_path, [SPACING_SWEEP]).results[0]
    assert result.error is None, result.error
    sweep = result.sweep_flows

    close, wide = (_sites_by_edge(sweep.pericyte_sites_at(i)) for i in (0, 1))

    # Spacing 50 then 100 µm, a 40 µm constriction centred 20 µm in, on each
    # of the four vessels the fixture's factors narrow.
    assert sorted(close) == sorted(wide) == [0, 1, 2, 3]
    assert close[1] == [20.0 + 50.0 * k for k in range(8)]
    assert wide[1] == [20.0, 120.0, 220.0, 320.0]
    assert all(
        not data.get("pericyte_centers_um")
        for *_edge, data in result.graph.edges(keys=True, data=True)
    )


def test_a_length_sweep_moves_the_first_site_with_the_length(tmp_path):
    result = _run(tmp_path, [LENGTH_SWEEP]).results[0]
    assert result.error is None, result.error

    short, long_ = (_sites_by_edge(result.sweep_flows.pericyte_sites_at(i)) for i in (0, 1))

    assert short[1] == [10.0, 110.0, 210.0, 310.0]  # 20 µm long
    assert long_[1] == [20.0, 120.0, 220.0, 320.0]  # 40 µm long


def test_a_dilation_and_pressure_sweep_places_each_dilations_sites_once(tmp_path):
    sweep = _run(tmp_path, [PRESSURE_AND_PERICYTE_SWEEP]).results[0].sweep_flows

    # 2 dilations x 2 pressures: the pressures at one dilation share its sites.
    assert sweep.pericyte_sites_at(0, 0) is sweep.pericyte_sites_at(0, 1)
    assert sweep.pericyte_sites_at(1, 0) is sweep.pericyte_sites_at(1, 1)
    assert len(sweep.pericyte_sites_at(1, 1)) > 0


def test_a_pressure_only_sweep_places_no_pericytes(tmp_path):
    sweep = _run(tmp_path, [PRESSURE_SWEEP]).results[0].sweep_flows
    assert sweep.pericyte_sites is None
    assert sweep.pericyte_sites_at(0) is None


def test_a_sweeps_pericyte_layer_holds_every_grid_points_sites(tmp_path):
    result = _run(tmp_path, [SPACING_SWEEP]).results[0]
    group = ResultLayers().stage_finished(
        "run_perturbations", PerturbationRun(results=[result], output_dir=tmp_path)
    )
    name = perturbation_pericyte_layer_name(result.name)
    (pericytes,) = [spec for spec in group.layers if spec.name == name]

    assert pericytes.kind == "points"
    assert pericytes.visible is False
    assert pericytes.layer_set == result.name
    # It follows its network's sliders rather than growing its own.
    assert pericytes.sweep is None
    assert len(pericytes.sweep_points) == result.sweep_flows.n_points
    first, second = pericytes.sweep_points
    np.testing.assert_array_equal(pericytes.data, first[0])
    assert len(first[0]) == 32 and len(second[0]) == 16
    # The fixture's vessels run along the last axis, end to end.
    capillary = second[0][second[1]["edge_index"] == 1]
    assert capillary[:, 2].tolist() == [EDGE_LENGTH_UM + c for c in (20.0, 120.0, 220.0, 320.0)]
    # One colour per branch order across every grid point, so none changes
    # colour as the slider moves.
    assert {label for label, _colour in pericytes.colour_cycle} == {"Art1", "B01", "Ven1"}
