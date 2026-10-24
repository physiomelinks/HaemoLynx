"""Unsolved vessels in the viewer: what is drawn for what the solve could not reach.

With ``boundary_handling`` at ``leave_unsolved`` the solve marks each vessel
solved or not (``graph.FLOW_SOLVED``). The vessels layers carry that as a
``flow_solution`` column and blank every flow-based column on the unsolved
ones, so a flow colouring leaves them out of its range (and the panel draws
them in the uncoloured grey, see ``test_gui_unsolved_vessels_widget.py``);
the flow arrows skip them. Pure: no napari, no Qt.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from haemolynx.graph import FLOW_SOLVED
from haemolynx.gui.results import (
    FLOW_SOLUTION,
    SOLVED_FLOW_COLUMNS,
    VESSEL_LABELS,
    VESSELS,
    ResultLayers,
    mask_unsolved_flow_columns,
    perturbation_layer_names,
)
from haemolynx.pipeline import PerturbationResult
from haemolynx.visualization.flow_direction import (
    edge_flow_direction_sign,
    flow_direction_vectors,
)
from test_gui_results import a_perturbation_run, built, solved_graph, spec_named

#: Per vessel along the line: its flow, and whether the solve reached it. The
#: unsolved one carries the largest flow, which must not stretch the range.
FLOWS = (1e-12, 1e-11, 5e-10)
SOLVED = (True, True, False)


def _graph():
    graph = solved_graph()
    for (_u, _v, _k, data), flow, solved in zip(graph.edges(keys=True, data=True), FLOWS, SOLVED):
        data["flow_abs"] = data["flow_signed"] = flow
        data["pressure_drop"] = flow * 1e15
        data[FLOW_SOLVED] = solved
    return graph


def _solve_layers(graph):
    return built(graph).stage_finished(
        "solve",
        SimpleNamespace(node_list=list(graph.nodes), pressure=np.array([100.0, 90.0, 80.0, 70.0])),
    )


def test_the_flow_based_columns_are_the_solves_and_not_the_geometrys():
    assert {
        "flow_abs", "flow_abs_log10", "flow_signed", "pressure_drop", "pressure_u",
        "pressure_v", "flow_dir_z", "flow_heading_deg", "discharge_haematocrit",
        "transit_time_s", "occlusion_flow_loss",
    } <= SOLVED_FLOW_COLUMNS
    assert not {"length", "diameter_um", "resistance", "branch_order", "segment_id"} & SOLVED_FLOW_COLUMNS


def test_the_vessels_layer_names_each_vessel_solved_or_unsolved():
    vessels = spec_named(_solve_layers(_graph()), VESSELS)

    assert list(vessels.features[FLOW_SOLUTION]) == ["Solved", "Solved", "Unsolved"]


@pytest.mark.parametrize("state", ["blind_end", "closed_end", "no_boundary"])
def test_a_vessel_behind_a_dead_end_is_drawn_unsolved_and_says_why(state):
    """PLAN_dead_ends.md decision D4 (Harvey, 2026-10-15): behind a dead end
    inside the tissue a vessel is unsolved, like every other vessel no
    inlet-to-outlet path runs along -- its flows blank, so the panel draws it
    grey and leaves the solve's zero out of the colour range. Its flow state
    stays a column of its own, and the hover says why."""
    from haemolynx.graph import FLOW_STATE
    from haemolynx.gui.branch_hover import FLOW_STATE_TEXT, _metric_value

    graph = _graph()
    states = ("solved", "solved", state)
    for (_u, _v, _k, data), each in zip(graph.edges(keys=True, data=True), states):
        data[FLOW_STATE] = each
    last = list(graph.edges(keys=True, data=True))[2]
    last[3]["flow_abs"] = last[3]["flow_signed"] = last[3]["pressure_drop"] = 0.0

    group = _solve_layers(graph)
    vessels = spec_named(group, VESSELS)

    assert list(vessels.features["flow_state"]) == list(states)
    assert list(spec_named(group, VESSEL_LABELS).features[FLOW_SOLUTION]) == [
        "Solved", "Solved", "Unsolved",
    ]
    for name in ("flow_abs", "flow_abs_log10", "flow_signed", "pressure_drop"):
        values = np.asarray(vessels.features[name], dtype=float)
        assert np.isnan(values[2]) and np.isfinite(values[:2]).all(), name
    assert vessels.contrast_limits == pytest.approx(FLOWS[:2])
    hover = _metric_value(graph, last[0], last[1], last[3], "flow_solution")
    assert hover == FLOW_STATE_TEXT[state] and hover.startswith("Unsolved")
    assert _metric_value(graph, last[0], last[1], last[3], "flow") is None
    # The vessels layer's own column is the hover's wording, and a sweep's
    # slider re-masks the flows it swaps in by that column.
    assert list(vessels.features[FLOW_SOLUTION]) == ["Solved", "Solved", hover]
    swapped = {**vessels.features, "flow_abs": np.array([2e-12, 3e-11, 0.0])}
    mask_unsolved_flow_columns(swapped)
    np.testing.assert_allclose(swapped["flow_abs"][:2], [2e-12, 3e-11])
    assert np.isnan(swapped["flow_abs"][2])


def test_a_flow_colouring_leaves_the_unsolved_vessel_out_of_its_range():
    group = _solve_layers(_graph())
    vessels = spec_named(group, VESSELS)

    flow = np.asarray(vessels.features["flow_abs"], dtype=float)
    np.testing.assert_allclose(flow[:2], FLOWS[:2])
    assert np.isnan(flow[2])
    for name in ("flow_abs_log10", "flow_signed", "pressure_drop", "flow_dir_z"):
        values = np.asarray(vessels.features[name], dtype=float)
        assert np.isnan(values[2]) and np.isfinite(values[:2]).all(), name
    # The geometry is no less real for being unsolved.
    assert np.asarray(vessels.features["length"], dtype=float)[2] == 10.0
    assert vessels.colour_by == "flow_abs"
    assert vessels.contrast_limits == pytest.approx(FLOWS[:2])
    # The per-vessel table the panel colours from says the same.
    assert np.isnan(np.asarray(spec_named(group, VESSEL_LABELS).features["flow_abs"])[2])


def test_before_a_solve_nothing_is_called_unsolved():
    group = built(solved_graph()).stage_finished(
        "solve", SimpleNamespace(node_list=[0, 1, 2, 3], pressure=np.zeros(4))
    )
    vessels = spec_named(group, VESSELS)

    assert list(vessels.features[FLOW_SOLUTION]) == ["", "", ""]
    assert np.isfinite(np.asarray(vessels.features["flow_abs"], dtype=float)).all()


def test_a_perturbations_vessels_blank_the_same_unsolved_vessels():
    result = PerturbationResult(name="art_dilate_20", type="arteriole_diameter_change", graph=_graph())
    group = built().stage_finished("run_perturbations", a_perturbation_run(result))
    vessels = spec_named(group, perturbation_layer_names("art_dilate_20")[0])

    assert list(vessels.features[FLOW_SOLUTION]) == ["Solved", "Solved", "Unsolved"]
    assert np.isnan(np.asarray(vessels.features["flow_abs"], dtype=float)[2])


def _sweep_result():
    from haemolynx.haemodynamics.sweep_flows import SweepFlowGrid

    graph = _graph()
    return PerturbationResult(
        name="dilate_grid",
        type="pericyte_dilation_sweep",
        graph=graph,
        sweep_flows=SweepFlowGrid(
            axis_names=("dilation_percent",),
            axis_values={"dilation_percent": np.array([0, 10])},
            flow_abs=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        ),
    )


def test_every_sweep_grid_point_blanks_the_unsolved_vessel(monkeypatch):
    from haemolynx.gui import _widget

    result = _sweep_result()
    spec = next(
        s
        for s in ResultLayers().stage_finished("run_perturbations", a_perturbation_run(result)).layers
        if s.name == perturbation_layer_names(result.name)[0]
    )
    flow = np.asarray(spec.features["flow_abs"], dtype=float)
    np.testing.assert_allclose(flow[:2], [1.0, 2.0])
    assert np.isnan(flow[2])

    # The slider swaps in the next point's flows; the unsolved one stays blank.
    layer = SimpleNamespace(features=dict(spec.features), metadata={_widget.OURS: {}})
    _widget._store_sweep_metadata(layer, spec)
    monkeypatch.setattr(_widget, "_colour_layer", lambda *a, **k: None)
    _widget._apply_sweep_index(layer, (1,))

    flow = np.asarray(layer.features["flow_abs"], dtype=float)
    np.testing.assert_allclose(flow[:2], [4.0, 5.0])
    assert np.isnan(flow[2])
    assert np.isnan(np.asarray(layer.features["flow_abs_log10"], dtype=float)[2])


def test_masking_needs_the_solution_column_and_copies_what_it_blanks():
    flow = np.array([1.0, 2.0])
    untouched = {"flow_abs": flow}
    mask_unsolved_flow_columns(untouched)
    assert untouched["flow_abs"] is flow

    columns = {
        FLOW_SOLUTION: np.array(["Solved", "Unsolved"], dtype=object),
        "flow_abs": flow,
        "length": np.array([3.0, 4.0]),
    }
    mask_unsolved_flow_columns(columns)
    np.testing.assert_array_equal(columns["flow_abs"], [1.0, np.nan])
    np.testing.assert_array_equal(columns["length"], [3.0, 4.0])
    np.testing.assert_array_equal(flow, [1.0, 2.0])


def test_an_unsolved_vessel_has_no_flow_direction_and_no_arrow():
    assert edge_flow_direction_sign({"flow_signed": -1e-12, FLOW_SOLVED: False}) is None
    assert edge_flow_direction_sign({"flow_signed": -1e-12, FLOW_SOLVED: True}) == -1
    # Not yet marked: the flow decides, as it always has.
    assert edge_flow_direction_sign({"flow_signed": 1e-12}) == 1

    vectors, features = flow_direction_vectors(_graph())
    assert len(vectors) == 2
    np.testing.assert_allclose(features["flow_abs"], FLOWS[:2])


# --- one list: the viewer and every export blank the solve's attributes alike --------


def test_the_viewer_and_every_export_blank_one_list_of_the_solves_attributes(tmp_path):
    """PLAN_dale_review_fixes.md item 5, step 1: ``graph.SOLVE_DERIVED_EDGE_ATTRIBUTES``
    is the one list. The viewer blanks every one of them on an unsolved
    vessel, and the VTK vessels file and a perturbation's edges CSV blank
    exactly the ones they write where the flow is no answer at all
    (``no_boundary``): nothing more, nothing less. *Where* differs on
    purpose: the viewer greys every unsolved vessel, while the exports keep a
    blind end's exact zeros (``graph.flow_is_undefined``)."""
    import csv

    import networkx as nx

    pv = pytest.importorskip("pyvista")
    from haemolynx.graph import FLOW_STATE, SOLVE_DERIVED_EDGE_ATTRIBUTES
    from haemolynx.gui.results import EDGE_COLUMNS, FLOW_DERIVED_EDGE_COLUMNS, OPTIONAL_EDGE_COLUMNS
    from haemolynx.pipeline.stages import (
        PERTURBATION_EDGE_ATTRIBUTES,
        PERTURBATION_EDGE_COLUMNS,
        _write_perturbation_csvs,
    )
    from haemolynx.visualization import graph_to_vtk
    from haemolynx.visualization.vtk_io import ARRAY_UNITS

    solve_list = set(SOLVE_DERIVED_EDGE_ATTRIBUTES)

    # The viewer: every one of them, and every column it reads straight off
    # the solve is one of them (bar the directions it draws from them here).
    assert solve_list <= SOLVED_FLOW_COLUMNS
    off_the_solve = {
        name for name, stage in {**EDGE_COLUMNS, **OPTIONAL_EDGE_COLUMNS}.items()
        if stage == "solve" and name != FLOW_STATE
    }
    assert off_the_solve - FLOW_DERIVED_EDGE_COLUMNS <= solve_list
    columns = {
        FLOW_SOLUTION: np.array(["Solved", "Unsolved"], dtype=object),
        **{name: np.array([0.25, 0.25]) for name in solve_list},
        **{name: np.array([5.0, 5.0]) for name in ("length", "diameter_um", "resistance", "conductance")},
    }
    mask_unsolved_flow_columns(columns)
    assert {name for name, values in columns.items() if name != FLOW_SOLUTION and np.isnan(values[1])} == solve_list

    # A solved vessel and one no inlet and outlet reach, both carrying every
    # attribute the solve writes.
    G = nx.MultiGraph()
    pos = {0: (0.0, 0.0, 0.0), 1: (10.0, 0.0, 0.0), 2: (50.0, 0.0, 0.0), 3: (60.0, 0.0, 0.0)}
    for node, p in pos.items():
        G.add_node(node, pos=np.asarray(p))
    for (u, v), state in (((0, 1), "solved"), ((2, 3), "no_boundary")):
        G.add_edge(
            u, v, key=0, voxels=[pos[u], pos[v]], length=10.0, diameter_um=5.0, diameter_source="fwhm",
            branch_order="B01", resistance=1e15, conductance=1e-15,
            **{name: 0.25 for name in solve_list}, **{FLOW_SOLVED: state == "solved", FLOW_STATE: state},
        )

    vessels = pv.read(graph_to_vtk(G, tmp_path / "run")["vessels_path"])
    numeric = {
        name: np.asarray(vessels.cell_data[name])
        for name in vessels.cell_data.keys()
        if np.issubdtype(np.asarray(vessels.cell_data[name]).dtype, np.floating)
    }
    assert not any(np.isnan(values[0]) for values in numeric.values())
    blanked = {name for name, values in numeric.items() if np.isnan(values[1])}
    # Every one the file documents is written, and blank on the second vessel.
    assert blanked == solve_list & set(ARRAY_UNITS) == solve_list & set(vessels.cell_data.keys())

    result = PerturbationResult(name="p", type="arteriole_diameter_change", graph=G)
    result.output_dir = tmp_path
    _write_perturbation_csvs(
        result,
        baseline_graph=G,
        solved={"equivalent_resistance": 1.0, "total_inlet_flow": 1.0, "total_outlet_flow": -1.0},
        baseline={"equivalent_resistance": 1.0},
        settings={"inlet_p_bc": 100.0, "outlet_p_bc": 0.0},
        overrides={},
    )
    with (tmp_path / "p_edges.csv").open(encoding="utf-8") as handle:
        solved_row, unsolved_row = csv.DictReader(handle)
    assert set(PERTURBATION_EDGE_ATTRIBUTES) <= set(PERTURBATION_EDGE_COLUMNS)
    assert "" not in solved_row.values()
    assert {column for column, value in unsolved_row.items() if value == ""} == {
        column for column, name in PERTURBATION_EDGE_ATTRIBUTES.items() if name in solve_list
    } == {"pressure_drop_pa", "flow_signed_m3_s"}
