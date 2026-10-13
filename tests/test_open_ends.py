"""Boundary conditions where vessels leave the image (haemodynamics/open_ends.py,
PLAN_dead_ends.md Phase 4): every model, every solver taking them, and the
flow states they leave."""
from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from haemolynx import graph as graph_module
from haemolynx import haemodynamics as hd
from haemolynx.graph.dead_ends import DEAD_END_KIND, LEAVES_IMAGE
from haemolynx.haemodynamics import open_ends as oe
from haemolynx.pipeline import default_schema, stages

P_IN, P_OUT = 4500.0, 1000.0
R = 1.0e15  # Pa.s/m^3, every vessel


def _network() -> nx.MultiGraph:
    """Inlet 0 on the x = 0 face, outlet 3 on the x = 99 face; vessel 1-4 runs
    out through the y = 0 face (4 leaves the image); 2-5 ends blind inside;
    6-7 is a piece of its own. Extent (10, 10, 99) um."""
    G = nx.MultiGraph()
    pos = {0: (5, 5, 0), 1: (5, 5, 30), 2: (5, 5, 60), 3: (5, 5, 99), 4: (5, 0, 30), 5: (5, 8, 60),
           6: (5, 5, 80), 7: (5, 8, 85)}
    for n, p in pos.items():
        G.add_node(n, pos=np.asarray(p, dtype=float))
    for u, v in ((0, 1), (1, 2), (2, 3), (1, 4), (2, 5), (6, 7)):
        length = float(np.linalg.norm(np.subtract(pos[u], pos[v])))
        G.add_edge(u, v, length=length, diameter_um=5.0, resistance=R, conductance=1.0 / R)
    G.graph["image_extent_um"] = (10.0, 10.0, 99.0)
    G.graph["image_voxel_size_zyx"] = (1.0, 1.0, 1.0)
    for n in (0, 3, 4):
        G.nodes[n][DEAD_END_KIND] = LEAVES_IMAGE
    G.nodes[5][DEAD_END_KIND] = "unknown"
    return G


def _solve(G, conditions, *, solver="dense"):
    conductance, nodes = hd.build_conductance_matrix_from_graph(G, solver=solver)
    matrix, names, index, extra_in, extra_out = oe.augment(conductance, nodes, conditions)
    flow = hd.solve_flow_from_conductance_matrix(
        matrix, names, inlet_p_bc=P_IN, outlet_p_bc=P_OUT,
        inlet_nodes=[0, *extra_in], outlet_nodes=[3, *extra_out], node_to_idx=index,
    )
    pressure = flow["pressure"][: len(nodes)]
    hd.set_edge_flows(G, nodes, pressure)
    return dict(zip(nodes, pressure)), conductance, nodes


def test_only_open_ends_in_a_piece_with_an_inlet_or_outlet_are_candidates():
    G = _network()
    G.nodes[7][DEAD_END_KIND] = LEAVES_IMAGE  # leaves the image, but in a piece of its own
    assert oe.open_end_candidates(G, [0], [3]) == [4]
    assert oe.open_end_candidates(G, [], []) == []


def test_closed_leaves_the_open_end_without_flow_and_says_why():
    G = _network()
    pressure, _c, _n = _solve(G, oe.open_end_conditions("closed", G, inlet_nodes=[0], outlet_nodes=[3],
                                                       inlet_p_bc=P_IN, outlet_p_bc=P_OUT))
    assert G.edges[1, 4, 0]["flow_abs"] <= 1e-12 * G.edges[0, 1, 0]["flow_abs"]
    graph_module.mark_flow_solved_edges(G, [0], [3])
    states = {(u, v): d[graph_module.FLOW_STATE] for u, v, d in G.edges(data=True)}
    assert states == {(0, 1): "solved", (1, 2): "solved", (2, 3): "solved",
                      (1, 4): "closed_end", (2, 5): "blind_end", (6, 7): "no_boundary"}


def test_terminal_resistance_drains_through_k_times_its_own_vessel():
    G = _network()
    conditions = oe.open_end_conditions("terminal_resistance", G, inlet_nodes=[0], outlet_nodes=[3],
                                        inlet_p_bc=P_IN, outlet_p_bc=P_OUT, terminal_resistance_multiple=10.0)
    assert set(conditions.to_outlet) == {4}
    assert conditions.to_outlet[4] == pytest.approx(1.0 / (10.0 * R))
    pressure, _c, _n = _solve(G, conditions)
    # Node 4 drains to the outlet pressure through 10 R; flow 1 -> 4 -> ghost.
    assert G.edges[1, 4, 0]["flow_abs"] == pytest.approx((pressure[1] - P_OUT) / (11.0 * R), rel=1e-9)
    # Kirchhoff at node 1: in from 0, out to 2 and 4.
    q01 = (pressure[0] - pressure[1]) / R
    assert q01 == pytest.approx((pressure[1] - pressure[2]) / R + G.edges[1, 4, 0]["flow_abs"], rel=1e-9)
    graph_module.mark_flow_solved_edges(G, [0], [3], conditions.nodes)
    assert G.edges[1, 4, 0][graph_module.FLOW_STATE] == "solved"
    assert G.edges[2, 5, 0][graph_module.FLOW_STATE] == "blind_end"


def test_outlet_holds_the_open_end_at_the_outlet_pressure():
    G = _network()
    conditions = oe.open_end_conditions("outlet", G, inlet_nodes=[0], outlet_nodes=[3],
                                        inlet_p_bc=P_IN, outlet_p_bc=P_OUT)
    pressure, _c, _n = _solve(G, conditions)
    assert pressure[4] == pytest.approx(P_OUT)
    assert G.edges[1, 4, 0]["flow_abs"] == pytest.approx((pressure[1] - P_OUT) / R, rel=1e-9)


def test_the_estimate_is_reproduced_exactly_by_its_conductances():
    G = _network()
    conductance, nodes = hd.build_conductance_matrix_from_graph(G)
    conditions = oe.estimate_open_end_conditions(
        G, conductance, nodes, inlet_nodes=[0], outlet_nodes=[3], inlet_p_bc=P_IN, outlet_p_bc=P_OUT, ends=[4],
    )
    assert conditions.info["converged"]
    assert len(conditions.nodes) == 1
    low, high = conditions.info["pressure_range_pa"]
    pressure, _c, _n = _solve(G, conditions)
    assert pressure[4] == pytest.approx(low, rel=1e-6)
    # Flow conserved at every node but the boundaries.
    residuals = hd.flow_conservation_residuals(G, boundary_nodes=[0, 3, 4])
    largest = max(abs(d["flow_abs"]) for *_e, d in G.edges(data=True))
    assert all(abs(r) <= 1e-9 * largest for r in residuals.values())
    # The targets came from the closed solve.
    assert P_OUT < conditions.info["pressure_target_pa"] < P_IN
    assert conditions.info["shear_target_pa"] > 0


def test_the_estimate_keeps_shear_near_its_target_where_closing_starves_a_vessel():
    """A vessel the closed solve leaves still is given flow by the estimate,
    in the direction that brings its shear stress nearest the target."""
    G = _network()
    conductance, nodes = hd.build_conductance_matrix_from_graph(G)
    conditions = oe.estimate_open_end_conditions(
        G, conductance, nodes, inlet_nodes=[0], outlet_nodes=[3], inlet_p_bc=P_IN, outlet_p_bc=P_OUT, ends=[4],
        shear_weight=10.0,
    )
    _solve(G, conditions)
    tau = G.edges[1, 4, 0]["flow_abs"] * R * 5.0 / (4.0 * G.edges[1, 4, 0]["length"])
    assert tau == pytest.approx(conditions.info["shear_target_pa"], rel=0.5)


def test_no_open_end_means_no_condition_and_the_closed_solve():
    G = _network()
    G.nodes[4][DEAD_END_KIND] = "unknown"
    for model in oe.OPEN_END_MODELS:
        conductance, nodes = hd.build_conductance_matrix_from_graph(G)
        conditions = oe.open_end_conditions(model, G, inlet_nodes=[0], outlet_nodes=[3], inlet_p_bc=P_IN,
                                            outlet_p_bc=P_OUT, conductance=conductance, node_list=nodes)
        assert not conditions
    with pytest.raises(ValueError):
        oe.open_end_conditions("open", G, inlet_nodes=[0], outlet_nodes=[3], inlet_p_bc=P_IN, outlet_p_bc=P_OUT)


def test_dense_and_sparse_agree_with_open_ends():
    for model in ("terminal_resistance", "outlet", "estimated"):
        dense_graph, sparse_graph = _network(), _network()
        conductance, nodes = hd.build_conductance_matrix_from_graph(dense_graph)
        conditions = oe.open_end_conditions(model, dense_graph, inlet_nodes=[0], outlet_nodes=[3], inlet_p_bc=P_IN,
                                            outlet_p_bc=P_OUT, conductance=conductance, node_list=nodes)
        dense, _c, _n = _solve(dense_graph, conditions, solver="dense")
        sparse, _c, _n = _solve(sparse_graph, conditions, solver="sparse")
        assert dense == pytest.approx(sparse, rel=1e-9)


def test_the_conditions_travel_with_the_graph_and_follow_its_nodes():
    G = _network()
    conditions = oe.open_end_conditions("terminal_resistance", G, inlet_nodes=[0], outlet_nodes=[3],
                                        inlet_p_bc=P_IN, outlet_p_bc=P_OUT)
    oe.keep_on_graph(G, conditions)
    again = oe.conditions_on_graph(G)
    assert again.model == "terminal_resistance" and again.to_outlet == conditions.to_outlet
    G.remove_node(4)
    assert not oe.conditions_on_graph(G)
    assert not oe.conditions_on_graph(nx.MultiGraph())


def test_open_end_flows_count_what_enters_and_leaves_through_them():
    G = _network()
    conditions = oe.open_end_conditions("terminal_resistance", G, inlet_nodes=[0], outlet_nodes=[3],
                                        inlet_p_bc=P_IN, outlet_p_bc=P_OUT)
    pressure, conductance, nodes = _solve(G, conditions)
    index = {n: i for i, n in enumerate(nodes)}
    through = oe.open_end_flows(conductance, np.array([pressure[n] for n in nodes]), index, conditions,
                                inlet_p_bc=P_IN, outlet_p_bc=P_OUT)
    assert through["in"] == 0.0
    assert through["out"] == pytest.approx(G.edges[1, 4, 0]["flow_abs"], rel=1e-9)


# --- in the pipeline ----------------------------------------------------------------


def _settings(model: str, **extra) -> dict:
    settings = default_schema().defaults()
    settings.update(
        run_haemodynamics=True, haematocrit_model="fixed", do_equiv_resistance_calculation=False,
        inlet_p_bc=P_IN, outlet_p_bc=P_OUT, inlet_nodes=[0], outlet_nodes=[3], open_end_model=model,
        haemodynamics_solver="dense", **extra,
    )
    return settings


def _run_solve(settings):
    G = _network()
    model = stages.HaemodynamicModel(graph=G)
    boundaries = stages.BoundaryNodes(inlet_nodes=[0], outlet_nodes=[3], graph=G)
    return stages.solve(settings, model, boundaries), G, boundaries


def test_the_solve_stage_applies_the_model_keeps_it_on_the_graph_and_reports_it():
    solution, G, _b = _run_solve(_settings("terminal_resistance"))
    assert G.graph[oe.GRAPH_RECORD]["model"] == "terminal_resistance"
    assert solution.statistics["open_ends"]["open_ends"] == 1
    assert solution.statistics["open_ends"]["outflow_m3_s"] == pytest.approx(G.edges[1, 4, 0]["flow_abs"])
    assert G.edges[1, 4, 0][graph_module.FLOW_SOLVED] is True
    closed, G_closed, _b = _run_solve(_settings("closed"))
    assert "open_ends" not in closed.statistics
    assert G_closed.edges[1, 4, 0][graph_module.FLOW_STATE] == "closed_end"
    # Network resistance counts the designated inlets' inflow: the open end
    # draws more of it.
    assert solution.network_resistance < closed.network_resistance


def test_a_perturbation_re_solves_with_the_baselines_open_ends():
    settings = _settings("terminal_resistance")
    _solution, G, boundaries = _run_solve(settings)
    baseline = {n: G.nodes[n]["pressure"] for n in G}
    copy = stages._perturbation_copy(G)
    solved = stages._solve_network(copy, settings, boundaries)
    assert {n: copy.nodes[n]["pressure"] for n in copy} == pytest.approx(baseline, rel=1e-9)
    assert solved["total_inlet_flow"] + solved["total_outlet_flow"] == pytest.approx(0.0, abs=1e-25)


def test_a_sweep_point_scales_with_the_drop_and_keeps_the_open_ends():
    from haemolynx.haemodynamics.pericyte_sweep import solve_sweep_point

    settings = _settings("estimated")
    _solution, G, _b = _run_solve(settings)
    baseline = {n: G.nodes[n]["pressure"] for n in G}
    unit, nodes, _iterated = solve_sweep_point(
        stages._perturbation_copy(G), settings, recompute_resistances=lambda: None, inlet_nodes=[0], outlet_nodes=[3],
    )
    at_run = unit.at(P_IN, P_OUT)
    assert len(at_run["pressure"]) == len(nodes)
    assert dict(zip(nodes, at_run["pressure"])) == pytest.approx(baseline, rel=1e-9, abs=1e-6)


def _no_flow(data) -> tuple:
    return data["flow_abs"], data["flow_signed"], data["pressure_drop"]


@pytest.mark.parametrize("model, solver, behind", [
    ("terminal_resistance", "dense", {(2, 5)}),
    ("terminal_resistance", "sparse", {(2, 5)}),
    ("closed", "dense", {(1, 4), (2, 5)}),
])
def test_vessels_behind_dead_ends_carry_exactly_no_flow_after_the_solve(model, solver, behind):
    """Nothing drives a flow along the blind branch 2-5 (nor, closed, along
    1-4), so its flows and pressure drop are exactly 0, not the 4.5e-28 m^3/s
    of rounding the pressure solve left there; its pressures stay the
    junction's."""
    from haemolynx.haemodynamics.resistance import flow_abs_log10_value

    settings = _settings(model)
    settings["haemodynamics_solver"] = solver
    _solution, G, _b = _run_solve(settings)
    edges = {(u, v): d for u, v, d in G.edges(data=True)}
    assert {e for e, d in edges.items() if d[graph_module.FLOW_STATE] in ("blind_end", "closed_end")} == behind
    for u, v in behind:
        assert _no_flow(edges[u, v]) == (0.0, 0.0, 0.0)
        assert edges[u, v]["flow_abs_log10"] == flow_abs_log10_value(0.0)
        assert P_OUT < edges[u, v]["pressure_u"] == G.nodes[u]["pressure"] < P_IN
        assert edges[u, v]["pressure_v"] == G.nodes[v]["pressure"]
    assert all(edges[e]["flow_abs"] > 0 for e in ((0, 1), (1, 2), (2, 3)))


@pytest.mark.parametrize("haematocrit_model", ["fixed", "distributed_iterative"])
def test_a_perturbation_and_a_sweep_leave_the_blind_branch_exactly_no_flow(haematocrit_model):
    """The re-solves keep the baseline's flow states, and its zeros (closed,
    dense: both re-solves left -4.5e-28 m^3/s on 2-5)."""
    from haemolynx.haemodynamics.pericyte_sweep import solve_sweep_point, sweep_rows_at_pressures

    settings = _settings("closed")
    _solution, G, boundaries = _run_solve(settings)
    settings["haematocrit_model"] = haematocrit_model
    perturbed = stages._perturbation_copy(G)
    stages._solve_network(perturbed, settings, boundaries, recompute_resistances=lambda: None)
    for u, v in ((1, 4), (2, 5)):
        assert _no_flow(perturbed.edges[u, v, 0]) == (0.0, 0.0, 0.0)
    assert perturbed.edges[1, 2, 0]["flow_abs"] > 0

    swept = stages._perturbation_copy(G)
    unit, nodes, _iterated = solve_sweep_point(
        swept, settings, recompute_resistances=lambda: None, inlet_nodes=[0], outlet_nodes=[3],
    )
    _rows, flows = sweep_rows_at_pressures(unit, swept, nodes, [P_IN, 2.0 * P_IN], P_OUT, {})
    behind = [i for i, (u, v) in enumerate(swept.edges()) if (u, v) in ((1, 4), (2, 5))]
    for point in flows:
        for name in ("flow_abs", "flow_signed", "pressure_drop"):
            assert point[name][behind].tolist() == [0.0, 0.0]


def test_the_vtk_writes_the_blind_branchs_flow_as_exactly_zero(tmp_path):
    pv = pytest.importorskip("pyvista")
    from haemolynx.visualization import graph_to_vtk

    _solution, G, _b = _run_solve(_settings("terminal_resistance"))
    vessels = pv.read(graph_to_vtk(G, tmp_path / "blind")["vessels_path"])
    states = list(vessels.cell_data["flow_state"])
    for name in ("flow_abs", "flow_signed", "pressure_drop"):
        assert vessels.cell_data[name][states.index("blind_end")] == 0.0
    assert vessels.cell_data["flow_abs"][states.index("solved")] > 0


def test_the_haematocrit_loop_solves_with_the_open_ends():
    from haemolynx.haemodynamics.haematocrit_distribution import iterate_flow_and_haematocrit

    G = _network()
    conditions = oe.open_end_conditions("terminal_resistance", G, inlet_nodes=[0], outlet_nodes=[3],
                                        inlet_p_bc=P_IN, outlet_p_bc=P_OUT)
    result = iterate_flow_and_haematocrit(
        G, recompute_resistances=lambda: None, inlet_haematocrit=0.45, inlet_p_bc=P_IN, outlet_p_bc=P_OUT,
        inlet_nodes=[0], outlet_nodes=[3], max_iterations=3, tolerance=1e-6, open_ends=conditions,
    )
    assert len(result["pressure"]) == len(result["node_list"]) == G.number_of_nodes()
    assert G.edges[1, 4, 0]["flow_abs"] > 0
    expected, _c, _n = _solve(_network(), conditions)
    assert dict(zip(result["node_list"], result["pressure"])) == pytest.approx(expected, rel=1e-9)


def test_a_perturbation_may_not_change_the_open_end_model():
    from haemolynx.haemodynamics.perturbations import INCOMPARABLE_OVERRIDES

    for name in ("open_end_model", "open_end_terminal_resistance_multiple", "open_end_estimate_shear_weight"):
        assert name in INCOMPARABLE_OVERRIDES
    # D1 (PLAN_dead_ends.md, 2026-10-12): the Fry estimate, chosen from the
    # E14.5 sensitivity report.
    assert default_schema()["open_end_model"].default == "estimated"
