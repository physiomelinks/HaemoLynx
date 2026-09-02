"""Tests for haemodynamics module."""
import pytest
import numpy as np
import networkx as nx

from haemolynx.haemodynamics import (
    PoiseuilleModel,
    build_conductance_matrix_from_graph,
    calc_laplacian_from_conductance_matrix,
    calc_two_point_from_laplacian_matrix_nodeID,
    flow_conservation_residuals,
    set_edge_flows,
    solve_flow_from_conductance_matrix,
)

# Named rather than defaulted: these pin the capillary power law's own
# calibration, and the default law is `pries` now.
MODEL = PoiseuilleModel(
    constriction_length=40.0,
    constriction_spacing=100.0,
    viscosity_law="capillary_power_law",
)


def test_calculate_viscosity():
    # Physical Pa.s, pinned at 3.0 mPa.s for a 5 um capillary, rising as vessels narrow.
    assert MODEL.calculate_viscosity(5.0) == pytest.approx(3.0e-3)
    assert MODEL.calculate_viscosity(3.0) > MODEL.calculate_viscosity(6.0)


def test_get_diameter_at_position():
    d = MODEL.get_diameter_at_position(0, 100, 5.0, 4.0)
    assert 4.0 <= d <= 5.0


def test_resistance_integrand():
    r = MODEL.resistance_integrand(10, 100, 5.0, 4.0)
    assert r > 0


def test_calculate_integrated_resistance():
    R = MODEL.calculate_integrated_resistance(50.0, 5.0, 4.0, num_points=50)
    assert R > 0
    assert R < float("inf")
    assert MODEL.calculate_integrated_resistance(0, 5, 4) == float("inf")


def test_set_poiseuille_resistances_with_constrictions(multigraph_with_branch_order):
    G = multigraph_with_branch_order.copy()
    config = {"BO1": {"d1": 6.2, "d2": 6.2}}
    out_graph, res = MODEL.set_poiseuille_resistances_with_constrictions(G, config)
    assert isinstance(out_graph, nx.MultiGraph)
    assert res["edges_set"] >= 0


def test_set_poiseuille_resistances_with_constrictions_fwhm_baseline(multigraph_with_branch_order):
    G = multigraph_with_branch_order.copy()
    G[0][1][0]["fwhm_diameter_um"] = 4.0
    out_graph, res = MODEL.set_poiseuille_resistances_with_constrictions(
        G,
        {"BO1": 6.0},
        prefer_edge_fwhm_baseline=True,
        constriction_factor_by_branch_order={"BO1": 0.8},
    )
    assert isinstance(out_graph, nx.MultiGraph)
    assert res["edges_set"] == 1
    assert res["used_fwhm_baseline"] == 1
    assert out_graph[0][1][0]["resistance"] > 0
    assert out_graph[0][1][0]["conductance"] == pytest.approx(
        1.0 / out_graph[0][1][0]["resistance"]
    )


def test_set_poiseuille_resistances_with_constrictions_fwhm_fallback(multigraph_with_branch_order):
    G = multigraph_with_branch_order.copy()
    out_graph, res = MODEL.set_poiseuille_resistances_with_constrictions(
        G,
        {"BO1": 6.0},
        prefer_edge_fwhm_baseline=True,
        constriction_factor_by_branch_order={"BO1": 0.5},
    )
    assert res["used_fwhm_baseline"] == 0
    assert res["edges_set"] == 1
    r_fallback = out_graph[0][1][0]["resistance"]

    G2 = multigraph_with_branch_order.copy()
    G2, _ = MODEL.set_poiseuille_resistances_with_constrictions(
        G2,
        # equivalent to the fallback above: d1=6.0 with constriction factor 0.5
        {"BO1": {"d1": 6.0, "d2": 3.0}},
    )
    assert np.isclose(r_fallback, G2[0][1][0]["resistance"])


def test_set_poiseuille_edge_resistances(multigraph_with_branch_order):
    G = multigraph_with_branch_order.copy()
    out_graph, res = MODEL.set_poiseuille_edge_resistances(
        G, [(0, 1)], 6.0
    )
    assert isinstance(out_graph, nx.MultiGraph)
    assert "updated" in res


def test_calc_laplacian_from_conductance_matrix():
    C = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    L = calc_laplacian_from_conductance_matrix(C)
    assert np.allclose(L, L.T)
    assert np.allclose(np.sum(L, axis=1), 0)


def test_build_conductance_matrix_from_graph():
    G = nx.MultiGraph()
    G.add_nodes_from([0, 1, 2])
    G.add_edge(0, 1, conductance=1.5)
    G.add_edge(0, 1, conductance=2.5)  # Parallel edge should be summed.
    G.add_edge(1, 2, conductance=1.0)
    G.add_edge(0, 2, conductance=-3.0)  # Non-positive values are ignored.

    C, node_list = build_conductance_matrix_from_graph(G)
    node_to_idx = {node_id: idx for idx, node_id in enumerate(node_list)}

    i0 = node_to_idx[0]
    i1 = node_to_idx[1]
    i2 = node_to_idx[2]

    assert C.shape == (3, 3)
    assert np.allclose(C, C.T)
    assert np.isclose(C[i0, i1], 4.0)
    assert np.isclose(C[i1, i2], 1.0)
    assert np.isclose(C[i0, i2], 0.0)


def test_calc_two_point_from_laplacian_matrix_nodeID():
    G = nx.MultiGraph()
    G.add_nodes_from([0, 1, 2])
    G.add_edge(0, 1, conductance=1)
    G.add_edge(1, 2, conductance=1)
    C = np.zeros((3, 3))
    C[0, 1] = C[1, 0] = 1
    C[1, 2] = C[2, 1] = 1
    L = calc_laplacian_from_conductance_matrix(C)
    R = calc_two_point_from_laplacian_matrix_nodeID(L, G, 0, 2)
    assert R > 0


def test_poiseuille_setters_are_named_for_what_they_set():
    """These functions write `resistance`/`conductance`; `weight` no longer exists.

    Guards against the old names creeping back via copy-paste from the OLD/
    scripts or from a stale branch.
    """
    import haemolynx.haemodynamics.pericyte_mask as pericyte_mask
    import haemolynx.haemodynamics.probability as probability

    for module, new_name, old_name in (
        (PoiseuilleModel, "set_poiseuille_resistances", "set_poiseuille_weights"),
        (
            PoiseuilleModel,
            "set_poiseuille_resistances_with_constrictions",
            "set_poiseuille_weights_with_constrictions",
        ),
        (
            PoiseuilleModel,
            "set_poiseuille_edge_resistances",
            "set_poiseuille_edge_weights",
        ),
        (
            pericyte_mask,
            "set_poiseuille_resistances_with_pericyte_mask",
            "set_poiseuille_weights_with_pericyte_mask",
        ),
        (
            probability,
            "set_poiseuille_resistances_with_probabilistic_periodic_constrictions",
            "set_poiseuille_weights_with_probabilistic_periodic_constrictions",
        ),
    ):
        assert hasattr(module, new_name), f"{new_name} missing"
        assert not hasattr(module, old_name), f"{old_name} should have been renamed"


def test_result_summaries_report_edges_set_not_weights_set(multigraph_with_branch_order):
    """The per-call summary key names edges, not the attribute that no longer exists."""
    G = multigraph_with_branch_order.copy()
    _G, results = MODEL.set_poiseuille_resistances(G, {"BO1": 6.0})
    assert "edges_set" in results
    assert "weights_set" not in results


@pytest.mark.parametrize("resistance_scale", [1.0, 1e16])
def test_two_point_resistance_of_series_edges_is_scale_invariant(resistance_scale):
    """Two edges in series must sum, at unit scale and at SI scale.

    Physiological conductances are ~1e-16 m^3/(Pa.s); a fixed absolute
    eigenvalue cut-off discards every mode at that scale and returns zero.
    """
    conductance_value = 1.0 / resistance_scale
    G = nx.MultiGraph()
    G.add_nodes_from([0, 1, 2])
    G.add_edge(0, 1, conductance=conductance_value)
    G.add_edge(1, 2, conductance=conductance_value)

    C, node_list = build_conductance_matrix_from_graph(G)
    L = calc_laplacian_from_conductance_matrix(C)
    R = calc_two_point_from_laplacian_matrix_nodeID(L, G, 0, 2)

    assert R == pytest.approx(2.0 * resistance_scale, rel=1e-9)


# ---------------------------------------------------------------------------
# Flow conservation on networks with disconnected or non-conductive parts.
#
# Regression for the cropped-nerve failure: edges without a conductance leave
# nodes with all-zero Laplacian rows, np.linalg.solve raises, and the old
# whole-system lstsq fallback then produced pressures that were not audited
# anywhere. The solve must instead restrict itself to what the boundary
# conditions can reach, and Kirchhoff's current law must hold exactly there.
# ---------------------------------------------------------------------------

SI_CONDUCTANCE = 1e-16  # m^3/(Pa.s), the real magnitude for a capillary


def _disconnected_network() -> nx.MultiGraph:
    """A network shaped like the cropped nerve: a conductive diamond carrying
    both boundary conditions, edges that never got a conductance, and a
    conductive component no boundary condition reaches."""
    G = nx.MultiGraph()
    G.add_edge(0, 1, conductance=2.0 * SI_CONDUCTANCE)
    G.add_edge(1, 3, conductance=1.0 * SI_CONDUCTANCE)
    G.add_edge(0, 2, conductance=1.0 * SI_CONDUCTANCE)
    G.add_edge(2, 3, conductance=3.0 * SI_CONDUCTANCE)
    G.add_edge(1, 2, conductance=0.5 * SI_CONDUCTANCE)
    # Edges that never got a conductance (no branch order -> no diameter).
    G.add_edge(3, 4)
    G.add_edge(4, 5)
    # A conductive component with no path to any boundary node.
    G.add_edge(6, 7, conductance=2.0 * SI_CONDUCTANCE)
    G.add_edge(7, 8, conductance=1.0 * SI_CONDUCTANCE)
    return G


def _solve_and_set_flows(G, *, inlet_p_bc=1000.0, outlet_p_bc=500.0):
    conductance, node_list = build_conductance_matrix_from_graph(G)
    flow = solve_flow_from_conductance_matrix(
        conductance,
        node_list,
        inlet_p_bc=inlet_p_bc,
        outlet_p_bc=outlet_p_bc,
        inlet_nodes=[0],
        outlet_nodes=[3],
    )
    set_edge_flows(G, node_list, flow["pressure"])
    return flow


def test_flow_is_conserved_despite_disconnected_parts():
    G = _disconnected_network()
    _solve_and_set_flows(G)

    max_flow = max(abs(d["flow_signed"]) for _, _, d in G.edges(data=True) if "flow_signed" in d)
    assert max_flow > 1e-18  # a real flow, not roundoff noise

    residuals = flow_conservation_residuals(G, boundary_nodes=[0, 3])
    assert set(residuals) == {1, 2, 6, 7, 8}
    for node, residual in residuals.items():
        assert abs(residual) < 1e-9 * max_flow, (
            f"Flow not conserved at node {node}: residual {residual:.3e} "
            f"vs max flow {max_flow:.3e}"
        )


def test_unreached_components_get_zero_pressure_and_zero_flow():
    G = _disconnected_network()
    flow = _solve_and_set_flows(G)

    idx = {n: i for i, n in enumerate(flow["node_list"])}
    for node in (6, 7, 8):
        assert flow["pressure"][idx[node]] == 0.0
    for u, v, data in G.edges(data=True):
        if {u, v} <= {6, 7, 8}:
            assert data["flow_signed"] == 0.0


def test_disconnected_parts_do_not_change_the_connected_solution():
    """The diamond must solve exactly as it would on its own."""
    disconnected = _disconnected_network()
    flow_full = _solve_and_set_flows(disconnected)

    diamond = nx.MultiGraph()
    for u, v, data in disconnected.edges(data=True):
        if "conductance" in data and {u, v} <= {0, 1, 2, 3}:
            diamond.add_edge(u, v, conductance=data["conductance"])
    flow_alone = _solve_and_set_flows(diamond)

    idx_full = {n: i for i, n in enumerate(flow_full["node_list"])}
    idx_alone = {n: i for i, n in enumerate(flow_alone["node_list"])}
    for node in (0, 1, 2, 3):
        assert flow_full["pressure"][idx_full[node]] == pytest.approx(
            flow_alone["pressure"][idx_alone[node]], rel=1e-12
        )


def test_singular_network_does_not_fall_back_to_lstsq(monkeypatch):
    """The old whole-system lstsq fallback must not fire for a network that is
    merely disconnected; only what the boundary conditions reach is solved."""

    def _no_lstsq(*args, **kwargs):
        raise AssertionError("np.linalg.lstsq must not be needed here")

    monkeypatch.setattr(np.linalg, "lstsq", _no_lstsq)
    G = _disconnected_network()
    _solve_and_set_flows(G)


def test_solve_warns_when_boundaries_pin_only_one_pressure(caplog):
    """Boundary nodes that all land outside the conductive part (or on one
    side of it) leave a component with a single imposed pressure: zero flow."""
    import logging

    G = nx.MultiGraph()
    G.add_edge(0, 1, conductance=SI_CONDUCTANCE)
    G.add_edge(1, 2, conductance=SI_CONDUCTANCE)
    G.add_edge(3, 4)  # the outlet (3) hangs off a non-conductive edge

    conductance, node_list = build_conductance_matrix_from_graph(G)
    with caplog.at_level(logging.WARNING, logger="haemolynx.haemodynamics.resistance"):
        flow = solve_flow_from_conductance_matrix(
            conductance,
            node_list,
            inlet_p_bc=1000.0,
            outlet_p_bc=500.0,
            inlet_nodes=[0],
            outlet_nodes=[3],
        )
    assert any("only" in rec.message and "zero" in rec.message for rec in caplog.records)

    set_edge_flows(G, node_list, flow["pressure"])
    for u, v, data in G.edges(data=True):
        if "conductance" in data:
            assert data["flow_signed"] == pytest.approx(0.0, abs=1e-30)


def test_set_edge_flows_writes_node_pressures():
    G = _disconnected_network()
    flow = _solve_and_set_flows(G)

    idx = {n: i for i, n in enumerate(flow["node_list"])}
    for node in G.nodes():
        assert G.nodes[node]["pressure"] == flow["pressure"][idx[node]]


def test_flow_conservation_residuals_reports_an_imbalance():
    """The auditor itself must flag pressures that violate Kirchhoff."""
    G = nx.MultiGraph()
    G.add_edge(0, 1, conductance=1.0)
    G.add_edge(1, 2, conductance=1.0)
    G.nodes[0]["pressure"] = 3.0
    G.nodes[1]["pressure"] = 1.0  # correct value for conservation would be 2.0
    G.nodes[2]["pressure"] = 1.0

    residuals = flow_conservation_residuals(G, boundary_nodes=[0, 2])
    assert residuals[1] == pytest.approx(-2.0)
