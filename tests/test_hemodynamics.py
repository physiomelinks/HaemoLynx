"""Tests for haemodynamics module."""
import pytest
import numpy as np
import networkx as nx

from ImageLynx.haemodynamics import (
    PoiseuilleModel,
    build_conductance_matrix_from_graph,
    calc_laplacian_from_conductance_matrix,
    calc_two_point_from_laplacian_matrix_nodeID,
)

MODEL = PoiseuilleModel(constriction_length=40.0, constriction_spacing=100.0)


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
    import ImageLynx.haemodynamics.pericyte_mask as pericyte_mask
    import ImageLynx.haemodynamics.probability as probability

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
