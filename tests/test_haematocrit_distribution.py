"""Pries-Secomb bifurcation haematocrit distribution.

The rest of the haemodynamics package treats haematocrit as one scalar
shared by every edge (`viscosity.DEFAULT_HAEMATOCRIT`,
`PoiseuilleModel.haematocrit`). This tests the second model: local discharge
haematocrit computed from Pries, Ley, Claassen & Gaehtgens' (1989)
bifurcation phase-separation law -- driven by parent/daughter diameters and
the flow fraction at the bifurcation, not by bifurcation angle -- and the
outer loop that iterates it together with the flow solve to a fixed point.
"""
from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from haemolynx.haemodynamics.haematocrit_distribution import (
    _fqe_pries_secomb,
    distribute_discharge_haematocrit,
    iterate_flow_and_haematocrit,
    pries_secomb_daughter_haematocrit,
)
from haemolynx.haemodynamics.poiseuille import PoiseuilleModel
from haemolynx.haemodynamics.resistance import build_conductance_matrix_from_graph


def _directed(G: nx.MultiGraph, frm, to, *, diameter_um: float, flow: float, key: int = 0) -> None:
    """Add edge frm-to and set flow_signed so it reads frm->to.

    networkx does not guarantee an undirected MultiGraph's `.edges()`
    reports (u, v) in the order edges were added -- it is stable across
    repeated calls on the same unmutated graph (which is all
    `flow_signed`'s sign convention ever relies on in real use, since
    `set_edge_flows` writes it and `distribute_discharge_haematocrit` reads
    it from the same graph object with no topology change in between), but
    a hand-built test fixture has to check which orientation it actually
    got before deciding the sign to write.
    """
    G.add_edge(frm, to, key=key, diameter_um=diameter_um)
    actual_u, actual_v, actual_k = next(
        (u, v, k)
        for u, v, k in G.edges(keys=True)
        if {u, v} == {frm, to} and k == key
    )
    sign = 1.0 if (actual_u, actual_v) == (frm, to) else -1.0
    G[actual_u][actual_v][actual_k]["flow_signed"] = sign * float(flow)


def _hct(G: nx.MultiGraph, a, b, key: int = 0) -> float:
    return G[a][b][key]["discharge_haematocrit"]


# --- pries_secomb_daughter_haematocrit / _fqe_pries_secomb ------------------


def test_symmetric_bifurcation_splits_haematocrit_with_the_flow():
    """Equal diameters, an even flow split: no phase separation to apply."""
    h = pries_secomb_daughter_haematocrit(
        fractional_flow=0.5,
        parent_diameter_um=20.0,
        own_diameter_um=15.0,
        other_diameter_um=15.0,
        parent_haematocrit=0.45,
    )
    assert h == pytest.approx(0.45, abs=1e-9)


def test_the_narrow_low_flow_daughter_is_skimmed_below_its_own_flow_share():
    """The qualitative signature of phase separation: a daughter carrying a
    small share of the flow carries an even smaller share of the red cells
    than a naive flow-proportional split would give it."""
    fqe = _fqe_pries_secomb(
        fractional_flow=0.2,
        parent_diameter_um=20.0,
        own_diameter_um=10.0,
        other_diameter_um=15.0,
        parent_haematocrit=0.45,
    )
    assert fqe < 0.2  # skimmed: FQE below its own FQB.

    h_narrow = pries_secomb_daughter_haematocrit(
        fractional_flow=0.2,
        parent_diameter_um=20.0,
        own_diameter_um=10.0,
        other_diameter_um=15.0,
        parent_haematocrit=0.45,
    )
    assert h_narrow < 0.45  # below the parent's own haematocrit.


def test_a_flow_fraction_at_or_below_the_plasma_skimming_threshold_gets_no_red_cells():
    """Below X0 = 0.4 / D_parent, the daughter is plasma-skimmed entirely."""
    x0 = 0.4 / 20.0
    fqe = _fqe_pries_secomb(
        fractional_flow=x0 / 2.0,
        parent_diameter_um=20.0,
        own_diameter_um=10.0,
        other_diameter_um=15.0,
        parent_haematocrit=0.45,
    )
    assert fqe == 0.0
    h = pries_secomb_daughter_haematocrit(
        fractional_flow=x0 / 2.0,
        parent_diameter_um=20.0,
        own_diameter_um=10.0,
        other_diameter_um=15.0,
        parent_haematocrit=0.45,
    )
    assert h == 0.0


def test_a_flow_fraction_at_or_above_one_minus_the_threshold_gets_every_red_cell():
    x0 = 0.4 / 20.0
    fqe = _fqe_pries_secomb(
        fractional_flow=1.0 - x0 / 2.0,
        parent_diameter_um=20.0,
        own_diameter_um=15.0,
        other_diameter_um=10.0,
        parent_haematocrit=0.45,
    )
    assert fqe == 1.0


def test_a_daughter_with_no_flow_carries_no_haematocrit():
    h = pries_secomb_daughter_haematocrit(
        fractional_flow=0.0,
        parent_diameter_um=20.0,
        own_diameter_um=10.0,
        other_diameter_um=15.0,
        parent_haematocrit=0.45,
    )
    assert h == 0.0


def test_the_computed_haematocrit_never_reaches_one():
    """viscosity.py's Pries formulas raise for haematocrit >= 1.0 -- this
    must stay a valid input to them regardless of how extreme the split is."""
    h = pries_secomb_daughter_haematocrit(
        fractional_flow=0.999,
        parent_diameter_um=3.5,  # a narrow parent pushes X0 up, sharpening the split.
        own_diameter_um=15.0,
        other_diameter_um=4.0,
        parent_haematocrit=0.6,
    )
    assert 0.0 <= h < 1.0


@pytest.mark.parametrize("bad_diameter", [0.0, -1.0])
def test_non_positive_diameters_are_refused(bad_diameter):
    with pytest.raises(ValueError, match="positive"):
        _fqe_pries_secomb(
            fractional_flow=0.5,
            parent_diameter_um=bad_diameter,
            own_diameter_um=10.0,
            other_diameter_um=10.0,
            parent_haematocrit=0.45,
        )


# --- distribute_discharge_haematocrit ---------------------------------------


def test_a_diverging_bifurcation_conserves_red_cell_mass():
    """0 -> 1 (parent), 1 -> 2 (narrow daughter), 1 -> 3 (wide daughter)."""
    G = nx.MultiGraph()
    _directed(G, 0, 1, diameter_um=20.0, flow=1.0)
    _directed(G, 1, 2, diameter_um=10.0, flow=0.2)
    _directed(G, 1, 3, diameter_um=15.0, flow=0.8)

    diag = distribute_discharge_haematocrit(G, inlet_haematocrit=0.45)
    assert diag == {"dead_edges": 0, "compound_junctions": 0, "unresolved_edges": 0}

    h_parent = _hct(G, 0, 1)
    h_narrow = _hct(G, 1, 2)
    h_wide = _hct(G, 1, 3)
    assert h_parent == pytest.approx(0.45)
    assert h_narrow < h_parent < h_wide  # the qualitative skimming signature.
    # Mass balance: parent's RBC flux equals the sum of both daughters'.
    assert 1.0 * h_parent == pytest.approx(0.2 * h_narrow + 0.8 * h_wide)


def test_a_converging_confluence_mixes_by_flow_weighted_average():
    """0 -> 2 (flow 0.3), 1 -> 2 (flow 0.7), 2 -> 3."""
    G = nx.MultiGraph()
    _directed(G, 0, 2, diameter_um=10.0, flow=0.3)
    _directed(G, 1, 2, diameter_um=10.0, flow=0.7)
    _directed(G, 2, 3, diameter_um=15.0, flow=1.0)

    distribute_discharge_haematocrit(G, inlet_haematocrit=0.45)
    # Both inlets seed at the same default, so the mix is trivially that too.
    assert _hct(G, 2, 3) == pytest.approx(0.45)

    # A genuine mix: one inflow to the confluence goes through its own
    # upstream bifurcation first, so it arrives at a different haematocrit
    # than the second, independent source feeding the same node.
    G3 = nx.MultiGraph()
    _directed(G3, 0, 1, diameter_um=20.0, flow=1.0)
    _directed(G3, 1, 2, diameter_um=10.0, flow=0.2)
    _directed(G3, 1, 3, diameter_um=15.0, flow=0.8)
    _directed(G3, 4, 2, diameter_um=8.0, flow=0.5)   # a second, independent source into node 2
    _directed(G3, 2, 5, diameter_um=12.0, flow=0.7)
    distribute_discharge_haematocrit(G3, inlet_haematocrit=0.45)
    h_narrow_daughter = _hct(G3, 1, 2)
    h_second_source = _hct(G3, 4, 2)
    assert h_second_source == pytest.approx(0.45)
    assert h_narrow_daughter != pytest.approx(0.45)  # it went through a real split first.
    expected_mix = (0.2 * h_narrow_daughter + 0.5 * h_second_source) / 0.7
    assert _hct(G3, 2, 5) == pytest.approx(expected_mix)


def test_a_pass_through_chain_carries_haematocrit_unchanged():
    G = nx.MultiGraph()
    _directed(G, 0, 1, diameter_um=15.0, flow=0.5)
    _directed(G, 1, 2, diameter_um=15.0, flow=0.5)
    distribute_discharge_haematocrit(G, inlet_haematocrit=0.45)
    assert _hct(G, 0, 1) == pytest.approx(0.45)
    assert _hct(G, 1, 2) == pytest.approx(0.45)


def test_an_anastomotic_loop_does_not_raise_and_still_conserves_mass():
    """Two parallel paths reconverge: 0->1->3 and 0->2->3 -- the undirected
    topology has a cycle, but the directed (flow-oriented) graph cannot
    (node pressure is single-valued), so this must resolve in one pass."""
    G = nx.MultiGraph()
    _directed(G, 0, 1, diameter_um=15.0, flow=0.6)
    _directed(G, 0, 2, diameter_um=10.0, flow=0.4)
    _directed(G, 1, 3, diameter_um=15.0, flow=0.6)
    _directed(G, 2, 3, diameter_um=10.0, flow=0.4)

    diag = distribute_discharge_haematocrit(G, inlet_haematocrit=0.45)
    assert diag["unresolved_edges"] == 0
    h_a = _hct(G, 1, 3)
    h_b = _hct(G, 2, 3)
    assert 0.6 * h_a + 0.4 * h_b == pytest.approx(0.45)


def test_a_dead_zero_flow_edge_gets_the_inlet_default_and_is_excluded_from_ordering():
    G = nx.MultiGraph()
    _directed(G, 0, 1, diameter_um=15.0, flow=0.5)
    _directed(G, 1, 2, diameter_um=15.0, flow=0.5)
    # A dangling, unsolved (zero-flow) branch off node 1.
    G.add_edge(1, 3, key=0, diameter_um=8.0, flow_signed=0.0)

    diag = distribute_discharge_haematocrit(G, inlet_haematocrit=0.45)
    assert diag["dead_edges"] == 1
    assert diag["unresolved_edges"] == 0
    assert G[1][3][0]["discharge_haematocrit"] == pytest.approx(0.45)


def test_three_way_divergence_falls_back_to_flow_proportional():
    """1 in, 3 out: no 2-daughter Pries-Secomb split defined -- flow-
    proportional (== inlet default here, since there is nothing upstream
    to have skewed it away from that) is the documented fallback."""
    G = nx.MultiGraph()
    _directed(G, 0, 1, diameter_um=20.0, flow=1.0)
    _directed(G, 1, 2, diameter_um=8.0, flow=0.2)
    _directed(G, 1, 3, diameter_um=8.0, flow=0.3)
    _directed(G, 1, 4, diameter_um=8.0, flow=0.5)
    distribute_discharge_haematocrit(G, inlet_haematocrit=0.45)
    for edge in ((0, 1), (1, 2), (1, 3), (1, 4)):
        assert _hct(G, *edge) == pytest.approx(0.45)


# --- iterate_flow_and_haematocrit -------------------------------------------


def _bifurcating_network(true_diameter_wide=20.0, true_diameter_narrow=8.0):
    """0 (inlet) -> 1 -> {2 (outlet, wide), 3 (outlet, narrow)}."""
    G = nx.MultiGraph()
    G.add_node(0, pos=(0.0, 0.0, 0.0))
    G.add_node(1, pos=(10.0, 0.0, 0.0))
    G.add_node(2, pos=(20.0, 5.0, 0.0))
    G.add_node(3, pos=(20.0, -5.0, 0.0))
    G.add_edge(
        0, 1, key=0, length=10.0, diameter_um=15.0, branch_order="B01",
    )
    G.add_edge(
        1, 2, key=0, length=20.0, diameter_um=true_diameter_wide, branch_order="B02",
    )
    G.add_edge(
        1, 3, key=0, length=20.0, diameter_um=true_diameter_narrow, branch_order="B03",
    )
    return G


def _seed_uniform_resistances(G, model: PoiseuilleModel) -> None:
    from haemolynx.haemodynamics.poiseuille import set_edge_resistance

    for _u, _v, _key, data in G.edges(keys=True, data=True):
        resistance = model.resistance_of_uniform_segment(data["length"], data["diameter_um"])
        set_edge_resistance(data, resistance)


def test_iterate_flow_and_haematocrit_converges_and_changes_the_baseline():
    G = _bifurcating_network()
    model = PoiseuilleModel(40.0, 100.0, viscosity_law="pries", haematocrit=0.45)
    _seed_uniform_resistances(G, model)

    def _recompute():
        _seed_uniform_resistances(G, model)
        for _u, _v, key, data in G.edges(keys=True, data=True):
            h = data.get("discharge_haematocrit")
            if h is not None:
                from haemolynx.haemodynamics.poiseuille import set_edge_resistance

                resistance = model.resistance_of_uniform_segment(
                    data["length"], data["diameter_um"], haematocrit=h
                )
                set_edge_resistance(data, resistance)

    baseline_narrow_resistance = G[1][3][0]["resistance"]

    result = iterate_flow_and_haematocrit(
        G,
        recompute_resistances=_recompute,
        inlet_haematocrit=0.45,
        inlet_p_bc=1000.0,
        outlet_p_bc=0.0,
        inlet_nodes=[0],
        outlet_nodes=[2, 3],
        max_iterations=30,
        tolerance=1e-3,
    )

    assert result["converged"] is True
    assert result["iterations"] <= 30
    assert result["pressure"] is not None
    assert len(result["node_list"]) == 4

    # The narrow branch should have picked up a lower discharge haematocrit
    # (phase separation), which changes its viscosity and hence resistance
    # relative to the uniform-haematocrit baseline -- proving this is not a
    # no-op.
    final_h = G[1][3][0]["discharge_haematocrit"]
    assert final_h < 0.45
    assert G[1][3][0]["resistance"] != pytest.approx(baseline_narrow_resistance)

    # Regression: resistance/conductance used to lag one iteration behind
    # the haematocrit this function reports, since recompute_resistances
    # was skipped on the converging pass. Every edge's stored resistance
    # must be derived from *exactly* its own final discharge_haematocrit,
    # not the pass before it.
    for u, v, key, data in G.edges(keys=True, data=True):
        expected = model.resistance_of_uniform_segment(
            data["length"], data["diameter_um"], haematocrit=data["discharge_haematocrit"]
        )
        assert data["resistance"] == pytest.approx(expected), (u, v, key)
        assert data["conductance"] == pytest.approx(1.0 / expected), (u, v, key)


def test_bifurcating_network_produces_a_pinned_median_haematocrit_signature():
    """A wider baseline test that the model runs on a realistic small
    network and gives the expected qualitative signature, not just for a
    single hand-picked bifurcation."""
    G = _bifurcating_network(true_diameter_wide=18.0, true_diameter_narrow=6.0)
    model = PoiseuilleModel(40.0, 100.0, viscosity_law="pries", haematocrit=0.45)
    _seed_uniform_resistances(G, model)

    def _recompute():
        from haemolynx.haemodynamics.poiseuille import set_edge_resistance

        for _u, _v, _key, data in G.edges(keys=True, data=True):
            h = data.get("discharge_haematocrit")
            resistance = model.resistance_of_uniform_segment(
                data["length"], data["diameter_um"], haematocrit=h
            )
            set_edge_resistance(data, resistance)

    iterate_flow_and_haematocrit(
        G,
        recompute_resistances=_recompute,
        inlet_haematocrit=0.45,
        inlet_p_bc=1000.0,
        outlet_p_bc=0.0,
        inlet_nodes=[0],
        outlet_nodes=[2, 3],
        max_iterations=40,
        tolerance=1e-3,
    )
    assert G[1][3][0]["discharge_haematocrit"] < 0.45  # narrow branch: skimmed.
    assert G[1][2][0]["discharge_haematocrit"] > 0.45  # wide branch: enriched.


# --- end to end through the pipeline stages ---------------------------------


def test_solve_stage_with_the_toggle_on_converges_and_changes_resistances():
    """`stages.build_haemodynamic_model` + `stages.solve`, the real call
    chain a pipeline run makes, with haematocrit_model set to
    distributed_iterative -- proving the wiring (not just the module in
    isolation) works."""
    from haemolynx.pipeline import BoundaryNodes, HaemodynamicModel, default_schema, resolve_settings
    from haemolynx.pipeline.stages import build_haemodynamic_model, solve

    schema = default_schema()

    def _run(distribute: bool):
        G = _bifurcating_network()
        G.graph["image_voxel_size_zyx"] = (1.0, 1.0, 1.0)
        model = HaemodynamicModel(graph=G)
        boundaries = BoundaryNodes(
            inlet_nodes=[0], outlet_nodes=[2, 3], resistance_node_pair=(0, 2),
        )
        values = {setting.name: setting.default for setting in schema}
        values.update(
            {
                "run_haemodynamics": True,
                "viscosity_law": "pries",
                "haematocrit": 0.45,
                "all_diams_const": False,
                "haematocrit_model": (
                    "distributed_iterative" if distribute else "fixed"
                ),
                "haematocrit_distribution_max_iterations": 30,
                "haematocrit_distribution_tolerance": 1e-3,
                "inlet_p_bc": 1000.0,
                "outlet_p_bc": 0.0,
                "inlet_nodes": [0],
                "outlet_nodes": [2, 3],
                "do_equiv_resistance_calculation": False,
                "diameter_by_branch_order": {"B01": 15.0, "B02": 20.0, "B03": 8.0},
            }
        )
        settings = resolve_settings(values, schema=schema, config_path=None)
        model = build_haemodynamic_model(settings, model, schema)
        return solve(settings, model, boundaries, schema)

    off = _run(False)
    assert "haematocrit_distribution" not in off.statistics

    on = _run(True)
    diag = on.statistics["haematocrit_distribution"]
    assert diag["converged"] is True
    assert on.pressure is not None
    assert len(on.node_list) == 4
    # The distributed-haematocrit run's narrow-branch resistance must differ
    # from the uniform-haematocrit baseline -- proving this is not a no-op
    # all the way through the real stage call chain.
    assert on.graph[1][3][0]["resistance"] != pytest.approx(
        off.graph[1][3][0]["resistance"]
    )


def test_iterate_flow_and_haematocrit_reports_the_iteration_count_honestly():
    """A single allowed iteration cannot claim convergence -- there is
    nothing to compare it against."""
    G = _bifurcating_network()
    model = PoiseuilleModel(40.0, 100.0, viscosity_law="pries", haematocrit=0.45)
    _seed_uniform_resistances(G, model)

    result = iterate_flow_and_haematocrit(
        G,
        recompute_resistances=lambda: None,
        inlet_haematocrit=0.45,
        inlet_p_bc=1000.0,
        outlet_p_bc=0.0,
        inlet_nodes=[0],
        outlet_nodes=[2, 3],
        max_iterations=1,
        tolerance=1e-4,
    )
    assert result["iterations"] == 1
    assert result["converged"] is False
