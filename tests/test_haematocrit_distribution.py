"""Pries-Secomb bifurcation haematocrit distribution.

The rest of the haemodynamics package treats haematocrit as one scalar
shared by every edge (`viscosity.DEFAULT_HAEMATOCRIT`,
`PoiseuilleModel.haematocrit`). This tests the second model: local discharge
haematocrit computed from Pries, Ley, Claassen & Gaehtgens' (1989)
bifurcation phase-separation law, with Pries & Secomb's (2005) parameters --
driven by parent/daughter diameters and the flow fraction at the bifurcation,
not by bifurcation angle -- the three rules for the junctions that law does
not cover (``haematocrit_junction_rule``), and the outer loop that iterates it
together with the flow solve to a fixed point.
"""
from __future__ import annotations

import math

import networkx as nx
import numpy as np
import pytest

from haemolynx.haemodynamics.haematocrit_distribution import (
    JUNCTION_RULES,
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


def _x0_2005(parent_diameter_um: float, parent_haematocrit: float) -> float:
    """Pries & Secomb (2005): X0 = 0.964 (1 - Hd) / D_parent."""
    return 0.964 * (1.0 - parent_haematocrit) / parent_diameter_um


def test_a_flow_fraction_at_or_below_the_plasma_skimming_threshold_gets_no_red_cells():
    """Below X0 = 0.964 (1 - Hd) / D_parent, the daughter is plasma-skimmed entirely."""
    x0 = _x0_2005(20.0, 0.45)
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
    x0 = _x0_2005(20.0, 0.45)
    fqe = _fqe_pries_secomb(
        fractional_flow=1.0 - x0 / 2.0,
        parent_diameter_um=20.0,
        own_diameter_um=15.0,
        other_diameter_um=10.0,
        parent_haematocrit=0.45,
    )
    assert fqe == 1.0


@pytest.mark.parametrize("parent_haematocrit", [0.2, 0.45, 0.6])
def test_the_skimming_threshold_is_pries_and_secombs_2005_one(parent_haematocrit):
    """Regression: X0 was the 1990 value, 0.4 / D_parent, paired with the 2005
    A and B. The 2005 threshold is 0.964 (1 - Hd) / D_parent -- higher at any
    haematocrit below 0.585 -- so a daughter drawing a flow share between the
    two got red cells it should not have."""
    parent_diameter = 20.0
    x0 = _x0_2005(parent_diameter, parent_haematocrit)

    def fqe(fractional_flow: float) -> float:
        return _fqe_pries_secomb(
            fractional_flow=fractional_flow,
            parent_diameter_um=parent_diameter,
            own_diameter_um=10.0,
            other_diameter_um=15.0,
            parent_haematocrit=parent_haematocrit,
        )

    assert fqe(x0 * 0.999) == 0.0
    assert fqe(x0 * 1.001) > 0.0
    assert fqe(1.0 - x0 * 0.999) == 1.0
    assert fqe(1.0 - x0 * 1.001) < 1.0


def test_a_daughter_between_the_1990_and_2005_thresholds_is_skimmed_bare():
    """The case the old threshold got wrong, at the default haematocrit: a 2.3%
    flow share off a 20 um parent is above 0.4 / 20 = 2% but below
    0.964 * 0.55 / 20 = 2.65%."""
    assert 0.4 / 20.0 < 0.023 < _x0_2005(20.0, 0.45)
    h = pries_secomb_daughter_haematocrit(
        fractional_flow=0.023,
        parent_diameter_um=20.0,
        own_diameter_um=10.0,
        other_diameter_um=15.0,
        parent_haematocrit=0.45,
    )
    assert h == 0.0


def test_the_law_matches_its_published_form_at_an_interior_split():
    """A hand evaluation of Pries & Secomb (2005), Eq. 3-5, diameters in um."""
    d_parent, d_own, d_other, hd, fqb = 12.0, 6.0, 9.0, 0.45, 0.3
    ratio_sq = (d_own / d_other) ** 2
    a = -13.29 * (ratio_sq - 1) / (ratio_sq + 1) * (1 - hd) / d_parent
    b = 1 + 6.98 * (1 - hd) / d_parent
    x0 = 0.964 * (1 - hd) / d_parent
    scaled = (fqb - x0) / (1 - 2 * x0)
    expected = 1 / (1 + math.exp(-(a + b * math.log(scaled / (1 - scaled)))))

    assert _fqe_pries_secomb(
        fractional_flow=fqb,
        parent_diameter_um=d_parent,
        own_diameter_um=d_own,
        other_diameter_um=d_other,
        parent_haematocrit=hd,
    ) == pytest.approx(expected, rel=1e-12)


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
    assert diag == {
        "dead_edges": 0,
        "compound_junctions": 0,
        "non_bifurcation_junctions": 0,
        "unresolved_edges": 0,
    }

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


# --- junctions the law does not cover: haematocrit_junction_rule -------------


def _trifurcation(outflow_order=(2, 3, 4)) -> nx.MultiGraph:
    """An upstream bifurcation, so the trifurcation's parent is not at the
    inlet haematocrit: 9 -> 0 -> {5 (a narrow side branch), 1}, then 1 splits
    three ways into 2 (8 um, flow 0.2), 3 (12 um, 0.5) and 4 (10 um, 0.3).
    *outflow_order* is the order the three are added in."""
    daughters = {2: (8.0, 0.2), 3: (12.0, 0.5), 4: (10.0, 0.3)}
    G = nx.MultiGraph()
    _directed(G, 9, 0, diameter_um=20.0, flow=1.1)
    _directed(G, 0, 5, diameter_um=6.0, flow=0.1)
    _directed(G, 0, 1, diameter_um=20.0, flow=1.0)
    for node in outflow_order:
        diameter, flow = daughters[node]
        _directed(G, 1, node, diameter_um=diameter, flow=flow)
    return G


def _two_in_two_out() -> nx.MultiGraph:
    """Two inflows at different haematocrits meet two outflows at node 4:
    0 -> 1 -> {2 (narrow), 4}, and 3 -> 4 straight from an inlet; then
    4 -> 5 (wide) and 4 -> 6 (narrow)."""
    G = nx.MultiGraph()
    _directed(G, 0, 1, diameter_um=20.0, flow=1.0)
    _directed(G, 1, 2, diameter_um=6.0, flow=0.15)
    _directed(G, 1, 4, diameter_um=14.0, flow=0.85)
    _directed(G, 3, 4, diameter_um=9.0, flow=0.35)
    _directed(G, 4, 5, diameter_um=16.0, flow=0.9)
    _directed(G, 4, 6, diameter_um=7.0, flow=0.3)
    return G


def test_every_rule_gives_a_plain_bifurcation_the_law():
    """The rules only decide what the law leaves open."""
    results = []
    for rule in JUNCTION_RULES:
        G = nx.MultiGraph()
        _directed(G, 0, 1, diameter_um=20.0, flow=1.0)
        _directed(G, 1, 2, diameter_um=10.0, flow=0.2)
        _directed(G, 1, 3, diameter_um=15.0, flow=0.8)
        distribute_discharge_haematocrit(G, inlet_haematocrit=0.45, junction_rule=rule)
        results.append((_hct(G, 1, 2), _hct(G, 1, 3)))
    expected = pries_secomb_daughter_haematocrit(
        fractional_flow=0.2,
        parent_diameter_um=20.0,
        own_diameter_um=10.0,
        other_diameter_um=15.0,
        parent_haematocrit=0.45,
    )
    for narrow, _wide in results:
        assert narrow == pytest.approx(expected, rel=1e-12)
    assert results[1:] == [results[0]] * (len(results) - 1)


@pytest.mark.parametrize("rule", ["no_separation", "split_junctions"])
def test_a_trifurcation_gets_no_phase_separation_by_default(rule):
    """Secomb's dishem.cpp: every outflow of a junction the law does not cover
    gets the mixed inflow's haematocrit. split_junctions has already split
    every such junction by the time this runs, and mixes any it could not."""
    G = _trifurcation()
    diag = distribute_discharge_haematocrit(G, inlet_haematocrit=0.45, junction_rule=rule)

    parent = _hct(G, 0, 1)
    assert parent != pytest.approx(0.45)  # the upstream split moved it
    for daughter in (2, 3, 4):
        assert _hct(G, 1, daughter) == pytest.approx(parent, rel=1e-12)
    assert diag["non_bifurcation_junctions"] == 1


def test_two_inflows_meeting_two_outflows_get_no_phase_separation_by_default():
    """Regression: a 2-in 2-out junction was split with the law, taking a
    flow-weighted mean inflow diameter as its parent. Secomb's published rule
    (dishem.cpp) mixes the inflows and applies no separation."""
    G = _two_in_two_out()
    diag = distribute_discharge_haematocrit(G, inlet_haematocrit=0.45)

    mixed = (0.85 * _hct(G, 1, 4) + 0.35 * _hct(G, 3, 4)) / 1.2
    assert _hct(G, 1, 4) != pytest.approx(_hct(G, 3, 4))  # a genuine mix
    assert _hct(G, 4, 5) == pytest.approx(mixed, rel=1e-12)
    assert _hct(G, 4, 6) == pytest.approx(mixed, rel=1e-12)
    assert diag["compound_junctions"] == 1
    assert diag["non_bifurcation_junctions"] == 1


def test_a_chain_of_bifurcations_follows_secombs_generalised_rule():
    """dishem_generalized.cpp, outflows largest flow first: 3 (0.5) split off
    against 4 with the parent's diameter, then 4 (0.3) against 2 with 4's
    diameter as the parent of what is left, and 2 takes the rest."""
    G = _trifurcation()
    distribute_discharge_haematocrit(
        G, inlet_haematocrit=0.45, junction_rule="sequential_bifurcations"
    )
    h_parent = _hct(G, 0, 1)

    rbc = h_parent * 1.0
    fqe_3 = _fqe_pries_secomb(
        fractional_flow=0.5 / 1.0,
        parent_diameter_um=20.0,
        own_diameter_um=12.0,
        other_diameter_um=10.0,
        parent_haematocrit=h_parent,
    )
    expected_3 = fqe_3 * rbc / 0.5
    rbc *= 1 - fqe_3
    fqe_4 = _fqe_pries_secomb(
        fractional_flow=0.3 / 0.5,
        parent_diameter_um=10.0,
        own_diameter_um=10.0,
        other_diameter_um=8.0,
        parent_haematocrit=rbc / 0.5,
    )
    expected_4 = fqe_4 * rbc / 0.3
    expected_2 = rbc * (1 - fqe_4) / 0.2

    assert _hct(G, 1, 3) == pytest.approx(expected_3, rel=1e-12)
    assert _hct(G, 1, 4) == pytest.approx(expected_4, rel=1e-12)
    assert _hct(G, 1, 2) == pytest.approx(expected_2, rel=1e-12)
    # Red cells are conserved, and they really were separated: the narrow,
    # low-flow branch is skimmed below its parent.
    assert 0.2 * _hct(G, 1, 2) + 0.5 * _hct(G, 1, 3) + 0.3 * _hct(G, 1, 4) == pytest.approx(
        1.0 * h_parent
    )
    assert _hct(G, 1, 2) < h_parent


def test_a_chain_of_bifurcations_does_not_depend_on_how_the_graph_was_built():
    """Secomb takes outflows in file order; here the order is by flow, so
    adding the same vessels in another order gives the same answer."""
    answers = []
    for order in [(2, 3, 4), (4, 2, 3), (3, 4, 2)]:
        G = _trifurcation(order)
        distribute_discharge_haematocrit(
            G, inlet_haematocrit=0.45, junction_rule="sequential_bifurcations"
        )
        answers.append([_hct(G, 1, node) for node in (2, 3, 4)])
    for answer in answers[1:]:
        assert answer == pytest.approx(answers[0], rel=1e-12)


def test_a_chain_splits_two_inflows_meeting_two_outflows_from_the_widest_inflow():
    """dishem_generalized.cpp: the mixed inflow, with the widest inflow (14 um,
    not the flow-weighted mean) as the parent."""
    G = _two_in_two_out()
    distribute_discharge_haematocrit(
        G, inlet_haematocrit=0.45, junction_rule="sequential_bifurcations"
    )
    mixed = (0.85 * _hct(G, 1, 4) + 0.35 * _hct(G, 3, 4)) / 1.2
    fqe_5 = _fqe_pries_secomb(
        fractional_flow=0.9 / 1.2,
        parent_diameter_um=14.0,
        own_diameter_um=16.0,
        other_diameter_um=7.0,
        parent_haematocrit=mixed,
    )
    assert _hct(G, 4, 5) == pytest.approx(fqe_5 * mixed * 1.2 / 0.9, rel=1e-12)
    assert _hct(G, 4, 6) == pytest.approx((1 - fqe_5) * mixed * 1.2 / 0.3, rel=1e-12)
    assert _hct(G, 4, 6) < mixed < _hct(G, 4, 5)


def test_an_unknown_junction_rule_is_refused():
    G = _trifurcation()
    with pytest.raises(ValueError, match="junction_rule"):
        distribute_discharge_haematocrit(G, inlet_haematocrit=0.45, junction_rule="average")


def test_the_schema_offers_exactly_these_rules_as_a_dropdown():
    """The setting and the module name the same three rules, in the same
    order, the dropdown labels each one, and it only shows once the
    distributed haematocrit model is picked."""
    from haemolynx.gui.form import CHOICE_VALUE_LABELS, widget_type_for
    from haemolynx.pipeline import default_schema

    setting = default_schema()["haematocrit_junction_rule"]
    assert tuple(setting.choices) == JUNCTION_RULES
    assert setting.default == "no_separation"
    assert widget_type_for(setting) == "ComboBox"
    assert "haematocrit_model=distributed_iterative" in setting.requires
    labels = CHOICE_VALUE_LABELS["haematocrit_junction_rule"]
    assert [labels[rule][:2] for rule in JUNCTION_RULES] == ["1.", "2.", "3."]


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


def test_a_daughter_near_the_skimming_threshold_converges_within_the_default_iterations():
    """Regression: the 8 um daughter here draws ~5% of the flow, just above the
    2005 threshold (3.5% off a 15 um parent). Unrelaxed, its haematocrit
    swung between ~0.02 and ~0.12 on alternate passes and was still 0.002
    from settling after 60; with NetFlow's relaxation schedule it settles
    well within the setting's default 20, at the same fixed point."""
    G = _bifurcating_network()
    model = PoiseuilleModel(40.0, 100.0, viscosity_law="pries", haematocrit=0.45)
    _seed_uniform_resistances(G, model)

    def _recompute():
        from haemolynx.haemodynamics.poiseuille import set_edge_resistance

        for _u, _v, _key, data in G.edges(keys=True, data=True):
            resistance = model.resistance_of_uniform_segment(
                data["length"], data["diameter_um"], haematocrit=data.get("discharge_haematocrit")
            )
            set_edge_resistance(data, resistance)

    result = iterate_flow_and_haematocrit(
        G,
        recompute_resistances=_recompute,
        inlet_haematocrit=0.45,
        inlet_p_bc=1000.0,
        outlet_p_bc=0.0,
        inlet_nodes=[0],
        outlet_nodes=[2, 3],
        max_iterations=20,
        tolerance=1e-3,
    )
    assert result["converged"] is True

    # A fixed point: one more distribution over the flow it left changes
    # nothing beyond the tolerance.
    settled = {edge: G.edges[edge]["discharge_haematocrit"] for edge in G.edges(keys=True)}
    distribute_discharge_haematocrit(G, inlet_haematocrit=0.45)
    for edge, h in settled.items():
        assert G.edges[edge]["discharge_haematocrit"] == pytest.approx(h, abs=2e-3)
    assert 0.0 < settled[(1, 3, 0)] < 0.45  # skimmed, not emptied


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


TRIFURCATION_DIAMETERS = {"B01": 15.0, "B02": 12.0, "B03": 8.0, "B04": 10.0}


def _trifurcating_network() -> nx.MultiGraph:
    """0 (inlet) -> 1, which divides three ways to outlets 2, 3 and 4. Positions
    are (z, y, x): 3 and 4 leave 1 in the closest directions, so a split peels
    those two off onto a connector together."""
    positions = {
        0: (0.0, 0.0, 0.0),
        1: (0.0, 0.0, 50.0),
        2: (0.0, 60.0, 80.0),
        3: (0.0, -10.0, 110.0),
        4: (0.0, -40.0, 100.0),
    }
    G = nx.MultiGraph()
    for node, pos in positions.items():
        G.add_node(node, pos=np.asarray(pos))
    for (u, v), order in zip([(0, 1), (1, 2), (1, 3), (1, 4)], TRIFURCATION_DIAMETERS):
        a, b = np.asarray(positions[u]), np.asarray(positions[v])
        G.add_edge(
            u, v, key=0,
            length=float(np.linalg.norm(b - a)),
            voxels=[list(positions[u]), list(positions[v])],
            diameter_um=TRIFURCATION_DIAMETERS[order],
            branch_order=order,
        )
    G.graph["image_voxel_size_zyx"] = (1.0, 1.0, 1.0)
    return G


def _solve_trifurcation(rule: str):
    from haemolynx.pipeline import (
        BoundaryNodes,
        HaemodynamicModel,
        apply_network_handling,
        default_schema,
        resolve_settings,
    )
    from haemolynx.pipeline.stages import build_haemodynamic_model, solve

    schema = default_schema()
    values = {setting.name: setting.default for setting in schema}
    values.update(
        {
            "run_haemodynamics": True,
            "viscosity_law": "pries",
            "haematocrit": 0.45,
            "all_diams_const": False,
            "haematocrit_model": "distributed_iterative",
            "haematocrit_junction_rule": rule,
            "haematocrit_distribution_max_iterations": 60,
            "haematocrit_distribution_tolerance": 1e-4,
            "inlet_p_bc": 1000.0,
            "outlet_p_bc": 0.0,
            "inlet_nodes": [0],
            "outlet_nodes": [2, 3, 4],
            "do_equiv_resistance_calculation": False,
            "diameter_by_branch_order": dict(TRIFURCATION_DIAMETERS),
        }
    )
    settings = resolve_settings(values, schema=schema, config_path=None)
    model = HaemodynamicModel(graph=_trifurcating_network())
    boundaries = BoundaryNodes(inlet_nodes=[0], outlet_nodes=[2, 3, 4], resistance_node_pair=(0, 2))
    model = apply_network_handling(settings, model, boundaries)
    model = build_haemodynamic_model(settings, model, schema)
    return solve(settings, model, boundaries, schema)


def _outflow_haematocrits(G: nx.MultiGraph, node) -> dict:
    """``{neighbour: (|flow|, haematocrit)}`` for every vessel leaving *node*
    (the ones running down to a lower pressure)."""
    here = G.nodes[node]["pressure"]
    return {
        other: (data["flow_abs"], data["discharge_haematocrit"])
        for _node, other, data in G.edges(node, data=True)
        if G.nodes[other]["pressure"] < here
    }


def test_the_solve_stage_divides_a_trifurcation_by_the_chosen_junction_rule():
    """All three dropdown options, through the real stage calls a run makes."""
    mixed = _solve_trifurcation("no_separation")
    chained = _solve_trifurcation("sequential_bifurcations")
    split = _solve_trifurcation("split_junctions")
    for solution in (mixed, chained, split):
        assert solution.statistics["haematocrit_distribution"]["converged"] is True

    # 1. Every daughter carries the parent's haematocrit.
    diag = mixed.statistics["haematocrit_distribution"]
    assert diag["junction_rule"] == "no_separation"
    assert diag["non_bifurcation_junctions"] == 1
    for _flow, h in _outflow_haematocrits(mixed.graph, 1).values():
        assert h == pytest.approx(0.45, rel=1e-12)

    # 2. Separated, red cells conserved.
    diag = chained.statistics["haematocrit_distribution"]
    assert diag["junction_rule"] == "sequential_bifurcations"
    assert diag["non_bifurcation_junctions"] == 1
    daughters = _outflow_haematocrits(chained.graph, 1)
    assert len({round(h, 9) for _flow, h in daughters.values()}) == 3
    total_flow = sum(flow for flow, _h in daughters.values())
    assert sum(flow * h for flow, h in daughters.values()) == pytest.approx(0.45 * total_flow)

    # 3. The network itself was split: no junction is left for a rule to
    # decide, and the connector carries a resistance like any vessel.
    diag = split.statistics["haematocrit_distribution"]
    assert diag["junction_rule"] == "split_junctions"
    assert diag["non_bifurcation_junctions"] == 0
    assert max(degree for _node, degree in split.graph.degree()) == 3
    (connector,) = [
        data for _u, _v, data in split.graph.edges(data=True) if data.get("junction_split_connector")
    ]
    assert connector["branch_order"] == "B04"  # the wider of the two peeled onto it
    assert connector["resistance"] > 0
    assert split.graph.number_of_edges() == 5
    h_3 = split.graph.edges[next(e for e in split.graph.edges(3, keys=True))]["discharge_haematocrit"]
    h_4 = split.graph.edges[next(e for e in split.graph.edges(4, keys=True))]["discharge_haematocrit"]
    assert h_3 != pytest.approx(h_4)


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


def _recompute_for(G, model: PoiseuilleModel):
    """Every edge's resistance from its own discharge haematocrit."""
    from haemolynx.haemodynamics.poiseuille import set_edge_resistance

    def _recompute():
        for _u, _v, _key, data in G.edges(keys=True, data=True):
            resistance = model.resistance_of_uniform_segment(
                data["length"], data["diameter_um"],
                haematocrit=data.get("discharge_haematocrit", 0.45),
            )
            set_edge_resistance(data, resistance)

    return _recompute


def _iterate(G, model, **kwargs):
    arguments = dict(
        recompute_resistances=_recompute_for(G, model),
        inlet_haematocrit=0.45,
        inlet_p_bc=1000.0,
        outlet_p_bc=0.0,
        inlet_nodes=[0],
        outlet_nodes=[2, 3],
        max_iterations=2,
        tolerance=1e-9,
    )
    arguments.update(kwargs)
    return iterate_flow_and_haematocrit(G, **arguments)


@pytest.mark.parametrize("solver", ["dense", "sparse"])
def test_the_returned_flows_obey_the_resistances_left_on_the_graph(solver):
    """Regression: the loop recomputed resistances after its last solve and
    returned that solve's flows, so a vessel's flow was not its pressure drop
    over its reported resistance -- by a whole haematocrit update when the
    loop stopped at its cap unconverged, as two passes at this tolerance do.
    A final solve from the last resistances closes the gap."""
    G = _bifurcating_network()
    model = PoiseuilleModel(40.0, 100.0, viscosity_law="pries", haematocrit=0.45)
    _seed_uniform_resistances(G, model)

    result = _iterate(G, model, solver=solver)

    assert result["converged"] is False
    for u, v, key, data in G.edges(keys=True, data=True):
        assert data["flow_signed"] == pytest.approx(
            data["conductance"] * data["pressure_drop"], rel=1e-12
        ), (u, v, key)
    from haemolynx.haemodynamics.resistance import flow_conservation_residuals

    residual = flow_conservation_residuals(G, boundary_nodes=[0, 2, 3])[1]
    assert abs(residual) < 1e-12 * G.edges[0, 1, 0]["flow_abs"]
    index = {node: i for i, node in enumerate(result["node_list"])}
    for node in G.nodes:
        assert G.nodes[node]["pressure"] == result["pressure"][index[node]]


@pytest.mark.parametrize("final_solve, extra", [(True, 1), (False, 0)])
def test_the_final_solve_is_one_more_solve_and_can_be_left_to_the_caller(
    monkeypatch, final_solve, extra
):
    import haemolynx.haemodynamics.haematocrit_distribution as module

    calls = []
    real = module.solve_flow_from_conductance_matrix

    def counting(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(module, "solve_flow_from_conductance_matrix", counting)
    G = _bifurcating_network()
    model = PoiseuilleModel(40.0, 100.0, viscosity_law="pries", haematocrit=0.45)
    _seed_uniform_resistances(G, model)

    result = _iterate(G, model, max_iterations=3, final_solve=final_solve)

    assert result["iterations"] == 3
    assert len(calls) == 3 + extra


def test_the_sparse_solver_reaches_the_dense_fixed_point():
    model = PoiseuilleModel(40.0, 100.0, viscosity_law="pries", haematocrit=0.45)
    results = {}
    for solver in ("dense", "sparse"):
        G = _bifurcating_network()
        _seed_uniform_resistances(G, model)
        result = _iterate(G, model, solver=solver, max_iterations=30, tolerance=1e-6)
        assert result["converged"] is True
        results[solver] = (G, result)
    (dense_G, dense), (sparse_G, sparse) = results["dense"], results["sparse"]
    assert sparse["iterations"] == dense["iterations"]
    np.testing.assert_allclose(sparse["pressure"], dense["pressure"], rtol=1e-10)
    for u, v, key, data in dense_G.edges(keys=True, data=True):
        assert sparse_G.edges[u, v, key]["discharge_haematocrit"] == pytest.approx(
            data["discharge_haematocrit"], rel=1e-9
        )
