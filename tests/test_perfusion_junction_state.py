"""Open item 24: Tier 3 carries blood across a node as pressures, not as content per litre.

The march mixed content per litre of blood at each node and inverted it for each daughter at
PCO2 40 and pH 7.4, with that daughter's haematocrit; a failed inversion fell back to arterial
pressures. Phase separation gives daughters a different H from the parent, so the content did
not describe the daughter's blood. On WKY-A a plasma-skimmed daughter (H near 0) received
arterial CO2 content at H 0.45, more than its curve can hold at any PCO2, and the implicit
step found no root.

Now a node's state is the blood's pressures (and the PCO2 and pH its O2 content was evaluated
at). One inflow passes its state through; two or more are mixed and inverted jointly. Each
daughter's content is evaluated at the node state with its own H. Both curves are affine in H,
so this conserves O2 and CO2 wherever the rheology conserves red cell and plasma flux.
"""
from dataclasses import replace

import networkx as nx
import numpy as np
import pytest

from ImageLynx.haemodynamics.perfusion import (
    _increasing_inverse,
    _mixed_blood_state,
    calculate_blood_co2_content,
    calculate_blood_oxygen_content,
    solve_multi_species_perfusion,
)
from ImageLynx.haemodynamics.resistance import POISEUILLE_FLOW_TO_UM3_PER_S
from test_perfusion_march_flow_units import _AREA, _V_CELL, _Config

_MEASURED = replace(_Config(), permeability_o2_cm_s=9.1e-2, permeability_co2_cm_s=9.1e-2,
                    picard_max_iterations=50, picard_tolerance=1e-4)


class _LineGrid:
    """n independent cells; the solver only needs these attributes."""

    def __init__(self, n):
        self.n_cells = n
        self.cell_volume = _V_CELL
        self.dims = (n, 1, 1)
        self.res = (10.0, 10.0, 10.0)


def _network(edges):
    """edges: (u, v, q um^3/s, H). One grid cell per edge, in the order given."""
    G = nx.MultiGraph()
    cells = {}
    for i, (u, v, q, h) in enumerate(edges):
        f = q / POISEUILLE_FLOW_TO_UM3_PER_S
        G.add_edge(u, v, key=0, flow_abs=f, hematocrit=h, length=10.0, downstream=v)
        cells[i] = [{'edge': (u, v, 0), 'flow': q, 'hematocrit': h, 'length': 10.0,
                     'surface_area': _AREA}]
    # flow_signed is relative to the orientation the graph iterates the edge in, as the
    # rheology solver writes it, and that need not be the order the edge was added in.
    for u, v, d in G.edges(data=True):
        d["flow_signed"] = d["flow_abs"] if v == d["downstream"] else -d["flow_abs"]
    return _LineGrid(len(edges)), G, cells


def test_both_content_curves_are_affine_in_haematocrit():
    """The premise of the split: sum q_d C(P, H_d) = q C(P, H) when sum q_d H_d = q H."""
    q, h = 1.0, 0.45
    daughters = [(0.5, 0.89), (0.5, 0.01)]
    assert sum(qd * hd for qd, hd in daughters) == pytest.approx(q * h)
    for po2, pco2, ph in [(100.0, 40.0, 7.4), (35.0, 52.0, 7.25)]:
        o2 = sum(qd * calculate_blood_oxygen_content(po2, hd, pco2, ph) for qd, hd in daughters)
        co2 = sum(qd * calculate_blood_co2_content(pco2, hd, po2) for qd, hd in daughters)
        assert o2 == pytest.approx(q * calculate_blood_oxygen_content(po2, h, pco2, ph), rel=1e-12)
        assert co2 == pytest.approx(q * calculate_blood_co2_content(pco2, h, po2), rel=1e-12)


def test_the_inverse_round_trips_and_refuses_content_the_blood_cannot_hold():
    for p in (5.0, 40.0, 100.0, 400.0):
        c = calculate_blood_oxygen_content(p, 0.45, 40.0, 7.4)
        assert _increasing_inverse(lambda x: calculate_blood_oxygen_content(x, 0.45, 40.0, 7.4),
                                   c, "O2") == pytest.approx(p, abs=1e-8)
    for p in (10.0, 40.0, 60.0):
        c = calculate_blood_co2_content(p, 0.01, 100.0)
        assert _increasing_inverse(lambda x: calculate_blood_co2_content(x, 0.01, 100.0),
                                   c, "CO2") == pytest.approx(p, abs=1e-8)
    # The WKY-A case: arterial content at H 0.45 into blood at H 0.01.
    arterial = calculate_blood_co2_content(40.0, 0.45, 100.0)
    with pytest.raises(ValueError, match="No CO2 partial pressure"):
        _increasing_inverse(lambda x: calculate_blood_co2_content(x, 0.01, 100.0), arterial, "CO2")


def _stream(q, h, po2, pco2, ph):
    return {"q": q, "h": h,
            "c_o2": calculate_blood_oxygen_content(po2, h, pco2, ph),
            "c_co2": calculate_blood_co2_content(pco2, h, po2),
            "state": {"po2": po2, "pco2": pco2, "o2_pco2": pco2, "o2_ph": ph}}


def test_identical_streams_mix_to_their_own_state():
    s = _stream(1.0, 0.45, 70.0, 45.0, 7.35)
    state = _mixed_blood_state([s, dict(s, q=3.0)])
    assert state["po2"] == pytest.approx(70.0, abs=1e-8)
    assert state["pco2"] == pytest.approx(45.0, abs=1e-8)


def test_a_mixture_holds_exactly_the_streams_gas_and_sits_between_them():
    a = _stream(400.0, 0.30, 90.0, 42.0, 7.38)
    b = _stream(600.0, 0.60, 30.0, 55.0, 7.25)
    state = _mixed_blood_state([a, b])
    q = a["q"] + b["q"]
    h = (a["q"] * a["h"] + b["q"] * b["h"]) / q
    c_o2 = calculate_blood_oxygen_content(state["po2"], h, state["o2_pco2"], state["o2_ph"])
    c_co2 = calculate_blood_co2_content(state["pco2"], h, state["po2"])
    assert c_o2 * q == pytest.approx(a["c_o2"] * a["q"] + b["c_o2"] * b["q"], rel=1e-9)
    assert c_co2 * q == pytest.approx(a["c_co2"] * a["q"] + b["c_co2"] * b["q"], rel=1e-9)
    assert 30.0 < state["po2"] < 90.0
    assert 42.0 < state["pco2"] < 55.0
    assert state["o2_ph"] == pytest.approx(0.4 * 7.38 + 0.6 * 7.25)


def test_a_plasma_skimmed_daughter_no_longer_breaks_the_march():
    """The old march raised 'Implicit CO2 step found no root' here, as on WKY-A."""
    grid, G, cells = _network([(0, 1, 1e3, 0.45), (1, 2, 5e2, 0.89), (1, 3, 5e2, 0.01)])
    po2, pco2, _ = solve_multi_species_perfusion(grid, G, [0], cells, _MEASURED)
    assert np.all(np.isfinite(po2)) and np.all(np.isfinite(pco2))
    assert np.all(po2 <= _MEASURED.po2_arterial_mmHg)


def test_two_inlets_merge_into_one_stream():
    grid, G, cells = _network([(0, 2, 4e2, 0.30), (1, 2, 6e2, 0.60), (2, 3, 1e3, 0.48)])
    po2, pco2, _ = solve_multi_species_perfusion(grid, G, [0, 1], cells, _MEASURED)
    assert np.all(np.isfinite(po2)) and np.all(np.isfinite(pco2))
    # Merged blood lies between the parents' outlets, and its tissue sits below it.
    assert po2[2] < max(po2[0], po2[1])


def test_a_node_with_outflow_and_no_inflow_is_refused_not_given_arterial_blood():
    grid, G, cells = _network([(0, 1, 1e3, 0.45), (5, 6, 1e3, 0.45)])
    with pytest.raises(ValueError, match="Node 5 sends blood out but receives none"):
        solve_multi_species_perfusion(grid, G, [0], cells, _MEASURED)


def test_a_cycle_in_the_flow_direction_is_refused():
    grid, G, cells = _network([(0, 1, 1e3, 0.45), (1, 2, 1e3, 0.45), (2, 0, 1e3, 0.45)])
    with pytest.raises(ValueError, match="has a cycle"):
        solve_multi_species_perfusion(grid, G, [0], cells, _MEASURED)


def test_rounding_level_flow_is_stagnant_not_an_unfed_source():
    """WKY-A: node 764 had both edges flowing outward at ~1e-14 of the largest flow."""
    grid, G, cells = _network([(0, 1, 1e3, 0.45), (1, 2, 1e3, 0.45),
                               (4, 3, 1e3 * 1e-14, 0.45), (4, 5, 1e3 * 2e-14, 0.45)])
    po2, pco2, _ = solve_multi_species_perfusion(grid, G, [0], cells, _MEASURED)
    assert np.all(np.isfinite(po2)) and np.all(np.isfinite(pco2))
    # The stagnant cells get nothing from the blood: no O2 in, so their tissue is anoxic.
    np.testing.assert_allclose(po2[2:], 0.0, atol=1e-12)


def test_flow_above_the_rounding_level_at_an_unfed_node_still_raises():
    grid, G, cells = _network([(0, 1, 1e3, 0.45), (4, 3, 1e3 * 1e-9, 0.45)])
    with pytest.raises(ValueError, match="Node 4 sends blood out but receives none"):
        solve_multi_species_perfusion(grid, G, [0], cells, _MEASURED)


def test_a_plasma_skimmed_daughter_keeps_tissue_co2_above_arterial():
    """With a full Haldane effect in plasma, the H 0.01 daughter's tissue sat at 39.948 mmHg.

    Solved tightly so the check is on the fixed point, not on where the loop stopped.
    """
    config = replace(_MEASURED, picard_max_iterations=200, picard_tolerance=1e-12)
    grid, G, cells = _network([(0, 1, 1e3, 0.45), (1, 2, 5e2, 0.89), (1, 3, 5e2, 0.01)])
    po2, pco2, _ = solve_multi_species_perfusion(grid, G, [0], cells, config)
    assert np.all(pco2 >= config.pco2_arterial)
    assert np.all(po2 <= config.po2_arterial_mmHg)
