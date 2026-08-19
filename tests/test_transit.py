"""Transit time along flow-directed paths, for H2 §2.4.

Reported as a within-specimen ratio rather than an absolute. The absolute sits under the
±45% calibre floor of S15, and the pressure, viscosity and length units in this pipeline are
not reconciled to a single system, so the magnitude is in arbitrary units. A ratio of two
transit times computed the same way is both dimensionless and free of the shared error.
"""
import networkx as nx
import numpy as np
import pytest

from ImageLynx.haemodynamics.transit import edge_transit_times, transit_time_from_inlets


def _chain(n_edges, diameter=8.0, length=10.0, flow=2.0):
    G = nx.MultiGraph()
    for i in range(n_edges):
        G.add_edge(i, i + 1, key=0, length=length, assigned_diameter_um=diameter,
                   flow_abs=flow, flow_signed=flow)
    return G


def test_edge_transit_time_is_volume_over_flow():
    G = _chain(1, diameter=8.0, length=10.0, flow=2.0)
    tau = edge_transit_times(G)
    expected = (np.pi * 4.0 ** 2 * 10.0) / 2.0
    assert list(tau.values())[0] == pytest.approx(expected)


def test_transit_time_scales_with_the_square_of_diameter():
    thin = list(edge_transit_times(_chain(1, diameter=4.0)).values())[0]
    thick = list(edge_transit_times(_chain(1, diameter=8.0)).values())[0]
    assert thick == pytest.approx(4.0 * thin)


def test_an_edge_with_no_flow_has_no_finite_transit_time():
    """Blood that does not move does not arrive, and infinity is the honest answer."""
    G = _chain(1, flow=0.0)
    assert not np.isfinite(list(edge_transit_times(G).values())[0])


def test_transit_time_accumulates_along_a_chain():
    G = _chain(3)
    times = transit_time_from_inlets(G, inlets=[0])
    single = list(edge_transit_times(G).values())[0]
    assert times[1] == pytest.approx(single)
    assert times[2] == pytest.approx(2 * single)
    assert times[3] == pytest.approx(3 * single)


def test_an_inlet_is_at_zero():
    G = _chain(2)
    assert transit_time_from_inlets(G, inlets=[0])[0] == pytest.approx(0.0)


def test_the_fastest_route_is_taken_when_two_paths_reach_a_node():
    """Two parallel branches of different transit time reconverging."""
    G = nx.MultiGraph()
    G.add_edge(0, 1, key=0, length=10.0, assigned_diameter_um=8.0, flow_abs=4.0, flow_signed=4.0)
    G.add_edge(0, 2, key=0, length=10.0, assigned_diameter_um=8.0, flow_abs=1.0, flow_signed=1.0)
    G.add_edge(1, 3, key=0, length=1.0, assigned_diameter_um=8.0, flow_abs=4.0, flow_signed=4.0)
    G.add_edge(2, 3, key=0, length=1.0, assigned_diameter_um=8.0, flow_abs=1.0, flow_signed=1.0)

    times = transit_time_from_inlets(G, inlets=[0])
    via_fast = (np.pi * 16 * 10.0) / 4.0 + (np.pi * 16 * 1.0) / 4.0
    assert times[3] == pytest.approx(via_fast)


def test_flow_direction_is_respected_not_just_adjacency():
    """An edge carrying blood away from a node cannot deliver blood to it."""
    G = nx.MultiGraph()
    G.add_edge(0, 1, key=0, length=10.0, assigned_diameter_um=8.0, flow_abs=2.0, flow_signed=2.0)
    # Signed negative: flow runs 2 -> 1, so node 2 is not reachable from the inlet through it.
    G.add_edge(1, 2, key=0, length=10.0, assigned_diameter_um=8.0, flow_abs=2.0,
               flow_signed=-2.0)
    times = transit_time_from_inlets(G, inlets=[0])
    assert np.isfinite(times[1])
    assert not np.isfinite(times[2])


def test_a_node_with_no_route_from_any_inlet_is_infinite_not_missing():
    """Every node gets an entry, so a caller cannot mistake absence for zero."""
    G = _chain(2)
    G.add_edge(90, 91, key=0, length=10.0, assigned_diameter_um=8.0, flow_abs=1.0,
               flow_signed=1.0)
    times = transit_time_from_inlets(G, inlets=[0])
    assert set(times) == set(G.nodes())
    assert not np.isfinite(times[90])


def test_an_edge_without_a_diameter_is_refused():
    G = nx.MultiGraph()
    G.add_edge(0, 1, key=0, length=10.0, flow_abs=2.0, flow_signed=2.0)
    with pytest.raises(ValueError, match="diameter"):
        edge_transit_times(G)


def test_a_cycle_in_the_flow_directions_does_not_hang():
    """Flow directions come from a numerical solve and can contain a small loop."""
    G = nx.MultiGraph()
    common = dict(length=10.0, assigned_diameter_um=8.0, flow_abs=2.0)
    G.add_edge(0, 1, key=0, flow_signed=2.0, **common)     # 0 -> 1
    G.add_edge(1, 2, key=0, flow_signed=2.0, **common)     # 1 -> 2
    G.add_edge(1, 2, key=1, flow_signed=-2.0, **common)    # 2 -> 1, closing the loop

    times = transit_time_from_inlets(G, inlets=[0])
    assert np.isfinite(times[1])
    assert np.isfinite(times[2])
