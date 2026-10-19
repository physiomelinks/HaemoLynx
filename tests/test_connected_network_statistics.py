"""Geometry and topology statistics a second time on the inlet/outlet-connected
network, when the run keeps its disconnected vessels."""
from __future__ import annotations

import networkx as nx
import pytest

from haemolynx import statistics
from haemolynx.pipeline import default_schema, stages

SCHEMA = default_schema()
VESSEL_UM = 10.0


def _graph() -> nx.MultiGraph:
    """Nodes 0-1-2 form the inlet-to-outlet path (2 vessels); 3-4 is a
    disconnected vessel of its own."""
    G = nx.MultiGraph()
    for node, z in {0: 0, 1: 10, 2: 20, 3: 100, 4: 110}.items():
        G.add_node(node, pos=(float(z), 0.0, 0.0))
    for u, v in ((0, 1), (1, 2), (3, 4)):
        G.add_edge(u, v, length=VESSEL_UM, diameter_um=4.0,
                   voxels=[G.nodes[u]["pos"], G.nodes[v]["pos"]])
    return G


def _settings(**overrides) -> dict:
    settings = SCHEMA.defaults()
    settings.update(
        statistics=True,
        statistics_mode="fast",
        inlet_nodes=[0],
        outlet_nodes=[2],
        arteriole_boundary_nodes=[],
        venule_boundary_nodes=[],
    )
    settings.update(overrides)
    return settings


def _connected(G, **overrides):
    return stages.connected_network_statistics(
        _settings(**overrides), G, (30, 30, 130), (1.0, 1.0, 1.0), None
    )


def _length_row(stats: dict) -> float:
    key = next(k for k in stats if "Total" in k and "Length" in k)
    return stats[key]


def test_measures_are_only_geometry_and_topology():
    assert statistics.CONNECTED_NETWORK_MEASURES <= set(statistics.STATISTIC_MEASURES)
    # Anything reading inlets, outlets or flow already sees only what is connected.
    for measure in ("bottlenecks", "shunts", "transit_time", "flux_heterogeneity",
                    "current_flow", "strahler", "algebraic_connectivity",
                    "network_robustness"):
        assert measure not in statistics.CONNECTED_NETWORK_MEASURES


def test_connected_version_leaves_out_the_disconnected_vessel():
    G = _graph()
    connected = _connected(G)
    assert connected is not None
    full = statistics.compute_comprehensive_vessel_statistics(
        G, node_positions=nx.get_node_attributes(G, "pos"),
        enabled_measures=statistics.CONNECTED_NETWORK_MEASURES,
    )
    assert _length_row(full) == pytest.approx(3 * VESSEL_UM)
    assert _length_row(connected) == pytest.approx(2 * VESSEL_UM)


def test_the_runs_graph_is_not_changed():
    G = _graph()
    _connected(G)
    assert G.number_of_edges() == 3
    assert G.number_of_nodes() == 5


def test_none_when_disconnected_vessels_are_already_removed():
    assert _connected(_graph(), boundary_handling="remove_disconnected") is None


def test_none_without_an_inlet_or_outlet():
    assert _connected(_graph(), inlet_nodes=[]) is None
    assert _connected(_graph(), outlet_nodes=[]) is None
