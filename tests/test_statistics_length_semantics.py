"""Regression tests: length metrics must not be perturbed by haemodynamics.

Statistics previously read the overloaded ``weight`` attribute before
``length``. Because the pipeline runs haemodynamics before statistics, and
haemodynamics overwrote ``weight`` with conductance, every reported "length"
was actually a conductance -- `Total Edge Length (microns)` fell from 120.0 to
0.514 on the fixture below.
"""
import networkx as nx
import pytest

from haemolynx import statistics as st
from haemolynx.haemodynamics.poiseuille import PoiseuilleModel, set_edge_resistance

MODEL = PoiseuilleModel(constriction_length=40.0, constriction_spacing=100.0)

SEGMENT_LENGTH_UM = 60.0
DIAMETER_UM = 5.0


def _two_segment_graph() -> tuple[nx.MultiGraph, dict]:
    """Two collinear 60 um segments; total length is exactly 120 um."""
    G = nx.MultiGraph()
    positions = {
        0: [0.0, 0.0, 0.0],
        1: [0.0, 0.0, SEGMENT_LENGTH_UM],
        2: [0.0, 0.0, 2 * SEGMENT_LENGTH_UM],
    }
    for node, pos in positions.items():
        G.add_node(node, pos=pos)
    for u, v in ((0, 1), (1, 2)):
        G.add_edge(
            u,
            v,
            length=SEGMENT_LENGTH_UM,
            branch_order="B01",
            voxels=[positions[u], positions[v]],
        )
    return G, positions


def test_length_metrics_are_identical_before_and_after_haemodynamics():
    G, positions = _two_segment_graph()

    before = st.compute_comprehensive_vessel_statistics(
        G, node_positions=positions, statistics_mode="fast"
    )
    G, _ = MODEL.set_poiseuille_resistances(G, {"B01": DIAMETER_UM})
    after = st.compute_comprehensive_vessel_statistics(
        G, node_positions=positions, statistics_mode="fast"
    )

    for metric in (
        "Total Edge Length (microns)",
        "Average Edge Length (microns)",
        "Average Tortuosity Index",
        "Average Curvature",
    ):
        assert after[metric] == pytest.approx(before[metric]), (
            f"{metric} changed when haemodynamics ran: {before[metric]} -> {after[metric]}"
        )


def test_total_edge_length_is_the_true_geometric_length():
    G, positions = _two_segment_graph()
    G, _ = MODEL.set_poiseuille_resistances(G, {"B01": DIAMETER_UM})
    stats = st.compute_comprehensive_vessel_statistics(
        G, node_positions=positions, statistics_mode="fast"
    )
    assert stats["Total Edge Length (microns)"] == pytest.approx(2 * SEGMENT_LENGTH_UM)
    assert stats["Average Edge Length (microns)"] == pytest.approx(SEGMENT_LENGTH_UM)


def test_statistics_reject_a_graph_carrying_the_removed_weight_attribute():
    G, positions = _two_segment_graph()
    next(iter(G.edges(data=True)))[-1]["weight"] = 1.234
    with pytest.raises(ValueError, match="removed 'weight' attribute"):
        st.compute_comprehensive_vessel_statistics(
            G, node_positions=positions, statistics_mode="fast"
        )


def test_betweenness_resistance_model_uses_resistance_not_inverse_length():
    """The three distance models must be resistance, length and flow, not 1/weight."""
    G, _ = _two_segment_graph()
    G, _ = MODEL.set_poiseuille_resistances(G, {"B01": DIAMETER_UM})

    measurements = st.compute_betweenness_and_community_measurements(G)

    assert set(measurements) == {"edge_resistance", "edge_length", "edge_flow_abs"}
    for model in measurements.values():
        assert set(model) == {"Betweenness", "Communities"}


def test_betweenness_flow_model_treats_higher_flow_as_shorter_distance():
    """Flow weighting must prefer the busiest path, not the least-travelled one.

    Two parallel routes from 0 to 3: one (through 1) carries far more flow
    than the other (through 2). Weighting by inverse |flow| makes the
    high-flow route the shortest, so every 0->3 shortest path should run
    through node 1 and none through node 2.
    """
    G = nx.Graph()
    G.add_nodes_from((0, 1, 2, 3))
    G.add_edge(0, 1, flow_abs=100.0)
    G.add_edge(1, 3, flow_abs=100.0)
    G.add_edge(0, 2, flow_abs=1.0)
    G.add_edge(2, 3, flow_abs=1.0)

    result = st.compute_weighted_betweenness_summary(
        G, source_attr="flow_abs", inverse_source_attr=True
    )

    by_node = {row["node"]: row["value"] for row in result["Betweenness Top Nodes"]}
    assert by_node[1] > 0.0
    assert by_node[2] == pytest.approx(0.0)
