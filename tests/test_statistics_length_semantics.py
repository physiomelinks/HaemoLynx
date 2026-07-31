"""Regression tests: length metrics must not be perturbed by haemodynamics.

Statistics previously read the overloaded ``weight`` attribute before
``length``. Because the pipeline runs haemodynamics before statistics, and
haemodynamics overwrote ``weight`` with conductance, every reported "length"
was actually a conductance -- `Total Edge Length (microns)` fell from 120.0 to
0.514 on the fixture below.
"""
import networkx as nx
import pytest

from ImageLynx import statistics as st
from ImageLynx.haemodynamics.poiseuille import PoiseuilleModel, set_edge_resistance

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
    G, _ = MODEL.set_poiseuille_weights(G, {"B01": DIAMETER_UM})
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
    G, _ = MODEL.set_poiseuille_weights(G, {"B01": DIAMETER_UM})
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
    """The two distance models must be resistance and length, not 1/weight."""
    G, _ = _two_segment_graph()
    G, _ = MODEL.set_poiseuille_weights(G, {"B01": DIAMETER_UM})

    measurements = st.compute_betweenness_and_community_measurements(G)

    assert set(measurements) == {"edge_resistance", "edge_length"}
    for model in measurements.values():
        assert set(model) == {"Betweenness", "Communities"}
