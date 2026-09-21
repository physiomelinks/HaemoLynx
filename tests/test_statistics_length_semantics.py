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


def test_compute_weighted_communities_summary_accepts_a_precomputed_partition(monkeypatch):
    """Regression: precomputed= must be usable in place of recomputing the
    partition via communities_for_weighting, and produce the same summary."""
    from haemolynx.graph.communities import communities_for_weighting
    from haemolynx.statistics import network_measures as nm

    G, _ = _two_segment_graph()
    G, _ = MODEL.set_poiseuille_resistances(G, {"B01": DIAMETER_UM})

    baseline = st.compute_weighted_communities_summary(
        G, source_attr="resistance", inverse_source_attr=False
    )
    precomputed = communities_for_weighting(G, "resistance")

    def _boom(*_args, **_kwargs):
        raise AssertionError("communities_for_weighting should not be called again")

    monkeypatch.setattr(nm, "communities_for_weighting", _boom)
    reused = st.compute_weighted_communities_summary(
        G, source_attr="resistance", inverse_source_attr=False, precomputed=precomputed
    )
    assert reused == baseline


def test_compute_betweenness_and_community_measurements_reuses_precomputed_communities(monkeypatch):
    from haemolynx.graph.communities import communities_for_weighting
    from haemolynx.statistics import network_measures as nm

    G, _ = _two_segment_graph()
    G, _ = MODEL.set_poiseuille_resistances(G, {"B01": DIAMETER_UM})

    baseline = st.compute_betweenness_and_community_measurements(G)
    precomputed = communities_for_weighting(G, "resistance")

    calls = []
    original = nm.communities_for_weighting

    def _spy(graph_arg, weighting, *args, **kwargs):
        calls.append(weighting)
        return original(graph_arg, weighting, *args, **kwargs)

    monkeypatch.setattr(nm, "communities_for_weighting", _spy)
    reused = st.compute_betweenness_and_community_measurements(
        G, precomputed_communities={"resistance": precomputed}
    )

    assert "resistance" not in calls  # reused, not recomputed
    assert set(calls) == {"length", "flow"}  # the other two still compute normally
    assert reused["edge_resistance"]["Communities"] == baseline["edge_resistance"]["Communities"]


# --- compute_flow_hierarchy ---------------------------------------------------


def test_flow_hierarchy_pure_tree_flow_is_one():
    """A branching tree with flow moving strictly root-to-leaves has no
    directed cycle at all -- every edge counts, hierarchy is exactly 1.0."""
    G = nx.MultiGraph()
    G.add_edge(0, 1, flow_signed=2.0, flow_abs=2.0)
    G.add_edge(1, 2, flow_signed=1.0, flow_abs=1.0)
    G.add_edge(1, 3, flow_signed=1.0, flow_abs=1.0)
    s = st.compute_flow_hierarchy(G)
    assert s["Flow Hierarchy"] == pytest.approx(1.0)
    assert s["Flow Hierarchy Edges Skipped"] == 0


def test_flow_hierarchy_negative_signed_flow_reverses_the_edge_direction():
    """A negative flow_signed must reverse that edge relative to a positive
    one -- confirms direction, not just magnitude, drives the directed
    graph flow_hierarchy is computed from. 0<-1->2->0 (1 is a source, not
    part of the 2-node path) has no directed cycle."""
    G = nx.MultiGraph()
    G.add_edge(0, 1)
    G.add_edge(1, 2)
    G.add_edge(2, 0)
    for u, v, key in list(G.edges(keys=True)):
        if {u, v} == {0, 1}:
            points_u_to_v = v == 0  # want 1 -> 0
        elif {u, v} == {1, 2}:
            points_u_to_v = v == 2  # want 1 -> 2
        else:
            points_u_to_v = v == 0  # want 2 -> 0
        G[u][v][key]["flow_signed"] = 1.0 if points_u_to_v else -1.0
        G[u][v][key]["flow_abs"] = 1.0

    s = st.compute_flow_hierarchy(G)
    assert s["Flow Hierarchy"] == pytest.approx(1.0)


def test_flow_hierarchy_detects_a_genuine_flow_recirculation():
    """A 3-node loop where every edge's solved flow points consistently
    around the loop is a real directed cycle -- physically impossible at
    steady state, but exactly what flow_hierarchy exists to catch.

    flow_signed's sign means "u -> v" for whatever (u, v) G.edges() itself
    reports -- which, for an undirected (Multi)Graph, is decided by
    adjacency-dict order, not by the order add_edge(u, v) was called with.
    So the sign for each edge below is picked from the graph's own reported
    (u, v), not assumed from the add_edge calls, to actually build a cycle
    rather than accidentally build a DAG that happens to look like one.
    """
    G = nx.MultiGraph()
    G.add_edge(0, 1)
    G.add_edge(1, 2)
    G.add_edge(2, 0)
    cycle = {0: 1, 1: 2, 2: 0}  # the directed cycle this test wants: 0->1->2->0
    for u, v, key in list(G.edges(keys=True)):
        wants_u_to_v = cycle[u] == v
        G[u][v][key]["flow_signed"] = 1.0 if wants_u_to_v else -1.0
        G[u][v][key]["flow_abs"] = 1.0

    s = st.compute_flow_hierarchy(G)
    assert s["Flow Hierarchy"] == pytest.approx(0.0)


def test_flow_hierarchy_skips_edges_with_no_determinate_flow():
    G = nx.MultiGraph()
    G.add_edge(0, 1, flow_signed=1.0, flow_abs=1.0)
    G.add_edge(1, 2, flow_signed=0.0, flow_abs=0.0)  # no determinate direction
    G.add_edge(2, 3)  # never solved at all
    s = st.compute_flow_hierarchy(G)
    assert s["Flow Hierarchy Edges Skipped"] == 2
    assert s["Flow Hierarchy"] == pytest.approx(1.0)  # the one remaining edge


def test_flow_hierarchy_no_solved_flow_at_all_is_na():
    G = nx.MultiGraph()
    G.add_edge(0, 1, length=1.0)
    s = st.compute_flow_hierarchy(G)
    assert s["Flow Hierarchy"] == "N/A (no solved flow)"
    assert s["Flow Hierarchy Edges Skipped"] == 1
