"""Post-processing rules behind the "10. Post processing" tab.

Junctions where four or more vessels meet: list them, delete vessels, split
with a connector -- pure graph logic, pinned on small hand-built networks
whose answers are known.
"""
from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from haemolynx.graph import (
    DEFAULT_SPLIT_CONNECTOR_LENGTH_UM,
    add_vessel_between,
    delete_vessels,
    edge_keys,
    high_degree_junctions,
    junction_vessels,
    mean_incident_diameter,
    prune_disconnected_branches,
    split_junction,
    vessel_path_between,
)
from haemolynx.graph._helpers import calculate_path_length


def _straight(a, b, step: float = 1.0) -> list[tuple[float, float, float]]:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    count = max(2, int(np.ceil(np.linalg.norm(b - a) / step)) + 1)
    return [tuple(float(c) for c in p) for p in np.linspace(a, b, count)]


def _network(positions: dict, edges: list[tuple], **edge_attrs) -> nx.MultiGraph:
    G = nx.MultiGraph()
    for node, pos in positions.items():
        G.add_node(node, pos=np.asarray(pos, dtype=float))
    for u, v in edges:
        voxels = _straight(positions[u], positions[v])
        G.add_edge(u, v, voxels=voxels, length=calculate_path_length(voxels), **edge_attrs)
    return G


# --- high_degree_junctions / junction_vessels --------------------------------


def _star(degree: int, centre: int = 0, first_arm: int = 1) -> tuple[dict, list]:
    positions = {centre: (0.0, 0.0, 0.0)}
    edges = []
    for i in range(degree):
        angle = 2 * np.pi * i / degree
        node = first_arm + i
        positions[node] = (0.0, 20 * np.sin(angle), 20 * np.cos(angle))
        edges.append((centre, node))
    return positions, edges


def test_high_degree_junctions_lists_four_or_more_busiest_first():
    p4, e4 = _star(4, centre=0, first_arm=1)
    p5, e5 = _star(5, centre=100, first_arm=101)
    p3, e3 = _star(3, centre=200, first_arm=201)
    G = _network({**p4, **p5, **p3}, e4 + e5 + e3)
    assert high_degree_junctions(G) == [100, 0]
    assert high_degree_junctions(G, min_degree=3) == [100, 0, 200]


def test_junction_vessels_reports_branch_id_length_diameter_and_order():
    positions, edges = _star(4)
    G = _network(positions, edges, branch_order="B02")
    G.add_edge(1, 2, voxels=_straight(positions[1], positions[2]), length=28.0)
    first = next(iter(G.edges(0, keys=True)))
    G.edges[first]["diameter_um"] = 5.5

    rows = junction_vessels(G, 0)
    keys = edge_keys(G)
    assert [row.other_node for row in rows] == [1, 2, 3, 4]
    for row in rows:
        assert keys[row.branch_id] == row.edge
        assert row.length_um == pytest.approx(20.0, abs=1e-6)
        assert row.branch_order == "B02"
    assert rows[0].diameter_um == 5.5
    assert all(row.diameter_um is None for row in rows[1:])
    # The 1-2 vessel does not touch node 0.
    assert all(0 in (row.u, row.v) for row in rows)
    with pytest.raises(ValueError):
        junction_vessels(G, 12345)


# --- delete_vessels ----------------------------------------------------------


def _four_way_through() -> nx.MultiGraph:
    """Junction 0 with arms to 1..4; arms 1 and 3 carry on to 5 and 6."""
    positions, edges = _star(4)
    positions.update({5: (0.0, 0.0, 40.0), 6: (0.0, 0.0, -40.0)})
    return _network(positions, edges + [(1, 5), (3, 6)])


def test_delete_one_vessel_at_a_four_way_leaves_a_bifurcation():
    G = _four_way_through()
    edge = next(e for e in edge_keys(G) if set(e[:2]) == {0, 2})
    delete_vessels(G, [edge])
    assert G.degree(0) == 3
    assert 2 not in G  # its far end was left with nothing
    assert G.number_of_edges() == 5


def test_delete_two_vessels_merges_the_pass_through_left_behind():
    G = _four_way_through()
    doomed = [e for e in edge_keys(G) if set(e[:2]) in ({0, 2}, {0, 4})]
    changed = delete_vessels(G, doomed)
    # 0 is now 1-0-3: merged away into one vessel from 1 to 3.
    assert 0 not in G
    assert {0, 1, 3} <= changed
    assert G.has_edge(1, 3)
    merged = G.get_edge_data(1, 3)[0]
    assert merged["length"] == pytest.approx(40.0, abs=1e-6)
    assert merged["length"] == pytest.approx(calculate_path_length(merged["voxels"]))


def test_delete_never_merges_a_protected_node():
    G = _four_way_through()
    doomed = [e for e in edge_keys(G) if set(e[:2]) in ({0, 2}, {0, 4})]
    delete_vessels(G, doomed, protected=[0])
    assert 0 in G and G.degree(0) == 2
    assert not G.has_edge(1, 3)


def test_delete_refuses_to_strand_a_boundary_node_and_changes_nothing():
    G = _four_way_through()
    before = sorted(edge_keys(G))
    edge = next(e for e in edge_keys(G) if set(e[:2]) == {1, 5})
    with pytest.raises(ValueError, match="boundary node"):
        delete_vessels(G, [edge], protected=[5])
    assert sorted(edge_keys(G)) == before


def test_delete_rejects_a_vessel_that_is_not_there():
    G = _four_way_through()
    with pytest.raises(ValueError, match="No vessel"):
        delete_vessels(G, [(0, 999, 0)])


# --- split_junction ----------------------------------------------------------


def _four_way_with_one_close_pair(length: float = 50.0) -> nx.MultiGraph:
    """Arms at 10 and -10 degrees from +x (a close pair) and at 150 and 210."""
    positions = {0: (0.0, 0.0, 0.0)}
    edges = []
    for node, degrees in zip((1, 2, 3, 4), (10, -10, 150, 210)):
        angle = np.radians(degrees)
        positions[node] = (0.0, length * np.sin(angle), length * np.cos(angle))
        edges.append((0, node))
    return _network(positions, edges, diameter_um=6.0, branch_order="B03")


def test_split_four_way_makes_two_bifurcations_joined_by_a_15um_vessel():
    G = _four_way_with_one_close_pair()
    new_nodes = split_junction(G, 0)
    assert len(new_nodes) == 1
    new = new_nodes[0]
    assert G.degree(0) == 3 and G.degree(new) == 3
    # The close pair (10 and -10 degrees) moved out along +x.
    assert {other for _, other in G.edges(new) if other != 0} == {1, 2}
    assert {other for _, other in G.edges(0) if other != new} == {3, 4}
    assert np.allclose(G.nodes[new]["pos"], (0.0, 0.0, DEFAULT_SPLIT_CONNECTOR_LENGTH_UM))

    connector = G.get_edge_data(0, new)[0]
    assert connector["length"] == pytest.approx(DEFAULT_SPLIT_CONNECTOR_LENGTH_UM)
    assert connector["junction_split_connector"] is True
    # Every arm is 6 um across, so the connector takes their mean as an override.
    assert connector["diameter_um"] == pytest.approx(6.0)
    assert connector["diameter_source"] == "override"
    assert "branch_order" not in connector


def test_split_trims_moved_vessels_and_remeasures_them():
    G = _four_way_with_one_close_pair()
    new = split_junction(G, 0)[0]
    for other in (1, 2):
        data = G.get_edge_data(new, other)[0]
        assert np.allclose(data["voxels"][0], G.nodes[new]["pos"])
        assert data["length"] == pytest.approx(calculate_path_length(data["voxels"]))
        # The part the connector now covers is gone: shorter than the arm was.
        assert 30.0 < data["length"] < 50.0
        # Non-geometric attributes ride along.
        assert data["diameter_um"] == 6.0 and data["branch_order"] == "B03"
    for other in (3, 4):
        assert G.get_edge_data(0, other)[0]["length"] == pytest.approx(50.0, abs=1e-6)


def test_split_five_way_peels_twice_to_bifurcations_only():
    positions, edges = _star(5)
    G = _network(positions, edges)
    new_nodes = split_junction(G, 0, connector_length_um=10.0)
    assert len(new_nodes) == 2
    assert max(G.degree(n) for n in [0, *new_nodes]) == 3
    assert G.number_of_edges() == 5 + 2
    assert all(
        G.get_edge_data(0, n)[0]["length"] == pytest.approx(10.0) for n in new_nodes
    )
    assert nx.is_connected(G)


def test_split_leaves_a_bifurcation_alone_and_respects_reserved_ids():
    positions, edges = _star(3)
    G = _network(positions, edges)
    assert split_junction(G, 0) == []
    G4 = _four_way_with_one_close_pair()
    new = split_junction(G4, 0, reserved_ids={5, 6, 7})
    assert new == [8]


def test_split_rejects_a_non_positive_connector():
    G = _four_way_with_one_close_pair()
    with pytest.raises(ValueError, match="positive"):
        split_junction(G, 0, connector_length_um=0.0)


# --- a new vessel between two nodes -------------------------------------------


def _two_ends() -> nx.MultiGraph:
    """Two separate 2-vessel chains whose ends 1 and 4 a new vessel can join.

    0 -(4 um)- 1 -(6 um)- 2   and   3 -(8 um)- 4 -(no diameter)- 5, all in
    the z = 0 plane, with 1 at (0, 0, 10) and 4 at (0, 20, 10).
    """
    positions = {
        0: (0.0, 0.0, 0.0), 1: (0.0, 0.0, 10.0), 2: (0.0, 0.0, 20.0),
        3: (0.0, 20.0, 0.0), 4: (0.0, 20.0, 10.0), 5: (0.0, 20.0, 20.0),
    }
    G = _network(positions, [(0, 1), (1, 2), (3, 4), (4, 5)])
    for (u, v), diameter in zip([(0, 1), (1, 2), (3, 4)], [4.0, 6.0, 8.0]):
        G.edges[u, v, 0]["diameter_um"] = diameter
    return G


def test_mean_incident_diameter_averages_the_vessels_at_both_nodes():
    G = _two_ends()
    assert mean_incident_diameter(G, [1, 4]) == pytest.approx((4 + 6 + 8) / 3)
    assert mean_incident_diameter(G, [1]) == pytest.approx(5.0)
    # A vessel with no diameter is skipped, and none at all gives None.
    assert mean_incident_diameter(G, [5]) is None
    # A vessel joining both nodes counts once.
    G.add_edge(1, 4, voxels=_straight((0, 0, 10), (0, 20, 10)), length=20.0, diameter_um=2.0)
    assert mean_incident_diameter(G, [1, 4]) == pytest.approx((4 + 6 + 8 + 2) / 4)


def _bent_mask() -> np.ndarray:
    """A 1-voxel-thick L-shaped vessel from voxel (0, 0, 10) to (0, 20, 10)
    that detours through y..x = (0..20, 30): routing should follow it."""
    mask = np.zeros((1, 25, 35), dtype=bool)
    mask[0, 0, 10:31] = True
    mask[0, 0:21, 30] = True
    mask[0, 20, 10:31] = True
    return mask


def test_path_is_routed_through_the_mask_when_it_can_be():
    from haemolynx.graph import mask_cost_field

    G = _two_ends()
    mask = _bent_mask()
    points, how = vessel_path_between(
        G, 1, 4, cost_field=mask_cost_field(mask), mask=mask, voxel_size_zyx=(1.0, 1.0, 1.0)
    )
    assert how == "routed"
    assert np.allclose(points[0], (0, 0, 10)) and np.allclose(points[-1], (0, 20, 10))
    # It went round the L (out to x = 30), not straight across.
    assert max(p[2] for p in points) >= 29
    assert calculate_path_length(points) > 50


def test_path_falls_back_to_straight_without_an_image_or_off_the_vessel():
    from haemolynx.graph import mask_cost_field

    G = _two_ends()
    straight = [(0.0, 0.0, 10.0), (0.0, 20.0, 10.0)]
    assert vessel_path_between(G, 1, 4) == (straight, "straight")
    # A mask with nothing between the nodes: any route leaves the vessel.
    empty = np.zeros((1, 25, 35), dtype=bool)
    empty[0, 0, 0] = True
    points, how = vessel_path_between(
        G, 1, 4, cost_field=mask_cost_field(empty), mask=empty
    )
    assert (points, how) == (straight, "straight")


def test_add_vessel_between_measures_it_and_keeps_the_diameter_as_override():
    G = _two_ends()
    u, v, key = add_vessel_between(G, 1, 4, diameter_um=6.0)
    data = G.edges[u, v, key]
    assert data["length"] == pytest.approx(20.0)
    assert data["diameter_um"] == pytest.approx(6.0)
    assert data["diameter_source"] == "override"
    assert data["post_processing_added"] is True
    assert G.degree(1) == 3 and G.degree(4) == 3
    # Without a diameter the Diameters stage decides it later.
    u, v, key = add_vessel_between(G, 0, 3, points_um=[(0, 0, 0), (0, 10, 0), (0, 20, 0)])
    assert "diameter_um" not in G.edges[u, v, key]
    assert G.edges[u, v, key]["length"] == pytest.approx(20.0)


def test_add_vessel_between_rejects_a_self_loop_or_a_missing_node():
    G = _two_ends()
    with pytest.raises(ValueError, match="two different"):
        add_vessel_between(G, 1, 1)
    with pytest.raises(ValueError, match="not in the network"):
        add_vessel_between(G, 1, 99)



# --- prune_disconnected_branches ----------------------------------------------


def _main_and_cut_off_pieces() -> nx.MultiGraph:
    """Inlet 0 -> 1 -> outlet 2 (kept); 3 -> inlet 4 (inlet only);
    5 -> 6 (no boundary node at all)."""
    positions = {n: (0.0, float(n), 0.0) for n in range(7)}
    return _network(positions, [(0, 1), (1, 2), (3, 4), (5, 6)])


def test_prune_drops_every_piece_without_both_an_inlet_and_an_outlet():
    G = _main_and_cut_off_pieces()
    pruned, stats = prune_disconnected_branches(G, [0, 4], [2])
    assert sorted(pruned.nodes) == [0, 1, 2]
    assert stats["removed_components"] == 2
    assert stats["removed_vessels"] == 2
    assert stats["removed_boundary_nodes"] == [4]
    assert G.number_of_edges() == 4  # the input graph is left as it was


def test_prune_with_nothing_to_drop_changes_nothing():
    positions = {n: (0.0, float(n), 0.0) for n in range(3)}
    G = _network(positions, [(0, 1), (1, 2)])
    pruned, stats = prune_disconnected_branches(G, [0], [2])
    assert stats["removed_components"] == 0 and stats["removed_vessels"] == 0
    assert pruned.number_of_edges() == 2


def test_prune_refuses_to_remove_every_vessel():
    G = _main_and_cut_off_pieces()
    with pytest.raises(ValueError, match="every vessel"):
        prune_disconnected_branches(G, [4], [2])  # no piece has both


def test_split_connector_diameter_is_the_mean_of_the_junctions_vessels():
    G = _four_way_with_one_close_pair()
    for (u, v, k), diameter in zip(edge_keys(G), [4.0, 6.0, 8.0, 10.0]):
        G.edges[u, v, k]["diameter_um"] = diameter
    new = split_junction(G, 0)[0]
    assert G.get_edge_data(0, new)[0]["diameter_um"] == pytest.approx(7.0)
    # With no diameters at all, the connector gets none (the table decides).
    positions, edges = _star(4)
    bare = _network(positions, edges)
    new = split_junction(bare, 0)[0]
    assert "diameter_um" not in bare.get_edge_data(0, new)[0]
