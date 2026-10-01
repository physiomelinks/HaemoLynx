"""Post-processing rules behind the "6. Post processing" tab.

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
    IntensityCostField,
    VesselEnd,
    add_traced_vessel,
    add_vessel_between,
    dead_end_vessels,
    delete_dead_end_vessels,
    delete_vessels,
    edge_keys,
    high_degree_junctions,
    junction_vessels,
    mask_cost_field,
    mean_incident_diameter,
    prune_disconnected_branches,
    smooth_traced_path,
    split_high_degree_junctions,
    split_junction,
    split_vessel_at,
    trace_path,
    vessel_path,
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


# --- split_high_degree_junctions (the split_junctions haematocrit rule) -------


def test_split_every_high_degree_junction_leaves_only_bifurcations():
    p4, e4 = _star(4, centre=0, first_arm=1)
    p5, e5 = _star(5, centre=100, first_arm=101)
    p3, e3 = _star(3, centre=200, first_arm=201)
    G = _network({**p4, **p5, **p3}, e4 + e5 + e3, diameter_um=6.0, branch_order="B02")

    split = split_high_degree_junctions(G)

    assert sorted(split) == [0, 100]
    assert len(split[0]) == 1 and len(split[100]) == 2
    assert max(degree for _node, degree in G.degree()) == 3
    assert G.degree(200) == 3  # a bifurcation is left as it was
    assert G.number_of_edges() == 12 + 3


def test_a_split_connector_takes_the_branch_order_of_the_vessels_peeled_onto_it():
    """A run does not regenerate branch orders afterwards, and an edge with no
    branch_order gets no resistance, so each connector is given one: the order
    its two vessels share, else the wider one's."""
    G = _four_way_with_one_close_pair()
    shared = split_high_degree_junctions(G)
    (new,) = shared[0]
    assert G.get_edge_data(0, new)[0]["branch_order"] == "B03"

    G = _four_way_with_one_close_pair()
    # The close pair (arms 1 and 2) is peeled together; give them different
    # orders and widths.
    G.edges[0, 1, 0].update(branch_order="B04", diameter_um=5.0)
    G.edges[0, 2, 0].update(branch_order="Art2", diameter_um=9.0)
    (new,) = split_high_degree_junctions(G)[0]
    connector = G.get_edge_data(0, new)[0]
    assert connector["branch_order"] == "Art2"
    assert connector["junction_split_connector"] is True


def test_split_high_degree_junctions_leaves_an_unlabelled_connector_unlabelled():
    positions, edges = _star(4)
    G = _network(positions, edges)
    (new,) = split_high_degree_junctions(G)[0]
    assert "branch_order" not in G.get_edge_data(0, new)[0]


def test_a_run_splits_each_junction_with_connectors_as_long_as_its_vessels_are_wide():
    """Two merged bifurcations were about a vessel diameter apart, so each
    junction's connector length is its own vessels' mean diameter -- one
    length for a capillary junction, another for an arteriole's."""
    p4, e4 = _star(4, centre=0, first_arm=1)
    q4, f4 = _star(4, centre=100, first_arm=101)
    G = _network({**p4, **q4}, e4 + f4)
    for u, v, k in edge_keys(G):
        G.edges[u, v, k]["diameter_um"] = 6.0 if 0 in (u, v) else 12.0

    split = split_high_degree_junctions(G)

    for junction, width in ((0, 6.0), (100, 12.0)):
        (new,) = split[junction]
        connector = G.get_edge_data(junction, new)[0]
        assert connector["length"] == pytest.approx(width)
        assert connector["diameter_um"] == pytest.approx(width)
        assert np.linalg.norm(G.nodes[new]["pos"] - G.nodes[junction]["pos"]) == pytest.approx(width)


def test_a_run_split_takes_a_fixed_connector_length_when_given_one():
    G = _four_way_with_one_close_pair()
    (new,) = split_high_degree_junctions(G, connector_length_um=20.0)[0]
    assert G.get_edge_data(0, new)[0]["length"] == pytest.approx(20.0)


def test_a_junction_with_no_diameters_falls_back_to_the_tabs_connector_length():
    positions, edges = _star(4)
    G = _network(positions, edges)
    (new,) = split_high_degree_junctions(G)[0]
    assert G.get_edge_data(0, new)[0]["length"] == pytest.approx(
        DEFAULT_SPLIT_CONNECTOR_LENGTH_UM
    )


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


# --- tracing a new vessel (Add vessel) -----------------------------------------


def _one_vessel(voxels, **attrs) -> nx.MultiGraph:
    """Node 0 at the first point of *voxels*, node 1 at the last, one vessel."""
    G = nx.MultiGraph()
    G.add_node(0, pos=np.asarray(voxels[0], dtype=float))
    G.add_node(1, pos=np.asarray(voxels[-1], dtype=float))
    G.add_edge(0, 1, voxels=list(voxels), length=calculate_path_length(voxels), **attrs)
    return G


def test_split_vessel_at_cuts_between_path_points_and_keeps_its_attributes():
    # A straight two-point vessel: the cut lands between its only two points.
    G = _one_vessel(
        [(0.0, 0.0, 0.0), (0.0, 0.0, 20.0)],
        diameter_um=5.0, diameter_source="measured", branch_order="B02",
    )
    node = split_vessel_at(G, (0, 1, 0), (0.0, 3.0, 7.0), reserved_ids={2})
    assert node == 3  # the next free id, past the reserved one
    assert np.allclose(G.nodes[node]["pos"], (0.0, 0.0, 7.0))
    assert not G.has_edge(0, 1)
    first, second = G.edges[0, node, 0], G.edges[node, 1, 0]
    assert first["length"] == pytest.approx(7.0) and second["length"] == pytest.approx(13.0)
    for half in (first, second):
        assert half["diameter_um"] == 5.0 and half["diameter_source"] == "measured"
        assert half["branch_order"] == "B02"
    assert np.allclose(first["voxels"][-1], (0, 0, 7)) and np.allclose(second["voxels"][0], (0, 0, 7))


def test_split_vessel_at_follows_a_path_stored_the_other_way_round():
    G = nx.MultiGraph()
    G.add_node(0, pos=np.asarray((0.0, 0.0, 0.0)))
    G.add_node(1, pos=np.asarray((0.0, 0.0, 20.0)))
    G.add_edge(0, 1, voxels=_straight((0, 0, 20), (0, 0, 0)), length=20.0)
    node = split_vessel_at(G, (0, 1, 0), (0.0, 0.0, 5.0))
    # The half at node 0 is the 5 um nearest it, not the 15 um at node 1.
    assert G.edges[0, node, 0]["length"] == pytest.approx(5.0)
    assert G.edges[node, 1, 0]["length"] == pytest.approx(15.0)
    assert np.allclose(G.edges[0, node, 0]["voxels"][0], (0, 0, 0))


def test_split_vessel_at_an_end_joins_that_node_instead():
    G = _one_vessel(_straight((0, 0, 0), (0, 0, 20)))
    assert split_vessel_at(G, (0, 1, 0), (0.0, 0.0, 0.6)) == 0
    assert split_vessel_at(G, (0, 1, 0), (0.0, 0.0, 19.5)) == 1
    assert sorted(G.nodes) == [0, 1] and G.number_of_edges() == 1


def test_trace_path_follows_the_mask_and_ends_exactly_on_its_points():
    mask = _bent_mask()
    points = trace_path(mask_cost_field(mask), (0.0, 0.0, 10.2), (0.0, 20.0, 9.8))
    assert points[0] == (0.0, 0.0, 10.2) and points[-1] == (0.0, 20.0, 9.8)
    assert max(p[2] for p in points) >= 29  # round the L, not straight across
    # No cost field: the straight line.
    assert trace_path(None, (0, 0, 10), (0, 20, 10)) == [(0.0, 0.0, 10.0), (0.0, 20.0, 10.0)]


def test_trace_path_scales_voxels_to_microns():
    mask = np.zeros((1, 25, 35), dtype=bool)
    mask[0, 0:21, 10] = True  # straight down y at voxel x = 10
    points = trace_path(
        mask_cost_field(mask), (0.0, 0.0, 5.0), (0.0, 10.0, 5.0), voxel_size_zyx=(1.0, 0.5, 0.5)
    )
    assert all(p[2] == pytest.approx(5.0) for p in points)
    assert calculate_path_length(points) == pytest.approx(10.0)


def _bright_l(noise_seed: int = 0) -> np.ndarray:
    """Raw intensities: the L of _bent_mask, bright, over dim noise."""
    rng = np.random.default_rng(noise_seed)
    raw = rng.uniform(0.0, 20.0, size=(1, 25, 35)).astype(np.float32)
    raw[_bent_mask()] = 200.0
    return raw


def test_intensity_cost_is_cheap_on_bright_voxels_and_dear_on_dark_ones():
    field = IntensityCostField(_bright_l(), sigma_vox=0.0)
    window = field[0:1, 0:25, 0:35]
    assert window.shape == (1, 25, 35)
    assert window.min() == pytest.approx(1.0)
    assert window.max() <= 1.0 / 0.01 + 1e-9
    assert window[0, 0, 20] == pytest.approx(1.0)  # on the L
    assert window[0, 10, 20] > 10.0  # in the dark middle
    with pytest.raises(ValueError, match="3D"):
        IntensityCostField(np.zeros((5, 5)))


def test_trace_path_follows_the_bright_vessel_in_raw_data():
    points = trace_path(IntensityCostField(_bright_l()), (0.0, 0.0, 10.0), (0.0, 20.0, 10.0))
    assert max(p[2] for p in points) >= 29
    assert calculate_path_length(points) > 50


def _staircase(steps: int = 20, dx: float = 1.0) -> list[tuple[float, float, float]]:
    """A voxel staircase: alternately one step in y and *dx* in x."""
    stairs = [(0.0, 0.0, 0.0)]
    for i in range(steps):
        y, x = stairs[-1][1], stairs[-1][2]
        stairs.append((0.0, y + 1.0, x) if i % 2 == 0 else (0.0, y, x + dx))
    return stairs


def test_smooth_traced_path_takes_out_the_staircase_without_moving_the_ends():
    stairs = _staircase()
    smoothed, outcome = smooth_traced_path(stairs)
    assert outcome in ("smoothed", "relaxed")
    assert smoothed[0] == stairs[0] and smoothed[-1] == stairs[-1]
    assert calculate_path_length(smoothed) < calculate_path_length(stairs)
    assert calculate_path_length(smoothed) >= np.hypot(10.0, 10.0) - 1e-9
    assert smooth_traced_path(stairs[:2]) == (stairs[:2], "too_short")


def test_add_traced_vessel_from_a_node_to_a_point_on_a_vessel_forms_a_node():
    G = _two_ends()
    # 1 -> a waypoint -> 4 um along 3 -> 4 (which runs x = 0..10 at y = 20).
    path = [(0.0, 0.0, 10.0), (0.0, 10.0, 6.0), (0.0, 20.0, 4.0)]
    traced = add_traced_vessel(
        G, VesselEnd.at_node(1), VesselEnd.on_vessel((3, 4, 0), (0.0, 20.5, 4.0)), path,
        smoothing=None,
    )
    new = traced.new_nodes[0]
    assert traced.split_vessels == ((3, 4, 0),)
    assert np.allclose(G.nodes[new]["pos"], (0.0, 20.0, 4.0))
    assert not G.has_edge(3, 4)
    assert G.edges[3, new, 0]["length"] == pytest.approx(4.0)
    assert G.edges[new, 4, 0]["length"] == pytest.approx(6.0)
    assert G.edges[3, new, 0]["diameter_um"] == G.edges[new, 4, 0]["diameter_um"] == 8.0

    data = G.edges[traced.edge]
    assert set(traced.edge[:2]) == {1, new}
    assert data["post_processing_added"] is True
    assert data["length"] == pytest.approx(calculate_path_length(path))
    # The mean of the vessels at node 1 (4 and 6 um) and at the new node (8, 8).
    assert data["diameter_um"] == pytest.approx(6.5) == traced.diameter_um
    assert data["diameter_source"] == "override"


def test_add_traced_vessel_with_both_ends_on_one_vessel_cuts_it_twice():
    G = _one_vessel(_straight((0, 0, 0), (0, 0, 40)), diameter_um=5.0)
    path = [(0.0, 0.0, 10.0), (0.0, 10.0, 10.0), (0.0, 10.0, 30.0), (0.0, 0.0, 30.0)]
    traced = add_traced_vessel(
        G,
        VesselEnd.on_vessel((0, 1, 0), (0.0, 0.0, 10.0)),
        VesselEnd.on_vessel((0, 1, 0), (0.0, 0.0, 30.0)),
        path,
        smoothing=None,
    )
    a, b = traced.new_nodes
    assert np.allclose(G.nodes[a]["pos"], (0, 0, 10)) and np.allclose(G.nodes[b]["pos"], (0, 0, 30))
    lengths = {
        frozenset((u, v)): d["length"]
        for u, v, d in G.edges(data=True) if not d.get("post_processing_added")
    }
    assert lengths == {
        frozenset((0, a)): pytest.approx(10.0),
        frozenset((a, b)): pytest.approx(20.0),
        frozenset((b, 1)): pytest.approx(10.0),
    }
    assert G.edges[traced.edge]["length"] == pytest.approx(40.0)
    assert G.number_of_edges() == 4


def test_add_traced_vessel_puts_the_path_ends_on_the_nodes_and_smooths_it():
    G = _two_ends()
    stairs = [(0.0, y, 10.0 + x) for _z, y, x in _staircase(dx=1.0)]
    stairs[0], stairs[-1] = (0.0, 0.3, 10.2), (0.0, 9.8, 19.8)
    G.nodes[5]["pos"] = np.asarray((0.0, 10.0, 20.0))
    # {}: the pipeline's own smoothing defaults (Taubin, 10 passes).
    traced = add_traced_vessel(G, VesselEnd.at_node(1), VesselEnd.at_node(5), stairs, smoothing={})
    data = G.edges[traced.edge]
    assert np.allclose(data["voxels"][0], (0, 0, 10)) and np.allclose(data["voxels"][-1], (0, 10, 20))
    assert traced.smoothing in ("smoothed", "relaxed")
    assert data["centreline_smoothing"] == traced.smoothing
    assert data["length"] == pytest.approx(calculate_path_length(data["voxels"]))
    assert data["length"] < calculate_path_length(stairs)


def test_add_traced_vessel_refuses_a_loop_back_to_its_start_and_changes_nothing():
    G = _one_vessel(_straight((0, 0, 0), (0, 0, 40)))
    before = (sorted(G.nodes), sorted(edge_keys(G)))
    with pytest.raises(ValueError, match="start and end at node 0"):
        add_traced_vessel(G, VesselEnd.at_node(0), VesselEnd.at_node(0), [(0, 0, 0), (0, 5, 0)])
    with pytest.raises(ValueError, match="start and end at node 0"):
        # A point on the vessel right by node 0 is node 0.
        add_traced_vessel(
            G, VesselEnd.at_node(0), VesselEnd.on_vessel((0, 1, 0), (0.0, 0.0, 0.5)),
            [(0, 0, 0), (0, 5, 0), (0, 0, 0.5)],
        )
    with pytest.raises(ValueError, match="same point"):
        add_traced_vessel(
            G,
            VesselEnd.on_vessel((0, 1, 0), (0.0, 0.0, 20.0)),
            VesselEnd.on_vessel((0, 1, 0), (0.0, 0.0, 20.4)),
            [(0, 0, 20), (0, 5, 20), (0, 0, 20.4)],
        )
    assert (sorted(G.nodes), sorted(edge_keys(G))) == before


def test_vessel_path_runs_from_the_u_end():
    G = _one_vessel(_straight((0, 0, 0), (0, 0, 10)))
    assert np.allclose(vessel_path(G, (0, 1, 0))[0], (0, 0, 0))
    assert np.allclose(vessel_path(G, (1, 0, 0))[0], (0, 0, 10))
    with pytest.raises(ValueError, match="No vessel"):
        vessel_path(G, (0, 1, 7))


# --- what each edit marks for the post_process stage ---------------------------

from haemolynx.graph import (  # noqa: E402
    IS_ZERO_RESISTANCE,
    clear_edit_marks,
    edited_edges,
    has_pending_edits,
)
from haemolynx.graph.post_processing import EDITED, PENDING  # noqa: E402


def _edited(G) -> set:
    return {frozenset(edge[:2]) for edge in edited_edges(G)}


def test_a_network_nobody_edited_has_nothing_pending():
    G = _four_way_through()
    assert not has_pending_edits(G)
    assert edited_edges(G) == []
    assert not has_pending_edits(None)


def test_adding_a_vessel_marks_it_and_nothing_else():
    G = _two_ends()
    before = set(G.edges(keys=True))
    edge = add_vessel_between(G, 1, 2)
    assert _edited(G) == {frozenset(edge[:2])}
    assert has_pending_edits(G)
    assert all(EDITED not in G.edges[e] for e in before)


def test_cutting_a_vessel_marks_both_halves():
    G = _one_vessel([(0.0, 0.0, 0.0), (0.0, 0.0, 10.0)])
    node = split_vessel_at(G, (0, 1, 0), (0.0, 0.0, 4.0))
    assert _edited(G) == {frozenset((0, node)), frozenset((node, 1))}


def test_splitting_a_junction_marks_its_connector_and_the_vessels_it_moved():
    G = _four_way_with_one_close_pair()
    untouched = set(G.edges(keys=True))
    new_nodes = split_junction(G, 0)
    (new_node,) = new_nodes
    marked = _edited(G)
    assert frozenset((0, new_node)) in marked, "the connector"
    moved = [e for e in G.edges(new_node, keys=True) if 0 not in e[:2]]
    assert len(moved) == 2 and all(frozenset(e[:2]) in marked for e in moved)
    still_there = [e for e in untouched if G.has_edge(*e)]
    assert all(EDITED not in G.edges[e] for e in still_there)


def test_a_run_splitting_junctions_itself_marks_nothing():
    """The haematocrit junction rule splits after post_process, for the model only."""
    positions, edges = _star(4)
    G = _network(positions, edges)
    split_high_degree_junctions(G)
    assert not has_pending_edits(G)


def test_a_deletion_marks_the_network_and_the_vessel_its_merge_made():
    G = _four_way_through()
    delete_vessels(G, [(0, 3, 0), (0, 4, 0)])
    assert G.has_edge(1, 2)
    assert _edited(G) == {frozenset((1, 2))}
    assert G.graph[PENDING] is True


def test_a_deletion_with_no_merge_still_leaves_the_network_pending():
    G = _four_way_through()
    delete_vessels(G, [(0, 3, 0)])
    assert edited_edges(G) == []
    assert has_pending_edits(G), "branch orders downstream of it can change"


def test_pruning_marks_the_pruned_network_pending():
    G = _main_and_cut_off_pieces()
    pruned, stats = prune_disconnected_branches(G, [0], [2])
    assert stats["removed_vessels"]
    assert has_pending_edits(pruned)


def test_clearing_the_marks_leaves_nothing_pending():
    G = _two_ends()
    add_vessel_between(G, 1, 2)
    clear_edit_marks(G)
    assert not has_pending_edits(G)
    assert all(EDITED not in data for *_e, data in G.edges(data=True))


# --- deletions and thick-vessel bridges ---------------------------------------


def _bridged() -> nx.MultiGraph:
    """Thin vessel 0-1, bridge 1-2 into a thick vessel's centreline 3-2-4.

    Node 1 is where the thin vessel meets the thick vessel's surface; node 2
    is on the thick vessel's own centreline.
    """
    positions = {
        0: (0.0, 0.0, 0.0),
        1: (0.0, 0.0, 10.0),
        2: (0.0, 0.0, 15.0),
        3: (0.0, -20.0, 15.0),
        4: (0.0, 20.0, 15.0),
        5: (0.0, 0.0, -10.0),
        6: (0.0, 10.0, 0.0),
        7: (0.0, -10.0, 0.0),
    }
    G = _network(positions, [(0, 1), (3, 2), (2, 4), (5, 0), (0, 6), (0, 7)])
    voxels = _straight(positions[1], positions[2])
    G.add_edge(1, 2, voxels=voxels, length=calculate_path_length(voxels), **{IS_ZERO_RESISTANCE: True})
    return G


def test_deleting_a_thin_vessel_deletes_the_bridge_it_opened_through():
    G = _bridged()
    delete_vessels(G, [(0, 1, 0)], protected={3, 4})

    assert 1 not in G, "the bridge's outer end went with it"
    assert not any(data.get(IS_ZERO_RESISTANCE) for *_e, data in G.edges(data=True))
    # The thick vessel's centreline is whole again: one vessel through node 2.
    assert 2 not in G and G.has_edge(3, 4)
    assert _edited(G) == {frozenset((3, 4))}


def test_a_bridge_is_never_merged_with_the_vessel_it_opens():
    """Deleting a centreline piece leaves node 2 between the bridge and the
    rest of the thick vessel; merged, one resistance rule would cover both."""
    G = _bridged()
    delete_vessels(G, [(2, 4, 0)], protected={0, 3})

    assert 2 in G
    bridges = [data for *_e, data in G.edges(2, data=True) if data.get(IS_ZERO_RESISTANCE)]
    ordinary = [data for *_e, data in G.edges(2, data=True) if not data.get(IS_ZERO_RESISTANCE)]
    assert len(bridges) == 1 and len(ordinary) == 1
# --- dead-end vessels -------------------------------------------------------


def _with_dead_ends() -> nx.MultiGraph:
    """Inlet 0 -> 1 -> {2, 3} -> outlet 4 (a loop blood crosses), plus dead ends.

    1 -> 5 -> 6 is a dead-end branch; 5 -> 7 -> 8 -> 5 a loop hanging off it
    at one node; 4 has a self-loop; 9 - 10 is a piece with neither boundary.
    """
    positions = {
        0: (0, 0, 0), 1: (10, 0, 0), 2: (20, 10, 0), 3: (20, -10, 0), 4: (30, 0, 0),
        5: (10, 20, 0), 6: (10, 30, 0), 7: (0, 30, 0), 8: (0, 20, 0),
        9: (50, 0, 0), 10: (60, 0, 0),
    }
    G = _network(
        positions,
        [(0, 1), (1, 2), (1, 3), (2, 4), (3, 4), (1, 5), (5, 6), (5, 7), (7, 8), (8, 5), (9, 10)],
        diameter_um=5.0,
    )
    G.add_edge(4, 4, voxels=[(30.0, 0.0, 0.0), (31.0, 1.0, 0.0), (30.0, 0.0, 0.0)], length=3.0)
    return G


def _pairs(edges) -> set:
    return {frozenset(e[:2]) for e in edges}


def test_dead_end_vessels_lists_everything_off_the_inlet_to_outlet_paths():
    G = _with_dead_ends()
    dead = dead_end_vessels(G, [0], [4])
    assert _pairs(dead) == {
        frozenset(p) for p in [(1, 5), (5, 6), (5, 7), (7, 8), (8, 5), (9, 10), (4, 4)]
    }
    # In branchID order, as the tab's table lists them.
    index = {key: i for i, key in enumerate(edge_keys(G))}
    assert [index[e] for e in dead] == sorted(index[e] for e in dead)


def test_a_loop_blood_crosses_is_not_a_dead_end_but_one_off_a_single_node_is():
    # 1 -> 2 -> 4 and 1 -> 3 -> 4 both carry flow; 5 -> 7 -> 8 -> 5 only
    # touches the rest at 5, so every node on it sits at 5's pressure.
    dead = _pairs(dead_end_vessels(_with_dead_ends(), [0], [4]))
    for through in [(0, 1), (1, 2), (1, 3), (2, 4), (3, 4)]:
        assert frozenset(through) not in dead
    assert frozenset((7, 8)) in dead


def test_parallel_vessels_between_inlet_and_outlet_both_carry_flow():
    G = _network({0: (0, 0, 0), 1: (10, 0, 0), 2: (20, 0, 0)}, [(0, 1), (1, 2), (1, 2)])
    assert dead_end_vessels(G, [0], [2]) == []


def test_several_inlets_and_outlets_each_keep_their_own_paths():
    # 0 -> 1 -> 2 and 3 -> 1 -> 4: every vessel is on some inlet-to-outlet path.
    positions = {0: (0, 0, 0), 1: (10, 0, 0), 2: (20, 0, 0), 3: (10, 10, 0), 4: (10, -10, 0)}
    G = _network(positions, [(0, 1), (1, 2), (3, 1), (1, 4)])
    assert dead_end_vessels(G, [0, 3], [2, 4]) == []
    # With 3 no longer an inlet, its vessel leads nowhere.
    assert _pairs(dead_end_vessels(G, [0], [2, 4])) == {frozenset((1, 3))}


def test_dead_end_vessels_needs_an_inlet_and_an_outlet_in_the_network():
    G = _with_dead_ends()
    with pytest.raises(ValueError, match="at least one inlet and one outlet"):
        dead_end_vessels(G, [0], [])
    with pytest.raises(ValueError, match="at least one inlet and one outlet"):
        dead_end_vessels(G, [0], [99])


def test_delete_all_dead_ends_leaves_only_the_network_between_inlet_and_outlet():
    G = _with_dead_ends()
    lost = delete_dead_end_vessels(G, [0], [4])
    assert lost == []
    assert _pairs(edge_keys(G)) == {frozenset(p) for p in [(0, 1), (1, 2), (1, 3), (2, 4), (3, 4)]}
    assert set(G.nodes) == {0, 1, 2, 3, 4}
    assert dead_end_vessels(G, [0], [4]) == []
    # Outlet 4, a pass-through once its self-loop went, is kept, not merged.
    assert G.degree(4) == 2


def test_deleting_one_dead_end_tidies_its_nodes_like_any_delete():
    G = _with_dead_ends()
    edge = next(e for e in edge_keys(G) if set(e[:2]) == {5, 6})
    delete_dead_end_vessels(G, [0], [4], [edge])
    assert 6 not in G  # left with no vessel
    assert G.degree(5) == 3  # the branch 1 -> 5 and the loop remain, still dead ends
    assert frozenset((1, 5)) in _pairs(dead_end_vessels(G, [0], [4]))


def test_delete_dead_end_vessels_refuses_a_vessel_blood_crosses():
    G = _with_dead_ends()
    edge = next(e for e in edge_keys(G) if set(e[:2]) == {1, 2})
    before = G.number_of_edges()
    with pytest.raises(ValueError, match="not a dead end"):
        delete_dead_end_vessels(G, [0], [4], [edge])
    assert G.number_of_edges() == before


def test_a_boundary_node_on_a_dead_end_goes_with_it_but_one_still_used_stays():
    G = _with_dead_ends()
    # 6 is an arteriole boundary node at the tip of the dead-end branch.
    lost = delete_dead_end_vessels(G, [0], [4], protected={0, 4, 6})
    assert lost == [6]
    assert 6 not in G and 0 in G and 4 in G


def test_a_protected_node_keeping_a_vessel_is_not_merged():
    # 0 -> 1 -> 2 with a dead end 1 -> 3: deleting it leaves 1 a pass-through.
    positions = {0: (0, 0, 0), 1: (10, 0, 0), 2: (20, 0, 0), 3: (10, 10, 0)}
    G = _network(positions, [(0, 1), (1, 2), (1, 3)])
    delete_dead_end_vessels(G, [0], [2], protected={0, 1, 2})
    assert 1 in G and G.degree(1) == 2
    G = _network(positions, [(0, 1), (1, 2), (1, 3)])
    delete_dead_end_vessels(G, [0], [2], protected={0, 2})
    assert 1 not in G and G.has_edge(0, 2)  # merged into one vessel


def test_delete_dead_end_vessels_refuses_to_remove_every_vessel():
    G = _network({0: (0, 0, 0), 1: (10, 0, 0), 2: (20, 0, 0)}, [(0, 1)])
    G.add_node(2, pos=np.zeros(3))
    G.add_edge(2, 2, voxels=[(20.0, 0.0, 0.0), (20.0, 0.0, 0.0)], length=0.0)
    with pytest.raises(ValueError, match="remove every vessel"):
        delete_dead_end_vessels(G, [0], [2])
    assert G.number_of_edges() == 2
