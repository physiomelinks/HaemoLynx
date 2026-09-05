"""Splitting a graph edge where it crosses a fat/thick-vessel mask boundary."""
from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from haemolynx.graph.thick_vessel_junctions import (
    IS_ZERO_RESISTANCE,
    insert_thick_vessel_junction_nodes,
)


def _straight_edge_graph(x0: int, x1: int, **edge_attrs):
    G = nx.MultiGraph()
    G.add_node("A", pos=(0.0, 0.0, float(x0)))
    G.add_node("B", pos=(0.0, 0.0, float(x1)))
    voxels = [(0.0, 0.0, float(x)) for x in range(x0, x1 + 1)]
    G.add_edge("A", "B", key=0, voxels=voxels, length=float(x1 - x0), **edge_attrs)
    return G


def _mask(shape_x: int, *thick_ranges: range) -> np.ndarray:
    mask = np.zeros((1, 1, shape_x), dtype=bool)
    for r in thick_ranges:
        mask[0, 0, r.start : r.stop] = True
    return mask


def test_fully_exterior_edge_is_untouched():
    G = _straight_edge_graph(0, 5)
    mask = _mask(30, range(20, 25))

    result = insert_thick_vessel_junction_nodes(G, mask, voxel_size_zyx=(1.0, 1.0, 1.0))

    assert result.number_of_nodes() == 2
    assert result.number_of_edges() == 1
    (_u, _v, data) = next(iter(result.edges(data=True)))
    assert IS_ZERO_RESISTANCE not in data


def test_fully_interior_edge_is_untouched_and_not_tagged():
    """The fat vessel's own centreline must keep its real resistance."""
    G = _straight_edge_graph(10, 15)
    mask = _mask(30, range(0, 30))

    result = insert_thick_vessel_junction_nodes(G, mask, voxel_size_zyx=(1.0, 1.0, 1.0))

    assert result.number_of_nodes() == 2
    assert result.number_of_edges() == 1
    (_u, _v, data) = next(iter(result.edges(data=True)))
    assert IS_ZERO_RESISTANCE not in data


def test_a_single_crossing_splits_the_edge_and_tags_only_the_interior_side():
    G = _straight_edge_graph(0, 20, branch_order="BO1")
    mask = _mask(25, range(10, 25))

    result = insert_thick_vessel_junction_nodes(G, mask, voxel_size_zyx=(1.0, 1.0, 1.0))

    assert result.number_of_nodes() == 3
    assert result.number_of_edges() == 2
    by_tag = {
        bool(data.get(IS_ZERO_RESISTANCE)): data for _u, _v, data in result.edges(data=True)
    }
    assert set(by_tag) == {False, True}
    # Both segments keep the original edge's other attributes.
    assert by_tag[False]["branch_order"] == "BO1"
    assert by_tag[True]["branch_order"] == "BO1"
    # Lengths must add back up to the original edge length -- no voxels lost
    # or double-counted at the new boundary node.
    assert by_tag[False]["length"] + by_tag[True]["length"] == pytest.approx(20.0)
    # The new node sits at the last exterior sample, one voxel before the
    # mask boundary at x=10 -- still outside `mask`, same convention
    # cut_graph_at_large_vessel_volumes uses for its own cut nodes.
    new_node = [n for n in result.nodes if n not in ("A", "B")][0]
    assert result.nodes[new_node]["pos"] == pytest.approx((0.0, 0.0, 9.0))


def test_multiple_crossings_tag_only_each_interior_run():
    G = _straight_edge_graph(0, 25)
    mask = _mask(30, range(5, 10), range(15, 20))

    result = insert_thick_vessel_junction_nodes(G, mask, voxel_size_zyx=(1.0, 1.0, 1.0))

    assert result.number_of_nodes() == 6
    assert result.number_of_edges() == 5
    tagged = [data for _u, _v, data in result.edges(data=True) if data.get(IS_ZERO_RESISTANCE)]
    untagged = [
        data for _u, _v, data in result.edges(data=True) if not data.get(IS_ZERO_RESISTANCE)
    ]
    assert len(tagged) == 2
    assert len(untagged) == 3
    # No gap or overlap: every segment's length adds back to the original span.
    total_length = sum(data["length"] for data in tagged + untagged)
    assert total_length == pytest.approx(25.0)


def test_consecutive_segments_share_their_boundary_point():
    """Each split edge's voxels must actually start at its own node's position.

    A segment built from the wrong slice of the sampled polyline can end up
    one sample short at its start -- geometrically discontinuous with the
    node it is supposed to originate from.
    """
    G = _straight_edge_graph(0, 25)
    mask = _mask(30, range(5, 10), range(15, 20))

    result = insert_thick_vessel_junction_nodes(G, mask, voxel_size_zyx=(1.0, 1.0, 1.0))

    for node in result.nodes:
        if node in ("A", "B"):
            continue
        node_pos = tuple(result.nodes[node]["pos"])
        # G.edges(node) always reports `node` as the first element regardless
        # of which end it was originally added as, so check either end of
        # the stored polyline rather than assuming one from that ordering.
        for _u, _v, _key, data in result.edges(node, keys=True, data=True):
            first = tuple(float(c) for c in data["voxels"][0])
            last = tuple(float(c) for c in data["voxels"][-1])
            assert first == pytest.approx(node_pos) or last == pytest.approx(
                node_pos
            ), f"neither end of {data['voxels']} touches node {node} at {node_pos}"


def test_a_single_voxel_flicker_at_the_boundary_does_not_create_a_degenerate_edge():
    """Voxel-grid discretisation noise must not split the edge into a
    near-zero-length stub of its own -- only the real crossing at x=12
    should produce a split."""
    G = _straight_edge_graph(0, 20)
    mask = np.zeros((1, 1, 25), dtype=bool)
    mask[0, 0, 8] = True  # an isolated single True voxel, surrounded by False
    mask[0, 0, 12:] = True  # the real thick region

    result = insert_thick_vessel_junction_nodes(G, mask, voxel_size_zyx=(1.0, 1.0, 1.0))

    # The flicker at x=8 is absorbed into a neighbouring run, not its own edge.
    assert result.number_of_edges() == 2
