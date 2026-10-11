"""Bridges where a vessel opens into a thick vessel.

A thick vessel 6 um in radius runs along y. Its thick-vessel region is the
segmentation within 3 um of it, as skeletonisation finds it: the thick vessel
and the first 3 um of each vessel fused to it. A bridge is the straight piece
from where a joining vessel leaves that region -- the thick vessel's
segmented boundary -- to the thick vessel's centreline. Inside the region
there is only the centreline and bridges; everything else keeps its
resistance.
"""
from __future__ import annotations

import networkx as nx
import numpy as np
import pytest
from scipy.ndimage import distance_transform_edt

from haemolynx.graph.thick_vessel_junctions import (
    IS_ZERO_RESISTANCE,
    insert_thick_vessel_junction_nodes,
)

VS = (1.0, 1.0, 1.0)
SHAPE = (40, 64, 64)
RADIUS = 6.0
MARGIN = 3.0
#: A bridge reaches from the centreline to the region's boundary.
REACH = RADIUS + MARGIN
T0, T1 = (20.0, 2.0, 20.0), (20.0, 61.0, 20.0)


def _tube(a, b, radius, shape=SHAPE):
    """Voxels within *radius* of the segment *a*-*b*."""
    p = np.stack(np.indices(shape), axis=-1).astype(float)
    a, b = np.asarray(a, float), np.asarray(b, float)
    ab = b - a
    t = np.clip(((p - a) @ ab) / (ab @ ab), 0.0, 1.0)
    return np.linalg.norm(p - (a + t[..., None] * ab), axis=-1) <= radius


def _line(*corners, step=1.0):
    points = [np.asarray(corners[0], float)]
    for a, b in zip(corners[:-1], corners[1:]):
        a, b = np.asarray(a, float), np.asarray(b, float)
        count = max(1, int(np.ceil(np.linalg.norm(b - a) / step)))
        points += [a + (b - a) * (k / count) for k in range(1, count + 1)]
    return [tuple(float(c) for c in p) for p in points]


def _graph(*edges, **edge_attrs):
    """``(u, v, corners...)`` per edge, its path straight between the corners."""
    G = nx.MultiGraph()
    for u, v, *corners in edges:
        path = _line(*corners)
        G.add_node(u, pos=path[0])
        G.add_node(v, pos=path[-1])
        length = float(np.sum(np.linalg.norm(np.diff(np.asarray(path), axis=0), axis=1)))
        G.add_edge(u, v, voxels=path, length=length, **edge_attrs)
    return G


def _thick(a=(20.0, 0.0, 20.0), b=(20.0, 63.0, 20.0)):
    """The thick vessel's lumen (vessels fused to it are added to it), its
    body and its centreline tree."""
    body = _tube(a, b, RADIUS)
    return body.copy(), body, _tube(a, b, 0.5)


def _region(lumen, body):
    """The segmentation within the margin of the thick vessel's body."""
    return lumen & (distance_transform_edt(~body) <= MARGIN)


def _split(G, lumen, body, tree, **kwargs):
    return insert_thick_vessel_junction_nodes(
        G, _region(lumen, body), voxel_size_zyx=VS, segmentation=lumen, centreline=tree, **kwargs
    )


def _bridges(G):
    return [(u, v, d) for u, v, d in G.edges(data=True) if d.get(IS_ZERO_RESISTANCE)]


def _thin_vessel(lumen, a, b):
    lumen |= _tube(a, b, 1.5)


JOINING = (("T0", "J", T0, (20, 30, 20)), ("J", "T1", (20, 30, 20), T1),
           ("X", "J", (20, 30, 63), (20, 30, 20)))


def test_a_vessel_ending_on_the_centreline_gets_one_bridge_to_the_boundary():
    """The vessel keeps its length up to the thick vessel's segmented
    boundary; the bridge is the rest, straight to the centreline; the thick
    vessel keeps its own centreline, untouched."""
    lumen, body, tree = _thick()
    _thin_vessel(lumen, (20, 30, 20), (20, 30, 63))
    G = _graph(*JOINING)

    result = _split(G, lumen, body, tree)

    (u, v, bridge), = _bridges(result)
    assert "J" in (u, v), "the bridge ends on the thick centreline's junction"
    assert REACH - 0.5 <= bridge["length"] <= REACH + 1.0
    thin = [d for a, b, d in result.edges(data=True) if "X" in (a, b)]
    assert len(thin) == 1 and not thin[0].get(IS_ZERO_RESISTANCE)
    assert thin[0]["length"] == pytest.approx(43.0 - bridge["length"], abs=0.6)
    for a, b in (("T0", "J"), ("J", "T1")):
        assert not result.edges[a, b, 0].get(IS_ZERO_RESISTANCE)
    assert result.edges["T0", "J", 0]["voxels"] == G.edges["T0", "J", 0]["voxels"]


def test_a_vessel_skimming_the_regions_edge_is_not_a_bridge():
    """Fused to the thick vessel's wall, never entering its lumen: a vessel
    passing by, all of it kept with its resistance."""
    lumen, body, tree = _thick()
    _thin_vessel(lumen, (27.5, 30, 0), (27.5, 30, 40))
    G = _graph(("T0", "T1", T0, T1), ("G1", "G2", (27.5, 30, 0), (27.5, 30, 40)))

    result = _split(G, lumen, body, tree)

    assert not _bridges(result)
    assert result.number_of_edges() == 2
    assert result.edges["G1", "G2", 0]["voxels"] == G.edges["G1", "G2", 0]["voxels"]


def test_a_centreline_leaving_the_region_and_coming_back_stays_one_vessel():
    """The thick vessel's own centreline is never a bridge, wherever the
    region round it has a gap."""
    lumen, body, tree = _thick()
    body[:, 28:33, :] = False
    tree[:, 28:33, :] = False
    G = _graph(("T0", "T1", T0, T1))

    result = _split(G, lumen, body, tree)

    assert not _bridges(result)
    assert result.number_of_edges() == 1
    assert result.edges["T0", "T1", 0]["voxels"] == G.edges["T0", "T1", 0]["voxels"]


def test_a_vessel_merged_with_the_centreline_it_joined_is_cut_at_the_boundary():
    """One edge running in from outside and on along the thick vessel's
    centreline (the join met the tree's tip): the vessel up to the boundary,
    one bridge, and the centreline as vessel."""
    lumen, body, tree = _thick(a=(20.0, 30.0, 20.0))
    _thin_vessel(lumen, (20, 30, 20), (20, 30, 63))
    G = _graph(("X", "T1", (20, 30, 63), (20, 30, 20), T1))

    result = _split(G, lumen, body, tree)

    (u, v, bridge), = _bridges(result)
    assert bridge["length"] <= REACH + 1.0
    assert result.degree("X") == 1 and result.degree("T1") == 1
    along = [d for a, b, d in result.edges(data=True) if "T1" in (a, b)]
    assert len(along) == 1 and not along[0].get(IS_ZERO_RESISTANCE)
    assert along[0]["length"] > 25.0


def test_a_vessel_joined_onto_another_vessels_join_gets_a_bridge_of_its_own():
    """B was joined onto A's path inside the lumen, not to the centreline:
    each gets its own bridge to the centreline, and the node inside the
    lumen goes."""
    lumen, body, tree = _thick()
    _thin_vessel(lumen, (20, 30, 20), (20, 30, 63))
    _thin_vessel(lumen, (20, 30, 24), (20, 42, 63))
    G = _graph(("T0", "J", T0, (20, 30, 20)), ("J", "T1", (20, 30, 20), T1),
               ("X", "N", (20, 30, 63), (20, 30, 24)), ("N", "J", (20, 30, 24), (20, 30, 20)),
               ("Y", "N", (20, 42, 63), (20, 30, 24)))

    result = _split(G, lumen, body, tree)

    bridges = _bridges(result)
    assert len(bridges) == 2
    assert "N" not in result
    for u, v, d in bridges:
        assert d["length"] <= REACH + 3.0
        inner = min((u, v), key=lambda n: abs(result.nodes[n]["pos"][2] - 20.0))
        assert result.nodes[inner]["pos"][2] == pytest.approx(20.0, abs=0.6)
    for x in ("X", "Y"):
        assert result.degree(x) == 1


def test_a_vessel_crossing_the_thick_one_gets_two_bridges_to_one_node():
    lumen, body, tree = _thick()
    _thin_vessel(lumen, (20, 30, 0), (20, 30, 63))
    G = _graph(("T0", "T1", T0, T1), ("X1", "X2", (20, 30, 63), (20, 30, 0)))

    result = _split(G, lumen, body, tree)

    bridges = _bridges(result)
    assert len(bridges) == 2
    (middle,) = set.intersection(*({u, v} for u, v, _d in bridges))
    assert result.nodes[middle]["pos"] == pytest.approx((20.0, 30.0, 20.0), abs=0.6)
    assert result.degree(middle) == 4
    assert all(REACH - 0.5 <= d["length"] <= REACH + 1.0 for *_e, d in bridges)


def test_a_vessel_entering_the_lumen_but_never_reaching_the_centreline_is_bridged():
    """Inside the thick vessel, opening into it, though its centreline
    stops short of the tree: inside the region there is only the centreline
    and bridges."""
    lumen, body, tree = _thick()
    _thin_vessel(lumen, (20, 30, 23), (20, 30, 63))
    G = _graph(("T0", "T1", T0, T1), ("X", "S", (20, 30, 63), (20, 30, 23)))

    result = _split(G, lumen, body, tree)

    (u, v, bridge), = _bridges(result)
    assert REACH - 0.5 <= bridge["length"] <= REACH + 1.0
    assert "S" not in result


def test_the_centreline_tree_is_found_from_the_region_when_not_given():
    lumen, body, _tree = _thick()
    _thin_vessel(lumen, (20, 30, 20), (20, 30, 63))
    G = _graph(*JOINING)

    result = insert_thick_vessel_junction_nodes(
        G, _region(lumen, body), voxel_size_zyx=VS, segmentation=lumen
    )

    (_u, _v, bridge), = _bridges(result)
    assert REACH - 1.0 <= bridge["length"] <= REACH + 1.0


def test_without_a_segmentation_the_region_stands_in_for_the_lumen():
    lumen, body, tree = _thick()
    _thin_vessel(lumen, (20, 30, 20), (20, 30, 63))
    G = _graph(*JOINING)

    result = insert_thick_vessel_junction_nodes(
        G, _region(lumen, body), voxel_size_zyx=VS, centreline=tree
    )

    (_u, _v, bridge), = _bridges(result)
    assert REACH - 0.5 <= bridge["length"] <= REACH + 1.0


def test_every_piece_starts_and_ends_on_its_own_nodes():
    lumen, body, tree = _thick()
    _thin_vessel(lumen, (20, 30, 20), (20, 30, 63))
    _thin_vessel(lumen, (20, 30, 24), (20, 42, 63))
    G = _graph(("T0", "J", T0, (20, 30, 20)), ("J", "T1", (20, 30, 20), T1),
               ("X", "N", (20, 30, 63), (20, 30, 24)), ("N", "J", (20, 30, 24), (20, 30, 20)),
               ("Y", "N", (20, 42, 63), (20, 30, 24)))

    result = _split(G, lumen, body, tree)

    for u, v, d in result.edges(data=True):
        ends = {tuple(np.round(d["voxels"][0], 6)), tuple(np.round(d["voxels"][-1], 6))}
        assert ends == {tuple(np.round(result.nodes[n]["pos"], 6)) for n in (u, v)}
        assert d["length"] == pytest.approx(
            float(np.sum(np.linalg.norm(np.diff(np.asarray(d["voxels"]), axis=0), axis=1)))
        )


def test_a_vessel_with_no_thick_centreline_in_the_graph_to_join_is_left_whole():
    """The thick vessel's tree is there but the graph has no centreline
    along it: nothing to bridge to, so the vessel is not cut off."""
    lumen, body, tree = _thick()
    _thin_vessel(lumen, (20, 30, 20), (20, 30, 63))
    G = _graph(("X", "J", (20, 30, 63), (20, 30, 20)))

    result = _split(G, lumen, body, tree)

    assert not _bridges(result)
    assert result.number_of_edges() == 1
    assert result.edges["X", "J", 0]["voxels"] == G.edges["X", "J", 0]["voxels"]


def test_no_thick_region_leaves_the_graph_as_it_is():
    G = _graph(("X", "Y", (20, 30, 63), (20, 30, 0)))

    result = insert_thick_vessel_junction_nodes(
        G, np.zeros(SHAPE, dtype=bool), voxel_size_zyx=VS
    )

    assert nx.utils.edges_equal(result.edges(data=False), G.edges(data=False))
    assert not _bridges(result)


# --- only the edges asked about (a hand-edited vessel's bridge) ------------------


def _two_joining_vessels():
    lumen, body, tree = _thick()
    _thin_vessel(lumen, (20, 30, 20), (20, 30, 63))
    _thin_vessel(lumen, (20, 45, 20), (20, 45, 63))
    G = _graph(("T0", "J", T0, (20, 30, 20)), ("J", "K", (20, 30, 20), (20, 45, 20)),
               ("K", "T1", (20, 45, 20), T1),
               ("X", "J", (20, 30, 63), (20, 30, 20)), ("Y", "K", (20, 45, 63), (20, 45, 20)))
    return G, lumen, body, tree


def test_edges_limits_the_rebuild_to_the_edges_named():
    G, lumen, body, tree = _two_joining_vessels()

    result = _split(G, lumen, body, tree, edges=[("J", "X", 0)])

    (u, v, _bridge), = _bridges(result)
    assert "J" in (u, v)
    assert result.edges["Y", "K", 0]["voxels"] == G.edges["Y", "K", 0]["voxels"]


def test_a_named_edge_already_a_bridge_is_left_as_it_is():
    G, lumen, body, tree = _two_joining_vessels()
    G.edges["X", "J", 0][IS_ZERO_RESISTANCE] = True

    result = _split(G, lumen, body, tree, edges=[("X", "J", 0)])

    assert result.edges["X", "J", 0][IS_ZERO_RESISTANCE] is True
    assert result.edges["X", "J", 0]["voxels"] == G.edges["X", "J", 0]["voxels"]
    assert len(_bridges(result)) == 1


def test_the_vessel_and_its_bridge_keep_every_other_attribute_of_the_edge_they_came_from():
    lumen, body, tree = _thick()
    _thin_vessel(lumen, (20, 30, 20), (20, 30, 63))
    G = _graph(("T0", "J", T0, (20, 30, 20)), ("J", "T1", (20, 30, 20), T1))
    G.add_node("X", pos=(20.0, 30.0, 63.0))
    G.add_edge("X", "J", voxels=_line((20, 30, 63), (20, 30, 20)), length=43.0,
               post_processing_edited=True, post_processing_added=True)

    result = _split(G, lumen, body, tree, edges=[("X", "J", 0)])

    pieces = [d for a, b, d in result.edges(data=True) if d.get("post_processing_edited")]
    assert len(pieces) == 2
    assert {bool(d.get(IS_ZERO_RESISTANCE)) for d in pieces} == {True, False}
    assert all(d["post_processing_added"] for d in pieces)


def test_a_bridge_is_never_attached_to_a_vessel_ending_blind():
    """A stub off the thick vessel, ending inside the image, goes with its
    bridge; a vessel running out of the image keeps its bridge."""
    lumen, body, tree = _thick()
    _thin_vessel(lumen, (20, 30, 20), (20, 30, 40))
    _thin_vessel(lumen, (20, 45, 20), (20, 45, 63))
    G = _graph(("T0", "J", T0, (20, 30, 20)), ("J", "K", (20, 30, 20), (20, 45, 20)),
               ("K", "T1", (20, 45, 20), T1),
               ("STUB", "J", (20, 30, 40), (20, 30, 20)), ("OUT", "K", (20, 45, 63), (20, 45, 20)))

    result = _split(G, lumen, body, tree)

    (u, v, _bridge), = _bridges(result)
    assert "K" in (u, v)
    assert "STUB" not in result
    assert "J" not in result, "the junction only the stub made is merged away"
    assert result.degree("OUT") == 1


# --- the network with its bridges contracted (what statistics measure) --------------


def test_contracting_bridges_joins_each_vessel_to_the_thick_centreline():
    """The bridge's two ends become its centreline end, and the graph shares
    the original's attribute dicts, so what is written on it lands there."""
    from haemolynx.graph.thick_vessel_junctions import contract_bridges

    G = _graph(("T0", "J", T0, (20, 30, 20)), ("J", "T1", (20, 30, 20), T1),
               ("X", "W", (20, 30, 63), (20, 30, 29)))
    G.add_edge("W", "J", voxels=_line((20, 30, 29), (20, 30, 20)), length=9.0,
               **{IS_ZERO_RESISTANCE: True})

    H, became = contract_bridges(G)

    assert set(H.nodes) == {"T0", "J", "T1", "X"}
    assert became["W"] == "J"
    assert {frozenset(e) for e in H.edges()} == {
        frozenset(("T0", "J")), frozenset(("J", "T1")), frozenset(("J", "X"))
    }
    assert not any(d.get(IS_ZERO_RESISTANCE) for *_e, d in H.edges(data=True))
    (key,) = H["J"]["X"]
    H.edges["J", "X", key]["written"] = 1
    assert G.edges["X", "W", 0]["written"] == 1
    assert H.nodes["J"] is G.nodes["J"]
