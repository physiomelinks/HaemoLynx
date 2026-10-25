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

from haemolynx.graph.thick_vessel_bridge_shapes import (
    BRIDGE_SHAPES,
    diagnose_thick_vessel_bridges,
    format_thick_vessel_bridges_report,
)
from haemolynx.graph.thick_vessel_junctions import (
    IS_ZERO_RESISTANCE,
    THICK_FORK_BRIDGE,
    THICK_TAPER,
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


def test_an_oblique_vessel_is_bridged_along_its_own_line_to_where_it_met_the_centreline():
    """Entering at 49 degrees to the thick vessel's axis: the bridge runs from
    where it leaves the region on along its own line to where it met the
    centreline, longer than the radius by the slant; the rest is vessel."""
    lumen, body, tree = _thick()
    _thin_vessel(lumen, (20, 30, 20), (20, 60, 55))
    G = _graph(("T0", "J", T0, (20, 30, 20)), ("J", "T1", (20, 30, 20), T1),
               ("X", "J", (20, 60, 55), (20, 30, 20)))

    result = _split(G, lumen, body, tree)

    (u, v, bridge), = _bridges(result)
    assert "J" in (u, v)
    slant = REACH / np.sin(np.arctan2(35.0, 30.0))
    assert slant - 1.0 <= bridge["length"] <= slant + 1.0
    along = np.subtract(*map(np.asarray, (bridge["voxels"][0], bridge["voxels"][-1])))
    assert abs(along @ np.array([0.0, 30.0, 35.0])) / (np.linalg.norm(along) * np.hypot(30, 35)) > 0.99
    (thin,) = [d for a, b, d in result.edges(data=True) if "X" in (a, b)]
    assert thin["length"] == pytest.approx(np.hypot(30.0, 35.0) - bridge["length"], abs=0.6)


def _side_by_side(gap: float, *, reach_the_centreline: bool = True):
    """Two thin vessels entering the thick one at y = 30 and 30 + *gap*,
    their centrelines meeting it at junctions J and K, or stopping 3 um
    short of it."""
    lumen, body, tree = _thick()
    y2 = 30.0 + gap
    _thin_vessel(lumen, (20, 30, 20), (20, 30, 63))
    _thin_vessel(lumen, (20, y2, 20), (20, y2, 63))
    if reach_the_centreline:
        G = _graph(("T0", "J", T0, (20, 30, 20)), ("J", "K", (20, 30, 20), (20, y2, 20)),
                   ("K", "T1", (20, y2, 20), T1),
                   ("X", "J", (20, 30, 63), (20, 30, 20)), ("Y", "K", (20, y2, 63), (20, y2, 20)))
    else:
        G = _graph(("T0", "T1", T0, T1),
                   ("X", "S", (20, 30, 63), (20, 30, 23)), ("Y", "R", (20, y2, 63), (20, y2, 23)))
    return G, lumen, body, tree


@pytest.mark.parametrize("reach_the_centreline", [True, False])
def test_two_vessels_entering_side_by_side_share_one_centreline_node(reach_the_centreline):
    """Design C.4: within one radius of each other (here 4 um, the thick
    vessel's radius 6 um), two vessels meet the thick centreline at one
    junction -- the one already there, or else where one of them would --
    not at two with a 4 um stub of thick vessel between."""
    G, lumen, body, tree = _side_by_side(4.0, reach_the_centreline=reach_the_centreline)

    result = _split(G, lumen, body, tree)

    bridges = _bridges(result)
    assert len(bridges) == 2
    (shared,) = set.intersection(*({u, v} for u, v, _d in bridges))
    assert result.nodes[shared]["pos"][2] == pytest.approx(20.0, abs=0.6)
    assert result.nodes[shared]["pos"][1] in (pytest.approx(30.0, abs=0.6), pytest.approx(34.0, abs=0.6))
    assert result.degree(shared) == 4
    thick = [d for a, b, d in result.edges(data=True) if not d.get(IS_ZERO_RESISTANCE) and not {a, b} & {"X", "Y"}]
    assert len(thick) == 2, "the thick centreline: one edge each side of the shared node"
    assert sum(d["length"] for d in thick) == pytest.approx(59.0, abs=0.6)
    for u, v, d in bridges:
        assert REACH - 0.5 <= d["length"] <= np.hypot(REACH, 4.0) + 1.0


def test_two_vessels_further_apart_than_the_radius_keep_a_node_each():
    G, lumen, body, tree = _side_by_side(10.0)

    result = _split(G, lumen, body, tree)

    bridges = _bridges(result)
    assert len(bridges) == 2
    assert not set.intersection(*({u, v} for u, v, _d in bridges))
    assert {"J", "K"} <= set(result.nodes)


def _curved_thick_vessel():
    """A thick vessel 5 um in radius bent into a C round a centre 9 um from
    its centreline (0 to 270 degrees), and a thin vessel coming up the C's
    hole that enters its inner wall at 0 degrees and runs round inside the
    lumen, along the wall, to meet the centreline at 150 degrees."""
    shape, centre, bend, radius = (17, 64, 64), np.array([8.0, 32.0, 32.0]), 9.0, 5.0
    p = np.stack(np.indices(shape), axis=-1).astype(float)
    rho = np.hypot(p[..., 1] - centre[1], p[..., 2] - centre[2])
    angle = np.degrees(np.arctan2(p[..., 1] - centre[1], p[..., 2] - centre[2])) % 360
    body = ((rho - bend) ** 2 + (p[..., 0] - centre[0]) ** 2 <= radius**2) & (angle <= 270)

    def at(r, degrees, z=8.0):
        t = np.radians(degrees)
        return (z, centre[1] + r * np.sin(t), centre[2] + r * np.cos(t))

    def arc(r, start, stop, step=1.0):
        count = max(2, int(np.ceil(abs(np.radians(stop - start)) * r / step)) + 1)
        return [at(r, a) for a in np.linspace(start, stop, count)]

    tree = np.zeros(shape, dtype=bool)
    for point in arc(bend, 0, 270, step=0.3):
        tree[tuple(np.rint(point).astype(int))] = True
    lumen = body | _tube(at(2.5, 0, 0.0), at(2.5, 0, 8.0), 1.2, shape) | _tube(at(2.5, 0, 8.0), at(4.5, 0), 1.2, shape)
    G = nx.MultiGraph()
    for u, v, points in (("T0", "J", arc(bend, 0, 150)), ("J", "T1", arc(bend, 150, 270)),
                         ("X", "J", [at(2.5, 0, 0.0), at(2.5, 0, 8.0), *arc(5.0, 0, 150), at(bend, 150)])):
        points = [tuple(float(c) for c in q) for q in points]
        G.add_node(u, pos=points[0])
        G.add_node(v, pos=points[-1])
        G.add_edge(u, v, voxels=points, length=float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1))))
    return G, lumen, body, tree


def test_a_bridge_round_a_curved_thick_vessel_keeps_the_way_its_vessel_ran():
    """Design C.5: straight, the bridge would cut across the C's hole, 7 um
    of background; it keeps the way the vessel ran through the lumen
    instead, and is marked so."""
    from haemolynx.graph.thick_vessel_junctions import DRAWN_BRIDGE, _longest_outside_um

    G, lumen, body, tree = _curved_thick_vessel()

    result = insert_thick_vessel_junction_nodes(G, body, voxel_size_zyx=VS, segmentation=lumen, centreline=tree)

    (u, v, bridge), = _bridges(result)
    assert "J" in (u, v)
    assert bridge[DRAWN_BRIDGE] is True
    path = np.asarray(bridge["voxels"])
    assert _longest_outside_um(path, lumen, np.ones(3)) == 0.0
    straight = np.vstack([path[0], path[-1]])
    assert _longest_outside_um(straight, lumen, np.ones(3)) > 5.0
    assert bridge["length"] == pytest.approx(float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1))))
    assert bridge["length"] > float(np.linalg.norm(path[-1] - path[0])) + 5.0
    assert {tuple(path[0]), tuple(path[-1])} == {tuple(result.nodes[n]["pos"]) for n in (u, v)}
    (thin,) = [d for a, b, d in result.edges(data=True) if "X" in (a, b)]
    assert not thin.get(IS_ZERO_RESISTANCE) and thin["length"] < 10.0
    report = diagnose_thick_vessel_bridges(result, body, tree, lumen, voxel_size_zyx=VS)
    assert report["drawn_path_count"] == 1


def test_a_straight_bridge_inside_the_lumen_is_not_drawn_along_its_path():
    from haemolynx.graph.thick_vessel_junctions import DRAWN_BRIDGE

    lumen, body, tree = _thick()
    _thin_vessel(lumen, (20, 30, 20), (20, 30, 63))

    result = _split(_graph(*JOINING), lumen, body, tree)

    (_u, _v, bridge), = _bridges(result)
    assert DRAWN_BRIDGE not in bridge and len(bridge["voxels"]) == 2


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


def _blind_vessel_off_the_thick_one():
    """A thin vessel leaving the thick one at y = 30 and ending blind inside
    the image 30 um out: 21 um beyond the bridge, which reaches the wall."""
    lumen, body, tree = _thick()
    _thin_vessel(lumen, (20, 30, 20), (20, 30, 50))
    G = _graph(("T0", "J", T0, (20, 30, 20)), ("J", "T1", (20, 30, 20), T1),
               ("TIP", "J", (20, 30, 50), (20, 30, 20)))
    return G, lumen, body, tree


def test_a_blind_vessel_beyond_a_bridge_stays_when_longer_than_a_stub():
    """Harvey's grid Y (2026-10-15): of the blind vessels the split took, the
    ones marked keep were 34-154 um long -- the thick vessel itself, long
    capillaries -- the ones marked remove 6 and 16 um. One goes only when the
    stub prune would take it: shorter than the prune's length and its 3
    lumen radii of the thick vessel (here 6 um: 18 um)."""
    G, lumen, body, tree = _blind_vessel_off_the_thick_one()

    kept = _split(G, lumen, body, tree)
    (u, v, bridge), = _bridges(kept)
    assert "TIP" in kept and kept.degree("TIP") == 1
    assert 30.0 - bridge["length"] >= 3 * RADIUS

    # At 4 radii (24 um) it is a stub, and goes with its bridge.
    removed = _split(G, lumen, body, tree, min_stub_length_radius_multiple=4.0)
    assert "TIP" not in removed and _bridges(removed) == []
    assert [sorted(e[:2]) for e in removed.edges] == [["T0", "T1"]]


def test_the_run_judges_blind_vessels_beyond_bridges_with_its_stub_settings():
    from haemolynx.pipeline.stages import thick_junction_bridge_arguments

    settings = {"bridge_require_mask_support": True, "bridge_max_background_gap_um": 2.0,
                "bridge_min_mask_fraction": 0.8, "min_stub_length": 25.0, "min_stub_length_radius_multiple": 1.0}
    arguments = thick_junction_bridge_arguments(settings)
    assert arguments["min_stub_length"] == 25.0 and arguments["min_stub_length_radius_multiple"] == 1.0
    G, lumen, body, tree = _blind_vessel_off_the_thick_one()

    result = _split(G, lumen, body, tree, **arguments)

    assert "TIP" not in result, "21 um beyond its bridge is under the run's 25 um"


def test_the_run_reads_the_centreline_tolerance_from_its_setting():
    """``thick_bridge_centreline_tolerance_um``: a vessel stopping 1 um off
    the thick vessel's axis. At the default (1.5 um, or 30% of the 6 um
    radius) its stretch inside the lumen goes and it is bridged to the thick
    centreline. At 4.5 um its last 3.5 um are within the tolerance of the
    tree, so the split takes them for the thick vessel's own centreline and
    the bridge stops there, 5 um long, instead of reaching the thick one."""
    from haemolynx.pipeline import default_schema
    from haemolynx.pipeline.stages import thick_junction_bridge_arguments

    settings = default_schema().defaults() | {"thick_vessel_extend_centreline_ends": False}
    assert thick_junction_bridge_arguments(settings)["centreline_tolerance_um"] == 1.5
    lumen, body, tree = _thick()
    _thin_vessel(lumen, (20, 30, 21), (20, 30, 63))
    G = _graph(("T0", "T1", T0, T1), ("X", "S", (20, 30, 63), (20, 30, 21)))

    default = _split(G, lumen, body, tree, **thick_junction_bridge_arguments(settings))
    wide = _split(G, lumen, body, tree, **thick_junction_bridge_arguments(
        settings | {"thick_bridge_centreline_tolerance_um": 4.5}
    ))

    (_u, _v, bridge), = _bridges(default)
    assert "S" not in default and REACH - 0.5 <= bridge["length"] <= REACH + 1.0
    (_u, _v, short), = _bridges(wide)
    assert short["length"] == pytest.approx(5.0, abs=0.6)
    assert "S" in wide


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


# --- what shape each bridge has (the plan's Phase 0 diagnostic) ---------------------


def _bridge_of_each_shape():
    """The thick vessel's centreline, and one hand-drawn bridge of each of
    the 2026-10-09 audit's shapes, each at its own place along y."""
    lumen, body, tree = _thick()
    G = _graph(("T0", "T1", T0, T1))
    bridge = {IS_ZERO_RESISTANCE: True}
    shapes = {
        # From the region's wall straight in to the centreline: 7.5 um, 1.2 radii.
        "radius_shaped": [("R1", "R0", (20, 10, 27.5), (20, 10, 20))],
        # From the wall to the centreline 10 um along: 12.5 um, 2 radii.
        "longer_than_radius": [("L1", "L0", (20, 30, 27.5), (20, 20, 20))],
        # Along the thick vessel's own centreline.
        "along_centreline": [("A0", "A1", (20, 34, 20), (20, 44, 20))],
        # From the wall to a node off the centreline another bridge reaches.
        # (C2-C0 is radius-shaped: C2 is within a voxel's step of the wall.)
        "on_another_bridge": [("C1", "C2", (20, 50, 27.5), (20, 50, 24.5)),
                              ("C2", "C0", (20, 50, 24.5), (20, 50, 20))],
        # Across the margin, both ends outside the lumen, never on the centreline.
        "wall_to_wall": [("W1", "W2", (27, 58, 20), (20, 58, 27))],
        # Inside the lumen, short of the centreline.
        "other": [("O1", "O0", (20, 62, 26), (20, 62, 23.5))],
    }
    for edges in shapes.values():
        G = nx.compose(G, _graph(*edges, **bridge))
    return G, lumen, body, tree, shapes


def test_each_bridge_is_reported_with_its_shape():
    G, lumen, body, tree, shapes = _bridge_of_each_shape()

    report = diagnose_thick_vessel_bridges(G, _region(lumen, body), tree, lumen, voxel_size_zyx=VS)

    shape_of = {frozenset((row["u"], row["v"])): row["shape"] for row in report["bridges"]}
    assert shape_of[frozenset(("C2", "C0"))] == "radius_shaped"
    for shape, edges in shapes.items():
        assert shape_of[frozenset(edges[0][:2])] == shape
    assert report["bridge_count"] == 7
    assert report["counts"] == {
        "radius_shaped": 2, "longer_than_radius": 1, "along_centreline": 1, "on_another_bridge": 1,
        "wall_to_wall": 1, "other": 1, "fork": 0,
    }
    assert set(report["counts"]) == set(BRIDGE_SHAPES)
    (radius,) = [row for row in report["bridges"] if row["inner_node"] == "R0"]
    assert radius["inner_node"] == "R0" and radius["outer_at_wall"]
    assert radius["length_um"] == pytest.approx(7.5)
    assert radius["local_radius_um"] == pytest.approx(RADIUS, abs=0.5)
    assert radius["length_over_radius"] == pytest.approx(7.5 / radius["local_radius_um"])
    assert report["median_length_um"]["longer_than_radius"] == pytest.approx(12.5)
    assert report["median_length_um"]["radius_shaped"] == pytest.approx((7.5 + 4.5) / 2)
    assert report["drawn_path_count"] == 0
    G.remove_edges_from([(u, v) for u, v, d in G.edges(data=True) if d.get(IS_ZERO_RESISTANCE)])
    assert G.number_of_edges() == 1, "read-only: the graph is the one handed in"


def test_a_bridge_ending_on_the_graphs_thick_centreline_off_the_tree_reaches_it():
    """The split bridges to the graph's own thick centreline, which strays off
    the tree by a few microns (on E14.5, 2026-10-24, every bridge ending off
    the tree ended on it, 2-6 um away): judged by the tree alone, the
    2026-10-09 audit's rule, it never reaches the centreline."""
    lumen, body, tree = _thick()
    G = _graph(("T0", "S", T0, (20, 30, 23.5)), ("S", "T1", (20, 30, 23.5), T1))
    G = nx.compose(G, _graph(("B1", "S", (20, 30, 29.5), (20, 30, 23.5)), **{IS_ZERO_RESISTANCE: True}))

    (row,) = diagnose_thick_vessel_bridges(G, _region(lumen, body), tree, lumen, voxel_size_zyx=VS)["bridges"]
    assert row["inner_node"] == "S" and not row["inner_on_tree"]
    assert row["inner_on_graph_centreline_off_tree"] and row["shape"] == "radius_shaped"
    report = diagnose_thick_vessel_bridges(G, _region(lumen, body), tree, lumen, voxel_size_zyx=VS)
    assert report["off_tree_count"] == 1 and report["unintended_count"] == 0
    (row,) = diagnose_thick_vessel_bridges(
        G, _region(lumen, body), tree, lumen, voxel_size_zyx=VS, graph_centreline=False
    )["bridges"]
    assert row["shape"] == "other"


def test_a_bridge_too_short_to_follow_the_tree_does_not_lie_along_it():
    """2.5 um from the region's edge 2 um off the tree to the tree: most of it
    is within the tolerance of the tree whatever its direction."""
    lumen, body, tree = _thick()
    G = nx.compose(_graph(("T0", "T1", T0, T1)),
                   _graph(("E1", "E0", (20, 30, 22.5), (20, 30, 20)), **{IS_ZERO_RESISTANCE: True}))
    thin_region = _region(lumen, body) & (np.abs(np.indices(SHAPE)[2] - 20) <= 2)

    (row,) = diagnose_thick_vessel_bridges(G, thin_region, tree, lumen, voxel_size_zyx=VS)["bridges"]

    assert row["share_on_centreline"] > 0.5
    assert row["shape"] == "radius_shaped"


def test_the_bridge_shape_report_reads_as_one_line():
    G, lumen, body, tree, _shapes = _bridge_of_each_shape()

    line = format_thick_vessel_bridges_report(
        diagnose_thick_vessel_bridges(G, _region(lumen, body), tree, lumen, voxel_size_zyx=VS)
    )

    assert line.startswith("Thick-vessel bridges: 7 bridge(s): 2 radius-shaped, 1 reaching the centreline")
    assert "1 wall to wall through the margin" in line and "length over the thick vessel's lumen radius" in line
    assert format_thick_vessel_bridges_report(
        diagnose_thick_vessel_bridges(_graph(("T0", "T1", T0, T1)), _region(lumen, body), tree, lumen)
    ) == "Thick-vessel bridges: 0 bridge(s)."


def test_the_splits_bridges_reach_from_the_wall_to_the_centreline():
    """What the split makes is the shape it means to: the region's wall to
    the centreline, the radius and the margin (here 9 um, 1.5 radii)."""
    lumen, body, tree = _thick()
    _thin_vessel(lumen, (20, 30, 20), (20, 30, 63))
    _thin_vessel(lumen, (20, 45, 0), (20, 45, 63))
    G = nx.compose(_graph(*JOINING), _graph(("Y1", "Y2", (20, 45, 63), (20, 45, 0))))
    result = _split(G, lumen, body, tree)

    report = diagnose_thick_vessel_bridges(result, _region(lumen, body), tree, lumen, voxel_size_zyx=VS)

    assert report["bridge_count"] == 3
    assert {row["shape"] for row in report["bridges"]} <= {"radius_shaped", "longer_than_radius"}
    for row in report["bridges"]:
        assert row["outer_at_wall"] and row["inner_to_tree_um"] <= 1.0
        assert row["length_over_radius"] == pytest.approx(REACH / row["local_radius_um"], abs=0.2)


def test_a_flicker_is_judged_by_its_length_in_microns_not_in_samples():
    """A stretch under 2 um between two others takes its neighbours' label.
    Sampled every 0.49 um (a 0.98 um voxel path densified to half a voxel),
    1.5 um is four samples: counted in samples it was a stretch of its own,
    and a vessel dipping out of the region for 1.5 um came back as a
    1.5 um vessel between two bridges, a loop inside the thick lumen."""
    from haemolynx.graph.thick_vessel_junctions import _INSIDE, _OUTSIDE, _runs

    labels = [_INSIDE] * 10 + [_OUTSIDE] * 4 + [_INSIDE] * 10
    along = np.arange(len(labels)) * 0.49
    assert _runs(labels, min_run_um=2.0, along=along) == [(0, 23, _INSIDE)]
    along = np.arange(len(labels)) * 0.6
    assert [value for _s, _e, value in _runs(labels, min_run_um=2.0, along=along)] == [_INSIDE, _OUTSIDE, _INSIDE]
    # An end run is judged the same way.
    labels = [_OUTSIDE] * 3 + [_INSIDE] * 10
    assert _runs(labels, min_run_um=2.0, along=np.arange(13) * 0.49) == [(0, 12, _INSIDE)]


# --- Harvey's examples, 2026-10-13 (PLAN_bridging.md, F1-F1c) -------------------


def _wall_case(stop_x: float, *, mask_from_x: float = 20.0):
    """The thick vessel along y, and a thin vessel along x whose centreline
    stops at *stop_x*, outside the thick-vessel region (its boundary is at
    x = 29); the thin vessel's mask runs from *mask_from_x* out to the face."""
    lumen, body, tree = _thick()
    _thin_vessel(lumen, (20, 30, mask_from_x), (20, 30, 63))
    G = _graph(("T0", "T1", T0, T1), ("X", "S", (20, 30, 63), (20, 30, stop_x)))
    return G, lumen, body, tree


def test_a_vessel_stopping_at_the_wall_is_drawn_on_and_bridged():
    """Skeletonisation ended the thin vessel's centreline 4 um short of the
    thick-vessel region, so nothing reached the thick centreline: drawn on
    through the mask, it gets the one bridge every joining vessel gets."""
    from haemolynx.graph.thick_vessel_junctions import JOINED_AT_WALL_UM

    G, lumen, body, tree = _wall_case(33.0)

    result = _split(G, lumen, body, tree)

    (u, v, bridge), = _bridges(result)
    assert REACH - 0.5 <= bridge["length"] <= REACH + 1.0
    assert "S" not in result
    inner = min((u, v), key=lambda n: abs(result.nodes[n]["pos"][2] - 20.0))
    assert result.nodes[inner]["pos"] == pytest.approx((20.0, 30.0, 20.0), abs=0.6)
    assert result.degree(inner) == 3, "the bridge splits the thick vessel's centreline"
    (thin,) = [d for a, b, d in result.edges(data=True) if "X" in (a, b)]
    assert not thin.get(IS_ZERO_RESISTANCE)
    assert thin["length"] == pytest.approx(43.0 - bridge["length"], abs=0.6)
    assert thin[JOINED_AT_WALL_UM] == pytest.approx(13.0, abs=0.6)


def test_a_thick_vessels_end_at_the_image_face_stays_when_a_vessel_joins_beside_it():
    """Regression (E14.5, the wall join at 15 um): skeletonisation stops the
    tree short of the image face, and a vessel joins the thick centreline
    13 um in, so the thick vessel's stretch to the face is an edge of its own
    inside the region, off the tree. Met at that node by the joining
    vessel's stretch, it was dropped with it as one connector, and the
    arteriole's inlet at the face went with it. It is the thick vessel's own
    end: kept, and the joining vessel still gets its one bridge."""
    body = _tube((20.0, -5.0, 20.0), (20.0, 63.0, 20.0), RADIUS)
    lumen = body.copy()
    tree = _tube((20.0, 12.0, 20.0), (20.0, 63.0, 20.0), 0.5)
    _thin_vessel(lumen, (20, 13, 20), (20, 13, 63))
    G = _graph(("F", "N", (20, 0, 20), (20, 13, 20)), ("N", "T1", (20, 13, 20), T1),
               ("X", "N", (20, 13, 63), (20, 13, 20)))

    result = _split(G, lumen, body, tree)

    assert "F" in result and result.degree("F") == 1, "the thick vessel's end at the face stays"
    (face_edge,) = [d for a, b, d in result.edges(data=True) if "F" in (a, b)]
    assert not face_edge.get(IS_ZERO_RESISTANCE)
    assert face_edge["length"] == pytest.approx(13.0, abs=0.6)
    ((_u, _v, bridge),) = _bridges(result)
    assert bridge["length"] == pytest.approx(REACH, abs=1.0), "the joining vessel's one bridge, wall to centreline"


@pytest.mark.parametrize(("past_lumen_um", "joined"), [(13.0, True), (17.0, False)])
def test_the_wall_join_reaches_15_um_past_the_lumen(past_lumen_um, joined):
    """Harvey's grids P and P2 (2026-10-25): every dead end 10-14 um past the
    lumen heading for it was marked "join". One stopping 13 um past the wall
    (the reach was 10 um) is drawn on through the mask; one 17 um past is not."""
    from haemolynx.graph.thick_vessel_junctions import JOINED_AT_WALL_UM, WALL_JOIN_REACH_UM

    assert WALL_JOIN_REACH_UM == 15.0
    stop_x = 20.0 + RADIUS + past_lumen_um
    G, lumen, body, tree = _wall_case(stop_x)

    result = _split(G, lumen, body, tree)

    assert bool(_bridges(result)) is joined
    assert ("S" not in result) is joined
    if joined:
        (thin,) = [d for a, b, d in result.edges(data=True) if "X" in (a, b)]
        assert thin[JOINED_AT_WALL_UM] == pytest.approx(stop_x - 20.0, abs=0.6)


def test_a_vessel_stopping_at_the_wall_is_left_when_the_join_is_switched_off():
    G, lumen, body, tree = _wall_case(33.0)

    result = _split(G, lumen, body, tree, join_dead_ends_at_walls=False)

    assert not _bridges(result)
    assert result.degree("S") == 1


def test_a_vessel_stopping_across_background_from_the_wall_is_not_drawn_on():
    """Its segmented vessel stops 9 um from the thick vessel's lumen: more
    background than a dropout, so not the same vessel running on."""
    G, lumen, body, tree = _wall_case(36.0, mask_from_x=35.0)

    result = _split(G, lumen, body, tree)

    assert not _bridges(result)
    assert result.degree("S") == 1


def _dead_end_kinds_after_split(result, lumen, body, tree):
    """Every dead end's kind on the split's *result*, the thick centreline
    recorded on it as the build records it (``stages.mark_dead_end_kinds``)."""
    from haemolynx.graph import dead_ends as de
    from haemolynx.graph.thick_vessel_junctions import ThickCentreline, thick_centreline_points_um
    from haemolynx.preprocessing import MaskSupport

    follower = ThickCentreline.from_region(_region(lumen, body), VS, segmentation=lumen, centreline=tree)
    de.record_thick_centreline(result, thick_centreline_points_um(tree, VS), follower.radius_um, tolerance_um=1.5)
    ends = de.classify_dead_ends(result, extent_um=de.image_extent_um(SHAPE, VS), support=MaskSupport(lumen, VS))
    return {tip: end.kind for tip, end in ends.items()}


@pytest.mark.parametrize(("stop_x", "mask_from_x", "joined"), [(33.0, 20.0, True), (35.0, 34.0, False)],
                         ids=["at the wall", "across background"])
def test_a_vessel_the_wall_join_leaves_is_a_thick_wall_dead_end(stop_x, mask_from_x, joined):
    """Phase 1.4 (iii): a vessel stopping across 6 um of background from the
    thick vessel's lumen gets no bridge, and its dead end is ``thick_wall``
    -- for review, where "Join to the vessel ahead" lands on the thick
    centreline. Stopping at the wall (ii), it is joined and no dead end."""
    lumen, body, tree = _thick()
    _thin_vessel(lumen, (20, 30, mask_from_x), (20, 30, 63))
    _thin_vessel(lumen, (20, 30, 50), (20, 63, 50))
    G = _graph(("T0", "T1", T0, T1), ("X", "J", (20, 30, 63), (20, 30, 50)),
               ("J", "S", (20, 30, 50), (20, 30, stop_x)), ("J", "Y", (20, 30, 50), (20, 63, 50)))

    result = _split(G, lumen, body, tree)
    kinds = _dead_end_kinds_after_split(result, lumen, body, tree)

    assert bool(_bridges(result)) is joined
    assert ("S" not in kinds) is joined
    if not joined:
        assert kinds["S"] == "thick_wall"
    assert kinds["X"] == kinds["Y"] == "leaves_image"


def test_a_vessel_running_alongside_the_thick_one_is_not_drawn_on_to_it():
    """Ending beside the thick vessel but heading along it (90 degrees off
    the way to its centreline), the mask joined to it all along: not a vessel
    opening into it."""
    lumen, body, tree = _thick()
    _thin_vessel(lumen, (20, 10, 31), (20, 40, 31))
    lumen[18:23, 10:41, 20:32] = True
    G = _graph(("T0", "T1", T0, T1), ("P", "S", (20, 10, 31), (20, 40, 31)))

    result = _split(G, lumen, body, tree)

    assert not any("S" in (u, v) for u, v, _d in _bridges(result))
    assert result.has_node("S") and result.degree("S") == 1


def test_a_wall_join_whose_route_turns_back_on_the_tip_is_refused():
    """Regression (review of 100293f..e1cf508): the join judged its turn on
    the straight line to the centreline, but draws the route through the
    mask. Here the straight line, dead ahead, crosses 5 um of background; the
    mask's only way round leaves the tip backwards, up a vessel 3 um behind
    it, and the join drew that, as no gap, optimise or orphan bridge may
    (grid R: all 13 such bridges wrong)."""
    from haemolynx.graph.thick_vessel_junctions import JOINED_AT_WALL_UM

    lumen, body, tree = _thick()
    _thin_vessel(lumen, (20, 30, 33), (20, 30, 63))
    _thin_vessel(lumen, (20, 30, 37), (20, 38, 37))
    _thin_vessel(lumen, (20, 38, 37), (20, 38, 20))
    G = _graph(("T", "T1", (20, 38, 20), T1), ("X", "S", (20, 30, 63), (20, 30, 34)))

    result = _split(G, lumen, body, tree)

    assert not any(d.get(JOINED_AT_WALL_UM) for *_e, d in result.edges(data=True))
    assert result.has_node("S") and result.degree("S") == 1


def test_a_short_wall_join_beside_a_vessel_in_the_same_lumen_is_refused():
    """Regression (review of 100293f..e1cf508): the same-lumen test left out
    the samples within a lumen radius of either end, so a join under about
    8 um, as most are, had none judged and was never refused. B stops 12 um
    from the thick centreline, 2 um beside A, which runs on into the thick
    vessel through the same segmented vessel: drawn on, B would be a second
    strand beside A. It is judged at its middle, as every reconnect is."""
    from haemolynx.graph.thick_vessel_junctions import JOINED_AT_WALL_UM

    lumen, body, tree = _thick()
    lumen[18:23, 26:35, 20:40] = True
    G = _graph(("T0", "J", T0, (20, 29, 20)), ("J", "T1", (20, 29, 20), T1),
               ("A", "J", (20, 29, 39), (20, 29, 20)), ("B", "S", (20, 31, 39), (20, 31, 32)))

    result = _split(G, lumen, body, tree)

    assert not any(d.get(JOINED_AT_WALL_UM) for *_e, d in result.edges(data=True))
    assert result.has_node("S") and result.degree("S") == 1


@pytest.mark.parametrize("stop_x", [28.0, 30.0])
def test_the_orphan_reconnect_leaves_a_tip_at_a_thick_wall_to_the_split(stop_x):
    """PLAN_bridging.md Phase 1.4: the orphan reconnect joins a dead end to
    a *node* within its reach (the run's 3 um), and that stays as it is --
    the junction step owns thick walls. The thin vessel stops 1 um inside
    the thick-vessel region (x = 28) or 1 um outside it with the mask
    running on (x = 30); the thick centreline has a node 3 um along it from
    the thin vessel's line. The reconnect, run as graph building runs it,
    leaves the tip alone, and the split gives the vessel its one bridge."""
    from haemolynx.graph.optimise import reconnect_orphan_and_dangling_nodes
    from haemolynx.pipeline import default_schema
    from haemolynx.preprocessing import MaskSupport

    lumen, body, tree = _thick()
    _thin_vessel(lumen, (20, 30, 20), (20, 30, 63))
    node_3um_along = (20.0, 33.0, 20.0)
    G = _graph(("T0", "N", T0, node_3um_along), ("N", "T1", node_3um_along, T1),
               ("X", "S", (20, 30, 63), (20, 30, stop_x)))
    G.graph["voxel_size"] = VS
    threshold = float(default_schema()["final_orphan_reconnect_threshold"].default)

    reconnected = reconnect_orphan_and_dangling_nodes(
        G.copy(), reconnect_threshold=threshold, mask_support=MaskSupport(lumen, VS)
    )

    assert threshold == pytest.approx(3.0)
    assert reconnected.degree("S") == 1
    assert not any(d.get("orphan_reconnect") for *_e, d in reconnected.edges(data=True))
    result = _split(reconnected, lumen, body, tree)
    (u, v, bridge), = _bridges(result)
    assert "S" not in result
    assert REACH - 0.5 <= bridge["length"] <= REACH + 1.0


def test_a_post_processing_split_draws_on_only_the_edges_it_names():
    G, lumen, body, tree = _wall_case(33.0)

    result = _split(G, lumen, body, tree, edges=[("T0", "T1", 0)])

    assert not _bridges(result)
    assert result.degree("S") == 1


def _taper():
    """A thick vessel ending at y = 30, narrowing there into a thin one that
    carries on along the same line."""
    lumen, body, tree = _thick(b=(20.0, 30.0, 20.0))
    _thin_vessel(lumen, (20, 30, 20), (20, 63, 20))
    return lumen, body, tree


@pytest.mark.parametrize("split_at_the_end", [True, False])
def test_a_thin_vessel_carrying_on_from_a_thick_vessels_end_is_bridged(split_at_the_end):
    """Harvey's grid P (2026-10-25): of the vessels carrying on from a thick
    centreline's end along its line, which the split used to leave as vessel
    (the thick vessel tapering), he marked 3 of 4 as vessels opening into it,
    to bridge. Now bridged like any other: from where it leaves the thick
    region (the 6 um cap and the 3 um margin past y = 30) to the thick
    centreline's end, whether the graph has a node there or one edge
    through, and the thin vessel is vessel from there on."""
    lumen, body, tree = _taper()
    if split_at_the_end:
        G = _graph(("T0", "J", T0, (20, 30, 20)), ("J", "T1", (20, 30, 20), T1))
    else:
        G = _graph(("T0", "T1", T0, T1))

    result = _split(G, lumen, body, tree)

    (u, v, bridge), = _bridges(result)
    outer, inner = sorted((u, v), key=lambda n: -result.nodes[n]["pos"][1])
    assert result.nodes[outer]["pos"] == pytest.approx((20.0, 39.5, 20.0), abs=0.6)
    assert result.nodes[inner]["pos"][1] == pytest.approx(30.0, abs=1.6), "on the thick centreline's end"
    (thin,) = [d for a, b, d in result.edges(data=True) if "T1" in (a, b)]
    assert thin["length"] == pytest.approx(T1[1] - 39.5, abs=0.6)
    assert not any(d.get(THICK_TAPER) for *_e, d in result.edges(data=True))


def _fork(branch_radius: float):
    """The thick vessel along y, and a branch of the thick centreline along x
    from y = 30, *branch_radius* wide: thick too, so the tree runs into it."""
    lumen, body, tree = _thick()
    branch = _tube((20, 30, 20), (20, 30, 63), branch_radius)
    lumen |= branch
    body |= branch
    tree |= _tube((20, 30, 20), (20, 30, 63), 0.5)
    G = _graph(("T0", "J", T0, (20, 30, 20)), ("J", "T1", (20, 30, 20), T1),
               ("J", "X", (20, 30, 20), (20, 30, 63)))
    return G, lumen, body, tree


def test_a_narrower_branch_of_the_thick_centreline_is_bridged_where_it_leaves_the_trunk():
    """Harvey: the thick centreline branches into a smaller one with no
    bridge. The branch's first 6 um lie inside the trunk's lumen: that
    stretch is the bridge, and the branch is vessel from the trunk's wall."""
    G, lumen, body, tree = _fork(4.0)

    result = _split(G, lumen, body, tree)

    (u, v, bridge), = _bridges(result)
    assert "J" in (u, v)
    assert RADIUS - 0.5 <= bridge["length"] <= RADIUS + 1.5
    (branch,) = [d for a, b, d in result.edges(data=True) if "X" in (a, b)]
    assert branch["length"] == pytest.approx(43.0 - bridge["length"], abs=0.6)
    for a, b in (("T0", "J"), ("J", "T1")):
        assert result.edges[a, b, 0]["voxels"] == G.edges[a, b, 0]["voxels"]
    # Both its ends lie on the tree, so it is marked: the bridge-shape report
    # would read it as lying along the thick centreline.
    assert bridge[THICK_FORK_BRIDGE] is True
    shapes = diagnose_thick_vessel_bridges(result, _region(lumen, body), tree, lumen, voxel_size_zyx=VS)
    assert [row["shape"] for row in shapes["bridges"]] == ["fork"]


def test_a_fork_of_equal_thick_vessels_gets_no_bridge():
    G, lumen, body, tree = _fork(RADIUS)

    result = _split(G, lumen, body, tree)

    assert not _bridges(result)


def test_a_narrower_branch_is_left_when_fork_bridges_are_switched_off():
    G, lumen, body, tree = _fork(4.0)

    result = _split(G, lumen, body, tree, bridge_forks=False)

    assert not _bridges(result)


def test_a_wall_join_targets_only_what_the_split_calls_thick_centreline():
    """Regression (E14.5, 2026-10-13): the join took any point near the tree
    as thick centreline, a vessel crossing the thick one included (within
    the tree's tolerance for a few microns: a flicker the split does not call
    centreline). Joined at one, the split read the join's new node in the
    input graph, which does not have it, and raised."""
    from scipy.spatial import cKDTree

    from haemolynx.graph.thick_vessel_junctions import _centreline_samples

    lumen, body, tree = _thick()
    _thin_vessel(lumen, (20, 36, 0), (20, 36, 63))
    G = _graph(("T0", "T1", T0, T1), ("V1", "V2", (20, 36, 0), (20, 36, 63)))
    tree_points = np.argwhere(tree) * np.asarray(VS)
    kd = cKDTree(tree_points)

    def near_tree(points):
        distance, _nearest = kd.query(points)
        return distance, distance <= 1.8

    _points, owners, _samples = _centreline_samples(G, near_tree, _region(lumen, body), VS)

    assert owners and {frozenset(o[:2]) for o in owners} == {frozenset(("T0", "T1"))}


def test_a_thick_vessel_ending_in_two_thin_ones_bridges_both():
    """Where the thick vessel ends, one thin vessel carries on along its line
    and another turns off: each opens into the thick vessel and gets its
    bridge to the centreline's end (the one along the line was left as
    vessel until grid P, 2026-10-25)."""
    lumen, body, tree = _taper()
    _thin_vessel(lumen, (20, 30, 20), (20, 30, 63))
    G = _graph(("T0", "J", T0, (20, 30, 20)), ("J", "T1", (20, 30, 20), T1),
               ("J", "X", (20, 30, 20), (20, 30, 63)))

    result = _split(G, lumen, body, tree)

    bridges = _bridges(result)
    assert len(bridges) == 2 and all("J" in (u, v) for u, v, _d in bridges)
    outer = sorted(tuple(result.nodes[v if u == "J" else u]["pos"]) for u, v, _d in bridges)
    assert outer[0] == pytest.approx((20.0, 30.0, 29.5), abs=0.6), "the one turning off along x"
    assert outer[1] == pytest.approx((20.0, 39.5, 20.0), abs=0.6), "the one carrying on along y"


@pytest.mark.parametrize("mask_support", [True, False])
def test_a_run_draws_vessels_on_from_walls_only_with_mask_supported_bridging(mask_support):
    """The wall join is a mask-supported bridge: the run's switch turns it
    off, and its limits on background are the run's own."""
    from haemolynx.pipeline.stages import thick_junction_bridge_arguments

    settings = {"bridge_require_mask_support": mask_support,
                "bridge_max_background_gap_um": 1.0, "bridge_min_mask_fraction": 0.9}
    arguments = thick_junction_bridge_arguments(settings)
    G, lumen, body, tree = _wall_case(33.0)

    result = _split(G, lumen, body, tree, **arguments)

    assert arguments["max_background_gap_um"] == 1.0 and arguments["min_mask_fraction"] == 0.9
    assert ("S" not in result) is mask_support


def test_a_taper_mark_on_a_saved_graph_is_kept_for_the_clean_up():
    """The split makes no taper now, but a graph saved before 2026-10-25 may
    carry the mark (a resumed run): an edge holding it keeps it through the
    split, so the clean-up after it leaves that edge alone as it did."""
    lumen, body, tree = _taper()
    G = _graph(("T0", "J", T0, (20, 30, 20)), ("J", "T1", (20, 30, 20), T1))
    G.edges["J", "T1", 0][THICK_TAPER] = True

    result = _split(G, lumen, body, tree)

    (thin,) = [d for a, b, d in result.edges(data=True) if "T1" in (a, b)]
    assert thin.get(THICK_TAPER) is True
    assert not any(d.get(THICK_TAPER) for a, b, d in result.edges(data=True) if "T0" in (a, b))


def _hook(far: int = 26):
    """A thin vessel leaving the thick one at y = 20, out along x, round and
    back towards it, stopping at its wall at y = *far*: joined, it would close
    a loop about 50 um round, round a hole of 7 um (16 um at *far* = 40)."""
    lumen, body, tree = _thick()
    for a, b in (((20, 20, 20), (20, 20, 40)), ((20, 20, 40), (20, far, 40)), ((20, far, 40), (20, far, 20))):
        _thin_vessel(lumen, a, b)
    G = _graph(("T0", "A", T0, (20, 20, 20)), ("A", "T1", (20, 20, 20), T1),
               ("A", "S", (20, 20, 20), (20, 20, 40), (20, far, 40), (20, far, 33)))
    return G, lumen, body, tree


def test_a_wall_join_closing_a_short_loop_is_refused():
    """On E14.5 all 10 short loops the build left ran through a vessel drawn on
    from a wall: the clean-up breaks every short loop but one through a
    bridge, so a join may not make one it would have broken."""
    from haemolynx.graph.thick_vessel_junctions import JOINED_AT_WALL_UM

    G, lumen, body, tree = _hook()

    result = _split(G, lumen, body, tree)

    # Not joined; the hook then ends blind and goes with its bridge at A.
    assert not any(d.get(JOINED_AT_WALL_UM) for *_e, d in result.edges(data=True))


def test_a_wall_join_closing_a_short_loop_round_a_wide_hole_is_made():
    """The clean-up leaves a short loop round a hole over 10 um (grids Q and
    Q2), so the join may close one; with that limit at 0 it may not."""
    from haemolynx.graph.thick_vessel_junctions import JOINED_AT_WALL_UM

    G, lumen, body, tree = _hook(far=40)

    joined = _split(G, lumen, body, tree)
    refused = _split(G, lumen, body, tree, short_loop_keep_hole_um=0.0)

    assert any(d.get(JOINED_AT_WALL_UM) for *_e, d in joined.edges(data=True))
    assert nx.cycle_basis(nx.Graph(joined))
    assert not any(d.get(JOINED_AT_WALL_UM) for *_e, d in refused.edges(data=True))


def test_with_the_short_loop_rule_off_a_wall_join_may_close_a_loop():
    G, lumen, body, tree = _hook()

    from haemolynx.graph.thick_vessel_junctions import JOINED_AT_WALL_UM

    result = _split(G, lumen, body, tree, max_short_loop_um=0.0)

    assert any(d.get(JOINED_AT_WALL_UM) for *_e, d in result.edges(data=True))
    assert nx.cycle_basis(nx.Graph(result))


# --- the thick centreline drawn from the tree where the graph strays or misses it (6a)


def _gap(stray: bool = False, detour: bool = False):
    """The thick vessel along y; its graph centreline follows the tree up to
    A (y = 10) and from B (y = 50), and between them nothing, a stray strand
    4 um off the tree (*stray*), or only a way round, from the thick
    vessel's two ends out through a vessel beside it (*detour*, ~150 um
    against the tree's 40). X, a thin vessel, enters the thick vessel at
    y = 30, too far from A or B to be bridged to either."""
    lumen, body, tree = _thick()
    _thin_vessel(lumen, (20, 30, 20), (20, 30, 63))
    edges = [("T0", "A", T0, (20, 10, 20)), ("B", "T1", (20, 50, 20), T1),
             ("X", "S", (20, 30, 63), (20, 30, 23))]
    if stray:
        edges.append(("A", "B", (20, 10, 20), (20, 30, 24), (20, 50, 20)))
    if detour:
        edges.append(("T0", "T1", T0, (20, 2, 55), (20, 61, 55), T1))
        for a, b in (((20, 2, 20), (20, 2, 55)), ((20, 2, 55), (20, 61, 55)), ((20, 61, 55), (20, 61, 20))):
            _thin_vessel(lumen, a, b)
    return _graph(*edges), lumen, body, tree


def _from_tree(G):
    from haemolynx.graph.thick_vessel_junctions import FROM_TREE

    return [(u, v, d) for u, v, d in G.edges(data=True) if d.get(FROM_TREE)]


def test_a_gap_in_the_graphs_thick_centreline_is_drawn_from_the_tree():
    """No graph centreline between A and B: X, reaching the thick vessel
    there, had nothing to be bridged to. Drawn from the tree, the thick
    vessel runs A to B and X gets its bridge."""
    G, lumen, body, tree = _gap()

    off = _split(G, lumen, body, tree, fill_tree_gaps=False)
    on = _split(G, lumen, body, tree)

    assert not _from_tree(off) and not any("S" in (u, v) or "X" in (u, v) for u, v, _d in _bridges(off))
    # Drawn A to B, then cut where X's bridge lands on it.
    pieces = _from_tree(on)
    assert sum(d["length"] for *_e, d in pieces) == pytest.approx(40.0, abs=1.5)
    assert nx.has_path(nx.Graph([(u, v) for u, v, _d in pieces]), "A", "B")
    assert len(_bridges(on)) == 1
    assert nx.has_path(on, "X", "T0") and nx.has_path(on, "X", "T1")


def test_a_strand_joining_the_gaps_sides_as_directly_is_left_as_the_vessel():
    """The graph's thick centreline wandering 4 um off the tree between A and
    B: it joins them about as directly as the tree (TREE_FILL_MIN_DETOUR),
    so it is the vessel there, nothing is drawn from the tree and it stays
    (grid V, 2026-10-15: of the strays and short ways across a junction the
    fill drew, Harvey kept the graph's own)."""
    G, lumen, body, tree = _gap(stray=True)

    on = _split(G, lumen, body, tree, fill_tree_gaps=True)

    assert not _from_tree(on)
    strand = [d for u, v, d in on.edges(data=True) if not d.get(IS_ZERO_RESISTANCE)
              and np.max(np.asarray(d["voxels"], float)[:, 2]) >= 23.5 and min(np.asarray(d["voxels"])[:, 1]) < 30]
    assert strand, "the graph's own strand A-B is still there"


def test_a_gap_the_graph_joins_only_the_long_way_round_is_drawn_from_the_tree():
    """A and B joined only through the thick vessel's ends and a vessel
    beside it, nearly four times the tree's path: the tree's path is drawn
    (grid V's right ones were 2.2-32 times), and X is bridged to it."""
    G, lumen, body, tree = _gap(detour=True)

    off = _split(G, lumen, body, tree, fill_tree_gaps=False)
    on = _split(G, lumen, body, tree)

    def x_bridged(H):
        """A bridge from X's side landing on the thick centreline at y = 30."""
        return any(
            any(np.allclose(H.nodes[n]["pos"], (20.0, 30.0, 20.0), atol=1.0) for n in (u, v))
            for u, v, _d in _bridges(H)
        )

    assert not _from_tree(off) and not x_bridged(off)
    assert sum(d["length"] for *_e, d in _from_tree(on)) == pytest.approx(40.0, abs=1.5)
    assert x_bridged(on)
    assert nx.number_connected_components(on) == 1


def test_a_post_processing_split_fills_only_the_gaps_its_edges_reach():
    """Regression (review of e1cf508): the fill ran over the whole network
    whatever *edges* named, so every post-processing regenerate drew thick
    centreline far from any edit -- where a hand edit had taken a stretch
    out, too -- and left it unmeasured. Now only a gap a named edge reaches,
    as far as a bridge from it looks for the thick centreline, is filled, and
    what is drawn takes *fill_attributes* (the stage marks it edited)."""
    G, lumen, body, tree = _gap()
    G.add_node("F1", pos=(20.0, 2.0, 50.0))
    G.add_node("F2", pos=(20.0, 20.0, 50.0))
    G.add_edge("F1", "F2", voxels=_line((20, 2, 50), (20, 20, 50)), length=18.0)

    far = _split(G, lumen, body, tree, edges=[("F1", "F2", 0)], fill_attributes={"edited": True})
    near = _split(G, lumen, body, tree, edges=[("X", "S", 0)], fill_attributes={"edited": True})

    assert not _from_tree(far)
    assert sorted(far.edges(keys=True)) == sorted(G.edges(keys=True))
    assert sum(d["length"] for *_e, d in _from_tree(near)) == pytest.approx(40.0, abs=1.5)
    assert all(d.get("edited") for *_e, d in _from_tree(near))
    assert len(_bridges(near)) == 1


def test_the_graphs_own_way_is_measured_along_its_vessels():
    """Between two points on one edge, along it; across edges, through the
    nodes; with no way within the cutoff, none."""
    from haemolynx.graph.thick_vessel_junctions import _graph_way_um

    G = _graph(("a", "b", (0, 0, 0), (0, 0, 10)), ("b", "c", (0, 0, 10), (0, 0, 30)))
    G.add_node("z", pos=(0.0, 5.0, 0.0))
    G.add_edge("y", "z", voxels=_line((0, 9, 0), (0, 5, 0)), length=4.0)
    G.nodes["y"]["pos"] = (0.0, 9.0, 0.0)
    ab, bc = ("a", "b", 0), ("b", "c", 0)
    assert _graph_way_um(G, (ab, 2.0, 10.0), (ab, 7.0, 10.0), cutoff=100.0) == pytest.approx(5.0)
    assert _graph_way_um(G, (ab, 2.0, 10.0), (bc, 5.0, 20.0), cutoff=100.0) == pytest.approx(13.0)
    assert _graph_way_um(G, (ab, 2.0, 10.0), (bc, 15.0, 20.0), cutoff=10.0) == np.inf
    assert _graph_way_um(G, (ab, 2.0, 10.0), (("y", "z", 0), 1.0, 4.0), cutoff=100.0) == np.inf


def test_the_run_turns_the_fill_on_with_its_setting():
    from haemolynx.pipeline.stages import thick_junction_bridge_arguments

    settings = {"bridge_require_mask_support": True, "bridge_max_background_gap_um": 2.0,
                "bridge_min_mask_fraction": 0.8, "remove_short_loops_under_um": 200.0}
    # On by default since grid V (2026-10-15).
    assert thick_junction_bridge_arguments(settings)["fill_tree_gaps"] is True
    settings["thick_vessel_fill_centreline_gaps"] = False
    assert thick_junction_bridge_arguments(settings)["fill_tree_gaps"] is False
    from haemolynx.pipeline import default_schema

    assert default_schema()["thick_vessel_fill_centreline_gaps"].default is True


# --- how much of a centreline follows the thick vessel's tree -------------------


def test_the_length_of_a_centreline_on_the_thick_vessels_tree():
    """``ThickCentreline.along_um``: what the lumen clean-up asks to keep the
    thick vessel's own strand. The tree of a 12 um tube runs down its middle;
    a strand there lies on it for its length, one 4 um off it (past the
    larger of 1.5 um and 0.3 radii) not at all, and one leaving the tube
    only for the part inside."""
    from haemolynx.graph.thick_vessel_junctions import ThickCentreline

    region = _tube(T0, T1, RADIUS)
    tree = ThickCentreline.from_region(region, VS, segmentation=region)
    middle = _line((20.0, 15.0, 20.0), (20.0, 45.0, 20.0))
    assert tree.along_um(middle) == pytest.approx(30.0, abs=1.0)
    assert tree.along_um(middle + np.array([0.0, 0.0, 4.0])) == 0.0
    leaving = _line((20.0, 30.0, 20.0), (20.0, 30.0, 50.0))
    assert tree.along_um(leaving) < 3.0
    assert tree.along_um(middle[:1]) == 0.0
    assert ThickCentreline.from_region(np.zeros(SHAPE, dtype=bool), VS) is None


# --- a thick centreline stopping at a dead end, drawn on (grid Y, 2026-10-15) ----


def _stopping(tree_to: float = 26.0, body_to: float = 30.0, graph_to: float | None = None, ahead: str = "vessel"):
    """A thick vessel along y up to *body_to* (its region and tree end there,
    the tree at *tree_to*), its graph centreline T0-E stopping at *graph_to*
    (the tree's end by default), and its segmented lumen running on along y
    as a thin vessel: to a vessel crossing at y = 56 (X1-X2, *ahead*
    "vessel"), or out of the image at y = 63 ("face")."""
    body = _tube((20.0, 2.0, 20.0), (20.0, body_to, 20.0), RADIUS)
    lumen = body | _tube((20.0, body_to, 20.0), (20.0, 63.0 if ahead == "face" else 56.0, 20.0), 2.0)
    edges = [("T0", "E", T0, (20.0, tree_to if graph_to is None else graph_to, 20.0))]
    if ahead == "vessel":
        lumen |= _tube((20.0, 56.0, 2.0), (20.0, 56.0, 62.0), 2.0)
        edges.append(("X1", "X2", (20.0, 56.0, 2.0), (20.0, 56.0, 62.0)))
    tree = _tube((20.0, 2.0, 20.0), (20.0, tree_to, 20.0), 0.5)
    return _graph(*edges), lumen, body, tree


def _extended(G):
    from haemolynx.graph.thick_vessel_junctions import EXTENDED_UM

    return [d[EXTENDED_UM] for *_e, d in G.edges(data=True) if d.get(EXTENDED_UM)]


def test_a_thick_centreline_stopping_where_its_tree_ends_is_drawn_on_to_the_vessel_ahead():
    """Grid W 16, 20 and 23: the thick region, and the tree drawn through it,
    end where the graph's thick centreline does, while the segmented vessel
    runs on. Drawn on through the segmentation, the thick vessel reaches the
    vessel ahead; left off (the default), it stays a dead end."""
    G, lumen, body, tree = _stopping()

    off = _split(G, lumen, body, tree)
    on = _split(G, lumen, body, tree, extend_thick_ends=True)

    assert off.degree("E") == 1 and not nx.has_path(off, "T0", "X1")
    assert "E" not in on
    assert nx.has_path(on, "T0", "X1") and nx.has_path(on, "T0", "X2")
    assert max(_extended(on)) == pytest.approx(30.0, abs=2.0)
    assert not [n for n in on if on.degree(n) == 1 and on.nodes[n]["pos"][1] < 56.0 and n != "T0"]


def test_a_thick_centreline_is_drawn_on_along_its_tree_first_where_the_tree_runs_on():
    """The tree runs on 20 um past where the graph's thick centreline stops,
    with no graph beside it (one side only, so not the fill's gap): the
    thick vessel is drawn along the tree, then on through the segmentation."""
    from haemolynx.graph.thick_vessel_junctions import EXTENDED_UM

    G, lumen, body, tree = _stopping(tree_to=40.0, body_to=44.0, graph_to=20.0)

    on = _split(G, lumen, body, tree, extend_thick_ends=True)

    assert "E" not in on and nx.has_path(on, "T0", "X1")
    drawn = [np.asarray(d["voxels"], float) for *_e, d in on.edges(data=True) if d.get(EXTENDED_UM)]
    along = np.vstack(drawn)
    on_tree = along[(along[:, 1] > 22.0) & (along[:, 1] < 38.0)]
    assert len(on_tree) and np.all(np.hypot(on_tree[:, 0] - 20.0, on_tree[:, 2] - 20.0) <= 1.0)
    assert max(_extended(on)) == pytest.approx(36.0, abs=2.0)


def test_a_thick_centreline_whose_vessel_leaves_the_image_first_is_not_drawn_on():
    """The segmented vessel runs on out of the image with nothing on the way:
    an open end, left as it is. Grid Y: drawn to a face, right at 1 of 3
    places (wrong 10 um across a vessel to a face and 10 um through a 1 um
    neck; right a 4 um step out to one)."""
    G, lumen, body, tree = _stopping(ahead="face")

    on = _split(G, lumen, body, tree, extend_thick_ends=True)

    assert not _extended(on)
    assert on.degree("E") == 1


def test_a_vessel_further_ahead_than_the_reach_is_not_drawn_on_to():
    """The vessel ahead 46 um on through the segmentation, past
    EXTEND_REACH_UM: grid Y's ways Harvey wanted were 32 um or shorter, the
    one too long 54 um."""
    from haemolynx.graph.thick_vessel_junctions import EXTEND_REACH_UM

    G, lumen, body, tree = _stopping(tree_to=10.0, body_to=14.0)

    on = _split(G, lumen, body, tree, extend_thick_ends=True)

    assert EXTEND_REACH_UM < 44.0
    assert not _extended(on) and on.degree("E") == 1


def test_a_thick_centreline_is_not_drawn_on_sideways_to_a_vessel_beside_its_end():
    """Nothing ahead -- the segmented vessel ends in the thick vessel's rounded
    end -- and a thin vessel leaving it at right angles beside the dead end:
    the way on leaves within 60 degrees of the thick vessel's heading, so it
    is not drawn there (grid W 20: the cheapest way turned 91 degrees to a
    vessel beside the thick one, not on to the one ahead Harvey named)."""
    G, lumen, body, tree = _stopping(ahead="none")
    lumen |= _tube((20.0, 26.0, 20.0), (20.0, 26.0, 62.0), 1.5)
    G.add_node("S1", pos=(20.0, 26.0, 62.0))
    G.add_node("S2", pos=(20.0, 26.0, 40.0))
    G.add_edge("S1", "S2", voxels=_line((20, 26, 62), (20, 26, 40)), length=22.0)

    on = _split(G, lumen, body, tree, extend_thick_ends=True, join_dead_ends_at_walls=False)

    assert not _extended(on)
    assert on.degree("E") == 1 and not nx.has_path(on, "T0", "S1")


def test_a_post_processing_split_draws_on_only_the_thick_ends_of_its_edges():
    G, lumen, body, tree = _stopping()

    other = _split(G, lumen, body, tree, extend_thick_ends=True, edges=[("X1", "X2", 0)])
    named = _split(G, lumen, body, tree, extend_thick_ends=True, edges=[("T0", "E", 0)],
                   fill_attributes={"edited": True})

    assert not _extended(other) and other.degree("E") == 1
    assert _extended(named) and nx.has_path(named, "T0", "X1")
    from haemolynx.graph.thick_vessel_junctions import EXTENDED_UM

    assert all(d.get("edited") for *_e, d in named.edges(data=True) if d.get(EXTENDED_UM))


def test_the_run_draws_thick_centrelines_on_by_default_and_off_with_its_setting():
    from haemolynx.pipeline import default_schema
    from haemolynx.pipeline.stages import thick_junction_bridge_arguments

    settings = {"bridge_require_mask_support": True, "bridge_max_background_gap_um": 2.0,
                "bridge_min_mask_fraction": 0.8, "remove_short_loops_under_um": 200.0}
    # On by default since 2026-10-20 (grid Y, 17/18; Harvey's E14.5 Claudin run).
    assert thick_junction_bridge_arguments(settings)["extend_thick_ends"] is True
    assert default_schema()["thick_vessel_extend_centreline_ends"].default is True
    settings["thick_vessel_extend_centreline_ends"] = False
    assert thick_junction_bridge_arguments(settings)["extend_thick_ends"] is False


# --- the inscribed radius read a block at a time; one measurement a run (2026-10-18) ---


def _inscribed_radius_read_whole(points_um, lumen, vs):
    """The inscribed radius as it was measured before it went block by block:
    the points' box, padded by 60 um, read and dilated at once."""
    from scipy.ndimage import binary_dilation, generate_binary_structure
    from scipy.spatial import cKDTree

    vs = np.asarray(vs, dtype=float)
    pad = np.ceil(60.0 / vs).astype(int)
    lo = np.maximum(np.floor(points_um.min(axis=0) / vs).astype(int) - pad, 0)
    hi = np.minimum(np.ceil(points_um.max(axis=0) / vs).astype(int) + pad + 1, np.asarray(lumen.shape))
    crop = np.asarray(lumen[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]], dtype=bool)
    wall = binary_dilation(crop, structure=generate_binary_structure(3, 1)) & ~crop
    if not wall.any():
        return np.full(len(points_um), np.inf)
    return cKDTree((np.argwhere(wall) + lo) * vs).query(points_um)[0]


def _small_blocks(monkeypatch, block_voxels):
    import functools

    from haemolynx.graph import thick_vessel_junctions as tvj
    from haemolynx.preprocessing.memmap_support import iter_blocks

    monkeypatch.setattr(tvj, "iter_blocks", functools.partial(iter_blocks, block_voxels=block_voxels))


@pytest.mark.parametrize("vs", [(1.0, 1.0, 1.0), (2.0, 0.7, 0.9)])
@pytest.mark.parametrize("seed", range(3))
def test_the_inscribed_radius_read_a_block_at_a_time_is_the_box_read_whole(monkeypatch, vs, seed):
    """Bit for bit, on points inside, outside and on the wall of a thick
    vessel in speckled tissue, with blocks far smaller than the box."""
    from haemolynx.graph.thick_vessel_junctions import _inscribed_radius_um

    rng = np.random.default_rng(seed)
    shape = (30, 90, 50)
    lumen = _tube((15, 0, 25), (15, 89, 25), 8.0, shape=shape) | (rng.random(shape) > 0.995)
    on_tree = np.argwhere(_tube((15, 0, 25), (15, 89, 25), 0.5, shape=shape)) * np.asarray(vs)
    anywhere = rng.uniform(0.0, 1.0, size=(200, 3)) * (np.asarray(shape) - 1) * np.asarray(vs)
    points = np.vstack([on_tree, anywhere])
    _small_blocks(monkeypatch, 4_000)

    for subset in (points, on_tree[:1], anywhere[:7]):
        expected = _inscribed_radius_read_whole(subset, lumen, vs)
        assert np.array_equal(_inscribed_radius_um(subset, lumen, np.asarray(vs)), expected)


def test_a_point_further_than_the_reach_from_any_wall_is_measured_against_every_block(monkeypatch):
    """A lumen 140 um across with one speck of background in it: the centre's
    wall is 100 um off, beyond the 60 um the first reading looks, and a lumen
    with no outside at all is inf, as read whole."""
    from haemolynx.graph.thick_vessel_junctions import _inscribed_radius_um

    vs = (2.0, 2.0, 2.0)
    lumen = np.ones((70, 70, 70), dtype=bool)
    lumen[6, 6, 6] = False
    centre = np.array([[70.0, 70.0, 70.0], [68.0, 72.0, 70.0]])
    _small_blocks(monkeypatch, 3_000)

    measured = _inscribed_radius_um(centre, lumen, np.asarray(vs))

    assert np.array_equal(measured, _inscribed_radius_read_whole(centre, lumen, vs))
    assert measured[0] == pytest.approx(np.sqrt(3) * 58.0)
    full = np.ones_like(lumen)
    assert np.array_equal(_inscribed_radius_um(centre, full, np.asarray(vs)), [np.inf, np.inf])


def test_a_block_no_point_is_within_reach_of_is_never_read(monkeypatch):
    """Two points 260 um apart along a thick vessel: their box runs the
    vessel's length, and the blocks of it more than 60 um from both are left
    unread."""
    from haemolynx.graph.thick_vessel_junctions import _inscribed_radius_um
    from haemolynx.preprocessing.memmap_support import iter_blocks

    lumen = _tube((15, 0, 25), (15, 300, 25), 6.0, shape=(30, 301, 50))
    _small_blocks(monkeypatch, 20_000)
    reads = []

    class Counting(np.ndarray):
        def __getitem__(self, key):
            reads.append(key)
            return super().__getitem__(key)

    points = np.array([[15.0, 20.0, 25.0], [15.0, 280.0, 25.0]])
    radius = _inscribed_radius_um(points, lumen.view(Counting), np.ones(3))

    assert np.array_equal(radius, _inscribed_radius_read_whole(points, lumen, (1.0, 1.0, 1.0)))
    blocks = list(iter_blocks((30, 301, 50), halo=1, block_voxels=20_000))
    assert 0 < len(reads) < len(blocks)
    assert all(key[1].start <= 81 or key[1].stop >= 220 for key in reads)


def test_a_split_handed_the_runs_support_and_tree_radius_is_the_split_without(monkeypatch):
    """build_network measures the tree's radius once (ThickCentreline) and
    shares one MaskSupport: the split it makes is the same, node for node."""
    from haemolynx.graph import thick_vessel_junctions as tvj
    from haemolynx.preprocessing import MaskSupport

    G, lumen, body, tree = _wall_case(33.0)
    region = _region(lumen, body)
    plain = insert_thick_vessel_junction_nodes(G, region, voxel_size_zyx=VS, segmentation=lumen, centreline=tree)
    follower = tvj.ThickCentreline.from_region(region, VS, segmentation=lumen, centreline=tree)
    support = MaskSupport(lumen, VS)
    measured = []
    real = tvj._tree_radius_um
    monkeypatch.setattr(tvj, "_tree_radius_um", lambda *a: measured.append(1) or real(*a))

    shared = insert_thick_vessel_junction_nodes(
        G, region, voxel_size_zyx=VS, segmentation=lumen, centreline=tree,
        mask_support=support, tree_radius_um=follower.radius_um,
    )

    assert _exact(shared) == _exact(plain)
    assert "S" not in shared, "the wall join ran"
    assert measured == [], "the tree's radius was not measured again"
    assert support._distance is not None, "the join read the run's support"
    assert np.array_equal(
        follower.radius_um,
        tvj._tree_radius_um(np.argwhere(tree) * np.asarray(VS), lumen, region, np.asarray(VS)),
    )


def _exact(G):
    """Every node, edge, key and attribute, floats to the last bit."""

    def plain(value):
        if isinstance(value, np.ndarray):
            return ("array", value.dtype.str, value.shape, value.tobytes())
        if isinstance(value, (float, np.floating)):
            return ("float", repr(float(value)))
        if isinstance(value, (list, tuple)):
            return tuple(plain(v) for v in value)
        if isinstance(value, dict):
            return tuple(sorted((str(k), plain(v)) for k, v in value.items()))
        return value

    nodes = sorted((repr(n), plain(d)) for n, d in G.nodes(data=True))
    edges = sorted((repr(u), repr(v), repr(k), plain(d)) for u, v, k, d in G.edges(keys=True, data=True))
    return nodes, edges
