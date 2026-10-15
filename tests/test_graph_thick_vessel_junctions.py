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
def test_a_thin_vessel_carrying_on_from_a_thick_vessels_end_gets_no_bridge(split_at_the_end):
    """One vessel narrowing (Harvey: these should not have a bridge), not a
    vessel opening into another: the thin vessel stays vessel to the thick
    centreline's end, whether the graph has a node there or one edge through."""
    lumen, body, tree = _taper()
    if split_at_the_end:
        G = _graph(("T0", "J", T0, (20, 30, 20)), ("J", "T1", (20, 30, 20), T1))
    else:
        G = _graph(("T0", "T1", T0, T1))

    result = _split(G, lumen, body, tree)

    assert not _bridges(result)
    assert sorted(len(d["voxels"]) for *_e, d in result.edges(data=True)) == sorted(
        len(d["voxels"]) for *_e, d in G.edges(data=True)
    )
    assert sum(d["length"] for *_e, d in result.edges(data=True)) == pytest.approx(59.0, abs=0.6)


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


def test_a_thick_vessel_ending_in_two_thin_ones_bridges_only_the_one_turning_off():
    """Where the thick vessel ends, one thin vessel carries on along its line
    (the thick vessel narrowing: no bridge) and another turns off (opening
    into it: a bridge). Two bridges met at that end on E14.5 more often than
    one, and the taper rule skipped them all."""
    lumen, body, tree = _taper()
    _thin_vessel(lumen, (20, 30, 20), (20, 30, 63))
    G = _graph(("T0", "J", T0, (20, 30, 20)), ("J", "T1", (20, 30, 20), T1),
               ("J", "X", (20, 30, 20), (20, 30, 63)))

    result = _split(G, lumen, body, tree)

    (u, v, bridge), = _bridges(result)
    outer = u if result.nodes[u]["pos"][2] > result.nodes[v]["pos"][2] else v
    assert result.nodes[outer]["pos"][2] > 25.0, "the bridge is the one turning off along x"
    along = [d for a, b, d in result.edges(data=True) if "T1" in (a, b)]
    assert len(along) == 1 and along[0]["voxels"] == G.edges["J", "T1", 0]["voxels"]


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


def test_a_taper_is_marked_for_the_clean_up_that_follows():
    from haemolynx.graph.thick_vessel_junctions import THICK_TAPER

    lumen, body, tree = _taper()
    G = _graph(("T0", "J", T0, (20, 30, 20)), ("J", "T1", (20, 30, 20), T1))

    result = _split(G, lumen, body, tree)

    assert result.edges["J", "T1", 0].get(THICK_TAPER) is True
    assert not result.edges["T0", "J", 0].get(THICK_TAPER)


def _hook():
    """A thin vessel leaving the thick one at y = 20, out along x, round and
    back towards it, stopping at its wall at y = 30: joined, it would close a
    loop about 60 um round."""
    lumen, body, tree = _thick()
    for a, b in (((20, 20, 20), (20, 20, 40)), ((20, 20, 40), (20, 30, 40)), ((20, 30, 40), (20, 30, 20))):
        _thin_vessel(lumen, a, b)
    G = _graph(("T0", "A", T0, (20, 20, 20)), ("A", "T1", (20, 20, 20), T1),
               ("A", "S", (20, 20, 20), (20, 20, 40), (20, 30, 40), (20, 30, 33)))
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

    off = _split(G, lumen, body, tree)
    on = _split(G, lumen, body, tree, fill_tree_gaps=True)

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

    off = _split(G, lumen, body, tree)
    on = _split(G, lumen, body, tree, fill_tree_gaps=True)

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
    assert thick_junction_bridge_arguments(settings)["fill_tree_gaps"] is False
    settings["thick_vessel_fill_centreline_gaps"] = True
    assert thick_junction_bridge_arguments(settings)["fill_tree_gaps"] is True
