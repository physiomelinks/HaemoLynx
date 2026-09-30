"""``graph.prune.prune_vascular_stubs``: stubs judged by their vessel's size,
and never at an image face.

One fixed length (10 um by default) could only get one of two cases right: a
skeleton spur on a 20 um-radius vessel is longer than that and survived, while
a real 8 um capillary end is shorter and went. And terminals where the image
cut a vessel -- the open ends inlets and outlets are chosen from -- were pruned
like any spur.
"""
from __future__ import annotations

import networkx as nx
import numpy as np

from haemolynx.graph import build_graph_from_skeleton
from haemolynx.graph.prune import prune_vascular_stubs


def _straight(G, u, v):
    path = [list(map(float, G.nodes[u]["pos"])), list(map(float, G.nodes[v]["pos"]))]
    G.add_edge(u, v, voxels=path, length=float(np.linalg.norm(np.subtract(path[1], path[0]))))


def _two_junctions():
    """A backbone through a thick-vessel junction and a capillary junction,
    each with one terminal: a 25 um spur off the thick one, a real 8 um
    capillary end off the thin one."""
    G = nx.MultiGraph()
    positions = {
        "start": (50, 50, 0),
        "thick": (50, 50, 100),
        "thin": (50, 50, 200),
        "end": (50, 50, 300),
        "spur": (50, 75, 100),
        "capillary_end": (50, 58, 200),
    }
    for node, pos in positions.items():
        G.add_node(node, pos=np.asarray(pos, dtype=float))
    for u, v in (("start", "thick"), ("thick", "thin"), ("thin", "end"),
                 ("thick", "spur"), ("thin", "capillary_end")):
        _straight(G, u, v)
    radii = {100.0: 20.0, 200.0: 2.5}

    def radius_at(position):
        return radii.get(float(position[2]), 0.0)

    return G, radius_at


def test_a_fixed_length_prunes_the_real_capillary_and_keeps_the_spur():
    """What the single threshold did: exactly the wrong way round here."""
    G, _radius_at = _two_junctions()

    out = prune_vascular_stubs(G, min_stub_length=10.0, max_iterations=1)

    assert "spur" in out and "capillary_end" not in out


def test_a_radius_relative_threshold_prunes_the_spur_and_keeps_the_capillary():
    G, radius_at = _two_junctions()

    out = prune_vascular_stubs(
        G, min_stub_length=10.0, max_iterations=1, radius_at=radius_at, radius_multiple=1.5
    )

    assert "spur" not in out  # 25 um < 1.5 x 20 um
    assert "capillary_end" in out  # 8 um >= 1.5 x 2.5 um


def test_min_stub_length_still_applies_where_no_radius_can_be_read():
    G, _ = _two_junctions()

    out = prune_vascular_stubs(
        G, min_stub_length=10.0, max_iterations=1, radius_at=lambda _p: 0.0, radius_multiple=1.5
    )

    assert "spur" in out and "capillary_end" not in out


def test_a_short_stub_ending_at_an_image_face_is_kept():
    """Regression: a vessel cut by the image -- an inlet or outlet candidate
    -- was pruned like any spur if its last segment was short."""
    G = nx.MultiGraph()
    for node, pos in (("a", (50, 50, 20)), ("b", (50, 50, 80)), ("c", (50, 20, 50)),
                      ("junction", (50, 50, 50)), ("at_face", (50, 50, 96)), ("inside", (50, 56, 50))):
        G.add_node(node, pos=np.asarray(pos, dtype=float))
    for u, v in (("a", "junction"), ("b", "junction"), ("c", "junction"),
                 ("b", "at_face"), ("junction", "inside")):
        _straight(G, u, v)

    out = prune_vascular_stubs(
        G, min_stub_length=20.0, max_iterations=1, image_extent_um=(100.0, 100.0, 100.0)
    )

    assert "at_face" in out  # 16 um long, but 4 um from the x=100 face
    assert "inside" not in out  # 6 um long, 44 um from every face


def test_graph_building_prunes_a_spur_by_the_vessel_it_leaves():
    """End to end: a spur 7 voxels long off a 6-voxel-radius vessel is pruned
    at 1.5 radii, though it survives the fixed 5 um threshold."""
    shape = (21, 41, 60)
    skeleton = np.zeros(shape, dtype=bool)
    skeleton[10, 20, :] = True  # the vessel's centreline, running face to face
    skeleton[10, 21:28, 30] = True  # a 7-voxel spur from its middle
    zz, yy, xx = np.indices(shape)
    mask = (zz - 10) ** 2 + (yy - 20) ** 2 <= 36
    mask |= (np.abs(zz - 10) <= 1) & (yy >= 20) & (yy <= 28) & (np.abs(xx - 30) <= 1)

    common = dict(
        graph_reconnect_threshold=0.0,
        final_orphan_reconnect_threshold=0.0,
        cluster_collapse_distance=0.0,
        min_stub_length=5.0,
    )
    fixed = build_graph_from_skeleton(skeleton, **common)
    by_radius = build_graph_from_skeleton(
        skeleton, segmentation_mask=mask, min_stub_length_radius_multiple=1.5, **common
    )

    def reaches_the_spur_tip(G):
        return any(np.allclose(G.nodes[n]["pos"], (10, 27, 30)) for n in G.nodes)

    assert reaches_the_spur_tip(fixed)
    assert not reaches_the_spur_tip(by_radius)
    # The vessel's own ends are at the image faces: kept either way.
    for G in (fixed, by_radius):
        ends = [G.nodes[n]["pos"][2] for n in G.nodes if G.degree(n) == 1]
        assert 0.0 in ends and 59.0 in ends
