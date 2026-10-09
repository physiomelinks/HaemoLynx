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


def test_the_pipeline_default_prunes_a_blind_spur_three_radii_long_and_no_more():
    """Lee thinning leaves spurs about as long as a capillary is wide, ending
    where the mask does. At 1.5 radii (3 um on a 2 um capillary) they stayed:
    on E14.5, 1,037 of 1,234 dead ends were blind ends kept by that rule."""
    from haemolynx.pipeline import default_schema

    multiple = default_schema()["min_stub_length_radius_multiple"].default
    G = nx.MultiGraph()
    for node, pos in (("start", (50, 50, 0)), ("junction", (50, 50, 50)), ("end", (50, 50, 100)),
                      ("spur", (50, 55, 50)), ("capillary_end", (50, 43, 50))):
        G.add_node(node, pos=np.asarray(pos, dtype=float))
    for u, v in (("start", "junction"), ("junction", "end"),
                 ("junction", "spur"), ("junction", "capillary_end")):
        _straight(G, u, v)

    def radius_at(position):
        return 2.0 if float(position[2]) == 50.0 else 0.0

    def prune(radius_multiple):
        return prune_vascular_stubs(
            G, min_stub_length=10.0, max_iterations=1,
            radius_at=radius_at, radius_multiple=radius_multiple,
        )

    assert "spur" in prune(1.5)  # 5 um >= 1.5 x 2 um
    out = prune(multiple)
    assert "spur" not in out  # 5 um < 3 x 2 um
    assert "capillary_end" in out  # 7 um >= 3 x 2 um
    assert {"start", "end"} <= set(out.nodes)


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


# --- dead ends judged as whole chains, and against other vessels' lumens -------


def _wide_trunk_with(*paths):
    """A 12 um trunk along x (centre y = 7) through a junction at x = 10, and
    the given extra paths, each an edge; nodes are named by position."""
    from lumen_artefact_fixtures import polyline_graph

    return polyline_graph([[(7, 7, 0), (7, 7, 10)], [(7, 7, 10), (7, 7, 59)], *paths])


def _rules(mask):
    from haemolynx.graph.assemble import mask_continues_past, mask_lumen_test
    from haemolynx.preprocessing import MaskSupport

    support = MaskSupport(mask, (1.0, 1.0, 1.0))
    return dict(
        inside_lumen=mask_lumen_test(mask, (1.0, 1.0, 1.0)),
        radius_at=lambda p: float(support.radius(np.asarray(p, dtype=float).reshape(1, 3))[0]),
        mask_continues_at=mask_continues_past(support),
        lumen_radius_at=support.radius,
    )


def test_a_hair_along_a_vessels_wall_is_pruned_whatever_its_length():
    """30 um along the inside of a 12 um trunk: long enough for the length
    rule, its tip far from the junction -- but it never leaves the lumen."""
    from lumen_artefact_fixtures import wide_vessel

    G = _wide_trunk_with([(7, 7, 10), (7, 10, 14), (7, 10, 44)])
    rules = _rules(wide_vessel())
    without = prune_vascular_stubs(G, min_stub_length=10.0, radius_multiple=1.5,
                                   **{k: v for k, v in rules.items() if k != "lumen_radius_at"})
    judged = prune_vascular_stubs(G, min_stub_length=10.0, radius_multiple=1.5, **rules)
    assert (7.0, 10.0, 44.0) in without
    assert (7.0, 10.0, 44.0) not in judged
    assert {(7.0, 7.0, 0.0), (7.0, 7.0, 10.0), (7.0, 7.0, 59.0)} <= set(judged)


def test_a_branch_that_leaves_the_vessel_stays():
    from lumen_artefact_fixtures import capsule

    mask = capsule(np.zeros((15, 40, 60), dtype=bool), (7, 7, -10), (7, 7, 70), 6)
    capsule(mask, (7, 7, 30), (7, 33, 30), 2)
    G = _wide_trunk_with([(7, 7, 10), (7, 7, 30)], [(7, 7, 30), (7, 31, 30)])
    G.remove_edge((7.0, 7.0, 10.0), (7.0, 7.0, 59.0))
    G.add_edge((7.0, 7.0, 30.0), (7.0, 7.0, 59.0), voxels=[[7.0, 7.0, float(x)] for x in range(30, 60)], length=29.0)
    judged = prune_vascular_stubs(G, min_stub_length=10.0, radius_multiple=1.5, **_rules(mask))
    assert (7.0, 31.0, 30.0) in judged


def test_a_hair_beside_a_real_branch_does_not_take_the_branch_with_it():
    """Two dead ends leave one junction side by side; neither is judged against
    the other, so the real branch, which leaves the trunk, stays."""
    from lumen_artefact_fixtures import capsule

    mask = capsule(np.zeros((15, 40, 60), dtype=bool), (7, 7, -10), (7, 7, 70), 6)
    capsule(mask, (7, 7, 30), (7, 33, 30), 2)
    G = _wide_trunk_with([(7, 7, 10), (7, 7, 30)], [(7, 7, 30), (7, 31, 30)], [(7, 7, 30), (7, 8, 31), (7, 20, 31)])
    G.remove_edge((7.0, 7.0, 10.0), (7.0, 7.0, 59.0))
    G.add_edge((7.0, 7.0, 30.0), (7.0, 7.0, 59.0), voxels=[[7.0, 7.0, float(x)] for x in range(30, 60)], length=29.0)
    judged = prune_vascular_stubs(G, min_stub_length=10.0, radius_multiple=1.5, **_rules(mask))
    assert (7.0, 31.0, 30.0) in judged


def test_a_dead_end_in_short_pieces_is_judged_whole():
    """Three 5 um pieces joined at degree-2 nodes: each is under the 10 um
    threshold, the dead end is not. Judged piece by piece it was worn away."""
    from lumen_artefact_fixtures import polyline_graph

    G = polyline_graph([
        [(0, 0, 0), (0, 0, 20)], [(0, 0, 20), (0, 0, 40)], [(0, 0, 20), (0, 5, 20)],
        [(0, 5, 20), (0, 10, 20)], [(0, 10, 20), (0, 15, 20)],
    ])
    pruned = prune_vascular_stubs(G, min_stub_length=10.0)
    assert (0.0, 15.0, 20.0) in pruned
    short = prune_vascular_stubs(G, min_stub_length=20.0)
    assert not any(n in short for n in ((0.0, 5.0, 20.0), (0.0, 10.0, 20.0), (0.0, 15.0, 20.0)))
    assert (0.0, 0.0, 20.0) in short or (0.0, 0.0, 0.0) in short

