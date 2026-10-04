"""Direction-aware node-cluster collapse: hand-verified geometry, no pipeline involved.

Mirrors tests/test_cartwheel_guard.py's style, since this module exists
specifically to stop collapse_node_clusters from producing the shape that
guard flags -- see graph/direction_aware_collapse.py's own docstring.
"""
from __future__ import annotations

import math

import numpy as np
import networkx as nx

from haemolynx.graph._helpers import calculate_path_length, path_separation
import pytest

from haemolynx.graph.cartwheel_guard import detect_cartwheel_hubs
from haemolynx.graph.collapse import collapse_node_clusters
from haemolynx.graph.direction_aware_collapse import (
    DEFAULT_DUPLICATE_PATH_TOLERANCE_UM,
    DEFAULT_MAX_RADIAL_DISPERSION,
    DEFAULT_MIN_DEGREE_FOR_DISPERSION_CHECK,
    collapse_node_clusters_direction_aware,
)


def _wheel_graph(
    n_spokes: int = 8, hub_jitter: float = 0.3, spoke_length: float = 20.0
) -> nx.MultiGraph:
    """*n_spokes* tightly-clustered hub nodes, each with one edge leaving in
    its own evenly-spaced direction -- the shape cartwheel_guard exists to
    flag, and the shape a naive collapse would produce by merging every hub
    node into one representative carrying every spoke."""
    G = nx.MultiGraph()
    rng = np.random.default_rng(0)
    for i in range(n_spokes):
        offset = rng.normal(scale=hub_jitter, size=3)
        G.add_node(i, pos=np.array([0.0, 0.0, 0.0]) + offset)
    for i in range(n_spokes):
        angle = 2 * math.pi * i / n_spokes
        direction = np.array([0.0, math.cos(angle), math.sin(angle)])
        ext_pos = direction * spoke_length
        ext_id = 100 + i
        G.add_node(ext_id, pos=ext_pos)
        c_pos = G.nodes[i]["pos"]
        G.add_edge(
            i, ext_id,
            length=float(np.linalg.norm(ext_pos - c_pos)),
            voxels=[c_pos.tolist(), ext_pos.tolist()],
        )
    return G


def _coherent_bundle_graph(
    n_branches: int = 8, spread_degrees: float = 20.0, spoke_length: float = 150.0
) -> nx.MultiGraph:
    """*n_branches* tightly-clustered hub nodes, each leaving in a direction
    within a narrow cone -- a real busy junction where every branch
    continues roughly one way, not a cartwheel. Collapsing all of them
    should stay allowed. *spoke_length* is long enough that neighbouring
    external endpoints stay outside the default 5 um collapse distance from
    each other despite the narrow angular spread -- this fixture is only
    testing whether the *hub* side stays free to merge, not creating a
    second cluster on the external side by accident."""
    G = nx.MultiGraph()
    rng = np.random.default_rng(1)
    spread = math.radians(spread_degrees)
    for i in range(n_branches):
        offset = rng.normal(scale=0.3, size=3)
        G.add_node(i, pos=np.array([0.0, 0.0, 0.0]) + offset)
        angle = -spread / 2 + spread * i / max(n_branches - 1, 1)
        direction = np.array([0.0, math.cos(angle), math.sin(angle)])
        ext_pos = direction * spoke_length
        ext_id = 100 + i
        G.add_node(ext_id, pos=ext_pos)
        c_pos = G.nodes[i]["pos"]
        G.add_edge(
            i, ext_id,
            length=float(np.linalg.norm(ext_pos - c_pos)),
            voxels=[c_pos.tolist(), ext_pos.tolist()],
        )
    return G


# --- the core claim: refuses to build what cartwheel_guard would flag ------


def test_a_wheel_shaped_cluster_is_not_collapsed_into_one_cartwheel_hub():
    G = _wheel_graph()

    legacy = collapse_node_clusters(G, distance_threshold=5.0)
    aware = collapse_node_clusters_direction_aware(G, distance_threshold=5.0)

    # The legacy behaviour really does produce the pathology this exists to
    # fix -- otherwise this fixture proves nothing.
    legacy_hubs = detect_cartwheel_hubs(
        legacy,
        min_degree=DEFAULT_MIN_DEGREE_FOR_DISPERSION_CHECK,
        max_radial_dispersion=DEFAULT_MAX_RADIAL_DISPERSION,
    )
    assert legacy_hubs, "fixture bug: legacy collapse did not even produce a cartwheel hub"
    assert max(dict(legacy.degree()).values()) == 8

    aware_hubs = detect_cartwheel_hubs(
        aware,
        min_degree=DEFAULT_MIN_DEGREE_FOR_DISPERSION_CHECK,
        max_radial_dispersion=DEFAULT_MAX_RADIAL_DISPERSION,
    )
    assert aware_hubs == [], "direction-aware collapse still produced a flaggable cartwheel hub"
    # And it did not just refuse to collapse anything at all.
    assert aware.number_of_nodes() < G.number_of_nodes()


def test_a_coherent_busy_junction_still_collapses_fully():
    """The fix must not become "never merge more than a few nodes" -- a
    real junction where every branch leaves roughly the same way should
    collapse exactly as the legacy method does."""
    G = _coherent_bundle_graph()

    legacy = collapse_node_clusters(G, distance_threshold=5.0)
    aware = collapse_node_clusters_direction_aware(G, distance_threshold=5.0)

    assert aware.number_of_nodes() == legacy.number_of_nodes()
    assert aware.number_of_edges() == legacy.number_of_edges()
    assert max(dict(aware.degree()).values()) == 8


def test_a_small_cluster_below_min_degree_always_collapses_regardless_of_shape():
    """Two hub nodes each with one external spoke pointing the opposite way
    -- as wheel-shaped as two spokes can be -- but degree 2 is below the
    default floor of 6, so it merges exactly as the legacy method would."""
    G = nx.MultiGraph()
    G.add_node(0, pos=np.array([0.0, 0.0, 0.0]))
    G.add_node(1, pos=np.array([1.0, 0.0, 0.0]))
    G.add_node(10, pos=np.array([0.0, 0.0, -20.0]))
    G.add_node(11, pos=np.array([0.0, 0.0, 20.0]))
    G.add_edge(0, 10, length=20.0, voxels=[[0, 0, 0], [0, 0, -20]])
    G.add_edge(1, 11, length=19.0, voxels=[[1, 0, 0], [0, 0, 20]])

    aware = collapse_node_clusters_direction_aware(G, distance_threshold=5.0)

    assert aware.number_of_nodes() == 3
    assert set(dict(aware.degree()).values()) == {2, 1, 1}


# --- deduplicating redundant parallel wiring after a merge -----------------


def test_two_cluster_members_reaching_the_same_neighbour_keep_only_the_shorter_edge():
    G = nx.MultiGraph()
    G.add_node(0, pos=np.array([0.0, 0.0, 0.0]))
    G.add_node(1, pos=np.array([1.0, 0.0, 0.0]))
    G.add_node(2, pos=np.array([20.0, 0.0, 0.0]))
    # Lengths are what the paths measure: the representative's own edge takes
    # a detour, the member's runs straight.
    G.add_edge(0, 2, length=20.9, voxels=[[0, 0, 0], [10, 3, 0], [20, 0, 0]])
    G.add_edge(1, 2, length=19.0, voxels=[[1, 0, 0], [20, 0, 0]])

    out = collapse_node_clusters_direction_aware(G, distance_threshold=5.0)

    assert out.number_of_nodes() == 2
    assert out.number_of_edges() == 1
    remaining = list(out.edges(data=True))[0][2]
    # The straight path, now starting at the merged node's centroid (0.5, 0, 0).
    assert len(remaining["voxels"]) == 2
    assert remaining["length"] == pytest.approx(19.5)


def _bowed_path(start, end, bulge, n_points=31):
    """*n_points* from *start* to *end*, bowing *bulge* um out along y at the middle."""
    t = np.linspace(0.0, 1.0, n_points)
    points = np.outer(1.0 - t, start) + np.outer(t, end)
    points[:, 1] += bulge * np.sin(np.pi * t)
    return points.tolist()


@pytest.mark.parametrize("member_takes", ["the bowed vessel", "the straight vessel"])
def test_cluster_members_reaching_one_neighbour_by_separate_vessels_keep_both(member_takes):
    """Regression test: two vessels from a cluster to one neighbour, 10 um
    apart, are two vessels. Deduplication used to keep only the shorter edge
    whatever its route, so merging the member either dropped its own vessel
    (longer) or deleted the representative's (shorter) -- as at E14.5 MCA
    node 4932, whose 31 um vessel to node 3643 vanished behind the
    representative's 28 um one, 11 um away."""
    G = nx.MultiGraph()
    G.add_node(0, pos=np.array([0.0, 0.0, 0.0]))
    G.add_node(1, pos=np.array([1.5, 0.0, 0.0]))
    G.add_node(2, pos=np.array([30.0, 0.0, 0.0]))
    straight = _bowed_path([0.0, 0.0, 0.0], [30.0, 0.0, 0.0], bulge=0.0)
    bowed = _bowed_path([1.5, 0.0, 0.0], [30.0, 0.0, 0.0], bulge=10.0)
    if member_takes == "the straight vessel":
        straight = _bowed_path([1.5, 0.0, 0.0], [30.0, 0.0, 0.0], bulge=0.0)
        bowed = _bowed_path([0.0, 0.0, 0.0], [30.0, 0.0, 0.0], bulge=10.0)
    for path in (straight, bowed):
        node = 0 if path[0] == [0.0, 0.0, 0.0] else 1
        G.add_edge(node, 2, voxels=path, length=calculate_path_length(path))

    out = collapse_node_clusters_direction_aware(G, distance_threshold=5.0)

    assert out.number_of_nodes() == 2  # 0 and 1 merged into one representative
    assert out.number_of_edges(0, 2) == 2
    centroid = [0.75, 0.0, 0.0]
    expected = sorted(calculate_path_length([centroid] + path[1:]) for path in (straight, bowed))
    assert sorted(d["length"] for _, _, d in out.edges(data=True)) == pytest.approx(expected)


def test_a_members_noise_route_is_merged_even_when_its_rewired_end_jumps_past_the_tolerance():
    """A chained cluster 0-1-2-3-4 (4.5 um steps) collapses onto node 2, its
    centroid. Node 4 reaches neighbour 5 by skeleton noise -- an approach
    onto node 2's route, then a voxel staircase never more than 1 um off it
    -- so that is node 2's vessel twice, and only the shorter copy stays.
    Re-ending node 4's path at node 2 adds a straight 9 um jump running up
    to 8 um from node 2's route: that is where the member was, not a second
    vessel, so it must not count."""
    G = nx.MultiGraph()
    positions = {
        0: [0, 0, 0], 1: [4.5, 0, 0], 2: [9, 0, 0], 3: [13.5, 0, 0], 4: [18, 0, 0],
        5: [9, 30, 0], 6: [9, -30, 0], 7: [-30, 0, 0],
    }
    for node, pos in positions.items():
        G.add_node(node, pos=np.array(pos, dtype=float))
    route = [[9.0, float(y), 0.0] for y in range(31)]
    noise = (
        [[18.0 - k, float(k), 0.0] for k in range(9)]
        + [[9.0, float(y), float(y % 2)] for y in range(9, 30)]
        + [[9.0, 30.0, 0.0]]
    )
    for u, v, path in (
        (2, 5, route),
        (4, 5, noise),
        (2, 6, [[9.0, 0.0, 0.0], [9.0, -30.0, 0.0]]),  # node 2 is the best-connected,
        (0, 7, [[0.0, 0.0, 0.0], [-30.0, 0.0, 0.0]]),  # so it represents the cluster
    ):
        G.add_edge(u, v, voxels=path, length=calculate_path_length(path))
    # Whole-path separation alone would call the rewired noise a second vessel.
    assert path_separation(route, [[9.0, 0.0, 0.0]] + noise[1:]) > DEFAULT_DUPLICATE_PATH_TOLERANCE_UM

    out = collapse_node_clusters_direction_aware(G, distance_threshold=5.0)

    assert sorted(out.nodes) == [2, 5, 6, 7]
    assert out.number_of_edges(2, 5) == 1
    remaining = next(iter(out.get_edge_data(2, 5).values()))
    assert remaining["voxels"] == route
    assert remaining["length"] == pytest.approx(30.0)


def test_two_centrelines_of_one_wide_vessel_are_merged():
    """Two cluster members 4 um apart, each reaching the neighbour along its
    own centreline 4 um from the other: one wide vessel skeletonised twice
    (on real data, such pairs mostly have lumen between them), so it keeps
    one edge -- though the paths are further apart than degree-2 merging's
    3 um tolerance."""
    G = nx.MultiGraph()
    G.add_node(0, pos=np.array([0.0, 0.0, 0.0]))
    G.add_node(1, pos=np.array([0.0, 4.0, 0.0]))
    G.add_node(2, pos=np.array([30.0, 2.0, 0.0]))
    near = [[float(x), 0.0, 0.0] for x in range(30)] + [[30.0, 2.0, 0.0]]
    far = [[float(x), 4.0, 0.0] for x in range(30)] + [[30.0, 2.0, 0.0]]
    G.add_edge(0, 2, voxels=near, length=calculate_path_length(near))
    G.add_edge(1, 2, voxels=far, length=calculate_path_length(far))
    assert 3.0 < path_separation(near, far) <= DEFAULT_DUPLICATE_PATH_TOLERANCE_UM

    out = collapse_node_clusters_direction_aware(G, distance_threshold=5.0)

    assert out.number_of_nodes() == 2
    assert out.number_of_edges(0, 2) == 1
    remaining = next(iter(out.get_edge_data(0, 2).values()))
    # Equally long, so the representative's own copy stays, re-ended at the centroid.
    assert remaining["voxels"] == [[0.0, 2.0, 0.0]] + near[1:]


def test_a_genuine_pre_existing_loop_survives_a_merged_members_shorter_edge():
    """Regression test: the representative already has a real loop -- two
    pre-existing parallel edges to the same external neighbour, lengths 8
    and 12 -- before any cluster member merges into it. A cluster member
    contributing a shorter edge to that same neighbour (noise, or simply a
    closer path) must not delete either genuine pre-existing edge: only
    edges introduced *during this collapse* are fair game for the
    keep-the-shorter rule (see test_two_cluster_members_reaching_the_same_
    neighbour_keep_only_the_shorter_edge above), not edges that predate it."""
    G = nx.MultiGraph()
    G.add_node(0, pos=np.array([0.0, 0.0, 0.0]))
    G.add_node(1, pos=np.array([1.0, 0.0, 0.0]))
    G.add_node(99, pos=np.array([20.0, 0.0, 0.0]))
    # Two genuine, differently shaped loop paths, and a straighter one from
    # the member; lengths are what the paths measure.
    loop_a = [[0, 0, 0], [10, 2, 0], [20, 0, 0]]
    loop_b = [[0, 0, 0], [10, 6, 0], [20, 0, 0]]
    G.add_edge(0, 99, length=20.4, voxels=loop_a)
    G.add_edge(0, 99, length=23.3, voxels=loop_b)
    G.add_edge(1, 99, length=19.0, voxels=[[1, 0, 0], [20, 0, 0]])

    out = collapse_node_clusters_direction_aware(G, distance_threshold=5.0)

    assert out.number_of_nodes() == 2  # 0 and 1 merged into one representative
    lengths = sorted(data["length"] for _, _, data in out.edges(0, data=True))
    moved = [0.5, 0.0, 0.0]  # the merged node's centroid
    expected = sorted(
        calculate_path_length(path)
        for path in ([moved] + loop_a[1:], [moved] + loop_b[1:], [moved, [20, 0, 0]])
    )
    assert lengths == pytest.approx(expected), (
        "both genuine pre-existing edges must survive alongside the merged "
        "member's shorter edge, none of them deleted"
    )


# --- distance_only parity: reusing the option must not add new settings ---


@pytest.mark.parametrize(
    "build_graph", [_wheel_graph, _coherent_bundle_graph], ids=["wheel", "coherent_bundle"]
)
def test_the_legacy_method_is_unaffected_by_this_modules_existence(build_graph):
    """collapse_node_clusters itself is never called by this module, and
    never imported for its behaviour (only two small, pure, read-only
    helpers) -- confirmed by re-running it directly and comparing."""
    G = build_graph()
    a = collapse_node_clusters(G, distance_threshold=5.0)
    b = collapse_node_clusters(G, distance_threshold=5.0)
    assert set(a.nodes) == set(b.nodes)
    assert set(a.edges(keys=True)) == set(b.edges(keys=True))


# --- validation --------------------------------------------------------


@pytest.mark.parametrize("bad_dispersion", [-0.1, 1.1])
def test_a_max_radial_dispersion_outside_0_1_is_rejected(bad_dispersion):
    G = nx.MultiGraph()
    with pytest.raises(ValueError, match="max_radial_dispersion"):
        collapse_node_clusters_direction_aware(
            G, max_radial_dispersion=bad_dispersion
        )


def test_a_min_degree_below_two_is_rejected():
    G = nx.MultiGraph()
    with pytest.raises(ValueError, match="min_degree_for_dispersion_check"):
        collapse_node_clusters_direction_aware(
            G, min_degree_for_dispersion_check=1
        )


def test_a_graph_with_fewer_than_two_positioned_nodes_is_left_alone():
    G = nx.MultiGraph()
    G.add_node(0, pos=np.array([0.0, 0.0, 0.0]))
    out = collapse_node_clusters_direction_aware(G, distance_threshold=5.0)
    assert list(out.nodes) == [0]
